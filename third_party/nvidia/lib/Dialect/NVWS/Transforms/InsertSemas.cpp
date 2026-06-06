// v4 commit 5: discovery + ACCESS DAG + OWNERSHIP DAG + RAW-SYNC DAG +
// OPT-SYNC DAG + semaphore IR emission.
//
// Per meta2nvws-plan/per-edge-sema-plan.v4.md Implementation Plan, this
// commit adds the final stage of the v4 pipeline:
//
//   discover backing buffers
//     -> build ACCESS DAG per buffer
//     -> build OWNERSHIP DAG per (logicalGroupId, resourceKey)
//     -> derive RAW-SYNC DAG per (logicalGroupId, resourceKey)
//     -> derive OPT-SYNC DAG via fanout/fanin combines
//     -> render nvws.semaphore.* IR from the OPT-SYNC DAG
//
// When NVWS_INSERT_SEMA_DUMP_DAG=1, the pass prints each planned stage to
// stderr before emitting IR.

#include "Utilities.h"
#include "InsertSemasModel.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;
using namespace triton::nvws::insert_semas;

// ---------------------------------------------------------------------------
// Discovery helpers (buffer.id / offset / overlap classes).
// ---------------------------------------------------------------------------

// Layout-correct extent. Delegates to `mlir::triton::getMemDescSize` so
// TMEM allocs report columns (from tensor_memory_encoding) and SMEM
// allocs report bytes (product(shape) * elementBitWidth/8). The
// `buffer.offset` attr on each alloc is in the same native unit.
static int64_t getAllocExtent(MemDescType type) {
  return static_cast<int64_t>(mlir::triton::getMemDescSize(type));
}

static bool intervalsOverlap(int64_t aLo, int64_t aHi, int64_t bLo,
                             int64_t bHi) {
  return aLo < bHi && bLo < aHi;
}

static bool isTmemAlloc(Operation *op) { return isa<TMEMAllocOp>(op); }
static bool isLocalAlloc(Operation *op) { return isa<LocalAllocOp>(op); }

// A local alloc is a candidate semaphore-managed backing buffer if its
// memdesc type has the multi-stage layout that insert-allocas chose.
// Heuristic that matches the prior implementation: the alloc result type
// has a rank greater than the source tensor's rank (the leading dim
// is the multi-stage depth).
static bool isLocalSemaphoreBackingType(MemDescType type) {
  return type.getMutableMemory();
}

// ---------------------------------------------------------------------------
// Alias chain + touch construction (v4 §Access Events).
// ---------------------------------------------------------------------------

static FailureOr<AliasInfo> lookupAlias(BufferGroup &group, Value v) {
  auto it = group.aliases.find(v);
  if (it == group.aliases.end()) return failure();
  return it->second;
}

static LogicalResult addAlias(BufferGroup &group, Operation *op) {
  if (op->getNumResults() != 1) return success();
  if (!isa<MemDescType>(op->getResult(0).getType())) return success();

  std::optional<AliasInfo> source;
  unsigned sourceOperand = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    auto it = group.aliases.find(operand);
    if (it == group.aliases.end()) continue;
    source = it->second;
    sourceOperand = idx;
    break;
  }
  if (!source) return success();

  if (!isSupportedAliasOp(op)) {
    if (group.isTmem())
      return op->emitError("nvws-insert-semas: unsupported memdesc alias use ")
             << op->getName();
    return op->emitError("local semaphore: unsupported SMEM memdesc alias use ")
           << op->getName();
  }

  AliasInfo alias = *source;
  alias.steps.push_back({op, sourceOperand});
  group.aliases.insert({op->getResult(0), alias});
  group.aliasOps.push_back(op);
  return success();
}

static void addTouch(BufferGroup &group, AccessEvent &event, Value v,
                     AccessEffect effect) {
  auto alias = lookupAlias(group, v);
  if (failed(alias)) return;
  BufferMember &member = group.members[alias->memberIdx];
  event.touches.push_back(
      {alias->memberIdx, member.resourceKey, effect, v, *alias});
}

static bool aliasesSameResource(BufferGroup &group, Value a, Value b) {
  auto aAlias = lookupAlias(group, a);
  auto bAlias = lookupAlias(group, b);
  if (failed(aAlias) || failed(bAlias))
    return false;
  if (aAlias->memberIdx >= group.members.size() ||
      bAlias->memberIdx >= group.members.size())
    return false;
  return group.members[aAlias->memberIdx].resourceKey ==
         group.members[bAlias->memberIdx].resourceKey;
}

static bool isAccumulatorImmediatelyOverwritten(BufferGroup &group,
                                                MMAv5OpInterface mma) {
  Operation *op = mma.getOperation();
  if (!op || !op->getBlock() || op->getNumResults() == 0)
    return false;
  Operation *next = op->getNextNode();
  if (!next)
    return false;
  auto store = dyn_cast<TMEMStoreOp>(next);
  if (!store || store.getDep() != op->getResult(0))
    return false;
  return aliasesSameResource(group, mma.getAccumulator(), store.getDst()) &&
         sameOwner(getPartitionId(op), getPartitionId(store));
}

// ---------------------------------------------------------------------------
// Physical conflict key (v4 §Physical Conflict Key).
// Union-find over members of the same group: members whose offset intervals
// overlap share a resourceKey.
// ---------------------------------------------------------------------------

// Union-find members of one buffer.id group by overlap in their native
// (memory-space) interval [offset, offset + size). Works uniformly for
// TMEM (columns) and SMEM (bytes) because `getAllocExtent` selects the
// right unit per memory space.
static void assignResourceKeys(BufferGroup &group) {
  SmallVector<int64_t> parent(group.members.size());
  for (auto [idx, _] : llvm::enumerate(group.members)) parent[idx] = idx;
  std::function<int64_t(int64_t)> find = [&](int64_t i) -> int64_t {
    if (parent[i] == i) return i;
    parent[i] = find(parent[i]);
    return parent[i];
  };
  auto unite = [&](int64_t a, int64_t b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[b] = a;
  };
  for (unsigned i = 0; i < group.members.size(); ++i)
    for (unsigned j = i + 1; j < group.members.size(); ++j) {
      BufferMember &lhs = group.members[i];
      BufferMember &rhs = group.members[j];
      // Plan §Physical Conflict Key: members whose native intervals overlap
      // MUST share a resourceKey (overlap ⇒ same key), with no exception.
      // A reuse handoff between overlapping members of different element
      // types (e.g. an f32 MMA accumulator whose columns are reused to stage
      // an f16 MMA operand) is still a physical conflict and must be
      // synchronized through the shared resource; separating them by element
      // type would silently drop the reuse edge and race.
      if (intervalsOverlap(lhs.offset, lhs.offset + lhs.extent, rhs.offset,
                           rhs.offset + rhs.extent))
        unite(i, j);
    }
  for (auto [idx, member] : llvm::enumerate(group.members))
    member.resourceKey = find(idx);
}

// ---------------------------------------------------------------------------
// Discovery: build one SmallVector<BufferGroup> covering BOTH ttng.tmem_alloc
// and ttg.local_alloc, uniformly. v4 §Uniform Access-DAG Builder.
// ---------------------------------------------------------------------------

template <typename AllocOpT>
static BufferGroup makeGroup(MemorySpaceKind memory, int64_t logicalId,
                             MutableArrayRef<AllocOpT> allocs) {
  BufferGroup group;
  group.memory = memory;
  group.logicalId = logicalId;
  for (auto [idx, allocOp] : llvm::enumerate(allocs)) {
    auto type = cast<MemDescType>(allocOp.getResult().getType());
    BufferMember member;
    member.allocOp = allocOp;
    member.value = allocOp.getResult();
    member.type = type;
    member.offset = getBufferOffset(allocOp);
    member.extent =
        memory == MemorySpaceKind::Local && !type.getShape().empty()
            ? type.getShape().front()
            : getAllocExtent(type);
    group.aliases.insert(
        {allocOp.getResult(), AliasInfo{static_cast<unsigned>(idx), {}}});
    group.members.push_back(member);
  }
  assignResourceKeys(group);
  return group;
}

static SmallVector<BufferGroup, 0>
collectAllBackingGroups(triton::FuncOp funcOp) {
  SmallVector<BufferGroup, 0> groups;
  int64_t nextSyntheticId = 0;

  // TMEM: group by buffer.id.
  llvm::MapVector<int64_t, SmallVector<TMEMAllocOp>> tmemBuckets;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    std::optional<int64_t> id = getBufferId(allocOp);
    int64_t key = id.value_or(nextSyntheticId++);
    tmemBuckets[key].push_back(allocOp);
  });
  for (auto &[id, allocs] : tmemBuckets)
    groups.push_back(makeGroup<TMEMAllocOp>(MemorySpaceKind::Tmem, id, allocs));

  // Local: group by buffer.id (same pattern as TMEM). Allocs without a
  // buffer.id attr get a fresh synthetic id so they remain singleton
  // groups. Members within a shared buffer.id bucket whose
  // [offset, offset+extent) intervals overlap union into one resourceKey.
  llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> localBuckets;
  funcOp.walk([&](LocalAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    if (!isLocalSemaphoreBackingType(cast<MemDescType>(allocOp.getType())))
      return;
    std::optional<int64_t> id = getBufferId(allocOp);
    int64_t key = id.value_or(nextSyntheticId++);
    localBuckets[key].push_back(allocOp);
  });
  for (auto &[id, allocs] : localBuckets)
    groups.push_back(
        makeGroup<LocalAllocOp>(MemorySpaceKind::Local, id, allocs));

  return groups;
}

// ---------------------------------------------------------------------------
// Access-event collection (v4 §Access Events). Walks the function in
// program order; for each terminal access op produces an AccessEvent with
// per-member touches.
// ---------------------------------------------------------------------------

static LogicalResult collectEvents(BufferGroup &group, triton::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (isSemaphoreBackingAlloc(op)) return WalkResult::advance();

    if (failed(addAlias(group, op))) return WalkResult::interrupt();
    if (isSupportedAliasOp(op)) return WalkResult::advance();

    AccessEvent event;
    event.op = op;
    event.owner = getPartitionId(op);
    event.tagSourceOp = event.owner ? getTagSourceOp(op) : nullptr;

    if (group.isTmem()) {
      if (auto allocOp = dyn_cast<TMEMAllocOp>(op)) {
        auto alias = lookupAlias(group, allocOp.getResult());
        if (succeeded(alias) && allocOp.getSrc()) {
          event.sourcefulAllocStore = true;
          addTouch(group, event, allocOp.getResult(), AccessEffect::Write);
        }
      } else if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
        addTouch(group, event, loadOp.getSrc(), AccessEffect::Read);
      } else if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
        addTouch(group, event, storeOp.getDst(), AccessEffect::Write);
      } else if (auto mma = dyn_cast<MMAv5OpInterface>(op)) {
        // v4: mma is always W on its accumulator regardless of
        // useAccumulator. The read-side semantics inside mma are handled
        // by single-partition program order; for cross-owner sync only
        // the overwrite matters.
        bool accumulatorImmediatelyOverwritten =
            isAccumulatorImmediatelyOverwritten(group, mma);
        for (Value operand : op->getOperands()) {
          auto alias = lookupAlias(group, operand);
          if (failed(alias)) continue;
          if (operand == mma.getAccumulator() && accumulatorImmediatelyOverwritten)
            continue;
          AccessEffect effect = operand == mma.getAccumulator()
                                    ? AccessEffect::Write
                                    : AccessEffect::Read;
          addTouch(group, event, operand, effect);
        }
      }
    } else {
      if (auto allocOp = dyn_cast<LocalAllocOp>(op)) {
        auto alias = lookupAlias(group, allocOp.getResult());
        if (succeeded(alias) && allocOp.getSrc()) {
          event.sourcefulAllocStore = true;
          addTouch(group, event, allocOp.getResult(), AccessEffect::Write);
        }
      } else if (auto storeOp = dyn_cast<LocalStoreOp>(op)) {
        addTouch(group, event, storeOp.getDst(), AccessEffect::Write);
      } else if (auto loadOp = dyn_cast<LocalLoadOp>(op)) {
        addTouch(group, event, loadOp.getSrc(), AccessEffect::Read);
      } else if (auto descLoad =
                     dyn_cast<triton::nvws::DescriptorLoadOp>(op)) {
        addTouch(group, event, descLoad.getResult(), AccessEffect::Write);
      } else if (auto descGather =
                     dyn_cast<triton::nvws::DescriptorGatherOp>(op)) {
        addTouch(group, event, descGather.getResult(), AccessEffect::Write);
      } else {
        for (Value operand : op->getOperands())
          addTouch(group, event, operand, AccessEffect::Read);
      }
    }

    if (!event.touches.empty()) group.events.push_back(std::move(event));
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

// ---------------------------------------------------------------------------
// Helpers used by the planner.
// ---------------------------------------------------------------------------

namespace {

// Recursive region-ownership planner (v4 §Structured Region Ownership).
// One Planner per ResourcePlan.
struct Planner {
  triton::FuncOp funcOp;
  ResourcePlan &plan;
  const DenseMap<Operation *, unsigned> &rank;
  // Events touching this resource, sorted by program order.
  SmallVector<Operation *> orderedEventOps;

  Planner(triton::FuncOp f, ResourcePlan &p,
          const DenseMap<Operation *, unsigned> &r)
      : funcOp(f), plan(p), rank(r) {}

  // First in-scope event (by program order) anywhere inside `region`.
  // v4 §Debug DAG Dumps scope-barrier rule: an event is in-scope for the
  // body region of a WS-tagged scf.for, but not for ancestor regions
  // outside that for-loop.
  std::optional<PartitionId> firstEventOwnerIn(Region &region) {
    Operation *foundOp = nullptr;
    region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (plan.useOwner.find(op) == plan.useOwner.end())
        return WalkResult::advance();
      Operation *tagSource = plan.useTagSource.lookup(op);
      if (!isEventInScopeForRegion(tagSource, op, &region))
        return WalkResult::advance();
      foundOp = op;
      return WalkResult::interrupt();
    });
    if (!foundOp) return std::nullopt;
    return plan.useOwner.find(foundOp)->second;
  }

  // Owner of next in-scope event for `contextRegion`, in program order
  // after `op`'s subtree. Used to choose the branchOwner for an scf.if
  // sitting inside `contextRegion`.
  std::optional<PartitionId> nextEventOwnerAfter(Operation *op,
                                                 Region &contextRegion) {
    unsigned cutoff = maxRankInSubtree(op, rank);
    for (Operation *evOp : orderedEventOps) {
      if (rank.lookup(evOp) <= cutoff) continue;
      Operation *tagSource = plan.useTagSource.lookup(evOp);
      if (!isEventInScopeForRegion(tagSource, evOp, &contextRegion))
        continue;
      return plan.useOwner.find(evOp)->second;
    }
    return std::nullopt;
  }

  // Plan `region`. Returns the chosen exit owner. For loop bodies
  // (isLoopBody=true) entry == exit == carried owner.
  std::optional<PartitionId> planRegion(Region &region,
                                        std::optional<PartitionId> entry,
                                        bool isLoopBody = false);
};

std::optional<PartitionId>
Planner::planRegion(Region &region, std::optional<PartitionId> entry,
                    bool isLoopBody) {
  RegionOwnership rec;
  rec.entry = entry;
  rec.carried = isLoopBody;
  bool subtreeHasEvents = false;

  std::optional<PartitionId> current = entry;
  if (isLoopBody) {
    // Loop-body carried owner: pick the first use inside the body. The
    // body's natural last use may differ from this; a handoff back to
    // `current` before scf.yield is required (raw-sync edge, commit 3).
    auto firstInside = firstEventOwnerIn(region);
    if (firstInside) current = firstInside;
    rec.entry = current;
  }

  for (Block &block : region) {
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // v4: scf.if branch owner = post-if next-use in this region;
        // filtered to events in-scope for *this* region (the one
        // containing the scf.if).
        auto postIf = nextEventOwnerAfter(&op, region);
        std::optional<PartitionId> branchOwner = postIf ? postIf : current;

        planRegion(ifOp.getThenRegion(), branchOwner,
                   /*isLoopBody=*/false);
        plan.regionOwners[&ifOp.getThenRegion()].entry = branchOwner;
        plan.regionOwners[&ifOp.getThenRegion()].exit = branchOwner;

        // Always emit a record for the else region (even if empty).
        RegionOwnership elseRec{branchOwner, branchOwner, /*carried=*/false,
                                /*hasEventsInSubtree=*/false};
        plan.regionOwners[&ifOp.getElseRegion()] = elseRec;
        if (!ifOp.getElseRegion().empty()) {
          planRegion(ifOp.getElseRegion(), branchOwner,
                     /*isLoopBody=*/false);
          plan.regionOwners[&ifOp.getElseRegion()].entry = branchOwner;
          plan.regionOwners[&ifOp.getElseRegion()].exit = branchOwner;
        }
        bool thenHas =
            plan.regionOwners[&ifOp.getThenRegion()].hasEventsInSubtree;
        bool elseHas =
            plan.regionOwners[&ifOp.getElseRegion()].hasEventsInSubtree;
        if (thenHas || elseHas) {
          subtreeHasEvents = true;
          current = branchOwner;
        }
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        planRegion(forOp.getRegion(), current, /*isLoopBody=*/true);
        RegionOwnership bodyRec =
            plan.regionOwners.lookup(&forOp.getRegion());
        if (bodyRec.hasEventsInSubtree) {
          subtreeHasEvents = true;
          // v4 scope-barrier rule: a WS-tagged scf.for is a scope
          // boundary. Its body owners are anchored to the for-loop and
          // do not propagate to the parent region. A plain scf.for
          // shares its parent's scope so its body exit owner is
          // in-scope for the parent and may propagate.
          if (!hasWarpSpecializeTag(&op))
            current = bodyRec.exit;
        }
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        if (isLoopBody)
          plan.yieldOwner[yieldOp.getOperation()] = rec.entry;
        continue;
      }
      auto useIt = plan.useOwner.find(&op);
      if (useIt != plan.useOwner.end()) {
        subtreeHasEvents = true;
        // Only update `current` when this event is in scope for the
        // region we're planning. Out-of-scope events still count for
        // transitive presence (subtreeHasEvents) but do not propagate
        // their owner across the WS scope barrier.
        Operation *tagSource = plan.useTagSource.lookup(&op);
        if (isEventInScopeForRegion(tagSource, &op, &region))
          current = useIt->second;
      }
    }
  }

  rec.exit = isLoopBody ? rec.entry : current;
  rec.hasEventsInSubtree = subtreeHasEvents;
  plan.regionOwners[&region] = rec;
  return rec.exit;
}

} // namespace

static ResourcePlan
planResource(triton::FuncOp funcOp, unsigned groupIdx, BufferGroup &group,
             int64_t resourceKey,
             const DenseMap<Operation *, unsigned> &rank) {
  ResourcePlan plan;
  plan.resource = {group.logicalId, resourceKey};
  plan.groupIdx = groupIdx;
  for (auto [idx, member] : llvm::enumerate(group.members))
    if (member.resourceKey == resourceKey)
      plan.memberIndices.push_back(static_cast<unsigned>(idx));

  for (AccessEvent &event : group.events) {
    bool found = false;
    for (auto [tIdx, touch] : llvm::enumerate(event.touches)) {
      if (touch.resourceKey == resourceKey) {
        if (!found) {
          std::optional<PartitionId> owner = getPartitionId(event.op);
          plan.useOwner[event.op] = owner;
          plan.useTagSource[event.op] =
              owner ? getTagSourceOp(event.op) : nullptr;
          plan.useTouchIdx[event.op] = static_cast<unsigned>(tIdx);
          found = true;
        }
        plan.useTouchIdxs[event.op].push_back(static_cast<unsigned>(tIdx));
      }
    }
  }

  Planner planner(funcOp, plan, rank);
  for (auto &kv : plan.useOwner)
    planner.orderedEventOps.push_back(kv.first);
  llvm::sort(planner.orderedEventOps, [&](Operation *a, Operation *b) {
    return rank.lookup(a) < rank.lookup(b);
  });

  // v4: the function region is never annotated, so we don't seed it with
  // a precomputed entry owner. The planner threads in-scope events
  // (root + intrinsic-tagged) through `current` if any exist; otherwise
  // function-region entry/exit stay nullopt.
  planner.planRegion(funcOp.getBody(), /*entry=*/std::nullopt,
                     /*isLoopBody=*/false);
  return plan;
}

// ---------------------------------------------------------------------------
// v4 §Dependency Edges / §Raw DAG Edge Model (commit 3, post-revision).
// Per-edge counter model:
//   - one counter per cross-owner edge (initial 0)
//   - source partition releases (counter += 1) after its op
//   - destination partition acquires (counter -= 1, blocks if 0) before
//     its op
//   - no FULL/EMPTY, no ready/done/handoff kind, no S_init
//   - sequential name "S_e<N>" per resource
//
// Edges are derived from version tracking applied to the augmented
// program-order stream (real accesses + virtual ENTER/YIELD rows + region
// op partitions). Concurrent readers of the same version produce no edge
// between themselves. Loop-carry close edges emerge naturally as
// cross-owner edges whose destination is a YIELD virtual row.
// ---------------------------------------------------------------------------

static bool isExplicitOffsetSourcefulTmemSelfContained(BufferGroup &group,
                                                       int64_t resourceKey) {
  if (!group.isTmem())
    return false;
  bool sawMember = false;
  for (BufferMember &member : group.members) {
    if (member.resourceKey != resourceKey)
      continue;
    sawMember = true;
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getSrc() ||
        !member.allocOp->hasAttr(kBufferOffsetAttrName))
      return false;
  }
  if (!sawMember)
    return false;

  for (const AccessEvent &event : group.events) {
    SmallVector<const AccessTouch *, 2> touches;
    collectTouchesForResource(event, resourceKey, touches);
    if (touches.empty())
      continue;
    if (!isa<TMEMAllocOp, TMEMLoadOp>(event.op))
      return false;
    for (const AccessTouch *touch : touches) {
      if (touch->memberIdx >= group.members.size())
        return false;
      auto allocOp =
          dyn_cast<TMEMAllocOp>(group.members[touch->memberIdx].allocOp);
      if (!allocOp || !sameOwner(event.owner, getPartitionId(allocOp)))
        return false;
      if (isa<TMEMAllocOp>(event.op) && !touchWrites(*touch))
        return false;
      if (isa<TMEMLoadOp>(event.op) && !touchReads(*touch))
        return false;
    }
  }
  bool sawOwner = false;
  int previousPartition = -1;
  for (const AccessEvent &event : group.events) {
    if (!eventTouchesResource(event, resourceKey) || !isa<TMEMAllocOp>(event.op))
      continue;
    if (!event.owner || (!sawOwner && event.owner->first != 0) ||
        (sawOwner && event.owner->first < previousPartition))
      return false;
    sawOwner = true;
    previousPartition = event.owner->first;
  }
  return true;
}

static std::string makeEdgeName(unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "S" << serial;
  return s;
}

static RegionOwnership ownershipForAnchor(Operation *op, Region *yieldRegion,
                                          const ResourcePlan &rp) {
  if (yieldRegion) {
    auto it = rp.regionOwners.find(yieldRegion);
    return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
  }
  if (!op)
    return {};
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto it = rp.regionOwners.find(&forOp.getRegion());
    return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto thenIt = rp.regionOwners.find(&ifOp.getThenRegion());
    if (thenIt != rp.regionOwners.end() && thenIt->second.hasEventsInSubtree)
      return thenIt->second;
    auto elseIt = rp.regionOwners.find(&ifOp.getElseRegion());
    return elseIt == rp.regionOwners.end() ? RegionOwnership{} : elseIt->second;
  }
  Region *parentRegion = op->getParentRegion();
  auto it = rp.regionOwners.find(parentRegion);
  return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
}

// Record an edge anchored at its destination. Both the release and the
// acquire render right before the destination's row at the destination's
// tree depth.
static void recordEdge(SyncPlan &sp, SyncEdge edge) {
  unsigned idx = sp.edges.size();
  if (edge.dstOp)
    sp.beforeOp[edge.dstOp].push_back(idx);
  else if (edge.dstYieldRegion)
    sp.beforeYield[edge.dstYieldRegion].push_back(idx);
  sp.edges.push_back(std::move(edge));
}

// Build the raw-sync DAG via a single unified walk of the OWNERSHIP-DAG.
// State is (writer, readers). Walk every row in program order:
//   - R{P}: if writer cross-partition and P not in readers, emit (writer→P).
//           Add P to readers.
//   - W{P}: close cross-partition readers; if no readers and cross writer,
//           emit handoff; set writer=P, clear readers.
//   - Structural row (scf.for-op / scf.if-op / YIELD body): same as W's
//     close+handoff logic, EXCEPT (a) writer is updated only when an edge
//     actually fires, (b) handoff is skipped when no real access exists
//     downstream (release-into-void suppression).
//
// scf.if: each branch walks from a snapshot; planner guarantees both
// branches converge on the same partition at exit.
//
// This unified rule replaces the prior bifurcated walker+loop-carry pair,
// and is sound by direct trace verification across all 86 commit3 OWNERSHIP
// DAGs.
static SyncPlan buildSyncPlan(BufferGroup &group, const ResourcePlan &rp,
                              triton::FuncOp funcOp) {
  SyncPlan sp;
  sp.resource = rp.resource;
  sp.groupIdx = rp.groupIdx;
  sp.memberIndices = rp.memberIndices;

  // Collect access ops for this resource.
  DenseMap<Operation *, const AccessEvent *> opToEvent;
  for (const AccessEvent &e : group.events)
    if (eventTouchesResource(e, rp.resource.second)) {
      sp.accessOps.insert(e.op);
      opToEvent[e.op] = &e;
    }

  // Program-order rank per op (depth-first pre-order over the function).
  DenseMap<Operation *, unsigned> opRank;
  {
    unsigned ctr = 0;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) { opRank[op] = ctr++; });
  }
  // Max rank of any access op — used to answer "any access after this row?".
  unsigned maxAccessRank = 0;
  bool anyAccess = false;
  for (Operation *op : sp.accessOps) {
    auto it = opRank.find(op);
    if (it == opRank.end()) continue;
    if (!anyAccess || it->second > maxAccessRank) {
      maxAccessRank = it->second;
      anyAccess = true;
    }
  }
  auto hasDownstreamAccess = [&](unsigned cutoff) {
    return anyAccess && maxAccessRank > cutoff;
  };

  // Owner = std::optional<PartitionId>: nullopt represents "root" (an
  // event outside any WS scope, still a valid producer/consumer).
  // State.writerSet distinguishes "no writer has happened yet" (false)
  // from "writer is root or some partition" (true).
  //
  // writerOp / writerYieldRegion identify where the writer became
  // `writer` (a real W access op, a structural row's op for region
  // transitions, or a yield region for body-yield closes). writerEpoch
  // is bumped every time the writer is updated, so edges that share a
  // (srcOwner, srcEpoch) provably came from the same producer event.
  using Owner = std::optional<PartitionId>;
  struct State {
    bool writerSet = false;
    Owner writer;
    Operation *writerOp = nullptr;
    Region *writerYieldRegion = nullptr;
    unsigned writerEpoch = 0;
    // Parallel arrays indexed together: `readers[i]` is the partition,
    // `readerOps[i]` / `readerYields[i]` is where its last access for
    // this resource happened (used as the SRC anchor when a close edge
    // fires from this reader).
    SmallVector<Owner, 4> readers;
    SmallVector<Operation *, 4> readerOps;
    SmallVector<Region *, 4> readerYields;
    SmallVector<unsigned, 4> readerEpochs;
  };

  auto containsReader = [](const State &s, const Owner &p) {
    return llvm::any_of(s.readers, [&](const Owner &q) { return sameOwner(p, q); });
  };

  unsigned serial = 0;
  unsigned writerEpochCtr = 0;

  auto readerIdx = [&](const State &s, const Owner &p) -> int {
    for (unsigned i = 0; i < s.readers.size(); ++i)
      if (sameOwner(s.readers[i], p)) return static_cast<int>(i);
    return -1;
  };

  auto populateEdgeInfo = [&](SyncEdge &edge, SyncEdgeKind kind,
                              unsigned producedEpoch, Owner preOwner,
                              Owner postOwner) {
    edge.kind = kind;
    edge.producedVersion = {rp.resource.second, producedEpoch};
    edge.preOwner = preOwner;
    edge.postOwner = postOwner;
    edge.srcRegionOwner =
        ownershipForAnchor(edge.srcOp, edge.srcYieldRegion, rp);
    edge.dstRegionOwner =
        ownershipForAnchor(edge.dstOp, edge.dstYieldRegion, rp);
    edge.releaseOutsideControlFlow =
        edge.srcOp && isa<scf::ForOp, scf::IfOp>(edge.srcOp);
    edge.acquireOutsideControlFlow =
        edge.dstOp && isa<scf::ForOp, scf::IfOp>(edge.dstOp);
    edge.carrierChoice = CarrierTokenChoice::Destination;
    if (edge.srcOp)
      edge.releaseStageCluster = getStageCluster(edge.srcOp);
    if (edge.dstOp)
      edge.acquireStageCluster = getStageCluster(edge.dstOp);
    edge.asyncPayload = getAsyncPayload(edge.srcOp);
    if (edge.srcOp) {
      auto it = opToEvent.find(edge.srcOp);
      if (it != opToEvent.end()) {
        SmallVector<const AccessTouch *, 4> touches;
        collectTouchesForResource(*it->second, rp.resource.second, touches);
        for (const AccessTouch *touch : touches)
          edge.touches.push_back(*touch);
      }
    }
    if (edge.dstOp) {
      auto it = opToEvent.find(edge.dstOp);
      if (it != opToEvent.end()) {
        SmallVector<const AccessTouch *, 4> touches;
        collectTouchesForResource(*it->second, rp.resource.second, touches);
        for (const AccessTouch *touch : touches)
          edge.touches.push_back(*touch);
      }
    }
  };

  // Apply the "close cross-readers + optional handoff" rule. Used by W
  // (isW=true) and structural rows (isW=false). Returns true if any edge
  // was emitted at this row.
  // `consumerGuaranteed`: the destination owner P is reached by a use that is
  // guaranteed to exist even when no access follows this row in static program
  // order — specifically a loop-carry back-edge (this row's YIELD flows to the
  // loop's next-iteration ENTER, whose body has events). An owner change into
  // such a carry is a real handoff and must emit, so it bypasses the
  // static-downstream "release-into-void" suppression below.
  auto emitClose = [&](State &state, const Owner &P, Operation *anchorOp,
                       Region *anchorYieldRegion, bool isW,
                       unsigned anchorRank,
                       bool consumerGuaranteed = false) -> bool {
    SmallVector<unsigned, 4> crossIdxs;
    for (unsigned i = 0; i < state.readers.size(); ++i)
      if (!sameOwner(state.readers[i], P)) crossIdxs.push_back(i);
    bool fired = false;
    if (!crossIdxs.empty()) {
      for (unsigned i : crossIdxs) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        bool sameOwnerReadAfterWrite =
            isW && state.writerSet && sameOwner(state.readers[i], state.writer);
        edge.srcOwner =
            sameOwnerReadAfterWrite ? state.writer : state.readers[i];
        edge.dstOwner = P;
        edge.srcOp = state.readerOps[i];
        edge.srcYieldRegion = state.readerYields[i];
        if (sameOwnerReadAfterWrite) {
          edge.srcEpoch = state.writerEpoch;
        } else {
          // Each reader's last-access is itself a fresh source event for
          // Combine A grouping purposes; assign a unique epoch per
          // contributing reader so cross-reader closes are not falsely
          // collapsed into a ReadyFanout.
          edge.srcEpoch = ++writerEpochCtr;
        }
        if (anchorOp)
          edge.dstOp = anchorOp;
        else
          edge.dstYieldRegion = anchorYieldRegion;
        populateEdgeInfo(edge,
                         sameOwnerReadAfterWrite ? SyncEdgeKind::Handoff
                                                 : SyncEdgeKind::Done,
                         sameOwnerReadAfterWrite ? state.writerEpoch
                                                 : state.readerEpochs[i],
                         edge.srcOwner, P);
        edge.forceFullSemaphore = sameOwnerReadAfterWrite;
        recordEdge(sp, edge);
      }
      state.readers.clear();
      state.readerOps.clear();
      state.readerYields.clear();
      state.readerEpochs.clear();
      fired = true;
    } else if (state.readers.empty() && state.writerSet &&
               !sameOwner(state.writer, P)) {
      // Handoff: W always fires; a structural row fires if a real access
      // exists downstream in program order (release-into-void suppression) OR
      // the destination is reached via a guaranteed loop-carry consumer (op2
      // is the next iteration, at a lower static rank).
      if (isW || consumerGuaranteed || hasDownstreamAccess(anchorRank)) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        edge.srcOwner = state.writer;
        edge.dstOwner = P;
        edge.srcOp = state.writerOp;
        edge.srcYieldRegion = state.writerYieldRegion;
        edge.srcEpoch = state.writerEpoch;
        if (anchorOp)
          edge.dstOp = anchorOp;
        else
          edge.dstYieldRegion = anchorYieldRegion;
        populateEdgeInfo(edge, SyncEdgeKind::Handoff, state.writerEpoch,
                         state.writer, P);
        recordEdge(sp, edge);
        fired = true;
      }
    }
    if (isW) {
      state.writerSet = true;
      state.writer = P;
      state.writerOp = anchorOp;
      state.writerYieldRegion = anchorYieldRegion;
      state.writerEpoch = ++writerEpochCtr;
      state.readers.clear();
      state.readerOps.clear();
      state.readerYields.clear();
      state.readerEpochs.clear();
    } else if (fired) {
      // Structural row only promotes itself to writer when it actually
      // settled a transition.
      state.writerSet = true;
      state.writer = P;
      state.writerOp = anchorOp;
      state.writerYieldRegion = anchorYieldRegion;
      state.writerEpoch = ++writerEpochCtr;
    }
    return fired;
  };

  auto emitRead = [&](State &state, const Owner &P, Operation *anchorOp) {
    if (state.writerSet && !sameOwner(state.writer, P) &&
        !containsReader(state, P)) {
      SyncEdge edge;
      edge.name = makeEdgeName(serial++);
      edge.srcOwner = state.writer;
      edge.dstOwner = P;
      edge.srcOp = state.writerOp;
      edge.srcYieldRegion = state.writerYieldRegion;
      edge.srcEpoch = state.writerEpoch;
      edge.dstOp = anchorOp;
      populateEdgeInfo(edge, SyncEdgeKind::Ready, state.writerEpoch,
                       state.writer, P);
      recordEdge(sp, edge);
    }
    int idx = readerIdx(state, P);
    unsigned readEpoch = state.writerSet ? state.writerEpoch : ++writerEpochCtr;
    if (idx < 0) {
      state.readers.push_back(P);
      state.readerOps.push_back(anchorOp);
      state.readerYields.push_back(nullptr);
      state.readerEpochs.push_back(readEpoch);
    } else {
      state.readerOps[idx] = anchorOp;
      state.readerYields[idx] = nullptr;
      state.readerEpochs[idx] = readEpoch;
    }
  };

  std::function<void(Region &, State &)> walkRegion;
  walkRegion = [&](Region &region, State &state) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          auto regIt = rp.regionOwners.find(&forOp.getRegion());
          if (regIt == rp.regionOwners.end() ||
              !regIt->second.hasEventsInSubtree)
            continue; // No events in subtree — skip entirely.
          Owner pp = regIt->second.entry; // may be nullopt (== root partition)
          unsigned rank = opRank.lookup(&op);
          // v4 §Region-Boundary Edges: a root/external producer (nullopt
          // effective owner) entering a warp-specialized loop is
          // *carrier-inherit*, not a handoff. Region entry already orders
          // the producer's writes before every partition, so the loop's
          // entry partition adopts the producer's state with no semaphore
          // edge. Applies uniformly to true-root and annotated-external
          // (both have nullopt effective owner). Without this, the
          // root->{P} edge would emit a bogus handoff (and, for a loop
          // that re-acquires per iteration against a single pre-loop
          // release, deadlock).
          if (hasWarpSpecializeTag(&op) && state.writerSet && !state.writer &&
              pp.has_value()) {
            state.writer = pp;
            state.writerOp = &op;
            state.writerYieldRegion = nullptr;
            state.writerEpoch = ++writerEpochCtr;
            state.readers.clear();
            state.readerOps.clear();
            state.readerYields.clear();
            state.readerEpochs.clear();
          } else {
            // Transition at the scf.for header row.
            emitClose(state, pp, &op, /*anchorYieldRegion=*/nullptr,
                      /*isW=*/false, rank);
          }
          walkRegion(forOp.getRegion(), state);
          // Transition at the body's YIELD row (loop-carry close). op2 is the
          // loop's next-iteration ENTER (body has events by the precondition
          // above), so an owner change here is a real carry handoff and must
          // emit even with nothing after the loop in static order.
          Operation *yieldOp = forOp.getRegion().front().getTerminator();
          unsigned yieldRank = opRank.lookup(yieldOp);
          emitClose(state, pp, /*anchorOp=*/nullptr, &forOp.getRegion(),
                    /*isW=*/false, yieldRank, /*consumerGuaranteed=*/true);
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          // The scf.if's partition is the entry partition of whichever
          // branch has events. May be nullopt (== root partition).
          auto thenIt = rp.regionOwners.find(&ifOp.getThenRegion());
          auto elseIt = rp.regionOwners.find(&ifOp.getElseRegion());
          bool thenHas = thenIt != rp.regionOwners.end() &&
                         thenIt->second.hasEventsInSubtree;
          bool elseHas = elseIt != rp.regionOwners.end() &&
                         elseIt->second.hasEventsInSubtree;
          if (!thenHas && !elseHas) continue;
          Owner pp = thenHas ? thenIt->second.entry : elseIt->second.entry;
          unsigned rank = opRank.lookup(&op);
          // Transition at the scf.if header row.
          emitClose(state, pp, &op, /*anchorYieldRegion=*/nullptr,
                    /*isW=*/false, rank);
          // Walk each branch from a snapshot of the post-header state.
          State snap = state;
          State thenExit = snap;
          walkRegion(ifOp.getThenRegion(), thenExit);
          {
            Operation *thenYield =
                ifOp.getThenRegion().front().getTerminator();
            unsigned ry = opRank.lookup(thenYield);
            emitClose(thenExit, pp, /*anchorOp=*/nullptr,
                      &ifOp.getThenRegion(), /*isW=*/false, ry);
          }
          State elseExit = snap;
          bool hasElse = !ifOp.getElseRegion().empty();
          if (hasElse) {
            walkRegion(ifOp.getElseRegion(), elseExit);
            Operation *elseYield =
                ifOp.getElseRegion().front().getTerminator();
            unsigned ry = opRank.lookup(elseYield);
            emitClose(elseExit, pp, /*anchorOp=*/nullptr,
                      &ifOp.getElseRegion(), /*isW=*/false, ry);
          }
          // Planner enforces branches converge. Use last walked branch.
          state = hasElse ? elseExit : thenExit;
          continue;
        }
        if (isa<scf::YieldOp>(op)) continue;
        if (!sp.accessOps.contains(&op)) continue;
        auto evIt = opToEvent.find(&op);
        if (evIt == opToEvent.end()) continue;
        const AccessEvent *ev = evIt->second;
        // ev->owner is nullopt for "root" (outside any WS scope); still
        // a valid participating partition.
        const Owner &P = ev->owner;
        bool reads = eventConsumes(*ev, rp.resource.second);
        bool writes = eventProduces(*ev, rp.resource.second);
        unsigned rank = opRank.lookup(&op);
        // For RMW (both reads and writes): apply R first (emits RAW edge
        // from prior writer), then W (closes cross-readers, becomes new
        // writer). For W-only: skip R. For R-only: skip W.
        if (reads) emitRead(state, P, &op);
        if (writes)
          emitClose(state, P, &op, /*anchorYieldRegion=*/nullptr,
                    /*isW=*/true, rank);
      }
    }
  };

  State state;
  walkRegion(funcOp.getBody(), state);

  // -------- Initial writable permit (RAW-SYNC-DAG dump only). ----------
  // v4 plan "Every access begins with an acquire": the first access of a
  // semaphore-managed (>=2 distinct owners) resource acquires the initial
  // writable permit. Cyclic chains reuse the loop-carry back edge into the
  // loop's entry owner; acyclic chains mint a distinct permit released at the
  // terminal (top-level) consumer. This only annotates the dump; the emit is
  // unchanged. Single-owner resources stay bare (no semaphore).
  {
    SmallVector<Owner, 4> distinctOwners;
    Operation *firstAccess = nullptr, *lastAccess = nullptr;
    unsigned firstRank = 0, lastRank = 0;
    for (auto &kv : opToEvent) {
      Operation *op = kv.first;
      const Owner &o = kv.second->owner;
      if (!llvm::any_of(distinctOwners,
                        [&](const Owner &q) { return sameOwner(o, q); }))
        distinctOwners.push_back(o);
      unsigned r = opRank.lookup(op);
      if (!firstAccess || r < firstRank) { firstAccess = op; firstRank = r; }
      if (!lastAccess || r > lastRank) { lastAccess = op; lastRank = r; }
    }
    if (distinctOwners.size() >= 2 && firstAccess) {
      scf::ForOp wsLoop;
      for (Operation *p = firstAccess->getParentOp(); p; p = p->getParentOp())
        if (auto f = dyn_cast<scf::ForOp>(p))
          if (hasWarpSpecializeTag(f)) wsLoop = f;
      Operation *anchorOp = wsLoop ? wsLoop.getOperation() : firstAccess;
      scf::ForOp cycleLoop = wsLoop;
      if (!cycleLoop)
        for (Operation *op : sp.accessOps) {
          scf::ForOp outer;
          for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
            if (auto f = dyn_cast<scf::ForOp>(p))
              if (hasWarpSpecializeTag(f)) outer = f;
          if (outer) { cycleLoop = outer; break; }
        }
      std::optional<PartitionId> loopOwner =
          cycleLoop ? regionOpPartition(cycleLoop.getOperation(), rp)
                    : std::nullopt;
      std::string reuseName;
      int reuseIdx = -1;
      if (cycleLoop && loopOwner)
        for (auto [ei, e] : llvm::enumerate(sp.edges))
          if (e.dstYieldRegion == &cycleLoop.getRegion() &&
              sameOwner(e.dstOwner, loopOwner)) {
            reuseName = e.name;
            reuseIdx = (int)ei;
            break;
          }
      sp.initialPermitBeforeOp = anchorOp;
      if (!reuseName.empty()) {
        sp.initialPermitName = reuseName; // cyclic: reuse loop-carry back edge
        sp.initialPermitEdgeIdx = reuseIdx;
      } else {
        sp.initialPermitName = makeEdgeName((unsigned)sp.edges.size()); // mint
        if (lastAccess && lastAccess->getParentOp() == funcOp.getOperation())
          sp.initialPermitReleaseAfterOp = lastAccess; // acyclic terminal
      }
    }
  }

  // ---- Semaphore-identity unification (commit 3) ----
  // Reaching-acquire rule: an access's buffer permit is provided by the nearest
  // acquire on each control-flow path. The loop back-edge means a loop's first
  // carried access is reached by BOTH the acquire entering the loop and the
  // carried (last) acquire of the body — so those semaphores MUST be the same.
  // Walk forward (modelling each loop body's back-edge) and union the semaphores
  // feeding each access. Element ids: 0..edges-1 = edges, edges.size() = seed.
  // No op-kind / read-vs-write / parity assumption.
  {
    unsigned seedId = static_cast<unsigned>(sp.edges.size());
    sp.semaRep.resize(seedId + 1);
    for (unsigned i = 0; i <= seedId; ++i)
      sp.semaRep[i] = i;
    auto unite = [&](unsigned a, unsigned b) {
      a = sp.semaFind(a);
      b = sp.semaFind(b);
      if (a != b)
        sp.semaRep[std::max(a, b)] = std::min(a, b);
    };
    DenseMap<Operation *, SmallVector<unsigned, 2>> acqByOp;
    DenseMap<Region *, SmallVector<unsigned, 2>> acqByYield;
    for (auto [i, e] : llvm::enumerate(sp.edges)) {
      if (e.dstOp)
        acqByOp[e.dstOp].push_back(static_cast<unsigned>(i));
      else if (e.dstYieldRegion)
        acqByYield[e.dstYieldRegion].push_back(static_cast<unsigned>(i));
    }
    // Edges acquired at the same anchor (op or region-yield) are one acquire
    // for that access — unite them and return the canonical id.
    auto uniteAnchor = [&](ArrayRef<unsigned> es) -> unsigned {
      for (unsigned e : es)
        unite(es.front(), e);
      return es.front();
    };
    // Forward walk threading the "current permit" (the most recent acquire). At
    // each loop, the iter_arg carrying this resource has init = the permit
    // entering the loop and yield = the permit live at the body's end; the
    // loop's first carried access is therefore fed by BOTH, so they are the same
    // semaphore — unite(enterPermit, bodyExit). Returns the permit live at the
    // region's end.
    std::function<unsigned(Region &, unsigned)> walk =
        [&](Region &region, unsigned incoming) -> unsigned {
      unsigned current = incoming;
      for (Block &blk : region) {
        for (Operation &op : blk) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            bool carriesResource = false;
            forOp.getRegion().walk([&](Operation *o) {
              if (sp.accessOps.contains(o))
                carriesResource = true;
            });
            if (carriesResource) {
              unsigned enter = current;
              auto it = acqByOp.find(&op);
              if (it != acqByOp.end())
                enter = uniteAnchor(it->second);
              unsigned bodyExit = walk(forOp.getRegion(), enter);
              unite(enter, bodyExit); // carry: iter_arg init == yield
              current = bodyExit;
              continue;
            }
          }
          // scf.if: a guarded acquire still produces the permit live after the
          // branch. Descend into both regions with the same incoming permit and
          // unite their exits (they feed the one scf.if result / carried slot),
          // so a re-acquire nested in a branch is unified with the loop carry.
          if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
            unsigned thenExit = walk(ifOp.getThenRegion(), current);
            unsigned elseExit = current;
            if (!ifOp.getElseRegion().empty())
              elseExit = walk(ifOp.getElseRegion(), current);
            unite(thenExit, elseExit);
            current = sp.semaFind(thenExit);
            continue;
          }
          auto it = acqByOp.find(&op);
          if (it != acqByOp.end())
            current = uniteAnchor(it->second);
        }
      }
      auto yit = acqByYield.find(&region);
      if (yit != acqByYield.end())
        current = uniteAnchor(yit->second);
      return current;
    };
    walk(funcOp.getBody(), seedId);
  }
  return sp;
}

// ---------------------------------------------------------------------------
// v4 §Final Combine Subpass / §End-to-End examples — OPT-SYNC DAG.
// Classify each raw SyncEdge into one of:
//   - Singleton:   one src -> one dst (default, identical render to RAW)
//   - ReadyFanout: one producer event -> N consumer events (Combine A)
//   - DoneFanin:   N reader events -> one next-writer event   (Combine B)
//   - LinearChain: not detected here; LinearChain is a render-time
//                  optimization at emit-stage and need not change the
//                  dump structure. We emit it explicitly only when the
//                  pattern is unambiguous: a singleton handoff whose
//                  src/dst chain forms a linear sequence with no other
//                  fanout/fanin participants. For commit 4 we leave
//                  linear handoffs as Singletons; combine C is an emit
//                  optimization deferred to commit 5.
// ---------------------------------------------------------------------------

static bool edgeRequiresRelease(const SyncEdge &edge) {
  if (!edge.srcOwner)
    return false;
  if (edge.dstOwner && sameOwner(edge.srcOwner, edge.dstOwner))
    return false;
  return true;
}

template <typename AnchorT>
static void addPlannedRelease(
    DenseMap<AnchorT *, SmallVector<PlannedRelease, 2>> &anchors,
    AnchorT *anchor, const SyncPlan &sp, unsigned groupIdx,
    ArrayRef<unsigned> edgeIdxs) {
  if (!anchor)
    return;
  PlannedRelease action;
  action.groupIdx = groupIdx;
  for (unsigned edgeIdx : edgeIdxs) {
    if (edgeIdx >= sp.edges.size())
      continue;
    if (!edgeRequiresRelease(sp.edges[edgeIdx]))
      continue;
    action.edgeIdxs.push_back(edgeIdx);
  }
  if (action.edgeIdxs.empty())
    return;
  SmallVector<PlannedRelease, 2> &planned = anchors[anchor];
  if (!llvm::is_contained(planned, action))
    planned.push_back(std::move(action));
}

template <typename AnchorT>
static void addPlannedRelease(
    DenseMap<AnchorT *, SmallVector<PlannedRelease, 2>> &anchors,
    AnchorT *anchor, const SyncPlan &sp, unsigned groupIdx, unsigned edgeIdx) {
  addPlannedRelease(anchors, anchor, sp, groupIdx,
                    ArrayRef<unsigned>(&edgeIdx, 1));
}

static std::string makeGroupName(unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "S_g" << serial;
  return s;
}

static bool isIfYieldRegion(Region *region);
static bool opReadsOnlyResource(Operation *op, BufferGroup &group,
                                int64_t resourceKey);
static Operation *findTerminalReadReleaseAnchor(Operation *op,
                                                const SyncPlan &sp,
                                                BufferGroup &group,
                                                int64_t resourceKey);

static bool collectLinearChainEdges(const SyncPlan &sp, ArrayRef<bool> claimed,
                                    SmallVectorImpl<unsigned> &chain) {
  chain.clear();
  for (unsigned i = 0; i < sp.edges.size(); ++i)
    if (!claimed[i])
      chain.push_back(i);
  if (chain.size() < 2) return false;

  SmallVector<ReadyFanoutKey, 4> srcKeys;
  SmallVector<DoneFaninKey, 4> dstKeys;
  for (auto [pos, idx] : llvm::enumerate(chain)) {
    const SyncEdge &e = sp.edges[idx];
    if ((!e.srcOp && !e.srcYieldRegion) || (!e.dstOp && !e.dstYieldRegion))
      return false;
    if (e.srcYieldRegion)
      return false;
    if (e.dstYieldRegion && !isIfYieldRegion(e.dstYieldRegion) &&
        (pos + 1 != chain.size() || e.kind != SyncEdgeKind::Done))
      return false;
    if (e.srcOp && isa<scf::ForOp, scf::IfOp>(e.srcOp))
      return false;
    ReadyFanoutKey srcKey{e.srcOwner, e.srcEpoch, e.srcOp, e.srcYieldRegion};
    if (llvm::is_contained(srcKeys, srcKey))
      return false;
    srcKeys.push_back(srcKey);
    DoneFaninKey dstKey{e.dstOwner, e.dstOp, e.dstYieldRegion};
    if (llvm::is_contained(dstKeys, dstKey))
      return false;
    dstKeys.push_back(dstKey);
  }

  for (unsigned i = 1; i < chain.size(); ++i) {
    const SyncEdge &prev = sp.edges[chain[i - 1]];
    const SyncEdge &cur = sp.edges[chain[i]];
    if (!sameOwner(prev.dstOwner, cur.srcOwner))
      return false;
  }

  return true;
}

static bool yieldedAccessTokenRequiresCarrier(Region *region,
                                              const SyncPlan &sp) {
  if (!region) return false;
  for (Block &block : *region) {
    auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
    if (!yieldOp) continue;
    for (Value operand : yieldOp.getOperands()) {
      if (!isa<AsyncTokenType>(operand.getType())) continue;
      auto result = dyn_cast<OpResult>(operand);
      if (result && sp.accessOps.contains(result.getOwner()))
        return true;
    }
  }
  return false;
}

static bool syncYieldRequiresCarrier(Region *region, const SyncPlan &sp) {
  if (!region) return false;
  Operation *parent = region->getParentOp();
  // A sync edge targeting a structured yield makes the semaphore token the
  // state that crosses the CFG boundary. Loops need it for re-entry even when
  // the source access did not already produce an async token in the input IR.
  if (isa_and_nonnull<scf::ForOp, scf::IfOp>(parent))
    return true;
  return yieldedAccessTokenRequiresCarrier(region, sp);
}

static Operation *findFirstAccessInRegion(Region &region,
                                          const OptSyncDag &dag) {
  Operation *firstAccess = nullptr;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (!dag.accessOps.contains(op))
      return WalkResult::advance();
    firstAccess = op;
    return WalkResult::interrupt();
  });
  return firstAccess;
}

static bool isIfYieldRegion(Region *region) {
  return region && isa<scf::IfOp>(region->getParentOp());
}

static Operation *findFirstAccessAfter(Operation *op, const OptSyncDag &dag) {
  if (!op || !op->getBlock())
    return nullptr;
  auto it = op->getIterator();
  for (++it; it != op->getBlock()->end(); ++it) {
    Operation *candidate = &*it;
    if (dag.accessOps.contains(candidate))
      return candidate;
    Operation *nestedAccess = nullptr;
    candidate->walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult {
      if (!dag.accessOps.contains(nested))
        return WalkResult::advance();
      nestedAccess = nested;
      return WalkResult::interrupt();
    });
    if (nestedAccess)
      return nestedAccess;
  }
  return nullptr;
}

static scf::ForOp getContainingWsFor(Operation *op) {
  if (!op)
    return {};
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    if (hasWarpSpecializeTag(forOp))
      return forOp;
  for (auto forOp = op->getParentOfType<scf::ForOp>(); forOp;
       forOp = forOp->getParentOfType<scf::ForOp>())
    if (hasWarpSpecializeTag(forOp))
      return forOp;
  return {};
}

static bool operationIsAttached(Operation *op) {
  return op && op->getBlock() && op->getBlock()->getParent();
}

static Operation *getIfBranchEntryAnchor(Operation *op) {
  if (!operationIsAttached(op))
    return op;
  Region *parentRegion = op->getBlock()->getParent();
  if (!parentRegion ||
      !isa_and_nonnull<scf::IfOp>(parentRegion->getParentOp()))
    return op;
  Operation *first = &op->getBlock()->front();
  if (isa<scf::YieldOp>(first))
    return op;
  return first;
}

static Operation *findTmemLoopExitReadForEdge(const SyncEdge &edge,
                                              const SyncPlan &sp,
                                              const OptSyncDag &dag,
                                              BufferGroup &group,
                                              int64_t resourceKey) {
  if (!group.isTmem() || !edge.dstYieldRegion || edge.dstOwner || !edge.srcOp)
    return nullptr;
  auto forOp =
      dyn_cast_or_null<scf::ForOp>(edge.dstYieldRegion->getParentOp());
  if (!forOp || !hasWarpSpecializeTag(forOp) ||
      !forOp->isProperAncestor(edge.srcOp))
    return nullptr;
  Operation *afterLoop = findFirstAccessAfter(forOp.getOperation(), dag);
  return findTerminalReadReleaseAnchor(afterLoop, sp, group, resourceKey);
}

static Operation *findFirstWriter(const SyncPlan &sp, BufferGroup &group,
                                  std::optional<PartitionId> &owner) {
  for (AccessEvent &event : group.events) {
    if (!sp.accessOps.contains(event.op))
      continue;
    if (!eventProduces(event, sp.resource.second))
      continue;
    owner = event.owner;
    return event.op;
  }
  return nullptr;
}

static scf::ForOp getSkippedInitialLoopCarrierFor(const SyncGroup &syncGroup,
                                                  const SyncPlan &sp,
                                                  BufferGroup &group) {
  if (!group.isTmem() || syncGroup.edgeIdxs.empty())
    return {};
  std::optional<PartitionId> firstWriterOwner;
  Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner);
  if (!firstWriter)
    return {};
  const SyncEdge &firstEdge = sp.edges[syncGroup.edgeIdxs.front()];
  if (firstEdge.srcOp != firstWriter)
    return {};
  auto forOp = dyn_cast_or_null<scf::ForOp>(firstEdge.dstOp);
  if (!forOp)
    return {};
  for (unsigned memberIdx : sp.memberIndices) {
    if (memberIdx >= group.members.size())
      continue;
    if (group.members[memberIdx].allocOp->hasAttr("buffer.copy"))
      return {};
  }
  if (forOp.getOperation()->isProperAncestor(firstWriter))
    return {};
  return forOp;
}

static bool edgeDefersToSkippedLoopExit(const SyncEdge &edge,
                                        scf::ForOp forOp) {
  if (!forOp || !edge.srcOp || !edge.dstOp)
    return false;
  Operation *forOperation = forOp.getOperation();
  return forOperation->isProperAncestor(edge.srcOp) &&
         !forOperation->isProperAncestor(edge.dstOp);
}

static bool edgeDstReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey);
static bool edgeDstWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);
static bool edgeSrcReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey);
static bool edgeSrcWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);
static bool edgeNeedsTerminalReadRelease(const SyncEdge &edge,
                                         BufferGroup &group,
                                         int64_t resourceKey);
static bool isRootToWsLoopEntryEdge(const SyncEdge &edge, BufferGroup &group);
static bool edgeUsesEmpty(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);

static bool hasOutgoingEdgeFromOp(const SyncPlan &sp, Operation *op) {
  if (!op)
    return false;
  for (const SyncEdge &edge : sp.edges)
    if (edge.srcOp == op)
      return true;
  return false;
}

static Operation *getNonWsForAncestorExitingTo(Operation *srcOp,
                                               Operation *dstOp) {
  if (!srcOp || !dstOp)
    return nullptr;
  for (Operation *parent = srcOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp || hasWarpSpecializeTag(parent))
      continue;
    if (!parent->isProperAncestor(dstOp))
      return parent;
  }
  return nullptr;
}

static bool isOriginalTmemTokenForMember(Value token, BufferGroup &group,
                                         ArrayRef<unsigned> memberIndices) {
  if (!group.isTmem() || !token || !isa<AsyncTokenType>(token.getType()))
    return false;
  for (unsigned memberIdx : memberIndices) {
    if (memberIdx >= group.members.size())
      continue;
    auto allocOp = dyn_cast<TMEMAllocOp>(group.members[memberIdx].allocOp);
    if (!allocOp || !allocOp.getToken())
      continue;
    if (token == allocOp.getToken())
      return true;
  }
  return false;
}

static bool isTmemAccessTokenForResource(Value token, BufferGroup &group,
                                         int64_t resourceKey) {
  if (!group.isTmem() || !token || !isa<AsyncTokenType>(token.getType()))
    return false;
  Operation *def = token.getDefiningOp();
  if (!def)
    return false;
  for (const AccessEvent &event : group.events)
    if (event.op == def && eventTouchesResource(event, resourceKey))
      return true;
  return false;
}

static bool isReusableTmemTokenForResource(Value token, BufferGroup &group,
                                           ArrayRef<unsigned> memberIndices,
                                           int64_t resourceKey) {
  return isOriginalTmemTokenForMember(token, group, memberIndices) ||
         isTmemAccessTokenForResource(token, group, resourceKey);
}

static bool tokenResultIsYielded(Operation *op) {
  if (!op)
    return false;
  for (Value result : op->getResults()) {
    if (!isa<AsyncTokenType>(result.getType()))
      continue;
    for (Operation *user : result.getUsers())
      if (isa<scf::YieldOp>(user))
        return true;
  }
  return false;
}

static bool shouldDeferTerminalLoopReadToExit(const SyncEdge &edge,
                                              BufferGroup &group,
                                              int64_t resourceKey) {
  if (!group.isTmem() || group.members.size() != 1 || !edge.dstYieldRegion ||
      !edge.srcOp || !tokenResultIsYielded(edge.srcOp))
    return false;
  if (group.members.empty() ||
      !group.members.front().allocOp->hasAttr("buffer.copy"))
    return false;
  auto forOp =
      dyn_cast_or_null<scf::ForOp>(edge.dstYieldRegion->getParentOp());
  return forOp && hasWarpSpecializeTag(forOp) &&
         forOp->isProperAncestor(edge.srcOp) &&
         edgeSrcReads(edge, group, resourceKey) &&
         !edgeSrcWrites(edge, group, resourceKey);
}

static std::optional<unsigned>
findReusableTmemTokenSlot(scf::ForOp forOp, BufferGroup &group,
                          ArrayRef<unsigned> memberIndices,
                          int64_t resourceKey) {
  if (!group.isTmem())
    return std::nullopt;
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (isReusableTmemTokenForResource(init, group, memberIndices, resourceKey))
      return static_cast<unsigned>(idx);
  }
  return std::nullopt;
}

static SmallVector<unsigned, 4>
findReusableTmemTokenSlots(scf::ForOp forOp, BufferGroup &group,
                           ArrayRef<unsigned> memberIndices,
                           int64_t resourceKey) {
  SmallVector<unsigned, 4> slots;
  if (!group.isTmem())
    return slots;
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs()))
    if (isReusableTmemTokenForResource(init, group, memberIndices, resourceKey))
      slots.push_back(static_cast<unsigned>(idx));
  return slots;
}

static scf::YieldOp getForYieldOp(scf::ForOp forOp) {
  for (Operation &op :
       llvm::reverse(forOp.getRegion().front().getOperations()))
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op))
      return yieldOp;
  llvm_unreachable("scf.for body has no scf.yield");
}

static scf::YieldOp appendToForYield(scf::ForOp forOp,
                                     ArrayRef<Value> newOperands) {
  scf::YieldOp yieldOp = getForYieldOp(forOp);
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());
  OpBuilder builder(yieldOp);
  auto newYield = scf::YieldOp::create(builder, yieldOp.getLoc(), operands);
  newYield->setAttrs(yieldOp->getAttrs());
  yieldOp->erase();
  return newYield;
}

static OptSyncDag buildOptSyncDag(const SyncPlan &sp, const ResourcePlan &plan,
                                  BufferGroup &group) {
  OptSyncDag dag;
  dag.resource = sp.resource;
  dag.groupIdx = sp.groupIdx;
  dag.memberIndices = sp.memberIndices;
  dag.accessOps = sp.accessOps;
  dag.edgeToGroup.assign(sp.edges.size(), 0u);
  for (auto [edgeIdx, edge] : llvm::enumerate(sp.edges)) {
    unsigned idx = static_cast<unsigned>(edgeIdx);
    if (edge.dstOp)
      dag.dstBranchEntryAnchor[idx] = getIfBranchEntryAnchor(edge.dstOp);
    if (edge.srcYieldRegion) {
      auto forOp = dyn_cast_or_null<scf::ForOp>(
          edge.srcYieldRegion->getParentOp());
      if (forOp && hasWarpSpecializeTag(forOp))
        dag.srcYieldParentWarpFor.insert(idx);
    }
    if (!isIfYieldRegion(edge.dstYieldRegion))
      continue;
    Operation *ifOp = edge.dstYieldRegion->getParentOp();
    if (Operation *joinAccess = findFirstAccessAfter(ifOp, dag))
      dag.ifYieldJoinAccess[idx] = joinAccess;
  }
  unsigned groupSerial = 0;

  // Initial writable permit: one planned initially-released EMPTY state per
  // managed (logicalGroupId, resourceKey), consumed by the first writer.
  std::optional<PartitionId> firstWriterOwner;
  if (!sp.edges.empty())
    if (Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner)) {
      SyncGroup g;
      g.name = makeGroupName(groupSerial++);
      g.kind = SyncGroupKind::InitialEmpty;
      g.initialOp = firstWriter;
      g.initialOwner = firstWriterOwner;
      dag.groups.push_back(std::move(g));
  }

  // First, partition edges by ReadyFanout key. Any bucket with >=2
  // entries collapses to one ReadyFanout group; size-1 buckets defer to
  // DoneFanin classification below.
  SmallVector<ReadyFanoutKey, 0> rfKeys;
  SmallVector<SmallVector<unsigned, 2>, 0> rfBuckets;
  auto rfFind = [&](const ReadyFanoutKey &k) -> int {
    for (unsigned i = 0; i < rfKeys.size(); ++i)
      if (rfKeys[i] == k) return static_cast<int>(i);
    return -1;
  };
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    const SyncEdge &e = sp.edges[i];
    ReadyFanoutKey k{e.srcOwner, e.srcEpoch, e.srcOp, e.srcYieldRegion};
    int idx = rfFind(k);
    if (idx < 0) {
      rfKeys.push_back(k);
      rfBuckets.emplace_back();
      idx = static_cast<int>(rfKeys.size() - 1);
    }
    rfBuckets[idx].push_back(i);
  }

  // Track which edges have been claimed by ReadyFanout groups so they
  // do not later participate in DoneFanin classification.
  SmallVector<bool> claimed(sp.edges.size(), false);

  for (unsigned i = 0; i < rfBuckets.size(); ++i) {
    if (rfBuckets[i].size() < 2) continue;
    // Sanity: a ReadyFanout group must have a usable src anchor (either
    // a real op or a yield-region). Otherwise it's a degenerate group
    // with no place to render the shared release, so leave as singletons.
    const SyncEdge &probe = sp.edges[rfBuckets[i].front()];
    if (!probe.srcOp && !probe.srcYieldRegion) continue;
    // A ReadyFanout shares ONE semaphore that every consumer edge acquires, at
    // its own consumer partition. A semaphore may be released from any
    // partition but acquired in only one, so only collapse when all consumer
    // edges have the same acquirer (dstOwner). Otherwise leave them as per-edge
    // Singletons so each consumer gets its own semaphore.
    if (llvm::any_of(rfBuckets[i], [&](unsigned e) {
          return !sameOwner(sp.edges[e].dstOwner, probe.dstOwner);
        }))
      continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::ReadyFanout;
    g.edgeIdxs.append(rfBuckets[i].begin(), rfBuckets[i].end());
    for (unsigned e : g.edgeIdxs) claimed[e] = true;
    dag.groups.push_back(std::move(g));
  }

  // DoneFanin among the not-yet-claimed edges.
  SmallVector<DoneFaninKey, 0> dfKeys;
  SmallVector<SmallVector<unsigned, 2>, 0> dfBuckets;
  auto dfFind = [&](const DoneFaninKey &k) -> int {
    for (unsigned i = 0; i < dfKeys.size(); ++i)
      if (dfKeys[i] == k) return static_cast<int>(i);
    return -1;
  };
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    if (claimed[i]) continue;
    const SyncEdge &e = sp.edges[i];
    DoneFaninKey k{e.dstOwner, e.dstOp, e.dstYieldRegion};
    int idx = dfFind(k);
    if (idx < 0) {
      dfKeys.push_back(k);
      dfBuckets.emplace_back();
      idx = static_cast<int>(dfKeys.size() - 1);
    }
    dfBuckets[idx].push_back(i);
  }
  for (unsigned i = 0; i < dfBuckets.size(); ++i) {
    if (dfBuckets[i].size() < 2) continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::DoneFanin;
    g.edgeIdxs.append(dfBuckets[i].begin(), dfBuckets[i].end());
    for (unsigned e : g.edgeIdxs) claimed[e] = true;
    dag.groups.push_back(std::move(g));
  }

  // Remaining unclaimed singleton edges that form a program-order handoff chain
  // render as one compact LinearChain group (Combine C).
  SmallVector<unsigned, 4> linearChain;
  if (collectLinearChainEdges(sp, claimed, linearChain)) {
    // A LinearChain collapses all of its writable-permit (EMPTY) handoffs onto
    // one shared EMPTY semaphore. That is sound only if every EMPTY edge is
    // acquired by the SAME partition: a semaphore may be released from any
    // partition, but it must always be acquired in exactly one. If the chain's
    // EMPTY edges span more than one acquirer (dstOwner) — e.g. a merged
    // resource whose qk write (p1) and P write (p5) are both writer phases —
    // collapsing would make one semaphore acquired by several partitions, which
    // is illegal and deadlocks. Leave such edges unclaimed so they fall back to
    // per-edge Singletons, each with a single acquirer (the RAW model).
    //
    // The same rule applies to the chain's data-ready edges, which may share a
    // single semaphore at emit: only collapse when both the writable (EMPTY)
    // edges and the data-ready (non-EMPTY) edges each have a single acquirer.
    SmallVector<std::optional<PartitionId>, 2> emptyAcquirers, fullAcquirers;
    for (unsigned idx : linearChain) {
      const SyncEdge &e = sp.edges[idx];
      auto &bucket = edgeUsesEmpty(e, group, sp.resource.second) ? emptyAcquirers
                                                                 : fullAcquirers;
      if (llvm::none_of(bucket,
                        [&](const std::optional<PartitionId> &o) {
                          return sameOwner(o, e.dstOwner);
                        }))
        bucket.push_back(e.dstOwner);
    }
    if (emptyAcquirers.size() <= 1 && fullAcquirers.size() <= 1) {
      SyncGroup g;
      g.name = makeGroupName(groupSerial++);
      g.kind = SyncGroupKind::LinearChain;
      g.edgeIdxs.append(linearChain.begin(), linearChain.end());
      for (unsigned e : g.edgeIdxs) claimed[e] = true;
      dag.groups.push_back(std::move(g));
    }
  }

  // Remaining: Singleton groups for each unclaimed edge.
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    if (claimed[i]) continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::Singleton;
    g.edgeIdxs.push_back(i);
    dag.groups.push_back(std::move(g));
  }

  // Populate edgeToGroup and the anchor maps.
  for (unsigned gi = 0; gi < dag.groups.size(); ++gi) {
    const SyncGroup &g = dag.groups[gi];
    for (unsigned ei : g.edgeIdxs) dag.edgeToGroup[ei] = gi;

    switch (g.kind) {
    case SyncGroupKind::InitialEmpty:
      if (g.initialOp)
        dag.acquireBeforeOp[g.initialOp].push_back(gi);
      break;
    case SyncGroupKind::Singleton: {
      // Release+acquire both at dst (matches RAW model).
      const SyncEdge &e = sp.edges[g.edgeIdxs.front()];
      if (e.dstOp) {
        if (isRootToWsLoopEntryEdge(e, group)) {
          dag.threadForOps.insert(e.dstOp);
          break;
        }
        addPlannedRelease(dag.releaseBeforeOp, e.dstOp, sp, gi,
                          g.edgeIdxs.front());
        Operation *acquireAnchor = e.dstOp;
        if (!group.isTmem()) {
          if (auto forOp = dyn_cast<scf::ForOp>(e.dstOp)) {
            if (hasWarpSpecializeTag(forOp.getOperation()))
              if (Operation *firstAccess =
                      findFirstAccessInRegion(forOp.getRegion(), dag))
                acquireAnchor = firstAccess;
          }
        }
        dag.acquireBeforeOp[acquireAnchor].push_back(gi);
        if (edgeNeedsTerminalReadRelease(e, group, sp.resource.second) &&
            findTerminalReadReleaseAnchor(e.dstOp, sp, group,
                                          sp.resource.second) &&
            !hasOutgoingEdgeFromOp(sp, e.dstOp) && !tokenResultIsYielded(e.dstOp))
          addPlannedRelease(dag.releaseAfterOp, e.dstOp, sp, gi,
                            g.edgeIdxs.front());
      } else if (e.dstYieldRegion) {
        if (group.isTmem()) {
          auto forOp = dyn_cast_or_null<scf::ForOp>(
              e.dstYieldRegion->getParentOp());
          if (forOp && hasWarpSpecializeTag(forOp) && !e.dstOwner &&
              e.srcOp && forOp->isProperAncestor(e.srcOp)) {
            addPlannedRelease(dag.releaseAfterOp, forOp.getOperation(), sp, gi,
                              g.edgeIdxs.front());
            if (Operation *afterLoop =
                    findFirstAccessAfter(forOp.getOperation(), dag)) {
              dag.acquireBeforeOp[afterLoop].push_back(gi);
              if (Operation *terminalRead = findTerminalReadReleaseAnchor(
                      afterLoop, sp, group, sp.resource.second))
                addPlannedRelease(dag.releaseAfterOp, terminalRead, sp, gi,
                                  g.edgeIdxs.front());
            }
            break;
          }
        }
        addPlannedRelease(dag.releaseBeforeYield, e.dstYieldRegion, sp, gi,
                          g.edgeIdxs.front());
        if (syncYieldRequiresCarrier(e.dstYieldRegion, sp))
          dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
      }
      break;
    }
    case SyncGroupKind::ReadyFanout: {
      // Shared release AFTER src (one row); per-consumer acquires at dst.
      const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
      if (probe.srcOp)
        addPlannedRelease(dag.releaseAfterOp, probe.srcOp, sp, gi, g.edgeIdxs);
      else if (probe.srcYieldRegion)
        addPlannedRelease(dag.releaseAfterYield, probe.srcYieldRegion, sp, gi,
                          g.edgeIdxs);
      // Acquires anchor at each consumer's dst.
      for (unsigned ei : g.edgeIdxs) {
        const SyncEdge &e = sp.edges[ei];
        if (e.dstOp)
          dag.acquireBeforeOp[e.dstOp].push_back(gi);
        else if (e.dstYieldRegion)
          dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
      }
      break;
    }
    case SyncGroupKind::DoneFanin: {
      // Per-reader releases AFTER each src; shared acquire at the dst.
      for (unsigned ei : g.edgeIdxs) {
        const SyncEdge &e = sp.edges[ei];
        if (e.srcOp)
          addPlannedRelease(dag.releaseAfterOp, e.srcOp, sp, gi, ei);
        else if (e.srcYieldRegion)
          addPlannedRelease(dag.releaseAfterYield, e.srcYieldRegion, sp, gi,
                            ei);
      }
      const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
      if (probe.dstOp)
        dag.acquireBeforeOp[probe.dstOp].push_back(gi);
      else if (probe.dstYieldRegion &&
               syncYieldRequiresCarrier(probe.dstYieldRegion, sp))
        dag.acquireBeforeYield[probe.dstYieldRegion].push_back(gi);
      break;
    }
    case SyncGroupKind::LinearChain: {
      // Compact linear handoff: each edge releases after its source and
      // acquires before its destination, sharing one EMPTY/FULL pair.
      Operation *initialOp = nullptr;
      for (const SyncGroup &candidate : dag.groups)
        if (candidate.kind == SyncGroupKind::InitialEmpty) {
          initialOp = candidate.initialOp;
          break;
        }
      scf::ForOp skippedCarrierFor =
          getSkippedInitialLoopCarrierFor(g, sp, group);
      if (skippedCarrierFor) {
        dag.skippedInitialLoopCarrierRegion[gi] =
            &skippedCarrierFor.getRegion();
        for (unsigned edgeIdx : g.edgeIdxs)
          if (edgeDefersToSkippedLoopExit(sp.edges[edgeIdx],
                                          skippedCarrierFor))
            dag.edgesDeferringToSkippedLoopExit.insert(edgeIdx);
      }
      bool chainEntersLoop =
          group.isTmem() && !g.edgeIdxs.empty() &&
          isa_and_nonnull<scf::ForOp>(sp.edges[g.edgeIdxs.front()].dstOp);
      bool chainHasIfYield =
          llvm::any_of(g.edgeIdxs, [&](unsigned edgeIdx) {
            return isIfYieldRegion(sp.edges[edgeIdx].dstYieldRegion);
          });
      DenseMap<Operation *, unsigned> ifYieldCounts;
      for (unsigned edgeIdx : g.edgeIdxs) {
        Region *yieldRegion = sp.edges[edgeIdx].dstYieldRegion;
        if (!isIfYieldRegion(yieldRegion))
          continue;
        ++ifYieldCounts[yieldRegion->getParentOp()];
      }
      for (unsigned ei : g.edgeIdxs) {
        const SyncEdge &e = sp.edges[ei];
        if (group.isTmem()) {
          if (auto forOp = dyn_cast_or_null<scf::ForOp>(e.dstOp)) {
            if (Operation *firstAccess =
                    findFirstAccessInRegion(forOp.getRegion(), dag)) {
              dag.threadForOps.insert(forOp.getOperation());
              if (skippedCarrierFor && forOp == skippedCarrierFor &&
                  ei == g.edgeIdxs.front() && initialOp && e.srcOp == initialOp)
                continue;
              addPlannedRelease(dag.releaseBeforeOp, firstAccess, sp, gi, ei);
              dag.acquireBeforeOp[firstAccess].push_back(gi);
              continue;
            }
          }
        }
        if (isIfYieldRegion(e.dstYieldRegion)) {
          auto ifOp = cast<scf::IfOp>(e.dstYieldRegion->getParentOp());
          if (ifYieldCounts.lookup(ifOp.getOperation()) > 1) {
            if (Operation *joinAccess =
                    findFirstAccessAfter(ifOp.getOperation(), dag)) {
              dag.threadIfOps.insert(ifOp.getOperation());
              SmallVector<unsigned, 4> branchEdges;
              for (unsigned branchEdgeIdx : g.edgeIdxs)
                if (sp.edges[branchEdgeIdx].dstYieldRegion &&
                    sp.edges[branchEdgeIdx].dstYieldRegion->getParentOp() ==
                        ifOp.getOperation())
                  branchEdges.push_back(branchEdgeIdx);
              addPlannedRelease(dag.releaseBeforeOp, joinAccess, sp, gi,
                                branchEdges);
              if (!llvm::is_contained(dag.acquireBeforeOp[joinAccess], gi))
                dag.acquireBeforeOp[joinAccess].push_back(gi);
              continue;
            }
          }
        }
        if (chainEntersLoop && e.dstYieldRegion &&
            !isIfYieldRegion(e.dstYieldRegion)) {
          if (shouldDeferTerminalLoopReadToExit(e, group, sp.resource.second)) {
            auto forOp = dyn_cast_or_null<scf::ForOp>(
                e.dstYieldRegion->getParentOp());
            if (forOp)
              if (Operation *firstAccess =
                      findFirstAccessInRegion(forOp.getRegion(), dag)) {
                dag.loopEntryHandoffAccess[ei] = firstAccess;
                addPlannedRelease(dag.releaseBeforeOp, firstAccess, sp, gi, ei);
                dag.acquireBeforeOp[firstAccess].push_back(gi);
                dag.threadForOps.insert(forOp.getOperation());
                continue;
              }
            dag.terminalLoopReadEdgesDeferringToExit.insert(ei);
            continue;
          }
          if (e.srcOp && edgeSrcReads(e, group, sp.resource.second) &&
              !edgeSrcWrites(e, group, sp.resource.second) &&
              isa<TMEMLoadOp>(e.srcOp)) {
            addPlannedRelease(dag.releaseAfterOp, e.srcOp, sp, gi, ei);
            if (syncYieldRequiresCarrier(e.dstYieldRegion, sp) &&
                !llvm::is_contained(dag.acquireBeforeYield[e.dstYieldRegion],
                                    gi))
              dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
          }
          continue;
        }
        if (e.dstYieldRegion) {
          if (shouldDeferTerminalLoopReadToExit(e, group, sp.resource.second)) {
            auto forOp = dyn_cast_or_null<scf::ForOp>(
                e.dstYieldRegion->getParentOp());
            if (forOp)
              if (Operation *firstAccess =
                      findFirstAccessInRegion(forOp.getRegion(), dag)) {
                dag.loopEntryHandoffAccess[ei] = firstAccess;
                addPlannedRelease(dag.releaseBeforeOp, firstAccess, sp, gi, ei);
                dag.acquireBeforeOp[firstAccess].push_back(gi);
                dag.threadForOps.insert(forOp.getOperation());
                continue;
              }
            dag.terminalLoopReadEdgesDeferringToExit.insert(ei);
            continue;
          }
          if (isIfYieldRegion(e.dstYieldRegion) && e.srcOp &&
              edgeSrcReads(e, group, sp.resource.second) &&
              !edgeSrcWrites(e, group, sp.resource.second))
            addPlannedRelease(dag.releaseAfterOp, e.srcOp, sp, gi, ei);
          else
            addPlannedRelease(dag.releaseBeforeYield, e.dstYieldRegion, sp, gi,
                              ei);
        } else if (chainHasIfYield && e.dstOp) {
          if (e.srcOp && e.srcOp->getBlock() == e.dstOp->getBlock() &&
              e.srcOp->isBeforeInBlock(e.dstOp))
            addPlannedRelease(dag.releaseAfterOp, e.srcOp, sp, gi, ei);
          else
            addPlannedRelease(dag.releaseBeforeOp, getIfBranchEntryAnchor(e.dstOp),
                              sp, gi, ei);
        } else if (e.srcOp) {
          if (!edgeDefersToSkippedLoopExit(e, skippedCarrierFor)) {
            Operation *releaseAnchor =
                getNonWsForAncestorExitingTo(e.srcOp, e.dstOp);
            addPlannedRelease(dag.releaseAfterOp,
                              releaseAnchor ? releaseAnchor : e.srcOp, sp, gi,
                              ei);
          }
        } else if (e.srcYieldRegion) {
          addPlannedRelease(dag.releaseAfterYield, e.srcYieldRegion, sp, gi,
                            ei);
        }
        if (e.dstOp)
          dag.acquireBeforeOp[e.dstOp].push_back(gi);
        else if (e.dstYieldRegion &&
                 syncYieldRequiresCarrier(e.dstYieldRegion, sp))
          dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
        if (ei == g.edgeIdxs.back() &&
            edgeNeedsTerminalReadRelease(e, group, sp.resource.second) &&
            findTerminalReadReleaseAnchor(e.dstOp, sp, group,
                                          sp.resource.second) &&
            !hasOutgoingEdgeFromOp(sp, e.dstOp) &&
            !tokenResultIsYielded(e.dstOp))
          addPlannedRelease(dag.releaseAfterOp, e.dstOp, sp, gi, ei);
      }
      break;
    }
    }
  }

  for (auto [edgeIdx, edge] : llvm::enumerate(sp.edges))
    if (Operation *loopExitRead = findTmemLoopExitReadForEdge(
            edge, sp, dag, group, sp.resource.second))
      dag.tmemLoopExitRead[static_cast<unsigned>(edgeIdx)] = loopExitRead;

  auto markThreadedOp = [&](Operation *op) {
    if (!op) return;
    if (isa<scf::ForOp>(op))
      dag.threadForOps.insert(op);
    else if (isa<scf::IfOp>(op))
      dag.threadIfOps.insert(op);
  };
  auto markOpAnchors = [&](const auto &anchors) {
    for (auto &kv : anchors)
      if (isa<scf::ForOp>(kv.first))
        markThreadedOp(kv.first);
  };
  markOpAnchors(dag.releaseBeforeOp);
  markOpAnchors(dag.acquireBeforeOp);
  markOpAnchors(dag.releaseAfterOp);
  for (auto &kv : dag.acquireBeforeYield)
    markThreadedOp(kv.first->getParentOp());

  auto markEscapingSourceToken = [&](Operation *srcOp, Operation *dstAnchor) {
    if (!srcOp || !dstAnchor)
      return;
    for (Operation *parent = srcOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(parent))
        continue;
      if (parent == dstAnchor || parent->isProperAncestor(dstAnchor))
        break;
      if (auto ifOp = dyn_cast<scf::IfOp>(parent))
        if (ifOp.getElseRegion().empty())
          continue;
      markThreadedOp(parent);
    }
  };
  for (const SyncEdge &edge : sp.edges) {
    Operation *dstAnchor =
        edge.dstOp ? edge.dstOp
                   : (edge.dstYieldRegion ? edge.dstYieldRegion->getParentOp()
                                          : nullptr);
    markEscapingSourceToken(edge.srcOp, dstAnchor);
  }

  auto carriedForRegion = [&](scf::ForOp forOp) {
    auto it = plan.regionOwners.find(&forOp.getRegion());
    return it != plan.regionOwners.end() && it->second.carried &&
           it->second.hasEventsInSubtree &&
           sameOwner(it->second.entry, it->second.exit);
  };
  auto canThreadIfRegion = [&](scf::IfOp ifOp) {
    if (!ifOp || ifOp.getElseRegion().empty())
      return false;
    auto thenIt = plan.regionOwners.find(&ifOp.getThenRegion());
    auto elseIt = plan.regionOwners.find(&ifOp.getElseRegion());
    bool thenHas = thenIt != plan.regionOwners.end() &&
                   thenIt->second.hasEventsInSubtree;
    bool elseHas = elseIt != plan.regionOwners.end() &&
                   elseIt->second.hasEventsInSubtree;
    return thenHas || elseHas;
  };
  auto hasEnclosingCarriedFor = [&](Operation *op) {
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
         parent = parent->getParentOp())
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        if (carriedForRegion(forOp))
          return true;
    return false;
  };
  auto carrierCrossesIfBoundary = [&](scf::IfOp ifOp) {
    return hasEnclosingCarriedFor(ifOp.getOperation()) ||
           findFirstAccessAfter(ifOp.getOperation(), dag);
  };
  auto propagateCarriedThreading = [&]() {
    // Edge groups and sync anchors are fixed at this point. This pass derives
    // only SSA carrier plumbing from that completed OPT-SYNC-DAG: when a
    // child structured op is already known to carry the semaphore state, every
    // enclosing loop/if that the state crosses must also return it. Re-entry
    // into a carried loop is itself the next use, so loop propagation must not
    // depend on finding a later access after the child in the same iteration.
    // An if has no re-entry by itself; only thread it when the carrier exits
    // the if to a later access or to an enclosing carried loop.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *, 8> seeds;
      seeds.append(dag.threadForOps.begin(), dag.threadForOps.end());
      seeds.append(dag.threadIfOps.begin(), dag.threadIfOps.end());
      for (Operation *seed : seeds) {
        for (Operation *parent = seed ? seed->getParentOp() : nullptr; parent;
             parent = parent->getParentOp()) {
          if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            if (carriedForRegion(forOp))
              changed |= dag.threadForOps.insert(parent).second;
            continue;
          }
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            if (canThreadIfRegion(ifOp) && carrierCrossesIfBoundary(ifOp))
              changed |= dag.threadIfOps.insert(parent).second;
          }
        }
      }
    }
  };

  auto removeGroupFromOpAnchor =
      [&](DenseMap<Operation *, SmallVector<unsigned, 2>> &anchors,
          Operation *op, unsigned groupIdx) {
        auto it = anchors.find(op);
        if (it == anchors.end()) return;
        it->second.erase(std::remove(it->second.begin(), it->second.end(),
                                     groupIdx),
                         it->second.end());
        if (it->second.empty())
          anchors.erase(it);
      };
  auto addGroupToOpAnchor =
      [&](DenseMap<Operation *, SmallVector<unsigned, 2>> &anchors,
          Operation *op, unsigned groupIdx) {
        SmallVector<unsigned, 2> &groups = anchors[op];
        if (!llvm::is_contained(groups, groupIdx))
          groups.push_back(groupIdx);
      };

  auto findLinearChainCarrierLoop = [&](const SyncGroup &syncGroup) -> scf::ForOp {
    if (!group.isTmem() || syncGroup.kind != SyncGroupKind::LinearChain)
      return {};
    SmallVector<scf::ForOp, 4> candidates;
    auto addCandidate = [&](scf::ForOp forOp) {
      if (forOp && hasWarpSpecializeTag(forOp) &&
          !llvm::is_contained(candidates, forOp))
        candidates.push_back(forOp);
    };
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(
              edge.dstYieldRegion ? edge.dstYieldRegion->getParentOp()
                                  : nullptr))
        addCandidate(forOp);
      addCandidate(getContainingWsFor(edge.srcOp));
      addCandidate(getContainingWsFor(edge.dstOp));
    }
    for (scf::ForOp forOp : candidates)
      if (findReusableTmemTokenSlot(forOp, group, dag.memberIndices,
                                    dag.resource.second))
        return forOp;
    return {};
  };

  for (const SyncGroup &syncGroup : dag.groups)
    if (scf::ForOp forOp = findLinearChainCarrierLoop(syncGroup))
      markThreadedOp(forOp.getOperation());

  // A token may be produced by an inner carried loop and be the ownership state
  // that an enclosing carried loop must reuse on its next iteration. Those
  // same-owner reentry requirements are visible in the ownership DAG as
  // `YIELD {P}` rows, not necessarily as extra sync edges, so promote threading
  // through all enclosing carried regions before placing the initial acquire.
  propagateCarriedThreading();

  // When the first writer is inside a loop whose resource state is loop-carried,
  // the initial EMPTY acquire must produce the loop-carried carrier before the
  // scf.for. The semaphore.buffer remains at the first writer and consumes the
  // iter_arg token inside the body.
  for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
    if (syncGroup.kind != SyncGroupKind::InitialEmpty || !syncGroup.initialOp)
      continue;
    Operation *threadedRegionOp = nullptr;
    for (Operation *parent = syncGroup.initialOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (dag.threadForOps.contains(parent) ||
            findReusableTmemTokenSlot(forOp, group, dag.memberIndices,
                                      dag.resource.second))
          threadedRegionOp = parent;
        continue;
      }
      // Initial-empty acquire seeds the first writer or a loop carrier. Do not
      // re-anchor it to an enclosing scf.if: branch-local semaphore creates do
      // not dominate the parent if.
      if (isa<scf::IfOp>(parent))
        continue;
    }
    if (!threadedRegionOp)
      continue;
    markThreadedOp(threadedRegionOp);
    unsigned groupIdx = static_cast<unsigned>(idx);
    removeGroupFromOpAnchor(dag.acquireBeforeOp, syncGroup.initialOp, groupIdx);
    addGroupToOpAnchor(dag.acquireBeforeOp, threadedRegionOp, groupIdx);
  }

  // §6.3 released-bit fact (stage-2 diagnostic). Forward program-order scan over
  // semaphore classes keyed (groupIdx, acquirer). The scan is uniform: for every
  // edge the destination is an *acquire* and the source is a *release* (an empty
  // edge has the reader release and the next writer acquire; a full edge has the
  // producer release and the consumer acquire — both place the acquire at dst and
  // the release at src). The InitialEmpty seed group contributes a single acquire
  // at its first writer with no preceding release. A class is created
  // `is_released = true` iff its earliest event is an acquire (§6.3): nothing
  // releases it before its first acquire. M1 expects exactly one such class for a
  // seeded resource; this is asserted/inspected, never repaired (§1).
  {
    DenseMap<Operation *, unsigned> rank;
    if (!dag.groups.empty()) {
      Operation *anchor = group.members.front().allocOp;
      if (auto funcOp = anchor->getParentOfType<triton::FuncOp>())
        buildProgramOrderRank(funcOp, rank);
    }
    auto rankOf = [&](Operation *op, Region *region) -> std::optional<unsigned> {
      Operation *keyOp = op ? op : (region ? region->getParentOp() : nullptr);
      if (!keyOp)
        return std::nullopt;
      auto it = rank.find(keyOp);
      if (it == rank.end())
        return std::nullopt;
      return it->second;
    };
    struct ClassEvents {
      unsigned groupIdx = 0;
      std::optional<PartitionId> acquirer;
      std::optional<unsigned> firstAcquire;
      std::optional<unsigned> firstRelease;
    };
    SmallVector<ClassEvents, 4> classes;
    auto classFor = [&](unsigned gIdx,
                        std::optional<PartitionId> acquirer) -> ClassEvents & {
      for (ClassEvents &c : classes)
        if (c.groupIdx == gIdx && sameOwner(c.acquirer, acquirer))
          return c;
      classes.push_back(ClassEvents{gIdx, acquirer, std::nullopt, std::nullopt});
      return classes.back();
    };
    auto noteEvent = [](std::optional<unsigned> &slot, std::optional<unsigned> r) {
      if (r && (!slot || *r < *slot))
        slot = r;
    };
    for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
      unsigned gIdx = static_cast<unsigned>(idx);
      if (syncGroup.kind == SyncGroupKind::InitialEmpty) {
        ClassEvents &c = classFor(gIdx, syncGroup.initialOwner);
        noteEvent(c.firstAcquire, rankOf(syncGroup.initialOp, nullptr));
        continue;
      }
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        ClassEvents &c = classFor(gIdx, edge.dstOwner);
        noteEvent(c.firstAcquire, rankOf(edge.dstOp, edge.dstYieldRegion));
        noteEvent(c.firstRelease, rankOf(edge.srcOp, edge.srcYieldRegion));
      }
    }
    for (const ClassEvents &c : classes) {
      bool released = c.firstAcquire &&
                      (!c.firstRelease || *c.firstAcquire < *c.firstRelease);
      if (released)
        dag.releasedSemaphores.push_back({c.groupIdx, c.acquirer});
    }
  }
  return dag;
}

// ---------------------------------------------------------------------------
// v4 commit 5 emission and post-emission cleanups.
// ---------------------------------------------------------------------------

#include "InsertSemasEmitterCore.cpp.inc"
#include "InsertSemasEmitterVerify.cpp.inc"
#include "InsertSemasCleanup.cpp.inc"

// ---------------------------------------------------------------------------
// Top-level pipeline (commit 5 stage).
// ---------------------------------------------------------------------------

struct PlannedResource {
  ResourcePlan plan;
  SyncPlan syncPlan;
  OptSyncDag optDag;
};

static bool tmemPlanNeedsMultiStage(BufferGroup &group,
                                    ArrayRef<PlannedResource> planned) {
  if (!group.isTmem() || group.members.size() != 1)
    return false;
  bool hasLinearChain = false;
  for (const PlannedResource &resource : planned)
    for (const SyncGroup &syncGroup : resource.optDag.groups)
      hasLinearChain |= syncGroup.kind == SyncGroupKind::LinearChain;
  if (!hasLinearChain)
    return false;

  for (auto [storeIdx, storeEvent] : llvm::enumerate(group.events)) {
    if (!storeEvent.owner || !isa<TMEMStoreOp>(storeEvent.op))
      continue;
    for (size_t mmaIdx = storeIdx + 1; mmaIdx < group.events.size();
         ++mmaIdx) {
      AccessEvent &mmaEvent = group.events[mmaIdx];
      if (!mmaEvent.owner || !isa<MMAv5OpInterface>(mmaEvent.op) ||
          sameOwner(storeEvent.owner, mmaEvent.owner))
        continue;
      auto parentLoop = mmaEvent.op->getParentOfType<scf::ForOp>();
      if (!parentLoop || hasWarpSpecializeTag(parentLoop))
        continue;
      for (size_t loadIdx = mmaIdx + 1; loadIdx < group.events.size();
           ++loadIdx) {
        AccessEvent &loadEvent = group.events[loadIdx];
        if (loadEvent.owner && isa<TMEMLoadOp>(loadEvent.op) &&
            sameOwner(storeEvent.owner, loadEvent.owner))
          return false;
      }
    }
  }
  return true;
}

static int computeTmemSemaphoreNumStagesFromPlans(
    BufferGroup &group, ArrayRef<PlannedResource> planned, int numTmemBlocks,
    bool useMetaPartitioner) {
  bool isMultiStaged = tmemPlanNeedsMultiStage(group, planned);
  if (isMultiStaged) {
    for (BufferMember &member : group.members) {
      auto allocOp = cast<TMEMAllocOp>(member.allocOp);
      for (auto user : allocOp.getResult().getUsers()) {
        if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
          if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
            auto wsLoop = getOuterWSLoop(loop);
            // Determine if the MMA accumulator can be multibuffered.
            bool accIsMultiBuffered =
                // MMAs in subsequent iterations can be overlapped.
                !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
                // The accumulator is reset at some point, thus allowing
                // multibuffering.
                isAccMultibufferingPossible(mmaOp, loop) &&
                // The user didn't disable it with a flag.
                !getDisallowAccMultiBuffer(wsLoop) &&
                canDoubleBufferAcc(mmaOp, numTmemBlocks);
            isMultiStaged = isMultiStaged && accIsMultiBuffered;
          }
        }
      }
    }
  }
  auto numStages =
      useMetaPartitioner ? 1 + 0 * isMultiStaged : 1 + 1 * isMultiStaged;
  return numStages;
}

static LogicalResult runOnFunction(triton::FuncOp funcOp,
                                   bool useMetaPartitioner) {
  // Only process functions that contain a warp-specialized loop, matching
  // the prior pipeline gating.
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted()) return success();

  // Phase 1: discovery.
  SmallVector<BufferGroup, 0> groups = collectAllBackingGroups(funcOp);

  // Phase 2: collect access events per group.
  for (BufferGroup &group : groups)
    if (failed(collectEvents(group, funcOp))) return failure();

  // Phase 3: build program-order rank, used by the ownership planner.
  DenseMap<Operation *, unsigned> rank;
  buildProgramOrderRank(funcOp, rank);

  bool dumpDag = shouldDumpDag();
  if (dumpDag) {
    llvm::errs() << "==== NVWS InsertSemas (commit 5: discovery + ACCESS DAG + "
                    "OWNERSHIP DAG + RAW-SYNC DAG + OPT-SYNC DAG + EMIT) ====\n";
    llvm::errs() << "function: " << funcOp.getName() << "\n";
    llvm::errs() << "backing buffers: " << groups.size() << "\n";
  }

  int numTmemBlocks = 0;
  DenseMap<unsigned, int> numStagesByGroup;
  SetVector<Operation *> eraseAfterEmission;

  DenseMap<BackingKey, GroupBacking> backings;
  for (auto en : llvm::enumerate(groups)) {
    BufferGroup &group = en.value();
    if (dumpDag) {
      dumpBackingGroupHeader(group);
      dumpAccessDag(group, funcOp);
    }
    std::set<int64_t> keys;
    for (auto &m : group.members) keys.insert(m.resourceKey);
    SmallVector<PlannedResource, 4> plannedResources;
    for (int64_t key : keys) {
      if (isExplicitOffsetSourcefulTmemSelfContained(group, key))
        continue;
      buildProgramOrderRank(funcOp, rank);
      ResourcePlan plan = planResource(funcOp,
                                       static_cast<unsigned>(en.index()),
                                       group, key, rank);
      SyncPlan sp = buildSyncPlan(group, plan, funcOp);
      OptSyncDag opt = buildOptSyncDag(sp, plan, group);
      if (dumpDag) {
        dumpOwnershipDag(plan, group, funcOp);
        dumpRawSyncDag(sp, plan, group, funcOp);
        dumpOptSyncDag(opt, sp, plan, group, funcOp);
        bool seeded = llvm::any_of(opt.groups, [](const SyncGroup &g) {
          return g.kind == SyncGroupKind::InitialEmpty;
        });
        llvm::errs() << "RELEASED-SEMAPHORES buffer.id=" << opt.resource.first
                     << " resourceKey=" << opt.resource.second
                     << " seeded=" << (seeded ? "yes" : "no")
                     << " count=" << opt.releasedSemaphores.size();
        if (seeded && opt.releasedSemaphores.size() != 1)
          llvm::errs() << " <<M1-VIOLATION: seeded resource must have exactly 1>>";
        if (!seeded && !opt.releasedSemaphores.empty())
          llvm::errs() << " <<M1-VIOLATION: edge-free resource must have 0>>";
        llvm::errs() << "\n";
        for (auto &[gIdx, acquirer] : opt.releasedSemaphores) {
          llvm::errs() << "  seed: group=" << gIdx << " (" << opt.groups[gIdx].name
                       << ", kind=" << static_cast<int>(opt.groups[gIdx].kind)
                       << ") acquirer=";
          if (acquirer)
            llvm::errs() << "{p" << acquirer->first << ",ws" << acquirer->second
                         << "}";
          else
            llvm::errs() << "{root}";
          llvm::errs() << "\n";
        }
        dumpEmitSchedule(opt, sp, plan, group, funcOp);
      }
      plannedResources.push_back(
          {std::move(plan), std::move(sp), std::move(opt)});
    }
    if (group.isTmem()) {
      int numStages = computeTmemSemaphoreNumStagesFromPlans(
          group, plannedResources, numTmemBlocks, useMetaPartitioner);
      numStagesByGroup[static_cast<unsigned>(en.index())] = numStages;
      updateNumTmemBlocks(group, numStages, numTmemBlocks);
    }
    for (PlannedResource &planned : plannedResources) {
      // Earlier resources in the same backing group may have rewritten scf.for
      // / scf.if signatures while adding carrier tokens. Rebuild the current
      // resource plan against the live IR so commit5 emission consumes the
      // completed OPT-SYNC-DAG with live structured-op anchors.
      buildProgramOrderRank(funcOp, rank);
      ResourcePlan emitPlan =
          planResource(funcOp, static_cast<unsigned>(en.index()), group,
                       planned.plan.resource.second, rank);
      SyncPlan emitSyncPlan = buildSyncPlan(group, emitPlan, funcOp);
      OptSyncDag emitOptDag = buildOptSyncDag(emitSyncPlan, emitPlan, group);
      if (failed(emitResource(funcOp, group, emitPlan, emitSyncPlan, emitOptDag,
                              backings, numStagesByGroup, eraseAfterEmission)))
        return failure();
    }
    bool hasBacking = llvm::any_of(backings, [&](const auto &entry) {
      return entry.first.first == static_cast<unsigned>(en.index());
    });
    if (hasBacking)
      poisonOriginalTmemAllocTokens(group);
    eraseUnusedOriginals(group);
  }
  for (Operation *op : llvm::reverse(eraseAfterEmission))
    op->erase();
  splitSemaphoreIfForLoopScheduler(funcOp);
  hoistInitialEmptyAcquires(funcOp);
  coalesceSemaphoreForCarriers(funcOp);
  coalesceTmemAllocsByBufferIdIntoViews(funcOp);
  eraseDeadTmemAllocs(funcOp);
  if (dumpDag)
    llvm::errs() << "\n";

  return success();
}

static void stripTemporarySemaphoreAttrs(triton::FuncOp funcOp) {
  funcOp.walk([&](Operation *op) { op->removeAttr("nvws.semaphore.backing"); });
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  using NVWSInsertSemasBase::NVWSInsertSemasBase;

  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp, useMetaPartitioner)))
        return WalkResult::interrupt();
      stripTemporarySemaphoreAttrs(funcOp);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
