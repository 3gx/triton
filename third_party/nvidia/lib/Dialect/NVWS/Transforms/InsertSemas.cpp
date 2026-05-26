// v4 commit 3: discovery + ACCESS DAG + OWNERSHIP DAG + RAW-SYNC DAG
// (dump-only).
//
// Per meta2nvws-plan/per-edge-sema-plan.v4.md Implementation Plan, this
// commit adds the third stage of the v4 pipeline:
//
//   discover backing buffers
//     -> build ACCESS DAG per buffer
//     -> build OWNERSHIP DAG per (logicalGroupId, resourceKey)
//     -> derive RAW-SYNC DAG per (logicalGroupId, resourceKey)
//
// The pass mutates no IR. It prints the backing-buffer list, the ACCESS
// DAG per buffer, a structured region OWNERSHIP DAG per backing resource,
// and a raw per-edge SYNC DAG per backing resource to stderr for manual
// verification.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

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

// ---------------------------------------------------------------------------
// v4 §Access Events data model.
// ---------------------------------------------------------------------------

enum class MemorySpaceKind { Tmem, Local };
// v4: only two access effects. R is provably read-only ops (tmem_load,
// local_load). Everything else (store, alloc-with-source, mma regardless
// of useAccumulator, descriptor load, etc.) is W. There is no ReadWrite.
enum class AccessEffect { Read, Write };

static bool hasRead(AccessEffect e) { return e == AccessEffect::Read; }
static bool hasWrite(AccessEffect e) { return e == AccessEffect::Write; }

// Effective owner per v4 §Effective Owner. nullopt = root/external; not
// equal to (wsTag=0, partition=0).
using PartitionId = std::pair<int /*partition*/, int /*wsTag*/>;

// One step in the alias/view chain from a member's alloc to a value
// referenced by a terminal op.
struct AliasStep {
  Operation *op = nullptr;
  unsigned sourceOperand = 0;
};

struct AliasInfo {
  unsigned memberIdx = 0;
  SmallVector<AliasStep> steps;
};

// One physical member of a backing group (one alloc).
struct BufferMember {
  Operation *allocOp = nullptr;
  Value value;
  MemDescType type;
  int64_t offset = 0;
  int64_t extent = 1;
  int64_t resourceKey = 0;
};

// Per-touch record on an AccessEvent.
struct AccessTouch {
  unsigned memberIdx = 0;
  int64_t resourceKey = 0;
  AccessEffect effect = AccessEffect::Read;
  Value accessValue;
  AliasInfo alias;
};

// One access event from the access DAG.
struct AccessEvent {
  Operation *op = nullptr;
  std::optional<PartitionId> owner;
  // The op that supplies this event's wsTag (v4 §Debug DAG Dumps scope
  // barrier rule):
  //   - nullptr if the event has no wsTag (owner is root/external).
  //   - == op itself if the op carries `ttg.warp_specialize.tag` directly
  //     on itself (intrinsic; self-named scope).
  //   - the enclosing WS-tagged scf.for otherwise (extrinsic).
  // Owner propagation: an extrinsic owner only propagates to regions
  // inside the body of `tagSourceOp`. An intrinsic owner propagates to
  // any region. A root owner propagates everywhere.
  Operation *tagSourceOp = nullptr;
  SmallVector<AccessTouch, 2> touches;
  bool sourcefulAllocStore = false;
};

// One backing buffer group.
struct BufferGroup {
  MemorySpaceKind memory = MemorySpaceKind::Tmem;
  int64_t logicalId = 0;
  SmallVector<BufferMember> members;
  DenseMap<Value, AliasInfo> aliases;
  SmallVector<Operation *> aliasOps;
  SmallVector<AccessEvent, 0> events;

  bool isTmem() const { return memory == MemorySpaceKind::Tmem; }
};

// ---------------------------------------------------------------------------
// Partition / owner helpers (v4 §Effective Owner).
// ---------------------------------------------------------------------------

static std::optional<int> tryGetWsTag(Operation *op) {
  while (op && !hasWarpSpecializeTag(op))
    op = op->getParentOfType<scf::ForOp>();
  if (!op) return std::nullopt;
  return *getWarpSpecializeTag(op);
}

// v4 §Debug DAG Dumps scope barrier rule: the op that supplies `op`'s
// wsTag, used to constrain owner propagation.
//   - nullptr: op has no WS context (root/external).
//   - op itself: op carries `ttg.warp_specialize.tag` directly (intrinsic).
//   - the enclosing WS-tagged scf.for: extrinsic — bound to that for-loop.
static Operation *getTagSourceOp(Operation *op) {
  if (!op) return nullptr;
  if (hasWarpSpecializeTag(op)) return op;
  Operation *p = op->getParentOfType<scf::ForOp>();
  while (p && !hasWarpSpecializeTag(p))
    p = p->getParentOfType<scf::ForOp>();
  return p;
}

static std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0) {
  if (!hasPartition(op))
    return std::nullopt;
  auto partitionIds = getPartitionIds(op);
  if (op->getNumRegions() > 0) {
    auto outputs = getPartitionOutputs(op);
    if (pos >= static_cast<int>(outputs.size()))
      return std::nullopt;
    partitionIds = outputs[pos];
  }
  if (partitionIds.size() != 1)
    return std::nullopt;
  auto tag = tryGetWsTag(op);
  if (!tag)
    return std::nullopt; // partition attr outside any WS loop = root/external
  return std::make_pair(*partitionIds.begin(), *tag);
}

// ---------------------------------------------------------------------------
// Discovery helpers (buffer.id / offset / overlap classes).
// ---------------------------------------------------------------------------

static constexpr StringLiteral kBufferIdAttrName = "buffer.id";
static constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
static constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";

static std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

static std::optional<int64_t> getBufferId(Operation *op) {
  return getI64Attr(op, kBufferIdAttrName);
}

static int64_t getBufferOffset(Operation *op) {
  return getI64Attr(op, kBufferOffsetAttrName).value_or(0);
}

static int64_t getTmemColumnExtent(MemDescType type) {
  // Approximation: TMEM extent = product of last-two dims for column width.
  auto shape = type.getShape();
  if (shape.size() < 2) return 1;
  return shape[shape.size() - 1];
}

static bool intervalsOverlap(int64_t aLo, int64_t aHi, int64_t bLo,
                             int64_t bHi) {
  return aLo < bHi && bLo < aHi;
}

static bool isTmemAlloc(Operation *op) { return isa<TMEMAllocOp>(op); }
static bool isLocalAlloc(Operation *op) { return isa<LocalAllocOp>(op); }

static bool isSemaphoreBackingAlloc(Operation *op) {
  // After commit 5 the pass marks materialized backing allocs so a later
  // run can skip them. At commit 1 nothing is marked yet, so always false.
  return op->hasAttr("nvws.semaphore.backing");
}

static bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

// A local alloc is a candidate semaphore-managed backing buffer if its
// memdesc type has the multi-stage layout that insert-allocas chose.
// Heuristic that matches the prior implementation: the alloc result type
// has a rank greater than the source tensor's rank (the leading dim
// is the multi-stage depth).
static bool isLocalSemaphoreBackingType(MemDescType type) {
  return type.getMutableMemory();
}

static AsyncOp getAsyncPayload(Operation *op) {
  if (isa<MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  return AsyncOp::NONE;
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
    return op->emitError("nvws-insert-semas: unsupported memdesc alias use ")
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

// ---------------------------------------------------------------------------
// Physical conflict key (v4 §Physical Conflict Key).
// Union-find over members of the same group: members whose offset intervals
// overlap share a resourceKey.
// ---------------------------------------------------------------------------

static void assignTmemResourceKeys(BufferGroup &group) {
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
      if (group.members.size() >= 3 &&
          lhs.type.getElementType() != rhs.type.getElementType())
        continue;
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

static BufferGroup makeTmemGroup(int64_t logicalId,
                                 MutableArrayRef<TMEMAllocOp> allocs) {
  BufferGroup group;
  group.memory = MemorySpaceKind::Tmem;
  group.logicalId = logicalId;
  for (auto [idx, allocOp] : llvm::enumerate(allocs)) {
    auto type = cast<MemDescType>(allocOp.getResult().getType());
    BufferMember member;
    member.allocOp = allocOp;
    member.value = allocOp.getResult();
    member.type = type;
    member.offset = getBufferOffset(allocOp);
    member.extent = getTmemColumnExtent(type);
    group.aliases.insert(
        {allocOp.getResult(), AliasInfo{static_cast<unsigned>(idx), {}}});
    group.members.push_back(member);
  }
  assignTmemResourceKeys(group);
  return group;
}

static BufferGroup makeLocalGroup(int64_t logicalId, LocalAllocOp allocOp) {
  BufferGroup group;
  group.memory = MemorySpaceKind::Local;
  group.logicalId = logicalId;
  auto type = cast<MemDescType>(allocOp.getResult().getType());
  group.members.push_back(
      {allocOp, allocOp.getResult(), type, /*offset=*/0, /*extent=*/1,
       /*resourceKey=*/0});
  group.aliases.insert({allocOp.getResult(), AliasInfo{0, {}}});
  return group;
}

static SmallVector<BufferGroup, 0>
collectAllBackingGroups(triton::FuncOp funcOp) {
  SmallVector<BufferGroup, 0> groups;
  int64_t nextSyntheticId = 0;

  // TMEM: group by buffer.id.
  llvm::MapVector<int64_t, SmallVector<TMEMAllocOp>> tmemBuckets;
  DenseMap<Operation *, int64_t> tmemSynthetic;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    std::optional<int64_t> id = getBufferId(allocOp);
    if (!id) {
      tmemSynthetic[allocOp.getOperation()] = nextSyntheticId++;
      id = tmemSynthetic[allocOp.getOperation()];
    }
    tmemBuckets[*id].push_back(allocOp);
  });
  for (auto &[id, allocs] : tmemBuckets)
    groups.push_back(makeTmemGroup(id, allocs));

  // Local: one group per LocalAllocOp.
  funcOp.walk([&](LocalAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    if (!isLocalSemaphoreBackingType(cast<MemDescType>(allocOp.getType())))
      return;
    int64_t id = getBufferId(allocOp).value_or(nextSyntheticId++);
    groups.push_back(makeLocalGroup(id, allocOp));
  });

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
        for (Value operand : op->getOperands()) {
          auto alias = lookupAlias(group, operand);
          if (failed(alias)) continue;
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
// Dump (commit 1 stage output).
// ---------------------------------------------------------------------------

// Anchor's WS-loop scope tag (v4 §Debug DAG Dumps display rule).
//   - If the anchor IS a WS-tagged scf.for op, the anchor IS the WS scope.
//   - Otherwise, the WS scope is the nearest enclosing WS-tagged scf.for
//     ancestor (not counting the anchor itself, even if the anchor is an
//     intrinsic-tag event).
// This is the key difference from `tryGetWsTag`: an intrinsic-tag event
// at func depth has no enclosing WS scope, so its rows print in tagged
// form, while the same event nested inside its WS-loop prints untagged.
static std::optional<int> getAnchorWsScopeTag(Operation *op) {
  if (!op) return std::nullopt;
  if (isa<scf::ForOp>(op) && hasWarpSpecializeTag(op))
    return *getWarpSpecializeTag(op);
  Operation *p = op->getParentOfType<scf::ForOp>();
  while (p && !hasWarpSpecializeTag(p))
    p = p->getParentOfType<scf::ForOp>();
  if (!p) return std::nullopt;
  return *getWarpSpecializeTag(p);
}

// Context-sensitive owner display (v4 §Debug DAG Dumps).
//
// - root/external → "root"
// - row anchored inside the same WS-tagged for-loop as the owner's wsTag
//   → "{<partition>}" (tag implicit)
// - row anchored outside any WS for-loop, or in a different WS for-loop
//   → "{@<wsTag>.<partition>}" (tag explicit)
static std::string ownerStr(Operation *anchor,
                            std::optional<PartitionId> owner) {
  if (!owner) return "root";
  std::string s;
  llvm::raw_string_ostream os(s);
  auto anchorTag = anchor ? getAnchorWsScopeTag(anchor) : std::nullopt;
  if (anchorTag && *anchorTag == owner->second)
    os << "{" << owner->first << "}";
  else
    os << "{@" << owner->second << "." << owner->first << "}";
  return s;
}

static char accessKindChar(bool reads, bool writes) {
  return writes ? 'W' : 'R';
}

static std::string treePrefix(unsigned depth) {
  std::string s;
  for (unsigned i = 0; i < depth; ++i) s += "|  ";
  return s;
}

// WS-tagged scf.for renders as "scf.for (WS, tag=N)"; plain scf.for as
// just "scf.for".
static std::string forOpLabel(scf::ForOp forOp) {
  if (!hasWarpSpecializeTag(forOp))
    return "scf.for";
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "scf.for (WS, tag=" << *getWarpSpecializeTag(forOp) << ")";
  return s;
}

// v4 §Debug DAG Dumps: a regioned op is included only when its subtree
// contains at least one event for the group/resource being printed.
static bool accessSubtreeHasEvent(
    Operation *op, DenseMap<Operation *, unsigned> &eventIdxByOp) {
  bool found = false;
  op->walk([&](Operation *o) -> WalkResult {
    if (eventIdxByOp.count(o)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static void dumpAccessDagBlock(Block &block, BufferGroup &group,
                               DenseMap<Operation *, unsigned> &eventIdxByOp,
                               unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!accessSubtreeHasEvent(&op, eventIdxByOp)) continue;
      llvm::errs() << treePrefix(depth) << "|- " << forOpLabel(forOp) << "\n";
      for (Block &b : forOp.getRegion())
        dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!accessSubtreeHasEvent(&op, eventIdxByOp)) continue;
      llvm::errs() << treePrefix(depth) << "|- scf.if\n";
      for (Block &b : ifOp.getThenRegion())
        dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      if (!ifOp.getElseRegion().empty())
        for (Block &b : ifOp.getElseRegion())
          dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      continue;
    }
    auto it = eventIdxByOp.find(&op);
    if (it == eventIdxByOp.end()) continue;
    AccessEvent &event = group.events[it->second];
    for (AccessTouch &touch : event.touches) {
      bool reads = hasRead(touch.effect);
      bool writes = hasWrite(touch.effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  "
                   << "m" << touch.memberIdx << "  "
                   << op.getName().getStringRef() << " "
                   << ownerStr(&op, event.owner) << "\n";
    }
  }
}

static void dumpBackingGroupHeader(BufferGroup &group) {
  llvm::errs() << "NVWS-SEMA-DAG buffer.id=" << group.logicalId
               << " memory=" << (group.isTmem() ? "tmem" : "local") << "\n";
  llvm::errs() << "  members:";
  for (auto [idx, member] : llvm::enumerate(group.members)) {
    llvm::errs() << " m" << idx << "(offset=" << member.offset
                 << ",extent=" << member.extent
                 << ",resourceKey=" << member.resourceKey << ")";
  }
  llvm::errs() << "\n";
}

static void dumpAccessDag(BufferGroup &group, triton::FuncOp funcOp) {
  DenseMap<Operation *, unsigned> eventIdxByOp;
  for (auto [idx, event] : llvm::enumerate(group.events))
    eventIdxByOp[event.op] = static_cast<unsigned>(idx);
  llvm::errs() << "ACCESS-DAG\n";
  for (Block &b : funcOp.getBody())
    dumpAccessDagBlock(b, group, eventIdxByOp, /*depth=*/0);
}

// ---------------------------------------------------------------------------
// v4 §Structured Region Ownership data model (commit 2).
// Per-resource side plan. The authoritative ownership state is this plan,
// not anything in IR. v4 plan: "Do not write ownership markers into IR as
// the source of truth."
// ---------------------------------------------------------------------------

using ResourceId = std::pair<int64_t /*logicalGroupId*/, int64_t /*resourceKey*/>;

struct RegionOwnership {
  std::optional<PartitionId> entry;
  std::optional<PartitionId> exit;
  bool carried = false;
  // True iff this region (or any nested region) contains at least one
  // event for the resource this plan is tracking. v4: a regioned op
  // (scf.if, scf.for) is annotated only when its subtree has access.
  bool hasEventsInSubtree = false;
};

struct ResourcePlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  // Members of `groups[groupIdx]` that share this resourceKey.
  SmallVector<unsigned, 4> memberIndices;
  // RegionOwnership per Region (function body, scf.if then/else,
  // scf.for body).
  DenseMap<Region *, RegionOwnership> regionOwners;
  // Direct-use owner per AccessEvent op that touches this resource.
  DenseMap<Operation *, std::optional<PartitionId>> useOwner;
  // tagSourceOp per AccessEvent op, mirrored from AccessEvent for fast
  // in-scope filtering inside the Planner.
  DenseMap<Operation *, Operation *> useTagSource;
  // Index into AccessEvent::touches of the touch that hit this resource
  // (used to render the right member name in the dump).
  DenseMap<Operation *, unsigned> useTouchIdx;
  // scf.yield owner stamp for loop bodies that carry this resource.
  DenseMap<Operation *, std::optional<PartitionId>> yieldOwner;
};

// v4 §Debug DAG Dumps scope barrier rule. An event with owner anchored
// at `tagSourceOp` propagates to a region `r` iff:
//   - tagSourceOp is null (root), or
//   - tagSourceOp == event.op (intrinsic; op self-names its scope), or
//   - tagSourceOp is r's parent op or an ancestor of r's parent op
//     (r is inside the body of the WS-tagged for-loop that supplies the
//     tag).
static bool isEventInScopeForRegion(Operation *tagSourceOp,
                                    Operation *eventOp, Region *region) {
  if (!tagSourceOp) return true;
  if (tagSourceOp == eventOp) return true;
  Operation *parent = region->getParentOp();
  while (parent) {
    if (parent == tagSourceOp) return true;
    parent = parent->getParentOp();
  }
  return false;
}

// ---------------------------------------------------------------------------
// Helpers used by the planner.
// ---------------------------------------------------------------------------

static void buildProgramOrderRank(triton::FuncOp funcOp,
                                  DenseMap<Operation *, unsigned> &rank) {
  rank.clear();
  unsigned i = 0;
  funcOp.walk([&](Operation *op) { rank[op] = i++; });
}

static unsigned maxRankInSubtree(Operation *op,
                                 const DenseMap<Operation *, unsigned> &rank) {
  unsigned mx = rank.lookup(op);
  op->walk([&](Operation *o) {
    auto it = rank.find(o);
    if (it != rank.end()) mx = std::max(mx, it->second);
  });
  return mx;
}

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
    for (auto [tIdx, touch] : llvm::enumerate(event.touches)) {
      if (touch.resourceKey == resourceKey) {
        plan.useOwner[event.op] = event.owner;
        plan.useTagSource[event.op] = event.tagSourceOp;
        plan.useTouchIdx[event.op] = static_cast<unsigned>(tIdx);
        break;
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

// One raw cross-owner edge. Both src and dst can be either a real access
// op or a virtual marker (region op for entering, region's YIELD for
// leaving). The renderer decides where to place release/acquire rows
// based on src/dst classification.
struct SyncEdge {
  std::string name;            // "S_e<N>"
  Operation *srcOp = nullptr;  // real access OR region op
  Operation *dstOp = nullptr;  // real access OR region op
  // If non-null, src/dst is a YIELD virtual row for this region (the
  // edge anchors at the region's YIELD point rather than at a real op).
  Region *srcYieldRegion = nullptr;
  Region *dstYieldRegion = nullptr;
  std::optional<PartitionId> srcOwner;
  std::optional<PartitionId> dstOwner;
};

struct SyncPlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncEdge> edges;
  // For each edge, both the release and the acquire render at the SAME
  // anchor point: right before the destination's row, at the
  // destination's tree depth. The anchor is keyed by either the
  // destination op (real access or region op) or the destination region
  // (for a YIELD-anchored close edge).
  DenseMap<Operation *, SmallVector<unsigned, 2>> beforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> beforeYield;
  // Ops with at least one touch for this resource (access rows in dump).
  DenseSet<Operation *> accessOps;
};

static bool touchReads(const AccessTouch &t) { return hasRead(t.effect); }
static bool touchWrites(const AccessTouch &t) { return hasWrite(t.effect); }

static const AccessTouch *
findTouchForResource(const AccessEvent &event, int64_t resourceKey) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey) return &t;
  return nullptr;
}

// True iff `event` reads/writes the resource identified by `resourceKey`.
static bool eventConsumes(const AccessEvent &event, int64_t resourceKey) {
  if (auto *t = findTouchForResource(event, resourceKey))
    return touchReads(*t);
  return false;
}
static bool eventProduces(const AccessEvent &event, int64_t resourceKey) {
  if (auto *t = findTouchForResource(event, resourceKey))
    return touchWrites(*t);
  return false;
}

static std::string makeEdgeName(unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "S_e" << serial;
  return s;
}

static bool sameOwner(const std::optional<PartitionId> &a,
                      const std::optional<PartitionId> &b) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return *a == *b;
}

// Return the partition for a region op (scf.for, scf.if) as seen by this
// resource: it equals the partition of the op's child region's ENTER/
// YIELD pair for this resource. Computed via the planner's RegionOwnership
// for one of the op's regions.
static std::optional<PartitionId>
regionOpPartition(Operation *op, const ResourcePlan &rp) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto it = rp.regionOwners.find(&forOp.getRegion());
    if (it == rp.regionOwners.end()) return std::nullopt;
    if (!it->second.hasEventsInSubtree) return std::nullopt;
    return it->second.entry;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto it = rp.regionOwners.find(&ifOp.getThenRegion());
    if (it != rp.regionOwners.end() && it->second.hasEventsInSubtree)
      return it->second.entry;
    auto it2 = rp.regionOwners.find(&ifOp.getElseRegion());
    if (it2 != rp.regionOwners.end() && it2->second.hasEventsInSubtree)
      return it2->second.entry;
    return std::nullopt;
  }
  return std::nullopt;
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

// When `dstOp` lives inside an `scf.for` body that is a descendant of
// `srcOp`'s region, lift the edge's dst to the outermost such scf.for op
// (at src's region depth). For `scf.if` branches, leave dst alone so the
// sync remains conditional inside the branch. Returns the (possibly
// lifted) op.
static Operation *liftDstForRegionTransition(Operation *srcOp,
                                             Operation *dstOp) {
  if (!srcOp || !dstOp) return dstOp;
  Region *srcReg = srcOp->getParentRegion();
  Region *dstReg = dstOp->getParentRegion();
  if (srcReg == dstReg) return dstOp;
  // Walk up from dstReg until we find a region whose parent region is
  // srcReg. The corresponding parent op is the candidate region op.
  Operation *liftedOp = dstOp;
  for (Region *r = dstReg; r;) {
    Operation *parentOp = r->getParentOp();
    if (!parentOp) break;
    Region *parentReg = parentOp->getParentRegion();
    if (parentReg == srcReg) {
      if (isa<scf::ForOp>(parentOp)) liftedOp = parentOp;
      // scf.if: leave dst as the original access so release/acquire
      // stay inside the conditional branch.
      break;
    }
    r = parentReg;
  }
  return liftedOp;
}

// Build the raw-sync DAG using version tracking. Edges are emitted at
// each cross-owner version transition. Loop-carry close edges go from
// the last cross-owner consumer/producer to the body's YIELD virtual op.
static SyncPlan buildSyncPlan(BufferGroup &group, const ResourcePlan &rp,
                              triton::FuncOp funcOp) {
  SyncPlan sp;
  sp.resource = rp.resource;
  sp.groupIdx = rp.groupIdx;
  sp.memberIndices = rp.memberIndices;

  // Events touching this resource, in program order.
  SmallVector<const AccessEvent *> events;
  for (const AccessEvent &e : group.events)
    if (findTouchForResource(e, rp.resource.second)) {
      events.push_back(&e);
      sp.accessOps.insert(e.op);
    }

  // Per-resource version tracking. Holds the live producer event (or
  // nullptr if none yet) and the set of cross-partition consumers that
  // have read the current version.
  const AccessEvent *currentProducer = nullptr;
  SmallVector<const AccessEvent *, 4> currentConsumers;
  unsigned serial = 0;

  for (const AccessEvent *E : events) {
    bool consumes = eventConsumes(*E, rp.resource.second);
    bool produces = eventProduces(*E, rp.resource.second);

    if (consumes) {
      if (currentProducer && !sameOwner(currentProducer->owner, E->owner)) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        edge.srcOp = currentProducer->op;
        edge.dstOp = liftDstForRegionTransition(currentProducer->op, E->op);
        edge.srcOwner = currentProducer->owner;
        edge.dstOwner = E->owner;
        recordEdge(sp, edge);
      }
      currentConsumers.push_back(E);
    }

    if (produces) {
      for (const AccessEvent *C : currentConsumers) {
        if (sameOwner(C->owner, E->owner)) continue;
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        edge.srcOp = C->op;
        edge.dstOp = liftDstForRegionTransition(C->op, E->op);
        edge.srcOwner = C->owner;
        edge.dstOwner = E->owner;
        recordEdge(sp, edge);
      }
      // Handoff fires only when there was no consumer at all (writer →
      // writer without an intervening reader). If consumers existed but
      // were same-partition as E, they handle the ordering via program
      // order; no handoff is needed.
      if (currentConsumers.empty() && !consumes && currentProducer &&
          !sameOwner(currentProducer->owner, E->owner)) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        edge.srcOp = currentProducer->op;
        edge.dstOp =
            liftDstForRegionTransition(currentProducer->op, E->op);
        edge.srcOwner = currentProducer->owner;
        edge.dstOwner = E->owner;
        recordEdge(sp, edge);
      }
      currentProducer = E;
      currentConsumers.clear();
    }
  }

  // Loop-carry close: at the YIELD of each annotated `scf.for` body whose
  // subtree contains events for this resource, emit edges from each
  // cross-owner live consumer/producer back to the body-carried owner so
  // the next iteration starts cleanly. The body-carried owner is on the
  // body's RegionOwnership entry/exit, equivalently the scf.for's
  // partition.
  funcOp.walk([&](scf::ForOp forOp) {
    Region &body = forOp.getRegion();
    auto it = rp.regionOwners.find(&body);
    if (it == rp.regionOwners.end() || !it->second.hasEventsInSubtree)
      return;
    auto carriedOwner = it->second.entry;

    // Find the last live state (producer + consumers) at the end of the
    // body for this resource. We re-walk events in this for-loop's
    // subtree to compute it locally — the global currentProducer/
    // currentConsumers at the very end of `events` may not correspond
    // to this specific for-loop.
    const AccessEvent *bodyProducer = nullptr;
    SmallVector<const AccessEvent *, 4> bodyConsumers;
    for (const AccessEvent *E : events) {
      if (!forOp->isProperAncestor(E->op)) continue;
      bool consumes = eventConsumes(*E, rp.resource.second);
      bool produces = eventProduces(*E, rp.resource.second);
      if (consumes) bodyConsumers.push_back(E);
      if (produces) {
        bodyProducer = E;
        bodyConsumers.clear();
      }
    }

    // Close edges: from each cross-owner consumer to YIELD, or from the
    // last producer to YIELD if no consumers remain and the producer is
    // cross-owner from the carried owner.
    bool emittedAny = false;
    for (const AccessEvent *C : bodyConsumers) {
      if (sameOwner(C->owner, carriedOwner)) continue;
      SyncEdge edge;
      edge.name = makeEdgeName(serial++);
      edge.srcOp = C->op;
      edge.dstYieldRegion = &body;
      edge.srcOwner = C->owner;
      edge.dstOwner = carriedOwner;
      recordEdge(sp, edge);
      emittedAny = true;
    }
    if (!emittedAny && bodyProducer &&
        !sameOwner(bodyProducer->owner, carriedOwner)) {
      SyncEdge edge;
      edge.name = makeEdgeName(serial++);
      edge.srcOp = bodyProducer->op;
      edge.dstYieldRegion = &body;
      edge.srcOwner = bodyProducer->owner;
      edge.dstOwner = carriedOwner;
      recordEdge(sp, edge);
    }
  });

  return sp;
}

// ---------------------------------------------------------------------------
// OWNERSHIP-DAG dump (commit 2 stage output).
// ---------------------------------------------------------------------------

// v4: a regioned op is annotated only when its subtree carries at least
// one access for this resource.
static bool regionHasEvents(Region &region, ResourcePlan &plan) {
  auto it = plan.regionOwners.find(&region);
  return it != plan.regionOwners.end() && it->second.hasEventsInSubtree;
}

// Build region-op label with the carried partition annotation:
//   `scf.for (WS, tag=N) {P}` for WS-tagged for-ops
//   `scf.for {P}`              for plain (non-WS) for-ops
//   `scf.if {P}`               for if-ops
// The partition is the body.ENTER/YIELD partition (and matches for both
// branches of scf.if per the invariant).
static std::string regionOpLabel(Operation *op, const ResourcePlan &plan) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (hasWarpSpecializeTag(op))
      os << "scf.for (WS, tag=" << *getWarpSpecializeTag(op) << ")";
    else
      os << "scf.for";
  } else if (isa<scf::IfOp>(op)) {
    os << "scf.if";
  }
  auto part = regionOpPartition(op, plan);
  os << " " << ownerStr(op, part);
  return s;
}

// Render an `ENTER {P}` or `YIELD {P}` virtual row at the given depth.
static void renderEnterRow(unsigned depth, Operation *anchor,
                           std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- ENTER " << ownerStr(anchor, partition)
               << "\n";
}
static void renderYieldRow(unsigned depth, Operation *anchor,
                           std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- YIELD " << ownerStr(anchor, partition)
               << "\n";
}

// ---------------------------------------------------------------------------
// OWNERSHIP-DAG dump (commit 2/3, new convention).
// ---------------------------------------------------------------------------

static void dumpOwnershipBlock(Block &block, ResourcePlan &plan,
                               BufferGroup &group, unsigned depth);

static void dumpOwnershipRegion(Region &region, ResourcePlan &plan,
                                BufferGroup &group, Operation *anchorOp,
                                unsigned depth) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end()) part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region) dumpOwnershipBlock(b, plan, group, depth);
  renderYieldRow(depth, anchorOp, part);
}

static void dumpOwnershipBlock(Block &block, ResourcePlan &plan,
                               BufferGroup &group, unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionHasEvents(forOp.getRegion(), plan)) continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpOwnershipRegion(forOp.getRegion(), plan, group, &op, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionHasEvents(ifOp.getThenRegion(), plan);
      bool elseHas = regionHasEvents(ifOp.getElseRegion(), plan);
      if (!thenHas && !elseHas) continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpOwnershipRegion(ifOp.getThenRegion(), plan, group, &op, depth + 2);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpOwnershipRegion(ifOp.getElseRegion(), plan, group, &op, depth + 2);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op)) continue; // YIELD is rendered by dumpOwnershipRegion
    auto useIt = plan.useOwner.find(&op);
    if (useIt == plan.useOwner.end()) continue;
    unsigned tIdx = plan.useTouchIdx.lookup(&op);
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    if (!event || tIdx >= event->touches.size()) continue;
    AccessTouch &touch = event->touches[tIdx];
    bool reads = hasRead(touch.effect);
    bool writes = hasWrite(touch.effect);
    llvm::errs() << treePrefix(depth) << "|- "
                 << accessKindChar(reads, writes) << "  m" << touch.memberIdx
                 << "  " << op.getName().getStringRef() << "  use "
                 << ownerStr(&op, useIt->second) << "\n";
  }
}

static void dumpOwnershipDag(ResourcePlan &plan, BufferGroup &group,
                             triton::FuncOp funcOp) {
  llvm::errs() << "OWNERSHIP-DAG buffer.id=" << plan.resource.first
               << " resourceKey=" << plan.resource.second << " members:";
  for (unsigned idx : plan.memberIndices) llvm::errs() << " m" << idx;
  llvm::errs() << "\n";
  // v4: the function region is never annotated and has no ENTER/YIELD.
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpOwnershipBlock(b, plan, group, /*depth=*/1);
}

// ---------------------------------------------------------------------------
// RAW-SYNC-DAG dump (commit 3, new convention).
// Same region-tree shape as OWNERSHIP-DAG; per-edge sync rows (r / a)
// rendered at the same depth as the access rows they sit between. No
// FULL/EMPTY column; no kind label. Each edge is a single counter
// `S_e<N>`.
// ---------------------------------------------------------------------------

static void renderAcquireRow(unsigned depth, Operation *anchor,
                             const SyncEdge &edge) {
  llvm::errs() << treePrefix(depth) << "|- a  " << edge.name << "  acquire  "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

static void renderReleaseRow(unsigned depth, Operation *anchor,
                             const SyncEdge &edge) {
  llvm::errs() << treePrefix(depth) << "|- r  " << edge.name << "  release  "
               << ownerStr(anchor, edge.srcOwner) << " -> "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

// Print the release+acquire pair for each edge anchored at `key`, in the
// order they were recorded. Both rows render at the same depth.
static void printEdgesAt(SmallVector<unsigned, 2> *edgeIdxs, SyncPlan &sp,
                         Operation *anchor, unsigned depth) {
  if (!edgeIdxs) return;
  for (unsigned idx : *edgeIdxs) {
    const SyncEdge &edge = sp.edges[idx];
    renderReleaseRow(depth, anchor, edge);
    renderAcquireRow(depth, anchor, edge);
  }
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
                             unsigned depth);

static void dumpRawSyncRegion(Region &region, SyncPlan &sp,
                              const ResourcePlan &plan, BufferGroup &group,
                              Operation *anchorOp, unsigned depth) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end()) part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region) dumpRawSyncBlock(b, sp, plan, group, depth);
  // YIELD-anchored close edges render right before the YIELD row.
  auto yIt = sp.beforeYield.find(&region);
  if (yIt != sp.beforeYield.end())
    printEdgesAt(&yIt->second, sp, anchorOp, depth);
  renderYieldRow(depth, anchorOp, part);
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
                             unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      bool show = false;
      forOp.walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show && !sp.beforeYield.count(&forOp.getRegion())) continue;
      // Edges anchored "before" this region op (release+acquire pair).
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth);
      // Region-op header row (planner-derived partition).
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpRawSyncRegion(forOp.getRegion(), sp, plan, group, &op, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool show = false;
      ifOp->walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show) continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpRawSyncRegion(ifOp.getThenRegion(), sp, plan, group, &op, depth + 2);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpRawSyncRegion(ifOp.getElseRegion(), sp, plan, group, &op,
                          depth + 2);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op)) continue;
    if (!sp.accessOps.contains(&op)) continue;
    // Edges anchored before this access row.
    auto bIt = sp.beforeOp.find(&op);
    if (bIt != sp.beforeOp.end())
      printEdgesAt(&bIt->second, sp, &op, depth);
    // Access row.
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    const AccessTouch *touch =
        event ? findTouchForResource(*event, sp.resource.second) : nullptr;
    if (touch) {
      bool reads = hasRead(touch->effect);
      bool writes = hasWrite(touch->effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
  }
}

static void dumpRawSyncDag(SyncPlan &sp, const ResourcePlan &plan,
                           BufferGroup &group, triton::FuncOp funcOp) {
  llvm::errs() << "RAW-SYNC-DAG buffer.id=" << sp.resource.first
               << " resourceKey=" << sp.resource.second << " edges="
               << sp.edges.size() << "\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1);
}

// ---------------------------------------------------------------------------
// Top-level pipeline (commit 3 stage).
// ---------------------------------------------------------------------------

static LogicalResult runOnFunction(triton::FuncOp funcOp) {
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

  // Dump (commit 3: discovery + ACCESS DAG + OWNERSHIP DAG + RAW-SYNC DAG).
  llvm::errs() << "==== NVWS InsertSemas (commit 3: discovery + ACCESS DAG + "
                  "OWNERSHIP DAG + RAW-SYNC DAG) ====\n";
  llvm::errs() << "function: " << funcOp.getName() << "\n";
  llvm::errs() << "backing buffers: " << groups.size() << "\n";
  for (auto en : llvm::enumerate(groups)) {
    BufferGroup &group = en.value();
    dumpBackingGroupHeader(group);
    dumpAccessDag(group, funcOp);
    std::set<int64_t> keys;
    for (auto &m : group.members) keys.insert(m.resourceKey);
    for (int64_t key : keys) {
      ResourcePlan plan = planResource(funcOp,
                                       static_cast<unsigned>(en.index()),
                                       group, key, rank);
      dumpOwnershipDag(plan, group, funcOp);
      SyncPlan sp = buildSyncPlan(group, plan, funcOp);
      dumpRawSyncDag(sp, plan, group, funcOp);
    }
  }
  llvm::errs() << "\n";

  return success();
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
