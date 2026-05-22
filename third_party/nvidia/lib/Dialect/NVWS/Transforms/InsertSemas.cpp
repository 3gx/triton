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
enum class AccessEffect { Read, Write, ReadWrite };

static bool hasRead(AccessEffect e) {
  return e == AccessEffect::Read || e == AccessEffect::ReadWrite;
}
static bool hasWrite(AccessEffect e) {
  return e == AccessEffect::Write || e == AccessEffect::ReadWrite;
}

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

static bool isConstFalse(Value v) {
  if (!v) return false;
  if (auto def = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto boolAttr = dyn_cast<BoolAttr>(def.getValue()))
      return !boolAttr.getValue();
  }
  return false;
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
        for (Value operand : op->getOperands()) {
          auto alias = lookupAlias(group, operand);
          if (failed(alias)) continue;
          AccessEffect effect =
              operand == mma.getAccumulator()
                  ? (isConstFalse(mma.useAccumulator()) ? AccessEffect::Write
                                                        : AccessEffect::ReadWrite)
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
// v4 §Dependency Edges / §Planned SyncEdgeInfo (commit 3).
// Raw per-edge synchronization graph derived from the per-resource
// OWNERSHIP-DAG. One edge per cross-owner dependency. No IR mutation.
// ---------------------------------------------------------------------------

enum class SyncEdgeKind { Ready, Done, Handoff };
enum class SemaState { Full, Empty };

static StringRef stateStr(SemaState s) {
  return s == SemaState::Full ? "FULL" : "EMPTY";
}

// One raw cross-owner edge. v4 §Planned SyncEdgeInfo subset; the full
// SyncEdgeInfo (planned release/acquire anchors, async payload, carrier
// token, backing data buffers) is materialized in later commits when the
// pass starts emitting IR.
struct SyncEdge {
  std::string name;
  SyncEdgeKind kind = SyncEdgeKind::Ready;
  SemaState state = SemaState::Full;
  // srcOp = null for the initial writable permit.
  Operation *srcOp = nullptr;
  Operation *dstOp = nullptr;
  std::optional<PartitionId> srcOwner;
  std::optional<PartitionId> dstOwner;
  // Anchor region + position used by the dump. v4 §Debug DAG Dumps places
  // release/acquire rows either inline next to the source/target access
  // rows (same region) or at the deeper region's entry/exit (cross-region).
  enum class Position { InLine, AtEntry, AtExit };
  Region *anchorRegion = nullptr;
  Position position = Position::InLine;
};

struct SyncPlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncEdge> edges;
  // Per-access-op incoming/outgoing edge indices. Only InLine edges are
  // indexed here; cross-region edges live in entry/exit maps.
  DenseMap<Operation *, SmallVector<unsigned, 2>> incomingByAccess;
  DenseMap<Operation *, SmallVector<unsigned, 2>> outgoingByAccess;
  // Per-region entry/exit edge indices (cross-region case).
  DenseMap<Region *, SmallVector<unsigned, 2>> entryEdges;
  DenseMap<Region *, SmallVector<unsigned, 2>> exitEdges;
  // Ops with at least one touch for this resource (access rows in dump).
  DenseSet<Operation *> accessOps;
  // Initial permit edge (S_init). Anchored at funcDirectAnchor inside
  // initialAnchorRegion: printed immediately before that op at the region's
  // depth.
  std::optional<unsigned> initialEdgeIdx;
  Region *initialAnchorRegion = nullptr;
  Operation *initialAnchorBeforeOp = nullptr;
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

// Anchor region computation for a cross-region raw sync edge. Returns the
// deepest unshared region (closer to dst when dst is deeper, closer to src
// when src is deeper) and the position to render the edge rows.
static std::pair<Region *, SyncEdge::Position>
computeAnchor(Operation *srcOp, Operation *dstOp) {
  Region *regSrc = srcOp ? srcOp->getParentRegion() : nullptr;
  Region *regDst = dstOp ? dstOp->getParentRegion() : nullptr;
  if (regSrc == regDst)
    return {regSrc, SyncEdge::Position::InLine};

  // Case A: regDst is descendant of regSrc.
  for (Region *r = regDst; r; ) {
    Operation *parentOp = r->getParentOp();
    if (!parentOp) break;
    Region *parentReg = parentOp->getParentRegion();
    if (parentReg == regSrc)
      return {r, SyncEdge::Position::AtEntry};
    r = parentReg;
  }
  // Case B: regSrc is descendant of regDst.
  for (Region *r = regSrc; r; ) {
    Operation *parentOp = r->getParentOp();
    if (!parentOp) break;
    Region *parentReg = parentOp->getParentRegion();
    if (parentReg == regDst)
      return {r, SyncEdge::Position::AtExit};
    r = parentReg;
  }
  // Case C: neither descends from the other (sibling regions under a
  // common ancestor — e.g., events both inside different branches of an
  // outer scf.if). Not exercised by v4 examples; fall back to inline at
  // src side so the edge is at least visible.
  return {regSrc, SyncEdge::Position::InLine};
}

// Initial-permit anchor: walk up from `dstOp` until we hit `funcBody`.
// The function-direct-child op containing dstOp is the row we render the
// initial acquire before. For loop bodies this hoists the initial permit
// out of the loop, matching v4 example 5 (S_alpha_done acquired before
// scf.for).
static Operation *findInitialAnchorOp(Operation *dstOp, Region &funcBody) {
  Operation *cur = dstOp;
  while (cur && cur->getParentRegion() != &funcBody) {
    Operation *p = cur->getParentOp();
    if (!p) break;
    cur = p;
  }
  return cur;
}

static std::string formatOwnerPair(Operation *anchor,
                                   std::optional<PartitionId> src,
                                   std::optional<PartitionId> dst) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (src)
    os << ownerStr(anchor, src);
  else
    os << "init";
  os << " -> " << ownerStr(anchor, dst);
  return s;
}

static std::string makeEdgeName(SyncEdgeKind kind,
                                std::optional<PartitionId> srcOwner,
                                std::optional<PartitionId> dstOwner,
                                unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  char prefix = kind == SyncEdgeKind::Ready
                    ? 'R'
                    : (kind == SyncEdgeKind::Done ? 'D' : 'H');
  os << "S_" << prefix << "_";
  if (srcOwner)
    os << srcOwner->first;
  else
    os << "root";
  os << "_";
  if (dstOwner)
    os << dstOwner->first;
  else
    os << "root";
  os << "_e" << serial;
  return s;
}

// Add an edge to `sp` and wire it into per-op / per-region indices. InLine
// edges are anchored directly to the source/target access ops; region
// entry/exit edges are anchored to the deeper region only.
static unsigned recordEdge(SyncPlan &sp, SyncEdge edge) {
  unsigned idx = sp.edges.size();
  if (edge.position == SyncEdge::Position::InLine) {
    if (edge.srcOp) sp.outgoingByAccess[edge.srcOp].push_back(idx);
    if (edge.dstOp) sp.incomingByAccess[edge.dstOp].push_back(idx);
  } else if (edge.position == SyncEdge::Position::AtEntry) {
    sp.entryEdges[edge.anchorRegion].push_back(idx);
  } else if (edge.position == SyncEdge::Position::AtExit) {
    sp.exitEdges[edge.anchorRegion].push_back(idx);
  }
  sp.edges.push_back(std::move(edge));
  return idx;
}

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

  // Per-resource version-tracking state.
  SmallVector<const AccessEvent *, 4> versionProducers;
  SmallVector<const AccessEvent *, 4> versionConsumers;
  unsigned serial = 0;

  auto sameOwner = [](const std::optional<PartitionId> &a,
                      const std::optional<PartitionId> &b) {
    if (!a && !b) return true;
    if (!a || !b) return false;
    return *a == *b;
  };

  for (const AccessEvent *E : events) {
    bool consumes = eventConsumes(*E, rp.resource.second);
    bool produces = eventProduces(*E, rp.resource.second);

    if (consumes) {
      // Ready edges from each current-version producer with different owner.
      for (const AccessEvent *P : versionProducers) {
        if (sameOwner(P->owner, E->owner)) continue;
        SyncEdge edge;
        edge.kind = SyncEdgeKind::Ready;
        edge.state = SemaState::Full;
        edge.srcOp = P->op;
        edge.dstOp = E->op;
        edge.srcOwner = P->owner;
        edge.dstOwner = E->owner;
        edge.name = makeEdgeName(edge.kind, P->owner, E->owner, serial++);
        std::tie(edge.anchorRegion, edge.position) =
            computeAnchor(P->op, E->op);
        recordEdge(sp, edge);
      }
    }

    if (produces) {
      // E retires the current version. Done edges from each consumer with
      // different owner; if no consumers, handoff edges from producers
      // with different owner (writer->writer no read fanout).
      if (!versionConsumers.empty()) {
        for (const AccessEvent *C : versionConsumers) {
          if (sameOwner(C->owner, E->owner)) continue;
          SyncEdge edge;
          edge.kind = SyncEdgeKind::Done;
          edge.state = SemaState::Empty;
          edge.srcOp = C->op;
          edge.dstOp = E->op;
          edge.srcOwner = C->owner;
          edge.dstOwner = E->owner;
          edge.name = makeEdgeName(edge.kind, C->owner, E->owner, serial++);
          std::tie(edge.anchorRegion, edge.position) =
              computeAnchor(C->op, E->op);
          recordEdge(sp, edge);
        }
      } else if (!consumes) {
        for (const AccessEvent *P : versionProducers) {
          if (sameOwner(P->owner, E->owner)) continue;
          SyncEdge edge;
          edge.kind = SyncEdgeKind::Handoff;
          edge.state = SemaState::Empty;
          edge.srcOp = P->op;
          edge.dstOp = E->op;
          edge.srcOwner = P->owner;
          edge.dstOwner = E->owner;
          edge.name = makeEdgeName(edge.kind, P->owner, E->owner, serial++);
          std::tie(edge.anchorRegion, edge.position) =
              computeAnchor(P->op, E->op);
          recordEdge(sp, edge);
        }
      }
      versionProducers.clear();
      versionConsumers.clear();
      versionProducers.push_back(E);
    } else if (consumes) {
      versionConsumers.push_back(E);
    }
  }

  // Initial writable/readable permit (v4 §Initial Writable Permit). One
  // S_init per (logicalGroupId, resourceKey).
  if (!events.empty()) {
    const AccessEvent *first = events.front();
    bool firstProduces = eventProduces(*first, rp.resource.second);
    SyncEdge init;
    init.kind = SyncEdgeKind::Handoff;
    init.state = firstProduces ? SemaState::Empty : SemaState::Full;
    init.name = "S_init";
    init.srcOp = nullptr;
    init.dstOp = first->op;
    init.srcOwner = std::nullopt;
    init.dstOwner = first->owner;
    init.position = SyncEdge::Position::InLine;
    sp.initialAnchorRegion = &funcOp.getBody();
    sp.initialAnchorBeforeOp =
        findInitialAnchorOp(first->op, funcOp.getBody());
    init.anchorRegion = sp.initialAnchorRegion;
    sp.edges.push_back(std::move(init));
    sp.initialEdgeIdx = sp.edges.size() - 1;
  }

  return sp;
}

// ---------------------------------------------------------------------------
// OWNERSHIP-DAG dump (commit 2 stage output).
// ---------------------------------------------------------------------------

static std::string formatRegionLine(Operation *anchor, StringRef label,
                                    const RegionOwnership &rec) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << label << "  entry " << ownerStr(anchor, rec.entry) << " exit "
     << ownerStr(anchor, rec.exit);
  if (rec.carried) os << " carried";
  return s;
}

// v4: a regioned op is annotated only when its subtree carries at least
// one access for this resource.
static bool regionHasEvents(Region &region, ResourcePlan &plan) {
  auto it = plan.regionOwners.find(&region);
  return it != plan.regionOwners.end() && it->second.hasEventsInSubtree;
}

static void dumpOwnershipBlock(Block &block, ResourcePlan &plan,
                               BufferGroup &group, unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionHasEvents(forOp.getRegion(), plan)) continue;
      llvm::errs() << treePrefix(depth) << "|- " << forOpLabel(forOp)
                   << "                              structural\n";
      auto &bodyRec = plan.regionOwners[&forOp.getRegion()];
      llvm::errs() << treePrefix(depth + 1) << "|- "
                   << formatRegionLine(&op, "body region", bodyRec) << "\n";
      for (Block &b : forOp.getRegion())
        dumpOwnershipBlock(b, plan, group, depth + 2);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionHasEvents(ifOp.getThenRegion(), plan);
      bool elseHas = regionHasEvents(ifOp.getElseRegion(), plan);
      if (!thenHas && !elseHas) continue;
      llvm::errs() << treePrefix(depth)
                   << "|- scf.if                               structural\n";
      auto &thenRec = plan.regionOwners[&ifOp.getThenRegion()];
      llvm::errs() << treePrefix(depth + 1) << "|- "
                   << formatRegionLine(&op, "then region", thenRec) << "\n";
      for (Block &b : ifOp.getThenRegion())
        dumpOwnershipBlock(b, plan, group, depth + 2);
      auto &elseRec = plan.regionOwners[&ifOp.getElseRegion()];
      llvm::errs() << treePrefix(depth + 1) << "|- "
                   << formatRegionLine(&op, "else region", elseRec) << "\n";
      if (!ifOp.getElseRegion().empty())
        for (Block &b : ifOp.getElseRegion())
          dumpOwnershipBlock(b, plan, group, depth + 2);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto it = plan.yieldOwner.find(yieldOp.getOperation());
      if (it != plan.yieldOwner.end())
        llvm::errs() << treePrefix(depth)
                     << "|- scf.yield                            owner "
                     << ownerStr(&op, it->second) << "\n";
      continue;
    }
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
  // v4: the function region is never annotated. Print only the header
  // row with the function name; no entry/exit ownership annotation.
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpOwnershipBlock(b, plan, group, /*depth=*/1);
}

// ---------------------------------------------------------------------------
// RAW-SYNC-DAG dump (commit 3 stage output).
// Same region-tree shape as OWNERSHIP-DAG; raw per-edge semaphore rows
// rendered inline as siblings (acquires before access; releases as
// children after access) or at region entry/exit when the edge crosses
// a control-flow boundary.
// ---------------------------------------------------------------------------

static void renderAcquireRow(unsigned depth, Operation *anchor,
                             const SyncEdge &edge) {
  llvm::errs() << treePrefix(depth) << "|- a  " << edge.name
               << "  acquire " << stateStr(edge.state) << "  "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

static void renderReleaseRow(unsigned depth, Operation *anchor,
                             const SyncEdge &edge) {
  llvm::errs() << treePrefix(depth) << "|- r  " << edge.name
               << "  release " << stateStr(edge.state) << "  "
               << formatOwnerPair(anchor, edge.srcOwner, edge.dstOwner)
               << "\n";
}

// Print inline incoming acquires for the access op (same depth as the
// access row, immediately before it).
static void printInlineIncoming(Operation *op, SyncPlan &sp, unsigned depth) {
  auto it = sp.incomingByAccess.find(op);
  if (it == sp.incomingByAccess.end()) return;
  for (unsigned idx : it->second)
    renderAcquireRow(depth, op, sp.edges[idx]);
}

// Print inline outgoing releases for the access op (depth+1, as children).
static void printInlineOutgoing(Operation *op, SyncPlan &sp, unsigned depth) {
  auto it = sp.outgoingByAccess.find(op);
  if (it == sp.outgoingByAccess.end()) return;
  for (unsigned idx : it->second)
    renderReleaseRow(depth + 1, op, sp.edges[idx]);
}

// Print region-entry edges (release then acquire) at the region's content
// depth.
static void printRegionEntry(Region &region, SyncPlan &sp, unsigned depth) {
  auto it = sp.entryEdges.find(&region);
  if (it == sp.entryEdges.end()) return;
  for (unsigned idx : it->second) {
    const SyncEdge &edge = sp.edges[idx];
    renderReleaseRow(depth, edge.dstOp, edge);
    renderAcquireRow(depth, edge.dstOp, edge);
  }
}

// Print region-exit edges at the region's content depth (used right
// before the region's terminator).
static void printRegionExit(Region &region, SyncPlan &sp, unsigned depth) {
  auto it = sp.exitEdges.find(&region);
  if (it == sp.exitEdges.end()) return;
  for (unsigned idx : it->second) {
    const SyncEdge &edge = sp.edges[idx];
    renderReleaseRow(depth, edge.srcOp, edge);
    renderAcquireRow(depth, edge.dstOp, edge);
  }
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp, BufferGroup &group,
                             unsigned depth);

static void dumpRawSyncRegion(Region &region, SyncPlan &sp,
                              BufferGroup &group, unsigned depth) {
  printRegionEntry(region, sp, depth);
  for (Block &b : region) dumpRawSyncBlock(b, sp, group, depth);
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp, BufferGroup &group,
                             unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Same subtree-has-events gating as OWNERSHIP-DAG.
      bool show = false;
      forOp.walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show && !sp.entryEdges.count(&forOp.getRegion()) &&
          !sp.exitEdges.count(&forOp.getRegion()))
        continue;
      // Initial permit anchored before this for-op?
      if (sp.initialEdgeIdx && sp.initialAnchorBeforeOp == &op &&
          sp.initialAnchorRegion == block.getParent())
        renderAcquireRow(depth, &op, sp.edges[*sp.initialEdgeIdx]);
      llvm::errs() << treePrefix(depth) << "|- " << forOpLabel(forOp)
                   << "                              structural\n";
      dumpRawSyncRegion(forOp.getRegion(), sp, group, depth + 1);
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
      if (!show && !sp.entryEdges.count(&ifOp.getThenRegion()) &&
          !sp.exitEdges.count(&ifOp.getThenRegion()) &&
          !sp.entryEdges.count(&ifOp.getElseRegion()) &&
          !sp.exitEdges.count(&ifOp.getElseRegion()))
        continue;
      if (sp.initialEdgeIdx && sp.initialAnchorBeforeOp == &op &&
          sp.initialAnchorRegion == block.getParent())
        renderAcquireRow(depth, &op, sp.edges[*sp.initialEdgeIdx]);
      llvm::errs() << treePrefix(depth)
                   << "|- scf.if                               structural\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then region\n";
      dumpRawSyncRegion(ifOp.getThenRegion(), sp, group, depth + 2);
      llvm::errs() << treePrefix(depth + 1) << "|- else region\n";
      if (!ifOp.getElseRegion().empty())
        dumpRawSyncRegion(ifOp.getElseRegion(), sp, group, depth + 2);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      // Render any region-exit edges immediately before the terminator.
      Region *parentRegion = yieldOp->getParentRegion();
      if (parentRegion) printRegionExit(*parentRegion, sp, depth);
      continue;
    }
    if (!sp.accessOps.contains(&op)) continue;
    auto useIt = sp.incomingByAccess.find(&op);
    // Initial-permit acquire when this op is the func-direct anchor.
    if (sp.initialEdgeIdx && sp.initialAnchorBeforeOp == &op &&
        sp.initialAnchorRegion == block.getParent())
      renderAcquireRow(depth, &op, sp.edges[*sp.initialEdgeIdx]);
    // Inline incoming acquires (excluding the initial permit row which
    // was just rendered).
    if (useIt != sp.incomingByAccess.end()) {
      for (unsigned idx : useIt->second) {
        const SyncEdge &edge = sp.edges[idx];
        if (!edge.srcOp) continue; // initial permit handled above
        renderAcquireRow(depth, &op, edge);
      }
    }
    // Find the touch + render the access row.
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    const AccessTouch *touch =
        event ? findTouchForResource(*event, sp.resource.second) : nullptr;
    if (touch) {
      bool reads = touchReads(*touch);
      bool writes = touchWrites(*touch);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
    // Outgoing releases as children.
    printInlineOutgoing(&op, sp, depth);
  }
}

static void dumpRawSyncDag(SyncPlan &sp, BufferGroup &group,
                           triton::FuncOp funcOp) {
  llvm::errs() << "RAW-SYNC-DAG buffer.id=" << sp.resource.first
               << " resourceKey=" << sp.resource.second << " edges="
               << sp.edges.size() << "\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, group, /*depth=*/1);
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
      dumpRawSyncDag(sp, group, funcOp);
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
