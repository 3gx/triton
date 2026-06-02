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
  if (!op) return AsyncOp::NONE;
  if (auto localAlloc = dyn_cast<LocalAllocOp>(op))
    if (Value src = localAlloc.getSrc())
      if (Operation *def = src.getDefiningOp())
        return getAsyncPayload(def);
  if (isa<MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  StringRef name = op->getName().getStringRef();
  if (name == "tt.descriptor_load" || name == "tt.descriptor_gather" ||
      name == "nvws.descriptor_load" || name == "nvws.descriptor_gather")
    return AsyncOp::TMALoad;
  return AsyncOp::NONE;
}

static RankedTensorType getTensorTypeFromScalar(OpBuilder &builder,
                                                Value scalar) {
  auto mod = scalar.getParentRegion()->getParentOfType<ModuleOp>();
  auto nWarps = lookupNumWarps(mod);
  auto threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  int CTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
  Attribute encoding = getDefaultBlockedEncoding(builder.getContext(), {1},
                                                 nWarps, threadsPerWarp, CTAs);
  return RankedTensorType::get({1}, scalar.getType(), encoding);
}

static int getTxCount(Operation *descOp) {
  auto getTensorTypeAndDesc =
      [](Operation *op) -> std::pair<RankedTensorType, Value> {
    if (auto loadOp = dyn_cast<triton::DescriptorLoadOp>(op))
      return {loadOp.getType(), loadOp.getDesc()};
    if (auto gatherOp = dyn_cast<triton::DescriptorGatherOp>(op))
      return {gatherOp.getType(), gatherOp.getDesc()};
    llvm_unreachable("unsupported descriptor operation type");
  };
  auto [tensorType, desc] = getTensorTypeAndDesc(descOp);
  auto encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         getIntOrFloatOrPtrBitWidth(tensorType.getElementType()) / 8;
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

static bool sameOwner(const std::optional<PartitionId> &a,
                      const std::optional<PartitionId> &b);

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
  // All per-member touches that hit this resource. Mixed-effect terminal ops can
  // read one member and write another member in the same physical conflict class.
  DenseMap<Operation *, SmallVector<unsigned, 2>> useTouchIdxs;
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

enum class SyncEdgeKind { Ready, Done, Handoff };

struct ProducedVersionKey {
  int64_t resourceKey = 0;
  unsigned epoch = 0;
  bool operator==(const ProducedVersionKey &other) const {
    return resourceKey == other.resourceKey && epoch == other.epoch;
  }
};

enum class CarrierTokenChoice { None, Source, Destination };

// One raw cross-owner edge. Both src and dst can be either a real access
// op or a virtual marker (region op for entering, region's YIELD for
// leaving). The renderer decides where to place release/acquire rows
// based on src/dst classification.
//
// srcOp / srcYieldRegion identify the SOURCE anchor (the producer event
// or region transition where the writer became `srcOwner`). srcEpoch is
// a monotonically increasing tag bumped every time the unified walk
// updates the writer; edges with the same (srcOwner, srcEpoch) provably
// share the same producer event, which is what Combine A (Ready Fanout)
// keys on at the opt-sync stage.
struct SyncEdge {
  std::string name;            // "S_e<N>"
  SyncEdgeKind kind = SyncEdgeKind::Handoff;
  ProducedVersionKey producedVersion;
  Operation *srcOp = nullptr;  // real access OR region op
  Operation *dstOp = nullptr;  // real access OR region op
  // If non-null, src/dst is a YIELD virtual row for this region (the
  // edge anchors at the region's YIELD point rather than at a real op).
  Region *srcYieldRegion = nullptr;
  Region *dstYieldRegion = nullptr;
  std::optional<PartitionId> srcOwner;
  std::optional<PartitionId> dstOwner;
  unsigned srcEpoch = 0;
  RegionOwnership srcRegionOwner;
  RegionOwnership dstRegionOwner;
  std::optional<PartitionId> preOwner;
  std::optional<PartitionId> postOwner;
  bool releaseOutsideControlFlow = false;
  bool acquireOutsideControlFlow = false;
  bool forceFullSemaphore = false;
  CarrierTokenChoice carrierChoice = CarrierTokenChoice::None;
  StageCluster releaseStageCluster;
  StageCluster acquireStageCluster;
  AsyncOp asyncPayload = AsyncOp::NONE;
  SmallVector<AccessTouch, 2> touches;
};

struct SyncPlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncEdge, 0> edges;
  // For each edge, both the release and the acquire render at the SAME
  // anchor point: right before the destination's row, at the
  // destination's tree depth. The anchor is keyed by either the
  // destination op (real access or region op) or the destination region
  // (for a YIELD-anchored close edge).
  DenseMap<Operation *, SmallVector<unsigned, 2>> beforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> beforeYield;
  // Ops with at least one touch for this resource (access rows in dump).
  DenseSet<Operation *> accessOps;
  // Initial writable permit — RAW-SYNC-DAG dump only; does NOT affect emit.
  // Rendered as a standalone `a <name> acquire root` before
  // `initialPermitBeforeOp`. Cyclic chains reuse the loop-carry back edge;
  // acyclic chains mint a fresh counter and add a terminal
  // `r <name> release root` after `initialPermitReleaseAfterOp`.
  std::string initialPermitName;
  Operation *initialPermitBeforeOp = nullptr;
  Operation *initialPermitReleaseAfterOp = nullptr;
  // Index of the edge whose counter the (cyclic) initial permit reuses, or -1
  // when the permit is a freshly minted counter. The OPT-SYNC-DAG dump uses it
  // to render the entry acquire with the combined name (S_full/S_empty) when
  // that reused edge is itself part of a fan-in/out combine.
  int initialPermitEdgeIdx = -1;
};

static bool touchReads(const AccessTouch &t) { return hasRead(t.effect); }
static bool touchWrites(const AccessTouch &t) { return hasWrite(t.effect); }

static void collectTouchesForResource(
    const AccessEvent &event, int64_t resourceKey,
    SmallVectorImpl<const AccessTouch *> &touches) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey)
      touches.push_back(&t);
}

static bool eventTouchesResource(const AccessEvent &event, int64_t resourceKey) {
  return llvm::any_of(event.touches, [&](const AccessTouch &t) {
    return t.resourceKey == resourceKey;
  });
}

// True iff `event` reads/writes the resource identified by `resourceKey`.
static bool eventConsumes(const AccessEvent &event, int64_t resourceKey) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey && touchReads(t))
      return true;
  return false;
}
static bool eventProduces(const AccessEvent &event, int64_t resourceKey) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey && touchWrites(t))
      return true;
  return false;
}

static bool sameOwner(const std::optional<PartitionId> &a,
                      const std::optional<PartitionId> &b);

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
  auto emitClose = [&](State &state, const Owner &P, Operation *anchorOp,
                       Region *anchorYieldRegion, bool isW,
                       unsigned anchorRank) -> bool {
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
      // Handoff: W always fires; structural row fires only if a real
      // access exists downstream (release-into-void suppression).
      if (isW || hasDownstreamAccess(anchorRank)) {
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
          // Transition at the body's YIELD row (loop-carry close).
          Operation *yieldOp = forOp.getRegion().front().getTerminator();
          unsigned yieldRank = opRank.lookup(yieldOp);
          emitClose(state, pp, /*anchorOp=*/nullptr, &forOp.getRegion(),
                    /*isW=*/false, yieldRank);
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

enum class SyncGroupKind {
  InitialEmpty,
  Singleton,
  ReadyFanout,
  DoneFanin,
  LinearChain
};

struct SyncGroup {
  std::string name;            // "S_g<N>"
  SyncGroupKind kind = SyncGroupKind::Singleton;
  SmallVector<unsigned, 4> edgeIdxs; // indexes into SyncPlan::edges
  Operation *initialOp = nullptr;
  std::optional<PartitionId> initialOwner;
};

struct PlannedRelease {
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> edgeIdxs;
};

static bool operator==(const PlannedRelease &lhs,
                       const PlannedRelease &rhs) {
  return lhs.groupIdx == rhs.groupIdx && lhs.edgeIdxs == rhs.edgeIdxs;
}

struct OptSyncDag {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncGroup> groups;
  // Which raw edge belongs to which opt group (parallel to sp.edges).
  SmallVector<unsigned> edgeToGroup;
  // For Singleton/LinearChain edges: does the edge ride the EMPTY semaphore
  // (true) or the FULL semaphore (false)? Computed once with `group` in
  // buildOptSyncDag so the dump renderer can label rows S_empty/S_full to
  // match the emitted semaphores (and unify the cyclic back-edge with the
  // initial writable permit). Absent ⇒ not an EMPTY edge.
  DenseMap<unsigned, bool> edgeRendersEmpty;
  // Where to render the release row(s).
  //   Singleton:    `releaseBeforeOp[dstOp]`        (release+acquire pair
  //                 at dst — matches RAW shape).
  //   ReadyFanout:  `releaseAfterOp[srcOp]`         (one row right after
  //                 the producer access).
  //   DoneFanin:    `releaseAfterOp[eachSrcOp]`     (one row right after
  //                 each retiring reader access).
  DenseMap<Operation *, SmallVector<PlannedRelease, 2>> releaseBeforeOp;
  DenseMap<Region *, SmallVector<PlannedRelease, 2>> releaseBeforeYield;
  DenseMap<Operation *, SmallVector<PlannedRelease, 2>> releaseAfterOp;
  DenseMap<Region *, SmallVector<PlannedRelease, 2>> releaseAfterYield;
  // Where to render the acquire row(s) — always before the consumer.
  //   Singleton:    `acquireBeforeOp[dstOp]`.
  //   ReadyFanout:  `acquireBeforeOp[eachDstOp]`.
  //   DoneFanin:    `acquireBeforeOp[sharedDstOp]`.
  DenseMap<Operation *, SmallVector<unsigned, 2>> acquireBeforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> acquireBeforeYield;
  DenseMap<unsigned, Operation *> ifYieldJoinAccess;
  DenseMap<unsigned, Operation *> dstBranchEntryAnchor;
  DenseMap<unsigned, Operation *> tmemLoopExitRead;
  DenseMap<unsigned, Operation *> loopEntryHandoffAccess;
  DenseMap<unsigned, Region *> skippedInitialLoopCarrierRegion;
  DenseSet<unsigned> edgesDeferringToSkippedLoopExit;
  DenseSet<unsigned> terminalLoopReadEdgesDeferringToExit;
  DenseSet<unsigned> srcYieldParentWarpFor;
  DenseSet<Operation *> accessOps;
  // Planned CFG token-threading requirements. These are derived from OPT-SYNC
  // anchors at structured op/yield boundaries before emission; the emitter only
  // materializes the planned iter_args/results.
  DenseSet<Operation *> threadForOps;
  DenseSet<Operation *> threadIfOps;
};

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

// Key for Combine A: (srcOwner, srcEpoch, srcAnchor). Two edges with
// the same key provably came from the same writer event.
namespace {
struct ReadyFanoutKey {
  std::optional<PartitionId> owner;
  unsigned epoch = 0;
  Operation *srcOp = nullptr;
  Region *srcYield = nullptr;
  bool operator==(const ReadyFanoutKey &o) const {
    return owner == o.owner && epoch == o.epoch && srcOp == o.srcOp &&
           srcYield == o.srcYield;
  }
};
struct DoneFaninKey {
  std::optional<PartitionId> owner;
  Operation *dstOp = nullptr;
  Region *dstYield = nullptr;
  bool operator==(const DoneFaninKey &o) const {
    return owner == o.owner && dstOp == o.dstOp && dstYield == o.dstYield;
  }
};
} // namespace

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
static bool linearChainEdgeUsesEmpty(const SyncGroup &syncGroup,
                                     const SyncPlan &sp, const OptSyncDag &dag,
                                     const SyncEdge *edge, BufferGroup &group,
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
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::LinearChain;
    g.edgeIdxs.append(linearChain.begin(), linearChain.end());
    for (unsigned e : g.edgeIdxs) claimed[e] = true;
    dag.groups.push_back(std::move(g));
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

  // Per-edge EMPTY/FULL classification for the dump renderer, matching the
  // emitter's semaphore choice (getSemaphoreForGroup). Computed here because
  // `group` is in scope; the renderers only have the DAG.
  for (const SyncGroup &syncGroup : dag.groups) {
    bool isChain = syncGroup.kind == SyncGroupKind::LinearChain;
    bool isSingleton = syncGroup.kind == SyncGroupKind::Singleton;
    if (!isChain && !isSingleton)
      continue;
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      bool usesEmpty =
          isChain ? linearChainEdgeUsesEmpty(syncGroup, sp, dag,
                                             &sp.edges[edgeIdx], group,
                                             sp.resource.second)
                  : edgeUsesEmpty(sp.edges[edgeIdx], group, sp.resource.second);
      dag.edgeRendersEmpty[edgeIdx] = usesEmpty;
    }
  }
  return dag;
}

// ---------------------------------------------------------------------------
// v4 commit 5 — semaphore IR emission.
// ---------------------------------------------------------------------------

static bool shouldDumpDag() {
  const char *value = std::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  if (!value) return false;
  std::string s(value);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s == "1" || s == "true" || s == "on";
}

template <typename OpT, typename... Args>
static OpT createIntoPartition(
    OpBuilder &b, Location loc,
    std::pair<std::optional<PartitionId>, StageCluster> partitionIdStageCluster,
    Args &&...args) {
  std::optional<SetVector<int>> partitionIds = SetVector<int>();
  std::optional<int> wsTag;
  if (partitionIdStageCluster.first) {
    auto [id, tag] = *partitionIdStageCluster.first;
    wsTag = tag;
    partitionIds->insert(id);
  } else {
    partitionIds = std::nullopt;
  }
  auto op = triton::gpu::createInto<OpT>(b, loc, partitionIds,
                                         partitionIdStageCluster.second,
                                         std::forward<Args>(args)...);
  if (wsTag) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !hasWarpSpecializeTag(forOp))
      forOp = forOp->template getParentOfType<scf::ForOp>();
    if (!forOp)
      setWarpSpecializeTag(op, *wsTag);
  }
  return op;
}

static void copyBufferAttrs(Operation *src, Operation *dst) {
  for (StringRef attrName :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName}) {
    if (Attribute attr = src->getAttr(attrName))
      dst->setAttr(attrName, attr);
  }
}

static Operation *getSemaphoreInsertionAnchor(BufferGroup &group) {
  Operation *anchor = group.members.front().allocOp;
  auto outerWsLoop = anchor->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  return outerWsLoop ? outerWsLoop.getOperation() : anchor;
}

static Operation *getLocalSemaphoreCreateAnchor(BufferGroup &group) {
  Operation *anchor = group.members.front().allocOp;
  Block *block = anchor->getBlock();
  for (BufferMember &member : group.members) {
    if (member.allocOp->getBlock() != block)
      continue;
    if (anchor->isBeforeInBlock(member.allocOp))
      anchor = member.allocOp;
  }
  return anchor;
}

static bool canDoubleBufferAcc(MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  auto blockM = tmemDesc.getShape()[0];
  auto blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + (blockM * blockN * 2) > numTMEMRows * numTMEMColumns)
    return false;
  if (isa<TCGen5MMAScaledOp>(mmaOp) && blockN == 256)
    return false;
  return true;
}

static int computeTmemSemaphoreNumStages(BufferGroup &group, int numTmemBlocks,
                                         bool useMetaPartitioner) {
  bool isMultiStaged = true;
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
  auto numStages =
      useMetaPartitioner ? 1 + 0 * isMultiStaged : 1 + 1 * isMultiStaged;
  return numStages;
}

static void updateNumTmemBlocks(BufferGroup &group, int numStages,
                                int &numTmemBlocks) {
  for (BufferMember &member : group.members) {
    auto shape = member.type.getShape();
    if (shape.size() >= 2)
      numTmemBlocks += shape[0] * shape[1] * numStages;
  }
}

struct GroupBacking {
  SmallVector<unsigned, 4> memberIndices;
  DenseMap<unsigned, unsigned> memberToBackingIndex;
  SmallVector<Value, 4> buffers;
  SmallVector<Type, 4> bufferTypes;
  Value sharedEmptySemaphore;
  Value sharedFullSemaphore;
};

using BackingKey = std::pair<unsigned /*groupIdx*/, int64_t /*resourceKey*/>;

static GroupBacking &
ensureGroupBacking(BufferGroup &group, unsigned groupIdx,
                   int64_t resourceKey, ArrayRef<unsigned> memberIndices,
                   DenseMap<BackingKey, GroupBacking> &backings,
                   const DenseMap<unsigned, int> &numStagesByGroup) {
  BackingKey key{groupIdx, resourceKey};
  auto it = backings.find(key);
  if (it != backings.end())
    return it->second;

  GroupBacking backing;
  backing.memberIndices.append(memberIndices.begin(), memberIndices.end());
  OpBuilder b(getSemaphoreInsertionAnchor(group));
  b.setInsertionPoint(getSemaphoreInsertionAnchor(group));
  int depth = 1;
  if (group.isTmem()) {
    auto it = numStagesByGroup.find(groupIdx);
    assert(it != numStagesByGroup.end() &&
           "TMEM semaphore numStages must be precomputed before emission");
    depth = it->second;
  }
  for (auto [backingIdx, memberIdx] : llvm::enumerate(memberIndices)) {
    BufferMember &member = group.members[memberIdx];
    backing.memberToBackingIndex[memberIdx] = static_cast<unsigned>(backingIdx);
    MemDescType semBufType = member.type;
    if (group.isTmem()) {
      semBufType = getSemaphoreMultiBufferedType(member.type, depth);
      Operation *semAlloc = createAlloc(b, member.allocOp->getLoc(), semBufType,
                                        Value());
      semAlloc->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      copyBufferAttrs(member.allocOp, semAlloc);
      backing.buffers.append(semAlloc->getResults().begin(),
                             semAlloc->getResults().end());
    } else if (auto localAlloc = dyn_cast<LocalAllocOp>(member.allocOp);
               localAlloc && !localAlloc.getSrc()) {
      member.allocOp->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      backing.buffers.push_back(member.value);
    } else {
      semBufType = getMultiBufferedType(member.type, depth);
      Operation *semAlloc = createAlloc(b, member.allocOp->getLoc(), semBufType,
                                        Value());
      semAlloc->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      copyBufferAttrs(member.allocOp, semAlloc);
      backing.buffers.append(semAlloc->getResults().begin(),
                             semAlloc->getResults().end());
    }
    backing.bufferTypes.push_back(semBufType);
  }
  auto inserted = backings.insert({key, std::move(backing)});
  return inserted.first->second;
}

static Value createSemaphore(OpBuilder &b, Location loc,
                             const GroupBacking &backing, bool released) {
  auto baseTypes = TypeArrayAttr::get(b.getContext(), backing.bufferTypes);
  auto semaTy = SemaphoreType::get(b.getContext(), baseTypes);
  auto op = SemaphoreCreateOp::create(b, loc, semaTy, backing.buffers, released);
  return op.getResult();
}

static void setPartitionFromAnchor(Operation *op, Operation *anchor);

struct SyncGroupSemaphores {
  Value empty;
  Value full;
  DenseMap<unsigned, Value> fullByEdge;
};

struct ResourceSemaphores {
  DenseMap<unsigned, SyncGroupSemaphores> byGroup;
};

static std::optional<unsigned> findEdgeIndex(const SyncPlan &sp,
                                             const SyncEdge *edge) {
  if (!edge)
    return std::nullopt;
  for (auto [idx, candidate] : llvm::enumerate(sp.edges))
    if (&candidate == edge)
      return static_cast<unsigned>(idx);
  return std::nullopt;
}

static const AccessEvent *findEvent(BufferGroup &group, Operation *op) {
  for (AccessEvent &event : group.events)
    if (event.op == op)
      return &event;
  return nullptr;
}

static bool edgeDstWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (!edge.dstOp) return false;
  const AccessEvent *event = findEvent(group, edge.dstOp);
  return event && eventProduces(*event, resourceKey);
}

static bool edgeDstReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey) {
  if (!edge.dstOp) return false;
  const AccessEvent *event = findEvent(group, edge.dstOp);
  return event && eventConsumes(*event, resourceKey);
}

static bool edgeSrcReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey) {
  if (!edge.srcOp) return false;
  const AccessEvent *event = findEvent(group, edge.srcOp);
  return event && eventConsumes(*event, resourceKey);
}

static bool edgeSrcWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (!edge.srcOp) return false;
  const AccessEvent *event = findEvent(group, edge.srcOp);
  return event && eventProduces(*event, resourceKey);
}

static bool isRootToWsLoopEntryEdge(const SyncEdge &edge, BufferGroup &group) {
  if (!group.isTmem() || edge.srcOwner || !edge.dstOwner)
    return false;
  auto forOp = dyn_cast_or_null<scf::ForOp>(edge.dstOp);
  return forOp && hasWarpSpecializeTag(forOp.getOperation());
}

static bool edgeStartsAtRootTmemStoreInitializer(const SyncEdge &edge,
                                                 BufferGroup &group,
                                                 int64_t resourceKey) {
  if (!group.isTmem() || edge.srcOwner || !isa_and_nonnull<TMEMStoreOp>(edge.srcOp))
    return false;
  for (const AccessEvent &event : group.events) {
    if (!eventProduces(event, resourceKey))
      continue;
    return event.op == edge.srcOp && !event.owner;
  }
  return false;
}

static bool edgeUsesEmpty(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (edge.forceFullSemaphore)
    return false;
  if (edgeStartsAtRootTmemStoreInitializer(edge, group, resourceKey))
    return true;
  if (edge.kind == SyncEdgeKind::Done)
    return true;
  if (edge.kind == SyncEdgeKind::Ready)
    return false;
  if (group.isTmem() && isa_and_nonnull<MMAv5OpInterface>(edge.srcOp) &&
      !edge.dstOp && edge.dstYieldRegion && edge.dstOwner)
    return true;
  if (edgeSrcWrites(edge, group, resourceKey) &&
      edgeDstWrites(edge, group, resourceKey))
    return false;
  if (edgeDstWrites(edge, group, resourceKey))
    return true;
  if (edgeDstReads(edge, group, resourceKey))
    return false;
  if (edgeSrcReads(edge, group, resourceKey) &&
      !edgeSrcWrites(edge, group, resourceKey))
    return true;
  return false;
}

static bool edgeNeedsTerminalReadRelease(const SyncEdge &edge,
                                         BufferGroup &group,
                                         int64_t resourceKey) {
  return edge.dstOp && edgeDstReads(edge, group, resourceKey) &&
         !edgeDstWrites(edge, group, resourceKey) &&
         isa<TMEMLoadOp, LocalLoadOp>(edge.dstOp);
}

static bool opReadsOnlyResource(Operation *op, BufferGroup &group,
                                int64_t resourceKey) {
  const AccessEvent *event = findEvent(group, op);
  return event && eventConsumes(*event, resourceKey) &&
         !eventProduces(*event, resourceKey);
}

static bool operationOccursAtOrAfterInSequentialScope(Operation *anchor,
                                                      Operation *candidate);

static bool hasLaterSameOwnerResourceAccessBeforeOwnerChange(
    Operation *op, BufferGroup &group, int64_t resourceKey) {
  const AccessEvent *anchor = findEvent(group, op);
  if (!anchor || !eventTouchesResource(*anchor, resourceKey))
    return false;

  for (const AccessEvent &event : group.events) {
    if (!eventTouchesResource(event, resourceKey))
      continue;
    if (event.op == op) {
      continue;
    }
    if (!operationOccursAtOrAfterInSequentialScope(op, event.op))
      continue;
    if (!sameOwner(event.owner, anchor->owner))
      return false;
    return true;
  }
  return false;
}

static bool operationOccursAtOrAfterInSequentialScope(Operation *anchor,
                                                      Operation *candidate) {
  if (!anchor || !candidate)
    return false;
  if (anchor == candidate)
    return true;
  for (Operation *anchorScope = anchor; anchorScope;
       anchorScope = anchorScope->getParentOp()) {
    Block *block = anchorScope->getBlock();
    if (!block)
      continue;
    for (Operation *candidateScope = candidate; candidateScope;
         candidateScope = candidateScope->getParentOp()) {
      if (candidateScope->getBlock() != block)
        continue;
      if (anchorScope == candidateScope)
        return false;
      return anchorScope->isBeforeInBlock(candidateScope);
    }
  }
  return false;
}

static bool regionContainsOp(Region *region, Operation *op) {
  if (!region || !op)
    return false;
  for (Region *parentRegion = op->getParentRegion(); parentRegion;) {
    if (parentRegion == region)
      return true;
    Operation *parentOp = parentRegion->getParentOp();
    parentRegion = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return false;
}

static bool yieldOccursAtOrAfterInSequentialScope(Region *yieldRegion,
                                                  Operation *anchor) {
  if (!yieldRegion || !anchor)
    return false;
  Operation *parent = yieldRegion->getParentOp();
  if (!parent)
    return false;
  if (regionContainsOp(yieldRegion, anchor))
    return true;
  return operationOccursAtOrAfterInSequentialScope(anchor, parent);
}

static bool ownerHasPlannedOutgoingEdgeAtOrAfterRead(
    Operation *op, const SyncPlan &sp, BufferGroup &group,
    int64_t resourceKey) {
  const AccessEvent *anchor = findEvent(group, op);
  if (!anchor || !eventTouchesResource(*anchor, resourceKey))
    return false;
  for (const SyncEdge &edge : sp.edges) {
    if (!sameOwner(edge.srcOwner, anchor->owner))
      continue;
    if (edge.srcOp &&
        operationOccursAtOrAfterInSequentialScope(op, edge.srcOp))
      return true;
    if (edge.srcYieldRegion &&
        yieldOccursAtOrAfterInSequentialScope(edge.srcYieldRegion, op))
      return true;
  }
  return false;
}

static Operation *findTerminalReadReleaseAnchor(Operation *op,
                                                const SyncPlan &sp,
                                                BufferGroup &group,
                                                int64_t resourceKey) {
  if (!opReadsOnlyResource(op, group, resourceKey))
    return nullptr;
  if (hasLaterSameOwnerResourceAccessBeforeOwnerChange(op, group, resourceKey))
    return nullptr;
  if (ownerHasPlannedOutgoingEdgeAtOrAfterRead(op, sp, group, resourceKey))
    return nullptr;
  return op;
}

static bool linearChainNeedsPerEdgeFulls(const SyncGroup &syncGroup,
                                         const SyncPlan &sp,
                                         BufferGroup &group,
                                         int64_t resourceKey) {
  if (!group.isTmem())
    return false;
  std::optional<int64_t> firstOffset;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    for (const AccessTouch &touch : sp.edges[edgeIdx].touches) {
      if (touch.resourceKey != resourceKey ||
          touch.memberIdx >= group.members.size())
        continue;
      int64_t offset = group.members[touch.memberIdx].offset;
      if (!firstOffset) {
        firstOffset = offset;
        continue;
      }
      if (*firstOffset != offset)
        return true;
    }
  }
  return false;
}

static bool linearChainEdgeUsesEmpty(const SyncGroup &syncGroup,
                                     const SyncPlan &sp,
                                     const OptSyncDag &dag,
                                     const SyncEdge *edge, BufferGroup &group,
                                     int64_t resourceKey) {
  if (!edge)
    return false;
  unsigned ifYieldEdges = 0;
  for (unsigned edgeIdx : syncGroup.edgeIdxs)
    if (dag.ifYieldJoinAccess.contains(edgeIdx))
      ++ifYieldEdges;
  if (ifYieldEdges > 1)
    return sameOwner(edge->dstOwner,
                     sp.edges[syncGroup.edgeIdxs.front()].srcOwner);
  if (linearChainNeedsPerEdgeFulls(syncGroup, sp, group, resourceKey))
    return edgeUsesEmpty(*edge, group, resourceKey);
  std::optional<unsigned> currentEdgeIdx = findEdgeIndex(sp, edge);
  bool skipsInitialLoopCarrier =
      currentEdgeIdx &&
      dag.skippedInitialLoopCarrierRegion.contains(
          dag.edgeToGroup[*currentEdgeIdx]);
  bool loopEntryHandoff =
      llvm::any_of(syncGroup.edgeIdxs, [&](unsigned edgeIdx) {
        return dag.loopEntryHandoffAccess.contains(edgeIdx);
      });
  for (auto [pos, edgeIdx] : llvm::enumerate(syncGroup.edgeIdxs))
    if (&sp.edges[edgeIdx] == edge) {
      if (loopEntryHandoff)
        return (pos % 2) == 0;
      if (skipsInitialLoopCarrier && pos > 0)
        return (pos % 2) == 0;
      return (pos % 2) == 1;
    }
  return false;
}

static ResourceSemaphores createResourceSemaphores(const OptSyncDag &dag,
                                                   const SyncPlan &sp,
                                                   BufferGroup &group,
                                                   GroupBacking &backing) {
  Operation *anchor = group.isTmem() ? getSemaphoreInsertionAnchor(group)
                                     : getLocalSemaphoreCreateAnchor(group);
  if (!group.isTmem()) {
    for (Value buffer : backing.buffers) {
      Operation *def = buffer.getDefiningOp();
      if (!def)
        continue;
      if (!anchor || anchor->getBlock() != def->getBlock()) {
        anchor = def;
        continue;
      }
      if (anchor->isBeforeInBlock(def))
        anchor = def;
    }
  }
  OpBuilder b(anchor);
  if (group.isTmem())
    b.setInsertionPoint(anchor);
  else
    b.setInsertionPointAfter(anchor);
  ResourceSemaphores semas;
  Location loc = group.members.front().allocOp->getLoc();
  std::optional<PartitionId> initialEmptyOwner;
  for (const SyncGroup &syncGroup : dag.groups) {
    if (syncGroup.kind != SyncGroupKind::InitialEmpty)
      continue;
    initialEmptyOwner = syncGroup.initialOwner;
    break;
  }

  Value localSharedEmpty;
  auto createSharedEmpty = [&]() -> Value {
    Value &sharedEmpty =
        group.isTmem() ? backing.sharedEmptySemaphore : localSharedEmpty;
    if (sharedEmpty)
      return sharedEmpty;
    sharedEmpty = createSemaphore(b, loc, backing, /*released=*/true);
    if (!group.isTmem())
      setPartitionFromAnchor(sharedEmpty.getDefiningOp(), anchor);
    return sharedEmpty;
  };
  auto createFull = [&]() -> Value {
    Value full = createSemaphore(b, loc, backing, /*released=*/false);
    if (!group.isTmem())
      setPartitionFromAnchor(full.getDefiningOp(), anchor);
    return full;
  };
  auto createSharedFull = [&]() -> Value {
    if (!group.isTmem())
      return createFull();
    if (backing.sharedFullSemaphore)
      return backing.sharedFullSemaphore;
    backing.sharedFullSemaphore =
        createSemaphore(b, loc, backing, /*released=*/false);
    return backing.sharedFullSemaphore;
  };

  for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
    unsigned groupIdx = static_cast<unsigned>(idx);
    SyncGroupSemaphores pair;
    switch (syncGroup.kind) {
    case SyncGroupKind::InitialEmpty:
    case SyncGroupKind::DoneFanin:
      pair.empty = createSharedEmpty();
      break;
    case SyncGroupKind::ReadyFanout:
      pair.full = createFull();
      break;
    case SyncGroupKind::LinearChain:
      pair.empty = createSharedEmpty();
      if (linearChainNeedsPerEdgeFulls(syncGroup, sp, group,
                                       dag.resource.second)) {
        for (unsigned edgeIdx : syncGroup.edgeIdxs) {
          const SyncEdge &edge = sp.edges[edgeIdx];
          if (!linearChainEdgeUsesEmpty(syncGroup, sp, dag, &edge, group,
                                        dag.resource.second))
            pair.fullByEdge[edgeIdx] = createFull();
        }
        if (!pair.fullByEdge.empty())
          pair.full = pair.fullByEdge.begin()->second;
        break;
      }
      pair.full = createFull();
      break;
    case SyncGroupKind::Singleton: {
      const SyncEdge *edge = syncGroup.edgeIdxs.empty()
                                 ? nullptr
                                 : &sp.edges[syncGroup.edgeIdxs.front()];
      if (edge && edgeUsesEmpty(*edge, group, dag.resource.second)) {
        if (sameOwner(edge->dstOwner, initialEmptyOwner))
          pair.empty = createSharedEmpty();
        else
          pair.empty = createFull();
      } else {
        pair.full = createFull();
        std::optional<unsigned> edgeIdx = findEdgeIndex(sp, edge);
        if (edge &&
            ((edgeNeedsTerminalReadRelease(*edge, group, dag.resource.second) &&
              findTerminalReadReleaseAnchor(edge->dstOp, sp, group,
                                            dag.resource.second)) ||
             (edgeIdx && dag.tmemLoopExitRead.contains(*edgeIdx))))
          pair.empty = createSharedEmpty();
      }
      break;
    }
    }
    semas.byGroup[groupIdx] = pair;
  }
  return semas;
}

enum class SyncAnchorKind {
  AcquireBeforeOp,
  AcquireBeforeYield,
  ReleaseBeforeOp,
  ReleaseBeforeYield,
  ReleaseAfterOp,
  ReleaseAfterYield
};

static const SyncEdge *findEdgeForAnchor(const SyncGroup &group,
                                         const SyncPlan &sp,
                                         const OptSyncDag &dag,
                                         SyncAnchorKind kind,
                                         Operation *anchor,
                                         Region *yieldRegion,
                                         Operation *liveAnchor = nullptr) {
  if (group.kind == SyncGroupKind::InitialEmpty)
    return nullptr;
  if (group.kind == SyncGroupKind::ReadyFanout &&
      (kind == SyncAnchorKind::ReleaseAfterOp ||
       kind == SyncAnchorKind::ReleaseAfterYield))
    return &sp.edges[group.edgeIdxs.front()];
  if (group.kind == SyncGroupKind::DoneFanin &&
      (kind == SyncAnchorKind::AcquireBeforeOp ||
       kind == SyncAnchorKind::AcquireBeforeYield))
    return &sp.edges[group.edgeIdxs.front()];

  for (unsigned ei : group.edgeIdxs) {
    const SyncEdge &edge = sp.edges[ei];
    auto dstBranchIt = dag.dstBranchEntryAnchor.find(ei);
    Operation *dstBranchEntry =
        dstBranchIt == dag.dstBranchEntryAnchor.end() ? nullptr
                                                      : dstBranchIt->second;
    switch (kind) {
    case SyncAnchorKind::AcquireBeforeOp:
      if (dag.loopEntryHandoffAccess.lookup(ei) == anchor)
        return &edge;
      if (edge.dstOp == anchor || dstBranchEntry == anchor)
        return &edge;
      break;
    case SyncAnchorKind::AcquireBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeOp:
      if (dag.loopEntryHandoffAccess.lookup(ei) == anchor)
        return &edge;
      if (edge.dstOp == anchor || dstBranchEntry == anchor)
        return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterOp:
      if (edge.srcOp == anchor)
        return &edge;
      if (Operation *ancestorAnchor = liveAnchor ? liveAnchor : anchor)
        if (isa<scf::ForOp>(ancestorAnchor) && edge.srcOp &&
            ancestorAnchor->isProperAncestor(edge.srcOp) &&
            (!edge.dstOp || !ancestorAnchor->isProperAncestor(edge.dstOp)))
          return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterYield:
      if (edge.srcYieldRegion == yieldRegion) return &edge;
      break;
    }
  }
  if (group.kind == SyncGroupKind::LinearChain &&
      (kind == SyncAnchorKind::ReleaseBeforeOp ||
       kind == SyncAnchorKind::AcquireBeforeOp) &&
      anchor) {
    for (unsigned ei : group.edgeIdxs) {
      const SyncEdge &edge = sp.edges[ei];
      auto joinIt = dag.ifYieldJoinAccess.find(ei);
      if (joinIt != dag.ifYieldJoinAccess.end() && joinIt->second == anchor)
        return &edge;
    }
  }
  if (group.kind == SyncGroupKind::LinearChain &&
      kind == SyncAnchorKind::ReleaseAfterOp && anchor) {
    for (unsigned ei : group.edgeIdxs) {
      const SyncEdge &edge = sp.edges[ei];
      if (edge.dstOp == anchor)
        return &edge;
    }
  }
  return group.edgeIdxs.empty() ? nullptr : &sp.edges[group.edgeIdxs.front()];
}

static Value getSemaphoreForGroup(unsigned groupIdx, const SyncEdge *edge,
                                  const OptSyncDag &dag, const SyncPlan &sp,
                                  BufferGroup &group,
                                  ResourceSemaphores &semas) {
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  SyncGroupSemaphores pair = semas.byGroup.lookup(groupIdx);
  switch (syncGroup.kind) {
  case SyncGroupKind::InitialEmpty:
    return pair.empty;
  case SyncGroupKind::DoneFanin:
    return pair.empty;
  case SyncGroupKind::ReadyFanout:
    return pair.full;
  case SyncGroupKind::LinearChain:
    if (linearChainEdgeUsesEmpty(syncGroup, sp, dag, edge, group,
                                 dag.resource.second))
      return pair.empty;
    if (edge) {
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        if (&sp.edges[edgeIdx] != edge)
          continue;
        auto it = pair.fullByEdge.find(edgeIdx);
        if (it != pair.fullByEdge.end())
          return it->second;
        break;
      }
    }
    return pair.full;
  case SyncGroupKind::Singleton:
    if (edge && edgeUsesEmpty(*edge, group, dag.resource.second))
      return pair.empty;
    return pair.full;
  }
  llvm_unreachable("unhandled sync group kind");
}

static const SyncEdge *getRepresentativeReleaseEdge(const PlannedRelease &action,
                                                    const SyncPlan &sp) {
  if (action.edgeIdxs.empty())
    return nullptr;
  unsigned edgeIdx = action.edgeIdxs.front();
  if (edgeIdx >= sp.edges.size())
    return nullptr;
  return &sp.edges[edgeIdx];
}

struct AcquireRecord {
  unsigned groupIdx = 0;
  Value semaphore;
  Value token;
  std::optional<PartitionId> owner;
};

struct EmittedSyncRecord {
  unsigned groupIdx = 0;
  SyncAnchorKind kind = SyncAnchorKind::AcquireBeforeOp;
  Operation *anchor = nullptr;
  Region *yieldRegion = nullptr;
  Operation *op = nullptr;
  Value semaphore;
  Value token;
  StageCluster expectedStageCluster;
  SmallVector<unsigned, 4> edgeIdxs;
};

struct EmittedBufferRecord {
  Operation *accessOp = nullptr;
  Operation *retargetOp = nullptr;
  Operation *bufferOp = nullptr;
  Value semaphore;
  Value token;
  Value accessBuffer;
  SmallVector<Value, 4> buffers;
  unsigned memberIdx = 0;
  unsigned backingIdx = 0;
  StageCluster expectedStageCluster;
};

enum class ThreadRecordKind { ForIterArg, ForResult, IfResult };

struct ThreadRecord {
  Operation *op = nullptr;
  Value token;
  std::optional<PartitionId> owner;
  ThreadRecordKind kind = ThreadRecordKind::ForResult;
  Region *plannedRegion = nullptr;
  Region *plannedElseRegion = nullptr;
};

struct PoisonTokenRecord {
  Operation *op = nullptr;
  Operation *insertBefore = nullptr;
};

struct EmitState {
  ResourceSemaphores semas;
  DenseMap<Operation *, Value> eventToken;
  DenseMap<Operation *, Value> eventSemaphore;
  DenseMap<Operation *, SmallVector<Value, 4>> eventBuffers;
  DenseMap<Value, Value> rewrittenAccessValue;
  DenseMap<Operation *, unsigned> reusedForCarrierSlots;
  DenseMap<Operation *, SmallVector<unsigned, 4>> reusedForTokenSlots;
  DenseMap<Operation *, Value> reusedForPoisonTokens;
  SmallVector<Value, 4> currentBuffers;
  DenseSet<Operation *> protectedAccesses;
  DenseSet<Value> knownCarrierTokens;
  SmallVector<EmittedSyncRecord, 8> emittedAcquires;
  SmallVector<EmittedSyncRecord, 8> emittedReleases;
  SmallVector<EmittedBufferRecord, 8> emittedBuffers;
  SmallVector<ThreadRecord, 4> threadedTokens;
  SmallVector<PoisonTokenRecord, 8> poisonTokenResultsAfterEmission;
  SetVector<Operation *> eraseAfterEmission;
  DenseMap<PartitionId, StageCluster> stageCache;
  Value currentToken;
  Value currentSemaphore;
  std::optional<PartitionId> currentOwner;
};

static StageCluster stageForYieldOwner(std::optional<PartitionId> owner,
                                       EmitState &state) {
  if (!owner) return std::nullopt;
  auto it = state.stageCache.find(*owner);
  return it == state.stageCache.end() ? StageCluster{} : it->second;
}

static SetVector<int> partitionSetForOwner(std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (owner)
    ids.insert(owner->first);
  return ids;
}

static std::optional<SetVector<int>> nearestPartitionIds(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp())
    if (hasPartition(parent))
      return getPartitionIds(parent);
  return std::nullopt;
}

static void addPartitionIds(SetVector<int> &dst, const SetVector<int> &src) {
  dst.insert(src.begin(), src.end());
}

static SetVector<int> partitionSetForValue(Value value) {
  SetVector<int> ids;
  if (!value)
    return ids;
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *def = result.getOwner();
    if (!hasPartition(def))
      return ids;
    if (def->getNumRegions() > 0) {
      auto outputs = getPartitionOutputs(def);
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber < outputs.size())
        addPartitionIds(ids, outputs[resultNumber]);
    }
    if (ids.empty())
      addPartitionIds(ids, getPartitionIds(def));
    return ids;
  }
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return ids;
  Operation *parent = arg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
    if (arg.getOwner() == forOp.getBody() && arg != forOp.getInductionVar()) {
      unsigned iterArgIdx = arg.getArgNumber() - 1;
      if (hasPartition(forOp)) {
        auto outputs = getPartitionOutputs(forOp);
        if (iterArgIdx < outputs.size())
          addPartitionIds(ids, outputs[iterArgIdx]);
        if (ids.empty())
          addPartitionIds(ids, getPartitionIds(forOp));
      }
    }
  } else if (parent && hasPartition(parent)) {
    addPartitionIds(ids, getPartitionIds(parent));
  }
  return ids;
}

static std::optional<int> wsTagForValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto result = dyn_cast<OpResult>(value))
    return tryGetWsTag(result.getOwner());
  if (auto arg = dyn_cast<BlockArgument>(value))
    return tryGetWsTag(arg.getOwner()->getParentOp());
  return std::nullopt;
}

static SetVector<int>
partitionSetForTokenOrOwner(Value token, std::optional<PartitionId> owner,
                            Operation *fallbackAnchor = nullptr) {
  SetVector<int> ids = partitionSetForOwner(owner);
  if (ids.empty())
    addPartitionIds(ids, partitionSetForValue(token));
  if (ids.empty())
    if (auto parentIds = nearestPartitionIds(fallbackAnchor))
      addPartitionIds(ids, *parentIds);
  return ids;
}

static void setWarpTagOutsideWsLoop(Operation *op, int tag) {
  auto forOp = op->getParentOfType<scf::ForOp>();
  while (forOp && !hasWarpSpecializeTag(forOp))
    forOp = forOp->getParentOfType<scf::ForOp>();
  if (!forOp && !hasWarpSpecializeTag(op))
    setWarpSpecializeTag(op, tag);
}

static void addOwnerPartition(Operation *op, std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (hasPartition(op))
    ids = getPartitionIds(op);
  else if (auto parentIds = nearestPartitionIds(op->getParentOp()))
    ids = *parentIds;
  if (owner) {
    ids.insert(owner->first);
    setWarpTagOutsideWsLoop(op, owner->second);
  }
  if (!ids.empty())
    setPartition(op, ids);
}

static void setSingleOwnerPartition(Operation *op,
                                    std::optional<PartitionId> owner) {
  if (!op || !owner)
    return;
  SetVector<int> ids;
  ids.insert(owner->first);
  setPartition(op, ids);
  setWarpTagOutsideWsLoop(op, owner->second);
}

static void setPartitionFromAnchor(Operation *op, Operation *anchor) {
  if (!op || !anchor || hasPartition(op) || !hasPartition(anchor))
    return;
  auto ids = getPartitionIds(anchor);
  if (ids.size() == 1) {
    setPartition(op, ids);
    if (auto tag = tryGetWsTag(anchor))
      setWarpTagOutsideWsLoop(op, *tag);
  }
}

static bool parentRequiresPartition(Operation *op) {
  return op && nearestPartitionIds(op->getParentOp()).has_value();
}

static void setPartitionFromTokenIfParentPartitioned(Operation *op,
                                                     Value token) {
  if (!op || hasPartition(op) || !parentRequiresPartition(op))
    return;
  auto ids = partitionSetForValue(token);
  if (ids.size() != 1)
    return;
  setPartition(op, ids);
  if (auto tag = wsTagForValue(token))
    setWarpTagOutsideWsLoop(op, *tag);
}

static bool regionContainsAccess(Region &region, const OptSyncDag &dag) {
  bool found = false;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (dag.accessOps.contains(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static ArrayAttr asyncPayloadArray(OpBuilder &b, AsyncOp payload) {
  return b.getArrayAttr(
      SmallVector<Attribute>{AsyncOpAttr::get(b.getContext(), payload)});
}

static MemDescType withMutableMemory(MemDescType type, bool mutableMemory) {
  if (type.getMutableMemory() == mutableMemory)
    return type;
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          mutableMemory, type.getAllocShape());
}

static MemDescType getLocalSemaphoreBufferType(
    unsigned memberIdx, ArrayRef<const AccessTouch *> touches, Type backingType,
    bool mutableMemory) {
  for (const AccessTouch *touch : touches) {
    if (touch->memberIdx != memberIdx) continue;
    Value viewValue = touch->accessValue;
    if (!touch->alias.steps.empty()) {
      AliasStep first = touch->alias.steps.front();
      viewValue = first.op->getOperand(first.sourceOperand);
    }
    for (AliasStep step : touch->alias.steps) {
      if (step.op->getName().getStringRef() != "ttg.memdesc_index")
        break;
      viewValue = step.op->getResult(0);
    }
    if (auto accessTy = dyn_cast<MemDescType>(viewValue.getType()))
      return withMutableMemory(accessTy, mutableMemory);
  }
  return withMutableMemory(
      getSemaphoreViewBufferType(cast<MemDescType>(backingType)),
      mutableMemory);
}

static FailureOr<unsigned> getBackingIndex(Operation *op,
                                           const GroupBacking &backing,
                                           unsigned memberIdx) {
  auto it = backing.memberToBackingIndex.find(memberIdx);
  if (it == backing.memberToBackingIndex.end())
    return op->emitError("nvws-insert-semas: semaphore backing has no member "
                         "for planned resource touch");
  return it->second;
}

static SmallVector<Type, 4> getSemaphoreBufferViewTypes(BufferGroup &group,
                                                        const GroupBacking &backing,
                                                        ArrayRef<const AccessTouch *> touches,
                                                        bool mutableMemory) {
  SmallVector<Type, 4> viewTypes;
  for (auto [idx, type] : llvm::enumerate(backing.bufferTypes)) {
    auto memDescType = cast<MemDescType>(type);
    if (group.isTmem())
      viewTypes.push_back(getSemaphoreViewBufferType(memDescType));
    else
      viewTypes.push_back(getLocalSemaphoreBufferType(
          backing.memberIndices[static_cast<unsigned>(idx)], touches, type,
          mutableMemory));
  }
  return viewTypes;
}

static SemaphoreBufferOp
emitSemaphoreBuffer(OpBuilder &b, Location loc, Value sem, Value token,
                    std::optional<PartitionId> owner, StageCluster stageCluster,
                    BufferGroup &group, const GroupBacking &backing,
                    ArrayRef<const AccessTouch *> touches,
                    bool mutableMemory) {
  SmallVector<Type, 4> viewTypes =
      getSemaphoreBufferViewTypes(group, backing, touches, mutableMemory);
  return createIntoPartition<SemaphoreBufferOp>(
      b, loc, {owner, stageCluster}, sem, TypeRange(viewTypes), token);
}

static SemaphoreAcquireOp emitAcquire(OpBuilder &b, Location loc, Value sem,
                                      std::optional<PartitionId> owner,
                                      StageCluster stageCluster) {
  return createIntoPartition<SemaphoreAcquireOp>(
      b, loc, {owner, stageCluster}, sem, b.getType<AsyncTokenType>());
}

static SemaphoreReleaseOp emitRelease(OpBuilder &b, Location loc, Value sem,
                                      Value token,
                                      std::optional<PartitionId> owner,
                                      StageCluster stageCluster,
                                      AsyncOp payload) {
  return createIntoPartition<SemaphoreReleaseOp>(
      b, loc, {owner, stageCluster}, sem, token, asyncPayloadArray(b, payload));
}

static StageCluster expectedStampedStage(std::optional<PartitionId> owner,
                                         StageCluster stageCluster) {
  return owner ? stageCluster : StageCluster{};
}

static Operation *createNVWSDescriptorLoadOp(
    OpBuilder &b, Operation *ttDescLoadOp, Value dataBuf,
    std::optional<PartitionId> owner, StageCluster stageCluster, Location loc) {
  int txCount = getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = createIntoPartition<triton::nvws::DescriptorLoadOp>(
        b, loc, {owner, stageCluster}, descLoad.getDesc(),
        descLoad.getIndices(), txCount, dataBuf, descLoad.getCache(),
        descLoad.getEvict());
    newDescLoad->setAttrs(descLoad->getAttrs());
    setStageCluster(b, newDescLoad, stageCluster);
    if (owner)
      setPartition(newDescLoad, partitionSetForOwner(owner));
    return newDescLoad.getOperation();
  }
  if (auto descGather = dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = createIntoPartition<triton::nvws::DescriptorGatherOp>(
        b, loc, {owner, stageCluster}, descGather.getDesc(),
        descGather.getXOffsets(), descGather.getYOffset(), txCount, dataBuf);
    newDescGather->setAttrs(descGather->getAttrs());
    setStageCluster(b, newDescGather, stageCluster);
    if (owner)
      setPartition(newDescGather, partitionSetForOwner(owner));
    return newDescGather.getOperation();
  }
  llvm_unreachable("unknown descriptor op");
}

static Operation *latestSameBlockConsumer(Operation *anchor) {
  Operation *latest = anchor;
  Block *block = anchor->getBlock();
  SmallVector<Operation *, 8> worklist;
  DenseSet<Operation *> seen;
  for (Value result : anchor->getResults())
    for (Operation *user : result.getUsers())
      worklist.push_back(user);

  while (!worklist.empty()) {
    Operation *user = worklist.pop_back_val();
    if (!seen.insert(user).second)
      continue;
    Operation *ancestor = block->findAncestorOpInBlock(*user);
    if (!ancestor)
      continue;
    if (latest->isBeforeInBlock(ancestor))
      latest = ancestor;
    for (Value result : user->getResults())
      for (Operation *next : result.getUsers())
        worklist.push_back(next);
  }
  return latest;
}

static bool hasMemDescResult(Operation *op) {
  return llvm::any_of(op->getResults(),
                      [](Value result) { return isa<MemDescType>(result.getType()); });
}

static void collectTransitiveConsumers(Operation *producer, Block *anchorBlock,
                                       DenseSet<Operation *> &seen,
                                       SetVector<Operation *> &consumers) {
  if (!seen.insert(producer).second)
    return;
  for (Value result : producer->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (hasMemDescResult(user)) {
        collectTransitiveConsumers(user, anchorBlock, seen, consumers);
        continue;
      }
      Operation *ancestor = anchorBlock->findAncestorOpInBlock(*user);
      consumers.insert(ancestor ? ancestor : user);
    }
  }
}

static Operation *latestTransitiveConsumer(Operation *anchor) {
  SetVector<Operation *> consumers;
  DenseSet<Operation *> seen;
  collectTransitiveConsumers(anchor, anchor->getBlock(), seen, consumers);
  if (consumers.empty())
    return latestSameBlockConsumer(anchor);
  SmallVector<Operation *, 8> consumerOps(consumers.begin(), consumers.end());
  Operation *scope = nullptr;
  if (auto funcOp = anchor->getParentOfType<triton::FuncOp>())
    scope = funcOp.getOperation();
  PostDominanceInfo dom(scope ? scope : anchor->getParentOp());
  Operation *postDom = findNearestCommonPostDominator(consumerOps, dom);
  if (!postDom)
    return latestSameBlockConsumer(anchor);
  Operation *ancestor = anchor->getBlock()->findAncestorOpInBlock(*postDom);
  return ancestor ? ancestor : postDom;
}

static bool sameMemDescViewType(Type a, Type b) {
  if (a == b)
    return true;
  auto aTy = dyn_cast<MemDescType>(a);
  auto bTy = dyn_cast<MemDescType>(b);
  if (!aTy || !bTy)
    return false;
  return aTy.getShape() == bTy.getShape() &&
         aTy.getElementType() == bTy.getElementType() &&
         aTy.getEncoding() == bTy.getEncoding() &&
         aTy.getMemorySpace() == bTy.getMemorySpace() &&
         aTy.getMutableMemory() == bTy.getMutableMemory();
}

static Value materializeAliasForBuffer(OpBuilder &b, const AccessTouch &touch,
                                       Value memberBuffer) {
  Value cur = memberBuffer;
  for (AliasStep step : touch.alias.steps) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" &&
        old->getNumResults() == 1 &&
        sameMemDescViewType(old->getResult(0).getType(), cur.getType()))
      continue;
    IRMapping mapping;
    for (auto [idx, operand] : llvm::enumerate(old->getOperands()))
      mapping.map(operand, idx == step.sourceOperand ? cur : operand);
    Operation *cloned = b.clone(*old, mapping);
    cur = cloned->getResult(0);
  }
  return cur;
}

static void replaceUsesExcept(Value oldValue, Value newValue,
                              Operation *except) {
  SmallVector<OpOperand *> uses;
  DominanceInfo domInfo(
      except->getParentOfType<triton::FuncOp>().getOperation());
  for (OpOperand &use : oldValue.getUses())
    if (use.getOwner() != except && !isa<SemaphoreCreateOp>(use.getOwner()) &&
        domInfo.dominates(newValue, use.getOwner()))
      uses.push_back(&use);
  for (OpOperand *use : uses)
    use->set(newValue);
}

static void replaceTokenResults(Operation *op, Value token) {
  if (!token) return;
  for (Value result : op->getResults())
    if (isa<AsyncTokenType>(result.getType()))
      result.replaceAllUsesWith(token);
}

static void poisonTokenResults(OpBuilder &b, Operation *op,
                               Operation *insertBefore = nullptr) {
  bool hasTokenResult = llvm::any_of(op->getResults(), [](Value result) {
    return isa<AsyncTokenType>(result.getType());
  });
  if (!hasTokenResult)
    return;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(insertBefore ? insertBefore : op);
  Value poison =
      ub::PoisonOp::create(b, op->getLoc(), b.getType<AsyncTokenType>());
  replaceTokenResults(op, poison);
}

static void poisonOriginalTmemAllocTokens(BufferGroup &group) {
  if (!group.isTmem())
    return;
  Operation *anchor = nullptr;
  for (BufferMember &member : group.members) {
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getToken() || allocOp.getToken().use_empty())
      continue;
    anchor = allocOp.getOperation();
    break;
  }
  if (!anchor)
    return;

  OpBuilder b(anchor);
  b.setInsertionPoint(anchor);
  Value poison =
      ub::PoisonOp::create(b, anchor->getLoc(), b.getType<AsyncTokenType>());
  for (BufferMember &member : group.members) {
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getToken())
      continue;
    allocOp.getToken().replaceAllUsesWith(poison);
  }
}

static bool touchesWrittenAccumulator(ArrayRef<const AccessTouch *> touches,
                                      Value accumulator) {
  return llvm::any_of(touches, [&](const AccessTouch *touch) {
    return touch->accessValue == accumulator && touchWrites(*touch);
  });
}

static void clearOwnedTmemTokenOperands(Operation *op) {
  if (auto tmemLoad = dyn_cast<TMEMLoadOp>(op)) {
    tmemLoad.getDepMutable().clear();
    return;
  }
  if (auto tmemStore = dyn_cast<TMEMStoreOp>(op)) {
    tmemStore.getDepMutable().clear();
    return;
  }
  if (auto mma = dyn_cast<MMAv5OpInterface>(op))
    mma.getAccDepMutable().clear();
}

static bool accessOwnsAsyncToken(Operation *op,
                                 ArrayRef<const AccessTouch *> touches,
                                 BufferGroup &group) {
  if (!group.isTmem())
    return false;
  if (isa<TMEMAllocOp, TMEMLoadOp, TMEMStoreOp>(op))
    return true;
  if (auto mma = dyn_cast<MMAv5OpInterface>(op))
    return llvm::any_of(touches, [](const AccessTouch *touch) {
      return touchWrites(*touch);
    });
  return false;
}

static Operation *getBufferDefiningOp(ArrayRef<Value> buffers) {
  for (Value buffer : buffers)
    if (Operation *def = buffer.getDefiningOp())
      return def;
  return nullptr;
}

static bool allResultsUnused(Operation *op) {
  for (Value result : op->getResults())
    if (!result.use_empty())
      return false;
  return true;
}

static void eraseUnusedOriginals(BufferGroup &group) {
  for (Operation *op : llvm::reverse(group.aliasOps))
    if (allResultsUnused(op))
      op->erase();
  for (BufferMember &member : group.members)
    if (!isSemaphoreBackingAlloc(member.allocOp) && allResultsUnused(member.allocOp))
      member.allocOp->erase();
}

static bool isEligibleTmemReuseAlloc(TMEMAllocOp allocOp) {
  if (!getBufferId(allocOp))
    return false;
  if (allocOp.getSrc())
    return false;
  if (auto token = allocOp.getToken(); token && !token.use_empty())
    return false;

  auto type = allocOp.getResult().getType();
  if (type.getRank() < 2)
    return false;
  if (!isa<TensorMemorySpaceAttr>(type.getMemorySpace()))
    return false;
  if (!isa<TensorMemoryEncodingAttr>(type.getEncoding()))
    return false;

  return true;
}

static int64_t getI64AttrOr(Operation *op, StringRef name,
                            int64_t defaultValue) {
  return getI64Attr(op, name).value_or(defaultValue);
}

using TmemReuseKey = std::pair<int64_t, int64_t>;

static TmemReuseKey getTmemReuseKey(TMEMAllocOp allocOp) {
  return {*getBufferId(allocOp),
          getI64AttrOr(allocOp, kBufferCopyAttrName, -1)};
}

struct TmemReuseView {
  int64_t offset = 0;
  int64_t sliceSize = 0;
};

static std::optional<TmemReuseView> getTmemReuseView(
    TMEMAllocOp representative, TMEMAllocOp duplicate) {
  if (getBufferOffset(representative) != 0)
    return std::nullopt;

  auto baseType = representative.getResult().getType();
  auto duplicateType = duplicate.getResult().getType();
  if (baseType.getRank() != duplicateType.getRank())
    return std::nullopt;

  ArrayRef<int64_t> baseShape = baseType.getShape();
  ArrayRef<int64_t> duplicateShape = duplicateType.getShape();
  for (int i = 0, e = baseType.getRank() - 1; i < e; ++i)
    if (baseShape[i] != duplicateShape[i])
      return std::nullopt;

  int64_t duplicateOffset = getBufferOffset(duplicate);
  if (duplicateOffset < 0)
    return std::nullopt;

  int64_t baseBlockN = baseShape.back();
  int64_t duplicateBlockN = duplicateShape.back();
  int64_t baseElemWidth = baseType.getElementTypeBitWidth();
  int64_t duplicateElemWidth = duplicateType.getElementTypeBitWidth();

  int64_t sliceSize = 0;
  if (baseElemWidth == duplicateElemWidth) {
    sliceSize = duplicateBlockN;
  } else if (baseElemWidth == duplicateElemWidth * 2) {
    if (duplicateBlockN % 2 != 0)
      return std::nullopt;
    sliceSize = duplicateBlockN / 2;
  } else {
    return std::nullopt;
  }

  if (sliceSize <= 0 || duplicateOffset + sliceSize > baseBlockN)
    return std::nullopt;

  return TmemReuseView{duplicateOffset, sliceSize};
}

static bool canRepresentTmemReuseGroup(TMEMAllocOp representative,
                                       ArrayRef<TMEMAllocOp> group) {
  return llvm::all_of(group, [&](TMEMAllocOp duplicate) {
    return duplicate == representative ||
           getTmemReuseView(representative, duplicate).has_value();
  });
}

static TMEMAllocOp chooseTmemReuseRepresentative(ArrayRef<TMEMAllocOp> group) {
  for (TMEMAllocOp candidate : group)
    if (canRepresentTmemReuseGroup(candidate, group))
      return candidate;
  return {};
}

static bool moveRepresentativeBeforeGroup(TMEMAllocOp representative,
                                          ArrayRef<TMEMAllocOp> group) {
  Block *block = representative->getBlock();
  Operation *earliest = representative.getOperation();
  for (TMEMAllocOp allocOp : group) {
    if (allocOp->getBlock() != block)
      return false;
    if (allocOp->isBeforeInBlock(earliest))
      earliest = allocOp.getOperation();
  }
  if (earliest != representative.getOperation())
    representative->moveBefore(earliest);
  return true;
}

static Value createTmemReuseView(OpBuilder &builder,
                                 TMEMAllocOp representative,
                                 TMEMAllocOp duplicate,
                                 TmemReuseView view) {
  auto duplicateType = duplicate.getResult().getType();
  if (representative.getResult().getType() == duplicateType &&
      view.offset == 0)
    return representative.getResult();

  builder.setInsertionPoint(duplicate);
  auto subSlice = TMEMSubSliceOp::create(builder, duplicate.getLoc(),
                                         representative.getResult(),
                                         view.offset, view.sliceSize);
  auto reinterpret = MemDescReinterpretOp::create(
      builder, duplicate.getLoc(), duplicateType, subSlice);
  setPartitionFromAnchor(subSlice, duplicate);
  setPartitionFromAnchor(reinterpret, duplicate);
  if (StageCluster stageCluster = getStageCluster(duplicate)) {
    setStageCluster(builder, subSlice, stageCluster);
    setStageCluster(builder, reinterpret, stageCluster);
  }
  return reinterpret.getResult();
}

static void coalesceTmemAllocsByBufferIdIntoViews(triton::FuncOp funcOp) {
  SmallVector<TMEMAllocOp> allocs;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isEligibleTmemReuseAlloc(allocOp))
      allocs.push_back(allocOp);
  });

  llvm::MapVector<TmemReuseKey, SmallVector<TMEMAllocOp>> groups;
  for (TMEMAllocOp allocOp : allocs)
    groups[getTmemReuseKey(allocOp)].push_back(allocOp);

  OpBuilder builder(funcOp.getContext());
  for (auto &entry : groups) {
    SmallVector<TMEMAllocOp> &group = entry.second;
    if (group.size() < 2)
      continue;

    TMEMAllocOp representative = chooseTmemReuseRepresentative(group);
    if (!representative)
      continue;
    if (!moveRepresentativeBeforeGroup(representative, group))
      continue;

    DominanceInfo domInfo(funcOp);
    if (!llvm::all_of(group, [&](TMEMAllocOp duplicate) {
          return duplicate == representative ||
                 domInfo.dominates(representative.getOperation(),
                                   duplicate.getOperation());
        }))
      continue;

    for (TMEMAllocOp duplicate : group) {
      if (duplicate == representative)
        continue;
      std::optional<TmemReuseView> view =
          getTmemReuseView(representative, duplicate);
      if (!view)
        continue;
      Value replacement =
          createTmemReuseView(builder, representative, duplicate, *view);
      duplicate.getResult().replaceAllUsesWith(replacement);
    }
  }
}

static void eraseDeadTmemAllocs(triton::FuncOp funcOp) {
  SmallVector<TMEMAllocOp> allocs;
  funcOp.walk([&](TMEMAllocOp allocOp) { allocs.push_back(allocOp); });
  for (TMEMAllocOp allocOp : llvm::reverse(allocs))
    if (allResultsUnused(allocOp))
      allocOp.erase();
}

static LogicalResult emitAccessEvent(OpBuilder &b, AccessEvent &event,
                                     ArrayRef<const AccessTouch *> touches,
                                     ArrayRef<AcquireRecord> acquires,
                                     BufferGroup &group,
                                     const OptSyncDag &dag,
                                     const GroupBacking &backing,
                                     EmitState &state) {
  Operation *op = event.op;
  bool writes = llvm::any_of(touches, [](const AccessTouch *touch) {
    return touchWrites(*touch);
  });

  Value sem;
  Value token;
  if (!acquires.empty()) {
    sem = acquires.front().semaphore;
    token = acquires.front().token;
  } else if (state.currentToken && state.currentSemaphore) {
    sem = state.currentSemaphore;
    token = state.currentToken;
  } else if (writes) {
    return op->emitError("nvws-insert-semas: missing planned EMPTY/FULL carrier "
                         "token for writer");
  } else {
    return success();
  }

  StageCluster stageCluster = getStageCluster(op);
  Operation *bufferOperation = nullptr;
  SmallVector<Value, 4> buffers;
  StageCluster bufferExpectedStageCluster =
      expectedStampedStage(event.owner, stageCluster);
  bool canReuseCurrentBuffers =
      acquires.empty() && state.currentToken == token &&
      state.currentSemaphore == sem &&
      state.currentBuffers.size() == backing.bufferTypes.size();
  if (canReuseCurrentBuffers) {
    buffers.assign(state.currentBuffers.begin(), state.currentBuffers.end());
    bufferOperation = getBufferDefiningOp(buffers);
    bufferExpectedStageCluster =
        bufferOperation ? getStageCluster(bufferOperation) : StageCluster{};
  } else {
    SemaphoreBufferOp bufferOp =
        emitSemaphoreBuffer(b, op->getLoc(), sem, token, event.owner,
                            stageCluster, group, backing, touches, writes);
    if (!event.owner)
      setPartitionFromAnchor(bufferOp.getOperation(), op);
    if (!event.owner)
      setPartitionFromTokenIfParentPartitioned(bufferOp.getOperation(), token);
    bufferOperation = bufferOp.getOperation();
    buffers.assign(bufferOp.getBuffers().begin(), bufferOp.getBuffers().end());
    state.currentBuffers = buffers;
  }
  state.eventBuffers[op] = buffers;
  Operation *retargetOp = op;
  bool ownsAsyncToken = accessOwnsAsyncToken(op, touches, group);

  if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful TMEM alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    FailureOr<unsigned> backingIdx = getBackingIndex(op, backing, touch.memberIdx);
    if (failed(backingIdx))
      return failure();
    if (*backingIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[*backingIdx]);
    if (Value src = tmemAlloc.getSrc()) {
      auto vTrue = createIntoPartition<arith::ConstantIntOp>(
          b, op->getLoc(), {event.owner, getStageCluster(op)}, true, 1);
      auto store = createIntoPartition<TMEMStoreOp>(
          b, op->getLoc(), {event.owner, getStageCluster(op)}, Type(),
          accessBuffer, Value(), src, vTrue);
      retargetOp = store.getOperation();
      replaceUsesExcept(tmemAlloc.getResult(), accessBuffer, store);
      state.rewrittenAccessValue[tmemAlloc.getResult()] = accessBuffer;
    }
    state.emittedBuffers.push_back(EmittedBufferRecord{
        op, retargetOp, bufferOperation, sem, token, accessBuffer,
        state.eventBuffers[op], touch.memberIdx, *backingIdx,
        bufferExpectedStageCluster});
  } else if (auto localAlloc = dyn_cast<LocalAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful local alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    FailureOr<unsigned> backingIdx = getBackingIndex(op, backing, touch.memberIdx);
    if (failed(backingIdx))
      return failure();
    if (*backingIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[*backingIdx]);
    if (Value src = localAlloc.getSrc()) {
      if (Operation *def = src.getDefiningOp();
          def && isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(def)) {
        retargetOp = createNVWSDescriptorLoadOp(
            b, def, accessBuffer, event.owner, getStageCluster(op),
            op->getLoc());
      } else {
        Value storeValue = src;
        if (isa<FloatType, IntegerType>(src.getType())) {
          auto splat = createIntoPartition<triton::SplatOp>(
              b, op->getLoc(), {event.owner, getStageCluster(op)},
              getTensorTypeFromScalar(b, src), src);
          storeValue = splat;
        }
        auto store = createIntoPartition<LocalStoreOp>(
            b, op->getLoc(), {event.owner, getStageCluster(op)}, storeValue,
            accessBuffer);
        retargetOp = store.getOperation();
      }
      replaceUsesExcept(localAlloc.getResult(), accessBuffer, retargetOp);
      state.rewrittenAccessValue[localAlloc.getResult()] = accessBuffer;
    }
    state.emittedBuffers.push_back(EmittedBufferRecord{
        op, retargetOp, bufferOperation, sem, token, accessBuffer,
        state.eventBuffers[op], touch.memberIdx, *backingIdx,
        bufferExpectedStageCluster});
  } else {
    SmallVector<std::pair<OpOperand *, Value>, 4> replacements;
    for (const AccessTouch *touch : touches) {
      FailureOr<unsigned> backingIdx =
          getBackingIndex(op, backing, touch->memberIdx);
      if (failed(backingIdx))
        return failure();
      if (*backingIdx >= buffers.size())
        return op->emitError("nvws-insert-semas: semaphore buffer member index "
                             "out of range");
      Value accessBuffer =
          materializeAliasForBuffer(b, *touch, buffers[*backingIdx]);
      Value currentAccessValue = state.rewrittenAccessValue.lookup(
          touch->accessValue);
      for (OpOperand &operand : op->getOpOperands())
        if (operand.get() == touch->accessValue ||
            (currentAccessValue && operand.get() == currentAccessValue))
          replacements.push_back({&operand, accessBuffer});
      state.emittedBuffers.push_back(EmittedBufferRecord{
          op, retargetOp, bufferOperation, sem, token, accessBuffer,
          state.eventBuffers[op], touch->memberIdx, *backingIdx,
          bufferExpectedStageCluster});
    }
    for (auto [operand, accessBuffer] : replacements)
      operand->set(accessBuffer);
    if (ownsAsyncToken) {
      clearOwnedTmemTokenOperands(op);
      Operation *poisonAnchor = nullptr;
      if (auto createOp = sem.getDefiningOp<SemaphoreCreateOp>())
        if (!createOp.getBuffers().empty())
          poisonAnchor = createOp.getBuffers().front().getDefiningOp();
      if (!poisonAnchor)
        poisonAnchor =
            group.members.empty() ? bufferOperation : group.members.front().allocOp;
      state.poisonTokenResultsAfterEmission.push_back({op, poisonAnchor});
    }
  }

  if (!event.owner && retargetOp)
    setPartitionFromTokenIfParentPartitioned(retargetOp, token);

  state.eventToken[op] = token;
  state.eventSemaphore[op] = sem;
  state.protectedAccesses.insert(op);
  state.knownCarrierTokens.insert(token);
  state.currentToken = token;
  state.currentSemaphore = sem;
  state.currentOwner = event.owner;
  return success();
}

static bool valueScopeCanReachBlock(Value value, Block *block) {
  if (!value || !block)
    return false;
  Region *valueRegion = value.getParentRegion();
  Region *insertRegion = block->getParent();
  return valueRegion == insertRegion || valueRegion->isAncestor(insertRegion);
}

static FailureOr<Value> lookupReleaseToken(Location loc, const SyncEdge *edge,
                                           EmitState &state,
                                           Block *insertBlock) {
  if (edge && edge->srcOp) {
    auto it = state.eventToken.find(edge->srcOp);
    if (it != state.eventToken.end()) {
      if (!insertBlock || valueScopeCanReachBlock(it->second, insertBlock))
        return it->second;
      if (state.currentToken &&
          valueScopeCanReachBlock(state.currentToken, insertBlock))
        return state.currentToken;
    }
  }
  if (state.currentToken &&
      (!insertBlock || valueScopeCanReachBlock(state.currentToken, insertBlock)))
    return state.currentToken;
  emitError(loc, "nvws-insert-semas: planned release has no carrier token "
                 "producer");
  return failure();
}

static bool isConstantTrue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIntOp>();
  return constant && constant.value() != 0;
}

static bool isConditionalTmemStore(Operation *op) {
  auto store = dyn_cast_or_null<TMEMStoreOp>(op);
  return store && !isConstantTrue(store.getPred());
}

static bool nextLinearEdgeDstIsConditionalStore(const SyncGroup &syncGroup,
                                                const SyncPlan &sp,
                                                const SyncEdge *edge) {
  if (!edge)
    return false;
  for (auto [pos, edgeIdx] : llvm::enumerate(syncGroup.edgeIdxs)) {
    if (&sp.edges[edgeIdx] != edge)
      continue;
    if (pos + 1 >= syncGroup.edgeIdxs.size())
      return false;
    const SyncEdge &nextEdge = sp.edges[syncGroup.edgeIdxs[pos + 1]];
    return isConditionalTmemStore(nextEdge.dstOp);
  }
  return false;
}

static bool shouldForceNonePayload(const SyncGroup &syncGroup,
                                   const SyncPlan &sp, const SyncEdge *edge,
                                   SyncAnchorKind kind) {
  return syncGroup.kind == SyncGroupKind::LinearChain &&
         kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
         isa<MMAv5OpInterface>(edge->srcOp) && edge->dstOp &&
         isa<TMEMLoadOp>(edge->dstOp) &&
         nextLinearEdgeDstIsConditionalStore(syncGroup, sp, edge);
}

static bool releaseShouldPrecedeFollowingSemaphores(const SyncGroup &syncGroup,
                                                    const SyncEdge *edge,
                                                    BufferGroup &group,
                                                    int64_t resourceKey,
                                                    Operation *anchor) {
  if (!group.isTmem() || !edge || !isa<MMAv5OpInterface>(anchor))
    return false;
  bool linearReadRelease =
      syncGroup.kind == SyncGroupKind::LinearChain && edge->srcOp == anchor &&
      edgeSrcReads(*edge, group, resourceKey) &&
      !edgeSrcWrites(*edge, group, resourceKey);
  bool terminalReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      edge->dstOp == anchor && edgeDstReads(*edge, group, resourceKey) &&
      !edgeDstWrites(*edge, group, resourceKey);
  bool mmaSourceRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      edge->srcOp == anchor;
  return linearReadRelease || terminalReadRelease || mmaSourceRelease;
}

static const AccessEvent *findLastProducerInRegion(Region *region,
                                                   BufferGroup &group,
                                                   int64_t resourceKey) {
  const AccessEvent *lastProducer = nullptr;
  Operation *parent = region ? region->getParentOp() : nullptr;
  if (!parent)
    return nullptr;
  for (const AccessEvent &event : group.events)
    if (event.op && parent->isProperAncestor(event.op) &&
        eventProduces(event, resourceKey))
      lastProducer = &event;
  return lastProducer;
}

static bool hasLaterBackingGroupAccessInSameBlock(Operation *op,
                                                  BufferGroup &group) {
  if (!op || !op->getBlock())
    return false;
  bool seen = false;
  for (const AccessEvent &event : group.events) {
    if (event.op == op) {
      seen = true;
      continue;
    }
    if (!seen || !event.op || event.op->getBlock() != op->getBlock())
      continue;
    if (op->isBeforeInBlock(event.op))
      return true;
  }
  return false;
}

static void moveAfterExistingReleasesBeforeAcquire(Operation *op,
                                                   Operation *source) {
  Operation *insertBefore = source->getNextNode();
  while (insertBefore && isa<SemaphoreReleaseOp>(insertBefore))
    insertBefore = insertBefore->getNextNode();
  if (insertBefore && isa<SemaphoreAcquireOp>(insertBefore))
    op->moveBefore(insertBefore);
  else
    op->moveAfter(source);
}

static void moveAfterLoopBeforeFollowingSemaphores(Operation *op,
                                                   scf::ForOp forOp) {
  Operation *insertBefore = forOp->getNextNode();
  if (insertBefore &&
      isa<SemaphoreReleaseOp, SemaphoreAcquireOp>(insertBefore))
    op->moveBefore(insertBefore);
  else
    op->moveAfter(forOp);
}

static Operation *findLastSameBlockNonTokenResultUser(Operation *op) {
  if (!op || !op->getBlock())
    return nullptr;
  Operation *lastUser = nullptr;
  for (Value result : op->getResults()) {
    if (isa<AsyncTokenType>(result.getType()))
      continue;
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != op->getBlock() || user == op ||
          !op->isBeforeInBlock(user))
        continue;
      if (!lastUser || lastUser->isBeforeInBlock(user))
        lastUser = user;
    }
  }
  return lastUser;
}

static LogicalResult emitReleaseAction(OpBuilder &b, Location loc,
                                       SyncAnchorKind kind, Operation *anchor,
                                       Region *yieldRegion,
                                       const PlannedRelease &action,
                                       const OptSyncDag &dag, const SyncPlan &sp,
                                       BufferGroup &group, EmitState &state,
                                       StageCluster stageCluster,
                                       Operation *liveAnchor = nullptr) {
  unsigned groupIdx = action.groupIdx;
  if (groupIdx >= dag.groups.size())
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: planned release references an invalid group");
  if (action.edgeIdxs.empty())
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: planned release has no transition edge");
  for (unsigned edgeIdx : action.edgeIdxs) {
    if (edgeIdx >= sp.edges.size())
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release references an invalid edge");
    if (!edgeRequiresRelease(sp.edges[edgeIdx]))
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release is not backed by a partition "
          "transition edge");
    if (edgeIdx >= dag.edgeToGroup.size() || dag.edgeToGroup[edgeIdx] != groupIdx)
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release edge does not belong to its group");
  }
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge = getRepresentativeReleaseEdge(action, sp);
  std::optional<unsigned> edgeIdx = action.edgeIdxs.front();
  for (const EmittedSyncRecord &record : state.emittedReleases)
    if (record.groupIdx == groupIdx && record.kind == kind &&
        record.anchor == anchor && record.yieldRegion == yieldRegion &&
        record.edgeIdxs == action.edgeIdxs)
      return success();
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  bool terminalDstReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->dstOp == anchor &&
      edge->srcOp != anchor;
  bool terminalLoopExitReadRelease =
      syncGroup.kind == SyncGroupKind::Singleton &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && anchor &&
      edgeIdx && dag.tmemLoopExitRead.lookup(*edgeIdx) == anchor;
  bool sourceReadRelease =
      group.isTmem() && kind == SyncAnchorKind::ReleaseAfterOp && edge &&
      anchor && edge->srcOp == anchor &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second);
  bool delayReadReleaseForUsers = group.isTmem() && group.members.size() > 1;
  bool readCompletionRelease =
      delayReadReleaseForUsers &&
      (terminalDstReadRelease || terminalLoopExitReadRelease ||
       sourceReadRelease);
  if (terminalDstReadRelease || terminalLoopExitReadRelease)
    sem = state.semas.byGroup.lookup(groupIdx).empty;
  bool useStructuredCarrier =
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
      edge->srcOp != anchor && state.currentToken;
  useStructuredCarrier |= kind == SyncAnchorKind::ReleaseBeforeOp && edge &&
                          edge->srcOp && edge->srcOp != anchor &&
                          state.currentToken;
  SetVector<int> structuredCarrierPartition;
  if (useStructuredCarrier)
    structuredCarrierPartition = partitionSetForValue(state.currentToken);
  FailureOr<Value> token =
      useStructuredCarrier ? FailureOr<Value>(state.currentToken)
                           : lookupReleaseToken(loc, edge, state,
                                                b.getInsertionBlock());
  if (failed(token))
    return failure();
  std::optional<PartitionId> owner = edge ? edge->srcOwner : std::nullopt;
  if (terminalDstReadRelease || terminalLoopExitReadRelease)
    owner = edge->dstOwner;
  if (terminalLoopExitReadRelease)
    owner = getPartitionId(anchor);
  if (!owner && kind == SyncAnchorKind::ReleaseBeforeOp && state.currentOwner)
    owner = state.currentOwner;
  if (edge && !edge->srcOwner && kind == SyncAnchorKind::ReleaseBeforeOp &&
      syncGroup.kind == SyncGroupKind::LinearChain &&
      syncGroup.edgeIdxs.size() > 1 && &sp.edges[syncGroup.edgeIdxs.front()] == edge)
    owner = sp.edges[syncGroup.edgeIdxs[1]].dstOwner;
  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  AsyncOp payload =
      (terminalDstReadRelease || terminalLoopExitReadRelease)
          ? getAsyncPayload(anchor)
          : (edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
  if (group.isTmem() && edge && edge->srcYieldRegion &&
      !terminalDstReadRelease && !terminalLoopExitReadRelease &&
      payload == AsyncOp::NONE)
    if (const AccessEvent *producer = findLastProducerInRegion(
            edge->srcYieldRegion, group, dag.resource.second))
      if (sameOwner(producer->owner, edge->srcOwner))
        payload = getAsyncPayload(producer->op);
  if (shouldForceNonePayload(syncGroup, sp, edge, kind))
    payload = AsyncOp::NONE;
  SemaphoreReleaseOp release =
      emitRelease(b, loc, sem, *token, owner, stageCluster, payload);
  if (readCompletionRelease && anchor)
    if (Operation *lastUser = findLastSameBlockNonTokenResultUser(anchor))
      if (release->getBlock() == lastUser->getBlock() &&
          release->isBeforeInBlock(lastUser))
        release->moveAfter(lastUser);
  bool readOnlyMmaSource =
      edge && isa_and_nonnull<MMAv5OpInterface>(edge->srcOp) &&
      edge->dstYieldRegion && edge->dstOwner &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second) &&
      !hasLaterBackingGroupAccessInSameBlock(edge->srcOp, group);
  bool writeMmaSource =
      edge && isa_and_nonnull<MMAv5OpInterface>(edge->srcOp) &&
      edge->dstYieldRegion && edge->dstOwner &&
      edgeSrcWrites(*edge, group, dag.resource.second) &&
      !hasLaterBackingGroupAccessInSameBlock(edge->srcOp, group);
  bool readOnlyTmemLoadSource =
      edge && isa_and_nonnull<TMEMLoadOp>(edge->srcOp) &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second);
  bool moveReadOnlyTmemLoadSource =
      readOnlyTmemLoadSource && group.members.size() == 1;
  if (group.isTmem() && kind == SyncAnchorKind::ReleaseBeforeYield && edge &&
      (moveReadOnlyTmemLoadSource || readOnlyMmaSource || writeMmaSource) &&
      operationIsAttached(edge->srcOp) &&
      release->getBlock() == edge->srcOp->getBlock()) {
    if (writeMmaSource)
      moveAfterExistingReleasesBeforeAcquire(release.getOperation(), edge->srcOp);
    else if (moveReadOnlyTmemLoadSource)
      release->moveAfter(edge->srcOp);
    else {
      Operation *insertAfter = edge->srcOp;
      if (Operation *lastUser =
              findLastSameBlockNonTokenResultUser(edge->srcOp))
        if (insertAfter->isBeforeInBlock(lastUser))
          insertAfter = lastUser;
      release->moveAfter(insertAfter);
    }
  }
  if (group.isTmem() && kind == SyncAnchorKind::ReleaseBeforeOp && edge &&
      edge->srcYieldRegion && anchor && opReadsOnlyResource(anchor, group,
                                                            dag.resource.second)) {
    if (edgeIdx && dag.srcYieldParentWarpFor.contains(*edgeIdx)) {
      for (Operation *candidate = release->getPrevNode(); candidate;
           candidate = candidate->getPrevNode()) {
        if (isa<SemaphoreAcquireOp, SemaphoreReleaseOp>(candidate))
          continue;
        auto forOp = dyn_cast<scf::ForOp>(candidate);
        if (!forOp)
          break;
        if (hasWarpSpecializeTag(forOp) &&
            release->getBlock() == forOp->getBlock())
          moveAfterLoopBeforeFollowingSemaphores(release.getOperation(), forOp);
        break;
      }
    }
  }
  if (useStructuredCarrier && structuredCarrierPartition.size() == 1 && !owner) {
    setPartition(release.getOperation(), structuredCarrierPartition);
    if (auto tag = wsTagForValue(state.currentToken))
      setWarpTagOutsideWsLoop(release.getOperation(), *tag);
  }
  if (!owner) {
    std::optional<PartitionId> fallbackOwner =
        edge ? edge->dstOwner : std::nullopt;
    setSingleOwnerPartition(release.getOperation(), fallbackOwner);
    if (!fallbackOwner)
      setPartitionFromAnchor(release.getOperation(),
                             anchor ? anchor
                                    : (yieldRegion ? yieldRegion->getParentOp()
                                                   : nullptr));
  }
  state.emittedReleases.push_back(EmittedSyncRecord{
      groupIdx, kind, anchor, yieldRegion, release.getOperation(), sem, *token,
      expectedStampedStage(owner, stageCluster)});
  state.emittedReleases.back().edgeIdxs.append(action.edgeIdxs.begin(),
                                               action.edgeIdxs.end());
  return success();
}

static AcquireRecord emitAcquireForGroup(OpBuilder &b, Location loc,
                                         SyncAnchorKind kind, Operation *anchor,
                                         Region *yieldRegion,
                                         unsigned groupIdx,
                                         const OptSyncDag &dag,
                                         const SyncPlan &sp, BufferGroup &group,
                                         EmitState &state,
                                         StageCluster stageCluster) {
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge =
      findEdgeForAnchor(syncGroup, sp, dag, kind, anchor, yieldRegion);
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  std::optional<PartitionId> owner =
      edge ? edge->dstOwner : syncGroup.initialOwner;
  SemaphoreAcquireOp acquire = emitAcquire(b, loc, sem, owner, stageCluster);
  if (!owner) {
    std::optional<PartitionId> fallbackOwner =
        parentRequiresPartition(acquire.getOperation()) && edge
            ? edge->srcOwner
            : std::nullopt;
    setSingleOwnerPartition(acquire.getOperation(), fallbackOwner);
    setPartitionFromAnchor(acquire.getOperation(),
                           anchor ? anchor
                                  : (yieldRegion ? yieldRegion->getParentOp()
                                                 : nullptr));
  }
  Value token = acquire.getToken();
  state.emittedAcquires.push_back(EmittedSyncRecord{
      groupIdx, kind, anchor, yieldRegion, acquire.getOperation(), sem, token,
      expectedStampedStage(owner, stageCluster)});
  state.knownCarrierTokens.insert(token);
  state.currentToken = token;
  state.currentSemaphore = sem;
  state.currentOwner = owner;
  state.currentBuffers.clear();
  return AcquireRecord{groupIdx, sem, token, owner};
}

static LogicalResult
emitBeforeOpSync(Operation *anchor, const OptSyncDag &dag, const SyncPlan &sp,
                 BufferGroup &group, EmitState &state,
                 SmallVectorImpl<AcquireRecord> &acquires) {
  OpBuilder b(anchor);
  b.setInsertionPoint(anchor);
  auto rIt = dag.releaseBeforeOp.find(anchor);
  if (rIt != dag.releaseBeforeOp.end())
    for (const PlannedRelease &release : rIt->second)
      if (failed(emitReleaseAction(
              b, anchor->getLoc(), SyncAnchorKind::ReleaseBeforeOp, anchor,
              nullptr, release, dag, sp, group, state, getStageCluster(anchor))))
        return failure();
  auto aIt = dag.acquireBeforeOp.find(anchor);
  if (aIt != dag.acquireBeforeOp.end())
    for (unsigned gi : aIt->second)
      acquires.push_back(emitAcquireForGroup(
          b, anchor->getLoc(), SyncAnchorKind::AcquireBeforeOp, anchor, nullptr,
          gi, dag, sp, group, state, getStageCluster(anchor)));
  return success();
}

static LogicalResult emitAfterOpSync(Operation *anchor, Operation *insertAfter,
                                     const OptSyncDag &dag,
                                     const SyncPlan &sp, BufferGroup &group,
                                     EmitState &state) {
  auto rIt = dag.releaseAfterOp.find(anchor);
  if (rIt == dag.releaseAfterOp.end()) return success();
  Operation *releaseAfter = insertAfter;
  if (!group.isTmem() && isa<LocalLoadOp>(insertAfter))
    releaseAfter = latestTransitiveConsumer(insertAfter);
  if (anchor == insertAfter && operationIsAttached(releaseAfter) &&
      operationIsAttached(anchor) &&
      releaseAfter->isProperAncestor(anchor))
    return success();
  OpBuilder b(releaseAfter);
  bool beforeFollowingSemaphores = false;
  bool afterLoopRelease = group.isTmem() && isa<scf::ForOp>(releaseAfter);
  if (group.isTmem()) {
    for (const PlannedRelease &release : rIt->second) {
      const SyncGroup &syncGroup = dag.groups[release.groupIdx];
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (releaseShouldPrecedeFollowingSemaphores(
              syncGroup, edge, group, dag.resource.second, insertAfter)) {
        beforeFollowingSemaphores = true;
        break;
      }
    }
    beforeFollowingSemaphores |= afterLoopRelease;
  }
  if (isa<scf::YieldOp>(releaseAfter)) {
    b.setInsertionPoint(releaseAfter);
  } else if (beforeFollowingSemaphores) {
    Operation *insertBefore = releaseAfter->getNextNode();
    while (insertBefore && isa<SemaphoreReleaseOp>(insertBefore))
      insertBefore = insertBefore->getNextNode();
    if (insertBefore && isa<SemaphoreAcquireOp>(insertBefore))
      b.setInsertionPoint(insertBefore);
    else
      b.setInsertionPointAfter(releaseAfter);
  } else {
    b.setInsertionPointAfter(releaseAfter);
  }
  std::optional<Value> savedEventToken;
  bool overrideEventToken = releaseAfter != anchor && state.currentToken;
  if (overrideEventToken) {
    auto it = state.eventToken.find(anchor);
    if (it != state.eventToken.end())
      savedEventToken = it->second;
    state.eventToken[anchor] = state.currentToken;
  }
  SmallVector<PlannedRelease, 4> releaseActions(rIt->second.begin(),
                                                rIt->second.end());
  if (group.isTmem())
    llvm::stable_sort(releaseActions, [&](const PlannedRelease &lhs,
                                          const PlannedRelease &rhs) {
      const SyncGroup &lhsGroup = dag.groups[lhs.groupIdx];
      const SyncEdge *lhsEdge = getRepresentativeReleaseEdge(lhs, sp);
      const SyncGroup &rhsGroup = dag.groups[rhs.groupIdx];
      const SyncEdge *rhsEdge = getRepresentativeReleaseEdge(rhs, sp);
      bool lhsPrecedes = releaseShouldPrecedeFollowingSemaphores(
          lhsGroup, lhsEdge, group, dag.resource.second, insertAfter);
      bool rhsPrecedes = releaseShouldPrecedeFollowingSemaphores(
          rhsGroup, rhsEdge, group, dag.resource.second, insertAfter);
      return lhsPrecedes && !rhsPrecedes;
    });
  LogicalResult result = success();
  for (const PlannedRelease &release : releaseActions)
    if (failed(emitReleaseAction(
            b, insertAfter->getLoc(), SyncAnchorKind::ReleaseAfterOp, anchor,
            nullptr, release, dag, sp, group, state, getStageCluster(insertAfter),
            insertAfter))) {
      result = failure();
      break;
    }
  if (overrideEventToken) {
    if (savedEventToken)
      state.eventToken[anchor] = *savedEventToken;
    else
      state.eventToken.erase(anchor);
  }
  return result;
}

static LogicalResult emitAfterOpSync(Operation *anchor, const OptSyncDag &dag,
                                     const SyncPlan &sp, BufferGroup &group,
                                     EmitState &state) {
  return emitAfterOpSync(anchor, anchor, dag, sp, group, state);
}

static LogicalResult emitDeferredNestedAfterOpSync(Operation *releaseAfter,
                                                   const OptSyncDag &dag,
                                                   const SyncPlan &sp,
                                                   BufferGroup &group,
                                                   EmitState &state) {
  SmallVector<Operation *, 4> anchors;
  releaseAfter->walk([&](Operation *op) {
    if (op == releaseAfter)
      return;
    if (!isa<LocalLoadOp>(op))
      return;
    if (!dag.releaseAfterOp.contains(op))
      return;
    if (latestTransitiveConsumer(op) == releaseAfter)
      anchors.push_back(op);
  });
  for (Operation *anchor : anchors)
    if (failed(emitAfterOpSync(anchor, releaseAfter, dag, sp, group, state)))
      return failure();
  return success();
}

static bool linearChainAnchorsLoopExit(const SyncGroup &syncGroup,
                                       const SyncPlan &sp, Operation *forOp,
                                       Region *region) {
  if (!forOp || !region)
    return false;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    const SyncEdge &edge = sp.edges[edgeIdx];
    if (edge.srcOp == forOp || edge.dstOp == forOp ||
        edge.srcYieldRegion == region || edge.dstYieldRegion == region)
      return true;
  }
  return false;
}

static const PlannedRelease *
findPlannedAfterOpReleaseForGroup(Operation *anchor, unsigned groupIdx,
                                  const OptSyncDag &dag) {
  if (!anchor)
    return nullptr;
  auto releaseIt = dag.releaseAfterOp.find(anchor);
  if (releaseIt == dag.releaseAfterOp.end())
    return nullptr;
  for (const PlannedRelease &action : releaseIt->second)
    if (action.groupIdx == groupIdx)
      return &action;
  return nullptr;
}

static bool linearChainNeedsLoopExitDrain(unsigned groupIdx,
                                          const SyncGroup &syncGroup,
                                          Operation *forOp, Region *region,
                                          const OptSyncDag &dag,
                                          const SyncPlan &sp) {
  if (!forOp || !region)
    return false;
  if (findPlannedAfterOpReleaseForGroup(forOp, groupIdx, dag))
    return true;
  if (dag.skippedInitialLoopCarrierRegion.lookup(groupIdx) == region)
    for (unsigned edgeIdx : syncGroup.edgeIdxs)
      if (dag.edgesDeferringToSkippedLoopExit.contains(edgeIdx))
        return true;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    const SyncEdge &edge = sp.edges[edgeIdx];
    if (dag.loopEntryHandoffAccess.contains(edgeIdx))
      return true;
    if (edge.dstYieldRegion == region &&
        dag.terminalLoopReadEdgesDeferringToExit.contains(edgeIdx))
      return true;
  }
  return false;
}

static LogicalResult emitTmemLinearLoopExitDrain(scf::ForOp forOp,
                                                 Region *region,
                                                 const OptSyncDag &dag,
                                                 const SyncPlan &sp,
                                                 BufferGroup &group,
                                                 EmitState &state) {
  if (!group.isTmem() || !state.currentToken)
    return success();
  if (!hasWarpSpecializeTag(forOp))
    return success();
  SmallVector<unsigned, 2> groupIds;
  auto it = dag.acquireBeforeYield.find(region);
  if (it != dag.acquireBeforeYield.end())
    groupIds.append(it->second.begin(), it->second.end());
  else
    for (auto [idx, syncGroup] : llvm::enumerate(dag.groups))
      if (syncGroup.kind == SyncGroupKind::LinearChain &&
          linearChainAnchorsLoopExit(syncGroup, sp, forOp.getOperation(),
                                     region))
        groupIds.push_back(static_cast<unsigned>(idx));

  for (unsigned gi : groupIds) {
    const SyncGroup &syncGroup = dag.groups[gi];
    if (syncGroup.kind != SyncGroupKind::LinearChain ||
        syncGroup.edgeIdxs.size() < 2)
      continue;
    SyncGroupSemaphores pair = state.semas.byGroup.lookup(gi);
    if (!pair.full || !pair.empty)
      continue;
    if (linearChainNeedsPerEdgeFulls(syncGroup, sp, group, dag.resource.second))
      continue;
    if (!linearChainNeedsLoopExitDrain(gi, syncGroup, forOp.getOperation(),
                                       region, dag, sp))
      continue;

    const SyncEdge &firstEdge = sp.edges[syncGroup.edgeIdxs.front()];
    const SyncEdge &secondEdge = sp.edges[syncGroup.edgeIdxs[1]];
    auto loopExitPayload = [&]() {
      Operation *lastWriter = nullptr;
      for (const AccessEvent &event : group.events)
        if (event.op && forOp->isProperAncestor(event.op) &&
            eventProduces(event, dag.resource.second))
          lastWriter = event.op;
      return getAsyncPayload(lastWriter);
    };
    OpBuilder b(forOp);
    b.setInsertionPointAfter(forOp);
    Location loc = forOp.getLoc();
    auto emitDrainRelease = [&](Value sem, Value token,
                                std::optional<PartitionId> owner,
                                AsyncOp payload,
                                const SyncEdge &edge) -> LogicalResult {
      if (!edgeRequiresRelease(edge))
        return forOp->emitError(
            "nvws-insert-semas: loop-exit drain release is not backed by a "
            "partition transition edge");
      emitRelease(b, loc, sem, token, owner, StageCluster{}, payload);
      return success();
    };
    auto findPlannedAfterLoopRelease =
        [&](const SyncEdge &edge) -> const PlannedRelease * {
      std::optional<unsigned> edgeIdx = findEdgeIndex(sp, &edge);
      if (!edgeIdx)
        return nullptr;
      auto releaseIt = dag.releaseAfterOp.find(forOp.getOperation());
      if (releaseIt == dag.releaseAfterOp.end())
        return nullptr;
      for (const PlannedRelease &action : releaseIt->second)
        if (llvm::is_contained(action.edgeIdxs, *edgeIdx))
          return &action;
      return nullptr;
    };
    bool skipsInitialCarrier =
        dag.skippedInitialLoopCarrierRegion.lookup(gi) == region;
    const SyncEdge *loopEntryHandoffEdge = nullptr;
    for (unsigned edgeIdx : syncGroup.edgeIdxs)
      if (dag.loopEntryHandoffAccess.contains(edgeIdx)) {
        loopEntryHandoffEdge = &sp.edges[edgeIdx];
        break;
      }
    bool deferredTerminalLoopRead = llvm::any_of(
        syncGroup.edgeIdxs, [&](unsigned edgeIdx) {
          const SyncEdge &edge = sp.edges[edgeIdx];
          return edge.dstYieldRegion == region &&
                 dag.terminalLoopReadEdgesDeferringToExit.contains(edgeIdx);
        });
    if (skipsInitialCarrier) {
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (!dag.edgesDeferringToSkippedLoopExit.contains(edgeIdx))
          continue;
        Value sem = getSemaphoreForGroup(gi, &edge, dag, sp, group,
                                         state.semas);
        if (failed(emitDrainRelease(sem, state.currentToken, state.currentOwner,
                                    edge.asyncPayload, edge)))
          return failure();
        state.currentSemaphore = sem;
        state.currentBuffers.clear();
        return success();
      }
    }
    std::optional<PartitionId> drainOwner = firstEdge.dstOwner;
    std::optional<PartitionId> releaseOwner = state.currentOwner;
    AsyncOp drainPayload = AsyncOp::NONE;
    AsyncOp emptyPayload = AsyncOp::NONE;
    if (loopEntryHandoffEdge) {
      releaseOwner = loopEntryHandoffEdge->srcOwner;
      drainOwner = loopEntryHandoffEdge->dstOwner;
      emptyPayload = loopExitPayload();
    } else if (skipsInitialCarrier) {
      drainOwner = secondEdge.dstOwner;
      drainPayload = loopExitPayload();
    } else if (deferredTerminalLoopRead) {
      emptyPayload = loopExitPayload();
    } else if (firstEdge.dstOp == forOp.getOperation()) {
      drainPayload = secondEdge.asyncPayload;
    } else if (isa_and_nonnull<MMAv5OpInterface>(firstEdge.srcOp) &&
               edgeDstReads(firstEdge, group, dag.resource.second) &&
               !edgeDstWrites(firstEdge, group, dag.resource.second)) {
      drainPayload = loopExitPayload();
    } else if (isa_and_nonnull<MMAv5OpInterface>(secondEdge.srcOp) &&
               edgeDstReads(secondEdge, group, dag.resource.second) &&
               !edgeDstWrites(secondEdge, group, dag.resource.second)) {
      emptyPayload = secondEdge.asyncPayload;
    }
    const SyncEdge *fullReleaseEdge =
        loopEntryHandoffEdge ? loopEntryHandoffEdge : &firstEdge;
    if (const PlannedRelease *releaseAction =
            findPlannedAfterLoopRelease(*fullReleaseEdge)) {
      if (failed(emitReleaseAction(b, loc, SyncAnchorKind::ReleaseAfterOp,
                                   forOp.getOperation(), nullptr,
                                   *releaseAction, dag, sp, group, state,
                                   StageCluster{})))
        return failure();
    } else if (failed(emitDrainRelease(pair.full, state.currentToken,
                                       releaseOwner, drainPayload,
                                       *fullReleaseEdge))) {
      return failure();
    }
    SemaphoreAcquireOp acquire =
        emitAcquire(b, loc, pair.full, drainOwner, StageCluster{});
    state.knownCarrierTokens.insert(acquire.getToken());
    if (failed(emitDrainRelease(pair.empty, acquire.getToken(), drainOwner,
                                emptyPayload, secondEdge)))
      return failure();
    state.currentToken = acquire.getToken();
    state.currentSemaphore = pair.empty;
    state.currentOwner = drainOwner;
    return success();
  }
  return success();
}

static bool collectFirstReadOnlyRegionAccess(Region &region, const OptSyncDag &dag,
                                             BufferGroup &group,
                                             SmallVectorImpl<const AccessTouch *> &touches) {
  Operation *firstAccess = nullptr;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (!dag.accessOps.contains(op))
      return WalkResult::advance();
    for (AccessEvent &event : group.events) {
      if (event.op != op)
        continue;
      collectTouchesForResource(event, dag.resource.second, touches);
      firstAccess = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!firstAccess || touches.empty())
    return false;
  return llvm::all_of(touches, [](const AccessTouch *touch) {
    return !touchWrites(*touch);
  });
}

static bool canPrebufferLocalRegionEntry(Operation *anchor, Region &region,
                                         ArrayRef<AcquireRecord> acquires,
                                         const OptSyncDag &dag,
                                         BufferGroup &group,
                                         SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!isa<scf::ForOp>(anchor))
    return false;
  if (!dag.acquireBeforeOp.contains(anchor) || dag.releaseAfterOp.contains(anchor) ||
      dag.acquireBeforeYield.contains(&region))
    return false;
  return collectFirstReadOnlyRegionAccess(region, dag, group, touches);
}

static bool canPrebufferLocalIfEntry(scf::IfOp ifOp,
                                     ArrayRef<AcquireRecord> acquires,
                                     const OptSyncDag &dag, BufferGroup &group,
                                     SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!dag.acquireBeforeOp.contains(ifOp.getOperation()) ||
      dag.releaseAfterOp.contains(ifOp.getOperation()) ||
      dag.acquireBeforeYield.contains(&ifOp.getThenRegion()) ||
      dag.acquireBeforeYield.contains(&ifOp.getElseRegion()))
    return false;
  if (collectFirstReadOnlyRegionAccess(ifOp.getThenRegion(), dag, group,
                                       touches))
    return true;
  return collectFirstReadOnlyRegionAccess(ifOp.getElseRegion(), dag, group,
                                          touches);
}

static void prebufferLocalRegionEntry(OpBuilder &b, Operation *anchor,
                                      ArrayRef<const AccessTouch *> touches,
                                      const AcquireRecord &acquire,
                                      BufferGroup &group,
                                      const GroupBacking &backing,
                                      EmitState &state) {
  b.setInsertionPoint(anchor);
  SemaphoreBufferOp bufferOp =
      emitSemaphoreBuffer(b, anchor->getLoc(), acquire.semaphore, acquire.token,
                          acquire.owner, getStageCluster(anchor), group, backing,
                          touches, /*mutableMemory=*/false);
  if (!acquire.owner)
    setPartitionFromAnchor(bufferOp.getOperation(), anchor);
  if (!acquire.owner)
    setPartitionFromTokenIfParentPartitioned(bufferOp.getOperation(),
                                             acquire.token);
  state.currentBuffers.assign(bufferOp.getBuffers().begin(),
                              bufferOp.getBuffers().end());
}

static LogicalResult
emitBeforeYieldSync(Operation *yieldOp, Region *region, const OptSyncDag &dag,
                    const SyncPlan &sp, BufferGroup &group, EmitState &state,
                    SmallVectorImpl<AcquireRecord> &acquires) {
  OpBuilder b(yieldOp);
  b.setInsertionPoint(yieldOp);
  auto rIt = dag.releaseBeforeYield.find(region);
  if (rIt != dag.releaseBeforeYield.end())
    for (const PlannedRelease &release : rIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (failed(emitReleaseAction(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseBeforeYield, nullptr,
              region, release, dag, sp, group, state,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  auto aIt = dag.acquireBeforeYield.find(region);
  if (aIt != dag.acquireBeforeYield.end())
    for (unsigned gi : aIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            dag,
                            SyncAnchorKind::AcquireBeforeYield, nullptr, region);
      acquires.push_back(emitAcquireForGroup(
          b, yieldOp->getLoc(), SyncAnchorKind::AcquireBeforeYield, nullptr,
          region, gi, dag, sp, group, state,
          stageForYieldOwner(edge ? edge->dstOwner : std::nullopt, state)));
    }
  auto arIt = dag.releaseAfterYield.find(region);
  if (arIt != dag.releaseAfterYield.end())
    for (const PlannedRelease &release : arIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (failed(emitReleaseAction(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseAfterYield, nullptr,
              region, release, dag, sp, group, state,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  return success();
}

static bool shouldThreadForRegion(scf::ForOp forOp, const OptSyncDag &dag) {
  return dag.threadForOps.contains(forOp.getOperation());
}

static bool shouldThreadIfRegion(scf::IfOp ifOp, const OptSyncDag &dag) {
  return dag.threadIfOps.contains(ifOp.getOperation());
}

static Operation *getDominatingPoisonAnchor(Operation *op) {
  Operation *anchor = op;
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp())
    if (isa<scf::ForOp>(parent))
      anchor = parent;
  return anchor;
}

static bool linearChainEntersFor(Operation *forOp, const OptSyncDag &dag,
                                 const SyncPlan &sp) {
  if (!forOp)
    return false;
  for (const SyncGroup &syncGroup : dag.groups)
    if (syncGroup.kind == SyncGroupKind::LinearChain &&
        !syncGroup.edgeIdxs.empty() &&
        sp.edges[syncGroup.edgeIdxs.front()].dstOp == forOp)
      return true;
  return false;
}

static std::optional<PartitionId>
linearChainLoopYieldSourceOwner(Operation *forOp, const OptSyncDag &dag,
                                const SyncPlan &sp) {
  auto loop = dyn_cast_or_null<scf::ForOp>(forOp);
  if (!loop)
    return std::nullopt;
  Region *region = &loop.getRegion();
  for (auto [groupIdx, syncGroup] : llvm::enumerate(dag.groups)) {
    if (syncGroup.kind != SyncGroupKind::LinearChain ||
        syncGroup.edgeIdxs.empty() ||
        sp.edges[syncGroup.edgeIdxs.front()].dstOp != forOp)
      continue;
    for (unsigned edgeIdx : llvm::reverse(syncGroup.edgeIdxs)) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (edge.dstYieldRegion == region) {
        auto acquireIt = dag.acquireBeforeYield.find(region);
        if (acquireIt != dag.acquireBeforeYield.end() &&
            llvm::is_contained(acquireIt->second,
                               static_cast<unsigned>(groupIdx)))
          return edge.dstOwner;
        return edge.srcOwner;
      }
    }
  }
  return std::nullopt;
}

static std::optional<PartitionId>
linearChainIfYieldSourceOwner(Operation *ifOp, const OptSyncDag &dag,
                              const SyncPlan &sp) {
  if (!isa_and_nonnull<scf::IfOp>(ifOp))
    return std::nullopt;
  for (const SyncGroup &syncGroup : dag.groups) {
    if (syncGroup.kind != SyncGroupKind::LinearChain)
      continue;
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (isIfYieldRegion(edge.dstYieldRegion) &&
          edge.dstYieldRegion->getParentOp() == ifOp)
        return edge.srcOwner;
    }
  }
  return std::nullopt;
}

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src);

static FailureOr<scf::ForOp> threadCarrierThroughFor(OpBuilder &b,
                                                     scf::ForOp forOp,
                                                     EmitState &state,
                                                     Region *plannedRegion,
                                                     std::optional<PartitionId>
                                                         recordOwner,
                                                     BufferGroup &group,
                                                     ArrayRef<unsigned>
                                                         memberIndices,
                                                     int64_t resourceKey) {
  unsigned oldNumResults = forOp.getNumResults();
  auto oldPartitionIds =
      hasPartition(forOp) ? getPartitionIds(forOp) : SetVector<int>();
  auto oldPartitionOutputs =
      hasPartition(forOp) ? getPartitionOutputs(forOp)
                          : SmallVector<SetVector<int>, 4>();

  SmallVector<unsigned, 4> reusableSlots =
      findReusableTmemTokenSlots(forOp, group, memberIndices, resourceKey);
  Value init = state.currentToken;
  if (!init && !reusableSlots.empty())
    init = forOp->getOperand(3 + reusableSlots.front());
  if (!init) {
    forOp.emitError("nvws-insert-semas: planned scf.for carrier threading has "
                    "no token producer at loop entry");
    return failure();
  }
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(init, state.currentOwner, forOp.getOperation());
  if (!reusableSlots.empty()) {
    unsigned carrierSlot = reusableSlots.front();
    Value poison;
    if (reusableSlots.size() > 1) {
      OpBuilder::InsertionGuard guard(b);
      if (Operation *def = init.getDefiningOp())
        b.setInsertionPoint(def);
      else
        b.setInsertionPoint(forOp);
      poison =
          ub::PoisonOp::create(b, forOp.getLoc(), b.getType<AsyncTokenType>());
    }
    for (unsigned slot : reusableSlots)
      forOp->setOperand(3 + slot, slot == carrierSlot ? init : poison);

    state.currentToken = forOp.getRegionIterArg(carrierSlot);
    state.knownCarrierTokens.insert(state.currentToken);
    state.currentBuffers.clear();
    state.reusedForCarrierSlots[forOp.getOperation()] = carrierSlot;
    state.reusedForTokenSlots[forOp.getOperation()] = reusableSlots;
    if (poison)
      state.reusedForPoisonTokens[forOp.getOperation()] = poison;
    state.threadedTokens.push_back({forOp.getOperation(), state.currentToken,
                                    recordOwner,
                                    ThreadRecordKind::ForIterArg,
                                    plannedRegion, nullptr});
    if (hasPartition(forOp)) {
      addPartitionIds(oldPartitionIds, carrierPartition);
      if (carrierSlot < oldPartitionOutputs.size())
        oldPartitionOutputs[carrierSlot] = carrierPartition;
      setPartition(forOp, oldPartitionIds);
      setPartitionOutputs(forOp, oldPartitionOutputs);
    }
    return forOp;
  }

  b.setInsertionPoint(forOp);
  scf::ForOp newFor = addIterArgsToLoop(b, forOp, {init});
  state.currentToken = newFor.getRegionIterArg(oldNumResults);
  state.knownCarrierTokens.insert(state.currentToken);
  state.currentBuffers.clear();
  state.threadedTokens.push_back({newFor.getOperation(), state.currentToken,
                                  recordOwner,
                                  ThreadRecordKind::ForIterArg,
                                  plannedRegion, nullptr});
  if (hasPartition(newFor)) {
    addPartitionIds(oldPartitionIds, carrierPartition);
    oldPartitionOutputs.push_back(carrierPartition);
    setPartition(newFor, oldPartitionIds);
    setPartitionOutputs(newFor, oldPartitionOutputs);
  }
  return newFor;
}

static void closeCarrierForLoop(scf::ForOp forOp, EmitState &bodyState,
                                EmitState &parentState,
                                std::optional<PartitionId> ownerAtYield,
                                bool overrideOwnerAtYield,
                                Region *plannedRegion) {
  auto reusedIt = bodyState.reusedForCarrierSlots.find(forOp.getOperation());
  if (reusedIt != bodyState.reusedForCarrierSlots.end()) {
    unsigned slot = reusedIt->second;
    Value yieldedToken = bodyState.currentToken
                             ? bodyState.currentToken
                             : forOp.getRegionIterArg(slot);
    std::optional<PartitionId> resultOwner =
        overrideOwnerAtYield ? ownerAtYield
                             : (ownerAtYield ? ownerAtYield
                                             : bodyState.currentOwner);
    auto yieldOp = getForYieldOp(forOp);
    auto slotsIt = bodyState.reusedForTokenSlots.find(forOp.getOperation());
    ArrayRef<unsigned> tokenSlots =
        slotsIt == bodyState.reusedForTokenSlots.end()
            ? ArrayRef<unsigned>(slot)
            : ArrayRef<unsigned>(slotsIt->second);
    Value poison = bodyState.reusedForPoisonTokens.lookup(forOp.getOperation());
    for (unsigned tokenSlot : tokenSlots)
      yieldOp.setOperand(tokenSlot,
                         tokenSlot == slot ? yieldedToken : poison);
    SetVector<int> carrierPartition = partitionSetForTokenOrOwner(
        yieldedToken, resultOwner, forOp.getOperation());
    SetVector<int> yieldPartition;
    if (hasPartition(yieldOp))
      yieldPartition = getPartitionIds(yieldOp);
    addPartitionIds(yieldPartition, carrierPartition);
    if (!yieldPartition.empty())
      setPartition(yieldOp.getOperation(), yieldPartition);
    if (!carrierPartition.empty() && hasPartition(forOp)) {
      auto partitionIds = getPartitionIds(forOp);
      addPartitionIds(partitionIds, carrierPartition);
      auto partitionOutputs = getPartitionOutputs(forOp);
      if (slot < partitionOutputs.size())
        partitionOutputs[slot] = carrierPartition;
      setPartitionOutputs(forOp, partitionOutputs);
    }
    parentState = bodyState;
    parentState.currentToken = forOp.getResult(slot);
    parentState.knownCarrierTokens.insert(parentState.currentToken);
    parentState.currentOwner = resultOwner;
    parentState.currentBuffers.clear();
    parentState.threadedTokens.push_back({forOp.getOperation(),
                                          parentState.currentToken,
                                          parentState.currentOwner,
                                          ThreadRecordKind::ForResult,
                                          plannedRegion, nullptr});
    return;
  }

  Value yieldedToken = bodyState.currentToken
                           ? bodyState.currentToken
                           : forOp.getRegionIterArg(forOp.getNumResults() - 1);
  std::optional<PartitionId> resultOwner =
      overrideOwnerAtYield ? ownerAtYield
                           : (ownerAtYield ? ownerAtYield
                                           : bodyState.currentOwner);
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(yieldedToken, resultOwner, forOp.getOperation());
  appendToForYield(forOp, yieldedToken);
  if (!carrierPartition.empty()) {
    if (hasPartition(forOp)) {
      auto partitionIds = getPartitionIds(forOp);
      addPartitionIds(partitionIds, carrierPartition);
      auto partitionOutputs = getPartitionOutputs(forOp);
      if (partitionOutputs.size() == forOp.getNumResults())
        partitionOutputs.back() = carrierPartition;
      setPartitionOutputs(forOp, partitionOutputs);
    }
    scf::YieldOp yieldOp = getForYieldOp(forOp);
    SetVector<int> yieldPartition;
    if (hasPartition(yieldOp))
      yieldPartition = getPartitionIds(yieldOp);
    addPartitionIds(yieldPartition, carrierPartition);
    setPartition(yieldOp, yieldPartition);
  }
  parentState = bodyState;
  parentState.currentToken = forOp.getResult(forOp.getNumResults() - 1);
  parentState.knownCarrierTokens.insert(parentState.currentToken);
  parentState.currentOwner = resultOwner;
  parentState.currentBuffers.clear();
  parentState.threadedTokens.push_back({forOp.getOperation(),
                                        parentState.currentToken,
                                        parentState.currentOwner,
                                        ThreadRecordKind::ForResult,
                                        plannedRegion, nullptr});
}

static void closeExistingCarrierForLoop(scf::ForOp forOp, EmitState &bodyState,
                                        EmitState &parentState) {
  mergeProtectedAccesses(parentState, bodyState);
  if (!bodyState.currentToken)
    return;
  auto yieldOp = getForYieldOp(forOp);
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    if (operand != bodyState.currentToken || idx >= forOp.getNumResults())
      continue;
    Value result = forOp.getResult(idx);
    if (!isa<AsyncTokenType>(result.getType()))
      continue;
    parentState.currentToken = result;
    parentState.currentSemaphore = bodyState.currentSemaphore;
    parentState.currentOwner = bodyState.currentOwner;
    parentState.currentBuffers.clear();
    parentState.knownCarrierTokens.insert(result);
    return;
  }
}

static scf::IfOp threadCarrierThroughIf(OpBuilder &b, scf::IfOp ifOp) {
  b.setInsertionPoint(ifOp);
  return replaceIfOpWithNewSignature(b, ifOp,
                                     TypeRange{b.getType<AsyncTokenType>()});
}

static void stampTokenYieldPartition(scf::YieldOp yieldOp, Value token,
                                     std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (hasPartition(yieldOp))
    ids = getPartitionIds(yieldOp);
  addPartitionIds(
      ids, partitionSetForTokenOrOwner(token, owner, yieldOp.getOperation()));
  if (!ids.empty())
    setPartition(yieldOp.getOperation(), ids);
}

static void appendTokenToYield(scf::YieldOp yieldOp, Value token,
                               std::optional<PartitionId> owner) {
  yieldOp->insertOperands(yieldOp.getNumOperands(), token);
  stampTokenYieldPartition(yieldOp, token, owner);
}

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src) {
  for (Operation *op : src.protectedAccesses)
    dst.protectedAccesses.insert(op);
  for (auto &kv : src.eventToken)
    if (!dst.eventToken.contains(kv.first))
      dst.eventToken[kv.first] = kv.second;
  for (auto &kv : src.eventSemaphore)
    if (!dst.eventSemaphore.contains(kv.first))
      dst.eventSemaphore[kv.first] = kv.second;
  for (auto &kv : src.eventBuffers)
    if (!dst.eventBuffers.contains(kv.first))
      dst.eventBuffers[kv.first] = kv.second;
  for (Value token : src.knownCarrierTokens)
    dst.knownCarrierTokens.insert(token);
  auto appendSync = [](auto &dstVec, const auto &srcVec) {
    for (const auto &record : srcVec) {
      bool seen = llvm::any_of(dstVec, [&](const auto &existing) {
        return existing.op == record.op;
      });
      if (!seen)
        dstVec.push_back(record);
    }
  };
  appendSync(dst.emittedAcquires, src.emittedAcquires);
  appendSync(dst.emittedReleases, src.emittedReleases);
  for (const EmittedBufferRecord &record : src.emittedBuffers) {
    bool seen = llvm::any_of(dst.emittedBuffers,
                             [&](const EmittedBufferRecord &existing) {
                               return existing.accessOp == record.accessOp &&
                                      existing.memberIdx == record.memberIdx &&
                                      existing.accessBuffer ==
                                          record.accessBuffer;
                             });
    if (!seen)
      dst.emittedBuffers.push_back(record);
  }
  for (const ThreadRecord &record : src.threadedTokens) {
    bool seen = llvm::any_of(dst.threadedTokens, [&](const ThreadRecord &existing) {
      return existing.op == record.op && existing.token == record.token;
    });
    if (!seen)
      dst.threadedTokens.push_back(record);
  }
  for (const PoisonTokenRecord &record :
       src.poisonTokenResultsAfterEmission) {
    bool seen = llvm::any_of(
        dst.poisonTokenResultsAfterEmission,
        [&](const PoisonTokenRecord &existing) {
          return existing.op == record.op;
        });
    if (!seen)
      dst.poisonTokenResultsAfterEmission.push_back(record);
  }
  for (Operation *op : src.eraseAfterEmission)
    dst.eraseAfterEmission.insert(op);
  for (auto &kv : src.reusedForCarrierSlots)
    dst.reusedForCarrierSlots[kv.first] = kv.second;
  for (auto &kv : src.reusedForTokenSlots)
    dst.reusedForTokenSlots[kv.first] = kv.second;
  for (auto &kv : src.reusedForPoisonTokens)
    dst.reusedForPoisonTokens[kv.first] = kv.second;
  for (auto &kv : src.stageCache)
    dst.stageCache[kv.first] = kv.second;
}

static unsigned countPlannedAnchors(const DenseMap<Operation *, SmallVector<unsigned, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(const DenseMap<Region *, SmallVector<unsigned, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static const char *syncGroupKindName(SyncGroupKind kind) {
  switch (kind) {
  case SyncGroupKind::InitialEmpty: return "initial-empty";
  case SyncGroupKind::Singleton: return "singleton";
  case SyncGroupKind::ReadyFanout: return "ready-fanout";
  case SyncGroupKind::DoneFanin: return "done-fanin";
  case SyncGroupKind::LinearChain: return "linear-chain";
  }
  llvm_unreachable("unknown sync group kind");
}

static bool hasSameSrcKey(const SyncEdge &a, const SyncEdge &b) {
  return a.srcOwner == b.srcOwner && a.srcEpoch == b.srcEpoch &&
         a.srcOp == b.srcOp && a.srcYieldRegion == b.srcYieldRegion;
}

static bool hasSameDstKey(const SyncEdge &a, const SyncEdge &b) {
  return a.dstOwner == b.dstOwner && a.dstOp == b.dstOp &&
         a.dstYieldRegion == b.dstYieldRegion;
}

static LogicalResult verifyLinearChainGroup(triton::FuncOp funcOp,
                                            const SyncPlan &sp,
                                            const SyncGroup &group) {
  if (group.edgeIdxs.size() < 2)
    return funcOp.emitError("nvws-insert-semas: verifier: linear-chain group "
                            "has fewer than two edges");
  SmallVector<ReadyFanoutKey, 4> srcKeys;
  SmallVector<DoneFaninKey, 4> dstKeys;
  for (auto [pos, idx] : llvm::enumerate(group.edgeIdxs)) {
    const SyncEdge &edge = sp.edges[idx];
    if ((!edge.srcOp && !edge.srcYieldRegion) ||
        (!edge.dstOp && !edge.dstYieldRegion))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has an unanchored endpoint");
    if (edge.srcYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has a yield source");
    if (edge.dstYieldRegion && !isIfYieldRegion(edge.dstYieldRegion) &&
        (pos + 1 != group.edgeIdxs.size() || edge.kind != SyncEdgeKind::Done))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has a non-terminal yield target");
    if (edge.srcOp && isa<scf::ForOp, scf::IfOp>(edge.srcOp))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "crosses a structured CFG anchor");
    ReadyFanoutKey srcKey{edge.srcOwner, edge.srcEpoch, edge.srcOp,
                          edge.srcYieldRegion};
    if (llvm::is_contained(srcKeys, srcKey))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain has "
                              "fanout from one produced version");
    srcKeys.push_back(srcKey);
    DoneFaninKey dstKey{edge.dstOwner, edge.dstOp, edge.dstYieldRegion};
    if (llvm::is_contained(dstKeys, dstKey))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain has "
                              "fanin to one target event");
    dstKeys.push_back(dstKey);
  }
  for (unsigned i = 1; i < group.edgeIdxs.size(); ++i) {
    const SyncEdge &prev = sp.edges[group.edgeIdxs[i - 1]];
    const SyncEdge &cur = sp.edges[group.edgeIdxs[i]];
    if (!sameOwner(prev.dstOwner, cur.srcOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain owner "
                              "sequence is not contiguous");
  }
  return success();
}

static LogicalResult verifyPlanBeforeEmission(triton::FuncOp funcOp,
                                              BufferGroup &group,
                                              const ResourcePlan &plan,
                                              const SyncPlan &sp,
                                              const OptSyncDag &dag) {
  if (sp.resource != dag.resource || sp.groupIdx != dag.groupIdx)
    return funcOp.emitError("nvws-insert-semas: verifier: OPT-SYNC resource "
                            "does not match RAW-SYNC resource");
  if (dag.edgeToGroup.size() != sp.edges.size())
    return funcOp.emitError("nvws-insert-semas: verifier: edge-to-group map "
                            "does not cover every raw edge");

  for (Operation *op : dag.accessOps) {
    if (!plan.useOwner.contains(op))
      return op->emitError("nvws-insert-semas: verifier: planned access has no "
                           "ownership record");
    const AccessEvent *event = findEvent(group, op);
    if (!event || !eventTouchesResource(*event, dag.resource.second))
      return op->emitError("nvws-insert-semas: verifier: planned access has no "
                           "touch for this resource");
  }

  for (const SyncEdge &edge : sp.edges) {
    if (edge.producedVersion.resourceKey != sp.resource.second)
      return funcOp.emitError("nvws-insert-semas: verifier: edge produced "
                              "version does not include this resource key");
    if (edge.producedVersion.epoch == 0)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "produced version epoch");
    if (!sameOwner(edge.postOwner, edge.dstOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: edge post-owner "
                              "does not match destination owner");
    if (edge.carrierChoice == CarrierTokenChoice::None)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "carrier-token choice");
    if (!edge.dstOp && !edge.dstYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "planned acquire anchor");
    if (!edge.srcOp && !edge.srcYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "planned release anchor");
    if (edge.srcOp && edge.dstOp && edge.dstOp->isProperAncestor(edge.srcOp))
      return edge.dstOp->emitError(
          "nvws-insert-semas: verifier: acquire would wait on a release "
          "control-dependent on the same acquire");
    if ((edge.kind == SyncEdgeKind::Ready ||
         edge.kind == SyncEdgeKind::Handoff) &&
        !sameOwner(edge.preOwner, edge.srcOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: edge pre-owner "
                              "does not match source owner");
  }

  std::optional<PartitionId> firstWriterOwner;
  Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner);
  unsigned initialGroups = 0;
  for (auto [groupIdx, syncGroup] : llvm::enumerate(dag.groups)) {
    switch (syncGroup.kind) {
    case SyncGroupKind::InitialEmpty:
      ++initialGroups;
      if (!firstWriter)
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group without a writer");
      if (syncGroup.initialOp != firstWriter ||
          syncGroup.initialOwner != firstWriterOwner)
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group is not anchored at the first writer");
      if (!syncGroup.edgeIdxs.empty())
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group must not own raw edges");
      break;
    case SyncGroupKind::Singleton:
      if (syncGroup.edgeIdxs.size() != 1)
        return funcOp.emitError("nvws-insert-semas: verifier: singleton group "
                                "does not contain exactly one edge");
      break;
    case SyncGroupKind::ReadyFanout: {
      if (syncGroup.edgeIdxs.size() < 2)
        return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                "group has fewer than two edges");
      const SyncEdge &probe = sp.edges[syncGroup.edgeIdxs.front()];
      if (!probe.srcOp && !probe.srcYieldRegion)
        return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                "group has no release anchor");
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (edge.kind != SyncEdgeKind::Ready)
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "contains a non-ready edge");
        if (!hasSameSrcKey(probe, edge))
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "group does not share one produced version");
        if (!(probe.producedVersion == edge.producedVersion))
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "group does not share one produced version "
                                  "key");
        if (edge.dstOp) {
          const AccessEvent *dst = findEvent(group, edge.dstOp);
          if (dst && (!eventConsumes(*dst, dag.resource.second) ||
                      eventProduces(*dst, dag.resource.second)))
            return edge.dstOp->emitError(
                "nvws-insert-semas: verifier: ready-fanout target is not a "
                "read-only consumer");
        }
      }
      break;
    }
    case SyncGroupKind::DoneFanin: {
      if (syncGroup.edgeIdxs.size() < 2)
        return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                "group has fewer than two edges");
      const SyncEdge &probe = sp.edges[syncGroup.edgeIdxs.front()];
      SmallVector<std::optional<PartitionId>, 4> srcOwners;
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (edge.kind != SyncEdgeKind::Done)
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "contains a non-done edge");
        if (!hasSameDstKey(probe, edge))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "group does not share one target event");
        if (!(probe.producedVersion == edge.producedVersion))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "group does not retire one produced version "
                                  "key");
        if (llvm::is_contained(srcOwners, edge.srcOwner))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin has "
                                  "multiple releases from one source owner");
        srcOwners.push_back(edge.srcOwner);
      }
      break;
    }
    case SyncGroupKind::LinearChain:
      if (failed(verifyLinearChainGroup(funcOp, sp, syncGroup)))
        return failure();
      break;
    }

    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      if (edgeIdx >= sp.edges.size())
        return funcOp.emitError("nvws-insert-semas: verifier: ")
               << syncGroupKindName(syncGroup.kind)
               << " group references an out-of-range raw edge";
      if (dag.edgeToGroup[edgeIdx] != groupIdx)
        return funcOp.emitError("nvws-insert-semas: verifier: edge-to-group "
                                "map disagrees with OPT-SYNC group");
    }
  }
  if (firstWriter && initialGroups != 1)
    return firstWriter->emitError("nvws-insert-semas: verifier: expected one "
                                  "initial-empty group for first writer");
  if (!firstWriter && initialGroups != 0)
    return funcOp.emitError("nvws-insert-semas: verifier: unexpected "
                            "initial-empty group");

  auto emitReleasePlanError = [&](Operation *anchor, const Twine &message) {
    if (anchor)
      return anchor->emitError(message);
    return funcOp.emitError(message);
  };
  auto verifyReleaseAction = [&](const PlannedRelease &action,
                                 Operation *anchor) -> LogicalResult {
    if (action.groupIdx >= dag.groups.size())
      return emitReleasePlanError(
          anchor, "nvws-insert-semas: verifier: planned release references an "
                  "invalid group");
    if (action.edgeIdxs.empty())
      return emitReleasePlanError(
          anchor, "nvws-insert-semas: verifier: planned release has no "
                  "transition edge");
    for (unsigned edgeIdx : action.edgeIdxs) {
      if (edgeIdx >= sp.edges.size())
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release references "
                    "an invalid edge");
      if (edgeIdx >= dag.edgeToGroup.size() ||
          dag.edgeToGroup[edgeIdx] != action.groupIdx)
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release edge does "
                    "not belong to its group");
      if (!edgeRequiresRelease(sp.edges[edgeIdx]))
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release is not "
                    "backed by a partition transition edge");
    }
    return success();
  };
  auto verifyReleaseOpMap =
      [&](const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &map)
      -> LogicalResult {
    for (auto &kv : map)
      for (const PlannedRelease &action : kv.second)
        if (failed(verifyReleaseAction(action, kv.first)))
          return failure();
    return success();
  };
  auto verifyReleaseYieldMap =
      [&](const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &map)
      -> LogicalResult {
    for (auto &kv : map)
      for (const PlannedRelease &action : kv.second)
        if (failed(verifyReleaseAction(
                action, kv.first ? kv.first->getParentOp() : nullptr)))
          return failure();
    return success();
  };
  if (failed(verifyReleaseOpMap(dag.releaseBeforeOp)) ||
      failed(verifyReleaseOpMap(dag.releaseAfterOp)) ||
      failed(verifyReleaseYieldMap(dag.releaseBeforeYield)) ||
      failed(verifyReleaseYieldMap(dag.releaseAfterYield)))
    return failure();

  WalkResult walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionContainsAccess(forOp.getRegion(), dag))
        return WalkResult::advance();
      if (!dag.threadForOps.contains(forOp.getOperation()))
        return WalkResult::advance();
      auto it = plan.regionOwners.find(&forOp.getRegion());
      if (it == plan.regionOwners.end() || !it->second.carried ||
          !sameOwner(it->second.entry, it->second.exit)) {
        op->emitError("nvws-insert-semas: verifier: scf.for has no known "
                      "loop-carried owner/state for this resource");
        return WalkResult::interrupt();
      }
      if (!plan.yieldOwner.contains(getForYieldOp(forOp))) {
        op->emitError("nvws-insert-semas: verifier: scf.for has no planned "
                      "yield owner for this resource");
        return WalkResult::interrupt();
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionContainsAccess(ifOp.getThenRegion(), dag);
      bool elseHas = regionContainsAccess(ifOp.getElseRegion(), dag);
      if (!thenHas && !elseHas)
        return WalkResult::advance();
      if (!dag.threadIfOps.contains(ifOp.getOperation()))
        return WalkResult::advance();
      auto thenIt = plan.regionOwners.find(&ifOp.getThenRegion());
      auto elseIt = plan.regionOwners.find(&ifOp.getElseRegion());
      if (thenIt == plan.regionOwners.end() ||
          elseIt == plan.regionOwners.end() ||
          !sameOwner(thenIt->second.entry, thenIt->second.exit) ||
          !sameOwner(elseIt->second.entry, elseIt->second.exit) ||
          !sameOwner(thenIt->second.exit, elseIt->second.exit)) {
        op->emitError("nvws-insert-semas: verifier: scf.if join has no known "
                      "owner/state on every path for this resource");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

static bool hasSyncRecord(ArrayRef<EmittedSyncRecord> records, unsigned groupIdx,
                          SyncAnchorKind kind, Operation *anchor,
                          Region *yieldRegion) {
  return llvm::any_of(records, [&](const EmittedSyncRecord &record) {
    return record.groupIdx == groupIdx && record.kind == kind &&
           record.anchor == anchor && record.yieldRegion == yieldRegion &&
           record.op;
  });
}

static bool hasSyncRecord(ArrayRef<EmittedSyncRecord> records,
                          const PlannedRelease &action, SyncAnchorKind kind,
                          Operation *anchor, Region *yieldRegion) {
  return llvm::any_of(records, [&](const EmittedSyncRecord &record) {
    return record.groupIdx == action.groupIdx && record.edgeIdxs == action.edgeIdxs &&
           record.kind == kind && record.anchor == anchor &&
           record.yieldRegion == yieldRegion && record.op;
  });
}

static bool isRecordPlanned(const EmittedSyncRecord &record,
                            const DenseMap<Operation *, SmallVector<unsigned, 2>>
                                &opMap,
                            const DenseMap<Region *, SmallVector<unsigned, 2>>
                                &yieldMap) {
  if (record.anchor) {
    auto it = opMap.find(record.anchor);
    return it != opMap.end() && llvm::is_contained(it->second, record.groupIdx);
  }
  if (record.yieldRegion) {
    auto it = yieldMap.find(record.yieldRegion);
    return it != yieldMap.end() &&
           llvm::is_contained(it->second, record.groupIdx);
  }
  return false;
}

static bool isRecordPlanned(
    const EmittedSyncRecord &record,
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &opMap,
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &yieldMap) {
  auto matches = [&](ArrayRef<PlannedRelease> planned) {
    return llvm::any_of(planned, [&](const PlannedRelease &action) {
      return action.groupIdx == record.groupIdx &&
             action.edgeIdxs == record.edgeIdxs;
    });
  };
  if (record.anchor) {
    auto it = opMap.find(record.anchor);
    return it != opMap.end() && matches(it->second);
  }
  if (record.yieldRegion) {
    auto it = yieldMap.find(record.yieldRegion);
    return it != yieldMap.end() && matches(it->second);
  }
  return false;
}

static LogicalResult verifyPlannedSyncRecords(
    triton::FuncOp funcOp, SyncAnchorKind opKind, SyncAnchorKind yieldKind,
    const DenseMap<Operation *, SmallVector<unsigned, 2>> &opMap,
    const DenseMap<Region *, SmallVector<unsigned, 2>> &yieldMap,
    ArrayRef<EmittedSyncRecord> records) {
  for (auto &kv : opMap)
    for (unsigned groupIdx : kv.second)
      if (!hasSyncRecord(records, groupIdx, opKind, kv.first, nullptr))
        return kv.first->emitError("nvws-insert-semas: post-emission verifier: "
                                   "planned sync op is missing at op anchor");
  for (auto &kv : yieldMap)
    for (unsigned groupIdx : kv.second)
      if (!hasSyncRecord(records, groupIdx, yieldKind, nullptr, kv.first))
        return funcOp.emitError("nvws-insert-semas: post-emission verifier: "
                                "planned sync op is missing at yield anchor");
  return success();
}

static LogicalResult verifyPlannedSyncRecords(
    triton::FuncOp funcOp, SyncAnchorKind opKind, SyncAnchorKind yieldKind,
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &opMap,
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &yieldMap,
    ArrayRef<EmittedSyncRecord> records) {
  for (auto &kv : opMap)
    for (const PlannedRelease &action : kv.second)
      if (!hasSyncRecord(records, action, opKind, kv.first, nullptr))
        return kv.first->emitError("nvws-insert-semas: post-emission verifier: "
                                   "planned release is missing at op anchor");
  for (auto &kv : yieldMap)
    for (const PlannedRelease &action : kv.second)
      if (!hasSyncRecord(records, action, yieldKind, nullptr, kv.first))
        return funcOp.emitError("nvws-insert-semas: post-emission verifier: "
                                "planned release is missing at yield anchor");
  return success();
}

static LogicalResult verifyStageCluster(Operation *emitted,
                                        StageCluster expected) {
  if (!expected)
    return success();
  StageCluster actual = getStageCluster(emitted);
  if (!actual || *actual != *expected)
    return emitted->emitError("nvws-insert-semas: post-emission verifier: "
                              "emitted semaphore op is missing planned "
                              "stage/cluster attrs");
  return success();
}

static bool operationUsesValue(Operation *op, Value value) {
  for (OpOperand &operand : op->getOpOperands())
    if (operand.get() == value)
      return true;
  return false;
}

static Operation *nearestStructuredParent(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<scf::ForOp, scf::IfOp>(parent))
      return parent;
  return nullptr;
}

static bool tokenEscapesWithoutYield(Value token) {
  Operation *def = token.getDefiningOp();
  if (!def)
    return false;
  Operation *boundary = nearestStructuredParent(def);
  if (!boundary)
    return false;
  for (Operation *user : token.getUsers()) {
    if (boundary->isAncestor(user))
      continue;
    if (isa<scf::YieldOp>(user))
      continue;
    return true;
  }
  return false;
}

static bool partitionSetMatchesOwner(SetVector<int> partitions,
                                     std::optional<PartitionId> owner) {
  if (!owner)
    return true;
  return partitions.size() == 1 && *partitions.begin() == owner->first;
}

static LogicalResult verifyThreadRecord(const ThreadRecord &record,
                                        const ResourcePlan &plan,
                                        const SyncPlan &sp,
                                        const OptSyncDag &dag) {
  if (!record.token || !isa<AsyncTokenType>(record.token.getType()))
    return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                "planned CFG-threaded carrier is not an "
                                "async token");
  switch (record.kind) {
  case ThreadRecordKind::ForIterArg: {
    auto forOp = dyn_cast<scf::ForOp>(record.op);
    if (!forOp || !isa<BlockArgument>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.for iter_arg carrier is absent");
    Region *region = record.plannedRegion ? record.plannedRegion
                                           : &forOp.getRegion();
    auto it = plan.regionOwners.find(region);
    if (it == plan.regionOwners.end() ||
        !sameOwner(it->second.entry, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.for iter_arg carrier owner does not "
                                  "match the ownership plan");
    return success();
  }
  case ThreadRecordKind::ForResult: {
    auto forOp = dyn_cast<scf::ForOp>(record.op);
    if (!forOp || !isa<OpResult>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.for result carrier is absent");
    Region *region = record.plannedRegion ? record.plannedRegion
                                           : &forOp.getRegion();
    auto it = plan.regionOwners.find(region);
    std::optional<PartitionId> expectedOwner;
    if (it != plan.regionOwners.end())
      expectedOwner = it->second.exit;
    if (auto linearOwner =
            linearChainLoopYieldSourceOwner(forOp.getOperation(), dag, sp))
      expectedOwner = linearOwner;
    if (it == plan.regionOwners.end() || !sameOwner(expectedOwner, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.for result carrier owner does not "
                                  "match the ownership plan");
    if (hasPartition(forOp)) {
      auto outputs = getPartitionOutputs(forOp);
      auto result = cast<OpResult>(record.token);
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber >= outputs.size() ||
          !partitionSetMatchesOwner(outputs[resultNumber], record.owner))
        return record.op->emitError(
            "nvws-insert-semas: post-emission verifier: scf.for result "
            "carrier partition output does not match the ownership plan");
    }
    return success();
  }
  case ThreadRecordKind::IfResult: {
    auto ifOp = dyn_cast<scf::IfOp>(record.op);
    if (!ifOp || !isa<OpResult>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.if result carrier is absent");
    return success();
  }
  }
  llvm_unreachable("unknown thread record kind");
}

static LogicalResult verifyPostEmission(const OptSyncDag &dag, BufferGroup &group,
                                        const ResourcePlan &plan,
                                        const SyncPlan &sp,
                                        const EmitState &state) {
  unsigned plannedAcquires =
      countPlannedAnchors(dag.acquireBeforeOp) +
      countPlannedAnchors(dag.acquireBeforeYield);
  unsigned plannedReleases =
      countPlannedAnchors(dag.releaseBeforeOp) +
      countPlannedAnchors(dag.releaseBeforeYield) +
      countPlannedAnchors(dag.releaseAfterOp) +
      countPlannedAnchors(dag.releaseAfterYield);
  if (state.emittedAcquires.size() < plannedAcquires)
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: post-emission verifier: fewer acquires emitted "
        "than planned");
  if (state.emittedReleases.size() < plannedReleases)
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: post-emission verifier: fewer releases emitted "
        "than planned");

  auto funcOp = group.members.front().allocOp->getParentOfType<triton::FuncOp>();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::AcquireBeforeOp,
                                      SyncAnchorKind::AcquireBeforeYield,
                                      dag.acquireBeforeOp,
                                      dag.acquireBeforeYield,
                                      state.emittedAcquires)))
    return failure();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::ReleaseBeforeOp,
                                      SyncAnchorKind::ReleaseBeforeYield,
                                      dag.releaseBeforeOp,
                                      dag.releaseBeforeYield,
                                      state.emittedReleases)))
    return failure();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::ReleaseAfterOp,
                                      SyncAnchorKind::ReleaseAfterYield,
                                      dag.releaseAfterOp,
                                      dag.releaseAfterYield,
                                      state.emittedReleases)))
    return failure();

  for (const EmittedSyncRecord &record : state.emittedAcquires) {
    if (!isRecordPlanned(record, dag.acquireBeforeOp, dag.acquireBeforeYield))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "unplanned acquire was emitted");
    if (failed(verifyStageCluster(record.op, record.expectedStageCluster)))
      return failure();
    if (tokenEscapesWithoutYield(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "branch-local acquire token escaped without "
                                  "CFG threading");
  }
  auto releaseIsPlanned = [&](const EmittedSyncRecord &record) {
    switch (record.kind) {
    case SyncAnchorKind::ReleaseBeforeOp:
    case SyncAnchorKind::ReleaseBeforeYield:
      return isRecordPlanned(record, dag.releaseBeforeOp, dag.releaseBeforeYield);
    case SyncAnchorKind::ReleaseAfterOp:
    case SyncAnchorKind::ReleaseAfterYield:
      return isRecordPlanned(record, dag.releaseAfterOp, dag.releaseAfterYield);
    case SyncAnchorKind::AcquireBeforeOp:
    case SyncAnchorKind::AcquireBeforeYield:
      return false;
    }
    llvm_unreachable("unknown sync anchor kind");
  };
  for (const EmittedSyncRecord &record : state.emittedReleases) {
    if (!releaseIsPlanned(record))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "unplanned release was emitted");
    if (!state.knownCarrierTokens.contains(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "release token does not trace to a planned "
                                  "carrier acquire");
    if (failed(verifyStageCluster(record.op, record.expectedStageCluster)))
      return failure();
  }

  for (Operation *op : dag.accessOps) {
    const AccessEvent *event = findEvent(group, op);
    if (!event) continue;
    SmallVector<const AccessTouch *, 4> touches;
    collectTouchesForResource(*event, dag.resource.second, touches);
    if (touches.empty()) continue;
    if (!state.protectedAccesses.contains(op))
      return op->emitError("nvws-insert-semas: post-emission verifier: "
                           "planned access was not rewritten through "
                           "nvws.semaphore.buffer");
    SmallVector<unsigned, 4> consumedRecords;
    for (const AccessTouch *touch : touches) {
      auto bufferIt =
          llvm::find_if(state.emittedBuffers, [&](const EmittedBufferRecord &rec) {
            if (rec.accessOp != op || rec.memberIdx != touch->memberIdx)
              return false;
            unsigned recordIdx = static_cast<unsigned>(&rec - state.emittedBuffers.data());
            return !llvm::is_contained(consumedRecords, recordIdx);
          });
      if (bufferIt == state.emittedBuffers.end())
        return op->emitError("nvws-insert-semas: post-emission verifier: "
                             "planned access has no semaphore.buffer op for a "
                             "planned member touch");
      unsigned recordIdx =
          static_cast<unsigned>(&*bufferIt - state.emittedBuffers.data());
      consumedRecords.push_back(recordIdx);
      if (bufferIt->backingIdx >= bufferIt->buffers.size())
        return bufferIt->bufferOp->emitError(
            "nvws-insert-semas: post-emission verifier: semaphore.buffer member "
            "does not match planned touch");
      if (!operationUsesValue(bufferIt->retargetOp, bufferIt->accessBuffer))
        return bufferIt->retargetOp->emitError(
            "nvws-insert-semas: post-emission verifier: planned memory access "
            "does not use the planned nvws.semaphore.buffer result");
      if (!state.knownCarrierTokens.contains(bufferIt->token))
        return bufferIt->bufferOp->emitError(
            "nvws-insert-semas: post-emission verifier: semaphore.buffer token "
            "does not trace to a planned acquire");
      if (failed(verifyStageCluster(bufferIt->bufferOp,
                                    bufferIt->expectedStageCluster)))
        return failure();
    }
  }

  for (const ThreadRecord &record : state.threadedTokens)
    if (failed(verifyThreadRecord(record, plan, sp, dag)))
      return failure();
  return success();
}

static LogicalResult emitResourceBlock(Block &block, const OptSyncDag &dag,
                                       const SyncPlan &sp,
                                       const ResourcePlan &plan,
                                       BufferGroup &group,
                                       const GroupBacking &backing,
                                       EmitState &state,
                                       Region *plannedRegion);

static LogicalResult emitResourceRegion(Region &region, const OptSyncDag &dag,
                                        const SyncPlan &sp,
                                        const ResourcePlan &plan,
                                        BufferGroup &group,
                                        const GroupBacking &backing,
                                        EmitState &state,
                                        Region *plannedRegion = nullptr) {
  Region *regionKey = plannedRegion ? plannedRegion : &region;
  for (Block &block : region)
    if (failed(emitResourceBlock(block, dag, sp, plan, group, backing, state,
                                 regionKey)))
      return failure();
  return success();
}

static LogicalResult emitResourceBlock(Block &block, const OptSyncDag &dag,
                                       const SyncPlan &sp,
                                       const ResourcePlan &plan,
                                       BufferGroup &group,
                                       const GroupBacking &backing,
                                       EmitState &state,
                                       Region *plannedRegion) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<scf::YieldOp>(op)) {
      SmallVector<AcquireRecord, 2> yieldAcquires;
      if (failed(emitBeforeYieldSync(&op, plannedRegion, dag, sp, group,
                                     state, yieldAcquires)))
        return failure();
      continue;
    }

    AccessEvent *event = nullptr;
    SmallVector<const AccessTouch *, 4> touches;
    if (dag.accessOps.contains(&op)) {
      for (AccessEvent &candidate : group.events)
        if (candidate.op == &op) {
          event = &candidate;
          collectTouchesForResource(candidate, dag.resource.second, touches);
          break;
        }
      if (event && event->owner)
        state.stageCache[*event->owner] = getStageCluster(&op);
    }

    SmallVector<AcquireRecord, 2> acquires;
    if (failed(emitBeforeOpSync(&op, dag, sp, group, state, acquires)))
      return failure();

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Operation *oldForOp = forOp.getOperation();
      Region *plannedForRegion = &forOp.getRegion();
      scf::ForOp activeForOp = forOp;
      EmitState bodyState = state;
      bool threaded = shouldThreadForRegion(forOp, dag);
      SmallVector<const AccessTouch *, 4> prebufferTouches;
      if (threaded && canPrebufferLocalRegionEntry(
                          forOp.getOperation(), forOp.getRegion(), acquires,
                          dag, group, prebufferTouches)) {
        OpBuilder prebufferBuilder(forOp);
        prebufferLocalRegionEntry(prebufferBuilder, forOp.getOperation(),
                                  prebufferTouches, acquires.front(), group,
                                  backing, state);
        bodyState = state;
        threaded = false;
      }
      OpBuilder loopBuilder(forOp);
      if (threaded) {
        std::optional<PartitionId> recordOwner = state.currentOwner;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end())
          recordOwner = regionOwnerIt->second.entry;
        FailureOr<scf::ForOp> threadedFor =
            threadCarrierThroughFor(loopBuilder, forOp, bodyState,
                                    plannedForRegion, recordOwner, group,
                                    dag.memberIndices, dag.resource.second);
        if (failed(threadedFor))
          return failure();
        activeForOp = *threadedFor;
      }
      if (failed(emitResourceRegion(activeForOp.getRegion(), dag, sp, plan,
                                    group, backing, bodyState,
                                    plannedForRegion)))
        return failure();
      if (threaded) {
        std::optional<PartitionId> ownerAtYield = bodyState.currentOwner;
        bool overrideOwnerAtYield = false;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end() &&
            (!linearChainEntersFor(oldForOp, dag, sp) ||
             dag.acquireBeforeYield.contains(plannedForRegion))) {
          ownerAtYield = regionOwnerIt->second.exit;
          overrideOwnerAtYield = true;
        }
        closeCarrierForLoop(activeForOp, bodyState, state, ownerAtYield,
                            overrideOwnerAtYield, plannedForRegion);
        if (failed(emitTmemLinearLoopExitDrain(activeForOp, plannedForRegion,
                                               dag, sp, group, state)))
          return failure();
      } else {
        closeExistingCarrierForLoop(activeForOp, bodyState, state);
      }
      if (failed(emitAfterOpSync(oldForOp, activeForOp.getOperation(), dag, sp,
                                 group, state)))
        return failure();
      if (failed(emitDeferredNestedAfterOpSync(activeForOp.getOperation(), dag,
                                               sp, group, state)))
        return failure();
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      Operation *oldIfOp = ifOp.getOperation();
      Region *plannedThenRegion = &ifOp.getThenRegion();
      Region *plannedElseRegion = &ifOp.getElseRegion();
      SmallVector<const AccessTouch *, 4> prebufferTouches;
      if (canPrebufferLocalIfEntry(ifOp, acquires, dag, group,
                                   prebufferTouches)) {
        OpBuilder prebufferBuilder(ifOp);
        prebufferLocalRegionEntry(prebufferBuilder, ifOp.getOperation(),
                                  prebufferTouches, acquires.front(), group,
                                  backing, state);
      }
      bool threaded = shouldThreadIfRegion(ifOp, dag);
      scf::IfOp activeIfOp = ifOp;
      unsigned oldNumResults = ifOp.getNumResults();
      auto oldPartitionIds =
          hasPartition(ifOp) ? getPartitionIds(ifOp) : SetVector<int>();
      auto oldPartitionOutputs =
          hasPartition(ifOp) ? getPartitionOutputs(ifOp)
                             : SmallVector<SetVector<int>, 4>();
      SmallVector<unsigned, 4> reusableIfTokenResults;
      if (threaded)
        for (auto [idx, result] : llvm::enumerate(ifOp.getResults()))
          if (isa<AsyncTokenType>(result.getType()))
            reusableIfTokenResults.push_back(static_cast<unsigned>(idx));
      std::optional<unsigned> reusableIfTokenResult;
      if (!reusableIfTokenResults.empty())
        reusableIfTokenResult = reusableIfTokenResults.front();
      bool reuseExistingTokenResult = reusableIfTokenResult.has_value();
      if (threaded) {
        if (ifOp.getElseRegion().empty())
          return ifOp.emitError(
              "nvws-insert-semas: planned scf.if carrier threading requires "
              "an else path producer");
        if (!reuseExistingTokenResult) {
          OpBuilder ifBuilder(ifOp);
          activeIfOp = threadCarrierThroughIf(ifBuilder, ifOp);
        }
      }
      Value incomingToken = state.currentToken;
      Value incomingSemaphore = state.currentSemaphore;
      EmitState thenState = state;
      if (failed(emitResourceRegion(activeIfOp.getThenRegion(), dag, sp, plan,
                                    group, backing, thenState,
                                    plannedThenRegion)))
        return failure();
      EmitState elseState = state;
      if (!activeIfOp.getElseRegion().empty() &&
          failed(emitResourceRegion(activeIfOp.getElseRegion(), dag, sp, plan,
                                    group, backing, elseState,
                                    plannedElseRegion)))
        return failure();
      if (threaded) {
        Value thenToken =
            thenState.currentToken ? thenState.currentToken : incomingToken;
        Value elseToken =
            elseState.currentToken ? elseState.currentToken : incomingToken;
        if (!thenToken || !elseToken)
          return activeIfOp.emitError(
              "nvws-insert-semas: planned scf.if carrier threading has no "
              "token producer on every path");
        if (reuseExistingTokenResult) {
          unsigned tokenResultIdx = *reusableIfTokenResult;
          Value poison;
          if (reusableIfTokenResults.size() > 1) {
            Operation *poisonAnchor =
                getDominatingPoisonAnchor(activeIfOp.getOperation());
            OpBuilder poisonBuilder(poisonAnchor);
            poisonBuilder.setInsertionPoint(poisonAnchor);
            poison = ub::PoisonOp::create(poisonBuilder, activeIfOp.getLoc(),
                                          poisonBuilder.getType<AsyncTokenType>());
          }
          for (unsigned resultIdx : reusableIfTokenResults) {
            activeIfOp.thenYield().setOperand(
                resultIdx, resultIdx == tokenResultIdx ? thenToken : poison);
            activeIfOp.elseYield().setOperand(
                resultIdx, resultIdx == tokenResultIdx ? elseToken : poison);
          }
          stampTokenYieldPartition(activeIfOp.thenYield(), thenToken,
                                   thenState.currentOwner);
          stampTokenYieldPartition(activeIfOp.elseYield(), elseToken,
                                   elseState.currentOwner);
        } else {
          appendTokenToYield(activeIfOp.thenYield(), thenToken,
                             thenState.currentOwner);
          appendTokenToYield(activeIfOp.elseYield(), elseToken,
                             elseState.currentOwner);
        }
        thenState.knownCarrierTokens.insert(thenToken);
        elseState.knownCarrierTokens.insert(elseToken);
        std::optional<PartitionId> outOwner =
            thenState.currentOwner == elseState.currentOwner
                ? thenState.currentOwner
                : std::nullopt;
        SetVector<int> outPartition = partitionSetForOwner(outOwner);
        if (outPartition.empty()) {
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            thenToken, thenState.currentOwner,
                                            activeIfOp.getOperation()));
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            elseToken, elseState.currentOwner,
                                            activeIfOp.getOperation()));
        }
        if (hasPartition(activeIfOp)) {
          addPartitionIds(oldPartitionIds, outPartition);
          if (!reuseExistingTokenResult)
            oldPartitionOutputs.push_back(outPartition);
          setPartition(activeIfOp, oldPartitionIds);
          setPartitionOutputs(activeIfOp, oldPartitionOutputs);
        }
        Value thenSemaphore =
            thenState.currentSemaphore ? thenState.currentSemaphore
                                       : incomingSemaphore;
        Value elseSemaphore =
            elseState.currentSemaphore ? elseState.currentSemaphore
                                       : incomingSemaphore;
        Value joinedSemaphore =
            thenSemaphore == elseSemaphore
                ? thenSemaphore
                : (outOwner ? (thenSemaphore ? thenSemaphore : elseSemaphore)
                            : Value());
        EmitState joinedState = state;
        mergeProtectedAccesses(joinedState, thenState);
        mergeProtectedAccesses(joinedState, elseState);
        joinedState.currentToken =
            activeIfOp.getResult(reuseExistingTokenResult
                                     ? *reusableIfTokenResult
                                     : oldNumResults);
        joinedState.currentSemaphore = joinedSemaphore;
        joinedState.currentOwner = outOwner;
        joinedState.currentBuffers.clear();
        joinedState.knownCarrierTokens.insert(joinedState.currentToken);
        joinedState.threadedTokens.push_back(
            {activeIfOp.getOperation(), joinedState.currentToken, outOwner,
             ThreadRecordKind::IfResult, plannedThenRegion, plannedElseRegion});
        state = joinedState;
      } else {
        mergeProtectedAccesses(state, thenState);
        mergeProtectedAccesses(state, elseState);
      }
      if (failed(emitAfterOpSync(oldIfOp, activeIfOp.getOperation(), dag, sp,
                                 group, state)))
        return failure();
      if (failed(emitDeferredNestedAfterOpSync(activeIfOp.getOperation(), dag,
                                               sp, group, state)))
        return failure();
      if (threaded && !reuseExistingTokenResult)
        state.eraseAfterEmission.insert(oldIfOp);
      continue;
    }

    if (event && !touches.empty()) {
      OpBuilder b(&op);
      b.setInsertionPoint(&op);
      if (failed(emitAccessEvent(b, *event, touches, acquires, group, dag,
                                 backing, state)))
        return failure();
    }
    if (failed(emitAfterOpSync(&op, dag, sp, group, state)))
      return failure();
  }
  return success();
}

static LogicalResult emitResource(triton::FuncOp funcOp, BufferGroup &group,
                                  const ResourcePlan &plan, const SyncPlan &sp,
                                  const OptSyncDag &dag,
                                  DenseMap<BackingKey, GroupBacking> &backings,
                                  const DenseMap<unsigned, int> &numStagesByGroup,
                                  SetVector<Operation *> &eraseAfterEmission) {
  if (dag.groups.empty()) return success();
  if (failed(verifyPlanBeforeEmission(funcOp, group, plan, sp, dag)))
    return failure();
  GroupBacking &backing =
      ensureGroupBacking(group, dag.groupIdx, dag.resource.second,
                         plan.memberIndices, backings, numStagesByGroup);
  EmitState state;
  state.semas = createResourceSemaphores(dag, sp, group, backing);
  if (failed(emitResourceRegion(funcOp.getBody(), dag, sp, plan, group, backing,
                                state)))
    return failure();
  DenseSet<Operation *> poisonedTokenOps;
  for (const PoisonTokenRecord &record :
       state.poisonTokenResultsAfterEmission) {
    if (!record.op || !poisonedTokenOps.insert(record.op).second)
      continue;
    OpBuilder poisonBuilder(record.insertBefore ? record.insertBefore
                                                : record.op);
    poisonTokenResults(poisonBuilder, record.op, record.insertBefore);
  }
  if (failed(verifyPostEmission(dag, group, plan, sp, state)))
    return failure();
  for (Operation *op : llvm::reverse(state.eraseAfterEmission))
    eraseAfterEmission.insert(op);
  return success();
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
    SmallVector<unsigned, 1> fallbackTouchIdx{tIdx};
    auto allTouchIdxs = plan.useTouchIdxs.find(&op);
    ArrayRef<unsigned> touchIdxs =
        allTouchIdxs == plan.useTouchIdxs.end()
            ? ArrayRef<unsigned>(fallbackTouchIdx)
            : ArrayRef<unsigned>(allTouchIdxs->second);
    for (unsigned touchIdx : touchIdxs) {
      if (touchIdx >= event->touches.size()) continue;
      AccessTouch &touch = event->touches[touchIdx];
      bool reads = hasRead(touch.effect);
      bool writes = hasWrite(touch.effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m" << touch.memberIdx
                   << "  " << op.getName().getStringRef() << "  use "
                   << ownerStr(&op, useIt->second) << "\n";
    }
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

// Defined after the RAW dag dump; forward-declared for the OPT overlay below.
static std::string ownerSetStr(Operation *anchor,
                               ArrayRef<std::optional<PartitionId>> owners);

// OPT-SYNC-DAG dump context. When passed to the shared RAW walk, it drives the
// only deviations the contract allows: edges in a ReadyFanout group render one
// shared `S_full` release (with a `{{..}}` dst-set) plus a per-consumer
// `S_full` acquire; edges in a DoneFanin group render a per-reader `S_empty`
// release plus one shared `S_empty` acquire (`pending={{..}}`). Every other
// edge — singleton, linear chain, initial permit — renders per-edge exactly as
// RAW, so a resource with no fan-in/out combine is byte-identical to RAW.
// `rendered` dedups each group's single shared row.
struct OptDumpCtx {
  const OptSyncDag *dag;
  DenseSet<unsigned> rendered;
};

// Print the release+acquire pair for each edge anchored at `key`, in the order
// they were recorded. Both rows render at the same depth. With `octx` set
// (OPT-SYNC-DAG), only ReadyFanout/DoneFanin edges deviate from the per-edge
// RAW form; see OptDumpCtx.
static void printEdgesAt(SmallVector<unsigned, 2> *edgeIdxs, SyncPlan &sp,
                         Operation *anchor, unsigned depth,
                         OptDumpCtx *octx = nullptr) {
  if (!edgeIdxs) return;
  if (!octx) {
    for (unsigned idx : *edgeIdxs) {
      renderReleaseRow(depth, anchor, sp.edges[idx]);
      renderAcquireRow(depth, anchor, sp.edges[idx]);
    }
    return;
  }
  const OptSyncDag &dag = *octx->dag;
  SmallVector<unsigned, 2> faninHere;
  for (unsigned idx : *edgeIdxs) {
    const SyncEdge &edge = sp.edges[idx];
    unsigned gi = dag.edgeToGroup[idx];
    const SyncGroup &g = dag.groups[gi];
    if (g.kind == SyncGroupKind::ReadyFanout) {
      if (octx->rendered.insert(gi).second) {
        SmallVector<std::optional<PartitionId>, 4> dsts;
        for (unsigned ei : g.edgeIdxs) dsts.push_back(sp.edges[ei].dstOwner);
        llvm::errs() << treePrefix(depth) << "|- r  S_full  release  "
                     << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].srcOwner)
                     << " -> " << ownerSetStr(anchor, dsts) << "\n";
      }
      llvm::errs() << treePrefix(depth) << "|- a  S_full  acquire  "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
    } else if (g.kind == SyncGroupKind::DoneFanin) {
      llvm::errs() << treePrefix(depth) << "|- r  S_empty  release  "
                   << ownerStr(anchor, edge.srcOwner) << " -> "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
      if (!llvm::is_contained(faninHere, gi)) faninHere.push_back(gi);
    } else {
      renderReleaseRow(depth, anchor, edge);
      renderAcquireRow(depth, anchor, edge);
    }
  }
  // A DoneFanin group's per-reader releases are above; its single shared
  // acquire (pending-count) renders once, after them, at the dst.
  for (unsigned gi : faninHere) {
    if (!octx->rendered.insert(gi).second) continue;
    const SyncGroup &g = dag.groups[gi];
    SmallVector<std::optional<PartitionId>, 4> srcs;
    for (unsigned ei : g.edgeIdxs) srcs.push_back(sp.edges[ei].srcOwner);
    llvm::errs() << treePrefix(depth) << "|- a  S_empty  acquire  pending="
                 << ownerSetStr(anchor, srcs) << "  "
                 << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].dstOwner)
                 << "\n";
  }
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
                             unsigned depth, OptDumpCtx *octx = nullptr);

static void dumpRawSyncRegion(Region &region, SyncPlan &sp,
                              const ResourcePlan &plan, BufferGroup &group,
                              Operation *anchorOp, unsigned depth,
                              OptDumpCtx *octx = nullptr) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end()) part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region) dumpRawSyncBlock(b, sp, plan, group, depth, octx);
  // YIELD-anchored close edges render right before the YIELD row.
  auto yIt = sp.beforeYield.find(&region);
  if (yIt != sp.beforeYield.end())
    printEdgesAt(&yIt->second, sp, anchorOp, depth, octx);
  renderYieldRow(depth, anchorOp, part);
}

static void dumpRawSyncBlock(Block &block, SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
                             unsigned depth, OptDumpCtx *octx) {
  for (Operation &op : block) {
    // Initial writable permit acquire, rendered before its anchor op. In the
    // OPT dump, if the permit reuses an edge that is itself fan-in/out combined,
    // show the combined name so the entry matches the rest of the combine.
    if (&op == sp.initialPermitBeforeOp && !sp.initialPermitName.empty()) {
      StringRef entryName = sp.initialPermitName;
      if (octx && sp.initialPermitEdgeIdx >= 0) {
        SyncGroupKind k =
            octx->dag->groups[octx->dag->edgeToGroup[sp.initialPermitEdgeIdx]]
                .kind;
        if (k == SyncGroupKind::ReadyFanout)
          entryName = "S_full";
        else if (k == SyncGroupKind::DoneFanin)
          entryName = "S_empty";
      }
      llvm::errs() << treePrefix(depth) << "|- a  " << entryName
                   << "  acquire  root\n";
    }
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
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      // Region-op header row (planner-derived partition).
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpRawSyncRegion(forOp.getRegion(), sp, plan, group, &op, depth + 1,
                        octx);
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
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpRawSyncRegion(ifOp.getThenRegion(), sp, plan, group, &op, depth + 2,
                        octx);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpRawSyncRegion(ifOp.getElseRegion(), sp, plan, group, &op,
                          depth + 2, octx);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op)) continue;
    if (!sp.accessOps.contains(&op)) continue;
    // Edges anchored before this access row.
    auto bIt = sp.beforeOp.find(&op);
    if (bIt != sp.beforeOp.end())
      printEdgesAt(&bIt->second, sp, &op, depth, octx);
    // Access row.
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    SmallVector<const AccessTouch *, 4> touches;
    if (event)
      collectTouchesForResource(*event, sp.resource.second, touches);
    for (const AccessTouch *touch : touches) {
      bool reads = hasRead(touch->effect);
      bool writes = hasWrite(touch->effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
    // Acyclic initial-permit terminal release, after its last (top-level) access.
    if (&op == sp.initialPermitReleaseAfterOp && !sp.initialPermitName.empty())
      llvm::errs() << treePrefix(depth) << "|- r  " << sp.initialPermitName
                   << "  release  root\n";
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
// OPT-SYNC-DAG dump (commit 4). Rendered by the shared RAW walk
// (dumpRawSyncBlock) driven by an OptDumpCtx overlay in printEdgesAt — see the
// Contract: the OPT-SYNC-DAG is byte-identical to the RAW-SYNC-DAG except where
// a ReadyFanout (one shared `S_full` release with a `{{..}}` dst-set, plus a
// per-consumer `S_full` acquire) or DoneFanin (per-reader `S_empty` releases
// plus one `pending={{..}}` acquire) combine fires. `ownerSetStr` formats those
// owner sets.
// ---------------------------------------------------------------------------

static std::string ownerSetStr(Operation *anchor,
                               ArrayRef<std::optional<PartitionId>> owners) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{";
  bool first = true;
  for (const auto &o : owners) {
    if (!first) os << ",";
    os << ownerStr(anchor, o);
    first = false;
  }
  os << "}";
  return s;
}

static void dumpOptSyncDag(const OptSyncDag &dag, SyncPlan &sp,
                           const ResourcePlan &plan, BufferGroup &group,
                           triton::FuncOp funcOp) {
  unsigned nInitial = 0, nFanout = 0, nFanin = 0, nSingleton = 0, nLinear = 0;
  for (const SyncGroup &g : dag.groups) {
    switch (g.kind) {
    case SyncGroupKind::InitialEmpty: ++nInitial; break;
    case SyncGroupKind::ReadyFanout: ++nFanout; break;
    case SyncGroupKind::DoneFanin: ++nFanin; break;
    case SyncGroupKind::Singleton: ++nSingleton; break;
    case SyncGroupKind::LinearChain: ++nLinear; break;
    }
  }
  llvm::errs() << "OPT-SYNC-DAG buffer.id=" << dag.resource.first
               << " resourceKey=" << dag.resource.second
               << " groups=" << dag.groups.size()
               << " (initial-empty=" << nInitial
               << " singleton=" << nSingleton
               << " ready-fanout=" << nFanout
               << " done-fanin=" << nFanin
               << " linear-chain=" << nLinear << ")\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  // Render via the shared RAW walk with a fanout/fanin overlay. Per the
  // Contract, the OPT-SYNC-DAG is byte-identical to the RAW-SYNC-DAG except
  // where a ReadyFanout/DoneFanin combine fires (see OptDumpCtx / printEdgesAt).
  OptDumpCtx octx{&dag, {}};
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1, &octx);
}

static SetVector<int> unionPartitionIds(Operation *lhs, Operation *rhs) {
  SetVector<int> ids;
  if (lhs && hasPartition(lhs))
    addPartitionIds(ids, getPartitionIds(lhs));
  if (rhs && hasPartition(rhs))
    addPartitionIds(ids, getPartitionIds(rhs));
  return ids;
}

static SetVector<int> subtractPartitionIds(const SetVector<int> &ids,
                                           const SetVector<int> &excluded) {
  SetVector<int> result;
  for (int id : ids)
    if (!llvm::is_contained(excluded, id))
      result.insert(id);
  return result;
}

static void assignStageIfKnown(OpBuilder &b, Operation *op,
                               StageCluster stageCluster) {
  if (stageCluster)
    setStageCluster(b, op, stageCluster);
}

static bool semaphoreUsesTmem(Value semaphore) {
  auto semaType = dyn_cast<SemaphoreType>(semaphore.getType());
  if (!semaType || semaType.getBaseType().empty())
    return false;
  auto memDescType = dyn_cast<MemDescType>(semaType.getBaseType().front());
  return memDescType &&
         memDescType.getMemorySpace() ==
             TensorMemorySpaceAttr::get(semaphore.getContext());
}

static unsigned semaphoreBaseTypeCount(Value semaphore) {
  auto semaType = dyn_cast<SemaphoreType>(semaphore.getType());
  return semaType ? semaType.getBaseType().size() : 0;
}

struct SemaphoreIfSplitCandidate {
  scf::IfOp ifOp;
  bool branchIsThen = true;
  SemaphoreReleaseOp releaseOp;
  SemaphoreAcquireOp acquireOp;
  unsigned tokenResultIdx = 0;
  bool releaseOnly = false;
};

static SemaphoreReleaseOp findBranchReleaseForSplit(Block *block) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (isa<scf::YieldOp>(op))
      return nullptr;
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(&op))
      return releaseOp;
    if (isa<SemaphoreAcquireOp>(op))
      return nullptr;
    if (op.hasTrait<OpTrait::ConstantLike>() || isSupportedAliasOp(&op))
      continue;
    return nullptr;
  }
  return nullptr;
}

static SemaphoreAcquireOp findBranchTrailingAcquire(Block *block) {
  if (!block || !block->getTerminator())
    return nullptr;
  Operation *lastOp = block->getTerminator()->getPrevNode();
  return dyn_cast_or_null<SemaphoreAcquireOp>(lastOp);
}

static bool branchHasAcquireAfter(SemaphoreReleaseOp releaseOp) {
  if (!releaseOp)
    return false;
  for (Operation *op = releaseOp->getNextNode(); op; op = op->getNextNode()) {
    if (isa<scf::YieldOp>(op))
      return false;
    if (isa<SemaphoreAcquireOp>(op))
      return true;
  }
  return false;
}

static StageCluster inferPrecedingMmaStage(scf::IfOp ifOp) {
  for (Operation *op = ifOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<MMAv5OpInterface>(op))
      return getStageCluster(op);
  return StageCluster{};
}

static void splitSemaphoreIfForLoopScheduler(triton::FuncOp funcOp) {
  SmallVector<SemaphoreIfSplitCandidate, 4> ifOps;
  funcOp.walk([&](scf::IfOp ifOp) {
    if (ifOp.thenBlock()->empty())
      return;
    auto makeReleaseOnlyCandidate = [&](bool branchIsThen)
        -> std::optional<SemaphoreIfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto releaseOp = findBranchReleaseForSplit(block);
      if (!releaseOp || !semaphoreUsesTmem(releaseOp.getSemaphore()) ||
          !branchHasAcquireAfter(releaseOp))
        return std::nullopt;
      return SemaphoreIfSplitCandidate{
          ifOp, branchIsThen, releaseOp, SemaphoreAcquireOp(), 0,
          /*releaseOnly=*/true};
    };
    auto makeCandidate = [&](bool branchIsThen)
        -> std::optional<SemaphoreIfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto releaseOp = findBranchReleaseForSplit(block);
      auto acquireOp = findBranchTrailingAcquire(block);
      if (!releaseOp || !acquireOp)
        return std::nullopt;
      if (semaphoreUsesTmem(releaseOp.getSemaphore()) &&
          semaphoreBaseTypeCount(releaseOp.getSemaphore()) > 1)
        return std::nullopt;
      scf::YieldOp yieldOp =
          branchIsThen ? ifOp.thenYield() : ifOp.elseYield();
      auto pos =
          findValuePosInRange(yieldOp->getOperands(), acquireOp.getToken());
      if (!pos)
        return std::nullopt;
      return SemaphoreIfSplitCandidate{
          ifOp, branchIsThen, releaseOp, acquireOp,
          static_cast<unsigned>(*pos), /*releaseOnly=*/false};
    };

    if (auto candidate = makeCandidate(/*branchIsThen=*/true)) {
      ifOps.push_back(*candidate);
      return;
    }
    if (auto candidate = makeCandidate(/*branchIsThen=*/false)) {
      ifOps.push_back(*candidate);
      return;
    }

    if (auto candidate = makeReleaseOnlyCandidate(/*branchIsThen=*/true)) {
      ifOps.push_back(*candidate);
      return;
    }
    if (auto candidate = makeReleaseOnlyCandidate(/*branchIsThen=*/false)) {
      ifOps.push_back(*candidate);
      return;
    }

    Operation *firstOp = &ifOp.thenBlock()->front();
    auto acquireOp = dyn_cast_or_null<SemaphoreAcquireOp>(firstOp);
    if (acquireOp) {
      Operation *prev = ifOp->getPrevNode();
      if (prev && ifOp.getCondition().getDefiningOp() == prev)
        prev = prev->getPrevNode();
      auto releaseOp = dyn_cast_or_null<SemaphoreReleaseOp>(prev);
      if (!releaseOp)
        return;
      scf::YieldOp yieldOp = ifOp.thenYield();
      auto pos =
          findValuePosInRange(yieldOp->getOperands(), acquireOp.getToken());
      if (!pos)
        return;
      ifOps.push_back(SemaphoreIfSplitCandidate{
          ifOp, /*branchIsThen=*/true, releaseOp, acquireOp,
          static_cast<unsigned>(*pos), /*releaseOnly=*/false});
    }
  });

  for (SemaphoreIfSplitCandidate candidate : ifOps) {
    scf::IfOp ifOp = candidate.ifOp;
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();

    b.setInsertionPoint(ifOp);
    auto exitIf = scf::IfOp::create(
        b, loc, TypeRange{}, ifOp.getCondition(),
        /*withElseRegion=*/!candidate.branchIsThen);
    Block *exitBlock =
        candidate.branchIsThen ? exitIf.thenBlock() : exitIf.elseBlock();
    candidate.releaseOp->moveBefore(exitBlock, exitBlock->begin());
    exitIf->setAttrs(ifOp->getAttrs());
    StageCluster releaseStage = getStageCluster(candidate.releaseOp);
    if (!releaseStage)
      releaseStage = inferPrecedingMmaStage(ifOp);
    assignStageIfKnown(b, candidate.releaseOp, releaseStage);
    assignStageIfKnown(b, exitIf, releaseStage);
    SetVector<int> exitIds;
    if (hasPartition(candidate.releaseOp.getOperation()))
      exitIds = getPartitionIds(candidate.releaseOp.getOperation());
    else if (hasPartition(ifOp))
      exitIds = getPartitionIds(ifOp);
    if (!exitIds.empty())
      setPartition(exitIf, exitIds);
    setPartitionOutputs(exitIf, {});
    if (candidate.releaseOnly)
      continue;

    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(b, loc, TypeRange{b.getType<AsyncTokenType>()},
                                     ifOp.getCondition(),
                                     /*withElseRegion=*/true);
    Block *enterAcquireBlock =
        candidate.branchIsThen ? enterIf.thenBlock() : enterIf.elseBlock();
    candidate.acquireOp->moveBefore(enterAcquireBlock,
                                    enterAcquireBlock->begin());

    ifOp.getResult(candidate.tokenResultIdx)
        .replaceAllUsesWith(enterIf.getResult(0));

    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(
        b, loc,
        candidate.branchIsThen
            ? candidate.acquireOp.getToken()
            : ifOp.thenYield().getOperand(candidate.tokenResultIdx));
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(
        b, loc,
        candidate.branchIsThen
            ? ifOp.elseYield().getOperand(candidate.tokenResultIdx)
            : candidate.acquireOp.getToken());

    b.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(b, loc, b.getType<AsyncTokenType>());
    ifOp.thenYield().setOperand(candidate.tokenResultIdx, poison);
    ifOp.elseYield().setOperand(candidate.tokenResultIdx, poison);

    enterIf->setAttrs(ifOp->getAttrs());
    StageCluster acquireStage = getStageCluster(candidate.acquireOp);
    if (!releaseStage)
      releaseStage = acquireStage;
    assignStageIfKnown(b, enterIf, acquireStage);

    SetVector<int> enterExitIds =
        unionPartitionIds(candidate.releaseOp.getOperation(),
                          candidate.acquireOp.getOperation());
    if (!enterExitIds.empty()) {
      setPartition(exitIf, enterExitIds);
      setPartition(enterIf, enterExitIds);
      setPartitionOutputs(exitIf, {});
      SmallVector<SetVector<int>, 1> enterOutputs{enterExitIds};
      setPartitionOutputs(enterIf, enterOutputs);
    }

    SetVector<int> middleIds;
    if (hasPartition(ifOp))
      middleIds = subtractPartitionIds(getPartitionIds(ifOp), enterExitIds);
    if (middleIds.empty() && ifOp.getNumResults() > 0)
      middleIds = partitionSetForValue(ifOp.getResult(0));
    if (!middleIds.empty()) {
      SetVector<int> ifIds = middleIds;
      SmallVector<SetVector<int>, 4> outputs;
      outputs.reserve(ifOp.getNumResults());
      for (Value result : ifOp.getResults()) {
        SetVector<int> resultIds = partitionSetForValue(result);
        if (resultIds.empty())
          resultIds = middleIds;
        addPartitionIds(ifIds, resultIds);
        outputs.push_back(resultIds);
      }
      setPartition(ifOp, ifIds);
      setPartitionOutputs(ifOp, outputs);
    }
  }
}

static bool isReleasedCreate(SemaphoreCreateOp createOp) {
  auto attr = dyn_cast_or_null<BoolAttr>(createOp->getAttr("is_released"));
  return attr && attr.getValue();
}

static bool isFirstAcquireAfterCreate(SemaphoreCreateOp createOp,
                                      SemaphoreAcquireOp acquireOp) {
  Operation *create = createOp.getOperation();
  Operation *acquire = acquireOp.getOperation();
  if (create->getBlock() != acquire->getBlock() ||
      !create->isBeforeInBlock(acquire))
    return false;
  Value semaphore = createOp.getResult();
  for (Operation *user : semaphore.getUsers()) {
    if (user == acquire || user->getBlock() != create->getBlock())
      continue;
    if (!isa<SemaphoreAcquireOp, SemaphoreReleaseOp>(user))
      continue;
    if (create->isBeforeInBlock(user) && user->isBeforeInBlock(acquire))
      return false;
  }
  return true;
}

static Operation *findEarliestTokenUserInBlock(Value token, Block *block) {
  Operation *earliest = nullptr;
  Operation *defOp = token.getDefiningOp();
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    Operation *ancestor = block ? block->findAncestorOpInBlock(*user) : user;
    if (!ancestor || ancestor == defOp)
      continue;
    if (!earliest || ancestor->isBeforeInBlock(earliest))
      earliest = ancestor;
  }
  return earliest;
}

static bool opBetweenFeedsTarget(Operation *first, Operation *last,
                                 Operation *target) {
  if (!first || !last || !target || first == target)
    return false;
  for (Operation *op = first; op && op != target; op = op->getNextNode()) {
    if (isa<SemaphoreCreateOp, SemaphoreAcquireOp, SemaphoreReleaseOp,
            SemaphoreBufferOp>(op))
      continue;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user == target)
          return true;
      }
    }
  }
  return false;
}

static void hoistInitialEmptyAcquires(triton::FuncOp funcOp) {
  SmallVector<SemaphoreAcquireOp, 8> acquires;
  funcOp.walk([&](SemaphoreAcquireOp acquireOp) {
    auto createOp =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (!createOp || !semaphoreUsesTmem(createOp.getResult()) ||
        !isReleasedCreate(createOp) ||
        !isFirstAcquireAfterCreate(createOp, acquireOp))
      return;
    acquires.push_back(acquireOp);
  });

  for (SemaphoreAcquireOp acquireOp : acquires) {
    auto createOp =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (!createOp || !isFirstAcquireAfterCreate(createOp, acquireOp))
      continue;
    Operation *insertAfter = createOp.getOperation();
    for (Operation *next = insertAfter->getNextNode();
         next && isa<SemaphoreCreateOp>(next); next = next->getNextNode())
      insertAfter = next;
    if (Operation *firstUser =
            findEarliestTokenUserInBlock(acquireOp.getToken(),
                                         acquireOp->getBlock())) {
      if (createOp->isBeforeInBlock(firstUser) &&
          firstUser != acquireOp.getOperation()) {
        Operation *afterCreates = insertAfter->getNextNode();
        if (opBetweenFeedsTarget(afterCreates, firstUser, firstUser)) {
          if (firstUser->getPrevNode() != acquireOp.getOperation())
            acquireOp->moveBefore(firstUser);
        } else if (insertAfter->getNextNode() != acquireOp.getOperation()) {
          acquireOp->moveAfter(insertAfter);
        }
      }
    } else if (insertAfter->getNextNode() != acquireOp.getOperation()) {
      acquireOp->moveAfter(insertAfter);
    }
  }
}

static Value findLatestSemaphoreCarrierInit(scf::ForOp forOp,
                                            ArrayRef<unsigned> slots) {
  if (slots.empty())
    return {};

  Value latestInit = forOp.getInitArgs()[slots.front()];
  Operation *latestAcquire = nullptr;
  for (unsigned slot : slots) {
    auto acquireOp = forOp.getInitArgs()[slot].getDefiningOp<SemaphoreAcquireOp>();
    if (!acquireOp || acquireOp->getBlock() != forOp->getBlock() ||
        !acquireOp->isBeforeInBlock(forOp))
      continue;
    if (!latestAcquire || latestAcquire->isBeforeInBlock(acquireOp)) {
      latestAcquire = acquireOp;
      latestInit = forOp.getInitArgs()[slot];
    }
  }
  return latestInit;
}

static void collectSemaphoreBackingsForToken(Value token,
                                             SetVector<Value> &backings,
                                             DenseSet<Value> &visited) {
  if (!token || !visited.insert(token).second)
    return;

  auto addSemaphoreBacking = [&](Value semaphore) {
    auto createOp = semaphore.getDefiningOp<SemaphoreCreateOp>();
    if (createOp && !createOp.getBuffers().empty())
      backings.insert(createOp.getBuffers().front());
  };

  for (Operation *user : token.getUsers()) {
    if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(user)) {
      addSemaphoreBacking(bufferOp.getSemaphore());
      continue;
    }
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user)) {
      addSemaphoreBacking(releaseOp.getSemaphore());
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned controlOperands = forOp.getNumControlOperands();
      for (OpOperand &operand : forOp->getOpOperands()) {
        if (operand.get() != token ||
            operand.getOperandNumber() < controlOperands)
          continue;
        unsigned slot = operand.getOperandNumber() - controlOperands;
        collectSemaphoreBackingsForToken(forOp.getRegionIterArg(slot),
                                         backings, visited);
      }
    }
  }
}

static std::optional<Value> inferSemaphoreBackingForCarrierSlot(scf::ForOp forOp,
                                                                unsigned slot) {
  SetVector<Value> backings;
  DenseSet<Value> visited;
  collectSemaphoreBackingsForToken(forOp.getRegionIterArg(slot), backings,
                                   visited);
  if (backings.size() != 1)
    return std::nullopt;
  return backings.front();
}

static bool isPoisonAsyncToken(Value value) {
  return value && isa<AsyncTokenType>(value.getType()) &&
         value.getDefiningOp<ub::PoisonOp>();
}

static void poisonDuplicateUnbackedTokenSlots(
    scf::ForOp forOp, ArrayRef<unsigned> asyncTokenSlots,
    const DenseMap<unsigned, Value> &backingBySlot) {
  llvm::MapVector<Value, SmallVector<unsigned, 4>> slotsByInit;
  for (unsigned slot : asyncTokenSlots)
    slotsByInit[forOp.getInitArgs()[slot]].push_back(slot);

  scf::YieldOp yieldOp = getForYieldOp(forOp);
  unsigned controlOperands = forOp.getNumControlOperands();
  Value poison;
  for (auto &it : slotsByInit) {
    ArrayRef<unsigned> slots = it.second;
    if (slots.size() < 2)
      continue;
    bool hasSemaphoreBackedSlot = false;
    for (unsigned slot : slots)
      hasSemaphoreBackedSlot |= backingBySlot.contains(slot);
    if (!hasSemaphoreBackedSlot)
      continue;

    for (unsigned slot : slots) {
      if (backingBySlot.contains(slot) ||
          !isPoisonAsyncToken(yieldOp.getOperand(slot)))
        continue;
      if (!poison) {
        OpBuilder b(forOp);
        b.setInsertionPoint(forOp);
        poison =
            ub::PoisonOp::create(b, forOp.getLoc(), b.getType<AsyncTokenType>());
      }
      forOp->setOperand(controlOperands + slot, poison);
      yieldOp.setOperand(slot, poison);
    }
  }
}

static void coalesceSemaphoreForCarriers(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp, 8> loops;
  funcOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    llvm::MapVector<Value, SmallVector<unsigned, 4>> slotsByBacking;
    DenseMap<unsigned, Value> backingBySlot;
    SmallVector<unsigned, 4> asyncTokenSlots;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
      if (!isa<AsyncTokenType>(init.getType()))
        continue;
      unsigned slot = static_cast<unsigned>(idx);
      asyncTokenSlots.push_back(slot);
      auto acquireOp = init.getDefiningOp<SemaphoreAcquireOp>();
      if (acquireOp) {
        auto createOp =
            acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
        if (createOp && !createOp.getBuffers().empty()) {
          Value backing = createOp.getBuffers().front();
          backingBySlot[slot] = backing;
          slotsByBacking[backing].push_back(slot);
          continue;
        }
      }
      if (std::optional<Value> backing = inferSemaphoreBackingForCarrierSlot(
              forOp, slot)) {
        backingBySlot[slot] = *backing;
        slotsByBacking[*backing].push_back(slot);
      }
    }

    poisonDuplicateUnbackedTokenSlots(forOp, asyncTokenSlots, backingBySlot);

    for (auto &it : slotsByBacking) {
      ArrayRef<unsigned> semaphoreTokenSlots = it.second;
      if (semaphoreTokenSlots.size() < 2)
        continue;

      Value carrierInit =
          findLatestSemaphoreCarrierInit(forOp, semaphoreTokenSlots);
      if (!carrierInit)
        continue;

      unsigned canonicalSlot = semaphoreTokenSlots.front();
      Value canonicalIterArg = forOp.getRegionIterArg(canonicalSlot);
      Value canonicalResult = forOp.getResult(canonicalSlot);
      scf::YieldOp yieldOp = getForYieldOp(forOp);
      Value canonicalYield = yieldOp.getOperand(canonicalSlot);
      unsigned controlOperands = forOp.getNumControlOperands();
      forOp->setOperand(controlOperands + canonicalSlot, carrierInit);
      for (unsigned slot : semaphoreTokenSlots) {
        if (slot == canonicalSlot)
          continue;
        forOp.getRegionIterArg(slot).replaceAllUsesWith(canonicalIterArg);
        forOp.getResult(slot).replaceAllUsesWith(canonicalResult);
        yieldOp.setOperand(slot, canonicalYield);
        forOp->setOperand(controlOperands + slot, carrierInit);
      }
    }
  }
}

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
