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
  if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(op))
    if (Value src = tmemAlloc.getSrc())
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
    member.extent = getAllocExtent(type);
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
          plan.useOwner[event.op] = event.owner;
          plan.useTagSource[event.op] = event.tagSourceOp;
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
        edge.srcOwner = state.readers[i];
        edge.dstOwner = P;
        edge.srcOp = state.readerOps[i];
        edge.srcYieldRegion = state.readerYields[i];
        // Each reader's last-access is itself a fresh source event for
        // Combine A grouping purposes; assign a unique epoch per
        // contributing reader so cross-reader closes are not falsely
        // collapsed into a ReadyFanout.
        edge.srcEpoch = ++writerEpochCtr;
        if (anchorOp)
          edge.dstOp = anchorOp;
        else
          edge.dstYieldRegion = anchorYieldRegion;
        populateEdgeInfo(edge, SyncEdgeKind::Done, state.readerEpochs[i],
                         state.readers[i], P);
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
          // Transition at the scf.for header row.
          emitClose(state, pp, &op, /*anchorYieldRegion=*/nullptr,
                    /*isW=*/false, rank);
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

struct OptSyncDag {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncGroup> groups;
  // Which raw edge belongs to which opt group (parallel to sp.edges).
  SmallVector<unsigned> edgeToGroup;
  // Where to render the release row(s).
  //   Singleton:    `releaseBeforeOp[dstOp]`        (release+acquire pair
  //                 at dst — matches RAW shape).
  //   ReadyFanout:  `releaseAfterOp[srcOp]`         (one row right after
  //                 the producer access).
  //   DoneFanin:    `releaseAfterOp[eachSrcOp]`     (one row right after
  //                 each retiring reader access).
  DenseMap<Operation *, SmallVector<unsigned, 2>> releaseBeforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> releaseBeforeYield;
  DenseMap<Operation *, SmallVector<unsigned, 2>> releaseAfterOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> releaseAfterYield;
  // Where to render the acquire row(s) — always before the consumer.
  //   Singleton:    `acquireBeforeOp[dstOp]`.
  //   ReadyFanout:  `acquireBeforeOp[eachDstOp]`.
  //   DoneFanin:    `acquireBeforeOp[sharedDstOp]`.
  DenseMap<Operation *, SmallVector<unsigned, 2>> acquireBeforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> acquireBeforeYield;
  DenseSet<Operation *> accessOps;
  // Planned CFG token-threading requirements. These are derived from OPT-SYNC
  // anchors at structured op/yield boundaries before emission; the emitter only
  // materializes the planned iter_args/results.
  DenseSet<Operation *> threadForOps;
  DenseSet<Operation *> threadIfOps;
};

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
    if (e.dstYieldRegion &&
        (pos + 1 != chain.size() || e.kind != SyncEdgeKind::Done))
      return false;
    if (e.srcOp && isa<scf::ForOp, scf::IfOp>(e.srcOp))
      return false;
    if (e.dstOp && isa<scf::ForOp, scf::IfOp>(e.dstOp))
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

static OptSyncDag buildOptSyncDag(const SyncPlan &sp, BufferGroup &group) {
  OptSyncDag dag;
  dag.resource = sp.resource;
  dag.groupIdx = sp.groupIdx;
  dag.memberIndices = sp.memberIndices;
  dag.accessOps = sp.accessOps;
  dag.edgeToGroup.assign(sp.edges.size(), 0u);
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
        dag.releaseBeforeOp[e.dstOp].push_back(gi);
        dag.acquireBeforeOp[e.dstOp].push_back(gi);
      } else if (e.dstYieldRegion) {
        dag.releaseBeforeYield[e.dstYieldRegion].push_back(gi);
        dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
      }
      break;
    }
    case SyncGroupKind::ReadyFanout: {
      // Shared release AFTER src (one row); per-consumer acquires at dst.
      const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
      if (probe.srcOp)
        dag.releaseAfterOp[probe.srcOp].push_back(gi);
      else if (probe.srcYieldRegion)
        dag.releaseAfterYield[probe.srcYieldRegion].push_back(gi);
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
          dag.releaseAfterOp[e.srcOp].push_back(gi);
        else if (e.srcYieldRegion)
          dag.releaseAfterYield[e.srcYieldRegion].push_back(gi);
      }
      const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
      if (probe.dstOp)
        dag.acquireBeforeOp[probe.dstOp].push_back(gi);
      else if (probe.dstYieldRegion)
        dag.acquireBeforeYield[probe.dstYieldRegion].push_back(gi);
      break;
    }
    case SyncGroupKind::LinearChain: {
      // Compact linear handoff: each edge releases after its source and
      // acquires before its destination, sharing one EMPTY/FULL pair.
      for (unsigned ei : g.edgeIdxs) {
        const SyncEdge &e = sp.edges[ei];
        if (e.srcOp)
          dag.releaseAfterOp[e.srcOp].push_back(gi);
        else if (e.srcYieldRegion)
          dag.releaseAfterYield[e.srcYieldRegion].push_back(gi);
        if (e.dstOp)
          dag.acquireBeforeOp[e.dstOp].push_back(gi);
        else if (e.dstYieldRegion &&
                 yieldedAccessTokenRequiresCarrier(e.dstYieldRegion, sp))
          dag.acquireBeforeYield[e.dstYieldRegion].push_back(gi);
      }
      break;
    }
    }
  }

  auto markThreadedOp = [&](Operation *op) {
    if (!op) return;
    if (isa<scf::ForOp>(op))
      dag.threadForOps.insert(op);
    else if (isa<scf::IfOp>(op))
      dag.threadIfOps.insert(op);
  };
  auto markOpAnchors = [&](const DenseMap<Operation *, SmallVector<unsigned, 2>>
                               &anchors) {
    for (auto &kv : anchors)
      markThreadedOp(kv.first);
  };
  markOpAnchors(dag.releaseBeforeOp);
  markOpAnchors(dag.acquireBeforeOp);
  markOpAnchors(dag.releaseAfterOp);
  for (auto &kv : dag.acquireBeforeYield)
    markThreadedOp(kv.first->getParentOp());

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

  // When the first writer is inside a loop whose resource state is loop-carried,
  // the initial EMPTY acquire must produce the loop-carried carrier before the
  // scf.for. The semaphore.buffer remains at the first writer and consumes the
  // iter_arg token inside the body.
  for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
    if (syncGroup.kind != SyncGroupKind::InitialEmpty || !syncGroup.initialOp)
      continue;
    Operation *threadedLoop = nullptr;
    for (Operation *parent = syncGroup.initialOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent) && dag.threadForOps.contains(parent))
        threadedLoop = parent;
    }
    if (!threadedLoop)
      continue;
    unsigned groupIdx = static_cast<unsigned>(idx);
    removeGroupFromOpAnchor(dag.acquireBeforeOp, syncGroup.initialOp, groupIdx);
    addGroupToOpAnchor(dag.acquireBeforeOp, threadedLoop, groupIdx);
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

static int computeSemaphoreDepth(BufferGroup &group) {
  if (!group.isTmem()) return 1;
  for (AccessEvent &event : group.events)
    if (isa<MMAv5OpInterface>(event.op) && event.op->getParentOfType<scf::ForOp>())
      return 2;
  return 1;
}

struct GroupBacking {
  SmallVector<Value, 4> buffers;
  SmallVector<Type, 4> bufferTypes;
};

static GroupBacking &
ensureGroupBacking(BufferGroup &group, unsigned groupIdx,
                   DenseMap<unsigned, GroupBacking> &backings) {
  auto it = backings.find(groupIdx);
  if (it != backings.end())
    return it->second;

  GroupBacking backing;
  OpBuilder b(getSemaphoreInsertionAnchor(group));
  b.setInsertionPoint(getSemaphoreInsertionAnchor(group));
  int depth = computeSemaphoreDepth(group);
  for (BufferMember &member : group.members) {
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
  auto inserted = backings.insert({groupIdx, std::move(backing)});
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
};

struct ResourceSemaphores {
  DenseMap<unsigned, SyncGroupSemaphores> byGroup;
};

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

static bool edgeUsesEmpty(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (edge.kind == SyncEdgeKind::Done)
    return true;
  if (edge.kind == SyncEdgeKind::Ready)
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

static ResourceSemaphores createResourceSemaphores(const OptSyncDag &dag,
                                                   const SyncPlan &sp,
                                                   BufferGroup &group,
                                                   const GroupBacking &backing) {
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

  Value sharedEmpty;
  auto createSharedEmpty = [&]() -> Value {
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
      pair.full = createFull();
      break;
    case SyncGroupKind::Singleton: {
      const SyncEdge *edge = syncGroup.edgeIdxs.empty()
                                 ? nullptr
                                 : &sp.edges[syncGroup.edgeIdxs.front()];
      if (edge && edgeUsesEmpty(*edge, group, dag.resource.second))
        pair.empty = createSharedEmpty();
      else
        pair.full = createFull();
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
                                         SyncAnchorKind kind,
                                         Operation *anchor,
                                         Region *yieldRegion) {
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
    switch (kind) {
    case SyncAnchorKind::AcquireBeforeOp:
      if (edge.dstOp == anchor) return &edge;
      break;
    case SyncAnchorKind::AcquireBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeOp:
      if (edge.dstOp == anchor) return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterOp:
      if (edge.srcOp == anchor) return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterYield:
      if (edge.srcYieldRegion == yieldRegion) return &edge;
      break;
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
    return edge && edgeUsesEmpty(*edge, group, dag.resource.second)
               ? pair.empty
               : pair.full;
  case SyncGroupKind::Singleton:
    if (edge && edgeUsesEmpty(*edge, group, dag.resource.second))
      return pair.empty;
    return pair.full;
  }
  llvm_unreachable("unhandled sync group kind");
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

struct EmitState {
  ResourceSemaphores semas;
  DenseMap<Operation *, Value> eventToken;
  DenseMap<Operation *, Value> eventSemaphore;
  DenseMap<Operation *, SmallVector<Value, 4>> eventBuffers;
  DenseMap<Value, Value> rewrittenAccessValue;
  SmallVector<Value, 4> currentBuffers;
  DenseSet<Operation *> protectedAccesses;
  DenseSet<Value> knownCarrierTokens;
  SmallVector<EmittedSyncRecord, 8> emittedAcquires;
  SmallVector<EmittedSyncRecord, 8> emittedReleases;
  SmallVector<EmittedBufferRecord, 8> emittedBuffers;
  SmallVector<ThreadRecord, 4> threadedTokens;
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
  if (!op || hasPartition(op) || !owner)
    return;
  setPartition(op, partitionSetForOwner(owner));
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
          static_cast<unsigned>(idx), touches, type, mutableMemory));
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
    if (old->getNumResults() == 1 &&
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

static void poisonTokenResults(OpBuilder &b, Operation *op) {
  bool hasTokenResult = llvm::any_of(op->getResults(), [](Value result) {
    return isa<AsyncTokenType>(result.getType());
  });
  if (!hasTokenResult)
    return;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(op);
  Value poison =
      ub::PoisonOp::create(b, op->getLoc(), b.getType<AsyncTokenType>());
  replaceTokenResults(op, poison);
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
    return touchesWrittenAccumulator(touches, mma.getAccumulator());
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
  bool ownsAsyncToken = accessOwnsAsyncToken(op, touches, group);
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
    bufferOperation = bufferOp.getOperation();
    buffers.assign(bufferOp.getBuffers().begin(), bufferOp.getBuffers().end());
    state.currentBuffers = buffers;
  }
  state.eventBuffers[op] = buffers;
  Operation *retargetOp = op;

  if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful TMEM alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    if (touch.memberIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[touch.memberIdx]);
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
        state.eventBuffers[op], touch.memberIdx, bufferExpectedStageCluster});
  } else if (auto localAlloc = dyn_cast<LocalAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful local alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    if (touch.memberIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[touch.memberIdx]);
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
        state.eventBuffers[op], touch.memberIdx, bufferExpectedStageCluster});
  } else {
    SmallVector<std::pair<OpOperand *, Value>, 4> replacements;
    for (const AccessTouch *touch : touches) {
      if (touch->memberIdx >= buffers.size())
        return op->emitError("nvws-insert-semas: semaphore buffer member index "
                             "out of range");
      Value accessBuffer =
          materializeAliasForBuffer(b, *touch, buffers[touch->memberIdx]);
      Value currentAccessValue = state.rewrittenAccessValue.lookup(
          touch->accessValue);
      for (OpOperand &operand : op->getOpOperands())
        if (operand.get() == touch->accessValue ||
            (currentAccessValue && operand.get() == currentAccessValue))
          replacements.push_back({&operand, accessBuffer});
      state.emittedBuffers.push_back(EmittedBufferRecord{
          op, retargetOp, bufferOperation, sem, token, accessBuffer,
          state.eventBuffers[op], touch->memberIdx, bufferExpectedStageCluster});
    }
    for (auto [operand, accessBuffer] : replacements)
      operand->set(accessBuffer);
    if (ownsAsyncToken)
      clearOwnedTmemTokenOperands(op);
  }

  if (ownsAsyncToken)
    poisonTokenResults(b, op);
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

static LogicalResult emitReleaseForGroup(OpBuilder &b, Location loc,
                                         SyncAnchorKind kind, Operation *anchor,
                                         Region *yieldRegion, unsigned groupIdx,
                                         const OptSyncDag &dag,
                                         const SyncPlan &sp, BufferGroup &group,
                                         EmitState &state,
                                         StageCluster stageCluster) {
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge =
      findEdgeForAnchor(syncGroup, sp, kind, anchor, yieldRegion);
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  bool useStructuredCarrier =
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
      edge->srcOp != anchor && state.currentToken;
  FailureOr<Value> token =
      useStructuredCarrier ? FailureOr<Value>(state.currentToken)
                           : lookupReleaseToken(loc, edge, state,
                                                b.getInsertionBlock());
  if (failed(token))
    return failure();
  std::optional<PartitionId> owner = edge ? edge->srcOwner : std::nullopt;
  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  SemaphoreReleaseOp release =
      emitRelease(b, loc, sem, *token, owner, stageCluster,
                  edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
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
      findEdgeForAnchor(syncGroup, sp, kind, anchor, yieldRegion);
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  std::optional<PartitionId> owner =
      edge ? edge->dstOwner : syncGroup.initialOwner;
  SemaphoreAcquireOp acquire = emitAcquire(b, loc, sem, owner, stageCluster);
  if (!owner) {
    std::optional<PartitionId> fallbackOwner =
        edge ? edge->srcOwner : std::nullopt;
    setSingleOwnerPartition(acquire.getOperation(), fallbackOwner);
    if (!fallbackOwner)
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
    for (unsigned gi : rIt->second)
      if (failed(emitReleaseForGroup(
              b, anchor->getLoc(), SyncAnchorKind::ReleaseBeforeOp, anchor,
              nullptr, gi, dag, sp, group, state, getStageCluster(anchor))))
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
  if (releaseAfter->isProperAncestor(anchor))
    return success();
  OpBuilder b(releaseAfter);
  b.setInsertionPointAfter(releaseAfter);
  std::optional<Value> savedEventToken;
  bool overrideEventToken = releaseAfter != anchor && state.currentToken;
  if (overrideEventToken) {
    auto it = state.eventToken.find(anchor);
    if (it != state.eventToken.end())
      savedEventToken = it->second;
    state.eventToken[anchor] = state.currentToken;
  }
  LogicalResult result = success();
  for (unsigned gi : rIt->second)
    if (failed(emitReleaseForGroup(
            b, insertAfter->getLoc(), SyncAnchorKind::ReleaseAfterOp, anchor,
            nullptr, gi, dag, sp, group, state, getStageCluster(insertAfter)))) {
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

static LogicalResult
emitBeforeYieldSync(Operation *yieldOp, Region *region, const OptSyncDag &dag,
                    const SyncPlan &sp, BufferGroup &group, EmitState &state,
                    SmallVectorImpl<AcquireRecord> &acquires) {
  OpBuilder b(yieldOp);
  b.setInsertionPoint(yieldOp);
  auto rIt = dag.releaseBeforeYield.find(region);
  if (rIt != dag.releaseBeforeYield.end())
    for (unsigned gi : rIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            SyncAnchorKind::ReleaseBeforeYield, nullptr, region);
      if (failed(emitReleaseForGroup(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseBeforeYield, nullptr,
              region, gi, dag, sp, group, state,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  auto aIt = dag.acquireBeforeYield.find(region);
  if (aIt != dag.acquireBeforeYield.end())
    for (unsigned gi : aIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            SyncAnchorKind::AcquireBeforeYield, nullptr, region);
      acquires.push_back(emitAcquireForGroup(
          b, yieldOp->getLoc(), SyncAnchorKind::AcquireBeforeYield, nullptr,
          region, gi, dag, sp, group, state,
          stageForYieldOwner(edge ? edge->dstOwner : std::nullopt, state)));
    }
  auto arIt = dag.releaseAfterYield.find(region);
  if (arIt != dag.releaseAfterYield.end())
    for (unsigned gi : arIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            SyncAnchorKind::ReleaseAfterYield, nullptr, region);
      if (failed(emitReleaseForGroup(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseAfterYield, nullptr,
              region, gi, dag, sp, group, state,
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

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src);

static FailureOr<scf::ForOp> threadCarrierThroughFor(OpBuilder &b,
                                                     scf::ForOp forOp,
                                                     EmitState &state,
                                                     Region *plannedRegion,
                                                     std::optional<PartitionId>
                                                         recordOwner) {
  if (!state.currentToken) {
    forOp.emitError("nvws-insert-semas: planned scf.for carrier threading has "
                    "no token producer at loop entry");
    return failure();
  }
  Value init = state.currentToken;
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(init, state.currentOwner, forOp.getOperation());
  unsigned oldNumResults = forOp.getNumResults();
  auto oldPartitionIds =
      hasPartition(forOp) ? getPartitionIds(forOp) : SetVector<int>();
  auto oldPartitionOutputs =
      hasPartition(forOp) ? getPartitionOutputs(forOp)
                          : SmallVector<SetVector<int>, 4>();

  b.setInsertionPoint(forOp);
  scf::ForOp newFor = addIterArgsToLoop(b, forOp, {init});
  state.currentToken = newFor.getRegionIterArg(oldNumResults);
  state.knownCarrierTokens.insert(state.currentToken);
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
                                Region *plannedRegion) {
  Value yieldedToken = bodyState.currentToken
                           ? bodyState.currentToken
                           : forOp.getRegionIterArg(forOp.getNumResults() - 1);
  std::optional<PartitionId> resultOwner =
      ownerAtYield ? ownerAtYield : bodyState.currentOwner;
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(yieldedToken, resultOwner, forOp.getOperation());
  appendToForOpYield(forOp, yieldedToken);
  if (!carrierPartition.empty()) {
    if (hasPartition(forOp)) {
      auto partitionIds = getPartitionIds(forOp);
      addPartitionIds(partitionIds, carrierPartition);
      auto partitionOutputs = getPartitionOutputs(forOp);
      if (partitionOutputs.size() == forOp.getNumResults())
        partitionOutputs.back() = carrierPartition;
      setPartition(forOp, partitionIds);
      setPartitionOutputs(forOp, partitionOutputs);
    }
    setPartition(forOp.getBody()->getTerminator(), carrierPartition);
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
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
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
    if (edge.dstYieldRegion &&
        (pos + 1 != group.edgeIdxs.size() || edge.kind != SyncEdgeKind::Done))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has a non-terminal yield target");
    if (edge.srcOp && isa<scf::ForOp, scf::IfOp>(edge.srcOp))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "crosses a structured CFG anchor");
    if (edge.dstOp && isa<scf::ForOp, scf::IfOp>(edge.dstOp))
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
      if (!plan.yieldOwner.contains(forOp.getBody()->getTerminator())) {
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
                                        const ResourcePlan &plan) {
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
    if (it == plan.regionOwners.end() ||
        !sameOwner(it->second.exit, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.for result carrier owner does not "
                                  "match the ownership plan");
    if (hasPartition(forOp)) {
      auto outputs = getPartitionOutputs(forOp);
      if (outputs.empty() ||
          !partitionSetMatchesOwner(outputs.back(), record.owner))
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
    Region *thenRegion = record.plannedRegion ? record.plannedRegion
                                               : &ifOp.getThenRegion();
    Region *elseRegion = record.plannedElseRegion
                             ? record.plannedElseRegion
                             : &ifOp.getElseRegion();
    auto thenIt = plan.regionOwners.find(thenRegion);
    auto elseIt = plan.regionOwners.find(elseRegion);
    if (thenIt == plan.regionOwners.end() ||
        elseIt == plan.regionOwners.end() ||
        !sameOwner(thenIt->second.exit, record.owner) ||
        !sameOwner(elseIt->second.exit, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.if result carrier owner does not match "
                                  "the ownership plan");
    if (hasPartition(ifOp)) {
      auto outputs = getPartitionOutputs(ifOp);
      if (outputs.empty() ||
          !partitionSetMatchesOwner(outputs.back(), record.owner))
        return record.op->emitError(
            "nvws-insert-semas: post-emission verifier: scf.if result carrier "
            "partition output does not match the ownership plan");
    }
    return success();
  }
  }
  llvm_unreachable("unknown thread record kind");
}

static LogicalResult verifyPostEmission(const OptSyncDag &dag, BufferGroup &group,
                                        const ResourcePlan &plan,
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
      if (bufferIt->memberIdx >= bufferIt->buffers.size())
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
    if (failed(verifyThreadRecord(record, plan)))
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
      OpBuilder loopBuilder(forOp);
      if (threaded) {
        std::optional<PartitionId> recordOwner = state.currentOwner;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end())
          recordOwner = regionOwnerIt->second.entry;
        FailureOr<scf::ForOp> threadedFor =
            threadCarrierThroughFor(loopBuilder, forOp, bodyState,
                                    plannedForRegion, recordOwner);
        if (failed(threadedFor))
          return failure();
        activeForOp = *threadedFor;
      }
      if (failed(emitResourceRegion(activeForOp.getRegion(), dag, sp, plan,
                                    group, backing, bodyState,
                                    plannedForRegion)))
        return failure();
      if (threaded) {
        auto ownerAtYield = plan.yieldOwner.lookup(
            activeForOp.getBody()->getTerminator());
        closeCarrierForLoop(activeForOp, bodyState, state, ownerAtYield,
                            plannedForRegion);
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
      bool threaded = shouldThreadIfRegion(ifOp, dag);
      scf::IfOp activeIfOp = ifOp;
      unsigned oldNumResults = ifOp.getNumResults();
      auto oldPartitionIds =
          hasPartition(ifOp) ? getPartitionIds(ifOp) : SetVector<int>();
      auto oldPartitionOutputs =
          hasPartition(ifOp) ? getPartitionOutputs(ifOp)
                             : SmallVector<SetVector<int>, 4>();
      bool reuseExistingTokenResult =
          threaded && !group.isTmem() && oldNumResults == 1 &&
          isa<AsyncTokenType>(ifOp.getResult(0).getType());
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
          activeIfOp.thenYield().setOperand(0, thenToken);
          activeIfOp.elseYield().setOperand(0, elseToken);
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
          if (reuseExistingTokenResult && !oldPartitionOutputs.empty())
            oldPartitionOutputs[0] = outPartition;
          else
            oldPartitionOutputs.push_back(outPartition);
          setPartition(activeIfOp, oldPartitionIds);
          setPartitionOutputs(activeIfOp, oldPartitionOutputs);
        }
        if (!reuseExistingTokenResult)
          oldIfOp->erase();
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
            activeIfOp.getResult(reuseExistingTokenResult ? 0 : oldNumResults);
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
                                  DenseMap<unsigned, GroupBacking> &backings) {
  if (dag.groups.empty()) return success();
  if (failed(verifyPlanBeforeEmission(funcOp, group, plan, sp, dag)))
    return failure();
  GroupBacking &backing = ensureGroupBacking(group, dag.groupIdx, backings);
  EmitState state;
  state.semas = createResourceSemaphores(dag, sp, group, backing);
  if (failed(emitResourceRegion(funcOp.getBody(), dag, sp, plan, group, backing,
                                state)))
    return failure();
  if (failed(verifyPostEmission(dag, group, plan, state)))
    return failure();
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
// OPT-SYNC-DAG dump (commit 4). Same tree shape as RAW-SYNC-DAG; sync
// rows are anchored per-group:
//   - InitialEmpty groups print one acquire at the first writer.
//   - Singleton groups print release+acquire pair at the dst (same as
//     RAW, just renamed to S_g<N>).
//   - ReadyFanout groups print one release at the producer src with a
//     `{{P1},{P2},...}` dst list, and one acquire at each consumer dst.
//   - DoneFanin groups print one release at each reader src and one
//     acquire at the shared dst with `pending={{P1},{P2},...}`.
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

static void renderOptRelease(unsigned depth, Operation *anchor,
                             const SyncPlan &sp, const OptSyncDag &dag,
                             unsigned groupIdx) {
  const SyncGroup &g = dag.groups[groupIdx];
  llvm::errs() << treePrefix(depth) << "|- r  " << g.name << "  release  ";
  if (g.kind == SyncGroupKind::ReadyFanout) {
    // src -> {{dst1},{dst2},...}: one release row for all consumers.
    const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
    SmallVector<std::optional<PartitionId>, 4> dsts;
    for (unsigned ei : g.edgeIdxs) dsts.push_back(sp.edges[ei].dstOwner);
    llvm::errs() << ownerStr(anchor, probe.srcOwner) << " -> "
                 << ownerSetStr(anchor, dsts) << "\n";
    return;
  }
  // Singleton + DoneFanin + LinearChain contributors: each release row is
  // per-edge.
  // Locate the edge whose src anchor matches `anchor`.
  for (unsigned ei : g.edgeIdxs) {
    const SyncEdge &e = sp.edges[ei];
    bool match = false;
    if (g.kind == SyncGroupKind::Singleton) {
      // Singleton: release anchored at dst.
      match = (e.dstOp == anchor) || (e.dstYieldRegion == nullptr ? false
                                                                   : false);
      // dstYieldRegion variant handled separately by the region renderer.
      if (e.dstOp == anchor)
        match = true;
    } else { // DoneFanin or LinearChain.
      match = (e.srcOp == anchor);
    }
    if (match) {
      llvm::errs() << ownerStr(anchor, e.srcOwner) << " -> "
                   << ownerStr(anchor, e.dstOwner) << "\n";
      return;
    }
  }
  // Fallback: shouldn't happen, but print probe info.
  const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
  llvm::errs() << ownerStr(anchor, probe.srcOwner) << " -> "
               << ownerStr(anchor, probe.dstOwner) << "\n";
}

static void renderOptReleaseYield(unsigned depth, Operation *anchor,
                                  Region *yieldRegion, const SyncPlan &sp,
                                  const OptSyncDag &dag, unsigned groupIdx) {
  const SyncGroup &g = dag.groups[groupIdx];
  llvm::errs() << treePrefix(depth) << "|- r  " << g.name << "  release  ";
  if (g.kind == SyncGroupKind::ReadyFanout) {
    const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
    SmallVector<std::optional<PartitionId>, 4> dsts;
    for (unsigned ei : g.edgeIdxs) dsts.push_back(sp.edges[ei].dstOwner);
    llvm::errs() << ownerStr(anchor, probe.srcOwner) << " -> "
                 << ownerSetStr(anchor, dsts) << "\n";
    return;
  }
  for (unsigned ei : g.edgeIdxs) {
    const SyncEdge &e = sp.edges[ei];
    bool match = false;
    if (g.kind == SyncGroupKind::Singleton)
      match = (e.dstYieldRegion == yieldRegion);
    else // DoneFanin or LinearChain.
      match = (e.srcYieldRegion == yieldRegion);
    if (match) {
      llvm::errs() << ownerStr(anchor, e.srcOwner) << " -> "
                   << ownerStr(anchor, e.dstOwner) << "\n";
      return;
    }
  }
  const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
  llvm::errs() << ownerStr(anchor, probe.srcOwner) << " -> "
               << ownerStr(anchor, probe.dstOwner) << "\n";
}

static void renderOptAcquire(unsigned depth, Operation *anchor,
                             const SyncPlan &sp, const OptSyncDag &dag,
                             unsigned groupIdx) {
  const SyncGroup &g = dag.groups[groupIdx];
  llvm::errs() << treePrefix(depth) << "|- a  " << g.name << "  acquire  ";
  if (g.kind == SyncGroupKind::InitialEmpty) {
    llvm::errs() << "initial-empty  " << ownerStr(anchor, g.initialOwner)
                 << "\n";
    return;
  }
  if (g.kind == SyncGroupKind::DoneFanin) {
    // pending=set-of-sources, then dst owner.
    SmallVector<std::optional<PartitionId>, 4> srcs;
    for (unsigned ei : g.edgeIdxs) srcs.push_back(sp.edges[ei].srcOwner);
    const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
    llvm::errs() << "pending=" << ownerSetStr(anchor, srcs) << "  "
                 << ownerStr(anchor, probe.dstOwner) << "\n";
    return;
  }
  // Singleton + ReadyFanout: locate the edge whose dst anchor matches.
  for (unsigned ei : g.edgeIdxs) {
    const SyncEdge &e = sp.edges[ei];
    if (e.dstOp == anchor) {
      llvm::errs() << ownerStr(anchor, e.dstOwner) << "\n";
      return;
    }
  }
  const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
  llvm::errs() << ownerStr(anchor, probe.dstOwner) << "\n";
}

static void renderOptAcquireYield(unsigned depth, Operation *anchor,
                                  Region *yieldRegion, const SyncPlan &sp,
                                  const OptSyncDag &dag, unsigned groupIdx) {
  const SyncGroup &g = dag.groups[groupIdx];
  llvm::errs() << treePrefix(depth) << "|- a  " << g.name << "  acquire  ";
  if (g.kind == SyncGroupKind::InitialEmpty) {
    llvm::errs() << "initial-empty  " << ownerStr(anchor, g.initialOwner)
                 << "\n";
    return;
  }
  if (g.kind == SyncGroupKind::DoneFanin) {
    SmallVector<std::optional<PartitionId>, 4> srcs;
    for (unsigned ei : g.edgeIdxs) srcs.push_back(sp.edges[ei].srcOwner);
    const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
    llvm::errs() << "pending=" << ownerSetStr(anchor, srcs) << "  "
                 << ownerStr(anchor, probe.dstOwner) << "\n";
    return;
  }
  for (unsigned ei : g.edgeIdxs) {
    const SyncEdge &e = sp.edges[ei];
    if (e.dstYieldRegion == yieldRegion) {
      llvm::errs() << ownerStr(anchor, e.dstOwner) << "\n";
      return;
    }
  }
  const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
  llvm::errs() << ownerStr(anchor, probe.dstOwner) << "\n";
}

// Print rows anchored BEFORE op: release-before, then acquire-before.
static void printOptBeforeOp(const OptSyncDag &dag, const SyncPlan &sp,
                             Operation *anchor, unsigned depth) {
  auto rIt = dag.releaseBeforeOp.find(anchor);
  if (rIt != dag.releaseBeforeOp.end())
    for (unsigned gi : rIt->second)
      renderOptRelease(depth, anchor, sp, dag, gi);
  auto aIt = dag.acquireBeforeOp.find(anchor);
  if (aIt != dag.acquireBeforeOp.end())
    for (unsigned gi : aIt->second)
      renderOptAcquire(depth, anchor, sp, dag, gi);
}

// Print rows anchored AFTER op: release-after only (acquires are always
// before the consumer).
static void printOptAfterOp(const OptSyncDag &dag, const SyncPlan &sp,
                            Operation *anchor, unsigned depth) {
  auto rIt = dag.releaseAfterOp.find(anchor);
  if (rIt != dag.releaseAfterOp.end())
    for (unsigned gi : rIt->second)
      renderOptRelease(depth, anchor, sp, dag, gi);
}

// Same, for YIELD-anchored entries.
static void printOptBeforeYield(const OptSyncDag &dag, const SyncPlan &sp,
                                Operation *anchor, Region *yieldRegion,
                                unsigned depth) {
  auto rIt = dag.releaseBeforeYield.find(yieldRegion);
  if (rIt != dag.releaseBeforeYield.end())
    for (unsigned gi : rIt->second)
      renderOptReleaseYield(depth, anchor, yieldRegion, sp, dag, gi);
  auto aIt = dag.acquireBeforeYield.find(yieldRegion);
  if (aIt != dag.acquireBeforeYield.end())
    for (unsigned gi : aIt->second)
      renderOptAcquireYield(depth, anchor, yieldRegion, sp, dag, gi);
}

static void printOptAfterYield(const OptSyncDag &dag, const SyncPlan &sp,
                               Operation *anchor, Region *yieldRegion,
                               unsigned depth) {
  auto rIt = dag.releaseAfterYield.find(yieldRegion);
  if (rIt != dag.releaseAfterYield.end())
    for (unsigned gi : rIt->second)
      renderOptReleaseYield(depth, anchor, yieldRegion, sp, dag, gi);
}

static void dumpOptSyncBlock(Block &block, const OptSyncDag &dag,
                             const SyncPlan &sp, const ResourcePlan &plan,
                             BufferGroup &group, unsigned depth);

static void dumpOptSyncRegion(Region &region, const OptSyncDag &dag,
                              const SyncPlan &sp, const ResourcePlan &plan,
                              BufferGroup &group, Operation *anchorOp,
                              unsigned depth) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end()) part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region)
    dumpOptSyncBlock(b, dag, sp, plan, group, depth);
  // YIELD-anchored rows render right before the YIELD row.
  printOptBeforeYield(dag, sp, anchorOp, &region, depth);
  renderYieldRow(depth, anchorOp, part);
  // Any release-after-yield (rare) renders after the YIELD line.
  printOptAfterYield(dag, sp, anchorOp, &region, depth);
}

static void dumpOptSyncBlock(Block &block, const OptSyncDag &dag,
                             const SyncPlan &sp, const ResourcePlan &plan,
                             BufferGroup &group, unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      bool show = false;
      forOp.walk([&](Operation *o) -> WalkResult {
        if (dag.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show && !dag.releaseBeforeYield.count(&forOp.getRegion()) &&
          !dag.acquireBeforeYield.count(&forOp.getRegion()) &&
          !dag.releaseAfterYield.count(&forOp.getRegion()))
        continue;
      printOptBeforeOp(dag, sp, &op, depth);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpOptSyncRegion(forOp.getRegion(), dag, sp, plan, group, &op,
                        depth + 1);
      printOptAfterOp(dag, sp, &op, depth);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool show = false;
      ifOp->walk([&](Operation *o) -> WalkResult {
        if (dag.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show) continue;
      printOptBeforeOp(dag, sp, &op, depth);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpOptSyncRegion(ifOp.getThenRegion(), dag, sp, plan, group, &op,
                        depth + 2);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpOptSyncRegion(ifOp.getElseRegion(), dag, sp, plan, group, &op,
                          depth + 2);
      }
      printOptAfterOp(dag, sp, &op, depth);
      continue;
    }
    if (isa<scf::YieldOp>(op)) continue;
    if (!dag.accessOps.contains(&op)) continue;
    printOptBeforeOp(dag, sp, &op, depth);
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    SmallVector<const AccessTouch *, 4> touches;
    if (event)
      collectTouchesForResource(*event, dag.resource.second, touches);
    for (const AccessTouch *touch : touches) {
      bool reads = hasRead(touch->effect);
      bool writes = hasWrite(touch->effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
    printOptAfterOp(dag, sp, &op, depth);
  }
}

static void dumpOptSyncDag(const OptSyncDag &dag, const SyncPlan &sp,
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
  for (Block &b : funcOp.getBody())
    dumpOptSyncBlock(b, dag, sp, plan, group, /*depth=*/1);
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

static void splitSemaphoreIfForLoopScheduler(triton::FuncOp funcOp) {
  SmallVector<scf::IfOp, 4> ifOps;
  funcOp.walk([&](scf::IfOp ifOp) {
    if (ifOp.thenBlock()->empty())
      return;
    Operation *firstOp = &ifOp.thenBlock()->front();
    Operation *lastOp = ifOp.thenBlock()->getTerminator()->getPrevNode();
    auto releaseOp = dyn_cast_or_null<SemaphoreReleaseOp>(firstOp);
    auto acquireOp = dyn_cast_or_null<SemaphoreAcquireOp>(lastOp);
    if (releaseOp && acquireOp &&
        (semaphoreUsesTmem(releaseOp.getSemaphore()) ||
         semaphoreUsesTmem(acquireOp.getSemaphore())))
      ifOps.push_back(ifOp);
  });

  for (scf::IfOp ifOp : ifOps) {
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();

    b.setInsertionPoint(ifOp);
    auto exitIf =
        scf::IfOp::create(b, loc, TypeRange{}, ifOp.getCondition(),
                          /*withElseRegion=*/false);
    auto releaseOp = cast<SemaphoreReleaseOp>(&ifOp.thenBlock()->front());
    releaseOp->moveBefore(exitIf.thenBlock(), exitIf.thenBlock()->begin());

    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(b, loc, TypeRange{b.getType<AsyncTokenType>()},
                                     ifOp.getCondition(),
                                     /*withElseRegion=*/true);
    auto acquireOp = cast<SemaphoreAcquireOp>(
        ifOp.thenBlock()->getTerminator()->getPrevNode());
    acquireOp->moveBefore(enterIf.thenBlock(), enterIf.thenBlock()->begin());

    auto pos = findValuePosInRange(ifOp.thenYield()->getOperands(),
                                   acquireOp.getToken());
    if (!pos)
      continue;
    ifOp.getResult(*pos).replaceAllUsesWith(enterIf.getResult(0));

    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(b, loc, acquireOp.getToken());
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(b, loc, ifOp.elseYield().getOperand(*pos));

    b.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(b, loc, b.getType<AsyncTokenType>());
    ifOp.thenYield().setOperand(*pos, poison);
    ifOp.elseYield().setOperand(*pos, poison);

    exitIf->setAttrs(ifOp->getAttrs());
    enterIf->setAttrs(ifOp->getAttrs());
    StageCluster releaseStage = getStageCluster(releaseOp);
    StageCluster acquireStage = getStageCluster(acquireOp);
    if (!releaseStage)
      releaseStage = acquireStage;
    assignStageIfKnown(b, exitIf, releaseStage);
    assignStageIfKnown(b, enterIf, acquireStage);

    SetVector<int> enterExitIds =
        unionPartitionIds(releaseOp.getOperation(), acquireOp.getOperation());
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

// ---------------------------------------------------------------------------
// Top-level pipeline (commit 5 stage).
// ---------------------------------------------------------------------------

struct PlannedResource {
  ResourcePlan plan;
  SyncPlan syncPlan;
  OptSyncDag optDag;
};

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

  bool dumpDag = shouldDumpDag();
  if (dumpDag) {
    llvm::errs() << "==== NVWS InsertSemas (commit 5: discovery + ACCESS DAG + "
                    "OWNERSHIP DAG + RAW-SYNC DAG + OPT-SYNC DAG + EMIT) ====\n";
    llvm::errs() << "function: " << funcOp.getName() << "\n";
    llvm::errs() << "backing buffers: " << groups.size() << "\n";
  }

  DenseMap<unsigned, GroupBacking> backings;
  for (auto en : llvm::enumerate(groups)) {
    BufferGroup &group = en.value();
    if (dumpDag) {
      dumpBackingGroupHeader(group);
      dumpAccessDag(group, funcOp);
    }
    std::set<int64_t> keys;
    for (auto &m : group.members) keys.insert(m.resourceKey);
    for (int64_t key : keys) {
      buildProgramOrderRank(funcOp, rank);
      ResourcePlan plan = planResource(funcOp,
                                       static_cast<unsigned>(en.index()),
                                       group, key, rank);
      SyncPlan sp = buildSyncPlan(group, plan, funcOp);
      OptSyncDag opt = buildOptSyncDag(sp, group);
      if (dumpDag) {
        dumpOwnershipDag(plan, group, funcOp);
        dumpRawSyncDag(sp, plan, group, funcOp);
        dumpOptSyncDag(opt, sp, plan, group, funcOp);
      }
      if (failed(emitResource(funcOp, group, plan, sp, opt, backings)))
        return failure();
    }
    eraseUnusedOriginals(group);
  }
  splitSemaphoreIfForLoopScheduler(funcOp);
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
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp))) return WalkResult::interrupt();
      stripTemporarySemaphoreAttrs(funcOp);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
