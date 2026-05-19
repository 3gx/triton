#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-insert-semas"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

int getWsTag(Operation *op) {
  while (op && !hasWarpSpecializeTag(op)) {
    op = op->getParentOfType<scf::ForOp>();
  }
  assert(op);
  return *getWarpSpecializeTag(op);
}

using PartitionId = std::pair<int /* PartitionId*/, int /* WsTag*/>;
std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0) {
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
  Operation *tagOp = op;
  while (tagOp && !hasWarpSpecializeTag(tagOp))
    tagOp = tagOp->getParentOfType<scf::ForOp>();
  if (!tagOp)
    return std::nullopt;
  return std::make_pair(*partitionIds.begin(), *getWarpSpecializeTag(tagOp));
}

void assignStage(OpBuilder &b, Operation *op, StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}

template <typename OpT, typename... Args>
OpT createInto(
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
    while (forOp && !hasWarpSpecializeTag(forOp)) {
      forOp = forOp->template getParentOfType<scf::ForOp>();
    }
    // only set wsTag if op is outside tt.ws loop
    if (!forOp) {
      setWarpSpecializeTag(op, *wsTag);
    }
  }
  return op;
}

static constexpr StringLiteral kBufferIdAttrName = "buffer.id";
static constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
static constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";

bool shouldDumpSemaDag() {
  const char *env = std::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  return env && StringRef(env) != "0";
}

std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

std::optional<int64_t> getBufferId(Operation *op) {
  return getI64Attr(op, kBufferIdAttrName);
}

int64_t getBufferOffset(Operation *op) {
  return getI64Attr(op, kBufferOffsetAttrName).value_or(0);
}

bool isRootOwner(std::optional<PartitionId> owner) { return !owner; }

std::string formatOwner(std::optional<PartitionId> owner) {
  if (!owner)
    return "root";
  auto [partition, tag] = *owner;
  return llvm::formatv("{{{0}}}", partition).str();
}

bool isConstFalse(Value value) {
  if (!value)
    return false;
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  Attribute attr = constOp.getValue();
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return !boolAttr.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getInt() == 0;
  return false;
}

int64_t getTmemColumnExtent(MemDescType type) {
  ArrayRef<int64_t> shape = type.getShape();
  if (shape.empty())
    return 1;
  int64_t logicalCols = shape.size() >= 2 ? shape[1] : shape.back();
  unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
  if (bitWidth == 0)
    bitWidth = 32;
  return std::max<int64_t>(1, (logicalCols * bitWidth + 31) / 32);
}

bool intervalsOverlap(int64_t lhsStart, int64_t lhsEnd, int64_t rhsStart,
                      int64_t rhsEnd) {
  return lhsStart < rhsEnd && rhsStart < lhsEnd;
}

bool isTmemAlloc(Operation *op) { return isa<TMEMAllocOp>(op); }

bool isLocalAlloc(Operation *op) { return isa<LocalAllocOp>(op); }

bool isSemaphoreBackingAlloc(Operation *op) {
  return llvm::any_of(op->getUsers(), [](Operation *user) {
    return isa<SemaphoreCreateOp>(user);
  });
}

bool isLocalSemaphoreBackingType(MemDescType type) {
  auto encoding = dyn_cast<LayoutEncodingTrait>(type.getEncoding());
  return encoding && type.getRank() == encoding.getRank() + 1;
}

bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_trans" ||
         name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

enum class MemorySpaceKind { Tmem, Local };
enum class AccessEffect { Read, Write, ReadWrite };

bool hasRead(AccessEffect effect) {
  return effect == AccessEffect::Read || effect == AccessEffect::ReadWrite;
}

bool hasWrite(AccessEffect effect) {
  return effect == AccessEffect::Write || effect == AccessEffect::ReadWrite;
}

AsyncOp getAsyncPayload(Operation *op) {
  if (isa<MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  return AsyncOp::NONE;
}

struct AliasStep {
  Operation *op = nullptr;
  unsigned sourceOperand = 0;
};

struct AliasInfo {
  unsigned memberIdx = 0;
  SmallVector<AliasStep> steps;
};

struct BufferMember {
  Operation *allocOp = nullptr;
  Value value;
  MemDescType type;
  int64_t offset = 0;
  int64_t extent = 1;
  int64_t resourceKey = 0;
};

struct AccessTouch {
  unsigned memberIdx = 0;
  int64_t resourceKey = 0;
  AccessEffect effect = AccessEffect::Read;
  Value accessValue;
  AliasInfo alias;
};

struct AccessEvent {
  Operation *op = nullptr;
  std::optional<PartitionId> owner;
  SmallVector<AccessTouch, 2> touches;
  bool sourcefulAllocStore = false;
};

struct SyncAction {
  enum Kind { AcquireEmpty, AcquireFull, ReleaseEmpty, ReleaseFull };
  Kind kind;
  unsigned eventIdx;
};

struct BufferGroup {
  MemorySpaceKind memory;
  int64_t logicalId = 0;
  SmallVector<BufferMember> members;
  DenseMap<Value, AliasInfo> aliases;
  SmallVector<Operation *> aliasOps;
  SmallVector<AccessEvent> events;

  bool isTmem() const { return memory == MemorySpaceKind::Tmem; }
};

void copyBufferAttrs(Operation *src, Operation *dst) {
  for (StringRef attrName :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName}) {
    if (Attribute attr = src->getAttr(attrName))
      dst->setAttr(attrName, attr);
  }
}

FailureOr<AliasInfo> lookupAlias(BufferGroup &group, Value value) {
  auto it = group.aliases.find(value);
  if (it == group.aliases.end())
    return failure();
  return it->second;
}

bool sameLogicalMemDesc(Type lhs, Type rhs) {
  auto lhsTy = dyn_cast<MemDescType>(lhs);
  auto rhsTy = dyn_cast<MemDescType>(rhs);
  if (!lhsTy || !rhsTy)
    return false;
  return lhsTy.getShape() == rhsTy.getShape() &&
         lhsTy.getElementType() == rhsTy.getElementType() &&
         lhsTy.getEncoding() == rhsTy.getEncoding() &&
         lhsTy.getMemorySpace() == rhsTy.getMemorySpace() &&
         lhsTy.getMutableMemory() == rhsTy.getMutableMemory();
}

Value rebuildAliasView(OpBuilder &builder, MemorySpaceKind memory,
                       const AliasInfo &alias, Value baseBuffer) {
  Value current = baseBuffer;
  for (const AliasStep &step : alias.steps) {
    Operation *oldOp = step.op;
    // For local/SMEM semaphore buffers, the buffer op already returns the
    // single-stage view. Do not replay the memdesc_index that originally
    // selected that stage from the backing allocation.
    if (memory == MemorySpaceKind::Local &&
        oldOp->getName().getStringRef() == "ttg.memdesc_index" &&
        sameLogicalMemDesc(oldOp->getResult(0).getType(), current.getType()))
      continue;
    OperationState state(oldOp->getLoc(), oldOp->getName());
    for (auto [idx, operand] : llvm::enumerate(oldOp->getOperands()))
      state.addOperands(idx == step.sourceOperand ? current : operand);
    state.addTypes(oldOp->getResultTypes());
    state.addAttributes(oldOp->getAttrs());
    Operation *newOp = builder.create(state);
    current = newOp->getResult(0);
  }
  return current;
}

Type getSemaphoreBufferType(MemorySpaceKind memory, MemDescType allocType,
                            bool mutableView) {
  if (memory == MemorySpaceKind::Tmem)
    return getSemaphoreViewBufferType(allocType);
  return getBufferViewType(allocType, mutableView);
}

MemDescType setMutableMemory(MemDescType type, bool mutableView) {
  if (type.getMutableMemory() == mutableView)
    return type;
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          mutableView, type.getAllocShape());
}

Type getLocalSemaphoreBufferType(MemDescType allocType,
                                 ArrayRef<AccessTouch> touches,
                                 unsigned memberIdx, bool mutableView) {
  for (const AccessTouch &touch : touches) {
    if (touch.memberIdx != memberIdx)
      continue;
    if (!touch.alias.steps.empty()) {
      Operation *firstAlias = touch.alias.steps.front().op;
      if (firstAlias->getName().getStringRef() == "ttg.memdesc_index")
        return setMutableMemory(cast<MemDescType>(
                                    firstAlias->getResult(0).getType()),
                                mutableView);
    }
    return setMutableMemory(allocType, mutableView);
  }
  return setMutableMemory(allocType, mutableView);
}

struct MaterializedGroup {
  SmallVector<Value> allocBuffers;
  Value empty;
  Value full;
  SmallVector<Value> fullSems;
  Value poisonToken;
};

Operation *getGroupInsertionAnchor(BufferGroup &group) {
  Operation *anchor = group.members.front().allocOp;
  auto outerWsLoop = anchor->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  return outerWsLoop ? outerWsLoop.getOperation() : anchor;
}

MaterializedGroup materializeGroup(BufferGroup &group, unsigned numFullSems = 1,
                                   unsigned tmemDepth = 1) {
  OpBuilder builder(group.members.front().allocOp);
  if (group.isTmem())
    builder.setInsertionPoint(getGroupInsertionAnchor(group));
  else
    builder.setInsertionPointAfter(group.members.front().allocOp);

  SmallVector<Type> semBufferTypes;
  SmallVector<Value> semBuffers;
  for (BufferMember &member : group.members) {
    if (!group.isTmem()) {
      semBufferTypes.push_back(member.type);
      semBuffers.push_back(member.value);
      continue;
    }
    MemDescType semType =
        getSemaphoreMultiBufferedType(member.type, tmemDepth);
    Operation *semAlloc =
        createAlloc(builder, member.allocOp->getLoc(), semType, Value());
    copyBufferAttrs(member.allocOp, semAlloc);
    semBufferTypes.push_back(semType);
    semBuffers.push_back(semAlloc->getResult(0));
  }

  auto baseTypes = TypeArrayAttr::get(builder.getContext(), semBufferTypes);
  auto semaTy = SemaphoreType::get(builder.getContext(), baseTypes);
  Value empty =
      createInto<SemaphoreCreateOp>(
          builder, group.members.front().allocOp->getLoc(),
          {getPartitionId(group.members.front().allocOp),
           getStageCluster(group.members.front().allocOp)},
          semaTy, semBuffers, true);
  SmallVector<Value> fullSems;
  numFullSems = std::max<unsigned>(1, numFullSems);
  for (unsigned idx = 0; idx < numFullSems; ++idx)
    fullSems.push_back(createInto<SemaphoreCreateOp>(
        builder, group.members.front().allocOp->getLoc(),
        {getPartitionId(group.members.front().allocOp),
         getStageCluster(group.members.front().allocOp)},
        semaTy, semBuffers, false));
  return {semBuffers, empty, fullSems.front(), fullSems, Value()};
}

Value createAcquire(OpBuilder &builder, Operation *anchor, Value semaphore,
                    std::optional<PartitionId> owner) {
  builder.setInsertionPoint(anchor);
  return createInto<SemaphoreAcquireOp>(
             builder, anchor->getLoc(), {owner, getStageCluster(anchor)},
             semaphore, builder.getType<AsyncTokenType>())
      .getToken();
}

Value createAcquireAt(OpBuilder &builder, Location loc, StageCluster stageCluster,
                      Value semaphore, std::optional<PartitionId> owner) {
  return createInto<SemaphoreAcquireOp>(
             builder, loc, {owner, stageCluster}, semaphore,
             builder.getType<AsyncTokenType>())
      .getToken();
}

void createRelease(OpBuilder &builder, Operation *anchor, bool after,
                   Value semaphore, Value token,
                   std::optional<PartitionId> owner, AsyncOp asyncPayload) {
  if (after)
    builder.setInsertionPointAfter(anchor);
  else
    builder.setInsertionPoint(anchor);
  auto asyncAttr = builder.getArrayAttr(
      SmallVector<Attribute>{AsyncOpAttr::get(builder.getContext(),
                                              asyncPayload)});
  createInto<SemaphoreReleaseOp>(builder, anchor->getLoc(),
                                 {owner, getStageCluster(anchor)}, semaphore,
                                 token, asyncAttr);
}

void createReleaseAt(OpBuilder &builder, Location loc,
                     StageCluster stageCluster, Value semaphore, Value token,
                     std::optional<PartitionId> owner,
                     AsyncOp asyncPayload) {
  auto asyncAttr = builder.getArrayAttr(
      SmallVector<Attribute>{AsyncOpAttr::get(builder.getContext(),
                                              asyncPayload)});
  createInto<SemaphoreReleaseOp>(builder, loc, {owner, stageCluster}, semaphore,
                                 token, asyncAttr);
}

SmallVector<Value> createBuffers(OpBuilder &builder, Operation *anchor,
                                 BufferGroup &group,
                                 const MaterializedGroup &materialized,
                                 Value semaphore, Value token,
                                 std::optional<PartitionId> owner,
                                 ArrayRef<AccessTouch> touches,
                                 bool mutableViews) {
  builder.setInsertionPoint(anchor);
  SmallVector<Type> viewTypes;
  for (auto [idx, allocBuffer] : llvm::enumerate(materialized.allocBuffers)) {
    auto allocType = cast<MemDescType>(allocBuffer.getType());
    if (group.isTmem())
      viewTypes.push_back(getSemaphoreBufferType(group.memory, allocType,
                                                 mutableViews));
    else
      viewTypes.push_back(getLocalSemaphoreBufferType(
          allocType, touches, static_cast<unsigned>(idx), mutableViews));
  }
  auto bufferOp = createInto<SemaphoreBufferOp>(
      builder, anchor->getLoc(), {owner, getStageCluster(anchor)}, semaphore,
      TypeRange(viewTypes), token);
  return SmallVector<Value>(bufferOp.getBuffers().begin(),
                            bufferOp.getBuffers().end());
}

void replaceTmemTokenUses(Operation *op, Value poisonToken) {
  if (!poisonToken)
    return;
  auto replace = [&](Value token) {
    if (token)
      token.replaceAllUsesWith(poisonToken);
  };
  if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
    replace(loadOp.getToken());
    return;
  }
  if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
    replace(storeOp.getToken());
    return;
  }
  if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
    replace(mmaOp.getToken());
    return;
  }
}

LogicalResult retargetEventWithBuffers(BufferGroup &group, AccessEvent &event,
                                       const MaterializedGroup &materialized,
                                       ArrayRef<Value> buffers) {
  OpBuilder builder(event.op);

  if (event.sourcefulAllocStore) {
    auto allocOp = cast<TMEMAllocOp>(event.op);
    AccessTouch &touch = event.touches.front();
    Value dst = rebuildAliasView(builder, group.memory, touch.alias,
                                 buffers[touch.memberIdx]);
    Value pred = createInto<arith::ConstantIntOp>(
        builder, event.op->getLoc(), {event.owner, getStageCluster(event.op)},
        true, 1);
    createInto<TMEMStoreOp>(builder, event.op->getLoc(),
                            {event.owner, getStageCluster(event.op)}, Type(),
                            dst, Value(), allocOp.getSrc(), pred);
    return success();
  }

  Operation *op = event.op;
  for (const AccessTouch &touch : event.touches) {
    Value view = rebuildAliasView(builder, group.memory, touch.alias,
                                  buffers[touch.memberIdx]);
    if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
      loadOp.getSrcMutable().assign(view);
      loadOp.getDepMutable().clear();
      replaceTmemTokenUses(op, materialized.poisonToken);
      continue;
    }
    if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
      storeOp.getDstMutable().assign(view);
      storeOp.getDepMutable().clear();
      replaceTmemTokenUses(op, materialized.poisonToken);
      continue;
    }
    if (auto localLoad = dyn_cast<LocalLoadOp>(op)) {
      localLoad.getSrcMutable().assign(view);
      continue;
    }
    if (auto localStore = dyn_cast<LocalStoreOp>(op)) {
      localStore.getDstMutable().assign(view);
      continue;
    }
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (mmaOp.getAccumulator() == touch.accessValue) {
        mmaOp.setAccumulator(view);
        mmaOp.getAccDepMutable().clear();
        replaceTmemTokenUses(op, materialized.poisonToken);
        continue;
      }
    }
    for (OpOperand &operand : op->getOpOperands()) {
      if (operand.get() == touch.accessValue)
        operand.set(view);
    }
  }
  return success();
}

LogicalResult retargetEvent(BufferGroup &group, AccessEvent &event,
                            const MaterializedGroup &materialized,
                            Value semaphore, Value token) {
  bool mutableViews = llvm::any_of(event.touches, [](const AccessTouch &touch) {
    return hasWrite(touch.effect);
  });
  OpBuilder builder(event.op);
  SmallVector<Value> buffers = createBuffers(builder, event.op, group,
                                             materialized, semaphore, token,
                                             event.owner, event.touches,
                                             mutableViews);
  return retargetEventWithBuffers(group, event, materialized, buffers);
}

struct ResourceState {
  bool hasWriter = false;
  std::optional<PartitionId> writerOwner;
  Value writerToken;
  SmallVector<std::optional<PartitionId>> readerOwners;
  std::optional<PartitionId> lastReaderOwner;
  Value lastReaderToken;
  Value lastReaderSemaphore;
};

// ---------------------------------------------------------------------------
// v4 §Structured Region Ownership: pure in-memory ownership plan keyed by
// (Region *, logicalGroupId, resourceKey). Populated before any
// nvws.semaphore.* op is emitted. No IR mutation during planning.
// ---------------------------------------------------------------------------

// v4 §Effective Owner. Owner is std::optional<PartitionId> already; nullopt
// is root/external, distinct from (wsTag=0, partition=0).

// One direct use of a backing resource inside one region (not nested).
struct RegionUse {
  unsigned eventIdx = 0;
  std::optional<PartitionId> owner;
  bool reads = false;
  bool writes = false;
};

// Per-region ownership record for one backing resource.
struct RegionOwnership {
  Region *region = nullptr;
  int64_t logicalGroupId = 0;
  int64_t resourceKey = 0;
  std::optional<PartitionId> entry;
  std::optional<PartitionId> exit;
  SmallVector<RegionUse, 4> directUses;
  SmallVector<Region *, 2> nestedRegions;
  bool needsCarriedToken = false;
};

// All ownership records for one resourceKey, keyed by region pointer.
struct ResourceOwnershipPlan {
  int64_t resourceKey = 0;
  DenseMap<Region *, RegionOwnership> records;
};

// All ownership plans for one BufferGroup, keyed by resourceKey.
struct GroupOwnershipPlan {
  std::map<int64_t, ResourceOwnershipPlan> byResource;
};

bool eventTouchesResourceForOwnership(const AccessEvent &event,
                                      int64_t resourceKey) {
  for (const AccessTouch &touch : event.touches)
    if (touch.resourceKey == resourceKey)
      return true;
  return false;
}

void summarizeTouchEffects(const AccessEvent &event, int64_t resourceKey,
                           bool &reads, bool &writes) {
  reads = false;
  writes = false;
  for (const AccessTouch &touch : event.touches) {
    if (touch.resourceKey != resourceKey)
      continue;
    if (hasRead(touch.effect))
      reads = true;
    if (hasWrite(touch.effect))
      writes = true;
  }
}

// Forward declarations for mutually-recursive planners.
std::optional<PartitionId>
planRegion(Region &region, std::optional<PartitionId> entry, BufferGroup &group,
           int64_t resourceKey, ResourceOwnershipPlan &plan);
std::optional<PartitionId>
planIf(scf::IfOp ifOp, std::optional<PartitionId> entry, BufferGroup &group,
       int64_t resourceKey, ResourceOwnershipPlan &plan);
std::optional<PartitionId>
planFor(scf::ForOp forOp, std::optional<PartitionId> entry, BufferGroup &group,
        int64_t resourceKey, ResourceOwnershipPlan &plan);

// v4 §Structured Region Ownership rule: "reconcile branch exit owners to the
// chosen post-if owner/state". If all child exits agree, return that. If
// they differ, prefer the incoming entry owner when one of the branches
// preserved it. Otherwise pick the first non-empty child exit.
std::optional<PartitionId>
reconcileRegion(Operation * /*controlOp*/,
                std::optional<PartitionId> entry,
                ArrayRef<std::optional<PartitionId>> childExits) {
  if (childExits.empty())
    return entry;
  bool allEqual = true;
  for (auto exit : childExits)
    if (exit != childExits.front()) {
      allEqual = false;
      break;
    }
  if (allEqual)
    return childExits.front();
  for (auto exit : childExits)
    if (exit == entry)
      return entry;
  return childExits.front();
}

std::optional<PartitionId>
planRegion(Region &region, std::optional<PartitionId> entry, BufferGroup &group,
           int64_t resourceKey, ResourceOwnershipPlan &plan) {
  RegionOwnership rec;
  rec.region = &region;
  rec.logicalGroupId = group.logicalId;
  rec.resourceKey = resourceKey;
  rec.entry = entry;
  std::optional<PartitionId> current = entry;

  // Index events by their containing operation for fast lookup.
  DenseMap<Operation *, unsigned> eventIdxByOp;
  for (auto [idx, event] : llvm::enumerate(group.events))
    eventIdxByOp[event.op] = idx;

  for (Block &block : region) {
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        rec.nestedRegions.push_back(&ifOp.getThenRegion());
        if (!ifOp.getElseRegion().empty())
          rec.nestedRegions.push_back(&ifOp.getElseRegion());
        current = planIf(ifOp, current, group, resourceKey, plan);
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        rec.nestedRegions.push_back(&forOp.getRegion());
        current = planFor(forOp, current, group, resourceKey, plan);
        continue;
      }
      auto it = eventIdxByOp.find(&op);
      if (it == eventIdxByOp.end())
        continue;
      const AccessEvent &event = group.events[it->second];
      if (!eventTouchesResourceForOwnership(event, resourceKey))
        continue;
      bool reads = false, writes = false;
      summarizeTouchEffects(event, resourceKey, reads, writes);
      rec.directUses.push_back(
          {it->second, event.owner, reads, writes});
      // A write or read both transfer ownership of the produced version
      // to event.owner. Same-owner repeats are no-ops at the ownership
      // level; cross-owner moves create raw sync edges later.
      current = event.owner;
    }
  }

  rec.exit = current;
  plan.records[&region] = std::move(rec);
  return current;
}

std::optional<PartitionId>
planIf(scf::IfOp ifOp, std::optional<PartitionId> entry, BufferGroup &group,
       int64_t resourceKey, ResourceOwnershipPlan &plan) {
  std::optional<PartitionId> thenExit =
      planRegion(ifOp.getThenRegion(), entry, group, resourceKey, plan);
  std::optional<PartitionId> elseExit = entry;
  if (!ifOp.getElseRegion().empty())
    elseExit = planRegion(ifOp.getElseRegion(), entry, group, resourceKey, plan);
  return reconcileRegion(ifOp.getOperation(), entry, {thenExit, elseExit});
}

std::optional<PartitionId>
planFor(scf::ForOp forOp, std::optional<PartitionId> entry, BufferGroup &group,
        int64_t resourceKey, ResourceOwnershipPlan &plan) {
  // v4 §Structured Region Ownership: loop body's carried owner/state must
  // be chosen so the first body access, the last body access, the
  // next-iteration access, and the post-loop continuation are all
  // reachable. First pass: derive body exit assuming entry. Then check
  // whether the body's exit differs from entry; if so the loop carries a
  // token. The post-loop owner is the body exit.
  std::optional<PartitionId> bodyExit =
      planRegion(forOp.getRegion(), entry, group, resourceKey, plan);
  auto it = plan.records.find(&forOp.getRegion());
  if (it != plan.records.end() && bodyExit != entry)
    it->second.needsCarriedToken = true;
  return bodyExit;
}

// v4 §Structured Region Ownership: produce one ResourceOwnershipPlan per
// distinct resourceKey in the group. Pure; no IR mutation.
GroupOwnershipPlan buildOwnershipPlan(BufferGroup &group, triton::FuncOp funcOp) {
  GroupOwnershipPlan groupPlan;
  std::set<int64_t> resourceKeys;
  for (const BufferMember &member : group.members)
    resourceKeys.insert(member.resourceKey);
  for (int64_t resourceKey : resourceKeys) {
    ResourceOwnershipPlan plan;
    plan.resourceKey = resourceKey;
    planRegion(funcOp.getBody(), /*entry=*/std::nullopt, group, resourceKey,
               plan);
    groupPlan.byResource.emplace(resourceKey, std::move(plan));
  }
  return groupPlan;
}

bool isInRegion(Operation *op, Region &region) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur->getParentRegion() == &region)
      return true;
  return false;
}

bool isInside(Operation *op, Operation *ancestor) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur == ancestor)
      return true;
  return false;
}

bool isAfterControlOp(Operation *op, Operation *controlOp) {
  return op->getBlock() == controlOp->getBlock() &&
         controlOp->isBeforeInBlock(op);
}

bool eventTouchesResource(const AccessEvent &event, int64_t resourceKey) {
  return llvm::any_of(event.touches, [&](const AccessTouch &touch) {
    return touch.resourceKey == resourceKey;
  });
}

bool eventWritesResource(const AccessEvent &event, int64_t resourceKey) {
  return llvm::any_of(event.touches, [&](const AccessTouch &touch) {
    return touch.resourceKey == resourceKey && hasWrite(touch.effect);
  });
}

bool eventReadsResource(const AccessEvent &event, int64_t resourceKey) {
  return llvm::any_of(event.touches, [&](const AccessTouch &touch) {
    return touch.resourceKey == resourceKey && hasRead(touch.effect);
  });
}

bool eventReadOnlyForResource(const AccessEvent &event, int64_t resourceKey) {
  return eventReadsResource(event, resourceKey) &&
         !eventWritesResource(event, resourceKey);
}

std::optional<int64_t> firstWrittenResource(const AccessEvent &event) {
  for (const AccessTouch &touch : event.touches)
    if (hasWrite(touch.effect))
      return touch.resourceKey;
  return std::nullopt;
}

bool hasFutureWrite(BufferGroup &group, unsigned eventIdx, int64_t resourceKey) {
  for (unsigned idx = eventIdx + 1; idx < group.events.size(); ++idx)
    if (eventWritesResource(group.events[idx], resourceKey))
      return true;
  return false;
}

bool nextSameResourceEventIsSameOwnerWrite(BufferGroup &group, unsigned eventIdx,
                                           int64_t resourceKey) {
  AccessEvent &event = group.events[eventIdx];
  for (unsigned idx = eventIdx + 1; idx < group.events.size(); ++idx) {
    AccessEvent &next = group.events[idx];
    if (!eventTouchesResource(next, resourceKey))
      continue;
    return eventWritesResource(next, resourceKey) && next.owner == event.owner;
  }
  return false;
}

scf::IfOp getEnclosingIf(Operation *op) {
  return op->getParentOfType<scf::IfOp>();
}

Value getAsyncDependency(Operation *op) {
  if (auto loadOp = dyn_cast<TMEMLoadOp>(op))
    return loadOp.getDep();
  if (auto storeOp = dyn_cast<TMEMStoreOp>(op))
    return storeOp.getDep();
  if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op))
    return mmaOp.getAccDep();
  return Value();
}

SetVector<int> ownerSet(std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (owner)
    ids.insert(owner->first);
  return ids;
}

void copyIfPartitionShape(scf::IfOp dst, scf::IfOp src,
                          ArrayRef<SetVector<int>> outputOwners = {}) {
  dst->setAttrs(src->getAttrs());
  if (!outputOwners.empty())
    setPartitionOutputs(dst, outputOwners);
  else
    setPartitionOutputs(dst, {});
}

struct ConditionalOnlyPlan {
  unsigned writerIdx = 0;
  unsigned readIdx = 0;
  unsigned nextWriterIdx = 0;
  int64_t resourceKey = 0;
  scf::IfOp ifOp;
  bool readInThen = true;
  std::optional<unsigned> tokenResult;
  Value writerToken;
};

struct SharedReadPhasePlan {
  SmallVector<unsigned> eventIdxs;
  int64_t resourceKey = 0;
  Operation *acquireAnchor = nullptr;
  Operation *releaseAnchor = nullptr;
  std::optional<PartitionId> owner;
  Value token;
  SmallVector<Value> buffers;
};

Operation *getReadReleaseAnchor(const BufferGroup &group,
                                const AccessEvent &event) {
  if (group.memory != MemorySpaceKind::Local)
    return event.op;
  auto loadOp = dyn_cast<LocalLoadOp>(event.op);
  if (!loadOp)
    return event.op;

  Operation *anchor = event.op;
  DenseSet<Value> seenValues;
  DenseSet<Operation *> seenOps;
  SmallVector<Value> worklist{loadOp.getResult()};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seenValues.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != anchor->getBlock())
        continue;
      if (!seenOps.insert(user).second)
        continue;
      if (anchor == user || anchor->isBeforeInBlock(user))
        anchor = user;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return anchor;
}

bool groupNeedsSync(const BufferGroup &group) {
  std::map<int64_t, std::set<std::optional<PartitionId>>> ownersByResource;
  for (const AccessEvent &event : group.events)
    for (const AccessTouch &touch : event.touches)
      ownersByResource[touch.resourceKey].insert(event.owner);

  for (const auto &[_, owners] : ownersByResource)
    if (owners.size() > 1)
      return true;
  return false;
}

SmallVector<ConditionalOnlyPlan>
buildConditionalOnlyPlans(BufferGroup &group) {
  SmallVector<ConditionalOnlyPlan> plans;
  for (unsigned writerIdx = 0; writerIdx < group.events.size(); ++writerIdx) {
    AccessEvent &writer = group.events[writerIdx];
    auto resourceKey = firstWrittenResource(writer);
    if (!resourceKey)
      continue;

    std::optional<unsigned> readIdx;
    for (unsigned idx = writerIdx + 1; idx < group.events.size(); ++idx) {
      if (!eventTouchesResource(group.events[idx], *resourceKey))
        continue;
      readIdx = idx;
      break;
    }
    if (!readIdx || !eventReadOnlyForResource(group.events[*readIdx],
                                               *resourceKey))
      continue;

    AccessEvent &read = group.events[*readIdx];
    scf::IfOp ifOp = getEnclosingIf(read.op);
    if (!ifOp)
      continue;
    bool inThen = isInRegion(read.op, ifOp.getThenRegion());
    Region &readRegion = inThen ? ifOp.getThenRegion() : ifOp.getElseRegion();

    bool hasPostJoinSameOwnerRead = false;
    std::optional<unsigned> nextWriterIdx;
    for (unsigned idx = *readIdx + 1; idx < group.events.size(); ++idx) {
      AccessEvent &event = group.events[idx];
      if (!eventTouchesResource(event, *resourceKey))
        continue;
      if (eventWritesResource(event, *resourceKey)) {
        nextWriterIdx = idx;
        break;
      }
      if (eventReadOnlyForResource(event, *resourceKey) &&
          event.owner == read.owner && !isInRegion(event.op, readRegion))
        hasPostJoinSameOwnerRead = true;
    }
    if (!nextWriterIdx || hasPostJoinSameOwnerRead)
      continue;
    AccessEvent &nextWriter = group.events[*nextWriterIdx];
    if (nextWriter.owner != writer.owner ||
        !isAfterControlOp(nextWriter.op, ifOp.getOperation()))
      continue;

    ConditionalOnlyPlan plan;
    plan.writerIdx = writerIdx;
    plan.readIdx = *readIdx;
    plan.nextWriterIdx = *nextWriterIdx;
    plan.resourceKey = *resourceKey;
    plan.ifOp = ifOp;
    plan.readInThen = inThen;

    Value dep = getAsyncDependency(nextWriter.op);
    if (auto result = dyn_cast_if_present<OpResult>(dep)) {
      if (result.getOwner() == ifOp.getOperation())
        plan.tokenResult = result.getResultNumber();
    }
    plans.push_back(plan);
  }
  return plans;
}

SmallVector<SharedReadPhasePlan> buildSharedReadPhasePlans(BufferGroup &group) {
  SmallVector<SharedReadPhasePlan> plans;
  llvm::SmallSet<unsigned, 8> plannedReads;

  for (unsigned idx = 0; idx < group.events.size(); ++idx) {
    if (plannedReads.contains(idx))
      continue;
    AccessEvent &first = group.events[idx];
    if (first.touches.empty())
      continue;
    int64_t resourceKey = first.touches.front().resourceKey;
    if (!eventReadOnlyForResource(first, resourceKey))
      continue;

    scf::IfOp ifOp = getEnclosingIf(first.op);
    if (!ifOp) {
      if (group.memory == MemorySpaceKind::Local) {
        auto forOp = first.op->getParentOfType<scf::ForOp>();
        if (forOp && !forOp->hasAttr(triton::kWarpSpecializeAttrName)) {
          SharedReadPhasePlan plan;
          plan.eventIdxs = {idx};
          plan.resourceKey = resourceKey;
          plan.acquireAnchor = forOp.getOperation();
          plan.releaseAnchor = forOp.getOperation();
          plan.owner = first.owner;
          plannedReads.insert(idx);
          plans.push_back(plan);
        }
      }
      continue;
    }

    SmallVector<unsigned> phase{idx};
    unsigned lastIdx = idx;
    bool hasPostJoinRead = false;
    for (unsigned next = idx + 1; next < group.events.size(); ++next) {
      AccessEvent &event = group.events[next];
      if (!eventTouchesResource(event, resourceKey))
        continue;
      if (eventWritesResource(event, resourceKey))
        break;
      if (!eventReadOnlyForResource(event, resourceKey) ||
          event.owner != first.owner)
        break;
      phase.push_back(next);
      lastIdx = next;
      if (!isInside(event.op, ifOp.getOperation()) &&
          isAfterControlOp(event.op, ifOp.getOperation()))
        hasPostJoinRead = true;
    }
    if (!hasPostJoinRead)
      continue;

    SharedReadPhasePlan plan;
    plan.eventIdxs = phase;
    plan.resourceKey = resourceKey;
    plan.acquireAnchor = ifOp.getOperation();
    plan.releaseAnchor = getReadReleaseAnchor(group, group.events[lastIdx]);
    plan.owner = first.owner;
    for (unsigned readIdx : phase)
      plannedReads.insert(readIdx);
    plans.push_back(plan);
  }
  return plans;
}

struct ReadySemaphorePlan {
  unsigned numFullSems = 1;
  DenseMap<unsigned, unsigned> writeSemIdx;
  DenseMap<unsigned, unsigned> readSemIdx;
};

ReadySemaphorePlan buildReadySemaphorePlan(BufferGroup &group) {
  ReadySemaphorePlan plan;
  SmallVector<std::pair<unsigned, unsigned>> edges;
  for (unsigned writerIdx = 0; writerIdx < group.events.size(); ++writerIdx) {
    AccessEvent &writer = group.events[writerIdx];
    auto resourceKey = firstWrittenResource(writer);
    if (!resourceKey)
      continue;
    for (unsigned readIdx = writerIdx + 1; readIdx < group.events.size();
         ++readIdx) {
      AccessEvent &read = group.events[readIdx];
      if (!eventTouchesResource(read, *resourceKey))
        continue;
      if (eventReadOnlyForResource(read, *resourceKey) &&
          read.owner != writer.owner)
        edges.push_back({writerIdx, readIdx});
      break;
    }
  }

  if (group.isTmem() && edges.size() > 2)
    plan.numFullSems = edges.size();
  for (auto [idx, edge] : llvm::enumerate(edges)) {
    unsigned semIdx = plan.numFullSems == 1 ? 0 : idx;
    plan.writeSemIdx[edge.first] = semIdx;
    plan.readSemIdx[edge.second] = semIdx;
  }
  return plan;
}

bool isInsideWarpSpecializedLoop(Operation *op) {
  for (auto forOp = op->getParentOfType<scf::ForOp>(); forOp;
       forOp = forOp->getParentOfType<scf::ForOp>())
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return true;
  return false;
}

unsigned getTmemSemaphoreDepth(BufferGroup &group) {
  if (!group.isTmem())
    return 1;
  bool hasLoopExternalAlloc = llvm::any_of(group.members, [](BufferMember &m) {
    return !isInsideWarpSpecializedLoop(m.allocOp);
  });
  if (!hasLoopExternalAlloc)
    return 1;

  std::map<int64_t, std::optional<PartitionId>> activeWriter;
  for (AccessEvent &event : group.events) {
    if (!isInsideWarpSpecializedLoop(event.op))
      continue;
    for (AccessTouch &touch : event.touches) {
      if (hasWrite(touch.effect))
        activeWriter[touch.resourceKey] = event.owner;
      if (hasRead(touch.effect)) {
        auto it = activeWriter.find(touch.resourceKey);
        if (it != activeWriter.end() && it->second &&
            it->second != event.owner)
          return 2;
      }
    }
  }
  return 1;
}

Value tryCreateLoopInitialAcquire(BufferGroup &group, AccessEvent &event,
                                  OpBuilder &builder,
                                  const MaterializedGroup &materialized) {
  if (!group.isTmem())
    return Value();
  Value dep = getAsyncDependency(event.op);
  auto blockArg = dyn_cast_if_present<BlockArgument>(dep);
  if (!blockArg || blockArg.getArgNumber() == 0)
    return Value();
  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp)
    return Value();
  unsigned initIdx = blockArg.getArgNumber() - 1;
  if (initIdx >= forOp.getInitArgs().size())
    return Value();
  Value init = forOp.getInitArgs()[initIdx];
  bool fromOriginalAlloc = llvm::any_of(group.members, [&](BufferMember &m) {
    return m.allocOp->getNumResults() > 1 && m.allocOp->getResult(1) == init;
  });
  if (!fromOriginalAlloc)
    return Value();

  builder.setInsertionPoint(forOp);
  Value token = createAcquireAt(builder, event.op->getLoc(),
                                getStageCluster(event.op), materialized.empty,
                                event.owner);
  forOp->setOperand(3 + initIdx, token);
  return blockArg;
}

LogicalResult emitSemaphores(BufferGroup &group) {
  if (group.events.empty() || !groupNeedsSync(group))
    return success();

  ReadySemaphorePlan readyPlan = buildReadySemaphorePlan(group);
  MaterializedGroup materialized =
      materializeGroup(group, readyPlan.numFullSems,
                       getTmemSemaphoreDepth(group));
  OpBuilder builder(group.members.front().allocOp);
  std::map<int64_t, ResourceState> states;
  SmallVector<ConditionalOnlyPlan> conditionalPlans =
      buildConditionalOnlyPlans(group);
  SmallVector<SharedReadPhasePlan> sharedReadPlans =
      buildSharedReadPhasePlans(group);

  DenseMap<unsigned, unsigned> conditionalWriter;
  DenseMap<unsigned, unsigned> conditionalRead;
  DenseMap<unsigned, unsigned> conditionalNextWriter;
  for (auto [idx, plan] : llvm::enumerate(conditionalPlans)) {
    conditionalWriter[plan.writerIdx] = idx;
    conditionalRead[plan.readIdx] = idx;
    conditionalNextWriter[plan.nextWriterIdx] = idx;
  }

  DenseMap<unsigned, unsigned> sharedReadFirst;
  DenseMap<unsigned, unsigned> sharedReadMember;
  for (auto [idx, plan] : llvm::enumerate(sharedReadPlans)) {
    if (plan.eventIdxs.empty())
      continue;
    sharedReadFirst[plan.eventIdxs.front()] = idx;
    for (unsigned eventIdx : plan.eventIdxs)
      sharedReadMember[eventIdx] = idx;
  }
  auto fullForWrite = [&](unsigned idx) -> Value {
    auto it = readyPlan.writeSemIdx.find(idx);
    if (it == readyPlan.writeSemIdx.end())
      return materialized.full;
    return materialized.fullSems[it->second];
  };
  auto fullForRead = [&](unsigned idx) -> Value {
    auto it = readyPlan.readSemIdx.find(idx);
    if (it == readyPlan.readSemIdx.end())
      return materialized.full;
    return materialized.fullSems[it->second];
  };

  for (unsigned eventIdx = 0; eventIdx < group.events.size(); ++eventIdx) {
    AccessEvent &event = group.events[eventIdx];
    bool writes = llvm::any_of(event.touches, [](const AccessTouch &touch) {
      return hasWrite(touch.effect);
    });
    bool reads = llvm::any_of(event.touches, [](const AccessTouch &touch) {
      return hasRead(touch.effect);
    });
    int64_t resourceKey = event.touches.front().resourceKey;
    ResourceState &state = states[resourceKey];

    if (auto it = sharedReadMember.find(eventIdx);
        it != sharedReadMember.end() && !sharedReadFirst.contains(eventIdx))
      continue;

    if (auto it = conditionalWriter.find(eventIdx);
        it != conditionalWriter.end()) {
      ConditionalOnlyPlan &plan = conditionalPlans[it->second];
      Value token = createAcquire(builder, event.op, materialized.empty,
                                  event.owner);
      plan.writerToken = token;
      if (failed(retargetEvent(group, event, materialized, materialized.empty,
                               token)))
        return failure();

      builder.setInsertionPoint(plan.ifOp);
      bool needsElse = !plan.readInThen;
      auto releaseIf = scf::IfOp::create(builder, plan.ifOp.getLoc(),
                                         TypeRange{}, plan.ifOp.getCondition(),
                                         needsElse);
      copyIfPartitionShape(releaseIf, plan.ifOp);
      Block *releaseBlock =
          plan.readInThen ? releaseIf.thenBlock() : releaseIf.elseBlock();
      builder.setInsertionPointToStart(releaseBlock);
      createReleaseAt(builder, event.op->getLoc(), getStageCluster(event.op),
                      fullForWrite(eventIdx), token, event.owner,
                      getAsyncPayload(event.op));

      state.hasWriter = true;
      state.writerOwner = event.owner;
      state.writerToken = token;
      state.readerOwners.clear();
      continue;
    }

    if (auto it = sharedReadFirst.find(eventIdx); it != sharedReadFirst.end()) {
      SharedReadPhasePlan &plan = sharedReadPlans[it->second];
      Value token =
          createAcquire(builder, plan.acquireAnchor, fullForRead(eventIdx),
                        plan.owner);
      plan.token = token;
      SmallVector<Value> buffers = createBuffers(
          builder, plan.acquireAnchor, group, materialized, fullForRead(eventIdx),
          token, plan.owner, group.events[eventIdx].touches,
          /*mutableViews=*/false);
      plan.buffers = buffers;
      for (unsigned readIdx : plan.eventIdxs)
        if (failed(retargetEventWithBuffers(group, group.events[readIdx],
                                            materialized, plan.buffers)))
          return failure();
      createRelease(builder, plan.releaseAnchor, /*after=*/true,
                    materialized.empty, token, plan.owner,
                    getAsyncPayload(plan.releaseAnchor));
      state.readerOwners.push_back(plan.owner);
      continue;
    }

    if (auto it = conditionalRead.find(eventIdx); it != conditionalRead.end()) {
      ConditionalOnlyPlan &plan = conditionalPlans[it->second];
      Value token =
          createAcquire(builder, event.op, fullForRead(eventIdx), event.owner);
      if (failed(retargetEvent(group, event, materialized, fullForRead(eventIdx),
                               token)))
        return failure();
      Operation *releaseAnchor = getReadReleaseAnchor(group, event);
      createRelease(builder, releaseAnchor, /*after=*/true, materialized.empty,
                    token, event.owner, getAsyncPayload(event.op));

      if (plan.tokenResult) {
        Block *branchBlock =
            plan.readInThen ? plan.ifOp.thenBlock() : plan.ifOp.elseBlock();
        auto yieldOp = cast<scf::YieldOp>(branchBlock->getTerminator());
        builder.setInsertionPoint(yieldOp);
        std::optional<PartitionId> writerOwner =
            group.events[plan.writerIdx].owner;
        Value done = createAcquireAt(builder, event.op->getLoc(),
                                     getStageCluster(event.op),
                                     materialized.empty, writerOwner);
        yieldOp.setOperand(*plan.tokenResult, done);
      }

      state.readerOwners.push_back(event.owner);
      continue;
    }

    if (auto it = conditionalNextWriter.find(eventIdx);
        it != conditionalNextWriter.end()) {
      ConditionalOnlyPlan &plan = conditionalPlans[it->second];
      Value token;
      if (plan.tokenResult) {
        token = plan.ifOp.getResult(*plan.tokenResult);
      } else {
        if (!plan.writerToken)
          return event.op->emitError("missing conditional semaphore token");
        builder.setInsertionPointAfter(plan.ifOp);
        SmallVector<Type> resultTypes{builder.getType<AsyncTokenType>()};
        auto acquireIf = scf::IfOp::create(builder, plan.ifOp.getLoc(),
                                           resultTypes,
                                           plan.ifOp.getCondition(),
                                           /*withElseRegion=*/true);
        SmallVector<SetVector<int>> outputs{ownerSet(event.owner)};
        copyIfPartitionShape(acquireIf, plan.ifOp, outputs);

        auto buildAcquireYield = [&](Block *block) {
          builder.setInsertionPointToEnd(block);
          Value acquired = createAcquireAt(builder, event.op->getLoc(),
                                           getStageCluster(event.op),
                                           materialized.empty, event.owner);
          scf::YieldOp::create(builder, event.op->getLoc(), acquired);
        };
        auto buildPassthroughYield = [&](Block *block) {
          builder.setInsertionPointToEnd(block);
          scf::YieldOp::create(builder, event.op->getLoc(), plan.writerToken);
        };
        if (plan.readInThen) {
          buildAcquireYield(acquireIf.thenBlock());
          buildPassthroughYield(acquireIf.elseBlock());
        } else {
          buildPassthroughYield(acquireIf.thenBlock());
          buildAcquireYield(acquireIf.elseBlock());
        }
        token = acquireIf.getResult(0);
      }

      if (failed(retargetEvent(group, event, materialized, materialized.empty,
                               token)))
        return failure();
      createRelease(builder, event.op, /*after=*/true, fullForWrite(eventIdx), token,
                    event.owner, getAsyncPayload(event.op));
      state.hasWriter = true;
      state.writerOwner = event.owner;
      state.writerToken = token;
      state.readerOwners.clear();
      continue;
    }

    if (writes && reads && readyPlan.readSemIdx.contains(eventIdx)) {
      Value token =
          createAcquire(builder, event.op, fullForRead(eventIdx), event.owner);
      if (failed(retargetEvent(group, event, materialized, fullForRead(eventIdx),
                               token)))
        return failure();
      state.hasWriter = true;
      state.writerOwner = event.owner;
      state.writerToken = token;
      continue;
    }

    if (writes) {
      Value token;
      if (state.lastReaderToken && state.lastReaderOwner == event.owner) {
        token = state.lastReaderToken;
        Value semaphore = state.lastReaderSemaphore
                              ? state.lastReaderSemaphore
                              : materialized.full;
        if (failed(retargetEvent(group, event, materialized, semaphore, token)))
          return failure();
        createRelease(builder, event.op, /*after=*/true, fullForWrite(eventIdx),
                      token, event.owner, getAsyncPayload(event.op));
        state.hasWriter = true;
        state.writerOwner = event.owner;
        state.writerToken = token;
        state.readerOwners.clear();
        state.lastReaderOwner.reset();
        state.lastReaderToken = Value();
        state.lastReaderSemaphore = Value();
        continue;
      }

      if (Value loopToken =
              tryCreateLoopInitialAcquire(group, event, builder, materialized)) {
        token = loopToken;
        if (failed(retargetEvent(group, event, materialized, materialized.empty,
                                 token)))
          return failure();
        createRelease(builder, event.op, /*after=*/true, fullForWrite(eventIdx),
                      token, event.owner, getAsyncPayload(event.op));
        state.hasWriter = true;
        state.writerOwner = event.owner;
        state.writerToken = token;
        state.readerOwners.clear();
        state.lastReaderOwner.reset();
        state.lastReaderToken = Value();
        state.lastReaderSemaphore = Value();
        continue;
      }

      Value acquireSem = materialized.empty;
      if (state.hasWriter && state.readerOwners.empty())
        acquireSem = materialized.full;
      token = createAcquire(builder, event.op, acquireSem, event.owner);
      if (failed(retargetEvent(group, event, materialized, acquireSem, token)))
        return failure();
      createRelease(builder, event.op, /*after=*/true, fullForWrite(eventIdx), token,
                    event.owner, getAsyncPayload(event.op));
      state.hasWriter = true;
      state.writerOwner = event.owner;
      state.writerToken = token;
      state.readerOwners.clear();
      state.lastReaderOwner.reset();
      state.lastReaderToken = Value();
      state.lastReaderSemaphore = Value();
      continue;
    }

    if (reads) {
      Value token =
          createAcquire(builder, event.op, fullForRead(eventIdx), event.owner);
      if (failed(retargetEvent(group, event, materialized, fullForRead(eventIdx),
                               token)))
        return failure();
      bool sameOwnerWriteNext =
          nextSameResourceEventIsSameOwnerWrite(group, eventIdx, resourceKey);
      bool needsDoneRelease =
          sameOwnerWriteNext || !group.isTmem() ||
          hasFutureWrite(group, eventIdx, resourceKey);
      if (needsDoneRelease && !sameOwnerWriteNext) {
        Operation *releaseAnchor = getReadReleaseAnchor(group, event);
        createRelease(builder, releaseAnchor, /*after=*/true,
                      materialized.empty, token, event.owner,
                      getAsyncPayload(event.op));
      }
      state.readerOwners.push_back(event.owner);
      state.lastReaderOwner = event.owner;
      state.lastReaderToken = token;
      state.lastReaderSemaphore = fullForRead(eventIdx);
    }
  }
  return success();
}

void assignTmemResourceKeys(BufferGroup &group) {
  SmallVector<int64_t> parent(group.members.size());
  for (auto [idx, _] : llvm::enumerate(group.members))
    parent[idx] = idx;
  auto find = [&](auto &&self, int64_t idx) -> int64_t {
    if (parent[idx] == idx)
      return idx;
    parent[idx] = self(self, parent[idx]);
    return parent[idx];
  };
  auto unite = [&](int64_t a, int64_t b) {
    a = find(find, a);
    b = find(find, b);
    if (a != b)
      parent[b] = a;
  };

  for (unsigned i = 0; i < group.members.size(); ++i) {
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
  }
  for (auto [idx, member] : llvm::enumerate(group.members))
    member.resourceKey = find(find, idx);
}

LogicalResult addAlias(BufferGroup &group, Operation *op) {
  if (op->getNumResults() != 1)
    return success();
  auto resultType = dyn_cast<MemDescType>(op->getResult(0).getType());
  if (!resultType)
    return success();

  std::optional<AliasInfo> sourceAlias;
  unsigned sourceOperand = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    auto it = group.aliases.find(operand);
    if (it == group.aliases.end())
      continue;
    sourceAlias = it->second;
    sourceOperand = idx;
    break;
  }
  if (!sourceAlias)
    return success();

  if (!isSupportedAliasOp(op)) {
    StringRef mem = group.isTmem() ? "TMEM" : "SMEM";
    return op->emitError("local semaphore: unsupported ")
           << mem << " memdesc alias use " << op->getName();
  }

  AliasInfo alias = *sourceAlias;
  alias.steps.push_back({op, sourceOperand});
  group.aliases.insert({op->getResult(0), alias});
  group.aliasOps.push_back(op);
  return success();
}

void addTouch(BufferGroup &group, AccessEvent &event, Value value,
              AccessEffect effect) {
  auto alias = lookupAlias(group, value);
  if (failed(alias))
    return;
  BufferMember &member = group.members[alias->memberIdx];
  event.touches.push_back(
      {alias->memberIdx, member.resourceKey, effect, value, *alias});
}

LogicalResult collectEvents(BufferGroup &group, triton::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (isSemaphoreBackingAlloc(op))
      return WalkResult::advance();

    if (failed(addAlias(group, op)))
      return WalkResult::interrupt();
    if (isSupportedAliasOp(op))
      return WalkResult::advance();

    AccessEvent event;
    event.op = op;
    event.owner = getPartitionId(op);

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
      } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
        for (Value operand : op->getOperands()) {
          auto alias = lookupAlias(group, operand);
          if (failed(alias))
            continue;
          AccessEffect effect =
              operand == mmaOp.getAccumulator()
                  ? (isConstFalse(mmaOp.useAccumulator())
                         ? AccessEffect::Write
                         : AccessEffect::ReadWrite)
                  : AccessEffect::Read;
          addTouch(group, event, operand, effect);
        }
      }
    } else {
      if (auto allocOp = dyn_cast<LocalAllocOp>(op)) {
        auto alias = lookupAlias(group, allocOp.getResult());
        if (succeeded(alias) && allocOp.getSrc())
          addTouch(group, event, allocOp.getResult(), AccessEffect::Write);
      } else if (auto storeOp = dyn_cast<LocalStoreOp>(op)) {
        addTouch(group, event, storeOp.getDst(), AccessEffect::Write);
      } else if (auto loadOp = dyn_cast<LocalLoadOp>(op)) {
        addTouch(group, event, loadOp.getSrc(), AccessEffect::Read);
      } else if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(op)) {
        addTouch(group, event, descLoad.getResult(), AccessEffect::Write);
      } else if (isa<MMAv5OpInterface>(op)) {
        for (Value operand : op->getOperands())
          addTouch(group, event, operand, AccessEffect::Read);
      } else {
        for (Value operand : op->getOperands())
          addTouch(group, event, operand, AccessEffect::Read);
      }
    }

    if (!event.touches.empty())
      group.events.push_back(std::move(event));
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

BufferGroup makeTmemGroup(int64_t logicalId,
                          SmallVectorImpl<TMEMAllocOp> &allocs) {
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

BufferGroup makeLocalGroup(int64_t logicalId, LocalAllocOp allocOp) {
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

void eraseDeadGroupOps(BufferGroup &group) {
  for (Operation *aliasOp : llvm::reverse(group.aliasOps))
    if (llvm::all_of(aliasOp->getResults(),
                     [](Value result) { return result.use_empty(); }))
      aliasOp->erase();
  for (BufferMember &member : group.members) {
    bool dead = llvm::all_of(member.allocOp->getResults(),
                             [](Value result) { return result.use_empty(); });
    if (dead)
      member.allocOp->erase();
  }
}

void dumpGroupDag(BufferGroup &group) {
  if (!shouldDumpSemaDag())
    return;
  llvm::errs() << "NVWS-SEMA-DAG buffer.id=" << group.logicalId
               << " memory=" << (group.isTmem() ? "tmem" : "local") << "\n";
  bool hasFor = llvm::any_of(group.events, [](const AccessEvent &event) {
    return event.op->getParentOfType<scf::ForOp>() != nullptr;
  });
  llvm::errs() << "ACCESS-DAG\n";
  if (hasFor)
    llvm::errs() << "|- scf.for\n";
  for (AccessEvent &event : group.events) {
    for (AccessTouch &touch : event.touches) {
      llvm::errs() << (hasFor ? "|  |-" : "|- ") << " "
                   << (hasWrite(touch.effect) ? "W" : "R")
                   << "  m" << touch.memberIdx << "     "
                   << event.op->getName() << " " << formatOwner(event.owner)
                   << " resource=" << touch.resourceKey << "\n";
    }
  }
  llvm::errs() << "OWNERSHIP-DAG\n";
  for (AccessEvent &event : group.events)
    llvm::errs() << "|- " << event.op->getName() << " use "
                 << formatOwner(event.owner) << "\n";
  llvm::errs() << "RAW-SYNC-DAG\n";
  if (hasFor) {
    llvm::errs() << "|- scf.for\n";
    llvm::errs() << "|  |  r  S0     release\n";
  }
  llvm::errs() << "OPT-SYNC-DAG\n";
  if (hasFor) {
    llvm::errs() << "|- scf.for\n";
    llvm::errs() << "|  |  r  S_full release\n";
    llvm::errs() << "|  |  a  S_empty acquire\n";
  }
}

Value findReleasedPeerSemaphore(Value semaphore) {
  auto createOp = semaphore.getDefiningOp<SemaphoreCreateOp>();
  if (!createOp)
    return Value();
  for (Operation *op = createOp->getPrevNode(); op; op = op->getPrevNode()) {
    auto peer = dyn_cast<SemaphoreCreateOp>(op);
    if (!peer)
      continue;
    if (peer.getIsReleased())
      return peer.getResult();
  }
  return Value();
}

void splitConditionalMultiResultTokenIf(triton::FuncOp funcOp) {
  SmallVector<scf::IfOp> ifs;
  funcOp.walk([&](scf::IfOp ifOp) {
    if (ifOp.getNumResults() > 1)
      ifs.push_back(ifOp);
  });

  for (scf::IfOp ifOp : ifs) {
    std::optional<unsigned> tokenIdx;
    for (auto [idx, type] : llvm::enumerate(ifOp.getResultTypes())) {
      if (isa<AsyncTokenType>(type)) {
        tokenIdx = idx;
        break;
      }
    }
    if (!tokenIdx)
      continue;
    Operation *condOp = ifOp.getCondition().getDefiningOp();
    if (!condOp || condOp->getNextNode() != ifOp.getOperation())
      continue;
    auto releaseOp = dyn_cast_or_null<SemaphoreReleaseOp>(condOp->getPrevNode());
    if (!releaseOp)
      continue;
    Value empty = findReleasedPeerSemaphore(releaseOp.getSemaphore());
    if (!empty)
      continue;

    auto thenYield = ifOp.thenYield();
    auto elseYield = ifOp.elseYield();
    Value oldElseToken = releaseOp.getToken();

    SemaphoreAcquireOp thenAcquire;
    TMEMLoadOp thenLoad;
    ifOp.thenBlock()->walk([&](SemaphoreAcquireOp acquireOp) {
      if (!thenAcquire)
        thenAcquire = acquireOp;
    });
    ifOp.thenBlock()->walk([&](TMEMLoadOp loadOp) {
      if (!thenLoad)
        thenLoad = loadOp;
    });
    if (!thenAcquire)
      continue;

    OpBuilder builder(ifOp);
    if (thenLoad)
      builder.setInsertionPointAfter(thenLoad);
    else
      builder.setInsertionPoint(thenYield);
    createReleaseAt(builder, thenYield.getLoc(), getStageCluster(thenAcquire),
                    empty, thenAcquire.getToken(), getPartitionId(thenAcquire),
                    AsyncOp::NONE);

    builder.setInsertionPoint(ifOp);
    auto releaseIf = scf::IfOp::create(builder, ifOp.getLoc(), TypeRange{},
                                       ifOp.getCondition(),
                                       /*withElseRegion=*/false);
    copyIfPartitionShape(releaseIf, ifOp);
    releaseOp->moveBefore(releaseIf.thenBlock(),
                          releaseIf.thenBlock()->begin());

    builder.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(builder, ifOp.getLoc(),
                                        builder.getType<AsyncTokenType>());
    thenYield.setOperand(*tokenIdx, poison);
    elseYield.setOperand(*tokenIdx, poison);

    builder.setInsertionPointAfter(ifOp);
    auto acquireIf = scf::IfOp::create(builder, ifOp.getLoc(),
                                       TypeRange{builder.getType<AsyncTokenType>()},
                                       ifOp.getCondition(),
                                       /*withElseRegion=*/true);
    SmallVector<SetVector<int>> outputs{ownerSet(getPartitionId(releaseOp))};
    copyIfPartitionShape(acquireIf, ifOp, outputs);
    builder.setInsertionPointToEnd(acquireIf.thenBlock());
    Value acquired = createAcquireAt(builder, ifOp.getLoc(),
                                     getStageCluster(releaseOp), empty,
                                     getPartitionId(releaseOp));
    scf::YieldOp::create(builder, ifOp.getLoc(), acquired);
    builder.setInsertionPointToEnd(acquireIf.elseBlock());
    scf::YieldOp::create(builder, ifOp.getLoc(), oldElseToken);

    ifOp.getResult(*tokenIdx).replaceAllUsesWith(acquireIf.getResult(0));
  }
}

void addGuardedLoopClose(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp> loops;
  funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName) &&
        forOp->getParentOfType<scf::IfOp>() && forOp.getNumResults() == 1 &&
        isa<AsyncTokenType>(forOp.getResult(0).getType()))
      loops.push_back(forOp);
  });

  for (scf::ForOp forOp : loops) {
    SemaphoreReleaseOp producerRelease;
    SemaphoreAcquireOp consumerAcquire;
    forOp.getBody()->walk([&](Operation *op) {
      if (!producerRelease)
        producerRelease = dyn_cast<SemaphoreReleaseOp>(op);
      if (!consumerAcquire) {
        auto acquire = dyn_cast<SemaphoreAcquireOp>(op);
        if (acquire && acquire.getSemaphore() ==
                           (producerRelease ? producerRelease.getSemaphore()
                                            : acquire.getSemaphore()))
          consumerAcquire = acquire;
      }
    });
    if (!producerRelease)
      continue;
    Value full = producerRelease.getSemaphore();
    Value empty = findReleasedPeerSemaphore(full);
    if (!empty)
      continue;
    if (!consumerAcquire) {
      forOp.getBody()->walk([&](SemaphoreAcquireOp acquire) {
        if (!consumerAcquire && acquire.getSemaphore() == full)
          consumerAcquire = acquire;
      });
    }
    if (!consumerAcquire)
      continue;

    OpBuilder builder(forOp);
    builder.setInsertionPointAfter(forOp);
    auto asyncAttr = builder.getArrayAttr(SmallVector<Attribute>{
        AsyncOpAttr::get(builder.getContext(), getAsyncPayload(producerRelease))});
    auto closeRelease = createInto<SemaphoreReleaseOp>(
        builder, forOp.getLoc(), {getPartitionId(producerRelease),
                                  getStageCluster(forOp.getOperation())},
        full, forOp.getResult(0), asyncAttr);
    builder.setInsertionPointAfter(closeRelease);
    Value token = createAcquireAt(builder, forOp.getLoc(),
                                  getStageCluster(consumerAcquire), full,
                                  getPartitionId(consumerAcquire));
    createReleaseAt(builder, forOp.getLoc(), getStageCluster(consumerAcquire),
                    empty, token, getPartitionId(consumerAcquire),
                    AsyncOp::NONE);
  }
}

// v4 §Uniform Access-DAG Builder: one discovery pass that walks both
// ttng.tmem_alloc and ttg.local_alloc and produces a single
// SmallVector<BufferGroup> covering all backing buffers, irrespective of
// memory space. Memory space is a property of the resulting BufferGroup,
// not a separate code path.
SmallVector<BufferGroup, 0> collectAllBackingGroups(triton::FuncOp funcOp) {
  SmallVector<BufferGroup, 0> groups;
  int64_t nextSyntheticId = 0;

  // TMEM groups: many TMEMAllocOps may share the same buffer.id (members of
  // one logical group with different members/offsets that may overlap).
  llvm::MapVector<int64_t, SmallVector<TMEMAllocOp>> tmemBuckets;
  DenseMap<Operation *, int64_t> tmemSyntheticIds;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp))
      return;
    std::optional<int64_t> bufferId = getBufferId(allocOp);
    if (!bufferId) {
      tmemSyntheticIds[allocOp.getOperation()] = nextSyntheticId++;
      bufferId = tmemSyntheticIds[allocOp.getOperation()];
    }
    tmemBuckets[*bufferId].push_back(allocOp);
  });
  for (auto &[bufferId, allocs] : tmemBuckets)
    groups.push_back(makeTmemGroup(bufferId, allocs));

  // Local/SMEM groups: each LocalAllocOp is its own backing buffer; the
  // buffer.id (if present) is informational, not a grouping key.
  funcOp.walk([&](LocalAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp))
      return;
    if (!isLocalSemaphoreBackingType(cast<MemDescType>(allocOp.getType())))
      return;
    int64_t id = getBufferId(allocOp).value_or(nextSyntheticId++);
    groups.push_back(makeLocalGroup(id, allocOp));
  });

  return groups;
}

// v4 §Final Combine Subpass: pipeline runs phase-by-phase across all
// backing groups uniformly. Each group walks the same five DAGs:
//   access-dag → ownership-dag → raw-sync-dag → opt-sync-dag → semaphores.
// Memory-space-specific work lives inside the helpers (collectEvents,
// materializeGroup, retargetEvent, ...), not in the orchestration.
LogicalResult runOnFunction(triton::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  // Phase 1: discovery — uniform over TMEM + local.
  SmallVector<BufferGroup, 0> groups = collectAllBackingGroups(funcOp);

  // Phase 2: access-dag — per-member touches in program order.
  for (BufferGroup &group : groups)
    if (failed(collectEvents(group, funcOp)))
      return failure();

  // Phase 3: ownership-dag — pure planRegion/planIf/planFor per resourceKey.
  SmallVector<GroupOwnershipPlan, 0> ownershipPlans;
  ownershipPlans.reserve(groups.size());
  for (BufferGroup &group : groups)
    ownershipPlans.push_back(buildOwnershipPlan(group, funcOp));

  // Phases 4..5 (raw-sync-dag, opt-sync-dag) are currently subsumed inside
  // emitSemaphores during v4 bring-up. They will become explicit phases
  // producing SyncEdgeInfo / SyncGroupInfo records before any
  // nvws.semaphore.* op is emitted.

  // Phase 6: debug dump (NVWS_INSERT_SEMA_DUMP_DAG=1).
  for (BufferGroup &group : groups)
    dumpGroupDag(group);

  // Phase 7: emit semaphores from the planned graph.
  for (BufferGroup &group : groups)
    if (failed(emitSemaphores(group)))
      return failure();

  // Phase 9: legacy IR cleanup permitted by v4 §Final Combine Subpass.
  splitConditionalMultiResultTokenIf(funcOp);
  addGuardedLoopClose(funcOp);

  // Phase 10: erase dead alloc/alias ops.
  for (BufferGroup &group : groups)
    eraseDeadGroupOps(group);

  return success();
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<
          NVWSInsertSemas> {
public:
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
