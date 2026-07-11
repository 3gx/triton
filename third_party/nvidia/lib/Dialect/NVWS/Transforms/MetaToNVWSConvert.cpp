#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSMETATONVWSCONVERT
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isWarpSpecializeLoop(Operation *op) {
  auto loop = dyn_cast<scf::ForOp>(op);
  return loop && loop->hasAttr(kWarpSpecializeAttrName);
}

static bool isNestedInWarpSpecializeLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isWarpSpecializeLoop(parent))
      return true;
  }
  return false;
}

static FailureOr<SmallVector<int>> getNormalizedTaskIds(Operation *op) {
  auto taskIds = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id");
  if (!taskIds)
    return failure();

  SmallVector<int> ids(taskIds.asArrayRef().begin(),
                       taskIds.asArrayRef().end());
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.empty() || llvm::any_of(ids, [](int id) { return id < 0; }))
    return failure();
  return ids;
}

static LogicalResult materializePartition(Operation *op) {
  FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(op);
  if (failed(ids))
    return op->emitError(
        "MetaToNVWSConvert requires a non-empty, non-negative "
        "async_task_id assignment");

  auto expected = DenseI32ArrayAttr::get(op->getContext(), *ids);
  if (Attribute existing = op->getAttr(kPartitionAttrName)) {
    if (existing != expected)
      return op->emitError(
          "MetaToNVWSConvert found conflicting async_task_id and "
          "ttg.partition assignments");
  }
  op->setAttr(kPartitionAttrName, expected);
  return success();
}

static void insertAll(SetVector<int> &dst, const SetVector<int> &src) {
  dst.insert(src.begin(), src.end());
}

static void collectYieldedValues(Operation *op, unsigned resultIndex,
                                 SmallVectorImpl<Value> &values) {
  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    Operation *terminator = loop.getBody()->getTerminator();
    if (resultIndex < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIndex));
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      Operation *terminator = region->front().getTerminator();
      if (resultIndex < terminator->getNumOperands())
        values.push_back(terminator->getOperand(resultIndex));
    }
    return;
  }

  if (auto reduce = dyn_cast<ReduceOp>(op)) {
    Region &region = reduce.getCombineOp();
    if (region.empty())
      return;
    Operation *terminator = region.front().getTerminator();
    if (resultIndex < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIndex));
  }
}

static void insertPartitionForUser(SetVector<int> &result, Operation *user,
                                   Block *scope = nullptr) {
  if (hasPartition(user)) {
    insertAll(result, getPartitionIds(user));
    return;
  }
  if (!scope)
    return;
  if (Operation *ancestor = scope->findAncestorOpInBlock(*user)) {
    if (hasPartition(ancestor))
      insertAll(result, getPartitionIds(ancestor));
  }
}

static SetVector<int> inferAsyncTokenResultPartition(Operation *op,
                                                     unsigned resultIndex) {
  SetVector<int> result;
  if (resultIndex >= op->getNumResults() ||
      !isa<AsyncTokenType>(op->getResult(resultIndex).getType()))
    return result;

  if (auto loop = dyn_cast<scf::ForOp>(op)) {
    if (resultIndex >= loop.getNumRegionIterArgs())
      return result;
    BlockArgument arg = loop.getRegionIterArg(resultIndex);
    for (OpOperand &use : arg.getUses())
      insertPartitionForUser(result, use.getOwner(), loop.getBody());
    return result;
  }

  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIndex, yieldedValues);
  for (Value value : yieldedValues) {
    if (Operation *def = value.getDefiningOp())
      insertPartitionForUser(result, def);
  }
  return result;
}

static void insertYieldedProducerPartitions(
    SetVector<int> &result, Value value,
    DenseSet<Operation *> &consumedExternalTaskIds) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return;

  if (def->getNumRegions() != 0 && def->hasAttr(kPartitionOutputsAttrName)) {
    auto it = llvm::find(def->getResults(), value);
    if (it != def->result_end()) {
      unsigned index = std::distance(def->result_begin(), it);
      auto outputs = getPartitionOutputs(def);
      if (index < outputs.size()) {
        insertAll(result, outputs[index]);
        return;
      }
    }
  }

  if (hasPartition(def))
    insertAll(result, getPartitionIds(def));
  else if (FailureOr<SmallVector<int>> ids = getNormalizedTaskIds(def);
           succeeded(ids)) {
    result.insert(ids->begin(), ids->end());
    consumedExternalTaskIds.insert(def);
  }
}

static SetVector<int>
inferResultPartition(Operation *op, unsigned resultIndex,
                     DenseSet<Operation *> &consumedExternalTaskIds) {
  SetVector<int> result;
  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIndex, yieldedValues);
  for (Value value : yieldedValues)
    insertYieldedProducerPartitions(result, value, consumedExternalTaskIds);

  insertAll(result, inferAsyncTokenResultPartition(op, resultIndex));
  return result;
}

static LogicalResult materializePartitionOutputs(
    Operation *op, DenseSet<Operation *> &consumedExternalTaskIds) {
  if (!isa<scf::ForOp, scf::IfOp, ReduceOp>(op) || op->getNumResults() == 0)
    return success();

  SmallVector<SetVector<int>> outputs;
  outputs.reserve(op->getNumResults());
  SetVector<int> opPartitions = getPartitionIds(op);
  for (unsigned index = 0; index < op->getNumResults(); ++index) {
    SetVector<int> ids =
        inferResultPartition(op, index, consumedExternalTaskIds);
    if (ids.empty())
      return op->emitError(
          "MetaToNVWSConvert cannot determine partition.outputs for result ")
             << index;
    for (int id : ids) {
      if (!opPartitions.contains(id))
        return op->emitError("MetaToNVWSConvert result partition ")
               << id << " is absent from the Meta region assignment";
    }
    outputs.push_back(std::move(ids));
  }
  setPartitionOutputs(op, outputs);
  return success();
}

static int getTxCount(Operation *descOp) {
  RankedTensorType tensorType;
  Value desc;
  if (auto load = dyn_cast<DescriptorLoadOp>(descOp)) {
    tensorType = load.getType();
    desc = load.getDesc();
  } else {
    auto gather = cast<DescriptorGatherOp>(descOp);
    tensorType = gather.getType();
    desc = gather.getDesc();
  }
  Attribute encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         getIntOrFloatOrPtrBitWidth(tensorType.getElementType()) / 8;
}

static std::optional<int> getEnclosingWarpSpecializeTag(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isWarpSpecializeLoop(parent))
      return getWarpSpecializeTag(parent);
  }
  return std::nullopt;
}

static std::optional<int> getAssociatedWarpSpecializeTag(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isWarpSpecializeLoop(def))
    return getWarpSpecializeTag(def);
  if (auto tag = getWarpSpecializeTag(def))
    return tag;
  return getEnclosingWarpSpecializeTag(def);
}

static SetVector<int> inferAssociatedWarpSpecializeTags(Operation *op) {
  SetVector<int> tags;
  for (Value operand : op->getOperands())
    if (auto tag = getAssociatedWarpSpecializeTag(operand))
      tags.insert(*tag);
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (auto tag = getWarpSpecializeTag(user))
        tags.insert(*tag);
      if (auto tag = getEnclosingWarpSpecializeTag(user))
        tags.insert(*tag);
    }
  }
  return tags;
}

// PartitionLoops treats partitioned siblings of a WS loop as part of that
// loop's specialization unit. Materialize the tag association already implied
// by their def-use closure, exactly as the former NVWS scheduler finalizer did.
static LogicalResult tagExternalPartitionedOps(FuncOp func,
                                               ArrayRef<scf::ForOp> wsLoops) {
  SetVector<int> loopTags;
  for (scf::ForOp loop : wsLoops) {
    auto tag = getWarpSpecializeTag(loop);
    if (!tag)
      return loop.emitError(
          "MetaToNVWSConvert requires a tag on every Meta WS loop");
    loopTags.insert(*tag);
  }

  bool changed = false;
  do {
    changed = false;
    Operation *ambiguous = nullptr;
    SetVector<int> ambiguousTags;
    func.walk([&](Operation *op) {
      if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
          isWarpSpecializeLoop(op))
        return WalkResult::advance();

      SetVector<int> tags = inferAssociatedWarpSpecializeTags(op);
      if (tags.empty() && loopTags.size() == 1)
        tags.insert(loopTags.front());
      if (tags.size() > 1) {
        ambiguous = op;
        ambiguousTags = tags;
        return WalkResult::interrupt();
      }
      if (tags.size() == 1) {
        int inferred = tags.front();
        if (auto existing = getWarpSpecializeTag(op)) {
          if (*existing != inferred) {
            ambiguous = op;
            ambiguousTags.insert(*existing);
            ambiguousTags.insert(inferred);
            return WalkResult::interrupt();
          }
        } else {
          setWarpSpecializeTag(op, inferred);
          changed = true;
        }
      }
      return WalkResult::advance();
    });
    if (ambiguous)
      return ambiguous->emitError(
          "MetaToNVWSConvert found an ambiguous WS tag association for an "
          "external partitioned operation");
  } while (changed);

  Operation *untagged = nullptr;
  func.walk([&](Operation *op) {
    if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
        isWarpSpecializeLoop(op))
      return WalkResult::advance();
    if (!hasWarpSpecializeTag(op)) {
      untagged = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (untagged)
    return untagged->emitError(
        "MetaToNVWSConvert cannot associate external partitioned operation "
        "with a unique WS loop");
  return success();
}

static LogicalResult convertDescriptorStores(FuncOp func) {
  SmallVector<LocalStoreOp> stores;
  func.walk([&](LocalStoreOp store) {
    if (isa_and_nonnull<DescriptorLoadOp, DescriptorGatherOp>(
            store.getSrc().getDefiningOp()))
      stores.push_back(store);
  });

  for (LocalStoreOp store : stores) {
    Operation *descOp = store.getSrc().getDefiningOp();
    FailureOr<SmallVector<int>> descIds = getNormalizedTaskIds(descOp);
    if (failed(descIds))
      return descOp->emitError(
          "MetaToNVWSConvert descriptor producer is missing its Meta task "
          "assignment");
    SetVector<int> expectedPartitions;
    expectedPartitions.insert(descIds->begin(), descIds->end());
    SetVector<int> storePartitions;
    if (hasPartition(store)) {
      storePartitions = getPartitionIds(store);
    } else if (FailureOr<SmallVector<int>> storeIds =
                   getNormalizedTaskIds(store);
               succeeded(storeIds)) {
      storePartitions.insert(storeIds->begin(), storeIds->end());
    }
    if (storePartitions != expectedPartitions)
      return store.emitError(
          "MetaToNVWSConvert descriptor producer and planned local_store "
          "have different task assignments");

    OpBuilder builder(store);
    int txCount = getTxCount(descOp);
    Operation *newOp = nullptr;
    if (auto load = dyn_cast<DescriptorLoadOp>(descOp)) {
      newOp = triton::nvws::DescriptorLoadOp::create(
          builder, load.getLoc(), load.getDesc(), load.getIndices(), txCount,
          store.getDst(), load.getCache(), load.getEvict());
    } else {
      auto gather = cast<DescriptorGatherOp>(descOp);
      newOp = triton::nvws::DescriptorGatherOp::create(
          builder, gather.getLoc(), gather.getDesc(), gather.getXOffsets(),
          gather.getYOffset(), txCount, store.getDst());
    }
    newOp->setAttrs(descOp->getAttrs());
    newOp->setAttr(kPartitionAttrName,
                   DenseI32ArrayAttr::get(store.getContext(), *descIds));
    newOp->removeAttr("async_task_id");

    store.erase();
    if (descOp->use_empty())
      descOp->erase();
  }
  return success();
}

struct SmemPlanPolicy {
  int allocAlgo = 1;
  bool circularReuse = false;
};

static SmemPlanPolicy getEffectiveSmemPlanPolicy(FuncOp func) {
  SmemPlanPolicy policy;
  func.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr(kWarpSpecializeAttrName))
      return;
    SmallVector<scf::ForOp> loopChain{forOp};
    for (auto parent = forOp->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>())
      loopChain.push_back(parent);
    for (scf::ForOp loop : llvm::reverse(loopChain)) {
      if (auto attr =
              loop->getAttrOfType<IntegerAttr>("tt.smem_alloc_algo"))
        policy.allocAlgo = attr.getInt();
      if (auto attr =
              loop->getAttrOfType<BoolAttr>("tt.smem_circular_reuse"))
        policy.circularReuse = attr.getValue();
    }
  });
  return policy;
}

static bool sameClonePlan(LocalAllocOp lhs, LocalAllocOp rhs) {
  return lhs.getType() == rhs.getType() && !lhs.getSrc() && !rhs.getSrc() &&
         lhs->getAttr("buffer.copy") == rhs->getAttr("buffer.copy") &&
         lhs->getAttr("buffer.offset") == rhs->getAttr("buffer.offset") &&
         lhs->getAttr("allocation.reuseTarget") ==
             rhs->getAttr("allocation.reuseTarget");
}

// Meta buffer allocation can leave several source-free local_alloc ops for
// one copy-1 physical plan entry. CodePartition normally collapses these
// logical clones. The NVWS boundary must do the same before InsertSemas:
// same-partition groups intentionally need no semaphore, so InsertSemas has no
// reason to rewrite their backing on its own.
static void collapseCopyOneLocalClones(FuncOp func) {
  llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> groups;
  func.walk<WalkOrder::PreOrder>([&](LocalAllocOp alloc) {
    if (!alloc.isSharedMemoryAlloc())
      return;
    auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id");
    auto copy = alloc->getAttrOfType<IntegerAttr>("buffer.copy");
    if (id && copy && copy.getInt() == 1)
      groups[id.getInt()].push_back(alloc);
  });

  DenseSet<Operation *> erased;
  for (auto &[id, group] : groups) {
    (void)id;
    for (unsigned i = 0; i < group.size(); ++i) {
      LocalAllocOp representative = group[i];
      if (erased.contains(representative.getOperation()))
        continue;
      for (unsigned j = i + 1; j < group.size(); ++j) {
        LocalAllocOp clone = group[j];
        if (erased.contains(clone.getOperation()) ||
            !sameClonePlan(representative, clone) ||
            representative->getBlock() != clone->getBlock() ||
            !representative->isBeforeInBlock(clone))
          continue;
        clone.getResult().replaceAllUsesWith(representative.getResult());
        erased.insert(clone.getOperation());
        clone.erase();
      }
    }
  }
}

static unsigned computeMemDescBytes(MemDescType type) {
  int64_t numElements = 0;
  if (auto padded =
          dyn_cast<PaddedSharedEncodingAttr>(type.getEncoding())) {
    SmallVector<int64_t> unpaddedShape = getShapePerCTA(type);
    numElements = padded.getPaddedSize(unpaddedShape);
  } else {
    numElements = product<int64_t>(getAllocationShapePerCTA(type));
  }
  return static_cast<unsigned>(numElements * type.getElementTypeBitWidth() /
                               8);
}

static bool areReuseTypesCompatible(MemDescType host, MemDescType reuser) {
  return host.getEncoding() == reuser.getEncoding() &&
         host.getMemorySpace() == reuser.getMemorySpace() &&
         host.getElementType() == reuser.getElementType();
}

// Realize Meta's reuseTarget only when it is a sound physical alias. Newer
// Meta planners avoid producing incompatible candidates, but this guard keeps
// the conversion boundary correct for older canonical prefixes as well.
static LogicalResult realizeReuseTargets(FuncOp func) {
  DenseMap<int64_t, LocalAllocOp> hosts;
  llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> reusers;
  LogicalResult result = success();

  func.walk<WalkOrder::PreOrder>([&](LocalAllocOp alloc) {
    if (!alloc.isSharedMemoryAlloc())
      return;
    if (auto target =
            alloc->getAttrOfType<IntegerAttr>("allocation.reuseTarget")) {
      if (target.getInt() < 0) {
        result = alloc.emitError(
            "MetaToNVWSConvert requires allocation.reuseTarget to be a "
            "non-negative integer buffer id");
        return;
      }
      reusers[target.getInt()].push_back(alloc);
      return;
    }
    if (alloc->hasAttr("buffer.tmaStaging"))
      return;
    if (auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
      hosts.try_emplace(id.getInt(), alloc);
  });
  if (failed(result))
    return failure();

  auto i32 = IntegerType::get(func.getContext(), 32);
  for (auto &[targetId, candidates] : reusers) {
    auto hostIt = hosts.find(targetId);
    if (hostIt == hosts.end()) {
      for (LocalAllocOp candidate : candidates)
        candidate->removeAttr("allocation.reuseTarget");
      continue;
    }

    LocalAllocOp host = hostIt->second;
    auto hostType = cast<MemDescType>(host.getType());
    unsigned hostBytes = computeMemDescBytes(hostType);
    SmallVector<LocalAllocOp> viable;
    for (LocalAllocOp candidate : candidates) {
      auto candidateType = cast<MemDescType>(candidate.getType());
      bool dominates = host->getBlock() == candidate->getBlock() &&
                       host->isBeforeInBlock(candidate);
      if (!dominates || computeMemDescBytes(candidateType) > hostBytes ||
          !areReuseTypesCompatible(hostType, candidateType)) {
        // The proposal was consumed and rejected. Preserve the candidate's
        // original independent plan instead of falsely assigning targetId.
        candidate->removeAttr("allocation.reuseTarget");
        continue;
      }
      viable.push_back(candidate);
    }
    if (viable.empty())
      continue;

    OpBuilder builder(host);
    auto backing =
        LocalAllocOp::create(builder, host.getLoc(), hostType);
    backing->setAttrs(host->getAttrs());
    host.getResult().replaceAllUsesWith(backing.getResult());
    host.erase();
    hosts[targetId] = backing;

    Operation *insertAfter = backing.getOperation();
    for (LocalAllocOp candidate : viable) {
      auto candidateType = cast<MemDescType>(candidate.getType());
      builder.setInsertionPointAfter(insertAfter);
      auto view = MemDescReinterpretOp::create(
          builder, candidate.getLoc(), candidateType, backing.getResult());
      for (NamedAttribute attr : candidate->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name == "allocation.reuseTarget" ||
            name == "buffer.tmaStaging" || name == "buffer.id" ||
            name == "buffer.offset")
          continue;
        view->setAttr(attr.getName(), attr.getValue());
      }
      view->setAttr("buffer.id", IntegerAttr::get(i32, targetId));
      view->setAttr("buffer.offset", IntegerAttr::get(i32, 0));
      candidate.getResult().replaceAllUsesWith(view.getResult());
      candidate.erase();
      insertAfter = view.getOperation();
    }
  }
  return success();
}

static LogicalResult translateBufferPlan(FuncOp func) {
  SmemPlanPolicy policy = getEffectiveSmemPlanPolicy(func);

  if (policy.allocAlgo == 1 && !policy.circularReuse) {
    collapseCopyOneLocalClones(func);
  } else {
    // Algorithm 0 and explicit circular-reuse plans use repeated IDs as
    // distinct logical ring entries. Preserve the already selected ID/copy
    // policy and add only NVWS's representation attributes.
    llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> circularGroups;
    func.walk<WalkOrder::PreOrder>([&](LocalAllocOp alloc) {
      if (!alloc.isSharedMemoryAlloc() ||
          alloc->hasAttr("allocation.reuseTarget"))
        return;
      if (auto id = alloc->getAttrOfType<IntegerAttr>("buffer.id"))
        circularGroups[id.getInt()].push_back(alloc);
    });
    auto i32 = IntegerType::get(func.getContext(), 32);
    for (auto &[id, group] : circularGroups) {
      (void)id;
      if (group.size() < 2)
        continue;
      for (auto [start, alloc] : llvm::enumerate(group)) {
        alloc->setAttr("buffer.circular", UnitAttr::get(func.getContext()));
        alloc->setAttr("buffer.start",
                       IntegerAttr::get(i32, static_cast<int64_t>(start)));
      }
    }
  }

  return realizeReuseTargets(func);
}

static LogicalResult convertFunc(FuncOp func) {
  SmallVector<scf::ForOp> wsLoops;
  func.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      wsLoops.push_back(loop);
  });
  if (wsLoops.empty())
    return success();

  for (scf::ForOp loop : wsLoops) {
    if (!loop->hasAttr(kWarpSpecializeTagAttrName))
      return loop.emitError(
          "MetaToNVWSConvert requires the Meta warp-specialize tag");
  }

  WalkResult partitionResult = func.walk([&](Operation *op) {
    if (!isWarpSpecializeLoop(op) && !isNestedInWarpSpecializeLoop(op)) {
      op->removeAttr(kPartitionAttrName);
      op->removeAttr(kPartitionOutputsAttrName);
      op->removeAttr(kPartitionStagesAttrName);
      op->removeAttr(kWarpSpecializeTagAttrName);
      return WalkResult::advance();
    }
    if (isa<ub::PoisonOp>(op))
      return WalkResult::advance();
    if (failed(materializePartition(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (partitionResult.wasInterrupted())
    return failure();

  DenseSet<Operation *> consumedExternalTaskIds;
  WalkResult outputResult = func.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (!isWarpSpecializeLoop(op) && !isNestedInWarpSpecializeLoop(op))
      return WalkResult::advance();
    if (failed(
            materializePartitionOutputs(op, consumedExternalTaskIds)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (outputResult.wasInterrupted())
    return failure();

  if (failed(convertDescriptorStores(func)) ||
      failed(tagExternalPartitionedOps(func, wsLoops)) ||
      failed(translateBufferPlan(func)))
    return failure();

  // Remove each Meta ownership attribute that was consumed into NVWS
  // representation. Function-scope attributes not consulted by conversion
  // remain as unconsumed metadata.
  func.walk([&](Operation *op) {
    if (isWarpSpecializeLoop(op) || isNestedInWarpSpecializeLoop(op) ||
        consumedExternalTaskIds.contains(op))
      op->removeAttr("async_task_id");
  });
  return success();
}

class NVWSMetaToNVWSConvert
    : public mlir::triton::impl::NVWSMetaToNVWSConvertBase<
          NVWSMetaToNVWSConvert> {
public:
  using mlir::triton::impl::NVWSMetaToNVWSConvertBase<
      NVWSMetaToNVWSConvert>::NVWSMetaToNVWSConvertBase;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](FuncOp func) {
      if (failed(convertFunc(func))) {
        signalPassFailure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    (void)result;
  }
};

} // namespace
