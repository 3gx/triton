#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace {

struct WarpGroupBuilder : public OpBuilder {
  WarpGroupBuilder(Block *block, Block::iterator insertPoint,
                   size_t partitionId)
      : OpBuilder(block, insertPoint), partitionId(partitionId) {}

  IRMapping mapping;
  size_t partitionId;
};

// This is computed per loop and partition
enum class LoopVarCategory {
  // The given loop variable is not used by the given partition. For example,
  // the use-D flag for MMA is only used by the MMA partition, and thus
  // is `Unused` for any other partition.
  Unused,
  // The given loop variable is used by the given partition. For example, a loop
  // index might be used to compute a relevant stage or phase value for the
  // given partition.
  Used,
  // The results of warp_group op are defined to be those of the first
  // partition. If the original loop results include a tensor which is computed
  // only by a non-default partition, such tensor cannot be returned from the
  // first partition and and must be passed through shared memory. The
  // corresponding loop variable falls into this category.
  // Recognizing this category is necessary for the first partition. For other
  // partitions, some loop variables might be assigned this category, but that
  // information is not used.
  TensorResultFromOtherPartition,
};

SetVector<int> getResultPartitionIds(Operation *op, int index) {
  return getPartitionOutputs(op)[index];
}

SetVector<int> getIfOpResultPartitionIds(scf::IfOp ifOp, Value value) {
  for (auto result : ifOp.getResults()) {
    if (result == value) {
      auto pos = result.getResultNumber();
      return getResultPartitionIds(ifOp, pos);
    }
  }
  llvm_unreachable("value is not a result of if-stmt");
}

bool isTensorResultComputedBy(scf::ForOp loop, size_t resultIdx,
                              const Partition *partition,
                              const PartitionSet &partitions) {
  auto value = loop.getYieldedValues()[resultIdx];
  if (!isa<RankedTensorType>(value.getType()))
    return false;
  auto defOp = value.getDefiningOp();
  auto partitionIds = getPartitionIds(defOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    partitionIds = getIfOpResultPartitionIds(ifOp, value);
  }
  return llvm::is_contained(partitionIds, partition->getIndex());
}

SmallVector<LoopVarCategory> classifyLoopVars(scf::ForOp loop,
                                              const Partition *partition,
                                              const PartitionSet &partitions) {
  auto isTensorResultFromOtherPartition = [&](int i) {
    for (auto otherPartition : partitions.getPartitions()) {
      if (&otherPartition == partition) {
        continue;
      }
      if (isTensorResultComputedBy(loop, i, &otherPartition, partitions)) {
        return true;
      }
    }
    return false;
  };

  SmallVector<LoopVarCategory> categories(loop.getNumRegionIterArgs());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    auto partitionIds = getResultPartitionIds(loop, i);
    if (llvm::is_contained(partitionIds, partition->getIndex())) {
      categories[i] = LoopVarCategory::Used;
    } else if (isTensorResultFromOtherPartition(i) &&
               !loop.getResult(i).use_empty()) {
      categories[i] = LoopVarCategory::TensorResultFromOtherPartition;
    } else {
      categories[i] = LoopVarCategory::Unused;
    }
  }

  return categories;
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        ArrayRef<LoopVarCategory> loopVarCategories) {
  SmallVector<size_t> indices;
  // The null index means an invalid index, the corresponding loop variable in
  // the original loop is removed in the cloned loop
  SmallVector<std::optional<size_t>> reverseIndices(loop.getNumRegionIterArgs(),
                                                    std::nullopt);
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (loopVarCategories[i] == LoopVarCategory::Used) {
      reverseIndices[i] = indices.size();
      indices.push_back(i);
    }
  }
  return std::make_pair(indices, reverseIndices);
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        const PartitionSet &partitions) {
  auto loopVarCategories = classifyLoopVars(loop, partition, partitions);
  return getLoopVarIndicesToKeep(loop, partition, loopVarCategories);
}

void mapRange(ValueRange fromRange, ValueRange toRange, IRMapping &mapping) {
  for (auto [from, to] : llvm::zip(fromRange, toRange)) {
    mapping.map(from, to);
  }
}

bool isDefinedInBlockTree(Value value, Block *rootBlock) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    if (owner == rootBlock)
      return true;
    if (Operation *parentOp = owner->getParentOp())
      return rootBlock->findAncestorOpInBlock(*parentOp) != nullptr;
    return false;
  }
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  return defOp->getBlock() == rootBlock ||
         rootBlock->findAncestorOpInBlock(*defOp) != nullptr;
}

Value mapValueForPartition(WarpGroupBuilder &builder, Value value,
                           Block *rootBlock, Location loc) {
  if (builder.mapping.contains(value))
    return builder.mapping.lookup(value);
  if (!isDefinedInBlockTree(value, rootBlock))
    return value;
  return ub::PoisonOp::create(builder, loc, value.getType());
}

void cloneOpsInBlock(Block *block, Block *rootBlock,
                     SmallVector<WarpGroupBuilder> &builders,
                     const PartitionSet &partitions);

void cloneForOp(scf::ForOp forOp, Block *rootBlock,
                SmallVector<WarpGroupBuilder> &builders,
                const PartitionSet &partitions) {
  auto forOpPartitions = getPartitionIds(forOp);

  SmallVector<scf::ForOp> newForOps;
  for (int i : forOpPartitions) {
    auto &b = builders[i];
    auto partition = partitions.getPartition(i);
    auto [newLoopIndices, _] =
        getLoopVarIndicesToKeep(forOp, partition, partitions);
    auto lb = mapValueForPartition(b, forOp.getLowerBound(), rootBlock,
                                   forOp.getLoc());
    auto ub = mapValueForPartition(b, forOp.getUpperBound(), rootBlock,
                                   forOp.getLoc());
    auto step = mapValueForPartition(b, forOp.getStep(), rootBlock,
                                     forOp.getLoc());
    SmallVector<Value> initArgs;
    for (auto idx : newLoopIndices) {
      initArgs.push_back(mapValueForPartition(b, forOp.getInitArgs()[idx],
                                              rootBlock, forOp.getLoc()));
    }
    auto newForOp =
        scf::ForOp::create(b, forOp.getLoc(), lb, ub, step, initArgs);
    newForOp->setAttrs(forOp->getAttrs());
    if (forOp->hasAttr(kPartitionOutputsAttrName)) {
      newForOp->removeAttr(kPartitionOutputsAttrName);
    }
    newForOps.push_back(newForOp);

    b.mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    auto oldIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = newForOp.getRegionIterArgs();
    for (auto [newIdx, oldIdx] : llvm::enumerate(newLoopIndices)) {
      b.mapping.map(oldIterArgs[oldIdx], newIterArgs[newIdx]);
      b.mapping.map(forOp.getResult(oldIdx), newForOp.getResult(newIdx));
    }

    b.setInsertionPointToStart(newForOp.getBody());
  }

  cloneOpsInBlock(forOp.getBody(), rootBlock, builders, partitions);

  for (auto [i, newForOp] : llvm::zip(forOpPartitions, newForOps)) {
    builders[i].setInsertionPointAfter(newForOp);
    newForOp.walk([&](Operation *op) { op->removeAttr(kPartitionAttrName); });
    newForOp->removeAttr(kPartitionStagesAttrName);
  }
}

void cloneIfOp(scf::IfOp ifOp, Block *rootBlock,
               SmallVector<WarpGroupBuilder> &builders,
               const PartitionSet &partitions) {
  auto partitionIndices = getPartitionIds(ifOp);

  SmallVector<scf::IfOp> newIfOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];
    auto cond =
        mapValueForPartition(b, ifOp.getCondition(), rootBlock, ifOp.getLoc());
    SmallVector<Type> newIfResultTypes;
    SmallVector<int> newIfResultIndices;
    for (auto pos = 0; pos < ifOp.getResultTypes().size(); ++pos) {
      auto partitionIds = getResultPartitionIds(ifOp, pos);
      if (llvm::is_contained(partitionIds, b.partitionId)) {
        newIfResultTypes.push_back(ifOp.getResult(pos).getType());
        newIfResultIndices.push_back(pos);
      }
    }
    auto newIfOp = scf::IfOp::create(b, ifOp.getLoc(), newIfResultTypes, cond,
                                     ifOp.elseBlock() ? true : false);
    newIfOp->setAttrs(ifOp->getAttrs());
    if (ifOp->hasAttr(kPartitionOutputsAttrName)) {
      newIfOp->removeAttr(kPartitionOutputsAttrName);
    }
    newIfOps.push_back(newIfOp);

    for (auto [newIdx, oldIdx] : llvm::enumerate(newIfResultIndices)) {
      b.mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
    }
    assert(ifOp.thenBlock()->getNumArguments() == 0);

    b.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), rootBlock, builders, partitions);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [idx, newIfOp] : llvm::zip(partitionIndices, newIfOps)) {
      builders[idx].setInsertionPointToStart(newIfOp.elseBlock());
    }
    cloneOpsInBlock(elseBlock, rootBlock, builders, partitions);
  }

  for (auto [idx, newIfOp] : llvm::zip(partitionIndices, newIfOps)) {
    builders[idx].setInsertionPointAfter(newIfOp);
  }
}

void cloneReduceOp(triton::ReduceOp reduceOp, Block *rootBlock,
                   SmallVector<WarpGroupBuilder> &builders,
                   const PartitionSet &partitions) {
  auto partitionIndices = getPartitionIds(reduceOp);

  SmallVector<ReduceOp> newReduceOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];

    SmallVector<Value> srcs;
    for (auto src : reduceOp.getSrcs()) {
      srcs.push_back(
          mapValueForPartition(b, src, rootBlock, reduceOp.getLoc()));
    }
    auto axis = reduceOp.getAxis();
    auto newReduceOp =
        triton::ReduceOp::create(b, reduceOp.getLoc(), srcs, axis);
    newReduceOp->setAttrs(reduceOp->getAttrs());
    if (reduceOp->hasAttr(kPartitionOutputsAttrName)) {
      newReduceOp->removeAttr(kPartitionOutputsAttrName);
    }
    newReduceOps.push_back(newReduceOp);

    mapRange(reduceOp.getResults(), newReduceOp.getResults(), b.mapping);

    auto &region = newReduceOp.getRegion();
    Block *block = &region.emplaceBlock();
    for (auto arg : reduceOp.getRegion().getBlocks().front().getArguments()) {
      auto newArg = block->addArgument(arg.getType(), arg.getLoc());
      b.mapping.map(arg, newArg);
    }

    b.setInsertionPointToStart(block);
  }

  cloneOpsInBlock(reduceOp.getBody(), rootBlock, builders, partitions);

  for (auto [idx, newReduceOp] : llvm::zip(partitionIndices, newReduceOps)) {
    builders[idx].setInsertionPointAfter(newReduceOp);
  }
}

void cloneOp(Operation *op, Block *rootBlock,
             SmallVector<WarpGroupBuilder> &builders,
             const SetVector<int> &partitionIndices) {
  if (op->getNumRegions() != 0) {
    llvm::report_fatal_error(
        "Ops are expected to be regionless at this point.");
  }

  for (size_t idx : partitionIndices) {
    auto &builder = builders[idx];
    auto newOp = builder.clone(*op, builder.mapping);
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(newOp);
      for (auto [operandIdx, oldOperand] : llvm::enumerate(op->getOperands())) {
        auto mappedValue = mapValueForPartition(builder, oldOperand, rootBlock,
                                                op->getLoc());
        newOp->setOperand(operandIdx, mappedValue);
      }
    }
    mapRange(op->getResults(), newOp->getResults(), builder.mapping);
  }
}

void cloneOpsInBlock(Block *block, Block *rootBlock,
                     SmallVector<WarpGroupBuilder> &builders,
                     const PartitionSet &partitions) {
  for (auto &op_ : *block) {
    auto op = &op_;

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cloneForOp(forOp, rootBlock, builders, partitions);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      cloneIfOp(ifOp, rootBlock, builders, partitions);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      cloneReduceOp(reduceOp, rootBlock, builders, partitions);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (yieldOp.getOperands().empty()) {
        continue;
      }
      // empty yield has no partition annotations
      assert(hasPartition(op));
      auto partitionIndices = getPartitionIds(op);

      for (size_t idx : partitionIndices) {
        auto &builder = builders[idx];
        SmallVector<size_t> newOperandIndices;
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          newOperandIndices =
              getLoopVarIndicesToKeep(
                  forOp, partitions.getPartition(builder.partitionId),
                  partitions)
                  .first;
        } else {
          auto ifOp = cast<scf::IfOp>(yieldOp->getParentOp());
          for (size_t i = 0; i < yieldOp.getOperands().size(); ++i) {
            auto ids = getResultPartitionIds(ifOp, i);
            if (llvm::is_contained(ids, builder.partitionId)) {
              newOperandIndices.push_back(i);
            }
          }
        }

        if (newOperandIndices.empty())
          continue;

        SmallVector<Value> newYieldOperands;
        for (size_t i : newOperandIndices) {
          Value oldYieldOperand = yieldOp.getOperand(i);
          if (builder.mapping.contains(oldYieldOperand)) {
            newYieldOperands.push_back(builder.mapping.lookup(oldYieldOperand));
            continue;
          }
          if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
            Value iterArg = forOp.getRegionIterArgs()[i];
            newYieldOperands.push_back(mapValueForPartition(
                builder, iterArg, rootBlock, yieldOp.getLoc()));
            continue;
          }
          newYieldOperands.push_back(mapValueForPartition(
              builder, oldYieldOperand, rootBlock, yieldOp.getLoc()));
        }

        scf::YieldOp::create(builder, op->getLoc(), newYieldOperands);
      }
    } else {
      assert(hasPartition(op));
      auto partitionIndices = getPartitionIds(op);
      cloneOp(op, rootBlock, builders, partitionIndices);
    }
  }
}

} // namespace

LogicalResult triton::gpu::partitionLoop(scf::ForOp loop) {
  FailureOr<PartitionSet> partitionsOr = PartitionSet::fromLoop(loop);
  if (failed(partitionsOr))
    return failure();
  PartitionSet partitions = std::move(*partitionsOr);

  // Only the root node should have consumers at this point.
  for (const Partition &partition : partitions.getPartitions()) {
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      auto partitionIds = getPartitionIds(use.getOwner());
      if (llvm::is_contained(partitionIds, partition.getIndex()))
        return;

      // check if consumer partition set is a subset of the producer partitions
      auto defOpPartitionIds = getPartitionIds(output.getDefiningOp());
      bool isValidSubset = std::all_of(
          partitionIds.begin(), partitionIds.end(), [&](int consumerId) {
            return llvm::is_contained(defOpPartitionIds, consumerId);
          });

      if (isValidSubset)
        return; // Valid: consumer ⊆ producer

      failed = true;
      InFlightDiagnostic diag =
          mlir::emitWarning(output.getLoc(), "non-root partition #")
          << partition.getIndex() << " has direct SSA consumer";

      for (auto partitionId : partitionIds) {
        diag.attachNote(use.getOwner()->getLoc())
            << "use at distance " << distance << " in partition #"
            << partitionId << " here";
      }
    };
    partition.iterateUses(loop, callback);
    if (failed)
      return failure();
  }

  // There is nothing to do if the loop has 1 or fewer partitions.
  if (llvm::size(partitions.getPartitions()) <= 1)
    return success();

  auto numPartitions = partitions.getNumPartitions();
  auto defaultPartition = partitions.getPartition((int)0);
  auto loopVarCategories = classifyLoopVars(loop, defaultPartition, partitions);
  auto [loopVarIndices, newResultIndices] =
      getLoopVarIndicesToKeep(loop, defaultPartition, loopVarCategories);

  ImplicitLocOpBuilder topBuilder(loop.getLoc(), loop);
  SmallVector<Value> tensorResultAllocs(loop.getNumRegionIterArgs());
  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (loopVarCategories[i] ==
        LoopVarCategory::TensorResultFromOtherPartition) {
      auto ty = cast<RankedTensorType>(res.getType());
      auto memdesc = MemDescType::get(
          ty.getShape(), ty.getElementType(), getSharedEncoding(ty),
          SharedMemorySpaceAttr::get(ty.getContext()), /*mutable=*/true);
      tensorResultAllocs[i] = LocalAllocOp::create(topBuilder, memdesc);
    }
  }

  SmallVector<Type> resultTypes;
  for (auto i : loopVarIndices) {
    resultTypes.push_back(loop.getResultTypes()[i]);
  }

  SmallVector<int32_t> numWarps(numPartitions, lookupNumWarps(loop));
  auto wgOp = nvws::WarpGroupOp::create(topBuilder, resultTypes, numWarps,
                                        numPartitions);

  SmallVector<WarpGroupBuilder> builders;
  for (Region &region : wgOp.getPartitionRegions()) {
    auto partitionId = builders.size();
    auto &block = region.emplaceBlock();
    builders.push_back(WarpGroupBuilder(&block, block.end(), partitionId));
  }

  SmallVector<Operation *> opsToErase;
  for (auto &op_ : *loop->getBlock()) {
    auto op = &op_;
    if (!hasPartition(op))
      continue;
    auto wsTag = getWarpSpecializeTag(op);
    if (wsTag && *wsTag != partitions.getTag())
      continue;
    if (op == loop) {
      cloneForOp(loop, loop.getBody(), builders, partitions);
      opsToErase.push_back(loop);
    } else {
      cloneOp(op, loop->getBlock(), builders, getPartitionIds(op));
      opsToErase.push_back(op);
    }
  }

  for (auto [b, region, partition] : llvm::zip(
           builders, wgOp.getPartitionRegions(), partitions.getPartitions())) {
    if (!llvm::is_contained(getPartitionIds(loop), b.partitionId)) {
      nvws::WarpGroupYieldOp::create(b, wgOp.getLoc(), SmallVector<Value>{});
      continue;
    }
    auto newForOp = *region.front().getOps<scf::ForOp>().begin();
    auto outputs = newForOp.getResults();

    if (b.partitionId == 0) {
      nvws::WarpGroupYieldOp::create(b, wgOp.getLoc(), outputs);
    } else {
      // Tensor results computed by non-default partitions are communicated back
      // via SMEM.
      // The calls to getLoopVarIndicesToKeep and isTensorResultComputedBy
      // below are unnecessary if we can encode the partition index and the
      // corresponding result tensor index of newForOp in
      // LoopVarCategory::TensorResultFromOtherPartition. In the absence of such
      // language support, we end up computing the same information multiple
      // times.
      auto [_, reverseIndices] =
          getLoopVarIndicesToKeep(loop, &partition, partitions);
      for (size_t i = 0; i < loop.getNumRegionIterArgs(); ++i) {
        if (loopVarCategories[i] ==
                LoopVarCategory::TensorResultFromOtherPartition &&
            isTensorResultComputedBy(loop, i, &partition, partitions)) {
          assert(reverseIndices[i] && "A valid index is expected.");
          auto result = newForOp.getResult(*reverseIndices[i]);
          LocalStoreOp::create(b, wgOp.getLoc(), result, tensorResultAllocs[i]);
        }
      }
      nvws::WarpGroupReturnOp::create(b, wgOp.getLoc());
    }
  }

  topBuilder.setInsertionPointAfter(wgOp);

  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (res.use_empty())
      continue;

    if (loopVarCategories[i] ==
        LoopVarCategory::TensorResultFromOtherPartition) {
      auto ty = cast<RankedTensorType>(loop.getResult(i).getType());
      auto output = LocalLoadOp::create(topBuilder, ty, tensorResultAllocs[i]);
      LocalDeallocOp::create(topBuilder, tensorResultAllocs[i]);
      res.replaceAllUsesWith(output);
    } else if (llvm::any_of(res.getUsers(), [&](Operation *user) {
                 return !hasPartition(user) ||
                        (isa<scf::ForOp>(user) && hasWarpSpecializeTag(user));
               })) {
      // If some users are in the root partition (no partition attribute) or
      // used by another warp-specialized loop, we need to replace their uses
      // with the corresponding result from the warp group operation
      assert(newResultIndices[i] && "A valid index is expected.");
      res.replaceAllUsesWith(wgOp.getResult(*newResultIndices[i]));
    }
  }

  for (auto op : llvm::reverse(opsToErase)) {
    // Partitioned scalar defs can be shared with non-partitioned users in the
    // root block. Keep those producers alive if uses remain after cloning.
    if (!op->use_empty())
      continue;
    op->erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONLOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionLoops
    : triton::gpu::impl::TritonGPUPartitionLoopsBase<PartitionLoops> {
  using TritonGPUPartitionLoopsBase::TritonGPUPartitionLoopsBase;

  void runOnOperation() override;
};
} // namespace

void PartitionLoops::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be annotated with partitions.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(partitionLoop(loop)))
      return signalPassFailure();
  }
}
