#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/Sys/GetEnv.h"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace ttnvws = ::mlir::triton::nvws;
namespace mlir {

#define DEBUG_TYPE "nvgpu-ws-lower-mem"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Can a TMA copy land directly in a buffer with this encoding? Mirrors
// verifyTMAEncoding() in TritonNvidiaGPU/IR/Ops.cpp: the destination of an
// async_tma_copy_global_to_local must be NVMMA shared, untransposed, and (once
// the descriptor carries an encoding) agree with it. A consumer-owned buffer
// can legitimately fail this -- a block-scale load is consumed through a
// #ttg.shared_linear alloc that a memdesc_reshape/trans chain feeds to
// tmem_copy -- in which case the load needs its own NVMMA landing buffer.
static bool isValidTMADestEncoding(Attribute bufferEnc, Attribute descEnc) {
  auto nvmma = dyn_cast_or_null<ttg::NVMMASharedEncodingAttr>(bufferEnc);
  if (!nvmma || nvmma.getTransposed())
    return false;
  // No descriptor encoding yet: only the NVMMA requirement is checked.
  if (!descEnc)
    return true;
  auto descNvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(descEnc);
  // Encodings may differ in rank for rank-reducing loads, so compare fields.
  return descNvmma && descNvmma.getTransposed() == nvmma.getTransposed() &&
         descNvmma.getSwizzlingByteWidth() == nvmma.getSwizzlingByteWidth() &&
         descNvmma.getElementBitWidth() == nvmma.getElementBitWidth() &&
         descNvmma.getFp4Padded() == nvmma.getFp4Padded();
}

static LogicalResult
convertDescriptorLoadLikeToNVWS(tt::DescriptorLoadLikeOpInterface descOp) {
  Value result = descOp->getResult(0);
  auto tensorType = dyn_cast<RankedTensorType>(result.getType());
  if (!tensorType)
    return descOp->emitError(
        "expected a ranked tensor descriptor operation result");

  Value desc = descOp.getDesc();
  Attribute descEnc =
      cast<tt::TensorDescType>(desc.getType()).getSharedLayout();

  ttg::LocalStoreOp soleStore;
  ttg::LocalAllocOp soleAlloc;
  if (descOp->hasOneUse())
    soleStore = dyn_cast<ttg::LocalStoreOp>(*descOp->getUsers().begin());
  if (descOp->hasOneUse())
    soleAlloc = dyn_cast<ttg::LocalAllocOp>(*descOp->getUsers().begin());

  // Only reuse the consumer's buffer when the TMA copy can actually target
  // it; otherwise fall through to the register-consumed path below, which
  // allocates an NVMMA buffer and local_loads out of it (what the standard
  // descriptor lowering does for every operation).
  if (soleStore && !isValidTMADestEncoding(
                       soleStore.getDst().getType().getEncoding(), descEnc))
    soleStore = nullptr;
  if (soleAlloc &&
      !isValidTMADestEncoding(soleAlloc.getType().getEncoding(), descEnc))
    soleAlloc = nullptr;

  OpBuilderWithAsyncTaskIds builder(descOp);
  builder.setInsertionPoint(descOp);
  Value buffer;
  if (soleStore) {
    buffer = soleStore.getDst();
  } else if (soleAlloc) {
    auto oldType = cast<ttg::MemDescType>(soleAlloc.getType());
    auto bufferType = ttg::MemDescType::get(
        oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
        oldType.getMemorySpace(), /*mutableMemory=*/true);
    auto newAlloc = builder.createWithAsyncTaskIds<ttg::LocalAllocOp>(
        soleAlloc.getLoc(), bufferType);
    newAlloc->setAttrs(soleAlloc->getAttrs());
    triton::replaceUsesAndPropagateType(builder, soleAlloc,
                                        newAlloc.getResult());
    buffer = newAlloc.getResult();
    builder.setInsertionPointAfter(newAlloc);
  } else {
    auto encoding = ttng::getEncodingFromDescriptor(descOp, tensorType, desc);
    auto memorySpace = ttg::SharedMemorySpaceAttr::get(descOp.getContext());
    auto bufferType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), encoding,
        memorySpace, /*mutableMemory=*/true);
    buffer = builder
                 .createWithAsyncTaskIds<ttg::LocalAllocOp>(descOp.getLoc(),
                                                            bufferType)
                 .getResult();
  }

  if (Operation *bufferDef = buffer.getDefiningOp();
      bufferDef && bufferDef->getBlock() == descOp->getBlock() &&
      descOp->isBeforeInBlock(bufferDef))
    bufferDef->moveBefore(descOp);

  int64_t txCount =
      ttng::getDescriptorLoadBytes(cast<ttg::MemDescType>(buffer.getType()));
  Operation *nvwsOp = nullptr;
  if (auto load = dyn_cast<tt::DescriptorLoadOp>(descOp.getOperation())) {
    nvwsOp = builder.createWithAsyncTaskIds<ttnvws::DescriptorLoadOp>(
        load.getLoc(), load.getDesc(), load.getIndices(), txCount, buffer,
        load.getCache(), load.getEvict());
  } else {
    auto gather = cast<tt::DescriptorGatherOp>(descOp.getOperation());
    Value xOffsets = gather.getXOffsets();
    auto offsetsType = cast<RankedTensorType>(xOffsets.getType());
    if (offsetsType.getElementType().isInteger(16)) {
      xOffsets = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
          gather.getLoc(), offsetsType.clone(builder.getI32Type()), xOffsets);
    }
    nvwsOp = builder.createWithAsyncTaskIds<ttnvws::DescriptorGatherOp>(
        gather.getLoc(), gather.getDesc(), xOffsets, gather.getYOffset(),
        txCount, buffer);
  }
  nvwsOp->setAttrs(descOp->getAttrs());

  if (soleStore) {
    soleStore.erase();
  } else if (soleAlloc) {
    soleAlloc.erase();
  } else {
    // Register-consumed descriptor operation (e.g. softmax metadata). Tag the
    // local_load with the consumer partitions and their loop schedule, not the
    // producer's, so the NVWS descriptor operation's SMEM buffer is directly
    // the cross-partition channel rather than introducing a second buffer.
    builder.setAsyncTaskIdsFromValueUsers(result);
    builder.setLoopScheduleInfoFromOp(*descOp->getUsers().begin());
    auto localLoad = builder.createWithAsyncTaskIds<ttg::LocalLoadOp>(
        descOp.getLoc(), result.getType(), buffer);
    result.replaceAllUsesWith(localLoad.getResult());
  }
  descOp->erase();
  return success();
}

LogicalResult doConvertDescriptorLoadsToNVWS(triton::FuncOp funcOp) {
  SmallVector<tt::DescriptorLoadLikeOpInterface> loads;
  funcOp.walk(
      [&](tt::DescriptorLoadLikeOpInterface load) { loads.push_back(load); });

  for (tt::DescriptorLoadLikeOpInterface load : loads)
    if (failed(convertDescriptorLoadLikeToNVWS(load)))
      return failure();

  bool hasUnconvertedLoad = false;
  funcOp.walk([&](tt::DescriptorLoadLikeOpInterface load) {
    load->emitError("descriptor operation was not converted for AutoWS");
    hasUnconvertedLoad = true;
  });
  return failure(hasUnconvertedLoad);
}

#define GEN_PASS_DEF_NVGPUCONVERTDESCRIPTORLOADSTONVWS
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

struct NVGPUConvertDescriptorLoadsToNVWSPass
    : public impl::NVGPUConvertDescriptorLoadsToNVWSBase<
          NVGPUConvertDescriptorLoadsToNVWSPass> {
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(doConvertDescriptorLoadsToNVWS(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

Value createBufferView(OpBuilderWithAsyncTaskIds &builder, Value alloc,
                       Value idx) {
  assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
         "Expected MemDescType");
  auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape;
  assert(allocDescType.getShape().size() > 1 &&
         "Expected multi-dimensional memdesc (e.g., Nx...) for subview");
  shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
               allocDescType.getShape().end());
  auto viewDescType = triton::gpu::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory());
  return triton::gpu::MemDescIndexOp::create(builder, alloc.getLoc(),
                                             viewDescType, alloc, idx);
}

namespace {

Value getTMALoadBufferForStage(OpBuilderWithAsyncTaskIds &builder, Value buffer,
                               Value bufferIdx) {
  auto currentView = buffer.getDefiningOp<ttg::MemDescIndexOp>();
  if (!currentView)
    return buffer;
  return createBufferView(builder, currentView.getSrc(), bufferIdx);
}

Value getDescriptorLoadBuffer(ttnvws::DescriptorLoadOpInterface op) {
  if (auto load = dyn_cast<ttnvws::DescriptorLoadOp>(op.getOperation()))
    return load.getResult();
  return cast<ttnvws::DescriptorGatherOp>(op.getOperation()).getResult();
}

} // namespace

Operation *optimizeTMALoads(OpBuilderWithAsyncTaskIds &builder,
                            SmallVector<ttnvws::DescriptorLoadOpInterface>
                                &tmaLoads,
                            Value barrierAlloc, Value bufferIdx,
                            Value bufferIdxExtract, Value phase,
                            Operation *headProducer, Operation *headConsumer,
                            Operation *headConsumerSameLevel,
                            ArrayRef<int> additionalConsumerTaskIds,
                            DictionaryAttr consumerWaitConstraints) {
  auto loc = barrierAlloc.getLoc();

  // Compute the total size of the loads.
  int64_t sizeInBytes = 0;
  for (auto tmaLoad : tmaLoads)
    sizeInBytes += tmaLoad.getTxCount();

  // Create a barrier_expect with the appropriate size and insert it before the
  // first load.
  builder.setInsertionPoint(headProducer);
  builder.setAsyncTaskIdsFromOp(headProducer);
  builder.setLoopScheduleInfoFromOp(headProducer);
  auto prodBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdx);
  auto pred = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);
  builder.createWithAsyncTaskIds<ttng::BarrierExpectOp>(loc, prodBarrier,
                                                        sizeInBytes, pred);

  // Convert all the producers to async_tma_copy_global_to_local
  Operation *copy = nullptr;
  for (auto tmaLoad : tmaLoads) {
    builder.setInsertionPoint(tmaLoad);
    builder.setAsyncTaskIdsFromOp(tmaLoad);
    builder.setLoopScheduleInfoFromOp(tmaLoad);
    Value pipelineBuffer =
        getTMALoadBufferForStage(builder, getDescriptorLoadBuffer(tmaLoad),
                                 bufferIdx);
    if (auto load = dyn_cast<ttnvws::DescriptorLoadOp>(tmaLoad.getOperation())) {
      copy = builder.createWithAsyncTaskIds<ttng::AsyncTMACopyGlobalToLocalOp>(
          load.getLoc(), load.getDesc(), load.getIndices(), prodBarrier,
          pipelineBuffer, pred);
    } else {
      auto gather = cast<ttnvws::DescriptorGatherOp>(tmaLoad.getOperation());
      copy = builder.createWithAsyncTaskIds<ttng::AsyncTMAGatherOp>(
          gather.getLoc(), gather.getDesc(), gather.getXOffsets(),
          gather.getYOffset(), prodBarrier, pipelineBuffer, pred);
    }
  }

  // Create a wait_barrier before the first consumer.
  // For data-partitioned channels, shared ops (consBarrier, phase, pred)
  // need ALL consumer task IDs so they survive specializeRegion.
  builder.setInsertionPoint(headConsumerSameLevel);
  SmallVector<int> consumerTaskIds;
  for (int id : getAsyncTaskIds(headConsumer))
    consumerTaskIds.push_back(id);
  for (int id : additionalConsumerTaskIds)
    consumerTaskIds.push_back(id);
  builder.setAsynTaskIdsFromArray(consumerTaskIds);
  builder.setLoopScheduleInfoFromOp(headConsumerSameLevel);
  auto consBarrier =
      getBarrierForPipelineStage(builder, barrierAlloc, bufferIdxExtract);
  phase = builder.createWithAsyncTaskIds<arith::ExtUIOp>(
      loc, builder.getI32Type(), phase);
  Value waitPred =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 1);

  // Create one WaitBarrierOp per consumer task ID.
  builder.setAsyncTaskIdsFromOp(headConsumer);
  builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
      loc, consBarrier, phase, waitPred, /*deps=*/ValueRange{},
      consumerWaitConstraints);
  for (int extraTaskId : additionalConsumerTaskIds) {
    builder.setAsynTaskIdsFromArray({extraTaskId});
    builder.createWithAsyncTaskIds<ttng::WaitBarrierOp>(
        loc, consBarrier, phase, waitPred,
        /*deps=*/ValueRange{}, consumerWaitConstraints);
  }

  for (auto tmaLoad : tmaLoads)
    tmaLoad->erase();
  builder.clearLoopScheduleInfo();
  return copy;
}

} // namespace mlir
