/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

struct PartitionWsTagIds {
  std::optional<int> wsTag;
  SetVector<int> partitionIds;
};

std::optional<PartitionWsTagIds> getPartitionWsTagIds(Operation *op) {
  std::optional<PartitionWsTagIds> partitionWsTagIds;
  if (hasPartition(op)) {
    partitionWsTagIds =
        PartitionWsTagIds{std::nullopt, triton::gpu::getPartitionIds(op)};
    if (auto wsTag = getWarpSpecializeTag(op))
      partitionWsTagIds->wsTag = *wsTag;
  }
  return partitionWsTagIds;
}

void assignStageCluster(Operation *op,
                        std::optional<PartitionWsTagIds> partitionWsTagIds,
                        StageCluster stageCluster, OpBuilder &builder) {
  if (partitionWsTagIds) {
    setPartition(op, partitionWsTagIds->partitionIds);
    if (auto wsTag = partitionWsTagIds->wsTag)
      setWarpSpecializeTag(op, *wsTag);
    setStageCluster(builder, op, stageCluster);
  }
}

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs)
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  return kinds;
}

int getPendingCount(SemaphoreCreateOp op) {
  SetVector<int> releaseGroups;
  int count = 0;

  for (Operation *user : op->getUsers()) {
    if (!hasPartition(user))
      continue;
    auto partitionIds = getPartitionIds(user);
    assert(partitionIds.size() == 1);

    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;

    if (releaseGroups.count(partitionIds.front()))
      continue;
    releaseGroups.insert(partitionIds.front());

    for (auto kind : castAsyncOpAttrs(releaseOp.getAsyncOps())) {
      switch (kind) {
      case AsyncOp::TC5MMA:
      case AsyncOp::TMALoad:
      case AsyncOp::NONE:
      case AsyncOp::WGMMA:
      case AsyncOp::TMEMCopy:
        count += 1;
        break;
      default:
        llvm_unreachable("unsupported async op");
      }
    }
  }

  if (count == 0)
    count = 1;
  return count;
}

Value createAndInitMbar(SemaphoreCreateOp op, PatternRewriter &rewriter) {
  int numStages = std::max(1, op.getType().getNumStages());
  int pendingCount = getPendingCount(op);

  rewriter.setInsertionPoint(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(b, b.getI64Type(), numStages);
  for (int i = 0; i < numStages; ++i) {
    auto view = createSingleBufferView(b, mbars, i);
    InitBarrierOp::create(b, view, pendingCount);
  }

  return mbars;
}

bool detectFenceNeeded(
    SemaphoreCreateOp semaOp, ArrayRef<AsyncOp> asyncKinds,
    const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema) {
  bool isGenericProxy =
      llvm::any_of(asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
  if (!isGenericProxy)
    return false;

  auto semaType = cast<SemaphoreType>(semaOp.getType());
  auto semaBufType = cast<MemDescType>(semaType.getBaseType()[0]);
  auto tmem = TensorMemorySpaceAttr::get(semaOp.getContext());
  if (semaBufType.getMemorySpace() == tmem)
    return false;

  // Fence decision depends on cross-semaphore information:
  // a generic release (async_ops=[none]) needs a fence if another semaphore
  // sharing the same buffer has an async release (tc5mma/tma_load).
  //
  // We do not recompute this by scanning IR during pattern application because
  // greedy rewriting can erase peer semaphores in a non-deterministic order.
  // If we queried "live" IR here, fence insertion would depend on rewrite
  // order and could miss required fences.
  auto it = hasAsyncPeerBySema.find(semaOp.getOperation());
  return it != hasAsyncPeerBySema.end() && it->second;
}

void rewriteAcquire(SemaphoreAcquireOp op, PatternRewriter &rewriter,
                    Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  Value phaseBit =
      arith::ShRUIOp::create(rewriter, loc, op.getPhase(), op.getStage());
  assignStageCluster(phaseBit.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);
  auto c1 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
  assignStageCluster(c1, partitionWsTagIds, stageCluster, rewriter);
  phaseBit = arith::AndIOp::create(rewriter, loc, phaseBit, c1);
  assignStageCluster(phaseBit.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  auto waitOp = WaitBarrierOp::create(rewriter, loc, mbar, phaseBit);
  assignStageCluster(waitOp, partitionWsTagIds, stageCluster, rewriter);
}

void rewriteRelease(
    SemaphoreCreateOp semaOp, SemaphoreReleaseOp op, PatternRewriter &rewriter,
    Value mbars, const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema) {
  auto loc = op.getLoc();
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  if (detectFenceNeeded(semaOp, asyncKinds, hasAsyncPeerBySema)) {
    auto fence = FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    assignStageCluster(fence, partitionWsTagIds, stageCluster, rewriter);
  }

  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                     rewriter);

  for (auto asyncKind : asyncKinds) {
    Operation *arriveOp = nullptr;
    switch (asyncKind) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      arriveOp = ArriveBarrierOp::create(rewriter, loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      arriveOp = TCGen5CommitOp::create(rewriter, loc, mbar, Value(),
                                        ValueRange{});
      break;
    case AsyncOp::TMALoad:
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unknown async op");
    }
    if (arriveOp)
      assignStageCluster(arriveOp, partitionWsTagIds, stageCluster, rewriter);
  }
}

void rewriteBuffer(SemaphoreBufferOp op, PatternRewriter &rewriter,
                   ArrayRef<Value> buffers) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  for (auto [i, buffer] : llvm::enumerate(buffers)) {
    auto memDesc = cast<MemDescType>(buffer.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(memDesc.getEncoding())) {
      op.getBuffers()[i].replaceAllUsesWith(buffer);
      continue;
    }

    auto shape = memDesc.getShape();
    SmallVector<int64_t> viewShape(shape.begin() + 1, shape.end());
    auto viewType = MemDescType::get(viewShape, memDesc.getElementType(),
                                     memDesc.getEncoding(),
                                     memDesc.getMemorySpace(),
                                     /*mutableMemory=*/true);
    auto view = MemDescIndexOp::create(rewriter, loc, viewType, buffer, op.getStage());
    assignStageCluster(view, partitionWsTagIds, stageCluster, rewriter);
    op.getBuffers()[i].replaceAllUsesWith(view);
  }
}

void handleTMALoads(SemaphoreCreateOp op, PatternRewriter &rewriter, Value mbars) {
  for (Operation *user : op->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;

    auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
    if (!llvm::any_of(kinds, [](AsyncOp kind) { return kind == AsyncOp::TMALoad; }))
      continue;

    int txCount = 0;
    SetVector<Operation *> loadOps;
    for (Operation *tokUser : releaseOp.getToken().getUsers()) {
      auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser);
      if (!bufOp)
        continue;

      for (Value buf : bufOp.getBuffers()) {
        for (Operation *viewUser : buf.getUsers()) {
          if (auto loadOp = dyn_cast<DescriptorLoadOpInterface>(viewUser)) {
            loadOps.insert(loadOp);
            txCount += loadOp.getTxCount();
          }
        }
      }
    }
    if (loadOps.empty())
      continue;

    auto partitionWsTagIds = getPartitionWsTagIds(releaseOp);
    auto stageCluster = getStageCluster(releaseOp);

    rewriter.setInsertionPoint(loadOps.front());
    auto mbar = createSingleBufferView(rewriter, mbars, releaseOp.getStage());
    assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                       rewriter);

    auto pred = arith::ConstantIntOp::create(rewriter, op.getLoc(), 1, 1);
    assignStageCluster(pred, partitionWsTagIds, stageCluster, rewriter);

    auto expectOp =
        BarrierExpectOp::create(rewriter, op.getLoc(), mbar, txCount, pred);
    assignStageCluster(expectOp, partitionWsTagIds, stageCluster, rewriter);

    for (Operation *loadOp : loadOps) {
      rewriter.setInsertionPoint(loadOp);
      if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(loadOp)) {
        auto newOp = AsyncTMACopyGlobalToLocalOp::create(
            rewriter, loadOp->getLoc(), descLoad.getDesc(), descLoad.getIndices(),
            mbar, descLoad.getResult(), pred);
        assignStageCluster(newOp, getPartitionWsTagIds(loadOp),
                           getStageCluster(loadOp), rewriter);
      } else if (auto descGather =
                     dyn_cast<triton::nvws::DescriptorGatherOp>(loadOp)) {
        auto newOp = AsyncTMAGatherOp::create(
            rewriter, loadOp->getLoc(), descGather.getDesc(),
            descGather.getXOffsets(), descGather.getYOffset(), mbar,
            descGather.getResult(), pred);
        assignStageCluster(newOp, getPartitionWsTagIds(loadOp),
                           getStageCluster(loadOp), rewriter);
      } else {
        llvm_unreachable("unknown descriptor load op");
      }
      rewriter.eraseOp(loadOp);
    }
  }
}

class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  LowerSemaphoreCreate(
      MLIRContext *ctx,
      const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema)
      : OpRewritePattern<SemaphoreCreateOp>(ctx),
        hasAsyncPeerBySema(hasAsyncPeerBySema) {}

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto mbars = createAndInitMbar(op, rewriter);
    SmallVector<Value> buffers(op.getBuffers().begin(), op.getBuffers().end());

    // Handle TMA rewrites before erasing/rewriting semaphore users.
    handleTMALoads(op, rewriter, mbars);

    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    SetVector<Operation *> allUsers;
    for (Operation *user : op->getUsers())
      allUsers.insert(user);

    Operation *cleanupAnchor = op.getOperation();
    if (!allUsers.empty()) {
      auto sortedUsers = topologicalSort(allUsers);
      cleanupAnchor = op->getBlock()->findAncestorOpInBlock(*sortedUsers.back());
    }

    // Insert cleanup now; later "setInsertionPointAfter(lastUser)" rewrites
    // will naturally materialize before these ops.
    {
      ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      b.setInsertionPointAfter(cleanupAnchor);
      int numStages = std::max(1, op.getType().getNumStages());
      for (int i = 0; i < numStages; ++i) {
        auto view = createSingleBufferView(b, mbars, i);
        InvalBarrierOp::create(b, view);
      }
      LocalDeallocOp::create(b, mbars);
    }

    SmallVector<Operation *> users(op->getUsers().begin(), op->getUsers().end());
    for (Operation *userOp : users) {
      opToDelete.insert(userOp);
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(userOp)) {
        rewriteAcquire(acquireOp, rewriter, mbars);
      } else if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(userOp)) {
        rewriteRelease(op, releaseOp, rewriter, mbars, hasAsyncPeerBySema);
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(userOp)) {
        rewriteBuffer(bufferOp, rewriter, buffers);
      } else {
        llvm_unreachable("unexpected semaphore user");
      }
    }

    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken =
        ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    for (Operation *candidate : sorted) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(candidate))
        acquireOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }

private:
  const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema;
};

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
  using impl::NVWSLowerSemaphoreBase<
      NVWSLowerSemaphore>::NVWSLowerSemaphoreBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Precompute cross-semaphore async relationships before any rewrite:
    //
    // hasAsyncPeerBySema[S] == true iff there exists another semaphore S'
    // that shares S's backing buffer and has at least one async release
    // (tc5mma or tma_load).
    //
    // This is required for deterministic/correct fence lowering. Greedy
    // rewrites process semaphores independently and may erase one semaphore
    // before rewriting its peer; computing this relation ahead of time avoids
    // rewrite-order-dependent fence decisions.
    llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> bufferGroups;
    m.walk([&](SemaphoreCreateOp semaOp) {
      if (semaOp.getBuffers().empty())
        return;
      bufferGroups[semaOp.getBuffers()[0]].push_back(semaOp);
    });

    llvm::DenseMap<Operation *, bool> hasAsyncPeerBySema;
    for (auto &it : bufferGroups) {
      auto &semas = it.second;
      llvm::DenseMap<Operation *, bool> hasAsyncRelease;
      for (auto semaOp : semas) {
        bool hasAsync = false;
        for (Operation *user : semaOp->getUsers()) {
          auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
          if (!releaseOp)
            continue;
          auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
          bool hasAsyncConsumer = llvm::any_of(
              kinds, [](AsyncOp kind) { return kind == AsyncOp::TC5MMA; });
          bool hasAsyncProducer = llvm::any_of(
              kinds, [](AsyncOp kind) { return kind == AsyncOp::TMALoad; });
          if (hasAsyncConsumer || hasAsyncProducer) {
            hasAsync = true;
            break;
          }
        }
        hasAsyncRelease[semaOp.getOperation()] = hasAsync;
      }

      for (auto semaOp : semas) {
        bool hasAsyncPeer = false;
        for (auto otherSema : semas) {
          if (otherSema == semaOp)
            continue;
          if (hasAsyncRelease.lookup(otherSema.getOperation())) {
            hasAsyncPeer = true;
            break;
          }
        }
        hasAsyncPeerBySema[semaOp.getOperation()] = hasAsyncPeer;
      }
    }

    RewritePatternSet patterns(context);
    // Pass precomputed peer information into the pattern so each semaphore can
    // make a stable fence decision even after peers are rewritten/erased.
    patterns.add<LowerSemaphoreCreate>(context, hasAsyncPeerBySema);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(m, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

} // namespace triton
} // namespace mlir
