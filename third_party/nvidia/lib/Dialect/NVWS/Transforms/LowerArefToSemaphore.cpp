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

#include "Utilities.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERAREFTOSEMAPHORE
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

bool isOperandPipelineable(Value v, scf::ForOp forOp) {
  auto isPipelineable = [](Operation *op) {
    return isa<ArefPutEnterOp, ArefGetEnterOp, ArefBufferOp>(op);
  };

  Operation *foundDef = nullptr;
  return triton::nvidia_gpu::isOperandPipelineableBase(v, forOp, foundDef,
                                                       isPipelineable);
}

void setIsAsync(triton::nvidia_gpu::MMAv5OpInterface mmaOp,
                unsigned defaultNumStages) {
  bool isAsync = true;
  auto forOp = mmaOp->getParentOfType<scf::ForOp>();
  if (!forOp)
    return;

  unsigned numStages = getNumStagesOrDefault(forOp, defaultNumStages);
  if (numStages <= 1)
    return;

  if (auto scaledOp = dyn_cast<triton::nvidia_gpu::TCGen5MMAScaledOp>(
          mmaOp.getOperation())) {
    if (!triton::nvidia_gpu::areScalesPipelineable(scaledOp, forOp))
      isAsync = false;
    if (!isOperandPipelineable(scaledOp.getAScale(), forOp) ||
        !isOperandPipelineable(scaledOp.getBScale(), forOp)) {
      isAsync = false;
    }
  }
  mmaOp.setIsAsync(isAsync);
}

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs)
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  return kinds;
}

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value aref) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (auto arefUser : aref.getUsers()) {
    if (auto getEnter = dyn_cast<ArefGetEnterOp>(arefUser)) {
      if (hasPartition(getEnter) && getPartitionIds(getEnter).front() == 0) {
        // Ignore mmav5 ops in the default partition. They are not warp
        // specialized.
        continue;
      }

      for (auto consumer : getEnter->getUsers()) {
        if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
          mmav5Ops.insert(mmav5);
        } else if (auto forOp = consumer->getParentOfType<scf::ForOp>()) {
          auto users =
              getTopLevelUsersInLoop(consumer, forOp, [](Operation *user) {
                return isa<MMAv5OpInterface>(user);
              });
          for (auto user : users)
            mmav5Ops.insert(cast<MMAv5OpInterface>(user));
        }
      }
    }
  }
  return mmav5Ops;
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true, type.getAllocShape());
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

static SmallVector<Type> getMutableBufferTypes(TypeRange bufferTypes) {
  SmallVector<Type> mutableTypes;
  mutableTypes.reserve(bufferTypes.size());
  for (Type type : bufferTypes) {
    if (auto memTy = dyn_cast<MemDescType>(type)) {
      mutableTypes.push_back(MemDescType::get(
          memTy.getShape(), memTy.getElementType(), memTy.getEncoding(),
          memTy.getMemorySpace(), /*mutableMemory=*/true, memTy.getAllocShape()));
    } else {
      mutableTypes.push_back(type);
    }
  }
  return mutableTypes;
}

bool isProducerLoad(ArefCreateOp arefOp) {
  for (auto user : arefOp.getResult().getUsers()) {
    if (auto putOp = dyn_cast<ArefPutEnterOp>(user)) {
      if (llvm::any_of(putOp->getUsers(), [](auto user) {
            return isa<triton::nvws::DescriptorLoadOpInterface>(user);
          })) {
        return true;
      }
    }
  }
  return false;
}

void multiBufferAref(const SmallVector<ArefCreateOp> &arefOps, int numStages) {
  SmallVector<Operation *> allocsToErase;
  for (auto arefOp : arefOps) {
    SmallVector<Value> allocOps;
    SmallVector<Type> arefTypes;

    bool eligible = true;
    for (auto opnd : arefOp.getOperands()) {
      if (!opnd.getDefiningOp() || isa<TMEMAllocOp>(opnd.getDefiningOp()))
        eligible = false;
    }

    if (!eligible)
      continue;

    OpBuilder builder(arefOp);
    for (auto opnd : arefOp.getOperands()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto arefBufType = cast<MemDescType>(opnd.getType());
      arefBufType =
          getMultiBufferedType(getBufferViewType(arefBufType, true), numStages);
      Operation *newAlloc =
          triton::nvws::createAlloc(builder, oldAlloc->getLoc(), arefBufType,
                                    Value());
      allocOps.push_back(newAlloc->getResult(0));
      arefTypes.push_back(arefBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      allocsToErase.push_back(oldAlloc);
    }

    auto newAref =
        createArefCreateOp(builder, arefTypes, allocOps, arefOp.getLoc());

    arefOp.getResult().replaceAllUsesWith(newAref.getResult());
    arefOp.erase();
  }

  for (auto alloc : allocsToErase)
    alloc->erase();
}

template <typename EnterOp, typename ExitOp>
ExitOp createCombinedArefOps(SmallVector<EnterOp> &enterOps,
                             SmallVector<ExitOp> &exitOps, ArefCreateOp aref,
                             OpBuilder &builder,
                             Operation *combinedEnterInsertPoint = nullptr) {
  auto firstEnter = *llvm::min_element(enterOps, [](EnterOp a, EnterOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto lastExit = *llvm::max_element(exitOps, [](ExitOp a, ExitOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  SmallVector<Type> arefEnterBuffers;
  for (auto enterOp : enterOps)
    arefEnterBuffers.push_back(enterOp.getResult(0).getType());

  llvm::SmallSetVector<Attribute, 5> opAttrsSet;
  for (ExitOp exitOp : exitOps)
    opAttrsSet.insert(exitOp.getAsyncOps().begin(), exitOp.getAsyncOps().end());

  builder.setInsertionPointAfter(aref);
  auto zero = arith::ConstantIntOp::create(builder, aref.getLoc(), 0, 32);
  assignStageCluster(zero, getPartitionWsTagIds(firstEnter),
                     getStageCluster(firstEnter), builder);

  if (combinedEnterInsertPoint) {
    // Combined get enter must be placed after combined put enter.
    builder.setInsertionPointAfter(combinedEnterInsertPoint);
  } else {
    builder.setInsertionPoint(firstEnter);
  }
  auto combinedEnter =
      EnterOp::create(builder, firstEnter.getLoc(), arefEnterBuffers,
                      builder.getType<AsyncTokenType>(), aref, zero, zero);
  assignStageCluster(combinedEnter, getPartitionWsTagIds(firstEnter),
                     getStageCluster(firstEnter), builder);

  builder.setInsertionPoint(lastExit);
  llvm::SmallVector<Attribute> asyncOpAttrs(opAttrsSet.begin(),
                                            opAttrsSet.end());
  auto combinedExit = ExitOp::create(builder, firstEnter.getLoc(), aref,
                                     combinedEnter.getToken(), zero,
                                     builder.getArrayAttr(asyncOpAttrs));
  assignStageCluster(combinedExit, getPartitionWsTagIds(lastExit),
                     getStageCluster(lastExit), builder);

  std::function<void(Operation *, Operation *)> moveUserAfter =
      [&](Operation *op, Operation *target) {
        auto curBlock = target->getBlock();
        for (auto user : op->getUsers()) {
          auto userOp = curBlock->findAncestorOpInBlock(*user);
          if (userOp->isBeforeInBlock(target)) {
            userOp->moveAfter(target);
            moveUserAfter(userOp, userOp);
          }
        }
      };

  for (auto [idx, enterOp] : llvm::enumerate(enterOps)) {
    moveUserAfter(enterOp, combinedEnter);
    enterOp.getBuffers()[0].replaceAllUsesWith(combinedEnter.getBuffers()[idx]);
  }

  return combinedExit;
}

SmallVector<Operation *> findSharedMemorySinkOps(Value value) {
  SmallVector<Operation *> sinkOps;
  for (Operation *user : value.getUsers()) {
    if (isa<MMAv5OpInterface, LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      auto rec = findSharedMemorySinkOps(user->getResult(0));
      sinkOps.insert(sinkOps.end(), rec.begin(), rec.end());
    }
  }
  return sinkOps;
}

Operation *getDominantConsumer(ArefGetEnterOp getEnterOp, Block &container,
                               DominanceInfo &domInfo) {
  assert(getEnterOp->getNumResults() && "Expect a single-result ArefGenterOp");
  auto buf = getEnterOp->getResult(0);
  SmallVector<Operation *> sinkOps = findSharedMemorySinkOps(buf);
  if (sinkOps.empty())
    return nullptr;
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

// This is an optimization to combine arefs for TMA load into one, so that
// synchronization operations are coalesced.
void combineArefs(scf::ForOp loop) {
  // Combine getEnterOps in the same loop body, not across a loop.
  auto getEnterOps = loop.getOps<ArefGetEnterOp>();

  // Arefs whose get-enter ops share the same dominant consumer can be combined.
  DominanceInfo domInfo(loop);
  llvm::DenseMap<std::pair<Operation *, int>, SmallVector<ArefGetEnterOp>>
      liveBeforeGroups;
  for (auto getEnterOp : getEnterOps) {
    if (auto liveBeforeOp =
            getDominantConsumer(getEnterOp, *loop.getBody(), domInfo)) {
      assert(hasPartition(getEnterOp));
      auto partitionIds = getPartitionIds(getEnterOp);
      assert(partitionIds.size() == 1);
      liveBeforeGroups[{liveBeforeOp, partitionIds.front()}].push_back(
          getEnterOp);
    }
  }

  for (auto getEnterOpsGroup : llvm::make_second_range(liveBeforeGroups)) {
    if (getEnterOpsGroup.size() == 1)
      continue;

    SmallVector<ArefCreateOp> arefs;
    for (auto getEnterOp : getEnterOpsGroup)
      arefs.push_back(cast<ArefCreateOp>(getEnterOp.getAref().getDefiningOp()));

    SmallVector<ArefPutEnterOp> putEnterOps;
    SmallVector<ArefPutExitOp> putExitOps;
    SmallVector<ArefGetExitOp> getExitOps;
    SmallVector<int> producerGroupIds;
    for (auto aref : arefs) {
      for (auto user : aref->getUsers()) {
        if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(user)) {
          putEnterOps.push_back(putEnterOp);
          producerGroupIds.push_back(getPartitionIds(putEnterOp).front());
        } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
          putExitOps.push_back(putExitOp);
        } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
          getExitOps.push_back(getExitOp);
        }
      }
    }

    // Producer arefs must be in the same partition.
    if (llvm::any_of(producerGroupIds,
                     [&](auto id) { return id != producerGroupIds[0]; })) {
      continue;
    }

    SmallVector<Type> arefBufTypes;
    SmallVector<Value> arefBufs;
    for (auto aref : arefs) {
      arefBufTypes.push_back(aref.getOperands()[0].getType());
      arefBufs.push_back(aref.getOperands()[0]);
    }

    // Set insertion point at the last aref.create.
    auto lastAref = *llvm::max_element(arefs, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });

    OpBuilder builder(lastAref);
    auto aref =
        createArefCreateOp(builder, arefBufTypes, arefBufs, lastAref->getLoc());

    auto combinedPutExit =
        createCombinedArefOps(putEnterOps, putExitOps, aref, builder);
    createCombinedArefOps(getEnterOpsGroup, getExitOps, aref, builder,
                          combinedPutExit);

    for (auto putExitOp : putExitOps)
      putExitOp->erase();
    for (auto putEnterOp : putEnterOps)
      putEnterOp->erase();
    for (auto getExitOp : getExitOps)
      getExitOp->erase();
    for (auto getEnterOp : getEnterOpsGroup)
      getEnterOp->erase();
    for (auto aref : arefs)
      aref->erase();
  }
}

void hoistPoisonOps(triton::FuncOp funcOp) {
  auto *block = &funcOp.getBody().front();
  funcOp.walk([&](ub::PoisonOp op) { op->moveBefore(&block->front()); });
}

struct SemaValue {
  Value sem0;
  Value sem1;
  SmallVector<Value> buffers;
};

SemaValue createSemaphores(ArefCreateOp op, PatternRewriter &rewriter) {
  auto arefTy = op.getType();
  auto arefBufTypes = llvm::to_vector(
      llvm::map_range(arefTy.getBaseType(),
                      [](Type type) { return cast<MemDescType>(type); }));
  auto depth = getArefDepth(arefBufTypes[0]);

  rewriter.setInsertionPoint(op);
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto semaTy = SemaphoreType::get(b.getContext(), arefTy.getBaseType(), depth);
  auto sem0 = SemaphoreCreateOp::create(b, semaTy, op.getOperands(), true);
  auto sem1 = SemaphoreCreateOp::create(b, semaTy, op.getOperands(), false);
  return SemaValue{sem0, sem1, SmallVector<Value>(op.getOperands())};
}

class LowerArefToSemaCreate : public OpRewritePattern<ArefCreateOp> {
public:
  LowerArefToSemaCreate(MLIRContext *ctx, unsigned defaultNumStages)
      : OpRewritePattern(ctx), defaultNumStages(defaultNumStages) {}

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    // Step 1: collect all information from aref ops before any replacement.
    auto sema = createSemaphores(op, rewriter);
    auto mmav5Ops = getAsyncMMAv5Consumers(op.getResult());

    SmallVector<Operation *> users(op->getUsers().begin(), op->getUsers().end());

    DenseMap<ArefPutEnterOp, ArefPutExitOp> putPairs;
    for (Operation *user : users) {
      if (auto putEnter = dyn_cast<ArefPutEnterOp>(user)) {
        for (auto tokUser : putEnter.getToken().getUsers()) {
          if (auto putExit = dyn_cast<ArefPutExitOp>(tokUser)) {
            putPairs[putEnter] = putExit;
            break;
          }
        }
      }
    }

    DenseMap<ArefBufferOp, Value> bufferSemMap;
    for (Operation *user : users) {
      if (auto bufOp = dyn_cast<ArefBufferOp>(user)) {
        Value tok = bufOp.getToken();
        if (tok.getDefiningOp<ArefPutEnterOp>())
          bufferSemMap[bufOp] = sema.sem0;
        else if (tok.getDefiningOp<ArefGetEnterOp>())
          bufferSemMap[bufOp] = sema.sem1;
        else
          bufferSemMap[bufOp] = sema.sem0;
      }
    }

    // Step 2: create semaphore ops and replace uses.
    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    for (Operation *userOp : users) {
      opToDelete.insert(userOp);

      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        rewriter.setInsertionPointAfter(user);
        auto partitionWsTagIds = getPartitionWsTagIds(user);
        auto stageCluster = getStageCluster(user);
        auto acquire = SemaphoreAcquireOp::create(
            rewriter, user.getLoc(), sema.sem0,
            rewriter.getType<AsyncTokenType>());
        assignStageCluster(acquire, partitionWsTagIds, stageCluster, rewriter);

        auto bufOp = SemaphoreBufferOp::create(rewriter, user.getLoc(),
                                               sema.sem0,
                                               TypeRange(getMutableBufferTypes(
                                                   user.getBuffers().getTypes())),
                                               acquire.getToken());
        assignStageCluster(bufOp, partitionWsTagIds, stageCluster, rewriter);

        for (auto [oldBuffer, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers()))
          oldBuffer.replaceAllUsesWith(view);
        user.getToken().replaceAllUsesWith(acquire.getToken());

        if (auto exitOp = putPairs.lookup(user)) {
          auto kinds = castAsyncOpAttrs(exitOp.getAsyncOps());
          if (llvm::any_of(kinds, [](AsyncOp kind) {
                return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
              })) {
            for (auto mma : mmav5Ops)
              setIsAsync(mma, defaultNumStages);
          }
        }

      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        rewriter.setInsertionPointAfter(user);
        auto partitionWsTagIds = getPartitionWsTagIds(user);
        auto stageCluster = getStageCluster(user);
        auto acquire = SemaphoreAcquireOp::create(
            rewriter, user.getLoc(), sema.sem1,
            rewriter.getType<AsyncTokenType>());
        assignStageCluster(acquire, partitionWsTagIds, stageCluster, rewriter);

        auto bufOp = SemaphoreBufferOp::create(rewriter, user.getLoc(),
                                               sema.sem1,
                                               TypeRange(getMutableBufferTypes(
                                                   user.getBuffers().getTypes())),
                                               acquire.getToken());
        assignStageCluster(bufOp, partitionWsTagIds, stageCluster, rewriter);

        for (auto [oldBuffer, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers())) {
          oldBuffer.replaceAllUsesWith(view);
          propagateMutability(view);
        }
        user.getToken().replaceAllUsesWith(acquire.getToken());

      } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
        // Cross-release: put.exit releases sem1.
        rewriter.setInsertionPointAfter(user);
        auto release = SemaphoreReleaseOp::create(
            rewriter, user.getLoc(), sema.sem1, user.getToken(),
            user.getAsyncOps());
        assignStageCluster(release, getPartitionWsTagIds(user),
                           getStageCluster(user), rewriter);

      } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
        // Cross-release: get.exit releases sem0.
        rewriter.setInsertionPointAfter(user);
        auto release = SemaphoreReleaseOp::create(
            rewriter, user.getLoc(), sema.sem0, user.getToken(),
            user.getAsyncOps());
        assignStageCluster(release, getPartitionWsTagIds(user),
                           getStageCluster(user), rewriter);

      } else if (auto user = dyn_cast<ArefBufferOp>(userOp)) {
        rewriter.setInsertionPointAfter(user);
        Value sem = bufferSemMap.lookup(user);
        if (!sem)
          sem = sema.sem0;

        auto bufOp = SemaphoreBufferOp::create(
            rewriter, user.getLoc(), sem,
            TypeRange(getMutableBufferTypes(user.getBuffers().getTypes())),
            user.getToken());
        assignStageCluster(bufOp, getPartitionWsTagIds(user),
                           getStageCluster(user), rewriter);
        for (auto [oldBuffer, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers()))
          oldBuffer.replaceAllUsesWith(view);

      } else {
        llvm_unreachable("unexpected aref user");
      }
    }

    // Step 3: replace remaining aref tokens with poison and erase old ops.
    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken =
        ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    for (auto candidate : sorted) {
      if (auto enterOp = dyn_cast<ArefPutEnterOp>(candidate))
        enterOp.getToken().replaceAllUsesWith(replToken);
      else if (auto enterOp = dyn_cast<ArefGetEnterOp>(candidate))
        enterOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }

private:
  unsigned defaultNumStages;
};

class NVWSLowerArefToSemaphore
    : public impl::NVWSLowerArefToSemaphoreBase<NVWSLowerArefToSemaphore> {
  using impl::NVWSLowerArefToSemaphoreBase<
      NVWSLowerArefToSemaphore>::NVWSLowerArefToSemaphoreBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName)) {
        loop->walk([&](scf::ForOp op) { loops.push_back(op); });
      }
    });
    for (scf::ForOp loop : loops)
      combineArefs(loop);

    SmallVector<ArefCreateOp> arefOps;
    m.walk([&](ArefCreateOp arefOp) {
      if (isProducerLoad(arefOp))
        arefOps.push_back(arefOp);
    });
    multiBufferAref(arefOps, numStages);

    RewritePatternSet patterns(context);
    patterns.add<LowerArefToSemaCreate>(context, numStages);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(m, std::move(patterns), config)))
      return signalPassFailure();

    // Hoist all poison ops to the top of function from nvws.wg regions.
    // They are unannotated and will trip subsequent passes, so hoist them.
    m.walk([&](triton::FuncOp funcOp) { hoistPoisonOps(funcOp); });
  }
};

} // namespace

} // namespace triton
} // namespace mlir
