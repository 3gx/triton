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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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

bool isOperandPipelineable(Value v, scf::ForOp forOp) {
  auto isPipelineable = [](Operation *op) {
    return isa<SemaphoreAcquireOp, SemaphoreBufferOp>(op);
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

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value semaphore) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (Operation *semaUser : semaphore.getUsers()) {
    auto acquireOp = dyn_cast<SemaphoreAcquireOp>(semaUser);
    if (!acquireOp)
      continue;
    if (hasPartition(acquireOp) && getPartitionIds(acquireOp).front() == 0) {
      // Ignore MMAv5 ops in the default partition. They are not warp
      // specialized.
      continue;
    }

    for (Operation *tokUser : acquireOp.getToken().getUsers()) {
      auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
      if (!bufferOp)
        continue;

      for (Operation *consumer : bufferOp->getUsers()) {
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

bool hasProducerLoad(SemaphoreCreateOp semaOp) {
  for (Operation *user : semaOp->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp)
      continue;
    auto asyncKinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
    if (llvm::any_of(asyncKinds, [](AsyncOp kind) {
          return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
        })) {
      return true;
    }
  }
  return false;
}

void multiBufferSemaphore(ModuleOp module, int numStages) {
  if (numStages <= 1)
    return;

  llvm::DenseMap<Value, SmallVector<SemaphoreCreateOp>> bufferGroups;
  module.walk([&](SemaphoreCreateOp semaOp) {
    if (semaOp.getBuffers().empty())
      return;
    bufferGroups[semaOp.getBuffers().front()].push_back(semaOp);
  });

  SetVector<Operation *> staleAllocs;
  for (auto &it : bufferGroups) {
    auto &semas = it.second;
    if (semas.empty())
      continue;
    if (!llvm::any_of(semas, hasProducerLoad))
      continue;

    bool eligible = true;
    for (Value opnd : semas.front().getBuffers()) {
      if (!opnd.getDefiningOp() || isa<TMEMAllocOp>(opnd.getDefiningOp()))
        eligible = false;
    }
    if (!eligible)
      continue;

    OpBuilder builder(semas.front());
    SmallVector<Value> newBuffers;
    SmallVector<Type> newBufferTypes;
    newBuffers.reserve(semas.front().getBuffers().size());
    newBufferTypes.reserve(semas.front().getBuffers().size());

    for (Value opnd : semas.front().getBuffers()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto oldBufType = cast<MemDescType>(opnd.getType());
      auto newBufType =
          getMultiBufferedType(getBufferViewType(oldBufType, true), numStages);
      Operation *newAlloc =
          triton::nvws::createAlloc(builder, oldAlloc->getLoc(), newBufType,
                                    Value());
      newBuffers.push_back(newAlloc->getResult(0));
      newBufferTypes.push_back(newBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      staleAllocs.insert(oldAlloc);
    }

    for (SemaphoreCreateOp semaOp : semas) {
      OpBuilder semaBuilder(semaOp);
      auto semaTy = SemaphoreType::get(
          semaBuilder.getContext(),
          TypeArrayAttr::get(semaBuilder.getContext(), newBufferTypes),
          numStages);
      auto newSema =
          SemaphoreCreateOp::create(semaBuilder, semaOp.getLoc(), semaTy,
                                    newBuffers, semaOp.getIsReleased());
      newSema->setAttrs(semaOp->getAttrs());
      semaOp.getResult().replaceAllUsesWith(newSema.getResult());
      semaOp.erase();
    }
  }

  for (Operation *alloc : staleAllocs)
    alloc->erase();
}

// ---------------------------------------------------------------------------
// combineSemaphores: coalesce multiple semaphore pairs that feed the same
// dominant consumer in a warp-specialize for-loop.  Mirrors combineArefs()
// in LowerArefToSemaphore.cpp:337-420.
// ---------------------------------------------------------------------------

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

// 2-hop traversal: acquireOp → token → SemaphoreBufferOp → buffer results
// → findSharedMemorySinkOps → findNearestCommonDominator.
Operation *getDominantConsumer(SemaphoreAcquireOp acquireOp, Block &container,
                               DominanceInfo &domInfo) {
  SmallVector<Operation *> sinkOps;
  for (Operation *tokUser : acquireOp.getToken().getUsers()) {
    auto bufferOp = dyn_cast<SemaphoreBufferOp>(tokUser);
    if (!bufferOp)
      continue;
    for (Value buf : bufferOp.getBuffers()) {
      auto ops = findSharedMemorySinkOps(buf);
      sinkOps.insert(sinkOps.end(), ops.begin(), ops.end());
    }
  }
  if (sinkOps.empty())
    return nullptr;
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

// Acquire-token lineage: consumer acquires FULL; consumer release targets
// EMPTY (cross-release).  Follow consumer acquire → token →
// SemaphoreReleaseOp → getSemaphore() → the EMPTY SemaphoreCreateOp.
SemaphoreCreateOp getPartnerSemaphore(SemaphoreAcquireOp consumerAcquire) {
  for (Operation *tokUser : consumerAcquire.getToken().getUsers()) {
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(tokUser))
      return releaseOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
  }
  return nullptr;
}

void combineSemaphores(scf::ForOp loop) {
  // 1. Find consumer acquire ops (consumer = acquires FULL semaphore,
  //    isReleased == false).  Skip TMEM.
  SmallVector<SemaphoreAcquireOp> consumerAcquires;
  for (auto acquireOp : loop.getOps<SemaphoreAcquireOp>()) {
    auto semaCreate =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (!semaCreate || semaCreate.getIsReleased())
      continue;
    bool isTMEM = false;
    for (Value buf : semaCreate.getBuffers()) {
      if (buf.getDefiningOp() && isa<TMEMAllocOp>(buf.getDefiningOp()))
        isTMEM = true;
    }
    if (isTMEM)
      continue;
    consumerAcquires.push_back(acquireOp);
  }

  // 2. Group by (dominant consumer, partition ID).
  DominanceInfo domInfo(loop);
  llvm::DenseMap<std::pair<Operation *, int>, SmallVector<SemaphoreAcquireOp>>
      groups;
  for (auto acquireOp : consumerAcquires) {
    auto liveBeforeOp =
        getDominantConsumer(acquireOp, *loop.getBody(), domInfo);
    if (!liveBeforeOp)
      continue;
    assert(hasPartition(acquireOp));
    auto partitionIds = getPartitionIds(acquireOp);
    assert(partitionIds.size() == 1);
    groups[{liveBeforeOp, partitionIds.front()}].push_back(acquireOp);
  }

  // 3. Combine each group with size > 1.
  for (auto &[key, acquireGroup] : groups) {
    if (acquireGroup.size() <= 1)
      continue;

    // --- Step A: discover EMPTY/FULL pairs ---------------------------------
    struct SemaPair {
      SemaphoreCreateOp empty; // producer acquires (isReleased=true)
      SemaphoreCreateOp full;  // consumer acquires (isReleased=false)
    };
    SmallVector<SemaPair> pairs;

    // Collect per-acquire consumer ops in order matching pairs.
    SmallVector<SemaphoreBufferOp> consBufferOps;
    SmallVector<SemaphoreReleaseOp> consReleaseOps;

    SmallVector<int> producerPartitionIds;
    bool valid = true;
    for (auto consAcquire : acquireGroup) {
      auto fullSema =
          consAcquire.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
      auto emptySema = getPartnerSemaphore(consAcquire);
      if (!fullSema || !emptySema) {
        valid = false;
        break;
      }
      pairs.push_back({emptySema, fullSema});

      // Collect the single consumer buffer & release for this acquire.
      SemaphoreBufferOp consBuf = nullptr;
      SemaphoreReleaseOp consRel = nullptr;
      for (Operation *tokUser : consAcquire.getToken().getUsers()) {
        if (auto b = dyn_cast<SemaphoreBufferOp>(tokUser))
          consBuf = b;
        if (auto r = dyn_cast<SemaphoreReleaseOp>(tokUser))
          consRel = r;
      }
      if (!consBuf || !consRel) {
        valid = false;
        break;
      }
      consBufferOps.push_back(consBuf);
      consReleaseOps.push_back(consRel);

      // Collect producer partition IDs from EMPTY semaphore users.
      for (Operation *user : emptySema->getUsers()) {
        if (auto prodAcquire = dyn_cast<SemaphoreAcquireOp>(user)) {
          if (hasPartition(prodAcquire))
            producerPartitionIds.push_back(
                getPartitionIds(prodAcquire).front());
        }
      }
    }
    if (!valid)
      continue;

    // All producers must be in the same partition.
    if (llvm::any_of(producerPartitionIds, [&](int id) {
          return id != producerPartitionIds[0];
        }))
      continue;

    // --- Step B: build combined SemaphoreCreateOps -------------------------
    SmallVector<Value> allBufs;
    SmallVector<Type> allBufTypes;
    for (auto &pair : pairs) {
      for (Value buf : pair.full.getBuffers()) {
        allBufs.push_back(buf);
        allBufTypes.push_back(buf.getType());
      }
    }

    // Collect buffer result types from individual consumer buffer ops (NOT
    // from the alloc types, which may have a multi-buffer dimension).
    SmallVector<Type> consBufResultTypes;
    for (auto consBuf : consBufferOps) {
      for (auto res : consBuf.getBuffers())
        consBufResultTypes.push_back(res.getType());
    }
    SmallVector<Type> prodBufResultTypes;
    for (auto &pair : pairs) {
      for (Operation *user : pair.empty->getUsers()) {
        if (auto acq = dyn_cast<SemaphoreAcquireOp>(user)) {
          for (Operation *tokUser : acq.getToken().getUsers()) {
            if (auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
              for (auto res : bufOp.getBuffers())
                prodBufResultTypes.push_back(res.getType());
            }
          }
        }
      }
    }

    // Insert combined creates after the last old SemaphoreCreateOp.
    SmallVector<SemaphoreCreateOp> allCreates;
    for (auto &pair : pairs) {
      allCreates.push_back(pair.empty);
      allCreates.push_back(pair.full);
    }
    auto lastCreate = *llvm::max_element(allCreates, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });

    auto *ctx = loop->getContext();
    int depth = pairs[0].full.getType().getNumStages();
    auto combinedType = SemaphoreType::get(
        ctx, TypeArrayAttr::get(ctx, allBufTypes), depth);

    OpBuilder builder(lastCreate);
    builder.setInsertionPointAfter(lastCreate);
    // EMPTY must appear before FULL in IR so that the greedy rewriter
    // processes FULL first.  handleTMALoads on the FULL semaphore follows
    // the producer-release token chain through the EMPTY semaphore's
    // acquire/buffer ops; those must still be live at that point.
    auto combinedEmpty = SemaphoreCreateOp::create(
        builder, lastCreate->getLoc(), combinedType, allBufs, /*isReleased=*/true);
    auto combinedFull = SemaphoreCreateOp::create(
        builder, lastCreate->getLoc(), combinedType, allBufs, /*isReleased=*/false);

    // moveUserAfter: ensure buffer consumers appear after the combined op.
    std::function<void(Operation *, Operation *)> moveUserAfter =
        [&](Operation *op, Operation *target) {
          auto *curBlock = target->getBlock();
          for (auto *user : op->getUsers()) {
            auto *userOp = curBlock->findAncestorOpInBlock(*user);
            if (userOp && userOp->isBeforeInBlock(target)) {
              userOp->moveAfter(target);
              moveUserAfter(userOp, userOp);
            }
          }
        };

    // --- Step C: replace consumer-side ops ---------------------------------
    auto firstConsAcquire =
        *llvm::min_element(acquireGroup, [](auto a, auto b) {
          return a->isBeforeInBlock(b);
        });
    auto lastConsRelease =
        *llvm::max_element(consReleaseOps, [](auto a, auto b) {
          return a->isBeforeInBlock(b);
        });
    auto consPartition = getPartitionWsTagIds(firstConsAcquire);
    auto consStage = getStageCluster(firstConsAcquire);

    builder.setInsertionPoint(firstConsAcquire);
    auto combinedConsAcquire = SemaphoreAcquireOp::create(
        builder, firstConsAcquire.getLoc(), combinedFull,
        builder.getType<AsyncTokenType>());
    assignStageCluster(combinedConsAcquire, consPartition, consStage, builder);

    builder.setInsertionPointAfter(combinedConsAcquire);
    auto combinedConsBuf = SemaphoreBufferOp::create(
        builder, firstConsAcquire.getLoc(), combinedFull,
        TypeRange(consBufResultTypes), combinedConsAcquire.getToken());
    assignStageCluster(combinedConsBuf, consPartition, consStage, builder);

    // Replace buffer results with positional indexing.
    int bufOffset = 0;
    for (auto consBufferOp : consBufferOps) {
      moveUserAfter(consBufferOp, combinedConsBuf);
      for (auto [j, oldBuf] : llvm::enumerate(consBufferOp.getBuffers()))
        oldBuf.replaceAllUsesWith(combinedConsBuf.getBuffers()[bufOffset + j]);
      bufOffset += consBufferOp.getBuffers().size();
    }

    // Combined consumer release (cross-release: targets combinedEmpty).
    llvm::SmallSetVector<Attribute, 5> consAsyncOpsSet;
    for (auto relOp : consReleaseOps)
      consAsyncOpsSet.insert(relOp.getAsyncOps().begin(),
                             relOp.getAsyncOps().end());

    builder.setInsertionPoint(lastConsRelease);
    auto combinedConsRelease = SemaphoreReleaseOp::create(
        builder, lastConsRelease.getLoc(), combinedEmpty,
        combinedConsAcquire.getToken(),
        builder.getArrayAttr(
            SmallVector<Attribute>(consAsyncOpsSet.begin(),
                                  consAsyncOpsSet.end())));
    assignStageCluster(combinedConsRelease,
                       getPartitionWsTagIds(lastConsRelease),
                       getStageCluster(lastConsRelease), builder);

    // --- Step D: replace producer-side ops ---------------------------------
    SmallVector<SemaphoreAcquireOp> prodAcquireOps;
    SmallVector<SemaphoreBufferOp> prodBufferOps;
    SmallVector<SemaphoreReleaseOp> prodReleaseOps;
    for (auto &pair : pairs) {
      for (Operation *user : pair.empty->getUsers()) {
        if (auto acq = dyn_cast<SemaphoreAcquireOp>(user))
          prodAcquireOps.push_back(acq);
      }
    }
    for (auto prodAcquire : prodAcquireOps) {
      for (Operation *tokUser : prodAcquire.getToken().getUsers()) {
        if (auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser))
          prodBufferOps.push_back(bufOp);
        if (auto relOp = dyn_cast<SemaphoreReleaseOp>(tokUser))
          prodReleaseOps.push_back(relOp);
      }
    }

    auto firstProdAcquire =
        *llvm::min_element(prodAcquireOps, [](auto a, auto b) {
          return a->isBeforeInBlock(b);
        });
    auto lastProdRelease =
        *llvm::max_element(prodReleaseOps, [](auto a, auto b) {
          return a->isBeforeInBlock(b);
        });
    auto prodPartition = getPartitionWsTagIds(firstProdAcquire);
    auto prodStage = getStageCluster(firstProdAcquire);

    builder.setInsertionPoint(firstProdAcquire);
    auto combinedProdAcquire = SemaphoreAcquireOp::create(
        builder, firstProdAcquire.getLoc(), combinedEmpty,
        builder.getType<AsyncTokenType>());
    assignStageCluster(combinedProdAcquire, prodPartition, prodStage, builder);

    builder.setInsertionPointAfter(combinedProdAcquire);
    auto combinedProdBuf = SemaphoreBufferOp::create(
        builder, firstProdAcquire.getLoc(), combinedEmpty,
        TypeRange(prodBufResultTypes), combinedProdAcquire.getToken());
    assignStageCluster(combinedProdBuf, prodPartition, prodStage, builder);

    bufOffset = 0;
    for (auto prodBufferOp : prodBufferOps) {
      moveUserAfter(prodBufferOp, combinedProdBuf);
      for (auto [j, oldBuf] : llvm::enumerate(prodBufferOp.getBuffers()))
        oldBuf.replaceAllUsesWith(combinedProdBuf.getBuffers()[bufOffset + j]);
      bufOffset += prodBufferOp.getBuffers().size();
    }

    // Combined producer release (cross-release: targets combinedFull).
    llvm::SmallSetVector<Attribute, 5> prodAsyncOpsSet;
    for (auto relOp : prodReleaseOps)
      prodAsyncOpsSet.insert(relOp.getAsyncOps().begin(),
                             relOp.getAsyncOps().end());

    builder.setInsertionPoint(lastProdRelease);
    auto combinedProdRelease = SemaphoreReleaseOp::create(
        builder, lastProdRelease.getLoc(), combinedFull,
        combinedProdAcquire.getToken(),
        builder.getArrayAttr(
            SmallVector<Attribute>(prodAsyncOpsSet.begin(),
                                  prodAsyncOpsSet.end())));
    assignStageCluster(combinedProdRelease,
                       getPartitionWsTagIds(lastProdRelease),
                       getStageCluster(lastProdRelease), builder);

    // --- Step E: erase old ops (reverse order) -----------------------------
    for (auto relOp : consReleaseOps)
      relOp->erase();
    for (auto bufOp : consBufferOps)
      bufOp->erase();
    for (auto &acqOp : acquireGroup)
      acqOp->erase();
    for (auto relOp : prodReleaseOps)
      relOp->erase();
    for (auto bufOp : prodBufferOps)
      bufOp->erase();
    for (auto acqOp : prodAcquireOps)
      acqOp->erase();
    for (auto &pair : pairs) {
      pair.full->erase();
      pair.empty->erase();
    }
  }
}

bool requiresAssignSemaphoreStagePhase(ModuleOp module) {
  bool needsAssign = false;
  module.walk([&](Operation *op) {
    if (needsAssign)
      return;
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
      needsAssign = !acquireOp.getStage() || !acquireOp.getPhase();
    } else if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(op)) {
      needsAssign = !releaseOp.getStage();
    } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(op)) {
      needsAssign = !bufferOp.getStage();
    }
  });
  return needsAssign;
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

void replaceValueUsesAndPropagateType(OpBuilder &builder, Value oldVal,
                                      Value newVal) {
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Operation *> opsToDelete;
  SmallVector<OpOperand *> operandsToReplace;

  for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
    if (!use.getOwner()->hasTrait<OpTrait::MemDescViewTrait>()) {
      operandsToReplace.push_back(&use);
      continue;
    }

    Operation *user = use.getOwner();
    builder.setInsertionPoint(user);
    Value replacement;
    if (auto subview = dyn_cast<MemDescIndexOp>(user)) {
      auto oldType = subview.getType();
      bool isMutable = cast<MemDescType>(newVal.getType()).getMutableMemory();
      auto newDstType =
          MemDescType::get(oldType.getShape(), oldType.getElementType(),
                           oldType.getEncoding(), oldType.getMemorySpace(),
                           isMutable);
      replacement = MemDescIndexOp::create(builder, subview.getLoc(), newDstType,
                                           newVal, subview.getIndex());
    } else if (auto subslice = dyn_cast<MemDescSubsliceOp>(user)) {
      auto oldType = subslice.getType();
      bool isMutable = cast<MemDescType>(newVal.getType()).getMutableMemory();
      auto newDstType = MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), isMutable, oldType.getAllocShape());
      replacement = MemDescSubsliceOp::create(builder, subslice.getLoc(),
                                              newDstType, newVal,
                                              subslice.getOffsets());
    } else if (auto trans = dyn_cast<MemDescTransOp>(user)) {
      replacement = MemDescTransOp::create(builder, trans.getLoc(), newVal,
                                           trans.getOrder());
    } else if (auto reshape = dyn_cast<MemDescReshapeOp>(user)) {
      replacement = MemDescReshapeOp::create(builder, reshape.getLoc(), newVal,
                                             reshape.getType().getShape());
    } else {
      llvm_unreachable("unhandled memdesc view op");
    }

    replacement.getDefiningOp()->setAttrs(user->getAttrs());
    replaceValueUsesAndPropagateType(builder, user->getResult(0), replacement);
    opsToDelete.push_back(user);
  }

  for (OpOperand *operand : operandsToReplace)
    operand->set(newVal);
  for (Operation *op : opsToDelete)
    op->erase();
}

void rewriteBuffer(SemaphoreBufferOp op, PatternRewriter &rewriter,
                   ArrayRef<Value> buffers) {
  auto loc = op.getLoc();
  auto partitionWsTagIds = getPartitionWsTagIds(op);
  auto stageCluster = getStageCluster(op);

  for (auto [i, buffer] : llvm::enumerate(buffers)) {
    // replacement helper may erase ops adjacent to this insertion point,
    // so refresh it for each buffer result before creating new view ops.
    rewriter.setInsertionPointAfter(op);

    auto memDesc = cast<MemDescType>(buffer.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(memDesc.getEncoding())) {
      op.getBuffers()[i].replaceAllUsesWith(buffer);
      continue;
    }

    auto shape = memDesc.getShape();
    assert(shape.size() > 1 && "expected multi-buffered semaphore buffer");
    SmallVector<int64_t> viewShape(shape.begin() + 1, shape.end());
    auto viewType = MemDescType::get(viewShape, memDesc.getElementType(),
                                     memDesc.getEncoding(),
                                     memDesc.getMemorySpace(),
                                     /*mutableMemory=*/true);
    auto view = MemDescIndexOp::create(rewriter, loc, viewType, buffer, op.getStage());
    assignStageCluster(view, partitionWsTagIds, stageCluster, rewriter);
    replaceValueUsesAndPropagateType(rewriter, op.getBuffers()[i], view);
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

    SmallVector<Operation *> sortedLoadOps(loadOps.begin(), loadOps.end());
    bool sameBlock =
        llvm::all_of(sortedLoadOps, [&](Operation *op) {
          return op->getBlock() == sortedLoadOps.front()->getBlock();
        });
    if (sameBlock) {
      llvm::sort(sortedLoadOps, [](Operation *lhs, Operation *rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
    } else {
      SetVector<Operation *> opSet(sortedLoadOps.begin(), sortedLoadOps.end());
      auto topo = topologicalSort(opSet);
      sortedLoadOps.assign(topo.begin(), topo.end());
    }

    auto partitionWsTagIds = getPartitionWsTagIds(releaseOp);
    auto stageCluster = getStageCluster(releaseOp);

    rewriter.setInsertionPoint(sortedLoadOps.front());
    auto mbar = createSingleBufferView(rewriter, mbars, releaseOp.getStage());
    assignStageCluster(mbar.getDefiningOp(), partitionWsTagIds, stageCluster,
                       rewriter);

    auto pred = arith::ConstantIntOp::create(rewriter, op.getLoc(), 1, 1);
    assignStageCluster(pred, partitionWsTagIds, stageCluster, rewriter);

    auto expectOp =
        BarrierExpectOp::create(rewriter, op.getLoc(), mbar, txCount, pred);
    assignStageCluster(expectOp, partitionWsTagIds, stageCluster, rewriter);

    for (Operation *loadOp : sortedLoadOps) {
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
      const llvm::DenseMap<Operation *, bool> &hasAsyncPeerBySema,
      unsigned defaultNumStages, bool enableAsyncMarking)
      : OpRewritePattern<SemaphoreCreateOp>(ctx),
        hasAsyncPeerBySema(hasAsyncPeerBySema),
        defaultNumStages(defaultNumStages),
        enableAsyncMarking(enableAsyncMarking) {}

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (enableAsyncMarking) {
      for (Operation *user : op->getUsers()) {
        auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
        if (!releaseOp)
          continue;
        auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
        if (llvm::any_of(kinds, [](AsyncOp kind) {
              return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
            })) {
          for (auto mma : getAsyncMMAv5Consumers(op.getResult()))
            setIsAsync(mma, defaultNumStages);
          break;
        }
      }
    }

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
    // Poison tokens may be yielded by ws-loops and PartitionLoops requires
    // all ops to carry partition annotations.  Copy from the semaphore.
    if (hasPartition(op))
      setPartition(replToken, getPartitionIds(op));
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
  unsigned defaultNumStages;
  bool enableAsyncMarking;
};

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
  using impl::NVWSLowerSemaphoreBase<
      NVWSLowerSemaphore>::NVWSLowerSemaphoreBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    bool phase2Mode = requiresAssignSemaphoreStagePhase(m);
    if (phase2Mode) {
      // Combine semaphores sharing the same dominant consumer.
      SmallVector<scf::ForOp> loops;
      m.walk([&](scf::ForOp loop) {
        if (loop->hasAttr(triton::kWarpSpecializeAttrName))
          loop->walk([&](scf::ForOp op) { loops.push_back(op); });
      });
      for (scf::ForOp loop : loops)
        combineSemaphores(loop);

      multiBufferSemaphore(m, numStages);
      OpPassManager pm("builtin.module");
      pm.addPass(createNVWSAssignSemaphoreStagePhase());
      if (failed(runPipeline(pm, m)))
        return signalPassFailure();
    }

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
    patterns.add<LowerSemaphoreCreate>(context, hasAsyncPeerBySema, numStages,
                                       phase2Mode);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(m, std::move(patterns), config)))
      signalPassFailure();

    // Hoist all poison ops to the top of function.  They are unannotated and
    // will trip subsequent passes (e.g. PartitionLoops), so hoist them.
    m.walk([&](triton::FuncOp funcOp) {
      auto *block = &funcOp.getBody().front();
      funcOp.walk(
          [&](ub::PoisonOp op) { op->moveBefore(&block->front()); });
    });
  }
};

} // namespace

} // namespace triton
} // namespace mlir
