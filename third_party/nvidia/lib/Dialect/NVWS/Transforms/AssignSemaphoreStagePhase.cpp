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

#include "SemaphoreUtilities.h"
#include "Utilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSASSIGNSEMAPHORESTAGEPHASE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

struct PartitionWsTagIds {
  std::optional<int> wsTag;
  std::optional<SetVector<int>> partitionIds;
  StageCluster stageCluster;
};

PartitionWsTagIds
getPartitionWsTagIds(Operation *op,
                     std::optional<SetVector<int>> partitionIds = std::nullopt) {
  PartitionWsTagIds ids;
  ids.partitionIds = partitionIds;
  if (!ids.partitionIds && hasPartition(op))
    ids.partitionIds = getPartitionIds(op);
  if (auto wsTag = getWarpSpecializeTag(op))
    ids.wsTag = *wsTag;
  ids.stageCluster = getStageCluster(op);
  return ids;
}

template <typename OpT, typename... Args>
OpT createInto(ImplicitLocOpBuilder &b, PartitionWsTagIds partitionWsTagIds,
               Args &&...args) {
  auto op = triton::gpu::createInto<OpT>(
      b, b.getLoc(), partitionWsTagIds.partitionIds,
      partitionWsTagIds.stageCluster, std::forward<Args>(args)...);
  if (partitionWsTagIds.partitionIds && partitionWsTagIds.wsTag)
    setWarpSpecializeTag(op, *partitionWsTagIds.wsTag);
  return op;
}

SetVector<int> inferPartitionIds(Value value) {
  SetVector<int> ids;
  if (auto defOp = value.getDefiningOp()) {
    if (defOp->getNumRegions() == 0) {
      if (hasPartition(defOp)) {
        auto defIds = getPartitionIds(defOp);
        ids.insert(defIds.begin(), defIds.end());
      }
    } else if (auto pos = findValuePosInRange(defOp->getResults(), value)) {
      auto outputs = getPartitionOutputs(defOp);
      if (*pos < outputs.size()) {
        auto outIds = outputs[*pos];
        ids.insert(outIds.begin(), outIds.end());
      }
    }
  } else {
    for (Operation *user : value.getUsers()) {
      if (isa<scf::YieldOp>(user))
        continue;
      if (hasPartition(user)) {
        auto userIds = getPartitionIds(user);
        ids.insert(userIds.begin(), userIds.end());
      }
    }
  }
  return ids;
}

class AssignSemaphoreStage {
public:
  struct State {
    Value stage;
    Value wasObserved;
  };

  enum class AccessKind {
    None,
    Observation,
    FreshWrite,
    FreshWriteMMA,
  };

  AssignSemaphoreStage(FuncOp funcOp, ArrayRef<SemaphoreCreateOp> semaphores)
      : funcOp(funcOp), semaphores(semaphores.begin(), semaphores.end()) {
    for (auto sema : this->semaphores)
      semaphoreValues.insert(sema.getResult());

    auto semaType = cast<SemaphoreType>(this->semaphores.front().getType());
    depth = std::max(1, semaType.getNumStages());

    collectGroupPartitionIds();
  }

  LogicalResult run() {
    if (semaphores.empty())
      return success();

    ImplicitLocOpBuilder b(semaphores.front().getLoc(), semaphores.front());
    b.setInsertionPointAfter(semaphores.front());

    auto info = getGroupInfo(semaphores.front().getOperation());
    auto stage0 = createInto<arith::ConstantIntOp>(b, info, 0, 32);
    auto observed0 = createInto<arith::ConstantIntOp>(b, info, 0, 1);

    State state{stage0, observed0};
    auto *entry = &funcOp.getBody().front();
    assignStateInBlock(entry, state);

    DenseSet<Operation *> visited;
    SmallVector<SemaphoreAcquireOp> acquireOps;
    funcOp.walk([&](SemaphoreAcquireOp acquireOp) {
      if (isGroupSemaphore(acquireOp.getSemaphore()))
        acquireOps.push_back(acquireOp);
    });
    for (auto acquireOp : acquireOps)
      propagateStage(acquireOp.getToken(), acquireOp.getStage(), visited);

    return success();
  }

private:
  bool isGroupSemaphore(Value semaphore) const {
    return semaphoreValues.contains(semaphore);
  }

  bool isGroupToken(Value token) {
    if (!isa<AsyncTokenType>(token.getType()))
      return false;
    if (auto it = tokenMemo.find(token); it != tokenMemo.end())
      return it->second;
    if (!tokenVisited.insert(token).second)
      return false;

    bool result = false;
    if (auto acquireOp = token.getDefiningOp<SemaphoreAcquireOp>()) {
      result = isGroupSemaphore(acquireOp.getSemaphore());
    } else if (auto blockArg = dyn_cast<BlockArgument>(token)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        if (auto pos = findValuePosInRange(forOp.getRegionIterArgs(), token))
          result = isGroupToken(forOp.getInitArgs()[*pos]);
      }
    } else if (auto *defOp = token.getDefiningOp()) {
      if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
        unsigned idx = cast<OpResult>(token).getResultNumber();
        if (idx < forOp.getYieldedValues().size())
          result = isGroupToken(forOp.getYieldedValues()[idx]);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        unsigned idx = cast<OpResult>(token).getResultNumber();
        if (idx < ifOp.thenYield()->getNumOperands())
          result = isGroupToken(ifOp.thenYield()->getOperand(idx));
        if (!result && ifOp.elseBlock() &&
            idx < ifOp.elseYield()->getNumOperands()) {
          result = isGroupToken(ifOp.elseYield()->getOperand(idx));
        }
      }
    }

    tokenVisited.erase(token);
    tokenMemo[token] = result;
    return result;
  }

  bool isGroupView(Value bufferView) {
    if (!isa<MemDescType>(bufferView.getType()))
      return false;
    if (auto it = viewMemo.find(bufferView); it != viewMemo.end())
      return it->second;
    if (!viewVisited.insert(bufferView).second)
      return false;

    bool result = false;
    if (auto semaBuffer = bufferView.getDefiningOp<SemaphoreBufferOp>()) {
      result = isGroupSemaphore(semaBuffer.getSemaphore());
    } else if (auto blockArg = dyn_cast<BlockArgument>(bufferView)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        if (auto pos = findValuePosInRange(forOp.getRegionIterArgs(), bufferView))
          result = isGroupView(forOp.getInitArgs()[*pos]);
      }
    } else if (auto *defOp = bufferView.getDefiningOp()) {
      if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
        for (Value operand : defOp->getOperands()) {
          if (!isa<MemDescType>(operand.getType()))
            continue;
          if (isGroupView(operand)) {
            result = true;
            break;
          }
        }
      } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
        unsigned idx = cast<OpResult>(bufferView).getResultNumber();
        if (idx < forOp.getYieldedValues().size())
          result = isGroupView(forOp.getYieldedValues()[idx]);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
        unsigned idx = cast<OpResult>(bufferView).getResultNumber();
        if (idx < ifOp.thenYield()->getNumOperands())
          result = isGroupView(ifOp.thenYield()->getOperand(idx));
        if (!result && ifOp.elseBlock() &&
            idx < ifOp.elseYield()->getNumOperands()) {
          result = isGroupView(ifOp.elseYield()->getOperand(idx));
        }
      }
    }

    viewVisited.erase(bufferView);
    viewMemo[bufferView] = result;
    return result;
  }

  AccessKind classifyAccess(Operation *op) {
    if (auto loadOp = dyn_cast<LocalLoadOp>(op)) {
      if (isGroupView(loadOp.getSrc()))
        return AccessKind::Observation;
      return AccessKind::None;
    }

    if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
      if (isGroupView(loadOp.getSrc()))
        return AccessKind::Observation;
      return AccessKind::None;
    }

    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (isGroupView(mmaOp.getAccumulator()))
        return AccessKind::FreshWriteMMA;
      if (isGroupView(mmaOp.getA()) || isGroupView(mmaOp.getB()))
        return AccessKind::Observation;
      return AccessKind::None;
    }

    if (auto storeOp = dyn_cast<LocalStoreOp>(op)) {
      if (isGroupView(storeOp.getDst()))
        return AccessKind::FreshWrite;
      return AccessKind::None;
    }

    if (auto descLoad = dyn_cast<DescriptorLoadOp>(op)) {
      if (isGroupView(descLoad.getResult()))
        return AccessKind::FreshWrite;
      return AccessKind::None;
    }

    if (auto descGather = dyn_cast<DescriptorGatherOp>(op)) {
      if (isGroupView(descGather.getResult()))
        return AccessKind::FreshWrite;
      return AccessKind::None;
    }

    if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
      if (isGroupView(storeOp.getDst()))
        return AccessKind::FreshWrite;
      return AccessKind::None;
    }

    return AccessKind::None;
  }

  bool hasGroupUseInBlock(Block *block) {
    for (Operation &op : *block) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
        if (isGroupSemaphore(acquireOp.getSemaphore()))
          return true;
      }
      if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(op)) {
        if (isGroupSemaphore(releaseOp.getSemaphore()))
          return true;
      }
      if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(op)) {
        if (isGroupSemaphore(bufferOp.getSemaphore()))
          return true;
      }

      if (classifyAccess(&op) != AccessKind::None)
        return true;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (hasGroupUseInBlock(forOp.getBody()))
          return true;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (hasGroupUseInBlock(ifOp.thenBlock()))
          return true;
        if (ifOp.elseBlock() && hasGroupUseInBlock(ifOp.elseBlock()))
          return true;
      }
    }
    return false;
  }

  void appendLoopPartitionOutputs(scf::ForOp forOp,
                                  ArrayRef<Value> newOutputValues) {
    if (!hasPartition(forOp))
      return;

    auto forIds = getPartitionIds(forOp);
    auto forOutputs = getPartitionOutputs(forOp);
    for (Value value : newOutputValues) {
      auto ids = inferPartitionIds(value);
      forIds.insert(ids.begin(), ids.end());
      forOutputs.push_back(ids);
    }
    setPartition(forOp, forIds);
    setPartitionOutputs(forOp, forOutputs);
  }

  void appendIfPartitionOutputs(scf::IfOp ifOp, SetVector<int> ifIds,
                                SmallVector<SetVector<int>, 4> ifOutputs,
                                State thenState, State elseState) {
    SetVector<int> stageIds = inferPartitionIds(thenState.stage);
    auto elseStageIds = inferPartitionIds(elseState.stage);
    stageIds.insert(elseStageIds.begin(), elseStageIds.end());

    SetVector<int> observedIds = inferPartitionIds(thenState.wasObserved);
    auto elseObservedIds = inferPartitionIds(elseState.wasObserved);
    observedIds.insert(elseObservedIds.begin(), elseObservedIds.end());

    ifIds.insert(stageIds.begin(), stageIds.end());
    ifIds.insert(observedIds.begin(), observedIds.end());

    ifOutputs.push_back(stageIds);
    ifOutputs.push_back(observedIds);

    setPartition(ifOp, ifIds);
    setPartitionOutputs(ifOp, ifOutputs);
  }

  void assignStateInForOp(scf::ForOp forOp, State &state) {
    if (!hasGroupUseInBlock(forOp.getBody()))
      return;

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, {state.stage, state.wasObserved});

    for (size_t idx = 0; idx < nArgs; ++idx) {
      Value initArg = forOp.getInitArgs()[idx];
      if (!isGroupToken(initArg))
        continue;
      Value iterTok = forOp.getRegionIterArgs()[idx];
      tokToStagePosMap[{forOp.getOperation(), iterTok}] = nArgs;
    }

    State loopState{forOp.getRegionIterArgs()[nArgs],
                    forOp.getRegionIterArgs()[nArgs + 1]};
    loopState = assignStateInBlock(forOp.getBody(), loopState);

    appendToForOpYield(forOp, {loopState.stage, loopState.wasObserved});

    auto *yieldOp = forOp.getBody()->getTerminator();
    for (size_t idx = 0; idx < nArgs; ++idx) {
      Value initArg = forOp.getInitArgs()[idx];
      if (!isGroupToken(initArg))
        continue;
      tokToStagePosMap[{yieldOp, yieldOp->getOperand(idx)}] = nArgs;
    }

    appendLoopPartitionOutputs(forOp, {loopState.stage, loopState.wasObserved});

    state.stage = forOp.getResult(nArgs);
    state.wasObserved = forOp.getResult(nArgs + 1);
  }

  void assignStateInIfOp(scf::IfOp ifOp, State &state) {
    bool useThen = hasGroupUseInBlock(ifOp.thenBlock());
    bool useElse = ifOp.elseBlock() ? hasGroupUseInBlock(ifOp.elseBlock()) : false;
    if (!useThen && !useElse)
      return;

    auto oldIfIds = hasPartition(ifOp) ? std::optional(getPartitionIds(ifOp))
                                       : std::nullopt;
    auto oldIfOutputs = hasPartition(ifOp)
                            ? std::optional(getPartitionOutputs(ifOp))
                            : std::nullopt;

    OpBuilder builder(ifOp);
    size_t nResults = ifOp.getNumResults();
    auto newIfOp = replaceIfOpWithNewSignature(
        builder, ifOp, TypeRange{state.stage.getType(), state.wasObserved.getType()});

    State thenState =
        useThen ? assignStateInBlock(newIfOp.thenBlock(), state) : state;
    State elseState =
        (newIfOp.elseBlock() && useElse)
            ? assignStateInBlock(newIfOp.elseBlock(), state)
            : state;

    auto thenYield = newIfOp.thenYield();
    auto elseYield = newIfOp.elseYield();

    size_t stagePos = thenYield.getNumOperands();
    for (Value value : thenYield->getOperands()) {
      if (isGroupToken(value))
        tokToStagePosMap[{thenYield.getOperation(), value}] = stagePos;
    }
    for (Value value : elseYield->getOperands()) {
      if (isGroupToken(value))
        tokToStagePosMap[{elseYield.getOperation(), value}] = stagePos;
    }

    thenYield->insertOperands(thenYield.getNumOperands(),
                              {thenState.stage, thenState.wasObserved});
    elseYield->insertOperands(elseYield.getNumOperands(),
                              {elseState.stage, elseState.wasObserved});

    ifOp.erase();

    if (oldIfIds && oldIfOutputs)
      appendIfPartitionOutputs(newIfOp, *oldIfIds, *oldIfOutputs, thenState,
                               elseState);

    state.stage = newIfOp.getResult(nResults);
    state.wasObserved = newIfOp.getResult(nResults + 1);
  }

  State assignStateInBlock(Block *block, State state) {
    for (Operation &op : llvm::make_early_inc_range(*block)) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
        if (isGroupSemaphore(acquireOp.getSemaphore()))
          acquireOp.getStageMutable().assign(state.stage);
      }

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignStateInForOp(forOp, state);
        continue;
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignStateInIfOp(ifOp, state);
        continue;
      }

      auto access = classifyAccess(&op);
      if (access == AccessKind::None)
        continue;

      ImplicitLocOpBuilder b(op.getLoc(), &op);
      b.setInsertionPointAfter(&op);
      auto info = getGroupInfo(&op);

      if (access == AccessKind::Observation) {
        state.wasObserved = createInto<arith::ConstantIntOp>(b, info, 1, 1);
        continue;
      }

      Value isFresh;
      if (access == AccessKind::FreshWrite) {
        isFresh = createInto<arith::ConstantIntOp>(b, info, 1, 1);
      } else {
        auto mmaOp = cast<MMAv5OpInterface>(&op);
        auto c1 = createInto<arith::ConstantIntOp>(b, info, 1, 1);
        isFresh = createInto<arith::XOrIOp>(b, info, mmaOp.useAccumulator(), c1);
      }

      auto shouldAdvance = createInto<arith::AndIOp>(b, info, state.wasObserved,
                                                     isFresh);
      auto c1_i32 = createInto<arith::ConstantIntOp>(b, info, 1, 32);
      auto c0_i32 = createInto<arith::ConstantIntOp>(b, info, 0, 32);
      auto cDepth = createInto<arith::ConstantIntOp>(b, info, depth, 32);
      auto cFalse = createInto<arith::ConstantIntOp>(b, info, 0, 1);

      auto next = createInto<arith::AddIOp>(b, info, state.stage, c1_i32);
      auto wrap = createInto<arith::CmpIOp>(
          b, info, arith::CmpIPredicate::eq, next, cDepth);
      auto wrapped = createInto<arith::SelectOp>(b, info, wrap, c0_i32, next);

      state.stage =
          createInto<arith::SelectOp>(b, info, shouldAdvance, wrapped, state.stage);
      state.wasObserved = createInto<arith::SelectOp>(
          b, info, shouldAdvance, cFalse, state.wasObserved);
    }

    return state;
  }

  void collectGroupPartitionIds() {
    funcOp.walk([&](Operation *op) {
      if (!hasPartition(op))
        return;

      bool include = false;
      if (auto semaCreate = dyn_cast<SemaphoreCreateOp>(op)) {
        include = isGroupSemaphore(semaCreate.getResult());
      } else if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
        include = isGroupSemaphore(acquireOp.getSemaphore());
      } else if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(op)) {
        include = isGroupSemaphore(releaseOp.getSemaphore());
      } else if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(op)) {
        include = isGroupSemaphore(bufferOp.getSemaphore());
      } else {
        include = classifyAccess(op) != AccessKind::None;
      }

      if (!include)
        return;
      auto ids = getPartitionIds(op);
      groupPartitionIds.insert(ids.begin(), ids.end());
    });
  }

  PartitionWsTagIds getGroupInfo(Operation *anchor) {
    std::optional<SetVector<int>> ids;
    if (!groupPartitionIds.empty())
      ids = groupPartitionIds;
    return getPartitionWsTagIds(anchor, ids);
  }

  void propagateStage(Value token, Value stage, DenseSet<Operation *> &visited) {
    for (auto &tokUse : token.getUses()) {
      auto *owner = tokUse.getOwner();
      if (!visited.insert(owner).second)
        continue;

      if (auto stageOp = dyn_cast<ArefStageInterface>(owner)) {
        if (auto blockArg = dyn_cast<BlockArgument>(stage)) {
          auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
          if (forOp && hasPartition(stageOp) && hasPartition(forOp)) {
            auto stageOpIds = getPartitionIds(stageOp);
            auto forIds = getPartitionIds(forOp);
            forIds.insert(stageOpIds.begin(), stageOpIds.end());
            setPartition(forOp, forIds);

            if (auto pos = findValuePosInRange(forOp.getRegionIterArgs(), stage)) {
              auto forOutputs = getPartitionOutputs(forOp);
              if (*pos < forOutputs.size()) {
                forOutputs[*pos].insert(stageOpIds.begin(), stageOpIds.end());
                if (*pos + 1 < forOutputs.size()) {
                  forOutputs[*pos + 1].insert(stageOpIds.begin(),
                                              stageOpIds.end());
                }
                setPartitionOutputs(forOp, forOutputs);
              }
            }
          }
        }
        stageOp.setStage(stage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber() - forOp.getNumControlOperands();
        auto iterTok = forOp.getRegionIterArg(tokPos);
        auto it = tokToStagePosMap.find({forOp.getOperation(), iterTok});
        if (it == tokToStagePosMap.end())
          continue;
        propagateStage(iterTok, forOp.getRegionIterArgs()[it->second], visited);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber();
        auto it = tokToStagePosMap.find({yieldOp.getOperation(), token});
        if (it == tokToStagePosMap.end())
          continue;
        auto *parentOp = yieldOp->getParentOp();
        propagateStage(parentOp->getResult(tokPos),
                       parentOp->getResult(it->second), visited);
      }
    }
  }

  FuncOp funcOp;
  SmallVector<SemaphoreCreateOp> semaphores;
  DenseSet<Value> semaphoreValues;
  SetVector<int> groupPartitionIds;
  int depth = 1;

  DenseMap<Value, bool> tokenMemo;
  DenseSet<Value> tokenVisited;
  DenseMap<Value, bool> viewMemo;
  DenseSet<Value> viewVisited;
  DenseMap<std::pair<Operation *, Value>, int> tokToStagePosMap;
};

void updateOutputWithDefaultPartition(Operation *op, int pos) {
  auto opIds = getPartitionIds(op);
  opIds.insert(0);
  setPartition(op, opIds);

  auto opOutputsIds = getPartitionOutputs(op);
  opOutputsIds[pos].insert(0);
  setPartitionOutputs(op, opOutputsIds);
}

void visitBackwardSlice(scf::ForOp wsLoop, Value value,
                        std::function<void(Operation *)> callback,
                        DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return;
      auto pos = findValuePosInRange(forOp.getRegionIterArgs(), value);
      if (!pos)
        return;
      visitBackwardSlice(wsLoop, forOp.getInitArgs()[*pos], callback, visited);
    }
  } else if (auto defOp = value.getDefiningOp();
             isa<scf::IfOp, scf::ForOp>(defOp)) {
    auto pos = findValuePosInRange(defOp->getResults(), value);
    if (!pos)
      return;
    updateOutputWithDefaultPartition(defOp, *pos);
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      visitBackwardSlice(wsLoop, ifOp.thenYield()->getOperand(*pos), callback,
                         visited);
      if (ifOp.elseBlock())
        visitBackwardSlice(wsLoop, ifOp.elseYield()->getOperand(*pos), callback,
                           visited);
      visitBackwardSlice(wsLoop, ifOp.getCondition(), callback, visited);
    } else {
      auto forOp = cast<scf::ForOp>(defOp);
      visitBackwardSlice(wsLoop,
                         forOp.getBody()->getTerminator()->getOperand(*pos),
                         callback, visited);
      for (int idx = 0; idx < forOp.getNumControlOperands(); ++idx)
        visitBackwardSlice(wsLoop, forOp.getOperand(idx), callback, visited);
    }
  } else if (auto *defOp = value.getDefiningOp();
             defOp && wsLoop.getBody()->findAncestorOpInBlock(*defOp)) {
    callback(defOp);
    for (Value operand : defOp->getOperands())
      visitBackwardSlice(wsLoop, operand, callback, visited);
  }
}

LogicalResult assignSemaphorePhase(FuncOp funcOp) {
  auto initPhase = [](ImplicitLocOpBuilder &b, Operation *op) -> Value {
    auto sema = cast<SemaphoreCreateOp>(op);
    uint32_t init = sema.getIsReleased() ? 0xFFFFFFFFu : 0x00000000u;
    auto info = getPartitionWsTagIds(op);
    return createInto<arith::ConstantIntOp>(b, info, static_cast<int64_t>(init),
                                            32);
  };

  auto updatePhase = [](ImplicitLocOpBuilder &b, Value phase,
                        Operation *op) -> Value {
    auto acquireOp = cast<SemaphoreAcquireOp>(op);
    acquireOp.getPhaseMutable().assign(phase);

    auto info = getPartitionWsTagIds(op);
    auto c1 = createInto<arith::ConstantIntOp>(b, info, 1, 32);
    auto phaseBit = createInto<arith::ShLIOp>(b, info, c1, acquireOp.getStage());
    return createInto<arith::XOrIOp>(b, info, phase, phaseBit);
  };

  SmallVector<WarpGroupOp> wgOps;
  funcOp.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

  if (!wgOps.empty()) {
    for (auto wgOp : wgOps)
      ThreadValue<SemaphoreAcquireOp>::run(wgOp, initPhase, updatePhase);
    return success();
  }

  ThreadValue<SemaphoreAcquireOp> threadValue{updatePhase};
  auto *entry = &funcOp.getBody().front();
  auto useSet = threadValue.analyzeUseInBlock(entry, {});

  ThreadValue<SemaphoreAcquireOp>::ValueMap valueMap;
  for (auto key : useSet) {
    if (auto *def = key.getDefiningOp()) {
      ImplicitLocOpBuilder b(key.getLoc(), def);
      b.setInsertionPointAfter(def);
      valueMap[key] = initPhase(b, def);
    }
  }

  threadValue.assignValueInBlock(entry, valueMap);
  return success();
}

LogicalResult assignSemaphoreStagePhase(FuncOp funcOp) {
  DenseMap<Value, SmallVector<SemaphoreCreateOp>> bufferGroups;
  funcOp.walk([&](SemaphoreCreateOp op) {
    if (op.getBuffers().empty())
      return;
    bufferGroups[op.getBuffers()[0]].push_back(op);
  });

  for (auto &it : bufferGroups) {
    AssignSemaphoreStage assignStage(funcOp, it.second);
    if (failed(assignStage.run()))
      return failure();
  }

  if (failed(assignSemaphorePhase(funcOp)))
    return failure();

  auto callback = [&](Operation *op) {
    if (!isa<scf::YieldOp, scf::IfOp, scf::ForOp, triton::ReduceOp>(op) &&
        hasPartition(op)) {
      auto partitionIds = getPartitionIds(op);
      partitionIds.insert(0);
      setPartition(op, partitionIds);
    }
  };

  funcOp.walk([&](scf::ForOp forOp) {
    DenseSet<Value> visited;
    if (!forOp->hasAttr(kWarpSpecializeAttrName))
      return;

    for (auto result : forOp.getResults()) {
      if (!isa<IntegerType, FloatType>(result.getType()) || result.use_empty())
        continue;

      bool assignDefaultPartition =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            return !hasPartition(user) ||
                   (isa<scf::ForOp>(user) && hasWarpSpecializeTag(user));
          });
      if (!assignDefaultPartition)
        continue;

      updateOutputWithDefaultPartition(forOp, result.getResultNumber());
      auto arg = forOp.getBody()->getTerminator()->getOperand(
          result.getResultNumber());
      visitBackwardSlice(forOp, arg, callback, visited);
    }
  });

  return success();
}

class NVWSAssignSemaphoreStagePhase
    : public impl::NVWSAssignSemaphoreStagePhaseBase<
          NVWSAssignSemaphoreStagePhase> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](FuncOp funcOp) {
      if (failed(assignSemaphoreStagePhase(funcOp)))
        signalPassFailure();
    });
  }
};

} // namespace

} // namespace triton
} // namespace mlir
