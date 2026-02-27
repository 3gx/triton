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
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
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

// ---------------------------------------------------------------------------
// Per-partition processing, exactly like aref AssignStagePhase<T>.
// State = {stage, wasObserved, phase, token} per semaphore per partition.
// Each (semaphore, partition) pair gets its own iter_args through loops/ifs.
// ---------------------------------------------------------------------------

enum class AccessKind { None, Observation, FreshWrite, FreshWriteMMA };

struct AssignSemaphoreStagePhase {
  struct State {
    Value stage;
    Value wasObserved;
    Value phase;
    Value token;
  };

  Value semaphore;
  int partitionId;
  int depth;
  DenseMap<std::pair<Operation *, Value>, int> tokToStagePosMap;

  AssignSemaphoreStagePhase(Value semaphore, int partitionId, int depth)
      : semaphore(semaphore), partitionId(partitionId), depth(depth) {}

  // --- Op matching ---------------------------------------------------------

  SemaphoreAcquireOp getAcquireOp(Operation *op) {
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
      if (acquireOp.getSemaphore() == semaphore) {
        if (!hasPartition(op) ||
            llvm::is_contained(getPartitionIds(op), partitionId))
          return acquireOp;
      }
    }
    return {};
  }

  bool isGroupBuffer(SemaphoreBufferOp bufOp, Value token) {
    if (!bufOp)
      return false;
    if (bufOp.getSemaphore() != this->semaphore)
      return false;
    return token == bufOp.getToken();
  }

  bool isGroupView(Value bufferView) {
    if (!isa<MemDescType>(bufferView.getType()))
      return false;
    if (auto semaBuffer = bufferView.getDefiningOp<SemaphoreBufferOp>())
      return semaBuffer.getSemaphore() == this->semaphore;
    if (auto blockArg = dyn_cast<BlockArgument>(bufferView)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()))
        if (auto pos =
                findValuePosInRange(forOp.getRegionIterArgs(), bufferView))
          return isGroupView(forOp.getInitArgs()[*pos]);
    } else if (auto *defOp = bufferView.getDefiningOp()) {
      if (defOp->hasTrait<OpTrait::MemDescViewTrait>())
        for (Value operand : defOp->getOperands())
          if (isa<MemDescType>(operand.getType()) && isGroupView(operand))
            return true;
    }
    return false;
  }

  // --- Access classification -----------------------------------------------

  AccessKind classifyAccess(Operation *op) {
    if (auto loadOp = dyn_cast<LocalLoadOp>(op))
      return isGroupView(loadOp.getSrc()) ? AccessKind::Observation
                                          : AccessKind::None;
    if (auto loadOp = dyn_cast<TMEMLoadOp>(op))
      return isGroupView(loadOp.getSrc()) ? AccessKind::Observation
                                          : AccessKind::None;
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (isGroupView(mmaOp.getAccumulator()))
        return AccessKind::FreshWriteMMA;
      if (isGroupView(mmaOp.getA()) || isGroupView(mmaOp.getB()))
        return AccessKind::Observation;
      return AccessKind::None;
    }
    if (auto storeOp = dyn_cast<LocalStoreOp>(op))
      return isGroupView(storeOp.getDst()) ? AccessKind::FreshWrite
                                           : AccessKind::None;
    if (auto descLoad = dyn_cast<DescriptorLoadOp>(op))
      return isGroupView(descLoad.getResult()) ? AccessKind::FreshWrite
                                               : AccessKind::None;
    if (auto descGather = dyn_cast<DescriptorGatherOp>(op))
      return isGroupView(descGather.getResult()) ? AccessKind::FreshWrite
                                                 : AccessKind::None;
    if (auto storeOp = dyn_cast<TMEMStoreOp>(op))
      return isGroupView(storeOp.getDst()) ? AccessKind::FreshWrite
                                           : AccessKind::None;
    return AccessKind::None;
  }

  bool isInPartition(Operation *op) {
    return !hasPartition(op) ||
           llvm::is_contained(getPartitionIds(op), partitionId);
  }

  // --- Block analysis ------------------------------------------------------

  bool analyzeUseInBlock(Block *block, Value token) {
    for (auto &op : *block) {
      if (getAcquireOp(&op) ||
          isGroupBuffer(dyn_cast<SemaphoreBufferOp>(op), token) ||
          (isInPartition(&op) && classifyAccess(&op) != AccessKind::None))
        return true;
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        Value newTok;
        if (auto pos = findValuePosInRange(forOp.getInitArgs(), token))
          newTok = forOp.getRegionIterArgs()[*pos];
        if (analyzeUseInBlock(forOp.getBody(), newTok))
          return true;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (analyzeUseInBlock(ifOp.thenBlock(), token))
          return true;
        if (ifOp.elseBlock() && analyzeUseInBlock(ifOp.elseBlock(), token))
          return true;
      }
    }
    return false;
  }

  // --- Control-flow threading (verbatim aref pattern) ----------------------

  void assignStateInForOp(scf::ForOp forOp, State &state) {
    Value newTok;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token))
      newTok = forOp.getRegionIterArgs()[*pos];
    if (!analyzeUseInBlock(forOp.getBody(), newTok))
      return;

    SmallVector<Value> extraIterArgs{state.stage, state.wasObserved,
                                     state.phase};
    SmallVector<Value *> stateRefs{&state.stage, &state.wasObserved,
                                   &state.phase};
    llvm::MapVector<int, Value *> tokenRefs;

    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token)) {
      tokenRefs[*pos] = &state.token;
      state.token = forOp.getRegionIterArgs()[*pos];
    }

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();

    assert(hasPartition(forOp));
    auto forOpIds = getPartitionIds(forOp);
    auto forOpOutputsIds = getPartitionOutputs(forOp);
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *stateRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    auto stateInBlock = assignStateInBlock(forOp.getBody(), state);

    SmallVector<Value> extraYieldArgs{stateInBlock.stage,
                                      stateInBlock.wasObserved,
                                      stateInBlock.phase};
    appendToForOpYield(forOp, extraYieldArgs);
    tokToStagePosMap[{forOp, state.token}] = nArgs;
    tokToStagePosMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        nArgs;

    // Partition annotation — verbatim from aref
    for (auto arg : extraYieldArgs) {
      SetVector<int> argIds;
      if (auto defOp = arg.getDefiningOp()) {
        if (defOp->getNumRegions() == 0) {
          if (hasPartition(defOp))
            argIds = getPartitionIds(defOp);
        } else {
          auto pos = findValuePosInRange(defOp->getResults(), arg);
          if (pos) {
            auto outputs = getPartitionOutputs(defOp);
            if (*pos < outputs.size())
              argIds = outputs[*pos];
          }
        }
      } else {
        for (auto user : arg.getUsers()) {
          if (isa<scf::YieldOp>(user))
            continue;
          if (hasPartition(user)) {
            auto ids = getPartitionIds(user);
            argIds.insert(ids.begin(), ids.end());
          }
        }
      }
      if (argIds.empty())
        argIds = forOpIds;
      forOpIds.insert(argIds.begin(), argIds.end());
      forOpOutputsIds.push_back(argIds);
    }
    setPartition(forOp, forOpIds);
    setPartitionOutputs(forOp, forOpOutputsIds);

    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *stateRefs[idx - nArgs] = forOp.getResult(idx);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = forOp.getResult(idx);
  }

  void assignStateInIfOp(scf::IfOp ifOp, State &state) {
    auto useInThen = analyzeUseInBlock(ifOp.thenBlock(), state.token);
    auto useInElse =
        ifOp.elseBlock() ? analyzeUseInBlock(ifOp.elseBlock(), state.token)
                         : false;
    if (!useInThen && !useInElse)
      return;

    SmallVector<Type> extraIfResults{state.stage.getType(),
                                     state.wasObserved.getType(),
                                     state.phase.getType()};
    SmallVector<Value *> stateRefs{&state.stage, &state.wasObserved,
                                   &state.phase};

    OpBuilder builder(ifOp);
    size_t nResults = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    auto thenState = assignStateInBlock(newIfOp.thenBlock(), state);
    auto elseState = newIfOp.elseBlock()
                         ? assignStateInBlock(newIfOp.elseBlock(), state)
                         : state;

    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();

    llvm::MapVector<int, Value *> tokenRefs;
    if (auto pos =
            findValuePosInRange(thenYieldOp->getOperands(), state.token))
      tokenRefs[*pos] = &state.token;
    if (auto pos =
            findValuePosInRange(elseYieldOp->getOperands(), state.token))
      tokenRefs[*pos] = &state.token;

    tokToStagePosMap[{newIfOp.thenYield(), thenState.token}] =
        thenYieldOp.getNumOperands();
    tokToStagePosMap[{newIfOp.elseYield(), elseState.token}] =
        elseYieldOp.getNumOperands();

    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.stage);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.stage);
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                thenState.wasObserved);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                elseState.wasObserved);
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.phase);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.phase);

    // Partition annotation — verbatim from aref. NEVER widen ifOpIds.
    assert(hasPartition(ifOp));
    auto ifOpIds = getPartitionIds(ifOp);
    auto ifOpOutputsIds = getPartitionOutputs(ifOp);
    ifOp.erase();

    SetVector<int> stageIds;
    for (auto arg : {thenState.stage, elseState.stage})
      if (auto defOp = arg.getDefiningOp())
        if (hasPartition(defOp)) {
          auto argIds = getPartitionIds(defOp);
          stageIds.insert(argIds.begin(), argIds.end());
        }
    SetVector<int> observedIds;
    for (auto arg : {thenState.wasObserved, elseState.wasObserved})
      if (auto defOp = arg.getDefiningOp())
        if (hasPartition(defOp)) {
          auto argIds = getPartitionIds(defOp);
          observedIds.insert(argIds.begin(), argIds.end());
        }
    SetVector<int> phaseIds;
    for (auto arg : {thenState.phase, elseState.phase})
      if (auto defOp = arg.getDefiningOp())
        if (hasPartition(defOp)) {
          auto argIds = getPartitionIds(defOp);
          phaseIds.insert(argIds.begin(), argIds.end());
        }

    if (stageIds.empty()) stageIds = ifOpIds;
    if (observedIds.empty()) observedIds = ifOpIds;
    if (phaseIds.empty()) phaseIds = ifOpIds;

    ifOpOutputsIds.push_back(stageIds);
    ifOpOutputsIds.push_back(observedIds);
    ifOpOutputsIds.push_back(phaseIds);
    setPartition(newIfOp, ifOpIds);
    setPartitionOutputs(newIfOp, ifOpOutputsIds);

    for (size_t idx = nResults; idx < newIfOp.getResults().size(); ++idx)
      *stateRefs[idx - nResults] = newIfOp.getResult(idx);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = newIfOp.getResult(idx);
  }

  // --- Main block walk -----------------------------------------------------

  State assignStateInBlock(Block *block, State state) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      // Phase trigger: SemaphoreAcquireOp
      if (auto acquireOp = getAcquireOp(&op)) {
        acquireOp.getStageMutable().assign(state.stage);
        acquireOp.getPhaseMutable().assign(state.phase);

        ImplicitLocOpBuilder b(acquireOp.getLoc(), acquireOp);
        b.setInsertionPointAfter(acquireOp);
        std::optional<SetVector<int>> pids;
        if (hasPartition(&op))
          pids = getPartitionIds(&op);
        auto wsTag = getWarpSpecializeTag(&op);
        auto stageCluster = getStageCluster(&op);
        auto createOp = [&](auto opTy, auto... args) {
          using ty = decltype(opTy);
          auto created = triton::gpu::createInto<ty>(
              b, b.getLoc(), pids, stageCluster,
              std::forward<decltype(args)>(args)...);
          if (wsTag)
            setWarpSpecializeTag(created, *wsTag);
          return created;
        };

        auto c1 = createOp(arith::ConstantIntOp{}, 1, 32);
        auto phaseBit = createOp(arith::ShLIOp{}, c1, state.stage);
        state.phase = createOp(arith::XOrIOp{}, state.phase, phaseBit);
        state.token = acquireOp.getToken();
        continue;
      }

      // Control flow
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignStateInForOp(forOp, state);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignStateInIfOp(ifOp, state);
        continue;
      }

      // Stage trigger: data access ops (only for current partition)
      if (!isInPartition(&op))
        continue;
      auto access = classifyAccess(&op);
      if (access == AccessKind::None)
        continue;

      ImplicitLocOpBuilder b(op.getLoc(), &op);
      b.setInsertionPointAfter(&op);
      std::optional<SetVector<int>> pids;
      if (hasPartition(&op))
        pids = getPartitionIds(&op);
      auto wsTag = getWarpSpecializeTag(&op);
      auto stageCluster = getStageCluster(&op);
      auto createOp = [&](auto opTy, auto... args) {
        using ty = decltype(opTy);
        auto created = triton::gpu::createInto<ty>(
            b, b.getLoc(), pids, stageCluster,
            std::forward<decltype(args)>(args)...);
        if (wsTag)
          setWarpSpecializeTag(created, *wsTag);
        return created;
      };

      if (access == AccessKind::Observation) {
        state.wasObserved = createOp(arith::ConstantIntOp{}, 1, 1);
        continue;
      }

      Value isFresh;
      if (access == AccessKind::FreshWrite) {
        isFresh = createOp(arith::ConstantIntOp{}, 1, 1);
      } else {
        auto mmaOp = cast<MMAv5OpInterface>(&op);
        auto c1 = createOp(arith::ConstantIntOp{}, 1, 1);
        isFresh = createOp(arith::XOrIOp{}, mmaOp.useAccumulator(), c1);
      }

      auto shouldAdvance =
          createOp(arith::AndIOp{}, state.wasObserved, isFresh);
      auto c1_i32 = createOp(arith::ConstantIntOp{}, 1, 32);
      auto c0_i32 = createOp(arith::ConstantIntOp{}, 0, 32);
      auto cDepth = createOp(arith::ConstantIntOp{}, depth, 32);
      auto cFalse = createOp(arith::ConstantIntOp{}, 0, 1);

      auto next = createOp(arith::AddIOp{}, state.stage, c1_i32);
      auto wrap =
          createOp(arith::CmpIOp{}, arith::CmpIPredicate::eq, next, cDepth);
      auto wrapped = createOp(arith::SelectOp{}, wrap, c0_i32, next);

      state.stage =
          createOp(arith::SelectOp{}, shouldAdvance, wrapped, state.stage);
      state.wasObserved =
          createOp(arith::SelectOp{}, shouldAdvance, cFalse, state.wasObserved);
    }

    return state;
  }

  // --- Token-chain propagation (from aref) ---------------------------------

  void propagateStage(Value token, Value stage,
                      DenseSet<Operation *> &visited) {
    for (auto &tokUse : token.getUses()) {
      auto owner = tokUse.getOwner();
      if (visited.contains(owner))
        continue;
      visited.insert(owner);
      if (auto stageOp = dyn_cast<ArefStageInterface>(owner)) {
        if (auto blk = dyn_cast<BlockArgument>(stage)) {
          assert(hasPartition(stageOp));
          auto stageOpIds = getPartitionIds(stageOp);
          auto forOp = cast<scf::ForOp>(blk.getOwner()->getParentOp());
          auto pos = findValuePosInRange(forOp.getRegionIterArgs(), stage);
          assert(pos);

          assert(hasPartition(forOp));
          auto forOpIds = getPartitionIds(forOp);
          forOpIds.insert(stageOpIds.begin(), stageOpIds.end());
          setPartition(forOp, forOpIds);

          auto forOpOutputsIds = getPartitionOutputs(forOp);
          // Token-chain propagation only constrains the stage lane.
          if (*pos < forOpOutputsIds.size())
            forOpOutputsIds[*pos].insert(stageOpIds.begin(), stageOpIds.end());
          setPartitionOutputs(forOp, forOpOutputsIds);
        }
        stageOp.setStage(stage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber() - forOp.getNumControlOperands();
        auto iterTok = forOp.getRegionIterArg(tokPos);
        auto it = tokToStagePosMap.find({forOp, iterTok});
        if (it == tokToStagePosMap.end())
          continue;
        propagateStage(iterTok, forOp.getRegionIterArgs()[it->second], visited);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto it = tokToStagePosMap.find({yieldOp, token});
        if (it == tokToStagePosMap.end())
          continue;
        auto parentOp = yieldOp->getParentOp();
        propagateStage(parentOp->getResult(tokUse.getOperandNumber()),
                       parentOp->getResult(it->second), visited);
      }
    }
  }

  // --- Entry point (per semaphore, like aref per aref) ---------------------

  static LogicalResult run(SemaphoreCreateOp semaOp) {
    auto semaType = cast<SemaphoreType>(semaOp.getType());
    int depth = std::max(1, semaType.getNumStages());

    std::set<int> partitionIds;
    for (auto user : semaOp->getUsers()) {
      if (isa<SemaphoreAcquireOp>(user) && hasPartition(user)) {
        auto ids = getPartitionIds(user);
        partitionIds.insert(ids.begin(), ids.end());
      }
    }
    if (partitionIds.empty())
      partitionIds.insert({0, 0});

    ImplicitLocOpBuilder b(semaOp.getLoc(), semaOp);
    b.setInsertionPointAfter(semaOp);

    State initState;
    initState.stage = arith::ConstantIntOp::create(b, 0, 32);
    initState.wasObserved = arith::ConstantIntOp::create(b, 0, 1);
    uint32_t initPhase = semaOp.getIsReleased() ? 0xFFFFFFFFu : 0x00000000u;
    initState.phase =
        arith::ConstantIntOp::create(b, static_cast<int64_t>(initPhase), 32);

    for (auto pid : partitionIds) {
      AssignSemaphoreStagePhase impl(semaOp.getResult(), pid, depth);
      impl.assignStateInBlock(semaOp->getBlock(), initState);

      for (auto user : semaOp->getUsers())
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user);
            acquireOp && (!hasPartition(acquireOp) ||
                          getPartitionIds(acquireOp).front() == pid)) {
          DenseSet<Operation *> visited;
          impl.propagateStage(acquireOp.getToken(), acquireOp.getStage(),
                              visited);
        }
    }

    return success();
  }
};

// ---------------------------------------------------------------------------
// Backward-slice partition logic — verbatim from AssignStagePhase.cpp
// ---------------------------------------------------------------------------

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
    if (auto forOp =
            dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return;
      auto pos = findValuePosInRange(forOp.getRegionIterArgs(), value);
      assert(pos);
      visitBackwardSlice(wsLoop, forOp.getInitArgs()[*pos], callback, visited);
    }
  } else if (auto defOp = value.getDefiningOp();
             isa<scf::IfOp, scf::ForOp>(defOp)) {
    auto pos = findValuePosInRange(defOp->getResults(), value);
    assert(pos);
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
  } else if (wsLoop.getBody()->findAncestorOpInBlock(*defOp)) {
    callback(defOp);
    for (auto operand : defOp->getOperands())
      visitBackwardSlice(wsLoop, operand, callback, visited);
  }
}

SmallVector<SetVector<int>, 4> getPartitionOutputsSafe(Operation *op) {
  SetVector<int> ids = hasPartition(op) ? getPartitionIds(op) : SetVector<int>{};
  if (ids.empty())
    ids.insert(0);

  SmallVector<SetVector<int>, 4> outputs =
      op->hasAttr(kPartitionOutputsAttrName)
          ? getPartitionOutputs(op)
          : SmallVector<SetVector<int>, 4>{};

  if (outputs.size() < op->getNumResults())
    outputs.resize(op->getNumResults(), ids);
  else if (outputs.size() > op->getNumResults())
    outputs.resize(op->getNumResults());

  for (auto &outIds : outputs) {
    if (outIds.empty())
      outIds = ids;
  }

  return outputs;
}

SetVector<int> getValuePartitionIdsForOutput(Value value,
                                             SetVector<int> fallbackIds) {
  SetVector<int> ids;
  if (auto defOp = value.getDefiningOp()) {
    if (hasPartition(defOp)) {
      if (defOp->getNumRegions() == 0) {
        ids = getPartitionIds(defOp);
      } else if (auto pos = findValuePosInRange(defOp->getResults(), value)) {
        auto outputs = getPartitionOutputsSafe(defOp);
        if (*pos < outputs.size())
          ids = outputs[*pos];
      }
    }
  } else {
    for (auto user : value.getUsers()) {
      if (isa<scf::YieldOp>(user) || !hasPartition(user))
        continue;
      auto userIds = getPartitionIds(user);
      ids.insert(userIds.begin(), userIds.end());
    }
  }
  if (ids.empty())
    ids = fallbackIds;
  return ids;
}

void widenValueProducerPartitionImpl(Value value,
                                     const SetVector<int> &requiredIds,
                                     DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  auto defOp = value.getDefiningOp();
  if (!defOp || !hasPartition(defOp))
    return;

  bool changed = false;

  auto defOpIds = getPartitionIds(defOp);
  size_t beforeOpIds = defOpIds.size();
  defOpIds.insert(requiredIds.begin(), requiredIds.end());
  if (defOpIds.size() != beforeOpIds) {
    llvm::errs() << "[assign-sema-debug] widen op " << defOp->getName()
                 << " at " << defOp->getLoc() << "\n";
    setPartition(defOp, defOpIds);
    changed = true;
  }

  if (defOp->getNumRegions() != 0) {
    auto pos = findValuePosInRange(defOp->getResults(), value);
    if (pos) {
      auto outputs = getPartitionOutputsSafe(defOp);
      size_t beforeOutIds = outputs[*pos].size();
      outputs[*pos].insert(requiredIds.begin(), requiredIds.end());
      if (outputs[*pos].size() != beforeOutIds || changed)
        setPartitionOutputs(defOp, outputs);
    }
  }

  for (Value operand : defOp->getOperands()) {
    if (auto operandDefOp = operand.getDefiningOp();
        operandDefOp && operandDefOp->getNumRegions() != 0)
      continue;
    widenValueProducerPartitionImpl(operand, requiredIds, visited);
  }
}

void widenValueProducerPartition(Value value, const SetVector<int> &requiredIds) {
  DenseSet<Value> visited;
  widenValueProducerPartitionImpl(value, requiredIds, visited);
}

void widenForOpOutputsFromIterArgUsers(scf::ForOp forOp) {
  if (!hasPartition(forOp) || !forOp->hasAttr(kPartitionOutputsAttrName) ||
      !hasWarpSpecializeTag(forOp))
    return;

  auto forOpOutputsIds = getPartitionOutputs(forOp);
  bool changed = false;
  for (size_t idx = 0; idx < forOp.getNumResults(); ++idx) {
    auto ids = forOpOutputsIds[idx];
    for (Operation *user : forOp.getRegionIterArg(idx).getUsers()) {
      if (isa<scf::YieldOp>(user) || !hasPartition(user))
        continue;
      auto userIds = getPartitionIds(user);
      ids.insert(userIds.begin(), userIds.end());
    }
    if (ids.size() != forOpOutputsIds[idx].size()) {
      forOpOutputsIds[idx] = ids;
      changed = true;
    }
  }
  if (changed)
    setPartitionOutputs(forOp, forOpOutputsIds);
}

void alignForOpYieldValuePartitions(scf::ForOp forOp) {
  auto forOpOutputsIds = getPartitionOutputsSafe(forOp);
  auto yieldOp = forOp.getBody()->getTerminator();
  size_t n = std::min<size_t>(forOpOutputsIds.size(), yieldOp->getNumOperands());
  for (size_t idx = 0; idx < n; ++idx) {
    auto val = yieldOp->getOperand(idx);
    if (!isa<IntegerType, FloatType, IndexType>(val.getType()))
      continue;
    if (auto defOp = val.getDefiningOp(); defOp && hasPartition(defOp)) {
      auto have = getPartitionIds(defOp);
      llvm::errs() << "[align-for-scan] loop " << forOp.getLoc() << " idx "
                   << idx << " def " << defOp->getName() << " at "
                   << defOp->getLoc() << " have={";
      for (auto id : have)
        llvm::errs() << id << ",";
      llvm::errs() << "} want={";
      for (auto id : forOpOutputsIds[idx])
        llvm::errs() << id << ",";
      llvm::errs() << "}\n";
      bool needWiden = llvm::any_of(forOpOutputsIds[idx], [&](int id) {
        return !have.contains(id);
      });
      if (needWiden) {
        llvm::errs() << "[align-for-debug] loop " << forOp.getLoc() << " idx "
                     << idx << " def " << defOp->getName() << " at "
                     << defOp->getLoc() << "\n";
      }
    }
    widenValueProducerPartition(val, forOpOutputsIds[idx]);
  }
}

void alignIfOpYieldValuePartitions(scf::IfOp ifOp) {
  auto ifOpOutputsIds = getPartitionOutputsSafe(ifOp);

  auto thenYield = ifOp.thenYield();
  size_t nThen =
      std::min<size_t>(ifOpOutputsIds.size(), thenYield->getNumOperands());
  for (size_t idx = 0; idx < nThen; ++idx) {
    auto val = thenYield->getOperand(idx);
    if (!isa<IntegerType, FloatType, IndexType>(val.getType()))
      continue;
    widenValueProducerPartition(val, ifOpOutputsIds[idx]);
  }

  if (auto elseYield = ifOp.elseYield()) {
    size_t nElse =
        std::min<size_t>(ifOpOutputsIds.size(), elseYield->getNumOperands());
    for (size_t idx = 0; idx < nElse; ++idx) {
      auto val = elseYield->getOperand(idx);
      if (!isa<IntegerType, FloatType, IndexType>(val.getType()))
        continue;
      widenValueProducerPartition(val, ifOpOutputsIds[idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

LogicalResult assignSemaphoreStagePhase(triton::FuncOp funcOp) {
  SmallVector<SemaphoreCreateOp> semaOps;
  funcOp.walk([&](SemaphoreCreateOp op) { semaOps.push_back(op); });
  for (auto semaOp : semaOps) {
    if (failed(AssignSemaphoreStagePhase::run(semaOp)))
      return failure();
  }

  auto callback = [&](Operation *op) {
    if (!isa<scf::YieldOp, scf::IfOp, scf::ForOp, triton::ReduceOp>(op)) {
      assert(hasPartition(op));
      auto partitionIds = getPartitionIds(op);
      partitionIds.insert(0);
      setPartition(op, partitionIds);
    }
  };

  funcOp.walk([&](scf::ForOp forOp) {
    DenseSet<Value> visited;
    if (forOp->hasAttr(kWarpSpecializeAttrName)) {
      for (auto result : forOp.getResults()) {
        if (isa<IntegerType, FloatType>(result.getType()) &&
            !result.use_empty()) {
          auto arg = forOp.getBody()->getTerminator()->getOperand(
              result.getResultNumber());
          bool assignDefaultPartition =
              llvm::any_of(result.getUsers(), [&](Operation *user) {
                return !hasPartition(user) ||
                       (isa<scf::ForOp>(user) && hasWarpSpecializeTag(user));
              });
          if (assignDefaultPartition) {
            updateOutputWithDefaultPartition(forOp, result.getResultNumber());
            visitBackwardSlice(forOp, arg, callback, visited);
          }
        }
      }
    }
  });

  auto normalizeControlFlowPartitions = [&]() {
    funcOp.walk([&](Operation *op) {
      if (!isa<scf::ForOp, scf::IfOp>(op) || !hasPartition(op))
        return;

    auto opIds = getPartitionIds(op);
    if (opIds.empty())
      opIds.insert(0);

    auto opOutputs = getPartitionOutputsSafe(op);
    if (opOutputs.size() < op->getNumResults())
      opOutputs.resize(op->getNumResults(), opIds);

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto yieldOp = forOp.getBody()->getTerminator();
      size_t n = std::min<size_t>(opOutputs.size(), yieldOp->getNumOperands());
      for (size_t idx = 0; idx < n; ++idx) {
        SetVector<int> fallback = opOutputs[idx].empty() ? opIds : opOutputs[idx];
        opOutputs[idx] =
            getValuePartitionIdsForOutput(yieldOp->getOperand(idx), fallback);
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      auto thenYield = ifOp.thenYield();
      auto elseYield = ifOp.elseYield();
      size_t n = std::min<size_t>(opOutputs.size(), thenYield->getNumOperands());
      for (size_t idx = 0; idx < n; ++idx) {
        SetVector<int> fallback = opOutputs[idx].empty() ? opIds : opOutputs[idx];
        auto ids =
            getValuePartitionIdsForOutput(thenYield->getOperand(idx), fallback);
        if (elseYield && idx < elseYield->getNumOperands()) {
          auto elseIds =
              getValuePartitionIdsForOutput(elseYield->getOperand(idx), fallback);
          ids.insert(elseIds.begin(), elseIds.end());
        }
        opOutputs[idx] = ids;
      }
    }

    for (auto &ids : opOutputs) {
      if (ids.empty())
        ids = opIds;
      opIds.insert(ids.begin(), ids.end());
    }

    op->walk([&](Operation *nested) {
      if (nested == op || !hasPartition(nested))
        return;
      auto nestedIds = getPartitionIds(nested);
      opIds.insert(nestedIds.begin(), nestedIds.end());
    });

      setPartition(op, opIds);
      setPartitionOutputs(op, opOutputs);
    });
  };

  normalizeControlFlowPartitions();
  funcOp.walk([&](scf::ForOp forOp) { widenForOpOutputsFromIterArgUsers(forOp); });
  funcOp.walk([&](scf::ForOp forOp) {
    llvm::errs() << "[for-loop-debug] " << forOp.getLoc()
                 << " outputs_attr=" << forOp->hasAttr(kPartitionOutputsAttrName)
                 << "\n";
    alignForOpYieldValuePartitions(forOp);
  });
  funcOp.walk([&](scf::IfOp ifOp) { alignIfOpYieldValuePartitions(ifOp); });
  normalizeControlFlowPartitions();
  return success();
}

} // anonymous namespace

class NVWSAssignSemaphoreStagePhase
    : public impl::NVWSAssignSemaphoreStagePhaseBase<
          NVWSAssignSemaphoreStagePhase> {
public:
  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    m.walk([&](triton::FuncOp funcOp) {
      if (failed(assignSemaphoreStagePhase(funcOp)))
        signalPassFailure();
    });
  }
};

} // namespace triton
} // namespace mlir
