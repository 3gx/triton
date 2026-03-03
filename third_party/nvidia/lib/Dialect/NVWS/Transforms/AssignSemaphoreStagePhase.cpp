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
#include "llvm/ADT/DenseSet.h"
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
// State = {stage, phase, token} per semaphore per partition.
// Each (semaphore, partition) pair gets its own iter_args through loops/ifs.
// ---------------------------------------------------------------------------

enum class AccessKind { None, Observation, FreshWrite, FreshWriteMMA };

struct AssignSemaphoreStagePhase {
  struct State {
    Value stage; // stage index
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

  bool isTokenView(Value bufferView, Value token,
                   DenseSet<Value> &visitedViews) {
    if (!visitedViews.insert(bufferView).second)
      return false;
    if (!isa<MemDescType>(bufferView.getType()))
      return false;
    if (auto semaBuffer = bufferView.getDefiningOp<SemaphoreBufferOp>())
      return semaBuffer.getSemaphore() == this->semaphore &&
             semaBuffer.getToken() == token;
    if (auto blockArg = dyn_cast<BlockArgument>(bufferView)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp()))
        if (auto pos =
                findValuePosInRange(forOp.getRegionIterArgs(), bufferView))
          return isTokenView(forOp.getInitArgs()[*pos], token, visitedViews);
      return false;
    }

    auto *defOp = bufferView.getDefiningOp();
    if (!defOp)
      return false;

    if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      for (Value operand : defOp->getOperands())
        if (isa<MemDescType>(operand.getType()) &&
            isTokenView(operand, token, visitedViews))
          return true;
      return false;
    }

    return false;
  }

  bool isTokenView(Value bufferView, Value token) {
    DenseSet<Value> visitedViews;
    return isTokenView(bufferView, token, visitedViews);
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

  AccessKind classifyAccessForToken(Operation *op, Value token) {
    if (!token || !isInPartition(op))
      return AccessKind::None;
    if (auto loadOp = dyn_cast<LocalLoadOp>(op))
      return isTokenView(loadOp.getSrc(), token) ? AccessKind::Observation
                                                 : AccessKind::None;
    if (auto loadOp = dyn_cast<TMEMLoadOp>(op))
      return isTokenView(loadOp.getSrc(), token) ? AccessKind::Observation
                                                 : AccessKind::None;
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      if (isTokenView(mmaOp.getAccumulator(), token))
        return AccessKind::FreshWriteMMA;
      if (isTokenView(mmaOp.getA(), token) || isTokenView(mmaOp.getB(), token))
        return AccessKind::Observation;
      return AccessKind::None;
    }
    if (auto storeOp = dyn_cast<LocalStoreOp>(op))
      return isTokenView(storeOp.getDst(), token) ? AccessKind::FreshWrite
                                                  : AccessKind::None;
    if (auto descLoad = dyn_cast<DescriptorLoadOp>(op))
      return isTokenView(descLoad.getResult(), token) ? AccessKind::FreshWrite
                                                      : AccessKind::None;
    if (auto descGather = dyn_cast<DescriptorGatherOp>(op))
      return isTokenView(descGather.getResult(), token)
                 ? AccessKind::FreshWrite
                 : AccessKind::None;
    if (auto storeOp = dyn_cast<TMEMStoreOp>(op))
      return isTokenView(storeOp.getDst(), token) ? AccessKind::FreshWrite
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
          (token && isInPartition(&op) &&
           classifyAccess(&op) != AccessKind::None))
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

  bool isFirstUseFreshWriteAfterAcquire(SemaphoreAcquireOp acquireOp) {
    auto isTokenViewOp = [&](Operation *op, Value token) {
      if (auto semaBuffer = dyn_cast<SemaphoreBufferOp>(op))
        return semaBuffer.getSemaphore() == semaphore &&
               semaBuffer.getToken() == token;
      if (!op->hasTrait<OpTrait::MemDescViewTrait>())
        return false;
      for (Value operand : op->getOperands())
        if (isa<MemDescType>(operand.getType()) &&
            isTokenView(operand, token))
          return true;
      return false;
    };

    auto usesTokenView = [&](Operation *op, Value token) {
      for (Value operand : op->getOperands())
        if (isa<MemDescType>(operand.getType()) &&
            isTokenView(operand, token))
          return true;
      return false;
    };

    auto token = acquireOp.getToken();
    auto it = std::next(Block::iterator(acquireOp.getOperation()));
    auto end = acquireOp->getBlock()->end();
    for (; it != end; ++it) {
      Operation *op = &*it;
      if (isa<SemaphoreAcquireOp>(op))
        return false;
      if (isa<scf::ForOp, scf::IfOp>(op))
        return false;
      if (isTokenViewOp(op, token))
        continue;
      auto access = classifyAccessForToken(op, token);
      if (access == AccessKind::FreshWrite)
        return true;
      if (access != AccessKind::None)
        return false;
      if (usesTokenView(op, token))
        return false;
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

    SmallVector<Value> extraIterArgs{state.stage, state.phase};
    SmallVector<Value *> stateRefs{&state.stage, &state.phase};
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
                                      stateInBlock.phase};
    appendToForOpYield(forOp, extraYieldArgs);
    tokToStagePosMap[{forOp, state.token}] = nArgs;
    tokToStagePosMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        nArgs;

    // Partition annotation — match aref AssignStagePhase
    for (auto arg : extraYieldArgs) {
      SetVector<int> argIds;
      if (auto defOp = arg.getDefiningOp()) {
        if (defOp->getNumRegions() == 0) {
          assert(hasPartition(defOp));
          argIds = getPartitionIds(defOp);
        } else {
          auto pos = findValuePosInRange(defOp->getResults(), arg);
          argIds = getPartitionOutputs(defOp)[*pos];
        }
      } else {
        for (auto user : arg.getUsers()) {
          if (isa<scf::YieldOp>(user))
            continue;
          assert(hasPartition(user));
          auto ids = getPartitionIds(user);
          argIds.insert(ids.begin(), ids.end());
        }
      }
      if (argIds.empty())
        argIds.insert(partitionId);
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
                                     state.phase.getType()};
    SmallVector<Value *> stateRefs{&state.stage, &state.phase};

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
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.phase);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.phase);

    // Partition annotation — verbatim from aref. NEVER widen ifOpIds.
    assert(hasPartition(ifOp));
    auto ifOpIds = getPartitionIds(ifOp);
    auto ifOpOutputsIds = getPartitionOutputs(ifOp);
    ifOp.erase();

    SetVector<int> stageIds;
    auto collectIds = [&](Value arg, SetVector<int> &ids) {
      if (auto defOp = arg.getDefiningOp()) {
        if (!hasPartition(defOp))
          return;
        if (defOp->getNumRegions() == 0) {
          auto argIds = getPartitionIds(defOp);
          ids.insert(argIds.begin(), argIds.end());
        } else if (auto pos = findValuePosInRange(defOp->getResults(), arg)) {
          auto outputs = getPartitionOutputs(defOp);
          if (*pos < outputs.size()) {
            auto argIds = outputs[*pos];
            ids.insert(argIds.begin(), argIds.end());
          }
        }
      } else {
        for (auto user : arg.getUsers()) {
          if (isa<scf::YieldOp>(user) || !hasPartition(user))
            continue;
          auto argIds = getPartitionIds(user);
          ids.insert(argIds.begin(), argIds.end());
        }
      }
    };
    for (auto arg : {thenState.stage, elseState.stage})
      collectIds(arg, stageIds);
    SetVector<int> phaseIds;
    for (auto arg : {thenState.phase, elseState.phase})
      collectIds(arg, phaseIds);
    if (stageIds.empty())
      stageIds.insert(ifOpIds.begin(), ifOpIds.end());
    if (phaseIds.empty())
      phaseIds.insert(ifOpIds.begin(), ifOpIds.end());
    if (stageIds.empty())
      stageIds.insert(partitionId);
    if (phaseIds.empty())
      phaseIds.insert(partitionId);
    ifOpOutputsIds.push_back(stageIds);
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
        ImplicitLocOpBuilder b(acquireOp.getLoc(), acquireOp);
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

        Value rawStage = state.stage;
        Value acquireStage = rawStage;
        if (isFirstUseFreshWriteAfterAcquire(acquireOp)) {
          auto c1 = createOp(arith::ConstantIntOp{}, 1, 32);
          auto c0 = createOp(arith::ConstantIntOp{}, 0, 32);
          auto cDepth = createOp(arith::ConstantIntOp{}, depth, 32);
          auto next = createOp(arith::AddIOp{}, rawStage, c1);
          auto wrap =
              createOp(arith::CmpIOp{}, arith::CmpIPredicate::eq, next, cDepth);
          auto wrapped = createOp(arith::SelectOp{}, wrap, c0, next);
          acquireStage = wrapped;
        }
        state.stage = acquireStage;
        acquireOp.getStageMutable().assign(acquireStage);
        acquireOp.getPhaseMutable().assign(state.phase);

        b.setInsertionPointAfter(acquireOp);
        auto c1 = createOp(arith::ConstantIntOp{}, 1, 32);
        auto phaseBit = createOp(arith::ShLIOp{}, c1, acquireStage);
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

      // No post-access stage mutation. Stage updates are acquire-site only.
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
          forOpOutputsIds[*pos + 0].insert(stageOpIds.begin(),
                                           stageOpIds.end());
          forOpOutputsIds[*pos + 1].insert(stageOpIds.begin(),
                                           stageOpIds.end());
          setPartitionOutputs(forOp, forOpOutputsIds);
        }
        stageOp.setStage(stage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber() - forOp.getNumControlOperands();
        auto iterTok = forOp.getRegionIterArg(tokPos);
        auto stagePos = tokToStagePosMap.at({forOp, iterTok});
        propagateStage(iterTok, forOp.getRegionIterArgs()[stagePos], visited);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto stagePos = tokToStagePosMap.at({yieldOp, token});
        auto parentOp = yieldOp->getParentOp();
        propagateStage(parentOp->getResult(tokUse.getOperandNumber()),
                       parentOp->getResult(stagePos), visited);
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

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

LogicalResult assignSemaphoreStagePhase(triton::FuncOp funcOp) {
  SmallVector<SemaphoreCreateOp> semaOps;
  funcOp.walk([&](SemaphoreCreateOp op) { semaOps.push_back(op); });

  // Keep processing order deterministic and scoped by backing buffer.
  llvm::MapVector<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
  for (auto semaOp : semaOps) {
    auto buffers = semaOp.getBuffers();
    if (buffers.empty())
      continue;
    semaGroups[buffers.front()].push_back(semaOp);
  }

  for (auto &it : semaGroups) {
    for (auto semaOp : it.second) {
      if (failed(AssignSemaphoreStagePhase::run(semaOp)))
        return failure();
    }
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
