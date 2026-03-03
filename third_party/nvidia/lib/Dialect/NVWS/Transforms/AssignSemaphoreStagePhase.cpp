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
    Value stage;                // shared stage index (one per buffer group)
    SmallVector<Value> phases;  // phases[i] = phase for groupSemaphoresList[i]
    Value token;
  };

  SmallVector<Value> groupSemaphoresList;  // stable ordering
  DenseSet<Value> groupSemaphoresSet;      // O(1) lookup
  int partitionId;
  int depth;
  DenseMap<std::pair<Operation *, Value>, int> tokToStagePosMap;
  DenseMap<std::pair<Operation *, Value>, int> tokToNumPhasesMap;
  DenseMap<Value, bool> viewMemo;
  DenseSet<Value> viewVisited;

  AssignSemaphoreStagePhase(ArrayRef<SemaphoreCreateOp> semaOps, int partitionId,
                            int depth)
      : partitionId(partitionId), depth(depth) {
    for (auto semaOp : semaOps) {
      groupSemaphoresList.push_back(semaOp.getResult());
      groupSemaphoresSet.insert(semaOp.getResult());
    }
  }

  bool isGroupSemaphore(Value semaphore) const {
    return groupSemaphoresSet.contains(semaphore);
  }

  bool isMultiSemaphoreGroup() const { return groupSemaphoresList.size() > 1; }

  std::optional<int> getSemaphoreIndex(Value semaphore) const {
    for (int i = 0; i < (int)groupSemaphoresList.size(); ++i)
      if (groupSemaphoresList[i] == semaphore)
        return i;
    return {};
  }

  // --- Op matching ---------------------------------------------------------

  SemaphoreAcquireOp getAcquireOp(Operation *op) {
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
      if (isGroupSemaphore(acquireOp.getSemaphore())) {
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
    if (!isGroupSemaphore(bufOp.getSemaphore()))
      return false;
    return token == bufOp.getToken();
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
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        if (auto pos =
                findValuePosInRange(forOp.getRegionIterArgs(), bufferView))
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
            idx < ifOp.elseYield()->getNumOperands())
          result = isGroupView(ifOp.elseYield()->getOperand(idx));
      }
    }

    viewVisited.erase(bufferView);
    viewMemo[bufferView] = result;
    return result;
  }

  bool isTokenView(Value bufferView, Value token,
                   DenseSet<Value> &visitedViews) {
    if (!visitedViews.insert(bufferView).second)
      return false;
    if (!isa<MemDescType>(bufferView.getType()))
      return false;
    if (auto semaBuffer = bufferView.getDefiningOp<SemaphoreBufferOp>())
      return isGroupSemaphore(semaBuffer.getSemaphore()) &&
             semaBuffer.getToken() == token;
    if (auto blockArg = dyn_cast<BlockArgument>(bufferView)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
        if (auto pos =
                findValuePosInRange(forOp.getRegionIterArgs(), bufferView))
          return isTokenView(forOp.getInitArgs()[*pos], token, visitedViews);
      }
      return false;
    }

    auto *defOp = bufferView.getDefiningOp();
    if (!defOp)
      return false;

    if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      for (Value operand : defOp->getOperands()) {
        if (!isa<MemDescType>(operand.getType()))
          continue;
        if (isTokenView(operand, token, visitedViews))
          return true;
      }
      return false;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      unsigned idx = cast<OpResult>(bufferView).getResultNumber();
      if (idx < forOp.getYieldedValues().size())
        return isTokenView(forOp.getYieldedValues()[idx], token, visitedViews);
      return false;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      unsigned idx = cast<OpResult>(bufferView).getResultNumber();
      if (idx < ifOp.thenYield()->getNumOperands() &&
          isTokenView(ifOp.thenYield()->getOperand(idx), token, visitedViews))
        return true;
      if (ifOp.elseBlock() && idx < ifOp.elseYield()->getNumOperands() &&
          isTokenView(ifOp.elseYield()->getOperand(idx), token, visitedViews))
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

  bool isTokenViewOp(Operation *op, Value token) {
    if (auto semaBuffer = dyn_cast<SemaphoreBufferOp>(op))
      return isGroupSemaphore(semaBuffer.getSemaphore()) &&
             semaBuffer.getToken() == token;
    if (!op->hasTrait<OpTrait::MemDescViewTrait>())
      return false;
    for (Value operand : op->getOperands())
      if (isa<MemDescType>(operand.getType()) && isTokenView(operand, token))
        return true;
    return false;
  }

  bool usesTokenView(Operation *op, Value token) {
    for (Value operand : op->getOperands())
      if (isa<MemDescType>(operand.getType()) && isTokenView(operand, token))
        return true;
    return false;
  }

  // Scan a block from start looking for the first access. Returns true if it's
  // a FreshWrite. Does not recurse further into nested for/if.
  bool isFirstUseFreshWriteInBlock(Block *block, Value token) {
    for (auto &op : *block) {
      if (isa<SemaphoreAcquireOp>(&op))
        return false;
      if (isa<scf::ForOp, scf::IfOp>(&op))
        return false;
      if (isTokenViewOp(&op, token))
        continue;
      auto access = classifyAccessForToken(&op, token);
      if (access == AccessKind::FreshWrite)
        return true;
      if (access != AccessKind::None)
        return false;
      if (usesTokenView(&op, token))
        return false;
    }
    return false;
  }

  bool isFirstUseFreshWriteAfterAcquire(SemaphoreAcquireOp acquireOp) {
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

  // --- Control-flow threading -----------------------------------------------

  // Collect which semaphore indices have acquire ops in a block (recursive).
  void collectSemaphoresUsedInBlock(Block *block, DenseSet<int> &usedIndices) {
    for (auto &op : *block) {
      if (auto acquireOp = getAcquireOp(&op)) {
        if (auto idx = getSemaphoreIndex(acquireOp.getSemaphore()))
          usedIndices.insert(*idx);
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        collectSemaphoresUsedInBlock(forOp.getBody(), usedIndices);
      else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        collectSemaphoresUsedInBlock(ifOp.thenBlock(), usedIndices);
        if (ifOp.elseBlock())
          collectSemaphoresUsedInBlock(ifOp.elseBlock(), usedIndices);
      }
    }
  }

  // Get sorted list of semaphore indices that have acquire ops in a block.
  SmallVector<int> getSortedUsedIndices(Block *block) {
    DenseSet<int> usedSemaIndices;
    collectSemaphoresUsedInBlock(block, usedSemaIndices);
    SmallVector<int> sorted(usedSemaIndices.begin(), usedSemaIndices.end());
    llvm::sort(sorted);
    return sorted;
  }

  // Infer partition IDs for a yield argument value.
  SetVector<int> inferPartitionIds(Value arg) {
    SetVector<int> argIds;
    if (auto defOp = arg.getDefiningOp()) {
      if (defOp->getNumRegions() == 0) {
        if (hasPartition(defOp))
          argIds = getPartitionIds(defOp);
      } else if (auto pos = findValuePosInRange(defOp->getResults(), arg)) {
        if (hasPartition(defOp)) {
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
      argIds.insert(partitionId);
    return argIds;
  }

  void assignStateInForOp(scf::ForOp forOp, State &state) {
    Value newTok;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token))
      newTok = forOp.getRegionIterArgs()[*pos];
    if (!analyzeUseInBlock(forOp.getBody(), newTok))
      return;

    auto sortedUsedIndices = getSortedUsedIndices(forOp.getBody());

    // Build extra iter args: stage + phase for each used semaphore
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> stateRefs;
    extraIterArgs.push_back(state.stage);
    stateRefs.push_back(&state.stage);
    for (int idx : sortedUsedIndices) {
      extraIterArgs.push_back(state.phases[idx]);
      stateRefs.push_back(&state.phases[idx]);
    }

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

    for (size_t i = 0; i < stateRefs.size(); ++i)
      *stateRefs[i] = forOp.getRegionIterArgs()[nArgs + i];

    auto stateInBlock = assignStateInBlock(forOp.getBody(), state);

    // Build yield args matching iter args order
    SmallVector<Value> extraYieldArgs;
    extraYieldArgs.push_back(stateInBlock.stage);
    for (int idx : sortedUsedIndices)
      extraYieldArgs.push_back(stateInBlock.phases[idx]);

    appendToForOpYield(forOp, extraYieldArgs);
    tokToStagePosMap[{forOp, state.token}] = nArgs;
    tokToStagePosMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        nArgs;
    tokToNumPhasesMap[{forOp, state.token}] = sortedUsedIndices.size();
    tokToNumPhasesMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        sortedUsedIndices.size();

    // Partition annotations
    for (auto arg : extraYieldArgs) {
      auto argIds = inferPartitionIds(arg);
      forOpIds.insert(argIds.begin(), argIds.end());
      forOpOutputsIds.push_back(argIds);
    }
    setPartition(forOp, forOpIds);
    setPartitionOutputs(forOp, forOpOutputsIds);

    for (size_t i = 0; i < stateRefs.size(); ++i)
      *stateRefs[i] = forOp.getResult(nArgs + i);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = forOp.getResult(idx);
  }

  void assignStateInIfOp(scf::IfOp ifOp, State &state) {
    // Only thread state through if-ops that contain acquire ops.
    // Stage/phase only change at acquire sites, so if-ops with only buffer
    // accesses (no acquires) don't need state threaded through them.
    DenseSet<int> usedSemaIndices;
    collectSemaphoresUsedInBlock(ifOp.thenBlock(), usedSemaIndices);
    if (ifOp.elseBlock())
      collectSemaphoresUsedInBlock(ifOp.elseBlock(), usedSemaIndices);
    if (usedSemaIndices.empty())
      return;

    SmallVector<int> sortedUsedIndices(usedSemaIndices.begin(),
                                       usedSemaIndices.end());
    llvm::sort(sortedUsedIndices);

    // Build extra result types: stage + phases for used semaphores
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> stateRefs;
    extraIfResults.push_back(state.stage.getType());
    stateRefs.push_back(&state.stage);
    for (int idx : sortedUsedIndices) {
      extraIfResults.push_back(state.phases[idx].getType());
      stateRefs.push_back(&state.phases[idx]);
    }

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

    // Insert stage yield args
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.stage);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.stage);
    // Insert per-semaphore phase yield args
    for (int idx : sortedUsedIndices) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  thenState.phases[idx]);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  elseState.phases[idx]);
    }

    // Partition annotations — NEVER widen ifOpIds
    assert(hasPartition(ifOp));
    auto ifOpIds = getPartitionIds(ifOp);
    auto ifOpOutputsIds = getPartitionOutputs(ifOp);
    ifOp.erase();

    // Stage partition IDs
    SetVector<int> stageIds;
    for (auto arg : {thenState.stage, elseState.stage}) {
      auto ids = inferPartitionIds(arg);
      stageIds.insert(ids.begin(), ids.end());
    }
    ifOpOutputsIds.push_back(stageIds);

    // Per-semaphore phase partition IDs
    for (int idx : sortedUsedIndices) {
      SetVector<int> phaseIds;
      for (auto arg : {thenState.phases[idx], elseState.phases[idx]}) {
        auto ids = inferPartitionIds(arg);
        phaseIds.insert(ids.begin(), ids.end());
      }
      ifOpOutputsIds.push_back(phaseIds);
    }

    setPartition(newIfOp, ifOpIds);
    setPartitionOutputs(newIfOp, ifOpOutputsIds);

    for (size_t i = 0; i < stateRefs.size(); ++i)
      *stateRefs[i] = newIfOp.getResult(nResults + i);
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

        // Per-semaphore phase: look up and update only this semaphore's phase
        auto semaIdx = getSemaphoreIndex(acquireOp.getSemaphore());
        assert(semaIdx && "acquire op must reference a group semaphore");
        Value &semaPhase = state.phases[*semaIdx];
        acquireOp.getPhaseMutable().assign(semaPhase);

        b.setInsertionPointAfter(acquireOp);
        auto c1 = createOp(arith::ConstantIntOp{}, 1, 32);
        auto phaseBit = createOp(arith::ShLIOp{}, c1, acquireStage);
        semaPhase = createOp(arith::XOrIOp{}, semaPhase, phaseBit);
        state.token = acquireOp.getToken();
        continue;
      }

      // Control flow
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (isMultiSemaphoreGroup()) {
          assignStateInBlock(forOp.getBody(), state);
          continue;
        }
        assignStateInForOp(forOp, state);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (isMultiSemaphoreGroup()) {
          assignStateInBlock(ifOp.thenBlock(), state);
          if (ifOp.elseBlock())
            assignStateInBlock(ifOp.elseBlock(), state);
          continue;
        }
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
          // Widen stage slot and all phase slots that follow
          forOpOutputsIds[*pos].insert(stageOpIds.begin(), stageOpIds.end());
          int numPhases = (int)groupSemaphoresList.size();
          auto numIt = tokToNumPhasesMap.find({forOp, blk});
          if (numIt != tokToNumPhasesMap.end())
            numPhases = numIt->second;
          for (int k = 1;
               k <= numPhases && (*pos + k) < (int)forOpOutputsIds.size(); ++k)
            forOpOutputsIds[*pos + k].insert(stageOpIds.begin(),
                                             stageOpIds.end());
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

  // --- Entry point (per buffer group) --------------------------------------

  static LogicalResult run(ArrayRef<SemaphoreCreateOp> semaOps) {
    if (semaOps.empty())
      return success();
    auto firstSemaOp = semaOps.front();
    auto lastSemaOp = semaOps.back();
    int depth = 1;
    for (auto semaOp : semaOps)
      depth = std::max(depth, cast<SemaphoreType>(semaOp.getType()).getNumStages());
    depth = std::max(depth, 1);

    std::set<int> partitionIds;
    for (auto semaOp : semaOps) {
      for (auto user : semaOp->getUsers()) {
        if (isa<SemaphoreAcquireOp>(user) && hasPartition(user)) {
          auto ids = getPartitionIds(user);
          partitionIds.insert(ids.begin(), ids.end());
        }
      }
    }
    if (partitionIds.empty())
      partitionIds.insert(0);

    // Insert after the last semaOp so all semaphores are defined
    ImplicitLocOpBuilder b(lastSemaOp.getLoc(), lastSemaOp);
    b.setInsertionPointAfter(lastSemaOp);

    State initState;
    initState.stage = arith::ConstantIntOp::create(b, 0, 32);
    // Per-semaphore initial phases
    for (auto semaOp : semaOps) {
      uint32_t initPhase =
          semaOp.getIsReleased() ? 0xFFFFFFFFu : 0x00000000u;
      initState.phases.push_back(
          arith::ConstantIntOp::create(b, static_cast<int64_t>(initPhase), 32));
    }

    for (auto pid : partitionIds) {
      AssignSemaphoreStagePhase impl(semaOps, pid, depth);
      impl.assignStateInBlock(firstSemaOp->getBlock(), initState);

      for (auto semaOp : semaOps) {
        for (auto user : semaOp->getUsers()) {
          if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user);
              acquireOp && (!hasPartition(acquireOp) ||
                            llvm::is_contained(getPartitionIds(acquireOp),
                                               pid))) {
            DenseSet<Operation *> visited;
            impl.propagateStage(acquireOp.getToken(), acquireOp.getStage(),
                                visited);
          }
        }
      }
    }

    // Fallback: patch any missing stage/phase with per-semaphore init values
    for (size_t semaIdx = 0; semaIdx < semaOps.size(); ++semaIdx) {
      auto semaOp = semaOps[semaIdx];
      for (auto user : semaOp->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          if (!acquireOp.getStage())
            acquireOp.getStageMutable().assign(initState.stage);
          if (!acquireOp.getPhase())
            acquireOp.getPhaseMutable().assign(initState.phases[semaIdx]);
        }
        if (auto stageOp = dyn_cast<ArefStageInterface>(user))
          if (!stageOp.getStage())
            stageOp.setStage(initState.stage);
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
    if (failed(AssignSemaphoreStagePhase::run(it.second)))
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
