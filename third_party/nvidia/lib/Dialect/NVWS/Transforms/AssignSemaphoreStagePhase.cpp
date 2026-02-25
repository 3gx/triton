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
// Single-traversal architecture:
//   One walk assigns stage and phase together.
//   Stage remains a shared buffer property.
//   Phase keeps per-(partition, semaphore) lanes to preserve existing behavior.
// ---------------------------------------------------------------------------

enum class AccessKind { None, Observation, FreshWrite, FreshWriteMMA };

struct AssignSemaphoreStagePhase {
  using LaneKey = std::pair<int, int>; // (partitionId, semaphoreIndex)
  static constexpr StringLiteral kUseSinglePhaseAttrName =
      "nvws.use_single_phase";

  struct State {
    Value stage;                   // shared stage index (per buffer group)
    SmallVector<Value> basePhases; // basePhases[i] by semaphore index
    llvm::MapVector<LaneKey, Value> lanes; // phase lane values by (pid, semIdx)
    Value token;                           // token used for stage propagation
  };

  SmallVector<Value> groupSemaphoresList;          // stable ordering
  DenseSet<Value> groupSemaphoresSet;              // O(1) lookup
  DenseMap<Value, bool> useSinglePhaseBySemaphore; // pass-local mode tags
  SetVector<int> allGroupPartitionIds; // all partition IDs across all acquires
  std::optional<int> groupWsTag;       // warp specialize tag for the group
  int depth;
  DenseMap<std::pair<Operation *, Value>, int> tokToStagePosMap;
  DenseMap<Value, bool> viewMemo;
  DenseSet<Value> viewVisited;

  // --- Single-phase eligibility analysis ------------------------------------

  // Recursive walk: enters scf.for and scf.if regions to find acquires.
  // Returns false if a (semaphore, partition, virtual_stage) duplicate is found.
  bool walkBlockForEligibility(
      Block *block, int &virtualStage,
      DenseSet<std::tuple<Value, int, int>> &seen) {
    for (auto &op : *block) {
      if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(&op)) {
        if (isGroupSemaphore(acquireOp.getSemaphore())) {
          if (isFirstUseFreshWriteAfterAcquire(acquireOp))
            virtualStage++;
          // Two acquires of the same semaphore at same vs but different
          // partitions are concurrent (one mbarrier wait) → not a dup.
          int pid = hasPartition(&op) ? getPartitionIds(&op).front() : 0;
          if (!seen.insert({acquireOp.getSemaphore(), pid, virtualStage})
                   .second)
            return false; // duplicate → multiphase
        }
      } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (!walkBlockForEligibility(forOp.getBody(), virtualStage, seen))
          return false;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        // Check both branches. An acquire in either branch counts.
        int vsBefore = virtualStage;
        auto seenBefore = seen;
        if (!walkBlockForEligibility(ifOp.thenBlock(), virtualStage, seen))
          return false;
        if (ifOp.elseBlock()) {
          int vsThen = virtualStage;
          auto seenThen = seen;
          virtualStage = vsBefore;
          seen = seenBefore;
          if (!walkBlockForEligibility(ifOp.elseBlock(), virtualStage, seen))
            return false;
          // Merge: conservative (max) virtual_stage, union of seen.
          virtualStage = std::max(vsThen, virtualStage);
          seen.insert(seenThen.begin(), seenThen.end());
        }
      }
    }
    return true;
  }

  // Returns true if single-phase is safe for the entire buffer group.
  // All-or-nothing: the release-acquire ring forces A_P(s) = A_C(s),
  // so all semaphores in a group must use the same mode.
  bool computeSinglePhaseEligibility() {
    // depth==1 → always single-phase (one stage, nothing to cycle).
    if (depth == 1)
      return true;

    // Find the warp-specialized for-loop containing group acquires.
    scf::ForOp wsLoop;
    for (Value sema : groupSemaphoresList) {
      for (auto user : sema.getDefiningOp()->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          for (auto *parent = acquireOp->getParentOp(); parent;
               parent = parent->getParentOp()) {
            if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
              if (forOp->hasAttr(kWarpSpecializeAttrName)) {
                wsLoop = forOp;
                break;
              }
            }
          }
          if (wsLoop)
            break;
        }
      }
      if (wsLoop)
        break;
    }

    // No warp-specialized loop → conservative, use multiphase.
    if (!wsLoop)
      return false;

    // Walk loop body recursively, tracking virtual_stage.
    // Key: (semaphore, partition_id, virtual_stage).
    DenseSet<std::tuple<Value, int, int>> seen;
    int virtualStage = 0;
    if (!walkBlockForEligibility(wsLoop.getBody(), virtualStage, seen))
      return false;

    // Must have at least one advance per iteration.
    if (virtualStage == 0)
      return false;

    return true;
  }

  AssignSemaphoreStagePhase(ArrayRef<SemaphoreCreateOp> semaOps,
                            DenseMap<Value, bool> useSinglePhaseBySemaphore,
                            SetVector<int> allPartitionIds, int depth,
                            std::optional<int> wsTag = std::nullopt)
      : useSinglePhaseBySemaphore(std::move(useSinglePhaseBySemaphore)),
        allGroupPartitionIds(std::move(allPartitionIds)), groupWsTag(wsTag),
        depth(depth) {
    for (auto semaOp : semaOps) {
      Value sema = semaOp.getResult();
      groupSemaphoresList.push_back(sema);
      groupSemaphoresSet.insert(sema);
    }
  }

  bool isGroupSemaphore(Value semaphore) const {
    return groupSemaphoresSet.contains(semaphore);
  }

  std::optional<int> getSemaphoreIndex(Value semaphore) const {
    for (int i = 0; i < (int)groupSemaphoresList.size(); ++i)
      if (groupSemaphoresList[i] == semaphore)
        return i;
    return {};
  }

  bool shouldUseSinglePhase(Value semaphore) const {
    auto it = useSinglePhaseBySemaphore.find(semaphore);
    return it != useSinglePhaseBySemaphore.end() && it->second;
  }

  // --- Op matching ----------------------------------------------------------

  SemaphoreAcquireOp getAcquireOp(Operation *op) {
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(op)) {
      if (isGroupSemaphore(acquireOp.getSemaphore()))
        return acquireOp;
    }
    return {};
  }

  SmallVector<int> getAcquireProcessingOrder(Operation *op) const {
    SmallVector<int> pids;
    for (int pid : allGroupPartitionIds) {
      if (!hasPartition(op) || llvm::is_contained(getPartitionIds(op), pid))
        pids.push_back(pid);
    }
    return pids;
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
    if (auto descLoad = dyn_cast<nvws::DescriptorLoadOp>(op))
      return isGroupView(descLoad.getResult()) ? AccessKind::FreshWrite
                                               : AccessKind::None;
    if (auto descGather = dyn_cast<nvws::DescriptorGatherOp>(op))
      return isGroupView(descGather.getResult()) ? AccessKind::FreshWrite
                                                 : AccessKind::None;
    if (auto storeOp = dyn_cast<TMEMStoreOp>(op))
      return isGroupView(storeOp.getDst()) ? AccessKind::FreshWrite
                                           : AccessKind::None;
    return AccessKind::None;
  }

  AccessKind classifyAccessForToken(Operation *op, Value token) {
    if (!token)
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
      return isTokenView(descGather.getResult(), token) ? AccessKind::FreshWrite
                                                        : AccessKind::None;
    if (auto descLoad = dyn_cast<nvws::DescriptorLoadOp>(op))
      return isTokenView(descLoad.getResult(), token) ? AccessKind::FreshWrite
                                                      : AccessKind::None;
    if (auto descGather = dyn_cast<nvws::DescriptorGatherOp>(op))
      return isTokenView(descGather.getResult(), token) ? AccessKind::FreshWrite
                                                        : AccessKind::None;
    if (auto storeOp = dyn_cast<TMEMStoreOp>(op))
      return isTokenView(storeOp.getDst(), token) ? AccessKind::FreshWrite
                                                  : AccessKind::None;
    return AccessKind::None;
  }

  // --- Block analysis ------------------------------------------------------

  bool analyzeUseInBlock(Block *block, Value token) {
    for (auto &op : *block) {
      if (getAcquireOp(&op) ||
          isGroupBuffer(dyn_cast<SemaphoreBufferOp>(op), token) ||
          (token && classifyAccess(&op) != AccessKind::None))
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
      if (isTokenViewOp(&op, token))
        continue;
      auto access = classifyAccessForToken(&op, token);
      if (access == AccessKind::FreshWrite ||
          access == AccessKind::FreshWriteMMA)
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
    auto *block = acquireOp->getBlock();
    auto end = block->end();
    for (; it != end; ++it) {
      Operation *op = &*it;
      if (isTokenViewOp(op, token))
        continue;
      // Follow token through for-loop init_args into the loop body.
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (auto pos = findValuePosInRange(forOp.getInitArgs(), token)) {
          Value iterArgToken = forOp.getRegionIterArgs()[*pos];
          return isFirstUseFreshWriteInBlock(forOp.getBody(), iterArgToken);
        }
        continue;
      }
      auto access = classifyAccessForToken(op, token);
      if (access == AccessKind::FreshWrite ||
          access == AccessKind::FreshWriteMMA)
        return true;
      if (access != AccessKind::None)
        return false;
      if (usesTokenView(op, token))
        return false;
    }
    // If we reached end of a for-loop body without finding a use, the token
    // may be carried to the next iteration via yield/iter_args. Check if the
    // first use at the TOP of the loop body is a FreshWrite.
    if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
      auto yieldOp = cast<scf::YieldOp>(block->getTerminator());
      if (auto pos = findValuePosInRange(yieldOp.getOperands(), token)) {
        Value iterArgToken = forOp.getRegionIterArgs()[*pos];
        return isFirstUseFreshWriteInBlock(block, iterArgToken);
      }
    }
    return false;
  }

  // --- Control-flow threading -----------------------------------------------

  // Collect lane keys (pid, semaphore-index) touched by acquires in a block.
  void collectUsedLanesInBlock(Block *block,
                               DenseMap<int, DenseSet<int>> &usedByPartition) {
    for (auto &op : *block) {
      if (auto acquireOp = getAcquireOp(&op)) {
        if (auto idx = getSemaphoreIndex(acquireOp.getSemaphore())) {
          for (int pid : getAcquireProcessingOrder(&op))
            usedByPartition[pid].insert(*idx);
        }
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op))
        collectUsedLanesInBlock(forOp.getBody(), usedByPartition);
      else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        collectUsedLanesInBlock(ifOp.thenBlock(), usedByPartition);
        if (ifOp.elseBlock())
          collectUsedLanesInBlock(ifOp.elseBlock(), usedByPartition);
      }
    }
  }

  SmallVector<LaneKey> getOrderedUsedLanes(
      const DenseMap<int, DenseSet<int>> &usedByPartition) const {
    SmallVector<LaneKey> lanes;
    for (int pid : allGroupPartitionIds) {
      auto it = usedByPartition.find(pid);
      if (it == usedByPartition.end())
        continue;
      SmallVector<int> sorted(it->second.begin(), it->second.end());
      llvm::sort(sorted);
      for (int semaIdx : sorted)
        lanes.push_back({pid, semaIdx});
    }
    return lanes;
  }

  SmallVector<LaneKey> getOrderedUsedLanes(Block *block) {
    DenseMap<int, DenseSet<int>> usedByPartition;
    collectUsedLanesInBlock(block, usedByPartition);
    return getOrderedUsedLanes(usedByPartition);
  }

  Value getLanePhase(State &state, LaneKey key) {
    auto it = state.lanes.find(key);
    if (it != state.lanes.end())
      return it->second;
    Value initPhase = state.basePhases[key.second];
    state.lanes.insert({key, initPhase});
    return initPhase;
  }

  // Infer partition IDs for a yield argument value.
  SetVector<int> inferPartitionIds(Value arg, int fallbackPartitionId) {
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
      argIds.insert(fallbackPartitionId);
    return argIds;
  }

  void assignStateInForOp(scf::ForOp forOp, State &state) {
    Value newTok;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token))
      newTok = forOp.getRegionIterArgs()[*pos];

    bool hasStageUse = analyzeUseInBlock(forOp.getBody(), newTok);
    auto usedLanes = getOrderedUsedLanes(forOp.getBody());
    if (!hasStageUse && usedLanes.empty())
      return;

    llvm::MapVector<int, Value *> tokenRefs;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), state.token)) {
      tokenRefs[*pos] = &state.token;
      state.token = forOp.getRegionIterArgs()[*pos];
    }

    // Keep the final argument order stable with previous two-pass behavior:
    // stage first, then phase lanes in (pid-order, sema-index-order).
    SmallVector<Value> extraIterArgs;
    extraIterArgs.push_back(state.stage);
    for (LaneKey key : usedLanes)
      extraIterArgs.push_back(getLanePhase(state, key));

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    assert(hasPartition(forOp));
    auto forOpIds = getPartitionIds(forOp);
    auto forOpOutputsIds = getPartitionOutputs(forOp);
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    state.stage = forOp.getRegionIterArgs()[nArgs];
    for (auto [i, key] : llvm::enumerate(usedLanes))
      state.lanes[key] = forOp.getRegionIterArgs()[nArgs + 1 + i];

    auto stateInBlock = assignStateInBlock(forOp.getBody(), state);

    SmallVector<Value> extraYieldArgs;
    extraYieldArgs.push_back(stateInBlock.stage);
    for (LaneKey key : usedLanes)
      extraYieldArgs.push_back(getLanePhase(stateInBlock, key));
    appendToForOpYield(forOp, extraYieldArgs);
    tokToStagePosMap[{forOp, state.token}] = nArgs;
    tokToStagePosMap[{forOp.getBody()->getTerminator(), stateInBlock.token}] =
        nArgs;

    // Annotate stage with all group partition IDs.
    forOpIds.insert(allGroupPartitionIds.begin(), allGroupPartitionIds.end());
    forOpOutputsIds.push_back(SetVector<int>(allGroupPartitionIds.begin(),
                                             allGroupPartitionIds.end()));
    // Annotate phase lanes with per-lane partition IDs.
    for (auto [i, key] : llvm::enumerate(usedLanes)) {
      auto argIds = inferPartitionIds(extraYieldArgs[1 + i], key.first);
      forOpIds.insert(argIds.begin(), argIds.end());
      forOpOutputsIds.push_back(argIds);
    }
    setPartition(forOp, forOpIds);
    setPartitionOutputs(forOp, forOpOutputsIds);

    state.stage = forOp.getResult(nArgs);
    for (auto [i, key] : llvm::enumerate(usedLanes))
      state.lanes[key] = forOp.getResult(nArgs + 1 + i);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = forOp.getResult(idx);
  }

  void assignStateInIfOp(scf::IfOp ifOp, State &state) {
    DenseMap<int, DenseSet<int>> usedByPartition;
    collectUsedLanesInBlock(ifOp.thenBlock(), usedByPartition);
    if (ifOp.elseBlock())
      collectUsedLanesInBlock(ifOp.elseBlock(), usedByPartition);
    auto usedLanes = getOrderedUsedLanes(usedByPartition);
    if (usedLanes.empty())
      return;

    SmallVector<Type> extraIfResults;
    extraIfResults.push_back(state.stage.getType());
    for (LaneKey key : usedLanes)
      extraIfResults.push_back(getLanePhase(state, key).getType());

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
    if (auto pos = findValuePosInRange(thenYieldOp->getOperands(), state.token))
      tokenRefs[*pos] = &state.token;
    if (auto pos = findValuePosInRange(elseYieldOp->getOperands(), state.token))
      tokenRefs[*pos] = &state.token;
    tokToStagePosMap[{newIfOp.thenYield(), thenState.token}] =
        thenYieldOp.getNumOperands();
    tokToStagePosMap[{newIfOp.elseYield(), elseState.token}] =
        elseYieldOp.getNumOperands();

    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenState.stage);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseState.stage);
    for (LaneKey key : usedLanes) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  getLanePhase(thenState, key));
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  getLanePhase(elseState, key));
    }

    assert(hasPartition(ifOp));
    auto ifOpIds = getPartitionIds(ifOp);
    auto ifOpOutputsIds = getPartitionOutputs(ifOp);
    ifOp.erase();

    // Stage: all group partition IDs.
    ifOpIds.insert(allGroupPartitionIds.begin(), allGroupPartitionIds.end());
    ifOpOutputsIds.push_back(SetVector<int>(allGroupPartitionIds.begin(),
                                            allGroupPartitionIds.end()));
    // Phase: per-lane partition IDs.
    for (LaneKey key : usedLanes) {
      SetVector<int> phaseIds;
      for (Value arg :
           {getLanePhase(thenState, key), getLanePhase(elseState, key)}) {
        auto ids = inferPartitionIds(arg, key.first);
        phaseIds.insert(ids.begin(), ids.end());
      }
      ifOpOutputsIds.push_back(phaseIds);
    }

    setPartition(newIfOp, ifOpIds);
    setPartitionOutputs(newIfOp, ifOpOutputsIds);

    state.stage = newIfOp.getResult(nResults);
    for (auto [i, key] : llvm::enumerate(usedLanes))
      state.lanes[key] = newIfOp.getResult(nResults + 1 + i);
    for (auto [idx, tokenRef] : tokenRefs)
      *tokenRef = newIfOp.getResult(idx);
  }

  // --- Main block walk -----------------------------------------------------

  State assignStateInBlock(Block *block, State state) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto acquireOp = getAcquireOp(&op)) {
        ImplicitLocOpBuilder b(acquireOp.getLoc(), acquireOp);
        auto wsTag = getWarpSpecializeTag(&op);
        auto stageCluster = getStageCluster(&op);

        // Stage ops use StageOnly-style annotation.
        std::optional<SetVector<int>> stagePids;
        bool insideWsLoop = false;
        for (auto *parent = op.getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            if (forOp->hasAttr(kWarpSpecializeAttrName)) {
              insideWsLoop = true;
              break;
            }
          }
        }
        if (insideWsLoop)
          stagePids = allGroupPartitionIds;
        else if (hasPartition(&op))
          stagePids = getPartitionIds(&op);
        // Phase ops use PhaseOnly-style annotation.
        std::optional<SetVector<int>> phasePids;
        if (hasPartition(&op))
          phasePids = getPartitionIds(&op);

        auto effectiveWsTag = wsTag ? wsTag : groupWsTag;
        auto createOp = [&](std::optional<SetVector<int>> pids, auto opTy,
                            auto... args) {
          using ty = decltype(opTy);
          auto created = triton::gpu::createInto<ty>(
              b, b.getLoc(), pids, stageCluster,
              std::forward<decltype(args)>(args)...);
          if (effectiveWsTag)
            setWarpSpecializeTag(created, *effectiveWsTag);
          return created;
        };

        bool useSinglePhase = shouldUseSinglePhase(acquireOp.getSemaphore());

        // Stage update (StageOnly behavior).
        Value rawStage = state.stage;
        Value acquireStage = rawStage;
        bool advanceStage = isFirstUseFreshWriteAfterAcquire(acquireOp);
        if (advanceStage) {
          auto c1 = createOp(stagePids, arith::ConstantIntOp{}, 1, 32);
          auto c0 = createOp(stagePids, arith::ConstantIntOp{}, 0, 32);
          auto cDepth = createOp(stagePids, arith::ConstantIntOp{}, depth, 32);
          auto next = createOp(stagePids, arith::AddIOp{}, rawStage, c1);
          auto stageWrapped = createOp(stagePids, arith::CmpIOp{},
                                       arith::CmpIPredicate::eq, next, cDepth);
          auto wrapped = createOp(stagePids, arith::SelectOp{},
                                  stageWrapped, c0, next);
          acquireStage = wrapped;
        }
        state.stage = acquireStage;
        acquireOp.getStageMutable().assign(acquireStage);
        state.token = acquireOp.getToken();

        // Phase update (preserve per-partition PhaseOnly behavior).
        auto semaIdx = getSemaphoreIndex(acquireOp.getSemaphore());
        assert(semaIdx && "acquire op must reference a group semaphore");
        for (int pid : getAcquireProcessingOrder(&op)) {
          LaneKey key{pid, *semaIdx};
          Value lanePhase = getLanePhase(state, key);
          if (useSinglePhase) {
            auto c1 = createOp(phasePids, arith::ConstantIntOp{}, 1, 32);
            auto nextPhase =
                createOp(phasePids, arith::XOrIOp{}, lanePhase, c1);
            // Phase flips only when stage wraps (matching triton-01 semantics).
            // Recompute wrap condition in consumer's partition context to avoid
            // cross-partition SSA references from state.stageWrapped.
            auto c0 = createOp(phasePids, arith::ConstantIntOp{}, 0, 32);
            auto wrapped = createOp(phasePids, arith::CmpIOp{},
                                    arith::CmpIPredicate::eq, acquireStage, c0);
            lanePhase = createOp(phasePids, arith::SelectOp{}, wrapped,
                                 nextPhase, lanePhase);
          } else {
            auto c1 = createOp(phasePids, arith::ConstantIntOp{}, 1, 32);
            auto phaseBit =
                createOp(phasePids, arith::ShLIOp{}, c1, acquireStage);
            lanePhase =
                createOp(phasePids, arith::XOrIOp{}, lanePhase, phaseBit);
          }
          state.lanes[key] = lanePhase;
          acquireOp.getPhaseMutable().assign(lanePhase);
        }
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
    }

    return state;
  }

  // --- Token-chain propagation -----------------------------------------------

  void propagateStage(Value token, Value stage,
                      DenseSet<Operation *> &visited) {
    for (auto &tokUse : token.getUses()) {
      auto owner = tokUse.getOwner();
      if (visited.contains(owner))
        continue;
      visited.insert(owner);
      if (auto stageOp = dyn_cast<SemaphoreStageInterface>(owner)) {
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
          // Widen only the stage slot (phases are handled separately)
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

  // --- Entry point (per buffer group) --------------------------------------

  static LogicalResult run(ArrayRef<SemaphoreCreateOp> semaOps) {
    if (semaOps.empty())
      return success();
    auto firstSemaOp = semaOps.front();
    auto lastSemaOp = semaOps.back();
    int depth = 1;
    for (auto semaOp : semaOps)
      depth =
          std::max(depth, cast<SemaphoreType>(semaOp.getType()).getNumStages());
    depth = std::max(depth, 1);

    // Collect all partition IDs and wsTag across all acquires in the group.
    SetVector<int> allPartitionIds;
    std::optional<int> groupWsTag;
    for (auto semaOp : semaOps) {
      for (auto user : semaOp->getUsers()) {
        if (isa<SemaphoreAcquireOp>(user) && hasPartition(user)) {
          auto ids = getPartitionIds(user);
          allPartitionIds.insert(ids.begin(), ids.end());
          if (!groupWsTag)
            if (auto tag = getWarpSpecializeTag(user))
              groupWsTag = *tag;
        }
      }
    }
    if (allPartitionIds.empty())
      allPartitionIds.insert(0);

    // Compute single-phase eligibility per buffer group.
    // Create temporary instance for analysis (empty mode map).
    DenseMap<Value, bool> emptyModes;
    AssignSemaphoreStagePhase analyzer(semaOps, std::move(emptyModes),
                                       allPartitionIds, depth, groupWsTag);
    bool eligible = analyzer.computeSinglePhaseEligibility();

    DenseMap<Value, bool> useSinglePhaseBySemaphore;
    for (auto semaOp : semaOps) {
      semaOp->setAttr(kUseSinglePhaseAttrName,
                      BoolAttr::get(semaOp.getContext(), eligible));
      useSinglePhaseBySemaphore[semaOp.getResult()] = eligible;
    }

    // Insert after the last semaOp so all semaphores are defined.
    ImplicitLocOpBuilder b(lastSemaOp.getLoc(), lastSemaOp);
    b.setInsertionPointAfter(lastSemaOp);

    State initState;
    initState.stage = arith::ConstantIntOp::create(b, depth - 1, 32);
    // Per-semaphore initial phases:
    // single-phase:  isReleased=true  -> 0, isReleased=false -> 1
    // multiphase:    isReleased=true  -> 0, isReleased=false -> -1
    for (auto semaOp : semaOps) {
      bool useSinglePhase =
          useSinglePhaseBySemaphore.lookup(semaOp.getResult());
      uint32_t initPhase = 0;
      if (useSinglePhase)
        initPhase = semaOp.getIsReleased() ? 0x00000000u : 0x00000001u;
      else
        initPhase = semaOp.getIsReleased() ? 0x00000000u : 0xFFFFFFFFu;
      initState.basePhases.push_back(
          arith::ConstantIntOp::create(b, static_cast<int64_t>(initPhase), 32));
    }

    // Set wsTag on init constants so backward-slice annotation doesn't trigger
    // the assert(hasWarpSpecializeTag) in PartitionLoops.
    if (groupWsTag) {
      setWarpSpecializeTag(initState.stage.getDefiningOp(), *groupWsTag);
      for (auto phase : initState.basePhases)
        setWarpSpecializeTag(phase.getDefiningOp(), *groupWsTag);
    }

    AssignSemaphoreStagePhase impl(semaOps,
                                   std::move(useSinglePhaseBySemaphore),
                                   allPartitionIds, depth, groupWsTag);
    impl.assignStateInBlock(firstSemaOp->getBlock(), initState);

    // Propagate stage to release/buffer ops via token chain.
    for (auto semaOp : semaOps) {
      for (auto user : semaOp->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          DenseSet<Operation *> visited;
          impl.propagateStage(acquireOp.getToken(), acquireOp.getStage(),
                              visited);
        }
      }
    }

    // Verify: all acquires must have stage/phase assigned by the main walk.
    // All release/buffer ops must have stage set via propagateStage.
    for (size_t semaIdx = 0; semaIdx < semaOps.size(); ++semaIdx) {
      auto semaOp = semaOps[semaIdx];
      for (auto user : semaOp->getUsers()) {
        if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(user)) {
          assert(acquireOp.getStage() &&
                 "acquire missing stage after assign-semaphore-stage-phase");
          assert(acquireOp.getPhase() &&
                 "acquire missing phase after assign-semaphore-stage-phase");
        }
        if (auto stageOp = dyn_cast<SemaphoreStageInterface>(user))
          assert(stageOp.getStage() &&
                 "release/buffer missing stage after propagation");
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
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
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
