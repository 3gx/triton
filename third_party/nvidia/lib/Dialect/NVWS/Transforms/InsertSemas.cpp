// v4 commit 5: discovery + ACCESS DAG + OWNERSHIP DAG + RAW-SYNC DAG +
// OPT-SYNC DAG + semaphore IR emission.
//
// Per meta2nvws-plan/per-edge-sema-plan.v4.md Implementation Plan, this
// commit adds the final stage of the v4 pipeline:
//
//   discover backing buffers
//     -> build ACCESS DAG per buffer
//     -> build OWNERSHIP DAG per (logicalGroupId, resourceKey)
//     -> derive RAW-SYNC DAG per (logicalGroupId, resourceKey)
//     -> derive OPT-SYNC DAG via fanout/fanin combines
//     -> render nvws.semaphore.* IR from the OPT-SYNC DAG
//
// When NVWS_INSERT_SEMA_DUMP_DAG=1, the pass prints each planned stage to
// stderr before emitting IR.

#include "Utilities.h"
#include "InsertSemasModel.h"
#include "InsertSemasCommon.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;
using namespace triton::nvws::insert_semas;

// Stage implementations for the private InsertSemas pipeline.
#include "InsertSemasAccessDag.h"
#include "InsertSemasOwnerDag.h"
#include "InsertSemasRawSyncDag.h"
#include "InsertSemasOptSyncDag.h"
#include "InsertSemasEmitSchedule.h"
#include "InsertSemasEmitter.h"

// ---------------------------------------------------------------------------
// Top-level pipeline (commit 5 stage).
// ---------------------------------------------------------------------------

struct PlannedResource {
  ResourcePlan plan;
  SyncPlan syncPlan;
  OptSyncDag optDag;
};

static bool tmemPlanNeedsMultiStage(BufferGroup &group,
                                    ArrayRef<PlannedResource> planned) {
  if (!group.isTmem() || group.members.size() != 1)
    return false;
  bool hasLinearChain = false;
  for (const PlannedResource &resource : planned)
    for (const SyncGroup &syncGroup : resource.optDag.groups)
      hasLinearChain |= syncGroup.kind == SyncGroupKind::LinearChain;
  if (!hasLinearChain)
    return false;

  for (auto [storeIdx, storeEvent] : llvm::enumerate(group.events)) {
    if (!storeEvent.owner || !isa<TMEMStoreOp>(storeEvent.op))
      continue;
    for (size_t mmaIdx = storeIdx + 1; mmaIdx < group.events.size();
         ++mmaIdx) {
      AccessEvent &mmaEvent = group.events[mmaIdx];
      if (!mmaEvent.owner || !isa<MMAv5OpInterface>(mmaEvent.op) ||
          sameOwner(storeEvent.owner, mmaEvent.owner))
        continue;
      auto parentLoop = mmaEvent.op->getParentOfType<scf::ForOp>();
      if (!parentLoop || hasWarpSpecializeTag(parentLoop))
        continue;
      for (size_t loadIdx = mmaIdx + 1; loadIdx < group.events.size();
           ++loadIdx) {
        AccessEvent &loadEvent = group.events[loadIdx];
        if (loadEvent.owner && isa<TMEMLoadOp>(loadEvent.op) &&
            sameOwner(storeEvent.owner, loadEvent.owner))
          return false;
      }
    }
  }
  return true;
}

static int computeTmemSemaphoreNumStagesFromPlans(
    BufferGroup &group, ArrayRef<PlannedResource> planned, int numTmemBlocks,
    bool useMetaPartitioner) {
  bool isMultiStaged = tmemPlanNeedsMultiStage(group, planned);
  if (isMultiStaged) {
    for (BufferMember &member : group.members) {
      auto allocOp = cast<TMEMAllocOp>(member.allocOp);
      for (auto user : allocOp.getResult().getUsers()) {
        if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
          if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
            auto wsLoop = getOuterWSLoop(loop);
            // Determine if the MMA accumulator can be multibuffered.
            bool accIsMultiBuffered =
                // MMAs in subsequent iterations can be overlapped.
                !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
                // The accumulator is reset at some point, thus allowing
                // multibuffering.
                isAccMultibufferingPossible(mmaOp, loop) &&
                // The user didn't disable it with a flag.
                !getDisallowAccMultiBuffer(wsLoop) &&
                canDoubleBufferAcc(mmaOp, numTmemBlocks);
            isMultiStaged = isMultiStaged && accIsMultiBuffered;
          }
        }
      }
    }
  }
  auto numStages =
      useMetaPartitioner ? 1 + 0 * isMultiStaged : 1 + 1 * isMultiStaged;
  return numStages;
}

static LogicalResult runOnFunction(triton::FuncOp funcOp,
                                   bool useMetaPartitioner) {
  // Only process functions that contain a warp-specialized loop, matching
  // the prior pipeline gating.
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted()) return success();

  // Phase 1: discovery.
  SmallVector<BufferGroup, 0> groups = collectAllBackingGroups(funcOp);

  // Phase 2: collect access events per group.
  for (BufferGroup &group : groups)
    if (failed(collectEvents(group, funcOp))) return failure();

  // Phase 3: build program-order rank, used by the ownership planner.
  DenseMap<Operation *, unsigned> rank;
  buildProgramOrderRank(funcOp, rank);

  bool dumpDag = shouldDumpDag();
  if (dumpDag) {
    llvm::errs() << "==== NVWS InsertSemas (commit 5: discovery + ACCESS DAG + "
                    "OWNERSHIP DAG + RAW-SYNC DAG + OPT-SYNC DAG + EMIT) ====\n";
    llvm::errs() << "function: " << funcOp.getName() << "\n";
    llvm::errs() << "backing buffers: " << groups.size() << "\n";
  }

  int numTmemBlocks = 0;
  DenseMap<unsigned, int> numStagesByGroup;
  SetVector<Operation *> eraseAfterEmission;

  DenseMap<BackingKey, GroupBacking> backings;
  for (auto en : llvm::enumerate(groups)) {
    BufferGroup &group = en.value();
    if (dumpDag) {
      dumpBackingGroupHeader(group);
      dumpAccessDag(group, funcOp);
    }
    std::set<int64_t> keys;
    for (auto &m : group.members) keys.insert(m.resourceKey);
    SmallVector<PlannedResource, 4> plannedResources;
    for (int64_t key : keys) {
      if (isExplicitOffsetSourcefulTmemSelfContained(group, key))
        continue;
      buildProgramOrderRank(funcOp, rank);
      ResourcePlan plan = planResource(funcOp,
                                       static_cast<unsigned>(en.index()),
                                       group, key, rank);
      SyncPlan sp = buildSyncPlan(group, plan, funcOp);
      OptSyncDag opt = buildOptSyncDag(sp, plan, group);
      if (dumpDag) {
        dumpOwnershipDag(plan, group, funcOp);
        dumpRawSyncDag(sp, plan, group, funcOp);
        dumpOptSyncDag(opt, sp, plan, group, funcOp);
        bool seeded = llvm::any_of(opt.groups, [](const SyncGroup &g) {
          return g.kind == SyncGroupKind::InitialEmpty;
        });
        llvm::errs() << "RELEASED-SEMAPHORES buffer.id=" << opt.resource.first
                     << " resourceKey=" << opt.resource.second
                     << " seeded=" << (seeded ? "yes" : "no")
                     << " count=" << opt.releasedSemaphores.size();
        if (seeded && opt.releasedSemaphores.size() != 1)
          llvm::errs() << " <<M1-VIOLATION: seeded resource must have exactly 1>>";
        if (!seeded && !opt.releasedSemaphores.empty())
          llvm::errs() << " <<M1-VIOLATION: edge-free resource must have 0>>";
        llvm::errs() << "\n";
        for (auto &[gIdx, acquirer] : opt.releasedSemaphores) {
          llvm::errs() << "  seed: group=" << gIdx << " (" << opt.groups[gIdx].name
                       << ", kind=" << static_cast<int>(opt.groups[gIdx].kind)
                       << ") acquirer=";
          if (acquirer)
            llvm::errs() << "{p" << acquirer->first << ",ws" << acquirer->second
                         << "}";
          else
            llvm::errs() << "{root}";
          llvm::errs() << "\n";
        }
        dumpEmitSchedule(opt, sp, plan, group, funcOp);
      }
      plannedResources.push_back(
          {std::move(plan), std::move(sp), std::move(opt)});
    }
    if (group.isTmem()) {
      int numStages = computeTmemSemaphoreNumStagesFromPlans(
          group, plannedResources, numTmemBlocks, useMetaPartitioner);
      numStagesByGroup[static_cast<unsigned>(en.index())] = numStages;
      updateNumTmemBlocks(group, numStages, numTmemBlocks);
    }
    for (PlannedResource &planned : plannedResources) {
      // Earlier resources in the same backing group may have rewritten scf.for
      // / scf.if signatures while adding carrier tokens. Rebuild the current
      // resource plan against the live IR so commit5 emission consumes the
      // completed OPT-SYNC-DAG with live structured-op anchors.
      buildProgramOrderRank(funcOp, rank);
      ResourcePlan emitPlan =
          planResource(funcOp, static_cast<unsigned>(en.index()), group,
                       planned.plan.resource.second, rank);
      SyncPlan emitSyncPlan = buildSyncPlan(group, emitPlan, funcOp);
      OptSyncDag emitOptDag = buildOptSyncDag(emitSyncPlan, emitPlan, group);
      if (failed(emitResource(funcOp, group, emitPlan, emitSyncPlan, emitOptDag,
                              backings, numStagesByGroup, eraseAfterEmission)))
        return failure();
    }
    bool hasBacking = llvm::any_of(backings, [&](const auto &entry) {
      return entry.first.first == static_cast<unsigned>(en.index());
    });
    if (hasBacking)
      poisonOriginalTmemAllocTokens(group);
    eraseUnusedOriginals(group);
  }
  for (Operation *op : llvm::reverse(eraseAfterEmission))
    op->erase();
  splitSemaphoreIfForLoopScheduler(funcOp);
  poisonUnbackedCarrierTokenSlots(funcOp);
  coalesceTmemAllocsByBufferIdIntoViews(funcOp);
  eraseDeadTmemAllocs(funcOp);
  if (dumpDag)
    llvm::errs() << "\n";

  return success();
}

static void stripTemporarySemaphoreAttrs(triton::FuncOp funcOp) {
  funcOp.walk([&](Operation *op) { op->removeAttr("nvws.semaphore.backing"); });
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  using NVWSInsertSemasBase::NVWSInsertSemasBase;

  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp, useMetaPartitioner)))
        return WalkResult::interrupt();
      stripTemporarySemaphoreAttrs(funcOp);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
