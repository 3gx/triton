#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

// Helper to get printing flags with location info enabled
static OpPrintingFlags getOpPrintingFlagsWithLoc() {
  OpPrintingFlags flags;
  flags.enableDebugInfo();
  flags.printNameLocAsPrefix(true);
  return flags;
}

// Dump the IR to `meta-aws-logs/<NN>-<funcName>-<stepName>.mlir`. The path is
// relative to the process CWD; the convention is to launch triton-opt from the
// repo root where `meta-aws-logs/` lives. Stable numbering is used: each
// logical step in `runOnFuncOp` always has the same NN, so optional steps that
// don't fire simply leave a gap.
static void dumpStepIR(ModuleOp moduleOp, StringRef funcName, int stepNum,
                       StringRef stepName) {
  llvm::SmallString<128> dir("meta-aws-logs");
  if (auto ec = llvm::sys::fs::create_directories(dir)) {
    llvm::errs() << "[nvgpu-warp-specialization] failed to create " << dir
                 << ": " << ec.message() << "\n";
    return;
  }
  llvm::SmallString<256> path(dir);
  llvm::SmallString<128> filename;
  llvm::raw_svector_ostream(filename)
      << llvm::format("%02d-", stepNum) << funcName << "-" << stepName
      << ".mlir";
  llvm::sys::path::append(path, filename);
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "[nvgpu-warp-specialization] failed to open " << path
                 << ": " << ec.message() << "\n";
    return;
  }
  moduleOp.print(out, getOpPrintingFlagsWithLoc());
}

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
LogicalResult doMemoryPlanner(triton::FuncOp &funcOp, unsigned numBuffers,
                              StringRef readDecisionFile = "",
                              StringRef writeDecisionFile = "",
                              int smemAllocAlgo = 0, unsigned smemBudget = 0,
                              bool smemCircularReuse = false);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doBufferAllocation(triton::FuncOp &funcOp);
void doHoistLoopInvariantTMEMStore(triton::FuncOp &funcOp);
void removeRedundantTmemZeroStores(triton::FuncOp &funcOp);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doCodePartitionPost(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doPingPongPrep(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability, int defaultNumStages);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups,
                    int capability);
void doTMAStoreWaitReorder(triton::FuncOp &funcOp);
void doAnnotateTMAStoreWaits(triton::FuncOp &funcOp);
void doValidateTMAStoreAnnotations(triton::FuncOp &funcOp);
void doGenerateSubtiledRegion(triton::FuncOp &funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  PassManager pm(moduleOp.getContext());
  pm.addPass(triton::nvidia_gpu::
                 createTritonNvidiaGPUTestGenerateSubtiledRegionPass());
  // Convert tmem_load → reshape → trans → split chains in SubtiledRegionOp
  // setup regions into tmem_subslice + tmem_load pairs.
  pm.addPass(
      triton::nvidia_gpu::createTritonNvidiaGPUOptimizeTMemLayoutsPass());
  // Push setup values that are shared across all tiles into the tile body.
  pm.addPass(
      triton::nvidia_gpu::createTritonNvidiaGPUPushSharedSetupToTilePass());
  (void)pm.run(moduleOp);
}

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  // Remove the warp_specialize attribute from all loops in the function so
  // downstream passes (pipelining, latency assignment) don't mistakenly
  // treat the loop as warp-specialized.
  void removeWarpSpecializeAttr(triton::FuncOp funcOp) {
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->removeAttr(mlir::triton::kWarpSpecializeAttrName);
    });
  }

  void runOnFuncOp(triton::FuncOp funcOp, int defaultNumStages) {
    bool enabled = false;
    funcOp->walk([&](Operation *op) {
      if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id"))
        enabled = true;
      if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(
              triton::gpu::kPartitionAttrName))
        enabled = true;
    });
    if (!enabled) {
      SmallVector<scf::ForOp> loops;
      funcOp->walk([&](scf::ForOp forOp) {
        if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
          loops.push_back(forOp);
      });
      if (!loops.empty())
        enabled = true;
    }
    if (!enabled)
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4) {
      LDBG("Warp specialization requires num_warps=4, but got "
           << numWarps << ". Skipping.");
      removeWarpSpecializeAttr(funcOp);
      return;
    }

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
          if (!isa<scf::YieldOp>(&op))
            hasElse = true;
        }
      }
    });
    if (hasElse) {
      LDBG("Warp specialization does not support else blocks. Skipping.");
      removeWarpSpecializeAttr(funcOp);
      return;
    }

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    // FIXME: skip data partitioning for Blackwell.
    bool ForBlackWell = (capability / 10) > 9;
    unsigned numWarpGroups = ForBlackWell ? 2 : 3;

    auto dump = [&](int step, StringRef name) {
      if (true /* dumpIntermediateSteps - temporarily forced on */)
        dumpStepIR(moduleOp, funcOp.getName(), step, name);
    };

    dump(0, "input");

    int retCode = doTaskIdPropagate(funcOp);
    if (retCode == -1)
      signalPassFailure();
    dump(1, "doTaskIdPropagate");

    if (pingpongAutoWS) {
      doPingPongPrep(funcOp, numWarpGroups, capability, defaultNumStages);
      dump(2, "doPingPongPrep");
    }

    // Remove redundant TMEM zeroing stores before buffer allocation.
    // When a TMEMAllocOp is used as operand D of a TCGen5MMAOp with
    // useAccumulator=false (on the first iteration), any preceding
    // tmem_store of zeros is redundant — the MMA's useD=false already
    // zeros the accumulator. Removing the store prevents the autoWS
    // compiler from creating a cross-partition channel for it, which
    // would otherwise cause a race condition between the reduction
    // partition (zeroing) and the computation partition (reading) in
    // persistent kernels.
    removeRedundantTmemZeroStores(funcOp);
    dump(3, "removeRedundantTmemZeroStores");

    // Canonicalize the SMEM/TEM buffers.
    // Create buffers for register channels.
    doBufferAllocation(funcOp);
    dump(4, "doBufferAllocation");

    doHoistLoopInvariantTMEMStore(funcOp);
    dump(5, "doHoistLoopInvariantTMEMStore");

    if (failed(doMemoryPlanner(funcOp, numStages, /*readDecisionFile=*/"",
                               /*writeDecisionFile=*/"",
                               /*smemAllocAlgo=*/0, smemBudget))) {
      signalPassFailure();
      return;
    }
    dump(6, "doMemoryPlanner");

    if (generateSubtiledRegion) {
      doGenerateSubtiledRegion(funcOp);
      dump(7, "doGenerateSubtiledRegion");
    }

    doAnnotateTMAStoreWaits(funcOp);
    dump(8, "doAnnotateTMAStoreWaits");

    doValidateTMAStoreAnnotations(funcOp);
    dump(9, "doValidateTMAStoreAnnotations");

    doCodePartitionPost(funcOp, numStages);
    dump(10, "doCodePartitionPost");

    if (pingpongAutoWS) {
      doPingPongSync(funcOp, numWarpGroups, capability);
      dump(11, "doPingPongSync");
    }

    // Primary SubtiledRegionOp lowering path. By this point the tile body
    // has been optimized (OptimizeTMemLayouts + PushSharedSetupToTile ran
    // inside doGenerateSubtiledRegion), so tmem_loads are sunk close to
    // their consumers. doTokenLowering converts token annotations to
    // barrier annotations, then lowerSubtiledRegion unrolls the tile body
    // with per-tile barrier materialization.
    //
    // Multi-task SubtiledRegionOps were already lowered as fallbacks in
    // doCodePartition/doCodePartitionPost (before specializeRegion).
    doTokenLowering(funcOp, numWarpGroups - 1);
    dump(12, "doTokenLowering");

    {
      SmallVector<triton::nvidia_gpu::SubtiledRegionOp> remaining;
      funcOp.walk([&](triton::nvidia_gpu::SubtiledRegionOp op) {
        remaining.push_back(op);
      });
      for (auto op : remaining)
        triton::nvidia_gpu::lowerSubtiledRegion(op);
    }
    dump(13, "lowerSubtiledRegion");

    triton::gpu::doLoopSchedulePreprocessing(moduleOp, builder);
    dump(14, "doLoopSchedulePreprocessing");

    triton::gpu::scheduleLoops(moduleOp, defaultNumStages, true);
    dump(15, "scheduleLoops");

    doTMAStoreWaitReorder(funcOp);
    dump(16, "doTMAStoreWaitReorder");
  }

  void runOnOperation() override {
    assert(numStages >= 1 && "numStages must be at least 1");
    getOperation()->walk(
        [&](triton::FuncOp funcOp) { runOnFuncOp(funcOp, numStages); });

    // Cleanup code generated by warp specialization.
    RewritePatternSet patterns(&getContext());
    populateForOpDeadArgumentElimination(patterns);
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    mlir::triton::gpu::WarpSpecializeOp::getCanonicalizationPatterns(
        patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace mlir
