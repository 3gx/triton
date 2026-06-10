#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

using namespace mlir;
using triton::nvws::AsyncOp;
namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;
namespace nvws = triton::nvws;

// Stage implementations (single translation unit; one header per stage —
// fable/new-insert-semas-plan-2.md section 1).
#include "InsertSemas.h"
#include "InsertSemasAccessDag.h"
#include "InsertSemasOwnerDag.h"
#include "InsertSemasSyncDag.h"
#include "InsertSemasEmitIR.h"

// ---------------------------------------------------------------------------
// Dispatcher. Commit 1 of the plan: stage 1 (ACCESS-DAG) only — pure
// analysis + diagnostics + dump. The pass mutates nothing until the EMIT-IR
// commit lands; lit failures of test/NVWS/insert_semas* are expected.
// ---------------------------------------------------------------------------
// useMetaPartitioner has exactly ONE consumer in this pass: the TMEM
// backing-stage decision (meta => numStages=1; see computeBackingPlan in
// InsertSemasSyncDag.h). It influences nothing else.
LogicalResult runOnFunction(triton::FuncOp funcOp, bool useMetaPartitioner) {
  // Only process functions that contain a warp-specialized loop.
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  // Stage 1: discovery + pieces + access events + region effect summaries.
  SmallVector<GroupDag, 0> groups = collectGroups(funcOp);
  for (GroupDag &g : groups)
    if (failed(buildAccessDag(g, funcOp)))
      return failure();

  // Stage 2: Enter/Exit brackets + per-piece carried owners (in-place
  // extension; the ACCESS dump filters bracket rows).
  for (GroupDag &g : groups)
    if (failed(buildOwnerDag(g)))
      return failure();

  // Stage 3: the ownership walk -> edges -> semaphores (in-place sync-node
  // injection) -> entry acquires, crossings, requiredParts, BackingPlan.
  // numTmemBlocks accumulates the 1x/2x capacity check across groups.
  int numTmemBlocks = 0;
  for (GroupDag &g : groups)
    if (failed(buildSyncDag(g, funcOp, useMetaPartitioner, numTmemBlocks)))
      return failure();

  if (shouldDumpDag()) {
    llvm::errs() << "==== NVWS InsertSemas (commit 4: ACCESS-DAG + "
                    "OWNER-DAG + SYNC-DAG + EMIT) ====\n";
    llvm::errs() << "function: @" << funcOp.getName() << "\n";
    llvm::errs() << "groups: " << groups.size() << "\n";
    for (GroupDag &g : groups) {
      dumpGroupAccessDag(g, funcOp);
      dumpGroupOwnerDag(g, funcOp);
      dumpGroupSyncDag(g, funcOp);
    }
  }

  // Stage 4: EMIT-IR (the only mutating stage).
  return emitIR(funcOp, groups);
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
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
