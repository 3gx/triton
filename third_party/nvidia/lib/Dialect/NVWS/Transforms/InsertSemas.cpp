// Four-step dispatcher; see sema-docs/insert-semas/overview.md.
#include "InsertSemas.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"

namespace mlir::triton {
#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
using namespace nvws_semas;

FailureOr<PlacementMode> parsePlacementMode(StringRef value,
                                            Operation *anchor) {
  if (value == "auto")
    return PlacementMode::Auto;
  if (value == "first-touch")
    return PlacementMode::FirstTouch;
  anchor->emitError() << "nvws-insert-semas: invalid placement mode '" << value
                      << "' (expected auto or first-touch)";
  return failure();
}

LogicalResult runOnFunction(triton::FuncOp funcOp, bool useMetaPartitioner,
                            int lowerSemaphoreNumStages,
                            PlacementMode placementMode) {
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(triton::kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  FailureOr<SmallVector<GroupDag, 0>> groupsOr = collectGroups(funcOp);
  if (failed(groupsOr)) return failure();
  SmallVector<GroupDag, 0> groups = std::move(*groupsOr);
  if (llvm::any_of(groups, [&](GroupDag &g) { return failed(buildAccessDag(g, funcOp)); }))
    return failure();
  int numTmemBlocks = 0;
  if (llvm::any_of(groups, [&](GroupDag &g) {
        return failed(buildSyncDag(g, useMetaPartitioner,
                                   lowerSemaphoreNumStages, numTmemBlocks,
                                   placementMode));
      }))
    return failure();
  if (failed(finalizeSyncSchedule(groups)))
    return failure();
  dumpSyncDags(groups, funcOp);
  return emitIR(funcOp, groups);
}
} // namespace

class NVWSInsertSemas : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  using NVWSInsertSemasBase::NVWSInsertSemasBase;
  void runOnOperation() override {
    FailureOr<PlacementMode> mode =
        parsePlacementMode(placementMode, getOperation());
    if (failed(mode))
      return signalPassFailure();
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp, useMetaPartitioner, numStages, *mode)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir::triton
