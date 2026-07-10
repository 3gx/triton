// Four-step dispatcher; see sema-docs/insert-semas/overview.md.
#include "InsertSemas.h"
#include "mlir/IR/Diagnostics.h"
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
  if (value == "pou")
    return PlacementMode::POU;
  anchor->emitError() << "nvws-insert-semas: invalid placement mode '" << value
                      << "' (expected auto, first-touch, or pou)";
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

  auto buildPlan = [&](PlacementMode mode)
      -> FailureOr<SmallVector<GroupDag, 0>> {
    FailureOr<SmallVector<GroupDag, 0>> groupsOr = collectGroups(funcOp);
    if (failed(groupsOr))
      return failure();
    SmallVector<GroupDag, 0> candidate = std::move(*groupsOr);
    if (llvm::any_of(candidate, [&](GroupDag &g) {
          return failed(buildAccessDag(g, funcOp));
        }))
      return failure();
    int numTmemBlocks = 0;
    if (llvm::any_of(candidate, [&](GroupDag &g) {
          return failed(buildSyncDag(g, useMetaPartitioner,
                                     lowerSemaphoreNumStages, numTmemBlocks,
                                     mode));
        }))
      return failure();
    if (failed(finalizeSyncSchedule(candidate)))
      return failure();
    return candidate;
  };

  FailureOr<SmallVector<GroupDag, 0>> groupsOr = failure();
  if (placementMode != PlacementMode::Auto) {
    groupsOr = buildPlan(placementMode);
  } else {
    SmallVector<std::pair<Operation *, DictionaryAttr>, 0> authoredAttrs;
    funcOp.walk([&](Operation *op) {
      authoredAttrs.push_back({op, op->getAttrDictionary()});
    });
    {
      // Auto may abandon a complete optimized candidate after construction or
      // schedule validation. Suppress only errors from that disposable
      // attempt; warnings still flow to the normal diagnostic handlers.
      ScopedDiagnosticHandler capture(
          funcOp.getContext(), [](Diagnostic &diag) -> LogicalResult {
            return diag.getSeverity() == DiagnosticSeverity::Error
                       ? success()
                       : failure();
          });
      groupsOr = buildPlan(PlacementMode::Auto);
    }
    if (failed(groupsOr)) {
      for (auto &[op, attrs] : authoredAttrs)
        op->setAttrs(attrs);
      groupsOr = buildPlan(PlacementMode::FirstTouch);
    }
  }
  if (failed(groupsOr))
    return failure();
  SmallVector<GroupDag, 0> groups = std::move(*groupsOr);
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
