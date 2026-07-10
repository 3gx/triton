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

  using FirstTouchLoops = DenseMap<Operation *, DenseSet<Operation *>>;
  const DenseSet<Operation *> noFirstTouchLoops;
  std::optional<POURejection> rejection;
  auto buildPlan = [&](PlacementMode mode, const FirstTouchLoops &overrides)
      -> FailureOr<SmallVector<GroupDag, 0>> {
    rejection.reset();
    FailureOr<SmallVector<GroupDag, 0>> groupsOr = collectGroups(funcOp);
    if (failed(groupsOr))
      return failure();
    SmallVector<GroupDag, 0> candidate = std::move(*groupsOr);
    if (llvm::any_of(candidate, [&](GroupDag &g) {
          return failed(buildAccessDag(g, funcOp));
        }))
      return failure();
    int numTmemBlocks = 0;
    for (GroupDag &g : candidate) {
      Operation *key = g.pieceTable.members.front().allocOp;
      auto override = overrides.find(key);
      const DenseSet<Operation *> &loops =
          override == overrides.end() ? noFirstTouchLoops : override->second;
      FailureOr<std::optional<POURejection>> result =
          buildSyncDag(g, useMetaPartitioner, lowerSemaphoreNumStages,
                       numTmemBlocks, mode, loops);
      if (failed(result))
        return failure();
      if (*result) {
        rejection = std::move(**result);
        return failure();
      }
    }
    if (failed(finalizeSyncSchedule(candidate)))
      return failure();
    return candidate;
  };

  SmallVector<std::pair<Operation *, DictionaryAttr>, 0> authoredAttrs;
  funcOp.walk([&](Operation *op) {
    authoredAttrs.push_back({op, op->getAttrDictionary()});
  });
  auto restoreAttrs = [&] {
    for (auto &[op, attrs] : authoredAttrs)
      op->setAttrs(attrs);
  };
  auto emitRejection = [&](PlacementMode mode) {
    if (rejection)
      semaError(rejection->loop)
          << (mode == PlacementMode::POU
                  ? "point-of-use placement is unavailable for this loop: "
                  : "canonical first-touch placement could not satisfy this "
                    "loop: ")
          << rejection->reason;
  };

  FailureOr<SmallVector<GroupDag, 0>> groupsOr = failure();
  const FirstTouchLoops noOverrides;
  if (placementMode != PlacementMode::Auto) {
    groupsOr = buildPlan(placementMode, noOverrides);
  } else {
    FirstTouchLoops overrides;
    while (true) {
      // Auto may abandon a complete optimized candidate after construction or
      // schedule validation. Suppress only errors from that disposable
      // attempt; warnings still flow to the normal diagnostic handlers.
      {
        ScopedDiagnosticHandler capture(
            funcOp.getContext(), [](Diagnostic &diag) -> LogicalResult {
              return diag.getSeverity() == DiagnosticSeverity::Error
                         ? success()
                         : failure();
            });
        groupsOr = buildPlan(PlacementMode::Auto, overrides);
      }
      if (succeeded(groupsOr))
        break;
      restoreAttrs();
      if (rejection &&
          overrides[rejection->group].insert(rejection->loop).second)
        continue;
      groupsOr = buildPlan(PlacementMode::FirstTouch, noOverrides);
      break;
    }
  }
  if (failed(groupsOr)) {
    emitRejection(placementMode == PlacementMode::POU
                      ? PlacementMode::POU
                      : PlacementMode::FirstTouch);
    restoreAttrs();
    return failure();
  }
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
