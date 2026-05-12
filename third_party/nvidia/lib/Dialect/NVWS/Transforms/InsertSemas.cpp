#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-insert-semas"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

int getWsTag(Operation *op) {
  while (op && !hasWarpSpecializeTag(op)) {
    op = op->getParentOfType<scf::ForOp>();
  }
  assert(op);
  return *getWarpSpecializeTag(op);
}

using PartitionId = std::pair<int /* PartitionId*/, int /* WsTag*/>;
std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0) {
  if (!hasPartition(op))
    return std::nullopt;
  auto partitionIds = getPartitionIds(op);
  if (op->getNumRegions() > 0) {
    partitionIds = getPartitionOutputs(op)[pos];
  }
  assert(partitionIds.size() == 1);
  return std::make_pair(*partitionIds.begin(), getWsTag(op));
}

void assignStage(OpBuilder &b, Operation *op, StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}

template <typename OpT, typename... Args>
OpT createInto(
    OpBuilder &b, Location loc,
    std::pair<std::optional<PartitionId>, StageCluster> partitionIdStageCluster,
    Args &&...args) {
  std::optional<SetVector<int>> partitionIds = SetVector<int>();
  std::optional<int> wsTag;
  if (partitionIdStageCluster.first) {
    auto [id, tag] = *partitionIdStageCluster.first;
    wsTag = tag;
    partitionIds->insert(id);
  } else {
    partitionIds = std::nullopt;
  }
  auto op = triton::gpu::createInto<OpT>(b, loc, partitionIds,
                                         partitionIdStageCluster.second,
                                         std::forward<Args>(args)...);
  if (wsTag) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !hasWarpSpecializeTag(forOp)) {
      forOp = forOp->template getParentOfType<scf::ForOp>();
    }
    // only set wsTag if op is outside tt.ws loop
    if (!forOp) {
      setWarpSpecializeTag(op, *wsTag);
    }
  }
  return op;
}

static constexpr StringLiteral kBufferIdAttrName = "buffer.id";
static constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
static constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";


LogicalResult runOnFunction(triton::FuncOp funcOp) {

  return success();
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<
          NVWSInsertSemas> {
public:
  void runOnOperation() override {
    auto walkResult = getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace triton
} // namespace mlir
