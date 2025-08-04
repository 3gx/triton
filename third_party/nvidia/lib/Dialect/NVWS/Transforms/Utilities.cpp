#include "Utilities.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir::triton::nvws {

Operation *createAlloc(OpBuilder &builder, Location loc,
                       MemDescType memDescType, Value src) {
  if (isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
    return builder.create<LocalAllocOp>(loc, memDescType, src);
  } else {
    assert(isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()));
    return builder.create<TMEMAllocOp>(loc, memDescType, src);
  }
}

ArefCreateOp createArefCreateOp(OpBuilder &builder, ArrayRef<Type> arefTypes,
                                ValueRange allocOps, Location loc) {
  auto ctx = builder.getContext();
  auto arefTy = ArefType::get(ctx, TypeArrayAttr::get(ctx, arefTypes));
  return builder.create<ArefCreateOp>(loc, arefTy, allocOps);
}

using PartitionId = uint64_t;
PartitionId getPartitionId(uint32_t tag, uint32_t partition) {
  return uint64_t(tag) << 32 | partition;
}
uint32_t getPartitionTag(PartitionId partitionId) { return partitionId >> 32; }
uint32_t getPartitionIndex(PartitionId partitionId) {
  return partitionId & 0xFFFFFFFF;
}

std::optional<PartitionId> getPartitionId(Operation *op) {
  if (auto partitionAttr = op->getAttrOfType<IntegerAttr>(kPartitionAttrName)) {
    IntegerAttr tagAttr;
    while (op && !tagAttr) {
      tagAttr = op->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
      op = op->getParentOp();
    }
    if (tagAttr)
      return getPartitionId(tagAttr.getInt(), partitionAttr.getInt());
  }
  return {};
}

} // namespace mlir::triton::nvws
