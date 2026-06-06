#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_COMMON_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_COMMON_H_

#include "InsertSemasModel.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace mlir::triton::nvws::insert_semas {

std::optional<int> tryGetWsTag(Operation *op) {
  while (op && !gpu::hasWarpSpecializeTag(op))
    op = op->getParentOfType<scf::ForOp>();
  if (!op)
    return std::nullopt;
  return *gpu::getWarpSpecializeTag(op);
}

Operation *getTagSourceOp(Operation *op) {
  if (!op)
    return nullptr;
  if (gpu::hasWarpSpecializeTag(op))
    return op;
  Operation *p = op->getParentOfType<scf::ForOp>();
  while (p && !gpu::hasWarpSpecializeTag(p))
    p = p->getParentOfType<scf::ForOp>();
  return p;
}

std::optional<PartitionId> getPartitionId(Operation *op, int pos) {
  if (!gpu::hasPartition(op))
    return std::nullopt;
  auto partitionIds = gpu::getPartitionIds(op);
  if (op->getNumRegions() > 0) {
    auto outputs = gpu::getPartitionOutputs(op);
    if (pos >= static_cast<int>(outputs.size()))
      return std::nullopt;
    partitionIds = outputs[pos];
  }
  if (partitionIds.size() != 1)
    return std::nullopt;
  auto tag = tryGetWsTag(op);
  if (!tag)
    return std::nullopt;
  return std::make_pair(*partitionIds.begin(), *tag);
}

std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

std::optional<int64_t> getBufferId(Operation *op) {
  return getI64Attr(op, kBufferIdAttrName);
}

int64_t getBufferOffset(Operation *op) {
  return getI64Attr(op, kBufferOffsetAttrName).value_or(0);
}

bool isSemaphoreBackingAlloc(Operation *op) {
  return op->hasAttr("nvws.semaphore.backing");
}

bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

AsyncOp getAsyncPayload(Operation *op) {
  if (!op)
    return AsyncOp::NONE;
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op))
    if (Value src = localAlloc.getSrc())
      if (Operation *def = src.getDefiningOp())
        return getAsyncPayload(def);
  if (isa<nvidia_gpu::MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  StringRef name = op->getName().getStringRef();
  if (name == "tt.descriptor_load" || name == "tt.descriptor_gather" ||
      name == "nvws.descriptor_load" || name == "nvws.descriptor_gather")
    return AsyncOp::TMALoad;
  return AsyncOp::NONE;
}

RankedTensorType getTensorTypeFromScalar(OpBuilder &builder, Value scalar) {
  auto mod = scalar.getParentRegion()->getParentOfType<ModuleOp>();
  auto nWarps = gpu::lookupNumWarps(mod);
  auto threadsPerWarp = gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  int CTAs = gpu::TritonGPUDialect::getNumCTAs(mod);
  Attribute encoding = gpu::getDefaultBlockedEncoding(
      builder.getContext(), {1}, nWarps, threadsPerWarp, CTAs);
  return RankedTensorType::get({1}, scalar.getType(), encoding);
}

int getTxCount(Operation *descOp) {
  auto getTensorTypeAndDesc =
      [](Operation *op) -> std::pair<RankedTensorType, Value> {
    if (auto loadOp = dyn_cast<triton::DescriptorLoadOp>(op))
      return {loadOp.getType(), loadOp.getDesc()};
    if (auto gatherOp = dyn_cast<triton::DescriptorGatherOp>(op))
      return {gatherOp.getType(), gatherOp.getDesc()};
    llvm_unreachable("unsupported descriptor operation type");
  };
  auto [tensorType, desc] = getTensorTypeAndDesc(descOp);
  auto encoding = nvidia_gpu::getEncodingFromDescriptor(descOp, tensorType, desc);
  auto shapePerCTA = gpu::getShapePerCTA(encoding, tensorType.getShape());
  return product(shapePerCTA) *
         getIntOrFloatOrPtrBitWidth(tensorType.getElementType()) / 8;
}

bool isEventInScopeForRegion(Operation *tagSourceOp, Operation *eventOp,
                             Region *region) {
  if (!tagSourceOp)
    return true;
  if (tagSourceOp == eventOp)
    return true;
  Operation *parent = region->getParentOp();
  while (parent) {
    if (parent == tagSourceOp)
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

void buildProgramOrderRank(mlir::triton::FuncOp funcOp,
                           DenseMap<Operation *, unsigned> &rank) {
  rank.clear();
  unsigned i = 0;
  funcOp.walk([&](Operation *op) { rank[op] = i++; });
}

unsigned maxRankInSubtree(Operation *op,
                          const DenseMap<Operation *, unsigned> &rank) {
  unsigned mx = rank.lookup(op);
  op->walk([&](Operation *o) {
    auto it = rank.find(o);
    if (it != rank.end())
      mx = std::max(mx, it->second);
  });
  return mx;
}

} // namespace mlir::triton::nvws::insert_semas

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_COMMON_H_
