// Blackwell/NVWS IR-shape adapters used by the copied Meta data-partition
// algorithm. These helpers extend sliceability; they do not choose a
// partition dimension or change the partition-search policy.

static std::optional<Attribute> getSlicedTensorMemoryEncoding(
    MLIRContext *ctx, nvidia_gpu::TensorMemoryEncodingAttr tmem,
    ArrayRef<int64_t> slicedShape, unsigned dim) {
  if (slicedShape.size() < 2 || dim > 1)
    return std::nullopt;

  ArrayRef<int64_t> matrixShape = slicedShape.take_back(2);
  int64_t m = matrixShape[0];
  int64_t n = matrixShape[1];
  if (m <= 0 || n <= 0)
    return std::nullopt;

  unsigned blockM = tmem.getBlockM();
  unsigned blockN = tmem.getBlockN();
  unsigned ctaSplitM = tmem.getCTASplitM();
  unsigned ctaSplitN = tmem.getCTASplitN();

  if (dim == 0 && m < static_cast<int64_t>(blockM) * ctaSplitM) {
    if (m >= static_cast<int64_t>(64) * ctaSplitM) {
      blockM = 64;
    } else {
      return std::nullopt;
    }
  }

  if (n < static_cast<int64_t>(blockN) * ctaSplitN) {
    if (n % ctaSplitN != 0)
      return std::nullopt;
    blockN = n / ctaSplitN;
  }

  if (!llvm::is_contained({64u, 128u}, blockM) ||
      !llvm::isPowerOf2_32(blockN) || blockN > 512)
    return std::nullopt;
  if (m < static_cast<int64_t>(blockM) * ctaSplitM ||
      n < static_cast<int64_t>(blockN) * ctaSplitN)
    return std::nullopt;

  return triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
      ctx, blockM, blockN, tmem.getColStride(), ctaSplitM, ctaSplitN,
      tmem.getTwoCTAs());
}

static std::optional<MemDescType>
getSlicedMemDescType(MLIRContext *ctx, MemDescType type,
                     ArrayRef<int64_t> slicedShape, unsigned dim) {
  Attribute encoding = type.getEncoding();
  if (auto tmem =
          dyn_cast<nvidia_gpu::TensorMemoryEncodingAttr>(type.getEncoding())) {
    auto slicedEncoding =
        getSlicedTensorMemoryEncoding(ctx, tmem, slicedShape, dim);
    if (!slicedEncoding)
      return std::nullopt;
    encoding = *slicedEncoding;
  }
  return MemDescType::get(slicedShape, type.getElementType(), encoding,
                          type.getMemorySpace(), type.getMutableMemory());
}

static bool isGenericSliceableOp(Operation *op) {
  return !op->getName().getStringRef().contains('.') &&
         op->getNumRegions() == 0 &&
         !op->hasTrait<OpTrait::IsTerminator>();
}
