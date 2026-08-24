// RUN: not triton-opt %s --nvgpu-convert-descriptor-stores-to-nvws 2>&1 | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: descriptor_store with reduce_kind must be canonicalized to tt.descriptor_reduce before native Meta warp specialization
  tt.func public @legacy_reduce_kind(
      %desc: !tt.tensordesc<128x256xf32, #shared>, %i: i32,
      %src: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %desc[%i, %i], %src reduce_kind = add :
        !tt.tensordesc<128x256xf32, #shared>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
