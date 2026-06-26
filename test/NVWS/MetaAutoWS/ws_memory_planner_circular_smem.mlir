// RUN: triton-opt %s --nvws-memory-planner="num-buffers=2 smem-circular-reuse" | FileCheck %s
// RUN: triton-opt %s --nvws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=200000 smem-circular-reuse" | FileCheck %s

// Cross-stage descriptor-load SMEM tiles model the K/V pattern from persistent
// attention. Circular reuse must not fold two independent descriptor-load
// channels into one backing allocation.

// CHECK-LABEL: tt.func public @cross_stage_descriptor_loads_do_not_share_circular_backing
// CHECK: %{{.*}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 0 : i32}
// CHECK: %{{.*}} = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 1 : i32}

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cross_stage_descriptor_loads_do_not_share_circular_backing(
      %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>,
      %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %k = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step  : i32 {
      nvws.descriptor_load %desc_k[%i, %c0_i32] 16384 %k {async_task_id = array<i32: 2>, loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_v[%i, %c0_i32] 16384 %v {async_task_id = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    } {async_task_id = array<i32: 0, 1, 2>, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.partition.types = ["default", "gemm", "load"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
