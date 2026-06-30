// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-memory-planner="num-buffers=3 smem-alloc-algo=1 smem-budget=50000 smem-circular-reuse=true" 2>&1 | FileCheck %s

// CHECK: error: NVWS algorithm-1 circular local group has incompatible types

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @incompatible_phase4_candidates(
      %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>,
      %lb: i32, %ub: i32, %step: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      nvws.descriptor_load %a_desc[%iv, %iv] 16384 %a {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %b_desc[%iv, %iv] 16384 %b {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      "consume"(%a) {async_task_id = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
      "consume"(%b) {async_task_id = array<i32: 1>} : (!ttg.memdesc<64x128xf16, #shared, #smem, mutable>) -> ()
    } {tt.warp_specialize}
    tt.return
  }
}
