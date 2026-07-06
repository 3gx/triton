// RUN: triton-opt %s -allow-unregistered-dialect --nvws-memory-planner="num-buffers=3 smem-alloc-algo=0" | FileCheck %s

// Algorithm 0 pools innermost-loop SMEM allocations by element type. A
// sourceful immutable view in that pool must not suppress circular staging for
// the compatible mutable members consumed by InsertSemas.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @mutable_subset_of_mixed_pool
  // CHECK: ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = [[ID:[0-9]+]] : i32, buffer.start = 0 : i32}
  // CHECK-NEXT: ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = [[ID]] : i32, buffer.start = 1 : i32}
  // CHECK: ttg.local_alloc %{{.*}} {async_task_id = array<i32: 0>, buffer.copy = 3 : i32, buffer.id = [[ID]] : i32}
  tt.func @mutable_subset_of_mixed_pool(%lb: i32, %ub: i32, %step: i32) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %value, %a {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %value, %b {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %av = ttg.local_load %a {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %bv = ttg.local_load %b {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %repacked = ttg.local_alloc %bv {async_task_id = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %repacked_value = ttg.local_load %repacked {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
      "use"(%av, %repacked_value) : (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
    }
    tt.return
  }
}
