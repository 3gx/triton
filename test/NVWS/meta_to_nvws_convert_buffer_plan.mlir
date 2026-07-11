// RUN: triton-opt %s --nvws-meta-to-nvws-convert | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#acc = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @buffer_plan
  tt.func @buffer_plan(%value: tensor<64x64xf16, #blocked>, %lb: i32,
                       %ub: i32, %step: i32) {
    // CHECK: %[[CLONE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-NOT: buffer.id = 7
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[HOST:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret %[[HOST]] {buffer.copy = 2 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32}
    // CHECK-NOT: allocation.reuseTarget
    %reuse = ttg.local_alloc {allocation.reuseTarget = 3 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32}
    %single = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 23 : i32}
    // CHECK-NOT: buffer.circular
    // CHECK-NOT: buffer.start
    %incompatible = ttg.local_alloc {allocation.reuseTarget = 9 : i32, buffer.copy = 1 : i32, buffer.id = 23 : i32} : () -> !ttg.memdesc<64x64xf16, #shared64, #smem, mutable>
    // CHECK-NOT: async_task_id
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: %[[PLANNED:.*]] = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 24 : i32, ttg.partition = array<i32: 2>}
      %planned = ttg.local_alloc {buffer.copy = 2 : i32, buffer.id = 24 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, %[[PLANNED]] {ttg.partition = array<i32: 2>}
      ttg.local_store %value, %planned {async_task_id = array<i32: 2>} : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_load %[[PLANNED]] {ttg.partition = array<i32: 2>}
      %loaded = ttg.local_load %planned {async_task_id = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "gemm"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: tt.func @tmem_plan
  tt.func @tmem_plan(%value: tensor<128x128xf32, #acc>, %lb: i32,
                     %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %true = arith.constant {async_task_id = array<i32: 2>} true
      // CHECK: %[[TMEM:.*]], %[[TOKEN:.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 25 : i32, ttg.partition = array<i32: 2>}
      %buffer, %token = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 25 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: %[[STORED:.*]] = ttng.tmem_store %{{.*}}, %[[TMEM]][%[[TOKEN]]], %{{.*}} {ttg.partition = array<i32: 2>}
      %stored = ttng.tmem_store %value, %buffer[%token], %true {async_task_id = array<i32: 2>} : tensor<128x128xf32, #acc> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_load %[[TMEM]][%[[STORED]]] {ttg.partition = array<i32: 2>}
      %loaded, %load_token = ttng.tmem_load %buffer[%stored] {async_task_id = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc>
      scf.yield {async_task_id = array<i32: 0, 2>}
    } {async_task_id = array<i32: 0, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "unused", "gemm"],
       ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
