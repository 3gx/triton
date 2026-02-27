// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-tmem-semaphore -cse | FileCheck %s --implicit-check-not=nvws.aref

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @insert_tmem_semaphore_basic
  // CHECK: nvws.semaphore.create
  // CHECK: nvws.semaphore.acquire
  // CHECK: nvws.semaphore.buffer
  // CHECK: nvws.semaphore.release
  tt.func @insert_tmem_semaphore_basic(%ub: i32, %desc_a: !tt.tensordesc<tensor<128x64xf16, #shared>>, %desc_b: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>

    %acc, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init = ttng.tmem_store %zero, %acc[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %loop_token = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%iter = %init) -> (!ttg.async.token) : i32 {
      %off = arith.muli %i, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %a = tt.descriptor_load %desc_a[%c0_i32, %off] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %b = tt.descriptor_load %desc_b[%c0_i32, %off] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %lhs = ttg.local_alloc %a {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %rhs_src = ttg.local_alloc %b {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %rhs = ttg.memdesc_trans %rhs_src {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%iter], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}

    %result, %final_tok = ttng.tmem_load %acc[%loop_token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
