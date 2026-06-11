// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Regression for the if-split workaround's partition metadata with
// NON-default partitions (mirrors upstream 9860c26c's test): the split
// must derive every attribute - nothing may assume partitions 0/1.
// Middle if = content union (result-less here -> the body partition);
// dead token slot dropped; enter/exit ifs carry the acquire/release
// partitions. See fd75df910d.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_workaround_nondefault_partitions
  tt.func @if_split_workaround_nondefault_partitions(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked3> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    // Single-buffered (disallow_acc_multi_buffer)
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 4, 5>} : (i32) -> (i32, tensor<64x128xi32, #blocked3>, i32)
      %3 = tt.splat %2#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (!tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.addptr %arg4, %2#1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>, tensor<64x128xi32, #blocked3>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<64x128x!tt.ptr<f16>, #blocked3>
      %7 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 5>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %9 = ttng.tc_gen5_mma %7, %8, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 4>} : i1
      // The three ifs of the split, in emission order. Exit-if: the
      // release's partition. Middle-if: content union — result-less
      // body partition, dead token slot dropped, outputs emptied.
      // Enter-if: the trailing acquire's partition, routing the token.
      // CHECK: scf.if
      // CHECK: nvws.semaphore.release {{.*}}ttg.partition = array<i32: 4>
      // CHECK: } {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>}
      // CHECK: scf.if
      // CHECK: } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = []}
      // CHECK: scf.if
      // CHECK: nvws.semaphore.acquire {{.*}}ttg.partition = array<i32: 4>
      // CHECK: } {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]}
      %12 = scf.if %10 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 4>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 4>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 4>, ttg.partition.outputs = [array<i32: 4>]}
      scf.yield {ttg.partition = array<i32: 0, 4, 5>} %11, %5, %12 : i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 4, 5>, ttg.partition.outputs = [array<i32: 4>, array<i32: 4>, array<i32: 4>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    // After loop: no drain; there is no post-loop TMEM access.
    tt.return
  }
}
