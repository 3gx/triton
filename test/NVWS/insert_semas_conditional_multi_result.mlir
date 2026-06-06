// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @conditional_multi_result_if_token
  tt.func @conditional_multi_result_if_token(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %false = arith.constant false
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token

    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[V8:%.*]]:3 = scf.for [[V9:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V5:%.*]] = %{{.*}}, [[V6:%.*]] = [[V4]], [[V7:%.*]] = %{{.*}}) -> (i1, !ttg.async.token, i32)  : i32 {
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V2]], [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
    // CHECK: nvws.semaphore.release [[V3]], [[V6]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // CHECK: [[V11:%.*]]:3 = scf.if %{{.*}} -> (i32, !ttg.async.token, i1) {
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        // CHECK: [[V12:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V13:%.*]] = nvws.semaphore.buffer [[V3]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: nvws.semaphore.release [[V2]], [[V12]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V9]], %{{.*}}, %{{.*}} : i32, !ttg.async.token, i1
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V7]], %{{.*}}, [[V5]] : i32, !ttg.async.token, i1
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0>]}
      // CHECK: [[V14:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 1>} [[V15]] : !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 1>} [[V6]] : !ttg.async.token
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V11]]#2, [[V14]], %{{.*}} : i1, !ttg.async.token, i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %epilogue#2, %epilogue#1, %next : i1, !ttg.async.token, i32
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @guarded_tokenless_if_deferred_initial_acquire
  tt.func @guarded_tokenless_if_deferred_initial_acquire(%lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>, %guard: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %true = arith.constant true
    %false = arith.constant false
    // CHECK: [[V16:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>

    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    scf.if %guard {
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V20:%.*]] = scf.for [[V21:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V19:%.*]] = [[V18]]) -> (!ttg.async.token)  : i32 {
      %loop = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[V22:%.*]] = nvws.semaphore.buffer [[V16]], [[V19]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: nvws.semaphore.release [[V17]], [[V19]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V24:%.*]] = nvws.semaphore.buffer [[V17]], [[V23]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: nvws.semaphore.release [[V16]], [[V23]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[V25:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V25]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %load_tok : !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
      scf.yield
    }
    tt.return
  }
}
