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

    // CHECK: [[ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ACC]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ACC]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 1>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: scf.for {{.*}} iter_args({{.*}}, [[LOOP_TOK:%.*]] = [[ATOK]]
    %loop:3 = scf.for %iv = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%use_acc = %false, %tok = %acc_tok, %carry = %c0_i32) -> (i1, !ttg.async.token, i32) : i32 {
      // CHECK: [[ABUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[LOOP_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[MMA:%.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, [[ABUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %use_acc, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[COND:%.*]] = arith.cmpi
      // CHECK-NEXT: scf.if [[COND]] {
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[LOOP_TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: }
      // CHECK-NEXT: [[POISON:%.*]] = ub.poison : !ttg.async.token
      // CHECK-NEXT: [[BODY_IF:%.*]]:3 = scf.if [[COND]] -> (i32, !ttg.async.token, i1) {
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[CBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[CBUF]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: scf.yield {{.*}}, [[POISON]], {{.*}} : i32, !ttg.async.token, i1
      // CHECK: } else {
      // CHECK-NEXT: scf.yield {{.*}}, [[POISON]], {{.*}} : i32, !ttg.async.token, i1
      // CHECK-NEXT: }
      // CHECK-NEXT: [[ENTER_IF:%.*]] = scf.if [[COND]] -> (!ttg.async.token) {
      // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}}[[PTOK]] : !ttg.async.token
      // CHECK-NEXT: } else {
      // CHECK-NEXT: scf.yield {{.*}}[[LOOP_TOK]] : !ttg.async.token
      // CHECK-NEXT: }
      %epilogue:3 = scf.if %cond -> (i32, !ttg.async.token, i1) {
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %iv, %load_tok, %true : i32, !ttg.async.token, i1
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %carry, %mma, %use_acc : i32, !ttg.async.token, i1
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 0>]}
      %next = arith.addi %epilogue#0, %c1_i32 {ttg.partition = array<i32: 0, 1>} : i32
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

    // CHECK: [[GACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<
    // CHECK-NEXT: [[GEMPTY:%.*]] = nvws.semaphore.create [[GACC]] true
    // CHECK-NEXT: [[GFULL:%.*]] = nvws.semaphore.create [[GACC]] false
    // CHECK-NEXT: scf.if {{.*}} {
    // CHECK-NEXT: [[GATOK:%.*]] = nvws.semaphore.acquire [[GEMPTY]] {{.*}}ttg.partition = array<i32: 1>
    %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    scf.if %guard {
      // CHECK: scf.for {{.*}} iter_args([[GLOOP_TOK:%.*]] = [[GATOK]])
      %loop = scf.for %iv = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%tok = %acc_tok) -> (!ttg.async.token) : i32 {
        // CHECK: nvws.semaphore.buffer [[GEMPTY]], [[GLOOP_TOK]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[GMMA:%.*]] = ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %value, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%value) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %load_tok : !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
      scf.yield
    }
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize{{.*}}ttg.warp_specialize.tag = 1 : i32}
    // CHECK-NEXT: }
    // CHECK-NEXT: tt.return
    tt.return
  }
}
