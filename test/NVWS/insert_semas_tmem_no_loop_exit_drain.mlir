// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_loop_carried_linear_chain_no_exit_drain
  tt.func @tmem_loop_carried_linear_chain_no_exit_drain(
      %lb: i32, %ub: i32, %step: i32,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %cst_f16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %acc, %acc_tok = ttng.tmem_alloc %cst_f32 {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK: [[BASE:%.*]] = ttng.tmem_alloc {buffer.id = 920 : i32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
    // CHECK-NEXT: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 5>
    // CHECK-NEXT: [[LOOP:%.*]]:{{[0-9]+}} = scf.for {{.*}} iter_args({{.*}}[[ITER:%[^ ,)]+]] = [[ENTRY]]
    %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0, %tok = %acc_tok) -> (i32, !ttg.async.token) : i32 {
      // CHECK: [[WRITE_VIEW:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {ttg.partition = array<i32: 5>}
      // CHECK: ttng.tmem_store {{.*}}, [[WRITE_VIEW]], {{.*}} {ttg.partition = array<i32: 5>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>}
      // CHECK-NEXT: [[READ:%.*]] = nvws.semaphore.acquire [[FULL]] {{.*}}ttg.partition = array<i32: 1>
      // CHECK: ttng.tc_gen5_mma
      // CHECK: nvws.semaphore.release [[EMPTY]], [[READ]] [#nvws.async_op<tc5mma>] {{.*}}ttg.partition = array<i32: 1>
      // CHECK: [[NEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 5>
      // CHECK-NEXT: scf.yield {{.*}} [[NEXT]]
      %frag = ttng.tmem_alloc %cst_f16 {buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
      %read = ttng.tc_gen5_mma %frag, %rhs, %acc[%tok], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %next_i = arith.addi %i, %c0 {ttg.partition = array<i32: 5>} : i32
      scf.yield {ttg.partition = array<i32: 1, 5>} %next_i, %read : i32, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 5>, ttg.partition.outputs = [array<i32: 5>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}

    // CHECK-NOT: nvws.semaphore.release [[FULL]], [[LOOP]]
    // CHECK-NOT: nvws.semaphore.acquire [[FULL]]
    // CHECK-NOT: nvws.semaphore.release [[EMPTY]]
    // CHECK: tt.return
    "use"(%loop#0) : (i32) -> ()
    tt.return
  }
}
