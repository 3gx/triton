// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @root_entry_accumulator_adopts_without_semaphore_handoff
  tt.func @root_entry_accumulator_adopts_without_semaphore_handoff(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true

    // Root initializes the accumulator before entering the WS loop. This state
    // is adopted by the loop-carried partition-1 token; it must not emit a
    // root->partition semaphore release/acquire pair before the loop.
    // CHECK: [[ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ACC]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ACC]] false
    // CHECK-NEXT: [[INIT:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK: ttng.tmem_store
    // CHECK-NOT: nvws.semaphore.release
    // CHECK-NOT: nvws.semaphore.acquire
    // CHECK: scf.for {{.*}} iter_args({{.*}} = [[INIT]])
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      // CHECK: [[P1BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{%.*}}, {{%.*}} = ttng.tmem_load [[P1BUF]]
      %loaded, %load_tok = ttng.tmem_load %acc[%carry] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

      // CHECK: ttng.tmem_store
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{%.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{%.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>}
      %store = ttng.tmem_store %loaded, %acc[%load_tok], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK: ttng.tc_gen5_mma
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: {{%.*}} = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%store], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1, 2>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}

    // CHECK: nvws.semaphore.release [[FULL]], {{%.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: {{%.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    %out, %out_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
