// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @branch_local_initial_acquire_stays_with_create
  tt.func @branch_local_initial_acquire_stays_with_create(
      %guard: i1,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %false = arith.constant false
    %true = arith.constant true

    // The semaphore storage is branch-local here. The initial EMPTY acquire
    // must stay in the same branch after nvws.semaphore.create; hoisting it
    // above the scf.if violates SSA dominance.
    // CHECK: scf.if {{.*}} {
    // CHECK-NEXT: } else {
    // CHECK-NEXT: [[ACC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<{{[0-9]+}}x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ACC]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ACC]] false
    // CHECK-NEXT: [[INIT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 1>
    // CHECK-NEXT: scf.for {{.*}} iter_args({{.*}} = [[INIT]])
    // CHECK: nvws.semaphore.buffer [[EMPTY]]
    // CHECK: ttng.tc_gen5_mma
    // CHECK: nvws.semaphore.release [[FULL]]
    // CHECK: nvws.semaphore.acquire [[FULL]]
    scf.if %guard {
      scf.yield
    } else {
      %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %loop = scf.for %iv = %c0 to %c4 step %c1 iter_args(%carry = %tok) -> (!ttg.async.token) : i32 {
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %loaded, %load_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%loaded) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %load_tok : !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
      "token_user"(%loop) : (!ttg.async.token) -> ()
      scf.yield
    }
    tt.return
  }
}
