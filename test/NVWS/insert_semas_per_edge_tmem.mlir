// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // DAG: NVWS-SEMA-DAG buffer.id=300 memory=tmem
  // DAG: ACCESS-DAG
  // DAG: |- scf.for
  // DAG: |  |- W  m0
  // DAG: RAW-SYNC-DAG
  // DAG: |  |  |- r  S0
  // DAG: OPT-SYNC-DAG
  // DAG: |  |  |- r  S_full
  // DAG: |  |  |- a  S_empty
  // CHECK-LABEL: @tmem_single_producer_multi_consumer_fanout
  tt.func @tmem_single_producer_multi_consumer_fanout(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[BUF0:%.*]] = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
      // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BUF0]] true
      // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BUF0]] false
      %a, %ta = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK0]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW0]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %s0 = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[VIEW1:%.*]] = nvws.semaphore.buffer [[FULL]], [[TOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[VIEW1]]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: "use_p1"
      %v1, %t1 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK-NEXT: [[TOK2:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[VIEW2:%.*]] = nvws.semaphore.buffer [[FULL]], [[TOK2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[VIEW2]]{{.*}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: "use_p2"
      %v2, %t2 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK-NEXT: [[TOK3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK3]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW3]]{{.*}} {ttg.partition = array<i32: 0>}
      %s1 = ttng.tmem_store %cst, %a[%t2], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#alpha_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_qk_alpha_pacc_three_member_edges
  tt.func @tmem_qk_alpha_pacc_three_member_edges(
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %alpha_val = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #alpha_blocked>
    %true = arith.constant true
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[QK:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32}
      // CHECK-NEXT: [[ALPHA:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 64 : i32}
      // CHECK-NEXT: [[PACC:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32}
      // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[QK]], [[ALPHA]], [[PACC]] true
      // CHECK-NEXT: [[QK_READY:%.*]] = nvws.semaphore.create [[QK]], [[ALPHA]], [[PACC]] false
      // CHECK-NEXT: [[ALPHA_READY:%.*]] = nvws.semaphore.create [[QK]], [[ALPHA]], [[PACC]] false
      // CHECK-NEXT: [[PACC_READY:%.*]] = nvws.semaphore.create [[QK]], [[ALPHA]], [[PACC]] false
      %qk, %tq = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %alpha = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 5>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[QK_READY]], [[T0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %qk0 = ttng.tmem_store %cst, %qk[%tq], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[T1:%.*]] = nvws.semaphore.acquire [[QK_READY]] {ttg.partition = array<i32: 5>}
      // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 5>}
      %qkv, %qkt = ttng.tmem_load %qk[%qk0] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_qk"(%qkv) {ttg.partition = array<i32: 5>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 5>}
      // CHECK-NEXT: nvws.semaphore.release [[ALPHA_READY]], [[T1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>}
      ttng.tmem_store %alpha_val, %alpha, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #alpha_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[T2:%.*]] = nvws.semaphore.acquire [[ALPHA_READY]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #alpha_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x1xf32, #alpha_blocked>) -> ()

      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 5>}
      // CHECK-NEXT: nvws.semaphore.release [[PACC_READY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 5>}
      %pacc = ttng.tmem_alloc %cst16 {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

      // CHECK-NEXT: [[T3:%.*]] = nvws.semaphore.acquire [[PACC_READY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[PVIEW:%.*]]:3 = nvws.semaphore.buffer [[PACC_READY]], [[T3]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma [[PVIEW]]#2
      %mma = ttng.tc_gen5_mma %pacc, %rhs, %qk[%qkt], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[QK2:%.*]]:3 = nvws.semaphore.buffer [[EMPTY]], [[T4]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[QK2]]#0
      %qk1 = ttng.tmem_store %cst, %qk[%mma], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 5>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 5>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 5>, ttg.partition.outputs = [array<i32: 0, 1, 5>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_full_aliased_members_different_partitions
  tt.func @tmem_full_aliased_members_different_partitions(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK-NOT: nvws.semaphore.create
      %a = ttng.tmem_alloc %cst0 {buffer.id = 302 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %av, %at = ttng.tmem_load %a[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_a"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %b = ttng.tmem_alloc %cst1 {buffer.id = 302 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %bv, %bt = ttng.tmem_load %b[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_b"(%bv) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
