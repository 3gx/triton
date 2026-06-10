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
  // DAG: |  |  |- r  S0
  // DAG: |  |  |- r  S1
  // DAG: |  |  |- a  S2  acquire
  // CHECK-LABEL: @tmem_single_producer_multi_consumer_fanout
  tt.func @tmem_single_producer_multi_consumer_fanout(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V6:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V6]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V7:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V3]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V8]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V2]], [[V7]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %a, %ta = ttng.tmem_alloc {buffer.id = 300 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %s0 = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %v1, %t1 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
    // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V4]], [[V9]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V10]][] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      %v2, %t2 = ttng.tmem_load %a[%s0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_p2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V12:%.*]] = nvws.semaphore.buffer [[V2]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V12]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>

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
  // DAG: NVWS-SEMA-DAG buffer.id=301 memory=tmem
  // DAG: members: m0(offset=0,extent=128,resourceKey=0) m1(offset=64,extent=1,resourceKey=0) m2(offset=0,extent=64,resourceKey=0)
  // DAG: OWNERSHIP-DAG buffer.id=301 resourceKey=0 members: m0 m1 m2
  // CHECK-LABEL: @tmem_qk_alpha_pacc_three_member_edges
  tt.func @tmem_qk_alpha_pacc_three_member_edges(
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %alpha_val = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #alpha_blocked>
    %true = arith.constant true
    // CHECK: [[V13:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V14:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V15:%.*]] = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V13]], [[V14]], [[V15]] true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V13]], [[V14]], [[V15]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V18:%.*]] = nvws.semaphore.create [[V13]], [[V14]], [[V15]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V19:%.*]] = nvws.semaphore.create [[V13]], [[V14]], [[V15]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %qk, %tq = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V20:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V21:%.*]]:3 = nvws.semaphore.buffer [[V16]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V21]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V17]], [[V20]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V22:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V23:%.*]]:3 = nvws.semaphore.buffer [[V17]], [[V22]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V23]]#0[] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %alpha = ttng.tmem_alloc {buffer.id = 301 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 5>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      %qk0 = ttng.tmem_store %cst, %qk[%tq], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %qkv, %qkt = ttng.tmem_load %qk[%qk0] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_qk"(%qkv) {ttg.partition = array<i32: 5>} : (tensor<128x128xf32, #blocked>) -> ()
    // CHECK: nvws.semaphore.release [[V18]], [[V22]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V24:%.*]] = nvws.semaphore.acquire [[V18]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V25:%.*]]:3 = nvws.semaphore.buffer [[V18]], [[V24]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V25]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #blocked1>
    // CHECK: nvws.semaphore.release [[V19]], [[V24]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V26:%.*]] = nvws.semaphore.acquire [[V19]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V27:%.*]]:3 = nvws.semaphore.buffer [[V19]], [[V26]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V27]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: nvws.semaphore.release [[V16]], [[V26]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V28:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V29:%.*]]:3 = nvws.semaphore.buffer [[V16]], [[V28]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma [[V29]]#2, %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      ttng.tmem_store %alpha_val, %alpha, %true {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #alpha_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>

      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #alpha_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x1xf32, #alpha_blocked>) -> ()

      %pacc = ttng.tmem_alloc %cst16 {buffer.id = 301 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

      %mma = ttng.tc_gen5_mma %pacc, %rhs, %qk[%qkt], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_nested_linear_chain_no_outer_drain
  tt.func @tmem_nested_linear_chain_no_outer_drain(
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %true = arith.constant true

    // CHECK: [[V30:%.*]], [[V31:%.*]] = ttng.tmem_alloc {buffer.id = 704 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc, %atok = ttng.tmem_alloc {buffer.id = 704 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[V32:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 705 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V33:%.*]] = nvws.semaphore.create [[V32]] true : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V34:%.*]] = nvws.semaphore.create [[V32]] false : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V35:%.*]] = nvws.semaphore.acquire [[V33]] {ttg.partition = array<i32: 5>, ttg.warp_specialize.tag = 7 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V39:%.*]]:3 = scf.for [[V40:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V36:%.*]] = %{{.*}}, [[V37:%.*]] = [[V31]], [[V38:%.*]] = [[V35]]) -> (i32, !ttg.async.token, !ttg.async.token)  : i32 {
    // CHECK: [[V43:%.*]]:2 = scf.for [[V44:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V41:%.*]] = [[V37]], [[V42:%.*]] = [[V38]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
    // CHECK: [[V45:%.*]] = nvws.semaphore.buffer [[V33]], [[V42]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V45]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: nvws.semaphore.release [[V34]], [[V42]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V46:%.*]] = nvws.semaphore.acquire [[V34]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V47:%.*]] = nvws.semaphore.buffer [[V34]], [[V46]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma [[V47]], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}[%{{[-A-Za-z0-9_.$#]+}}], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: nvws.semaphore.release [[V33]], [[V46]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V48:%.*]] = nvws.semaphore.acquire [[V33]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: scf.yield {ttg.partition = array<i32: 1, 5>} %{{.*}}, [[V48]] : !ttg.async.token, !ttg.async.token
    %outer:2 = scf.for %iv0 = %lb to %ub step %step iter_args(%i = %c0, %outer_tok = %atok) -> (i32, !ttg.async.token) : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%inner_tok = %outer_tok) -> (!ttg.async.token) : i32 {
        %src = ttng.tmem_alloc %cst {buffer.copy = 1 : i32, buffer.id = 705 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

        %mma = ttng.tc_gen5_mma %src, %rhs, %acc[%inner_tok], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 5>} %mma : !ttg.async.token
      } {ttg.partition = array<i32: 1, 5>, ttg.partition.outputs = [array<i32: 1>]}
      %next = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 5>} : i32
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 5>} %{{.*}}, [[V43]]#0, [[V43]]#1 : i32, !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 5>} %next, %inner : i32, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 5>, ttg.partition.outputs = [array<i32: 0, 1, 5>, array<i32: 1>], ttg.warp_specialize.tag = 7 : i32}
    "use_i32"(%outer#0) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // DAG: NVWS-SEMA-DAG buffer.id=706 memory=tmem
  // DAG: RAW-SYNC-DAG
  // DAG: |  |- R  m0  ttng.tmem_load  {0}
  // DAG: |  |- R  m0  ttng.tmem_load  {0}
  // DAG: |  |- r  S1  release  {0} -> {5}
  // DAG: OPT-SYNC-DAG
  // DAG: |  |- R  m0  ttng.tmem_load  {0}
  // DAG: |  |- R  m0  ttng.tmem_load  {0}
  // DAG: |  |- r  {{S[^ ]+}}  release  {0} -> {5}
  // CHECK-LABEL: @tmem_same_owner_reads_close_at_yield
  tt.func @tmem_same_owner_reads_close_at_yield(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V49:%.*]] = ttng.tmem_alloc {buffer.id = 706 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V50:%.*]] = nvws.semaphore.create [[V49]] true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V51:%.*]] = nvws.semaphore.create [[V49]] false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V52:%.*]] = nvws.semaphore.acquire [[V50]] {ttg.partition = array<i32: 5>, ttg.warp_specialize.tag = 8 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V55:%.*]]:2 = scf.for [[V56:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V53:%.*]] = %{{.*}}, [[V54:%.*]] = [[V52]]) -> (i32, !ttg.async.token)  : i32 {
    // CHECK: [[V57:%.*]] = nvws.semaphore.buffer [[V50]], [[V54]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V57]][], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: nvws.semaphore.release [[V51]], [[V54]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V58:%.*]] = nvws.semaphore.acquire [[V51]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V59:%.*]] = nvws.semaphore.buffer [[V51]], [[V58]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V59]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a, %ta = ttng.tmem_alloc {buffer.id = 706 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %s = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %v0, %t0 = ttng.tmem_load %a[%s] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_first"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[V50]], [[V58]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %v1, %t1 = ttng.tmem_load %a[%s] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_second"(%v1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 5>} : i32
      // CHECK: [[V60:%.*]] = nvws.semaphore.acquire [[V50]] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 5>} %{{.*}}, [[V60]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 5>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 5>, ttg.partition.outputs = [array<i32: 0, 5>], ttg.warp_specialize.tag = 8 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
