// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @live_tag_source_after_prior_loop_threading
  tt.func @live_tag_source_after_prior_loop_threading(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %scratch = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = ttng.tmem_alloc {buffer.id = 900 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V4]] true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V4]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V4]] false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V13:%.*]]:3 = scf.for [[V14:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V10:%.*]] = %{{.*}}, [[V11:%.*]] = [[V8]], [[V12:%.*]] = [[V9]]) -> (i32, !ttg.async.token, !ttg.async.token)  : i32 {
    // CHECK: [[V15:%.*]] = nvws.semaphore.buffer [[V5]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: nvws.semaphore.release [[V6]], [[V11]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token

    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      %acc, %tok = ttng.tmem_alloc %cst {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[V18:%.*]] = scf.for [[V19:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V17:%.*]] = [[V16]]) -> (!ttg.async.token)  : i32 {
      %inner_tmem = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load_lhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load_rhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[V20:%.*]] = nvws.semaphore.buffer [[V6]], [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V7]], [[V17]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[V22:%.*]] = nvws.semaphore.buffer [[V7]], [[V21]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[V6]], [[V21]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_tmem"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V23]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: nvws.semaphore.release [[V5]], [[V18]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V24:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V25:%.*]] = nvws.semaphore.buffer [[V5]], [[V24]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V5]], [[V24]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      %out, %out_tok = ttng.tmem_load %acc[%inner_tmem] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_tmem_post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %c0_p2 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      %write = ttg.memdesc_index %scratch[%c0_p2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %payload = "local_payload"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
      // CHECK: [[V26:%.*]] = nvws.semaphore.buffer [[V2]], [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %payload, %write {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[V3]], [[V12]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V27:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V28:%.*]] = nvws.semaphore.buffer [[V3]], [[V27]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem>

      %c0_p1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %read = ttg.memdesc_index %scratch[%c0_p1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      scf.for %iv2 = %lb to %ub step %step : i32 {
        %loaded = ttg.local_load %read {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #blocked>
        "use_local"(%loaded) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      } {ttg.partition = array<i32: 1>}

      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: nvws.semaphore.release [[V2]], [[V27]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V29:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} %{{.*}}, [[V24]], [[V29]] : i32, !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// DAG-LABEL: NVWS-SEMA-DAG buffer.id=910 memory=local
// DAG: ACCESS-DAG
// DAG: |  |- W  m0  ttg.local_store {2}
// DAG: |  |  |- R  m0  ttg.local_load {1}
// DAG: OWNERSHIP-DAG buffer.id=910 resourceKey=0 members: m0
// DAG: |  |- scf.for (WS, tag=0) {2}
// DAG: |  |  |- ENTER {2}
// DAG: |  |  |- W  m0  ttg.local_store  use {2}
// DAG: |  |  |- scf.for {1}
// DAG: |  |  |  |- ENTER {1}
// DAG: |  |  |  |- R  m0  ttg.local_load  use {1}
// DAG: |  |  |  |- YIELD {1}
// DAG: |  |  |- YIELD {2}
