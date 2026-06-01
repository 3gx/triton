// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @live_tag_source_after_prior_loop_threading(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %scratch = ttg.local_alloc {buffer.id = 910 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>

    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      %acc, %tok = ttng.tmem_alloc %cst {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %inner_tmem = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load_lhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load_rhs"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_tmem"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      %out, %out_tok = ttng.tmem_load %acc[%inner_tmem] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_tmem_post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      %c0_p2 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      %write = ttg.memdesc_index %scratch[%c0_p2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %payload = "local_payload"() {ttg.partition = array<i32: 2>} : () -> tensor<128x128xf16, #blocked>
      ttg.local_store %payload, %write {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

      %c0_p1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %read = ttg.memdesc_index %scratch[%c0_p1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      scf.for %iv2 = %lb to %ub step %step : i32 {
        %loaded = ttg.local_load %read {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #blocked>
        "use_local"(%loaded) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      } {ttg.partition = array<i32: 1>}

      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
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
