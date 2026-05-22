// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @raw_edge_token_carried_if
  tt.func @raw_edge_token_carried_if(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // Different physical resources in the same logical buffer group force the
      // raw-edge scheduler while the first resource exercises an if-carried
      // generated semaphore token.
      // CHECK: [[A:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32}
      // CHECK-NEXT: [[B:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32}
      // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[A]], [[B]] true
      %a, %ta = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %b, %tb = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      %a0 = ttng.tmem_store %cst0, %a[%ta], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %cond = arith.cmpi eq, %iv, %c0 {ttg.partition = array<i32: 0, 1>} : i32

      // CHECK: [[IF_TOK:%.*]] = scf.if {{.*}} -> (!ttg.async.token) {
      // CHECK: nvws.semaphore.release [[EMPTY]],
      // CHECK-SAME: {ttg.partition = array<i32: 1>}
      // CHECK: [[DONE:%.*]] = nvws.semaphore.acquire [[EMPTY]]
      // CHECK-SAME: {ttg.partition = array<i32: 0>}
      // CHECK: scf.yield {{.*}}[[DONE]] : !ttg.async.token
      // CHECK: } else {
      // CHECK: scf.yield {{.*}} : !ttg.async.token
      // CHECK: }
      %if_tok = scf.if %cond -> (!ttg.async.token) {
        %av, %at = ttng.tmem_load %a[%a0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use_a"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %at : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %a0 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: nvws.semaphore.buffer [[EMPTY]], [[IF_TOK]]
      // CHECK-NEXT: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 0>}
      %a1 = ttng.tmem_store %cst1, %a[%if_tok], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %b0 = ttng.tmem_store %cst0, %b[%tb], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %bv, %bt = ttng.tmem_load %b[%b0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_b"(%bv) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
