// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-allocas | FileCheck %s --implicit-check-not="ttng.tmem_alloc %"

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // A tokenful initializer outside the WS loop becomes a tokenless backing and
  // store.  Only the loop seed becomes poison: the body argument -> MMA token
  // -> yield recurrence must remain for Meta MemoryPlanner.
  // CHECK-LABEL: tt.func @tokenful_initializer_preserves_mma_recurrence
  tt.func @tokenful_initializer_preserves_mma_recurrence(
      %src: tensor<128x128xf32, #blocked>,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared_t, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // CHECK: [[BACKING:%.*]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 17 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK-NEXT: [[INIT_PRED:%.*]] = arith.constant true
    // CHECK-NEXT: {{^ *}}ttng.tmem_store %{{.*}}, [[BACKING]], [[INIT_PRED]] : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc, %init = ttng.tmem_alloc %src {buffer.copy = 2 : i32, buffer.id = 17 : i32, buffer.offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for {{.*}} iter_args([[DEP:%.*]] = [[POISON]]) -> (!ttg.async.token)
    %loop = scf.for %iv = %lb to %ub step %step iter_args(%dep = %init) -> (!ttg.async.token) : i32 {
      // CHECK: [[MMA:%.*]] = ttng.tc_gen5_mma %{{.*}}, %{{.*}}, [[BACKING]][[[DEP]]], %{{.*}}, %{{.*}} {ttg.partition = array<i32: 1>}
      %next = ttng.tc_gen5_mma %lhs, %rhs, %acc[%dep], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_t, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.yield {{.*}}[[MMA]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 1>} %next : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "consume_token"(%loop) : (!ttg.async.token) -> ()
    tt.return
  }

  // Same-owner immutable TMEM is normalized too.  The backing becomes mutable
  // and the explicit store owns the original schedule point.
  // CHECK-LABEL: tt.func @immutable_same_owner
  tt.func @immutable_same_owner(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[IMM_BACKING:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: scf.for
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[IMM_SRC:%.*]] = "producer"()
      %value = "producer"() {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : () -> tensor<128x128xf32, #blocked>
      // CHECK: [[IMM_PRED:%.*]] = arith.constant {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} true
      // CHECK-NEXT: {{^ *}}ttng.tmem_store [[IMM_SRC]], [[IMM_BACKING]], [[IMM_PRED]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %immutable = ttng.tmem_alloc %value {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
      // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[IMM_BACKING]][] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %loaded, %done = ttng.tmem_load %immutable[] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
      "consume"(%loaded) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 2>, ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }

  // The backing for an initializer in an early-return work block stays in that
  // top-level CFG block; it must not be moved to the function entry block.
  // CHECK-LABEL: tt.func @sourceful_initializer_in_non_entry_block
  tt.func @sourceful_initializer_in_non_entry_block(
      %empty: i1, %lb: i32, %ub: i32, %step: i32) {
    cf.cond_br %empty, ^bb1, ^bb2
  ^bb1:
    tt.return
  ^bb2:
    // CHECK: ^bb2:
    // CHECK: [[CFG_SRC:%.*]] = arith.constant dense<0.000000e+00>
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-NEXT: [[CFG_BACKING:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[CFG_POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK-NEXT: scf.for
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[CFG_PRED:%.*]] = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} true
      // CHECK-NEXT: {{^ *}}ttng.tmem_store [[CFG_SRC]], [[CFG_BACKING]], [[CFG_PRED]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc, %init = ttng.tmem_alloc %zero {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK-NEXT: %{{.*}}, %{{.*}} = ttng.tmem_load [[CFG_BACKING]][[[CFG_POISON]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %loaded, %done = ttng.tmem_load %acc[%init] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "consume"(%loaded) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}
