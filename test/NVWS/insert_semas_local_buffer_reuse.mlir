// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Local-memory mirrors of TMEM buffer-reuse tests. These exercise the
// same v4 §Physical Conflict Key behaviors (buffer.id grouping +
// buffer.offset overlap classification) on ttg.local_alloc instead of
// ttng.tmem_alloc. Until the make-group path is unified the local
// allocs are treated as independent groups; once unified they will
// share a logical buffer group and the dump / emit shape will match
// the TMEM mirrors.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Mirror of @sourceful_tokenless_alias from tmem-buffer-reuse-semas.mlir.
  // Two local_allocs share buffer.id=400 and overlap at offsets 0 and 64
  // (extent 128 each → physical-conflict-key match). Two partitions
  // alternate: {1} writes/reads member 0, then {0} writes/reads member 1.
  // CHECK-LABEL: @local_sourceful_aliased_buffers
  tt.func @local_sourceful_aliased_buffers(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]], [[V2]] true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V2]] false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V8:%.*]]:2 = scf.for [[V9:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V6:%.*]] = %{{.*}}, [[V7:%.*]] = [[V5]]) -> (i32, !ttg.async.token)  : i32 {
    // CHECK: [[V10:%.*]]:2 = nvws.semaphore.buffer [[V3]], [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[V4]], [[V7]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V12:%.*]]:2 = nvws.semaphore.buffer [[V4]], [[V11]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %b = ttg.local_alloc %cst_0 {buffer.id = 400 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: nvws.semaphore.release [[V3]], [[V11]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{.*}}, [[V13]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // Mirror of @n_owner_alias_sequence from tmem-buffer-reuse-semas.mlir.
  // Two local_allocs share buffer.id=401 and overlap at offsets 0 and 64.
  // Three partitions form a linear chain: {0} writes m0, {1} reads m0,
  // {2} writes m1, {0} reads m1 — alternating EMPTY/FULL semaphore shape.
  // CHECK-LABEL: @local_n_owner_aliased_buffers
  tt.func @local_n_owner_aliased_buffers(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK: [[V14:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}} true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V15:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}} false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create %{{.*}}, %{{.*}} false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V18:%.*]]:2 = nvws.semaphore.buffer [[V14]], [[V17]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[V15]], [[V17]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V15]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V20:%.*]]:2 = nvws.semaphore.buffer [[V15]], [[V19]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem, 1x128x128>
      %a = ttg.local_alloc %cst0 {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
	      // CHECK: nvws.semaphore.release [[V16]], [[V19]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
	      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
	      // CHECK: [[V22:%.*]]:2 = nvws.semaphore.buffer [[V16]], [[V21]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      // CHECK: nvws.semaphore.release [[V14]], [[V21]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
	      // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V14]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
	      // CHECK: [[V24:%.*]]:2 = nvws.semaphore.buffer [[V14]], [[V23]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem>
	      %b = ttg.local_alloc %cst1 {buffer.id = 401 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
	      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
	      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[V14]], [[V23]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // Mirror of @tmem_non_overlapping_members_no_semaphore from
  // insert_semas_per_edge_tmem.mlir. Two local_allocs share
  // buffer.id=402 but live at non-overlapping offsets (0 and 256, both
  // extent 128). They are physically distinct (different resourceKey),
  // each touched by only one owner, so no semaphores are needed.
  // CHECK-LABEL: @local_non_overlapping_aliased_buffers
  tt.func @local_non_overlapping_aliased_buffers(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #blocked>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst0 {buffer.id = 402 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %av = ttg.local_load %a {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_a"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst1 {buffer.id = 402 : i32, buffer.offset = 256 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %bv = ttg.local_load %b {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use_b"(%bv) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 2 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
