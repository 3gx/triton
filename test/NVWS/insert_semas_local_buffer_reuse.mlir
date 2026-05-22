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
    // CHECK: [[ABUF:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[BBUF:%.*]] = ttg.local_alloc {buffer.id = 400 : i32, buffer.offset = 64 : i32}
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[BBUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[BBUF]] false
    // CHECK-NEXT: scf.for
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[AVIEW:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[ATOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{.*}}, [[AVIEW]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}} = ttg.local_load [[AVIEW]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: "use"
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ATOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst_0 {buffer.id = 400 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: [[BTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BVIEW:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[BTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{.*}}, [[BVIEW]]#1 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}} = ttg.local_load [[BVIEW]]#1 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use"
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]], [[BTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield
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
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[ABUF:%.*]] = ttg.local_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32}
      // CHECK-NEXT: [[BBUF:%.*]] = ttg.local_alloc {buffer.id = 401 : i32, buffer.offset = 64 : i32}
      // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[BBUF]] true
      // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[BBUF]] false
      %a = ttg.local_alloc %cst0 {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW0:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[TOK0]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{.*}}, [[VIEW0]]#0 {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[VIEW1:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[TOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}} = ttg.local_load [[VIEW1]]#0 {ttg.partition = array<i32: 1>}
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst1 {buffer.id = 401 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK-NEXT: "use"
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[TOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[VIEW2:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[TOK2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store {{.*}}, [[VIEW2]]#1 {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[TOK3:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW3:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[TOK3]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}} = ttg.local_load [[VIEW3]]#1 {ttg.partition = array<i32: 0>}
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
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
      // CHECK-NOT: nvws.semaphore.create
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
