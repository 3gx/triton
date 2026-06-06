// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @sourceful_tokenless_alias
  tt.func @sourceful_tokenless_alias(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc {buffer.id = 7 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>
    // CHECK-NEXT: scf.for {{.*}} iter_args({{.*}}, [[AITER:%[^ ]+]] = [[ATOK]]) -> (i32, !ttg.async.token)
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttng.tmem_alloc %cst {buffer.id = 7 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[AVIEW:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[AITER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[TRUE1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} true
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[AVIEW]]#0, [[TRUE1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[AVIEW]]#0[] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: "use"
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[AITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %b = ttng.tmem_alloc %cst_0 {buffer.id = 7 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[BTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BVIEW:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[BTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[TRUE0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} true
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BVIEW]]#1, [[TRUE0]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[BVIEW]]#1[] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use"
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]], [[BTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ANEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}} [[ANEXT]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @same_owner_alias_no_semaphore
  tt.func @same_owner_alias_no_semaphore(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-NOT: nvws.semaphore
    // CHECK: scf.for
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttng.tmem_alloc %cst {buffer.id = 8 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %b = ttng.tmem_alloc %cst_0 {buffer.id = 8 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 1>} : i32
      scf.yield {ttg.partition = array<i32: 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 1 : i32}
    // CHECK-NOT: nvws.semaphore
    // CHECK: tt.return
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @loop_token_slot_alias
  tt.func @loop_token_slot_alias(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 2 : i32}
    %a, %ta = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %b, %tb = ttng.tmem_alloc {buffer.id = 9 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args([[ITER:%.*]] = [[ATOK]], {{.*}} = [[POISON]])
    %r:2 = scf.for %iv = %lb to %ub step %step iter_args(%t0 = %ta, %t1 = %tb) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK-NEXT: [[PROD_VIEW:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[PROD_VIEW]]#0[], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[CONS_VIEW:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[CONS_VIEW]]#1[] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use"
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[NEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT]], [[POISON]] : !ttg.async.token, !ttg.async.token
      %s = ttng.tmem_store %cst, %a[%t0], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %v, %l = ttng.tmem_load %b[%t1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%v) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %s, %l : !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>], ttg.warp_specialize.tag = 2 : i32}
    // CHECK-NEXT: } {tt.warp_specialize
    // CHECK-NEXT: tt.return
    tt.return
  }

  // CHECK-LABEL: @if_branch_alias
  tt.func @if_branch_alias(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc {buffer.id = 10 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[MERGE_FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[ELSE_FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[POST_FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
    // CHECK-NEXT: [[ENTRY:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 3 : i32}
    // CHECK-NEXT: [[POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK-NEXT: scf.for {{.*}} iter_args({{.*}} = {{.*}}, [[ITER:%.*]] = [[ENTRY]])
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a, %ta = ttng.tmem_alloc {buffer.id = 10 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %b, %tb = ttng.tmem_alloc {buffer.id = 10 : i32, ttg.partition = array<i32: 0>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK-NEXT: [[IF:%.*]]:2 = scf.if {{.*}} -> (!ttg.async.token, !ttg.async.token) {
      %if:2 = scf.if %cond -> (!ttg.async.token, !ttg.async.token) {
        // CHECK-NEXT: [[TBUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: ttng.tmem_store {{.*}}, [[TBUF]]#0[], {{.*}} {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[TFULL:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[TLOAD:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[TFULL]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[TLOAD]]#1[] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: "use"
        // CHECK-NEXT: nvws.semaphore.release [[MERGE_FULL]], [[TFULL]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[TMRG:%.*]] = nvws.semaphore.acquire [[MERGE_FULL]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[TMRG]], [[POISON]] : !ttg.async.token, !ttg.async.token
      %s0 = ttng.tmem_store %cst, %a[%ta], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %v0, %l0 = ttng.tmem_load %b[%tb] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %s0, %l0 : !ttg.async.token, !ttg.async.token
      } else {
        // CHECK-NEXT: } else {
        // CHECK-NEXT: [[EBUF:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[EBUF]]#0[] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: "use"
        // CHECK-NEXT: nvws.semaphore.release [[ELSE_FULL]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[EFULL:%.*]] = nvws.semaphore.acquire [[ELSE_FULL]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[ESTORE:%.*]]:2 = nvws.semaphore.buffer [[ELSE_FULL]], [[EFULL]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: ttng.tmem_store {{.*}}, [[ESTORE]]#1[], {{.*}} {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: nvws.semaphore.release [[MERGE_FULL]], [[EFULL]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: [[EMRG:%.*]] = nvws.semaphore.acquire [[MERGE_FULL]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[EMRG]], [[POISON]] : !ttg.async.token, !ttg.async.token
        %v1, %l1 = ttng.tmem_load %a[%ta] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "use"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
        %s1 = ttng.tmem_store %cst, %b[%tb], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 0, 1>} %l1, %s1 : !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0>]}
      // CHECK: [[POSTA_VIEW:%.*]]:2 = nvws.semaphore.buffer [[MERGE_FULL]], [[IF]]#0 {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[POSTA_VIEW]]#0[], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[POST_FULL]], [[IF]]#0 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[POSTB:%.*]] = nvws.semaphore.acquire [[POST_FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[POSTB_VIEW:%.*]]:2 = nvws.semaphore.buffer [[POST_FULL]], [[POSTB]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[POSTB_VIEW]]#1[] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use"
      %s2 = ttng.tmem_store %cst, %a[%if#0], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %v2, %l2 = ttng.tmem_load %b[%if#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%v2) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]], [[POSTB]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[POSTNEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.yield {{.*}} [[POSTNEXT]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 3 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // CHECK-LABEL: @accumulator_and_operand_alias_same_partition
  tt.func @accumulator_and_operand_alias_same_partition(
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<128x128xf16, #linear>
    %true = arith.constant true
    // CHECK: [[ACC0:%.*]] = ttng.tmem_alloc {buffer.id = 11 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[ALIAS_SUB:%.*]] = ttng.tmem_subslice [[ACC0]] {N = 0 : i32}
    // CHECK-NEXT: [[ALIAS:%.*]] = ttg.memdesc_reinterpret [[ALIAS_SUB]] : {{.*}} -> !ttg.memdesc<1x128x128xf16
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ACC0]], [[ALIAS]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ACC0]], [[ALIAS]] false
    // CHECK-NEXT: [[ACC1:%.*]], [[ACC1_TOK:%.*]] = ttng.tmem_alloc {buffer.id = 12 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 4 : i32}
    %acc0, %tok0 = ttng.tmem_alloc {buffer.id = 11 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc1, %tok1 = ttng.tmem_alloc {buffer.id = 12 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args([[TMEM_TOK:%.*]] = [[ATOK]], [[ACC1_ITER:%.*]] = [[ACC1_TOK]])
    %r:2 = scf.for %iv = %lb to %ub step %step iter_args(%t0 = %tok0, %t1 = %tok1) -> (!ttg.async.token, !ttg.async.token) : i32 {
      // CHECK-NEXT: [[ACC_VIEW:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[TMEM_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[MMA0:%.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, [[ACC_VIEW]]#0[], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TMEM_TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[SRC_TOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[SRC_VIEW:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[SRC_TOK]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[TRUE2:%.*]] = arith.constant {ttg.partition = array<i32: 2>} true
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[SRC_VIEW]]#1, [[TRUE2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[SRC_TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[READ_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[READ_VIEW:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[READ_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[MMA1:%.*]] = ttng.tc_gen5_mma [[READ_VIEW]]#1, {{.*}}, [[ACC1]]{{\[}}[[ACC1_ITER]]{{\]}}, {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[READ_VIEW]]#0[] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: "use"
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 1, 2>} [[READ_TOK]], [[MMA1]] : !ttg.async.token, !ttg.async.token
      %mma0 = ttng.tc_gen5_mma %lhs, %rhs, %acc0[%t0], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %alias = ttng.tmem_alloc %cst_1 {buffer.id = 11 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
      %mma1 = ttng.tc_gen5_mma %alias, %rhs, %acc1[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %v, %load = ttng.tmem_load %acc0[%mma0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%v) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 1, 2>} %load, %mma1 : !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], ttg.warp_specialize.tag = 4 : i32}
    // CHECK-NOT: nvws.semaphore.release [[FULL]]
    // CHECK-NOT: nvws.semaphore.acquire [[FULL]]
    // CHECK-NOT: nvws.semaphore.release [[EMPTY]]
    // CHECK: tt.return
    tt.return
  }

  // CHECK-LABEL: @singleton_staged_alloc_preserves_buffer_attrs
  tt.func @singleton_staged_alloc_preserves_buffer_attrs(
      %lhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<128x128xf16, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // CHECK: [[BUF:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BUF]] false
    // CHECK-NEXT: [[INIT:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    %acc, %tok = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32, buffer.offset = 0 : i32} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[VIEW:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INIT]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW]]
    // CHECK-NOT: ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 13 : i32, buffer.offset = 0 : i32}
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[LOOP:%.*]] = scf.for {{.*}} iter_args([[ITER:%.*]] = [[INIT]])
    %r = scf.for %iv = %lb to %ub step %step iter_args(%t = %init) -> (!ttg.async.token) : i32 {
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[MMA_TOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ACC_VIEW:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[MMA_TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[MMA:%.*]] = ttng.tc_gen5_mma {{.*}}, {{.*}}, [[ACC_VIEW]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[MMA_TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOAD_TOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[LOAD_VIEW:%.*]] = nvws.semaphore.buffer [[FULL]], [[LOAD_TOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[LOAD_VIEW]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "use"
      // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[LOAD_TOK]] : !ttg.async.token
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%t], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %v, %load = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%v) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %load : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 5 : i32}
    // CHECK: nvws.semaphore.release [[EMPTY]], [[LOOP]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 5 : i32}
    // CHECK-NEXT: [[EXIT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 5 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[EXIT]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 5 : i32}
    tt.return
  }

  // CHECK-LABEL: @n_owner_alias_sequence
  tt.func @n_owner_alias_sequence(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc {buffer.id = 14 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
      // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] true
      // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
      // CHECK-NEXT: [[NEXT_FULL:%.*]] = nvws.semaphore.create [[ABUF]], [[ABUF]] false
      %a = ttng.tmem_alloc %cst0 {buffer.id = 14 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW0:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[TOK0]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}} = arith.constant {ttg.partition = array<i32: 0>} true
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW0]]#0, {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[VIEW1:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[TOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[VIEW1]]#0[] {ttg.partition = array<i32: 1>}
      %va, %ta = ttng.tmem_load %a[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      %b = ttng.tmem_alloc %cst1 {buffer.id = 14 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: "use"
      // CHECK-NEXT: nvws.semaphore.release [[NEXT_FULL]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[TOK2:%.*]] = nvws.semaphore.acquire [[NEXT_FULL]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[VIEW2:%.*]]:2 = nvws.semaphore.buffer [[NEXT_FULL]], [[TOK2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: {{.*}} = arith.constant {ttg.partition = array<i32: 2>} true
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW2]]#1, {{.*}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[TOK3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[VIEW3:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[TOK3]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[VIEW3]]#1[] {ttg.partition = array<i32: 0>}
      %vb, %tb = ttng.tmem_load %b[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 6 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
