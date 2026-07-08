// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#scalar = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#local_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two single-buffered TMEM groups are touched by partition 1 at clusters 1
  // and 3, while an independent partition-1 local buffer occupies the lane's
  // last in-loop cluster (4). Point-of-use lowering keeps each group's
  // in-loop acquires at its own first touching access (no shared tail
  // acquire at the cluster-4 frontier), threads no tokens through the inner
  // loop, and emits each group's loop-exit regain AFTER the loop: partition 1
  // drains the group's MMA-ready phase -- anchored at partition 0's last
  // in-loop load release, so the surviving release cannot move earlier --
  // and bridges it into the store gate with a tc5mma arrival. Partition 0's
  // own load->next-store edge is covered by its program order, so each store
  // gate keeps pending_count = 1 with partition 1 as its only releaser.
  // CHECK-LABEL: @cross_group_tail_acquire_schedule
  tt.func @cross_group_tail_acquire_schedule(
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>,
      %lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    // The frontier buffer is only ever accessed by partition 1, so it gets
    // no semaphores.
    // CHECK: [[FRONTIER:%.*]] = ttg.local_alloc {buffer.id = 402 : i32}
    %frontier = ttg.local_alloc {buffer.id = 402 : i32} : () -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

    // Both TMEM groups are hoisted out of the outer loop. Each gets a store
    // gate (true) plus an MMA-ready and a load-ready phase (all single
    // buffered, pending_count = 1), and its store gate is acquired once up
    // front; those two tokens are the only ones carried by the outer loop.
    // CHECK: [[ACC_A:%.*]] = ttng.tmem_alloc {buffer.id = 400 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[A_STORE:%.*]] = nvws.semaphore.create [[ACC_A]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[A_MMA:%.*]] = nvws.semaphore.create [[ACC_A]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[A_LOAD:%.*]] = nvws.semaphore.create [[ACC_A]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_B:%.*]] = ttng.tmem_alloc {buffer.id = 401 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[B_STORE:%.*]] = nvws.semaphore.create [[ACC_B]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[B_MMA:%.*]] = nvws.semaphore.create [[ACC_B]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[B_LOAD:%.*]] = nvws.semaphore.create [[ACC_B]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[A_TOK0:%.*]] = nvws.semaphore.acquire [[A_STORE]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[B_TOK0:%.*]] = nvws.semaphore.acquire [[B_STORE]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[OUT:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args(%{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}}, [[A_TOK:%.*]] = [[A_TOK0]], [[B_TOK:%.*]] = [[B_TOK0]]) -> (i32, !ttg.async.token, !ttg.async.token)
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // Partition 0 re-initializes each accumulator under its store gate,
      // then releases the MMA-ready phase.
      // CHECK: [[A_INIT_BUF:%.*]] = nvws.semaphore.buffer [[A_STORE]], [[A_TOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[A_INIT_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[A_MMA]], [[A_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_a, %tok_a = ttng.tmem_alloc %cst {buffer.id = 400 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[B_INIT_BUF:%.*]] = nvws.semaphore.buffer [[B_STORE]], [[B_TOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[B_INIT_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[B_MMA]], [[B_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %acc_b, %tok_b = ttng.tmem_alloc %cst {buffer.id = 401 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no tokens.
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} : i32 {
      %inner:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%a_token = %tok_a, %b_token = %tok_b) -> (!ttg.async.token, !ttg.async.token) : i32 {
        // Group A, cluster 1: point-of-use acquire at the MMA (not at the
        // cluster-4 tail), tc5mma arrival into the load-ready phase.
        // CHECK: [[A_MMA_TOK:%.*]] = nvws.semaphore.acquire [[A_MMA]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[A_MMA_BUF:%.*]] = nvws.semaphore.buffer [[A_MMA]], [[A_MMA_TOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[A_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[A_LOAD]], [[A_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_a = ttng.tc_gen5_mma %lhs, %rhs, %acc_a[%a_token], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // Group A, cluster 2: consumer load, buffer handed back to the
        // MMA-ready phase right at the read.
        // CHECK: [[A_LD_TOK:%.*]] = nvws.semaphore.acquire [[A_LOAD]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[A_LD_BUF:%.*]] = nvws.semaphore.buffer [[A_LOAD]], [[A_LD_TOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[A_LD_BUF]][] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[A_MMA]], [[A_LD_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %a_value, %a_read = ttng.tmem_load %acc_a[%mma_a] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_a"(%a_value) {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        // Group B, cluster 3: same point-of-use shape, at its own clusters.
        // CHECK: [[B_MMA_TOK:%.*]] = nvws.semaphore.acquire [[B_MMA]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[B_MMA_BUF:%.*]] = nvws.semaphore.buffer [[B_MMA]], [[B_MMA_TOK]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[B_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[B_LOAD]], [[B_MMA_TOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %mma_b = ttng.tc_gen5_mma %lhs, %rhs, %acc_b[%b_token], %true, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: [[B_LD_TOK:%.*]] = nvws.semaphore.acquire [[B_LOAD]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK: [[B_LD_BUF:%.*]] = nvws.semaphore.buffer [[B_LOAD]], [[B_LD_TOK]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[B_LD_BUF]][] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[B_MMA]], [[B_LD_TOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %b_value, %b_read = ttng.tmem_load %acc_b[%mma_b] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume_b"(%b_value) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

        %frontier_value = "frontier_value"() {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #scalar>
        // Partition 1's last in-loop access (cluster 4): no protocol ops
        // attach to it, and no tail acquires follow it.
        // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[FRONTIER]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
        ttg.local_store %frontier_value, %frontier {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<1xi32, #scalar> -> !ttg.memdesc<1xi32, #local_shared, #smem, mutable>

        scf.yield {ttg.partition = array<i32: 0, 1>} %a_read, %b_read : !ttg.async.token, !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0>} : i32
      // Loop-exit handoff, once per group after the inner loop: partition 1
      // drains the MMA-ready phase (waiting on partition 0's last in-loop
      // load release) and bridges it into the store gate with a tc5mma
      // arrival that also covers its own last MMA; partition 0 then acquires
      // the store gate for the next outer iteration's init store.
      // CHECK: [[A_EXIT:%.*]] = nvws.semaphore.acquire [[A_MMA]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[A_STORE]], [[A_EXIT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[A_NEXT:%.*]] = nvws.semaphore.acquire [[A_STORE]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[B_EXIT:%.*]] = nvws.semaphore.acquire [[B_MMA]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: nvws.semaphore.release [[B_STORE]], [[B_EXIT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[B_NEXT:%.*]] = nvws.semaphore.acquire [[B_STORE]] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{[-A-Za-z0-9_.$#]+}}, [[A_NEXT]], [[B_NEXT]] : i32, !ttg.async.token, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: "use_i32"([[OUT]]#0)
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}
