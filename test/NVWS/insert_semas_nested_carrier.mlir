// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The sourceful accumulator is stored by partition 0 before the inner loop
  // and read back by partition 0 after it, so OUTER_EMPTY carries the outer
  // permit across outer iterations.  The inner loop carries no semaphore
  // token: partition 1 acquires MMA_READY at the point of use adjacent to
  // the MMA, and a post-loop bridge acquire anchors the OUTER_EMPTY release
  // after the last in-loop read.
  // CHECK-LABEL: @outer_sourceful_alloc_inner_loop_reentry
  tt.func @outer_sourceful_alloc_inner_loop_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // Single-slot accumulator: one true(EMPTY) gate plus two false(FULL)
    // phases; the initial permit is acquired ahead of the outer loop.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32}
    // CHECK: [[MMA_READY:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // CHECK: [[ACC_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // CHECK: [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]]
    // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // Partition 0 initializes the accumulator under the carried permit and
      // hands the slot to partition 1's MMA.
      // CHECK: [[OUTER_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // The inner loop carries no semaphore token; MMA_READY is acquired at
      // the point of use next to the MMA on every iteration.
      // CHECK: nvws.semaphore.release [[MMA_READY]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[MMA_ACQ:%.*]] = nvws.semaphore.acquire [[MMA_READY]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[MMA_BUF:%.*]] = nvws.semaphore.buffer [[MMA_READY]], [[MMA_ACQ]] {ttg.partition = array<i32: 1>}
        // CHECK: ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>}
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // MMA completion flips ACC_FULL; partition 0 reads the accumulator
        // and recycles the slot back to MMA_READY for the next iteration.
        // CHECK: nvws.semaphore.release [[ACC_FULL]], [[MMA_ACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[READ_ACQ:%.*]] = nvws.semaphore.acquire [[ACC_FULL]] {ttg.partition = array<i32: 0>}
        // CHECK: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[ACC_FULL]], [[READ_ACQ]] {ttg.partition = array<i32: 0>}
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[READ_BUF]][] {ttg.partition = array<i32: 0>}
        %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[MMA_READY]], [[READ_ACQ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // Post-loop bridge: the leftover MMA_READY permit anchors the
      // OUTER_EMPTY release after the last in-loop read; partition 0 then
      // reads the final accumulator and carries the permit to the next
      // outer iteration.
      // CHECK: [[BRIDGE:%.*]] = nvws.semaphore.acquire [[MMA_READY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[BRIDGE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[OUTER_READ:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[OUTER_READ_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_READ]] {ttg.partition = array<i32: 0>}
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OUTER_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[OUTER_READ]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Completion differs between the conditional's stage-1 path and its
  // passthrough path.  The non-WS inner loop must therefore retain its token
  // carrier instead of moving the acquire to the first stage-0 MMA.
  // CHECK-LABEL: @branch_completion_requires_carrier
  tt.func @branch_completion_requires_carrier(
      %cond: i1, %lb: i32, %ub: i32, %step: i32,
      %lhs: !ttg.memdesc<128x64xf32, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf32, #shared, #smem>) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[BRANCH_ALLOC:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[BRANCH_EMPTY:%.*]] = nvws.semaphore.create [[BRANCH_ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BRANCH_READY:%.*]] = nvws.semaphore.create [[BRANCH_ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BRANCH_FULL:%.*]] = nvws.semaphore.create [[BRANCH_ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BRANCH_BACK:%.*]] = nvws.semaphore.create [[BRANCH_ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[FINAL_FULL:%.*]] = nvws.semaphore.create [[BRANCH_ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[BRANCH_INIT:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[BRANCH_OUTER:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[BRANCH_OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH_OUTER_TOKEN:%.*]] = [[BRANCH_INIT]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %i = %lb to %ub step %step iter_args(%tile = %c0) -> (i32) : i32 {
      // CHECK: [[INIT_BUF:%.*]] = nvws.semaphore.buffer [[BRANCH_EMPTY]], [[BRANCH_OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[INIT_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[BRANCH_READY]], [[BRANCH_OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[PRE_LOOP_ACQUIRE:%.*]] = nvws.semaphore.acquire [[BRANCH_READY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %acc, %tok = ttng.tmem_alloc %cst {buffer.copy = 1 : i32, buffer.id = 920 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: [[BRANCH_INNER:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[CARRIED_READY:%.*]] = [[PRE_LOOP_ACQUIRE]]) -> (!ttg.async.token)  : i32 {
      %inner = scf.for %j = %lb to %ub step %step iter_args(%iter = %tok) -> (!ttg.async.token) : i32 {
        // CHECK: [[FIRST_MMA_BUF:%.*]] = nvws.semaphore.buffer [[BRANCH_READY]], [[CARRIED_READY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: [[FIRST_MMA_TOKEN:%.*]] = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[FIRST_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma0 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%iter], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        // CHECK: [[BRANCH_RESULT:%.*]] = scf.if %{{[-A-Za-z0-9_.$#]+}} -> (!ttg.async.token) {
        %branch = scf.if %cond -> (!ttg.async.token) {
          // CHECK: [[BRANCH_MMA_BUF:%.*]] = nvws.semaphore.buffer [[BRANCH_READY]], [[CARRIED_READY]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: [[BRANCH_MMA_TOKEN:%.*]] = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[BRANCH_MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: nvws.semaphore.release [[BRANCH_FULL]], [[CARRIED_READY]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %mma1 = ttng.tc_gen5_mma %lhs, %rhs, %acc[%mma0], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: [[BRANCH_READ_ACQUIRE:%.*]] = nvws.semaphore.acquire [[BRANCH_FULL]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: [[BRANCH_READ_BUF:%.*]] = nvws.semaphore.buffer [[BRANCH_FULL]], [[BRANCH_READ_ACQUIRE]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK-NEXT: [[BRANCH_READ_VALUE:%.*]], [[BRANCH_READ_TOKEN:%.*]] = ttng.tmem_load [[BRANCH_READ_BUF]][] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          // CHECK-NEXT: nvws.semaphore.release [[BRANCH_BACK]], [[BRANCH_READ_ACQUIRE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          %value0, %read0 = ttng.tmem_load %acc[%mma1] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          "consume.branch"(%value0) {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          // CHECK: [[BRANCH_NEXT_ACQUIRE:%.*]] = nvws.semaphore.acquire [[BRANCH_BACK]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[BRANCH_NEXT_ACQUIRE]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %read0 : !ttg.async.token
        } else {
          // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[CARRIED_READY]] : !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1>} %mma0 : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}

        // CHECK: nvws.semaphore.release [[FINAL_FULL]], [[BRANCH_RESULT]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[FINAL_READ_ACQUIRE:%.*]] = nvws.semaphore.acquire [[FINAL_FULL]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[FINAL_READ_BUF:%.*]] = nvws.semaphore.buffer [[FINAL_FULL]], [[FINAL_READ_ACQUIRE]] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: [[FINAL_READ_VALUE:%.*]], [[FINAL_READ_TOKEN:%.*]] = ttng.tmem_load [[FINAL_READ_BUF]][] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        // CHECK-NEXT: nvws.semaphore.release [[BRANCH_READY]], [[FINAL_READ_ACQUIRE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %value1, %read1 = ttng.tmem_load %acc[%branch] {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "consume"(%value1) {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[NEXT_ITER_ACQUIRE:%.*]] = nvws.semaphore.acquire [[BRANCH_READY]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[NEXT_ITER_ACQUIRE]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %read1 : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: nvws.semaphore.release [[BRANCH_EMPTY]], [[BRANCH_INNER]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[POST_READ_ACQUIRE:%.*]] = nvws.semaphore.acquire [[BRANCH_EMPTY]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[POST_READ_BUF:%.*]] = nvws.semaphore.buffer [[BRANCH_EMPTY]], [[POST_READ_ACQUIRE]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: [[POST_READ_VALUE:%.*]], [[POST_READ_TOKEN:%.*]] = ttng.tmem_load [[POST_READ_BUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "consume.post"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.yield {{.*}}[[POST_READ_ACQUIRE]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "consume.outer"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // A relocated recurrence acquire must not leave a stale scheduling arc from
  // the stage-1 read to the next stage-0 MMA.  Its post-loop bridge also stays
  // at the owner-1 boundary instead of moving to the owner-0 final read.
  // CHECK-LABEL: @scheduled_relocated_acquire_boundaries
  tt.func @scheduled_relocated_acquire_boundaries(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[MMA_READY:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[ACC_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[OUTER_LOOP:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[OUTER_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: nvws.semaphore.release [[MMA_READY]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
      %inner = scf.for %iv1 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
        %lhs = "load1"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %rhs = "load2"(%iv1) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[MMA_ACQUIRE:%.*]] = nvws.semaphore.acquire [[MMA_READY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[MMA_BUF:%.*]] = nvws.semaphore.buffer [[MMA_READY]], [[MMA_ACQUIRE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: [[MMA_TOKEN:%.*]] = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[MMA_BUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: nvws.semaphore.release [[ACC_FULL]], [[MMA_ACQUIRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[READ_ACQUIRE:%.*]] = nvws.semaphore.acquire [[ACC_FULL]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK-NEXT: [[READ_BUF:%.*]] = nvws.semaphore.buffer [[ACC_FULL]], [[READ_ACQUIRE]] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK-NEXT: [[READ_VALUE:%.*]], [[READ_TOKEN:%.*]] = ttng.tmem_load [[READ_BUF]][] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %read_tok = ttng.tmem_load %acc[%mma] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[MMA_READY]], [[READ_ACQUIRE]] [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "use"(%val) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

      // CHECK: [[BRIDGE_ACQUIRE:%.*]] = nvws.semaphore.acquire [[MMA_READY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: nvws.semaphore.release [[OUTER_EMPTY]], [[BRIDGE_ACQUIRE]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[OUTER_READ_ACQUIRE:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK-NEXT: [[OUTER_READ_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_READ_ACQUIRE]] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK-NEXT: [[OUTER_READ_VALUE:%.*]], [[OUTER_READ_TOKEN:%.*]] = ttng.tmem_load [[OUTER_READ_BUF]][] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %out, %out_tok = ttng.tmem_load %acc[%inner] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %c0_i32 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[OUTER_READ_ACQUIRE]] : i32, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_reentry_without_post_access
  tt.func @three_level_reentry_without_post_access(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // CHECK: [[V1:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V6:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[V8:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V9:%.*]] = nvws.semaphore.buffer [[V2]], [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V9]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: nvws.semaphore.release [[V3]], [[V8]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V14:%.*]]:3 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V12:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V13:%.*]] = [[V10]]) -> (i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // CHECK: [[V17:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V15:%.*]] = [[V12]], [[V16:%.*]] = [[V13]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[V18:%.*]] = nvws.semaphore.buffer [[V3]], [[V16]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[V4]], [[V16]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK: [[V20:%.*]] = nvws.semaphore.buffer [[V4]], [[V19]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V20]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[V5]], [[V19]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {{.*}}[[V21]]
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        // CHECK: scf.yield {{.*}}[[V17]]#0, [[V17]]#1
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %inner : i32, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // CHECK: nvws.semaphore.release [[V2]], [[V14]]#2 [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @three_level_sourceful_alloc_reentry
  tt.func @three_level_sourceful_alloc_reentry(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true

    // The innermost recurrence gets its own initial permit.  OUTER_EMPTY still
    // carries the sourceful allocation across outer iterations.
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[INNER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32}
    // CHECK: [[OUTER_EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32}
    // CHECK: [[OUTER_TO_MIDDLE:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // CHECK: [[INNER_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // CHECK: [[MIDDLE_FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // CHECK: [[OUTER_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]]
    // CHECK: [[OUTER:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OUTER_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_TOKEN:%.*]] = [[OUTER_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
    %outer = scf.for %iv0 = %lb to %ub step %step iter_args(%tile = %c0_i32) -> (i32) : i32 {
      // CHECK: [[OUTER_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[OUTER_BUF]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>}
      %acc, %tok = ttng.tmem_alloc %cst {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[OUTER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[MIDDLE_ENTRY:%.*]] = nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
      // CHECK: [[MIDDLE:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[MIDDLE_IV:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[MIDDLE_TOKEN:%.*]] = [[MIDDLE_ENTRY]]) -> (i32, !ttg.async.token)  : i32 {
      %middle:2 = scf.for %iv1 = %lb to %ub step %step iter_args(%mid = %c0_i32, %mtok = %tok) -> (i32, !ttg.async.token) : i32 {
        // The innermost loop carries no semaphore token; its EMPTY acquire is
        // adjacent to the MMA on every iteration.
        // CHECK: scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}}  : i32 {
        %inner = scf.for %iv2 = %lb to %ub step %step iter_args(%tok1 = %mtok) -> (!ttg.async.token) : i32 {
          %lhs = "load1"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
          %rhs = "load2"(%iv2) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
          // CHECK: [[INNER_ACQ:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
          // CHECK-NEXT: [[INNER_BUF:%.*]] = nvws.semaphore.buffer [[INNER_EMPTY]], [[INNER_ACQ]] {ttg.partition = array<i32: 1>}
          %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%tok1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          // CHECK: nvws.semaphore.release [[INNER_FULL]], [[INNER_ACQ]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
          // CHECK: [[INNER_READ:%.*]] = nvws.semaphore.acquire [[INNER_FULL]] {ttg.partition = array<i32: 0>}
          // CHECK: [[INNER_READ_BUF:%.*]] = nvws.semaphore.buffer [[INNER_FULL]], [[INNER_READ]] {ttg.partition = array<i32: 0>}
          // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[INNER_READ_BUF]][] {ttg.partition = array<i32: 0>}
          %val, %read_tok = ttng.tmem_load %acc[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          // CHECK: nvws.semaphore.release [[INNER_EMPTY]], [[INNER_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %read_tok : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}

        // CHECK: [[FINAL_INNER:%.*]] = nvws.semaphore.acquire [[INNER_EMPTY]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[MIDDLE_FULL]], [[FINAL_INNER]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: [[MIDDLE_READ:%.*]] = nvws.semaphore.acquire [[MIDDLE_FULL]] {ttg.partition = array<i32: 0>}
        // CHECK: [[MIDDLE_READ_BUF:%.*]] = nvws.semaphore.buffer [[MIDDLE_FULL]], [[MIDDLE_READ]] {ttg.partition = array<i32: 0>}
        // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[MIDDLE_READ_BUF]][] {ttg.partition = array<i32: 0>}
        %mid_out, %mid_tok = ttng.tmem_load %acc[%inner] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK: nvws.semaphore.release [[OUTER_TO_MIDDLE]], [[MIDDLE_READ]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use"(%mid_out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %mid_next = arith.addi %mid, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        // The middle regain is also the bridge source for the next inner loop.
        // CHECK: [[MIDDLE_REGAIN:%.*]] = nvws.semaphore.acquire [[OUTER_TO_MIDDLE]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: nvws.semaphore.release [[INNER_EMPTY]], [[MIDDLE_REGAIN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // CHECK: scf.yield {{.*}}[[MIDDLE_REGAIN]]
        scf.yield {ttg.partition = array<i32: 0, 1>} %mid_next, %mid_tok : i32, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>]}

      // CHECK: nvws.semaphore.release [[OUTER_EMPTY]], [[MIDDLE]]#1 [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[OUTER_READ:%.*]] = nvws.semaphore.acquire [[OUTER_EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[OUTER_READ_BUF:%.*]] = nvws.semaphore.buffer [[OUTER_EMPTY]], [[OUTER_READ]] {ttg.partition = array<i32: 0>}
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[OUTER_READ_BUF]][] {ttg.partition = array<i32: 0>}
      %out, %out_tok = ttng.tmem_load %acc[%middle#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%out) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %next = arith.addi %tile, %middle#0 {ttg.partition = array<i32: 0>} : i32
      // CHECK: scf.yield {{.*}}[[OUTER_READ]]
      scf.yield {ttg.partition = array<i32: 0, 1>} %next : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%outer) : (i32) -> ()
    tt.return
  }
}
