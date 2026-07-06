// RUN: triton-opt %s --allow-unregistered-dialect --nvws-assign-stage-phase | FileCheck %s --check-prefix=ASP
// RUN: triton-opt %s --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s --allow-unregistered-dialect --nvws-assign-stage-phase --tritongpu-partition-loops --nvws-lower-warp-group | FileCheck %s --check-prefix=PARTITION

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // ASP-LABEL: @circular_cross_partition_acquire_sequence
  // ASP: [[EMPTY:%.*]] = nvws.semaphore.create
  // ASP: [[FULL:%.*]] = nvws.semaphore.create
  // ASP: [[A_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
  // ASP: [[A_BIT:%.*]] = arith.shli {{%.*}}, [[A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[A_WORD:%.*]] = arith.xori {{%.*}}, [[A_BIT]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[A_SHIFT:%.*]] = arith.shrui [[A_WORD]], [[A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[A_PHASE:%.*]] = arith.andi [[A_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: nvws.semaphore.acquire [[EMPTY]][[[A_STAGE]], [[A_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // ASP-NEXT: nvws.semaphore.acquire [[EMPTY]][[[A_STAGE]], [[A_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // ASP: [[B_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
  // ASP: [[B_BIT:%.*]] = arith.shli {{%.*}}, [[B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[B_WORD:%.*]] = arith.xori [[A_WORD]], [[B_BIT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[B_SHIFT:%.*]] = arith.shrui [[B_WORD]], [[B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[B_PHASE:%.*]] = arith.andi [[B_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: nvws.semaphore.acquire [[EMPTY]][[[B_STAGE]], [[B_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // ASP-NEXT: nvws.semaphore.acquire [[EMPTY]][[[B_STAGE]], [[B_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}

  // LOWER-LABEL: @circular_cross_partition_acquire_sequence
  // LOWER: [[A_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS:%.*]][[[A_STAGE:%.*]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER-NEXT: ttng.wait_barrier [[A_BARRIER]], [[A_PHASE:%.*]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER: [[A_SHADOW_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[A_STAGE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER-NEXT: ttng.wait_barrier [[A_SHADOW_BARRIER]], [[A_PHASE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER: [[B_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[B_STAGE:%.*]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER-NEXT: ttng.wait_barrier [[B_BARRIER]], [[B_PHASE:%.*]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER: [[B_SHADOW_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[B_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER-NEXT: ttng.wait_barrier [[B_SHADOW_BARRIER]], [[B_PHASE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}

  // PARTITION-LABEL: @circular_cross_partition_acquire_sequence
  // PARTITION: partition0
  // PARTITION: [[P0_A_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_A_BIT:%.*]] = arith.shli {{%.*}}, [[P0_A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_A_WORD:%.*]] = arith.xori {{%.*}}, [[P0_A_BIT]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_A_SHIFT:%.*]] = arith.shrui [[P0_A_WORD]], [[P0_A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_A_PHASE:%.*]] = arith.andi [[P0_A_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: nvws.semaphore.acquire [[P0_EMPTY:%.*]][[[P0_A_STAGE]], [[P0_A_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // PARTITION: [[P0_B_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_B_BIT:%.*]] = arith.shli {{%.*}}, [[P0_B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_B_WORD:%.*]] = arith.xori [[P0_A_WORD]], [[P0_B_BIT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_B_SHIFT:%.*]] = arith.shrui [[P0_B_WORD]], [[P0_B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_B_PHASE:%.*]] = arith.andi [[P0_B_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P0_B_TOKEN:%.*]] = nvws.semaphore.acquire [[P0_EMPTY]][[[P0_B_STAGE]], [[P0_B_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32}
  // PARTITION-NEXT: {{%.*}} = nvws.semaphore.buffer [[P0_EMPTY]][[[P0_B_STAGE]]], [[P0_B_TOKEN]]
  // PARTITION: partition1
  // PARTITION: [[P1_A_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_A_BIT:%.*]] = arith.shli {{%.*}}, [[P1_A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_A_WORD:%.*]] = arith.xori {{%.*}}, [[P1_A_BIT]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_A_SHIFT:%.*]] = arith.shrui [[P1_A_WORD]], [[P1_A_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_A_PHASE:%.*]] = arith.andi [[P1_A_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_A_TOKEN:%.*]] = nvws.semaphore.acquire [[P1_EMPTY:%.*]][[[P1_A_STAGE]], [[P1_A_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32}
  // PARTITION-NEXT: {{%.*}} = nvws.semaphore.buffer [[P1_EMPTY]][[[P1_A_STAGE]]], [[P1_A_TOKEN]]
  // PARTITION: [[P1_B_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_B_BIT:%.*]] = arith.shli {{%.*}}, [[P1_B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_B_WORD:%.*]] = arith.xori [[P1_A_WORD]], [[P1_B_BIT]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_B_SHIFT:%.*]] = arith.shrui [[P1_B_WORD]], [[P1_B_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: [[P1_B_PHASE:%.*]] = arith.andi [[P1_B_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
  // PARTITION: nvws.semaphore.acquire [[P1_EMPTY]][[[P1_B_STAGE]], [[P1_B_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32}
  // PARTITION-NEXT: scf.yield
  tt.func @circular_cross_partition_acquire_sequence(%lb: i32, %ub: i32,
                                                      %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = 307 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %a_value = "producer_a"() {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : () -> tensor<1xi32, #blocked>
      %a_offset = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %a_token = nvws.semaphore.acquire %empty[%a_offset] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %a_buffer = nvws.semaphore.buffer %empty[%a_offset], %a_token {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %a_value, %a_buffer {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %full[%a_offset], %a_token [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %b_value = "producer_b"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<1xi32, #blocked>
      %b_offset = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %b_token = nvws.semaphore.acquire %empty[%b_offset] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b_buffer = nvws.semaphore.buffer %empty[%b_offset], %b_token {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %b_value, %b_buffer {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %full[%b_offset], %b_token [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %b_consumer_offset = arith.constant {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} 0 : i32
      %b_consumer_token = nvws.semaphore.acquire %full[%b_consumer_offset] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b_consumer_buffer = nvws.semaphore.buffer %full[%b_consumer_offset], %b_consumer_token {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %b_loaded = ttg.local_load %b_consumer_buffer {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      "use_b"(%b_loaded) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : (tensor<1xi32, #blocked>) -> ()
      nvws.semaphore.release %empty[%b_consumer_offset], %b_consumer_token [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %a_consumer_offset = arith.constant {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} -1 : i32
      %a_consumer_token = nvws.semaphore.acquire %full[%a_consumer_offset] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %a_consumer_buffer = nvws.semaphore.buffer %full[%a_consumer_offset], %a_consumer_token {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %a_loaded = ttg.local_load %a_consumer_buffer {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32, #blocked>
      "use_a"(%a_loaded) {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : (tensor<1xi32, #blocked>) -> ()
      nvws.semaphore.release %empty[%a_consumer_offset], %a_consumer_token [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 13 : i32}
    tt.return
  }
}
