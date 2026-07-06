// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas=num-stages=4 --nvws-lower-semaphore=num-stages=4 --tritongpu-partition-loops --nvws-lower-warp-group --tritongpu-schedule-loops=num-stages=4 --tritongpu-pipeline=num-stages=4 | FileCheck %s --check-prefix=PIPE

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance a four-slot physical ring on every iteration.
  // Each logical buffer therefore reuses its slots after two iterations. Its
  // stage-3 EMPTY release followed by a stage-0 reacquire has required owner
  // delay +1, but the reverse FULL handoff has delay -3. The complete owner
  // cycle has delay -2 and is legal: the producer may block while the
  // independent consumer releases the old slot.
  // CHECK-LABEL: @legal_cross_partition_backpressure
  // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // CHECK: nvws.semaphore.release {{.*}} {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
  // PIPE-LABEL: @legal_cross_partition_backpressure
  // PIPE: ttg.warp_specialize
  // PIPE: default {
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_load
  // PIPE: partition0
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  // PIPE: partition1
  // PIPE: scf.for
  // PIPE: ttng.wait_barrier
  // PIPE: ttg.local_store
  tt.func @legal_cross_partition_backpressure(%lb: i32, %ub: i32, %step: i32) {
    %a = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.circular, buffer.copy = 4 : i32, buffer.id = 422 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

    scf.for %iv = %lb to %ub step %step : i32 {
      %av = "producer_a"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %av, %a {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      %bv = "producer_b"() {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : () -> tensor<128x64xf16, #blocked>
      ttg.local_store %bv, %b {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      %br = ttg.local_load %b {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_b"(%br) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      %ar = ttg.local_load %a {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume_a"(%ar) {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
