// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore -cse | FileCheck %s --check-prefix=LOWER --implicit-check-not=nvws.descriptor_load

// Two exact-alias members are filled by one partition before either is
// consumed by another partition.  The first fill is a TMA load and the second
// is synchronous.  Their one ownership handoff must retain both completion
// signals: the TMA completion and the explicit arrival after the local store.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @same_owner_mixed_completion
  // LOWER-LABEL: @same_owner_mixed_completion
  // LOWER: ttng.init_barrier %{{.*}}, 2
  // LOWER: ttng.barrier_expect %{{.*}}, 16384
  // LOWER: ttng.async_tma_copy_global_to_local
  // LOWER: ttng.arrive_barrier %{{.*}}, 1
  tt.func @same_owner_mixed_completion(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %i: i32, %lb: i32, %ub: i32, %step: i32) {
    %tma = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %sync = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>

    // CHECK: [[TMA_BASE:%.*]] = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32}
    // CHECK: [[SYNC_BASE:%.*]] = ttg.local_alloc {buffer.id = 610 : i32, buffer.offset = 0 : i32}
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[TMA_BASE]], [[SYNC_BASE]] true {pending_count = 1 : i32}
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create [[TMA_BASE]], [[SYNC_BASE]] false {pending_count = 2 : i32}
    // CHECK: scf.for
    scf.for %iv = %lb to %ub step %step : i32 {
      // CHECK: [[PRODUCER_TOKEN:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PRODUCER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[EMPTY]], [[PRODUCER_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] 16384 [[PRODUCER_BUFFERS]]#0 {ttg.partition = array<i32: 0>}
      nvws.descriptor_load %desc[%i, %i] 16384 %tma {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[PRODUCER_BUFFERS]]#1 {ttg.partition = array<i32: 0>}
      ttg.local_store %value, %sync {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      // CHECK: nvws.semaphore.release [[FULL]], [[PRODUCER_TOKEN]] [#nvws.async_op<none>, #nvws.async_op<tma_load>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}

      // CHECK: [[CONSUMER_TOKEN:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK: [[CONSUMER_BUFFERS:%.*]]:2 = nvws.semaphore.buffer [[FULL]], [[CONSUMER_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[CONSUMER_BUFFERS]]#1 {ttg.partition = array<i32: 1>}
      %sync_value = ttg.local_load %sync {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      // CHECK: ttg.local_load [[CONSUMER_BUFFERS]]#0 {ttg.partition = array<i32: 1>}
      %tma_value = ttg.local_load %tma {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      "consume"(%sync_value, %tma_value) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>) -> ()
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CONSUMER_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
