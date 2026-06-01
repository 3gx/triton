// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @outer_produced_inner_consumed
  tt.func @outer_produced_inner_consumed(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
    // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 2>
    // CHECK-NEXT: scf.for {{.*}} iter_args([[ITER:%.*]] = [[PTOK]])
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>

    scf.for %outer = %lb to %ub step %step : i32 {
      %c0_p = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      %write = ttg.memdesc_index %alloc[%c0_p] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %value = "producer"() {ttg.partition = array<i32: 2>} : () -> tensor<128x64xf16, #blocked>
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {{.*}}ttg.partition = array<i32: 2>
      // CHECK-NEXT: ttg.local_store {{.*}}, [[PBUF]] {{.*}}ttg.partition = array<i32: 2>
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ITER]] [#nvws.async_op<none>] {{.*}}ttg.partition = array<i32: 2>
      ttg.local_store %value, %write {ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

      %c0_c = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %read = ttg.memdesc_index %alloc[%c0_c] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {{.*}}ttg.partition = array<i32: 1>
      // CHECK-NEXT: [[CBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {{.*}}ttg.partition = array<i32: 1>
      // CHECK-NEXT: scf.for
      scf.for %inner = %lb to %ub step %step : i32 {
        // CHECK: [[LOADED:%.*]] = ttg.local_load [[CBUF]] {{.*}}ttg.partition = array<i32: 1>
        %loaded = ttg.local_load %read {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked>
        "use_tensor"(%loaded) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> ()
      } {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {{.*}}ttg.partition = array<i32: 1>
      // CHECK-NEXT: [[NEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 2>
      // CHECK-NEXT: scf.yield {{.*}}[[NEXT]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
