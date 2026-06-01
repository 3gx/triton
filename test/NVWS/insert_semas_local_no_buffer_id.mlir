// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_no_buffer_id
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>
  // CHECK-NEXT: scf.for {{.*}} iter_args([[ITER:%.*]] = [[PTOK]])
  tt.func @local_no_buffer_id(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ITER]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[ITER]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      %c1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOAD:%.*]] = ttg.local_load [[RBUF]] {ttg.partition = array<i32: 1>}
      %l = ttg.local_load %rv {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[NEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: scf.yield {{.*}}[[NEXT]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
