// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_no_buffer_id
  tt.func @local_no_buffer_id(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for [[V7:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem>
    // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V11]] : !ttg.async.token
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      %c1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
      %l = ttg.local_load %rv {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
