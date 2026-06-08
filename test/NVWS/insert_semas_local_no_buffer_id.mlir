// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_no_buffer_id
  tt.func @local_no_buffer_id(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for [[V7:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V10:%.*]] = nvws.semaphore.buffer [[V3]], [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem>
    // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttg.local_load [[V10]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> tensor<1xi32, #blocked>
    // CHECK: nvws.semaphore.release [[V2]], [[V9]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[V11:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V11]] : !ttg.async.token
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>

      %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
