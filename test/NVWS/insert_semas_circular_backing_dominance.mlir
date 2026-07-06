// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
!ty = tensor<128x128xf16, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The start-1 allocation appears first, while producer order remains
  // start0, start1. Folding must move the canonical start-0 backing before the
  // merged semaphore creates.
  // CHECK-LABEL: @circular_start_zero_backing_dominates_creates
  // CHECK: [[BASE:%.*]] = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[BASE]] false
  tt.func @circular_start_zero_backing_dominates_creates(
      %lb: i32, %ub: i32, %step: i32) {
    %start1_payload = "make_start1"() {ttg.partition = array<i32: 1>} : () -> !ty
    %start0_payload = "make_start0"() {ttg.partition = array<i32: 1>} : () -> !ty
    %start1 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 1 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %start0 = ttg.local_alloc {buffer.circular, buffer.copy = 2 : i32, buffer.id = 700 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %start0_payload, %start0 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %start0_value = ttg.local_load %start0 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      ttg.local_store %start1_payload, %start1 {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %start1_value = ttg.local_load %start1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ty
      "use"(%start0_value, %start1_value) {ttg.partition = array<i32: 2>} : (!ty, !ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
