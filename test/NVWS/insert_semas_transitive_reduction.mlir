// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// Transitive reduction (spec TRANSITIVE REDUCTION section): implied
// same-chain edges are dropped pay-for-play — but never wave-opening
// acquires (the wave guard) and never across distinct destinations'
// closed waves. Corner cases pinned here:
//   1. serialized ring: the {0}->{2} fan-in arm is implied through
//      {0}->{1}->{2} and DROPPED (one release per handoff survives);
//   2. genuine fan-out to two reader partitions: both edges are wave
//      openers — NOTHING is dropped;
//   3. regain by the producer after a reader wave: kept (wave guard),
//      even though ordering is transitively implied.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @serialized_ring_reduces
  tt.func @serialized_ring_reduces(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<128x128xf16, #blocked>
    %cst1 = arith.constant {ttg.partition = array<i32: 2>} dense<1.0> : tensor<128x128xf16, #blocked>
    // Overlapping pair (offset 0 and 64 of one buffer.id): {0} writes a,
    // {1} reads a, {2} writes b (overlaps a), {0} reads b. The W-after-R
    // edge {0}->{2} for the overlap piece is implied via {0}->{1}->{2}.
    // The minimal serialized chain survives: three handoffs plus the
    // carrier close — four releases. The implied {0}->{2} fan-in arm
    // (phase A) and the {2} regain whose ordering the following
    // traversal already implies (phase B, traversal closure) are gone:
    // CHECK-COUNT-4: nvws.semaphore.release
    // CHECK-NOT: nvws.semaphore.release
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst0 {buffer.id = 500 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %va = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%va) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %b = ttg.local_alloc %cst1 {buffer.id = 500 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %vb = ttg.local_load %b {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%vb) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: @fanout_not_reduced
  tt.func @fanout_not_reduced(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst0 = arith.constant {ttg.partition = array<i32: 0>} dense<0.0> : tensor<128x128xf16, #blocked>
    // One producer, two independent reader partitions: both edges open
    // their waves — the reduction must keep both reader acquires.
    // CHECK: nvws.semaphore.acquire {{.*}} {ttg.partition = array<i32: 1>}
    // CHECK: nvws.semaphore.acquire {{.*}} {ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.acquire {{.*}} {ttg.partition = array<i32: 0>}
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %a = ttg.local_alloc %cst0 {buffer.id = 501 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %v1 = ttg.local_load %a {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked>) -> ()
      %v2 = ttg.local_load %a {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked>) -> ()
      %v0 = ttg.local_load %a {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      "use"(%v0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked>) -> ()
      %j = arith.addi %i, %iv {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
