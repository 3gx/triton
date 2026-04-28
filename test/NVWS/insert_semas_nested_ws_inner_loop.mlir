// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// M2 PIN (nested-loop hold-rule extension, plan v3).
//
// A WS-tagged OUTER loop wrapping a non-WS INNER loop with ONE inner-confined
// ping-pong buffer (the persistent-FA qk shape, design v3 §8): tc_gen5_mma
// (partition 1) writes the tmem accumulator, tmem_load (partition 0) reads it,
// BOTH confined to the inner loop; nothing touches the buffer at the outer
// level.
//
// This pins the enabled native shape: no root entry acquire, no added carrier
// iter_args on either loop, and the writer acquire moved to the first toucher.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @nested_ws_inner_loop(
  tt.func @nested_ws_inner_loop(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Alloc grows to 1x (single-buffered) and the two ping-pong semaphores are
    // created at the root: EMPTY (true) gates the writer, FULL (false) the reader.
    // CHECK:           [[POISON:%.*]] = ub.poison : !ttg.async.token
    // CHECK:           [[BUF_ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK:           [[EMPTY:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK:           [[FULL:%.*]] = nvws.semaphore.create [[BUF_ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Both loops carry the poison token unchanged: no acquire hoisted to the root,
    // no new carrier iter_args beyond the existing token thread.
    // CHECK:           [[OUTER:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[OCARRY:%.*]] = [[POISON]]) -> (!ttg.async.token)  : i32 {
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      // CHECK:             [[INNER:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
        // CHECK:               {{.*}} = "loadA"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // CHECK:               {{.*}} = "loadB"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
        // Writer (partition 1): acquire EMPTY at first toucher, buffer, MMA, release FULL.
        // CHECK:               [[WTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[WBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[WTOK]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               {{.*}} = ttng.tc_gen5_mma %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}}, [[WBUF]][], %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK:               nvws.semaphore.release [[FULL]], [[WTOK]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // Reader (partition 0): acquire FULL, buffer, tmem_load, release EMPTY.
        // CHECK:               [[RTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        // CHECK:               [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[RTOK]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK:               %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[RBUF]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        // CHECK:               nvws.semaphore.release [[EMPTY]], [[RTOK]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK:               "use"(%{{[-A-Za-z0-9_.$#]+}}) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // Inner loop yields the poison token (carrier unchanged, no buffer threaded out).
        // CHECK:               scf.yield {ttg.partition = array<i32: 0, 1>} [[POISON]] : !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
      // CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      // CHECK:             scf.yield {ttg.partition = array<i32: 0, 1>} [[INNER]] : !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
    // CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    // No post-loop TMEM access: nothing emitted after the outer loop.
    // CHECK:           tt.return
    tt.return
  }
}
