// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

// M0 PIN (nested-loop hold-rule extension, plan v3 §M0 item 5 — the
// canDrop(If) golden; the true If-ENCLOSER target is ABSENT from the corpus).
//
// WS-tagged outer loop -> scf.if -> non-WS inner loop, with ONE inner-confined
// ping-pong buffer in the if-branch. The scf.if sits BETWEEN the WS loop and
// the inner for, so it is the inner loop's encloser. The SYNC-DAG today is
// outer holdrule{gated(nested-final)} + inner holdrule{gated(non-ws-loop)};
// the if threads the carrier.
//
// This pins TODAY's all-gated emission. After M2, canDrop(If)=false keeps the
// inner loop GATED (clause (ii) of edit 1 finds the scf.if in the chain) and
// the only change is the dump gate-reason label (non-ws-loop -> if-encloser);
// the EMITTED IR here must stay BYTE-IDENTICAL. This is the only artifact that
// lets M2 verify the C3 claim ("if-encloser stays byte-identical, only the
// label changes"). Optimizing this shape is the UNSCHEDULED M4.2 follow-up
// (flip canDrop(If) to true + build the else pass-through).


// The script is designed to make adding checks to
// a test case fast, it is *not* designed to be authoritative
// about what constitutes a good test! The CHECK should be
// minimized and named to reflect the test intent.

// CHECK: #[[$ATTR_0:.+]] = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK: #[[$ATTR_2:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
// CHECK: #[[$ATTR_3:.+]] = #ttg.shared_memory
// CHECK: #[[$ATTR_4:.+]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-LABEL:   tt.func @if_encloser_inner_loop(
// CHECK-SAME:  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i1) {
// CHECK:           %[[VAL_4:.*]] = ub.poison : !ttg.async.token
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>
// CHECK:           %[[VAL_7:.*]] = nvws.semaphore.create %[[VAL_6]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>
// CHECK:           %[[VAL_8:.*]] = nvws.semaphore.create %[[VAL_6]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>
// CHECK:           %[[VAL_9:.*]] = nvws.semaphore.acquire %[[VAL_7]] : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]> -> !ttg.async.token
// CHECK:           %[[VAL_10:.*]]:2 = scf.for %[[VAL_11:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] iter_args(%[[VAL_12:.*]] = %[[VAL_4]], %[[VAL_13:.*]] = %[[VAL_9]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
// CHECK:             %[[VAL_14:.*]]:2 = scf.if %[[VAL_3]] -> (!ttg.async.token, !ttg.async.token) {
// CHECK:               %[[VAL_15:.*]]:2 = scf.for %[[VAL_16:.*]] = %[[VAL_0]] to %[[VAL_1]] step %[[VAL_2]] iter_args(%[[VAL_17:.*]] = %[[VAL_12]], %[[VAL_18:.*]] = %[[VAL_13]]) -> (!ttg.async.token, !ttg.async.token)  : i32 {
// CHECK:                 %[[VAL_19:.*]] = "loadA"(%[[VAL_16]]) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #[[$ATTR_1]], #[[$ATTR_3]]>
// CHECK:                 %[[VAL_20:.*]] = "loadB"(%[[VAL_16]]) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #[[$ATTR_2]], #[[$ATTR_3]]>
// CHECK:                 %[[VAL_21:.*]] = nvws.semaphore.buffer %[[VAL_7]], %[[VAL_18]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable, 1x128x128>
// CHECK:                 %[[VAL_22:.*]] = ttng.tc_gen5_mma %[[VAL_19]], %[[VAL_20]], %[[VAL_21]][], %[[VAL_5]], %[[VAL_5]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #[[$ATTR_1]], #[[$ATTR_3]]>, !ttg.memdesc<64x128xf16, #[[$ATTR_2]], #[[$ATTR_3]]>, !ttg.memdesc<128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable, 1x128x128>
// CHECK:                 nvws.semaphore.release %[[VAL_8]], %[[VAL_18]] [#nvws.async_op<tc5mma>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>, !ttg.async.token
// CHECK:                 %[[VAL_23:.*]] = nvws.semaphore.acquire %[[VAL_8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]> -> !ttg.async.token
// CHECK:                 %[[VAL_24:.*]] = nvws.semaphore.buffer %[[VAL_8]], %[[VAL_23]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable, 1x128x128>
// CHECK:                 %[[VAL_25:.*]], %[[VAL_26:.*]] = ttng.tmem_load %[[VAL_24]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #[[$ATTR_0]]>
// CHECK:                 nvws.semaphore.release %[[VAL_7]], %[[VAL_23]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]>, !ttg.async.token
// CHECK:                 "use"(%[[VAL_25]]) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #[[$ATTR_0]]>) -> ()
// CHECK:                 %[[VAL_27:.*]] = nvws.semaphore.acquire %[[VAL_7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #[[$ATTR_4]], #ttng.tensor_memory, mutable>]> -> !ttg.async.token
// CHECK:                 scf.yield {ttg.partition = array<i32: 0, 1>} %[[VAL_4]], %[[VAL_27]] : !ttg.async.token, !ttg.async.token
// CHECK:               } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
// CHECK:               scf.yield {ttg.partition = array<i32: 0, 1>} %[[VAL_28:.*]]#0, %[[VAL_28]]#1 : !ttg.async.token, !ttg.async.token
// CHECK:             } else {
// CHECK:               scf.yield {ttg.partition = array<i32: 0, 1>} %[[VAL_12]], %[[VAL_13]] : !ttg.async.token, !ttg.async.token
// CHECK:             } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
// CHECK:             scf.yield {ttg.partition = array<i32: 0, 1>} %[[VAL_29:.*]]#0, %[[VAL_29]]#1 : !ttg.async.token, !ttg.async.token
// CHECK:           } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
// CHECK:           tt.return
// CHECK:         }
  tt.func @if_encloser_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %true = arith.constant true
    %res, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %o = scf.for %iv0 = %lb to %ub step %step iter_args(%t0 = %tok) -> (!ttg.async.token) : i32 {
      %r = scf.if %cond -> (!ttg.async.token) {
        %i = scf.for %iv = %lb to %ub step %step iter_args(%t1 = %t0) -> (!ttg.async.token) : i32 {
          %sA = "loadA"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf16, #shared, #smem>
          %sB = "loadB"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf16, #shared1, #smem>
          %mma = ttng.tc_gen5_mma %sA, %sB, %res[%t1], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          %val, %t2 = ttng.tmem_load %res[%mma] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
          "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
          scf.yield {ttg.partition = array<i32: 0, 1>} %t2 : !ttg.async.token
        } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
        scf.yield {ttg.partition = array<i32: 0, 1>} %i : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %t0 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
      scf.yield {ttg.partition = array<i32: 0, 1>} %r : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
