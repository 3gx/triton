// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect \
// RUN:   --nvws-lower-aref-to-semaphore \
// RUN:   --nvws-assign-semaphore-stage-phase \
// RUN:   --mlir-print-op-generic | FileCheck %s --check-prefix=ASSIGN
// DEBUG: triton-opt %s -split-input-file --allow-unregistered-dialect \
// DEBUG:   --nvws-lower-aref-to-semaphore \
// DEBUG:   --nvws-assign-semaphore-stage-phase \
// DEBUG:   --nvws-lower-semaphore \
// DEBUG:   -tritongpu-partition-loops

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // ASSIGN-LABEL: sym_name = "partition_loops_ws_tag_regression"
  // ASSIGN: "arith.constant"() <{value = false}> {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
  // ASSIGN: "arith.constant"() <{value = true}> {ttg.partition = array<i32: 0, 1>} : () -> i1
  // ASSIGN-NOT: "arith.constant"() <{value = true}> {ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
  // ASSIGN: "scf.if"(%{{.*}})
  // ASSIGN: "scf.yield"(%{{.*}}) {ttg.partition = array<i32: 0, 1>}
  // ASSIGN-NOT: "arith.constant"() <{value = true}> {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  tt.func @partition_loops_ws_tag_regression(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %aref = nvws.aref.create %buf : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    scf.for %iv = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !elt
      %pbuf, %ptok = nvws.aref.put.enter %aref[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
      ttg.local_store %v, %pbuf {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      nvws.aref.put.exit %aref[%c0_i32], %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %g1buf, %g1tok = nvws.aref.get.enter %aref[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
      %g1 = ttg.local_load %g1buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      scf.if %cond {
        "consumer1"(%g1) {ttg.partition = array<i32: 1>} : (!elt) -> ()
      } {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %aref[%c0_i32], %g1tok [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1>}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
