// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore | FileCheck %s
// RUN: triton-opt %S/lower_aref.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore | FileCheck %s --check-prefix=L2S-AREF --implicit-check-not=nvws.aref.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @basic
  // CHECK: %[[SEM0:.*]] = nvws.semaphore.create{{.*}} true
  // CHECK: %[[SEM1:.*]] = nvws.semaphore.create{{.*}} false
  // CHECK-NOT: nvws.aref.
  tt.func @basic(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %aref = nvws.aref.create %buf : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: %[[TOKP:.*]] = nvws.semaphore.acquire %[[SEM0]]
      // CHECK-NOT: [%
      // CHECK: %[[PB0:.*]] = nvws.semaphore.buffer %[[SEM0]], %[[TOKP]]
      // CHECK-NOT: [%
      %pb, %ptok = nvws.aref.put.enter %aref {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      // CHECK: %[[PB1:.*]] = nvws.semaphore.buffer %[[SEM0]], %[[TOKP]]
      %pb_from_tok = nvws.aref.buffer %aref, %ptok : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      ttg.local_store %v, %pb_from_tok {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      // CHECK: nvws.semaphore.release %[[SEM1]], %[[TOKP]] [#nvws.async_op<none>]
      nvws.aref.put.exit %aref, %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: %[[TOKG0:.*]] = nvws.semaphore.acquire %[[SEM1]]
      %gb0, %gtok0 = nvws.aref.get.enter %aref {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %x = ttg.local_load %gb0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> !ty
      // CHECK: nvws.semaphore.release %[[SEM0]], %[[TOKG0]] [#nvws.async_op<none>]
      nvws.aref.get.exit %aref, %gtok0 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer0"(%x) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // CHECK: %[[TOKG1:.*]] = nvws.semaphore.acquire %[[SEM1]]
      %gb1, %gtok1 = nvws.aref.get.enter %aref {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %y = ttg.local_load %gb1 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> !ty
      // CHECK: nvws.semaphore.release %[[SEM0]], %[[TOKG1]] [#nvws.async_op<none>]
      nvws.aref.get.exit %aref, %gtok1 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer1"(%y) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    }
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tma_preserved
  // CHECK: %[[SEM0:.*]] = nvws.semaphore.create{{.*}} true
  // CHECK: %[[SEM1:.*]] = nvws.semaphore.create{{.*}} false
  // CHECK: %[[TOK0:.*]] = nvws.semaphore.acquire %[[SEM0]]
  // CHECK-NOT: [%
  // CHECK: nvws.semaphore.buffer %[[SEM0]], %[[TOK0]]
  // CHECK-NOT: [%
  // CHECK: nvws.descriptor_load
  // CHECK: nvws.semaphore.release %[[SEM1]], %[[TOK0]] [#nvws.async_op<tma_load>]
  tt.func @tma_preserved(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %idx: i32) {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %aref = nvws.aref.create %buf : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>

    %pb, %ptok = nvws.aref.put.enter %aref {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.async.token
    nvws.descriptor_load %desc[%idx, %c0_i32] 16384 %pb {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    nvws.aref.put.exit %aref, %ptok [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token

    %gb, %gtok = nvws.aref.get.enter %aref {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.async.token
    %v = ttg.local_load %gb {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
    nvws.aref.get.exit %aref, %gtok [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    "use"(%v) : (tensor<128x64xf16, #blocked>) -> ()

    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// L2S-AREF-LABEL: @two_consumers
// L2S-AREF-DAG: nvws.semaphore.create
// L2S-AREF-DAG: nvws.semaphore.acquire
// L2S-AREF-DAG: nvws.semaphore.buffer
// L2S-AREF-DAG: nvws.semaphore.release
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @three_consumers
// L2S-AREF-DAG: "op_e"
// L2S-AREF-DAG: "op_f"
// L2S-AREF-DAG: nvws.semaphore.acquire
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @reuse_argument
// L2S-AREF-DAG: "op_a"
// L2S-AREF-DAG: "op_d"
// L2S-AREF-DAG: nvws.semaphore.acquire
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @warp_specialize_tma_matmul
// L2S-AREF-DAG: nvws.descriptor_load
// L2S-AREF-DAG: ttng.tc_gen5_mma
// L2S-AREF-DAG: [#nvws.async_op<tma_load>]
// L2S-AREF-DAG: [#nvws.async_op<tc5mma>]
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @load_used_as_reg_and_smem
// L2S-AREF-DAG: "use1"
// L2S-AREF-DAG: "use2"
// L2S-AREF-DAG: nvws.descriptor_load
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @load_used_as_reg_and_smem_same_partition
// L2S-AREF-DAG: "use1"
// L2S-AREF-DAG: "use2"
// L2S-AREF-DAG: nvws.semaphore.acquire
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @lower_aref_buffer
// L2S-AREF-DAG: scf.if
// L2S-AREF-DAG: ttng.tmem_load
// L2S-AREF-DAG: nvws.semaphore.release
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @aref_not_in_loop
// L2S-AREF-DAG: ttng.tc_gen5_mma
// L2S-AREF-DAG: [#nvws.async_op<tc5mma>]
// L2S-AREF-DAG: nvws.semaphore.acquire
// L2S-AREF: tt.return

// L2S-AREF-LABEL: @load_scale_mma_user
// L2S-AREF-DAG: ttng.tc_gen5_mma_scaled
// L2S-AREF-DAG: ttng.tmem_alloc
// L2S-AREF-DAG: ttng.tmem_load
// L2S-AREF: tt.return
