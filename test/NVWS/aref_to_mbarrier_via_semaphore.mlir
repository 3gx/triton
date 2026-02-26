// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect \
// RUN:   --nvws-lower-aref-to-semaphore \
// RUN:   --nvws-assign-semaphore-stage-phase \
// RUN:   --nvws-lower-semaphore | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers_e2e
  // CHECK-COUNT-3: ttng.init_barrier {{.*}}, 2
  // CHECK-COUNT-3: ttng.init_barrier {{.*}}, 1
  // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 0>
  // CHECK: ttng.arrive_barrier {{.*}} {ttg.partition = array<i32: 0>
  // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 1>
  // CHECK: ttng.arrive_barrier {{.*}} {ttg.partition = array<i32: 1>
  // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 2>
  // CHECK: ttng.arrive_barrier {{.*}} {ttg.partition = array<i32: 2>
  // CHECK: ttng.inval_barrier
  // CHECK: ttg.local_dealloc
  tt.func @two_consumers_e2e(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %arg3 = %arg0 to %arg1 step %arg2 : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      %pbuf, %ptok = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %v, %pbuf {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %c1buf, %c1tok = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %c1 = ttg.local_load %c1buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %c1tok [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer1"(%c1) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      %c2buf, %c2tok = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %c2 = ttg.local_load %c2buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %c2tok [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "consumer2"(%c2) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @aref_not_in_loop_e2e
  // CHECK: ttng.tmem_alloc
  // CHECK: ttng.tmem_store
  // CHECK: ttng.tc_gen5_mma
  // CHECK: ttng.tc_gen5_commit
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.tmem_load
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.inval_barrier
  tt.func @aref_not_in_loop_e2e(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %aref = nvws.aref.create %result : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %pbuf, %ptok = nvws.aref.put.enter %aref : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %pview = nvws.aref.buffer %aref, %ptok : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %init = ttng.tmem_store %cst, %pview[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    scf.for %iv = %c0_i32 to %arg0 step %c1_i32 : i32 {
      %off = arith.muli %iv, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %a = tt.descriptor_load %arg3[%arg1, %off] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %b = tt.descriptor_load %arg4[%arg2, %off] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %ab = ttg.local_alloc %a {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %bb = ttg.local_alloc %b {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %bt = ttg.memdesc_trans %bb {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %acc = nvws.aref.buffer %aref, %ptok {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %mma = ttng.tc_gen5_mma %ab, %bt, %acc[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    nvws.aref.put.exit %aref, %ptok [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %gbuf, %gtok = nvws.aref.get.enter %aref : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %gview = nvws.aref.buffer %aref, %gtok : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %out, %tok = ttng.tmem_load %gview[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %aref, %gtok [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
