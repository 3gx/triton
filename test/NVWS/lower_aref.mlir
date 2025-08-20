// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase  -canonicalize | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {

  //CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %2, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %3 = ttg.local_load %buffers_0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %4 = ttg.local_load %buffers_2 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

  //CHECK-LABEL: @three_consumers
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %2, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %3 = ttg.local_load %buffers_0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %4 = ttg.local_load %buffers_2 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      %20:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 3 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %30 = ttg.local_load %20#0 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %20#1 [#nvws.async_op<none>] {ttg.partition = 3 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_e"(%30) {ttg.partition = 3 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_f"(%30) {ttg.partition = 3 : i32} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32, 3 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

  //CHECK-LABEL: @complex_case
  tt.func @complex_case(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %3 = nvws.aref.create %2 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %4:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
      %buffers, %token = nvws.aref.put.enter %3[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %arg5, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %3[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %arg4, %buffers_0 {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %5 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %6 = ttg.local_load %buffers_2 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%6) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_4, %token_5 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %7 = ttg.local_load %buffers_4 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_5 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%7) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_c"(%7) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_6, %token_7 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %8 = ttg.local_load %buffers_6 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %3[%c0_i32], %token_7 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%8) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_8, %token_9 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %9 = ttg.local_load %buffers_8 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %3[%c0_i32], %token_9 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%9) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %5, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %2 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  //CHECK-LABEL: @reuse_argument
  tt.func @reuse_argument(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %2:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst_0) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %arg5, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %3 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %buffers_1, %token_2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %4 = ttg.local_load %buffers_1 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_2 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%4) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %buffers_3, %token_4 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %5 = ttg.local_load %buffers_3 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_4 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%5) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %3, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }


}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //CHECK-LABEL: @aref_lowering
  tt.func @aref_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32

    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %aref1 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    scf.for %i = %lb to %ub step %c1_i32 : i32{
      %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tma_load"(%1#0) {ttg.partition = 0 : i32}: (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
      "sts"(%1#1)  {ttg.partition = 0 : i32} : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tc5mma"(%2#0, %2#1) {ttg.partition = 1 : i32}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      scf.if %cond {
        %4:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tmem_store"(%4#0, %4#1) {ttg.partition = 0 : i32} : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.put.exit %aref1[%c0_i32], %4#2 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        %5:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tc5mma"(%5#0, %5#1) {ttg.partition = 1 : i32}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.get.exit %aref0[%c0_i32], %5#2 [#nvws.async_op<tc5mma>] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      }
      %6:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_store"(%6#0, %6#1) {ttg.partition = 0 : i32} : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.put.exit %aref1[%c0_i32], %6#2 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      %7:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_load"(%7#0, %7#1) {ttg.partition = 1 : i32} : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref1[%c0_i32], %7#2 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
    }
    tt.return
  }
}
