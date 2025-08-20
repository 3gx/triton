// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {

  //CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2 = "op_a"() : () -> tensor<1xi32, #blocked>
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 3 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %2, %3 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      }
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 3 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 4 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2 = "op_a"() : () -> tensor<1xi32, #blocked>
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %2, %3 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      }
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition3 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
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

    nvws.warp_group
    partition0  num_warps(4) {
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token

        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

        scf.if %cond {


          %2:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
          "tmem_store"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

          nvws.aref.put.exit %aref1[%c0_i32], %2#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

        }
      }

      scf.if %cond {

        %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      }

      %1:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_store"(%1#0, %1#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.put.exit %aref1[%c0_i32], %1#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      nvws.warp_group.return
    }
    partition1 num_warps(8) {
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

        scf.if %cond {
          %3:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
          "tmem_load"(%3#0, %3#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
          nvws.aref.get.exit %aref1[%c0_i32], %3#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        }
      }
      scf.if %cond {
        %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      }

      %2:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_load"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref1[%c0_i32], %2#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      nvws.warp_group.return
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  //CHECK-LABEL: @complex_case
  tt.func @complex_case(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %3 = nvws.aref.create %2 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    nvws.warp_group
    partition0 num_warps(4) {
      %5:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
        %6, %token6 = nvws.aref.put.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %arg5, %6 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %3[%c0_i32], %token6 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %7, %token7 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %arg4, %7 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token7 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %8 = "op_a"() : () -> tensor<1xi32, #blocked>
        scf.yield %8, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
      }
      nvws.warp_group.yield %5#0, %5#1 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%6) : (tensor<1xi32, #blocked>) -> ()
        %7:2 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %3[%c0_i32], %7#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%6) : (tensor<1xi32, #blocked>) -> ()
        "op_c"(%6) : (tensor<1xi32, #blocked>) -> ()
        %7:2 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %3[%c0_i32], %7#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    ttg.local_dealloc %2 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_argument(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    nvws.warp_group
    partition0 num_warps(4) {
      %2:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst_0) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %arg5, %3 {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %4 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
        scf.yield %4, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.yield %2#0, %2#1 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%6) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %7:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %7#1 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.return
    }
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
