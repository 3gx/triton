// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @assign_stage_basic(%arg0: i32, %arg1: i32, %arg2: i32) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.semaphore.create %0 true : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
      %3 = nvws.semaphore.buffer %1, %2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %4 = ttg.local_load %3 {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
      ttg.local_store %4, %3 {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      nvws.semaphore.release %1, %2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @shared_stage_two_semaphores
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @shared_stage_two_semaphores() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.semaphore.create %0 true : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    %2 = nvws.semaphore.create %0 false : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    %3 = nvws.semaphore.acquire %1 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %4 = nvws.semaphore.acquire %2 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %5 = nvws.semaphore.buffer %1, %3 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    ttg.local_store %cst, %5 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    nvws.semaphore.release %1, %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    nvws.semaphore.release %2, %4 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @if_observation
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @if_observation(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.semaphore.create %0 true : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    scf.for %arg4 = %arg1 to %arg2 step %arg3  : i32 {
      %2 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
      %3 = nvws.semaphore.buffer %1, %2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      scf.if %arg0 {
        %4 = ttg.local_load %3 {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        "use"(%4) {ttg.partition = array<i32: 0>} : (tensor<1xi32, #blocked>) -> ()
      } {ttg.partition = array<i32: 0>}
      %cst = arith.constant {ttg.partition = array<i32: 0>} dense<0> : tensor<1xi32, #blocked>
      ttg.local_store %cst, %3 {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      nvws.semaphore.release %1, %2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.semaphore.create %0 true : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    %2 = nvws.semaphore.create %0 false : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %3 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      %4 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %5 = nvws.semaphore.buffer %1, %4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      ttg.local_store %3, %5 {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.semaphore.release %2, %4 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      %6 = nvws.semaphore.acquire %2 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %7 = nvws.semaphore.buffer %2, %6 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %8 = ttg.local_load %7 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %1, %6 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_b"(%8) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()
      %9 = nvws.semaphore.acquire %2 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %10 = nvws.semaphore.buffer %2, %9 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      %11 = ttg.local_load %10 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.semaphore.release %1, %9 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_c"(%11) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%11) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:0", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @aref_lowering
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @aref_lowering(%arg0: !ttg.memdesc<3x64x16xf16, #shared, #smem>, %arg1: !ttg.memdesc<3x16x32xf16, #shared, #smem>, %arg2: !ttg.memdesc<3x64x16xf16, #shared, #smem>, %arg3: !ttg.memdesc<3x16x32xf16, #shared, #smem>, %arg4: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = nvws.semaphore.create %arg0, %arg1 true : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>
    %1 = nvws.semaphore.create %arg0, %arg1 false : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>
    %2 = nvws.semaphore.create %arg2, %arg3 true : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>
    %3 = nvws.semaphore.create %arg2, %arg3 false : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>
    scf.for %arg5 = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %4 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3> -> !ttg.async.token
      %5:2 = nvws.semaphore.buffer %0, %4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
      "op1"(%5#0) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared, #smem, mutable>) -> ()
      "op2"(%5#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<16x32xf16, #shared, #smem, mutable>) -> ()
      nvws.semaphore.release %1, %4 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token
      %6 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3> -> !ttg.async.token
      %7:2 = nvws.semaphore.buffer %1, %6 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
      "op3"(%7#0, %7#1) {ttg.partition = array<i32: 1>} : (!ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>) -> ()
      nvws.semaphore.release %0, %6 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token
      scf.if %arg4 {
      } else {
        %8 = nvws.semaphore.acquire %2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3> -> !ttg.async.token
        %9:2 = nvws.semaphore.buffer %2, %8 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
        "op4"(%9#0, %9#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>) -> ()
        nvws.semaphore.release %3, %8 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token
        %10 = nvws.semaphore.acquire %3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3> -> !ttg.async.token
        %11:2 = nvws.semaphore.buffer %3, %10 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
        "op5"(%11#0, %11#1) {ttg.partition = array<i32: 1>} : (!ttg.memdesc<64x16xf16, #shared, #smem, mutable>, !ttg.memdesc<16x32xf16, #shared, #smem, mutable>) -> ()
        nvws.semaphore.release %2, %10 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x64x16xf16, #shared, #smem>, !ttg.memdesc<3x16x32xf16, #shared, #smem>], 3>, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>}
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.warp_specialize.tag = 0 : i32}
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
  // CHECK-LABEL: @warp_specialize_tma_matmul
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %4 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %5 = ttng.tmem_store %cst, %4[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %9 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %10 = tt.descriptor_load %arg3[%arg1, %9] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %11 = tt.descriptor_load %arg4[%arg2, %9] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %12 = ttg.local_alloc %10 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %13 = ttg.local_alloc %11 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %14 = ttg.memdesc_trans %13 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %15 = nvws.semaphore.buffer %0, %2 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %16 = ttng.tc_gen5_mma %12, %14, %15[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.semaphore.release %1, %2 [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    %6 = nvws.semaphore.acquire %1 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %7 = nvws.semaphore.buffer %1, %6 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %8 = nvws.semaphore.buffer %1, %6 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_0, %token = ttng.tmem_load %8[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    nvws.semaphore.release %0, %6 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %4 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %5 = ttng.tmem_store %cst_0, %4[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %6 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %2) -> (!ttg.async.token)  : i32 {
      %7:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %8 = tt.descriptor_load %arg0[%7#0, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %9 = tt.descriptor_load %arg1[%7#1, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %10 = ttg.local_alloc %8 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %11 = ttg.local_alloc %9 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %12 = nvws.semaphore.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %13 = ttng.tc_gen5_mma %10, %11, %12[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      nvws.semaphore.release %1, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
      %14 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
      %15 = nvws.semaphore.buffer %1, %14 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %16 = nvws.semaphore.buffer %1, %14 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_1, %token = ttng.tmem_load %16[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %0, %14 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
      "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      %17 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
      %18 = nvws.semaphore.buffer %0, %17 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %19 = nvws.semaphore.buffer %0, %17 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %20 = ttng.tmem_store %cst, %19[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %17 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    nvws.semaphore.release %1, %6 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @assign_stage_buffer
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @assign_stage_buffer(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %4 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %5 = ttng.tmem_store %cst_0, %4[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %6 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %2) -> (!ttg.async.token)  : i32 {
      %7:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %8 = tt.descriptor_load %arg0[%7#0, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %9 = tt.descriptor_load %arg1[%7#1, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %10 = ttg.local_alloc %8 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %11 = ttg.local_alloc %9 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %12 = nvws.semaphore.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %13 = ttng.tc_gen5_mma %10, %11, %12[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %15 = scf.if %14 -> (!ttg.async.token) {
        nvws.semaphore.release %1, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
        %18 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
        %19 = nvws.semaphore.buffer %1, %18 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %20 = nvws.semaphore.buffer %1, %18 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_1, %token = ttng.tmem_load %20[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %0, %18 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
        "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %21 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
        %22 = nvws.semaphore.buffer %0, %21 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        scf.yield %21 : !ttg.async.token
      } else {
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %16 = nvws.semaphore.buffer %0, %15 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %17 = ttng.tmem_store %cst, %16[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %15 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    nvws.semaphore.release %1, %6 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %4 = nvws.semaphore.create %result_2 true : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %5 = nvws.semaphore.create %result_2 false : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %6 = nvws.semaphore.acquire %4 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %7 = nvws.semaphore.buffer %4, %6 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %8 = nvws.semaphore.buffer %4, %6 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %9 = ttng.tmem_store %cst_0, %8[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %10 = nvws.semaphore.create %result_3 true : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>
    %11 = nvws.semaphore.create %result_3 false : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>
    %12:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %2, %arg9 = %6) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %16 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %17 = ttg.local_alloc %16 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %18 = ttg.memdesc_trans %17 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %19 = nvws.semaphore.buffer %0, %arg8 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %20 = ttng.tc_gen5_mma %arg0, %18, %19[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      nvws.semaphore.release %1, %arg8 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
      %21 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
      %22 = nvws.semaphore.buffer %1, %21 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %23 = nvws.semaphore.buffer %1, %21 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %result_5, %token_6 = ttng.tmem_load %23[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.semaphore.release %0, %21 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
      %24 = "compute_row_max"(%result_5, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %25 = "sub_row_max"(%result_5, %24, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %26 = math.exp2 %25 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %27 = arith.subf %arg7, %24 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %28 = arith.subf %arg7, %24 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %29 = math.exp2 %27 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %30 = math.exp2 %28 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %31 = "tt.reduce"(%26) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %57 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %57 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %32 = arith.mulf %arg6, %30 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %33 = arith.addf %32, %31 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %34 = tt.expand_dims %29 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %35 = tt.broadcast %34 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %36 = nvws.semaphore.buffer %4, %arg9 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %result_7, %token_8 = ttng.tmem_load %36[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %37 = arith.mulf %result_7, %35 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %38 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %39 = ttg.local_alloc %38 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %40 = arith.truncf %26 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %41 = nvws.semaphore.acquire %10 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %42 = nvws.semaphore.buffer %10, %41 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %43 = nvws.semaphore.buffer %10, %41 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %44 = ttng.tmem_store %40, %43[%41], %true {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %11, %41 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %45 = ttng.tmem_store %37, %36[], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %5, %arg9 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %46 = nvws.semaphore.acquire %5 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %47 = nvws.semaphore.buffer %5, %46 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %48 = nvws.semaphore.buffer %5, %46 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %49 = nvws.semaphore.acquire %11 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %50 = nvws.semaphore.buffer %11, %49 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %51 = nvws.semaphore.buffer %11, %49 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %52 = ttng.tc_gen5_mma %51, %39, %48[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %10, %49 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      nvws.semaphore.release %4, %46 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %53 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
      %54 = nvws.semaphore.buffer %0, %53 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %55 = nvws.semaphore.acquire %4 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %56 = nvws.semaphore.buffer %4, %55 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      scf.yield %33, %24, %53, %55 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.semaphore.release %5, %12#3 [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    nvws.semaphore.release %1, %12#2 [#nvws.async_op<none>] : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
    %13 = nvws.semaphore.acquire %5 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %14 = nvws.semaphore.buffer %5, %13 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %15 = nvws.semaphore.buffer %5, %13 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_4, %token = ttng.tmem_load %15[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    nvws.semaphore.release %4, %13 [#nvws.async_op<none>] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    "use"(%12#0, %result_4, %12#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %4 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %5 = ttng.tmem_store %cst_0, %4[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %6 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %2) -> (!ttg.async.token)  : i32 {
      %7:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %8 = tt.descriptor_load %arg0[%7#0, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %9 = tt.descriptor_load %arg1[%7#1, %7#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %10 = ttg.local_alloc %8 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %11 = ttg.local_alloc %9 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %12 = nvws.semaphore.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %13 = ttng.tc_gen5_mma %10, %11, %12[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      %15 = scf.if %14 -> (!ttg.async.token) {
        nvws.semaphore.release %1, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
        %18 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
        %19 = nvws.semaphore.buffer %1, %18 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %20 = nvws.semaphore.buffer %1, %18 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_1, %token = ttng.tmem_load %20[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %0, %18 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
        "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        %21 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2> -> !ttg.async.token
        %22 = nvws.semaphore.buffer %0, %21 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        scf.yield %21 : !ttg.async.token
      } else {
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %16 = nvws.semaphore.buffer %0, %15 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %17 = ttng.tmem_store %cst, %16[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %15 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    nvws.semaphore.release %1, %6 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 2>, !ttg.async.token
    tt.return
  }
}
// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_tma_persistent_ws_kernel
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func public @matmul_tma_persistent_ws_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg6, %arg8], [%0, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %2 = arith.extsi %arg4 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg7, %arg8], [%2, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %4 = arith.extsi %arg5 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg6, %arg7], [%4, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg6, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.addi %arg8, %c127_i32 : i32
    %12 = arith.divsi %11, %c128_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>
    %17 = nvws.semaphore.create %15, %16 true : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>
    %18 = nvws.semaphore.create %15, %16 false : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %19 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %20 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    scf.for %arg9 = %6 to %13 step %c148_i32  : i32 {
      %21 = arith.divsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %22 = arith.muli %21, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %23 = arith.subi %8, %22 {ttg.partition = array<i32: 0, 2>} : i32
      %24 = arith.minsi %23, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %25 = arith.remsi %arg9, %24 {ttg.partition = array<i32: 0, 2>} : i32
      %26 = arith.addi %22, %25 {ttg.partition = array<i32: 0, 2>} : i32
      %27 = arith.remsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %28 = arith.divsi %27, %24 {ttg.partition = array<i32: 0, 2>} : i32
      %29 = arith.muli %26, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %30 = arith.muli %28, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %31 = nvws.semaphore.acquire %19 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %32 = nvws.semaphore.buffer %19, %31 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %33 = nvws.semaphore.buffer %19, %31 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %34 = ttng.tmem_store %cst, %33[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %20, %31 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %35 = nvws.semaphore.acquire %20 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %36 = nvws.semaphore.buffer %20, %35 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %37 = scf.for %arg10 = %c0_i32 to %12 step %c1_i32 iter_args(%arg11 = %false) -> (i1)  : i32 {
        %45 = arith.muli %arg10, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %46 = nvws.semaphore.acquire %17 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3> -> !ttg.async.token
        %47:2 = nvws.semaphore.buffer %17, %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        nvws.descriptor_load %1[%29, %45] 16384 %47#0 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        nvws.descriptor_load %3[%30, %45] 16384 %47#1 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        nvws.semaphore.release %18, %46 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>, !ttg.async.token
        %48 = nvws.semaphore.acquire %18 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3> -> !ttg.async.token
        %49:2 = nvws.semaphore.buffer %18, %48 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        %50 = ttg.memdesc_trans %49#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>
        %51 = nvws.semaphore.buffer %20, %35 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %52 = ttng.tc_gen5_mma %49#0, %50, %51[], %arg11, %true {is_async, loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable, 1x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %17, %48 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>, !ttg.memdesc<3x128x128xf8E4M3FN, #shared, #smem, mutable>], 3>, !ttg.async.token
        scf.yield %true : i1
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      nvws.semaphore.release %19, %35 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %38 = nvws.semaphore.acquire %19 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %39 = nvws.semaphore.buffer %19, %38 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %40 = nvws.semaphore.buffer %19, %38 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_0, %token = ttng.tmem_load %40[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      nvws.semaphore.release %20, %38 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %41 = nvws.semaphore.acquire %20 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %42 = nvws.semaphore.buffer %20, %41 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %19, %41 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %43 = tt.fp_to_fp %result_0 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %44 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %5[%29, %30], %44 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
// -----
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @for_loop_control_operand_ppg
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: arith.shli
  // CHECK: arith.xori
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  // CHECK: nvws.semaphore.acquire %{{.*}}[%{{.*}}, %{{.*}}]
  // CHECK: nvws.semaphore.release %{{.*}}[%{{.*}}]
  tt.func @for_loop_control_operand_ppg(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.ptr<i32>) {
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %1 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %4 = scf.for %arg4 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %2) -> (!ttg.async.token)  : i32 {
      %7 = tt.addptr %arg3, %arg4 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %8 = tt.load %7 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %9 = "lb1"(%arg4) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %10 = "step1"(%arg4) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %11 = scf.for %arg6 = %9 to %8 step %10 iter_args(%arg7 = %arg5) -> (!ttg.async.token)  : i32 {
        %16 = "load1"(%arg6) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %17 = "load2"(%arg6) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %18 = nvws.semaphore.buffer %0, %arg7 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        ttng.tc_gen5_mma %16, %17, %18, %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %arg7 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      nvws.semaphore.release %1, %11 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %12 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %13 = nvws.semaphore.buffer %1, %12 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      nvws.semaphore.release %0, %12 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
      %14 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
      %15 = nvws.semaphore.buffer %0, %14 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1, 2>} %14 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
    nvws.semaphore.release %1, %4 [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    %5 = nvws.semaphore.acquire %1 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %1, %5 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    nvws.semaphore.release %0, %5 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>], 1>, !ttg.async.token
    tt.return
  }
}
