#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @matmul_change_desc_in_prologue(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = ub.poison : !tt.tensordesc<tensor<128x64xf16, #shared>>
    %1 = ub.poison : !tt.tensordesc<tensor<64x128xf16, #shared>>
    %2 = ub.poison : !ttg.async.token
    %3 = ub.poison : !ttg.async.token
    %4 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %5 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %6 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %7 = nvws.semaphore.acquire %5 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %8 = nvws.semaphore.buffer %5, %7 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %9 = ttng.tmem_store %cst, %8[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %11 = nvws.semaphore.create %10 true : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %12 = nvws.semaphore.create %10 false : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %14 = nvws.semaphore.create %13 true : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %15 = nvws.semaphore.create %13 false : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %16 = nvws.semaphore.acquire %11 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %17 = nvws.semaphore.acquire %14 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %18:6 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0, %arg5 = %1, %arg6 = %7, %arg7 = %16, %arg8 = %17) -> (i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %19 = "prologue_cond"(%arg2) {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (i32) -> i1
      %20:2 = scf.if %19 -> (!tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>) {
        %37 = tt.make_tensor_descriptor %arg0, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
        %38 = tt.make_tensor_descriptor %arg1, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
        scf.yield {ttg.partition = array<i32: 2>} %37, %38 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %arg4, %arg5 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %21:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %22 = nvws.semaphore.buffer %11, %arg7 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg4[%21#0, %21#2] 16384 %22 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %12, %arg7 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %23 = nvws.semaphore.buffer %14, %arg8 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg5[%21#1, %21#2] 16384 %23 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %15, %arg8 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %24 = nvws.semaphore.buffer %5, %arg6 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %25 = nvws.semaphore.acquire %12 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %26 = nvws.semaphore.buffer %12, %25 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %27 = nvws.semaphore.acquire %15 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %28 = nvws.semaphore.buffer %15, %27 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %29 = ttng.tc_gen5_mma %26, %28, %24[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %14, %27 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %11, %25 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %30 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %31 = arith.select %30, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %30 {
        nvws.semaphore.release %6, %arg6 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %32 = ub.poison : !ttg.async.token
      %33 = scf.if %30 -> (!ttg.async.token) {
        %37 = nvws.semaphore.acquire %6 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %38 = nvws.semaphore.buffer %6, %37 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %38[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %5, %37 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %32 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %32 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %34 = scf.if %30 -> (!ttg.async.token) {
        %37 = nvws.semaphore.acquire %5 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %37 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg6 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %35 = nvws.semaphore.acquire %11 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %36 = nvws.semaphore.acquire %14 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %31, %20#0, %20#1, %34, %35, %36 : i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = ub.poison : !ttg.async.token
    %1 = ub.poison : !ttg.async.token
    %2 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %4 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %5 = nvws.semaphore.acquire %3 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %3, %5 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %7 = ttng.tmem_store %cst, %6[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %9 = nvws.semaphore.create %8 true : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %10 = nvws.semaphore.create %8 false : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %12 = nvws.semaphore.create %11 true : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %13 = nvws.semaphore.create %11 false : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %14 = nvws.semaphore.acquire %9 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %15 = nvws.semaphore.acquire %12 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %16:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %5, %arg5 = %14, %arg6 = %15) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %17:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %18 = tt.splat %17#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      %19 = nvws.semaphore.buffer %9, %arg5 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_gather %arg0[%18, %17#2] 16384 %19 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %10, %arg5 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %20 = nvws.semaphore.buffer %12, %arg6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg1[%17#1, %17#2] 16384 %20 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %13, %arg6 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %21 = nvws.semaphore.buffer %3, %arg4 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %22 = nvws.semaphore.acquire %10 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %23 = nvws.semaphore.buffer %10, %22 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %24 = nvws.semaphore.acquire %13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %25 = nvws.semaphore.buffer %13, %24 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %26 = ttng.tc_gen5_mma %23, %25, %21[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %12, %24 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %9, %22 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %27 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %28 = arith.select %27, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %27 {
        nvws.semaphore.release %4, %arg4 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %29 = ub.poison : !ttg.async.token
      %30 = scf.if %27 -> (!ttg.async.token) {
        %34 = nvws.semaphore.acquire %4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %35 = nvws.semaphore.buffer %4, %34 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %35[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %3, %34 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %29 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %29 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %31 = scf.if %27 -> (!ttg.async.token) {
        %34 = nvws.semaphore.acquire %3 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %34 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg4 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %32 = nvws.semaphore.acquire %9 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.acquire %12 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %28, %31, %32, %33 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
  tt.func @matmul_tma_and_regular_load(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = ub.poison : !ttg.async.token
    %1 = ub.poison : !ttg.async.token
    %2 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = nvws.semaphore.create %result true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %4 = nvws.semaphore.create %result false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %5 = nvws.semaphore.acquire %3 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %3, %5 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %7 = ttng.tmem_store %cst, %6[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %9 = nvws.semaphore.create %8 true : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %10 = nvws.semaphore.create %8 false : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %11 = nvws.semaphore.acquire %9 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 2 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %12:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %5, %arg6 = %11) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token, !ttg.async.token)  : i32 {
      %13:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : (i32) -> (i32, tensor<64x128xi32, #blocked2>, i32)
      %14 = tt.splat %13#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      %15 = tt.addptr %arg4, %13#1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
      %16 = tt.load %15 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>
      %17 = nvws.semaphore.buffer %9, %arg6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_gather %arg0[%14, %13#2] 16384 %17 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %10, %arg6 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %18 = ttg.local_alloc %16 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %19 = nvws.semaphore.buffer %3, %arg5 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %20 = nvws.semaphore.acquire %10 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %21 = nvws.semaphore.buffer %10, %20 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %22 = ttng.tc_gen5_mma %21, %18, %19[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %9, %20 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %23 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %24 = arith.select %23, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %23 {
        nvws.semaphore.release %4, %arg5 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %25 = ub.poison : !ttg.async.token
      %26 = scf.if %23 -> (!ttg.async.token) {
        %29 = nvws.semaphore.acquire %4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %30 = nvws.semaphore.buffer %4, %29 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %30[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %3, %29 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %25 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %25 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %27 = scf.if %23 -> (!ttg.async.token) {
        %29 = nvws.semaphore.acquire %3 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %29 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg5 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %28 = nvws.semaphore.acquire %9 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %24, %15, %27, %28 : i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}

