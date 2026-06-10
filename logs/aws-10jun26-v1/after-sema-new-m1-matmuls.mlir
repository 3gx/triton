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
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %2 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %3 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %4 = nvws.semaphore.acquire %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %5 = nvws.semaphore.buffer %2, %4 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %6 = ttng.tmem_store %cst, %5[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %7 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %8 = nvws.semaphore.create %7 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %9 = nvws.semaphore.create %7 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %11 = nvws.semaphore.create %10 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %12 = nvws.semaphore.create %10 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %13 = nvws.semaphore.acquire %8 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %14 = nvws.semaphore.acquire %11 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %15:6 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0, %arg5 = %1, %arg6 = %4, %arg7 = %13, %arg8 = %14) -> (i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %16 = "prologue_cond"(%arg2) {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (i32) -> i1
      %17:2 = scf.if %16 -> (!tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>) {
        %34 = tt.make_tensor_descriptor %arg0, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
        %35 = tt.make_tensor_descriptor %arg1, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
        scf.yield {ttg.partition = array<i32: 2>} %34, %35 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %arg4, %arg5 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %18:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %19 = nvws.semaphore.buffer %8, %arg7 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg4[%18#0, %18#2] 16384 %19 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %9, %arg7 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %20 = nvws.semaphore.buffer %11, %arg8 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg5[%18#1, %18#2] 16384 %20 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %12, %arg8 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %21 = nvws.semaphore.buffer %2, %arg6 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %22 = nvws.semaphore.acquire %9 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %23 = nvws.semaphore.buffer %9, %22 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %24 = nvws.semaphore.acquire %12 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %25 = nvws.semaphore.buffer %12, %24 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %26 = ttng.tc_gen5_mma %23, %25, %21[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %11, %24 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %8, %22 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %27 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %28 = arith.select %27, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %27 {
        nvws.semaphore.release %3, %arg6 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %29 = ub.poison : !ttg.async.token
      %30 = scf.if %27 -> (!ttg.async.token) {
        %34 = nvws.semaphore.acquire %3 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %35 = nvws.semaphore.buffer %3, %34 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %35[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %2, %34 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %29 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %29 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %31 = scf.if %27 -> (!ttg.async.token) {
        %34 = nvws.semaphore.acquire %2 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %34 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg6 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %32 = nvws.semaphore.acquire %8 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %33 = nvws.semaphore.acquire %11 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %28, %17#0, %17#1, %31, %32, %33 : i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token, !ttg.async.token, !ttg.async.token
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
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %1 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %4 = ttng.tmem_store %cst, %3[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %6 = nvws.semaphore.create %5 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %7 = nvws.semaphore.create %5 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %9 = nvws.semaphore.create %8 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %10 = nvws.semaphore.create %8 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>
    %11 = nvws.semaphore.acquire %6 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %12 = nvws.semaphore.acquire %9 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %13:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %2, %arg5 = %11, %arg6 = %12) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %14:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %15 = tt.splat %14#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      %16 = nvws.semaphore.buffer %6, %arg5 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_gather %arg0[%15, %14#2] 16384 %16 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %7, %arg5 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %17 = nvws.semaphore.buffer %9, %arg6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg1[%14#1, %14#2] 16384 %17 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %10, %arg6 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %18 = nvws.semaphore.buffer %0, %arg4 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %19 = nvws.semaphore.acquire %7 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %20 = nvws.semaphore.buffer %7, %19 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %21 = nvws.semaphore.acquire %10 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %22 = nvws.semaphore.buffer %10, %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %23 = ttng.tc_gen5_mma %20, %22, %18[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %9, %21 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %6, %19 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %24 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %25 = arith.select %24, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %24 {
        nvws.semaphore.release %1, %arg4 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %26 = ub.poison : !ttg.async.token
      %27 = scf.if %24 -> (!ttg.async.token) {
        %31 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %32 = nvws.semaphore.buffer %1, %31 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %32[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %0, %31 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %26 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %26 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %28 = scf.if %24 -> (!ttg.async.token) {
        %31 = nvws.semaphore.acquire %0 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %31 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg4 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %29 = nvws.semaphore.acquire %6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %30 = nvws.semaphore.acquire %9 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %25, %28, %29, %30 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token
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
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %1 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %2 = nvws.semaphore.acquire %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %3 = nvws.semaphore.buffer %0, %2 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %4 = ttng.tmem_store %cst, %3[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %6 = nvws.semaphore.create %5 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %7 = nvws.semaphore.create %5 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %8 = nvws.semaphore.acquire %6 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 2 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %9:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %2, %arg6 = %8) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token, !ttg.async.token)  : i32 {
      %10:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : (i32) -> (i32, tensor<64x128xi32, #blocked2>, i32)
      %11 = tt.splat %10#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      %12 = tt.addptr %arg4, %10#1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
      %13 = tt.load %12 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>
      %14 = nvws.semaphore.buffer %6, %arg6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_gather %arg0[%11, %10#2] 16384 %14 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %7, %arg6 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %15 = ttg.local_alloc %13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %16 = nvws.semaphore.buffer %0, %arg5 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %17 = nvws.semaphore.acquire %7 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %18 = nvws.semaphore.buffer %7, %17 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %19 = ttng.tc_gen5_mma %18, %15, %16[], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %6, %17 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %20 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %21 = arith.select %20, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %20 {
        nvws.semaphore.release %1, %arg5 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %22 = ub.poison : !ttg.async.token
      %23 = scf.if %20 -> (!ttg.async.token) {
        %26 = nvws.semaphore.acquire %1 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %27 = nvws.semaphore.buffer %1, %26 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_0, %token = ttng.tmem_load %27[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %0, %26 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %22 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %22 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      %24 = scf.if %20 -> (!ttg.async.token) {
        %26 = nvws.semaphore.acquire %0 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %26 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg5 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %25 = nvws.semaphore.acquire %6 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %21, %12, %24, %25 : i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}

