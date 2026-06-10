#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32, %arg5: !tt.ptr<f32>) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %1 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %2 = nvws.semaphore.create %result_2 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %3 = nvws.semaphore.create %result_2 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %4 = nvws.semaphore.create %result_2 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %5 = nvws.semaphore.acquire %2 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %6 = nvws.semaphore.buffer %2, %5 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %7 = ttng.tmem_store %cst_0, %6[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x256xf32, #shared1, #smem, mutable>
    %9 = nvws.semaphore.create %8 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %10 = nvws.semaphore.create %8 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %12 = nvws.semaphore.create %11 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %13 = nvws.semaphore.create %11 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %16 = nvws.semaphore.create %15 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %17 = nvws.semaphore.create %15 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %18 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %19 = nvws.semaphore.create %18 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>
    %20 = nvws.semaphore.create %18 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>
    %21 = ttg.local_alloc : () -> !ttg.memdesc<1x256xf32, #shared1, #smem, mutable>
    %22 = nvws.semaphore.create %21 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %23 = nvws.semaphore.create %21 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %24 = nvws.semaphore.acquire %0 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %25 = nvws.semaphore.acquire %9 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %26 = nvws.semaphore.acquire %12 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %27 = nvws.semaphore.acquire %16 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %28 = nvws.semaphore.acquire %19 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %29 = nvws.semaphore.acquire %22 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %30:9 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %24, %arg10 = %5, %arg11 = %25, %arg12 = %26, %arg13 = %27, %arg14 = %28, %arg15 = %29) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %33 = nvws.semaphore.buffer %9, %arg11 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      ttg.local_store %arg8, %33 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %10, %arg11 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %34 = nvws.semaphore.buffer %12, %arg12 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg1[%arg6, %c0_i32] 8192 %34 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %13, %arg12 [#nvws.async_op<tma_load>] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %35 = ttg.memdesc_trans %14 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %36 = nvws.semaphore.buffer %0, %arg9 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %37 = nvws.semaphore.acquire %13 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %38 = nvws.semaphore.buffer %13, %37 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %39 = ttg.memdesc_trans %38 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %40 = ttng.tc_gen5_mma %arg0, %39, %36[], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      nvws.semaphore.release %12, %37 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %1, %arg9 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %41 = nvws.semaphore.acquire %1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %42 = nvws.semaphore.buffer %1, %41 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %result_4, %token_5 = ttng.tmem_load %42[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.semaphore.release %0, %41 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %43 = "compute_row_max"(%result_4, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %44 = nvws.semaphore.buffer %22, %arg15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      ttg.local_store %43, %44 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %23, %arg15 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %45 = "sub_row_max"(%result_4, %43, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %46 = math.exp2 %45 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %47 = arith.subf %arg8, %43 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %48 = nvws.semaphore.acquire %10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %49 = nvws.semaphore.buffer %10, %48 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %50 = ttg.local_load %49 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      nvws.semaphore.release %9, %48 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %51 = nvws.semaphore.acquire %23 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %52 = nvws.semaphore.buffer %23, %51 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %53 = ttg.local_load %52 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      nvws.semaphore.release %22, %51 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %54 = arith.subf %50, %53 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %55 = math.exp2 %47 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %56 = math.exp2 %54 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %57 = "tt.reduce"(%46) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %86 = arith.addf %arg16, %arg17 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %86 {ttg.partition = array<i32: 0>} : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %58 = arith.mulf %arg7, %55 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %59 = arith.addf %58, %57 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %60 = tt.expand_dims %56 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %61 = tt.broadcast %60 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %62 = tt.addptr %arg5, %arg6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>, i32
      %63 = tt.load %62 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>
      %64 = tt.splat %63 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : f32 -> tensor<256x64xf32, #blocked>
      %65 = nvws.semaphore.buffer %2, %arg10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %result_6, %token_7 = ttng.tmem_load %65[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %66 = arith.mulf %result_6, %61 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      %67 = arith.addf %66, %64 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      %68 = nvws.semaphore.buffer %16, %arg13 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg2[%arg6, %c0_i32] 8192 %68 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %17, %arg13 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %69 = arith.truncf %46 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %70 = nvws.semaphore.buffer %19, %arg14 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.local_store %69, %70 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %20, %arg14 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %71 = ttng.tmem_store %67, %65[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %3, %arg10 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %72 = nvws.semaphore.acquire %3 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %73 = nvws.semaphore.buffer %3, %72 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %74 = nvws.semaphore.acquire %17 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %75 = nvws.semaphore.buffer %17, %74 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %76 = nvws.semaphore.acquire %20 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %77 = nvws.semaphore.buffer %20, %76 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %78 = ttng.tc_gen5_mma %77, %75, %73[], %true, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %19, %76 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %16, %74 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %2, %72 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %79 = nvws.semaphore.acquire %0 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %80 = nvws.semaphore.acquire %2 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %81 = nvws.semaphore.acquire %9 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %82 = nvws.semaphore.acquire %12 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %83 = nvws.semaphore.acquire %16 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %84 = nvws.semaphore.acquire %19 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %85 = nvws.semaphore.acquire %22 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %59, %43, %79, %80, %81, %82, %83, %84, %85 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 1>, array<i32: 0>, array<i32: 3>, array<i32: 3>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.semaphore.release %4, %30#3 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %31 = nvws.semaphore.acquire %4 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %32 = nvws.semaphore.buffer %4, %31 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_3, %token = ttng.tmem_load %32[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    "use"(%30#0, %result_3, %30#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

