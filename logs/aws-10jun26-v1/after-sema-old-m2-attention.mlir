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
    %0 = ub.poison : !ttg.async.token
    %1 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %2 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %3 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %4 = nvws.semaphore.acquire %2 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %5 = ub.poison : !ttg.async.token
    %6 = ub.poison : !ttg.async.token
    %7 = ub.poison : !ttg.async.token
    %8 = ub.poison : !ttg.async.token
    %9 = ub.poison : !ttg.async.token
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %10 = nvws.semaphore.create %result_2 true : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %11 = nvws.semaphore.create %result_2 false : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %12 = nvws.semaphore.create %result_2 false : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %13 = nvws.semaphore.acquire %10 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %14 = nvws.semaphore.buffer %10, %13 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %15 = ttng.tmem_store %cst_0, %14[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<1x256xf32, #shared1, #smem, mutable>
    %17 = nvws.semaphore.create %16 true : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %18 = nvws.semaphore.create %16 false : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %20 = nvws.semaphore.create %19 true : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %21 = nvws.semaphore.create %19 false : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>
    %23 = nvws.semaphore.create %22 true : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %24 = nvws.semaphore.create %22 false : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>
    %25 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %26 = nvws.semaphore.create %25 true : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>
    %27 = nvws.semaphore.create %25 false : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>
    %28 = ttg.local_alloc : () -> !ttg.memdesc<1x256xf32, #shared1, #smem, mutable>
    %29 = nvws.semaphore.create %28 true : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %30 = nvws.semaphore.create %28 false : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>
    %31 = nvws.semaphore.acquire %17 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %32 = nvws.semaphore.acquire %20 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %33 = nvws.semaphore.acquire %23 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %34 = nvws.semaphore.acquire %26 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %35 = nvws.semaphore.acquire %29 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %36:9 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %4, %arg10 = %13, %arg11 = %31, %arg12 = %32, %arg13 = %33, %arg14 = %34, %arg15 = %35) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %39 = nvws.semaphore.buffer %17, %arg11 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      ttg.local_store %arg8, %39 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %18, %arg11 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %40 = nvws.semaphore.buffer %20, %arg12 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg1[%arg6, %c0_i32] 8192 %40 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %21, %arg12 [#nvws.async_op<tma_load>] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %41 = nvws.semaphore.buffer %2, %arg9 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %42 = nvws.semaphore.acquire %21 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %43 = nvws.semaphore.buffer %21, %42 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %44 = ttg.memdesc_trans %43 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %45 = ttng.tc_gen5_mma %arg0, %44, %41[], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      nvws.semaphore.release %20, %42 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %3, %arg9 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %46 = nvws.semaphore.acquire %3 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %47 = nvws.semaphore.buffer %3, %46 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      %result_4, %token_5 = ttng.tmem_load %47[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.semaphore.release %2, %46 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %48 = "compute_row_max"(%result_4, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %49 = nvws.semaphore.buffer %29, %arg15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      ttg.local_store %48, %49 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %30, %arg15 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %50 = "sub_row_max"(%result_4, %48, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %51 = math.exp2 %50 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %52 = arith.subf %arg8, %48 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %53 = nvws.semaphore.acquire %18 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %54 = nvws.semaphore.buffer %18, %53 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %55 = ttg.local_load %54 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      nvws.semaphore.release %17, %53 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %56 = nvws.semaphore.acquire %30 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %57 = nvws.semaphore.buffer %30, %56 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %58 = ttg.local_load %57 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      nvws.semaphore.release %29, %56 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %59 = arith.subf %55, %58 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %60 = math.exp2 %52 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %61 = math.exp2 %59 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %62 = "tt.reduce"(%51) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %91 = arith.addf %arg16, %arg17 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %91 {ttg.partition = array<i32: 0>} : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %63 = arith.mulf %arg7, %60 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %64 = arith.addf %63, %62 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %65 = tt.expand_dims %61 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %66 = tt.broadcast %65 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %67 = tt.addptr %arg5, %arg6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>, i32
      %68 = tt.load %67 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>
      %69 = tt.splat %68 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : f32 -> tensor<256x64xf32, #blocked>
      %70 = nvws.semaphore.buffer %10, %arg10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %result_6, %token_7 = ttng.tmem_load %70[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %71 = arith.mulf %result_6, %66 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      %72 = arith.addf %71, %69 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      %73 = nvws.semaphore.buffer %23, %arg13 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg2[%arg6, %c0_i32] 8192 %73 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %24, %arg13 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %74 = arith.truncf %51 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %75 = nvws.semaphore.buffer %26, %arg14 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.local_store %74, %75 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      nvws.semaphore.release %27, %arg14 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %76 = ttng.tmem_store %72, %70[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %11, %arg10 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %77 = nvws.semaphore.acquire %11 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %78 = nvws.semaphore.buffer %11, %77 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %79 = nvws.semaphore.acquire %24 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %80 = nvws.semaphore.buffer %24, %79 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %81 = nvws.semaphore.acquire %27 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %82 = nvws.semaphore.buffer %27, %81 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %83 = ttng.tc_gen5_mma %82, %80, %78[], %true, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %26, %81 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %23, %79 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %10, %77 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %84 = nvws.semaphore.acquire %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %85 = nvws.semaphore.acquire %10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %86 = nvws.semaphore.acquire %17 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %87 = nvws.semaphore.acquire %20 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %88 = nvws.semaphore.acquire %23 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x64x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %89 = nvws.semaphore.acquire %26 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %90 = nvws.semaphore.acquire %29 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %64, %48, %84, %85, %86, %87, %88, %89, %90 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 1>, array<i32: 0>, array<i32: 3>, array<i32: 3>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.semaphore.release %12, %36#3 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %37 = nvws.semaphore.acquire %12 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %38 = nvws.semaphore.buffer %12, %37 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_3, %token = ttng.tmem_load %38[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    "use"(%36#0, %result_3, %36#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

