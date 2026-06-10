#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.descriptor_load %arg0[%1, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %4 = tt.splat %arg24 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %5 = tt.splat %arg24 : f32 -> tensor<128x128xf32, #linear>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %7 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %8 = nvws.semaphore.create %result_2 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %9 = nvws.semaphore.create %result_2 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %10 = nvws.semaphore.create %result_2 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %11 = nvws.semaphore.acquire %8 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %12 = nvws.semaphore.buffer %8, %11 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %13 = ttng.tmem_store %cst, %12[], %true : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf32, #shared1, #smem, mutable>
    %15 = nvws.semaphore.create %14 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>
    %16 = nvws.semaphore.create %14 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %18 = nvws.semaphore.create %17 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %19 = nvws.semaphore.create %17 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %20 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %21 = nvws.semaphore.create %20 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %22 = nvws.semaphore.create %20 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<1x128xf32, #shared1, #smem, mutable>
    %24 = nvws.semaphore.create %23 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>
    %25 = nvws.semaphore.create %23 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %26 = nvws.semaphore.create %result_3 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %27 = nvws.semaphore.create %result_3 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %28 = nvws.semaphore.acquire %6 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %29 = nvws.semaphore.acquire %26 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %30 = nvws.semaphore.acquire %15 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %31 = nvws.semaphore.acquire %18 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %32 = nvws.semaphore.acquire %21 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %33 = nvws.semaphore.acquire %24 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
    %34:10 = scf.for %arg25 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %cst_0, %arg27 = %cst_1, %arg28 = %false, %arg29 = %28, %arg30 = %11, %arg31 = %29, %arg32 = %30, %arg33 = %31, %arg34 = %32, %arg35 = %33) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %50 = nvws.semaphore.buffer %15, %arg32 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %arg26, %50 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %16, %arg32 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %51 = nvws.semaphore.buffer %18, %arg33 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg5[%arg25, %c0_i32] 32768 %51 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %19, %arg33 [#nvws.async_op<tma_load>] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %52 = nvws.semaphore.buffer %6, %arg29 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %53 = nvws.semaphore.acquire %19 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %54 = nvws.semaphore.buffer %19, %53 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %55 = ttg.memdesc_trans %54 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      %56 = ttng.tc_gen5_mma %3, %55, %52[], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      nvws.semaphore.release %18, %53 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %7, %arg29 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %57 = nvws.semaphore.acquire %7 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %58 = nvws.semaphore.buffer %7, %57 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_5, %token_6 = ttng.tmem_load %58[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #linear>
      nvws.semaphore.release %6, %57 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %59 = "tt.reduce"(%result_5) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg36: f32, %arg37: f32):
        %103 = arith.maxnumf %arg36, %arg37 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %103 {ttg.partition = array<i32: 0>} : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %60 = arith.mulf %59, %4 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %61 = arith.maxnumf %arg26, %60 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %62 = nvws.semaphore.buffer %24, %arg35 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %61, %62 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      nvws.semaphore.release %25, %arg35 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %63 = arith.mulf %result_5, %5 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %64 = tt.expand_dims %61 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %65 = tt.broadcast %64 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %66 = arith.subf %63, %65 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %67 = math.exp2 %66 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %68 = arith.subf %arg26, %61 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %69 = nvws.semaphore.acquire %16 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %70 = nvws.semaphore.buffer %16, %69 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      %71 = ttg.local_load %70 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      nvws.semaphore.release %15, %69 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %72 = nvws.semaphore.acquire %25 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %73 = nvws.semaphore.buffer %25, %72 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      %74 = ttg.local_load %73 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      nvws.semaphore.release %24, %72 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]>, !ttg.async.token
      %75 = arith.subf %71, %74 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %76 = math.exp2 %68 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %77 = math.exp2 %75 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %78 = "tt.reduce"(%67) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
      ^bb0(%arg36: f32, %arg37: f32):
        %103 = arith.addf %arg36, %arg37 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %103 {ttg.partition = array<i32: 0>} : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %79 = tt.expand_dims %77 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %80 = tt.broadcast %79 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %81 = nvws.semaphore.buffer %8, %arg30 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_7, %token_8 = ttng.tmem_load %81[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear>
      %82 = arith.mulf %result_7, %80 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128x128xf32, #linear>
      %83 = nvws.semaphore.buffer %21, %arg34 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg10[%arg25, %c0_i32] 32768 %83 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %22, %arg34 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %84 = arith.truncf %67 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %85 = nvws.semaphore.buffer %26, %arg31 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %true_9 = arith.constant {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} true
      ttng.tmem_store %84, %85, %true_9 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %27, %arg31 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %86 = ttng.tmem_store %82, %81[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %9, %arg30 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %87 = nvws.semaphore.acquire %9 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %88 = nvws.semaphore.buffer %9, %87 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %89 = nvws.semaphore.acquire %27 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %90 = nvws.semaphore.buffer %27, %89 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %91 = nvws.semaphore.acquire %22 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %92 = nvws.semaphore.buffer %22, %91 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %93 = ttng.tc_gen5_mma %85, %92, %88[], %arg28, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      nvws.semaphore.release %21, %91 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %26, %89 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.semaphore.release %8, %87 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %94 = arith.mulf %arg27, %76 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %95 = arith.addf %94, %78 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %96 = nvws.semaphore.acquire %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %97 = nvws.semaphore.acquire %8 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %98 = nvws.semaphore.acquire %26 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %99 = nvws.semaphore.acquire %15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      %100 = nvws.semaphore.acquire %18 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %101 = nvws.semaphore.acquire %21 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %102 = nvws.semaphore.acquire %24 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128xf32, #shared1, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %61, %95, %true, %96, %97, %98, %99, %100, %101, %102 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 0>, array<i32: 0>, array<i32: 3>, array<i32: 3>, array<i32: 0>], ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.semaphore.release %10, %34#4 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %35 = nvws.semaphore.acquire %10 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %36 = nvws.semaphore.buffer %10, %35 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_4, %token = ttng.tmem_load %36[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear>
    %37 = arith.truncf %result_4 : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
    %38 = ttg.convert_layout %37 : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked>
    tt.descriptor_store %arg15[%1, %c0_i32], %38 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked>
    %39 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %41 = tt.splat %39 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %42 = tt.addptr %41, %40 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %43 = arith.truncf %34#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>>
    %44 = ttg.convert_layout %43 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf16, #blocked1>
    tt.store %42, %44 : tensor<128x!tt.ptr<f16>, #blocked1>
    %45 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %46 = tt.splat %45 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %47 = tt.addptr %46, %40 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %48 = arith.truncf %34#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>>
    %49 = ttg.convert_layout %48 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf16, #blocked1>
    tt.store %47, %49 : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}


/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/test/unit/language/test_warp_specialization.py:427:27: warning: non-root partition #0 has direct SSA consumer
        acc = tl.dot(p, v, acc)
                          ^
/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/test/unit/language/test_warp_specialization.py:427:27: note: use at distance 0 in partition #2 here
