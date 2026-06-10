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
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
    %6:4 = scf.for %arg6 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg7 = %cst_1, %arg8 = %cst, %arg9 = %token, %arg10 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      ttg.local_store %arg8, %1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      nvws.descriptor_load %arg1[%arg6, %c0_i32] 8192 %2 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %7 = ttg.memdesc_trans %2 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 2>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %8 = ttng.tc_gen5_mma %arg0, %7, %result[%arg9], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%8] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %9 = "compute_row_max"(%result_6, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      ttg.local_store %9, %5 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<256xf32, #shared1, #smem, mutable>
      %10 = "sub_row_max"(%result_6, %9, %arg3) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %11 = math.exp2 %10 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %12 = arith.subf %arg8, %9 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = ttg.local_load %1 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = ttg.local_load %5 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256xf32, #shared1, #smem, mutable> -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.subf %13, %14 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = math.exp2 %12 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = math.exp2 %15 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = "tt.reduce"(%11) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %31 = arith.addf %arg11, %arg12 {ttg.partition = array<i32: 0>} : f32
        tt.reduce.return %31 {ttg.partition = array<i32: 0>} : f32
      }) {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = arith.mulf %arg7, %16 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %20 = arith.addf %19, %18 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = tt.expand_dims %17 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %22 = tt.broadcast %21 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %23 = tt.addptr %arg5, %arg6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>, i32
      %24 = tt.load %23 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !tt.ptr<f32>
      %25 = tt.splat %24 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : f32 -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg10] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %26 = arith.mulf %result_8, %22 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      %27 = arith.addf %26, %25 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked>
      nvws.descriptor_load %arg2[%arg6, %c0_i32] 8192 %3 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %28 = arith.truncf %11 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      ttg.local_store %28, %4 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %29 = ttng.tmem_store %27, %result_2[%token_9], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 1>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %30 = ttng.tc_gen5_mma %4, %3, %result_2[%29], %true, %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %20, %9, %token_7, %30 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 4 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 2>, array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_4, %token_5 = ttng.tmem_load %result_2[%6#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%6#0, %result_4, %6#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}


