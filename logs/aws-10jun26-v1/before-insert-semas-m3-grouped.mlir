#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @grouped_matmul_tma_kernel(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32) attributes {noinline = false} {
    %c1024_i64 = arith.constant 1024 : i64
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.divsi %arg3, %c128_i32 : i32
    %1 = arith.divsi %arg4, %c128_i32 : i32
    %2 = arith.muli %0, %1 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.divsi %arg5, %c64_i32 : i32
    %result, %token = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %5 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %6 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %7 = scf.for %arg7 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg8 = %token) -> (!ttg.async.token)  : i32 {
      %8 = tt.addptr %arg0, %arg7 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>, i32
      %9 = tt.load %8 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>
      %10 = tt.int_to_ptr %9 {ttg.partition = array<i32: 2>} : i64 -> !tt.ptr<f16>
      %11 = tt.addptr %arg1, %arg7 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>, i32
      %12 = tt.load %11 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>
      %13 = tt.int_to_ptr %12 {ttg.partition = array<i32: 2>} : i64 -> !tt.ptr<f16>
      %14 = tt.addptr %arg2, %arg7 {ttg.partition = array<i32: 0>} : !tt.ptr<i64>, i32
      %15 = tt.load %14 {ttg.partition = array<i32: 0>} : !tt.ptr<i64>
      %16 = tt.int_to_ptr %15 {ttg.partition = array<i32: 0>} : i64 -> !tt.ptr<f16>
      %17 = tt.make_tensor_descriptor %10, [%arg3, %arg5], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
      %18 = tt.make_tensor_descriptor %13, [%arg4, %arg5], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
      %19 = tt.make_tensor_descriptor %16, [%arg3, %arg4], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
      %20 = scf.for %arg9 = %3 to %2 step %c4_i32 iter_args(%arg10 = %arg8) -> (!ttg.async.token)  : i32 {
        %21 = arith.divsi %arg9, %1 {ttg.partition = array<i32: 0, 2>} : i32
        %22 = arith.remsi %arg9, %1 {ttg.partition = array<i32: 0, 2>} : i32
        %23 = arith.muli %21, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %24 = arith.muli %22, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %25:2 = scf.for %arg11 = %c0_i32 to %4 step %c1_i32 iter_args(%arg12 = %false, %arg13 = %arg10) -> (i1, !ttg.async.token)  : i32 {
          %28 = arith.muli %arg11, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
          nvws.descriptor_load %17[%23, %28] 16384 %5 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.descriptor_load %18[%24, %28] 16384 %6 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %29 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
          %30 = ttng.tc_gen5_mma %5, %29, %result[%arg13], %arg12, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
          scf.yield {ttg.partition = array<i32: 1, 2>} %true, %30 : i1, !ttg.async.token
        } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>]}
        %result_0, %token_1 = ttng.tmem_load %result[%25#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %26 = arith.truncf %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %27 = ttg.convert_layout %26 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
        tt.descriptor_store %19[%23, %24], %27 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %token_1 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %20 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}


