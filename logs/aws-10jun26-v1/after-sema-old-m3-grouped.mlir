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
    %5 = ub.poison : !ttg.async.token
    %6 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %7 = nvws.semaphore.create %result true : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %8 = nvws.semaphore.create %result false : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %9 = nvws.semaphore.acquire %7 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %10 = nvws.semaphore.buffer %7, %9 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %true_0 = arith.constant true
    ttng.tmem_store %cst, %10, %true_0 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %12 = nvws.semaphore.create %11 true : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %13 = nvws.semaphore.create %11 false : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %14 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %15 = nvws.semaphore.create %14 true : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %16 = nvws.semaphore.create %14 false : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %17 = nvws.semaphore.acquire %12 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %18 = nvws.semaphore.acquire %15 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %19:3 = scf.for %arg7 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg8 = %9, %arg9 = %17, %arg10 = %18) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %20 = tt.addptr %arg0, %arg7 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>, i32
      %21 = tt.load %20 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>
      %22 = tt.int_to_ptr %21 {ttg.partition = array<i32: 2>} : i64 -> !tt.ptr<f16>
      %23 = tt.addptr %arg1, %arg7 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>, i32
      %24 = tt.load %23 {ttg.partition = array<i32: 2>} : !tt.ptr<i64>
      %25 = tt.int_to_ptr %24 {ttg.partition = array<i32: 2>} : i64 -> !tt.ptr<f16>
      %26 = tt.addptr %arg2, %arg7 {ttg.partition = array<i32: 0>} : !tt.ptr<i64>, i32
      %27 = tt.load %26 {ttg.partition = array<i32: 0>} : !tt.ptr<i64>
      %28 = tt.int_to_ptr %27 {ttg.partition = array<i32: 0>} : i64 -> !tt.ptr<f16>
      %29 = tt.make_tensor_descriptor %22, [%arg3, %arg5], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
      %30 = tt.make_tensor_descriptor %25, [%arg4, %arg5], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
      %31 = tt.make_tensor_descriptor %28, [%arg3, %arg4], [%c1024_i64, %c1_i64] {ttg.partition = array<i32: 0>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
      %32 = ub.poison : !ttg.async.token
      %33:4 = scf.for %arg11 = %3 to %2 step %c4_i32 iter_args(%arg12 = %32, %arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %34 = arith.divsi %arg11, %1 {ttg.partition = array<i32: 0, 2>} : i32
        %35 = arith.remsi %arg11, %1 {ttg.partition = array<i32: 0, 2>} : i32
        %36 = arith.muli %34, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %37 = arith.muli %35, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %38:5 = scf.for %arg16 = %c0_i32 to %4 step %c1_i32 iter_args(%arg17 = %false, %arg18 = %arg12, %arg19 = %arg13, %arg20 = %arg14, %arg21 = %arg15) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
          %44 = arith.muli %arg16, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
          %45 = nvws.semaphore.buffer %12, %arg20 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.descriptor_load %29[%36, %44] 16384 %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.semaphore.release %13, %arg20 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %46 = nvws.semaphore.buffer %15, %arg21 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.descriptor_load %30[%37, %44] 16384 %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.semaphore.release %16, %arg21 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %47 = nvws.semaphore.buffer %7, %arg19 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
          %48 = nvws.semaphore.acquire %13 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %49 = nvws.semaphore.buffer %13, %48 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %50 = nvws.semaphore.acquire %16 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %51 = nvws.semaphore.buffer %16, %50 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %52 = ttg.memdesc_trans %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
          %53 = ttng.tc_gen5_mma %49, %52, %47[], %arg17, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
          nvws.semaphore.release %15, %50 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          nvws.semaphore.release %12, %48 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %54 = nvws.semaphore.acquire %12 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %55 = nvws.semaphore.acquire %15 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          scf.yield {ttg.partition = array<i32: 0, 1, 2>} %true, %5, %arg19, %54, %55 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
        } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>]}
        nvws.semaphore.release %8, %38#2 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %39 = nvws.semaphore.acquire %8 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %40 = nvws.semaphore.buffer %8, %39 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_1, %token = ttng.tmem_load %40[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %7, %39 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %41 = arith.truncf %result_1 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %42 = ttg.convert_layout %41 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
        tt.descriptor_store %31[%36, %37], %42 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
        %43 = nvws.semaphore.acquire %7 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %32, %43, %38#3, %38#4 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %33#1, %33#2, %33#3 : !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

