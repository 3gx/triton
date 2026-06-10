#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @grouped_matmul_tma_kernel(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32) attributes {noinline = false} {
    %0 = ub.poison : !ttg.async.token
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
    %1 = arith.divsi %arg3, %c128_i32 : i32
    %2 = arith.divsi %arg4, %c128_i32 : i32
    %3 = arith.muli %1, %2 : i32
    %4 = tt.get_program_id x : i32
    %5 = arith.divsi %arg5, %c64_i32 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %6 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %7 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %8 = nvws.semaphore.acquire %6 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %9 = nvws.semaphore.buffer %6, %8 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %true_0 = arith.constant true
    ttng.tmem_store %cst, %9, %true_0 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %11 = nvws.semaphore.create %10 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %12 = nvws.semaphore.create %10 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %13 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %14 = nvws.semaphore.create %13 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %15 = nvws.semaphore.create %13 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %16 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %17 = nvws.semaphore.acquire %11 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %18 = nvws.semaphore.acquire %14 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %19:4 = scf.for %arg7 = %c0_i32 to %arg6 step %c1_i32 iter_args(%arg8 = %0, %arg9 = %8, %arg10 = %17, %arg11 = %18) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
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
      %32:4 = scf.for %arg12 = %4 to %3 step %c4_i32 iter_args(%arg13 = %arg8, %arg14 = %arg9, %arg15 = %arg10, %arg16 = %arg11) -> (!ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %33 = arith.divsi %arg12, %2 {ttg.partition = array<i32: 0, 2>} : i32
        %34 = arith.remsi %arg12, %2 {ttg.partition = array<i32: 0, 2>} : i32
        %35 = arith.muli %33, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %36 = arith.muli %34, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
        %37:3 = scf.for %arg17 = %c0_i32 to %5 step %c1_i32 iter_args(%arg18 = %false, %arg19 = %arg15, %arg20 = %arg16) -> (i1, !ttg.async.token, !ttg.async.token)  : i32 {
          %43 = arith.muli %arg17, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
          %44 = nvws.semaphore.buffer %11, %arg19 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.descriptor_load %29[%35, %43] 16384 %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.semaphore.release %12, %arg19 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %45 = nvws.semaphore.buffer %14, %arg20 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.descriptor_load %30[%36, %43] 16384 %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          nvws.semaphore.release %15, %arg20 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %46 = ttg.memdesc_trans %16 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
          %47 = nvws.semaphore.buffer %6, %arg14 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
          %48 = nvws.semaphore.acquire %12 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %49 = nvws.semaphore.buffer %12, %48 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %50 = nvws.semaphore.acquire %15 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %51 = nvws.semaphore.buffer %15, %50 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
          %52 = ttg.memdesc_trans %51 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
          %53 = ttng.tc_gen5_mma %49, %52, %9[], %arg18, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
          nvws.semaphore.release %14, %50 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          nvws.semaphore.release %11, %48 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
          %54 = nvws.semaphore.acquire %11 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          %55 = nvws.semaphore.acquire %14 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
          scf.yield {ttg.partition = array<i32: 1, 2>} %true, %54, %55 : i1, !ttg.async.token, !ttg.async.token
        } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 2>]}
        nvws.semaphore.release %7, %arg14 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %38 = nvws.semaphore.acquire %7 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %39 = nvws.semaphore.buffer %7, %38 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_1, %token = ttng.tmem_load %9[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.semaphore.release %6, %38 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %40 = arith.truncf %result_1 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %41 = ttg.convert_layout %40 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
        tt.descriptor_store %31[%35, %36], %41 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
        %42 = nvws.semaphore.acquire %6 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %0, %42, %37#1, %37#2 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %32#0, %32#1, %32#2, %32#3 : !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

