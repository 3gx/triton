#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 1, 0], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 0, 1], [128, 0, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear5 = #ttg.linear<{register = [], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear6 = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.muli %arg2, %c4_i32 : i32
    %3 = arith.muli %2, %arg3 : i32
    %4 = arith.divsi %3, %1 : i32
    %5 = arith.remsi %3, %1 : i32
    %6 = arith.cmpi slt, %0, %5 : i32
    %7 = scf.if %6 -> (i32) {
      %103 = arith.addi %4, %c1_i32 : i32
      scf.yield %103 : i32
    } else {
      scf.yield %4 : i32
    }
    %8 = arith.muli %arg2, %arg3 : i32
    %9 = arith.muli %8, %c1024_i32 : i32
    %10 = tt.make_tensor_descriptor %arg4, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %11 = tt.make_tensor_descriptor %arg4, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %12 = tt.make_tensor_descriptor %arg5, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %13 = tt.make_tensor_descriptor %arg6, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %14 = tt.make_tensor_descriptor %arg7, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %15 = tt.make_tensor_descriptor %arg7, [%9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %16 = arith.muli %arg3, %c1024_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %18 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %19 = arith.mulf %arg0, %cst : f32
    %20 = tt.splat %19 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %21 = tt.splat %19 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %22 = tt.splat %19 : f32 -> tensor<128x128xf32, #linear>
    %23 = tt.splat %19 : f32 -> tensor<128x128xf32, #linear>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %25 = nvws.semaphore.create %24 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %26 = nvws.semaphore.create %24 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %27 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %28 = nvws.semaphore.create %27 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %29 = nvws.semaphore.create %27 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %30 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %31 = nvws.semaphore.create %30 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %32 = nvws.semaphore.create %30 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %34 = nvws.semaphore.create %33 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %35 = nvws.semaphore.create %33 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %36 = ub.poison : !ttg.async.token
    %37 = ub.poison : !ttg.async.token
    %38 = ub.poison : !ttg.async.token
    %39 = ub.poison : !ttg.async.token
    %40 = ub.poison : !ttg.async.token
    %result = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %41 = ttng.tmem_subslice %result {N = 64 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %42 = ttg.memdesc_reinterpret %41 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %43 = ttng.tmem_subslice %result {N = 66 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %44 = ttg.memdesc_reinterpret %43 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %45 = ttng.tmem_subslice %result {N = 65 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %46 = ttg.memdesc_reinterpret %45 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %47 = ttng.tmem_subslice %result {N = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
    %48 = ttg.memdesc_reinterpret %47 : !ttg.memdesc<1x128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %49 = nvws.semaphore.create %42, %44, %46, %result, %48 true : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %50 = nvws.semaphore.create %42, %44, %46, %result, %48 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %51 = nvws.semaphore.create %42, %44, %46, %result, %48 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %52 = nvws.semaphore.create %42, %44, %46, %result, %48 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %53 = nvws.semaphore.create %42, %44, %46, %result, %48 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %54 = nvws.semaphore.create %42, %44, %46, %result, %48 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %55 = nvws.semaphore.acquire %49 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %56 = ub.poison : !ttg.async.token
    %57 = ub.poison : !ttg.async.token
    %58 = ub.poison : !ttg.async.token
    %59 = ub.poison : !ttg.async.token
    %60 = ub.poison : !ttg.async.token
    %result_3 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %61 = ttng.tmem_subslice %result_3 {N = 64 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %62 = ttg.memdesc_reinterpret %61 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %63 = ttng.tmem_subslice %result_3 {N = 66 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %64 = ttg.memdesc_reinterpret %63 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %65 = ttng.tmem_subslice %result_3 {N = 65 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128>
    %66 = ttg.memdesc_reinterpret %65 : !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %67 = ttng.tmem_subslice %result_3 {N = 0 : i32} : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128>
    %68 = ttg.memdesc_reinterpret %67 : !ttg.memdesc<1x128x64xf32, #tmem2, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %69 = nvws.semaphore.create %62, %64, %66, %result_3, %68 true : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %70 = nvws.semaphore.create %62, %64, %66, %result_3, %68 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %71 = nvws.semaphore.create %62, %64, %66, %result_3, %68 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %72 = nvws.semaphore.create %62, %64, %66, %result_3, %68 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %73 = nvws.semaphore.create %62, %64, %66, %result_3, %68 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %74 = nvws.semaphore.create %62, %64, %66, %result_3, %68 false : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>
    %75 = nvws.semaphore.acquire %69 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %76 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %77 = nvws.semaphore.create %76 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %78 = nvws.semaphore.create %76 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %79 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %80 = nvws.semaphore.create %79 true : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %81 = nvws.semaphore.create %79 false : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %82 = ub.poison : !ttg.async.token
    %83 = ub.poison : !ttg.async.token
    %84 = ub.poison : !ttg.async.token
    %85 = ub.poison : !ttg.async.token
    %result_4 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %86 = nvws.semaphore.create %result_4 true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %87 = nvws.semaphore.create %result_4 false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %88 = nvws.semaphore.acquire %86 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %89 = ub.poison : !ttg.async.token
    %90 = ub.poison : !ttg.async.token
    %91 = ub.poison : !ttg.async.token
    %92 = ub.poison : !ttg.async.token
    %result_5 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %93 = nvws.semaphore.create %result_5 true : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %94 = nvws.semaphore.create %result_5 false : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %95 = nvws.semaphore.acquire %93 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %96 = nvws.semaphore.acquire %25 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %97 = nvws.semaphore.acquire %28 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %98 = nvws.semaphore.acquire %31 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %99 = nvws.semaphore.acquire %34 {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %100 = nvws.semaphore.acquire %77 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %101 = nvws.semaphore.acquire %80 {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %102:11 = scf.for %arg8 = %c0_i32 to %7 step %c1_i32 iter_args(%arg9 = %0, %arg10 = %55, %arg11 = %75, %arg12 = %88, %arg13 = %95, %arg14 = %96, %arg15 = %97, %arg16 = %98, %arg17 = %99, %arg18 = %100, %arg19 = %101) -> (i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %103 = arith.remsi %arg9, %c4_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %104 = arith.divsi %arg9, %c4_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %105 = arith.divsi %104, %arg3 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %106 = arith.remsi %104, %arg3 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %107 = arith.muli %105, %16 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %108 = arith.muli %106, %c1024_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %109 = arith.addi %107, %108 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %110 = arith.muli %103, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %111 = arith.addi %109, %110 {ttg.partition = array<i32: 2, 3>} : i32
      %112 = arith.addi %111, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %113 = arith.addi %111, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %114 = tt.splat %110 {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %115 = tt.splat %110 {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %116 = arith.addi %114, %17 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %117 = arith.addi %115, %18 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %118 = nvws.semaphore.buffer %25, %arg14 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %10[%111, %c0_i32] 32768 %118 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %26, %arg14 [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %119 = nvws.semaphore.buffer %28, %arg15 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %11[%113, %c0_i32] 32768 %119 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %29, %arg15 [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %120 = nvws.semaphore.buffer %86, %arg12 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %true_6 = arith.constant {ttg.partition = array<i32: 0>} true
      ttng.tmem_store %cst_0, %120, %true_6 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %121 = nvws.semaphore.buffer %93, %arg13 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %true_7 = arith.constant {ttg.partition = array<i32: 0>} true
      ttng.tmem_store %cst_0, %121, %true_7 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %122 = nvws.semaphore.acquire %26 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %123 = nvws.semaphore.acquire %29 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %124:13 = scf.for %arg20 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg21 = %109, %arg22 = %cst_2, %arg23 = %cst_1, %arg24 = %arg10, %arg25 = %arg12, %arg26 = %cst_2, %arg27 = %cst_1, %arg28 = %arg11, %arg29 = %arg13, %arg30 = %122, %arg31 = %123, %arg32 = %arg16, %arg33 = %arg17) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %186 = nvws.semaphore.buffer %31, %arg32 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.descriptor_load %12[%arg21, %c0_i32] 32768 %186 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.semaphore.release %32, %arg32 [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        %187 = nvws.semaphore.buffer %34, %arg33 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.descriptor_load %13[%arg21, %c0_i32] 32768 %187 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.semaphore.release %35, %arg33 [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        %188:5 = nvws.semaphore.buffer %49, %arg24 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %189 = nvws.semaphore.buffer %26, %arg30 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %190 = nvws.semaphore.acquire %32 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        %191 = nvws.semaphore.buffer %32, %190 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %192 = ttg.memdesc_trans %191 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        %193 = ttng.tc_gen5_mma %189, %192, %188#3[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %50, %arg24 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %194:5 = nvws.semaphore.buffer %69, %arg28 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %195 = nvws.semaphore.buffer %29, %arg31 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %196 = ttg.memdesc_trans %191 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>
        %197 = ttng.tc_gen5_mma %195, %196, %194#3[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %31, %190 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.semaphore.release %70, %arg28 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %198 = nvws.semaphore.acquire %50 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %199:5 = nvws.semaphore.buffer %50, %198 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_23, %token_24 = ttng.tmem_load %199#3[] {ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
        %200 = ttg.convert_layout %result_23 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %201 = nvws.semaphore.acquire %70 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %202:5 = nvws.semaphore.buffer %70, %201 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_25, %token_26 = ttng.tmem_load %202#3[] {ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
        %203 = ttg.convert_layout %result_25 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %204 = "tt.reduce"(%200) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %287 = arith.maxnumf %arg34, %arg35 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %287 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %205 = "tt.reduce"(%203) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %287 = arith.maxnumf %arg34, %arg35 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %287 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %206 = arith.mulf %204, %20 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %207 = arith.mulf %205, %21 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %208 = arith.maxnumf %arg23, %206 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %209 = arith.maxnumf %arg27, %207 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %210 = arith.mulf %200, %22 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %211 = arith.mulf %203, %23 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %212 = tt.expand_dims %208 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %213 = tt.expand_dims %209 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %214 = tt.broadcast %212 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %215 = tt.broadcast %213 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %216 = arith.subf %210, %214 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %217 = arith.subf %211, %215 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %218 = math.exp2 %216 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %219 = math.exp2 %217 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %220 = arith.subf %arg23, %208 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %221 = arith.subf %arg27, %209 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %222 = math.exp2 %220 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %223 = tt.expand_dims %222 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %true_27 = arith.constant {ttg.partition = array<i32: 5>} true
        ttng.tmem_store %223, %199#0, %true_27 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        nvws.semaphore.release %51, %198 [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %224 = math.exp2 %221 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %225 = tt.expand_dims %224 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %true_28 = arith.constant {ttg.partition = array<i32: 4>} true
        ttng.tmem_store %225, %202#0, %true_28 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
        nvws.semaphore.release %71, %201 [#nvws.async_op<none>] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %226 = "tt.reduce"(%218) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %287 = arith.addf %arg34, %arg35 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %287 {ttg.partition = array<i32: 5>} : f32
        }) {ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %227 = "tt.reduce"(%219) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg34: f32, %arg35: f32):
          %287 = arith.addf %arg34, %arg35 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %287 {ttg.partition = array<i32: 4>} : f32
        }) {ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %228 = nvws.semaphore.buffer %86, %arg25 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_29, %token_30 = ttng.tmem_load %228[] {tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
        %229 = nvws.semaphore.buffer %93, %arg29 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_31, %token_32 = ttng.tmem_load %229[] {tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
        %230 = tt.reshape %result_29 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %231 = tt.reshape %result_31 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %232 = tt.trans %230 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %233 = tt.trans %231 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %232 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_33, %outRHS_34 = tt.split %233 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %234 = nvws.semaphore.acquire %51 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %235:5 = nvws.semaphore.buffer %51, %234 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_35, %token_36 = ttng.tmem_load %235#0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
        nvws.semaphore.release %52, %234 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %236 = tt.reshape %result_35 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %237 = ttg.convert_layout %236 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %238 = tt.expand_dims %237 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %239 = nvws.semaphore.acquire %71 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %240:5 = nvws.semaphore.buffer %71, %239 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %result_37, %token_38 = ttng.tmem_load %240#0[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
        nvws.semaphore.release %72, %239 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %241 = tt.reshape %result_37 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %242 = ttg.convert_layout %241 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %243 = tt.expand_dims %242 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %244 = ttg.convert_layout %238 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %245 = ttg.convert_layout %243 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %246 = tt.broadcast %244 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %247 = tt.broadcast %245 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %248 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %246 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %249 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_33, %247 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %250 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %246 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %251 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_34, %247 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %252 = tt.join %248, %250 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %253 = tt.join %249, %251 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %254 = tt.trans %252 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %255 = tt.trans %253 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %256 = tt.reshape %254 {ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %257 = tt.reshape %255 {ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %258 = arith.truncf %218 {ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %259 = arith.truncf %219 {ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %260 = nvws.semaphore.acquire %52 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %261:5 = nvws.semaphore.buffer %52, %260 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %true_39 = arith.constant {ttg.partition = array<i32: 5>} true
        ttng.tmem_store %258, %261#4, %true_39 {ttg.partition = array<i32: 5>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %49, %260 [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %262 = nvws.semaphore.acquire %72 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %263:5 = nvws.semaphore.buffer %72, %262 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %true_40 = arith.constant {ttg.partition = array<i32: 4>} true
        ttng.tmem_store %259, %263#4, %true_40 {ttg.partition = array<i32: 4>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %69, %262 [#nvws.async_op<none>] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %264 = ttng.tmem_store %256, %228[], %true {tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %87, %arg25 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %265 = ttng.tmem_store %257, %229[], %true {tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %94, %arg29 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %266 = nvws.semaphore.acquire %49 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %267:5 = nvws.semaphore.buffer %49, %266 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %268 = nvws.semaphore.acquire %87 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %269 = nvws.semaphore.buffer %87, %268 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %270 = nvws.semaphore.acquire %35 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        %271 = nvws.semaphore.buffer %35, %270 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %272 = ttng.tc_gen5_mma %267#4, %271, %269[], %true, %true {tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %86, %268 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %273 = nvws.semaphore.acquire %69 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %274:5 = nvws.semaphore.buffer %69, %273 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %275 = nvws.semaphore.acquire %94 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %276 = nvws.semaphore.buffer %94, %275 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %277 = ttng.tc_gen5_mma %274#4, %271, %276[], %true, %true {tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        nvws.semaphore.release %34, %270 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.semaphore.release %93, %275 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %278 = arith.mulf %arg22, %222 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %279 = arith.mulf %arg26, %224 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %280 = arith.addf %278, %226 {ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %281 = arith.addf %279, %227 {ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %282 = arith.addi %arg21, %c128_i32 {ttg.partition = array<i32: 3>} : i32
        %283 = nvws.semaphore.acquire %86 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %284 = nvws.semaphore.acquire %93 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %285 = nvws.semaphore.acquire %31 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        %286 = nvws.semaphore.acquire %34 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %282, %280, %208, %266, %283, %281, %209, %273, %284, %arg30, %arg31, %285, %286 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>, array<i32: 1>, array<i32: 1>, array<i32: 3>, array<i32: 3>]}
      nvws.semaphore.release %28, %124#10 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %25, %124#9 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %73, %124#7 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.semaphore.release %53, %124#3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %125 = tt.expand_dims %124#6 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %true_8 = arith.constant {ttg.partition = array<i32: 4>} true
      %126 = nvws.semaphore.acquire %73 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %127:5 = nvws.semaphore.buffer %73, %126 {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      ttng.tmem_store %125, %127#2, %true_8 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %128 = tt.expand_dims %124#5 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %true_9 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %128, %127#1, %true_9 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      nvws.semaphore.release %74, %126 [#nvws.async_op<none>] {ttg.partition = array<i32: 4>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %129 = tt.expand_dims %124#2 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %true_10 = arith.constant {ttg.partition = array<i32: 5>} true
      %130 = nvws.semaphore.acquire %53 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %131:5 = nvws.semaphore.buffer %53, %130 {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      ttng.tmem_store %129, %131#2, %true_10 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      %132 = tt.expand_dims %124#1 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %true_11 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %132, %131#1, %true_11 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>
      nvws.semaphore.release %54, %130 [#nvws.async_op<none>] {ttg.partition = array<i32: 5>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %133 = nvws.semaphore.acquire %54 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %134:5 = nvws.semaphore.buffer %54, %133 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_12, %token = ttng.tmem_load %134#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
      %135 = tt.reshape %result_12 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %136 = ttg.convert_layout %135 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %137 = math.log2 %136 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %138 = nvws.semaphore.acquire %74 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %139:5 = nvws.semaphore.buffer %74, %138 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_13, %token_14 = ttng.tmem_load %139#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
      %140 = tt.reshape %result_13 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %141 = ttg.convert_layout %140 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %142 = math.log2 %141 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %result_15, %token_16 = ttng.tmem_load %134#2[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
      nvws.semaphore.release %49, %133 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %143 = tt.reshape %result_15 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %144 = ttg.convert_layout %143 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %145 = arith.addf %144, %137 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %result_17, %token_18 = ttng.tmem_load %139#2[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable, 1x128x1> -> tensor<128x1xf32, #linear5>
      nvws.semaphore.release %69, %138 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %146 = tt.reshape %result_17 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %147 = ttg.convert_layout %146 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %148 = arith.addf %147, %142 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %149 = tt.expand_dims %136 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %150 = tt.expand_dims %141 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %151 = tt.broadcast %149 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %152 = tt.broadcast %150 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %153 = nvws.semaphore.buffer %86, %124#4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_19, %token_20 = ttng.tmem_load %153[] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
      %154 = ttg.convert_layout %result_19 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %155 = nvws.semaphore.buffer %93, %124#8 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_21, %token_22 = ttng.tmem_load %155[] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #linear1>
      %156 = ttg.convert_layout %result_21 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %157 = arith.divf %154, %151 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %158 = arith.divf %156, %152 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %159 = arith.muli %104, %c1024_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %160 = tt.addptr %arg1, %159 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : !tt.ptr<f32>, i32
      %161 = tt.splat %160 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %162 = tt.splat %160 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %163 = tt.addptr %161, %116 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %164 = tt.addptr %162, %117 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %165 = ttg.convert_layout %145 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %166 = ttg.convert_layout %148 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      tt.store %163, %165 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %164, %166 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      %167 = arith.truncf %157 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %168 = nvws.semaphore.buffer %77, %arg18 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %167, %168 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %78, %arg18 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %169 = arith.truncf %158 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %170 = nvws.semaphore.buffer %80, %arg19 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %169, %170 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.semaphore.release %81, %arg19 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %171 = nvws.semaphore.acquire %78 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %172 = nvws.semaphore.buffer %78, %171 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %173 = ttg.local_load %172 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      nvws.semaphore.release %77, %171 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %174 = ttg.convert_layout %173 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      %175 = nvws.semaphore.acquire %81 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %176 = nvws.semaphore.buffer %81, %175 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %177 = ttg.local_load %176 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #linear>
      nvws.semaphore.release %80, %175 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %178 = ttg.convert_layout %177 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %14[%111, %c0_i32], %174 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      tt.descriptor_store %15[%112, %c0_i32], %178 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      %179 = arith.addi %arg9, %1 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %180 = nvws.semaphore.acquire %49 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %181 = nvws.semaphore.acquire %69 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %182 = nvws.semaphore.acquire %25 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %183 = nvws.semaphore.acquire %28 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %184 = nvws.semaphore.acquire %77 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      %185 = nvws.semaphore.acquire %80 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %179, %180, %181, %124#4, %124#8, %182, %183, %124#11, %124#12, %184, %185 : i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4, 5>, array<i32: 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>, array<i32: 3>, array<i32: 3>, array<i32: 3>, array<i32: 3>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}


