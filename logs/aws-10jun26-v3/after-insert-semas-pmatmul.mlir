#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [8, 2, 2], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked9 = #ttg.blocked<{sizePerThread = [1, 2, 16], threadsPerWarp = [8, 1, 4], warpsPerCTA = [8, 1, 1], order = [1, 2, 0]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 16, 2], threadsPerWarp = [8, 4, 1], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 1, 8, 4], warpsPerCTA = [1, 1, 1, 8, 1], order = [4, 3, 2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 128]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 0, 64]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 1, 0]], block = []}>
#linear4 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0]], warp = [[32, 0, 0], [64, 0, 0], [0, 0, 1]], block = []}>
#linear5 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [64, 0, 0]], lane = [[0, 16, 0], [0, 32, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0]], warp = [[8, 0, 0], [16, 0, 0], [32, 0, 0]], block = []}>
#linear6 = #ttg.linear<{register = [[0, 0, 1], [0, 64, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[0, 32, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear7 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [64, 0]], lane = [[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#linear8 = #ttg.linear<{register = [[0, 64], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[16, 0], [32, 0], [64, 0]], block = []}>
#linear9 = #ttg.linear<{register = [[0, 1], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], lane = [[16, 0], [32, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[0, 1], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[32, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear11 = #ttg.linear<{register = [[64], [1], [2], [4], [8]], lane = [[16], [32], [0], [0], [0]], warp = [[0], [0], [0]], block = []}>
#linear12 = #ttg.linear<{register = [[64], [1], [2], [4], [8], [16]], lane = [[32], [0], [0], [0], [0]], warp = [[0], [0], [0]], block = []}>
#linear13 = #ttg.linear<{register = [[0, 1, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [0, 0, 16]], lane = [[0, 0, 32], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear14 = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0]], lane = [[0, 32, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0]], warp = [[16, 0, 0], [32, 0, 0], [64, 0, 0]], block = []}>
#linear15 = #ttg.linear<{register = [[1, 0], [0, 1], [0, 2], [0, 4], [0, 8]], lane = [[0, 16], [0, 32], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear16 = #ttg.linear<{register = [[1, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear17 = #ttg.linear<{register = [[0, 1], [1, 0], [2, 0], [4, 0], [8, 0]], lane = [[16, 0], [32, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#linear18 = #ttg.linear<{register = [[0, 1], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[32, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8, rank = 5}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, rank = 3}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared6 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, fp4Padded = true}>
#shared7 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared8 = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [1, 0, 0, 0, 0]]}, alignment = 128>
#shared9 = #ttg.shared_linear<{offset = [[0, 1], [0, 2], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [128, 0]]}, alignment = 128>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_p_matmul_NNT_fp8e4nvxfp8e4nvxmxfp4_128x256x128x1_dequantize_mxfp8(%arg0: !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: !tt.tensordesc<tensor<1x128xf8E4M3FN, #shared1>>, %arg20: i32, %arg21: i32, %arg22: i64, %arg23: i64, %arg24: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: !tt.tensordesc<tensor<1x256x64xui8, #shared2>>, %arg31: i32, %arg32: i32, %arg33: i32, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg38: i32 {tt.divisibility = 16 : i32}, %arg39: i32 {tt.divisibility = 16 : i32}, %arg40: !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared3>>, %arg41: i32, %arg42: i32, %arg43: i32, %arg44: i32, %arg45: i32, %arg46: i64, %arg47: i64, %arg48: i64, %arg49: i64, %arg50: i64, %arg51: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg52: i32 {tt.divisibility = 16 : i32}, %arg53: i32 {tt.divisibility = 16 : i32}, %arg54: i32 {tt.divisibility = 16 : i32}, %arg55: i32 {tt.divisibility = 16 : i32}, %arg56: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg57: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg58: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg59: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg60: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg61: i32, %arg62: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<false> : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %0 = ub.poison : !tt.ptr<i32>
    %1 = ub.poison : tensor<128x4x!tt.ptr<i8>, #blocked1>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<128x64xf32, #linear>
    %c31_i32 = arith.constant 31 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %cst_2 = arith.constant dense<4.480000e+02> : tensor<128x2x1xf32, #blocked2>
    %cst_3 = arith.constant dense<8388607> : tensor<128x2x1xi32, #blocked2>
    %cst_4 = arith.constant dense<2139095040> : tensor<128x2x1xi32, #blocked2>
    %cst_5 = arith.constant dense<23> : tensor<128x2xi32, #blocked3>
    %true = arith.constant true
    %cst_6 = arith.constant dense<-1> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_7 = arith.constant dense<-1> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked4>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x2x1xf32, #blocked5>
    %cst_10 = arith.constant dense<1.000000e+00> : tensor<128x2x1xf32, #blocked5>
    %cst_11 = arith.constant dense<2139095040> : tensor<128x2x1xi32, #blocked5>
    %cst_12 = arith.constant dense<8388607> : tensor<128x2x1xi32, #blocked5>
    %cst_13 = arith.constant dense<4.480000e+02> : tensor<128x2x1xf32, #blocked5>
    %cst_14 = arith.constant dense<-1.000000e+00> : tensor<128x64xf32, #blocked4>
    %c1_i64 = arith.constant 1 : i64
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c127_i32 = arith.constant 127 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_15 = arith.constant dense<4> : tensor<128x4xi32, #blocked1>
    %c2_i32 = arith.constant 2 : i32
    %c192_i32 = arith.constant 192 : i32
    %c6_i32 = arith.constant 6 : i32
    %cst_16 = arith.constant dense<0> : tensor<128x4xi8, #blocked1>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked6>
    %c8_i32 = arith.constant 8 : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cst_18 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear1>
    %2 = tt.make_tensor_descriptor %arg11, [%arg1, %arg2, %arg3, %arg4, %arg5], [%arg6, %arg7, %arg8, %arg9, %c1_i64] : !tt.ptr<f8E4M3FN>, !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>
    %3 = tt.addptr %arg59, %c10_i32 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = tt.get_program_id x : i32
    %6 = arith.subi %5, %c10_i32 : i32
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %9 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %13 = arith.extsi %arg29 : i32 to i64
    %14 = tt.splat %13 : i64 -> tensor<128x1xi64, #blocked1>
    %15 = tt.splat %arg27 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked1>
    %16 = arith.extsi %12 : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<4xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %17 = tt.expand_dims %16 {axis = 0 : i32} : tensor<4xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x4xi64, #blocked1>
    %18 = tt.broadcast %17 : tensor<1x4xi64, #blocked1> -> tensor<128x4xi64, #blocked1>
    %19 = arith.addi %arg54, %c127_i32 : i32
    %20 = arith.divsi %19, %c128_i32 : i32
    %21 = arith.maxsi %20, %c1_i32 : i32
    %22 = arith.cmpi sgt, %21, %c0_i32 : i32
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked6>
    %24 = tt.splat %arg53 : i32 -> tensor<256xi32, #blocked6>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %26 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %27 = tt.splat %arg53 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %28 = tt.splat %arg53 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %29 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %30 = arith.addi %arg53, %c31_i32 : i32
    %31 = arith.divsi %30, %c32_i32 : i32
    %32 = tt.splat %31 : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %33 = arith.extsi %arg18 : i32 to i64
    %34 = tt.splat %33 : i64 -> tensor<128x1xi64, #blocked3>
    %35 = tt.splat %arg15 : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked3>
    %36 = arith.subi %4, %5 : i32
    %37 = arith.ceildivsi %36, %c10_i32 : i32
    %38 = arith.maxsi %21, %c1_i32 : i32
    %39 = arith.muli %37, %38 : i32
    %40 = arith.subi %5, %c10_i32 : i32
    %41 = arith.addi %arg53, %c127_i32 : i32
    %42 = arith.divsi %41, %c128_i32 : i32
    %43 = arith.subi %38, %c1_i32 : i32
    %44 = arith.subi %38, %c1_i32 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %45 = nvws.semaphore.create %result true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %46 = nvws.semaphore.create %result false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %47 = nvws.semaphore.acquire %45 : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %48 = nvws.semaphore.buffer %45, %47 : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
    %49 = ttng.tmem_store %cst_18, %48[], %true : tensor<128x256xf32, #linear1> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
    %50 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>
    %51 = nvws.semaphore.create %50 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>
    %52 = nvws.semaphore.create %50 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>
    %54 = nvws.semaphore.create %53 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>
    %55 = nvws.semaphore.create %53 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>
    %56 = ttg.local_alloc : () -> !ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>
    %57 = nvws.semaphore.create %56 true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>
    %58 = nvws.semaphore.create %56 false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>
    %59 = nvws.semaphore.acquire %51 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]> -> !ttg.async.token
    %60 = nvws.semaphore.acquire %54 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]> -> !ttg.async.token
    %61 = nvws.semaphore.acquire %57 {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]> -> !ttg.async.token
    %62:16 = scf.for %arg63 = %c0_i32 to %39 step %c1_i32 iter_args(%arg64 = %c0_i32, %arg65 = %40, %arg66 = %6, %arg67 = %c0_i32, %arg68 = %1, %arg69 = %0, %arg70 = %c0_i32, %arg71 = %c0_i32, %arg72 = %cst_0, %arg73 = %cst, %arg74 = %1, %arg75 = %false, %arg76 = %47, %arg77 = %59, %arg78 = %60, %arg79 = %61) -> (i32, i32, i32, i32, tensor<128x4x!tt.ptr<i8>, #blocked1>, !tt.ptr<i32>, i32, i32, tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x4x!tt.ptr<i8>, #blocked1>, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %63 = arith.cmpi eq, %arg64, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %64 = arith.select %63, %c0_i32, %arg67 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %65:8 = scf.if %63 -> (!tt.ptr<i32>, i32, i32, tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x4x!tt.ptr<i8>, #blocked1>, tensor<128x4x!tt.ptr<i8>, #blocked1>, i32) {
        %106 = arith.addi %arg65, %c10_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %107 = arith.remsi %106, %4 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %108 = arith.divsi %107, %c8_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %109 = arith.muli %108, %c8_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %110 = arith.subi %4, %109 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %111 = arith.minsi %110, %c8_i32 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %112 = arith.cmpi sge, %111, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        llvm.intr.assume %112 : i1 {ttg.partition = array<i32: 0>}
        %113 = arith.remsi %107, %111 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %114 = arith.addi %109, %113 {ttg.partition = array<i32: 0, 1, 2>} : i32
        %115 = arith.remsi %107, %c8_i32 {ttg.partition = array<i32: 2>} : i32
        %116 = arith.divsi %115, %111 {ttg.partition = array<i32: 2>} : i32
        %117 = tt.addptr %arg60, %114 {ttg.partition = array<i32: 0, 1, 2>} : !tt.ptr<i32>, i32
        %118 = tt.load %117 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
        %119 = arith.andi %118, %c65535_i32 {ttg.partition = array<i32: 1, 2>} : i32
        %120 = arith.shrsi %118, %c16_i32 {ttg.partition = array<i32: 1, 2>} : i32
        %121 = tt.addptr %arg58, %119 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
        %122 = tt.load %121 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
        %123 = arith.muli %120, %c128_i32 {ttg.partition = array<i32: 1, 2>} : i32
        %124 = tt.addptr %arg57, %119 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
        %125 = tt.load %124 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
        %126 = arith.muli %116, %c256_i32 {ttg.partition = array<i32: 2>} : i32
        %127 = tt.splat %123 {ttg.partition = array<i32: 1>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %128 = tt.splat %123 {ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %129 = arith.addi %127, %7 {ttg.partition = array<i32: 1>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %130 = arith.addi %128, %11 {ttg.partition = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %131 = tt.splat %125 {ttg.partition = array<i32: 1>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %132 = tt.splat %125 {ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %133 = arith.cmpi slt, %129, %131 {ttg.partition = array<i32: 1>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %134 = arith.cmpi slt, %130, %132 {ttg.partition = array<i32: 2>} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %135 = arith.extsi %122 {ttg.partition = array<i32: 1, 2>} : i32 to i64
        %136 = tt.addptr %arg56, %135 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i64
        %137 = tt.splat %136 {ttg.partition = array<i32: 1>} : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %138 = tt.splat %136 {ttg.partition = array<i32: 2>} : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
        %139 = tt.addptr %137, %129 {ttg.partition = array<i32: 1>} : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %140 = tt.addptr %138, %130 {ttg.partition = array<i32: 2>} : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %141 = tt.load %139, %133, %cst_7 {ttg.partition = array<i32: 1>} : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %142 = tt.load %140, %134, %cst_6 {ttg.partition = array<i32: 2>} : tensor<128x!tt.ptr<i32>, #ttg.slice<{dim = 0, parent = #blocked}>>
        %143 = arith.extsi %141 {ttg.partition = array<i32: 1>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %144 = tt.expand_dims %143 {axis = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi64, #blocked1>
        %145 = arith.muli %144, %14 {ttg.partition = array<i32: 1>} : tensor<128x1xi64, #blocked1>
        %146 = tt.addptr %15, %145 {ttg.partition = array<i32: 1>} : tensor<128x1x!tt.ptr<i8>, #blocked1>, tensor<128x1xi64, #blocked1>
        %147 = tt.broadcast %146 {ttg.partition = array<i32: 1>} : tensor<128x1x!tt.ptr<i8>, #blocked1> -> tensor<128x4x!tt.ptr<i8>, #blocked1>
        %148 = tt.addptr %147, %18 {ttg.partition = array<i32: 1>} : tensor<128x4x!tt.ptr<i8>, #blocked1>, tensor<128x4xi64, #blocked1>
        llvm.intr.assume %22 : i1 {ttg.partition = array<i32: 0, 1, 2>}
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %117, %119, %126, %133, %142, %148, %148, %106 : !tt.ptr<i32>, i32, i32, tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x4x!tt.ptr<i8>, #blocked1>, tensor<128x4x!tt.ptr<i8>, #blocked1>, i32
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1, 2>} %arg69, %arg70, %arg71, %arg72, %arg73, %arg74, %arg68, %arg65 : !tt.ptr<i32>, i32, i32, tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x4x!tt.ptr<i8>, #blocked1>, tensor<128x4x!tt.ptr<i8>, #blocked1>, i32
      } {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 2>, array<i32: 1>, array<i32: 1>, array<i32: 0, 1, 2>]}
      %66 = arith.muli %64, %c128_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %67 = arith.muli %64, %c64_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %68 = nvws.semaphore.buffer %51, %arg77 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
      nvws.descriptor_gather %arg19[%65#4, %66] 16384 %68 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x128xf8E4M3FN, #shared1>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
      nvws.semaphore.release %52, %arg77 [#nvws.async_op<tma_load>] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>, !ttg.async.token
      %69 = tt.expand_dims %65#3 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi1, #blocked1>
      %70 = tt.broadcast %69 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<128x1xi1, #blocked1> -> tensor<128x4xi1, #blocked1>
      %71 = tt.load %65#6, %70, %cst_16 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : tensor<128x4x!tt.ptr<i8>, #blocked1>
      %72 = ttg.local_alloc %71 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x4xi8, #blocked1>) -> !ttg.memdesc<128x4xi8, #shared5, #smem>
      %73 = ttg.local_load %72 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x4xi8, #shared5, #smem> -> tensor<128x4xi8, #linear2>
      %74 = nvws.semaphore.buffer %54, %arg78 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xi8, #shared4, #smem, mutable>
      nvws.descriptor_load %arg30[%65#1, %65#2, %67] 16384 %74 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x256x64xui8, #shared2>>, i32, i32, i32, !ttg.memdesc<256x64xi8, #shared4, #smem, mutable>
      nvws.semaphore.release %55, %arg78 [#nvws.async_op<tma_load>] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>, !ttg.async.token
      %75 = arith.divsi %67, %c16_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %76 = arith.muli %65#1, %42 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %77 = arith.divsi %65#2, %c128_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %78 = arith.addi %76, %77 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %79 = arith.divsi %75, %c4_i32 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %80 = nvws.semaphore.buffer %57, %arg79 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1x2x1x2x256xi8, #shared3, #smem, mutable>
      nvws.descriptor_load %arg40[%c0_i32, %78, %79, %c0_i32, %c0_i32] 1024 %80 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x2x1x2x256xui8, #shared3>>, i32, i32, i32, i32, i32, !ttg.memdesc<1x2x1x2x256xi8, #shared3, #smem, mutable>
      nvws.semaphore.release %58, %arg79 [#nvws.async_op<tma_load>] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>, !ttg.async.token
      %result_19 = ttng.tmem_alloc %73 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<128x4xi8, #linear2>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      %81 = nvws.semaphore.buffer %45, %arg76 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
      %82 = nvws.semaphore.acquire %52 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]> -> !ttg.async.token
      %83 = nvws.semaphore.buffer %52, %82 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>
      %84 = nvws.semaphore.acquire %55 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]> -> !ttg.async.token
      %85 = nvws.semaphore.buffer %55, %84 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xi8, #shared4, #smem, mutable>
      %86 = ttg.memdesc_trans %85 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xi8, #shared4, #smem, mutable> -> !ttg.memdesc<64x256xi8, #shared6, #smem, mutable>
      %87 = nvws.semaphore.acquire %58 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]> -> !ttg.async.token
      %88 = nvws.semaphore.buffer %58, %87 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1x2x1x2x256xi8, #shared3, #smem, mutable>
      %89 = ttg.memdesc_reshape %88 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x2x1x2x256xi8, #shared3, #smem, mutable> -> !ttg.memdesc<2x1x32x4x4xi8, #shared7, #smem, mutable>
      %90 = ttg.memdesc_trans %89 {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 0, 3, 2, 1, 4>, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x1x32x4x4xi8, #shared7, #smem, mutable> -> !ttg.memdesc<2x4x32x1x4xi8, #shared8, #smem, mutable>
      %91 = ttg.memdesc_reshape %90 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x4x32x1x4xi8, #shared8, #smem, mutable> -> !ttg.memdesc<256x4xi8, #shared9, #smem, mutable>
      %92 = ttng.tc_gen5_mma_scaled %83, %86, %81[], %result_19, %91, %arg75, %true lhs = e4m3 rhs = e2m1 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, mutable>, !ttg.memdesc<64x256xi8, #shared6, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<256x4xi8, #shared9, #smem, mutable>
      nvws.semaphore.release %57, %87 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %54, %84 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]>, !ttg.async.token
      nvws.semaphore.release %51, %82 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]>, !ttg.async.token
      %93 = tt.addptr %65#6, %cst_15 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<128x4x!tt.ptr<i8>, #blocked1>, tensor<128x4xi32, #blocked1>
      %94 = arith.addi %64, %c1_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %95 = arith.cmpi eq, %arg64, %43 {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %96 = arith.select %95, %false, %true {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      scf.if %95 {
        nvws.semaphore.release %46, %arg76 [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      %97 = ub.poison : !ttg.async.token
      %98:2 = scf.if %95 -> (i32, !ttg.async.token) {
        %106 = arith.addi %arg66, %c10_i32 {ttg.partition = array<i32: 0>} : i32
        %107 = arith.remsi %106, %4 {ttg.partition = array<i32: 0>} : i32
        %108 = arith.divsi %107, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %109 = arith.muli %108, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %110 = arith.subi %4, %109 {ttg.partition = array<i32: 0>} : i32
        %111 = arith.minsi %110, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %112 = arith.cmpi sge, %111, %c0_i32 {ttg.partition = array<i32: 0>} : i32
        llvm.intr.assume %112 : i1 {ttg.partition = array<i32: 0>}
        %113 = arith.remsi %107, %c8_i32 {ttg.partition = array<i32: 0>} : i32
        %114 = arith.divsi %113, %111 {ttg.partition = array<i32: 0>} : i32
        %115 = tt.load %65#0 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %116 = arith.andi %115, %c65535_i32 {ttg.partition = array<i32: 0>} : i32
        %117 = arith.shrsi %115, %c16_i32 {ttg.partition = array<i32: 0>} : i32
        %118 = tt.addptr %arg58, %116 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %119 = tt.load %118 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %120 = arith.muli %117, %c128_i32 {ttg.partition = array<i32: 0>} : i32
        %121 = arith.muli %114, %c256_i32 {ttg.partition = array<i32: 0>} : i32
        %122 = tt.addptr %arg57, %116 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>, i32
        %123 = tt.load %122 {ttg.partition = array<i32: 0>} : !tt.ptr<i32>
        %124 = tt.splat %120 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
        %125 = tt.splat %120 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
        %126 = tt.splat %120 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %127 = arith.addi %124, %8 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
        %128 = arith.addi %125, %9 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
        %129 = arith.addi %126, %10 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %130 = tt.splat %123 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
        %131 = tt.splat %123 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
        %132 = tt.splat %123 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %133 = arith.cmpi slt, %127, %130 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
        %134 = arith.cmpi slt, %128, %131 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
        %135 = arith.cmpi slt, %129, %132 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %136 = tt.splat %121 {ttg.partition = array<i32: 0>} : i32 -> tensor<256xi32, #blocked6>
        %137 = arith.addi %136, %23 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked6>
        %138 = arith.cmpi slt, %137, %24 {ttg.partition = array<i32: 0>} : tensor<256xi32, #blocked6>
        %139 = arith.muli %116, %arg52 {ttg.partition = array<i32: 0>} : i32
        %140 = tt.addptr %arg51, %139 {ttg.partition = array<i32: 0>} : !tt.ptr<f32>, i32
        %141 = tt.splat %140 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked6>
        %142 = tt.addptr %141, %137 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked6>, tensor<256xi32, #blocked6>
        %143 = tt.load %142, %138, %cst_17 {ttg.partition = array<i32: 0>} : tensor<256x!tt.ptr<f32>, #blocked6>
        %144 = nvws.semaphore.acquire %46 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %145 = nvws.semaphore.buffer %46, %144 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256>
        %result_20, %token = ttng.tmem_load %145[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x256> -> tensor<128x256xf32, #linear1>
        nvws.semaphore.release %45, %144 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %146 = tt.reshape %result_20 {ttg.partition = array<i32: 0>} : tensor<128x256xf32, #linear1> -> tensor<128x2x128xf32, #linear3>
        %147 = tt.trans %146 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x128xf32, #linear3> -> tensor<128x128x2xf32, #linear4>
        %148 = ttg.convert_layout %147 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #linear4> -> tensor<128x128x2xf32, #linear5>
        %149 = ttg.convert_layout %147 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #linear4> -> tensor<128x128x2xf32, #linear6>
        %outLHS, %outRHS = tt.split %148 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #linear5> -> tensor<128x128xf32, #linear7>
        %outLHS_21, %outRHS_22 = tt.split %149 {ttg.partition = array<i32: 0>} : tensor<128x128x2xf32, #linear6> -> tensor<128x128xf32, #linear8>
        %150 = tt.reshape %143 {ttg.partition = array<i32: 0>} : tensor<256xf32, #blocked6> -> tensor<2x128xf32, #blocked7>
        %151 = tt.trans %150 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x128xf32, #blocked7> -> tensor<128x2xf32, #blocked8>
        %152 = ttg.convert_layout %151 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #blocked8> -> tensor<128x2xf32, #linear9>
        %153 = ttg.convert_layout %151 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #blocked8> -> tensor<128x2xf32, #linear10>
        %outLHS_23, %outRHS_24 = tt.split %152 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #linear9> -> tensor<128xf32, #linear11>
        %outLHS_25, %outRHS_26 = tt.split %153 {ttg.partition = array<i32: 0>} : tensor<128x2xf32, #linear10> -> tensor<128xf32, #linear12>
        %154 = tt.reshape %outLHS {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear7> -> tensor<128x2x64xf32, #blocked9>
        %155 = tt.reshape %outLHS_21 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8> -> tensor<128x2x64xf32, #linear13>
        %156 = tt.trans %154 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #blocked9> -> tensor<128x64x2xf32, #blocked10>
        %157 = tt.trans %155 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear13> -> tensor<128x64x2xf32, #linear14>
        %158 = tt.reshape %outRHS {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear7> -> tensor<128x2x64xf32, #blocked9>
        %159 = tt.reshape %outRHS_22 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear8> -> tensor<128x2x64xf32, #linear13>
        %160 = tt.trans %158 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #blocked9> -> tensor<128x64x2xf32, #blocked10>
        %161 = tt.trans %159 {order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear13> -> tensor<128x64x2xf32, #linear14>
        %outLHS_27, %outRHS_28 = tt.split %156 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #blocked10> -> tensor<128x64xf32, #blocked4>
        %outLHS_29, %outRHS_30 = tt.split %157 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear14> -> tensor<128x64xf32, #linear>
        %outLHS_31, %outRHS_32 = tt.split %160 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #blocked10> -> tensor<128x64xf32, #blocked4>
        %outLHS_33, %outRHS_34 = tt.split %161 {ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear14> -> tensor<128x64xf32, #linear>
        %162 = tt.reshape %outLHS_23 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear11> -> tensor<2x64xf32, #linear15>
        %163 = tt.reshape %outLHS_25 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear12> -> tensor<2x64xf32, #linear16>
        %164 = tt.trans %162 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x64xf32, #linear15> -> tensor<64x2xf32, #linear17>
        %165 = tt.trans %163 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x64xf32, #linear16> -> tensor<64x2xf32, #linear18>
        %outLHS_35, %outRHS_36 = tt.split %164 {ttg.partition = array<i32: 0>} : tensor<64x2xf32, #linear17> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %outLHS_37, %outRHS_38 = tt.split %165 {ttg.partition = array<i32: 0>} : tensor<64x2xf32, #linear18> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %166 = tt.reshape %outRHS_24 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear11> -> tensor<2x64xf32, #linear15>
        %167 = tt.reshape %outRHS_26 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear12> -> tensor<2x64xf32, #linear16>
        %168 = tt.trans %166 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x64xf32, #linear15> -> tensor<64x2xf32, #linear17>
        %169 = tt.trans %167 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<2x64xf32, #linear16> -> tensor<64x2xf32, #linear18>
        %outLHS_39, %outRHS_40 = tt.split %168 {ttg.partition = array<i32: 0>} : tensor<64x2xf32, #linear17> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %outLHS_41, %outRHS_42 = tt.split %169 {ttg.partition = array<i32: 0>} : tensor<64x2xf32, #linear18> -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>>
        %170 = tt.expand_dims %outLHS_35 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xf32, #blocked4>
        %171 = tt.expand_dims %outLHS_37 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %172 = tt.broadcast %170 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #blocked4> -> tensor<128x64xf32, #blocked4>
        %173 = tt.broadcast %171 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #linear> -> tensor<128x64xf32, #linear>
        %174 = arith.addf %outLHS_27, %172 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %175 = arith.addf %outLHS_29, %173 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %176 = tt.splat %121 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %177 = tt.splat %121 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %178 = arith.addi %176, %25 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %179 = arith.addi %177, %26 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %180 = arith.cmpi slt, %178, %27 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %181 = arith.cmpi slt, %179, %28 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %182 = tt.expand_dims %133 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<128x1xi1, #blocked4>
        %183 = tt.expand_dims %134 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi1, #linear>
        %184 = tt.expand_dims %135 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi1, #blocked3>
        %185 = tt.expand_dims %180 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi1, #blocked4>
        %186 = tt.expand_dims %181 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi1, #linear>
        %187 = tt.broadcast %182 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
        %188 = tt.broadcast %183 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #linear> -> tensor<128x64xi1, #linear>
        %189 = tt.broadcast %185 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
        %190 = tt.broadcast %186 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
        %191 = arith.andi %187, %189 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>
        %192 = arith.andi %188, %190 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>
        %193 = math.absf %174 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %194 = math.absf %175 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %195 = arith.select %191, %193, %cst_14 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %196 = arith.select %192, %194, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>, tensor<128x64xf32, #linear>
        %197 = tt.reshape %195 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %198 = tt.reshape %196 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear> -> tensor<128x2x32xf32, #blocked2>
        %199 = "tt.reduce"(%197) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked5>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>>
        %200 = "tt.reduce"(%198) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked2>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
        %201 = tt.expand_dims %199 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<128x2x1xf32, #blocked5>
        %202 = tt.expand_dims %200 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<128x2x1xf32, #blocked2>
        %203 = arith.divf %201, %cst_13 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %204 = arith.divf %202, %cst_2 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2>
        %205 = tt.bitcast %203 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x1xi32, #blocked5>
        %206 = tt.bitcast %204 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2> -> tensor<128x2x1xi32, #blocked2>
        %207 = arith.addi %205, %cst_12 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %208 = arith.addi %206, %cst_3 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %209 = arith.andi %207, %cst_11 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %210 = arith.andi %208, %cst_4 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %211 = tt.bitcast %209 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5> -> tensor<128x2x1xf32, #blocked5>
        %212 = arith.cmpf oeq, %211, %cst_9 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %213 = arith.divf %cst_10, %211 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %214 = arith.select %212, %cst_9, %213 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi1, #blocked5>, tensor<128x2x1xf32, #blocked5>
        %215 = tt.reshape %174 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %216 = tt.broadcast %214 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x32xf32, #blocked5>
        %217 = arith.mulf %215, %216 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5>
        %218 = tt.reshape %217 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x64xf32, #blocked4>
        %219 = arith.select %191, %218, %cst_8 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %220 = tt.reshape %210 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2> -> tensor<128x2xi32, #blocked3>
        %221 = arith.shrui %220, %cst_5 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3>
        %222 = arith.trunci %221 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3> to tensor<128x2xi8, #blocked3>
        %223 = tt.fp_to_fp %219, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
        %224 = arith.divsi %121, %c32_i32 {ttg.partition = array<i32: 0>} : i32
        %225 = tt.splat %224 {ttg.partition = array<i32: 0>} : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %226 = arith.addi %225, %29 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %227 = arith.cmpi slt, %226, %32 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %228 = arith.addi %119, %120 {ttg.partition = array<i32: 0>} : i32
        %229 = tt.splat %228 {ttg.partition = array<i32: 0>} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %230 = arith.addi %229, %10 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %231 = arith.extsi %230 {ttg.partition = array<i32: 0>} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %232 = tt.expand_dims %231 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi64, #blocked3>
        %233 = arith.muli %232, %34 {ttg.partition = array<i32: 0>} : tensor<128x1xi64, #blocked3>
        %234 = tt.addptr %35, %233 {ttg.partition = array<i32: 0>} : tensor<128x1x!tt.ptr<i8>, #blocked3>, tensor<128x1xi64, #blocked3>
        %235 = arith.extsi %226 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %236 = tt.expand_dims %235 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
        %237 = tt.broadcast %234 {ttg.partition = array<i32: 0>} : tensor<128x1x!tt.ptr<i8>, #blocked3> -> tensor<128x2x!tt.ptr<i8>, #blocked3>
        %238 = tt.broadcast %236 {ttg.partition = array<i32: 0>} : tensor<1x2xi64, #blocked3> -> tensor<128x2xi64, #blocked3>
        %239 = tt.addptr %237, %238 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>, tensor<128x2xi64, #blocked3>
        %240 = tt.expand_dims %227 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi1, #blocked3>
        %241 = tt.broadcast %184 {ttg.partition = array<i32: 0>} : tensor<128x1xi1, #blocked3> -> tensor<128x2xi1, #blocked3>
        %242 = tt.broadcast %240 {ttg.partition = array<i32: 0>} : tensor<1x2xi1, #blocked3> -> tensor<128x2xi1, #blocked3>
        %243 = arith.andi %241, %242 {ttg.partition = array<i32: 0>} : tensor<128x2xi1, #blocked3>
        tt.store %239, %222, %243 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>
        %244 = arith.subi %c1073741824_i32, %123 {ttg.partition = array<i32: 0>} : i32
        %245 = arith.addi %244, %120 {ttg.partition = array<i32: 0>} : i32
        %246 = arith.addi %119, %123 {ttg.partition = array<i32: 0>} : i32
        %247 = tt.reshape %223 {ttg.partition = array<i32: 0>} : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        tt.descriptor_store %2[%c1073741824_i32, %246, %c0_i32, %245, %121], %247 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>, tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        %248 = tt.expand_dims %outRHS_36 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xf32, #blocked4>
        %249 = tt.expand_dims %outRHS_38 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %250 = tt.broadcast %248 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #blocked4> -> tensor<128x64xf32, #blocked4>
        %251 = tt.broadcast %249 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #linear> -> tensor<128x64xf32, #linear>
        %252 = arith.addf %outRHS_28, %250 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %253 = arith.addf %outRHS_30, %251 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %254 = arith.addi %121, %c64_i32 {ttg.partition = array<i32: 0>} : i32
        %255 = tt.splat %254 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %256 = tt.splat %254 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %257 = arith.addi %255, %25 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %258 = arith.addi %256, %26 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %259 = arith.cmpi slt, %257, %27 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %260 = arith.cmpi slt, %258, %28 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %261 = tt.expand_dims %259 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi1, #blocked4>
        %262 = tt.expand_dims %260 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi1, #linear>
        %263 = tt.broadcast %261 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
        %264 = tt.broadcast %262 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
        %265 = arith.andi %187, %263 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>
        %266 = arith.andi %188, %264 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>
        %267 = math.absf %252 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %268 = math.absf %253 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %269 = arith.select %265, %267, %cst_14 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %270 = arith.select %266, %268, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>, tensor<128x64xf32, #linear>
        %271 = tt.reshape %269 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %272 = tt.reshape %270 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear> -> tensor<128x2x32xf32, #blocked2>
        %273 = "tt.reduce"(%271) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked5>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>>
        %274 = "tt.reduce"(%272) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked2>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
        %275 = tt.expand_dims %273 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<128x2x1xf32, #blocked5>
        %276 = tt.expand_dims %274 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<128x2x1xf32, #blocked2>
        %277 = arith.divf %275, %cst_13 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %278 = arith.divf %276, %cst_2 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2>
        %279 = tt.bitcast %277 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x1xi32, #blocked5>
        %280 = tt.bitcast %278 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2> -> tensor<128x2x1xi32, #blocked2>
        %281 = arith.addi %279, %cst_12 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %282 = arith.addi %280, %cst_3 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %283 = arith.andi %281, %cst_11 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %284 = arith.andi %282, %cst_4 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %285 = tt.bitcast %283 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5> -> tensor<128x2x1xf32, #blocked5>
        %286 = arith.cmpf oeq, %285, %cst_9 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %287 = arith.divf %cst_10, %285 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %288 = arith.select %286, %cst_9, %287 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi1, #blocked5>, tensor<128x2x1xf32, #blocked5>
        %289 = tt.reshape %252 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %290 = tt.broadcast %288 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x32xf32, #blocked5>
        %291 = arith.mulf %289, %290 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5>
        %292 = tt.reshape %291 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x64xf32, #blocked4>
        %293 = arith.select %265, %292, %cst_8 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %294 = tt.reshape %284 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2> -> tensor<128x2xi32, #blocked3>
        %295 = arith.shrui %294, %cst_5 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3>
        %296 = arith.trunci %295 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3> to tensor<128x2xi8, #blocked3>
        %297 = tt.fp_to_fp %293, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
        %298 = arith.addi %224, %c2_i32 {ttg.partition = array<i32: 0>} : i32
        %299 = tt.splat %298 {ttg.partition = array<i32: 0>} : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %300 = arith.addi %299, %29 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %301 = arith.cmpi slt, %300, %32 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %302 = arith.extsi %300 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %303 = tt.expand_dims %302 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
        %304 = tt.broadcast %303 {ttg.partition = array<i32: 0>} : tensor<1x2xi64, #blocked3> -> tensor<128x2xi64, #blocked3>
        %305 = tt.addptr %237, %304 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>, tensor<128x2xi64, #blocked3>
        %306 = tt.expand_dims %301 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi1, #blocked3>
        %307 = tt.broadcast %306 {ttg.partition = array<i32: 0>} : tensor<1x2xi1, #blocked3> -> tensor<128x2xi1, #blocked3>
        %308 = arith.andi %241, %307 {ttg.partition = array<i32: 0>} : tensor<128x2xi1, #blocked3>
        tt.store %305, %296, %308 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>
        %309 = tt.reshape %297 {ttg.partition = array<i32: 0>} : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        tt.descriptor_store %2[%c1073741824_i32, %246, %c0_i32, %245, %254], %309 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>, tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        %310 = tt.expand_dims %outLHS_39 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xf32, #blocked4>
        %311 = tt.expand_dims %outLHS_41 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %312 = tt.broadcast %310 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #blocked4> -> tensor<128x64xf32, #blocked4>
        %313 = tt.broadcast %311 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #linear> -> tensor<128x64xf32, #linear>
        %314 = arith.addf %outLHS_31, %312 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %315 = arith.addf %outLHS_33, %313 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %316 = arith.addi %121, %c128_i32 {ttg.partition = array<i32: 0>} : i32
        %317 = tt.splat %316 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %318 = tt.splat %316 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %319 = arith.addi %317, %25 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %320 = arith.addi %318, %26 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %321 = arith.cmpi slt, %319, %27 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %322 = arith.cmpi slt, %320, %28 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %323 = tt.expand_dims %321 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi1, #blocked4>
        %324 = tt.expand_dims %322 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi1, #linear>
        %325 = tt.broadcast %323 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
        %326 = tt.broadcast %324 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
        %327 = arith.andi %187, %325 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>
        %328 = arith.andi %188, %326 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>
        %329 = math.absf %314 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %330 = math.absf %315 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %331 = arith.select %327, %329, %cst_14 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %332 = arith.select %328, %330, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>, tensor<128x64xf32, #linear>
        %333 = tt.reshape %331 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %334 = tt.reshape %332 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear> -> tensor<128x2x32xf32, #blocked2>
        %335 = "tt.reduce"(%333) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked5>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>>
        %336 = "tt.reduce"(%334) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked2>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
        %337 = tt.expand_dims %335 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<128x2x1xf32, #blocked5>
        %338 = tt.expand_dims %336 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<128x2x1xf32, #blocked2>
        %339 = arith.divf %337, %cst_13 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %340 = arith.divf %338, %cst_2 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2>
        %341 = tt.bitcast %339 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x1xi32, #blocked5>
        %342 = tt.bitcast %340 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2> -> tensor<128x2x1xi32, #blocked2>
        %343 = arith.addi %341, %cst_12 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %344 = arith.addi %342, %cst_3 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %345 = arith.andi %343, %cst_11 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %346 = arith.andi %344, %cst_4 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %347 = tt.bitcast %345 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5> -> tensor<128x2x1xf32, #blocked5>
        %348 = arith.cmpf oeq, %347, %cst_9 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %349 = arith.divf %cst_10, %347 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %350 = arith.select %348, %cst_9, %349 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi1, #blocked5>, tensor<128x2x1xf32, #blocked5>
        %351 = tt.reshape %314 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %352 = tt.broadcast %350 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x32xf32, #blocked5>
        %353 = arith.mulf %351, %352 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5>
        %354 = tt.reshape %353 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x64xf32, #blocked4>
        %355 = arith.select %327, %354, %cst_8 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %356 = tt.reshape %346 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2> -> tensor<128x2xi32, #blocked3>
        %357 = arith.shrui %356, %cst_5 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3>
        %358 = arith.trunci %357 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3> to tensor<128x2xi8, #blocked3>
        %359 = tt.fp_to_fp %355, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
        %360 = arith.addi %224, %c4_i32 {ttg.partition = array<i32: 0>} : i32
        %361 = tt.splat %360 {ttg.partition = array<i32: 0>} : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %362 = arith.addi %361, %29 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %363 = arith.cmpi slt, %362, %32 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %364 = arith.extsi %362 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %365 = tt.expand_dims %364 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
        %366 = tt.broadcast %365 {ttg.partition = array<i32: 0>} : tensor<1x2xi64, #blocked3> -> tensor<128x2xi64, #blocked3>
        %367 = tt.addptr %237, %366 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>, tensor<128x2xi64, #blocked3>
        %368 = tt.expand_dims %363 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi1, #blocked3>
        %369 = tt.broadcast %368 {ttg.partition = array<i32: 0>} : tensor<1x2xi1, #blocked3> -> tensor<128x2xi1, #blocked3>
        %370 = arith.andi %241, %369 {ttg.partition = array<i32: 0>} : tensor<128x2xi1, #blocked3>
        tt.store %367, %358, %370 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>
        %371 = tt.reshape %359 {ttg.partition = array<i32: 0>} : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        tt.descriptor_store %2[%c1073741824_i32, %246, %c0_i32, %245, %316], %371 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>, tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        %372 = tt.expand_dims %outRHS_40 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xf32, #blocked4>
        %373 = tt.expand_dims %outRHS_42 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %374 = tt.broadcast %372 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #blocked4> -> tensor<128x64xf32, #blocked4>
        %375 = tt.broadcast %373 {ttg.partition = array<i32: 0>} : tensor<1x64xf32, #linear> -> tensor<128x64xf32, #linear>
        %376 = arith.addf %outRHS_32, %374 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %377 = arith.addf %outRHS_34, %375 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %378 = arith.addi %121, %c192_i32 {ttg.partition = array<i32: 0>} : i32
        %379 = tt.splat %378 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %380 = tt.splat %378 {ttg.partition = array<i32: 0>} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %381 = arith.addi %379, %25 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %382 = arith.addi %380, %26 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %383 = arith.cmpi slt, %381, %27 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
        %384 = arith.cmpi slt, %382, %28 {ttg.partition = array<i32: 0>} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %385 = tt.expand_dims %383 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x64xi1, #blocked4>
        %386 = tt.expand_dims %384 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<64xi1, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi1, #linear>
        %387 = tt.broadcast %385 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #blocked4> -> tensor<128x64xi1, #blocked4>
        %388 = tt.broadcast %386 {ttg.partition = array<i32: 0>} : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
        %389 = arith.andi %187, %387 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>
        %390 = arith.andi %188, %388 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>
        %391 = math.absf %376 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4>
        %392 = math.absf %377 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear>
        %393 = arith.select %389, %391, %cst_14 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %394 = arith.select %390, %392, %cst_1 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #linear>, tensor<128x64xf32, #linear>
        %395 = tt.reshape %393 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %396 = tt.reshape %394 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear> -> tensor<128x2x32xf32, #blocked2>
        %397 = "tt.reduce"(%395) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked5>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>>
        %398 = "tt.reduce"(%396) <{axis = 2 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%arg80: f32, %arg81: f32):
          %434 = arith.maxnumf %arg80, %arg81 {ttg.partition = array<i32: 0>} : f32
          tt.reduce.return %434 {ttg.partition = array<i32: 0>} : f32
        }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<128x2x32xf32, #blocked2>) -> tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>>
        %399 = tt.expand_dims %397 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<128x2x1xf32, #blocked5>
        %400 = tt.expand_dims %398 {axis = 2 : i32, ttg.partition = array<i32: 0>} : tensor<128x2xf32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<128x2x1xf32, #blocked2>
        %401 = arith.divf %399, %cst_13 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %402 = arith.divf %400, %cst_2 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2>
        %403 = tt.bitcast %401 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x1xi32, #blocked5>
        %404 = tt.bitcast %402 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked2> -> tensor<128x2x1xi32, #blocked2>
        %405 = arith.addi %403, %cst_12 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %406 = arith.addi %404, %cst_3 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %407 = arith.andi %405, %cst_11 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5>
        %408 = arith.andi %406, %cst_4 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2>
        %409 = tt.bitcast %407 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked5> -> tensor<128x2x1xf32, #blocked5>
        %410 = arith.cmpf oeq, %409, %cst_9 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %411 = arith.divf %cst_10, %409 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5>
        %412 = arith.select %410, %cst_9, %411 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi1, #blocked5>, tensor<128x2x1xf32, #blocked5>
        %413 = tt.reshape %376 {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x2x32xf32, #blocked5>
        %414 = tt.broadcast %412 {ttg.partition = array<i32: 0>} : tensor<128x2x1xf32, #blocked5> -> tensor<128x2x32xf32, #blocked5>
        %415 = arith.mulf %413, %414 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5>
        %416 = tt.reshape %415 {ttg.partition = array<i32: 0>} : tensor<128x2x32xf32, #blocked5> -> tensor<128x64xf32, #blocked4>
        %417 = arith.select %389, %416, %cst_8 {ttg.partition = array<i32: 0>} : tensor<128x64xi1, #blocked4>, tensor<128x64xf32, #blocked4>
        %418 = tt.reshape %408 {ttg.partition = array<i32: 0>} : tensor<128x2x1xi32, #blocked2> -> tensor<128x2xi32, #blocked3>
        %419 = arith.shrui %418, %cst_5 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3>
        %420 = arith.trunci %419 {ttg.partition = array<i32: 0>} : tensor<128x2xi32, #blocked3> to tensor<128x2xi8, #blocked3>
        %421 = tt.fp_to_fp %417, rounding = rtne {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked4> -> tensor<128x64xf8E4M3FN, #blocked4>
        %422 = arith.addi %224, %c6_i32 {ttg.partition = array<i32: 0>} : i32
        %423 = tt.splat %422 {ttg.partition = array<i32: 0>} : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %424 = arith.addi %423, %29 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %425 = arith.cmpi slt, %424, %32 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %426 = arith.extsi %424 {ttg.partition = array<i32: 0>} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %427 = tt.expand_dims %426 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi64, #blocked3>
        %428 = tt.broadcast %427 {ttg.partition = array<i32: 0>} : tensor<1x2xi64, #blocked3> -> tensor<128x2xi64, #blocked3>
        %429 = tt.addptr %237, %428 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>, tensor<128x2xi64, #blocked3>
        %430 = tt.expand_dims %425 {axis = 0 : i32, ttg.partition = array<i32: 0>} : tensor<2xi1, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x2xi1, #blocked3>
        %431 = tt.broadcast %430 {ttg.partition = array<i32: 0>} : tensor<1x2xi1, #blocked3> -> tensor<128x2xi1, #blocked3>
        %432 = arith.andi %241, %431 {ttg.partition = array<i32: 0>} : tensor<128x2xi1, #blocked3>
        tt.store %429, %420, %432 {ttg.partition = array<i32: 0>} : tensor<128x2x!tt.ptr<i8>, #blocked3>
        %433 = tt.reshape %421 {ttg.partition = array<i32: 0>} : tensor<128x64xf8E4M3FN, #blocked4> -> tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        tt.descriptor_store %2[%c1073741824_i32, %246, %c0_i32, %245, %378], %433 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<1x1x1x128x64xf8E4M3FN, #shared>>, tensor<1x1x1x128x64xf8E4M3FN, #blocked11>
        scf.yield {ttg.partition = array<i32: 0, 1>} %106, %97 : i32, !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %arg66, %97 : i32, !ttg.async.token
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
      %99 = scf.if %95 -> (!ttg.async.token) {
        %106 = nvws.semaphore.acquire %45 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x256xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield {ttg.partition = array<i32: 1>} %106 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 1>} %arg76 : !ttg.async.token
      } {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      %100 = arith.addi %arg64, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %101 = arith.cmpi eq, %arg64, %44 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %102 = arith.select %101, %c0_i32, %100 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0, 1, 2>} : i32
      %103 = nvws.semaphore.acquire %51 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared1, #smem, mutable>]> -> !ttg.async.token
      %104 = nvws.semaphore.acquire %54 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x64xi8, #shared4, #smem, mutable>]> -> !ttg.async.token
      %105 = nvws.semaphore.acquire %57 {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1x2x1x2x256xi8, #shared3, #smem, mutable>]> -> !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %102, %65#7, %98#0, %94, %93, %65#0, %65#1, %65#2, %65#3, %65#4, %65#5, %96, %99, %103, %104, %105 : i32, i32, i32, i32, tensor<128x4x!tt.ptr<i8>, #blocked1>, !tt.ptr<i32>, i32, i32, tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x4x!tt.ptr<i8>, #blocked1>, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0>, array<i32: 2>, array<i32: 1>, array<i32: 0>, array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 2>, array<i32: 1>, array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

