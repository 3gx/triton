// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -nvws-insert-tmem-aref -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      %10 = ttng.tmem_store %cst, %result[%9], %8 {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
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
    %1:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst_1, %arg7 = %cst, %arg8 = %token, %arg9 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg8], %false, %true {ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%5] {ttg.partition = 0 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %6 = "compute_row_max"(%result_6, %arg3) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 {ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg7, %6 {ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = arith.subf %arg7, %6 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = math.exp2 %9 {ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = math.exp2 %10 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %26 = arith.addf %arg10, %arg11 : f32
        tt.reduce.return %26 : f32
      }) {ttg.partition = 0 : i32} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.mulf %arg6, %12 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.addf %14, %13 {ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = tt.expand_dims %11 {axis = 1 : i32, ttg.partition = 3 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %17 = tt.expand_dims %12 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %18 = tt.broadcast %16 {ttg.partition = 3 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %19 = tt.broadcast %17 {ttg.partition = 0 : i32} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg9] {ttg.partition = 3 : i32} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %20 = arith.mulf %result_8, %18 {ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked>
      %21 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %22 = ttg.local_alloc %21 {ttg.partition = 2 : i32} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %23 = arith.truncf %8 {ttg.partition = 0 : i32} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %result_10 = ttng.tmem_alloc %23 {ttg.partition = 0 : i32} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>
      %24 = ttng.tmem_store %20, %result_2[%token_9], %true {ttg.partition = 3 : i32} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %25 = ttng.tc_gen5_mma %result_10, %22, %result_2[%24], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %15, %6, %token_7, %25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>
module attributes {ttg.maxnreg = 80 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_16 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_18 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_19 = arith.constant dense<-1.000000e+06> : tensor<64x128xf32, #blocked>
    %start_m = tt.get_program_id x : i32
    %off_hz = tt.get_program_id y : i32
    %off_z = arith.divsi %off_hz, %H : i32
    %off_h = arith.remsi %off_hz, %H : i32
    %offset_y = arith.muli %N_CTX, %H : i32
    %offset_y_20 = arith.muli %off_z, %offset_y : i32
    %offset_y_21 = arith.muli %off_h, %N_CTX : i32
    %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 : i32
    %qo_offset_y = arith.muli %start_m, %c64_i32 : i32
    %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y : i32
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %offs_m_25 = tt.splat %qo_offset_y : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_26 = tt.splat %qo_offset_y : i32 -> tensor<64xi32, #blocked1>
    %offs_m_27 = arith.addi %offs_m_25, %offs_m : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_28 = arith.addi %offs_m_26, %offs_m_24 : tensor<64xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst_18 : f32
    %q = tt.descriptor_load %desc_q[%qo_offset_y_23, %c0_i32] : !tt.tensordesc<tensor<64x128xf8E5M2, #shared>> -> tensor<64x128xf8E5M2, #blocked2>
    %q_29 = ttg.local_alloc %q : (tensor<64x128xf8E5M2, #blocked2>) -> !ttg.memdesc<64x128xf8E5M2, #shared, #smem>
    %offsetv_y = arith.muli %offset_y_22, %c128_i32 : i32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<64x128xf32, #blocked>
    %qk_30, %qk_31 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc, %acc_32 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_33 = ttng.tmem_store %cst_17, %acc[%acc_32], %true : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetv_y_34:6 = scf.for %offsetv_y_57 = %c0_i32 to %qo_offset_y step %c128_i32 iter_args(%arg26 = %cst, %arg27 = %cst_16, %offset_y_58 = %offset_y_22, %offsetv_y_59 = %offsetv_y, %qk_60 = %qk_31, %acc_61 = %acc_33) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %k = tt.descriptor_load %desc_k[%offset_y_58, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %k_62 = ttg.local_alloc %k {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %k_63 = ttg.memdesc_trans %k_62 {loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %qk_64 = ttng.tc_gen5_mma %q_29, %k_63, %qk_30[%qk_60], %false, %true {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %qk_65, %qk_66 = ttng.tmem_load %qk_30[%qk_64] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %m_ij_67 = "tt.reduce"(%qk_65) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_93: f32, %m_ij_94: f32):
        %m_ij_95 = arith.maxnumf %m_ij_93, %m_ij_94 : f32
        tt.reduce.return %m_ij_95 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_68 = arith.mulf %m_ij_67, %m_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_69 = arith.maxnumf %arg27, %m_ij_68 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_70 = arith.mulf %qk_65, %qk {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %qk_71 = tt.expand_dims %m_ij_69 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %qk_72 = tt.broadcast %qk_71 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %qk_73 = arith.subf %qk_70, %qk_72 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %p = math.exp2 %qk_73 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %alpha = arith.subf %arg27, %m_ij_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_74 = arith.subf %arg27, %m_ij_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_75 = math.exp2 %alpha {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_76 = math.exp2 %alpha_74 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_93: f32, %l_ij_94: f32):
        %l_ij_95 = arith.addf %l_ij_93, %l_ij_94 : f32
        tt.reduce.return %l_ij_95 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_77 = tt.expand_dims %alpha_75 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_78 = tt.expand_dims %alpha_76 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_79 = tt.broadcast %acc_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_80 = tt.broadcast %acc_78 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_81, %acc_82 = ttng.tmem_load %acc[%acc_61] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %acc_83 = arith.mulf %acc_81, %acc_79 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked>
      %v = tt.descriptor_load %desc_v[%c0_i32, %offsetv_y_59] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %v_84 = ttg.local_alloc %v {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %v_85 = ttg.memdesc_trans %v_84 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %p_86 = tt.fp_to_fp %p {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
      %acc_87 = ttng.tmem_alloc %p_86 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf8E5M2, #blocked>) -> !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      %acc_88 = ttng.tmem_store %acc_83, %acc[%acc_82], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_89 = ttng.tc_gen5_mma %acc_87, %v_85, %acc[%acc_88], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %l_i = arith.mulf %arg26, %alpha_76 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_90 = arith.addf %l_i, %l_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_91 = arith.addi %offset_y_58, %c128_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %offsetv_y_92 = arith.addi %offsetv_y_59, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32} : i32
      scf.yield %l_i_90, %m_ij_69, %offsetk_y_91, %offsetv_y_92, %qk_66, %acc_89 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    %acc_35, %acc_36 = ttng.tmem_load %acc[%offsetv_y_34#5] : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
    %0 = arith.muli %start_m, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
    %1 = arith.addi %start_m, %c1_i32 : i32
    %2 = arith.muli %1, %c64_i32 : i32
    %offsetk_y = arith.addi %offset_y_22, %0 : i32
    %offsetv_y_37 = arith.addi %offsetv_y, %0 : i32
    %mask = tt.expand_dims %offs_m_27 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %mask_38 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_39 = tt.expand_dims %mask_38 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %mask_40 = tt.broadcast %mask : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %qk_41 = tt.splat %qk_scale : f32 -> tensor<64x128xf32, #blocked>
    %qk_42, %qk_43 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_44, %acc_45 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_46 = ttng.tmem_store %acc_35, %acc_44[%acc_45], %true : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetv_y_47:6 = scf.for %offsetv_y_57 = %0 to %2 step %c128_i32 iter_args(%offsetv_y_58 = %offsetv_y_34#0, %offsetv_y_59 = %offsetv_y_34#1, %offsetk_y_60 = %offsetk_y, %offsetv_y_61 = %offsetv_y_37, %qk_62 = %qk_43, %acc_63 = %acc_46) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %k = tt.descriptor_load %desc_k[%offsetk_y_60, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %k_64 = ttg.local_alloc %k {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %k_65 = ttg.memdesc_trans %k_64 {loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %qk_66 = ttng.tc_gen5_mma %q_29, %k_65, %qk_42[%qk_62], %false, %true {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %mask_67 = tt.splat %offsetv_y_57 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : i32 -> tensor<1x128xi32, #blocked>
      %mask_68 = arith.addi %mask_67, %mask_39 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<1x128xi32, #blocked>
      %mask_69 = tt.broadcast %mask_68 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %mask_70 = arith.cmpi sge, %mask_40, %mask_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xi32, #blocked>
      %qk_71, %qk_72 = ttng.tmem_load %qk_42[%qk_66] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %qk_73 = arith.mulf %qk_71, %qk_41 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %qk_74 = arith.select %mask_70, %cst_17, %cst_19 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xi1, #blocked>, tensor<64x128xf32, #blocked>
      %qk_75 = arith.addf %qk_73, %qk_74 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %m_ij_76 = "tt.reduce"(%qk_75) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_100: f32, %m_ij_101: f32):
        %m_ij_102 = arith.maxnumf %m_ij_100, %m_ij_101 : f32
        tt.reduce.return %m_ij_102 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_77 = arith.maxnumf %offsetv_y_59, %m_ij_76 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_78 = tt.expand_dims %m_ij_77 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %qk_79 = tt.broadcast %qk_78 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %qk_80 = arith.subf %qk_75, %qk_79 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %p = math.exp2 %qk_80 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %alpha = arith.subf %offsetv_y_59, %m_ij_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_81 = arith.subf %offsetv_y_59, %m_ij_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_82 = math.exp2 %alpha {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_83 = math.exp2 %alpha_81 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_100: f32, %l_ij_101: f32):
        %l_ij_102 = arith.addf %l_ij_100, %l_ij_101 : f32
        tt.reduce.return %l_ij_102 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_84 = tt.expand_dims %alpha_82 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_85 = tt.expand_dims %alpha_83 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_86 = tt.broadcast %acc_84 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_87 = tt.broadcast %acc_85 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_88, %acc_89 = ttng.tmem_load %acc_44[%acc_63] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %acc_90 = arith.mulf %acc_88, %acc_86 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked>
      %v = tt.descriptor_load %desc_v[%c0_i32, %offsetv_y_61] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %v_91 = ttg.local_alloc %v {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %v_92 = ttg.memdesc_trans %v_91 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %p_93 = tt.fp_to_fp %p {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
      %acc_94 = ttng.tmem_alloc %p_93 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf8E5M2, #blocked>) -> !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      %acc_95 = ttng.tmem_store %acc_90, %acc_44[%acc_89], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_96 = ttng.tc_gen5_mma %acc_94, %v_92, %acc_44[%acc_95], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %l_i = arith.mulf %offsetv_y_58, %alpha_83 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_97 = arith.addf %l_i, %l_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_98 = arith.addi %offsetk_y_60, %c128_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %offsetv_y_99 = arith.addi %offsetv_y_61, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32} : i32
      scf.yield %l_i_97, %m_ij_77, %offsetk_y_98, %offsetv_y_99, %qk_72, %acc_96 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    %acc_48, %acc_49 = ttng.tmem_load %acc_44[%offsetv_y_47#5] : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
    %m_i = math.log2 %offsetv_y_47#0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_i_50 = arith.addf %offsetv_y_47#1, %m_i : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %acc_51 = tt.expand_dims %offsetv_y_47#0 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %acc_52 = tt.broadcast %acc_51 : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %acc_53 = arith.divf %acc_48, %acc_52 : tensor<64x128xf32, #blocked>
    %m_ptrs = arith.muli %off_hz, %N_CTX : i32
    %m_ptrs_54 = tt.addptr %M, %m_ptrs : !tt.ptr<f32>, i32
    %m_ptrs_55 = tt.splat %m_ptrs_54 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %m_ptrs_56 = tt.addptr %m_ptrs_55, %offs_m_28 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %3 = ttg.convert_layout %m_i_50 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
    tt.store %m_ptrs_56, %3 : tensor<64x!tt.ptr<f32>, #blocked1>
    %4 = tt.fp_to_fp %acc_53, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
    %5 = ttg.convert_layout %4 : tensor<64x128xf8E5M2, #blocked> -> tensor<64x128xf8E5M2, #blocked2>
    tt.descriptor_store %desc_o[%qo_offset_y_23, %c0_i32], %5 : !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, tensor<64x128xf8E5M2, #blocked2>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 64, blockN = 128, unpacked = false>
module attributes {ttg.maxnreg = 80 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_o: !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, %desc_o_12: i32, %desc_o_13: i32, %desc_o_14: i64, %desc_o_15: i64, %N_CTX: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_16 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_18 = arith.constant 1.44269502 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_19 = arith.constant dense<-1.000000e+06> : tensor<64x128xf32, #blocked>
    %start_m = tt.get_program_id x : i32
    %off_hz = tt.get_program_id y : i32
    %off_z = arith.divsi %off_hz, %H : i32
    %off_h = arith.remsi %off_hz, %H : i32
    %offset_y = arith.muli %N_CTX, %H : i32
    %offset_y_20 = arith.muli %off_z, %offset_y : i32
    %offset_y_21 = arith.muli %off_h, %N_CTX : i32
    %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 : i32
    %qo_offset_y = arith.muli %start_m, %c64_i32 : i32
    %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y : i32
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %offs_m_25 = tt.splat %qo_offset_y : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_26 = tt.splat %qo_offset_y : i32 -> tensor<64xi32, #blocked1>
    %offs_m_27 = arith.addi %offs_m_25, %offs_m : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_28 = arith.addi %offs_m_26, %offs_m_24 : tensor<64xi32, #blocked1>
    %qk_scale = arith.mulf %sm_scale, %cst_18 : f32
    %q = tt.descriptor_load %desc_q[%qo_offset_y_23, %c0_i32] : !tt.tensordesc<tensor<64x128xf8E5M2, #shared>> -> tensor<64x128xf8E5M2, #blocked2>
    %q_29 = ttg.local_alloc %q : (tensor<64x128xf8E5M2, #blocked2>) -> !ttg.memdesc<64x128xf8E5M2, #shared, #smem>
    %offsetv_y = arith.muli %offset_y_22, %c128_i32 : i32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<64x128xf32, #blocked>
    %qk_30, %qk_31 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc, %acc_32 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %acc_33 = ttng.tmem_store %cst_17, %acc[%acc_32], %true : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetv_y_34:6 = scf.for %offsetv_y_57 = %c0_i32 to %qo_offset_y step %c128_i32 iter_args(%arg26 = %cst, %arg27 = %cst_16, %offset_y_58 = %offset_y_22, %offsetv_y_59 = %offsetv_y, %qk_60 = %qk_31, %acc_61 = %acc_33) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %k = tt.descriptor_load %desc_k[%offset_y_58, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %k_62 = ttg.local_alloc %k {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %k_63 = ttg.memdesc_trans %k_62 {loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %qk_64 = ttng.tc_gen5_mma %q_29, %k_63, %qk_30[%qk_60], %false, %true {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %qk_65, %qk_66 = ttng.tmem_load %qk_30[%qk_64] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %m_ij_67 = "tt.reduce"(%qk_65) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_93: f32, %m_ij_94: f32):
        %m_ij_95 = arith.maxnumf %m_ij_93, %m_ij_94 : f32
        tt.reduce.return %m_ij_95 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_68 = arith.mulf %m_ij_67, %m_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_69 = arith.maxnumf %arg27, %m_ij_68 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_70 = arith.mulf %qk_65, %qk {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %qk_71 = tt.expand_dims %m_ij_69 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %qk_72 = tt.broadcast %qk_71 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %qk_73 = arith.subf %qk_70, %qk_72 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %p = math.exp2 %qk_73 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %alpha = arith.subf %arg27, %m_ij_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_74 = arith.subf %arg27, %m_ij_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_75 = math.exp2 %alpha {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_76 = math.exp2 %alpha_74 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_93: f32, %l_ij_94: f32):
        %l_ij_95 = arith.addf %l_ij_93, %l_ij_94 : f32
        tt.reduce.return %l_ij_95 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_77 = tt.expand_dims %alpha_75 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_78 = tt.expand_dims %alpha_76 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_79 = tt.broadcast %acc_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_80 = tt.broadcast %acc_78 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_81, %acc_82 = ttng.tmem_load %acc[%acc_61] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %acc_83 = arith.mulf %acc_81, %acc_79 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked>
      %v = tt.descriptor_load %desc_v[%c0_i32, %offsetv_y_59] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %v_84 = ttg.local_alloc %v {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %v_85 = ttg.memdesc_trans %v_84 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %p_86 = tt.fp_to_fp %p {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
      %acc_87 = ttng.tmem_alloc %p_86 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf8E5M2, #blocked>) -> !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      %acc_88 = ttng.tmem_store %acc_83, %acc[%acc_82], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_89 = ttng.tc_gen5_mma %acc_87, %v_85, %acc[%acc_88], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %l_i = arith.mulf %arg26, %alpha_76 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_90 = arith.addf %l_i, %l_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_91 = arith.addi %offset_y_58, %c128_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %offsetv_y_92 = arith.addi %offsetv_y_59, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32} : i32
      scf.yield %l_i_90, %m_ij_69, %offsetk_y_91, %offsetv_y_92, %qk_66, %acc_89 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    %acc_35, %acc_36 = ttng.tmem_load %acc[%offsetv_y_34#5] : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
    %0 = arith.muli %start_m, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32
    %1 = arith.addi %start_m, %c1_i32 : i32
    %2 = arith.muli %1, %c64_i32 : i32
    %offsetk_y = arith.addi %offset_y_22, %0 : i32
    %offsetv_y_37 = arith.addi %offsetv_y, %0 : i32
    %mask = tt.expand_dims %offs_m_27 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %mask_38 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %mask_39 = tt.expand_dims %mask_38 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %mask_40 = tt.broadcast %mask : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %qk_41 = tt.splat %qk_scale : f32 -> tensor<64x128xf32, #blocked>
    // %qk_42, %qk_43 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    //%acc_44, %acc_45 = ttng.tmem_alloc : () -> (!ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    //%acc_46 = ttng.tmem_store %acc_35, %acc_44[%acc_45], %true : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc_46 = ttng.tmem_store %acc_35, %acc[%acc_36], %true : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>

//    %offsetv_y_47:6 = scf.for %offsetv_y_57 = %0 to %2 step %c128_i32 iter_args(%offsetv_y_58 = %offsetv_y_34#0, %offsetv_y_59 = %offsetv_y_34#1, %offsetk_y_60 = %offsetk_y, %offsetv_y_61 = %offsetv_y_37, %qk_62 = %qk_43, %acc_63 = %acc_46) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
    %offsetv_y_47:6 = scf.for %offsetv_y_57 = %0 to %2 step %c128_i32 iter_args(%offsetv_y_58 = %offsetv_y_34#0, %offsetv_y_59 = %offsetv_y_34#1, %offsetk_y_60 = %offsetk_y, %offsetv_y_61 = %offsetv_y_37, %qk_62 = %offsetv_y_34#4, %acc_63 = %acc_46) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %k = tt.descriptor_load %desc_k[%offsetk_y_60, %c0_i32] {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %k_64 = ttg.local_alloc %k {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %k_65 = ttg.memdesc_trans %k_64 {loop.cluster = 3 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      //%qk_66 = ttng.tc_gen5_mma %q_29, %k_65, %qk_42[%qk_62], %false, %true {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %qk_66 = ttng.tc_gen5_mma %q_29, %k_65, %qk_30[%qk_62], %false, %true {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %mask_67 = tt.splat %offsetv_y_57 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : i32 -> tensor<1x128xi32, #blocked>
      %mask_68 = arith.addi %mask_67, %mask_39 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<1x128xi32, #blocked>
      %mask_69 = tt.broadcast %mask_68 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %mask_70 = arith.cmpi sge, %mask_40, %mask_69 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xi32, #blocked>
      //%qk_71, %qk_72 = ttng.tmem_load %qk_42[%qk_66] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %qk_71, %qk_72 = ttng.tmem_load %qk_30[%qk_66] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>

      %qk_73 = arith.mulf %qk_71, %qk_41 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %qk_74 = arith.select %mask_70, %cst_17, %cst_19 {loop.cluster = 0 : i32, loop.stage = 3 : i32} : tensor<64x128xi1, #blocked>, tensor<64x128xf32, #blocked>
      %qk_75 = arith.addf %qk_73, %qk_74 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %m_ij_76 = "tt.reduce"(%qk_75) <{axis = 1 : i32}> ({
      ^bb0(%m_ij_100: f32, %m_ij_101: f32):
        %m_ij_102 = arith.maxnumf %m_ij_100, %m_ij_101 : f32
        tt.reduce.return %m_ij_102 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %m_ij_77 = arith.maxnumf %offsetv_y_59, %m_ij_76 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %qk_78 = tt.expand_dims %m_ij_77 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %qk_79 = tt.broadcast %qk_78 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %qk_80 = arith.subf %qk_75, %qk_79 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %p = math.exp2 %qk_80 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x128xf32, #blocked>
      %alpha = arith.subf %offsetv_y_59, %m_ij_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_81 = arith.subf %offsetv_y_59, %m_ij_77 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_82 = math.exp2 %alpha {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %alpha_83 = math.exp2 %alpha_81 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_ij = "tt.reduce"(%p) <{axis = 1 : i32}> ({
      ^bb0(%l_ij_100: f32, %l_ij_101: f32):
        %l_ij_102 = arith.addf %l_ij_100, %l_ij_101 : f32
        tt.reduce.return %l_ij_102 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %acc_84 = tt.expand_dims %alpha_82 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_85 = tt.expand_dims %alpha_83 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %acc_86 = tt.broadcast %acc_84 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      %acc_87 = tt.broadcast %acc_85 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
      //%acc_88, %acc_89 = ttng.tmem_load %acc_44[%acc_63] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
      %acc_88, %acc_89 = ttng.tmem_load %acc[%acc_63] {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>

      %acc_90 = arith.mulf %acc_88, %acc_86 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked>
      %v = tt.descriptor_load %desc_v[%c0_i32, %offsetv_y_61] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %v_91 = ttg.local_alloc %v {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 2 : i32} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %v_92 = ttg.memdesc_trans %v_91 {loop.cluster = 0 : i32, loop.stage = 3 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>
      %p_93 = tt.fp_to_fp %p {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
      %acc_94 = ttng.tmem_alloc %p_93 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : (tensor<64x128xf8E5M2, #blocked>) -> !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      //%acc_95 = ttng.tmem_store %acc_90, %acc_44[%acc_89], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_95 = ttng.tmem_store %acc_90, %acc[%acc_89], %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 3 : i32} : tensor<64x128xf32, #blocked> -> !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>

      //%acc_96 = ttng.tc_gen5_mma %acc_94, %v_92, %acc_44[%acc_95], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_96 = ttng.tc_gen5_mma %acc_94, %v_92, %acc[%acc_95], %true, %true {loop.cluster = 0 : i32, loop.stage = 3 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<64x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared1, #smem>, !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable>

      %l_i = arith.mulf %offsetv_y_58, %alpha_83 {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %l_i_97 = arith.addf %l_i, %l_ij {loop.cluster = 0 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %offsetk_y_98 = arith.addi %offsetk_y_60, %c128_i32 {loop.cluster = 4 : i32, loop.stage = 1 : i32} : i32
      %offsetv_y_99 = arith.addi %offsetv_y_61, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32} : i32
      scf.yield %l_i_97, %m_ij_77, %offsetk_y_98, %offsetv_y_99, %qk_72, %acc_96 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 1 : i32}
    // %acc_48, %acc_49 = ttng.tmem_load %acc_44[%offsetv_y_47#5] : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
    %acc_48, %acc_49 = ttng.tmem_load %acc[%offsetv_y_47#5] : !ttg.memdesc<64x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x128xf32, #blocked>
    %m_i = math.log2 %offsetv_y_47#0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %m_i_50 = arith.addf %offsetv_y_47#1, %m_i : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %acc_51 = tt.expand_dims %offsetv_y_47#0 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %acc_52 = tt.broadcast %acc_51 : tensor<64x1xf32, #blocked> -> tensor<64x128xf32, #blocked>
    %acc_53 = arith.divf %acc_48, %acc_52 : tensor<64x128xf32, #blocked>
    %m_ptrs = arith.muli %off_hz, %N_CTX : i32
    %m_ptrs_54 = tt.addptr %M, %m_ptrs : !tt.ptr<f32>, i32
    %m_ptrs_55 = tt.splat %m_ptrs_54 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
    %m_ptrs_56 = tt.addptr %m_ptrs_55, %offs_m_28 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
    %3 = ttg.convert_layout %m_i_50 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #blocked1>
    tt.store %m_ptrs_56, %3 : tensor<64x!tt.ptr<f32>, #blocked1>
    %4 = tt.fp_to_fp %acc_53, rounding = rtne : tensor<64x128xf32, #blocked> -> tensor<64x128xf8E5M2, #blocked>
    %5 = ttg.convert_layout %4 : tensor<64x128xf8E5M2, #blocked> -> tensor<64x128xf8E5M2, #blocked2>
    tt.descriptor_store %desc_o[%qo_offset_y_23, %c0_i32], %5 : !tt.tensordesc<tensor<64x128xf8E5M2, #shared>>, tensor<64x128xf8E5M2, #blocked2>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = 0 : i32}
      %10 = ttng.tmem_store %cst, %result[%9], %8 {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32}
    tt.return
  }
}
