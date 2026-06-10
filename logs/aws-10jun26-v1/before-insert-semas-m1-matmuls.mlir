#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @matmul_change_desc_in_prologue(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %0 = ub.poison : !tt.tensordesc<tensor<128x64xf16, #shared>>
    %1 = ub.poison : !tt.tensordesc<tensor<64x128xf16, #shared>>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %2 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %5:4 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0, %arg5 = %1, %arg6 = %2) -> (i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token)  : i32 {
      %6 = "prologue_cond"(%arg2) {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (i32) -> i1
      %7:2 = scf.if %6 -> (!tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>) {
        %13 = tt.make_tensor_descriptor %arg0, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #shared>>
        %14 = tt.make_tensor_descriptor %arg1, [%arg2, %arg2], [%c1_i64, %c1_i64] {ttg.partition = array<i32: 2>} : !tt.ptr<f16>, !tt.tensordesc<tensor<64x128xf16, #shared>>
        scf.yield {ttg.partition = array<i32: 2>} %13, %14 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } else {
        scf.yield {ttg.partition = array<i32: 2>} %arg4, %arg5 : !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>
      } {loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %8:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      nvws.descriptor_load %arg4[%8#0, %8#2] 16384 %3 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg5[%8#1, %8#2] 16384 %4 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %9 = ttng.tc_gen5_mma %3, %4, %result[%arg6], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      %12 = scf.if %10 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %11, %7#0, %7#1, %12 : i1, !tt.tensordesc<tensor<128x64xf16, #shared>>, !tt.tensordesc<tensor<64x128xf16, #shared>>, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 2>, array<i32: 2>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %3:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.splat %4#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      nvws.descriptor_gather %arg0[%5, %4#2] 16384 %1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %arg1[%4#1, %4#2] 16384 %2 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>>, i32, i32, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %6 = ttng.tc_gen5_mma %1, %2, %result[%arg4], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %7 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %8 = arith.select %7, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      %9 = scf.if %7 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %6 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %8, %9 : i1, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
  tt.func @matmul_tma_and_regular_load(%arg0: !tt.tensordesc<tensor<1x64xf16, #shared>>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %2:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token)  : i32 {
      %3:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : (i32) -> (i32, tensor<64x128xi32, #blocked2>, i32)
      %4 = tt.splat %3#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked1>
      %5 = tt.addptr %arg4, %3#1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>, tensor<64x128xi32, #blocked2>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked2>
      nvws.descriptor_gather %arg0[%4, %3#2] 16384 %1 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<128xi32, #blocked1>, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %7 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #blocked2>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %8 = ttng.tc_gen5_mma %1, %7, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %9 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %10 = arith.select %9, %false, %true {loop.cluster = 1 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : i1
      %11 = scf.if %9 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%8] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %8 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %10, %5, %11 : i1, tensor<64x128x!tt.ptr<f16>, #blocked2>, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}


