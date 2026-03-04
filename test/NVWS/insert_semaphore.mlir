// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semaphore | FileCheck %s --implicit-check-not=nvws.aref

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [128, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Two cross-partition values => two semaphore pairs (4 creates)
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // Producer side for first TMA load
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load {{.*}} 16384
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tma_load>]
      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // Producer side for second TMA load
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load {{.*}} 16384
      // CHECK: nvws.semaphore.release
      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>

      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // Consumer side: acquire + buffer for RHS
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.memdesc_trans
      // Consumer side: acquire + buffer for LHS
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttng.tc_gen5_mma
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Consumer releases
      // CHECK: nvws.semaphore.release
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tc5mma>]
      scf.yield {ttg.partition = array<i32: 0, 1>} %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @specialize_load_only
  tt.func @specialize_load_only(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.semaphore.release
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 1 : i32, loop.stage = 0, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @no_value_semaphore
  tt.func @no_value_semaphore(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-NOT: nvws.semaphore.create
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @value_semaphore_multiple_producers
  tt.func @value_semaphore_multiple_producers(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: [[VAL:%.*]] = "producer"
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: local_store
      // CHECK: nvws.semaphore.release
      // CHECK: "use0"([[VAL]])
      // CHECK: "use1"([[VAL]])
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: nvws.semaphore.release
      // CHECK: "use2"
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use0"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use1"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire + buffer + descriptor_load + release
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.semaphore.release
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Consumer 1 (register use): acquire + buffer + local_load + release
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
      // CHECK: "use1"
      // Consumer 2 (smem use): acquire + buffer + release
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: "use2"
      // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem_same_partition
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire + buffer + descriptor_load + release
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.semaphore.release
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Single consumer partition: acquire + buffer + local_load + uses + release
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: "use1"
      // CHECK: "use2"
      // CHECK: nvws.semaphore.release
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg4: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg5: !tt.tensordesc<tensor<128x8xi8, #shared2>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %result = ttng.tmem_alloc %cst_0 : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %0 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %1 = arith.muli %arg6, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %2 = tt.descriptor_load %arg3[%arg1, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %3 = tt.descriptor_load %arg4[%arg2, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = ttg.local_alloc %2 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      %6 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      // scales are a register descriptor_load — stays as tt.descriptor_load
      // CHECK: [[REG:%.*]] = tt.descriptor_load
      %4 = tt.descriptor_load %arg5[%arg1, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x8xi8, #shared2>> -> tensor<128x8xi8, #linear>
      // CHECK: tmem_alloc [[REG]]
      %result_1 = ttng.tmem_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      %result_2, %token = ttng.tmem_alloc %arg7 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %8 = ttng.tc_gen5_mma_scaled %5, %7, %result_2[%token], %result, %result_1, %true, %true lhs = e4m3 rhs = e4m3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %result_3, %token_4 = ttng.tmem_load %result_2[%8] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %result_3 : tensor<128x128xf32, #blocked>
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], tt.num_stages = 2 : i64, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_alloc_default_partition
  tt.func @local_alloc_default_partition(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x128xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    // Three cross-partition values => three semaphore pairs (6 creates)
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      // Producer for LHS TMA load
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load

      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: ttg.local_store

      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.memdesc_trans

      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %lhs_trans = ttg.memdesc_trans %5 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared1, #smem> -> !ttg.memdesc<128x128xf16, #shared, #smem>

      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>

      // CHECK: ttng.tc_gen5_mma
      %8 = ttng.tc_gen5_mma %lhs_trans, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @two_consumers
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_store
    // CHECK: nvws.semaphore.release

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // Consumer partition 1: acquire + buffer + load + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_b"

    "op_c"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    // Consumer partition 2: acquire + buffer + load + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_c"
    // CHECK: "op_d"
    "op_d"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @distance_one
tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_store
    // CHECK: nvws.semaphore.release
    %0 = "op_a"() {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ty
    // Consumer: acquire + buffer + load + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_b"
    "op_b"(%k) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @different_yield_partition
tt.func @different_yield_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer: acquire + buffer + load -> yield
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  // Two cross-partition iter_args => two semaphore pairs
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  %cst = arith.constant dense<0> : !ty
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
    // Producer puts for %l and %k
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release

    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: op_a

    // Consumer for %k in partition 1
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_b"
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer for %k in partition 2
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_c"
    // CHECK: "op_c"
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()

    // Consumer for %l in partition 1
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_d"
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer for %l in partition 2
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_d"
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @reuse_argument
tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty

  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer partition 1
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: op_d
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // Consumer partition 2
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: op_d
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [1, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch
tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // Three cross-partition iter_args => three semaphore pairs
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false

  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // Producer puts for %c, %b, %a
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer for %a
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: op_b
    "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> ()

    // Consumer for %b
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: op_c
    "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> ()

    // Consumer for %c
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: op_d
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %a, %a : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0, 0, 0], ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch2
tt.func @multiplicity_branch2(%lb: i32, %ub: i32, %step: i32) {
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // Three cross-partition iter_args => three semaphore pairs
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false

  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // Producer puts for %c, %b, %a
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // CHECK: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // Consumer for %a in partition 1
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: "op_b"
    %d = "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> !ty

    // Consumer for %b in partition 2
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: "op_c"
    %e = "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> !ty

    // Consumer for %c in partition 3
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: "op_d"
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %d, %e : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @self_recursion
tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NOT: nvws.semaphore.create
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @self_recursion_and_use
tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    // CHECK: "op_a"
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // Consumer: acquire + buffer + load + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release
    // CHECK: "op_b"

    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 1], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @conditional_consumer
tt.func @conditional_consumer(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "producer"
    // Producer: acquire + buffer + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    %cond = "rand"() {ttg.partition = array<i32: 1>} : () -> i1
    // CHECK: "rand"
    // Consumer: acquire + buffer + load + release wrapping if
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release
    // CHECK: scf.if
    %1 = scf.if %cond -> !ty {
      "something"() {ttg.partition = array<i32: 1>} : () -> ()
      scf.yield {ttg.partition = array<i32: 1>} %0 : !ty
    } else {
      %2 = "something"() {ttg.partition = array<i32: 1>} : () -> !ty
      scf.yield {ttg.partition = array<i32: 1>} %2 : !ty
    } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
    "keep"(%1) {ttg.partition = array<i32: 1>} : (!ty) -> ()
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @no_def_op
tt.func @no_def_op(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    // Producer: acquire + buffer + splat + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: splat
    // CHECK: local_store
    // CHECK: nvws.semaphore.release
    // Consumer: acquire + buffer + load + unsplat
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: local_load
    // CHECK: nvws.semaphore.release
    // CHECK: tt.unsplat
    // CHECK: addi
    arith.addi %k, %k {ttg.partition = array<i32: 1>} : i32
    scf.yield {ttg.partition = array<i32: 0>} %k : i32
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
  tt.return
}

// CHECK-LABEL: @scalar_consumers
tt.func @scalar_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> i32
    // CHECK: "op_a"
    // Producer: acquire + buffer + splat + store + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: tt.splat
    // CHECK: ttg.local_store
    // CHECK: nvws.semaphore.release

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (i32) -> ()
    // Consumer: acquire + buffer + load + unsplat + release
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.local_load
    // CHECK: nvws.semaphore.release
    // CHECK: tt.unsplat
    // CHECK: "op_b"

  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}


}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // Two cross-partition values => two semaphore pairs
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false

  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_b"
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer

    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]

    "op_c"(%1) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    scf.yield
  } {tt.warp_specialize, ttg.partition.stages = [0, 2], ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // Three cross-partition values => three semaphore pairs
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_b"
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer

    %2 = "op_c"(%1) {ttg.partition = array<i32: 2>} : (!ty) -> !ty
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_c"
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer

    "op_c"(%2) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>]
    // CHECK: "op_c"
    scf.yield
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 3], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}


// -----

// CHECK-LABEL: @inner_loop_fixed_operand
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @inner_loop_fixed_operand(%arg0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %arg3, %c128_i32 : i32
    %2 = arith.divsi %arg4, %c128_i32 : i32
    %3 = arith.divsi %arg5, %c128_i32 : i32
    %4 = arith.muli %1, %2 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %result, %token = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Two cross-partition values (outer LHS + inner RHS) => semaphore pairs
    // CHECK-COUNT-2: nvws.semaphore.create {{.*}} true
    // CHECK: scf.for
    // Producer for outer LHS TMA load
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: nvws.descriptor_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tma_load>]
    // Consumer for outer LHS
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: scf.for
    // Producer for inner RHS TMA load
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: nvws.descriptor_load
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tma_load>]
    // Consumer for inner RHS
    // CHECK: nvws.semaphore.acquire
    // CHECK: nvws.semaphore.buffer
    // CHECK: ttg.memdesc_trans
    // CHECK: ttng.tc_gen5_mma
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tc5mma>]
    %6 = scf.for %arg6 = %0 to %4 step %c148_i32 iter_args(%arg7 = %token) -> (!ttg.async.token)  : i32 {
      %7 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.muli %7, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.subi %1, %8 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.minsi %9, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %11 = arith.remsi %arg6, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %12 = arith.addi %8, %11 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.divsi %13, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %15 = arith.muli %12, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = arith.muli %14, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %17 = tt.descriptor_load %arg0[%15, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %18 = ttg.local_alloc %17 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %19:2 = scf.for %arg8 = %c0_i32 to %3 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %arg7) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg8, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg1[%16, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = ttg.memdesc_trans %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %26 = ttng.tc_gen5_mma %18, %25, %result[%arg10], %arg9, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %26 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %20 = tt.fp_to_fp %result_0 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %21 = ttg.convert_layout %20 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %arg2[%15, %16], %21 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %token_1 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: @semaphore_result_outside_scheduled_loop
tt.func @semaphore_result_outside_scheduled_loop(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: nvws.semaphore.create {{.*}} true
  // CHECK: nvws.semaphore.create {{.*}} false
  // Producer: acquire + buffer + store + release
  // CHECK: nvws.semaphore.acquire
  // CHECK: nvws.semaphore.release
  // Consumer: acquire + buffer + load + release
  // CHECK: nvws.semaphore.acquire
  // CHECK: nvws.semaphore.release
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 2>} : () -> !ty
    "op_b"(%0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    scf.for %j = %lb to %ub step %step : i32 {
      %x = arith.addi %lb, %lb {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
      scf.yield
    } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0>}
    scf.yield
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 2>, ttg.partition.stages = [0, 1], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}
}
