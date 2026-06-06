// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s

// Captured from meta-aws-logs/run-22may26-nvws-meta-tmem-crash/passes/
// 062-anonymous_VerifyWarpSpecializationPartitions.mlir — Meta flash
// attention forward (persistent) IR. Stripped of loc() attributes.
// Used as a high-coverage input for v4 insert-semas discovery +
// ACCESS DAG (and the full pipeline once later commits land).

// CHECK-LABEL: @_attn_fwd_persist
// CHECK: [[ALPHA_BUF:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32}
// CHECK: [[ALPHA_EMPTY:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} true
// CHECK-NEXT: [[ALPHA_FULL:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} false
// CHECK-NEXT: [[ALPHA_TO_ROOT_0:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} false
// CHECK-NEXT: [[ALPHA_STAGE:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} false
// CHECK-NEXT: [[ALPHA_FROM_GEMM:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} false
// CHECK-NEXT: [[ALPHA_TO_ROOT_1:%.*]] = nvws.semaphore.create {{.*}}[[ALPHA_BUF]]{{.*}} false
// CHECK-NEXT: [[ALPHA_ENTRY:%.*]] = nvws.semaphore.acquire [[ALPHA_EMPTY]] {ttg.partition = array<i32: 1>
// CHECK: [[INNER:%.*]]:11 = scf.for {{.*}} to %c16384_i32
// CHECK: nvws.semaphore.release [[ALPHA_FULL]], [[ALPHA_MMA_TOK:%arg[0-9]+]] [#nvws.async_op<tc5mma>] {{.*}}ttg.partition = array<i32: 1>
// CHECK-NEXT: [[ALPHA_FULL_ACQ:%.*]] = nvws.semaphore.acquire [[ALPHA_FULL]] {{.*}}ttg.partition = array<i32: 5>
// CHECK: nvws.semaphore.release [[ALPHA_TO_ROOT_0]], [[ALPHA_FULL_ACQ]] [#nvws.async_op<none>] {{.*}}ttg.partition = array<i32: 5>
// CHECK-NEXT: [[ALPHA_TO_ROOT_0_ACQ:%.*]] = nvws.semaphore.acquire [[ALPHA_TO_ROOT_0]] {{.*}}ttg.partition = array<i32: 0>
// CHECK: nvws.semaphore.release [[ALPHA_STAGE]], [[ALPHA_TO_ROOT_0_ACQ]] [#nvws.async_op<none>] {{.*}}ttg.partition = array<i32: 0>
// CHECK-NEXT: [[ALPHA_STAGE_ACQ:%.*]] = nvws.semaphore.acquire [[ALPHA_STAGE]] {{.*}}ttg.partition = array<i32: 5>
// CHECK: nvws.semaphore.release [[ALPHA_EMPTY]], [[ALPHA_STAGE_ACQ]] [#nvws.async_op<none>] {{.*}}ttg.partition = array<i32: 5>
// CHECK: nvws.semaphore.release [[ALPHA_FROM_GEMM]], [[INNER]]#3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
// CHECK-NEXT: [[ALPHA_FROM_GEMM_ACQ:%.*]] = nvws.semaphore.acquire [[ALPHA_FROM_GEMM]] {ttg.partition = array<i32: 5>}
// CHECK: nvws.semaphore.release [[ALPHA_TO_ROOT_1]], [[ALPHA_FROM_GEMM_ACQ]] [#nvws.async_op<none>] {ttg.partition = array<i32: 5>}
// CHECK-NEXT: [[ALPHA_TO_ROOT_1_ACQ:%.*]] = nvws.semaphore.acquire [[ALPHA_TO_ROOT_1]] {ttg.partition = array<i32: 0>}
// CHECK: nvws.semaphore.release [[ALPHA_EMPTY]], [[ALPHA_TO_ROOT_1_ACQ]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
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
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.max_reg_auto_ws = 152 : i32, ttg.maxnreg = 128 : i32, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_persist(%sm_scale: f32, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Z: i32, %H: i32 {tt.divisibility = 16 : i32}, %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %n_tile_num = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #linear>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %prog_id = tt.get_program_id x : i32
    %num_progs = tt.get_num_programs x : i32
    %total_tiles = arith.muli %Z, %n_tile_num : i32
    %total_tiles_3 = arith.muli %total_tiles, %H : i32
    %tiles_per_sm = arith.divsi %total_tiles_3, %num_progs : i32
    %0 = arith.remsi %total_tiles_3, %num_progs : i32
    %1 = arith.cmpi slt, %prog_id, %0 : i32
    %2 = scf.if %1 -> (i32) {
      %tiles_per_sm_19 = arith.addi %tiles_per_sm, %c1_i32 : i32
      scf.yield %tiles_per_sm_19 : i32
    } else {
      scf.yield %tiles_per_sm : i32
    }
    %desc_q_4 = arith.muli %Z, %H : i32
    %desc_q_5 = arith.muli %desc_q_4, %c16384_i32 : i32
    %desc_q_6 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_q_7 = tt.make_tensor_descriptor %desc_q, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_k_8 = tt.make_tensor_descriptor %desc_k, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_v_9 = tt.make_tensor_descriptor %desc_v, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_10 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %desc_o_11 = tt.make_tensor_descriptor %desc_o, [%desc_q_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x128xf16, #shared>>
    %offset_y = arith.muli %H, %c16384_i32 : i32
    %offs_m0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs_m0_12 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked>
    %qk_scale = arith.mulf %sm_scale, %cst : f32
    %m_ij = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %m_ij_13 = tt.splat %qk_scale : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %qk = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %qk_14 = tt.splat %qk_scale : f32 -> tensor<128x128xf32, #linear>
    %q0_0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %q0_1 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %k = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %v = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %alpha = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %alpha_15 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_16 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_17 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %offsetkv_y_18 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %4 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %tile_idx = scf.for %_ = %c0_i32 to %2 step %c1_i32 iter_args(%tile_idx_19 = %prog_id) -> (i32)  : i32 {
      %pid = arith.remsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_hz = arith.divsi %tile_idx_19, %n_tile_num {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_z = arith.divsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %off_h = arith.remsi %off_hz, %H {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_20 = arith.muli %off_z, %offset_y {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_21 = arith.muli %off_h, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %offset_y_22 = arith.addi %offset_y_20, %offset_y_21 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %qo_offset_y = arith.muli %pid, %c256_i32 {ttg.partition = array<i32: 0, 2, 3>} : i32
      %qo_offset_y_23 = arith.addi %offset_y_22, %qo_offset_y {ttg.partition = array<i32: 2, 3>} : i32
      %5 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 2>} : i32
      %q0 = arith.addi %qo_offset_y_23, %c128_i32 {ttg.partition = array<i32: 3>} : i32
      %offs_m0_24 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_25 = tt.splat %qo_offset_y {ttg.partition = array<i32: 0, 2, 3>} : i32 -> tensor<128xi32, #blocked>
      %offs_m0_26 = arith.addi %offs_m0_24, %offs_m0 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %offs_m0_27 = arith.addi %offs_m0_25, %offs_m0_12 {ttg.partition = array<i32: 0>} : tensor<128xi32, #blocked>
      %q0_0_28 = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      %q0_0_29 = ttg.memdesc_index %q0_0[%q0_0_28] {ttg.partition = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_q_6[%qo_offset_y_23, %c0_i32] 32768 %q0_0_29 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %q0_1_30 = arith.constant {ttg.partition = array<i32: 3>} 0 : i32
      %q0_1_31 = ttg.memdesc_index %q0_1[%q0_1_30] {ttg.partition = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_q_7[%q0, %c0_i32] 32768 %q0_1_31 {ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %qk_0, %qk_0_32 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 5>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %qk_1, %qk_1_33 = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0, 1, 4>} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_0, %acc_0_34 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 2 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_1, %acc_1_35 = ttng.tmem_alloc %cst_0 {buffer.copy = 1 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #linear>) -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %q0_0_36 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %q0_0_37 = ttg.memdesc_index %q0_0[%q0_0_36] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %q0_1_38 = arith.constant {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %q0_1_39 = ttg.memdesc_index %q0_1[%q0_1_38] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %offsetkv_y_40:9 = scf.for %offsetkv_y_88 = %c0_i32 to %c16384_i32 step %c128_i32 iter_args(%offset_y_89 = %offset_y_22, %arg12 = %cst_2, %arg13 = %cst_1, %qk_0_90 = %qk_0_32, %acc_91 = %acc_0_34, %arg16 = %cst_2, %arg17 = %cst_1, %qk_1_92 = %qk_1_33, %acc_93 = %acc_1_35) -> (i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k_94 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
        %k_95 = ttg.memdesc_index %k[%k_94] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.descriptor_load %desc_k_8[%offset_y_89, %c0_i32] 32768 %k_95 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %k_96 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
        %k_97 = ttg.memdesc_index %k[%k_96] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %k_98 = ttg.memdesc_trans %k_97 {loop.cluster = 1 : i32, loop.stage = 0 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
        %v_99 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} 0 : i32
        %v_100 = ttg.memdesc_index %v[%v_99] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        nvws.descriptor_load %desc_v_9[%offset_y_89, %c0_i32] 32768 %v_100 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %qk_101 = ttng.tc_gen5_mma %q0_0_37, %k_98, %qk_0[%qk_0_90], %false, %true {loop.cluster = 1 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %qk_102 = ttng.tc_gen5_mma %q0_1_39, %k_98, %qk_1[%qk_1_92], %false, %true {loop.cluster = 3 : i32, loop.stage = 0 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %qk_103, %qk_104 = ttng.tmem_load %qk_0[%qk_101] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_105 = ttg.convert_layout %qk_103 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %qk_106, %qk_107 = ttng.tmem_load %qk_1[%qk_102] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %qk_108 = ttg.convert_layout %qk_106 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
        %m_ij_109 = "tt.reduce"(%qk_105) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_110 = "tt.reduce"(%qk_108) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%m_ij_176: f32, %m_ij_177: f32):
          %m_ij_178 = arith.maxnumf %m_ij_176, %m_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %m_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_111 = arith.mulf %m_ij_109, %m_ij {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_112 = arith.mulf %m_ij_110, %m_ij_13 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_113 = arith.maxnumf %arg13, %m_ij_111 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %m_ij_114 = arith.maxnumf %arg17, %m_ij_112 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %qk_115 = arith.mulf %qk_105, %qk {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_116 = arith.mulf %qk_108, %qk_14 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %qk_117 = tt.expand_dims %m_ij_113 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_118 = tt.expand_dims %m_ij_114 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %qk_119 = tt.broadcast %qk_117 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_120 = tt.broadcast %qk_118 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
        %qk_121 = arith.subf %qk_115, %qk_119 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %qk_122 = arith.subf %qk_116, %qk_120 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %p = math.exp2 %qk_121 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear>
        %p_123 = math.exp2 %qk_122 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear>
        %alpha_124 = arith.subf %arg13, %m_ij_113 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_125 = arith.subf %arg17, %m_ij_114 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_126 = math.exp2 %alpha_124 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_127 = tt.expand_dims %alpha_126 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_128 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} true
        ttng.tmem_store %alpha_127, %alpha, %alpha_128 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %alpha_129 = math.exp2 %alpha_125 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %alpha_130 = tt.expand_dims %alpha_129 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_131 = arith.constant {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} true
        ttng.tmem_store %alpha_130, %alpha_15, %alpha_131 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
        %l_ij = "tt.reduce"(%p) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 5>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 5>} : f32
        }) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>, ttg.partition.outputs = [array<i32: 5>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_ij_132 = "tt.reduce"(%p_123) <{axis = 1 : i32, reduction_ordering = "unordered"}> ({
        ^bb0(%l_ij_176: f32, %l_ij_177: f32):
          %l_ij_178 = arith.addf %l_ij_176, %l_ij_177 {ttg.partition = array<i32: 4>} : f32
          tt.reduce.return %l_ij_178 {ttg.partition = array<i32: 4>} : f32
        }) {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>, ttg.partition.outputs = [array<i32: 4>]} : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc_133, %acc_134 = ttng.tmem_load %acc_0[%acc_91] {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 8>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %acc_135, %acc_136 = ttng.tmem_load %acc_1[%acc_93] {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 11>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
        %18 = tt.reshape %acc_133 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %19 = tt.reshape %acc_135 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x2x64xf32, #linear2>
        %20 = tt.trans %18 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %21 = tt.trans %19 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x64x2xf32, #linear3>
        %outLHS, %outRHS = tt.split %20 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %outLHS_137, %outRHS_138 = tt.split %21 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x64xf32, #linear4>
        %alpha_139, %alpha_140 = ttng.tmem_load %alpha[] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_141 = tt.reshape %alpha_139 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_142 = ttg.convert_layout %alpha_141 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_143 = tt.expand_dims %alpha_142 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %alpha_144, %alpha_145 = ttng.tmem_load %alpha_15[] {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
        %alpha_146 = tt.reshape %alpha_144 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
        %alpha_147 = ttg.convert_layout %alpha_146 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %acc0_148 = tt.expand_dims %alpha_147 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
        %acc0_149 = ttg.convert_layout %acc0_143 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_150 = ttg.convert_layout %acc0_148 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #linear4>
        %acc0_151 = tt.broadcast %acc0_149 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_152 = tt.broadcast %acc0_150 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_153 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc0_154 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outLHS_137, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 4 : i32, loop.stage = 0 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS, %acc0_151 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc1_155 = tt.elementwise_inline_asm "\0A        {\0A            .reg .b64 ra, rb, rc;\0A            mov.b64 ra, { $2, $3 };\0A            mov.b64 rb, { $4, $5 };\0A            mul.f32x2 rc, ra, rb;\0A            mov.b64 { $0, $1 }, rc;\0A        }\0A        " {constraints = "=r,=r,r,r,r,r", loop.cluster = 2 : i32, loop.stage = 1 : i32, packed_element = 2 : i32, pure = true, ttg.partition = array<i32: 0>} %outRHS_138, %acc0_152 : tensor<128x64xf32, #linear4>, tensor<128x64xf32, #linear4> -> tensor<128x64xf32, #linear4>
        %acc_156 = tt.join %acc0_153, %acc1 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_157 = tt.join %acc0_154, %acc1_155 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x64xf32, #linear4> -> tensor<128x64x2xf32, #linear3>
        %acc_158 = tt.trans %acc_156 {loop.cluster = 4 : i32, loop.stage = 0 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_159 = tt.trans %acc_157 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 0, 2, 1>, ttg.partition = array<i32: 0>} : tensor<128x64x2xf32, #linear3> -> tensor<128x2x64xf32, #linear2>
        %acc_160 = tt.reshape %acc_158 {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %acc_161 = tt.reshape %acc_159 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128x2x64xf32, #linear2> -> tensor<128x128xf32, #linear>
        %p_162 = arith.truncf %p {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %p_163 = arith.truncf %p_123 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
        %acc_164 = ttng.tmem_alloc %p_162 {buffer.copy = 1 : i32, buffer.id = 4 : i32, buffer.offset = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 5>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        %acc_165 = ttng.tmem_alloc %p_163 {buffer.copy = 1 : i32, buffer.id = 5 : i32, buffer.offset = 0 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : (tensor<128x128xf16, #linear>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
        %acc_166 = ttng.tmem_store %acc_160, %acc_0[%acc_134], %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.start = array<i32: 9>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_167 = ttng.tmem_store %acc_161, %acc_1[%acc_136], %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.start = array<i32: 12>, ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %v_168 = arith.constant {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
        %v_169 = ttg.memdesc_index %v[%v_168] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %acc_170 = ttng.tc_gen5_mma %acc_164, %v_169, %acc_0[%acc_166], %true, %true {loop.cluster = 4 : i32, loop.stage = 0 : i32, tmem.end = array<i32: 9>, tmem.start = array<i32: 8, 10>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %acc_171 = ttng.tc_gen5_mma %acc_165, %v_169, %acc_1[%acc_167], %true, %true {loop.cluster = 2 : i32, loop.stage = 1 : i32, tmem.end = array<i32: 12>, tmem.start = array<i32: 11, 13>, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %l_i0 = arith.mulf %arg12, %alpha_126 {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_172 = arith.mulf %arg16, %alpha_129 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_173 = arith.addf %l_i0, %l_ij {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %l_i0_174 = arith.addf %l_i0_172, %l_ij_132 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
        %offsetkv_y_175 = arith.addi %offset_y_89, %c128_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : i32
        scf.yield {ttg.partition = array<i32: 0, 1, 3, 4, 5>} %offsetkv_y_175, %l_i0_173, %m_ij_113, %qk_104, %acc_170, %l_i0_174, %m_ij_114, %qk_107, %acc_171 : i32, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, !ttg.async.token, !ttg.async.token
      } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.scheduled_max_stage = 1 : i32, tt.separate_epilogue_store = true, ttg.partition = array<i32: 0, 1, 3, 4, 5>, ttg.partition.outputs = [array<i32: 3>, array<i32: 5>, array<i32: 5>, array<i32: 1>, array<i32: 0>, array<i32: 4>, array<i32: 4>, array<i32: 1>, array<i32: 0>]}
      %offsetkv_y_41 = tt.expand_dims %offsetkv_y_40#6 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_42 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_41, %offsetkv_y_18, %offsetkv_y_42 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y_43 = tt.expand_dims %offsetkv_y_40#5 {axis = 1 : i32, ttg.partition = array<i32: 4>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_44 = arith.constant {ttg.partition = array<i32: 4>} true
      ttng.tmem_store %offsetkv_y_43, %offsetkv_y_17, %offsetkv_y_44 {ttg.partition = array<i32: 4>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y_45 = tt.expand_dims %offsetkv_y_40#2 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_46 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_45, %offsetkv_y_16, %offsetkv_y_46 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y_47 = tt.expand_dims %offsetkv_y_40#1 {axis = 1 : i32, ttg.partition = array<i32: 5>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %offsetkv_y_48 = arith.constant {ttg.partition = array<i32: 5>} true
      ttng.tmem_store %offsetkv_y_47, %offsetkv_y, %offsetkv_y_48 {ttg.partition = array<i32: 5>} : tensor<128x1xf32, #linear> -> !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable>
      %offsetkv_y_49, %offsetkv_y_50 = ttng.tmem_load %offsetkv_y[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_51 = tt.reshape %offsetkv_y_49 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_52 = ttg.convert_layout %offsetkv_y_51 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0 = math.log2 %offsetkv_y_52 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offsetkv_y_53, %offsetkv_y_54 = ttng.tmem_load %offsetkv_y_17[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_55 = tt.reshape %offsetkv_y_53 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_56 = ttg.convert_layout %offsetkv_y_55 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_57 = math.log2 %offsetkv_y_56 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offsetkv_y_58, %offsetkv_y_59 = ttng.tmem_load %offsetkv_y_16[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_60 = tt.reshape %offsetkv_y_58 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_61 = ttg.convert_layout %offsetkv_y_60 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_62 = arith.addf %offsetkv_y_61, %m_i0 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offsetkv_y_63, %offsetkv_y_64 = ttng.tmem_load %offsetkv_y_18[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #linear5>
      %offsetkv_y_65 = tt.reshape %offsetkv_y_63 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear5> -> tensor<128xf32, #linear6>
      %offsetkv_y_66 = ttg.convert_layout %offsetkv_y_65 {ttg.partition = array<i32: 0>} : tensor<128xf32, #linear6> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %m_i0_67 = arith.addf %offsetkv_y_66, %m_i0_57 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %acc0 = tt.expand_dims %offsetkv_y_52 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_68 = tt.expand_dims %offsetkv_y_56 {axis = 1 : i32, ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %acc0_69 = tt.broadcast %acc0 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc0_70 = tt.broadcast %acc0_68 {ttg.partition = array<i32: 0>} : tensor<128x1xf32, #linear> -> tensor<128x128xf32, #linear>
      %acc, %acc_71 = ttng.tmem_load %acc_0[%offsetkv_y_40#4] {tmem.end = array<i32: 10>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_72 = ttg.convert_layout %acc {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc_73, %acc_74 = ttng.tmem_load %acc_1[%offsetkv_y_40#8] {tmem.end = array<i32: 13>, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear1>
      %acc_75 = ttg.convert_layout %acc_73 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear1> -> tensor<128x128xf32, #linear>
      %acc0_76 = arith.divf %acc_72, %acc0_69 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %acc0_77 = arith.divf %acc_75, %acc0_70 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear>
      %m_ptrs0 = arith.muli %off_hz, %c16384_i32 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      %m_ptrs0_78 = tt.addptr %M, %m_ptrs0 {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : !tt.ptr<f32>, i32
      %m_ptrs0_79 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_80 = tt.splat %m_ptrs0_78 {ttg.partition = array<i32: 0>} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %m_ptrs0_81 = tt.addptr %m_ptrs0_79, %offs_m0_26 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %m_ptrs0_82 = tt.addptr %m_ptrs0_80, %offs_m0_27 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      %6 = ttg.convert_layout %m_i0_62 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      %7 = ttg.convert_layout %m_i0_67 {ttg.partition = array<i32: 0>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked>
      tt.store %m_ptrs0_81, %6 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      tt.store %m_ptrs0_82, %7 {ttg.partition = array<i32: 0>} : tensor<128x!tt.ptr<f32>, #blocked>
      %8 = arith.truncf %acc0_76 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %c0_i32_83 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %9 = ttg.memdesc_index %3[%c0_i32_83] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %8, %9 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %10 = arith.truncf %acc0_77 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #linear> to tensor<128x128xf16, #linear>
      %c0_i32_84 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %11 = ttg.memdesc_index %4[%c0_i32_84] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_store %10, %11 {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %c0_i32_85 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      %12 = ttg.memdesc_index %3[%c0_i32_85] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %13 = ttg.local_load %12 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #linear>
      %14 = ttg.convert_layout %13 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      %c0_i32_86 = arith.constant {ttg.partition = array<i32: 2>} 0 : i32
      %15 = ttg.memdesc_index %4[%c0_i32_86] {ttg.partition = array<i32: 2>} : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %16 = ttg.local_load %15 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #linear>
      %17 = ttg.convert_layout %16 {ttg.partition = array<i32: 2>} : tensor<128x128xf16, #linear> -> tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_10[%qo_offset_y_23, %c0_i32], %14 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      tt.descriptor_store %desc_o_11[%5, %c0_i32], %17 {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked1>
      %tile_idx_87 = arith.addi %tile_idx_19, %num_progs {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>} %tile_idx_87 : i32
    } {tt.data_partition_factor = 2 : i32, tt.merge_epilogue = true, tt.separate_epilogue_store = true, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3, 4, 5>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3, 4, 5>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32], ttg.partition.types = ["correction", "gemm", "epilogue_store", "load", "computation", "computation"], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
