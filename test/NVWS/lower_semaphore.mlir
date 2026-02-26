// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --implicit-check-not=nvws.semaphore

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_desc = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @basic
  tt.func @basic() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // Mbarrier allocation (2 slots for 2-buffer semaphore)
    // CHECK: [[MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // Per-slice barrier init
    // CHECK: [[SLICE0:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE0]], 1
    // CHECK: [[SLICE1:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE1]], 1
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // Acquire: mbar indexed by stage (%c1_i32), phase extracted from phase vec (%cm1_i32)
    // CHECK: [[MBAR_ACQ:%.*]] = ttg.memdesc_index [[MBAR]][%c1_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: [[PH_SHR:%.*]] = arith.shrui %c-1_i32, %c1_i32 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : i32
    // CHECK: [[C1_MASK:%.*]] = arith.constant {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} 1 : i32
    // CHECK: [[PH_BIT:%.*]] = arith.andi [[PH_SHR]], [[C1_MASK]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : i32
    // CHECK: ttng.wait_barrier [[MBAR_ACQ]], [[PH_BIT]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // Buffer indexed by stage
    // CHECK: [[BUF_VIEW:%.*]] = ttg.memdesc_index %{{.*}}[%c1_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttg.local_load [[BUF_VIEW]] {ttg.partition = array<i32: 0>}
    // CHECK: ttg.local_store {{%.*}}, [[BUF_VIEW]] {ttg.partition = array<i32: 0>}
    // Release: arrive on same mbar slice
    // CHECK: [[MBAR_REL:%.*]] = ttg.memdesc_index [[MBAR]][%c1_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier [[MBAR_REL]], 1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    %tok = nvws.semaphore.acquire %sem[%c1_i32, %cm1_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c1_i32], %tok {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
    ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    nvws.semaphore.release %sem[%c1_i32], %tok [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    // Cleanup: per-slice invalidation
    // CHECK: [[INV0:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV0]]
    // CHECK: [[INV1:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV1]]
    // CHECK: ttg.local_dealloc [[MBAR]]
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load
  tt.func @tma_load(%desc: !tt.tensordesc<tensor<128x64xf16, #shared_desc>>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>
    // CHECK: [[MBAR_TMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_TMA:%.*]] = ttg.memdesc_index [[MBAR_TMA]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_TMA]], 1
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1>

    // Acquire: wait on mbar, then TMA load with barrier_expect
    // CHECK: [[MBAR_TMA_ACQ:%.*]] = ttg.memdesc_index [[MBAR_TMA]][%c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.wait_barrier [[MBAR_TMA_ACQ]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[BUF_TMA:%.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[MBAR_TMA_EXP:%.*]] = ttg.memdesc_index [[MBAR_TMA]][%c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.barrier_expect [[MBAR_TMA_EXP]], 4096 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] [[BUF_TMA]], [[MBAR_TMA_EXP]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_load %desc[%c0_i32, %c0_i32] 4096 %view {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared_desc>>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
    // Cleanup
    // CHECK: [[INV_TMA:%.*]] = ttg.memdesc_index [[MBAR_TMA]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV_TMA]]
    // CHECK: ttg.local_dealloc [[MBAR_TMA]]
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_gather
  tt.func @tma_gather(%desc: !tt.tensordesc<tensor<1x128xf16, #shared_desc>>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %xoffs = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>
    // CHECK: [[MBAR_GAT:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_GAT:%.*]] = ttg.memdesc_index [[MBAR_GAT]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_GAT]], 1
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>

    // CHECK: [[MBAR_GAT_ACQ:%.*]] = ttg.memdesc_index [[MBAR_GAT]][%c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.wait_barrier [[MBAR_GAT_ACQ]], {{%.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[BUF_GAT:%.*]] = ttg.memdesc_index %{{.*}}[%c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[MBAR_GAT_EXP:%.*]] = ttg.memdesc_index [[MBAR_GAT]][%c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.barrier_expect [[MBAR_GAT_EXP]], 2048 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.async_tma_gather %arg0[{{%.*}}, %c0_i32] [[BUF_GAT]], [[MBAR_GAT_EXP]], {{%.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_gather %desc[%xoffs, %c0_i32] 2048 %view {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x128xf16, #shared_desc>>, tensor<8xi32>, i32, !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    // CHECK: [[INV_GAT:%.*]] = ttg.memdesc_index [[MBAR_GAT]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV_GAT]]
    // CHECK: ttg.local_dealloc [[MBAR_GAT]]
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tc5mma_commit
  tt.func @tc5mma_commit() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_MMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_MMA:%.*]] = ttg.memdesc_index [[MBAR_MMA]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_MMA]], 1
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>

    // CHECK: [[MBAR_MMA_ACQ:%.*]] = ttg.memdesc_index [[MBAR_MMA]][%c0_i32] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.wait_barrier [[MBAR_MMA_ACQ]], {{%.*}} {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // tc5mma release: tc_gen5_commit on the mbar
    // CHECK: [[MBAR_MMA_REL:%.*]] = ttg.memdesc_index [[MBAR_MMA]][%c0_i32] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.tc_gen5_commit [[MBAR_MMA_REL]] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tc5mma>] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token
    // CHECK: [[INV_MMA:%.*]] = ttg.memdesc_index [[MBAR_MMA]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV_MMA]]
    // CHECK: ttg.local_dealloc [[MBAR_MMA]]
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @wgmma_pending_count
  // Tests that async_ops=[wgmma, none] is supported by pending-count init
  // and lowers to two arrives.
  tt.func @wgmma_pending_count() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_WGMMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[WGMMA_SLICE:%.*]] = ttg.memdesc_index [[MBAR_WGMMA]][{{%.*}}]
    // CHECK: ttng.init_barrier [[WGMMA_SLICE]], 2
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    // CHECK-COUNT-2: ttng.arrive_barrier {{%.*}}, 1 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<wgmma>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_copy_pending_count
  // Tests that async_ops=[tmem_copy, none] is supported by pending-count init
  // and lowers to tc_gen5_commit + arrive_barrier.
  tt.func @tmem_copy_pending_count() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_TMEM:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[TMEM_SLICE:%.*]] = ttg.memdesc_index [[MBAR_TMEM]][{{%.*}}]
    // CHECK: ttng.init_barrier [[TMEM_SLICE]], 2
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    // CHECK: ttng.tc_gen5_commit {{%.*}} {ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier {{%.*}}, 1 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tmem_copy>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @fence_needed
  // Tests that async_op<none> inserts fence + arrive_barrier,
  // while async_op<tc5mma> uses tc_gen5_commit (no fence)
  tt.func @fence_needed() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // Two separate mbar allocs (one per semaphore)
    // CHECK: [[MBAR_GEN:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_GEN:%.*]] = ttg.memdesc_index [[MBAR_GEN]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_GEN]], 1
    %sem_generic = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>
    // CHECK: [[MBAR_TC5:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_TC5:%.*]] = ttg.memdesc_index [[MBAR_TC5]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_TC5]], 1
    %sem_other = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>

    // tc5mma semaphore: wait then tc_gen5_commit (no fence)
    // CHECK: ttng.wait_barrier {{%.*}}, {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.tc_gen5_commit {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    %tok_other = nvws.semaphore.acquire %sem_other[%c0_i32, %cm1_i32] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    nvws.semaphore.release %sem_other[%c0_i32], %tok_other [#nvws.async_op<tc5mma>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token

    // Generic semaphore: wait then fence + arrive_barrier
    // CHECK: ttng.wait_barrier {{%.*}}, {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: [[MBAR_GEN_REL:%.*]] = ttg.memdesc_index [[MBAR_GEN]][%c0_i32] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttng.arrive_barrier [[MBAR_GEN_REL]], 1 {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    %tok_generic = nvws.semaphore.acquire %sem_generic[%c0_i32, %cm1_i32] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    nvws.semaphore.release %sem_generic[%c0_i32], %tok_generic [#nvws.async_op<none>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_scales_passthrough
  // Tests that tensor_memory_scales buffers pass through without memdesc_index
  tt.func @tmem_scales_passthrough(%arg0: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %sem = nvws.semaphore.create %arg0 true : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>], 1>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>], 1> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>], 1>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: "use_scale"(%arg0) : (!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) -> ()
    // CHECK-NOT: ttg.memdesc_index {{.*}}#ttng.tensor_memory_scales_encoding
    "use_scale"(%view) : (!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) -> ()
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>], 1>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @two_consumers
  // Tests 1 producer + 2 consumers with 3-buffer semaphore pair
  // Verifies barrier init counts, stage-indexed mbar, and per-slice cleanup
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // EMPTY semaphore: init_barrier with count=2 (2 consumers must arrive)
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[ES0:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES0]], 2
    // CHECK: [[ES1:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES1]], 2
    // CHECK: [[ES2:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES2]], 2
    %sem_empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    // FULL semaphore: init_barrier with count=1 (1 producer must arrive)
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FS0:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS0]], 1
    // CHECK: [[FS1:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS1]], 1
    // CHECK: [[FS2:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS2]], 1
    %sem_full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    // CHECK: scf.for
    scf.for %i = %arg0 to %arg1 step %arg2 : i32 {
      %val = "op_a"() {ttg.partition = array<i32: 0>} : () -> !elt
      // Producer: wait EMPTY[stage], store, arrive FULL[stage]
      // CHECK: [[EMPTY_MBAR:%.*]] = ttg.memdesc_index [[EMPTY]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 0>}
      // CHECK: ttng.wait_barrier [[EMPTY_MBAR]], {{%.*}} {{{.*}}ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = ttg.memdesc_index %{{.*}}[{{%.*}}] {{{.*}}ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: [[FULL_MBAR_P:%.*]] = ttg.memdesc_index [[FULL]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 0>}
      // CHECK: ttng.arrive_barrier [[FULL_MBAR_P]], 1 {{{.*}}ttg.partition = array<i32: 0>}
      %ptok = nvws.semaphore.acquire %sem_empty[%c0_i32, %cm1_i32] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %sem_empty[%c0_i32], %ptok {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %val, %pbuf {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem_full[%c0_i32], %ptok [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token

      // Consumer 1: wait FULL[stage], load, arrive EMPTY[stage]
      // CHECK: [[FULL_MBAR_C1:%.*]] = ttg.memdesc_index [[FULL]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[FULL_MBAR_C1]], {{%.*}} {{{.*}}ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[EMPTY_MBAR_C1:%.*]] = ttg.memdesc_index [[EMPTY]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier [[EMPTY_MBAR_C1]], 1 {{{.*}}ttg.partition = array<i32: 1>}
      %gtok1 = nvws.semaphore.acquire %sem_full[%c0_i32, %cm1_i32] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %gbuf1 = nvws.semaphore.buffer %sem_full[%c0_i32], %gtok1 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v1 = ttg.local_load %gbuf1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %gtok1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_b"(%v1) {ttg.partition = array<i32: 1>} : (!elt) -> ()

      // Consumer 2: wait FULL[stage], load, arrive EMPTY[stage]
      // CHECK: [[FULL_MBAR_C2:%.*]] = ttg.memdesc_index [[FULL]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 2>}
      // CHECK: ttng.wait_barrier [[FULL_MBAR_C2]], {{%.*}} {{{.*}}ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK: [[EMPTY_MBAR_C2:%.*]] = ttg.memdesc_index [[EMPTY]][{{%.*}}] {{{.*}}ttg.partition = array<i32: 2>}
      // CHECK: ttng.arrive_barrier [[EMPTY_MBAR_C2]], 1 {{{.*}}ttg.partition = array<i32: 2>}
      %gtok2 = nvws.semaphore.acquire %sem_full[%c0_i32, %cm1_i32] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %gbuf2 = nvws.semaphore.buffer %sem_full[%c0_i32], %gtok2 {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v2 = ttg.local_load %gbuf2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %gtok2 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_c"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
      "op_d"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // Cleanup: per-slice inval for EMPTY (3 slices)
    // CHECK: [[EI0:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI0]]
    // CHECK: [[EI1:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI1]]
    // CHECK: [[EI2:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI2]]
    // CHECK: ttg.local_dealloc [[EMPTY]]
    // Cleanup: per-slice inval for FULL (3 slices)
    // CHECK: [[FI0:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI0]]
    // CHECK: [[FI1:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI1]]
    // CHECK: [[FI2:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI2]]
    // CHECK: ttg.local_dealloc [[FULL]]
    ttg.local_dealloc %buf : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @three_consumers
  // Tests 1 producer + 3 consumers: init_barrier count=3 for EMPTY
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // EMPTY: init_barrier with count=3 (3 consumers)
    // CHECK: [[EMPTY3:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[ES:%.*]] = ttg.memdesc_index [[EMPTY3]]
    // CHECK: ttng.init_barrier [[ES]], 3
    %sem_empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    // FULL: init_barrier with count=1 (1 producer)
    // CHECK: [[FULL3:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FS:%.*]] = ttg.memdesc_index [[FULL3]]
    // CHECK: ttng.init_barrier [[FS]], 1
    %sem_full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>
    scf.for %i = %arg0 to %arg1 step %arg2 : i32 {
      %val = "op_a"() {ttg.partition = array<i32: 0>} : () -> !elt
      // Producer
      %ptok = nvws.semaphore.acquire %sem_empty[%c0_i32, %cm1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %sem_empty[%c0_i32], %ptok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %val, %pbuf {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem_full[%c0_i32], %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      // Consumer 1
      %g1 = nvws.semaphore.acquire %sem_full[%c0_i32, %cm1_i32] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %b1 = nvws.semaphore.buffer %sem_full[%c0_i32], %g1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v1 = ttg.local_load %b1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_b"(%v1) {ttg.partition = array<i32: 1>} : (!elt) -> ()
      // Consumer 2
      %g2 = nvws.semaphore.acquire %sem_full[%c0_i32, %cm1_i32] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %b2 = nvws.semaphore.buffer %sem_full[%c0_i32], %g2 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v2 = ttg.local_load %b2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g2 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_c"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
      // Consumer 3
      %g3 = nvws.semaphore.acquire %sem_full[%c0_i32, %cm1_i32] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3> -> !ttg.async.token
      %b3 = nvws.semaphore.buffer %sem_full[%c0_i32], %g3 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v3 = ttg.local_load %b3 {ttg.partition = array<i32: 3>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g3 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>], 3>, !ttg.async.token
      "op_e"(%v3) {ttg.partition = array<i32: 3>} : (!elt) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32, 3 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>}
    // Cleanup: 3 inval per mbar alloc
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[EMPTY3]]
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[FULL3]]
    ttg.local_dealloc %buf : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cleanup_after_last_user
  // Tests that barrier invalidation and dealloc happen before "after_last_user"
  tt.func @cleanup_after_last_user() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_CL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK: [[CL_S0:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.init_barrier [[CL_S0]], 1
    // CHECK: [[CL_S1:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.init_barrier [[CL_S1]], 1
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // Acquire: mbar indexed by stage, phase computation, wait
    // CHECK: [[CL_MBAR:%.*]] = ttg.memdesc_index [[MBAR_CL]][%c0_i32]
    // CHECK: ttng.wait_barrier [[CL_MBAR]], {{%.*}}
    // CHECK: [[CL_BUF:%.*]] = ttg.memdesc_index %{{.*}}[%c0_i32]
    // CHECK: ttg.local_load [[CL_BUF]]
    // CHECK: "sink"
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32>
    "sink"(%v) : (tensor<1xi32>) -> ()
    // Release: arrive, then cleanup BEFORE "after_last_user"
    // CHECK: [[CL_MBAR_REL:%.*]] = ttg.memdesc_index [[MBAR_CL]][%c0_i32]
    // CHECK: ttng.arrive_barrier [[CL_MBAR_REL]], 1
    // CHECK: [[CL_INV0:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[CL_INV0]]
    // CHECK: [[CL_INV1:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[CL_INV1]]
    // CHECK: ttg.local_dealloc [[MBAR_CL]]
    // CHECK: "after_last_user"()
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    "after_last_user"() : () -> ()
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
