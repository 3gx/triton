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
    // CHECK: [[MBAR_BASIC:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK-COUNT-2: ttng.init_barrier
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // CHECK: [[PH_SHR:%.*]] = arith.shrui {{%.*}}, {{%.*}} {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : i32
    // CHECK: [[PH_BIT:%.*]] = arith.andi [[PH_SHR]], {{%.*}} {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : i32
    // CHECK: ttng.wait_barrier {{%.*}}, [[PH_BIT]] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: [[BUF_VIEW:%.*]] = ttg.memdesc_index {{%.*}}[{{%.*}}] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttg.local_load [[BUF_VIEW]] {ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier {{%.*}}, 1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    %tok = nvws.semaphore.acquire %sem[%c1_i32, %cm1_i32] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c1_i32], %tok {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
    ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    nvws.semaphore.release %sem[%c1_i32], %tok [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    // CHECK-COUNT-2: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_BASIC]]
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load
  tt.func @tma_load(%desc: !tt.tensordesc<tensor<128x64xf16, #shared_desc>>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1>

    // CHECK: ttng.barrier_expect {{%.*}}, 4096 {{.*}}, {{%.*}}
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_load %desc[%c0_i32, %c0_i32] 4096 %view {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared_desc>>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
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
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>

    // CHECK: ttng.barrier_expect {{%.*}}, 2048 {{.*}}, {{%.*}}
    // CHECK: ttng.async_tma_gather {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token -> !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_gather %desc[%xoffs, %c0_i32] 2048 %view {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<1x128xf16, #shared_desc>>, tensor<8xi32>, i32, !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tc5mma_commit
  tt.func @tc5mma_commit() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>

    // CHECK: ttng.tc_gen5_commit {{%.*}} {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tc5mma>] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @fence_needed
  tt.func @fence_needed() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %sem_generic = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>
    %sem_other = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>

    %tok_other = nvws.semaphore.acquire %sem_other[%c0_i32, %cm1_i32] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    // CHECK: ttng.tc_gen5_commit {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    nvws.semaphore.release %sem_other[%c0_i32], %tok_other [#nvws.async_op<tc5mma>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token

    %tok_generic = nvws.semaphore.acquire %sem_generic[%c0_i32, %cm1_i32] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1> -> !ttg.async.token
    // CHECK-COUNT-1: ttng.fence_async_shared {bCluster = false, loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier {{%.*}}, 1 {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem_generic[%c0_i32], %tok_generic [#nvws.async_op<none>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>], 1>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_scales_passthrough
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
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cleanup_after_last_user
  tt.func @cleanup_after_last_user() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    %tok = nvws.semaphore.acquire %sem[%c0_i32, %cm1_i32] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32>
    "sink"(%v) : (tensor<1xi32>) -> ()
    // CHECK: ttng.arrive_barrier
    // CHECK: [[INV0:%.*]] = ttg.memdesc_index {{%.*}}[{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV0]]
    // CHECK: [[INV1:%.*]] = ttg.memdesc_index {{%.*}}[{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV1]]
    // CHECK: ttg.local_dealloc {{%.*}}
    // CHECK: "after_last_user"()
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    "after_last_user"() : () -> ()
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
