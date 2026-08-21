// RUN: triton-opt %s -split-input-file --nvgpu-ws-tma-store-lowering | FileCheck %s

#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.early_tma_store_lowering" = true} {
// CHECK-LABEL: tma_store_basic
//       CHECK: ttg.local_alloc %arg2
//   CHECK-NOT: ttng.fence_async_shared
//       CHECK: %[[TOKEN:.*]] = ttng.async_tma_copy_local_to_global
//  CHECK-SAME: -> !ttg.async.token
//       CHECK: ttng.async_tma_store_token_wait %[[TOKEN]] : !ttg.async.token
  tt.func public @tma_store_basic(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store_reduce_skipped
//       CHECK: tt.descriptor_store
//   CHECK-NOT: ttng.async_tma_copy_local_to_global
//   CHECK-NOT: ttng.async_tma_store_token_wait
  tt.func public @tma_store_reduce_skipped(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 reduce_kind = add : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }
}

// -----

#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttg.test_nvws_tma_store_conversion" = true} {
// The sourceful staging allocation stays with the tensor producer (task 0),
// while the abstract descriptor store keeps the original epilogue task (2)
// and its schedule/partition metadata.
// CHECK-LABEL: tma_store_split_ownership
// CHECK: %[[SRC:.*]] = arith.addf {{.*}} {async_task_id = array<i32: 0, 2>}
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32}
// CHECK: ttg.local_store %[[SRC]], %[[ALLOC]] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32}
// CHECK: nvws.descriptor_store {{.*}} %[[ALLOC]]
// CHECK-SAME: {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
// CHECK-NOT: ttng.async_tma_copy_local_to_global
// CHECK-NOT: ttng.async_tma_store_token_wait
  tt.func public @tma_store_split_ownership(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    %src = arith.addf %arg2, %arg2 {async_task_id = array<i32: 0, 2>} : tensor<128x256xf32, #blocked>
    tt.descriptor_store %arg0[%arg1, %arg1], %src {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }

// A block-argument source has no producer task, so the allocation uses the
// descriptor operation's task as its conservative fallback.
// CHECK-LABEL: tma_store_block_arg_fallback
// CHECK: %[[FALLBACK_ALLOC:.*]] = ttg.local_alloc {async_task_id = array<i32: 4>}
// CHECK: ttg.local_store %arg2, %[[FALLBACK_ALLOC]] {async_task_id = array<i32: 4>}
// CHECK: nvws.descriptor_store {{.*}} %[[FALLBACK_ALLOC]]
// CHECK-SAME: {async_task_id = array<i32: 4>}
  tt.func public @tma_store_block_arg_fallback(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    tt.descriptor_store %arg0[%arg1, %arg1], %arg2 {async_task_id = array<i32: 4>} : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }

// Reduce-only functions must not be skipped when there are no plain stores.
// CHECK-LABEL: tma_reduce_only_abstract
// CHECK: %[[REDUCE_SRC:.*]] = arith.addf {{.*}} {async_task_id = array<i32: 1>}
// CHECK: %[[REDUCE_ALLOC:.*]] = ttg.local_alloc {async_task_id = array<i32: 1>, loop.cluster = 6 : i32, loop.stage = 2 : i32}
// CHECK: ttg.local_store %[[REDUCE_SRC]], %[[REDUCE_ALLOC]] {async_task_id = array<i32: 1>, loop.cluster = 6 : i32, loop.stage = 2 : i32}
// CHECK: nvws.descriptor_reduce add, {{.*}} %[[REDUCE_ALLOC]]
// CHECK-SAME: {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>}
// CHECK-NOT: ttng.async_tma_reduce
// CHECK-NOT: ttng.async_tma_store_token_wait
  tt.func public @tma_reduce_only_abstract(%arg0: !tt.tensordesc<128x256xf32, #nvmma_128>, %arg1: i32, %arg2: tensor<128x256xf32, #blocked>) {
    %src = arith.addf %arg2, %arg2 {async_task_id = array<i32: 1>} : tensor<128x256xf32, #blocked>
    tt.descriptor_reduce add, %arg0[%arg1, %arg1], %src {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !tt.tensordesc<128x256xf32, #nvmma_128>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
