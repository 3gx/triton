// RUN: triton-opt %s --nvgpu-test-ws-memory-planner="num-buffers=2 smem-alloc-algo=1 smem-budget=196608" | FileCheck %s

// Hoist descriptor-load staging allocations reached through memdesc views.

// CHECK-LABEL: @hoist_tma_load_staging_through_view
// CHECK: %[[ALLOC:.*]] = ttg.local_alloc {{.*}}!ttg.memdesc<1x128x64xf16
// CHECK: scf.for
// CHECK: ttg.memdesc_index %[[ALLOC]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @hoist_tma_load_staging_through_view(
      %desc: !tt.tensordesc<128x64xf16, #shared>) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %channel_buffer = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    scf.for %iv = %c0 to %c1 step %c1 : i32 {
      %staging_buffer = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
      %staging_view = ttg.memdesc_index %staging_buffer[%c0] : !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %tile = tt.descriptor_load %desc[%c0, %c0] {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      ttg.local_store %tile, %staging_view {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %tile, %channel_buffer {async_task_id = array<i32: 1>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %staging_value = ttg.local_load %staging_view {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %channel_value = ttg.local_load %channel_buffer {async_task_id = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked>
      %sum = arith.addf %staging_value, %channel_value {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked>
      scf.yield {async_task_id = array<i32: 0, 1>}
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }

  // Direct scf.while staging is specific to the abstract store/reduce path.
  // Both planned allocations must be hoisted so persistent iterations reuse
  // one physical ring instead of allocating a private copy in each partition.
  // CHECK-LABEL: @hoist_nvws_tma_staging_out_of_while
  // CHECK: %[[STORE:.*]] = ttg.local_alloc {{.*}}buffer.tmaStaging = 1 : i32{{.*}}!ttg.memdesc<128x64xf16
  // CHECK: %[[REDUCE:.*]] = ttg.local_alloc {{.*}}buffer.tmaStaging = 2 : i32{{.*}}!ttg.memdesc<128x64xf32
  // CHECK: scf.while
  // CHECK: ttg.local_store {{.*}}, %[[STORE]]
  // CHECK: nvws.descriptor_store {{.*}} %[[STORE]]
  // CHECK: ttg.local_store {{.*}}, %[[REDUCE]]
  // CHECK: nvws.descriptor_reduce add, {{.*}} %[[REDUCE]]
  tt.func public @hoist_nvws_tma_staging_out_of_while(
      %store_desc: !tt.tensordesc<128x64xf16, #shared>,
      %reduce_desc: !tt.tensordesc<128x64xf32, #shared32>,
      %store_src: tensor<128x64xf16, #blocked>,
      %reduce_src: tensor<128x64xf32, #blocked>, %bound: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %result = scf.while (%i = %c0) : (i32) -> i32 {
      %valid = arith.cmpi slt, %i, %bound : i32
      scf.condition(%valid) %i : i32
    } do {
    ^bb0(%i: i32):
      %store_staging = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %store_src, %store_staging {async_task_id = array<i32: 0>} : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_store %store_desc[%i, %c0] %store_staging {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %reduce_staging = ttg.local_alloc : () -> !ttg.memdesc<128x64xf32, #shared32, #smem, mutable>
      ttg.local_store %reduce_src, %reduce_staging {async_task_id = array<i32: 0>} : tensor<128x64xf32, #blocked> -> !ttg.memdesc<128x64xf32, #shared32, #smem, mutable>
      nvws.descriptor_reduce add, %reduce_desc[%i, %c0] %reduce_staging {async_task_id = array<i32: 1>} : !tt.tensordesc<128x64xf32, #shared32>, !ttg.memdesc<128x64xf32, #shared32, #smem, mutable>
      %next = arith.addi %i, %c1 : i32
      scf.yield %next : i32
    } attributes {async_task_id = array<i32: 0, 1>, tt.warp_specialize}
    tt.return
  }
}
