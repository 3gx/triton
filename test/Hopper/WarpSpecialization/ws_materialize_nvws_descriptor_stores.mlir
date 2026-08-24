// RUN: triton-opt %s --nvgpu-materialize-nvws-descriptor-stores | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @materialize_store_reduce
  // CHECK-NOT: nvws.descriptor_store
  // CHECK: %[[STORE_TOKEN:.*]] = ttng.async_tma_copy_local_to_global %arg0[%arg3, %arg3] %arg2
  // CHECK-SAME: {async_task_id = array<i32: 2>, loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
  // CHECK-NEXT: ttng.async_tma_store_token_wait %[[STORE_TOKEN]]
  // CHECK-SAME: {async_task_id = array<i32: 2>, can_rotate_by_buffer_count = 2 : i32, loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
  // CHECK-NOT: nvws.descriptor_reduce
  // CHECK: %[[REDUCE_TOKEN:.*]] = ttng.async_tma_reduce add, %arg1[%arg3, %arg3] %arg2
  // CHECK-SAME: {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>}
  // CHECK-NEXT: ttng.async_tma_store_token_wait %[[REDUCE_TOKEN]]
  // CHECK-SAME: {async_task_id = array<i32: 3>, loop.cluster = 5 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>}
  // CHECK-NOT: ttng.async_tma_store_wait
  tt.func public @materialize_store_reduce(
      %store_desc: !tt.tensordesc<128x64xf32, #shared>,
      %reduce_desc: !tt.tensordesc<128x64xf32, #shared>,
      %src: !ttg.memdesc<128x64xf32, #shared, #smem, mutable>, %i: i32) {
    nvws.descriptor_store %store_desc[%i, %i] %src
        {async_task_id = array<i32: 2>, can_rotate_by_buffer_count = 2 : i32,
         loop.cluster = 3 : i32, loop.stage = 1 : i32,
         ttg.partition = array<i32: 2>} :
        !tt.tensordesc<128x64xf32, #shared>,
        !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    nvws.descriptor_reduce add, %reduce_desc[%i, %i] %src
        {async_task_id = array<i32: 3>, loop.cluster = 5 : i32,
         loop.stage = 2 : i32, ttg.partition = array<i32: 3>} :
        !tt.tensordesc<128x64xf32, #shared>,
        !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
    tt.return
  }
}
