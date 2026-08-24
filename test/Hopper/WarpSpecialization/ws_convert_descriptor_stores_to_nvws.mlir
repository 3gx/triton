// RUN: triton-opt %s --nvgpu-convert-descriptor-stores-to-nvws | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @descriptor_store
  // CHECK: %[[SRC:.*]] = arith.addf {{.*}} {async_task_id = array<i32: 0, 2>}
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32}
  // CHECK: ttg.local_store %[[SRC]], %[[ALLOC]] {async_task_id = array<i32: 0>, loop.cluster = 4 : i32, loop.stage = 1 : i32}
  // CHECK: nvws.descriptor_store %arg0[%arg1, %arg1] %[[ALLOC]]
  // CHECK-SAME: {async_task_id = array<i32: 2>, loop.cluster = 4 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
  // CHECK-NOT: tt.descriptor_store
  // CHECK-NOT: ttng.async_tma_copy_local_to_global
  tt.func public @descriptor_store(
      %desc: !tt.tensordesc<128x256xf32, #shared>, %i: i32,
      %src: tensor<128x256xf32, #blocked>) {
    %value = arith.addf %src, %src {async_task_id = array<i32: 0, 2>} :
        tensor<128x256xf32, #blocked>
    tt.descriptor_store %desc[%i, %i], %value
        {async_task_id = array<i32: 2>, loop.cluster = 4 : i32,
         loop.stage = 1 : i32, ttg.partition = array<i32: 2>} :
        !tt.tensordesc<128x256xf32, #shared>, tensor<128x256xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @descriptor_reduce
  // CHECK: %[[SRC:.*]] = arith.addf {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: %[[ALLOC:.*]] = ttg.local_alloc {async_task_id = array<i32: 1>, loop.cluster = 6 : i32, loop.stage = 2 : i32}
  // CHECK: ttg.local_store %[[SRC]], %[[ALLOC]] {async_task_id = array<i32: 1>, loop.cluster = 6 : i32, loop.stage = 2 : i32}
  // CHECK: nvws.descriptor_reduce add, %arg0[%arg1, %arg1] %[[ALLOC]]
  // CHECK-SAME: {async_task_id = array<i32: 3>, loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>}
  // CHECK-NOT: tt.descriptor_reduce
  // CHECK-NOT: ttng.async_tma_reduce
  tt.func public @descriptor_reduce(
      %desc: !tt.tensordesc<128x256xf32, #shared>, %i: i32,
      %src: tensor<128x256xf32, #blocked>) {
    %value = arith.addf %src, %src {async_task_id = array<i32: 1>} :
        tensor<128x256xf32, #blocked>
    tt.descriptor_reduce add, %desc[%i, %i], %value
        {async_task_id = array<i32: 3>, loop.cluster = 6 : i32,
         loop.stage = 2 : i32, ttg.partition = array<i32: 3>} :
        !tt.tensordesc<128x256xf32, #shared>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
