// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @mixed_depth_inactive_peer
  tt.func @mixed_depth_inactive_peer(%lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // The active copy-2 peer is semaphore-managed. The inactive copy-1 peer
    // remains independent rather than making raw accesses to semaphore-owned
    // backing.
    // CHECK: [[ACTIVE:%[-A-Za-z0-9_.$#]+]] = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 9821 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[EMPTY:%[0-9]+]] = nvws.semaphore.create [[ACTIVE]] true {pending_count = 1 : i32}
    // CHECK-NEXT: [[FULL:%[0-9]+]] = nvws.semaphore.create [[ACTIVE]] false {pending_count = 1 : i32}
    // CHECK: [[INACTIVE:%[-A-Za-z0-9_.$#]+]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9821 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %active = ttng.tmem_alloc {buffer.copy = 2 : i32, buffer.id = 9821 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %inactive = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 9821 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %active_value = "make_active"() {ttg.partition = array<i32: 0>} : () -> tensor<128x128xf32, #blocked>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[WRITE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[WRITE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[EMPTY]], [[WRITE_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_store %{{[0-9]+}}, [[WRITE_BUFFER]], %true {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      ttng.tmem_store %active_value, %active, %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[READ_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[READ_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[FULL]], [[READ_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[ACTIVE_RESULT:%[-A-Za-z0-9_.$#]+]], %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[READ_BUFFER]][] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[READ_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %active_result, %active_token = ttng.tmem_load %active[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use_active"(%active_result) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}

    %inactive_value = "make_inactive"() : () -> tensor<128x128xf32, #blocked>
    // CHECK: ttng.tmem_store %{{[0-9]+}}, [[INACTIVE]], %true
    ttng.tmem_store %inactive_value, %inactive, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[INACTIVE_RESULT:%[-A-Za-z0-9_.$#]+]], %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[INACTIVE]][]
    %inactive_result, %inactive_token = ttng.tmem_load %inactive[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use_inactive"(%inactive_result) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
