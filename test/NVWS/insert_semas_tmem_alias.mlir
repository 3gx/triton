// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas="use-meta-partitioner=true" -cse | FileCheck %s --check-prefix=META

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_reinterpret_alias
  tt.func @tmem_reinterpret_alias(%ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // META: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32
    // META-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
    // META-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
    // META-NEXT: nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 0>
    // CHECK: [[ALLOC:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {{.*}}ttg.partition = array<i32: 0>
    %alloc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %r = scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%t = %tok) -> (!ttg.async.token) : i32 {
      %view0 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: [[BUF0:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 0>} : {{.*}} -> !ttg.memdesc<128x128xf32
      // CHECK-NEXT: [[VIEW0:%.*]] = ttg.memdesc_reinterpret [[BUF0]]
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[VIEW0]]
      // CHECK-NEXT: nvws.semaphore.release [[FULL]]
      %t0 = ttng.tmem_store %cst, %view0[%t], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %view1 = ttg.memdesc_reinterpret %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[LTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[BUF1:%.*]] = nvws.semaphore.buffer [[FULL]], [[LTOK]] {ttg.partition = array<i32: 1>} : {{.*}} -> !ttg.memdesc<128x128xf32
      // CHECK-NEXT: [[VIEW1:%.*]] = ttg.memdesc_reinterpret [[BUF1]]
      // CHECK-NEXT: {{.*}}, {{.*}} = ttng.tmem_load [[VIEW1]]
      %val, %t1 = ttng.tmem_load %view1[%t0] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1>} %t1 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    "use_token"(%r) : (!ttg.async.token) -> ()
    tt.return
  }
}
