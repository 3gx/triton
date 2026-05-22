// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas

// v4 container-pattern tests. All multi-member buffer.id groups in real
// Triton IR have one member that acts as the physical slot container
// (largest extent, covers [0, slot_size)); other members are sub-views
// inside the container. Union-find on overlap intervals unites all
// members through the container, so each buffer.id group collapses to
// a single resourceKey.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked64 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Disjoint sub-views m1 = [0, 128),
  // m2 = [128, 192), m3 = [192, 256). m0 unions with m1, m2, m3 via
  // overlap; m1, m2, m3 are pairwise disjoint. All collapse to one
  // resourceKey via m0.
  tt.func @container_with_disjoint_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked64>
    %true = arith.constant true
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 900 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 128 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m3, %t3 = ttng.tmem_alloc %cst64 {buffer.id = 900 : i32, buffer.offset = 192 : i32, ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> (!ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>
      %v3, %l3 = ttng.tmem_load %m3[%t3] {ttg.partition = array<i32: 3>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked64>
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x64xf32, #blocked64>) -> ()
      "use3"(%v3) {ttg.partition = array<i32: 3>} : (tensor<128x64xf32, #blocked64>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2, 3>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0, 1, 2, 3>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked256 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem128 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem256 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Container m0 = [0, 256). Two overlapping sub-views m1 = [0, 128),
  // m2 = [64, 192). m1 and m2 overlap each other AND both overlap m0.
  // All three collapse to one resourceKey.
  tt.func @container_with_overlapping_subviews(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst256 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked256>
    %cst128 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %m0, %t0 = ttng.tmem_alloc %cst256 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked256>) -> (!ttg.memdesc<128x256xf32, #tmem256, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m1, %t1 = ttng.tmem_alloc %cst128 {buffer.id = 901 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %m2, %t2 = ttng.tmem_alloc %cst128 {buffer.id = 901 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %v1, %l1 = ttng.tmem_load %m1[%t1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %v2, %l2 = ttng.tmem_load %m2[%t2] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem128, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use1"(%v1) {ttg.partition = array<i32: 1>} : (tensor<128x128xf32, #blocked>) -> ()
      "use2"(%v2) {ttg.partition = array<i32: 2>} : (tensor<128x128xf32, #blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
