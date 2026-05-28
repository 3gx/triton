// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore | FileCheck %s --check-prefix=LOWER
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore --tritongpu-partition-loops | FileCheck %s --check-prefix=PARTITION

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // LOWER-LABEL: @post_ws_tmem_read_carrier_tag
  // LOWER: ttng.tmem_load
  // LOWER-NEXT: [[BAR:%.*]] = ttg.memdesc_index {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // LOWER-NEXT: ttng.arrive_barrier [[BAR]], 1 {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  // PARTITION-LABEL: @post_ws_tmem_read_carrier_tag
  // PARTITION: nvws.warp_group
  // PARTITION: ttng.arrive_barrier {{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
  tt.func @post_ws_tmem_read_carrier_tag(
      %ub: i32,
      %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
      %rhs: !ttg.memdesc<64x128xf16, #shared1, #smem>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %acc, %tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %init = ttng.tmem_store %cst, %acc[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %loop = scf.for %iv = %c0 to %ub step %c1 iter_args(%carry = %init) -> (!ttg.async.token) : i32 {
      %mma = ttng.tc_gen5_mma %lhs, %rhs, %acc[%carry], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 1>} %mma : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %out, %load_tok = ttng.tmem_load %acc[%loop] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%out) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
