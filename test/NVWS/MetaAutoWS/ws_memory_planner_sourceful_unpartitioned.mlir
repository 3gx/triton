// RUN: triton-opt %s -allow-unregistered-dialect --nvws-memory-planner=num-buffers=1 | FileCheck %s

// A sourceful TMEM allocation outside a warp-specialized loop is not an NVWS
// communication channel. The TMEM planner may allocate it, but channel
// collection must not require a producer partition.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sourceful_unpartitioned_outside_ws
  // CHECK: scf.for
  // CHECK: ttng.tmem_alloc %{{.*}} {buffer.copy = 1 : i32, buffer.id = [[ID:[0-9]+]] : i32, buffer.offset = 0 : i32}
  // CHECK: ttng.tmem_load
  tt.func @sourceful_unpartitioned_outside_ws(%src: tensor<128x128xf32, #blocked>, %lb: i32, %ub: i32, %step: i32) {
    scf.for %iv = %lb to %ub step %step : i32 {
      %alloc, %token = ttng.tmem_alloc %src : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %out, %out_token = ttng.tmem_load %alloc[%token] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
      "use"(%out) : (tensor<128x128xf32, #linear>) -> ()
    }
    tt.return
  }
}
