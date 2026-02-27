// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semaphore | FileCheck %s --implicit-check-not=nvws.aref

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @insert_semaphore_basic
  // CHECK: nvws.semaphore.create
  // CHECK: nvws.semaphore.acquire
  // CHECK: nvws.semaphore.buffer
  // CHECK: nvws.semaphore.release
  tt.func @insert_semaphore_basic(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
      %ld = tt.descriptor_load %desc[%i, %i] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      "use"(%ld) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
