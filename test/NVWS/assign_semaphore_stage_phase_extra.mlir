// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck %s

#blocked_i32 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_f16 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @descriptor_gather_fresh_write
  // CHECK: nvws.semaphore.acquire {{.*}}[%[[STAGE:.*]], %[[PHASE:.*]]]
  // CHECK: nvws.descriptor_gather
  // CHECK: nvws.semaphore.release {{.*}}[%[[STAGE]]]
  tt.func @descriptor_gather_fresh_write(%desc: !tt.tensordesc<tensor<1x64xf16, #shared>>, %y: i32) {
    %off = arith.constant dense<0> : tensor<1xi32, #blocked_i32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>], 2>

    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable, 2x1x64>
    %old = ttg.local_load %view : !ttg.memdesc<1x64xf16, #shared, #smem, mutable, 2x1x64> -> tensor<1x64xf16, #blocked_f16>
    "use"(%old) : (tensor<1x64xf16, #blocked_f16>) -> ()
    nvws.descriptor_gather %desc[%off, %y] 128 %view : !tt.tensordesc<tensor<1x64xf16, #shared>>, tensor<1xi32, #blocked_i32>, i32, !ttg.memdesc<1x64xf16, #shared, #smem, mutable, 2x1x64>
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] : !nvws.semaphore<[!ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>], 2>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<2x1x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
