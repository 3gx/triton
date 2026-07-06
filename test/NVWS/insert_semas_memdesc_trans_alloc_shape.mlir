// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // The first MMA operand gives member A its exact view. Member B is cached at
  // the same time as a generic view, whose allocShape includes the three-stage
  // backing. Replaying B's transpose must infer its result type from that view.
  // CHECK-LABEL: @memdesc_trans_preserves_staged_alloc_shape
  tt.func @memdesc_trans_preserves_staged_alloc_shape(
      %desc_a: !tt.tensordesc<tensor<128x64xf16, #shared>>,
      %desc_b: !tt.tensordesc<tensor<256x64xf16, #shared>>,
      %acc: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>,
      %tok: !ttg.async.token) {
    %false = arith.constant false
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 700 : i32} : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %r = scf.for %iv = %c0 to %c1 step %c1 iter_args(%flag = %false) -> (i1) : i32 {
      nvws.descriptor_load %desc_a[%c0, %c0] 16384 %a {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      nvws.descriptor_load %desc_b[%c0, %c0] 32768 %b {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<256x64xf16, #shared>>, i32, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %bt = ttg.memdesc_trans %b {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>
      // CHECK: [[BUFS:%.*]]:2 = nvws.semaphore.buffer {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : {{.*}} -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64>
      // CHECK-NEXT: [[BT:%.*]] = ttg.memdesc_trans [[BUFS]]#1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf16, #shared, #smem, mutable, 3x256x64> -> !ttg.memdesc<64x256xf16, #shared{{.*}}, #smem, mutable, 3x64x256>
      // CHECK-NEXT: {{%.*}} = ttng.tc_gen5_mma [[BUFS]]#0, [[BT]],
      %mma = ttng.tc_gen5_mma %a, %bt, %acc[%tok], %flag, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared_t, #smem, mutable>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      "use_token"(%mma) {ttg.partition = array<i32: 0>} : (!ttg.async.token) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %true : i1
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    "use_i1"(%r) : (i1) -> ()
    tt.return
  }
}
