// RUN: triton-opt %s --nvws-meta-to-nvws-convert | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @buffer_plan
  tt.func @buffer_plan(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: %[[CLONE:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32}
    %a = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-NOT: buffer.id = 7
    %b = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 7 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: %[[HOST:.*]] = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32}
    %host = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 3 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret %[[HOST]] {buffer.copy = 2 : i32, buffer.id = 3 : i32, buffer.offset = 0 : i32}
    // CHECK-NOT: allocation.reuseTarget
    %reuse = ttg.local_alloc {allocation.reuseTarget = 3 : i32, buffer.copy = 2 : i32, buffer.id = 22 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32}
    %single = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 9 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 23 : i32}
    // CHECK-NOT: buffer.circular
    // CHECK-NOT: buffer.start
    %incompatible = ttg.local_alloc {allocation.reuseTarget = 9 : i32, buffer.copy = 1 : i32, buffer.id = 23 : i32} : () -> !ttg.memdesc<64x64xf16, #shared64, #smem, mutable>
    // CHECK-NOT: async_task_id
    scf.for %i = %lb to %ub step %step : i32 {
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
