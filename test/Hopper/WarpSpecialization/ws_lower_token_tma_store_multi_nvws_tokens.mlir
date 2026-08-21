// RUN: triton-opt %s --nvgpu-test-ws-lower-token | FileCheck %s
// RUN: triton-opt %s --nvgpu-test-ws-lower-token --nvgpu-test-tma-store-token-wait-reorder="enable-rotation=false" | FileCheck %s --check-prefix=REDUCE-LATE

// Regression test for B-17-F2 / T273495687.
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tma_store_wait_two_nvws_tokens
  // CHECK: ttg.local_alloc
  // CHECK: %[[EMPTY0:.*]] = ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: %[[EMPTY1:.*]] = ttg.local_alloc
  // CHECK: %[[BAR0:.*]] = ttg.memdesc_index %[[EMPTY0]][%{{.*}}]
  // CHECK: %[[BAR1:.*]] = ttg.memdesc_index %[[EMPTY1]][%{{.*}}]
  // CHECK: ttng.async_tma_store_token_wait %arg0 , %[[BAR0]][%{{.*}}], %[[BAR1]][%{{.*}}]
  // CHECK-NOT: nvws_token
  tt.func public @tma_store_wait_two_nvws_tokens(%store_tok: !ttg.async.token) {
    %tok0 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %tok1 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %idx0 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    %idx1 = arith.constant {async_task_id = array<i32: 1>} 1 : i32

    "ttng.async_tma_store_token_wait"(%store_tok, %tok0, %tok1, %idx0, %idx1)
        <{operandSegmentSizes = array<i32: 1, 0, 0, 2, 2>, async_task_id = array<i32: 1>}>
        : (!ttg.async.token, tensor<2x!nvws.token>, tensor<2x!nvws.token>, i32, i32) -> ()
    tt.return
  }

  // Each deferred release on an abstract store must resolve against its own
  // CreateTokenOp's empty-barrier array. Resolving the first token must leave
  // the second pair for the next token rather than clearing both.
  // CHECK-LABEL: @abstract_store_two_nvws_tokens
  // CHECK: ttg.local_alloc
  // CHECK: %[[ABSTRACT_EMPTY0:.*]] = ttg.local_alloc
  // CHECK: ttg.local_alloc
  // CHECK: %[[ABSTRACT_EMPTY1:.*]] = ttg.local_alloc
  // CHECK: %[[ABSTRACT_BAR0:.*]] = ttg.memdesc_index %[[ABSTRACT_EMPTY0]][%{{.*}}]
  // CHECK: %[[ABSTRACT_BAR1:.*]] = ttg.memdesc_index %[[ABSTRACT_EMPTY1]][%{{.*}}]
  // CHECK: nvws.descriptor_store {{.*}}, %[[ABSTRACT_BAR0]][%{{.*}}], %[[ABSTRACT_BAR1]][%{{.*}}]
  // CHECK-NOT: nvws_token
  tt.func public @abstract_store_two_nvws_tokens(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %tok0 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %tok1 = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %idx0 = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    %idx1 = arith.constant {async_task_id = array<i32: 1>} 1 : i32
    nvws.descriptor_store %desc[%idx0, %idx1] %src
        nvws_token %tok0[%idx0] nvws_token %tok1[%idx1]
        {async_task_id = array<i32: 1>}
        : !tt.tensordesc<16x16xf16, #shared>,
          !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
          token_types = tensor<2x!nvws.token>, tensor<2x!nvws.token>
    tt.return
  }

  // A reduce carries the same deferred staging-buffer release as a store.
  // Token lowering realizes the empty barrier, and late materialization moves
  // it onto the generated reduce completion wait.
  // REDUCE-LATE-LABEL: @abstract_reduce_completion
  // REDUCE-LATE: %[[FULL:.*]] = ttg.local_alloc
  // REDUCE-LATE-NEXT: %[[EMPTY:.*]] = ttg.local_alloc
  // REDUCE-LATE: ttg.barrier local
  // REDUCE-LATE: %[[BAR:.*]] = ttg.memdesc_index %[[EMPTY]][%{{.*}}]
  // REDUCE-LATE: %[[TOKEN:.*]] = ttng.async_tma_reduce add
  // REDUCE-LATE: ttng.async_tma_store_token_wait %[[TOKEN]] , %[[BAR]][%{{.*}}]
  // REDUCE-LATE-NOT: nvws.descriptor_reduce
  tt.func public @abstract_reduce_completion(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) {
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %idx = arith.constant {async_task_id = array<i32: 1>} 0 : i32
    nvws.descriptor_reduce add, %desc[%idx, %idx] %src
        nvws_token %token[%idx] {async_task_id = array<i32: 1>}
        : !tt.tensordesc<16x16xf16, #shared>,
          !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
          token_types = tensor<2x!nvws.token>
    tt.return
  }
}
