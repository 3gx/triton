// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_release_duplicate_async() {
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{async_ops contains duplicate async kind}}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>, #nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared32 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_element_type_mismatch(
      %desc: !tt.tensordesc<16x16xf16, #shared16>,
      %src: !ttg.memdesc<16x16xf32, #shared32, #smem, mutable>,
      %i: i32, %j: i32) {
    // expected-error @below {{descriptor block and tensor element types must match, but got descriptor element type 'f16' and tensor element type 'f32'}}
    nvws.descriptor_store %desc[%i, %j] %src
        : !tt.tensordesc<16x16xf16, #shared16>,
          !ttg.memdesc<16x16xf32, #shared32, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_element_count_mismatch(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<8x16xf16, #shared, #smem, mutable>,
      %i: i32, %j: i32) {
    // expected-error @below {{descriptor block and tensor must have the same number of elements, but got descriptor block with 256 elements tensor with 128 elements}}
    nvws.descriptor_store %desc[%i, %j] %src
        : !tt.tensordesc<16x16xf16, #shared>,
          !ttg.memdesc<8x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_coordinate_count(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, %i: i32) {
    // expected-error @below {{expected 2 coordinates, but got 1}}
    nvws.descriptor_store %desc[%i] %src
        : !tt.tensordesc<16x16xf16, #shared>,
          !ttg.memdesc<16x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_requires_shared_source(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #ttng.shared_cluster_memory, mutable>,
      %i: i32, %j: i32) {
    // expected-error @below {{source must use shared memory, but got #ttng.shared_cluster_memory}}
    nvws.descriptor_store %desc[%i, %j] %src
        : !tt.tensordesc<16x16xf16, #shared>,
          !ttg.memdesc<16x16xf16, #shared, #ttng.shared_cluster_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_rejects_none(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %i: i32, %j: i32) {
    // expected-error @below {{reduction kind must not be none}}
    "nvws.descriptor_reduce"(%desc, %i, %j, %src)
        <{kind = 0 : i32,
          operandSegmentSizes = array<i32: 1, 2, 1, 0, 0, 0, 0>}>
        : (!tt.tensordesc<16x16xf16, #shared>, i32, i32,
           !ttg.memdesc<16x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_requires_barrier_predicate_pairs(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %i: i32, %j: i32) {
    // expected-error @below {{expected one predicate for every completion barrier}}
    "nvws.descriptor_store"(%desc, %i, %j, %src, %barrier)
        <{operandSegmentSizes = array<i32: 1, 2, 1, 1, 0, 0, 0>}>
        : (!tt.tensordesc<16x16xf16, #shared>, i32, i32,
           !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
           !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_reduce_requires_token_index_pairs(
      %desc: !tt.tensordesc<16x16xf16, #shared>,
      %src: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %token: tensor<2x!nvws.token>, %i: i32, %j: i32) {
    // expected-error @below {{expected one index for every deferred NVWS token}}
    "nvws.descriptor_reduce"(%desc, %i, %j, %src, %token)
        <{kind = 1 : i32,
          operandSegmentSizes = array<i32: 1, 2, 1, 0, 0, 1, 0>}>
        : (!tt.tensordesc<16x16xf16, #shared>, i32, i32,
           !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
           tensor<2x!nvws.token>) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_leading_dims_mismatch(%d : !ttg.memdesc<1x1xi32, #shared0, #smem>, %e : !ttg.memdesc<2x1xi32, #shared0, #smem>) {
    // expected-error @below {{Leading dims of sliced semaphore inputs don't match}}
    %sem = nvws.semaphore.create %d, %e released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem>, !ttg.memdesc<2x1xi32, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_buffer_used_elsewhere(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>) {
    // expected-error @below {{Semaphore buffer is used elsewhere, Semaphore cannot guarantee async safety}}
    %sem = nvws.semaphore.create %d released = 1 : !nvws.semaphore<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>]>
    %tmp = ttng.tmem_alloc %d : (!ttg.memdesc<1x64x16xf16, #shared0, #smem>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_partial_overlap_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %c = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %a, %c : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %c : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_create_permuted_buffer_tuple_mismatch() {
    %c0_i32 = arith.constant 0 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem0 = nvws.semaphore.create %a, %b released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    // expected-error @below {{semaphores sharing a backing buffer must use identical ordered buffer operands}}
    %sem1 = nvws.semaphore.create %b, %a : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem1[%c0_i32, %c0_i32] : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    ttg.local_dealloc %a : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %b : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_arity_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Semaphore has different number of arguments than buffer}}
    %views:2 = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared0, #smem, mutable>, !ttg.memdesc<1xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_dimensions_mismatch() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Dimensions don't match}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<2xi32, #shared0, #smem, mutable>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @semaphore_buffer_result_must_be_mutable() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    %sem = nvws.semaphore.create %buf released = 1 : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]> -> !ttg.async.token
    // expected-error @below {{Semaphore buffer result memdesc must be mutable}}
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared0, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared0, #smem>
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared0, #smem, mutable>
    tt.return
  }
}
