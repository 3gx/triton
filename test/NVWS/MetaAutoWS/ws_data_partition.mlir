// RUN: triton-opt %s -split-input-file --nvws-ws-data-partition=num-warp-groups=3 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_ws_cooperative_kernel
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_ws_cooperative_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %2 = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %3 = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %4:2 = scf.for %arg7 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        // CHECK: %[[#GA1:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
        // CHECK: %[[#GA2:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
        // After reordering, B load is moved right after A loads:
        // CHECK: %[[#GB:]] = tt.load {{.*}} : tensor<64x256x!tt.ptr<f16>
        %8 = tt.load %2 {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
        // CHECK: %[[#LA1:]] = ttg.local_alloc %[[#GA1]]
        // CHECK: %[[#LA2:]] = ttg.local_alloc %[[#GA2]]
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %10 = tt.load %3 {async_task_id = array<i32: 0>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        // CHECK: %[[#LB:]] = ttg.local_alloc %[[#GB]]
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        // CHECK: %[[#C1:]] = ttng.warp_group_dot %[[#LA1]], %[[#LB]], {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        // CHECK: %[[#C2:]] = ttng.warp_group_dot %[[#LA2]], %[[#LB]], {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        %12 = ttng.warp_group_dot %9, %11, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %13 = arith.addi %arg9, %c64_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %12, %13 : tensor<128x256xf32, #mma>, i32
      } {async_task_id = array<i32: 0, 1, 2>}
      %5 = arith.truncf %4#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %7 = tt.splat %arg2 {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
     // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
     // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
     tt.store %7, %6 {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    } {tt.data_partition_factor = 2 : i32}
    tt.return
  }
}

// -----

// CHECK-LABEL: @cross_dim_partition
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cross_dim_partition(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32) {
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant {async_task_id = array<i32: 1, 2>} dense<true> : tensor<128x128xi1, #blocked>
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_program_id y {async_task_id = array<i32: 0, 1, 2>} : i32
    %2 = tt.load %arg1 {async_task_id = array<i32: 0, 1, 2>} : !tt.ptr<i32>
    %3 = arith.extsi %arg8 {async_task_id = array<i32: 0>} : i32 to i64
    ttng.tensormap_create %arg6, %arg0, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg2, [%c64_i32, %c128_i32], [%arg8, %arg9], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg3, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg5, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %4 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %5 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %6 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %7 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    %8 = tt.descriptor_load %4[%0, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<128x128xbf16
    %10 = tt.descriptor_load %5[%1, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    // After reordering, second dot's loads are also moved before first dot:
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x128xbf16, {{.*}} * !ttg.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x128xbf16, {{.*}} * !ttg.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
     %12 = ttng.warp_group_dot %9, %11, %cst {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem> * !ttg.memdesc<128x128xbf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    %13 = arith.truncf %12 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
    %14 = ttg.local_alloc %13 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #mma>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %15 = tt.descriptor_load %6[%0, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %16 = ttg.local_alloc %15 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %17 = ttg.memdesc_trans %16 {async_task_id = array<i32: 1, 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<128x64xbf16, {{.*}} * !ttg.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<128x64xbf16, {{.*}} * !ttg.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
    %18 = ttng.warp_group_dot %17, %14, %cst {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem> * !ttg.memdesc<128x128xbf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    %19 = ttg.convert_layout %18 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %20 = arith.truncf %19 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %21 = tt.splat %arg4 {async_task_id = array<i32: 1, 2>} : !tt.ptr<bf16> -> tensor<1x128x!tt.ptr<bf16>, #blocked>
    %22 = tt.broadcast %21 {async_task_id = array<i32: 1, 2>} : tensor<1x128x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %23 = tt.atomic_rmw fadd, relaxed, gpu, %22, %20, %cst_0 {async_task_id = array<i32: 1, 2>} : (tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xbf16, #blocked>, tensor<128x128xi1, #blocked>) -> tensor<128x128xbf16, #blocked>
    tt.return
  }
}

// -----

// Test that loads are reordered by first-use position after data partitioning.
// B's descriptor_load appears before A's, but A's local_alloc appears before
// B's. After partitioning, loads should be reordered to A0, A1, B because
// A's partitioned local_allocs (the first uses of A0/A1) precede B's.
// CHECK-LABEL: @reorder_loads_to_first_use
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reorder_loads_to_first_use(%desc_a: !tt.tensordesc<tensor<128x64xf16>>, %desc_b: !tt.tensordesc<tensor<64x256xf16>>, %arg2: !tt.ptr<f16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %4:2 = scf.for %arg7 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        // B's descriptor_load comes first in the input IR.
        %10 = tt.descriptor_load %desc_b[%arg9, %0] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
        %8 = tt.descriptor_load %desc_a[%0, %arg9] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
        // A's local_alloc comes before B's local_alloc.
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        // After reordering, A loads (split) should appear before B load:
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16
        // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        %12 = ttng.warp_group_dot %9, %11, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %12, %arg9 : tensor<128x256xf32, #mma>, i32
      } {async_task_id = array<i32: 0, 1, 2>}
      %5 = arith.truncf %4#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %7 = tt.splat %arg2 {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      tt.store %7, %6 {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    } {tt.data_partition_factor = 2 : i32}
    tt.return
  }
}

// -----

// Test host-side TMA: TensorDescType passed as function argument.
// CHECK-LABEL: @host_tma_data_partition
// Function signature should show sliced descriptor block types:
// CHECK-SAME: !tt.tensordesc<tensor<64x64xf16>>
// CHECK-SAME: !tt.tensordesc<tensor<64x256xf16>>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @host_tma_data_partition(%desc_a: !tt.tensordesc<tensor<128x64xf16>>, %desc_b: !tt.tensordesc<tensor<64x256xf16>>, %arg2: !tt.ptr<f16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %4:2 = scf.for %arg7 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        // Two descriptor_load ops should be created from slicing A:
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16
        %8 = tt.descriptor_load %desc_a[%0, %arg9] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked>
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // B is not partitioned (partition is along M dim):
        // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16
        %10 = tt.descriptor_load %desc_b[%arg9, %0] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #blocked1>
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        %12 = ttng.warp_group_dot %9, %11, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %12, %arg9 : tensor<128x256xf32, #mma>, i32
      } {async_task_id = array<i32: 0, 1, 2>}
      %5 = arith.truncf %4#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %7 = tt.splat %arg2 {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      tt.store %7, %6 {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    } {tt.data_partition_factor = 2 : i32}
    tt.return
  }
}

// -----


// HSTU cross-attention backward (reduce_dq) inner loop, reduced from the real
// kernel, with tt.data_partition_factor = 2 on the loop. Partitioning splits
// the KV dimension (dim 0, 256 -> 128) across two warp groups. The dk
// accumulator is a use_acc MMAv5 accumulator zero-initialized by a pre-loop
// `tmem_store %cst, %dk`; that store's token seeds the loop-carried dk token.
// The fix in getBackwardSliceToPartition (TMEMAllocOp branch) pulls the
// initializing store into the partition scheme so it is sliced per group. Without
// it the store stays 256-row, the sliced accumulators reuse the original loop
// token, the unsliced 256-row dk chain survives dead-arg elimination, and later
// passes reject the 256-row TMEM allocation. Assert no 256-row TMEM survives and
// that every MMA (5) was split into 2 (10 total).

// CHECK-LABEL: @_hstu_attn_bwd_redq
// CHECK-COUNT-10: ttng.tc_gen5_mma
// CHECK-NOT: ttng.tc_gen5_mma
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.cluster-dim-x" = 1 : i32, "ttg.cluster-dim-y" = 1 : i32, "ttg.cluster-dim-z" = 1 : i32, ttg.early_tma_store_lowering = true, ttg.min_reg_auto_ws = 24 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_hstu_attn_bwd_redq(%Q: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %K: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %V: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DO: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %total_seq_len_q: i32 {tt.divisibility = 16 : i32}, %total_seq_len_kv: i32 {tt.divisibility = 16 : i32}, %seq_offsets: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %seq_offsets_q: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %DQ: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %DK: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %DV: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %stride_qm: i32 {tt.divisibility = 16 : i32}, %stride_qh: i32 {tt.divisibility = 16 : i32}, %stride_kn: i32 {tt.divisibility = 16 : i32}, %stride_kh: i32 {tt.divisibility = 16 : i32}, %stride_vn: i32 {tt.divisibility = 16 : i32}, %stride_vh: i32 {tt.divisibility = 16 : i32}, %stride_dom: i32 {tt.divisibility = 16 : i32}, %stride_doh: i32 {tt.divisibility = 16 : i32}, %stride_dqm: i32 {tt.divisibility = 16 : i32}, %stride_dqh: i32 {tt.divisibility = 16 : i32}, %stride_dkn: i32 {tt.divisibility = 16 : i32}, %stride_dkh: i32 {tt.divisibility = 16 : i32}, %stride_dvn: i32 {tt.divisibility = 16 : i32}, %stride_dvh: i32 {tt.divisibility = 16 : i32}, %alpha: f32, %max_seq_len: i32 {tt.divisibility = 16 : i32}, %attn_scale: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Delta: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %AUTOTUNE_MAX_SEQ_LEN: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #linear>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #linear1>
    %off_hz = tt.get_program_id x : i32
    %seq_start_kv = tt.addptr %seq_offsets, %off_hz : !tt.ptr<i64>, i32
    %seq_start_kv_2 = tt.load %seq_start_kv : !tt.ptr<i64>
    %seq_start_kv_3 = arith.trunci %seq_start_kv_2 : i64 to i32
    %seq_end_kv = tt.addptr %seq_start_kv, %c1_i32 : !tt.ptr<i64>, i32
    %seq_end_kv_4 = tt.load %seq_end_kv : !tt.ptr<i64>
    %seq_end_kv_5 = arith.trunci %seq_end_kv_4 : i64 to i32
    %seq_len_kv = arith.subi %seq_end_kv_5, %seq_start_kv_3 : i32
    %seq_start_q = tt.addptr %seq_offsets_q, %off_hz : !tt.ptr<i64>, i32
    %seq_start_q_6 = tt.load %seq_start_q : !tt.ptr<i64>
    %seq_start_q_7 = arith.trunci %seq_start_q_6 : i64 to i32
    %seq_end_q = tt.addptr %seq_start_q, %c1_i32 : !tt.ptr<i64>, i32
    %seq_end_q_8 = tt.load %seq_end_q : !tt.ptr<i64>
    %seq_end_q_9 = arith.trunci %seq_end_q_8 : i64 to i32
    %seq_len_q = arith.subi %seq_end_q_9, %seq_start_q_7 : i32
    %0 = arith.cmpi eq, %seq_len_kv, %c0_i32 : i32
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %desc_q = tt.make_tensor_descriptor %Q, [%total_seq_len_q, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %desc_k = tt.make_tensor_descriptor %K, [%total_seq_len_kv, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<256x128xbf16, #shared>>
    %desc_do = tt.make_tensor_descriptor %DO, [%total_seq_len_q, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<64x128xbf16, #shared>>
    %desc_dk = tt.make_tensor_descriptor %DK, [%seq_end_kv_5, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<bf16>, !tt.tensordesc<tensor<256x128xbf16, #shared>>
    %desc_dq = tt.make_tensor_descriptor %DQ, [%seq_end_q_9, %c128_i32], [%c128_i64, %c1_i64] : !tt.ptr<f32>, !tt.tensordesc<tensor<64x128xf32, #shared1>>
    %M_off = tt.addptr %M, %seq_start_q_7 : !tt.ptr<f32>, i32
    %Delta_off = tt.addptr %Delta, %seq_start_q_7 : !tt.ptr<f32>, i32
    %offs_n = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %scaled_alpha = arith.mulf %alpha, %cst : f32
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %mask_m = tt.splat %seq_len_q : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %valid_mask_trans = tt.splat %seq_len_q : i32 -> tensor<1x64xi32, #linear>
    %valid_mask_trans_10 = tt.splat %seq_len_kv : i32 -> tensor<256x1xi32, #linear>
    %qk_trans = tt.splat %scaled_alpha : f32 -> tensor<256x64xf32, #linear>
    %m = tt.splat %M_off : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>
    %Di = tt.splat %Delta_off : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>
    %dk = tt.splat %alpha : f32 -> tensor<256x128xf32, #linear1>
    %dq_trans = tt.splat %alpha : f32 -> tensor<128x64xf32, #linear2>
    scf.for %start_n = %c0_i32 to %seq_len_kv step %c256_i32  : i32 {
      %kv_offset = arith.addi %seq_start_kv_3, %start_n : i32
      %k = tt.descriptor_load %desc_k[%kv_offset, %c0_i32] : !tt.tensordesc<tensor<256x128xbf16, #shared>> -> tensor<256x128xbf16, #blocked>
      %k_11 = ttg.local_alloc %k : (tensor<256x128xbf16, #blocked>) -> !ttg.memdesc<256x128xbf16, #shared, #smem>
      %offs_n_12 = tt.splat %start_n : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %offs_n_13 = arith.addi %offs_n_12, %offs_n : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>>
      %valid_mask_trans_14 = tt.expand_dims %offs_n_13 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<256x1xi32, #linear>
      %valid_mask_trans_15 = arith.cmpi slt, %valid_mask_trans_14, %valid_mask_trans_10 : tensor<256x1xi32, #linear>
      %valid_mask_trans_16 = tt.broadcast %valid_mask_trans_15 : tensor<256x1xi1, #linear> -> tensor<256x64xi1, #linear>
      %dq_trans_17 = ttg.memdesc_trans %k_11 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xbf16, #shared, #smem> -> !ttg.memdesc<128x256xbf16, #shared2, #smem>
      %qk_trans_18, %qk_trans_19 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dact_qk_trans, %dact_qk_trans_20 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_21, %dk_22 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dq_trans_23, %dq_trans_24 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_attn, %dk_attn_25 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %dk_26 = ttng.tmem_store %cst_1, %dk_21[%dk_22], %true : tensor<256x128xf32, #linear1> -> !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %dk_27:6 = scf.for %start_m = %c0_i32 to %seq_len_q step %c64_i32 iter_args(%dk_30 = %false, %qk_trans_31 = %qk_trans_19, %dact_qk_trans_32 = %dact_qk_trans_20, %dk_33 = %dk_26, %dq_trans_34 = %dq_trans_24, %dk_attn_35 = %dk_attn_25) -> (i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
        %offs_m_36 = tt.splat %start_m : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %offs_m_37 = arith.addi %offs_m_36, %offs_m : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %mask_m_38 = arith.cmpi slt, %offs_m_37, %mask_m : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %q_offset = arith.addi %seq_start_q_7, %start_m : i32
        %q = tt.descriptor_load %desc_q[%q_offset, %c0_i32] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
        %q_39 = ttg.local_alloc %q : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %qk_trans_40 = ttg.memdesc_trans %q_39 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %qk_trans_41 = ttng.tc_gen5_mma %k_11, %qk_trans_40, %qk_trans_18[%qk_trans_31], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \220\22, \22channels\22: [\22opndD,tmem,1,2\22]}"} : !ttg.memdesc<256x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %valid_mask_trans_42 = tt.expand_dims %offs_m_37 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi32, #linear>
        %valid_mask_trans_43 = arith.cmpi slt, %valid_mask_trans_42, %valid_mask_trans : tensor<1x64xi32, #linear>
        %valid_mask_trans_44 = tt.broadcast %valid_mask_trans_43 : tensor<1x64xi1, #linear> -> tensor<256x64xi1, #linear>
        %valid_mask_trans_45 = arith.andi %valid_mask_trans_44, %valid_mask_trans_16 : tensor<256x64xi1, #linear>
        %qk_trans_46, %qk_trans_47 = ttng.tmem_load %qk_trans_18[%qk_trans_41] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #linear>
        %qk_trans_48 = arith.mulf %qk_trans_46, %qk_trans : tensor<256x64xf32, #linear>
        %m_49 = tt.addptr %m, %offs_m_37 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %m_50 = tt.load %m_49, %mask_m_38 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>
        %pT = tt.expand_dims %m_50 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %pT_51 = tt.broadcast %pT : tensor<1x64xf32, #linear> -> tensor<256x64xf32, #linear>
        %pT_52 = arith.subf %qk_trans_48, %pT_51 : tensor<256x64xf32, #linear>
        %pT_53 = math.exp2 %pT_52 : tensor<256x64xf32, #linear>
        %pT_54 = arith.select %valid_mask_trans_45, %pT_53, %cst_0 : tensor<256x64xi1, #linear>, tensor<256x64xf32, #linear>
        %act_qk_trans = arith.truncf %pT_54 : tensor<256x64xf32, #linear> to tensor<256x64xbf16, #linear>
        %dk_55 = ttng.tmem_alloc %act_qk_trans : (tensor<256x64xbf16, #linear>) -> !ttg.memdesc<256x64xbf16, #tmem, #ttng.tensor_memory>
        %do = tt.descriptor_load %desc_do[%q_offset, %c0_i32] : !tt.tensordesc<tensor<64x128xbf16, #shared>> -> tensor<64x128xbf16, #blocked>
        %do_56 = ttg.local_alloc %do : (tensor<64x128xbf16, #blocked>) -> !ttg.memdesc<64x128xbf16, #shared, #smem>
        %dact_qk_trans_57 = ttg.memdesc_trans %do_56 {order = array<i32: 1, 0>} : !ttg.memdesc<64x128xbf16, #shared, #smem> -> !ttg.memdesc<128x64xbf16, #shared2, #smem>
        %dact_qk_trans_58 = ttng.tc_gen5_mma %k_11, %dact_qk_trans_57, %dact_qk_trans[%dact_qk_trans_32], %false, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndD,tmem,1,11\22]}"} : !ttg.memdesc<256x128xbf16, #shared, #smem>, !ttg.memdesc<128x64xbf16, #shared2, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_59 = ttng.tc_gen5_mma %dk_55, %do_56, %dk_21[%dk_33], %dk_30, %true {tt.autows = "{\22stage\22: \220\22, \22order\22: \222\22, \22channels\22: [\22opndA,tmem,1,2\22, \22opndD,tmem,1,7\22]}"} : !ttg.memdesc<256x64xbf16, #tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %Di_60 = tt.addptr %Di, %offs_m_37 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
        %Di_61 = tt.load %Di_60, %mask_m_38 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #linear}>>
        %dqk_trans = tt.expand_dims %Di_61 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xf32, #linear>
        %dqk_trans_62 = tt.broadcast %dqk_trans : tensor<1x64xf32, #linear> -> tensor<256x64xf32, #linear>
        %dact_qk_trans_63, %dact_qk_trans_64 = ttng.tmem_load %dact_qk_trans[%dact_qk_trans_58] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #linear>
        %dqk_trans_65 = arith.subf %dact_qk_trans_63, %dqk_trans_62 : tensor<256x64xf32, #linear>
        %dqk_trans_66 = arith.mulf %pT_54, %dqk_trans_65 : tensor<256x64xf32, #linear>
        %dqk_trans_67 = arith.truncf %dqk_trans_66 : tensor<256x64xf32, #linear> to tensor<256x64xbf16, #linear>
        %dqk_trans_68 = ttg.local_alloc %dqk_trans_67 : (tensor<256x64xbf16, #linear>) -> !ttg.memdesc<256x64xbf16, #shared, #smem>
        %dq_trans_69 = ttng.tc_gen5_mma %dq_trans_17, %dqk_trans_68, %dq_trans_23[%dq_trans_34], %false, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndB,smem,1,8\22, \22opndD,tmem,1,11\22]}"} : !ttg.memdesc<128x256xbf16, #shared2, #smem>, !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
        %dk_attn_70 = ttng.tc_gen5_mma %dqk_trans_68, %q_39, %dk_attn[%dk_attn_35], %false, %true {tt.autows = "{\22stage\22: \221\22, \22order\22: \221\22, \22channels\22: [\22opndA,smem,1,8\22, \22opndD,tmem,1,10\22]}"} : !ttg.memdesc<256x64xbf16, #shared, #smem>, !ttg.memdesc<64x128xbf16, #shared, #smem>, !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        %dk_attn_71, %dk_attn_72 = ttng.tmem_load %dk_attn[%dk_attn_70] : !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear1>
        %dk_73 = arith.mulf %dk_attn_71, %dk : tensor<256x128xf32, #linear1>
        %dk_74, %dk_75 = ttng.tmem_load %dk_21[%dk_59] : !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear1>
        %dk_76 = arith.addf %dk_74, %dk_73 : tensor<256x128xf32, #linear1>
        %dq_trans_77, %dq_trans_78 = ttng.tmem_load %dq_trans_23[%dq_trans_69] : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #linear2>
        %dq_trans_79 = arith.mulf %dq_trans_77, %dq_trans : tensor<128x64xf32, #linear2>
        %dq = tt.trans %dq_trans_79 {order = array<i32: 1, 0>} : tensor<128x64xf32, #linear2> -> tensor<64x128xf32, #linear3>
        %3 = ttg.convert_layout %dq : tensor<64x128xf32, #linear3> -> tensor<64x128xf32, #blocked1>
        tt.descriptor_reduce add, %desc_dq[%q_offset, %c0_i32], %3 : !tt.tensordesc<tensor<64x128xf32, #shared1>>, tensor<64x128xf32, #blocked1>
        %dk_80 = ttng.tmem_store %dk_76, %dk_21[%dk_75], %true : tensor<256x128xf32, #linear1> -> !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable>
        scf.yield %true, %qk_trans_47, %dact_qk_trans_64, %dk_80, %dq_trans_78, %dk_attn_72 : i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
      } {tt.merge_epilogue = true, tt.warp_specialize, tt.data_partition_factor = 2 : i32}
      %dk_28, %dk_29 = ttng.tmem_load %dk_21[%dk_27#3] : !ttg.memdesc<256x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<256x128xf32, #linear1>
      %1 = arith.truncf %dk_28 : tensor<256x128xf32, #linear1> to tensor<256x128xbf16, #linear1>
      %2 = ttg.convert_layout %1 : tensor<256x128xbf16, #linear1> -> tensor<256x128xbf16, #blocked>
      tt.descriptor_store %desc_dk[%kv_offset, %c0_i32], %2 : !tt.tensordesc<tensor<256x128xbf16, #shared>>, tensor<256x128xbf16, #blocked>
    }
    tt.return
  }
}
