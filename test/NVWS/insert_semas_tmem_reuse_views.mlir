// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @tmem_reuse_views_end_of_insert_semas
  tt.func @tmem_reuse_views_end_of_insert_semas() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // CHECK: [[BASE:%.*]] = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32
    // CHECK-NOT: ttng.tmem_alloc {{.*}}buffer.id = 42 : i32
    // CHECK: [[ALIAS_SUB:%.*]] = ttng.tmem_subslice [[BASE]] {N = 64 : i32}
    // CHECK-NEXT: [[ALIAS_VIEW:%.*]] = ttg.memdesc_reinterpret [[ALIAS_SUB]] : {{.*}} -> !ttg.memdesc<1x128x1xf32
    // CHECK-NEXT: [[HALF_SUB:%.*]] = ttng.tmem_subslice [[BASE]] {N = 0 : i32}
    // CHECK-NEXT: [[HALF_VIEW:%.*]] = ttg.memdesc_reinterpret [[HALF_SUB]] : {{.*}} -> !ttg.memdesc<1x128x128xf16
    %alias = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
    %base = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %half = ttng.tmem_alloc {buffer.copy = 1 : i32, buffer.id = 42 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: "use"([[ALIAS_VIEW]], [[BASE]], [[HALF_VIEW]])
    "use"(%alias, %base, %half) : (!ttg.memdesc<1x128x1xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x128x128xf16, #tmem, #ttng.tensor_memory, mutable>) -> ()

    scf.for %iv = %c0 to %c1 step %c1 {
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 0 : i32}

    // CHECK: tt.return
    // CHECK-NOT: ttng.tmem_alloc {{.*}}buffer.id = 42 : i32
    tt.return
  }
}
