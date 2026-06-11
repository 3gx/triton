// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

// Dedicated mirror of the meta-FA stats group (GROUP buffer.id=4 in
// insert_semas_meta_fa_fwd: m0[64,65) m1[66,67) m2[65,66) m3[0,128)
// m4[0,64)): many members on ONE backing buffer, with a MIX of
// overlapping and non-overlapping extents. Three functions:
//
//   1. @tmem_mixed_overlap_spanning_member  - the FA shape: a spanning
//      member bridges four pairwise-disjoint members into one component.
//   2. @tmem_disjoint_slivers_cross_partition - same slivers, same
//      cross-partition accesses, spanning member DELETED: the group
//      dissolves into independent components; each sliver pays only for
//      its own producer->consumer handoff, never for a neighbor.
//   3. @tmem_disjoint_slivers_same_owner - same slivers, each written
//      and read by ONE partition: zero semaphores in the whole function.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#half_blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#col_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // memory:   0        64  65  66  67      128
  // m3 (acc)  [==============================)    <- spans everything
  // m4 (p)    [========)
  // m0 (alpha)         [--)
  // m2 (l)                 [--)
  // m1 (m)                     [--)
  //
  // conflict graph (who overlaps whom):
  //
  //       m4 --- m3 --- m0       m3 is the HUB: it overlaps every
  //               | \            sliver, so every sliver must sync
  //              m2  m1          with m3's writes/reads.
  //                              No sliver<->sliver edge exists!
  //
  // components: ONE (m3 bridges them all into c0). One carrier chain
  // threads the whole group: every sliver's W/R weaves acquire/release
  // against the shared semaphores, but slivers never sync with each
  // other directly - overlap is priced per shared piece.
  //
  // Owners form a ring: slivers W{1}->R{2}, acc W{2}->R{0}, p W{0}->R{1}.
  // DAG-LABEL: function: @tmem_mixed_overlap_spanning_member
  // DAG: GROUP buffer.id=520 memory=tmem members=5
  // DAG: members: m0[64,65) m1[66,67) m2[65,66) m3[0,128) m4[0,64)
  // DAG: pieces: P0=[0,64){m3,m4}c0 P1=[64,65){m0,m3}c0 P2=[65,66){m2,m3}c0 P3=[66,67){m1,m3}c0 P4=[67,128){m3}c0
  // DAG: footprints: m0={P1} m1={P3} m2={P2} m3={P0,P1,P2,P3,P4} m4={P0}
  // DAG: SYNC-DAG
  // DAG: |  |- a  S7(2)  root  ; entry
  // DAG: |  |  |- W m0  ttng.tmem_store {1}
  // DAG: |  |  |- r  S0  {1} [none]
  // DAG: |  |  |- a  S0  {2}
  // DAG: |  |  |- R m0  ttng.tmem_load {2}
  // DAG: |  |  |- r  S1  {2} [none]
  // DAG: |  |  |- a  S1  {1}
  // DAG: |  |  |- W m1  ttng.tmem_store {1}
  // DAG: |  |  |- r  S2  {1} [none]
  // DAG: |  |  |- a  S2  {2}
  // DAG: |  |  |- R m1  ttng.tmem_load {2}
  // DAG: |  |  |- r  S3  {2} [none]
  // DAG: |  |  |- a  S3  {1}
  // DAG: |  |  |- W m2  ttng.tmem_store {1}
  // DAG: |  |  |- r  S4  {1} [none]
  // DAG: |  |  |- a  S4  {2}
  // DAG: |  |  |- R m2  ttng.tmem_load {2}
  // DAG: |  |  |- W m3  ttng.tmem_store {2}
  // DAG: |  |  |- r  S5  {2} [none]
  // DAG: |  |  |- r  S7  {2} [none]
  // DAG: |  |  |- a  S5  {0}
  // DAG: |  |  |- R m3  ttng.tmem_load {0}
  // DAG: |  |  |- r  S7  {0} [none]
  // DAG: |  |  |- W m4  ttng.tmem_store {0}
  // DAG: |  |  |- r  S6  {0} [none]
  // DAG: |  |  |- a  S6  {1}
  // DAG: |  |  |- R m4  ttng.tmem_load {1}
  // DAG: |  |  |- a  S7(2)  {1}
  // DAG: SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=1} S4{count=1} S5{count=1} S6{count=1} S7{count=2 entry inherit={@0.1}}
  // CHECK-LABEL: @tmem_mixed_overlap_spanning_member
  tt.func @tmem_mixed_overlap_spanning_member(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_subslice [[V1]] {N = 65 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V5:%.*]] = ttg.memdesc_reinterpret [[V4]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = ttng.tmem_subslice [[V1]] {N = 66 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V7:%.*]] = ttg.memdesc_reinterpret [[V6]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V8:%.*]] = ttng.tmem_subslice [[V1]] {N = 64 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V9:%.*]] = ttg.memdesc_reinterpret [[V8]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] true {nvws.dag_pending_count = 2 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V11:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V12:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V14:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V15:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V9]], [[V7]], [[V5]], [[V1]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V21:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V19:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V20:%.*]] = [[V18]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // m0: alpha sliver at column 64, produced by {1}, consumed by {2}.
      %alpha = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V22:%.*]]:5 = nvws.semaphore.buffer [[V10]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V22]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V11]], [[V20]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V11]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V24:%.*]]:5 = nvws.semaphore.buffer [[V11]], [[V23]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V24]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: nvws.semaphore.release [[V12]], [[V23]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_alpha"(%av) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m1: m sliver at column 66, produced by {1}, consumed by {2}.
      %m = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V25:%.*]] = nvws.semaphore.acquire [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V26:%.*]]:5 = nvws.semaphore.buffer [[V12]], [[V25]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V26]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V13]], [[V25]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V27:%.*]] = nvws.semaphore.acquire [[V13]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V28:%.*]]:5 = nvws.semaphore.buffer [[V13]], [[V27]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V28]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: nvws.semaphore.release [[V14]], [[V27]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_m"(%mv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m2: l sliver at column 65, produced by {1}, consumed by {2}.
      %l = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V29:%.*]] = nvws.semaphore.acquire [[V14]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V30:%.*]]:5 = nvws.semaphore.buffer [[V14]], [[V29]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V30]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V15]], [[V29]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V31:%.*]] = nvws.semaphore.acquire [[V15]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V32:%.*]]:5 = nvws.semaphore.buffer [[V15]], [[V31]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V32]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_l"(%lv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // m3: the spanning accumulator [0,128), produced by {2}, consumed
      // by {0}. It overlaps ALL other members.
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V33:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V32]]#3[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V16]], [[V31]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V10]], [[V31]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V34:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V35:%.*]]:5 = nvws.semaphore.buffer [[V16]], [[V34]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V35]]#3[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.semaphore.release [[V10]], [[V34]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_acc"(%accv) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // m4: p at [0,64) - disjoint from every sliver, overlaps only the
      // accumulator. Produced by {0}, consumed by {1}.
      %p = ttng.tmem_alloc {buffer.id = 520 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V35]]#4, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V17]], [[V34]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V36:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V37:%.*]]:5 = nvws.semaphore.buffer [[V17]], [[V36]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V37]]#4[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[V38:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V38]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // memory:   0        64  65  66  67      128
  // m3 (p)    [========)                       <- the spanning member is
  // m0 (alpha)         [--)                       GONE; p keeps [0,64)
  // m2 (l)                 [--)
  // m1 (m)                     [--)
  //
  // conflict graph:
  //
  //       p     alpha    l     m        no edges at all
  //
  // components: FOUR - each member is its own island. Sharing buffer.id
  // alone costs nothing. Accesses are the SAME cross-partition pattern
  // as above (slivers W{1}->R{2}, p W{0}->R{1}), so each member still
  // pays for its OWN producer->consumer handoff - but the golden diff
  // against function 1 shows exactly what the spanning overlap costs:
  // the cross-member weave, and nothing else.
  // DAG-LABEL: function: @tmem_disjoint_slivers_cross_partition
  // DAG: GROUP buffer.id=521 memory=tmem members=4
  // DAG: members: m0[64,65) m1[66,67) m2[65,66) m3[0,64)
  // DAG: pieces: P0=[0,64){m3}c0 P1=[64,65){m0}c1 P2=[65,66){m2}c2 P3=[66,67){m1}c3
  // DAG: SYNC-DAG
  // DAG: |  |- a  S3  root  ; entry
  // DAG: |  |- a  S5  root  ; entry
  // DAG: |  |- a  S6  root  ; entry
  // DAG: |  |- a  S7  root  ; entry
  // DAG: |  |  |- W m0  ttng.tmem_store {1}
  // DAG: |  |  |- r  S0  {1} [none]
  // DAG: |  |  |- a  S0  {2}
  // DAG: |  |  |- R m0  ttng.tmem_load {2}
  // DAG: |  |  |- r  S5  {2} [none]
  // DAG: |  |  |- W m1  ttng.tmem_store {1}
  // DAG: |  |  |- r  S1  {1} [none]
  // DAG: |  |  |- a  S1  {2}
  // DAG: |  |  |- R m1  ttng.tmem_load {2}
  // DAG: |  |  |- r  S7  {2} [none]
  // DAG: |  |  |- W m2  ttng.tmem_store {1}
  // DAG: |  |  |- r  S2  {1} [none]
  // DAG: |  |  |- a  S2  {2}
  // DAG: |  |  |- R m2  ttng.tmem_load {2}
  // DAG: |  |  |- r  S6  {2} [none]
  // DAG: |  |  |- a  S4  {0}
  // DAG: |  |  |- W m3  ttng.tmem_store {0}
  // DAG: |  |  |- r  S3  {0} [none]
  // DAG: |  |  |- a  S3  {1}
  // DAG: |  |  |- R m3  ttng.tmem_load {1}
  // DAG: |  |  |- r  S4  {1} [none]
  // DAG: |  |  |- a  S5  {1}
  // DAG: |  |  |- a  S6  {1}
  // DAG: |  |  |- a  S7  {1}
  // DAG: |  |  |- EXIT pieces{P0:W:{0},P1:W:{1},P2:W:{1},P3:W:{1}} yield{c0: a S3,c1: a S5,c2: a S6,c3: a S7}
  // DAG: SEMAS c0: S3{count=1 entry inherit={@1.0}} S4{count=1 entry inherit={@1.0}}
  // DAG: SEMAS c1: S0{count=1} S5{count=1 entry inherit={@1.1}}
  // DAG: SEMAS c2: S2{count=1} S6{count=1 entry inherit={@1.1}}
  // DAG: SEMAS c3: S1{count=1} S7{count=1 entry inherit={@1.1}}
  // CHECK-LABEL: @tmem_disjoint_slivers_cross_partition
  tt.func @tmem_disjoint_slivers_cross_partition(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 64 : i32} : () -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 66 : i32} : () -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V3:%.*]] = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 65 : i32} : () -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V10:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V11:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V12:%.*]] = nvws.semaphore.create [[V1]], [[V2]], [[V3]], [[V4]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V14:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V22:%.*]]:5 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V17:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V18:%.*]] = [[V13]], [[V19:%.*]] = [[V14]], [[V20:%.*]] = [[V15]], [[V21:%.*]] = [[V16]]) -> (i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %alpha = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V23:%.*]]:4 = nvws.semaphore.buffer [[V7]], [[V19]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V23]]#0, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V10]], [[V19]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V24:%.*]] = nvws.semaphore.acquire [[V10]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V25:%.*]]:4 = nvws.semaphore.buffer [[V10]], [[V24]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V25]]#0[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: nvws.semaphore.release [[V7]], [[V24]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_alpha"(%av) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      %m = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V26:%.*]]:4 = nvws.semaphore.buffer [[V9]], [[V21]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V26]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V11]], [[V21]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V27:%.*]] = nvws.semaphore.acquire [[V11]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V28:%.*]]:4 = nvws.semaphore.buffer [[V11]], [[V27]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V28]]#1[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: nvws.semaphore.release [[V9]], [[V27]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_m"(%mv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      %l = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V29:%.*]]:4 = nvws.semaphore.buffer [[V8]], [[V20]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V29]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V12]], [[V20]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V30:%.*]] = nvws.semaphore.acquire [[V12]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V31:%.*]]:4 = nvws.semaphore.buffer [[V12]], [[V30]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V31]]#2[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: nvws.semaphore.release [[V8]], [[V30]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_l"(%lv) {ttg.partition = array<i32: 2>} : (tensor<128x1xf32, #col_blocked>) -> ()

      %p = ttng.tmem_alloc {buffer.id = 521 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: [[V32:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V33:%.*]]:4 = nvws.semaphore.buffer [[V6]], [[V32]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V33]]#3, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[V32]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V34:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V35:%.*]]:4 = nvws.semaphore.buffer [[V5]], [[V34]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V35]]#3[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      // CHECK: nvws.semaphore.release [[V6]], [[V34]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[V36:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V37:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V38:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V34]], [[V36]], [[V37]], [[V38]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // Same disjoint layout as function 2, but every member is written AND
  // read by one partition (slivers in {1}, p in {0}): per-partition
  // program order covers everything, the pass must emit NOTHING.
  //
  //       p{0}   alpha{1}   l{1}   m{1}     no edges, no handoffs
  //
  // DAG-LABEL: function: @tmem_disjoint_slivers_same_owner
  // DAG: GROUP buffer.id=522 memory=tmem members=4
  // DAG: pieces: P0=[0,64){m3}c0 P1=[64,65){m0}c1 P2=[65,66){m2}c2 P3=[66,67){m1}c3
  // DAG: SYNC-DAG
  // Four single-owner components: no acquire/release rows at all.
  // DAG-NOT: |- a  S
  // DAG-NOT: |- r  S
  // CHECK-LABEL: @tmem_disjoint_slivers_same_owner
  tt.func @tmem_disjoint_slivers_same_owner(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V2:%.*]] = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V1:%.*]] = %{{[-A-Za-z0-9_.$#]+}}) -> (i32)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // CHECK-NOT: nvws.semaphore
      %alpha = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V3:%.*]] = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V3]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V3]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked2>
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_alpha"(%av) {ttg.partition = array<i32: 1>} : (tensor<128x1xf32, #col_blocked>) -> ()

      %m = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V4:%.*]] = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 66 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V4]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst1, %m, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V4]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked2>
      %mv, %mt = ttng.tmem_load %m[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_m"(%mv) {ttg.partition = array<i32: 1>} : (tensor<128x1xf32, #col_blocked>) -> ()

      // CHECK-NOT: nvws.semaphore
      %l = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: [[V5:%.*]] = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 65 : i32, ttg.partition = array<i32: 1>} : () -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V5]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst1, %l, %true {ttg.partition = array<i32: 1>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V5]][] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #blocked2>
      %lv, %lt = ttng.tmem_load %l[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      "use_l"(%lv) {ttg.partition = array<i32: 1>} : (tensor<128x1xf32, #col_blocked>) -> ()

      %p = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: [[V6:%.*]] = ttng.tmem_alloc {buffer.id = 522 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V6]], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V6]][] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 0>} : (tensor<128x64xf32, #half_blocked>) -> ()
      // CHECK-NOT: nvws.semaphore

      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1>} : i32
      scf.yield {ttg.partition = array<i32: 0, 1>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>], ttg.warp_specialize.tag = 2 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
