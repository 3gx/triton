// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s
// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse 2>&1 >/dev/null | FileCheck %s --check-prefix=DAG

// TOKEN RETENTION (spec wave-locality section, user ruling 11jun26):
// a partition's token from its previous wave stays usable by that
// partition. A touch whose ordering is already implied (every
// conflicting holder transitively synchronized behind it) rides the
// retained token instead of opening a new wave - no redundant
// semaphore (the meta-FA S2/S6 class: same source completion, same
// acquiring partition, later in its program order; the wait could
// never block).
//
//   1. @retention_merges_second_wave - {1} writes the spanning acc;
//      {2} reads it (acquire), stores a sliver, hands to {0}; {2}'s
//      LATER store into p (overlaps only the acc) needs no new
//      acquire: the acc completion was already waited on by {2}.
//   2. @retention_blocked_by_conflict - same shape, but {0} WRITES the
//      p region between {2}'s waves: the conflict demands a real edge,
//      so {2}'s second wave must open with an acquire.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#half_blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#col_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem64 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 1, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // DAG-LABEL: function: @retention_merges_second_wave
  // DAG: GROUP buffer.id=530 memory=tmem members=3
  // DAG: pieces: P0=[0,64){m0,m2}c0 P1=[64,65){m0,m1}c0 P2=[65,128){m0}c0
  // DAG: SYNC-DAG
  // DAG: |  |- a  S3(2)  root  ; entry
  // DAG: |  |- scf.for (WS, tag=0) pieces{P0:W:{1},P1:W:{1},P2:W:{1}} parts{0,1,2} thread{c0:{1}}
  // DAG: |  |  |- W m0  ttng.tmem_store {1}
  // DAG: |  |  |- r  S0  {1} [none]
  // DAG: |  |  |- a  S0  {2}
  // DAG: |  |  |- R m0  ttng.tmem_load {2}
  // DAG: |  |  |- W m1  ttng.tmem_store {2}
  // DAG: |  |  |- r  S1  {2} [none]
  // DAG: |  |  |- r  S3  {2} [none]
  // DAG: |  |  |- a  S1  {0}
  // DAG: |  |  |- R m1  ttng.tmem_load {0}
  // DAG: |  |  |- r  S3  {0} [none]
  // DAG: |  |  |- W m2  ttng.tmem_store {2}
  // DAG: |  |  |- r  S2  {2} [none]
  // DAG: |  |  |- a  S2  {1}
  // DAG: |  |  |- R m2  ttng.tmem_load {1}
  // DAG: |  |  |- a  S3(2)  {1}
  // DAG: |  |  |- EXIT pieces{P0:W:{1},P1:W:{1},P2:W:{1}} yield{c0: a S3}
  // DAG: SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=2 entry inherit={@0.1}}
  // CHECK-LABEL: @retention_merges_second_wave
  tt.func @retention_merges_second_wave(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<128x1xf32, #col_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 530 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = ttng.tmem_subslice [[V1]] {N = 64 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V5:%.*]] = ttg.memdesc_reinterpret [[V4]] : !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] true {nvws.dag_pending_count = 2 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.create [[V1]], [[V5]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V13:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V11:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V12:%.*]] = [[V10]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      // the spanning acc [0,128), produced by {1}
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 530 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V14:%.*]]:3 = nvws.semaphore.buffer [[V6]], [[V12]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: [[V15:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // {2}'s FIRST wave: acquire acc, read it, store the sliver
      // CHECK: nvws.semaphore.release [[V7]], [[V12]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V16:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V17:%.*]]:3 = nvws.semaphore.buffer [[V7]], [[V16]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V17]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x1xf32, #blocked2> -> !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %alpha = ttng.tmem_alloc {buffer.id = 530 : i32, buffer.offset = 64 : i32, ttg.partition = array<i32: 2>} : () -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V8]], [[V16]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V6]], [[V16]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V19:%.*]]:3 = nvws.semaphore.buffer [[V8]], [[V18]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V19]]#1[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1> -> tensor<128x1xf32, #blocked2>
      ttng.tmem_store %cst1, %alpha, %true {ttg.partition = array<i32: 2>} : tensor<128x1xf32, #col_blocked> -> !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable>
      // {0} consumes the sliver (the carrier moves to {0})
      // CHECK: nvws.semaphore.release [[V6]], [[V18]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %av, %at = ttng.tmem_load %alpha[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x1xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x1xf32, #col_blocked>
      // CHECK: [[V20:%.*]]:3 = nvws.semaphore.buffer [[V7]], [[V16]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V20]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      "use_alpha"(%av) {ttg.partition = array<i32: 0>} : (tensor<128x1xf32, #col_blocked>) -> ()
      // {2}'s LATER write into p [0,64) - overlaps only the acc, whose
      // completion {2} already waited on: rides the RETAINED token.
      %p = ttng.tmem_alloc {buffer.id = 530 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V9]], [[V16]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V9]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V22:%.*]]:3 = nvws.semaphore.buffer [[V9]], [[V21]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x1xf32, #tmem2, #ttng.tensor_memory, mutable, 2x128x1>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V22]]#2[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x1xf32, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V23]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }

  // Same shape, but {0} WRITES into the p region between {2}'s waves:
  // {0} becomes an unsynchronized holder of p's piece, so {2}'s second
  // write MUST take a real edge (a fresh acquire - no merge).
  // DAG-LABEL: function: @retention_blocked_by_conflict
  // DAG: GROUP buffer.id=531 memory=tmem members=3
  // DAG: pieces: P0=[0,64){m0,m1,m2}c0 P1=[64,128){m0}c0
  // DAG: SYNC-DAG
  // DAG: |  |- a  S4  root  ; entry
  // DAG: |  |- scf.for (WS, tag=1) pieces{P0:W:{1},P1:W:{1}} parts{0,1,2} thread{c0:{1}}
  // DAG: |  |  |- W m0  ttng.tmem_store {1}
  // DAG: |  |  |- r  S0  {1} [none]
  // DAG: |  |  |- a  S0  {2}
  // DAG: |  |  |- R m0  ttng.tmem_load {2}
  // DAG: |  |  |- r  S1  {2} [none]
  // DAG: |  |  |- r  S4  {2} [none]
  // DAG: |  |  |- a  S1  {0}
  // DAG: |  |  |- W m1  ttng.tmem_store {0}
  // DAG: |  |  |- r  S2  {0} [none]
  // DAG: |  |  |- a  S2  {2}
  // DAG: |  |  |- W m2  ttng.tmem_store {2}
  // DAG: |  |  |- r  S3  {2} [none]
  // DAG: |  |  |- a  S3  {1}
  // DAG: |  |  |- R m2  ttng.tmem_load {1}
  // DAG: |  |  |- a  S4  {1}
  // DAG: |  |  |- EXIT pieces{P0:W:{1},P1:W:{1}} yield{c0: a S4}
  // DAG: SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=1} S4{count=1 entry inherit={@1.1}}
  // CHECK-LABEL: @retention_blocked_by_conflict
  tt.func @retention_blocked_by_conflict(%lb: i32, %ub: i32, %step: i32) {
    %c0 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst64 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #half_blocked>
    %true = arith.constant true
    // CHECK: [[V1:%.*]] = ttng.tmem_alloc {buffer.id = 531 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[V2:%.*]] = ttng.tmem_subslice [[V1]] {N = 0 : i32} : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[V3:%.*]] = ttg.memdesc_reinterpret [[V2]] : !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x128> -> !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
    // CHECK: [[V4:%.*]] = nvws.semaphore.create [[V1]], [[V3]], [[V3]] true {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V5:%.*]] = nvws.semaphore.create [[V1]], [[V3]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V6:%.*]] = nvws.semaphore.create [[V1]], [[V3]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V7:%.*]] = nvws.semaphore.create [[V1]], [[V3]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V8:%.*]] = nvws.semaphore.create [[V1]], [[V3]], [[V3]] false {nvws.dag_pending_count = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>
    // CHECK: [[V9:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    // CHECK: [[V12:%.*]]:2 = scf.for %{{[-A-Za-z0-9_.$#]+}} = %{{[-A-Za-z0-9_.$#]+}} to %{{[-A-Za-z0-9_.$#]+}} step %{{[-A-Za-z0-9_.$#]+}} iter_args([[V10:%.*]] = %{{[-A-Za-z0-9_.$#]+}}, [[V11:%.*]] = [[V9]]) -> (i32, !ttg.async.token)  : i32 {
    %r = scf.for %iv = %lb to %ub step %step iter_args(%i = %c0) -> (i32) : i32 {
      %acc, %tacc = ttng.tmem_alloc {buffer.id = 531 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: [[V13:%.*]]:3 = nvws.semaphore.buffer [[V4]], [[V11]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: [[V14:%.*]] = ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V13]]#0[], %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %acc0 = ttng.tmem_store %cst, %acc[%tacc], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V5]], [[V11]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V15:%.*]] = nvws.semaphore.acquire [[V5]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V16:%.*]]:3 = nvws.semaphore.buffer [[V5]], [[V15]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: nvws.semaphore.release [[V6]], [[V15]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[V4]], [[V15]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V17:%.*]] = nvws.semaphore.acquire [[V6]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V18:%.*]]:3 = nvws.semaphore.buffer [[V6]], [[V17]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V18]]#1, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      %accv, %acct = ttng.tmem_load %acc[%acc0] {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // {0} writes the p region: a conflicting holder {2} has NOT synced behind
      %q = ttng.tmem_alloc {buffer.id = 531 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V7]], [[V17]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V19:%.*]] = nvws.semaphore.acquire [[V7]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V20:%.*]]:3 = nvws.semaphore.buffer [[V7]], [[V19]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: ttng.tmem_store %{{[-A-Za-z0-9_.$#]+}}, [[V20]]#2, %{{[-A-Za-z0-9_.$#]+}} {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      ttng.tmem_store %cst64, %q, %true {ttg.partition = array<i32: 0>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // {2}'s second write into the same region: must acquire
      %p = ttng.tmem_alloc {buffer.id = 531 : i32, buffer.offset = 0 : i32, ttg.partition = array<i32: 2>} : () -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      // CHECK: nvws.semaphore.release [[V8]], [[V19]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[V21:%.*]] = nvws.semaphore.acquire [[V8]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: [[V22:%.*]]:3 = nvws.semaphore.buffer [[V8]], [[V21]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>, !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}}, %{{[-A-Za-z0-9_.$#]+}} = ttng.tmem_load [[V22]]#2[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable, 2x128x64> -> tensor<128x64xf32, #blocked1>
      ttng.tmem_store %cst64, %p, %true {ttg.partition = array<i32: 2>} : tensor<128x64xf32, #half_blocked> -> !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable>
      %pv, %pt = ttng.tmem_load %p[] {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf32, #tmem64, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #half_blocked>
      "use_p"(%pv) {ttg.partition = array<i32: 1>} : (tensor<128x64xf32, #half_blocked>) -> ()
      %j = arith.addi %i, %c0 {ttg.partition = array<i32: 0, 1, 2>} : i32
      // CHECK: [[V23:%.*]] = nvws.semaphore.acquire [[V4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x128x64xf32, #tmem1, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {{.*}}[[V23]]
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %j : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 1 : i32}
    "use_i32"(%r) : (i32) -> ()
    tt.return
  }
}
