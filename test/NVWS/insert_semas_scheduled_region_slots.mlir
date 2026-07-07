// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas | FileCheck %s --check-prefix=SEMA
// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas --nvws-assign-stage-phase -cse | FileCheck %s --check-prefix=ASP

// These tests cover slot replay across an atomic scheduled region.  The
// scf.if itself owns the stage-2 schedule; its child operations intentionally
// have no loop.stage/loop.cluster attributes.

#blocked64 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked128 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // Two fresh writes advance the depth-5 cursor from A's slot to B's slot.
  // The conditional C wave is a same-owner overwrite after B has been read,
  // so it reuses B's slot and does not advance the cursor.  The region-closing
  // release therefore supplies A three logical iterations later:
  //
  //   A(i) = 2i mod 5, B/C(i) = 2i+1 mod 5, A(i+3) = B/C(i).
  //
  // The only authored non-zero displacement is the A-read to B-write handoff.
  // SEMA-LABEL: @depth5_regular_atomic_if
  // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
  // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
  // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] true
  // SEMA: [[A_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[A_TO_B:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[B_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[C_EXIT:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: scf.for
  // SEMA: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]]
  // SEMA: nvws.semaphore.release [[A_FULL]]{{.*}}, [[A_TOK]]
  // SEMA: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]]
  // SEMA: [[TO_B:%.*]] = arith.constant {{.*}} 1 : i32
  // SEMA: nvws.semaphore.release [[A_TO_B]][[[TO_B]]], [[A_READ_TOK]]
  // SEMA: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]]
  // SEMA: nvws.semaphore.release [[B_FULL]]{{.*}}, [[B_TOK]]
  // SEMA: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]]
  // SEMA: scf.if
  // SEMA: ttg.local_store {{.*}}, {{.*}} {ttg.partition = array<i32: 0>}
  // SEMA: nvws.semaphore.release [[C_FULL]]{{.*}}, [[B_READ_TOK]]
  // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]]
  // SEMA: ttg.local_load
  // SEMA: nvws.semaphore.release [[C_EXIT]]{{.*}}, [[C_READ_TOK]]
  // SEMA: nvws.semaphore.acquire [[C_EXIT]]
  // SEMA: nvws.semaphore.release [[ENTRY]]{{.*}}, {{.*}}

  // After ASP, B's fresh acquire advances to [[B_SLOT]].  C is rendered
  // through the C member of B's consumer buffer tuple, proving that the atomic
  // region did not add a third cursor advance.  The closing ENTRY release uses
  // that same slot.
  // ASP-LABEL: @depth5_regular_atomic_if
  // ASP: [[ENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[A_FULL:%.*]] = nvws.semaphore.create
  // ASP: [[A_TO_B:%.*]] = nvws.semaphore.create
  // ASP: [[B_FULL:%.*]] = nvws.semaphore.create
  // ASP: [[C_FULL:%.*]] = nvws.semaphore.create
  // ASP: [[C_EXIT:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[CURSOR:%.*]] = {{%.*}}
  // ASP: [[A_SLOT:%.*]] = arith.select {{.*}} : i32
  // ASP: [[A_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[A_SLOT]], {{%.*}}]
  // ASP: [[A_READ_TOK:%.*]] = nvws.semaphore.acquire [[A_FULL]][[[A_SLOT]], {{%.*}}]
  // ASP: [[TO_B_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {{.*}} : i32
  // ASP: [[TO_B_REM:%.*]] = arith.remsi [[TO_B_RAW]], {{%.*}} {{.*}} : i32
  // ASP: [[TO_B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[TO_B_REM]] {{.*}} : i32
  // ASP: nvws.semaphore.release [[A_TO_B]][[[TO_B_SLOT]]], [[A_READ_TOK]]
  // ASP: [[B_SLOT_RAW:%.*]] = arith.addi [[A_SLOT]], {{%.*}} {{.*}} : i32
  // ASP: [[B_SLOT:%.*]] = arith.select {{.*}}, {{.*}}, [[B_SLOT_RAW]] {{.*}} : i32
  // ASP: [[B_TOK:%.*]] = nvws.semaphore.acquire [[A_TO_B]][[[B_SLOT]], {{%.*}}]
  // ASP: [[B_READ_TOK:%.*]] = nvws.semaphore.acquire [[B_FULL]][[[B_SLOT]], {{%.*}}]
  // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
  // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[B_FULL]][[[B_SLOT]]], [[B_READ_TOK]]
  // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
  // ASP: nvws.semaphore.release [[C_FULL]][[[B_SLOT]]], [[B_READ_TOK]]
  // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[B_SLOT]], {{%.*}}]
  // ASP: nvws.semaphore.release [[C_EXIT]][[[B_SLOT]]], [[C_READ_TOK]]
  // ASP: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_EXIT]][[[B_SLOT]], {{%.*}}]
  // ASP: scf.yield {{.*}} [[C_EXIT_TOK]], [[B_SLOT]],
  // ASP: scf.yield {{.*}} [[B_READ_TOK]], [[B_SLOT]],
  // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0
  tt.func @depth5_regular_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                    %cond: i1) {
    %a = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 5 : i32, buffer.id = 900 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_a"(%a_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_b"(%b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>) -> ()
      scf.if %cond {
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Consecutive A/B writes by the same owner share one token and therefore
  // make one fresh depth-3 cursor advance.  Both reads and the atomic C region
  // use that stage.  The closing release returns the same slot to A after one
  // full three-iteration orbit.
  // SEMA-LABEL: @depth3_same_owner_atomic_if
  // SEMA: [[A_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
  // SEMA: [[B_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
  // SEMA: [[C_BASE:%.*]] = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32}
  // SEMA: [[ENTRY:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] true
  // SEMA: [[AB_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[C_FULL:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: [[C_EXIT:%.*]] = nvws.semaphore.create [[A_BASE]], [[B_BASE]], [[C_BASE]] false
  // SEMA: scf.for
  // SEMA: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]]
  // SEMA: ttg.local_store {{.*}}, {{.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // SEMA: ttg.local_store {{.*}}, {{.*}} {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // SEMA: nvws.semaphore.release [[AB_FULL]], [[AB_TOK]]
  // SEMA: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]]
  // SEMA: ttg.local_load
  // SEMA: ttg.local_load
  // SEMA: scf.if
  // SEMA: ttg.local_store {{.*}}, {{.*}} {ttg.partition = array<i32: 0>}
  // SEMA: nvws.semaphore.release [[C_FULL]], [[AB_READ_TOK]]
  // SEMA: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]]
  // SEMA: nvws.semaphore.release [[C_EXIT]], [[C_READ_TOK]]
  // SEMA: nvws.semaphore.acquire [[C_EXIT]]
  // SEMA: nvws.semaphore.release [[ENTRY]], {{.*}}

  // ASP-LABEL: @depth3_same_owner_atomic_if
  // ASP: [[ENTRY:%.*]] = nvws.semaphore.create
  // ASP: [[AB_FULL:%.*]] = nvws.semaphore.create
  // ASP: [[C_FULL:%.*]] = nvws.semaphore.create
  // ASP: [[C_EXIT:%.*]] = nvws.semaphore.create
  // ASP: scf.for {{.*}} iter_args([[CURSOR:%.*]] = {{%.*}}
  // ASP: [[AB_SLOT:%.*]] = arith.select {{.*}} : i32
  // ASP: [[AB_TOK:%.*]] = nvws.semaphore.acquire [[ENTRY]][[[AB_SLOT]], {{%.*}}]
  // ASP: nvws.semaphore.release [[AB_FULL]][[[AB_SLOT]]], [[AB_TOK]]
  // ASP: [[AB_READ_TOK:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[AB_SLOT]], {{%.*}}]
  // ASP: [[AB_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]]
  // ASP: ttg.local_load [[AB_VIEWS]]#0
  // ASP: ttg.local_load [[AB_VIEWS]]#1
  // ASP: [[IF_RESULTS:%.*]]:4 = scf.if
  // ASP: [[C_VIEWS:%.*]]:3 = nvws.semaphore.buffer [[AB_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]]
  // ASP: ttg.local_store {{.*}}, [[C_VIEWS]]#2 {ttg.partition = array<i32: 0>}
  // ASP: nvws.semaphore.release [[C_FULL]][[[AB_SLOT]]], [[AB_READ_TOK]]
  // ASP: [[C_READ_TOK:%.*]] = nvws.semaphore.acquire [[C_FULL]][[[AB_SLOT]], {{%.*}}]
  // ASP: nvws.semaphore.release [[C_EXIT]][[[AB_SLOT]]], [[C_READ_TOK]]
  // ASP: [[C_EXIT_TOK:%.*]] = nvws.semaphore.acquire [[C_EXIT]][[[AB_SLOT]], {{%.*}}]
  // ASP: scf.yield {{.*}} [[C_EXIT_TOK]], [[AB_SLOT]],
  // ASP: scf.yield {{.*}} [[AB_READ_TOK]], [[AB_SLOT]],
  // ASP: nvws.semaphore.release [[ENTRY]][[[IF_RESULTS]]#1], [[IF_RESULTS]]#0
  tt.func @depth3_same_owner_atomic_if(%lb: i32, %ub: i32, %step: i32,
                                       %cond: i1) {
    %a = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %c = ttg.local_alloc {buffer.copy = 3 : i32, buffer.id = 901 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %a_value = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked64>
    %b_value = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #blocked64>
    %c_value = arith.constant dense<2.000000e+00> : tensor<128x128xf16, #blocked128>

    scf.for %iv = %lb to %ub step %step : i32 {
      ttg.local_store %a_value, %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      ttg.local_store %b_value, %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : tensor<128x64xf16, #blocked64> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %a_read = ttg.local_load %a {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      %b_read = ttg.local_load %b {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #blocked64>
      "consume_ab"(%a_read, %b_read) {loop.cluster = 5 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked64>, tensor<128x64xf16, #blocked64>) -> ()
      scf.if %cond {
        ttg.local_store %c_value, %c {ttg.partition = array<i32: 0>} : tensor<128x128xf16, #blocked128> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
        %c_read = ttg.local_load %c {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked128>
        "consume_c"(%c_read) {ttg.partition = array<i32: 1>} : (tensor<128x128xf16, #blocked128>) -> ()
      } else {
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
    } {tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32}
    tt.return
  }
}
