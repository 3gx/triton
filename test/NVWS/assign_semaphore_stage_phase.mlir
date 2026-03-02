// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore --nvws-assign-semaphore-stage-phase | FileCheck %s --implicit-check-not=nvws.aref.











#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  tt.func @assign_stage_basic(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C0:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // CHECK: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C0]], [[PHASE:%.*]] = [[CM1]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[RAWSTAGE:%.*]] = arith.andi [[STAGE]], {{%.*}}
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[RAWSTAGE]], [[PHASE]]]
      // CHECK: [[C1:%.*]] = arith.constant {{.*}} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[RAWSTAGE]]
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]]
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %val = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      ttg.local_store %val, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // stage advancement: extract wasObserved from bit 31, compute shouldAdvance, wrap stage
      // CHECK: [[NEXT:%.*]] = arith.addi {{%.*}}, {{%.*}}
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], {{%.*}}
      // CHECK: [[WRAPPED:%.*]] = arith.select [[WRAP]], {{%.*}}, [[NEXT]]
      // CHECK: [[ADV_STAGE:%.*]] = arith.select {{%.*}}, [[WRAPPED]], {{%.*}}
      // CHECK: [[NEW_STAGE:%.*]] = arith.select {{%.*}}, [[ADV_STAGE]], {{%.*}}
      // CHECK: nvws.semaphore.release [[SEM]][[[RAWSTAGE]]], [[TOK]] [#nvws.async_op<none>]
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
      // CHECK: scf.yield {{.*}} [[NEW_STAGE]], [[PHASE_NEW]] : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}








// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @shared_stage_two_semaphores
  tt.func @shared_stage_two_semaphores() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM0:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[S0:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[SEM1:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: [[S1:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF0:%.*]] = arith.constant 0 : i32
    %sem0 = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    %sem1 = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // CHECK: [[MASK0:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RS0:%.*]] = arith.andi [[S0]], [[MASK0]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[SEM0]][[[RS0]], [[CM1]]]
    // CHECK: [[C1_0:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH0:%.*]] = arith.shli [[C1_0]], [[RS0]] : i32
    // CHECK: [[PE_NEW:%.*]] = arith.xori [[CM1]], [[SH0]] : i32
    // CHECK: [[MASK1:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RS1:%.*]] = arith.andi [[S1]], [[MASK1]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[SEM1]][[[RS1]], [[PF0]]]
    // CHECK: [[C1_1:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH1:%.*]] = arith.shli [[C1_1]], [[RS1]] : i32
    // CHECK: [[PF_NEW:%.*]] = arith.xori [[PF0]], [[SH1]] : i32
    %tok0 = nvws.semaphore.acquire %sem0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %tok1 = nvws.semaphore.acquire %sem1 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token

    %view0 = nvws.semaphore.buffer %sem0, %tok0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    %v = arith.constant dense<0> : !elt
    ttg.local_store %v, %view0 : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

    // CHECK: nvws.semaphore.release [[SEM0]][[[RS0]]], [[TOK0]] [#nvws.async_op<none>]
    // CHECK: nvws.semaphore.release [[SEM1]][[[RS1]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.semaphore.release %sem0, %tok0 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    nvws.semaphore.release %sem1, %tok1 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @if_observation
  tt.func @if_observation(%cond: i1, %lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[SEM:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C0:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    // CHECK: scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C0]], [[PHASE:%.*]] = [[CM1]]) -> (i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2147483647 : i32
      // CHECK: [[RS:%.*]] = arith.andi [[STAGE]], [[MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[TOK:%.*]] = nvws.semaphore.acquire [[SEM]][[[RS]], [[PHASE]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[C1:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[SHIFT:%.*]] = arith.shli [[C1]], [[RS]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PHASE_NEW:%.*]] = arith.xori [[PHASE]], [[SHIFT]] {ttg.partition = array<i32: 0>} : i32
      %tok = nvws.semaphore.acquire %sem {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
      %view = nvws.semaphore.buffer %sem, %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

      // CHECK: [[IF:%.*]]:2 = scf.if %{{.*}} -> (i32, i32)
      // CHECK:   arith.ori [[STAGE]], {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK:   scf.yield {ttg.partition = array<i32: 0>} {{%.*}}, [[PHASE_NEW]] : i32, i32
      // CHECK: } else {
      // CHECK:   scf.yield {ttg.partition = array<i32: 0>} [[STAGE]], [[PHASE_NEW]] : i32, i32
      // CHECK: }
      scf.if %cond {
        %x = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
        "use"(%x) {ttg.partition = array<i32: 0>} : (!elt) -> ()
      } {ttg.partition = array<i32: 0>}

      %v = arith.constant {ttg.partition = array<i32: 0>} dense<0> : !elt
      // Stage advancement after FreshWrite
      // CHECK: [[WASOBS:%.*]] = arith.shrui [[IF]]#0, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: arith.trunci [[WASOBS]] {ttg.partition = array<i32: 0>} : i32 to i1
      // CHECK: [[NEXT:%.*]] = arith.addi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ADV:%.*]] = arith.select {{%.*}}, {{%.*}}, [[IF]]#0 {ttg.partition = array<i32: 0>} : i32
      ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      // CHECK: nvws.semaphore.release [[SEM]][[[RS]]], [[TOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[ADV]], [[IF]]#1 : i32, i32
    } {ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %ub = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true
    // CHECK: [[C0:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false
    // CHECK: [[C0_CONS:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF0:%.*]] = arith.constant 0 : i32
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[LOOP:%.*]]:6 = scf.for {{.*}} iter_args([[STAGE:%.*]] = [[C0]], [[PE:%.*]] = [[CM1]], [[STAGE_C1:%.*]] = [[C0_CONS]], [[PF1:%.*]] = [[PF0]], [[STAGE_C2:%.*]] = [[C0_CONS]], [[PF2:%.*]] = [[PF0]]) -> (i32, i32, i32, i32, i32, i32)
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      // CHECK: [[MASK_P:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2147483647 : i32
      // CHECK: [[RAW_STAGE:%.*]] = arith.andi [[STAGE]], [[MASK_P]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW_STAGE]], [[PE]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[C1P:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSHIFT:%.*]] = arith.shli [[C1P]], [[RAW_STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE_NEW:%.*]] = arith.xori [[PE]], [[PSHIFT]] {ttg.partition = array<i32: 0>} : i32
      // wasObserved extraction and stage advancement
      // CHECK: [[C3:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 3 : i32
      // CHECK: [[NEXT:%.*]] = arith.addi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[WRAP:%.*]] = arith.cmpi eq, [[NEXT]], [[C3]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[WRAPPED:%.*]] = arith.select [[WRAP]], {{%.*}}, [[NEXT]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[SEL1:%.*]] = arith.select {{%.*}}, [[WRAPPED]], {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ADV_STAGE:%.*]] = arith.select {{%.*}}, [[SEL1]], [[STAGE]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: nvws.semaphore.release [[FULL]][[[RAW_STAGE]]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %2, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK: [[MASK_C1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 2147483647 : i32
      // CHECK: [[RAW_C1:%.*]] = arith.andi [[STAGE_C1]], [[MASK_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[RAW_C1]], [[PF1]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[C1G1:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSHIFT1:%.*]] = arith.shli [[C1G1]], [[RAW_C1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF1_NEW:%.*]] = arith.xori [[PF1]], [[GSHIFT1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[OBS_C1:%.*]] = arith.ori [[STAGE_C1]], {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]][[[RAW_C1]]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %3 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: [[MASK_C2:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 2147483647 : i32
      // CHECK: [[RAW_C2:%.*]] = arith.andi [[STAGE_C2]], [[MASK_C2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]][[[RAW_C2]], [[PF2]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[C1G2:%.*]] = arith.constant {ttg.partition = array<i32: 2>} 1 : i32
      // CHECK: [[GSHIFT2:%.*]] = arith.shli [[C1G2]], [[RAW_C2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[PF2_NEW:%.*]] = arith.xori [[PF2]], [[GSHIFT2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[OBS_C2:%.*]] = arith.ori [[STAGE_C2]], {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]][[[RAW_C2]]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %4 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[ADV_STAGE]], [[PE_NEW]], [[OBS_C1]], [[PF1_NEW]], [[OBS_C2]], [[PF2_NEW]] : i32, i32, i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 1>, array<i32: 2>, array<i32: 2>], ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}

    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @aref_lowering
  tt.func @aref_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %f : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %g : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32

    // CHECK: [[E0:%.*]] = nvws.semaphore.create {{%.*}} true
    // CHECK: [[CM1_0:%.*]] = arith.constant -1 : i32
    // CHECK: [[S0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[OBS0_INIT:%.*]] = arith.constant {{.*}} false
    // CHECK: [[F0:%.*]] = nvws.semaphore.create {{%.*}} false
    // CHECK: [[PF0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[E1:%.*]] = nvws.semaphore.create {{%.*}} true
    // CHECK: [[CM1_1:%.*]] = arith.constant -1 : i32
    // CHECK: [[S1_INIT:%.*]] = arith.constant {{.*}} 0 : i32
    // CHECK: [[OBS1_INIT:%.*]] = arith.constant {{.*}} false
    // CHECK: [[F1:%.*]] = nvws.semaphore.create {{%.*}} false
    // CHECK: [[PF1_INIT:%.*]] = arith.constant 0 : i32
    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %aref1 = nvws.aref.create %f, %g : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    // CHECK: [[LOOP:%.*]]:8 = scf.for {{.*}} iter_args([[S0:%.*]] = [[S0_INIT]], [[OBS0:%.*]] = [[OBS0_INIT]], [[S1:%.*]] = [[S1_INIT]], [[OBS1:%.*]] = [[OBS1_INIT]], [[PE0:%.*]] = [[CM1_0]], [[PF0:%.*]] = [[PF0_INIT]], [[PE1:%.*]] = [[CM1_1]], [[PF1:%.*]] = [[PF1_INIT]]) -> (i32, i1, i32, i1, i32, i32, i32, i32)
    scf.for %i = %lb to %ub step %c1_i32 : i32{
      // CHECK: [[PTOK0:%.*]] = nvws.semaphore.acquire [[E0]][[[S0]], [[PE0]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[C1_P0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[PSH0:%.*]] = arith.shli [[C1_P0]], [[S0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PE0_NEW:%.*]] = arith.xori [[PE0]], [[PSH0]] {ttg.partition = array<i32: 0>} : i32
      %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "op1"(%1#0) {ttg.partition = array<i32: 0>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
      "op2"(%1#1)  {ttg.partition = array<i32: 0>} : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: nvws.semaphore.release [[F0]][[[S0]]], [[PTOK0]] [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0]], [[PF0]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[C1_G0:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[GSH0:%.*]] = arith.shli [[C1_G0]], [[S0]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PF0_NEW:%.*]] = arith.xori [[PF0]], [[GSH0]] {ttg.partition = array<i32: 1>} : i32
      %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "op3"(%2#0, %2#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: nvws.semaphore.release [[E0]][[[S0]]], [[GTOK0]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: [[IFRES:%.*]]:2 = scf.if {{%.*}} -> (i32, i1)
      scf.if %cond {
      } else {
        %4:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "op4"(%4#0, %4#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.put.exit %aref1[%c0_i32], %4#2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        %5:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "op5"(%5#0, %5#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.get.exit %aref1[%c0_i32], %5#2 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1]], [[OBS1]] : i32, i1
      // CHECK: } else {
      // CHECK: nvws.semaphore.acquire [[E1]][[[S1]],
      // CHECK-SAME: {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[F1]][[[S1]]], {{%.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.acquire [[F1]][[[S1]],
      // CHECK-SAME: {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][[[S1]]], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[S1]], [[OBS1]] : i32, i1
      // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 0, 1>]}
      } {ttg.partition = array<i32: 0, 1>}

      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[S0]], [[OBS0]], [[IFRES]]#0, [[IFRES]]#1, [[PE0_NEW]], [[PF0_NEW]], [[PE1]], [[PF1]] : i32, i1, i32, i1, i32, i32, i32, i32
    // CHECK-NEXT: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>, array<i32: 0, 1, 2>], ttg.warp_specialize.tag = 0 : i32}
    } {ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<1x128x128xf32
    // CHECK: [[S0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<1x128x128xf32
    // CHECK: [[S1_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 0 : i32
    // Pre-loop acquire on EMPTY (with mask extraction)
    // CHECK: [[MASK0:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RAW0:%.*]] = arith.andi [[S0_INIT]], [[MASK0]] : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW0]], [[CM1]]]
    // CHECK: [[C1_PRE:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_PRE:%.*]] = arith.shli [[C1_PRE]], [[RAW0]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.xori [[CM1]], [[SH_PRE]] : i32
    // Pre-loop stage advancement (result feeds into for iter_args)
    // CHECK: [[ADV_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, [[S0_INIT]]
    %0 = nvws.aref.create %result : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0[%c0_i32], %token : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: scf.for {{.*}} : i32 {
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %4 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[RAW0]]], [[TOK0]] {ttg.partition = array<i32: 1>}
      %10 = nvws.aref.buffer %0[%c0_i32], %token {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // Post-loop release/acquire using loop results
    // CHECK: nvws.semaphore.release [[FULL]][[[RAW0]]], [[TOK0]] [#nvws.async_op<tc5mma>]
    nvws.aref.put.exit %0[%c0_i32], %token [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[MASK1:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RAW1:%.*]] = arith.andi [[S1_INIT]], [[MASK1]] : i32
    // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[RAW1]], [[PF_INIT]]]
    // CHECK: [[C1_POST:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_POST:%.*]] = arith.shli [[C1_POST]], [[RAW1]] : i32
    // CHECK: [[PF_POST:%.*]] = arith.xori [[PF_INIT]], [[SH_POST]] : i32
    %buffers_0, %token_1 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %3 = nvws.aref.buffer %0[%c0_i32], %token_1 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %3[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    // CHECK: [[ORI_MASK:%.*]] = arith.constant -2147483648 : i32
    // CHECK: arith.ori [[S1_INIT]], [[ORI_MASK]] : i32
    // CHECK: nvws.semaphore.release [[EMPTY]][[[RAW1]]], [[TOK1]] [#nvws.async_op<none>]
    nvws.aref.get.exit %0[%c0_i32], %token_1 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[S0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[S1_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 0 : i32
    // Pre-loop acquire and phase update (with mask extraction)
    // CHECK: [[MASK0:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RAW0:%.*]] = arith.andi [[S0_INIT]], [[MASK0]] : i32
    // CHECK: [[PRETOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW0]], [[CM1]]]
    // CHECK: [[C1_PRE:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_PRE:%.*]] = arith.shli [[C1_PRE]], [[RAW0]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.xori [[CM1]], [[SH_PRE]] : i32
    // Pre-loop stage advancement
    // CHECK: [[ADV_S:%.*]] = arith.select {{%.*}}, {{%.*}}, [[S0_INIT]]
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0[%c0_i32], %token : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = [[PRETOK]], [[FS:%.*]] = [[ADV_S]], [[FPE:%.*]] = [[PE_PRE]], [[FS1:%.*]] = [[S1_INIT]], [[FPF:%.*]] = [[PF_INIT]]) -> (!ttg.async.token, i32, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[FS]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      %9 = nvws.aref.buffer %0[%c0_i32], %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // In-loop stage advancement
      // CHECK: [[LADV:%.*]] = arith.select {{%.*}}, {{%.*}}, [[FS]]
      // CHECK: nvws.semaphore.release [[FULL]][[[FS]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.put.exit %0[%c0_i32], %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // CHECK: [[MASK_F:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2147483647 : i32
      // CHECK: [[RAW_FS1:%.*]] = arith.andi [[FS1]], [[MASK_F]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[RAW_FS1]], [[FPF]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[C1_G:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[GSH:%.*]] = arith.shli [[C1_G]], [[RAW_FS1]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[FPF_NEW:%.*]] = arith.xori [[FPF]], [[GSH]] {ttg.partition = array<i32: 0>} : i32
      %buffers_1, %token_2 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
      %11 = nvws.aref.buffer %0[%c0_i32], %token_2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: [[ORI_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} -2147483648 : i32
      // CHECK: [[FS1_OBS:%.*]] = arith.ori [[FS1]], [[ORI_MASK]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: nvws.semaphore.release [[EMPTY]][[[RAW_FS1]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_3, %token_4 = ttng.tmem_load %11[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      nvws.aref.get.exit %0[%c0_i32], %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: [[MASK_E:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 2147483647 : i32
      // CHECK: [[RAW_LADV:%.*]] = arith.andi [[LADV]], [[MASK_E]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW_LADV]], [[FPE]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[C1_P:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
      // CHECK: [[PSH:%.*]] = arith.shli [[C1_P]], [[RAW_LADV]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[FPE_NEW:%.*]] = arith.xori [[FPE]], [[PSH]] {ttg.partition = array<i32: 1>} : i32
      %buffers_5, %token_6 = nvws.aref.put.enter %0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
      %12 = nvws.aref.buffer %0[%c0_i32], %token_6 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 1>}
      // Second stage advancement
      // CHECK: [[LADV2:%.*]] = arith.select {{%.*}}, {{%.*}}, [[LADV]]
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[PTOK]], [[LADV2]], [[FPE_NEW]], [[FS1_OBS]], [[FPF_NEW]]
      %13 = ttng.tmem_store %cst, %12[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %token_6 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.aref.put.exit %0[%c0_i32], %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @assign_stage_buffer
  tt.func @assign_stage_buffer(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[S0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[S1_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 0 : i32
    // Pre-loop acquire on EMPTY (with mask extraction)
    // CHECK: [[MASK0:%.*]] = arith.constant 2147483647 : i32
    // CHECK: [[RAW0:%.*]] = arith.andi [[S0_INIT]], [[MASK0]] : i32
    // CHECK: [[PRETOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW0]], [[CM1]]]
    // CHECK: [[C1_PRE:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_PRE:%.*]] = arith.shli [[C1_PRE]], [[RAW0]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.xori [[CM1]], [[SH_PRE]] : i32
    // Pre-loop stage advancement
    // CHECK: [[ADV_S:%.*]] = arith.select {{%.*}}, {{%.*}}, [[S0_INIT]]
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0, %token : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = [[PRETOK]], [[FS:%.*]] = [[ADV_S]], [[FPE:%.*]] = [[PE_PRE]], [[FS1:%.*]] = [[S1_INIT]], [[FPF:%.*]] = [[PF_INIT]]) -> (!ttg.async.token, i32, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[FS]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      %9 = nvws.aref.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // In-loop stage advancement
      // CHECK: [[LADV:%.*]] = arith.select {{%.*}}, {{%.*}}, [[FS]]
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[IF:%.*]]:5 = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[FULL]][[[FS]]], [[FTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.aref.put.exit %0, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[MASK_F:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 2147483647 : i32
        // CHECK: [[RAW_FS1:%.*]] = arith.andi [[FS1]], [[MASK_F]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[RAW_FS1]], [[FPF]]] {ttg.partition = array<i32: 0>}
        // CHECK: [[C1_G:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
        // CHECK: [[GSH:%.*]] = arith.shli [[C1_G]], [[RAW_FS1]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[FPF_T:%.*]] = arith.xori [[FPF]], [[GSH]] {ttg.partition = array<i32: 0>} : i32
        %buffers_1, %token_2 = nvws.aref.get.enter %0 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        %15 = nvws.aref.buffer %0, %token_2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
        // CHECK: [[ORI_MASK:%.*]] = arith.constant {ttg.partition = array<i32: 0>} -2147483648 : i32
        // CHECK: [[FS1_OBS:%.*]] = arith.ori [[FS1]], [[ORI_MASK]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: nvws.semaphore.release [[EMPTY]][[[RAW_FS1]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.aref.get.exit %0, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[MASK_E:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 2147483647 : i32
        // CHECK: [[RAW_LADV:%.*]] = arith.andi [[LADV]], [[MASK_E]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[RAW_LADV]], [[FPE]]] {ttg.partition = array<i32: 1>}
        // CHECK: [[C1_P:%.*]] = arith.constant {ttg.partition = array<i32: 1>} 1 : i32
        // CHECK: [[PSH:%.*]] = arith.shli [[C1_P]], [[RAW_LADV]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[FPE_T:%.*]] = arith.xori [[FPE]], [[PSH]] {ttg.partition = array<i32: 1>} : i32
        %buffers_5, %token_6 = nvws.aref.put.enter %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], [[LADV]], [[FPE_T]], [[FS1_OBS]], [[FPF_T]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[FTOK]], [[LADV]], [[FPE]], [[FS1]], [[FPF]]
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>]}
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[IF]]#1], [[IF]]#0 {ttg.partition = array<i32: 1>}
      %13 = nvws.aref.buffer %0, %12 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // Second stage advancement after if
      // CHECK: [[LADV2:%.*]] = arith.select {{%.*}}, {{%.*}}, [[IF]]#1
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF]]#0, [[LADV2]], [[IF]]#2, [[IF]]#3, [[IF]]#4
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.aref.put.exit %0, %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}


// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[E0:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<2x256x64xf32
    // CHECK-DAG: [[F0:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<2x256x64xf32
    // CHECK-DAG: [[E1:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<1x256x64xf32
    // CHECK-DAG: [[F1:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<1x256x64xf32
    // CHECK-DAG: [[E2:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<1x256x64xf16
    // CHECK-DAG: [[F2:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<1x256x64xf16
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0 : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.async.token
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = nvws.aref.create %result_2 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers_3, %token_4 = nvws.aref.put.enter %1 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
    %2 = nvws.aref.buffer %1, %token_4 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %3 = ttng.tmem_store %cst_0, %2[], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_5 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>
    %4 = nvws.aref.create %result_5 : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[LOOP:%.*]]:16 = scf.for {{.*}} iter_args({{.*}}) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    %5:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %token_4) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      // CHECK: tt.descriptor_load {{.*}} {ttg.partition = array<i32: 2>}
      %7 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %8 = ttg.local_alloc %7 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %10 = nvws.aref.buffer %0, %arg8 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
      %11 = ttng.tc_gen5_mma %arg0, %9, %10[], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // Stage0 advancement: wrap then conditional advance
      // CHECK: [[S0_WRAP:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}}
      // CHECK: [[S0_ADV:%.*]] = arith.select {{%.*}}, [[S0_WRAP]], {{%.*}}
      // CHECK: nvws.semaphore.release [[F0]][{{%.*}}], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.put.exit %0, %arg8 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[S0_RAW_G:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[GTOK0:%.*]] = nvws.semaphore.acquire [[F0]][[[S0_RAW_G]], {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[C1_G0:%.*]] = arith.constant {ttg.partition = array<i32: 0>} 1 : i32
      // CHECK: [[GSH0:%.*]] = arith.shli [[C1_G0]], [[S0_RAW_G]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[PF0_NEW:%.*]] = arith.xori {{%.*}}, [[GSH0]] {ttg.partition = array<i32: 0>} : i32
      %buffers_10, %token_11 = nvws.aref.get.enter %0 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.async.token
      %12 = nvws.aref.buffer %0, %token_11 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>
      // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[E0]][[[S0_RAW_G]]], [[GTOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_12, %token_13 = ttng.tmem_load %12[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64> -> tensor<256x64xf32, #blocked>
      nvws.aref.get.exit %0, %token_11 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %13 = "compute_row_max"(%result_12, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = "sub_row_max"(%result_12, %13, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %15 = math.exp2 %14 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %16 = arith.subf %arg7, %13 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %17 = arith.subf %arg7, %13 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %18 = math.exp2 %16 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %19 = math.exp2 %17 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %20 = "tt.reduce"(%15) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %36 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>}: f32
        tt.reduce.return %36 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = arith.mulf %arg6, %19 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %22 = arith.addf %21, %20 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.expand_dims %18 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %24 = tt.broadcast %23 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %25 = nvws.aref.buffer %1, %arg9 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %result_14, %token_15 = ttng.tmem_load %25[] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
      %26 = arith.mulf %result_14, %24 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %27 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %28 = ttg.local_alloc %27 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %29 = arith.truncf %15 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %buffers_16, %token_17 = nvws.aref.put.enter %4 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
      %30 = nvws.aref.buffer %4, %token_17 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.acquire [[E2]][{{%.*}}, {{%.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 0>}
      // Stage2 advancement: wrap then conditional advance
      // CHECK: [[S2_WRAP:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}}
      // CHECK: [[S2_ADV:%.*]] = arith.select {{%.*}}, [[S2_WRAP]], {{%.*}}
      // CHECK: nvws.semaphore.release [[F2]][{{%.*}}], {{%.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %31 = ttng.tmem_store %29, %30[%token_17], %true {ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.aref.put.exit %4, %token_17 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %32 = ttng.tmem_store %26, %25[], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // Stage1 advancement: wrap then conditional advance
      // CHECK: [[S1_WRAP:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}}
      // CHECK: [[S1_ADV:%.*]] = arith.select {{%.*}}, [[S1_WRAP]], {{%.*}}
      // CHECK: nvws.semaphore.release [[F1]][{{%.*}}], {{%.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      nvws.aref.put.exit %1, %arg9 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %buffers_18, %token_19 = nvws.aref.get.enter %1 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
      %33 = nvws.aref.buffer %1, %token_19 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      %buffers_20, %token_21 = nvws.aref.get.enter %4 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
      %34 = nvws.aref.buffer %4, %token_21 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
      %35 = ttng.tc_gen5_mma %34, %28, %33[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
      // CHECK: nvws.semaphore.release [[E2]][{{%.*}}], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[E1]][{{%.*}}], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %4, %token_21 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf16, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.aref.get.exit %1, %token_19 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: nvws.semaphore.acquire [[E0]][{{%.*}}, {{%.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.acquire [[E1]][{{%.*}}, {{%.*}}] {ttg.partition = array<i32: 3>}
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>}
      %buffers_22, %token_23 = nvws.aref.put.enter %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 2x256x64>, !ttg.async.token
      %buffers_24, %token_25 = nvws.aref.put.enter %1 {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
      scf.yield %22, %13, %token_23, %token_25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>
    // CHECK: nvws.semaphore.release [[F1]][[[LOOP]]#8], [[LOOP]]#3 [#nvws.async_op<tc5mma>]
    // CHECK: nvws.semaphore.release [[F0]][[[LOOP]]#4], [[LOOP]]#2 [#nvws.async_op<none>]
    nvws.aref.put.exit %1, %5#3 [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    nvws.aref.put.exit %0, %5#2 [#nvws.async_op<none>] : <[!ttg.memdesc<2x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: [[S1_RAW_END:%.*]] = arith.andi [[LOOP]]#10, {{%.*}} : i32
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[F1]][[[S1_RAW_END]], [[LOOP]]#11]
    // CHECK: [[C1_END:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_END:%.*]] = arith.shli [[C1_END]], [[S1_RAW_END]] : i32
    // CHECK: [[PE_END:%.*]] = arith.xori [[LOOP]]#11, [[SH_END]] : i32
    // CHECK: ttng.tmem_load
    // CHECK: nvws.semaphore.release [[E1]][[[S1_RAW_END]]], [[TOK_END]] [#nvws.async_op<none>]
    %buffers_6, %token_7 = nvws.aref.get.enter %1 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>, !ttg.async.token
    %6 = nvws.aref.buffer %1, %token_7 : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64>
    %result_8, %token_9 = ttng.tmem_load %6[] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked>
    nvws.aref.get.exit %1, %token_7 [#nvws.async_op<none>] : <[!ttg.memdesc<1x256x64xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%5#0, %result_8, %5#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
    tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[S0_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[CM1:%.*]] = arith.constant -1 : i32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false : <[!ttg.memdesc<2x128x128xf32
    // CHECK: [[PF_INIT:%.*]] = arith.constant 0 : i32
    // CHECK: [[PF_INIT2:%.*]] = arith.constant 0 : i32
    // Pre-loop acquire on EMPTY
    // CHECK: [[S0_RAW_PRE:%.*]] = arith.andi [[S0_INIT]], {{%.*}} : i32
    // CHECK: [[PRETOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0_RAW_PRE]], [[CM1]]]
    // CHECK: [[C1_PRE:%.*]] = arith.constant 1 : i32
    // CHECK: [[SH_PRE:%.*]] = arith.shli [[C1_PRE]], [[S0_RAW_PRE]] : i32
    // CHECK: [[PE_PRE:%.*]] = arith.xori [[CM1]], [[SH_PRE]] : i32
    // Pre-loop stage advancement
    // CHECK: [[ADV_S:%.*]] = arith.select {{%.*}}, {{%.*}}, [[S0_INIT]]
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0, %token : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[FOR:%.*]]:5 = scf.for {{.*}} iter_args([[FTOK:%.*]] = [[PRETOK]], [[FS:%.*]] = [[ADV_S]], [[FPE:%.*]] = [[PE_PRE]], [[FPF:%.*]] = [[PF_INIT]], [[FPF2:%.*]] = [[PF_INIT2]]) -> (!ttg.async.token, i32, i32, i32, i32)
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[FS]]], [[FTOK]] {ttg.partition = array<i32: 1>}
      %9 = nvws.aref.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[IF:%.*]]:5 = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: nvws.semaphore.release [[FULL]][{{%.*}}], {{%.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.aref.put.exit %0, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        // CHECK: [[GS_RAW:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]][[[GS_RAW]], [[GP_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 0>}
        // CHECK: [[GSH:%.*]] = arith.shli {{%.*}}, [[GS_RAW]] {ttg.partition = array<i32: 0>} : i32
        // CHECK: [[GP_OUT:%.*]] = arith.xori [[GP_IN]], [[GSH]] {ttg.partition = array<i32: 0>} : i32
        %buffers_1, %token_2 = nvws.aref.get.enter %0 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        %15 = nvws.aref.buffer %0, %token_2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        // CHECK: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[EMPTY]][[[GS_RAW]]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
        %result_3, %token_4 = ttng.tmem_load %15[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.aref.get.exit %0, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        // CHECK: [[PS_RAW:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[PS_RAW]], [[PP_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 1>}
        // CHECK: [[PSH:%.*]] = arith.shli {{%.*}}, [[PS_RAW]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[PP_OUT:%.*]] = arith.xori [[PP_IN]], [[PSH]] {ttg.partition = array<i32: 1>} : i32
        %buffers_5, %token_6 = nvws.aref.put.enter %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[PTOK]], {{%.*}}, [[PP_OUT]], {{%.*}}, [[GP_OUT]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>]}
      // CHECK: nvws.semaphore.buffer [[EMPTY]][[[IF]]#1], [[IF]]#0 {ttg.partition = array<i32: 1>}
      %13 = nvws.aref.buffer %0, %12 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[IF]]#0, {{%.*}}, [[IF]]#2, [[IF]]#3, [[IF]]#4
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // CHECK: } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 0, 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR]]#1], [[FOR]]#0 [#nvws.async_op<none>]
    nvws.aref.put.exit %0, %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmul_tma_persistent_ws_kernel
  tt.func public @matmul_tma_persistent_ws_kernel(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg6, %arg8], [%0, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %2 = arith.extsi %arg4 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg7, %arg8], [%2, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %4 = arith.extsi %arg5 : i32 to i64
    %5 = tt.make_tensor_descriptor %arg2, [%arg6, %arg7], [%4, %c1_i64] : <f8E4M3FN>, <tensor<128x128xf8E4M3FN, #shared>>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg6, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg7, %c127_i32 : i32
    %10 = arith.divsi %9, %c128_i32 : i32
    %11 = arith.addi %arg8, %c127_i32 : i32
    %12 = arith.divsi %11, %c128_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>
    %16 = nvws.aref.create %15 : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %17 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>
    %18 = nvws.aref.create %17 : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK-DAG: [[AB_EMPTY:%.*]] = nvws.semaphore.create {{.*}} true : <[!ttg.memdesc<3x128x128xf8E4M3FN
    // CHECK-DAG: [[AB_FULL:%.*]] = nvws.semaphore.create {{.*}} false : <[!ttg.memdesc<3x128x128xf8E4M3FN
    // CHECK-DAG: [[ACC_EMPTY:%.*]] = nvws.semaphore.create {{.*}} true : <[!ttg.memdesc<1x128x128xf32
    // CHECK-DAG: [[ACC_FULL:%.*]] = nvws.semaphore.create {{.*}} false : <[!ttg.memdesc<1x128x128xf32
    %19 = nvws.aref.create %result : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[OUTER:%.*]]:8 = scf.for {{.*}} iter_args({{.*}}) -> (i32, i32, i32, i32, i32, i32, i32, i32)
    scf.for %arg9 = %6 to %13 step %c148_i32  : i32 {
      %20 = arith.divsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %21 = arith.muli %20, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %22 = arith.subi %8, %21 {ttg.partition = array<i32: 0, 2>} : i32
      %23 = arith.minsi %22, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %24 = arith.remsi %arg9, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %25 = arith.addi %21, %24 {ttg.partition = array<i32: 0, 2>} : i32
      %26 = arith.remsi %arg9, %14 {ttg.partition = array<i32: 0, 2>} : i32
      %27 = arith.divsi %26, %23 {ttg.partition = array<i32: 0, 2>} : i32
      %28 = arith.muli %25, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %29 = arith.muli %27, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      // CHECK: [[ASTAGE0:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK0:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE0]], [[APHASE0_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[ASHIFT0:%.*]] = arith.shli {{%.*}}, [[ASTAGE0]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE0_OUT:%.*]] = arith.xori [[APHASE0_IN]], [[ASHIFT0]] {ttg.partition = array<i32: 0>} : i32
      %buffers, %token = nvws.aref.put.enter %19 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
      %30 = nvws.aref.buffer %19, %token {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %31 = ttng.tmem_store %cst, %30[], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      // CHECK: nvws.semaphore.release [[ACC_FULL]][{{%.*}}], [[ATOK0]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      nvws.aref.put.exit %19, %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[ASTAGE1:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK1:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE1]], [[APHASE1_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[ASHIFT1:%.*]] = arith.shli {{%.*}}, [[ASTAGE1]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE1_OUT:%.*]] = arith.xori [[APHASE1_IN]], [[ASHIFT1]] {ttg.partition = array<i32: 1>} : i32
      %buffers_0, %token_1 = nvws.aref.get.enter %19 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
      // CHECK: [[INNER:%.*]]:5 = scf.for {{.*}} iter_args({{.*}}) -> (i1, i32, i32, i32, i32)
      %32 = scf.for %arg10 = %c0_i32 to %12 step %c1_i32 iter_args(%arg11 = %false) -> (i1)  : i32 {
        %36 = arith.muli %arg10, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[ABSTAGE_P:%.*]] = arith.andi {{%.*}}, {{%.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[ABTOK_P:%.*]] = nvws.semaphore.acquire [[AB_EMPTY]][[[ABSTAGE_P]], [[ABPHASE_P_IN:%[-a-zA-Z0-9_#]+]]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        // CHECK: [[ABSHIFT_P:%.*]] = arith.shli {{%.*}}, [[ABSTAGE_P]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        // CHECK: [[ABPHASE_P_OUT:%.*]] = arith.xori [[ABPHASE_P_IN]], [[ABSHIFT_P]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %buffers_8, %token_9 = nvws.aref.put.enter %16 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.async.token
        nvws.descriptor_load %1[%28, %36] 16384 %buffers_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[AB_FULL]][{{%.*}}], [[ABTOK_P]] [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
        nvws.aref.put.exit %16, %token_9 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token

        // CHECK: [[ABSTAGE_C:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[ABTOK_C:%.*]] = nvws.semaphore.acquire [[AB_FULL]][[[ABSTAGE_C]], [[ABPHASE_C_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 1>}
        // CHECK: [[ABSHIFT_C:%.*]] = arith.shli {{%.*}}, [[ABSTAGE_C]] {ttg.partition = array<i32: 1>} : i32
        // CHECK: [[ABPHASE_C_OUT:%.*]] = arith.xori [[ABPHASE_C_IN]], [[ABSHIFT_C]] {ttg.partition = array<i32: 1>} : i32
        %buffers_10, %token_11 = nvws.aref.get.enter %16 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, 1x128x128>, !ttg.async.token
        %buffers_12, %token_13 = nvws.aref.put.enter %18 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>, !ttg.async.token
        nvws.descriptor_load %3[%29, %36] 16384 %buffers_12 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, i32, i32, !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable, 1x128x128>
        nvws.aref.put.exit %18, %token_13 [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token
        %buffers_14, %token_15 = nvws.aref.get.enter %18 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, 1x128x128>, !ttg.async.token
        %37 = ttg.memdesc_trans %buffers_14 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, 1x128x128> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, 1x128x128>
        %38 = nvws.aref.buffer %19, %token_1 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %39 = ttng.tc_gen5_mma %buffers_10, %37, %38[], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, 1x128x128>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem, 1x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        // CHECK: nvws.semaphore.release [[AB_EMPTY]][{{%.*}}], [[ABTOK_C]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        nvws.aref.get.exit %18, %token_15 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.aref.get.exit %16, %token_11 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E4M3FN, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} {{%.*}}, {{%.*}}, [[ABPHASE_P_OUT]], {{%.*}}, [[ABPHASE_C_OUT]]
        scf.yield %true : i1
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][{{%.*}}], [[ATOK1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %19, %token_1 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %buffers_2, %token_3 = nvws.aref.put.enter %19 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
      %33 = nvws.aref.buffer %19, %token_3 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %result_4, %token_5 = ttng.tmem_load %33[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      nvws.aref.put.exit %19, %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[ASTAGE2:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[ATOK2:%.*]] = nvws.semaphore.acquire [[ACC_EMPTY]][[[ASTAGE2]], [[APHASE2_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[ASHIFT2:%.*]] = arith.shli {{%.*}}, [[ASTAGE2]] {ttg.partition = array<i32: 0>} : i32
      // CHECK: [[APHASE2_OUT:%.*]] = arith.xori [[APHASE2_IN]], [[ASHIFT2]] {ttg.partition = array<i32: 0>} : i32
      %buffers_6, %token_7 = nvws.aref.get.enter %19 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[ACC_FULL]][{{%.*}}], [[ATOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: [[ASTAGE3:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[ATOK3:%.*]] = nvws.semaphore.acquire [[ACC_FULL]][[[ASTAGE3]], [[APHASE3_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[ASHIFT3:%.*]] = arith.shli {{%.*}}, [[ASTAGE3]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: [[APHASE3_OUT:%.*]] = arith.xori [[APHASE3_IN]], [[ASHIFT3]] {ttg.partition = array<i32: 1>} : i32
      // CHECK: nvws.semaphore.release [[ACC_EMPTY]][{{%.*}}], [[ATOK3]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %19, %token_7 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %34 = tt.fp_to_fp %result_4 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %35 = ttg.convert_layout %34 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %5[%28, %29], %35 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, {{%.*}}, [[APHASE2_OUT]], {{%.*}}, [[APHASE3_OUT]]
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>, array<i32: 1>, array<i32: 1>, array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @for_loop_control_operand_ppg
  tt.func @for_loop_control_operand_ppg(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %arefBuf = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create {{.*}} true : <[!ttg.memdesc<1x128x128xf32
    // CHECK: [[FULL:%.*]] = nvws.semaphore.create {{.*}} false : <[!ttg.memdesc<1x128x128xf32
    // CHECK: [[S0:%.*]] = arith.andi {{%.*}}, {{%.*}} : i32
    // CHECK: [[TOK0:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S0]], [[P0_IN:%[-a-zA-Z0-9_#]+]]]
    // CHECK: [[SH0:%.*]] = arith.shli {{%.*}}, [[S0]] : i32
    // CHECK: [[P0_OUT:%.*]] = arith.xori [[P0_IN]], [[SH0]] : i32
    %aref = nvws.aref.create %arefBuf : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %_0, %tok = nvws.aref.put.enter %aref : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    // CHECK: [[FOR0:%.*]]:5 = scf.for {{.*}} iter_args({{.*}}) -> (!ttg.async.token, i32, i32, i32, i32)
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
      %ptrub = tt.addptr %ptr0, %iv0 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %ub1 = tt.load %ptrub {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %lb1 = "lb1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %step1 = "step1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      // CHECK: [[FOR1:%.*]]:3 = scf.for {{.*}} iter_args({{.*}}) -> (!ttg.async.token, i32, i32)
      %tok5 = scf.for %iv = %lb1 to %ub1 step %step1 iter_args(%tok2 = %tok1) -> (!ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %buf = nvws.aref.buffer %aref, %tok2 {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 2>}
        ttng.tc_gen5_mma %sA, %sB, %buf, %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %tok2 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 0, 2>, array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[FULL]][[[FOR1]]#1], [[FOR1]]#0 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>}
      nvws.aref.put.exit %aref, %tok5 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[S1:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[TOK1:%.*]] = nvws.semaphore.acquire [[FULL]][[[S1]], [[P1_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[SH1:%.*]] = arith.shli {{%.*}}, [[S1]] {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[P1_OUT:%.*]] = arith.xori [[P1_IN]], [[SH1]] {ttg.partition = array<i32: 0, 1>} : i32
      %_1, %token_2 = nvws.aref.get.enter %aref {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
      // CHECK: nvws.semaphore.release [[EMPTY]][[[S1]]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %aref, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      // CHECK: [[S2:%.*]] = arith.andi {{%.*}}, {{%.*}} {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[TOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]][[[S2]], [[P2_IN:%[-a-zA-Z0-9_#]+]]] {ttg.partition = array<i32: 2>}
      // CHECK: [[SH2:%.*]] = arith.shli {{%.*}}, [[S2]] {ttg.partition = array<i32: 2>} : i32
      // CHECK: [[P2_OUT:%.*]] = arith.xori [[P2_IN]], [[SH2]] {ttg.partition = array<i32: 2>} : i32
      %buf1, %tok6 = nvws.aref.put.enter %aref {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[TOK2]], [[FOR1]]#1, [[P2_OUT]], {{%.*}}, [[P1_OUT]]
      scf.yield {ttg.partition = array<i32: 1, 2>} %tok6 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 0, 2>, array<i32: 2>, array<i32: 0, 1>, array<i32: 0, 1>]}
    // CHECK: nvws.semaphore.release [[FULL]][[[FOR0]]#1], [[FOR0]]#0 [#nvws.async_op<tc5mma>]
    // CHECK: [[S_END:%.*]] = arith.andi {{%.*}}, {{%.*}} : i32
    // CHECK: [[TOK_END:%.*]] = nvws.semaphore.acquire [[FULL]][[[S_END]], [[P_END_IN:%[-a-zA-Z0-9_#]+]]]
    // CHECK: [[SH_END:%.*]] = arith.shli {{%.*}}, [[S_END]] : i32
    // CHECK: [[P_END_OUT:%.*]] = arith.xori [[P_END_IN]], [[SH_END]] : i32
    // CHECK: nvws.semaphore.release [[EMPTY]][[[S_END]]], [[TOK_END]] [#nvws.async_op<none>]
    nvws.aref.put.exit %aref, %tok0 [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %_2, %token_2 = nvws.aref.get.enter %aref : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token
    nvws.aref.get.exit %aref, %token_2 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}
