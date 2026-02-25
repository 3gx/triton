// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck %s
// RUN: triton-opt %S/assign_stage_phase.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore --nvws-assign-semaphore-stage-phase | FileCheck %s --check-prefix=ASP-AREF --implicit-check-not=nvws.aref.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @assign_stage_basic
  // CHECK: %[[SEM:.*]] = nvws.semaphore.create
  // CHECK: %[[TOK:.*]] = nvws.semaphore.acquire %[[SEM]][%[[STAGE:.*]], %[[PHASE:.*]]]
  // CHECK: arith.shli
  // CHECK: %[[VIEW:.*]] = nvws.semaphore.buffer %[[SEM]][%[[STAGE]]], %[[TOK]]
  // CHECK: %[[ADV:.*]] = arith.andi
  // CHECK: nvws.semaphore.release %[[SEM]][%[[STAGE]]], %[[TOK]] [#nvws.async_op<none>]
  tt.func @assign_stage_basic(%lb: i32, %ub: i32, %step: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    scf.for %i = %lb to %ub step %step : i32 {
      %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
      %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      %val = ttg.local_load %view : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      ttg.local_store %val, %view : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
      nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token
    }

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// ASP-AREF-LABEL: @two_consumers
// ASP-AREF-DAG: nvws.semaphore.acquire
// ASP-AREF-DAG: arith.shli
// ASP-AREF-DAG: arith.andi
// ASP-AREF-DAG: ttg.partition.outputs = [
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @aref_lowering
// ASP-AREF-DAG: scf.if
// ASP-AREF-DAG: [#nvws.async_op<tma_load>, #nvws.async_op<none>]
// ASP-AREF-DAG: [#nvws.async_op<tc5mma>]
// ASP-AREF-DAG: ttg.partition.outputs = [
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @warp_specialize_tma_matmul
// ASP-AREF-DAG: ttng.tc_gen5_mma
// ASP-AREF-DAG: tt.warp_specialize
// ASP-AREF-DAG: [#nvws.async_op<tc5mma>]
// ASP-AREF-DAG: ttng.tmem_load
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @matmul_tma_acc_with_unconditional_user
// ASP-AREF-DAG: ttng.tmem_store
// ASP-AREF-DAG: ttng.tmem_load
// ASP-AREF-DAG: [#nvws.async_op<none>]
// ASP-AREF-DAG: [#nvws.async_op<tc5mma>]
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @assign_stage_buffer
// ASP-AREF-DAG: scf.if
// ASP-AREF-DAG: nvws.semaphore.buffer
// ASP-AREF-DAG: [#nvws.async_op<tc5mma>]
// ASP-AREF-DAG: ttng.tmem_load
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @matmul_tma_acc_with_conditional_user
// ASP-AREF-DAG: scf.if
// ASP-AREF-DAG: ttng.tmem_load
// ASP-AREF-DAG: [#nvws.async_op<none>]
// ASP-AREF-DAG: [#nvws.async_op<tc5mma>]
// ASP-AREF-DAG: tt.descriptor_store
// ASP-AREF: tt.return

// ASP-AREF-LABEL: @for_loop_control_operand_ppg
// ASP-AREF-DAG: tt.load
// ASP-AREF-DAG: scf.for
// ASP-AREF-DAG: ttg.partition.outputs = [
// ASP-AREF-DAG: ttng.tc_gen5_mma
// ASP-AREF: tt.return

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @shared_stage_two_semaphores
  // CHECK: %[[SEM0:.*]] = nvws.semaphore.create
  // CHECK: %[[SEM1:.*]] = nvws.semaphore.create
  // CHECK: %[[TOK0:.*]] = nvws.semaphore.acquire %[[SEM0]][%[[S:.*]], %{{.*}}]
  // CHECK: %[[TOK1:.*]] = nvws.semaphore.acquire %[[SEM1]][%[[S]], %{{.*}}]
  tt.func @shared_stage_two_semaphores() {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem0 = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    %sem1 = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>

    %tok0 = nvws.semaphore.acquire %sem0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %tok1 = nvws.semaphore.acquire %sem1 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token

    %view0 = nvws.semaphore.buffer %sem0, %tok0 : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    %v = arith.constant dense<0> : !elt
    ttg.local_store %v, %view0 : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

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
  // CHECK: %[[IF:.*]]:2 = scf.if %{{.*}} -> (i32, i1)
  // CHECK: %[[ADV:.*]] = arith.andi %[[IF]]#1,
  tt.func @if_observation(%cond: i1) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>

    scf.if %cond {
      %x = ttg.local_load %view : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> !elt
      "use"(%x) : (!elt) -> ()
    }

    %v = arith.constant dense<0> : !elt
    ttg.local_store %v, %view : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>], 2>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
