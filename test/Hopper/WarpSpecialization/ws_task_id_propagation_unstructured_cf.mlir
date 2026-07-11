// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

// Regression: task-id backward propagation must handle UNSTRUCTURED control
// flow (`cf.cond_br` / `cf.br`) instead of aborting. Before the fix,
// visitBranchOperand asserted the branch was an scf op and crashed
// NVGPUWarpSpecialization on kernels whose control flow lowers to cf ops. The
// common source is an early-exit `return` / bounds guard at the top of a kernel.

// Case 1: cf.cond_br whose successors take forwarded operands used by different
// task-anchored ops -> the condition gets the union of both task ids.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @unstructured_cond_br_task_id
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}} : i32) {async_task_id = array<i32: 0, 1>}
  tt.func public @unstructured_cond_br_task_id(%arg0: i32, %cond: i1) {
    cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg0 : i32)
  ^bb1(%x: i32):
    %a = arith.addi %x, %x {async_task_id = array<i32: 0>} : i32
    cf.br ^bb3
  ^bb2(%y: i32):
    %b = arith.muli %y, %y {async_task_id = array<i32: 1>} : i32
    cf.br ^bb3
  ^bb3:
    tt.return
  }
}

// -----

// Case 2: an early-exit guard whose successors take no forwarded operands. The
// empty union is benign and the branch condition remains task-id-less.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @early_return_guard_task_id
  // CHECK: cf.cond_br
  // CHECK: arith.addi %{{.*}} {async_task_id = array<i32: 0>}
  tt.func public @early_return_guard_task_id(%n: i32, %cond: i1) {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    tt.return
  ^bb2:
    %a = arith.addi %n, %n {async_task_id = array<i32: 0>} : i32
    tt.return
  }
}
