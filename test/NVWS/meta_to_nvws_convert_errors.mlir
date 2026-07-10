// RUN: not triton-opt %s -allow-unregistered-dialect --nvws-meta-to-nvws-convert 2>&1 | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK: error: MetaToNVWSConvert requires a non-empty, non-negative async_task_id assignment
  tt.func @missing_meta_assignment(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      "test.missing"() : () -> ()
      scf.yield {async_task_id = array<i32: 0>}
    } {async_task_id = array<i32: 0>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32],
       ttg.partition.types = ["default"],
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
