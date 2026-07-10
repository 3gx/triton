// RUN: triton-opt %s -allow-unregistered-dialect --nvws-meta-to-nvws-convert | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @partitions_and_result
  // CHECK: %[[LOOP:.*]] = scf.for
  // CHECK: "test.producer"() {ttg.partition = array<i32: 1>}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} {{.*}}
  // CHECK: } {tt.warp_specialize
  // CHECK-SAME: ttg.partition = array<i32: 0, 1, 2>
  // CHECK-SAME: ttg.partition.outputs = [array<i32: 1>]
  // CHECK-SAME: ttg.warp_specialize.tag = 7 : i32
  // CHECK: "test.outside"(%[[LOOP]]) {async_task_id = array<i32: 0>} :
  tt.func @partitions_and_result(%lb: i32, %ub: i32, %step: i32,
                                 %init: tensor<32xf32, #blocked>) {
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%arg = %init) -> (tensor<32xf32, #blocked>) : i32 {
      %next = "test.producer"() {async_task_id = array<i32: 1>} : () -> tensor<32xf32, #blocked>
      scf.yield {async_task_id = array<i32: 0, 1>} %next : tensor<32xf32, #blocked>
    } {async_task_id = array<i32: 2, 0, 1, 2>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32],
       ttg.partition.types = ["default", "compute", "load"],
       ttg.warp_specialize.tag = 7 : i32}
    "test.outside"(%result) {async_task_id = array<i32: 0>, ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 7 : i32} : (tensor<32xf32, #blocked>) -> ()
    tt.return
  }

  // A yielded producer outside the WS loop remains in Meta representation
  // until conversion. Its source attribute is consumed into partition.outputs
  // without adding ttg.partition to the external constant.
  // CHECK-LABEL: tt.func @external_yielded_meta_producer
  // CHECK: %[[TRUE:.*]] = arith.constant true
  // CHECK: %[[BOOL_LOOP:.*]] = scf.for
  // CHECK: scf.yield {ttg.partition = array<i32: 0>} %[[TRUE]]
  // CHECK: ttg.partition.outputs = [array<i32: 0, 1>]
  // CHECK: "test.bool.outside"(%[[BOOL_LOOP]]) {async_task_id = array<i32: 0>}
  tt.func @external_yielded_meta_producer(
      %lb: i32, %ub: i32, %step: i32, %init: i1) {
    %true = arith.constant {async_task_id = array<i32: 0, 1>} true
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%flag = %init) -> (i1) : i32 {
      "test.bool.consumer"(%flag) {async_task_id = array<i32: 1>} : (i1) -> ()
      scf.yield {async_task_id = array<i32: 0>} %true : i1
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["default", "compute"],
       ttg.warp_specialize.tag = 9 : i32}
    "test.bool.outside"(%result) {async_task_id = array<i32: 0>} : (i1) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @token_result
  // CHECK: "test.token.consumer"({{.*}}) {ttg.partition = array<i32: 1>}
  // CHECK: %[[NEXT:.*]] = "test.token.producer"() {ttg.partition = array<i32: 0>}
  // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %[[NEXT]]
  // CHECK: ttg.partition.outputs = [array<i32: 0, 1>]
  // CHECK: "test.token.outside"({{.*}}) {async_task_id = array<i32: 0>}
  tt.func @token_result(%lb: i32, %ub: i32, %step: i32,
                        %init: !ttg.async.token) {
    %result = scf.for %i = %lb to %ub step %step
        iter_args(%arg = %init) -> (!ttg.async.token) : i32 {
      "test.token.consumer"(%arg) {async_task_id = array<i32: 1>} : (!ttg.async.token) -> ()
      %next = "test.token.producer"() {async_task_id = array<i32: 0>} : () -> !ttg.async.token
      scf.yield {async_task_id = array<i32: 0, 1>} %next : !ttg.async.token
    } {async_task_id = array<i32: 0, 1>, tt.warp_specialize,
       ttg.partition.stages = [0 : i32, 0 : i32],
       ttg.partition.types = ["producer", "consumer"],
       ttg.warp_specialize.tag = 8 : i32}
    "test.token.outside"(%result) {async_task_id = array<i32: 0>} : (!ttg.async.token) -> ()
    tt.return
  }
}
