// RUN: triton-opt --split-input-file %s | FileCheck %s


// CHECK-LABEL: @warp_group_nothing
tt.func @warp_group_nothing() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  tt.return
}

// CHECK-LABEL: @warp_1_partition
tt.func @warp_1_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT:  num_warps(4) {
  partition0  num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_2_partition
tt.func @warp_2_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT: partition0  num_warps(8) {
  partition0  num_warps(8) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT:   nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @token_producer_consumer
tt.func @token_producer_consumer() {

  // CHECK: nvws.create_token
  // CHECK: nvws.producer_acquire
  // CHECK: nvws.producer_commit
  // CHECK: nvws.consumer_wait
  // CHECK: nvws.consumer_release

  %0 = nvws.create_token {loadType = 1 : i32, numBuffers = 3 : i32} : tensor<3x!nvws.token>

  %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
  %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false

  nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  tt.return
}
