// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_conditional_only
  tt.func @local_if_conditional_only(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V1:%.*]] = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V2:%.*]] = nvws.semaphore.create [[V1]] true : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V3:%.*]] = nvws.semaphore.create [[V1]] false : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V4:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V6:%.*]] = scf.for [[V7:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V5:%.*]] = [[V4]]) -> (!ttg.async.token)  : i32 {
    // CHECK: [[V8:%.*]] = nvws.semaphore.buffer [[V2]], [[V5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V8]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      scf.if %cond {
        // CHECK: nvws.semaphore.release [[V3]], [[V5]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[V9:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: [[V10:%.*]] = nvws.semaphore.acquire [[V3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[V11:%.*]] = nvws.semaphore.buffer [[V3]], [[V10]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttg.local_load [[V11]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> tensor<1xi32, #blocked>
        // CHECK: nvws.semaphore.release [[V2]], [[V10]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{.*}} : !ttg.async.token
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } else {
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{.*}} : !ttg.async.token
      // CHECK: [[V12:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
      // CHECK: [[V13:%.*]] = nvws.semaphore.acquire [[V2]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[V13]] : !ttg.async.token
      // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[V5]] : !ttg.async.token
      // CHECK: [[V14:%.*]] = nvws.semaphore.buffer [[V2]], [[V12]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V14]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V12]] : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_conditional_only_else
  tt.func @local_if_conditional_only_else(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V15:%.*]] = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V16:%.*]] = nvws.semaphore.create [[V15]] true : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V17:%.*]] = nvws.semaphore.create [[V15]] false : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V18:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: [[V20:%.*]] = scf.for [[V21:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[V19:%.*]] = [[V18]]) -> (!ttg.async.token)  : i32 {
    // CHECK: [[V22:%.*]] = nvws.semaphore.buffer [[V16]], [[V19]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V22]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      scf.if %cond {
      } else {
        // CHECK: nvws.semaphore.release [[V17]], [[V19]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[V23:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{.*}} : !ttg.async.token
        // CHECK: [[V24:%.*]] = nvws.semaphore.acquire [[V17]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[V25:%.*]] = nvws.semaphore.buffer [[V17]], [[V24]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem>
        // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttg.local_load [[V25]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> tensor<1xi32, #blocked>
        // CHECK: nvws.semaphore.release [[V16]], [[V24]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} %{{.*}} : !ttg.async.token
        // CHECK: [[V26:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[V19]] : !ttg.async.token
        // CHECK: [[V27:%.*]] = nvws.semaphore.acquire [[V16]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0>} [[V27]] : !ttg.async.token
        // CHECK: [[V28:%.*]] = nvws.semaphore.buffer [[V16]], [[V26]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V28]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1>} [[V26]] : !ttg.async.token
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use_else"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_consumption_continues_after_join
  tt.func @local_if_consumption_continues_after_join(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[V29:%.*]] = ttg.local_alloc {buffer.id = 201 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %alloc = ttg.local_alloc {buffer.id = 201 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[V30:%.*]] = nvws.semaphore.create [[V29]] true : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[V31:%.*]] = nvws.semaphore.create [[V29]] false : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[V32:%.*]] = nvws.semaphore.acquire [[V30]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V33:%.*]] = nvws.semaphore.buffer [[V30]], [[V32]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V33]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // CHECK: nvws.semaphore.release [[V31]], [[V32]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V34:%.*]] = nvws.semaphore.acquire [[V31]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V35:%.*]] = nvws.semaphore.buffer [[V31]], [[V34]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem>
      // CHECK: %{{[-A-Za-z0-9_.$#]+}} = ttg.local_load [[V35]] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> tensor<1xi32, #blocked>
      scf.if %cond {
        %l = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } else {
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      // CHECK: nvws.semaphore.release [[V30]], [[V34]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: [[V36:%.*]] = nvws.semaphore.acquire [[V30]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[V37:%.*]] = nvws.semaphore.buffer [[V30]], [[V36]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{[-A-Za-z0-9_.$#]+}}, [[V37]] {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use_after"(%l2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %v2, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_root_external_distinct_from_ws_tag_zero
  tt.func @local_root_external_distinct_from_ws_tag_zero(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 202 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    ttg.local_store %v, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %l = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "use"(%l) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
