// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_if_conditional_only
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @local_if_conditional_only(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 200 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}} {ttg.partition = array<i32: 0>}
      ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: scf.if
      scf.if %cond {
        %c1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
        %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
        // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[LOAD:%.*]] = ttg.local_load [[RBUF]] {ttg.partition = array<i32: 1>}
        %l = ttg.local_load %rv {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
        // CHECK: "use_then"([[LOAD]]) {ttg.partition = array<i32: 1>}
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      } else {
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c2 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv2 = ttg.memdesc_index %alloc[%c2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[NEXTTOK:%.*]] = scf.if {{.*}} -> (!ttg.async.token)
      // CHECK-NEXT: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK: scf.yield {{.*}}[[PTOK]]
      // CHECK: nvws.semaphore.buffer [[EMPTY]], [[NEXTTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store {{.*}} {ttg.partition = array<i32: 0>}
      ttg.local_store %v2, %pv2 {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
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
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @local_if_conditional_only_else(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 203 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // CHECK: scf.if
      scf.if %cond {
      } else {
        %c1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
        %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
        // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: [[LOAD:%.*]] = ttg.local_load [[RBUF]] {ttg.partition = array<i32: 1>}
        %l = ttg.local_load %rv {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
        // CHECK: "use_else"([[LOAD]]) {ttg.partition = array<i32: 1>}
        "use_else"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c2 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv2 = ttg.memdesc_index %alloc[%c2] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[NEXTTOK:%.*]] = scf.if {{.*}} -> (!ttg.async.token)
      // CHECK: nvws.semaphore.buffer [[EMPTY]], [[NEXTTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store
      ttg.local_store %v2, %pv2 {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
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
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @local_if_consumption_continues_after_join(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 201 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %cond = "cond"() {ttg.partition = array<i32: 0, 1>} : () -> i1
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]]
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[RBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: scf.if
      scf.if %cond {
        %c1 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
        %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
        // CHECK: ttg.local_load [[RBUF]] {ttg.partition = array<i32: 1>}
        %l = ttg.local_load %rv {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
        "use_then"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } else {
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = []}
      %c2 = arith.constant {ttg.partition = array<i32: 1>} 0 : i32
      %rv2 = ttg.memdesc_index %alloc[%c2] {ttg.partition = array<i32: 1>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
      // CHECK: [[LOAD2:%.*]] = ttg.local_load [[RBUF]] {ttg.partition = array<i32: 1>}
      %l2 = ttg.local_load %rv2 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      // CHECK: "use_after"([[LOAD2]]) {ttg.partition = array<i32: 1>}
      "use_after"(%l2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      %v2 = "producer2"() {ttg.partition = array<i32: 0>} : () -> !ty
      %c3 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %pv2 = ttg.memdesc_index %alloc[%c3] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.buffer [[EMPTY]], [[PTOK2]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store
      ttg.local_store %v2, %pv2 {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
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
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @local_root_external_distinct_from_ws_tag_zero(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 202 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %v = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    %c0 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
    %pv = ttg.memdesc_index %alloc[%c0] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[RTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    ttg.local_store %v, %pv {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: nvws.semaphore.release [[FULL]], [[RTOK]]
    // CHECK: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      %c1 = arith.constant {ttg.partition = array<i32: 0>} 0 : i32
      %rv = ttg.memdesc_index %alloc[%c1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<1x1xi32, #shared, #smem, mutable> -> !ttg.memdesc<1xi32, #shared, #smem>
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      %l = ttg.local_load %rv {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem> -> !ty
      "use"(%l) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.stages = [0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
