// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=pou' -cse | FileCheck %s --check-prefixes=CHECK,POU
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=first-touch' -cse | FileCheck %s --check-prefixes=CHECK,FT

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // POU-LABEL: @branch_pass_with_outstanding_completion
  // FT-LABEL: @branch_pass_with_outstanding_completion
  tt.func @branch_pass_with_outstanding_completion(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[PASS_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10100 : i32}
    // CHECK: [[LOOP_SEMA:%[0-9]+]] = nvws.semaphore.create [[PASS_BASE]] true {pending_count = 2 : i32}
    %buf = ttg.local_alloc {buffer.id = 10100 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
    // POU: [[W1_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[LOOP_SEMA]] {ttg.partition = array<i32: 1>}
    // POU-NEXT: [[W1_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[LOOP_SEMA]], [[W1_TOKEN]] {ttg.partition = array<i32: 1>}
    // FT: [[W1_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[LOOP_SEMA]], [[W1_TOKEN:%[a-z0-9]+]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[W1_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[TO_0:%[0-9]+]], [[W1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[R0_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_0]]
    // CHECK-NEXT: [[R0_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
    // CHECK-NEXT: ttg.local_load [[R0_BUFFER]]
    %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
    scf.if %cond {
      // CHECK: scf.if
      // CHECK: nvws.semaphore.release [[TO_1_THEN:%[0-9]+]], [[R0_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[R1_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_1_THEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[R1_THEN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_1_THEN]], [[R1_THEN_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[R1_THEN_BUFFER]] {ttg.partition = array<i32: 1>}
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: nvws.semaphore.release [[LOOP_SEMA]], [[R1_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[R0_THEN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
      // CHECK-NEXT: ttg.local_load [[R0_THEN_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[LOOP_SEMA]], [[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %r0b = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then0"(%r0b) {ttg.partition = array<i32: 0>} : (!one) -> ()
    } else {
      // CHECK: } else {
      // CHECK: nvws.semaphore.release [[TO_1_ELSE:%[0-9]+]], [[R0_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[R1_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_1_ELSE]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[R1_ELSE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_1_ELSE]], [[R1_ELSE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_load [[R1_ELSE_BUFFER]] {ttg.partition = array<i32: 1>}
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: nvws.semaphore.release [[LOOP_SEMA]], [[R1_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[R0_ELSE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_0]], [[R0_TOKEN]]
      // CHECK-NEXT: ttg.local_load [[R0_ELSE_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[LOOP_SEMA]], [[R0_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
      %r0b = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "else0"(%r0b) {ttg.partition = array<i32: 0>} : (!one) -> ()
    } {ttg.partition = array<i32: 0, 1>}
    // POU-NOT: nvws.semaphore.acquire [[LOOP_SEMA]]
    // FT: [[TAIL_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[LOOP_SEMA]] {ttg.partition = array<i32: 1>}
    // FT: scf.yield {{.*}}[[TAIL_TOKEN]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // POU-LABEL: @closed_exact_token_enters_inner_loop
  // FT-LABEL: @closed_exact_token_enters_inner_loop
  tt.func @closed_exact_token_enters_inner_loop(
      %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[REUSE_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10101 : i32}
    // CHECK: [[REUSE_ENTRY:%[0-9]+]] = nvws.semaphore.create [[REUSE_BASE]] true
    %buf = ttg.local_alloc {buffer.id = 10101 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() {ttg.partition = array<i32: 1>} : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
      // CHECK: [[W_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[REUSE_ENTRY]], [[W_TOKEN:%[a-z0-9]+]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[W_BUFFER]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[TO_ZERO:%[0-9]+]], [[W_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_ZERO]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_ZERO]], [[ZERO_TOKEN]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_load [[ZERO_BUFFER]] {ttg.partition = array<i32: 0>}
      %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
      scf.for %j = %lb to %ub step %step : i32 {
        // CHECK: scf.for
        // CHECK: [[INNER_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[REUSE_ENTRY]], [[W_TOKEN]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: ttg.local_load [[INNER_BUFFER]] {ttg.partition = array<i32: 1>}
        %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
        "use1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      } {ttg.partition = array<i32: 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // POU-LABEL: @conditional_completion_is_later_edge_source
  // FT-LABEL: @conditional_completion_is_later_edge_source
  tt.func @conditional_completion_is_later_edge_source(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[SOURCE_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10102 : i32}
    // CHECK: [[SOURCE_ENTRY:%[0-9]+]] = nvws.semaphore.create [[SOURCE_BASE]] true
    %buf = ttg.local_alloc {buffer.id = 10102 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: [[SOURCE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[SOURCE_ENTRY]], [[SOURCE_TOKEN:%[a-z0-9]+]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[SOURCE_BUFFER]] {ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.if %cond {
      // CHECK: scf.if
      // CHECK: nvws.semaphore.release [[TO_ZERO:%[0-9]+]], [[SOURCE_TOKEN]] [#nvws.async_op<none>]
      // CHECK-NEXT: [[ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_ZERO]]
      // CHECK-NEXT: [[ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_ZERO]], [[ZERO_TOKEN]]
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[ZERO_BUFFER]]
      // CHECK-NEXT: nvws.semaphore.release [[JOIN:%[0-9]+]], [[ZERO_TOKEN]] [#nvws.async_op<none>]
      ttg.local_store %x, %buf {ttg.partition = array<i32: 0>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: nvws.semaphore.release [[JOIN]], [[SOURCE_TOKEN]] [#nvws.async_op<none>]
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: [[R1_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[JOIN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[R1_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[JOIN]], [[R1_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_load [[R1_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[TO_TWO:%[0-9]+]], [[R1_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
    // CHECK: [[R2_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[TO_TWO]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[R2_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[TO_TWO]], [[R2_TOKEN]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: ttg.local_load [[R2_BUFFER]] {ttg.partition = array<i32: 2>}
    %r2 = ttg.local_load %buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use2"(%r2) {ttg.partition = array<i32: 2>} : (!one) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // POU-LABEL: @conditional_alternatives_normalize_counts
  // FT-LABEL: @conditional_alternatives_normalize_counts
  tt.func @conditional_alternatives_normalize_counts(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[COUNT_BASE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10103 : i32}
    // CHECK: [[COUNT_ENTRY:%[0-9]+]] = nvws.semaphore.create [[COUNT_BASE]] true
    %buf = ttg.local_alloc {buffer.id = 10103 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %x = "value"() : () -> !one
    scf.for %i = %lb to %ub step %step : i32 {
    // CHECK: [[COUNT_WRITE_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[COUNT_ENTRY]], [[COUNT_WRITE_TOKEN:%[a-z0-9]+]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[COUNT_WRITE_BUFFER]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.semaphore.release [[COUNT_TO_ZERO:%[0-9]+]], [[COUNT_WRITE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
    ttg.local_store %x, %buf {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK-NEXT: [[COUNT_ZERO_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[COUNT_TO_ZERO]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[COUNT_ZERO_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[COUNT_TO_ZERO]], [[COUNT_ZERO_TOKEN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_load [[COUNT_ZERO_BUFFER]] {ttg.partition = array<i32: 0>}
    %r0 = ttg.local_load %buf {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "use0"(%r0) {ttg.partition = array<i32: 0>} : (!one) -> ()
    scf.if %cond {
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      "then1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
      // CHECK: nvws.semaphore.release [[JOIN:%[0-9]+]], {{%[0-9]+}} [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      %r2 = ttg.local_load %buf {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      // CHECK: ttg.local_load {{%[0-9]+}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[JOIN]], {{%[0-9]+}} [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      "then2"(%r2) {ttg.partition = array<i32: 2>} : (!one) -> ()
    } else {
      %r1 = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
      // CHECK: } else {
      // CHECK: ttg.local_load {{%[0-9]+}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[JOIN]], {{%[0-9]+}} [#nvws.async_op<none>] {arrive_count = 2 : i32, ttg.partition = array<i32: 1>}
      "else1"(%r1) {ttg.partition = array<i32: 1>} : (!one) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: [[JOIN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[JOIN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[JOIN_BUFFER:%[0-9]+]] = nvws.semaphore.buffer [[JOIN]], [[JOIN_TOKEN]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: ttg.local_load [[JOIN_BUFFER]] {ttg.partition = array<i32: 1>}
    %r1c = ttg.local_load %buf {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !one
    "continue1"(%r1c) {ttg.partition = array<i32: 1>} : (!one) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!one = tensor<1xi32, #blocked>
!two = tensor<2xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // POU-LABEL: @mixed_owner_conditional_joins_at_whole_write
  // FT-LABEL: @mixed_owner_conditional_joins_at_whole_write
  tt.func @mixed_owner_conditional_joins_at_whole_write(
      %cond: i1, %lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[MIXED_WHOLE:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[MIXED_LEFT:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32}
    // CHECK-NEXT: [[MIXED_RIGHT:%[0-9]+]] = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 1 : i32}
    // CHECK-NEXT: [[MIXED_ENTRY_0:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] true
    // CHECK-NEXT: [[MIXED_ENTRY_1:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] true
    // CHECK-NEXT: [[MIXED_JOIN:%[0-9]+]] = nvws.semaphore.create [[MIXED_WHOLE]], [[MIXED_LEFT]], [[MIXED_RIGHT]] false {pending_count = 2 : i32}
    %whole = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    %left = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 0 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %right = ttg.local_alloc {buffer.id = 10104 : i32, buffer.offset = 1 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %one = "one"() : () -> !one
    %two = "two"() : () -> !two
    scf.for %i = %lb to %ub step %step : i32 {
    scf.if %cond {
      // CHECK: scf.if
      // CHECK: [[LEFT_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LEFT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_0]], [[LEFT_THEN_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[LEFT_THEN_BUFFER]]#1 {ttg.partition = array<i32: 1>}
      ttg.local_store %one, %left {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[RIGHT_THEN_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[RIGHT_THEN_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_1]], [[RIGHT_THEN_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[RIGHT_THEN_BUFFER]]#2 {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[MIXED_JOIN]], [[RIGHT_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[MIXED_JOIN]], [[LEFT_THEN_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %one, %right {ttg.partition = array<i32: 2>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } else {
      // CHECK: } else {
      // CHECK: [[LEFT_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_0]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LEFT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_0]], [[LEFT_ELSE_TOKEN]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[LEFT_ELSE_BUFFER]]#1 {ttg.partition = array<i32: 1>}
      ttg.local_store %one, %left {ttg.partition = array<i32: 1>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK-NEXT: [[RIGHT_ELSE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_ENTRY_1]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[RIGHT_ELSE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_ENTRY_1]], [[RIGHT_ELSE_TOKEN]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[RIGHT_ELSE_BUFFER]]#2 {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[MIXED_JOIN]], [[RIGHT_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[MIXED_JOIN]], [[LEFT_ELSE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
      ttg.local_store %one, %right {ttg.partition = array<i32: 2>} : !one -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    } {ttg.partition = array<i32: 0, 1, 2>}
    // CHECK: [[WHOLE_TOKEN:%[0-9]+]] = nvws.semaphore.acquire [[MIXED_JOIN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[WHOLE_BUFFER:%[0-9]+]]:3 = nvws.semaphore.buffer [[MIXED_JOIN]], [[WHOLE_TOKEN]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store {{%[0-9]+}}, [[WHOLE_BUFFER]]#0 {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.semaphore.release [[MIXED_ENTRY_0]], [[WHOLE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.semaphore.release [[MIXED_ENTRY_1]], [[WHOLE_TOKEN]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
    ttg.local_store %two, %whole {ttg.partition = array<i32: 0>} : !two -> !ttg.memdesc<2xi32, #shared, #smem, mutable>
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>,
       ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
