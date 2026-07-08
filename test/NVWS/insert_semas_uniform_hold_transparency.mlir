// RUN: triton-opt %s -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // S1: prefix For, p1-anchored transparent region.
  // The p1 hold is carried into the inner loop (transparent): the store acquires
  // EMPTY once at the top, the inner loop carries that same token, and the trailing
  // p2 read after the inner loop gates a fresh acquire on the regained handle.
  // CHECK-LABEL: @uniform_hold_s1_prefix_for_p1
  tt.func @uniform_hold_s1_prefix_for_p1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 981 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 2>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S2: owner-change at op1 -> inner; negative expected.
  // The p1 hold is now transparent across the whole nest: a function-level
  // EMPTY acquire threads the outer and middle loops as a carrier, the store
  // writes on the carried handle and releases FULL at the write. The p2 inner
  // region lowers point-of-use (a fresh FULL acquire at the first touching
  // access each iteration), a post-loop bridge acquire gates the trailing p2
  // read, and the middle-loop tail re-acquires EMPTY to renew the carrier.
  // CHECK-LABEL: @uniform_hold_s2_owner_change_cut
  tt.func @uniform_hold_s2_owner_change_cut(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 982 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[MID:%.*]] = scf.for {{.*}} iter_args([[MCARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[MCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[MCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for {{.*}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F3]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F2]], [[T3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T4]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[MID]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S3: trailing read after regain; negative expected.
  // The function-level carrier threads the outer and middle loops for the p1
  // store (transparent hold). The p2 inner region lowers point-of-use: a fresh
  // FULL acquire at the first touching access each iteration. After the inner
  // loop, a bridge acquire regains the handle in p2 and releases EMPTY at the
  // same anchor as before; the trailing p1 read acquires the regained EMPTY
  // handle, which is yielded as the next iteration's carrier.
  // CHECK-LABEL: @uniform_hold_s3_trailing_read_after_regain
  tt.func @uniform_hold_s3_trailing_read_after_regain(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 983 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[MID:%.*]] = scf.for {{.*}} iter_args([[MCARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[MCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[MCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: scf.for {{.*}}  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F3]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[T2]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>}
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T4]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[MID]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S4: prefix For, p2-anchored owner mirror of S1.
  // CHECK-LABEL: @uniform_hold_s4_prefix_for_p2
  tt.func @uniform_hold_s4_prefix_for_p2(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 984 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: "producer1"
        // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 2>}
        %v1 = "producer1"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v1, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
        scf.for %i2 = %lb to %ub step %step : i32 {
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F3]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"(%i2) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
        // CHECK: nvws.semaphore.release [[F4]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S5: region-spanning at WS-body depth 1, no middle, no op4.
  // p1 store acquires EMPTY at top, the inner loop carries the token (point-of-use
  // store into the carried buffer at the region tail), and the regained handle is
  // drained right after the inner loop.
  // CHECK-LABEL: @uniform_hold_s5_ws_body_depth1
  tt.func @uniform_hold_s5_ws_body_depth1(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 985 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[CARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[CARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[CARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S6: same-owner trailing read after the inner region.
  // The p1 store reuses a function-level carried EMPTY handle, the inner loop
  // carries it, and the trailing same-owner p1 read reuses the regained token
  // (no extra acquire) — the carrier threads outer loop and inner loop.
  // CHECK-LABEL: @uniform_hold_s6_same_owner_trailing_read
  tt.func @uniform_hold_s6_same_owner_trailing_read(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 986 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[OCARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 2>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[INNER]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_load [[B3]] {ttg.partition = array<i32: 1>}
      %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consumer4"(%v4) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S7: cross-owner store out / load in.
  // p1 store acquires EMPTY then releases FULL; the inner p2 load acquires FULL
  // above the inner loop, uses the same token inside (no carrier needed — the
  // inner loop is plain), and the regained EMPTY handle is released after.
  // CHECK-LABEL: @uniform_hold_s7_cross_owner_store_load
  tt.func @uniform_hold_s7_cross_owner_store_load(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 987 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: scf.for {{.*}}  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S8: If-as-prefix-region inside the WS body.
  // p1 store acquires EMPTY at top; the scf.if carries the token (point-of-use
  // store at the region tail in the then-branch, pass-through in else), and the
  // regained handle is drained right after the if.
  // CHECK-LABEL: @uniform_hold_s8_if_prefix_region
  tt.func @uniform_hold_s8_if_prefix_region(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 988 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer1"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
      %v1 = "producer1"(%i0) {ttg.partition = array<i32: 1>} : (i32) -> !ty
      ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
      scf.if %cond {
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B2]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
      // CHECK: } else {
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T0]] : !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[IF]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S9: If inside the inner loop.
  // The store/load/store all live inside the conditional; a single carrier is
  // threaded from a function-level acquire through outer loop, inner loop, and
  // the scf.if (point-of-use store on the carried buffer, else passes through).
  // CHECK-LABEL: @uniform_hold_s9_if_inside_inner_loop
  tt.func @uniform_hold_s9_if_inside_inner_loop(%lb: i32, %ub: i32, %step: i32, %cond: i1) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %alloc = ttg.local_alloc {buffer.id = 989 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: [[OUTER:%.*]] = scf.for {{.*}} iter_args([[OCARRY:%.*]] = [[T0]]) -> (!ttg.async.token)  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[OCARRY]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: [[IF:%.*]] = scf.if %{{.*}} -> (!ttg.async.token) {
        scf.if %cond {
          // CHECK: "producer1"
          // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ICARRY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: ttg.local_load [[B0]] {ttg.partition = array<i32: 1>}
          // CHECK: nvws.semaphore.release [[F2]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v1 = "producer1"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
          ttg.local_store %v1, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
          "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
          // CHECK: "producer3"
          // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[T1]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: ttg.local_store %{{.*}}, [[B1]] {ttg.partition = array<i32: 2>}
          // CHECK: nvws.semaphore.release [[EMPTY]], [[T1]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
          %v3 = "producer3"() {ttg.partition = array<i32: 2>} : () -> !ty
          ttg.local_store %v3, %alloc {ttg.partition = array<i32: 2>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
          // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[T2]] : !ttg.async.token
        // CHECK: } else {
          // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[ICARRY]] : !ttg.async.token
        // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
        } {ttg.partition = array<i32: 1, 2>}
        // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[IF]] : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>}
      // CHECK: } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.yield {ttg.partition = array<i32: 1, 2>} [[INNER]] : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // S10: fan-out / multi-consumer.
  // Multiplicity keeps the read-side hold carrier-bearing inside the inner
  // loop, but the producer-side EMPTY hold is now point-of-use inside the
  // outer loop (no function-level carrier). Redundant-arrival elimination
  // collapses the F2 empty-cycle to a single arrival: p1's store->next-read
  // obligation is enforced transitively through its F5 handoff into p0's
  // surviving F2 arrival (p1 releases F5 after the store, p0 acquires F5,
  // reads, then releases F2 ahead of p2's next acquire), so F2's
  // pending/arrive counts drop from 2 to 1 and p1's own F2 arrival is gone.
  // The two reads still fan out via the F3/F4 handoffs.
  // CHECK-LABEL: @uniform_hold_s10_fanout_multi_consumer
  tt.func @uniform_hold_s10_fanout_multi_consumer(%lb: i32, %ub: i32, %step: i32) {
    // CHECK: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F2:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F4:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[F5:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %alloc = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // CHECK: scf.for {{.*}}  : i32 {
    scf.for %i0 = %lb to %ub step %step : i32 {
      // CHECK: "producer0"
      // CHECK: [[T0:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[B0:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[T0]] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: ttg.local_store %{{.*}}, [[B0]] {ttg.partition = array<i32: 3>}
      // CHECK: nvws.semaphore.release [[F2]], [[T0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %v0 = "producer0"(%i0) {ttg.partition = array<i32: 3>} : (i32) -> !ty
      ttg.local_store %v0, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // CHECK: [[T1:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      // CHECK: [[INNER:%.*]] = scf.for {{.*}} iter_args([[ICARRY:%.*]] = [[T1]]) -> (!ttg.async.token)  : i32 {
      scf.for %i1 = %lb to %ub step %step : i32 {
        // CHECK: nvws.semaphore.release [[F3]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // CHECK: [[B1:%.*]] = nvws.semaphore.buffer [[F2]], [[ICARRY]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B1]] {ttg.partition = array<i32: 2>}
        // CHECK: nvws.semaphore.release [[F4]], [[ICARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v1 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer1"(%v1) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // CHECK: [[T2:%.*]] = nvws.semaphore.acquire [[F3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B2:%.*]] = nvws.semaphore.buffer [[F3]], [[T2]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B2]] {ttg.partition = array<i32: 1>}
        %v2 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer2"(%v2) {ttg.partition = array<i32: 1>} : (!ty) -> ()
        // CHECK: "producer3"
        // CHECK: [[T3:%.*]] = nvws.semaphore.acquire [[F4]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B3:%.*]] = nvws.semaphore.buffer [[F4]], [[T3]] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_store %{{.*}}, [[B3]] {ttg.partition = array<i32: 1>}
        // CHECK: nvws.semaphore.release [[F5]], [[T3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        // The old p1 arrival on F2 here is eliminated: it is covered by
        // p1's F5 release above, p0's F5 acquire, and p0's F2 release below.
        // CHECK-NOT: nvws.semaphore.release [[F2]], [[T3]]
        %v3 = "producer3"(%i1) {ttg.partition = array<i32: 1>} : (i32) -> !ty
        ttg.local_store %v3, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: [[T4:%.*]] = nvws.semaphore.acquire [[F5]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: [[B4:%.*]] = nvws.semaphore.buffer [[F5]], [[T4]] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // CHECK: ttg.local_load [[B4]] {ttg.partition = array<i32: 0>}
        // CHECK: nvws.semaphore.release [[F2]], [[T4]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %v4 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "consumer4"(%v4) {ttg.partition = array<i32: 0>} : (!ty) -> ()
        // CHECK: [[T5:%.*]] = nvws.semaphore.acquire [[F2]] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
        // CHECK: scf.yield {ttg.partition = array<i32: 0, 1, 2, 3>} [[T5]] : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1, 2, 3>}
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
