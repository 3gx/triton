// RUN: triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=auto' -cse -o %t.auto
// RUN: triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=first-touch' -cse -o %t.first
// RUN: diff %t.auto %t.first
// RUN: FileCheck %s --check-prefix=AUTO < %t.auto
// RUN: not triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=pou' -cse 2>&1 | FileCheck %s --check-prefix=POU

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // AUTO-LABEL: tt.func @completed_pou_fallback
  // AUTO: [[ALLOC:%.*]] = ttg.local_alloc
  // AUTO: [[ENTRY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // AUTO: [[INITIAL:%.*]] = nvws.semaphore.acquire [[ENTRY]]
  // AUTO: scf.for {{.*}} iter_args([[OUTER:%.*]] = [[INITIAL]]) -> (!ttg.async.token)
  // AUTO: scf.for {{.*}} iter_args([[CARRY:%.*]] = [[OUTER]]) -> (!ttg.async.token)
  // AUTO: [[FIRST_BUF:%.*]] = nvws.semaphore.buffer [[ENTRY]], [[CARRY]]
  // AUTO: "touch0"([[FIRST_BUF]])
  // AUTO: "touch1"([[FIRST_BUF]])
  // AUTO: nvws.semaphore.release {{%.*}}, [[CARRY]]
  // AUTO: "touch2"
  // AUTO: [[NEXT:%.*]] = nvws.semaphore.acquire
  // AUTO-NEXT: scf.yield {{.*}} [[NEXT]] : !ttg.async.token
  // POU: error: nvws-insert-semas: point-of-use placement is unavailable for this loop: fixed loop.stage constraints require a carried recurrence
  tt.func @completed_pou_fallback(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.copy = 1 : i32, buffer.id = 991 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %outer = %lb to %ub step %step : i32 {
      scf.for %inner = %lb to %ub step %step : i32 {
        "touch0"(%alloc) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        "touch1"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
        "touch2"(%alloc) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      } {tt.scheduled_max_stage = 1 : i32, ttg.partition = array<i32: 0, 1>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
