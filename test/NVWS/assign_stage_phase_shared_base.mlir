// RUN: triton-opt %s --allow-unregistered-dialect --nvws-assign-stage-phase | FileCheck %s --check-prefix=ASP
// RUN: triton-opt %s --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --check-prefix=LOWER

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // ASP-LABEL: @base_phase_shared_across_partitions
  // ASP: [[EMPTY:%.*]] = nvws.semaphore.create
  // ASP: {{%.*}}:2 = scf.for {{.*}} iter_args([[STAGE_IN:%[^ ]+]] = {{%[^,]+}}, [[PHASE_IN:%[^ ]+]] = {{%[^)]+}}) -> (i32, i32)
  // ASP: [[P2_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P2_WORD:%.*]] = arith.xori [[PHASE_IN]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P2_SHIFT:%.*]] = arith.shrui [[P2_WORD]], [[P2_STAGE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P2_PHASE:%.*]] = arith.andi [[P2_SHIFT]], {{%.*}} {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: nvws.semaphore.acquire [[EMPTY]][[[P2_STAGE]], [[P2_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // ASP-NEXT: nvws.semaphore.acquire [[EMPTY]][[[P2_STAGE]], [[P2_PHASE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // ASP: [[P1_STAGE:%.*]] = arith.select {{%.*}}, {{%.*}}, {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P1_WORD:%.*]] = arith.xori [[P2_WORD]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P1_SHIFT:%.*]] = arith.shrui [[P1_WORD]], [[P1_STAGE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: [[P1_PHASE:%.*]] = arith.andi [[P1_SHIFT]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : i32
  // ASP: nvws.semaphore.acquire [[EMPTY]][[[P1_STAGE]], [[P1_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // ASP-NEXT: nvws.semaphore.acquire [[EMPTY]][[[P1_STAGE]], [[P1_PHASE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // ASP: scf.yield {{.*}}, [[P1_WORD]] : i32, i32
  // ASP: ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1, 2>]

  // LOWER-LABEL: @base_phase_shared_across_partitions
  // LOWER: [[P2_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS:%.*]][[[P2_STAGE:%.*]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER-NEXT: ttng.wait_barrier [[P2_BARRIER]], [[P2_PHASE:%.*]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER: [[P2_SHADOW_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[P2_STAGE]]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER-NEXT: ttng.wait_barrier [[P2_SHADOW_BARRIER]], [[P2_PHASE]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER: [[P1_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[P1_STAGE:%.*]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER-NEXT: ttng.wait_barrier [[P1_BARRIER]], [[P1_PHASE:%.*]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
  // LOWER: [[P1_SHADOW_BARRIER:%.*]] = ttg.memdesc_index [[EMPTY_BARRIERS]][[[P1_STAGE]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  // LOWER-NEXT: ttng.wait_barrier [[P1_SHADOW_BARRIER]], [[P1_PHASE]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
  tt.func @base_phase_shared_across_partitions(%lb: i32, %ub: i32,
                                                %step: i32) {
    %base = ttg.local_alloc {buffer.circular, buffer.copy = 3 : i32, buffer.id = 306 : i32, buffer.start = 0 : i32} : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %empty = nvws.semaphore.create %base true {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    %full = nvws.semaphore.create %base false {pending_count = 1 : i32} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %lb to %ub step %step : i32 {
      %z0 = arith.constant {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} 0 : i32
      %t0 = nvws.semaphore.acquire %empty[%z0] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b0 = nvws.semaphore.buffer %empty[%z0], %t0 {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%b0) {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      nvws.semaphore.release %full[%z0], %t0 [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      %z1 = arith.constant {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} 0 : i32
      %t1 = nvws.semaphore.acquire %empty[%z1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b1 = nvws.semaphore.buffer %empty[%z1], %t1 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      "test_store"(%b1) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
      nvws.semaphore.release %full[%z1], %t1 [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.scheduled_max_stage = 0 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [], ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }
}
