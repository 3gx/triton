// RUN: triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=auto' -cse | FileCheck %s --check-prefix=AUTO
// RUN: triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=first-touch' -cse | FileCheck %s --check-prefix=FIRST
// RUN: not triton-opt %s -allow-unregistered-dialect '--nvws-insert-semas=placement-mode=invalid' 2>&1 | FileCheck %s --check-prefix=BAD

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // AUTO-LABEL: tt.func @placement_mode
  // AUTO: [[BASE:%.*]] = ttg.local_alloc
  // AUTO: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true
  // AUTO: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
  // AUTO: [[WRITE:%.*]] = nvws.semaphore.acquire [[EMPTY]]

  // FIRST-LABEL: tt.func @placement_mode
  // FIRST: [[BASE:%.*]] = ttg.local_alloc
  // FIRST: [[EMPTY:%.*]] = nvws.semaphore.create [[BASE]] true
  // FIRST: [[INITIAL:%.*]] = nvws.semaphore.acquire [[EMPTY]]
  // FIRST: [[LOOP:%.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args([[CARRY:%.*]] = [[INITIAL]]) -> (!ttg.async.token) : i32 {
  // FIRST: nvws.semaphore.buffer [[EMPTY]], [[CARRY]]
  // FIRST: [[NEXT:%.*]] = nvws.semaphore.acquire [[EMPTY]]
  // FIRST: scf.yield {{.*}} [[NEXT]] : !ttg.async.token
  tt.func @placement_mode(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 990 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %iv = %lb to %ub step %step : i32 {
      %value = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      ttg.local_store %value, %alloc {ttg.partition = array<i32: 0>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %loaded = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
      "consumer"(%loaded) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// BAD: error: nvws-insert-semas: invalid placement mode 'invalid' (expected auto or first-touch)
