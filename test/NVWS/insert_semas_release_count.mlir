// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=EMIT
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore -cse | FileCheck %s --check-prefix=LOWER
//
// EMIT/LOWER pin the first-class count contract end to end on THIS
// pass-produced shape (fable/integrate-pending-count-plan.md): after
// handoff-granularity redundant-edge elimination the ready release
// carries arrive_count = 1 in emitted IR, and the lowering transcribes
// both counts into the mbarrier init/arrive.
// LOWER-COUNT-5: ttng.init_barrier {{.*}}, 1
// LOWER: ttng.arrive_barrier {{.*}}, 1 {ttg.partition = array<i32: 3>}

// Release arrive-multiplicity (spec section 5.2, uniform pending count):
// a semaphore's pending count is a per-semaphore constant — every acquire
// site sees the same count and every acquire cycle must receive exactly
// that many arrives. Shape: producer {3} stores outside the inner loop;
// inside, {2} (carried) and {1} read, {1} CORRECTS the buffer in place
// after an explicit {2}->{1} WAR handoff, and {0} consumes the corrected
// value AFTER the store. The last version's holders at the inner EXIT are
// {1} (the writer) and {0} (its reader), but redundant-edge elimination
// at handoff granularity drops {1}'s regain arrival: {1}'s store is
// already ordered before {0}'s surviving arrival through the S5 handoff
// ({1} r S5 after the store -> {0} a S5 -> {0}'s read -> {0} r FULL), so
// the regain is fan-in-1 and the For-row unification merges the outer
// single-source ready edge onto the SAME semaphore at the uniform pending
// count 1. The lone outer release arrives once: r S(1). The emitted and
// lowered IR checks below pin that multiplicity.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // EMIT-LABEL: @release_multiplicity_unified_fanin_regain
  tt.func @release_multiplicity_unified_fanin_regain(%lb: i32, %ub: i32, %step: i32) {
    // EMIT: [[ALLOC:%.*]] = ttg.local_alloc {buffer.id = 700 : i32}
    %alloc = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    // EMIT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true {pending_count = 1 : i32}
    // EMIT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT: [[S3:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT: [[S4:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT: [[S5:%.*]] = nvws.semaphore.create [[ALLOC]] false {pending_count = 1 : i32}
    // EMIT-NOT: nvws.semaphore.acquire
    // EMIT: scf.for
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 3>} : () -> !ty
      // EMIT: [[ACQ3:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 3>} : {{.*}} -> !ttg.async.token
      // EMIT: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ACQ3]] {ttg.partition = array<i32: 3>}
      // EMIT: ttg.local_store {{%.*}}, [[BUF3]] {ttg.partition = array<i32: 3>}
      // EMIT: nvws.semaphore.release [[FULL]], [[ACQ3]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 3>}
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      // EMIT: [[ACQ2:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>} : {{.*}} -> !ttg.async.token
      // EMIT: [[INNER:%.*]] = scf.for {{.*}} iter_args([[JCARRY:%.*]] = [[ACQ2]]) -> (!ttg.async.token)
      scf.for %j = %lb to %ub step %step : i32 {
        // EMIT: nvws.semaphore.release [[S3]], [[JCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
        // EMIT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[JCARRY]] {ttg.partition = array<i32: 2>}
        // EMIT: ttg.local_load [[BUF2]] {ttg.partition = array<i32: 2>}
        // EMIT: nvws.semaphore.release [[S4]], [[JCARRY]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        // EMIT: [[ACQ1R:%.*]] = nvws.semaphore.acquire [[S3]] {ttg.partition = array<i32: 1>} : {{.*}} -> !ttg.async.token
        // EMIT: [[BUF1R:%.*]] = nvws.semaphore.buffer [[S3]], [[ACQ1R]] {ttg.partition = array<i32: 1>}
        // EMIT: ttg.local_load [[BUF1R]] {ttg.partition = array<i32: 1>}
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        %c = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        // EMIT: [[ACQ1W:%.*]] = nvws.semaphore.acquire [[S4]] {ttg.partition = array<i32: 1>} : {{.*}} -> !ttg.async.token
        // EMIT: [[BUF1W:%.*]] = nvws.semaphore.buffer [[S4]], [[ACQ1W]] {ttg.partition = array<i32: 1>}
        // EMIT: ttg.local_store {{%.*}}, [[BUF1W]] {ttg.partition = array<i32: 1>}
        // EMIT: nvws.semaphore.release [[S5]], [[ACQ1W]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 1>}
        // {1}'s own arrival into [[FULL]] is elided — covered through {0}:
        // EMIT-NOT: nvws.semaphore.release
        ttg.local_store %c, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        // EMIT: [[ACQ0:%.*]] = nvws.semaphore.acquire [[S5]] {ttg.partition = array<i32: 0>} : {{.*}} -> !ttg.async.token
        // EMIT: [[BUF0:%.*]] = nvws.semaphore.buffer [[S5]], [[ACQ0]] {ttg.partition = array<i32: 0>}
        // EMIT: ttg.local_load [[BUF0]] {ttg.partition = array<i32: 0>}
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        // EMIT: nvws.semaphore.release [[FULL]], [[ACQ0]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 0>}
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
        // EMIT: [[ACQ2B:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>} : {{.*}} -> !ttg.async.token
        // EMIT: scf.yield {ttg.partition = array<i32: 0, 1, 2>} [[ACQ2B]]
      } {ttg.partition = array<i32: 0, 1, 2>}
      // EMIT: nvws.semaphore.release [[EMPTY]], [[INNER]] [#nvws.async_op<none>] {arrive_count = 1 : i32, ttg.partition = array<i32: 2>}
      // EMIT-NOT: nvws.semaphore.acquire
      // EMIT: tt.warp_specialize
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// The outer ready release by {3} carries arrive multiplicity 1: the inner
// EXIT regain is fan-in-1 after handoff-granularity redundant-edge
// elimination. {1} (the corrector) no longer arrives into the carried
// ready semaphore — its store is certified transitively by {0}, the
// coverer: {1}'s r S5 sits after the in-place store, {0} acquires S5
// before its read, and {0}'s surviving r FULL (anchored after that read,
// its old program point) closes into carried {2}. Both acquire sites show
// the uniform pending count (1). The stable ENTER source lets {1}'s read
// overlap {2}'s read, so the in-loop store takes an explicit {2}->{1} WAR
// edge; {1}'s own read is ordered by program order. With arrive
// multiplicity 1, the v5 uniform hold rule's eligibility check now
// accepts point-of-use for the outer ready hold: there is no preheader
// acquire or carried outer token, and {3} acquires EMPTY at its store
// site inside the loop.
