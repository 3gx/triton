// RUN: env NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -o /dev/null 2>&1 | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas -cse | FileCheck %s --check-prefix=EMIT
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semas --nvws-lower-semaphore -cse | FileCheck %s --check-prefix=LOWER
//
// EMIT/LOWER pin the first-class count contract end to end on THIS
// pass-produced shape (fable/integrate-pending-count-plan.md): the
// scaled release carries arrive_count = 2 in emitted IR, and the
// lowering transcribes both counts into the mbarrier init/arrive.
// EMIT: nvws.semaphore.create {{.*}} {pending_count = 2 : i32}
// EMIT: nvws.semaphore.release {{.*}} {arrive_count = 2 : i32, ttg.partition = array<i32: 3>}
// LOWER: ttng.init_barrier {{.*}}, 2
// LOWER: ttng.arrive_barrier {{.*}}, 2

// Release arrive-multiplicity (spec section 5.2, uniform pending count):
// a semaphore's pending count is a per-semaphore constant — every acquire
// site sees the same count and every acquire cycle must receive exactly
// that many arrives. Shape: producer {3} stores outside the inner loop;
// inside, {2} (carried) and {1} read, {1} CORRECTS the buffer in place
// (the store's WAR obligations resolve early: {0}-less here — {2}'s via
// the transitive-sync skip), and {0} consumes the corrected value AFTER
// the store. The last version's holders at the inner EXIT are therefore
// {1} (the writer) and {0} (its reader) — a fan-in-2 regain — while the
// For-row unification merges the outer single-source ready edge onto the
// SAME semaphore. The lone outer release must arrive twice: r S(2). At
// commit 3 this is checked on the SYNC-DAG dump; commit 4 extends it with
// IR-level checks (async_ops arity carries the multiplicity).

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK: function: @release_multiplicity_unified_fanin_regain
  tt.func @release_multiplicity_unified_fanin_regain(%lb: i32, %ub: i32, %step: i32) {
    %alloc = ttg.local_alloc {buffer.id = 700 : i32} : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    scf.for %i = %lb to %ub step %step : i32 {
      %v = "producer"() {ttg.partition = array<i32: 3>} : () -> !ty
      ttg.local_store %v, %alloc {ttg.partition = array<i32: 3>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      scf.for %j = %lb to %ub step %step : i32 {
        %l2 = ttg.local_load %alloc {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use2"(%l2) {ttg.partition = array<i32: 2>} : (!ty) -> ()
        %l1 = ttg.local_load %alloc {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        %c = "correct"(%l1) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
        ttg.local_store %c, %alloc {ttg.partition = array<i32: 1>} : !ty -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
        %l0 = ttg.local_load %alloc {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !ty
        "use0"(%l0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      } {ttg.partition = array<i32: 0, 1, 2>}
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 0 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// The SYNC-DAG view (stderr dump). The outer ready release by {3} carries
// arrive multiplicity 2 — one release op, two arrives — because it shares
// the semaphore with the inner fan-in-2 regain ({1} the corrector and {0}
// its post-store reader both close into carried {2}); both acquire sites
// show the uniform pending count (2). The in-loop store itself takes NO
// edge: {2}'s WAR is discharged transitively ({1} synced behind {2} via
// its own read edge), {1}'s own read by program order. Under the v5 uniform
// hold rule, the outer S3 component remains carrier-bearing: the prefix's
// release has arrive multiplicity 2, so condition E rejects point-of-use
// (`rel-count`) to preserve the one-carrier-slot rule.
// CHECK: SYNC-DAG
// CHECK: |- a  S3  root  ; entry
// CHECK-NEXT: |- scf.for {{.*}} holdrule{c0:gated(rel-count)}
// CHECK-NEXT: |- ENTER
// CHECK-NEXT: |- W m0  ttg.local_store {3}
// CHECK-NEXT: |- r  S0(2)  {3} [none]
// CHECK-NEXT: |- a  S0(2)  {2}
// CHECK: |- R m0  ttg.local_load {2}
// CHECK-NEXT: |- r  S1  {2} [none]
// CHECK-NEXT: |- a  S1  {1}
// CHECK-NEXT: |- R m0  ttg.local_load {1}
// CHECK-NEXT: |- W m0  ttg.local_store {1}
// CHECK-NEXT: |- r  S2  {1} [none]
// CHECK-NEXT: |- r  S0  {1} [none]
// CHECK-NEXT: |- a  S2  {0}
// CHECK-NEXT: |- R m0  ttg.local_load {0}
// CHECK-NEXT: |- r  S0  {0} [none]
// CHECK-NEXT: |- a  S0(2)  {2}
// CHECK: |- r  S3  {2} [none]
// CHECK-NEXT: |- a  S3  {3}
// CHECK-NEXT: |- EXIT pieces{P0:W:{3}} yield{c0: a S3}
// CHECK: SEMAS c0: S0{count=2} S1{count=1} S2{count=1} S3{count=1 entry inherit={@0.3}}
