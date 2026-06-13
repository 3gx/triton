# Plan v2: implement the generic holding-region construction for nested loops

Implements `fable/holdrule-nest-completion-design-v2.md` — the one uniform
recursive construction (§2: sequence across every level, cut at context
change, anchor each hold at its endpoints' regions; single-loop is the
depth-1 degenerate). Scope: **insert-semas ONLY**. No feature phasing, no
classes-with-fallbacks, no gated-fallback path. The milestones are a
build/verify ORDER for one change, not a staging of features.

## Starting state (commit 220a53289b)

Semaphore combine is ENABLED. With combine on,
`automatic-warp-specialization.mlir` CRASHES on
`@grouped_matmul_tma_kernel`: combine's `getPartitionIds`
(`analyzeCombinedSemaphoreGroup`, LowerAref.cpp:791) asserts on the
attr-less ROOT-OUTSIDE entry acquire that the current rotated emission
hoists OUTSIDE the `tt.ws` loop for a nested crossing. That crash is the
concrete arbiter: the rotated shape is the only one that hoists an acquire
to root attr-less; the §2 construction keeps every acquire stamped INSIDE
the loop it belongs to. When the construction places the nested kernels,
the attr-less acquire is gone and combine stops asserting. Implement with
combine ON.

## Scope — insert-semas only

DOES:
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp`
  — replace `gateCrossing`'s binary verdict with the §2 placement
  (`classifyCrossing`); `applyHoldRulePlacement` realizes the placement;
  `computeHoldRuleGates`.
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp`
  — the poison-init seatbelt verifier; the in-loop / carrier-copy emission
  paths already exist and the emitter already stamps in-loop anchors'
  partition+stage via `emitInto` (walks to the ws-tagged ancestor,
  :48-77 — verified correct for anchors inside nested non-ws loops).
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h` — a
  `Crossing` placement field + dump.
- `test/NVWS/insert_semas_*.mlir` — golden regeneration + new goldens.

DOES NOT touch: AssignStagePhase, LowerAref/lower-semaphore beyond the
already-committed combine re-enable, PartitionLoops, pipeliner/
ScheduleLoops, partition scheduling. Per design-v2 §6, downstream passes
do not constrain the placement; if a §2 placement is not representable
downstream, that is a blocker (below), never a reason to widen the
critical region.

## The construction to implement (design-v2 §2–§3.5)

Per **component (buffer)**, over the whole nest: sequence the accesses
across every level; cut at execution-context change (owner or
enclosing-predicate set); each maximal uncut run is a hold; acquire before
its first element (in that element's region), release after its last (in
that element's region) — the two ends may sit at different levels; at a
cut between two holds, the release/acquire sit at the innermost region
containing that cut. A loop participates as an ENTER/EXIT pseudo-access
pair when a hold spans it or its boundary pairs unequal multiplicity.
Facts already on the DAG suffice (verified): For/If `pieceInfo` footprints
(stage 1-2), the backward outside-endpoint scan (`gateCrossing`
:1390-1403), the forward scan (:1413-1431), `parent`/`children` chain.

The placement yields three shapes (design-v2 §3 — **consequences of the one
construction, not separate classes needing guards**):
- **in-loop** (hold confined to one loop) → point-of-use acquire/release
  inside it; no entry acquire, no carrier, no slot; first acquire pairs
  with the create's initial credit. This is today's UNGATED path
  (EmitIR :477-479 slot skip; renderRegion :734-779 carrier erase).
- **spanning-inner** (one owner holds the buffer across a whole inner
  loop) → the acquire/release anchor at the enclosing level, before/after
  the inner loop; the token reaches the inner accesses by plain SSA
  capture (renderRegion copies the carrier into the body :759, `getView`
  mints the view inside :610-618; corpus proof `local_read_lifetime`).
  The node model places the off-region pair natively (Acquire spliced
  before a For-row :977/:1007, Release after :1009-1020 with async payload
  back-filled :337-344; a For-row dst with no in-body re-acquire takes its
  own semaphore :951).
- **carried** (same-owner hold across a loop's iteration bracket) → one
  carrier iter_arg at that loop; init = adopted live value or the entry
  acquire.

Per design-v2 §3.5, a buffer confined to one loop emits single-loop-like
regardless of enclosing nesting (per-buffer reduction), and single-loop is
the depth-1 instance of this same construction — **so flat kernels emit
byte-identical** (the verification obligation below).

`non-ws-loop` (:1365) is DELETED (a crossing over a non-ws loop is
classified by §2). `nested-final` (:1371) is **relaxed loop-scoped, NOT
deleted**: it fires whenever the carrier's final producer is in a nested
region, which is a **For** in a genuine nest (→ relax, classify by §2) but
an **`scf.if`** in a depth-1 conditional (→ KEEP gated, the
conditionality-cut emission is correct and must stay byte-identical).
Verified: `local_cfg`, `if_split_metadata`, `conditional_multi_result` all
dump `gated(nested-final)` with NO nested loop — deleting `nested-final`
blindly would regress these depth-1 conditionals. So the relax condition
is: `nested-final` ⇒ relax IFF the final's enclosing nested region is a
`scf.for`; otherwise keep. The existing outside-endpoint detectors
(`entry-consumed`, `result-consumed`, `region-feed`, `release-feed`,
`entry-sema-mismatch`, `trailing-use`) become evidence for the
spanning/carried placement. The prefix-shape reasons (`no-buf`,
`rel-count`, `rel-before-buf`) and `region-crossing` / `no-final` /
`no-entry-acquire` remain gated (malformed-input or unsupported-shape
diagnostics) — see the relax-only oracle for the full reason partition.

One real placement fix needed (B5, insert-semas-side):
`applyHoldRulePlacement` (:1504-1519) unconditionally unlinks
`holdFeedAcquire` (:1517); an in-loop hold with no outside endpoint has no
feed → null-deref today. Skip the feed unlink when there is no feed and
mark the in-body semaphore `isEntry` (template: the iteration re-acquire
that seeds the next iteration, InsertSemasSyncDag.cpp:1003; first acquire
pairs with the initial credit).

## Downstream realizability (out of scope; blocker on fire)

Per design-v2 §6, the placement has no exceptions and downstream passes do
not constrain it. The emitter stamps attrs as today (`emitInto`); in-loop
anchors at any depth get the same partition+stage stamping that flat
kernels already get (verified). Whether a placement is representable by a
DOWNSTREAM pass (AssignStagePhase stage/phase threading, lower-semaphore
pending-count, the pipeliner) is a realization question, NOT a placement
decision:
- Acquire-before-release order within a body and the absence of
  cross-partition releases split across an `scf.if` for one semaphore are
  natural properties of the point-of-use shape (so the pending-count wave
  analysis, SemaphorePendingCount.cpp:64-108, is satisfied without special
  handling).
- A `[tma_load]` producer's loads and its release are one hold in one
  region, so the construction never separates them across a loop boundary
  (the lowering's arrive-rides-the-load / BarrierExpect-at-first-load
  assumption, LowerAref.cpp:204-264, is met by where the hold sits).
- If a placement nonetheless trips a downstream pass (e.g. AssignStagePhase
  `DenseMap::at` :1183 on a single-group carrier+point-of-use coexistence),
  that is a **blocker** → STOP, reduce, report; close it in the realization
  or by a downstream ruling — never by widening the region back to rotated.

## Verifiers / in-pass seatbelts

- **Relax-only oracle** (covers ALL gateCrossing reasons; partition is
  exhaustive so no reason is unclassified):
  - **may relax** (newly placed in-loop/spanning vs the old rotated form):
    `non-ws-loop`; `nested-final` ONLY when the nested final is inside a
    `scf.for`; and the outside-endpoint detectors `entry-consumed`,
    `result-consumed`, `region-feed`, `release-feed`, `entry-sema-mismatch`,
    `trailing-use` (used as spanning/carried evidence).
  - **must stay gated** (the construction must NOT change their emission):
    `nested-final` when the nested final is inside an `scf.if` (depth-1
    conditional — byte-identical), and `region-crossing`, `no-final`,
    `no-entry-acquire`, `no-buf`, `rel-count`, `rel-before-buf`
    (unsupported-shape / malformed-input diagnostics).
  Hard error if a crossing in the "must stay gated" set is placed
  in-loop/spanning, or if any reason falls outside this partition. Run on
  every input.
- **`verifySingleCarrierPerGroup`** (EmitIR :1501-1531) — kept, hard; firing
  is a blocker (see above), not a fallback trigger.
- **`verifyNoUseAfterRelease`** (EmitIR :1436-1470) — kept; an in-loop
  hold's release must postdate the component's last view in every path.
- **Poison-init verifier (new):** a materialized (carrier) slot's init must
  resolve to a real token, never `ctx.poison` (today `verifyTokenLocality`
  passes poison silently, :1360-1361; renderRegion inits absent carriers to
  poison :737-739). Catches a mis-placement invisible until runtime.
- Fix the stale `crossCheckHoldRule` comment (SyncDag:1349).
- `holdrule{}` dump prints the placement (`in-loop` / `spanning@<level>` /
  `carried@<level>`).

## Build (after ANY C++ change, before ANY test) — per AGENTS.md

```
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/ && ninja triton triton-opt
```
FIRST BUILD, THEN RUN lit-tests. A C++ edit with no rebuild silently tests
the old pass.

## Gates (HARD; each with a 60s timeout) — combine ON

lit-tests are run per AGENTS.md: from the build folder
(`build/cmake.linux-x86_64-cpython-3.12/`), with
`/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test[/<path>]`.

1. **AWS mlir gate** (compile), from the build folder:
   ```
   timeout 60 /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/TritonGPU/automatic-warp-specialization.mlir
   ```
   Crash → GREEN: `@grouped_matmul_tma_kernel` is placed by §2, the
   attr-less outside-loop acquire goes away, combine stops asserting. Its
   CHECK lines regenerate to the placed emission, audited line-by-line.
   (Full NVWS suite at M2: `... llvm-lit -v test/NVWS`.)
2. **4 warp-spec pytests** (`TRITON_ALWAYS_COMPILE=1`, 60s each — pytest
   only where this plan's runtime gate calls for it, per AGENTS.md "do not
   run pytest unless explicitly told"):
   ```
   test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]
   test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]
   test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]
   test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024]
   ```
   (file: `python/test/unit/language/test_warp_specialization.py`)
3. **grouped gemm** (`TRITON_ALWAYS_COMPILE=1`, 60s):
   ```
   python/test/unit/language/test_warp_specialization.py::test_grouped_gemm[16-4096-8192-1024]
   ```
4. **run_nvws.sh** (60s): `timeout 60 sh run_nvws.sh` (NOT run_nvws_1.sh).

Any hang: `third_party/tlx/killgpu.sh`, capture IR, STOP.

## Milestones (build/verify order for ONE change)

### M0 — baseline + change-detector goldens
1. Build at tip; run all gates; record state. With combine ON the baseline
   is: AWS gate KNOWINGLY RED (the `@grouped_matmul_tma_kernel` crash = the
   target); 4 pytests, grouped_gemm, run_nvws.sh, NVWS lit suite GREEN.
2. Dump `holdrule{...}` (`NVWS_INSERT_SEMA_DUMP_DAG=1`) for the corpus
   already captured in `logs/nested-12jun26-v1/` (grouped_gemm, attn /
   matmul persistent, meta_fa_fwd) — the before-picture to diff against.
   Per the design-v2 §7 verification, every component places by §2; M1's
   dump must match that placement.
3. Add change-detector goldens pinning CURRENT (rotated) emission, flipped
   + audited at M2:
   - `insert_semas_nested_continuation.mlir` — perfect double-nest
     cross-owner ping-pong (accesses only in the inner loop) + triple-nest
     re-entry (in-loop).
   - `insert_semas_nested_device.mlir` — outer-body prologue feeding
     per-inner-iteration reads + epilogue after an inner loop
     (spanning-inner). (`local_read_lifetime` already covers the read-only
     single-owner spanning variant.)

### M1 — implement the §2 placement (the change)
Implement together:
1. `gateCrossing` → `classifyCrossing`: delete `non-ws-loop`; relax
   `nested-final` ONLY when the final's nested region is a `scf.for` (keep
   it gated for `scf.if` — depth-1 conditional, byte-identical); compute
   the §2 placement (in-loop / spanning-inner anchored at the enclosing
   level / carried) from the outside-endpoint scans + the multiplicity
   test (same-region-only equal); keep malformed-input
   diagnostics. Emit the placement into the dump.
2. `applyHoldRulePlacement`: in-loop → point-of-use + feed-less `isEntry`
   (B5); spanning-inner → the off-region acquire/release pair, no slot
   below the cut; carried → one carrier slot at the bracket only.
3. Verifiers: relax-only oracle, poison-init, keep
   `verifySingleCarrierPerGroup` + `verifyNoUseAfterRelease`; stale-comment
   fix.
Build, then dump `holdrule{...}` on the corpus and diff against M0 +
design-v2 §7: confirm every component places as §7 says (grouped_matmul
A/B in-loop, acc composing across levels; persistent attention `q`
spanning-inner; etc.) and no attr-less outside-loop acquire remains on any
combine-eligible component.

Gate M1: insert-semas runs clean on the corpus pre-lower IR; relax-only
oracle + single-carrier + no-use-after-release + poison-init hold on the
full corpus; placement matches design-v2 §7.

### M2 — goldens + compile gate
1. Regenerate every flipped lit golden. Per the design-v2 §7 corpus
   verification, the expected diffs are:
   - `nested_carrier` ×3 — rotated → §2 composition (in-loop + carried +
     spanning-inner handoff).
   - `meta_fa_fwd` — persistent FA placed by §2 (`q` spanning-inner, etc.).
   - `sequential_ws_loops` (E4) — regenerates to **symmetric continuation**
     (both loops native, loop A's release pairs with loop B's acquire on
     one semaphore pair; no entry acquire / carrier / bottom re-acquire /
     seam release) — design-v2 §7.4. This is an improvement, not a
     regression; audit it as such.
   - the AWS file's `@grouped_matmul_tma_kernel`.
   - `insert_semas.mlir` — the 5 hidden nested funcs (`@hoisted_alloc`,
     `@nested_loop_yes/no_double_buffer`, `@nested_loop_yes/no_double_buffer_scaled`)
     flip (rotated → §2 spanning/in-loop). The other 23 funcs are depth-1
     and MUST stay byte-identical — incl. the conditional funcs that dump
     `gated(nested-final)` via an `scf.if`, which the loop-scoped relax
     keeps unchanged.
   - `insert_semas_per_edge_tmem.mlir` — `@tmem_nested_linear_chain_no_outer_drain`
     flips (inner-confined buffer, rotated → in-loop); 4 other funcs
     byte-identical.
   - `insert_semas_live_tag_source.mlir` — its acc component flips
     (carried + spanning-inner), like grouped_gemm's acc.
   - the M0 change-detectors; any other the dump flips.
   Audit each diff line-by-line: every change is a placement move
   (slot/entry-acquire removal for in-loop, off-region pair for
   spanning-inner, carrier kept for carried) — nothing unexplained. Flat
   goldens (and every depth-1 conditional) MUST stay byte-identical
   (design-v2 §3.5 / §7.3); a flat churn is a regression to stop on.
2. Full NVWS lit suite green.
3. AWS mlir gate GREEN with combine ON (the M0 crash resolved).

Gate M2: NVWS lit suite green; AWS gate crash→green with combine ON; every
golden diff audited; flat goldens byte-identical.

### M3 — runtime gates (the arbiters), combine ON
Run gates 2–4 (4 warp-spec pytests, grouped_gemm, run_nvws.sh), 60s each,
`TRITON_ALWAYS_COMPILE=1`. Persistent attention/matmul and grouped_gemm are
the placed kernels, now running through enabled combine.

Gate M3: all green, no hang, combine ON.

## Blocker protocol (BINDING)

A blocker is anything not resolvable inside insert-semas. If a §2 placement
makes a DOWNSTREAM pass crash or miscompile — AssignStagePhase
(`DenseMap::at`, AssignStagePhase.cpp:1183), the single-carrier verifier
firing on a real kernel, lower-semaphore (pending-count
authored-vs-analysis mismatch, LowerAref.cpp:151-154), PartitionLoops, the
pipeliner, or a runtime hang/race — then:
1. STOP. Do not edit any pass outside insert-semas.
2. `third_party/tlx/killgpu.sh` if a GPU run hangs; capture the IR
   (pre-lower + the failing pass's input).
3. Reduce to a minimal `.mlir` reproducer (one ws loop + one nested loop,
   smallest component set) that fails both ways: passes when the crossing
   stays rotated, fails under the §2 placement.
4. Return to the user with the reduced example, the exact pass + file:line,
   and what a fix would require (which pass, which invariant). Do NOT
   proceed.

## Churn budget
- New goldens: the two M0 change-detectors (+ any reduced reproducers from
  blockers).
- Regenerated (per design-v2 §7 + the corpus coverage check, all
  expected): `nested_carrier` ×3, `meta_fa_fwd`,
  `sequential_ws_loops` (→ symmetric continuation), the AWS file's
  grouped_matmul kernel, `insert_semas.mlir` (5 hidden nested funcs only),
  `insert_semas_per_edge_tmem.mlir` (`@tmem_nested_linear_chain_no_outer_drain`
  only), `insert_semas_live_tag_source.mlir` (acc component), the
  change-detectors. Every diff audited as a placement move. All other
  goldens — incl. every depth-1 conditional — byte-identical (a flat/
  conditional churn = stop).

## References
- Design: `fable/holdrule-nest-completion-design-v2.md` (§2 construction,
  §3 placements, §3.5 per-buffer / single-loop degenerate, §6 no
  exceptions, §7 full-corpus verification).
- Rule: `fable/rule-v2-corpus-verification.md` §7.2 / §7.6.
- Corpus bodies: `logs/nested-12jun26-v1/`.
- Build/ground rules: `AGENTS.md`, `fable/hold-rule-implementation-plan.md`.
- Combine re-enable + crash: commit 220a53289b; the assert at
  LowerAref.cpp:791.
