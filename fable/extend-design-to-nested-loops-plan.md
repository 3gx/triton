# Plan: extend hold-rule point-of-use emission to nested ws loops

Scope: **insert-semas ONLY**. Goal: stop nested/persistent kernels from
emitting the rotated boundary device (the −47TF shape) where the rule
derives CONTINUATION. Design basis: `fable/holdrule-nest-completion-design.md`.

## What this plan does and does NOT touch

DOES (insert-semas pass, three files):
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp`
  — the classifier (`gateCrossing` / `computeHoldRuleGates`),
  `applyHoldRulePlacement`, and the in-pass MERGE/point-of-use
  coexistence guard (B3 fallback, see N1).
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp`
  — (a) anchor stage/cluster inheritance at render (B1, the one
  required emission change — EmitIR:836-896) and (b) the poison-init
  verifier. The ungated emission path otherwise already exists.
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h` — only
  if a Crossing field is needed for the classification dump.
- `test/NVWS/insert_semas_*.mlir` — golden regeneration + new goldens.

DOES NOT touch (out of scope — see Blocker protocol):
- AssignStagePhase, LowerAref/lower-semaphore, PartitionLoops, the
  pipeliner / ScheduleLoops, partition scheduling, any other pass.
  B1/B3 above are insert-semas-side mitigations that PREVENT downstream
  breakage; they are in scope precisely so these passes need no edit.

## Known data this plan is built on (verified this session, file:line)

1. The only thing forcing the rotated shape on nested crossings is two
   short-circuits in `gateCrossing`: `non-ws-loop`
   (InsertSemasSyncDag.cpp:1365) and `nested-final` (:1371). Both fire
   BEFORE any shape analysis; inherited from the deleted fixup's
   flat-only domain.
2. The CONTINUATION emission path already exists — it is today's UNGATED
   path (EmitIR M2 skip :477-479; renderRegion ungated :734-779; carrier
   erase :762-779). Flipping a nested crossing gated→ungated routes
   through existing machinery.
3. Stamping inside non-ws inner loops is already correct (emitInto walks
   to the ws-tagged ancestor, EmitIR:48-77).
4. `applyHoldRulePlacement` unconditionally unlinks `holdFeedAcquire`
   (:1517); a continuation with no outside endpoint has no feed → this
   null-derefs today (must be handled — B5 in the design doc).
5. Corpus flip set is known from the dumps: persistent attention 5/6
   gated, grouped_gemm all gated, persistent matmul mixed.

## v1 scope: CONTINUATION class only

- A nested crossing is reclassified to UNGATED (point-of-use) IFF it is
  CONTINUATION: the component has NO outside endpoint at this loop
  boundary (all its accesses are inside this loop / the recurring
  cycle). Equal-multiplicity is concluded ONLY when endpoints share the
  same region (design doc §3 zero-trip safety rule).
- Everything else stays GATED, unchanged: MERGE (same-owner bracket
  carrier), and the DEVICE shape (outer prologue feeding inner reads)
  falls back to GATED in v1 (native DEVICE is a future step — it is not
  needed for the gate kernels, which flip on CONTINUATION).
- All protocol-shape gate reasons (trailing-use, entry-consumed,
  region-feed, release-feed, result-consumed, no-buf, rel-count,
  rel-before-buf, entry-sema-mismatch) are UNCHANGED.
- **B3 coexistence guard (in-pass, required for v1):** if a CONTINUATION
  flip would put a point-of-use acquire and a MERGE carrier of the SAME
  semaphore group in the SAME loop body, the component stays GATED.
  Persistent attention is exactly this shape (K/V = CONTINUATION, acc =
  MERGE in the outer loop); without the guard the coexistence crashes
  AssignStagePhase (propagateStage `DenseMap::at`, :1183) downstream.
  The guard keeps the fix inside insert-semas. (TMA-split, B4, cannot
  arise in v1: CONTINUATION co-locates producer and release; DEVICE,
  the only splitting class, stays gated.) Acquire-before-releases body
  order (B2) is auto-satisfied by point-of-use emission.

In-pass oracle (the safety invariant): the classifier may only RELAX —
every crossing it newly ungates must have had gate reason `non-ws-loop`
or `nested-final` under the old logic. It must NEVER convert a
protocol-shape gate to point-of-use. Assert this on every input.

## Build steps (after ANY C++ change, before ANY test)

Per `fable/hold-rule-implementation-plan.md` ground rules:
```
cd build/cmake.linux-x86_64-cpython-3.12/ && ninja triton triton-opt
```
A C++ edit with no rebuild silently tests the old pass.

## Gates (HARD; each with a 60s timeout)

lit runner (this machine): `lit` module lives only under python3.6
site-packages —
`PYTHONPATH=/home/egaburov/.local/lib/python3.6/site-packages /usr/bin/python3 -W ignore /home/egaburov/.local/bin/lit`.

1. **AWS mlir gate** (compile):
   ```
   timeout 60 <lit> -s build/cmake.linux-x86_64-cpython-3.12/test/TritonGPU/automatic-warp-specialization.mlir
   ```
   Must end GREEN. This file contains `@grouped_matmul_tma_kernel`, which
   v1 may reclassify; if its CHECK lines change, they are regenerated and
   audited line-by-line as a continuation flip at N2 (same as any other
   flipped golden). Green is the bar.
2. **4 warp-spec pytests** (`TRITON_ALWAYS_COMPILE=1`, 60s each):
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

## Milestones

### N0 — baseline + before-picture (no behavior change)
1. Build clean at tip; run ALL gates above; confirm green. Record the
   numbers/pass-state — this is the regression baseline.
2. Dump and save `holdrule{...}` decisions (`NVWS_INSERT_SEMA_DUMP_DAG=1`)
   for: grouped_gemm, persistent attention, persistent matmul, the
   run_nvws kernel. This is the exact flip set (which crossings are
   `non-ws-loop`/`nested-final` today) — the change must touch only these.
3. Add a reduced golden `test/NVWS/insert_semas_nested_continuation.mlir`
   (perfect double-nest cross-owner ping-pong, accesses only in the inner
   loop) capturing the CURRENT (gated, rotated) emission, with a header
   noting it pins pre-change behavior and will flip at N1.

Gate N0: all hard gates green; baseline recorded.

### N1 — the classifier (the one real change)
1. In `gateCrossing`: replace the `non-ws-loop` (:1365) and
   `nested-final` (:1371) short-circuits with CONTINUATION detection
   using facts already on the DAG (the backward outside-endpoint scan
   :1390-1403 and forward scan :1413-1431; For/If pieceInfo footprints).
   A crossing with no outside endpoint at the boundary → UNGATED
   (continuation). Otherwise keep the existing reason and stay gated.
   Same-region equal-multiplicity only (zero-trip safety).
2. B3 coexistence guard: in the classifier, before flipping a crossing
   to CONTINUATION, check the loop body for a MERGE carrier of the same
   semaphore group; if present, keep the crossing GATED. This is what
   keeps persistent attention (K/V continuation + acc MERGE in one outer
   loop) from crashing AssignStagePhase downstream — the guard is the
   reason no downstream edit is needed.
3. Handle the feed-less continuation in `applyHoldRulePlacement` (:1504-
   1519): when there is no `holdFeedAcquire`, do not unlink it; mark the
   in-body semaphore `isEntry` (template: back-edge placement :1003).
   First acquire pairs with the create's initial credit.
4. B1 anchor stage/cluster inheritance (EmitIR render, :836-896): a
   point-of-use acquire/release emitted inside a re-scheduled inner loop
   must carry stage/cluster inherited from its protected access
   (aq.stage ≤ first-access.stage, rel.stage ≥ last-access.stage). The
   flat kernel already proves in-body waits survive the inner-loop
   pipeliner when stages are inherited; this extends the same to nested
   anchors so the pipeliner needs no change.
5. In-pass oracle assert (the relax-only invariant above) — runs on
   every input, hard error on violation.
6. Add the poison-init verifier in EmitIR: a materialized (gated) slot's
   init must resolve to a real token, never `ctx.poison` (today
   verifyTokenLocality passes poison silently, :1360-1361).
7. Fix the stale `crossCheckHoldRule` comment (SyncDag:1349).
8. Update the `holdrule{}` dump to print the class
   (`continuation`/`gated(<reason>)`).

Build, then run the reduced golden + the standalone insert-semas dumps
on the gate kernels: confirm the flip set matches N0's expectation
(only `non-ws-loop`/`nested-final` crossings became `continuation`).

Gate N1: insert-semas pass runs clean (no crash) on the gate kernels'
pre-lower IR; oracle assert holds on the full corpus.

### N2 — goldens + compile gate
1. Regenerate the insert-semas lit goldens that flip
   (`nested_carrier`, `meta_fa_fwd`, the new continuation golden, any
   others the dump shows). Audit each diff line-by-line: every change
   must be a gated→continuation flip (slot/entry-acquire removal,
   in-body anchor) — nothing else.
2. Full NVWS lit suite green.
3. AWS mlir gate green (CHECK lines regenerated + audited if
   `@grouped_matmul_tma_kernel` flipped).

Gate N2: NVWS lit suite green; AWS gate green; every golden diff audited
as a continuation flip.

### N3 — runtime gates (the arbiters)
Run gates 2–4 (4 warp-spec pytests, grouped_gemm, run_nvws.sh), 60s each,
`TRITON_ALWAYS_COMPILE=1`. These are the correctness arbiters for the
flip — persistent attention/matmul and grouped_gemm are exactly the
kernels that reclassify.

Gate N3: all green, no hang.

## Blocker protocol (BINDING)

A blocker is anything that cannot be resolved inside the three
insert-semas files above. Specifically: if flipping a crossing to
continuation makes a DOWNSTREAM pass crash or miscompile —
AssignStagePhase (e.g. propagateStage `DenseMap::at`,
AssignStagePhase.cpp:1183), lower-semaphore (pending_count
authored-vs-analysis mismatch, LowerAref.cpp:151-154), PartitionLoops,
the pipeliner/ScheduleLoops, or a runtime hang/race — then:

1. STOP. Do not edit any pass outside insert-semas.
2. `third_party/tlx/killgpu.sh` if a GPU run hangs; capture the IR
   (pre-lower + the crashing pass's input).
3. Reduce to a minimal `.mlir` reproducer (single ws loop + one nested
   loop, smallest component set) that demonstrates the root cause, both
   ways (passes when the crossing stays gated, fails when continuation).
4. Come back to the user with: the reduced example, the exact pass +
   file:line of the failure, and what the fix would require (which pass,
   which invariant). Do NOT proceed.

This is expected for: DEVICE-class shapes (kept gated in v1 — do not
flip), any continuation whose downstream stage/phase or pending_count
derivation breaks. Those define the v1/v2 boundary empirically rather
than by theorizing.

## Churn budget
- New goldens: `insert_semas_nested_continuation.mlir` (+ any reduced
  reproducers from blockers).
- Regenerated: `nested_carrier`, `meta_fa_fwd`, and the AWS file if
  `@grouped_matmul_tma_kernel` flips. Every diff audited as a
  continuation flip; no unexplained CHECK changes.
