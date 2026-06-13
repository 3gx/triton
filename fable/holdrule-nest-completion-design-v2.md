# HOLD-RULE COMPLETION: boundary classes for arbitrary loop nests

Status: DESIGN PROPOSAL, 12jun26 (evening). Nothing implemented; tip is
green at 5c11da9c87. Authored after the persistent-class finding: the
point-of-use realization covers FLAT ws loops only, and every nested
kernel (persistent attention, persistent/grouped matmul, triple nests)
falls back to the rotated boundary device — the measured −47TF shape.
This doc is the completion design: same rule, full structural range.

Companion evidence (this session, all verified on real kernels):
- persistent attention pytest: 5 of 6 component classes gated
  (nested-final / non-ws-loop); flat variant: 5 of 6 point-of-use.
- test_grouped_gemm: triple nest, EVERYTHING gated.
- test_warp_specialize_tma_matmul_persistent[True-False-8-…]: ws tag on
  the INNER K-loop → its smem components are point-of-use; placement of
  the ws tag relative to the nest currently decides emission shape.
- The gate reasons are not derived from the rule: `non-ws-loop`
  (InsertSemasSyncDag.cpp:1365) and `nested-final` (:1371) fire BEFORE
  any shape analysis — inherited verbatim from the deleted fixup's
  conversion domain.

## 0. The contract (unchanged rule, completed realization)

THE HOLD RULE itself is unchanged (rule-v2-corpus-verification.md §7.2).
What changes: gateCrossing's binary gated/pointofuse verdict becomes a
per-(component, loop-boundary) CLASSIFICATION with three outcomes, each
with its own realization:

| class | DAG evidence at this boundary | realization |
|---|---|---|
| CONTINUATION | component has NO outside endpoint at this loop, or the outside endpoint belongs to the same recurring cycle | NOTHING at the boundary. In-body point-of-use aq/rel. First-ever acquire pairs with the create's initial credit. No entry acquire op, no carrier, no slot at ANY depth. |
| DEVICE | unequal multiplicity: an endpoint that executes once per ENTER/EXIT event of this loop pairs against a per-iteration endpoint inside | aq/rel pair rendered in the innermost region containing the cut (outer body): aq before the loop / rel after the loop. Token reaches inner accesses by plain SSA capture — no iter_arg. |
| MERGE | same-owner hold merged across this loop's bracket pair | carrier iter_arg at EXACTLY this loop; token threading only below it; entry/adoption seeds the init. |

Multiplicity reference frame (the spec amendment §6 below makes this
binding): multiplicity is counted RELATIVE TO THIS BOUNDARY — "once" =
once per ENTER/EXIT event of this loop, "per-iteration" = once per
iteration of this loop. Outer-prologue ops are once-per-ENTER of the
inner loop regardless of how many times the outer loop runs.

## 1. The classes in pseudo-IR

### 1.1 CONTINUATION — the persistent-kernel core (user's sketch)

```
S_e = create true ; S_f = create false
for outer (ws) {
  for tile {
    for k {
      aq  S_e {p2}              // 1st-ever firing: initial credit;
      descriptor_load {p2}      //  every later firing: previous rel,
      rel S_f [tma] {p2}        //  across ANY loop boundary — mbarrier
      aq  S_f {p1}              //  phase just keeps flipping
      mma {p1}
      rel S_e [tc5mma] {p1}
    }
  }
}
```
Ledger: aq S_e fires N_total times; credits = 1 init + (N_total−1) rels.
Tokens are body-local; nothing crosses any boundary. Zero-trip safe by
construction (aq and rel co-located: 0 trips = 0 waits AND 0 arrives;
partitions execute CLONED nests, so producer and consumer skip
together).

### 1.2 DEVICE — imperfect nest, outer prologue (Q-load class)

```
for outer (ws) {
  aq  S_b {p3}                  // bracket cut, point of use
  q_load {p3}                   // once per outer iteration
  rel S_a {p3}
  aq  S_a {p1}                  // ← DEVICE pair at the INNER boundary,
  for inner {                   //    rendered in the OUTER body
    mma reads q [token of S_a]  //    (per-inner-iteration reads)
  }
  rel S_b {p1}                  // after the loop; payload back-filled
}                               //  from the inner accesses
```
The S_a token is captured into the inner region as plain SSA — the
emitter ALREADY renders this when no in-body acquire exists
(renderRegion copies carrier maps wholesale, EmitIR:759; corpus proof:
insert_semas_local_read_lifetime.mlir pins exactly this shape).

### 1.3 MERGE — same-owner across the bracket (acc class)

Unchanged from E0/nested_carrier: merged tail|head hold → carrier at the
merging loop, nested token threading below. This is today's gated shape
applied ONLY where the rule actually derives a merge.

### 1.4 What flips on the persistent attention kernel

Today: K/V smem, softmax/stat comps = gated(nested-final + non-ws-loop)
→ rotated. Under classification: all of them = CONTINUATION (accesses
only inside the kv loop, recurring cycle) → in-body point-of-use, which
is byte-equivalent to what the FLAT attention kernel already gets. The
acc = MERGE at the outer loop (init/epilogue same owner across the
persistent bracket) — keeps its carrier, as it does in the flat kernel
(gated entry-consumed).

## 2. Why the realization is small: the model already supports it

Verified by code-walk (file:line):
- Region rows are first-class edge endpoints; Acquire splices BEFORE a
  For row (:977,:1007), Release AFTER it (:1009-1020) with async
  payloads back-filled from the inner accesses (:337-344) — the DEVICE
  shape exists in the node model today.
- For-row dst groups with no in-body regain get their OWN semaphore
  (:951) — pure DEVICE pairing exists.
- Entry-less first-acquire-pairs-with-credit exists: the back-edge
  placement (:1003) sets isEntry with NO pre-loop instance; the emitter
  is purely node-driven and tolerates absent entry nodes
  (emitEntryAcquires EmitIR:304-333, seeding :1571-1582).
- Slot suppression is nest-depth-agnostic already: the M2 skip
  (EmitIR:477-479) + renderRegion ungated path (:734-779). The emitter
  never sees an ungated nested crossing ONLY because gateCrossing
  blanket-gates them upstream.
- Stamping inside non-ws inner loops is already correct: emitInto walks
  up to the ws-tagged ancestor; partition+stage attrs stay (EmitIR:48-77).
- computeRequiredParts runs after placement and derives clone sets from
  the rows recursively (:1330-1341) — reclassification updates them for
  free.

The work concentrates in: the classifier (replacing gateCrossing's
reason-1–3 short-circuits), applyHoldRulePlacement (a third outcome:
continuation-without-feed — today it null-derefs, see B5), and the four
cross-pass blockers below.

## 3. Blockers the design must carry (red-team verified, file:line)

**B1 — PIPELINING (the load-bearing discovery).** Inner non-ws loops are
re-scheduled and pipelined AFTER insert-semas (internal ScheduleLoops at
AutomaticWarpSpecialization.cpp:126 → compiler.py:442 expander; lowered
waits/arrives are latency-class ops, ScheduleLoops.cpp:797-801). An
in-body anchor with NO loop.stage in a scheduled loop either asserts
(useMetaWS, ScheduleLoops.cpp:810-814) or is silently pushed to the last
stage (no SSA edge wait→access) → RACE. The gated form dodges this only
because OUTER loops are never pipelined (ScheduleLoops.cpp:39-41). The
flat variant is the existence proof that in-body waits survive
pipelining WHEN stage attrs are inherited. → Design invariant: every
point-of-use anchor inside a scheduled loop carries stage/cluster
inherited from its protected access: aq.stage ≤ first-access.stage,
rel.stage ≥ last-access.stage, per (component, loop). The expander
already predicates all four protocol ops (PipeliningUtility.cpp:232-282).

**B2 — pending_count wave analysis is backedge- and predicate-blind**
(SemaphorePendingCount.cpp:64-108). A continuation body whose releases
LEAD the acquire in walk order gets its waves unioned → wrong derived
count → either authored-vs-analysis hard error (LowerAref.cpp:151-154)
or, if authored to match the union, a runtime hang (the verified FA-meta
mechanism). → Design invariant: continuation emission keeps
acquire-before-releases order per body (natural for point-of-use), and
the classifier must never emit cross-partition releases split across
then/else for one semaphore. If a future shape needs leading releases,
the analysis must become loop-aware first — separate, gated work.

**B3 — AssignStagePhase carrier bookkeeping** (AssignStagePhase.cpp:
947-948, :1103, crash :1183). state.token is overwritten by EVERY group
acquire in lexical order; a MERGE carrier mixed with a later same-group
point-of-use acquire in one body registers the WRONG token →
propagateStage DenseMap::at crash. verifySingleCarrierPerGroup
(EmitIR:1501-1531) rejects two SLOTS but not slot+in-body-same-group. →
Design invariant: per (semaphore group, loop body), a MERGE carrier and
point-of-use acquires of the SAME group must not coexist — the
classifier must verify this and fall back to gated (rotated) for that
component if it occurs; OR AssignStagePhase registers the yielded token
explicitly (an AssignStagePhase fix = outside InsertSemas, needs a
ruling).

**B4 — TMALoad payload locality** (LowerAref.cpp:204-264, :241-250,
:349-356). A [tma_load] release emits NO arrive at the release point —
the arrival rides the TMA op and the single BarrierExpect is inserted at
the FIRST LOAD. A DEVICE pair separating loads from their release across
a loop boundary deadlocks at zero trip and over-expects at N>1. →
Design invariant: a [tma_load]-payload release must stay in the same
block as all loads its semaphore covers. CONTINUATION satisfies this
naturally; the classifier must FORBID the DEVICE class from splitting a
tma_load producer (fall back to gated).

**B5 — applyHoldRulePlacement null-deref** (InsertSemasSyncDag.cpp:1517).
It unconditionally unlinks holdFeedAcquire; CONTINUATION has no feed by
definition. The classifier's third outcome must skip the feed unlink and
instead mark the in-body semaphore isEntry (template: the :1003
back-edge path).

Plus three safety rules from the red team:
- Equal-multiplicity ⇒ CONTINUATION may only be concluded when both
  endpoints share the SAME region (zero-trip attack: cross-region static
  equality breaks under dynamic trip counts → over-arrive).
- Poison-init verifier: a gated/DEVICE slot whose init resolves to
  ctx.poison is a silent mis-classification today (EmitIR:737-739;
  verifyTokenLocality deliberately passes poison :1360-1361). Add a
  verifier: every real slot's init must be a real token.
- Single-phase eligibility (AssignStagePhase.cpp:137-208) simulates one
  body pass; new continuation patterns should advance the ring at least
  once per inner body touching a depth>1 group (verifier, not assumed).

## 4. What changes, where

1. `gateCrossing` → `classifyBoundary` (InsertSemasSyncDag.cpp): delete
   the `non-ws-loop`/`nested-final` short-circuits; compute the outside-
   endpoint facts it ALREADY computes (backward scan :1390-1403, forward
   scan :1413-1431) plus the multiplicity frame; emit class + reason
   into the dump (`holdrule{c0:continuation}` / `device@<row>` /
   `merge` / `gated(<fallback reason>)`). Gated stays as the SAFE
   FALLBACK class for: B3 coexistence, B4 tma-split, prefix-shape
   reasons (no-buf/rel-count/rel-before-buf), trailing-use.
2. `applyHoldRulePlacement`: third outcome (B5); DEVICE keeps the
   already-injected For-row pair; MERGE keeps the slot only at the
   merge loop (suppress slots below the cut's innermost region).
3. Emitter: near-zero for slots (skip extends to classified
   continuation at any depth); anchor stage/cluster inheritance (B1) at
   render (Acquire/Release rendering, EmitIR:836-896); poison-init
   verifier (new).
4. Out-of-InsertSemas items NEEDING A RULING before any edit:
   AssignStagePhase token registration (B3 alternative), pending-count
   loop-aware waves (B2 future), ScheduleLoops interplay is observation
   only (no edit anticipated; B1 is solved inside insert-semas by attr
   inheritance).

## 5. Validation plan

Goldens (write PINNING CURRENT gated behavior first, then flip with the
classifier and audit the diff — the M0→M2 pattern):
1. `insert_semas_nested_continuation.mlir` @perfect_double_nest_pingpong
   + @triple_nest_reentry — the user's sketch, depths 2 and 3.
2. `insert_semas_nested_device.mlir` @outer_prologue_device_inner_pingpong
   (device + in-body point-of-use coexisting — uncovered today) +
   @inner_exit_epilogue_device (native; @hoisted_alloc pins it gated).
3. `insert_semas_sequential_inner_loops.mlir` — one buffer, two
   sequential inner loops inside one ws loop (per-outer-iteration seam).
4. Reduced persistent-attention golden + reduced grouped-matmul golden
   (small enough to review per-component classes; meta_fa_fwd stays as
   the full-capture change-detector).
5. Zero-trip continuation + predicated inner loop (lower through
   partition-loops as crash probes; LOWER prefix where shapes allow).
Existing nested_carrier: cases reclassify (case 2 middle boundary =
continuation; case 1/3 keep MERGE at the merging level + DEVICE at
epilogue boundaries) — expected wholesale regeneration, audited line by
line.

Gates (same ladder as M0–M3, plus the motivating ones):
- compile: NVWS lit suite; AWS hard gate (WILL change for grouped_gemm
  if its kernels reclassify — that golden's diff is part of the review,
  not blindly regenerated);
- runtime in order: 4 warp-spec pytests, 2 moe, run_nvws.sh, 06-fa
  parity (flat must be UNCHANGED — fingerprint byte-diff), then the
  NEW arbiter: persistent attention + persistent matmul + grouped_gemm
  pytests, and the perf A/B on the persistent FA tutorial kernel
  (run_nvws.sh kernel) — the number this whole design exists to move.

In-pass oracle (M1 pattern): classifier dual-run — for every corpus
input, assert classified CONTINUATION ⊆ today's reasons {non-ws-loop,
nested-final} ∪ {pointofuse}, i.e. the classifier may only RELAX the
gate, never convert a protocol-shape gate (trailing-use, rel-count, …)
into point-of-use. Plus the C3 balance verifier and the poison-init
verifier as permanent seatbelts.

## 6. Spec amendments (rule-v2 + semas-report3)

1. §7.2 rule 1: define the multiplicity reference frame (the corpus
   agent's finding: "once-per-run" is ambiguous for nests — "once" must
   read "once per ENTER/EXIT event of THIS loop"). One sentence.
2. §7.2 rule 4: add the three-class realization table (§0 above) and
   the B1–B4 invariants as realization constraints alongside C1–C3.
3. E-series: add E5 (nested continuation, the user's sketch), E6
   (device-at-inner-boundary with coexisting in-body point-of-use).
4. semas-report3 Addendum B mirrors.

## 7. Open rulings for the user

1. B3 resolution: classifier fallback (stay inside InsertSemas, some
   shapes stay rotated) vs AssignStagePhase registration fix (one
   targeted edit outside InsertSemas, removes the fallback). Default
   proposal: fallback first, measure, then decide.
2. AWS golden: grouped_gemm kernels live in the AWS lit file — its
   CHECK lines will change if classification flips them. The M0-M3 rule
   was "AWS unmodified"; the completion intentionally changes emission,
   so the AWS diff becomes a REVIEWED artifact. Confirm.
3. DEVICE class in v1 or v2: persistent attention/matmul flip needs
   CONTINUATION (+MERGE unchanged); the q_load DEVICE class is needed
   for full imperfect-nest coverage but could ship second. Default
   proposal: v1 = CONTINUATION only (classifier emits device→gated
   fallback), v2 = DEVICE native. Smaller blast radius per step.
