# Holding Rule Nested-Loop Extension — implementation plan (v3)

Status: PLAN (14jun26). Design authority:
`fable/holding-rule-exnteison-nested-loop-design-v3.md` §3 (the `canDrop` rule +
edits 1–3) and
§5 (soundness). Nothing here introduces design content; divergence from the
design during implementation = STOP and report.

Scope: **solid-01 only**, branch
`egx/meta/sema10a-meta-new-sema-fresh-v5-fable-perf-1-codex`. triton-03.git
untouched.

What is being built: native point-of-use for the holding rule on **non-WS inner
loops in WS scope**, at **any nesting depth** (the rule is depth-generic), so the
persistent-FA inner KV loop matches the flat-FA inner loop (kills the nested
−47 TF stall). Edits land in `InsertSemasSyncDag.cpp` (stages 3–7, the behavioral
core) and `InsertSemas.h` (the `holdKind` enum on `Crossing`, the
`firstTouchForced()` helper). EMIT (`InsertSemasEmitIR.cpp`) is **verified
unchanged** at M2, not edited. **NO downstream-pass edits.**

**Completeness (read this — it is a FULL plan, not a partial one):** M0→M3 build
and verify the entire feature; M4 is *optional perf only*. **Nothing is deferred
because of a design gap.** The single deferred item (M4 — the If-*encloser* perf
optimization) is for a loop shape **no current kernel has**, and that shape is
already **correct** without it (it stays gated). Every other shape in the corpus
— flat, depth-2, depth-3 (grouped GEMM), if-in-loop, acc, fan-in, sequential — is
built and gated in M0→M3. The milestones are an **ordered build with a
verification checkpoint at each step**, NOT "build some, skip the rest": M0 lands
inert scaffolding, M1 computes-and-prints (proves the rule on the real corpus
before any emission), M2 turns it on, M3 confirms on hardware. A "STOP" in the
blocker protocol fires only on a *concretely demonstrated* divergence — it is the
safety latch of a complete plan, not an admission the plan is partial.

---

## Blocker protocol (BINDING — campaign-standard)

If implementation hits an apparent blocker:

1. **Verify it is REAL before stopping the clock.** A blocker must be shown
   mechanically: a failing command/test with output, the exact IR before/after,
   or a re-derivation against ground truth (triton-opt output, lowering source at
   file:line), independently checked. Theorizing/pattern-matching is NOT a
   blocker.
2. **STOP.** No workarounds, no scope expansion, no downstream-pass edits, no
   "temporary" hacks.
3. **Report three things:** root cause (file:line, verified); a reduced
   illustrated example in the established notation (`op {p}`, aq/rel, holds/cuts)
   showing input / what the design prescribes / where it breaks; and the minimal
   decision needed, with options.

Special blocker class for this change: **any mishandling by a downstream pass**
(partition-loops, AssignStagePhase, SemaphorePendingCount, LowerAref, lowering)
of the carrier-free nested-native shape is a STOP-and-report — the design forbids
fixing it here (design §5). This is the one residual risk; M3 is where it
surfaces.

---

## Ground rules (campaign-standard)

- DO NOT modify downstream passes. A blocker there = STOP, report.
- One variable at a time; baseline-first; no speculative edits.
- Build after C++ changes:
  `cd build/cmake.linux-x86_64-cpython-3.12/ && ninja triton triton-opt`.
- Do NOT run `pre-commit` / clang-format — irrelevant to this work.
- Never hand-mutate test partition metadata.
- A/B discipline for perf: `TRITON_ALWAYS_COMPILE=1` both legs + one-time per-leg
  IR fingerprint (env knobs are not in the cache key).
- If a test runs minutes: `third_party/tlx/killgpu.sh`.

---

## Gates

### Compile-time
- **SACROSANCT HARD GATE — `test/TritonGPU/automatic-warp-specialization.mlir`
  must pass green AND must NOT be modified, every milestone.** (Correction 14jun26: the
  `-tritongpu-automatic-warp-specialization` pass **runs insert-semas internally**
  — `AutomaticWarpSpecialization.cpp:122` calls `createNVWSInsertSemas(...)` — so
  this lit DOES exercise the change end-to-end, and it contains
  `@grouped_matmul_tma_kernel` (line 289). An earlier note that "AWS doesn't run
  insert-semas" was wrong; it grepped only the RUN-line megapass name, not its
  sub-passes.) Run it from the build folder:
  ```
  timeout 60 /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/TritonGPU/automatic-warp-specialization.mlir
  ```
  **The test file is NEVER modified — zero changes. No CHECK regeneration, no
  input edits, no skips, nothing.** It must pass with the file EXACTLY as
  committed, at every milestone. The implementation must make the *code* produce
  output that satisfies the existing CHECK lines; if the change would alter the
  AWS emission and break a CHECK (incl. `@grouped_matmul_tma_kernel`), that is a
  **STOP-and-report** signal (the change leaked where it must not), NEVER a license
  to touch the test.
- **`test/NVWS/insert_semas*.mlir` MAY FAIL** during this change and is
  **investigated ONLY after** the AWS gate AND all runtime gates are green. It is
  NOT a blocker on its own. (Full NVWS suite regen happens at M2:
  `timeout 60 .../llvm-lit -v test/NVWS`.) Note the WS-outer→non-WS-inner funcs in
  `insert_semas.mlir` (`@hoisted_alloc`, `@nested_loop_{yes,no}_double_buffer{,_scaled}`)
  have a crossing-less inner so edit 1 never fires — expected zero-diff there.
- **REGRESSION PIN — flat FA is byte-identical.** The regular-FA SYNC-DAG and
  emitted IR (`fa-14jun26-v2/regular/`) must not change at any milestone (design
  Invariant 2). Any diff there = STOP, the change leaked into the flat path.

### Runtime — in this order, at M3 (re-run after any later change)

Gate priority: **AWS compile gate green FIRST**, then all of these green, and ONLY
THEN is `insert_semas*` investigated. (pytest is run ONLY where a gate below calls
for it — per AGENTS.md "do not run pytest unless explicitly told"; these gates are
that explicit instruction.) **Every runtime gate runs with
`PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/`,
`TRITON_ALWAYS_COMPILE=1`, `timeout 60`.**

1. **4× warp-spec pytests** (file `python/test/unit/language/test_warp_specialization.py`):
   - `test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`
   - `test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]`
   - `test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]`
   - **`test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024]`**
     ← the target shape; this is the correctness arbiter for the whole change.
2. **grouped GEMM** (the depth-3 arbiter):
   `python/test/unit/language/test_warp_specialization.py::test_grouped_gemm[16-4096-8192-1024]`
3. **MOE**:
   `python/triton_kernels/tests/test_matmul.py::test_op[True-True-True-True-None-128-768-512-1024-plain-bfloat16-mxfloat4_e2m1-1-1-False-True-None-False-False-False-True-None]`
4. **`timeout 60 python 06-fa.py`** 
5. **`timeout 60 sh run_nvws.sh`**
6. **`timeout 60 sh run_nvws_1.sh`**
All three above must functionally pass. Just capture the TFLOPS


Any hang: `third_party/tlx/killgpu.sh`, capture IR, STOP.

---

## Milestones

### M0 — pre-work (NO behavior change)

1. **Golden for the nested point-of-use shape.** Add a lit
   `test/NVWS/insert_semas_nested_ws_inner_loop.mlir`: a WS-tagged outer loop
   wrapping a non-WS inner loop with one inner-confined ping-pong buffer (the
   persistent-`k` shape, design §8). Run the CURRENT pass, capture output as the
   golden, and record in the header that it pins the **gated (today's)** shape —
   carrier through both loops + bottom regain. M2 will regenerate this to the
   native shape; pinning the before-state makes the M2 diff the whole story.
2. **Capture the persistent-FA baseline** under
   `NVWS_INSERT_SEMA_DUMP_DAG=1` (already at `fa-14jun26-v2/persistent/`) — the
   per-buffer gate reasons are the before-fingerprint.
3. **Verifier asserts** (verifySyncDag layer): (a) at most one carrier slot per
   semaphore group; (b) no buffer use after release. Hard errors naming the
   component. (Compensates for the absent crossCheckHoldRule oracle.)
4. **Record (comment, no code)** the traced fact: of gateCrossing's 5 checks,
   only the `non-ws-loop` veto (`:1365`) and check (c) (`:1390`) break under
   nesting; (a),(b),(d),(e) survive (design §1 table). This is the answer key for
   M2 — any other check needing a nested edit = new information, STOP.
5. **canDrop(If) golden (audit gap — the true If-encloser target is ABSENT from
   the corpus).** Add a lit `test/NVWS/insert_semas_if_encloser_inner_loop.mlir`:
   `WS for → scf.if → non-WS for` with one inner-confined buffer. Capture today's
   all-gated output; this is the only artifact that lets M2 verify the C3 claim
   ("if-encloser stays byte-identical, only the label changes"). Without it that
   claim is untestable.
6. **Zero-diff coverage pins (audit MISS-1/4/7).** Beyond the flat-FA pin, pin
   these existing functions byte-identical (edits must not perturb them):
   `insert_semas.mlir` 5 nested funcs (crossing-less inner); the fan-in-under-
   nesting case `@release_multiplicity_unified_fanin_regain`
   (`insert_semas_release_count.mlir`) — an outer loop already native today with a
   multi-count (fan-in) release; and the write-only nested chains
   `@tmem_nested_linear_chain_no_outer_drain` (`insert_semas_per_edge_tmem.mlir`).
7. **Three-valued `holdKind`.** Land the enum (`GATED` / `POINT_OF_USE` /
   `PASSTHROUGH_DROP`, design §3 edit 3) and the `applyHoldRulePlacement` guard
   (`:1511` — move a node ONLY for `POINT_OF_USE`) in M0 as inert scaffolding
   (nothing sets `PASSTHROUGH_DROP` yet), so M2 only flips producers. Verifier
   asserts a `PASSTHROUGH_DROP`/native crossing never reaches the move path.
8. **`NVWS_FIRST_TOUCH=1` fallback flag (verified trivial — ~6 lines).** Add
   `firstTouchForced()` to `InsertSemas.h` (verbatim copy of `shouldDumpDag()`,
   bare `::getenv`, NOT in the cache key) and one early-return at the top of
   `gateCrossing` (right after it sets `holdGated=true; holdFirstToucher=nullptr`
   at `:1362-1364`): `if (firstTouchForced()) return gated("first-touch-flag");`.
   This forces EVERY crossing gated → the first-touch boundary-device shape
   everywhere, disabling both the existing flat hold rule and the new nested one
   in one chokepoint. `applyHoldRulePlacement` already skips gated (no-op); EMIT
   renders gated; OWNER-DAG (first-touch ownership) is unchanged — no other edit.
   **Default off = hold rule.** Rationale: `egx/...-fable-perf` is pure
   first-touch and a strict ancestor (current = it + 19 additive commits, gate
   logic net-new), so its placement IS the current `gated` path; forcing gated
   reproduces it. **Land this in M0 so M1/M2/M3 can use it as the first-touch
   baseline leg** (the A/B partner for hold-rule perf) and as a one-flip safety
   fallback if M3 ever goes red.
   - **Fidelity gate for the flag:** build `egx/...-fable-perf` once, run both it
     and `NVWS_FIRST_TOUCH=1` (current build) over the whole `insert_semas*` lit
     corpus with `TRITON_ALWAYS_COMPILE=1`, and require **byte-identical** output.
     Confirms the forced-gated path equals `...-fable-perf` (the 15 deletions in
     the 19 commits are comments/loop-restructure, expected to be inert here). Any
     diff = STOP, report (the gated path was perturbed by the hold-rule work).

Gate M0: AWS sacrosanct gate green + flat-FA pin + all the zero-diff pins above +
`insert_semas*` green with zero golden churn outside the new tests (M0 is inert —
scaffolding defaults off); build clean.

### M1 — SYNC-DAG side-band (compute + PRINT, no behavior change)

In `gateCrossing` / `computeHoldRuleGates`, compute the **nested** decision but
do not act on it:

1. Behind a dump-only path, evaluate edits 1–3 (design §3): for each non-WS inner
   `For` in WS scope, run (a)–(e) with the nesting-aware (c), and print the
   would-be outcome (native + the `holdFirstToucher`, or the real gate reason)
   AND, for each enclosing `For`, the would-be `nested-native` (drop) vs
   `nested-final` (keep). Emission stays on today's gated path.
2. Dump format: extend the SYNC dump's `holdrule{...}` line with the computed
   nested outcome, so goldens pin it before any behavior change.

Gate M1 (oracle — two arms): AWS sacrosanct gate green + flat-FA pin (M1 is
side-band: only dump lines added, no emission change → `insert_semas*` stays
green).
- **Positive arm:** on the persistent FA capture and the new lit, the computed
  native positions for an inner-confined loop equal the positions the flat hold
  rule produces for the same body (the flat regular-FA shape is the oracle —
  design Invariant 3).
- **Negated arm (audit H5):** components that are NOT inner-confined (acc-class —
  enclosing-body access, post-loop read, trailing use) must compute **stay-gated**
  with the SAME gate reason as today (`trailing-use` / `result-consumed` /
  `region-crossing`), and every enclosing loop whose child stayed gated must
  compute `nested-final` (not `nested-native`). This positively asserts the
  optimization fires ONLY where intended.
DAG-prefix goldens regenerated for the new dump lines only.

### M2 — enable native nested emission

1. Apply the design §3 rule (native ⟺ WS-scope ∧ all enclosers `canDrop`) via
   edits 1–3: **edit 1** replaces the blanket veto (`:1365-1366`) with the
   eligibility predicate — clause (i) `hasWarpSpecializeTag(outerWSLoop(F))` +
   clause (ii) `allEnclosersCanDrop(F)` (a small helper; `canDrop(For)=true`,
   `canDrop(If)=false` for now); **edit 2** makes (c)'s feed scan walk up
   enclosing loops; **edit 3** is `canDrop(For)` — the `nested-final` →
   `nested-native` parent-resolution, setting the child→native crossing to
   `POINT_OF_USE` and the enclosing crossing to `PASSTHROUGH_DROP` (the M0
   `holdKind` scaffolding). `applyHoldRulePlacement` (already recursive) moves the
   regain to point-of-use and unlinks the (now-found) root entry acquire ONLY for
   `POINT_OF_USE`; it must **skip** `PASSTHROUGH_DROP` (whose `finals[0]` is the
   child `For`, not an Acquire — moving it would relocate the child loop). EMIT
   threads no carrier for either (existing `holdGated==false` path). Net: inner
   native, enclosers emit nothing.
2. Verify EMIT needs no change: `renderRegion` already renders the native nested
   body (design §4). If it does NOT, STOP — that contradicts the trace and is new
   information.
   - **If-encloser → GATE via `canDrop(If)=false`, never abort.** Clause (ii)
     (`allEnclosersCanDrop`) returns false when an `scf.if` is in the chain, so
     the inner loop stays **gated** (reason `if-encloser`) — byte-identical to
     today's all-gated emission, which compiles and runs. Graceful degradation
     (loses the speedup for that one shape, never correctness); aborting would
     regress a shape that compiles today. This guard is what keeps edits 1+2 from
     making the inner loop native under a still-gated `if` (inconsistent carrier).
     (`if` INSIDE the inner loop is a different, already-handled shape, gated by
     check (e) region-crossing — not this guard.)
3. Golden regen. **Expected churn ONLY:**
   - persistent-FA inner-confined buffers (k, v, qk, p, …): lose carrier
     `iter_arg`s on both loops, lose the root entry acquire, lose the bottom
     regain; writer acquire moves to point-of-use. The new lit regenerates from
     gated → native.
   - **grouped GEMM (depth-3) buffers 2/3**: innermost `POINT_OF_USE`, middle + WS
     `PASSTHROUGH_DROP` (carriers dropped on all THREE levels). Buffer-1 (acc,
     spans innermost+middle) **zero diff** (stays gated). This is the depth-3
     compile evidence.
   - flat FA and all single-WS-loop lits: **zero diff** (the pin).
   - acc-class buffers (enclosing-loop accesses): **zero diff** (stay gated).
   Any other diff = STOP, the diff is information.

Gate M2: **AWS gate green with the test file UNMODIFIED** (never edited; if it
breaks, STOP — the change leaked into the AWS emission) + flat-FA pin + the M0
verifier asserts pass on the whole corpus. `insert_semas*` MAY churn/fail here —
regenerate the NVWS suite (`llvm-lit -v test/NVWS`) and **investigate it ONLY
after AWS green AND all M3 runtime gates green**, never as a blocker.

### M3 — runtime gates (perf runs ONLY here)

Run the runtime sequence in order (AWS compile gate green + UNMODIFIED first →
4× warp-spec pytests → grouped_gemm pytest → MOE → `06-fa.py` (must functionally
pass; absolute TFLOPS recorded, flag A/B direction gated) → run_nvws.sh →
run_nvws_1.sh — see the Gates §Runtime list for exact
commands, all with `PYTHONPATH=.../python/`, `TRITON_ALWAYS_COMPILE=1`,
`timeout 60`). The persistent-FA pytest is the correctness arbiter, grouped GEMM
the depth-3 arbiter. 06-fa's **absolute** TFLOPS is recorded, not threshold-gated;
but the `NVWS_FIRST_TOUCH=1` A/B **direction** (flag-on < flag-off) IS gated — see
the flag verification below. `insert_semas*` is investigated only after all the
correctness gates are green.

**M3 is the standard runtime confirmation that EVERY GPU-compiler change needs —
"does it run correctly on hardware" — not a doubt unique to this design.** It is
**low-risk** for concrete reasons: (1) the *gated* nested shape already runs
partition-stamped protocol ops inside the nested loop today and the persistent
test passes (exit 0), so "protocol ops in a nested loop" is already handled
downstream; (2) the mbarrier counter is per-allocation and cycles across any
number of back-edges in hardware; (3) the native change *removes* ops (the SSA
carrier), which is easier for downstream, not harder. It runs at BOTH depths:
persistent FA (two back-edges) and **grouped GEMM (three back-edges, the depth-3
arbiter — add its correctness run here if a pytest exists; else compile-golden +
run_nvws coverage)**. The only thing that cannot be proven statically is the
*runtime* behavior of the downstream passes we are forbidden to edit; M3 settles
it in minutes. If the carrier-free nested shape deadlocks, miscompiles, or hangs,
capture IR + the failing pass and STOP-report (do NOT edit downstream passes).
Specifically watch:
- `AssignStagePhase` phase assignment for the inner non-WS loop without a carrier;
- `SemaphorePendingCount` counts vs the gated baseline (should be unchanged —
  native releases stay partition-stamped);
- `LowerAref` not silently skipping any native op.

**`NVWS_FIRST_TOUCH=1` verification (the second A/B leg — proves the flag AND the
gain are real).** After M2, re-run with the flag on:
- **Correctness gate:** with `NVWS_FIRST_TOUCH=1`, `automatic-warp-specialization.mlir`
  (UNMODIFIED) AND every runtime gate (items 1–6, same
  `PYTHONPATH=.../python/`, `TRITON_ALWAYS_COMPILE=1`, `timeout 60`) must pass —
  the first-touch fallback is correct end-to-end (it is the `...-fable-perf`
  behavior).
- **Perf-direction gate:** `06-fa.py` TFLOPS with `NVWS_FIRST_TOUCH=1`
  (first-touch) must be **LOWER** than the default leg (hold rule). The flag
  reintroduces the nested stall, so this single comparison simultaneously proves
  (a) the flag really selects first-touch and (b) the hold-rule nested
  point-of-use actually delivered the gain. **If flag-on ≥ flag-off TFLOPS →
  STOP** (the flag is not selecting first-touch, or the gain did not materialize).

Gate M3: all green in order (default leg); persistent FA correct; **the
`NVWS_FIRST_TOUCH=1` verification passes — flag-on green on AWS + all runtime
gates, and flag-on 06-fa TFLOPS < flag-off (hold-rule) 06-fa TFLOPS**; perf —
persistent gains or is neutral, flat does not regress.

### M4 — follow-up queue (SEQUENCED AFTER M3, each its own ask)

- **M4.1 — NONE (no depth milestone): depth is handled generically in the core
  change (M2/M3).** The rule does not branch on depth (design §6), so M2's edits apply at
  every nesting level uniformly and M3 runs whatever depths the corpus contains.
  The gates already span depths: persistent FA (depth-2) and grouped GEMM
  (depth-3, `@grouped_matmul_tma_kernel`) are both in M2's compile golden and M3's
  runtime/probe — not because depth-3 is special, but because they are the kernels
  we have. A deeper (depth-4+) inner-confined kernel, if one ever appears, needs
  no new code — only its golden added to the gate.
- **M4.2 — If-encloser nested-native (UNSCHEDULED — build only if a kernel needs
  it).** This is the *inner-loop-inside-an-if-branch* shape ONLY; an `if` INSIDE
  the inner loop (codex Example 5) is already handled and untouched. The encloser
  shape is absent from the current corpus, and M2 already keeps it **correct** by
  GATING it (`canDrop(If)=false` → graceful degradation, today's behavior). M4.2
  is purely the *optimization* of that shape: **flip `canDrop(If)` to true** and
  build the `if`-drop (explicit pass-through on untaken/empty branches, codex
  Invariant 7). Because `canDrop` is a capability helper, the eligibility rule
  itself does not change — only the helper. Built only if a kernel has the shape
  AND needs the perf.
- **M4.3 — persistent GEMM / other nested WS kernels.** Sweep the corpus for
  other WS-scope nested loops and confirm the transparency invariant holds
  beyond FA.

---

## Risk register

1. **Downstream carrier-free handling** (the one real risk): AssignStagePhase /
   partition-loops / LowerAref must accept the inner cycle ordered purely by the
   mbarrier counter across two back-edges. Mitigation: the gated persistent shape
   already runs protocol ops (partition-stamped) inside the nested loop today, so
   the structural location is exercised; the delta is carrier removal only
   (design §5). Surfaces at M3; a break is STOP-report.
2. **Flat-path leakage**: an edit unintentionally changes the single-WS-loop
   path. Mitigation: the flat-FA byte-identical pin at every milestone.
3. **(c) over-reach**: the upward feed scan finds the *wrong* acquire (a
   different component's, or across a real boundary). Mitigation: the existing
   sema-match check (`:1406`) + the M0 verifier (one carrier per group); M1
   oracle equality against the flat shape.
4. **Oracle insensitivity** (recorded): lit+gates were historically blind to
   placement nuance. Mitigation: M1 prints the computed nested positions and pins
   them; M0 adds the use-after-release / single-carrier asserts.
5. **Emission races** (AutoWS): new nondeterminism shows as machine-dependent
   hangs — `run_nvws.sh` on this GPU is in-gate.

---

## Approval checklist (plain language)

- [ ] **1. Edits.** Three SYNC-DAG edits in `InsertSemasSyncDag.cpp` (relax veto
      via `outerWSLoop`; nesting-aware feed scan; `nested-final`→`nested-native`
      parent drop) + `InsertSemas.h` header (`holdKind` enum, `firstTouchForced`).
      EMIT verified-unchanged; no downstream-pass edits. (design §3)
- [ ] **2. Seatbelts first.** M0 (no behavior change): one lit pinning today's
      gated nested shape + the canDrop(If) golden; **three** verifier asserts
      (single-carrier, no-use-after-release, no-`PASSTHROUGH_DROP`-move); the
      zero-diff pins (5 nested funcs in insert_semas.mlir, fan-in, tmem chains);
      `holdKind` scaffolding + `NVWS_FIRST_TOUCH=1` (default off).
- [ ] **3. Answer key.** Correctness of the inner loop = the native positions
      equal the flat hold rule's for the same body, verified at M1 before any
      emission change; flat FA byte-identical throughout.
- [ ] **4. Tripwire.** Only pre-approved golden churn: persistent inner-confined
      buffers gated→native + the new lit. ANY other diff is brought as a diff,
      never regenerated over.
- [ ] **5. Gates.** Compile SACROSANCT HARD GATE:
      `test/TritonGPU/automatic-warp-specialization.mlir` green **with the file
      NEVER modified** (it runs insert-semas internally via
      `AutomaticWarpSpecialization.cpp:122`; if it breaks → STOP, do not edit it)
      + flat-FA pin. `insert_semas*.mlir` MAY fail and is investigated ONLY after
      AWS + all runtime gates green. Runtime in order at M3 (4 pytests,
      grouped_gemm, MOE, 06-fa.py, run_nvws.sh, run_nvws_1.sh); 
      persistent-FA the arbiter (persistent forward attention pytest)
      grouped GEMM the depth-3 arbiter (group gemm pytest)
- [ ] **6. Downstream blocker = STOP.** If the carrier-free nested shape breaks a
      downstream pass, capture + report; do not fix it in this change.
