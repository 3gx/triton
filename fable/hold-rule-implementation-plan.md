# THE HOLD RULE — native emission implementation plan

Status: COMPLETE (12jun26). M0-M2 committed (47a3e1653f); M3 runtime
gates PASSED in order: 4x warp-spec pytests, 2x moe block-128, run_nvws.sh
(no hang), 06-fa.py PARITY by same-session A/B — native 623/627/611 vs
fixup-era binary 625/610 TFLOPS back-to-back on the same GPU (the recorded
657-668 band is stale for that day's GPU state; the fixup-era binary does
not reach it either). Fingerprint: holdrule on the real 06-fa kernel = acc
gated(entry-consumed), qk/p/m_i/m_ij/k/v pointofuse — the validated
partition. M4 follow-ups (absorber flags first) remain queued.

Scope: **solid-01 only** (this repo, branch
`egx/meta/sema10a-meta-new-sema-fresh-v5-fable-perf-1`). triton-03.git is
untouched until the user asks.

Design authority: `fable/rule-v2-corpus-verification.md` §0 (the rule, the
three constraints, the all-cases table) and `fable/semas-report3.md`
Addendum B (what changes in the spec, what stays in force). Nothing in this
plan introduces design content; divergence from those two documents during
implementation = STOP and report.

## Blocker protocol (BINDING)

No deviation from this plan or the spec (Addendum B + verification doc §0).
If implementation hits an apparent blocker:

1. **Verify it is REAL before stopping the clock on it.** A blocker claim
   must be demonstrated mechanically, never inferred: a failing
   command/test with its output, the exact IR before/after, or a
   re-derivation against ground truth (`triton-opt` output, lowering
   source at file:line) — independently checked (second derivation or
   adversarial agent verification). Theorizing, assuming, hypothesizing,
   speculating, or pattern-matching from memory does NOT constitute a
   blocker. (Campaign precedent: the M0 STOP that was retracted as
   premature — that class of escalation is forbidden.)
2. **STOP.** No workarounds, no scope expansion, no edits to downstream
   passes, no "temporary" hacks while waiting.
3. **Report to the user with exactly three things:**
   - **root cause** — with file:line evidence, verified as in (1);
   - **a reduced illustrated example** — pseudo-IR in the established
     notation (`op {p}`, aq/rel, holds/cuts), showing the input, what the
     plan/spec prescribes, and precisely where it breaks — the §6/§7.6
     example style;
   - **what is needed** — the minimal decision or ruling required to
     proceed, with options if they exist.

## Ground rules (campaign-standard)

- DO NOT modify downstream passes (partition-loops, AssignStagePhase,
  LowerAref beyond what M4 names, lowering). A blocker there = STOP, report.
- One variable at a time; baseline-first; no speculative edits.
- Build after C++ changes: `cd build/cmake.linux-x86_64-cpython-3.12/ &&
  ninja triton triton-opt`.
- Formatter before any review: `pre-commit run --all`.
- Never hand-mutate test partition metadata.
- A/B discipline for any perf comparison: `TRITON_ALWAYS_COMPILE=1` both
  legs + one-time per-leg IR fingerprint (env knobs are not in the cache
  key).
- If a test runs minutes: `third_party/tlx/killgpu.sh`.

## What is being built

Native emission of the hold rule in InsertSemas (stages 2–4), replacing the
post-emission fixup. End state:

- `rewriteCarriedTokensToPointOfUse` (InsertSemasEmitIR.cpp:1491, invoked
  :1674), BOTH `hasOneUse` guards (:1510/:1516),
  `componentHasRides`/`tokenRides` (:1436/:1415), and the poison-slot
  machinery are **deleted**.
- The SYNC-DAG produces point-of-use shapes for ungated components and the
  §5.3 boundary-device shape for gated ones (Addendum B.2.1's gate),
  natively.
- Emitted IR for the current corpus is the fixup's output **minus the dead
  poison iter_arg slots** (the only sanctioned golden churn), plus the
  in-loop acquire-row position being native rather than moved.

## Gates (campaign-standard, per user spec)

### Compile-time

- **HARD GATE — `test/TritonGPU/automatic-warp-specialization.mlir` MUST
  pass with the test file UNMODIFIED.** Checked at every milestone.
- `test/NVWS/insert_semas*.mlir` and other insert-semas lits are NOT
  gates: they may fail/churn during bring-up (M1/M2) and are regenerated
  at M2 (churn budget below). Suite must be green by end of M2.

### Runtime — in this order, at M3 (and re-run after any later change)

1. **4x warp-spec pytests** —
   `python/test/unit/language/test_warp_specialization.py`
   (`TRITON_ALWAYS_COMPILE=1`, 60s timeouts):
   - `test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`
   - `test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]`
   - `test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]`
   - `test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024]`
2. **2x moe tests** — the previously-failing pair from the 12jun26
   regression (fail-moe.log):
   `pytest python/triton_kernels/tests/test_matmul.py::test_op[True-False-False-False-None-128-720-576-768-plain-bfloat16-mxfloat4_e2m1-1-1-False-True-None-False-False-False-True-None]`
   `pytest python/triton_kernels/tests/test_matmul.py::test_op[True-False-False-False-None-128-720-576-768-batched-bfloat16-mxfloat4_e2m1-10-1-False-True-None-False-False-False-True-None]`
3. **`sh run_nvws.sh`** (note: run_nvws.sh, not run_nvws_1.sh).
4. **`python 06-fa.py`** — perf parity vs current tip (657–668 TF band),
   A/B discipline (`TRITON_ALWAYS_COMPILE=1`, per-leg IR fingerprint).

Any hang: `third_party/tlx/killgpu.sh`, capture IR, STOP.

## Milestones

### M0 — pre-work (no behavior change)

1. **E4 golden**: add `test/NVWS/insert_semas_sequential_ws_loops.mlir` —
   two sequential ws-tagged loops sharing one buffer (shape per
   verification doc E4). Run the CURRENT pass, capture its output as the
   golden, and record in the test header what continuation behavior it
   pins (same-semaphore protocol across the seam; no boundary device).
   If the current pass emits something else, STOP and report — that is
   new information about the seam, not a bug to fix silently.
2. **Verifier asserts** (InsertSemas verifier layer, spec §7 + B.3):
   (a) at most one carrier token slot per semaphore group;
   (b) no token has a buffer use after its release use.
   Both as hard errors with the diagnostic naming the component.
3. Record (comment, no code): protocol ops must not be emitted inside
   partition-less `scf.if` (AssignStagePhase `assignStateInIfOp` asserts
   partition metadata); root-side anchors hoist above predicates (current
   emitter already behaves this way).

Gate M0: compile-time hard gate + full lit suite green with zero golden
churn outside the new test; build clean.

### M1 — SYNC-DAG: the gate and the point-of-use handback

In InsertSemasSyncDag.cpp (+ OwnerDag fact plumbing if needed):

1. Compute per (For region, component) the **boundary-device gate**
   (Addendum B.2.1): does any boundary cut pair a once-per-run outside end
   against a per-iteration inside end? (Once-events: pre/post-region
   accesses incl. nested-region multiplicity mismatches. Same-cycle
   outside ends — sequential-loop seams — do NOT fire it.)
2. Gated components: unchanged today's construction (entry acquire,
   carrier, regain anchored at EXIT).
3. Ungated components: compute the new handback destination — the carried
   owner's next REAL access (the next-iteration first toucher) — and the
   resulting point-of-use acquire/release positions. **M1 is SIDE-BAND
   ONLY: node injection and emission stay unchanged** (the rotated shape +
   in-tree fixup keep producing the IR); the new positions are computed
   and PRINTED, not acted on. Behavior switches at M2.
4. Dump format: SEMAS/SYNC-DAG dumps print, per (region, component), the
   gate decision AND the computed point-of-use positions, so goldens pin
   both before any behavior change.

Gate M1 (oracle): compile-time hard gate, plus: on every lit input, the
dump's COMPUTED point-of-use positions for ungated components equal the
positions the in-tree fixup produces in IR today (the fixup is the oracle;
it stays in-tree and active through M1 — IR output is byte-identical at
M1, only dumps grow). DAG-prefix goldens regenerated for the new dump
lines by milestone end.

### M2 — EMIT-IR: native rendering, fixup deletion

1. Switch the SYNC-DAG node injection to the M1-computed positions for
   ungated components (no ENTER/EXIT augmentation rows; merged wrap holds
   keep carrier + entry-as-init) and emit natively: in-loop acquire at its
   DAG position; token born and dead in-body (no iter_arg, no poison, no
   entry acquire).
2. Delete the fixup wholesale: `rewriteCarriedTokensToPointOfUse`
   (InsertSemasEmitIR.cpp:1491, invoked :1674), BOTH hasOneUse guards
   (:1510 bottomAcq and :1516 entryAcq), `componentHasRides` (:1436) /
   `tokenRides` (:1415), and the poison-slot creation/cleanup for
   converted slots.
3. Golden regen for IR goldens. Expected churn ONLY: dead poison iter_arg
   slots disappear (loop signatures shorten); acquire positions identical
   to the fixup's post-rewrite output. Any other diff = STOP, diff is
   information.
   Regen tool quirks (recorded campaign knowledge): `fable/
   gen_insert_semas_checks.py <test> <triton-opt> --apply`; it DROPS the
   mixed_overlap CHECK-NOT lines (buffer.id=522) and BREAKS the tmem_alias
   META prefix — restore both by hand; DAG-prefix sections are not
   regenerated by the tool.

Gate M2: compile-time hard gate + full lit suite green (churn within
budget).

### M3 — runtime gates

Run the runtime gate sequence exactly as specified in the Gates section,
in order: 4x pytest -> 2x moe -> `sh run_nvws.sh` -> 06-fa.py.
**Perf runs only here, nothing earlier** (user rule: no perf tests unless
explicitly part of a gate).

Gate M3: all green in order; perf within band.

### M4 — follow-up queue, SEQUENCED AFTER M3 (not dropped; not in this
### change, so perf effects stay attributable one variable at a time)

In planned order, each as its own ask once M3 is green:

- **M4.1 — `ABSORBER_IN_ROOT_P0=1` / `ABSORBER_IN_ROOT_ALL=1` flags +
  A/B** (FIRST follow-up; agreed experiment).
- M4.2 — Elision peephole in lower-semaphore (root-consumer exit pairs;
  `[none]` absorber elision with needFence residue) — verification doc
  §7.4 table.
  M4.1 prerequisites recorded: AssignStagePhase last-pid-wins fix
  (:1108-1137), Exit-group semaphore adoption path
  (InsertSemasSyncDag.cpp:900-953), final-iteration peel of the bottom
  acquire (count ledger: verification doc E2). Both flags into
  `include/triton/Tools/Sys/GetEnv.hpp`.
- M4.3 — Combine re-enable (separate thread: needs
  `fable/attr-less-acquire-release-handoff.md` §5.1 tolerance guards).

## Risk register

1. **Stale-dump trap**: `logs/fa-11jun26-v3-root/passes-03/048` predates
   the fixup commits — never use it as ground truth; re-run the pass.
2. **Oracle insensitivity** (recorded memory): lit+gates were blind to
   repair-pass behavior before; M1's DAG-dump gate (positions printed and
   pinned) is the mitigation here.
3. **Emission races** (AutoWS): any new nondeterminism shows up as
   machine-dependent hangs — run_nvws_1.sh on THIS GPU is in-gate.
4. **Token retention remains PARKED** (spec Addendum A); nothing in this
   plan touches it.

## Approval checklist for the user (plain language)

- [x] **1. Spec.** The spec update (Addendum B) correctly encodes the
      design we discussed. USER DELEGATES correctness here to the
      implementer's due diligence; the binding contract is
      rule-v2-corpus-verification.md §0 + the enforcement in item 3.
- [x] **2. Seatbelts first.** (USER APPROVED) Before any change: one new lit test for the
      two-sequential-loops case (locks in today's behavior), and two
      loud-abort safety checks for the two known-dangerous shapes. No
      behavior change.
- [x] **3. Answer key.** (implementer's discipline, owned) Correctness of the rewrite is defined as:
      the new code places every aq/rel IDENTICALLY to today's pass output
      (the output we validated and perf-tested), verified by mechanical
      comparison on all lit files + the FA kernel BEFORE the old code is
      deleted. A mismatch = my bug, caught at compile time, not on GPU.
- [x] **4. Tripwire.** (implementer's discipline, owned) Golden regeneration ACCEPTS whatever the pass
      emits — a bug at regen time gets baked into the tests and the suite
      goes green over broken output (the recorded oracle-gap failure
      class). Protection: the only pre-approved golden change is the dead
      poison iter_args disappearing; ANY other diff is brought to the
      user as a diff, never regenerated over.
- [x] **5. Gates.** Compile: automatic-warp-specialization.mlir UNMODIFIED
      must pass at every step. Runtime, in order: 4x named pytest, 2x
      named moe, sh run_nvws.sh, 06-fa.py (657–668 band). APPROVED.
- [x] **6. Sequenced, not dropped.** (settled: flags = M4.1 after M3) The ABSORBER_IN_ROOT flags + A/B
      experiment IS happening — as M4.1, the first follow-up after M3 is
      green, so its perf effect is attributable (one variable at a time).
      Likewise the elision peephole (M4.2) and combine re-enable (M4.3).
      None of them lands inside THIS change.
