# Uniform Hold Construction — implementation plan (v5, rev 3)

Design: `fable/holding-rule-uniform-hold-construction-design-v5.md` (rev 4).
Builds on the LANDED v3 (`...-nested-loop-design-v3.md` + `-plan-v3.md`).

**Goal:** replace `gateCrossing`'s 5-check 3-valued-`holdKind` procedure with one
compositional `Hold` builder whose output is **optimal point-of-use at every nesting
level where it is provably safe** (for/if), with first-touch (`CARRIER`) as a
*derived* outcome wherever the token crosses a region boundary.

**NO FLAGS** (rev 3). There is no `NVWS_REGION_SPANNING_HOLDS` toggle — the rule is
uniform and always produces its optimal output — and the pre-existing
`NVWS_FIRST_TOUCH` knob is **deleted** (M0). Verification does not rely on a
byte-identical "scaffold" phase: the M1 oracle compares each component against
`legacy ∪ the expected-flip matrix` (design §2.2/§8). Staging of the easy vs.
region-tail subclasses is by **implementation completeness** (region-tail placement
lands last), not a runtime switch. An emergency revert, if ever needed, is a code
revert — not an env var.

**No perf claim.** Value = uniformity + correctness-by-construction (the §3
transparency proof) + future-proofing. The decision conditions are *consolidated*,
not deleted; v5 lands only if the M2 moving-parts count strictly drops.

---

## Blocker protocol (BINDING — campaign-standard)

1. **Verify it is REAL before stopping** — failing command/test with output, exact
   IR before/after, or re-derivation against ground truth (`triton-opt`, source
   file:line). Pattern-matching is not a blocker.
2. **STOP.** No workarounds, no scope expansion, no downstream-pass edits.
3. **Report:** root cause (file:line, verified); a reduced illustrated example
   (`op {p}`, aq/rel, holds/cuts) showing input / what the design prescribes /
   where it breaks; the minimal decision with options.

Special blocker classes:
- **M1 oracle divergence** — `mapOutcome(Hold.outcome)` ≠ the expected outcome
  (`legacy` on a non-flip component, or the matrix's value on a flip component) on
  any corpus function ⇒ STOP-report: the §2.2 map is incomplete. Do not silence the
  oracle.
- **Any downstream-pass mishandling** of a region-spanning point-of-use shape
  (AssignStagePhase / SemaphorePendingCount / LowerAref) ⇒ STOP-report (design
  forbids fixing it here).

---

## Ground rules (campaign-standard)

- DO NOT modify downstream passes. A blocker there = STOP, report.
- One variable at a time; baseline-first; no speculative edits.
- Build after C++ changes:
  `cd build/cmake.linux-x86_64-cpython-3.12/ && ninja triton triton-opt`.
- Do NOT run `pre-commit` / clang-format.
- Never hand-mutate test partition metadata.
- A/B discipline for any dump comparison: `TRITON_ALWAYS_COMPILE=1` + per-leg IR
  fingerprint.
- If a test runs minutes: `third_party/tlx/killgpu.sh`.

---

## Gates

### Compile-time
- **SACROSANCT HARD GATE — `test/TritonGPU/automatic-warp-specialization.mlir`
  passes green AND is NOT modified, every milestone.** (`-tritongpu-automatic-warp-
  specialization` runs insert-semas internally — `AutomaticWarpSpecialization.cpp:122`
  — and contains `@grouped_matmul_tma_kernel`.)
  ```
  timeout 60 <LLVM_LIT> -v test/TritonGPU/automatic-warp-specialization.mlir
  # <LLVM_LIT> = your llvm build's bin/llvm-lit (this tree:
  #   /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit)
  ```
  A broken CHECK = STOP-and-report (the change leaked), NEVER a test edit.
- **NON-FLIP EQUIVALENCE PIN (every milestone):** every `test/NVWS/insert_semas*.mlir`
  function **not** in the flip set, plus the **flat-FA** and **v3-native** dumps, is
  **byte-identical** to today. (*flat-FA* = the regular single-WS-loop FA dump
  `logs/fa-14jun26-v2/regular/`, the v3 Invariant-2 pin; *v3-native* = the persistent-FA
  / nested-native dumps v3 already made point-of-use, e.g. `insert_semas_nested_ws_inner_loop`
  and the persistent KV buffers — v5 must reproduce v3's native shape.) A diff on a
  non-flip function = STOP-and-report (a plumbing leak), not a regen.
  ```
  timeout 60 <LLVM_LIT> -v test/NVWS
  ```
- **FLIP-SET GOLDENS (M2–M3):** the **only** lits that change are the §3-transparent
  region cases the M0 audit finalized — at M2 (easy, regain = bottom Acquire):
  `sample11`, `sample5`, `raw_if_token`; at M3 (hard, `regionTail`):
  `conditional_multi_result`. Each is pinned as a golden whose diff is audited to be
  exactly "gated device → point-of-use." **Any OTHER lit changing = STOP-and-report**
  (a plumbing leak, or a non-transparent case mis-classified — a §3 verifier bug).
- **TRANSPARENCY VERIFIER (M3):** the §3 clauses are asserted in `verifySyncDag`;
  a non-transparent region reaching `POINT_OF_USE` is a hard compile error.

  DO NOT FUCKING FIX LIT TEST, THAT IS DONE AFTER RUNTIME GATES BELOW ARE PASSING!!!

### Runtime — in this order, at M2 (first behavior change) and re-run at M3
With `PYTHONPATH=.../python/`, `TRITON_ALWAYS_COMPILE=1`, `timeout 60`:
1. **4× warp-spec pytests** (`test_warp_specialization.py`):
   `test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`,
   `..._tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]`,
   `..._attention_forward[False-4-True-3-128-128-1024-1024]`,
   `..._attention_persistent_forward[True-8-True-3-128-128-1024-1024]` (arbiter).
2. **grouped GEMM** `::test_grouped_gemm[16-4096-8192-1024]` (depth-3 arbiter).
4. `timeout 60 python 06-fa.py`
5. `timeout 60 sh run_nvws.sh`
6. `timeout 60 sh run_nvws_1.sh`
All functionally pass (exit 0, no hang). TFLOPS captured, **not** a gate (no perf
claim). Any hang: `killgpu.sh`, capture IR, STOP.

---

## Milestones

### M0 — pre-work (NO behavior change)
1. **Moving-parts census (N_before).** A "moving part" = one logical decision point
   (a check/bail), one emit path, or one verifier arm — **not** lines of code or
   data-structure size (so the M2 `N_after < N_before` gate is about decision
   complexity, not byte count). Comment block enumerating the current decision's
   parts: WS-scope + `allEnclosersCanDrop` preconditions; checks (a)–(e)
   with each bail tag; helpers `regionResultConsumedAfter`,
   `prefixRowIsSingleBufferView`; the 3-valued `holdKind`; the
   `applyHoldRulePlacement` POINT_OF_USE-only guard; the `verifyHoldKinds` 3-arm
   shape check; the separate PASSTHROUGH_DROP branch. Fixes N_before. A part found
   later that is not listed = new information, STOP.
2. **Per-condition lit pins.** Confirm/author lits that each fail if one validity
   condition is dropped: `entry-sema-mismatch` (`meta_fa_fwd` acc, `nested_carrier`);
   `rel-count`/fan-in (`@release_multiplicity_unified_fanin_regain`);
   `prefix-not-buffer-view` (`root_entry_tmem`/`per_edge_tmem`); the §3
   transparency clauses (zero-trip, if pass-through, same-sema).

2a. **Transparency lit MATRIX (positive + negative, across owner permutations).**
   The §3 transparency predicate is the safety lynchpin and is **subtle** — note that
   in review, careful analysis repeatedly mis-called it by confusing a region's
   *last-access* owner with its *returned-token* owner (e.g. sample11's inner does
   `p1→p2` accesses but **returns** owner `p1` via its bottom re-acquire, `yield a
   S1 {1}` — it IS transparent). Conclusion: **the mechanical §3 verifier, not
   eyeball judgement, is the arbiter**, and the matrix below must be a hard gate.
   Each row is a required lit (✓ = already a corpus golden, verified from its dump):

   | shape (owner = partition) | corpus | gate today | v5 outcome | polarity | tests |
   |---|---|---|---|---|---|
   | `op1{1}` ; inner[`{1}`,`{2}`] ; `op4{2}` | sample11 ✓ | region-crossing | **POINT_OF_USE** (easy) | **positive** | §3.2 inner returns H=`{1}` (mirror in op4 owner) |
   | `op1{2}` ; inner[`{2}`,`{1}`] ; `op4{1}` | sample5 ✓ | region-crossing | **POINT_OF_USE** (easy) | **positive** | §3.2 owner-mirror of sample11 (H=`{2}`) |
   | `op1{1}` ; inner[`{2}`,`{1}`] ; `op4{1}` | sample10 ✓ | **trailing-use** | **CARRIER** | **negative** | cond B: `op4` reads after the regain → not the tail |
   | `meta_fa_fwd` acc (inner entry-sema-mismatch) | ✓ | region-crossing | **CARRIER** | **negative** | §3.3 inner forces parent feed≠regain |
   | real-kernel accumulators (carried across back-edge) | ✓ | nested-final | **CARRIER** | **negative** | cond D fall-through (loop-carried) |
   | `op1{1}` ; inner[`{2}`,`{1}`] ; `op4{2}` | NEW | (author) | audit | **negative-exp** | owner-change cut at `op1`/inner; not an H-spanning hold |
   | `store{1}` ; inner[`load{1}`,`store{2}`] ; `load{1}` (no middle, WS body) | NEW | (author) | audit | mixed | region-spanning at WS-body depth-1 + same-owner trailing read |
   | `store{1}` ; inner[`load{2}`] (cross-owner, store out / load in) | NEW | (author) | audit | mixed | cross-owner store/load span (cf. sample3 pointofuse/passthrough) |
   | `op1{1}` ; `if{1}{ op2{2} }` (If in WS body) | NEW | (author) | audit | **positive-exp** | If-as-prefix-region transparency (then touches, else pass-through) |
   | `for{1}{ if{1}{ op1{1}; op2{2} } }` (If inside inner loop) | NEW | (author) | audit | mixed | If-as-body-region inside a (sub)loop crossing |
   | `raw_if_token` (If prefix, cross-sema branch yields) | ✓ | region-crossing | **POINT_OF_USE** (easy) | **positive** (subtle) | §3.2 owner-not-semaphore (then→S1 / else→S3, both owner 0) |
   | `conditional_multi_result` (If-tail = regain) | ✓ | nested-final | **POINT_OF_USE** (hard, regionTail) | **positive** | §6 region-tail placement/verifier |
   | `producer{3}` ; inner[`read0{2}`,`read1{1}`,`store{1}`,`load{0}`] (fan-out) | NEW | (author) | audit | **negative-exp** | §3.4 one-carrier-slot / `rel-count` multiplicity (fan-out) |

   "audit"/"-exp" = the M0.4 audit + the §3 verifier determine the actual outcome;
   the expected polarity is the test's *intent*. The NEW rows must be authored as
   synthetic lits (each in `test/NVWS/`); the ✓ rows reuse existing goldens. A
   positive lit that the verifier leaves `CARRIER`, or a negative lit it makes
   `POINT_OF_USE`, is a STOP-and-report (the predicate is wrong). **The literal
   loop-shape IR for every row (S1–S10) is in Appendix A** — that is the exact set
   of shapes to verify.
3. **Corpus decision snapshot.** Re-dump every `insert_semas*.mlir` + the 6 real
   kernels (`NVWS_INSERT_SEMA_DUMP_DAG=1`); store per-component `holdrule{...}` as
   the M1 oracle table.
4. **Transparency-flip audit (fixes the M2/M3 golden set).** Classify EVERY
   `region-crossing` (cond E) and `nested-final` (cond A) case. A case flips only if
   its region is §3-transparent **and** the hold passes B/C/D/E (design §2: A does
   not short-circuit; allowing the region lets the decision fall through). The flip set = those
   that pass (expected: `sample11`, `sample5`, `raw_if_token` easy;
   `conditional_multi_result` hard). **Confirm every real-kernel accumulator
   (`attn_persistent`, `grouped_gemm`, `matmul_persistent`, `pfa`, `meta_fa_fwd`)
   stays `CARRIER`** — the robust reason is **condition D** (loop-carried/post-loop)
   and/or **PRE-2** (if-encloser), *independent of transparency* (design §8);
   `meta_fa_fwd` additionally fails §3.3. This is the "no real-kernel change" claim.
   If any case unexpectedly flips, disclose it as a real (still §3-safe) change and
   add it to the goldens.
5. **Delete the vestigial `NVWS_FIRST_TOUCH`** — the `firstTouchForced()`
   early-return at `gateCrossing:1434-1435` and its helper. A uniform rule never
   force-gates everything, so the knob is dead weight. It is default-off today, so
   removal does not change default behavior; verify the NVWS suite is byte-identical
   after removal.

### M1 — uniform builder + side-band oracle (compute + ASSERT, no emission change)
1. Implement the `Hold` builder (design §2–§5). It always computes the optimal
   outcome — **no toggle**. The region-tail (hard) placement is not built yet, so a
   transparent region-tail is conservatively left `CARRIER` (== legacy) until M3.
2. **In-pass oracle:** assert `mapOutcome(Hold.outcome)` equals the **expected**
   outcome per `For` crossing — `legacy` on every non-flip component, the **matrix**
   value (design §2.2) on each flip component. Mismatch = hard error naming
   function + component + both values. A non-flip mismatch is a plumbing bug; a flip
   mismatch is a decision bug — so it still bisects. Iterate until green corpus-wide.
   (Concept pre-verified for the non-flip set: 127/127, 137/137 — design §8.)
3. Print the uniform decision beside the legacy one in the DAG dump for audit.
4. Gates: AWS green; NVWS suite byte-identical (no emission change yet); oracle green.

### M2 — switch emission; easy subclass goes point-of-use; delete the legacy gate
1. Drive emission from `Hold.outcome` (design §6 — `outcome==POINT_OF_USE` where
   `!holdGated` is read today). The easy transparent cases (`region-crossing`,
   bottom-Acquire regain: `sample11`/`sample5`/`raw_if_token`) now **emit
   point-of-use**. Delete `gateCrossing`'s 5-check body, the 3-valued `holdKind`
   arms, and the placement/verifier special-cases the unified `Hold` subsumes. Keep
   the oracle as a permanent regression assert (builder-vs-emitted-shape).
2. Land the §3 **transparency verifier** in `verifySyncDag`: every clause asserted;
   a non-transparent region reaching `POINT_OF_USE` is a hard error.
3. **Goldens:** the flip-set easy lits (`sample11`/`sample5`/`raw_if_token`) change,
   each diff audited as "gated → point-of-use"; **every non-flip lit + flat-FA +
   v3-native byte-identical** (any other change = STOP). Confirm `meta_fa_fwd` acc +
   `sample1` + all real-kernel accumulators stay `CARRIER` via **condition D /
   PRE-2**, independent of transparency (design §8); `meta_fa_fwd` also §3.3.
4. **N_after census + net accounting.** If `N_after >= N_before` ⇒ STOP-and-report
   (the consolidation did not happen — do not land a same-size rewrite of a
   load-bearing pass with no perf upside).
5. **Runtime gates** (full list above). First behavior change — the
   `attention_persistent` arbiter and grouped GEMM are the correctness arbiters.

### M3 — region-TAIL subclass (the §6 hard path) — the last thing built
1. Placement: handle `regionTail` (`finals[0]` a region, e.g. the `scf.if` tail) —
   do not relocate it; place the point-of-use acquire before `holdFirstToucher`;
   closing release consumes the region **result** token.
2. Verifier: `verifyHoldKinds` accepts `POINT_OF_USE` with `regionTail==true`;
   stale-token assert (carrier after a region row equals the region result, §6.3).
3. Confirm emit threads the region result to the release (design §6.3).
4. Goldens: `conditional_multi_result` (and any transparent For-tail the M0 audit
   found) change to point-of-use; audited. **Non-flip lits stay byte-identical.**
   Runtime gates re-run. With M2 (easy) and M3 (region-tail) both green, the uniform
   rule is fully realized — no flag to flip, the optimal output simply ships.

---

## Risk register

| risk | likelihood | mitigation |
|---|---|---|
| §2.2 map incomplete → M1 oracle divergence | med | oracle vs `legacy ∪ matrix` corpus-wide at compile time; divergence = STOP-report |
| Builder emits different IR on a NON-flip component (plumbing bug) | med | M2 non-flip lits + flat-FA/v3 byte-identical; diff = STOP |
| A NON-transparent region made point-of-use → race | low | §3 verifier (M2) hard-errors it; per-condition lit pins; runtime arbiters |
| A lit OUTSIDE the finalized flip set changes | med | flip-set goldens; any extra change = STOP-report |
| "Simpler" illusory (count flat) | med | M2 N_before/N_after hard gate |
| Downstream pass mishandles point-of-use across the back-edge | low | runtime gates (persistent-FA + grouped GEMM arbiters); divergence = STOP-report |
| region-tail stale-token release | low | §6.3 emit assert + verifier (M3) |

---

## Approval checklist (plain language)

- [ ] M0 census lists every current moving part (N_before fixed);
      `NVWS_FIRST_TOUCH` deleted (NVWS suite byte-identical after removal).
- [ ] M0 transparency audit done: the flip set is **finalized** (expected:
      `sample11`/`sample5`/`raw_if_token` at M2, `conditional_multi_result` at M3).
      All real-kernel accumulators stay `CARRIER` via **condition D / PRE-2,
      independent of transparency** (the "no real-kernel change" claim);
      `meta_fa_fwd` additionally §3.3. A surprise flip is **disclosed for a go/no-go
      decision and, if approved, added to the flip set** — never silent, never an
      automatic code STOP.
- [ ] M1 oracle green on **every** corpus function (uniform == `legacy ∪ matrix`,
      no emission change).
- [ ] M2 easy flip set (`sample11`/`sample5`/`raw_if_token`) → point-of-use; §3
      verifier green; **every non-flip lit + flat-FA + v3-native byte-identical**
      (any other change = STOP); legacy gate deleted; `N_after < N_before`
      (consolidation real, measured); runtime gates pass.
- [ ] M3 region-tail: `conditional_multi_result` → point-of-use; verifier +
      stale-token assert green; non-flip lits byte-identical; runtime gates pass.
      Uniform rule fully realized — **no flag exists to flip**.
- [ ] No new runtime flag; `NVWS_FIRST_TOUCH` removed. No downstream-pass edits.
      AWS hard gate never modified.

---

## Appendix A — Verification loop shapes (S1–S10)

These are the **exact loop shapes** the implementer must verify, in the order
introduced. Owner `{n}` / `{pn}` = the executing partition. The M0.2a matrix
classifies each (positive = should flip to point-of-use; negative = must stay
gated/`CARRIER`). `✓ corpus` = reuse the named existing golden; `NEW` = author a
synthetic lit under `test/NVWS/`. Decisions cited were verified from the dumps.

**S1 — region-spanning, prefix For, `p1`-anchored** — `✓ corpus = sample11`
(`@case_d_p1_inner_p1_p2_p2`). gate today `region-crossing`; v5 **POINT_OF_USE
(easy)**; **positive**. (Inner returns owner `{1}` via `yield a S1 {1}` → transparent.)
```
for (tt.ws) {
  for middle {
    op1 {1}
    for inner { op2 {1}; op3 {2} }
    op4 {2}
  }
}
```

**S2 — owner-change at op1→inner** — `NEW`. Expected **negative** (the `{1}` access
and the `{2}`-entering inner are different owners → the hold cuts at `op1`; not an
H-spanning hold). Author + let the M0 audit/verifier confirm.
```
for (tt.ws) {
  for middle {
    op1 {1}
    for inner { op2 {2}; op3 {1} }
    op4 {2}
  }
}
```

**S3 — trailing read after the regain** — `✓ corpus = sample10`
(`@case_c_p1_inner_p2_p1_p1`). gate today `trailing-use`; v5 **CARRIER**
(condition B fails — `op4 {p1}` reads after the regain); **negative**.
```
for (tt.ws) {
  for middle {
    op1 {p1}
    for inner { op2 {p2}; op3 {p1} }
    op4 {p1}
  }
}
```

**S4 — region-spanning, prefix For, `p2`-anchored (owner-mirror of S1)** —
`✓ corpus = sample5` (`@outer_inner_same_owner_pairs`). gate today `region-crossing`;
v5 **POINT_OF_USE (easy)**; **positive**. (Inner returns owner `{2}` via
`yield a S1 {2}` → transparent.)
```
for (tt.ws) {
  for middle {
    op1 {p2}
    for inner { op2 {p2}; op3 {p1} }
    op4 {p1}
  }
}
```

**S5 — region-spanning at WS-body depth-1 (no middle, no op4)** — `NEW`. Tests the
region-spanning hold directly under the WS loop; **mixed** (positive iff the inner
returns the entry owner `{1}`; else the `{2}` escapes as live-out).
```
for {tt.ws} {
  op1 {1}
  for {1} { op2 {1}; op3 {2} }
}
```

**S6 — concrete store/load, WS-body depth-1, same-owner trailing read** — `NEW`.
**Mixed** (region-spanning positive vs. trailing-use on the final `load A {1}`).
```
for {tt.ws} {
  store A {1}
  for {1} { load A {1}; store A {2} }
  load A {1}
}
```

**S7 — cross-owner store-out / load-in** — `NEW` (cf. `sample3`
`@outer_access_inner_loop`, which is point-of-use/passthrough-drop today). Tests a
store `{1}` whose only in-loop consumer is a different owner `{2}`.
```
for {tt.ws} {
  store A {1}
  for {1} { load A {2} }
}
```

**S8 — If in the WS body (If-as-prefix-region)** — `NEW`. Expected **positive** iff
the `if` is §3-transparent (then-branch touches `{2}`, else-branch passes the
incoming `{1}` token through). The smaller cousin of `raw_if_token`.
```
for {tt.ws} {
  op1 {1}
  if {1} { op2 {2} }
}
```

**S9 — If inside the inner loop (If-as-body-region)** — `NEW`. Tests an `if` doing
`{1}→{2}` inside an inner loop's crossing (the codex "if inside the loop" case);
**mixed**, audit/verifier-determined.
```
for {tt.ws} {
  for {1} {
    if {1} { op1 {1}; op2 {2} }
  }
}
```

**S10 — fan-out / multi-consumer** — `NEW`. Expected **negative**: the inner has
multiple readers (`{2}`,`{1}`) plus `store{1}`/`load{0}` off one producer `{3}` →
hits `rel-count` / the §3.4 one-carrier-slot constraint (multiplicity), so it must
stay `CARRIER`.
```
for {tt.ws} {
  producer {3}
  for {3} { read0 {2}; read1 {1}; store {1}; load {0} }
}
```

**Negative anchors that must stay `CARRIER` (also pinned):** `meta_fa_fwd` acc
(inner `entry-sema-mismatch`, §3.3) and every real-kernel accumulator (loop-carried
→ condition D). **Positive hard-subclass:** `conditional_multi_result` (If-tail =
regain → POINT_OF_USE via `regionTail`, M3).
