# Holding Rule — Uniform Hold Construction (DESIGN v5, rev 4)

Status: **DRAFT / IMPLEMENTATION-CANDIDATE** (15jun26, rev 4).

Changelog: **rev 1** — uniform builder as a behavior-preserving refactor (M4 region-
spanning held as a separate ask). **rev 2** — §2 made a total ordered procedure,
`Carrier` enum collapsed 5→3, `trailing-use` count fixed 11→12. **rev 3** — M4
folded in as the logical conclusion; §2.2 line numbers fixed; condition A region-tail
and §3.3 same-sema clarified; "no real-kernel change" grounded on condition D.
**rev 4 — NO FLAGS.** Removed `NVWS_REGION_SPANNING_HOLDS` entirely (the rule is
uniform — there is nothing to toggle) and deleted the pre-existing vestigial
`NVWS_FIRST_TOUCH`. Verification compares against `legacy ∪ the expected-flip matrix`
instead of an empty diff; staging is by implementation completeness, not a flag.

v5 unifies `InsertSemas`'s acquire/release placement into **one compositional
hold-builder**: **point-of-use is the rule, and the first-touch carrier device is a
*derived* outcome** that appears exactly where a hold's token crosses its region
boundary (live-in / live-out). The five special-case checks of `gateCrossing` and
the three-valued `holdKind` collapse into one recursive rule. There are **no
runtime flags**: the rule always produces its optimal output, and first-touch is
"no more, no less" than what the boundary liveness requires.

- v3 (`holding-rule-exnteison-nested-loop-design-v3.md`, LANDED): inner-confined
  components go native at any depth (three edits to the gated-by-default gate).
- v4 (`holding-rule-exnteison-nested-loop-design-v4.md`, DRAFT): proposed letting a
  native hold **span** a nested region row, but bolted it onto the same gate.
- **v5 (this doc):** the uniform rule whose natural output already includes v4's
  region-spanning. There is **no separate "v4 feature"** and **no toggle** —
  region-spanning point-of-use is simply what the rule produces for a transparent
  region; first-touch is what it produces when the token escapes the region.

**The arc (one project, staged by what's built — §8, plan):** build the uniform
builder (region-tail handling lands last, so those cases stay `CARRIER` until then)
→ a side-band oracle proves it reproduces legacy on every non-flip component and
matches the expected-flip matrix on the rest → switch emission, the easy transparent
cases go point-of-use → build the region-tail placement/verifier, those cases go
point-of-use. No flag flip anywhere; correctness rests on the §3 verifier + the lit
matrix + runtime gates.

**Honest value statement (no perf claim).** v5 produces **no measured speedup** on
any current real kernel (the only real-kernel `region-crossing` case,
`meta_fa_fwd`'s accumulator, is provably *not* hold-transparent and correctly
stays gated — §3). Its value is: one uniform rule instead of a gate pile;
**correctness by construction** (the §3 transparency proof guarantees no
race/hang where it goes native); and it handles any future region-spanning kernel
for free. If raw perf were the only goal, v5 would not be worth doing.

Companion plan: `fable/holding-rule-uniform-hold-construction-plan-v5.md`.

---

## 1. The artifact removed, and the goal

`gateCrossing` (`InsertSemasSyncDag.cpp:1423-1544`) is a **gated-by-default**
procedure: a `For` crossing is `GATED` (root entry acquire + carrier `iter_arg` +
bottom regain) unless five checks (a)–(e) all pass → `POINT_OF_USE`; a sixth
branch promotes the parent of a native child to `PASSTHROUGH_DROP`. The
three-valued `holdKind` (`InsertSemas.h:132-136`) is then special-cased by
placement (`applyHoldRulePlacement:1581`), verifier (`verifyHoldKinds:1864`), and
emitter (`!holdGated`).

The **goal** is the compositional rule: build a component's holds over each
region, place acquire-before-first / release-after-last, and let the first-touch
device appear *only* where the hold's token is live across the region boundary.
Point-of-use is the default; the carrier emerges, it is not the default.

---

## 2. The decision: one ordered procedure, three outcomes

For every `For` `F` and component `c`, in **post-order** (children decided before
parents — `computeHoldRuleGates:1546-1554` already recurses post-order, and the
parent outcome reads the child's, step A):

A hold for `c` in `F` is classified into exactly one of `{POINT_OF_USE, CARRIER,
CHILD_OWNS}`. It is **POINT_OF_USE** (no carrier; acquire rides the initial credit
on iteration 0 and the mbarrier phase counter across the back-edge) **iff every**
condition below holds. The order mirrors `gateCrossing` so equivalence is
mechanical; the first failing condition fixes the outcome and a `reason`.

```
PRE-1 scope:    F is in WS scope (outerWSLoop has the tag).
                FAIL -> no protocol (outside WS scope).               [non-ws-scope]
PRE-2 encloser: no If between F and the WS loop (allEnclosersCanDrop).
                FAIL -> CARRIER                                       [if-encloser]
A regain-tail:  the hold's last carrier-producer (finals[0]) is
                  - a bottom Acquire of c in F's body          -> continue to B
                  - a child For whose outcome is !holdGated
                    (POINT_OF_USE or CHILD_OWNS)               -> CHILD_OWNS
                  - absent                                      -> CARRIER  [no-final]
                  - a REGION (an If row, or a gated child For): the region-TAIL case
                      if §3-transparent: set regionTail=true, CONTINUE to B
                        (the region plays the regain's role; B/C/D/E still run —
                        POINT_OF_USE only if they ALSO pass. In particular D still
                        catches a loop-carried/post-loop accumulator -> CARRIER even
                        though the region is transparent. A does NOT short-circuit;
                        this is the "fall through to B/C/D" of §2.3/§8.)
                      else: CARRIER                               [region-not-transparent]
                    (region-tail placement is the last thing built — plan M3; until
                     it exists a transparent region-tail is conservatively CARRIER,
                     i.e. unchanged from today. No flag — staging by what's implemented.)
B no-trailing:  nothing touches c after that regain in F's body.
                FAIL -> CARRIER                                       [trailing-use]
C clean-feed:   the backward feed (climbing enclosers) is an Acquire of c with
                nothing consuming c between, AND feed.sema == regain.sema.
                FAIL -> CARRIER  [entry-consumed|region-feed|release-feed|
                                  no-entry-acquire|entry-sema-mismatch]
D dead-result:  c's loop result is not consumed after F (no live-out / seam).
                FAIL -> CARRIER                                       [result-consumed]
E clean-prefix: the carrier-fed prefix is one buffer-view + one release, view
                first, not a sourceful TMEM alloc; AND every region row in the
                prefix is HOLD-TRANSPARENT (§3).
                FAIL -> CARRIER  [no-buf|rel-count|rel-before-buf|
                                  prefix-not-buffer-view|region-not-transparent]
ALL PASS     -> POINT_OF_USE
```

**Three outcomes, mapping 1:1 to today's `holdKind`:** `POINT_OF_USE ↔
POINT_OF_USE`, `CARRIER ↔ GATED`, `CHILD_OWNS ↔ PASSTHROUGH_DROP`.

### 2.1 Liveness vs. non-liveness

Conditions **A, B, C, D** are **liveness** facts — where `c`'s token is born (A),
whether it is the hold's tail (B), whether it is fed from an enclosing scope (C:
live-in), whether it escapes forward (D: live-out). When any fails the token
crosses the region boundary, so a carrier must thread it: *this is why first-touch
appears exactly where needed* — the thesis, correct for A–D.

Conditions **PRE-2, the `sema`-match in C, and the validity part of E** are **not
liveness** — they are the §4 emission-shape / multiplicity / single-semaphore
validity. They gate `CARRIER` even when the token looks confined. So the decision
is the **conjunction**, not a liveness lookup.

### 2.2 Total reason→outcome map (the equivalence obligation)

Every `gateCrossing` reason, with the `Hold` outcome the uniform rule produces.
This table is the M1 oracle's expected-outcome map. (Line numbers cite the
`return gated("...")` / outcome-set statement in `InsertSemasSyncDag.cpp`, verified
by grep 15jun26.) The `region-crossing` (cond E) and `nested-final` (cond A) rows
are the only ones whose v5 outcome differs from today — they are the **expected
flip set** (the rest must match legacy exactly).

| gateCrossing reason (file:line of the statement) | v5 outcome | class |
|---|---|---|
| (all pass) `:1540` | **POINT_OF_USE** | confined cycle |
| `PASSTHROUGH_DROP` (child For !holdGated) `:1451` | **CHILD_OWNS** | child owns cycle |
| `non-ws-scope` `:1438` | (no protocol) | precondition |
| `if-encloser` `:1440` | **CARRIER** | live-in (conditional entry) |
| `no-final` `:1447,1456` | **CARRIER** | no carrier-producer |
| `trailing-use` `:1465,1467,1470` | **CARRIER** | liveness (tail extends) |
| `entry-consumed` `:1484` | **CARRIER** | live-in via prior access |
| `region-feed` `:1486` | **CARRIER** | live-in via region |
| `release-feed` `:1489` | **CARRIER** | live-in via release |
| `no-entry-acquire` `:1496` | **CARRIER** | no clean feed |
| `entry-sema-mismatch` `:1498` | **CARRIER** | §4 validity (sema bracket) |
| `result-consumed` `:1505` | **CARRIER** | liveness (live-out) |
| `no-buf` `:1535` | **CARRIER** | §4 validity (no view) |
| `rel-count` / `rel-before-buf` `:1536,1537` | **CARRIER** | §4 validity (multiplicity) |
| `prefix-not-buffer-view` `:1539` | **CARRIER** | §4 validity (sourceful alloc) |
| `nested-final` `:1456,1459` (region-TAIL, cond A), §3-transparent | **continue to B/C/D/E** → POINT_OF_USE (regionTail) iff B–E also pass, else CARRIER (e.g. D for accumulators) | flip (region-tail, M3) |
| `nested-final`, NOT transparent | **CARRIER** | `region-not-transparent` |
| `region-crossing` `:1521` (prefix region, cond E), §3-transparent | **POINT_OF_USE** | flip (easy, M2) |
| `region-crossing`, NOT transparent | **CARRIER** | `region-not-transparent` |

The map is total: every `gateCrossing` branch lands in exactly one outcome.
(`gateCrossing`'s pre-existing `first-touch-flag` early-return at `:1434-1435` —
the `NVWS_FIRST_TOUCH` knob — is **deleted** in M0 as vestigial; a uniform rule
never force-gates everything, so it has no row here.)

### 2.3 First-touch is *derived*, not toggled (no flag)

There is **no flag and no "cut"**: the rule always produces its optimal output, and
`CARRIER` (first-touch) is simply what the conjunction yields when a hold's token
crosses its region boundary. The two places a region row appears — condition E's
**prefix** region and condition A's **regain-tail** region — are allowed in a
point-of-use hold iff the region is **hold-transparent** (§3), and otherwise yield
`CARRIER` (`region-not-transparent`). First-touch comes out exactly where a region
has an access whose ownership escapes — "no more, no less."

**Allowing a transparent region does NOT force point-of-use** — it only lets the
decision continue through conditions B, C, D, which still apply. A genuine
loop-carried accumulator fails **D** (its result is live across the back-edge /
read post-loop) and stays `CARRIER` regardless of region transparency. This is the
robust reason the real-kernel accumulators do not change (§8): no flag protects
them — condition D does, by construction.

The only thing that "stages" the rollout is **implementation completeness**: the
region-tail placement (§6 hard subclass) is built last, so until then a transparent
region-tail is conservatively left `CARRIER` (unchanged from today). That is a
property of unfinished code, not a runtime switch.

---

## 3. Hold-transparency — the no-race-by-construction guarantee

A nested region row `R` (`For` or `If`) inside a hold owned by `H` for component
`c` is **hold-transparent** iff **all**:

1. `R` has a crossing for `c`, and the token entering `R` is owner `H`.
2. On **every dynamic path** through `R`, the token returned to the parent is
   owned by **partition `H`** (the *owner*, not a specific semaphore — different
   paths may return tokens sitting on different semaphores; see the note below):
   - **For:** the iter_arg is `H`; every iteration yields an `H`-owned token (the
     in-body regain re-acquires `H`); and the **zero-trip path returns the
     incoming `H` token** (standard `scf.for` semantics: a 0-trip loop yields its
     init args). `R`'s own crossing may be `CARRIER` internally — it may ping-pong
     `H→K→H` inside — provided it returns an `H`-owned token.
   - **If:** every branch yields an `H`-owned token; a branch that does not touch
     `c` yields the **incoming token unchanged** (pass-through, already emitted at
     `InsertSemasEmitIR.cpp:810-812`); a branch that touches `c` re-acquires `H`
     before yielding.

   **Owner, not semaphore (the `raw_if_token` subtlety).** Two branches may return
   `H`-owned tokens on *different* semaphores. `raw_if_token`'s then-branch returns
   an `S1`-acquired token (after its internal `S0→S1` ping-pong) while the else-branch
   passes the incoming `S3` token — **both owned by partition 0**. This is benign:
   the carrier is SSA def-use glue, and the post-region release names its own
   semaphore (`S2`) regardless of which token threads it. The semaphore-level
   invariant is clause 3 (the *parent* brackets feed=regain on one semaphore), not a
   cross-branch equality. **Empirical anchor:** this exact branch-threading
   (then→`S1`, else→pass) is already what today's *gated* `raw_if_token` emits and it
   passes; making it point-of-use removes only the outer carrier/regain, not the
   branch threading — so the transparency is established by the existing working
   emission. (Because this
   is the subtlest easy case, the M0 audit + the §3 verifier confirm it explicitly;
   if either rejects the cross-sema yield, `raw_if_token` simply stays `CARRIER` —
   it is not load-bearing.)
3. **Same-semaphore bracket (evaluated at the PARENT, not inside `R`):** after `R`,
   the parent's feeding acquire and its regain must ride the **same** semaphore
   (`feed.sema == regain.sema`, i.e. `gateCrossing` condition C does not raise
   `entry-sema-mismatch`). `R` MAY perform internal cross-semaphore handoffs — e.g.
   `raw_if_token`'s then-branch ping-pongs on `S0`/`S1` internally — provided the
   parent's hold still brackets on one semaphore (`raw_if_token`: feed and regain
   are both `S3`). A region is **not** transparent precisely when it forces the
   parent's `feed.sema != regain.sema` (`entry-sema-mismatch`) — which is the
   `meta_fa_fwd` accumulator.
4. `R` uses exactly one carrier slot for `c` (one semaphore group), and there is
   no buffer use after release on the returned token.

**This predicate is the guarantee.** Where it holds, point-of-use is sound by
construction: the parent owns `c` from its point-of-use acquire to its release;
`R` is just a transparent participant; iteration-0 rides the initial credit; the
phase counter advances only on actual arrive/wait (§9). Where it fails, the rule
keeps `CARRIER` — so it can never emit a racy point-of-use. The verifier (plan
M3) checks every clause; a violation is a hard compile error, not a runtime race.

Condition 3 is exactly the `meta_fa_fwd` accumulator: its inner region returns a
different-semaphore token (`entry-sema-mismatch`), so it is not transparent and
**correctly stays `CARRIER`** (and even if it were transparent, condition D would
hold it — §8).

---

## 4. Non-liveness validity conditions (preserved verbatim)

These gate `CARRIER` regardless of liveness; a builder that dropped any would
emit different IR or race. They are conditions of the §2 conjunction and clauses
of §3 transparency, copied from the current checks:

1. **`entry-sema-mismatch`** (C / §3.3, `:1497`). Hold rides one semaphore.
   (`meta_fa_fwd` acc, `sequential_ws_loops` loop A.)
2. **`rel-count` / `rel-before-buf`** (E, `:1536-1537`). One release before the
   re-acquire; view first. (`@release_multiplicity_unified_fanin_regain`.)
3. **`prefix-not-buffer-view`** (E, `:1538`, `prefixRowIsSingleBufferView:1410`).
   Sourceful TMEM alloc is not a bare view.
4. **`no-buf`** (E, `:1534`). A hold with a release but no buffer-view row.

---

## 5. Data model

Replace the per-`Crossing` gate fields (`holdKind`, `holdGated`,
`holdFirstToucher`, `holdFeedAcquire`) with one record per (For, component):

```cpp
struct Hold {
  CompId comp;
  Owner owner;
  SmallVector<Node *> rows;     // access + hold-transparent region rows, in order
  Node *entryAcquire;           // the acquire bounding the hold start (feed)
  Node *closingRelease;         // the release bounding the hold end
  Node *regain;                 // = finals[0]; the hold's last carrier-producer:
                                //   an Acquire, OR (region-spanning) a region row
  bool  regionTail;             // true: regain is a region row (the §6 hard path)
  enum Outcome { POINT_OF_USE, CARRIER, CHILD_OWNS } outcome;
  const char *reason;           // dump/analysis only: which §2 condition decided
};
```

**Three outcomes, matching the three emission-distinct shapes.** (A gated `For`
threads the carrier *both ways* unconditionally — iter_arg in `EmitIR:770`, yield
out `:782-783` — so there is no in-only/out-only emitted shape; live-in vs
live-out survives only as `reason`.) `regionTail` flags the §6 hard subclass for
placement. `Crossing.finals` is retained as the yield-wiring fact.

---

## 6. Emission — easy subclass is free, the region-tail needs placement/verifier work

- `renderChain` (`:827-920`): `Acquire` sets `rs.carrier[comp]` (`:870`); `Release`
  consumes it (`:878`). `renderRegion` (`:729-825`) threads a `CARRIER` For via
  iter_arg/yield (`:770,783`), erases for `POINT_OF_USE` (`:766,778`), and threads
  an `If` row via branch pass-through (`:807-818`, result→carrier at `:818`).

- **Easy subclass — region in the PREFIX, regain is a bottom `Acquire`
  (`region-crossing` with `finals[0]` an Acquire: `sample11`, `sample5` (For row);
  `raw_if_token` (If row, both branches owner-`H`: then re-acquires, else
  passes through)):** the parent is `POINT_OF_USE`; the prefix region stays
  `CARRIER`/threads a **hold-local** token (born at the moved point-of-use acquire,
  restored from the region result at `:783`/`:818`, consumed by the closing
  release). The existing `applyHoldRulePlacement` (move regain→holdFirstToucher,
  unlink feed) and emit machinery handle it with no change beyond the §2/§3 gate
  allowing the transparent region. **No new emit path.**

- **Hard subclass — `regionTail`, the region IS the hold's last carrier-producer
  (`nested-final` with `finals[0]` a transparent region: `conditional_multi_result`,
  the If-tail; or a transparent gated child `For` tail):** `finals[0]` is the
  region, not an Acquire. This needs:
  1. **placement:** do **not** move `finals[0]` (there is no bottom regain to
     relocate); place the point-of-use acquire before `holdFirstToucher` and let
     the closing release consume the region's **result** token.
  2. **verifier:** `verifyHoldKinds` must accept `POINT_OF_USE` with
     `regionTail==true` (`finals[0]` a region), not assert `finals[0]->kind ==
     Acquire`.
  3. **emit (verify, likely already correct):** the release reads `rs.carrier`,
     which the region sets to its result (`For:783` / `If:818`); this must be
     confirmed to consume the region result, never the pre-region token (the §10.2
     stale-token hazard).

  This is the only genuinely incremental work; it is **in scope** as the final
  milestone, not a separate ask.

---

## 7. For / If uniformity

- **Rows are uniform.** A `Hold.rows` element may be an access, a `For` row, or an
  `If` row; only the §3 token-threading proof is region-specific (`For`:
  iter_arg/yield/zero-trip; `If`: branch pass-through with the incoming-token
  fallback).
- **The native anchor is `For`-only by construction.** A native hold rides the
  loop initial credit + back-edge phase counter; an `scf.if` has no back-edge and
  no iteration, so there is no cycle for it to own. `computeHoldRuleGates:1551`
  runs the decision only on `Node::For`. An `If` participates only as a
  hold-transparent **row** (§3), a conditional **wrapper** for the initial acquire
  (`branch_local_init`), or an **encloser** that forces `CARRIER` (`allEnclosersCanDrop(If)=false`).

---

## 8. Verification arc (no flags; oracle vs `legacy ∪ expected-flip matrix`)

(Milestones are defined in the plan; this section uses the same M0–M3 labels.) The
uniform rule always produces its optimal output, so it does **not** reproduce today
byte-identically — it differs on exactly the enumerated transparent region cases
(the **expected flip set**). Verification therefore compares per component against
`legacy ∪ the flip matrix`, which keeps full bisectability without any flag:

0. **M0:** census (`N_before`), lit matrix, **delete the vestigial `NVWS_FIRST_TOUCH`**,
   transparency audit fixing the flip set.
1. **M1 side-band oracle (no emission change):** run the builder beside `gateCrossing`;
   assert per component that `mapOutcome(Hold.outcome)` equals **legacy** on every
   non-flip component and the **matrix's expected outcome** on the flip set. A
   mismatch on a non-flip component = plumbing bug; on a flip component = decision
   bug — so it still bisects cleanly. (Concept pre-verified 15jun26: the non-flip
   set matched legacy 127/127 and 137/137 in two sweeps.) The region-tail (hard)
   placement is not built yet, so transparent region-tail cases are conservatively
   `CARRIER` at M1 (== legacy).
2. **M2 switch emission + §3 verifier + delete legacy:** the easy transparent cases
   (`region-crossing`, bottom-Acquire regain) now **emit point-of-use** — their lits
   change to the audited flip-set goldens; **every non-flip lit stays byte-identical**
   (any unexpected change = STOP). Land the §3 verifier (a non-transparent region
   reaching `POINT_OF_USE` is a hard error). Measure `N_after`. Runtime gates.
3. **M3 region-tail (hard) subclass:** build the `regionTail` placement/verifier; the
   transparent region-tail cases (e.g. `conditional_multi_result`) flip; goldens +
   runtime gates.

(`regionTail`, the `Hold` struct, the §3 verifier, and the region-tail placement are
all **net-new** code these milestones add — the current
`gateCrossing`/`Crossing`/`applyHoldRulePlacement`/`verifyHoldKinds` implement none
of v5 yet; that is what M1–M3 build.)

**Expected flip set (v5 outcomes that DIFFER from today — fixed by the M0 audit; any
flip not on this list = §3 verifier bug = STOP).** Two gate reasons differ:
`region-crossing` (prefix region, cond E) and `nested-final` (region-tail, cond A).
A case flips to `POINT_OF_USE` iff its region row(s) are §3 hold-transparent **and**
B/C/D/E also pass.

| case | gate reason | regain | §3 transparent? | v5 outcome | subclass |
|---|---|---|---|---|---|
| `sample11`, `sample5` | region-crossing | bottom Acquire | yes (For, same owner) | **POINT_OF_USE** | easy (M2) |
| `raw_if_token` | region-crossing | bottom Acquire | yes — owner 0 on both branches (then→`S1`, else→pass `S3`; cross-sema but same owner, §3.2 note); **subtlest case, verifier-confirmed** | **POINT_OF_USE** | easy (M2) |
| `conditional_multi_result` | nested-final | the `scf.if` | yes (If-tail) | **POINT_OF_USE** | hard (M3) |
| `meta_fa_fwd` buf 4 & 5 | region-crossing | — | **no** (§3.3: inner forces parent feed≠regain) | **CARRIER** | unchanged |
| `sample1` (sibling regions) | region-crossing | — | **no** (§3.3: inner forces parent feed≠regain) | **CARRIER** | unchanged |
| every other `region-crossing`/`nested-final` case — incl. `insert_semas`, `if_split`, `local_cfg`, `nested_carrier`, `per_edge_tmem`, `if_encloser_inner_loop`, and the real-kernel accumulators (`attn_persistent`, `grouped_gemm`, `matmul_persistent`, `pfa`, `meta_fa_fwd`) | region-crossing / nested-final | — | **no** — see reason below; **M0 audit confirms each** | **CARRIER** | unchanged |

(The "24 total" in the inventory is the corpus-wide `nested-final` substring count,
not a per-row figure.)

**Why the real-kernel accumulators do not change (the robust reason).** They are
**not confined cycles**: an FA/GEMM accumulator is carried across the WS-loop
back-edge and read in the epilogue, so it fails **condition D** (`result-consumed` /
live-out). Allowing a transparent region only lets the decision fall through to
B/C/D — and **D still gates the accumulator** — so the rule *cannot* make it
point-of-use, independent of whether its inner region is §3-transparent. (Several,
e.g. `meta_fa_fwd`, also fail §3.3 directly.) The `if_encloser_inner_loop` case is
held by PRE-2 (`allEnclosersCanDrop(If)=false`). The **"no real-kernel change"**
claim thus rests on D/PRE-2, not on a fragile transparency judgement; the M0 audit
confirms each per-case. If any case were unexpectedly transparent *and* otherwise
point-of-use-eligible, it would flip — a *real* (still §3-safe) change to disclose
and add to the goldens, never a silent one.

Corpus reason inventory the M1 oracle reproduces (raw `grep -rho` substring counts
over `v4-corpus/{nvws,nested}`, 15jun26): `nested-final ×24, entry-consumed ×17,
trailing-use ×12, entry-sema-mismatch ×7, region-crossing ×5, result-consumed ×4,
prefix-not-buffer-view ×3, rel-count ×2, if-encloser ×1`; `region-feed ×0` here
(`×1` in sample2); `pointofuse ×56` (nvws 37 + nested 19), `passthrough-drop ×17`.
(These are raw substring counts over **both** subdirs; `insert_semas_meta_fa_fwd`
appears in both `nvws/` (lit) and `nested/` (real-kernel) dumps, so its entries —
incl. its 6 `pointofuse` — are counted in each. The point of the count is
reproducibility of the grep, not a per-function tally.)

---

## 9. Soundness

- **Liveness emergence is sound** (mbarrier is one persistent allocation; phase
  counter cycles per back-edge and only advances on actual arrive/wait — v3 §5,
  `LowerAref.cpp:135-174`). Point-of-use rides the initial credit; `CARRIER`
  threads the SSA token across the boundary. Both are in production (v3).
- **Region-spanning point-of-use is sound by the §3 proof** — the parent owns `c`
  end-to-end; the transparent region returns owner-`H` on every path (incl.
  zero-trip and `if` not-taken); same-semaphore bracket; one carrier slot. The
  verifier discharges every clause at compile time.
- **The §4 conditions are sound by preservation** (copied; the M1 oracle proves
  they were not weakened).
- **No downstream-pass change.** Non-flip components emit identical ops; only the
  transparent region cases change, and their emitted ops are the same protocol ops
  in point-of-use position (verified at M2/M3). A divergence in AssignStagePhase /
  SemaphorePendingCount / LowerAref is a STOP-and-report.

---

## 10. Counterexamples — what breaks if a guarantee is dropped

1. **Drop §3.3 (same-sema):** `meta_fa_fwd` acc made native; feed/regain semaphores
   differ → point-of-use acquire pairs with the wrong permit → race. (Why it stays
   `CARRIER`.)
2. **Drop §3.2 zero-trip/path proof:** a region whose not-taken/zero-trip path does
   not return `H` → the parent releases a token that never became valid → dropped
   sync / hang.
3. **Drop `rel-count`:** a fan-in hold with two releases before re-acquire, made
   point-of-use, fires a permit per partial release → early consumer permit.
4. **Hard subclass, stale release (§6.3):** if the closing release consumes the
   pre-region token instead of the region result → race. The verifier asserts the
   carrier equals the region result after a region row.

---

## 11. Scope boundaries (non-goals)

No perf claim. No downstream-pass edits. No If-encloser native (`canDrop(If)=false`).
No token retention, no combine, no elision. OWNER-DAG policy unchanged. **No runtime
flags** — `NVWS_REGION_SPANNING_HOLDS` is not introduced, and the pre-existing
`NVWS_FIRST_TOUCH` is deleted (M0). The uniform rule always emits its optimal output;
there is nothing to toggle. (If an emergency revert is ever needed, it is a code
revert, not an env var — consistent with "robust by construction, no emergency
valves.")

---

## 12. Design verdict

The thesis — *optimal point-of-use by construction; first-touch only where the
token crosses the region boundary* — is realized by the §2 procedure directly, with
**no flag and no cut**, and made safe by the §3 transparency proof. Region-spanning
is **not** a separable feature; it is what the uniform rule produces for a
transparent region. First-touch is "no more, no less" than what boundary liveness
(and the §4 validity conditions) require.

Two honest caveats, eyes open:
1. **No measured perf payoff** — the value is uniformity, correctness-by-
   construction, and future-proofing, not speed. (Established: the only real-kernel
   `region-crossing` case is non-transparent and stays gated.)
2. **The simplification is consolidation, not deletion** of conditions — the §2
   conjunction remains; what collapses is the 3-valued `holdKind` + its placement /
   verifier / parent-drop special-casing into one `Hold` outcome and one recursion,
   **plus** the removal of both flags (`NVWS_REGION_SPANNING_HOLDS` never added,
   `NVWS_FIRST_TOUCH` deleted). Worth it iff the plan-M2 moving-parts count drops.

Accept v5 iff: the M1 oracle is green corpus-wide against `legacy ∪ the flip matrix`
(non-flip set pre-verified), M2 keeps every non-flip lit byte-identical while the
easy transparent cases go point-of-use (with a measured moving-parts reduction and
the §3 verifier green), and M3 lands the region-tail subclass — all runtime gates
passing. Otherwise v3 stands.

---

## Appendix B. Retained-token redundant-semaphore reduction

This appendix records a post-v5 design amendment. It supersedes the §11 "No token
retention, no combine, no elision" scope line only for the specific same-owner
retained-token optimization described here. The uniform hold rule itself remains
unchanged: this is a SYNC-DAG construction improvement, not a hold-outcome rule.

### B.1 Ownership of the optimization

Redundant semaphore removal belongs in SYNC-DAG construction. If a handoff is
proved redundant, the builder must not create the edge. No edge means no
semaphore group, no acquire/release nodes, and nothing for the emitter to delete.
The emitter must remain a mechanical renderer of the finalized SYNC-DAG.

The professional implementation rule is therefore:

1. Decide redundancy while walking the token game in `InsertSemasSyncDag`.
2. Encode the result as DAG facts: either the normal handoff edge exists, or the
   row is allowed to use an owner-retained token.
3. In `InsertSemasEmitIR`, render those facts only. The emitter may select the
   retained token/semaphore that the DAG facts name, but it must not synthesize,
   delete, coalesce, or second-guess semaphores.

### B.2 Eligibility rule

A forced wave handoff from owner `P` back to owner `Q` may be suppressed iff all
of the following hold:

1. `Q` held the component earlier in the same chain, so a partition-local token
   for `Q` exists (`hadWave`).
2. The normal token game requires zero ordering edges for this touch: for a write,
   every conflicting holder is already transitively synchronized behind `Q`; for
   a read, `Q` is rereading a version it already holds (`retentionEligible`).
3. `Q`'s retained token has not already fed a release in the same block
   (`releasedWave`). This preserves the existing no-use-after-release verifier:
   no token may materialize a new `semaphore.buffer` after a release that consumes
   that token.
4. The retention scope is the current chain only. Region rows clear retention and
   rescope it from the regained carried owner; retained tokens do not cross a
   region boundary implicitly.

If any condition fails, the builder keeps the normal forced handoff. This is the
safe fallback and it is intentionally over-synchronizing rather than unsound.

### B.3 Required verifier and emitter facts

The SYNC verifier must accept an `Access` or `Release` owned by `Q` when `Q` has a
valid retained token for the component, even if the current carrier owner is a
different partition. That is not a relaxation of token locality: the retained
token is still a token acquired by `Q` and consumed by `Q`.

The emitter needs only enough state to render that DAG shape:

- current carrier token and its semaphore per component;
- owner key for the current carrier;
- retained `(component, owner) -> {token, semaphore}` facts;
- owner-keyed view cache, because a buffer view is bound to the token/partition
  that produced it.

When a retained owner needs a buffer or release, the emitter uses the retained
token and the semaphore that produced that token. This is mechanical rendering.
It is not an emitter-side semaphore optimization.

### B.4 Failure policy

If `verifyNoUseAfterRelease`, carrier-locality, or token/view-locality rejects a
retained-token shape, the fix belongs in SYNC-DAG construction. Do not weaken the
emitter verifier and do not add an emitter cleanup pass. The correct repair is to
make the retention proof more conservative so the builder emits the normal
semaphore whenever the retained token cannot be rendered safely.
