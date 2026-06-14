# Holding Rule — Nested-Loop Extension (DESIGN v3)

Status: DESIGN (14jun26) — **AUDITED & CLOSED.** An 8-agent adversarial coverage
audit (ground-truth `triton-opt` DAG dumps over all 25 `insert_semas*` lits + 6
nested IR inputs) ran; every confirmed hole is fixed here (three-outcome
`holdKind`; If-finals via gateCrossing-is-For-only; depth is not a rule
parameter; (a)–(e) enforce confinement; edit-2 termination). No open design hole
remains. The only thing not provable on paper is the M3 *runtime* behavior of
downstream passes (standard for any GPU-compiler change, low-risk — plan §M3).
Companion plan: `fable/holding-rule-exnteison-nested-loop-plan-v3.md`.
Builds on: `fable/holding-regions-explained.md` (hold rule vs first-touch),
`fable/rule-v2-corpus-verification.md` §0 (the rule + 3 constraints),
`triton-solid-03.git/fable/codex-compositional-holding-rule-design.md` (the
compositional framing this realizes), and the captured dumps under
`fa-14jun26-v2/{regular,persistent}/`.

Scope: **solid-01 only**, this repo. Edits land in `InsertSemasSyncDag.cpp`
(stages 3–7, the behavioral core: edits 1–3 + the placement guard) and
`InsertSemas.h` (the `holdKind` enum on `Crossing`, the `firstTouchForced()`
helper). EMIT (`InsertSemasEmitIR.cpp`) is **verified unchanged** at M2, not
edited. NO downstream-pass edits (partition-loops, AssignStagePhase, LowerAref,
lowering) — a blocker there is STOP-and-report (campaign ground rule).

---

## 0. Problem (the persistent-FA stall, measured)

Same FA kernel, two shapes, both passing (exit 0), dumps in `fa-14jun26-v2/`:

- **regular** (`test_warp_specialize_attention_forward`) — the KV loop **is** the
  WS-tagged loop. Its in-loop ping-pong buffers (k, v, qk, p, …) are
  **point-of-use** native (the −47 TF fix). Only `acc` is gated.
- **persistent** (`test_warp_specialize_attention_persistent_forward`) — the
  persistent loop is WS-tagged; the KV loop is a **nested, non-WS-tagged** inner
  loop. The *same* in-loop buffers come out **gated**: root entry acquire +
  carrier `iter_arg` threaded through **both** loops + bottom regain inside the
  inner loop. They pay the exact stall the hold rule removes.

The two SYNC-DAG dumps differ by one move per buffer — writer acquire at TOP
(regular, native) vs BOTTOM (persistent, regain). Nesting flips it.

```
regular  group #4 (k/v)                     persistent group #2 (k)
scf.for(WS,tag=0) holdrule{pointofuse}      scf.for(WS,tag=0) holdrule{gated(nested-final)}
  a S1 {3}     ◀ acquire at TOP               scf.for      holdrule{gated(non-ws-loop)}
  W descriptor_load {3}                          W descriptor_load {2}
  r S0 {3}[tma_load]                              r S0 {2}[tma_load]
  a S0 {2}                                        a S0 {1}
  R tc_gen5_mma {2}                               R tc_gen5_mma {1}
  r S1 {2}[tc5mma]                                r S1 {1}[tc5mma]
  EXIT yield{native}                              a S1 {2}   ◀ acquire at BOTTOM (regain)
                                                  EXIT yield{c0: a S1}
                                                EXIT yield{c0: scf.for}  ◀ carrier through outer too
```

---

## 1. Why nesting flips it — the exact mechanism

`gateCrossing` (`InsertSemasSyncDag.cpp:1361-1467`) decides GATED (keep the
rotated first-touch device) vs UNGATED (native point-of-use). Its **first**
check is an unconditional veto:

```cpp
// InsertSemasSyncDag.cpp:1365-1366
if (!F->op || !gpu::hasWarpSpecializeTag(F->op))
    return gated("non-ws-loop");
```

The nested KV loop has no WS tag → vetoed → bottom regain. The outer WS loop
then can't go native either: its carrier's final access is the inner-for region
row, not a top-level bottom acquire:

```cpp
// InsertSemasSyncDag.cpp:1371-1373
Node *regain = c.finals.empty() ? nullptr : c.finals[0];
if (!regain || regain->kind != Node::Acquire)
    return gated(regain ? "nested-final" : "no-final");
```

### Which of the five checks survive a nested `F` (traced 14jun26)

| check | what it scans | nested-`F` verdict |
|---|---|---|
| veto (`non-ws-loop`) | `hasWarpSpecializeTag(F)` | **BREAKS** — the blanket veto (the trigger) |
| (a) regain is bottom Acquire | `finals[0]->kind` | survives |
| (b) trailing-use | `regain->next` within F's body | survives |
| (c) feed/entry acquire | **`F->prev` (siblings only)** | **BREAKS** — see below |
| (d) result-dead + parent-crossing | `F->next` + `F->parent->crossings` (1425-1431) | survives (parent check already anticipates nesting) |
| (e) one-round prefix | `F->children[0]` (F's body) | survives |

So **only the veto and check (c)** break. (a),(b),(d),(e) already work on a
nested `F`; (d)'s parent-crossing branch (`:1425-1431`) was even written
anticipating nesting.

### Why (c) breaks (the load-bearing detail)

The entry acquire for an inner-confined component is created by
`insertEntryAcquires` spliced **before the outermost loop that touches the
component** (`:1165-1173`, `spliceBefore(acq, rows.front())`). For the nested
case `rows.front()` is the **outer** loop, so the entry acquire lands in root,
above the outer loop. Check (c) scans `F->prev` (`:1390`) — the **inner** loop's
sibling chain — which never reaches it → `gated("no-entry-acquire")`. The entry
acquire is real and correct; it's just one nesting level up from where (c)
looks.

---

## 2. Goal / invariant

> **The inner loop's holding regions are computed IDENTICALLY whether or not the
> loop is nested, at any depth, for every component whose cycle is confined to
> that loop.** Nesting is transparent to the inner computation.

Equivalently (the user's "effective tt.ws"): a non-WS inner loop that lives in
WS scope is analyzed by the *existing* point-of-use logic exactly as if it were
the top-level WS loop; the enclosing loops emit nothing for inner-confined
components. This is the codex compositional model realized: *child solved from
fresh local state seeded at ENTER; parent sees only the child summary*
(codex Design Invariants 2, 3, 5).

---

## 3. The fix: one rule, realized by edits in SYNC-DAG

### The rule (uniform — not an `if` carve-out)

> A loop `F` goes **native** (point-of-use) for a component ⟺
> **(i)** `F` is in WS scope (`hasWarpSpecializeTag(outerWSLoop(F))`), **and**
> **(ii)** every enclosing region between `F` and that WS loop **`canDrop`** its
> device for the component (can emit nothing and let the child own the cycle).
>
> `canDrop(For) = true` — a `For` encloser drops (edit 3).
> `canDrop(If)  = false` *for now* — the `else` pass-through is unbuilt (§6).
> When a non-droppable region sits in the chain, `F` stays **gated**.

This is the codex composition collapsed to one predicate: *drop at every level
that can; go native at the level that owns the cycle.* It is deliberately **not**
an `if`-special-case — `canDrop` is a **capability**, and the only reason it is
`false` for `If` today is that we have not built If-drop. Flip `canDrop(If)` to
`true` when that lands (§6 / plan M4.2) and the rule widens with **zero** other
edits.

Two facts make this cheap and safe to realize:

- **The machinery is already nesting-capable.** `computeHoldRuleGates` (`:1469`)
  and `applyHoldRulePlacement` (`:1504`) already recurse into nested `For`s;
  `gateCrossing` runs **post-order** (children decided before parents, `:1471-
  1473`); EMIT's `renderRegion` (`InsertSemasEmitIR.cpp:729-793`) already renders
  a native component inside a nested body; `outerWSLoop` (`:1554-1561`) already
  walks to the WS ancestor.
- **Gating (clause ii false) IS today's behavior.** Today *every* nested loop is
  gated by the blanket veto (`:1365`) — that is the bug. The fix only *removes*
  gating where the chain is all-droppable; everywhere else emission is unchanged
  and **only the gate-reason label moves** (e.g. `non-ws-loop` → `if-encloser`).
  So the change can never regress a shape it does not optimize.

### Edit 1 — replace the blanket veto (`:1365-1366`) with the eligibility predicate

```cpp
auto forOp = F->op ? dyn_cast<scf::ForOp>(F->op) : nullptr;
if (!forOp || !gpu::hasWarpSpecializeTag(outerWSLoop(forOp)))
    return gated("non-ws-scope");          // clause (i): not in WS scope
if (!allEnclosersCanDrop(forOp))           // clause (ii): a non-dropper in the chain
    return gated("if-encloser");           // today: an scf.if between F and the WS loop
```

`allEnclosersCanDrop(F)` walks F's parent chain up to the WS loop and returns
false iff any ancestor fails `canDrop` (today: any `scf.if`). This is the single
place clause (ii) lives — it both *enables* the fix on `For`-chains and *preserves
today's gated emission* for `If`-enclosed shapes. (`canDrop` is one small helper,
not an inline `isa<scf::IfOp>` bail, so M4.2 is a one-line flip.)

### Edit 2 — make check (c) nesting-aware

When the backward `F->prev` scan exhausts the sibling chain without finding the
feed, **continue up through enclosing loops**. Explicit, terminating form (each
inner scan keeps today's body verbatim — same Acquire-of-comp match and the same
`entry-consumed` / `region-feed` / `release-feed` bails at `:1396-1402`):

```cpp
Node *feed = nullptr;
for (Node *cur = F; !feed && cur; cur = cur->parent) {   // climb to function root
  for (Node *m = cur->prev; m; m = m->prev) {            // today's scan body, unchanged
    if (m->kind == Acquire && semas[m->sema].component == comp) { feed = m; break; }
    if (m->kind == Access && nodeInvolvesComp(m, comp))   return gated("entry-consumed");
    if ((m->kind == For || m->kind == If) && crossesComp(m, comp)) return gated("region-feed");
    if (m->kind == Release && semas[m->sema].component == comp)    return gated("release-feed");
  }
  if (cur->parent && cur->parent->kind != For && cur->parent->kind != If) break; // root
}
```

Termination: the climb stops at the function chain (`parent` no longer a
region). The per-step bails are **per-component** (matched by
`semas[..].component == comp`), so reuse of the same semaphore *type* across
groups cannot cross-match. The first matching Acquire is the feed; the
`feed->sema == regain->sema` check (`:1406`) is unchanged. This makes
`c.holdFeedAcquire` the root entry acquire so `applyHoldRulePlacement` (`:1517`)
unlinks it — iteration-0 falls back to the semaphore's initial credit, exactly as
in the flat native case.

### Edit 3 — `canDrop(For)`: parent resolution `nested-final` → `nested-native`

This edit *is* `canDrop(For)`. When an enclosing `For`'s `finals[0]` is a nested
**`For`** (`:1371-1373`, today's `nested-final`), do not gate immediately.
Because gating is post-order, that child `For`'s crossing for the same component
is **already decided**. If it went **native**, the child owns the whole cycle, so
the parent has **no protocol** for it (`PASSTHROUGH_DROP`). If it stayed
**gated**, or if `finals[0]` is an `If`, keep today's `nested-final` device (a
genuine boundary crossing — e.g. acc).

**Checks (a)–(e) still run and enforce confinement (no separate precondition).**
After edit 1 relaxes the veto, a nested `For`'s own (a)–(e) decide native vs
gated exactly as for a flat loop. A component that is *not* inner-confined — an
access in an enclosing body (acc), a post-loop read, a trailing use — gates the
inner `For` via `result-consumed`/`trailing-use`/`region-crossing` and never goes
native. So "inner-confined" is not a new check we add; it is what (a)–(e) already
compute. (This is why the apparent "non-confined component goes native" hole does
not exist: the inner `For` gates itself.)

**Crossing state is THREE-valued, not a boolean (corrected 14jun26).** The
binary `holdGated` cannot represent the parent drop. There are three outcomes:

| `holdKind` | `finals[0]` | `applyHoldRulePlacement` | EMIT carrier |
|---|---|---|---|
| `GATED` | Acquire or child row | **no move** | threads carrier (today's device) |
| `POINT_OF_USE` | **Acquire** (bottom regain) | move regain → `holdFirstToucher`, unlink `holdFeedAcquire` | none |
| `PASSTHROUGH_DROP` | **child `For` row, that For is `!holdGated`** | **moves/unlinks NOTHING** | none |

**`finals[0]` that is an `If` (or a `For` that stayed gated) ⇒ keep `GATED`
(`nested-final`), never drop.** `computeHoldRuleGates` calls `gateCrossing` only
on `Node::For` (`:1474`), so an `If` crossing never gets a gate decision — "the
inner crossing went native" is only meaningful for a `For` child. A component
whose cycle escapes into an `scf.if` therefore always keeps the device. This is
consistent with `canDrop(If)=false`.

The parent of a native `For` child is `PASSTHROUGH_DROP`. It must NOT ride the existing
`!holdGated` placement path: `applyHoldRulePlacement` (`:1511-1517`)
unconditionally does `regain = c.finals[0]; unlinkFromChain(regain);
spliceBefore(regain, holdFirstToucher)` for any non-gated crossing — for a
`PASSTHROUGH_DROP` that would relocate/destroy the **child loop** (`finals[0]` is
the child `For`, `holdFirstToucher` is null). So `applyHoldRulePlacement` must
move a node **only** for `POINT_OF_USE`; the minimal guard is `if (c.holdGated ||
c.finals.empty() || c.finals[0]->kind != Node::Acquire) continue;`, but an
explicit `holdKind` is preferred (encodes intent, lets the M0 verifier assert
it). EMIT needs no change — its existing `holdGated==false` path threads no
carrier and recurses into the child, which is exactly the drop. The parent does
**not** drop the entry acquire; the **inner** crossing's `holdFeedAcquire` unlink
(edit 2) removes the root entry acquire.

Edit 1's clause (ii) guarantees the parent reached here can actually drop (it is
a `For`, since an `If` would have kept the child gated), so native-child and
`PASSTHROUGH_DROP`-parent are always consistent.

Result for an inner-confined component on a `For`-chain: inner loop = native
point-of-use; every enclosing loop = `PASSTHROUGH_DROP` (emit nothing). Identical
to the flat shape, wrapped.

---

## 4. What EMIT needs: ≈ nothing

- `renderRegion`/`renderChain` already recurse and render native components in
  nested bodies (`InsertSemasEmitIR.cpp:729-793`, `827-920`); the moved acquire
  renders at its `holdFirstToucher` inside the nested body with no extra logic.
- **`PASSTHROUGH_DROP` also needs no EMIT change**: EMIT threads the carrier only
  when `holdGated` (`:734-735,762-767,777-779`), so a `holdGated==false` parent
  threads nothing and recurses into the child — exactly the drop. The ONLY
  placement-stage change is the `applyHoldRulePlacement` guard (edit 3) so it
  does not try to move `PASSTHROUGH_DROP`'s non-Acquire `finals[0]`.
- There is **no** `crossCheckHoldRule` oracle to satisfy — it was deleted in M2
  (stale comment at `InsertSemasSyncDag.cpp:1349`; deletion noted at
  `InsertSemasEmitIR.cpp:1631-1635`). The SYNC-DAG gate is the single source of
  truth; the plan adds a verifier to compensate (plan M0).

So the behavioral change is **SYNC-DAG-confined**: edits 1–3, in `gateCrossing` +
`insertEntryAcquires` feed/cleanup + the parent-resolution branch, all in
`InsertSemasSyncDag.cpp`. The only other touch is the `InsertSemas.h` header (the
`holdKind` enum, the `firstTouchForced()` helper). EMIT is verified unchanged; no
downstream-pass edits.

---

## 5. Soundness

**Semantics are sound.** A semaphore lowers to a **single persistent mbarrier**
allocated once at `semaphore.create` (`LowerAref.cpp:158-174`), not loop-carried;
acquire/release mutate its phase in place (`WaitBarrier`/`ArriveBarrier`). The
hardware phase counter cycles across **every** back-edge — inner `(k,i)→(k,i+1)`
and outer `(k,i_last)→(k+1,0)` alike — with no SSA token. The initial credit
(`semaphore.create true/false` → `InitBarrier` pending_count,
`LowerAref.cpp:135-173`) seeds iteration-0 of the whole nest. So dropping the
SSA carrier and riding the counter across the outer back-edge is correct at the
hardware level.

**The narrowing insight (de-risks the change).** The *current* gated persistent
shape **already** places acquire/release/buffer ops inside the nested non-WS
inner loop, and the test **passes** (exit 0). And those ops are **partition-
stamped** (`ttg.partition = array<i32: …>` in `after-nvws-insert-semas.mlir`)
even though the loop has no WS tag. So "protocol ops inside a nested non-WS loop,
partition-stamped" is already exercised and handled by downstream passes today.
The native delta is strictly *smaller*: (a) drop the carrier `iter_arg`/`yield`
from both loops, (b) move one writer acquire bottom→point-of-use, (c) drop the
root entry acquire (ride the initial credit). It does **not** introduce a new
structural location for protocol ops.

**The residual risk to discharge (runtime, plan M3).** Removing the SSA carrier
means the cross-iteration ordering for the inner cycle is carried purely by the
mbarrier counter across **two** back-edges, where the flat case used one. The
downstream phase machinery must still assign correctly:

- `AssignStagePhase::computeSinglePhaseEligibility` (`AssignStagePhase.cpp:172-210`)
  finds the housing WS loop via `getOuterWSLoop` (`:187`) — which walks parents,
  so a native acquire in the nested loop should still resolve to the outer WS
  loop. **Verify** the phase progression it computes for the inner non-WS loop
  without the carrier.
- `SemaphorePendingCount` (`SemaphorePendingCount.cpp:85-165`) counts
  partitioned releases; our native releases remain partition-stamped, so the
  count should be unaffected. **Verify** pending counts match the gated baseline.
- `LowerAref` protocol rewrites copy partition/stage via `assignStageCluster`;
  partition-stamped native ops carry that metadata. **Verify** no op is silently
  skipped.

If any of these mishandles the carrier-free nested shape, that is a **STOP-and-
report blocker** (no downstream-pass edits in this change), per the campaign
rule and the documented oracle-gap risk.

---

## 6. Scope boundaries (non-goals)

- **OWNER-DAG policy is unchanged** (codex Invariant 1). First-toucher ownership
  is identical; only acquire/release *placement* moves.
- **Only inner-confined components go native.** A component with accesses in an
  enclosing loop body (acc: pre-inner write, post-inner read) keeps its boundary
  device — `gated(nested-final)` survives via edit 3's "inner stayed gated" arm,
  and acc's own gate reasons (entry-consumed / result-consumed) are untouched.
- **If-regions — two positions, only one is even a question** (both fall out of
  the §3 rule; neither is a hand-written carve-out):
  - *`if` INSIDE the inner loop* (loop body contains an `if`; codex Example 5):
    **already handled** by the existing if-machinery (conditionality cut,
    in-branch pairs, pass-through). The inner loop's check (e) sees the `if` as a
    body region-crossing and gates accordingly today; edit 3 does not touch it.
    NOT an issue.
  - *inner loop INSIDE an `if`-branch* (the `if` is the loop's parent / encloser):
    the only shape that exercises `canDrop(If)`. Absent from the corpus
    (persistent FA is `WS loop → KV for`, no `if` between), and it **compiles and
    runs today** all-gated. Because `canDrop(If)=false`, edit 1's clause (ii)
    keeps the inner loop **gated** here — *byte-identical to today*, only the
    gate-reason label changes (`non-ws-loop` → `if-encloser`). This is graceful
    degradation (loses the speedup for that one shape, never correctness);
    **aborting would be wrong** — it would regress a shape that compiles today.
    *Optimizing* it later = building If-drop (flip `canDrop(If)` to true + the
    `else` pass-through, codex Invariant 7) — the **unscheduled** follow-up,
    built only if a kernel has the shape AND needs the perf.
- **No combine re-enable, no token retention, no elision** (all parked
  elsewhere).
- **Depth is NOT a parameter of the rule.** There is one recursive rule applied
  at every `For`; nothing branches on nesting depth. `computeHoldRuleGates`
  recurses all levels; edit 1's `outerWSLoop`/`allEnclosersCanDrop` and edit 2's
  feed-scan walk to the WS ancestor / function root at *any* depth; edit 3 drops
  any `For` whose child is `!holdGated` (both `POINT_OF_USE` and
  `PASSTHROUGH_DROP` qualify), so the drop chains up the whole `For`-stack
  bottom-up (post-order) to whatever depth it goes. Depth-2, depth-3, depth-N take
  the identical path. There is no depth cap and no per-depth code.
  - *Depth is only a test-coverage fact, never a handling difference:* the
    kernels we have happen to span depths — persistent FA is depth-2, grouped GEMM
    (`@grouped_matmul_tma_kernel`) is depth-3 (buffers 2/3: `W`+`R` both in the
    innermost loop → innermost native, all enclosers drop; buffer-1 acc spans
    levels → stays gated). Both are just inputs to the same rule and both are in
    the M2/M3 gates. Structural-only nests (`@three_level_*`,
    `@quad_for_if_for_if_access`) have no active cycle and stay inert.
  - The one genuinely depth-touching concern is *downstream*, not in this pass:
    the mbarrier counter is per-allocation (handles N back-edges in HW), and
    `AssignStagePhase` walks parents — so phase assignment for an inner loop at any
    depth is one M3 runtime probe (grouped GEMM exercises three back-edges). Same
    probe, more back-edges; not a new code path.

---

## 7. Design invariants (must hold after the change)

1. OWNER-DAG policy unchanged.
2. A flat single WS loop produces byte-identical IR (the regular FA dump is a
   regression pin) — the change is inert when there is no WS-scope nesting.
3. Inner hold-rule placement is native/point-of-use whenever the *flat* hold
   rule would permit it for the same body (the transparency invariant).
4. Boundary device appears only when the holding rule genuinely requires it
   (multiplicity mismatch / enclosing-loop access) — never merely because of
   nesting.
5. Each enclosing level emits nothing for a component fully owned by a descendant
   (compositional pass-through).
6. No protocol op is emitted into a partition-less region (the AssignStagePhase
   precondition); native ops inherit the partition stamp from the WS context.
7. At most one carrier per semaphore group; no buffer use after release (the M0
   verifier asserts both).
8. **Graceful, never abort.** A shape the fix does not optimize (a non-droppable
   encloser — today any `scf.if` — in the chain) emits **byte-identically to
   today**; only the gate-reason label may change. The optimization narrows its
   own scope by gating; it never refuses to compile a shape that compiles today.

---

## 8. Reduced illustrated example (the persistent `k` buffer)

Input (after partitioning, before insert-semas):

```
for {tt.ws} {                 // persistent loop (WS-tagged)
  for {                       // KV loop (nested, non-WS)
    W k {2}                   // descriptor_load
    R k {1}                   // tc_gen5_mma
  }
}
```

**Today (gated):**

```
a S1 root ; entry                       // root entry acquire (created by insertEntryAcquires)
for {tt.ws} iter_args(%t = ...) {        // carrier through outer loop
  for iter_args(%u = %t) {               // carrier through inner loop
    W k {2}
    r S0 {2} [tma_load]
    a S0 {1}
    R k {1}
    r S1 {1} [tc5mma]
    a S1 {2}                             // bottom regain — the stall
    yield %u'
  }
  yield %t'
}
```

**After (nested-native), identical to the flat shape wrapped by the outer loop:**

```
for {tt.ws} {                            // outer: NOTHING for k
  for {                                  // inner: native point-of-use
    a S1 {2}                             // acquire at point of use (iter-0 = initial credit)
    W k {2}
    r S0 {2} [tma_load]
    a S0 {1}
    R k {1}
    r S1 {1} [tc5mma]
  }
}
```

Edit 1 lets the inner `for` past the veto; edit 2 finds the root entry acquire so
it is unlinked; edit 3 makes the outer `for` drop its carrier; the existing
`applyHoldRulePlacement` moves `a S1 {2}` to point-of-use; the existing
`renderRegion` emits the native body. No downstream pass changes.
