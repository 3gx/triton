# Commit-5 Mechanical Emitter — Design & Implementation Plan

Status: proposal (awaiting approval before any code is written)
Scope: `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp` only,
plus its pass-option TableGen (`Passes.td`) and NVWS lit tests under `test/NVWS/`.

This document contains only claims that were verified by reading the current
source at HEAD `393b0f8b03`. Every code reference is a line number in
`InsertSemas.cpp` unless stated otherwise. Items that still require empirical
confirmation are listed explicitly in §9 and are *not* assumed elsewhere.

---

## 1. Purpose

Replace the commit-5 semaphore-emission core with a **mechanical** emitter:
given a valid sync DAG, emit `nvws.semaphore.create` / `.acquire` / `.release`
ops deterministically, with **no analysis or heuristics at emit time**.

Two hard requirements:

1. **DAG-source interchangeability.** The emitter must produce correct IR from
   either the unoptimized **RAW-SYNC-DAG** (per-edge) or the optimized
   **OPT-SYNC-DAG** (merged groups). The optimization must be
   correctness-preserving: turning it off may change *how many* semaphores are
   emitted, never *whether the program is correct*.
2. **No heuristics in the emitter.** Every emit-time *semantic decision* —
   whether an action exists, which semaphore it uses, the released bit — must be a
   lookup of a fact already settled in the DAG, or a deterministic partition of
   DAG data by an intrinsic edge property. No op-kind sniffing, no positional
   parity, no event-graph re-walks. (Deterministically placing an
   already-decided action at a canonical slot computed from its stamped endpoint
   is *materialization*, not a decision, and is permitted — see §5 stage 3 and the
   §6.7 placement policy. The rule here governs semantic decisions only.)

A pass flag must allow disabling the OPT stage, and a lit test must prove the
emitter produces correct IR on the resulting RAW DAG.

### Hard rule — escalate, never add a heuristic

If, during implementation, a case is hit that appears to **require** a heuristic
to emit correctly — i.e. a *semantic decision* (whether an action exists, which
semaphore it uses, the released bit) that cannot be expressed as (a) a lookup of
a fact already settled in the DAG, or (b) a deterministic partition of DAG data
by an intrinsic edge property — then:

1. **STOP IMMEDIATELY.** Do not add the heuristic. Do not add a "temporary"
   special case, fallback, or `if (op-kind …)` workaround.
2. **Root-cause it.** The required information is missing from the DAG (a
   commit-1…4 gap) or the model in §2 is incomplete. Determine which.
3. **Build a minimal reproducing example** (smallest IR / DAG that forces the
   decision) and capture the exact point where a heuristic would be needed.
4. **Report to the user** with the example and the root cause, and wait.

A heuristic in the emitter is, by definition of this plan, a defect — the same
class of defect this rewrite exists to remove. The correct fix for a missing
fact is to **settle it upstream in the DAG** (and surface it as a new DAG
field), not to re-derive it at emit time. That decision is the user's to make,
not the implementer's.

### Hard rule — escalate any M1/M2/M3 violation, never improvise

If a valid input DAG produces output that violates a model invariant from §2 —
**M1** (exactly one `is_released = true` semaphore per resource that has a seed;
zero for an edge-free resource), **M2** (every
other semaphore created unreleased and explicitly released), or **M3** (each
semaphore acquired by root and at most one concrete owner) — treat it identically
to a heuristic block:

1. **STOP IMMEDIATELY.** Do not "fix it up" with a special case, a merge, a
   split, a re-key, or any emit-time adjustment to make the invariant hold.
2. **Root-cause it.** A violation means either the DAG is invalid (an upstream
   commit-1…4 bug) or the model in §2 is wrong/incomplete. Determine which.
3. **Build a minimal reproducing example** and capture the exact violating
   semaphore (resource, group, acquirer set, released bit).
4. **Report to the user** with the example and root cause, and wait.

The M1/M3 assertions in `verifyPostEmission` (Step 5), plus the M2
schedule-dump / code-review audit (Step 7), exist precisely to catch these and
**hard-fail or block review**, not to trigger silent repair. No improvisation:
the resolution is the user's to direct.

---

## 2. The semaphore model (established)

A `nvws.semaphore` has exactly one piece of creation state: `is_released`
(verified: `NVWSOps.td:50`, `I1Attr:$is_released`). There is no "empty" or
"full" kind of semaphore — those words are an emitter invention and are removed
by this plan. A semaphore is created either already-released or not, is acquired
(`semaphore.acquire`, returns a token — `NVWSOps.td:58-75`), and is released
(`semaphore.release`, consumes a token — `NVWSOps.td:77-94`).

**Model invariants (per resource):**

The model is stated in terms of one concept only — the **owner** (partition).
There is no reader/writer subdivision anywhere in the emitter: it keys on the
acquiring owner and the seed marker, nothing else. `root` means the existing
`std::nullopt` / root-external owner; a **concrete owner** means a present
`PartitionId`.

- **M1 — single released seed.** A resource with at least one synchronized
  handoff (i.e. an `InitialEmpty` seed group, F4) has **exactly one** semaphore
  created `is_released = true`; a resource with no edges emits no semaphores and
  has zero. The seed is the permit claimed by the *first owner* of the resource,
  whose acquire is the first acquire in sequential program order, in the prologue
  block ahead of the warp-specialized `scf.for`.
- **M2 — everything else is unreleased.** Every other semaphore is created
  `is_released = false` and is made acquirable only by an explicit
  `semaphore.release` from some owner.
- **M3 — single concrete acquirer, optional root.** A semaphore's acquire-owner
  set may contain `root` and at most one concrete owner `P`. Valid sets are
  `{root}`, `{P}`, and `{root, P}`. Invalid sets are `{P, Q}` or
  `{root, P, Q}` for `P != Q`. The acquirer of every edge is `edge.dstOwner`,
  because the acquire is always emitted immediately before the edge's
  destination; `root` is compatible with whichever single concrete owner also
  acquires that semaphore, but it is not compatible with two different concrete
  owners.

**Why M1/M2 are correct (not assumed).** At program start the buffer is in
exactly one well-defined state, so exactly one permit can legitimately begin
satisfied — that is the single `released` seed (M1). Every other state of the
buffer becomes true only when a real op completes, so every other handoff is an
explicit release on an unreleased semaphore (M2). The single released seed is also
what keeps the sync graph a **DAG**: a loop-reused buffer would otherwise need a
back edge (free → next write); seeding one permit released breaks that would-be
cycle, and the `scf.for` carry re-circulates it (F12). *(Rationale for which permit is
the released one — used by the upstream seed selector `findFirstWriter` (F4),
not by the emitter: the first access to a buffer is necessarily a write, because
reading never-written memory is meaningless; the first owner therefore holds the
buffer-free permit, and a free buffer is exactly the state that starts
satisfied.)*

**Why M3 prevents the known hangs.** Each `(group, concrete owner)` class is a
distinct semaphore, with `root` allowed as an optional compatible acquirer
(§6.2). The seed is claimed by the first concrete owner only, possibly with
`root`; any later different concrete owner (e.g. meta_fa's partitions `p1` and
`p5`) acquires a *different*, unreleased semaphore that some earlier owner
released to it — never the seed. Two distinct concrete owners can therefore
never both acquire the same semaphore, which is the multi-acquirer condition that
produced the meta_fa and `per_edge @three_member` hangs.

---

## 3. Verified facts about the current code

These are the facts the design relies on. All confirmed by reading the source.

- **F1.** The OPT-SYNC-DAG is built from the RAW SyncPlan by
  `buildOptSyncDag` (`2164`). The emitter consumes `OptSyncDag` + `SyncPlan`
  via `emitResource` (`6624`); both are built at the emit site (`7784-7786`).
- **F2.** `OptSyncDag` already carries all emit anchors: `acquireBeforeOp`,
  `acquireBeforeYield`, `releaseBeforeOp`, `releaseBeforeYield`,
  `releaseAfterOp`, `releaseAfterYield` (struct `1687-1731`), plus the
  token-threading decisions `threadForOps` / `threadIfOps` (`1729-1730`).
- **F3.** The emitter binds a semaphore to an edge through a **single
  chokepoint**, `getSemaphoreForGroup` (`3490`), called from exactly three emit
  sites (`4656`, `4811`, `5113`). Semaphore *creation* is `createResourceSemaphores`
  (`3270`).
- **F4.** The seed is already identified: `buildOptSyncDag` creates an
  `InitialEmpty` group from `findFirstWriter`, recording `initialOp`
  (first writer op) and `initialOwner` (first writer partition) (`2192-2201`).
- **F5.** Grouping is already mostly acquirer-homogeneous:
  - `ReadyFanout` only forms when **all** consumer edges share one `dstOwner`
    (`2241-2244`); otherwise the edges stay singletons.
  - `DoneFanin` buckets are keyed by `dstOwner` (`2264`), so each is one
    `dstOwner` class.
  - `Singleton` has one edge → one `dstOwner`.
  - **`LinearChain` is the only kind that can contain two distinct `dstOwner`s**
    (a ping-pong alternates ownership between two owners, so its edges carry two
    distinct `dstOwner`s — one per acquiring owner).
- **F6 (defect).** `createSharedEmpty` (`3309-3317`) caches **one** semaphore per
  resource and **ignores its `acquirer` argument**, contradicting its own comment
  (`3304-3308`). This is the direct cause of multi-acquirer seeds.
- **F7 (defect).** Empty/unreleased classification is computed in the DAG as
  `edgeRendersEmpty` (`2782-2794`) **and independently re-derived in the emitter**
  by the heuristics `edgeUsesEmpty` (`3066-3090`) and `linearChainEdgeUsesEmpty`
  (`3234-3268`), called at `3354/3363/3378/3504/3519`. The two can disagree.
- **F8 (defect).** `SyncGroupSemaphores` (`2991-2995`) holds `empty` **and**
  `full` **and** `fullByEdge` — one group may mint up to three semaphores, which
  forces per-edge selection logic to exist.
- **F9 (defect).** The semaphore factory re-runs event-graph analysis
  (`edgeNeedsTerminalReadRelease` `3092`, `findTerminalReadReleaseAnchor` `3197`,
  `linearChainNeedsPerEdgeFulls` `3210`) to decide extra permits (`3387-3392`).
- **F10.** The pass-option mechanism is TableGen `Option<...>` in
  `Passes.td:157-161` (`useMetaPartitioner`), plumbed as a pass member and passed
  to `runOnFunction` (`7708-7709`, `7823`).
- **F11.** Post-emission cleanups run unconditionally after all resources:
  `splitSemaphoreIfForLoopScheduler`, `hoistInitialEmptyAcquires`,
  `coalesceSemaphoreForCarriers`, `coalesceTmemAllocsByBufferIdIntoViews`,
  `eraseDeadTmemAllocs` (`7799-7803`).
- **F12.** The sync-edge graph is **forward-only**: every edge runs ENTER → op →
  … → op → YIELD, one direction. There is **no back edge** (undefined in a DAG)
  and no cycle. Cross-iteration buffer reuse is **not an edge** — it is the
  `scf.for` carry: an acquire's token is yielded as a region result and threaded
  back as an `iter_arg`. So the emitter never needs to know whether a buffer is in
  a loop; one forward token-threading covers loop and straight-line alike.
- **F12a.** The current `SyncPlan` initial-permit fields (`initialPermitEdgeIdx`,
  `initialPermitBeforeOp`, `initialPermitReleaseAfterOp`, `1049-1056`) are
  **RAW-SYNC-DAG dump-only and do NOT affect emit** (comment `1044`,
  `1582-1588`), and they are framed around loop-yield reuse — which the forward
  model does not use. They are **not** the seed mechanism for this plan. The seed
  is computed by the forward-first-acquire rule (§6.3) and promoted to a real DAG
  fact (§9-V6); this also satisfies external-review finding #1.

---

## 4. Defects this plan fixes (all verified above)

1. The emitter's correctness depends on the OPT grouping choice — proven by
   `per_edge @three_member` (all-singleton DAG) hanging with multi-acquirer, and
   meta_fa only ceasing to hang after commit-4 *changed* its grouping. An
   optimization must not decide correctness. (F5, F6)
2. Two disagreeing sources of truth for the released/unreleased bit. (F7)
3. A semaphore factory that mints up to three semaphores per group and an
   emit-time chokepoint that re-runs heuristics to choose between them.
   (F3, F8)
4. Analysis heuristics living inside the emit path. (F7, F9)

---

## 5. Architecture — three stages

The pipeline is three stages with a clean contract between each. Stage 2 is new
(commit 4.5); it is what makes "the emitter is mechanical" a structural fact
rather than a claim.

1. **Sync DAG (commits 1–4; RAW or OPT).** The forward-only edge graph (F12) with
   anchors (F2) and the `releasedSemaphores` seed fact (§6.3). Unchanged except
   for the two carve-outs in §11.
2. **Emit schedule (commit 4.5) — `buildEmitSchedule`.** Consumes RAW *or* OPT and
   produces an explicit, deterministically ordered list of typed emit actions
   (§6.7): `CreateSemaphore`, `Acquire`, `Release`, `Buffer`, `ThreadToken`, each
   with a stable key and a DAG-derived endpoint / placement basis. **All semantic
   decisions live here** — semaphore identity (§6.2), released bit (§6.3), whether
   an acquire/release/buffer/thread action exists, which edge it belongs to, and
   the forward token-thread plan (§6.6). The schedule is a pure value; it mutates
   no IR. It does not need to precompute every final insertion iterator where a
   deterministic placement rule is sufficient.
3. **Emit (commit 5) — `materializeSchedule` + `verifyPostEmission`.** Walks the
   schedule in order, materializes each action with the low-level primitives
   below, then asserts M1/M3 (Step 5). M2 is checked by schedule-dump /
   code-review audit (§9-V6a), not by `verifyPostEmission`. Commit 5 is a
   `switch` over action kinds plus a fixed canonical placement policy. It performs
   **no semantic analysis** — no deciding whether an action is needed, no
   semaphore selection, no released-bit choice, no empty/full choice, and no
   event-graph walk.

This is the concrete form of the §1 "no heuristics in the emitter" requirement:
all semantic judgement is in stage 2; stage 3 only executes with canonical,
deterministic placement. Interchangeability (§7) is a property of stage 2 — RAW
and OPT both yield a valid schedule, differing only in how many
`CreateSemaphore`/`Acquire`/`Release` actions (merging), never in correctness.

### Materialization backend (reused, not rewritten)

Stage 3 invokes the existing low-level surgery, driven by schedule actions instead
of by an inline IR walk:
- `createSemaphore` + insertion anchors (`3274-3293`) ← `CreateSemaphore`.
- `emitAcquire` (`3840`) / `emitRelease` (`3847`) ← `Acquire` / `Release`.
- `SemaphoreBufferOp` view + access retarget ← `Buffer`.
- `threadCarrierThroughFor/If` and the `scf.for`/`scf.if` rewriting (`6399-6608`)
  ← `ThreadToken`.

The carrier-threading surgery is **not** rewritten — only its *trigger* moves
from inline decisions to explicit `ThreadToken` actions. Stage 3 may inspect IR
containment and the action endpoints to compute a canonical insertion point; it
must not inspect op kind, access direction, ownership, event-graph structure, or
old empty/full state to decide what action to emit or which semaphore to use.

### Deleted outright

The decision code in `emitResourceBlock` (`6343-6622`) that selects semantic
actions, chooses empty/full, or synthesizes terminal/drain releases inline is
removed — those decisions become stage-2 actions. Deterministic final placement
from an already-stamped action endpoint remains a stage-3 materialization detail.
`createResourceSemaphores` (`3270-3400`), `getSemaphoreForGroup` (`3490-3524`),
`SyncGroupSemaphores` (`2991-2995`), and the
emit-path classifiers `edgeUsesEmpty` / `linearChainEdgeUsesEmpty` /
`linearChainNeedsPerEdgeFulls` / `edgeNeedsTerminalReadRelease` /
`findTerminalReadReleaseAnchor` (emit-path only; §9-V2) are deleted, replaced by
§6 + the schedule. Any inline decision that cannot be reduced to a schedule action
without a heuristic is an escalation per §1 (tracked by §9-V7).

---

## 6. Stage-2 schedule-construction rules

§6.1–§6.6 are the rules `buildEmitSchedule` (commit 4.5, §5 stage 2) uses to turn
a DAG into schedule actions; §6.7 assembles them into the ordered schedule. They
contain no op-kind tests, no positional parity, and no event-graph walks — only
`dstOwner`, the group structure, and the `dag.releasedSemaphores` seed set
(§6.3), all intrinsic DAG data computed upstream.

### 6.1 Data model

```
// One semaphore per (group, acquirer class): root or one concrete partition.
struct SemaphoreId { unsigned groupIdx; std::optional<PartitionId> acquirer; };
// Resource-level table.
DenseMap<SemaphoreKey, Value> semaphores;     // SemaphoreKey == hashable SemaphoreId
```

`SyncGroupSemaphores`’ `empty`/`full`/`fullByEdge` triple is removed.

### 6.2 Identity rule (deterministic)

For each group `g` in `dag.groups` **that has edges**, partition its edges by
`edge.dstOwner`. Each distinct `dstOwner` class is **one semaphore**, keyed
`(g.idx, dstOwner)`. This remains the conservative deterministic construction:
the plan does **not** add a heuristic that merges `root` with a concrete owner
just because such sharing would be valid.

- Singleton / ReadyFanout / DoneFanin → exactly one class (single `dstOwner`,
  verified F5) → one semaphore.
- LinearChain → one class per distinct `dstOwner` (two for a 2-party ping-pong:
  one per acquiring owner).
- `InitialEmpty` has **no edges** (F4), so it is *not* partitioned here. It is
  the seed *marker*; §6.3 marks exactly one edge-keyed semaphore as the released
  seed (the forward-first-acquire rule), uniformly — no loop/single-use split.

This makes the concrete part of M3 **structural**: stage 2 never intentionally
assigns one semaphore to two different concrete owners. `root` is special only
for validation/materialized IR: if the same semaphore is observed as acquired by
`root` and one concrete owner, that is M3-valid; if it is acquired by two
different concrete owners, with or without `root`, that is an M3 violation.

**Deterministic creation order** (external-review finding #4). Semaphore creation
must iterate `dag.groups` in order, then each group's `edgeIdxs` in order,
collecting first-seen `dstOwner`. Do **not** iterate a `DenseMap` to decide create
order (non-deterministic). The `(group, dstOwner)` lookup table may be a
`DenseMap`; only the *creation* walk must be ordered.

### 6.3 Released-bit rule (deterministic, forward-only)

There is no loop/cycle distinction (F12). A semaphore is created
`is_released = true` **iff its earliest event in program order is an acquire** —
nothing releases it before its first acquire. Concretely: the first access to a
buffer is a write, so the first owner acquires the buffer-free permit before
anyone releases it → that one semaphore is the seed. Every other semaphore has a
release ahead of its first acquire (a producer made data ready, or an earlier
owner freed the buffer) → created `is_released = false`.

This is computed **once**, in `buildOptSyncDag`'s common finalization (the
post-grouping stage at `2331`, which runs for both RAW and OPT), by a forward
program-order scan (program order is already built — `buildProgramOrderRank`,
`750`) and recorded as the DAG fact `dag.releasedSemaphores : set of (groupIdx,
acquirer)` — promotion covered by §9-V6, which also satisfies external-review
finding #1 (seed identity is a DAG fact, not re-derived in the emitter). The
emitter performs no scan; it only reads the set. The `InitialEmpty` alias resolves
to that single member as its `semaId` (§6.4).

Exactly one released semaphore per seeded resource (M1; an edge-free resource
yields zero), asserted in `verifyPostEmission` (Step 5). All other semaphores are
`is_released = false` and become acquirable only by an explicit release (M2).
M2's first-event property is validated by inspecting `buildEmitSchedule` and the
schedule dumps / FileChecks; it is **not** a formal `verifyPostEmission`
obligation.

### 6.4 Semaphore-id resolution (stage 2)

Each `Acquire`/`Release`/`Buffer` action's `semaId` is computed once in stage 2
(replacing the current `getSemaphoreForGroup` chokepoint, F3, which is deleted):

```
SemaphoreId semaIdFor(groupIdx, edge):
    if dag.groups[groupIdx].kind == InitialEmpty:
        return seedSemaId;                 // the single member of releasedSemaphores (§6.3)
    return SemaphoreId{ groupIdx, edge.dstOwner };
```

`edge` is non-null for any group that has edges (every non-`InitialEmpty` group
does). No `switch` on group kind, no `edge`-direction branch. Stage 3 never calls
this — it reads the `semaId` already stamped on each action.

### 6.5 CreateSemaphore actions (stage 2)

```
// One CreateSemaphore action per (edge-bearing group, acquirer), in the §6.2
// deterministic order. Released iff the DAG marked it the seed (§6.3).
for each group g in dag.groups with edges:                 // dag.groups order
    for each distinct acquirer a in g.edges (first-seen dstOwner order):
        released = dag.releasedSemaphores.contains({g.idx, a})
        schedule.add(CreateSemaphore{ semaId={g.idx, a}, released,
                                      placementBasis = semaphoreCreationScope })
```

Stage 3 materializes each `CreateSemaphore` with `createSemaphore` at the existing
insertion point / partition stamping (`getSemaphoreInsertionAnchor` /
`getLocalSemaphoreCreateAnchor`, `3274-3293`, reused verbatim).

### 6.6 Token threading (forward, uniform)

Each acquire yields a token consumed by its release. The token flows **forward**
along the DAG: when its release is past a region boundary, the token is yielded as
a region result; when that region is an `scf.for`, the result is threaded as an
`iter_arg` to the next iteration; threading continues forward until the token
reaches its release. This is **one uniform mechanism** for every resource — loop
and straight-line alike. Stage 2 records it as `ThreadToken` actions (§6.7); stage
3 materializes them with the existing carrier-threading backend
(`threadForOps` / `threadIfOps`, `threadCarrierThroughFor/If`, F2/§5). No
loop-specific decision is made and no back edge is introduced.

### 6.7 The emit schedule (commit 4.5)

`buildEmitSchedule(dag, sp, group, plan)` returns an `EmitSchedule` — an ordered
`SmallVector<EmitAction>`. It is built **per resource, against live IR**, right
before that resource is materialized, matching the existing per-resource re-plan
loop (`7775-7786`), so every endpoint / placement basis is a live `Operation*` /
`Region*` when stage 3 consumes it (no stale pointers across the IR rewrites
prior resources make).

Resources are scheduled and materialized in a stable outer order: the order of
the resource list produced by the DAG/resource planner. If that list is ever
backed by a map or otherwise loses source order, it must be sorted by a stable
resource key before schedule construction. Cross-resource interleaving is not a
semantic decision; stable outer ordering exists only for deterministic IR and
test output.

Action kinds — each carries a **stable key** (ordering, dedup, verification) and
a DAG-derived **endpoint / placement basis**:

| Kind            | Stable key             | Endpoint / placement basis               | Rule |
|-----------------|------------------------|------------------------------------------|------|
| CreateSemaphore | `(groupIdx, acquirer)` | semaphore creation scope / owner         | §6.5 |
| Acquire         | `(semaId, edgeIdx)`    | destination endpoint: dstOp, yield, or region entry | §6.4 |
| Release         | `(semaId, edgeIdx)`    | source endpoint: srcOp, yield, or region entry (only if the DAG carries a release fact for the edge) | §6.4 |
| Buffer          | `(semaId, edgeIdx)`    | the access op being retargeted           | access event |
| ThreadToken     | `(tokenKey, regionOp)` | the `scf.for` / `scf.if` to thread plus token endpoints | §6.6 |

`released` on `CreateSemaphore` = `dag.releasedSemaphores.contains(key)` (§6.3).
`Acquire`/`Release`/`Buffer` carry the `semaId` from §6.4. `ThreadToken` carries
the forward token-flow plan (§6.6): the iter_args/results to add so an acquire's
token reaches its release.

A `Release` action is emitted only from an ordinary DAG release fact / release
anchor for that edge. Stage 2 may translate that fact into a schedule action;
stage 3 must not infer release necessity from op kind, access direction, terminal
read shape, drain behavior, or event-graph inspection.

**Stage-3 canonical placement policy.** The schedule does not over-specify every
final insertion iterator. Stage 3 computes placement mechanically from the action
endpoint / placement basis with these fixed rules:
- A destination-side `Acquire` is placed immediately before the destination-side
  `Buffer` when there is one, otherwise immediately before the destination op or
  yield.
- A `Buffer` is placed immediately before the op that needs the retargeted
  buffer, after any `Acquire` for the same `(semaId, edgeIdx)` at that endpoint.
- A source-side `Release` is placed at the source side: after `srcOp` when that is
  the endpoint, at the region/block entry when the endpoint is `ENTRY`, or
  immediately before the yield/terminator when the endpoint is `YIELD` / exit.
- For an `ENTRY -> dst` edge, destination-side acquire/buffer materialization is
  placed at the beginning of the target region/block before the first relevant
  destination op, preserving local `Acquire` then `Buffer` order.
- For a `src -> YIELD` edge, exit-side materialization is placed immediately
  before the yield/terminator.

These placement rules are deterministic materialization, not semantic analysis.
Stage 3 may use IR containment and action endpoints to find the canonical slot;
it must not inspect op kind, access direction, ownership, event-graph structure,
or old empty/full state to decide that an action exists, choose a semaphore, or
choose the released bit.

**Deterministic order** (the only ordering stage 3 obeys):
1. all `CreateSemaphore` actions first, in §6.5 order;
2. then the rest by **program-order rank** of the endpoint / placement basis
   (`buildProgramOrderRank`, `750`); ties at one rank are broken by fixed local
   priority `Release`, then `Acquire`, then `Buffer`, then `ThreadToken`, then by
   stable key. This priority is checked by the §8.3 sweep.

The schedule is independently dumpable (under `NVWS_INSERT_SEMA_DUMP_DAG`) and is
what the §8.1 interchangeability test asserts. Commit 4.5 must print this as its
own named diagnostic section in the same dump path/style as the commit-1/2/3/4
DAG diagnostics, before materialization is wired. The dump must show, per
resource, the stable action order, action kind, `semaId`, `edgeIdx` when present,
`released` for `CreateSemaphore`, endpoint / placement basis, and token-threading
endpoints.

---

## 7. OPT on/off flag and RAW interchangeability

### 7.1 RAW ≡ all-singleton OPT

Verified construction: `buildOptSyncDag` already produces `Singleton` groups for
every edge that is not merged. A RAW DAG is therefore an OPT DAG in which the
merge steps (ReadyFanout `2229-2251`, DoneFanin, LinearChain) are skipped, leaving
the `InitialEmpty` seed (F4) plus one `Singleton` per edge.

### 7.2 Flag

Add to `Passes.td` (mirroring F10):

```
Option<"disableOptSyncDag", "disable-opt-sync-dag", "bool", /*default*/"false",
       "Emit from the unoptimized per-edge RAW-SYNC-DAG (skip edge merging).">
```

Plumb it through `runOnFunction` (F10 path) and into `buildOptSyncDag` as a new
parameter `bool optimize`. The **merge-formation** blocks (ReadyFanout
`2229-2251`, DoneFanin, LinearChain) and the **common finalization** (anchor-map +
threading population, which runs *after* grouping at `2331`) must be **factored
apart**. When `optimize == false`, only merge formation is skipped — every
unclaimed edge stays a `Singleton` (`2322-2329`) — and the common finalization at
`2331+` still runs, so the RAW DAG has identical anchors/threading to OPT.
Returning *before* `2331` would yield groups with no emission anchors; that is the
bug to avoid (external-review finding #3). The returned type is unchanged, and the
`releasedSemaphores` fact (§6.3) is computed in that same finalization for both
modes — so neither the schedule builder (stage 2) nor the emitter (stage 3)
branches on the flag.

An environment-variable fallback is **not** added (the pass-option is the single
source of truth, and lit tests drive it directly).

### 7.3 Why this proves the requirement

`buildEmitSchedule` (§6.7) consumes RAW or OPT through the same rules (§6). Both
yield a valid schedule; they differ only in how many
`CreateSemaphore`/`Acquire`/`Release` actions (merging), never in correctness —
stage 3 executes whichever schedule it is given. The §8.1 lit test asserts both
schedules materialize correct IR.

---

## 8. Test plan

### 8.1 Lit (no GPU)

1. **Interchangeability test** — new file `test/NVWS/insert_semas_raw_emit.mlir`
   (or a `-split-input-file` section), two RUN lines over the same input
   (RUN pattern verified from `test/NVWS/insert_semas_local_cfg.mlir:1`):
   ```
   // RUN: triton-opt %s -split-input-file -allow-unregistered-dialect \
   // RUN:   --nvws-insert-semas -cse | FileCheck %s --check-prefix=OPT
   // RUN: triton-opt %s -split-input-file -allow-unregistered-dialect \
   // RUN:   --nvws-insert-semas=disable-opt-sync-dag=true -cse | FileCheck %s --check-prefix=RAW
   ```
   The RAW prefix asserts per-edge semaphores; both prefixes assert the same
   acquire/release **structure** and that **exactly one** `semaphore.create`
   carries `true` (released) per resource. The commit-4.5 schedule dump (§6.7) may
   additionally be FileChecked directly, so a schedule regression is caught before
   materialization.

2. **Existing NVWS lit suite** must pass. The known-stale CHECK files
   (`insert_semas_local_cfg.mlir`, `insert_semas_per_edge_tmem.mlir`,
   `insert_semas_meta_fa_fwd.mlir`, and the others enumerated previously) are
   updated **inline, above each op, with exact captured tokens (no wildcards)**
   only after the emitted IR is confirmed correct — never to codify a hang.

3. **`test/TritonGPU/automatic-warp-specialization.mlir` must pass unmodified.**
   (Hard constraint carried from prior work.)

### 8.2 GPU correctness (run only on request, then `killgpu.sh`)

- matmul: `pytest -s python/test/unit/language/test_warp_specialization.py::test_warp_specialize_tma_matmul_persistent[...]`
- meta_fa fwd: `NVWS_USE_SSA_TMEM=1 TRITON_ALWAYS_COMPILE=1 TRITON_NVWS_USE_META=1 python python/tutorials/fused-attention-ws-device-tma.py`
- Both must pass with `disable-opt-sync-dag` **false** and **true** (the flag is
  threadable to the Python compile path via the existing pass-option plumbing;
  if not, GPU verification of RAW mode is via the dumped IR + lit, and this is
  recorded as a limitation rather than silently skipped).

### 8.3 Invariant sweeps

Re-run the per-`tt.func` single-concrete-acquirer-with-optional-root /
no-dropped-release checker over the emitted IR of every NVWS lit test, in both
OPT and RAW modes. Zero violations required.

---

## 9. Items requiring empirical confirmation before/within implementation

These are **not** assumed by the design; each is a gated step.

- **V1 (Step 0, blocking).** Dump `(resource → {is_released==true count})` and
  `(group → set of dstOwners)` for matmul and meta_fa on the current binary.
  Confirm: exactly one semaphore per resource whose earliest program-order event
  is an acquire (the forward-first seed, §6.3); LinearChain is the only
  multi-`dstOwner` group kind. If either is false, the §6 rules are revised before
  coding.
- **V2 (Step 1).** Confirm by call-graph which of `edgeUsesEmpty`,
  `linearChainEdgeUsesEmpty`, `linearChainNeedsPerEdgeFulls`,
  `edgeNeedsTerminalReadRelease`, `findTerminalReadReleaseAnchor` are reachable
  *only* from the emit path. Symbols also used by the DAG dumps are kept for the
  dumps and removed only from emit.
- **V3 (Step 3).** Confirm the forward-first-acquire rule (§6.3) yields exactly
  one released semaphore per seeded resource in matmul and meta_fa; that it is the
  first owner's buffer-free permit; and that its acquire is the first acquire in
  program order (the prologue / loop-entry acquire, matching M1). Uniform — no
  loop/straight-line split.
- **V4 (Step 5).** Confirm the post-emission cleanups (F11) remain correct under
  the new single-seed scheme (they currently assume the old shared-empty); adjust
  or assert as needed. `hoistInitialEmptyAcquires` in particular is named for the
  old model and must be reviewed.
- **V5 (Step 4).** Before deleting the emit-path synthesizers
  (`findTerminalReadReleaseAnchor` / `edgeNeedsTerminalReadRelease`, F9) **and**
  before rerouting the loop-exit drain (`5022-5156`, external-review finding #2),
  confirm every release they currently synthesize — terminal reads *and*
  loop-exit drains — already exists as an ordinary forward edge in the
  RAW-SYNC-DAG, and that the drain's target is selectable without the op-kind
  branch (`5138`). If any does **not**, that is a missing DAG fact: STOP and
  escalate per the §1 Hard Rule — do not re-add the synthesizer or the op-kind
  branch.
- **V6 (Step 3 prerequisite; Step 6 extends it to RAW mode).** Add the
  forward-first-acquire result as a real DAG fact `dag.releasedSemaphores` (a set
  of `(groupIdx, acquirer)`), computed in `buildOptSyncDag`'s post-grouping
  finalization (`2331+`) by the forward program-order scan — mandatory before
  `buildEmitSchedule` reads it; this is external-review finding #1. Step 3 may
  add it for the existing OPT path before schedule construction; Step 6's
  factoring must preserve the same finalization for RAW mode. Then confirm every seed
  semaphore's **release** is an ordinary forward edge release reachable by token
  threading (§6.6), for both loop and straight-line resources. If any seed has no
  ordinary release edge, that is a missing DAG fact: STOP and escalate (§1) — do
  not synthesize it.
- **V6a (Step 7 review gate, not a verifier pass).** Confirm from the
  `buildEmitSchedule` code and schedule dumps that every unreleased semaphore's
  first possible acquire is preceded by a DAG-scheduled `Release` action. This is
  the M2 first-event consistency audit. Do **not** add this as formal
  `verifyPostEmission` logic unless explicitly requested.
- **V7 (Step 4).** Confirm every semantic decision the current `emitResourceBlock`
  (`6343-6622`) makes inline — action existence, prebuffering, deferred/after-op
  sync, poison-token insertion, if/for carrier threading and partition stamping —
  reduces to one of the five schedule actions (§6.7) plus the deterministic
  placement policy. For each that does **not**, STOP and escalate (§1): either add
  the missing action kind/fact upstream (with approval) or mark the migration
  blocked. Do not leave a semantic decision inline in stage 3.

---

## 10. Implementation steps (ordered, each independently verifiable)

0. **V1 dump.** Land nothing; produce the evidence in §9-V1. Gate the rest on it.
1. **Carve the deletes.** Identify the exact symbol set to delete vs keep (§5
   "Deleted outright"; §9-V2). No behavior change yet; produce the list.
2. **Schedule data model.** Define `EmitAction` (the five kinds, §6.7) and
   `EmitSchedule`, plus the `(groupIdx, acquirer)` semaphore id (§6.1). Build
   green.
3. **`buildEmitSchedule` (commit 4.5).** Implement §6.2–§6.7: emit
   CreateSemaphore/Acquire/Release/Buffer/ThreadToken actions with stable keys and
   endpoints / placement bases in the deterministic order; consume the
   `releasedSemaphores` DAG fact (§9-V6). Add the named commit-4.5 schedule
   diagnostic described in §6.7 and dump it; do not wire it to emission yet.
4. **`materializeSchedule` (commit 5).** Replace the inline `emitResource` walk
   with an ordered switch over actions calling the §5 backend primitives. Delete
   the emit-path heuristics (§9-V2) and the inline decisions in
   `emitResourceBlock`; each must reduce to a schedule action (§9-V7) or escalate
   (§1).
5. **Post-emit verifier.** Add to `verifyPostEmission`: M1 (exactly one
   `is_released=true` semaphore for any resource that emits semaphores; zero for
   an edge-free resource) and M3 (each semaphore acquired by root and at most one
   concrete owner).
   Each is a hard-fail per the §1 Hard Rule, not a repair trigger. M2 first-event
   consistency is validated by the schedule-dump / code-review audit (§9-V6a),
   not by `verifyPostEmission`. Also review the F11 cleanups (§9-V4).
6. **OPT flag.** Add `disable-opt-sync-dag` (§7), thread it, and factor
   `buildOptSyncDag` so RAW skips **merge formation only** while the common
   finalization (anchors/threading + `releasedSemaphores`) still runs (§7.2). Do
   not return before the finalization.
7. **Lit.** Add the interchangeability test (§8.1.1); update stale CHECKs inline
   with exact tokens; confirm `automatic-warp-specialization.mlir` unmodified.
8. **GPU.** On request only: matmul + meta_fa, OPT and RAW (§8.2), then
   `killgpu.sh`.
9. **Invariant sweeps** (§8.3). Zero violations.

Each step first builds from
`/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/`
with `ninja triton triton-opt`, then runs the relevant lit tests from that same
build directory with
`/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test`.
Do not run pytest unless explicitly requested.

At **every** step, the Hard Rules in §1 apply: if a case seems to need a
heuristic, **or** if a valid DAG produces an M1/M2/M3 violation, stop,
root-cause, build a minimal example, and report — do not add the heuristic, do
not improvise a repair for the violation, and do not proceed past the blocked
step.

---

## 11. Non-goals

- No change to ACCESS-DAG, OWNERSHIP-DAG, RAW-SYNC-DAG (`buildSyncPlan`), or the
  OPT-SYNC-DAG *merge criteria* (the ReadyFanout/DoneFanin/LinearChain formation
  logic), **except** the two changes this plan requires: (a) factoring
  `buildOptSyncDag` so the flag can skip merge formation while the common
  finalization still runs (§7.2), and (b) adding the derived `releasedSemaphores`
  seed fact in that finalization (§6.3, §9-V6). The same-`dstOwner` guards
  already present in ReadyFanout (`2241-2244`) are retained.
- No change to stage/phase assignment (separate pass `AssignStagePhase.cpp`).
- No change to the older, unrelated `nvws-insert-semaphore` pass
  (`InsertSemaphore.cpp`).
- No performance work; correctness and structural soundness only.

---

## 12. Acceptance criteria

- All semantic emit decisions live in the commit-4.5 schedule (§6.7); commit-5 is
  an ordered `switch` over schedule actions, deterministic canonical placement,
  and verification, with zero op-kind / parity / event-walk heuristics. Semaphore
  identity is a pure function of `(group, dstOwner, seed-marker)`, and M3
  validation treats `root` as compatible with at most one concrete owner.
- The same input emits correct IR with `disable-opt-sync-dag` false and true,
  proven by the §8.1.1 lit test.
- `verifyPostEmission` enforces M1 (one released seed per seeded resource; zero
  for an edge-free resource) and M3 (single concrete acquirer per semaphore, with
  optional `root`); all pass on the full lit suite in both modes. M2 first-event
  consistency is verified by schedule-dump FileChecks / code review (§9-V6a), not
  by `verifyPostEmission`.
- matmul and meta_fa pass on GPU in both modes (or RAW-mode GPU limitation is
  documented, not hidden).
- `test/TritonGPU/automatic-warp-specialization.mlir` passes unmodified.
- No heuristic was added to the emitter, and no M1/M2/M3 violation was
  improvised around. Any case that appeared to need a heuristic, or that
  produced an invariant violation, was escalated per the §1 Hard Rules
  (stopped, root-caused, exampled, reported) and resolved upstream — not at
  emit time.
