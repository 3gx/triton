# Per-Edge Semaphore Plan v4 — Follow-up Work

Checkpoint state: `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp`
restored from the prior `b74750bee4` implementation as a structural skeleton.
70 of 72 `test/NVWS/` lit tests pass with this baseline. Remaining work below
must close the gap to the v4 contract in `per-edge-sema-plan.v4.md`.

## Failing lit tests at checkpoint

### `test/NVWS/insert_semas.mlir`

The strict CHECK lines (post-b74750bee4) demand carrier-token threading
through `scf.for`. Specifically `@warp_specialize_tma_matmul`:

- initial `tmem_store` before the `tt.warp_specialize` loop must keep its
  `acquire EMPTY` token live and yield it as the loop's iter-arg
- the loop body's `tc_gen5_mma` consumes the carried token via
  `nvws.semaphore.buffer EMPTY, TOK`
- `nvws.semaphore.release FULL` is emitted only after the loop, against the
  loop's result token
- subsequent `tmem_load` then acquires FULL once

Current behavior eagerly releases FULL after the initial store and re-acquires
EMPTY inside the loop. The v4 contract's §Carrier Token and §Initial Writable
Permit sections describe the correct shape.

### `test/NVWS/tmem-buffer-reuse-semas.mlir`

`@sourceful_tokenless_alias` (and similar) expect same-owner consecutive
accesses to share one acquire window:

- `{1}` acquires EMPTY once
- `{1}` stores A, then loads A, then releases FULL (signaling `{0}`)
- `{0}` acquires FULL once, stores B, loads B, releases EMPTY

Current behavior emits one acquire/release pair per event, even when the next
event is same-owner same-resource.

## Concrete v4 deltas required

Listed in approximate dependency order.

### 1. Owner-phase coalescing in `emitSemaphores`

Required by §Dependency Edges and §Structured Region Ownership. Add a
pre-emission step that groups consecutive events for the same
`(resourceKey, owner)` into one phase, then emit one acquire / one release per
phase. Writer-then-same-owner-reads must coalesce; same-owner read fanout into
a single phase. Defer release to the last event of the phase.

The data shape can stay close to `SharedReadPhasePlan` but generalized to
cover writer-led phases and post-write same-owner read continuations.

### 2. Carrier-token threading through `scf.for`

Required by §Carrier Token and §Control Flow Rules. When the structured
ownership plan implies a resource's owner/state crosses an `scf.for`
boundary:

- the acquire happens once before the loop
- the loop gains an extra `iter_arg` of type `!ttg.async.token` initialized
  to the acquired token
- inside the body, all access events use `nvws.semaphore.buffer S, %iter_arg`
- the body yields the token forward (unchanged if no transition inside the
  body)
- release on the loop result happens after the loop

This is also already partially expressed by `tryCreateLoopInitialAcquire` in
the restored code; extend that helper to be the general mechanism rather than
a special case, and drive it from the structured ownership plan rather than
ad hoc heuristics.

### 3. Region-shaped structured ownership planner

Required by §Structured Region Ownership. Replace block-level ownership
state with explicit `RegionOwnership` records keyed by
`(Region *, logicalGroupId, resourceKey)`. Pure functions:

- `planRegion(region, entryOwnership) -> exitOwnership + records`
- `planIf(ifOp, entryOwnership) -> joinOwnership + branch records`
- `planFor(forOp, entryOwnership) -> loopExitOwnership + loop records`
- `reconcileRegion(controlOp, childExitOwnerships) -> chosen exitOwnership`

The planner must be pure — no `nvws.semaphore.*` ops, no IR mutation. Then
derive `SyncEdge`s from owner transitions between regions and direct uses.

### 4. Explicit `SyncEdgeInfo` / `SyncGroupInfo` data structures

Required by §Planned SyncEdgeInfo and §Final Combine Subpass. Replace the
inline edge-emission loop with:

1. derive `SyncEdgeInfo` records from the ownership plan
2. run `optimizeSyncEdgesForFanoutFanin` → produces `SyncGroupInfo`
3. emit IR from `SyncGroupInfo`

This separates planning from emission and is what the pre-emission verifier
will consume.

### 5. `NVWS_INSERT_SEMA_DUMP_DAG=1` debug dump

Required by §Debug DAG Dumps. Four tree views per logical buffer group:

1. `ACCESS-DAG` — access events under CFG shape
2. `OWNERSHIP-DAG` — one tree per backing resource
3. `RAW-SYNC-DAG` — same trees with raw `SyncEdge`s inline
4. `OPT-SYNC-DAG` — same trees post fanout/fanin optimization

The formatting contract is strict (mechanical column alignment, lowercase
member names vs uppercase semaphore names, `R`/`W` for memory rows and `r`/`a`
for semaphore rows). CHECK lines in `test/NVWS/insert_semas_per_edge_tmem.mlir`
already exercise this dump.

### 6. Pre/post-emission ownership verifiers

Required by §Plan Verification and Enablement. Two passes:

- pre-emission: validates the ownership plan and `SyncGroup` graph
  (hard-diagnostic on violation)
- post-emission: walks emitted IR, confirms each planned anchor exists at the
  planned location, no branch-local token escapes its branch, etc.

Must run unconditionally during bring-up. Can be gated by a debug flag
later.

### 7. Remove obsolete two-owner helpers

Required by §Stage 6. The restored skeleton no longer uses
`TMEMSemaphore::Kind{PING,PONG}`, but other two-owner helpers may still be
present. Audit and remove:

- ping/pong kind toggling
- `pickOtherPartition`-style helpers
- two-owner close/reconcile helpers
- target-partition-only semaphore selection
- original TMEM async-token DAG scheduling

Replacement logic must target the exact dependency edge or exact combined
edge group.

## Acceptance criteria not yet satisfied

From `per-edge-sema-plan.v4.md` §Acceptance Criteria:

- structured region ownership assigned per `(logicalGroupId, resourceKey)`
  before raw `SyncEdge` creation — partially implemented; not driven by a
  region planner yet
- raw cross-owner `SyncEdge`s optimized into `SyncGroup`s before any
  `nvws.semaphore.*` IR is emitted — not yet; emission is inline
- `NVWS_INSERT_SEMA_DUMP_DAG=1` dumps in old-style CFG tree form — not yet
- existing fanout lit tests keep compact one-`FULL`/one-`EMPTY` shape —
  passes today
- existing linear handoff reuse tests keep compact one-`EMPTY`/one-`FULL`
  shape — passes today
- qk/alpha/pacc shared buffer not merged on target-partition only —
  passes today (`@tmem_qk_alpha_pacc_three_member_edges`)
- N-owner sequences supported — passes today (`@n_owner_alias_sequence`)
- unsupported patterns emit hard diagnostics — partial; pre-emission verifier
  not yet built

## Logical order to resume

1. Owner-phase coalescing (delta 1) — fixes
   `@sourceful_tokenless_alias` and friends.
2. Carrier-token threading (delta 2) — fixes
   `@warp_specialize_tma_matmul`.
3. Region-shaped planner + explicit `SyncEdgeInfo` / `SyncGroupInfo`
   (deltas 3–4) — required by the v4 contract independently of the failing
   tests; refactor target.
4. `NVWS_INSERT_SEMA_DUMP_DAG` (delta 5) — additive once planner emits
   structured records.
5. Verifiers (delta 6).
6. Obsolete-helper cleanup (delta 7).
