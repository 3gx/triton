# Per-Edge Semaphore Plan v4 — Status and Follow-up Work

## Current state

`third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp` now
implements the v4 §Uniform Access-DAG Builder architecture end-to-end:

```
unified discovery (TMEM + Local)
  → access-dag  (collectEvents)
  → ownership-dag (buildOwnershipPlan: planRegion / planIf / planFor /
                   reconcileRegion)
  → raw-sync-dag  (buildRawSyncDag: SyncEdgeInfo per cross-owner edge)
  → opt-sync-dag  (optimizeSyncEdgesForFanoutFanin: combines A/B/C)
  → debug dump   (NVWS_INSERT_SEMA_DUMP_DAG=1, four CFG-shaped trees)
  → pre-emission verifier
  → emit         (emitFromOptSyncDag, consumes optSyncDag)
  → post-emission verifier
  → legacy CFG cleanup
  → erase dead ops
```

70 of 72 `test/NVWS/` lit tests pass. The `insert_semas_per_edge_tmem.mlir`
DAG-dump CHECK lines are exercised end-to-end via
`NVWS_INSERT_SEMA_DUMP_DAG=1`.

## What's covered

- §Goal items 1–4: CFG-shaped access DAG, structured region ownership,
  raw `SyncEdgeInfo` per cross-owner dep, fanout/fanin combine optimization.
- §Uniform Access-DAG Builder: one discovery pass + one phased pipeline
  over both TMEM and Local groups.
- §Access Events + §Physical Conflict Key: per-member touches with
  `resourceKey` from offset-interval union-find.
- §Structured Region Ownership: `RegionOwnership` records keyed by
  `(Region *, logicalGroupId, resourceKey)`. Pure planners.
- §Effective Owner: root/external (nullopt) distinct from `(wsTag=0,
  partition=0)`.
- §Dependency Edges: explicit `SyncEdgeKind { Ready, Done, Handoff }` with
  source/target events, owners, regions, anchors.
- §Final Combine Subpass: `SyncGroupKind { ReadyFanout, DoneFanin,
  LinearChain, Singleton }` from `optimizeSyncEdgesForFanoutFanin` with
  Combine A/B/C safety checks.
- §Debug DAG Dumps: four CFG-shaped tree views with strict alignment
  (R/W vs r/a, lowercase members vs uppercase semaphores, structural
  scf.for/scf.if rows, region entry/exit annotations, carried-token marker).
- §Plan Verification: pre-emission verifiers (`verifyOwnershipPlan`,
  `verifyRawSyncDag`, `verifyOptSyncDag`) and post-emission verifier
  (`verifyEmittedIR`). Hard diagnostics.
- §Combine A (ready fanout), §Combine B (done fanin), §Combine C (linear
  chain preservation): all three implemented with safety checks.
- §Structured Region Ownership "writer-phase" coalescing: when a sourceful
  TMEM alloc is followed by same-owner consecutive reads of the same
  resource, all reads share one acquire window and one nvws.semaphore.buffer
  view; release is deferred to the last event of the phase. Alternating-
  owner linear chains use the prior phase's release semaphore as the next
  phase's acquire semaphore.
- §End-to-End Example 1 (simple fanout/fanin),
  §End-to-End Example 2 (conditional-only if),
  §End-to-End Example 3 (if post-consume),
  §End-to-End Example 5 (qk/alpha/pacc shared TMEM): all pass.

## What's remaining

Two NVWS tests still fail. Both require carrier-token-through-scf.for /
scf.if emission patterns:

### `test/NVWS/insert_semas.mlir`

`@warp_specialize_tma_matmul` expects an outer-writer phase that crosses
into a warp-specialized scf.for:

```
acquire EMPTY                    // root, OUTSIDE loop
buffer EMPTY, ATOK               // root
tmem_store ..., BUF              // root, OUTSIDE loop
scf.for iter_args(TOK = ATOK)    // carry acquire token as iter_arg
  buffer EMPTY, TOK              // partition 1, INSIDE loop
  tc_gen5_mma ..., BUF           // partition 1
release FULL, TOK2 [tc5mma]      // partition 1, AFTER loop
```

The legacy `tryCreateLoopInitialAcquire` handles the related shape where
the *inner* MMA's iter_arg-sourced dep traces directly to the alloc's
token, but not when an outer `tmem_store` chains the alloc's token into
the iter_arg. The §Carrier Token semantics demand:

- detect when a writer outside scf.for has its async-token result used as
  scf.for iter_arg init, and the corresponding iter_arg flows to inner
  events touching the same resource
- acquire once before the writer, retarget the writer through the
  carrier buffer view, replace the iter_arg init with the acquire token
- inside the loop, inner events reuse the carrier token (no acquire)
- release fires on the loop's result token after the loop

### `test/NVWS/tmem-buffer-reuse-semas.mlir`

`@sourceful_tokenless_alias` and `@n_owner_alias_sequence` now produce
the correct same-owner phase / linear-chain shape thanks to the
writer-phase coalescing patch. Four sub-tests still fail:

- `@loop_token_slot_alias`: two tmem_allocs returning tokens, both
  carried as scf.for iter_args. Expects one acquire OUTSIDE the loop
  yielding `ATOK`, scf.for iter_args(`ATOK`, `POISON`), inner producer
  writes via the carrier, releases FULL, inner consumer acquires FULL,
  releases EMPTY, then re-acquires EMPTY for the next iteration. The
  `POISON` token replaces the second alloc's init.
- `@if_branch_alias`: alloc tokens reaching into scf.if branches with
  branch-yielded tokens.
- `@accumulator_and_operand_alias_same_partition`: MMA accumulator alias
  pattern with two tmem_allocs sharing a buffer.id.
- `@singleton_staged_alloc_preserves_buffer_attrs`: depth-1 backing alloc
  inherits `buffer.copy` attribute, original alloc erased.

All four share the same root: tmem_alloc's token result is used as
scf.for iter_arg init, and inner events consume the iter_arg as their
async dep. The needed transform is an extension of
`tryCreateLoopInitialAcquire` (or its replacement) that:

1. detects the outer-alloc → scf.for-iter_arg → inner-consumer chain
2. acquires EMPTY before the loop
3. replaces the iter_arg's init with the acquire token (or POISON for
   carried tokens unrelated to this group)
4. inside the loop, inner events use the carried token as carrier
5. releases FULL after the loop using the loop's matching result

### Stage 6 cleanup

`ResourceState`'s legacy state-machine fields (`writerOwner`,
`lastReaderOwner`, etc.) still drive parts of the emit body. Replacing
those with reads from the `RegionOwnership` records is the v4 §Stage 6
work. The data is already computed in `ownershipPlan`; the rewrite is a
mechanical refactor of `emitSemaphores`'s decision logic to consult the
plan instead of the running state machine.

## Suggested resume order

1. Outer-writer → loop-carrier transform (fixes `@warp_specialize_tma_matmul`
   and `@singleton_staged_alloc_preserves_buffer_attrs`).
2. Multi-token loop carry with POISON init (fixes
   `@loop_token_slot_alias` and the alias-token tests in the same family).
3. If-branch carrier-token threading (fixes `@if_branch_alias`).
4. Stage 6 emitSemaphores rewrite to consume `RegionOwnership` directly,
   retiring `ResourceState`.
