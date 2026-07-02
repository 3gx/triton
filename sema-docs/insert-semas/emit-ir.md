# EMIT-IR

## Rule

EMIT-IR materializes the finalized SYNC-DAG. It does not rediscover owners,
edges, holds, token-reuse decisions, pending counts, or stage offsets. A
`Node::reuseTokenOwner` mark is a decision to render, not a decision EMIT-IR
may make. Its one schedule exception is the loop-scheduler
workaround (`workaroundLoopScheduler`), which
splits qualifying `scf.if` operations so a release leading a branch is
hoisted before the `if` and a trailing acquire follows it — a shape the loop
scheduler otherwise mishandles:

```text
before the workaround                    after
%t = scf.if %c {                         scf.if %c { r S0 %t0 {1} }   ; hoisted guard
  r  S0 %t0 {1}     ; leading release    scf.if %c { ... }            ; body: protocol ops gone
  ...                                    %t = scf.if %c {             ; trailing guard
  %t1 = a  S1 {1}   ; trailing acquire     %t1 = a S1 {1}; yield %t1
  yield %t1                              } else { yield %t0 }
} else { yield %t0 }
```

Two rarer variants hoist only the release (a
TMEM semaphore whose branch re-acquires later), or — when the acquire leads
the branch and the release already sits outside the `if` — perform the same
split by moving that release into the hoisted guard and the leading acquire
into the trailing guard. A release
moved this way with no assigned
pipeline stage may inherit one from the first MMA that precedes it in the
block. Model terms are defined in the
[InsertSemas overview](overview.md#core-objects).

## Mechanical sequence

1. For TMEM groups that received semaphores, strip the input IR's own TMEM
   dependency tokens: TTGIR threads `!ttg.async.token` values through
   `tmem_alloc`/`tmem_load`/`tmem_store`/MMA to order them — unrelated to
   semaphore tokens — and the semaphore protocol now carries that ordering.
   Remove those operands and results and delete the token entries that
   become dead in loop and `if` signatures.
2. Allocate the planned multi-buffered backing and create semaphores with the
   DAG's initially-released and pending-count facts.
3. Emit the entry acquires.
4. Extend `scf.for`/`scf.if` signatures once for crossings that require a
   token result.
5. Walk the SYNC-DAG and render accesses, acquires, releases, and region
   yields.
6. Fold the split groups that share one planned `buffer.id` — the circular
   SMEM and mixed-depth TMEM arrangements of
   [ACCESS-DAG's Groups section](access-dag.md#groups) — onto their shared
   allocation, now that each group's protocol exists.
7. Apply the loop-scheduler workaround and remove dead aliases/allocations.
8. Verify partition outputs, token/view locality, unmarked buffer use after
   release, and at most one semaphore token in a loop's iter-args per group.

For each group, the rendering walk's `RenderState` keeps one ordered list of
token records — token value, semaphore, and optional owner — plus buffer views
cached by member and owner. The last token is used by default. An acquire
replaces any earlier record for its resolved owner and appends its result. A
node marked with `reuseTokenOwner` may instead use its owner's record without
changing the order. At a region boundary, only the token selected for
threading remains. A loop hold with no token iter-arg or result passes no token
through its boundary; a region with no crossing leaves the outer state
unchanged.

## Node mapping

| DAG node | Emitted form |
|---|---|
| `Acquire` | `nvws.semaphore.acquire`; records its result for the resolved owner and makes it the last token |
| `Release` | `nvws.semaphore.release` with the assigned completion kind and `arrive_count`; a marked node uses its owner's token when that token is not last |
| `Access` | `nvws.semaphore.buffer`, replayed view chain, and the retargeted access; a marked node builds the view from its owner's token, otherwise from the last token |
| sourceful alloc | explicit SMEM/TMEM store into the semaphore buffer view |
| `For`/`If` crossing | token init/result/yield position when the hold crosses the boundary |
| `ENTER`/`EXIT` | no operation of their own; the token iter-args, results, and yields added on the parent `for`/`if` realize the boundary |

POINT_OF_USE loop holds — acquires moved to the first body access — receive
no token iter-arg: their moved acquire creates the token in the body and the
closing release uses it there. If SYNC-DAG instead keeps a token iter-arg and
result because an eligibility check fails — printed as bare
`holdrule{gated}` unless the blocker is `trailing-use` or `result-consumed` —
signature rewriting adds the ordinary loop token slot. EMIT-IR performs no
stage comparison of its own.

## Schedule preservation

Generated protocol operations receive the owner and `loop.stage`/
`loop.cluster` already stored on their SYNC-DAG nodes. Access views inherit
the retargeted access's schedule. Stage offsets are emitted in the semaphore
operand named `stage`, to be resolved by `AssignStagePhase`; they are not
`loop.stage` values (see the
[NVWS-AWS terminology](../nvws-aws-overview.md#terminology)).

Circular members keep independent per-group DAGs while SYNC-DAG assigns
their stage offsets. EMIT-IR transcribes each access's offset to its
`nvws.semaphore.buffer` and each protocol node's offset to the
acquire/release `stage` operands, and only then folds the equivalent backings
and semaphore creates onto the shared multi-buffered allocation. Mixed-depth
TMEM groups likewise keep independent depths and share only a checked
subslice/reinterpretation of the physical allocation.

If the loop-scheduler workaround moves a release that has no assigned
pipeline stage, it may copy the schedule from the first MMA preceding it in
the block; other operations may sit between them. It does not otherwise
reschedule the DAG.

## Output contract

Every access rewritten by EMIT-IR uses a semaphore buffer view produced from
the token selected by the finalized DAG. Within one block, no *unmarked*
buffer view may follow a release of its token. A view recorded in
`reusedTokenBufferOps` may do so only because its SYNC-DAG node explicitly
proved owner-specific token reuse; this is the sole buffer-after-release
verifier exemption, not a general weakening. One loop carries at most one
semaphore token per group — circular SMEM backings are excluded from that
last verifier.
`LowerSemaphore` can therefore assign buffer stages and phases without
reconstructing ownership.

## Code map

[`InsertSemasEmitIR.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp):

- `emitBackingsAndCreates` and `emitEntryAcquires`
- `rewriteSignatures`
- `RenderState`, `getView`, `renderAccess`, `renderRegion`, and `renderChain`
- `foldCircularGroups`, `coalesceBackings`, and
  `coalesceMixedDepthTmemBackings`
- `workaroundLoopScheduler`
- `verifyNoUseAfterRelease`, `emitIR`, and the other post-emission verifiers
