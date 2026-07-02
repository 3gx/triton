# EMIT-IR

## Rule

EMIT-IR materializes the finalized SYNC-DAG. It does not rediscover owners,
handoff edges, holds, retained-token eligibility, pending counts, or stage
offsets. A `Node::retainedTokenOwner` mark is a decision to render, not a
decision EMIT-IR may make. Its one schedule exception is the loop-scheduler
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

The rendering walk keeps the following state for each group (`RenderState`):
the current token, used by an unmarked access or release; the semaphore it came
from and its optional owner; a map from each retained owner to its earlier token
and semaphore; and buffer views cached by member and owner. Every acquire makes
its result the current token. If its resolved owner is a partition owner, the
walk also remembers it for retained use. When a region returns a token, that
token is the only one retained afterward. A loop hold
with no token iter-arg or result passes no retained token through its boundary.
A region with no crossing leaves the outer state unchanged. Retention proofs
therefore do not leak across control-flow boundaries.

## Node mapping

| DAG node | Emitted form |
|---|---|
| `Acquire` | `nvws.semaphore.acquire`; becomes the current token and, when its resolved owner is a partition owner, is remembered for retained use |
| `Release` | `nvws.semaphore.release` with the assigned completion kind and `arrive_count`; a marked node selects that owner's retained token when the current token belongs to another owner |
| `Access` | `nvws.semaphore.buffer`, replayed view chain, and the retargeted access; a marked node builds the view from that owner's retained token |
| sourceful alloc | explicit SMEM/TMEM store into the semaphore buffer view |
| `For`/`If` crossing | token init/result/yield position when the hold crosses the boundary |
| `ENTER`/`EXIT` | no operation of their own; the token iter-args, results, and yields added on the parent `for`/`if` realize the boundary |

POINT_OF_USE loop holds — acquires moved to the first body access — receive
no token iter-arg: their moved acquire creates the token in the body and the
closing release uses it there. If SYNC-DAG instead keeps a token iter-arg and
result for `cross-stage-final-acquire` — printed as
`holdrule{gated(cross-stage-final-acquire)}` — signature rewriting adds the
ordinary loop token slot. EMIT-IR performs no stage comparison of its own.

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
the token held by that access's owner. Within one block, no *unmarked* buffer
view may follow a release of its token. A view recorded in
`retainedBufferOps` may do so only because its SYNC-DAG node explicitly proved
same-owner retention; this is the sole buffer-after-release verifier
exemption, not a general weakening. One loop carries at most one semaphore
token per group — circular SMEM backings are excluded from that last verifier.
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
