# EMIT-IR

## Rule

EMIT-IR materializes the finalized SYNC-DAG. It does not rediscover owners,
handoff edges, holds, pending counts, or stage offsets. Its one schedule
exception is the loop-scheduler workaround (`workaroundLoopScheduler`), which
splits qualifying `scf.if` operations so a release leading a branch is
hoisted before the `if` and a trailing acquire follows it — a shape the loop
scheduler otherwise mishandles. Two rarer variants hoist only the release (a
TMEM semaphore whose branch re-acquires later), or move an already-outside
release into the hoisted guard when the acquire leads the branch. A release
moved this way with no assigned
pipeline stage may inherit one from the first MMA that precedes it in the
block. Model terms are defined in the
[InsertSemas overview](overview.md#core-objects).

## Mechanical sequence

1. For TMEM groups that received semaphores, remove the old async-token
   operands and results from their allocation, loads, stores, and MMAs, and
   delete the token entries that become dead in loop and `if` signatures.
2. Allocate the planned multi-buffered backing and create semaphores with the
   DAG's initially-released and pending-count facts.
3. Emit the entry acquires.
4. Extend `scf.for`/`scf.if` signatures once for crossings that require a
   token result.
5. Walk the SYNC-DAG and render accesses, acquires, releases, and region
   yields.
6. Fold the circular and planned physical-alias groups onto their shared
   allocation, now that each group's protocol exists.
7. Apply the loop-scheduler workaround and remove dead aliases/allocations.
8. Verify partition outputs, token/view locality, hold boundaries, and at
   most one token in a loop's iter-args per physical semaphore group.

## Node mapping

| DAG node | Emitted form |
|---|---|
| `Acquire` | `nvws.semaphore.acquire`; becomes the current component token |
| `Release` | `nvws.semaphore.release` with the assigned completion kind and `arrive_count` |
| `Access` | `nvws.semaphore.buffer`, replayed view chain, and the retargeted access |
| sourceful alloc | explicit SMEM/TMEM store into the semaphore buffer view |
| `For`/`If` crossing | token init/result/yield position when the hold crosses the boundary |
| `ENTER`/`EXIT` | no operation of their own; the token iter-args, results, and yields added on the parent `for`/`if` realize the boundary |

POINT_OF_USE loop holds receive no token iter-arg: their moved acquire
creates the token in the body and the closing release consumes it there.

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
the token held by that access's owner. No buffer view follows its token's
release, and one loop carries at most one token per physical semaphore group
— circular SMEM backings are excluded from that last verifier.
`LowerSemaphore` can therefore assign buffer stages and phases without
reconstructing ownership.

## Code map

[`InsertSemasEmitIR.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp):

- `emitBackingsAndCreates` and `emitEntryAcquires`
- `rewriteSignatures`
- `renderAccess`, `renderRegion`, and `renderChain`
- `foldCircularGroups`, `coalesceBackings`, and
  `coalesceMixedDepthTmemBackings`
- `workaroundLoopScheduler`
- `emitIR` and the post-emission verifiers
