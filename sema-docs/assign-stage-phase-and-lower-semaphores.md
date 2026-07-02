# AssignStagePhase and LowerSemaphore

## Division of responsibility

`InsertSemas` assigns ownership, `loop.stage`/`loop.cluster`, pending and
arrive counts, and optional stage offsets. `AssignStagePhase` computes the
final buffer stage for each acquire, buffer, and release, plus the wait phase
for each acquire. `LowerSemaphore` consumes those values to emit hardware
barrier operations. Buffer stage, current buffer stage, stage offset, phase,
pipeline stage, fresh write, and semaphore group are defined in the
[NVWS-AWS terminology](nvws-aws-overview.md#terminology).

```text
current buffer stage (advanced on a fresh write) + stage offset
  -> AssignStagePhase: buffer stage, wait phase
  -> LowerSemaphore: mbarrier view, wait, arrive/commit
```

## LowerSemaphore order

1. When the group's effective depth exceeds 1 — `tt.num_stages` on the
   owning WS loop when present, else the `num-stages` option, maximized
   across WS loops sharing the group (`getSemaphoreGroupNumStages`) —
   semaphore groups whose release is fed by a TMA load and whose SMEM
   backings carry no `buffer.copy` are widened to that many copies
   (`multiBufferSemaphore`).
2. Run `NVWSAssignStagePhase` over every semaphore group.
3. Allocate and initialize one mbarrier per buffer stage.
4. Lower acquire to a barrier view plus wait.
5. Lower release to the corresponding view plus arrive, MMA commit, or TMA
   completion path.
6. Replace semaphore buffers with buffer-stage-indexed memdesc views, except
   TMEM scale encodings, which reuse the original buffer; remove semaphore IR.
7. Coalesce eligible, dominating, shape/width/bounds-compatible TMEM aliases
   around a zero-offset representative; then remove planning attributes even
   for groups that could not be coalesced.

Rewrites derived from an annotated semaphore operation copy its partition,
optional WS tag, `loop.stage`, and `loop.cluster` (`assignStageCluster`).
Barrier allocation/initialization and invalidation/deallocation receive no
metadata. Poison tokens — the placeholder values substituted for erased
semaphore tokens that WS loops still yield — are the one exception among the
scaffolding: they receive the semaphore's partition IDs, because
`PartitionLoops` requires every operation to carry one, but no WS tag or
pipeline stage.

## Buffer-stage assignment

Each semaphore group carries one current buffer stage. It advances only on
a fresh write — an acquire whose first reachable buffer access writes the
backing; other acquires leave it unchanged. The current buffer stage starts
at `depth - 1`, so the first fresh write advances to 0.

An initially released semaphore starts at phase 0. Any other semaphore starts
at phase 1 in single-phase mode and all-ones (`-1`) in multiphase mode; the
two modes are defined under [Phase assignment](#phase-assignment) below.

A stage-offset operand from `InsertSemas` is not a final buffer stage:

```text
baseStage   = advance_if_fresh_write(state.stage)
finalStage  = positive_mod(baseStage + stageOffset, depth)
state.stage = baseStage
```

Stage-offset mode is enabled only when the group contains an acquire with a
stage-offset operand but no phase operand. (`InsertSemas` never emits a
`phase` operand; one appears only in already-assigned or hand-written IR.) In that mode the unshifted
`baseStage` follows the acquire token through loops and `if` results, while
each release/buffer keeps its own stage offset. Outside that mode,
propagation replaces an existing release/buffer `stage` operand with the
current buffer stage.

The emitted buffer-stage arithmetic carries the semaphore operation's
`loop.stage`/`loop.cluster`. The pipeline stage determines *when* the
arithmetic runs; the buffer stage it computes determines *which* backing copy
and mbarrier the operation addresses.

## Phase assignment

Phase state was originally keyed by `(partition, semaphore)` (the
`egx/nvws-semaphore` branch). That key is sufficient only when all acquires
of the key execute in one `loop.stage`.

For a key whose acquires execute in more than one `loop.stage`, the pass:

1. requires the affected key's candidate acquires in one direct loop body,
   each carrying a static `loop.stage` (a missing one fails the pass),
   rejects the group's acquires in nested regions there, and analyzes all of
   the group's direct acquires in that body as one path-invariant sequence;
2. requires every affected stage offset to be constant;
3. computes `A`, the sequence's nonzero number of fresh-write advances per
   iteration; each direct acquire's position in that advance sequence,
   `advancePosition`; and `G = gcd(depth, A)`;
4. assigns each acquire the class
   `positive_mod(advancePosition + stageOffset, G)` — acquires in one class
   address the same buffer stages over the loop's lifetime;
5. fails the pass if one class is touched by more than one `loop.stage`;
6. keeps one phase word per `(partition, semaphore, loop.stage)`, holding one
   phase bit per buffer stage;
7. updates only the word belonging to the acquire's `loop.stage`.

The phase update for buffer stage `s` is:

```text
phaseWord ^= 1 << s
waitPhase = (phaseWord >> s) & 1
```

Worked trace — depth 4, one loop body with two fresh-write acquires (so
`A = 2` advances per iteration and `G = gcd(4, 2) = 2`), acquire X in
`loop.stage 0` and acquire Y in `loop.stage 1`, both with stage offset 0:

```text
classes:  X: positive_mod(0 + 0, 2) = 0     Y: positive_mod(1 + 0, 2) = 1
          each class is touched by ONE loop.stage -> the split is accepted

buffer stages over iterations (start depth-1 = 3, +1 per fresh write):
          iter 0        iter 1        iter 2
X:        3 -> 0        1 -> 2        3 -> 0 ...   only even stages {0, 2}
Y:        0 -> 1        2 -> 3        0 -> 1 ...   only odd  stages {1, 3}

phase words, one per (partition, semaphore, loop.stage):
word[ls0] covers X's stages {0, 2}:  iter 0: ^= 1<<0   iter 1: ^= 1<<2
word[ls1] covers Y's stages {1, 3}:  iter 0: ^= 1<<1   iter 1: ^= 1<<3
iter 2 revisits stage 0: the bit flips back — the phase, not the stage,
tells the wait apart from iteration 0's.
```

Acquires in one class address the same buffer stages forever (`stage ≡
class (mod G)`), so giving each `loop.stage` its own word is safe exactly
when no class is shared between two `loop.stage` values — that is what step
5 checks. Had Y carried stage offset `-1`, both acquires would land in
class 0 while sitting in different `loop.stage`s, and the pass fails.

The phase arithmetic is emitted in its selected `loop.stage` and partition. A
diagnostic reports an acquire whose buffer-stage SSA value is produced in
another or an unknown `loop.stage`. Stage offsets force multiphase mode; so
does a multi-`loop.stage` phase split anywhere in the group, independent of
the eligibility checks below.
Single-phase mode — one single-bit phase per `(partition, semaphore)` instead
of the multiphase per-buffer-stage word; only the mode choice is group-wide —
additionally requires `gcd(depth, advances per iteration) == 1`, which
guarantees that repeated advancing visits every copy before revisiting any.

Depth 1 is always single-phase and skips the single-phase eligibility checks
(stage offsets, WS loop, duplicate visits, nonzero advance, and the gcd
requirement).

## Structured control flow metadata

Buffer-stage and phase state is threaded through only the `for` and `if`
regions that use it. A buffer-stage result is stamped with all of the
semaphore group's partitions, and that partition set is extended when another
partition consumes its block argument; phase result partitions are inferred
from their final SSA values. After assignment, every invariant iter-arg in a
WS loop is removed, including invariants this pass did not introduce.

## Changes from `egx/nvws-semaphore`

| Area | Branch behavior | Current behavior |
|---|---|---|
| Stage-offset operands | Replaced by the base stage | Preserved as modulo offsets, only for a group in stage-offset mode |
| Phase key across `loop.stage` values | One phase value per partition/semaphore | Proven stage-disjoint phase word per `loop.stage` |
| Single-phase proof | Requires a WS loop and rejects duplicate semaphore/partition/virtual-stage visits (virtual stage = the un-wrapped advance count within one simulated iteration) | Retains those checks and adds the gcd requirement; stage offsets force multiphase |
| Buffer-stage propagation | Propagates the acquire's final buffer stage | Propagates the base stage; applies stage offsets at each user |
| Pending count | Re-derived during lowering | Required from `semaphore.create` |
| Release multiplicity | One count-1 arrive/commit per applicable async kind; no release-side arrival for TMA load | The assigned `arrive_count` controls each lowerable release contribution |
| Semaphore combining | Enabled before AssignStagePhase | Disabled; preserves the InsertSemas protocol and entry acquires |
| TMEM reuse | Separate lowered allocs | Eligible aliases coalesced only after dominance and compatible-view checks |

## Code map

- Buffer-stage and phase analysis:
  [`AssignStagePhase.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp),
  `AssignStagePhase::run`, `assignStateInBlock`, and `propagateStage`.
- Multi-`loop.stage` phase proof: the same file,
  `computeMultiStagePhaseLanes` and `proveStageDisjointSlotOwnership`.
- Barrier lowering:
  [`LowerAref.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp),
  `NVWSLowerSemaphore::runOnOperation`, `rewriteAcquire`, `rewriteRelease`, and
  `rewriteBuffer`.
