# NVWS multi-stage phase design v2

Status: DESIGN ONLY (18jun26).

Scope: `--nvws-assign-stage-phase` phase assignment for a semaphore that is
acquired by the same partition in more than one static `loop.stage`.

This design intentionally does not introduce circular-buffer concepts into
AssignStagePhase. It does not change semaphore placement, token threading,
stage cursor threading, buffer/release stage propagation, LowerAref, pending
counts, or mbarrier allocation.

## Problem

AssignStagePhase currently carries phase state per `(partition, semaphore)`:

```text
PhaseKey = (partition, semaphore)
```

That is sufficient when every acquire of that key is scheduled in one
`loop.stage`.

It is insufficient when the same `(partition, semaphore)` is acquired in
multiple pipeline stages:

```mlir
for (..., %phaseVec, ...) {
  %stage0 = ...
  %phaseVec0 = update(%phaseVec, %stage0) ; {partition=1, loop.stage=0}
  acq %sem[%stage0, phase(%phaseVec0, %stage0)]
    {partition=1, loop.stage=0}

  %stage1 = ...
  %phaseVec1 = update(%phaseVec0, %stage1) ; {partition=1, loop.stage=1}
  acq %sem[%stage1, phase(%phaseVec1, %stage1)]
    {partition=1, loop.stage=1}

  yield ..., %phaseVec1, ...
}
```

The sequential phase history is correct, but the one loop-carried phase value is
produced in `loop.stage=1` and consumed by the next iteration's
`loop.stage=0`. The pipeliner cannot legally overlap those stages.

The issue is SSA granularity. A single `i32` phase vector cannot be "partly"
produced in multiple stages. Each SSA value has one defining stage.

## Requirements

1. If one `(partition, semaphore)` is acquired in `N` distinct static
   `loop.stage` values, AssignStagePhase must carry `N` phase-vector SSA values
   for that key.
2. The multi-stage case always uses multiphase phase vectors. Do not try to
   combine this design with single-phase scalar phase state.
3. If every `(partition, semaphore)` is acquired in at most one static
   `loop.stage`, AssignStagePhase must keep the current behavior.
4. `State.stage` remains exactly the existing single shared stage cursor for the
   semaphore backing group.
5. Buffer and release stage assignment continues to follow the acquire token via
   the existing token propagation path.
6. The design is generic NVWS semaphore phase handling. It must not inspect
   circular-buffer metadata.

## Detection

Before assigning phases for one semaphore group, ASP computes:

```text
AcquireStages[(partition, semaphore)] = set of static loop.stage values
```

For each acquire:

```text
partition set = acquire.ttg.partition if present, otherwise all group partitions
stage         = acquire.loop.stage
```

If an acquire for an affected multi-stage key has no static `loop.stage`, it
cannot participate in the split. That is malformed for this design and should
be diagnosed. Acquires for keys that do not need multi-stage splitting keep the
current behavior.

Phase-key selection:

```text
if size(AcquireStages[(partition, semaphore)]) <= 1:
  use PhaseKey(partition, semaphore)

if size(AcquireStages[(partition, semaphore)]) > 1:
  use PhaseKey(partition, semaphore, phaseLane)
  for every phaseLane in AcquireStages[(partition, semaphore)]
```

`phaseLane` is a static `loop.stage` value.

If any `(partition, semaphore)` in the group needs phase lanes, the whole
semaphore group must use multiphase mode:

```text
useSinglePhaseForGroup = false
```

This is consistent with the current ASP model where single-phase eligibility is
group-wide.

## Core Rule

For a multi-stage `(partition, semaphore)`, every logical acquire updates every
phase lane for that key.

The acquire itself uses only the lane matching the acquire's own static
`loop.stage`.

For two lanes:

```mlir
for (..., %phaseVec_s0, %phaseVec_s1, ...) {
  %stage0 = ...

  %phaseVec_s0_a = update(%phaseVec_s0, %stage0)
    {partition=1, loop.stage=0}
  %phaseVec_s1_a = update(%phaseVec_s1, %stage0)
    {partition=1, loop.stage=1}

  acq %sem[%stage0, phase(%phaseVec_s0_a, %stage0)]
    {partition=1, loop.stage=0}

  %stage1 = ...

  %phaseVec_s0_b = update(%phaseVec_s0_a, %stage1)
    {partition=1, loop.stage=0}
  %phaseVec_s1_b = update(%phaseVec_s1_a, %stage1)
    {partition=1, loop.stage=1}

  acq %sem[%stage1, phase(%phaseVec_s1_b, %stage1)]
    {partition=1, loop.stage=1}

  yield ..., %phaseVec_s0_b, %phaseVec_s1_b, ...
}
```

Both lanes carry the same logical phase history. They are duplicated so that
each pipeline stage consumes a phase value produced by the same pipeline stage
in the previous iteration.

## Multiphase Formula

The existing multiphase update is:

```text
phaseVec' = phaseVec xor (1 << stage)
phaseBit  = (phaseVec' >> stage) & 1
```

The multi-stage design applies the same formula independently to each lane:

```mlir
%bit_s0      = 1 << %stage       ; {loop.stage=0}
%phase_s0'   = %phase_s0 ^ %bit_s0
%phaseBit_s0 = (%phase_s0' >> %stage) & 1

%bit_s1      = 1 << %stage       ; {loop.stage=1}
%phase_s1'   = %phase_s1 ^ %bit_s1
%phaseBit_s1 = (%phase_s1' >> %stage) & 1
```

The `loop.stage=0` acquire uses `%phaseBit_s0`. The `loop.stage=1` acquire
uses `%phaseBit_s1`.

The input `%stage` value is the stage value computed by existing ASP logic. This
design does not duplicate or re-key `State.stage`.

## Acquire Stage Availability

The duplicated phase update for every lane uses the acquire's computed stage
value:

```text
phaseVecLane' = phaseVecLane xor (1 << acquireStage)
```

This design does not require `acquireStage` to be produced in the same
`loop.stage` as every duplicated phase update. If `acquireStage` is produced in
a different pipeline stage, ASP still emits the split phase-vector updates.

That case is a known performance limitation, not a correctness reason to avoid
the phase split. The resulting IR may carry a cross-stage dependency through the
stage value, and the pipeliner may fail to overlap the loop. ASP should emit a
diagnostic so the reason is visible:

```text
multi-stage phase split uses acquire stage value produced in another loop.stage;
pipeline scheduling may fail until stage computation is made lane-local
```

This keeps v2 narrowly scoped: it fixes phase-vector SSA granularity without
also solving stage-cursor SSA granularity.

## Loop And If Threading

ASP already threads one extra loop iter_arg / if result per collected
`PhaseKey`.

This design keeps that mechanism. It only changes which phase keys are
collected:

```text
single-stage key:
  PhaseKey(partition, semaphore)

multi-stage keys:
  PhaseKey(partition, semaphore, phaseLane=0)
  PhaseKey(partition, semaphore, phaseLane=1)
  ...
```

For `scf.for`:

```text
input phase-vector lane values become iter_args
body updates each lane in program order
yield returns every lane's final phase vector
```

For `scf.if`:

```text
each branch carries the same set of required phase keys
branches that do not update a key forward the incoming value unchanged
if results carry one phase vector per key
```

This is compositional because the duplicated phase updates remain inside the
same SCF control path as the logical acquire. The transform must not speculate
phase updates across control-flow boundaries.

## Scheduling Contract

For each duplicated phase update:

```text
loop.stage = phaseLane
partition  = the acquire partition for this PhaseKey
```

Cluster placement should preserve the original acquire's relative cluster when
legal. If that placement is not legal for the target `phaseLane`, the
implementation must choose a legal cluster in that stage. If it cannot do so,
the implementation should emit a diagnostic that the produced IR is expected to
remain non-pipelineable for that shape.

Changing `PhaseKey` alone is insufficient. The arithmetic that produces a lane
must also be emitted with the lane's `loop.stage`.

## Stage And Token Contract

`State.stage` is unchanged:

```text
one shared stage cursor per semaphore backing group
same update rule as today
same loop/if threading as today
```

Acquire stage assignment is unchanged:

```text
acquire.stage = stage computed by existing ASP logic
```

After the acquire receives its stage and lane-selected phase bit, buffer and
release stage assignment continues through existing token propagation:

```text
acquire token -> buffer/release stage propagation
```

This design only changes how the acquire phase bit is computed.

## Correctness Sketch

For one affected `(partition, semaphore)`, let the sequential multiphase vector
after `k` acquires be `P[k]`.

Invariant:

```text
after processing k logical acquires, every phase lane L contains P[k]
```

Base:

```text
all lanes are initialized from the same initial phase vector for the semaphore
```

Step:

```text
logical acquire k uses computed stage S
every lane applies P[k+1] = P[k] xor (1 << S)
therefore every lane contains the same sequential phase vector P[k+1]
```

Use:

```text
an acquire scheduled in loop.stage L consumes lane L
lane L contains the correct sequential phase vector for that acquire
```

Pipeline recurrence:

```text
lane L is yielded from arithmetic scheduled in loop.stage L
next iteration's acquire in loop.stage L consumes lane L
there is no forced loop-carried dependency through the phase vector from another
pipeline stage
```

This does not claim the loop is always pipelineable. If the stage value used by
the phase update is produced in another `loop.stage`, that separate stage-value
dependency can still block pipelining. That case is diagnosed by the acquire
stage availability rule above.

## Non-Goals

1. Do not add cursor lanes.
2. Do not change `State.stage`.
3. Do not change buffer/release stage propagation.
4. Do not add circular-buffer semantics to ASP.
5. Do not support single-phase mode for multi-stage acquire keys.
6. Do not introduce phase state per physical slot.
7. Do not reject phase splitting only because the acquire stage value is
   produced in another `loop.stage`; warn and proceed.

## Implementation Checklist

1. Extend `PhaseKey` with an optional `phaseLane`.
2. Include `phaseLane` in `PhaseKey::operator==` and `operator<`.
3. Detect `AcquireStages[(partition, semaphore)]`.
4. Force group multiphase if any key has more than one acquire stage.
5. Collect every lane key for a multi-stage `(partition, semaphore)`.
6. In `assignStateInBlock`, update every lane for each logical acquire of a
   multi-stage key.
7. Assign `acquire.phase` from only the lane matching `acquire.loop.stage`.
8. Emit each lane's phase arithmetic with `loop.stage = phaseLane`.
9. Keep the existing `State.stage`, `tokenLogicalStage`, `tokToStagePosMap`, and
   `propagateStage` semantics unchanged.
10. If a duplicated phase update consumes an acquire stage value produced in a
    different `loop.stage`, emit a diagnostic that pipelining may fail because
    stage computation is not lane-local.
11. Existing inputs with no multi-stage `(partition, semaphore)` must produce
    identical IR except for changes caused by explicitly unsupported malformed
    inputs.
