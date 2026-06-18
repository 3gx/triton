# NVWS Multi-Stage Phase Design V3

Status: DESIGN ONLY (18jun26).

Scope: `--nvws-assign-stage-phase` phase assignment when the same
`(partition, semaphore)` is acquired in more than one static `loop.stage`.

## Contract

For an affected `(partition, semaphore)`:

```text
affected := acquired in more than one static loop.stage
```

ASP carries one phase vector per acquiring `loop.stage`:

```text
PhaseKey = (partition, semaphore, optional stageLane)
```

`stageLane` is absent for the current behavior and present only for affected
keys.

If one key in a semaphore backing group is affected, the whole backing group
uses multiphase phase vectors:

```text
useSinglePhaseForGroup = false
```

This is required even if the old single-phase eligibility analysis would have
accepted the group.

## Phase Update Rule

For an affected acquire in static `loop.stage = S`, ASP selects:

```text
key = PhaseKey(partition, semaphore, stageLane = S)
```

It updates only that selected key:

```text
bit       = 1 << acquireStage
next      = phase[key] xor bit
waitPhase = (next >> acquireStage) & 1

phase[key] = next
acquire.phase = waitPhase
```

No other stage lane for the same semaphore is updated by this acquire.

Unaffected keys keep the existing ASP behavior.

## Generic Transform

The transform is generic over:

```text
number of static loop stages
number of acquires per stage
semaphore depth
constant authored offsets
```

For each affected `(partition, semaphore)`, ASP:

```text
1. proves the steady-state event sequence is path-invariant
2. computes each acquire's slotClass
3. verifies every touched slotClass has one owner loop.stage
4. creates one phase vector per owner loop.stage
5. updates only the selected stage lane for each acquire
```

No offset, stage count, semaphore name, or K/V shape is special-cased.

## Illustrative Accepted Example

This example is only an illustration of the generic rule.

Proof inputs:

```text
D = 4
A = 4
G = gcd(4, 4) = 4
```

Event table:

```text
event  loopStage  freshWrite  authoredOffset  advancePosition  classOffset  slotClass
A      0          yes         -1              1                0            0
B      0          yes         -1              2                1            1
C      1          yes         -1              3                2            2
D      1          yes         -1              4                3            3
```

Slot ownership:

```text
slotClass 0 -> loop.stage 0
slotClass 1 -> loop.stage 0
slotClass 2 -> loop.stage 1
slotClass 3 -> loop.stage 1
```

Every touched slot class has one owner stage, so ASP may split phase vectors by
stage.

Input shape:

```text
for {
  acq[-1] {stage 0}
  acq[-1] {stage 0}
  acq[-1] {stage 1}
  acq[-1] {stage 1}
}
```

Expected phase shape:

```text
for (..., phase_s0, phase_s1) {
  phase_s0 = update(phase_s0, slot[-1]) {stage 0}
  acq[-1] uses phase_s0                 {stage 0}

  phase_s0 = update(phase_s0, slot[-1]) {stage 0}
  acq[-1] uses phase_s0                 {stage 0}

  phase_s1 = update(phase_s1, slot[-1]) {stage 1}
  acq[-1] uses phase_s1                 {stage 1}

  phase_s1 = update(phase_s1, slot[-1]) {stage 1}
  acq[-1] uses phase_s1                 {stage 1}

  yield phase_s0, phase_s1
}
```

There are exactly two phase vectors because the key is acquired in exactly two
static stages. There is no update to `phase_s1` for the stage-0 acquires.

## Illustrative Rejected Example

This example uses the earlier four-acquire shape under the interpretation that
all four acquires are fresh-write and bracket operands are authored offsets.

Proof inputs:

```text
D = 4
A = 4
G = gcd(4, 4) = 4
```

Event table:

```text
event  loopStage  freshWrite  authoredOffset  advancePosition  classOffset  slotClass
A      0          yes         -1              1                0            0
B      0          yes         -2              2                0            0
C      0          yes         -1              3                2            2
D      1          yes          0              4                4            0
```

Slot ownership:

```text
slotClass 0 -> loop.stage 0 and loop.stage 1
slotClass 2 -> loop.stage 0
```

This fails the safety proof. ASP must fail for this affected key with:

```text
multi-stage phase split cannot prove disjoint stage-owned slots
```

## Disjoint-Slot Safety Proof

Per-stage phase vectors are correct only if different stages touch disjoint
physical slots over time.

For one fixed steady-state loop body:

```text
D = semaphore depth
A = number of State.stage advances in one loop body
G = gcd(D, A)
```

For each acquire event:

```text
classOffset = advancePosition + authoredOffset
slotClass   = positive_mod(classOffset, G)
```

`advancePosition` follows current ASP stage semantics: a fresh-write acquire
advances `State.stage` before authored offsets are applied.

Two events can touch the same physical slot over time iff:

```text
classOffset(e) == classOffset(f) mod G
```

Therefore, for every affected `(partition, semaphore, slotClass)`, all acquires
in that slot class must have the same static `loop.stage`.

If one slot class is acquired by more than one `loop.stage`, ASP must reject the
multi-stage split for that key with a diagnostic.

## Path-Invariance Gate

The GCD proof is valid only when the loop body has a fixed event sequence:

```text
fixed A
fixed authored offsets
fixed classOffset for every affected acquire event
```

ASP must reject the multi-stage split if control flow, nested loops, dynamic
offsets, or any other unproven shape can make `A` or an affected acquire's
`classOffset` path-dependent.

## State And Token Contract

ASP does not change:

```text
State.stage update
acquire.stage assignment
tokenLogicalStage
tokToStagePosMap
propagateStage()
buffer/release stage propagation
```

Buffer and release stage assignment continues to follow the acquire token.

This design changes only acquire phase computation.

## State.stage Diagnostic

The multi-stage phase split removes the phase-vector recurrence. It does not
prove that the unchanged shared `State.stage` SSA is pipeline-friendly.

After ASP emits stage and phase arithmetic, ASP must run a non-fatal diagnostic
scan. If affected phase arithmetic uses an `acquireStage` value produced in
another `loop.stage`, or if same-stage production cannot be proven, emit:

```text
multi-stage phase split uses acquire stage value produced in another or unknown loop.stage;
pipeline scheduling may fail until stage computation is made stage-local
```

Do not reject and do not change IR for this diagnostic.

## Non-Goals

1. Do not add cursor lanes.
2. Do not change `State.stage`.
3. Do not change token propagation.
4. Do not change buffer/release stage propagation.
5. Do not add circular-buffer knowledge to ASP.
6. Do not create one phase vector per slot class.
7. Do not update every stage lane for every acquire.
