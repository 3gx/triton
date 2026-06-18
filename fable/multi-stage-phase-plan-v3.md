# NVWS Multi-Stage Phase Implementation Plan V3

Status: PLAN ONLY (18jun26).

Design source: `fable/multi-stage-phase-design-v3.md`.

Goal: implement one phase vector per static `loop.stage` for an affected
`(partition, semaphore)`, with a GCD proof that different stages own disjoint
physical slots.

## Boundaries

Do not change:

```text
InsertSemas
LowerAref
pending-count logic
mbarrier allocation
State.stage update
acquire.stage assignment
tokenLogicalStage
tokToStagePosMap
propagateStage()
buffer/release stage propagation
```

Do not add circular-buffer predicates to ASP.

## M1: Extend PhaseKey

Add an optional `stageLane` to `PhaseKey`:

```text
PhaseKey(partitionId, semaphoreOrder, semaphore, optional stageLane)
```

Rules:

1. `stageLane` is absent for current behavior.
2. `stageLane` is present only for affected keys.
3. Absent lane and lane `0` are distinct.
4. `operator==` and `operator<` include `stageLane`.

## M2: Detect Affected Keys

For each semaphore backing group, collect acquires by:

```text
baseKey = (partition, semaphore)
```

For each acquire record:

```text
partition set   = acquire.ttg.partition if present, otherwise all group partitions
loopStage       = static loop.stage
authoredOffset  = constant stage-only operand before ASP, otherwise 0
freshWrite      = isFirstUseFreshWriteAfterAcquire(acquire)
```

A key is affected if it is acquired in more than one static `loop.stage`.

If an affected key has missing static stage, dynamic authored offset, nested
group acquire sequence, or path-dependent group acquire sequence, fail with a
diagnostic.

## M3: Prove Stage-Disjoint Slot Ownership

Accept only one statically walkable steady-state loop body for the first
implementation.

Compute:

```text
D = semaphore depth
A = number of fresh-write acquires for the backing group in one loop body
G = gcd(D, A)
```

Walk the group acquires in source order:

```text
if acquire is fresh-write:
  advancePosition += 1

classOffset = advancePosition + authoredOffset
slotClass   = positive_mod(classOffset, G)
```

For each affected `(partition, semaphore, slotClass)`, require:

```text
all acquires in this slotClass have the same static loopStage
```

If not, fail with:

```text
multi-stage phase split cannot prove disjoint stage-owned slots
```

The slot class is proof metadata only. It is not the phase key.

## M4: Produce Stage-Lane Phase Keys

Add helpers:

```text
getPhaseKeys(partition, semaphore)
getSelectedPhaseKey(partition, acquire)
```

For unaffected keys:

```text
getPhaseKeys -> { PhaseKey(partition, semaphore) }
selected     -> PhaseKey(partition, semaphore)
```

For affected keys:

```text
getPhaseKeys -> one PhaseKey per static loopStage that acquires this key
selected     -> PhaseKey(partition, semaphore, stageLane = acquire.loopStage)
```

Use `getPhaseKeys` in `analyzeSemaphoreUseInBlockImpl` so existing `scf.for`
and `scf.if` threading adds one carrier per stage lane.

## M5: Force Multiphase

If any key in the backing group is affected:

```text
useSinglePhaseForGroup = false
```

Initial constants must use the final group mode:

```text
single-phase:
  isReleased=true  -> 0
  isReleased=false -> 1

multiphase:
  isReleased=true  -> 0
  isReleased=false -> -1
```

Every affected stage lane starts from the same initial multiphase constant for
that semaphore.

## M6: Emit Per-Stage Phase Updates

For each acquire and active partition:

```text
selectedKey = getSelectedPhaseKey(partition, acquire)
phaseState  = getPhase(state, selectedKey)
bit         = 1 << acquireStage
next        = phaseState xor bit
phase       = (next >> acquireStage) & 1

state.phases[selectedKey] = next
acquire.phase = phase
```

Do not update any other stage lane for the same semaphore.

For unaffected keys, keep current ASP phase-update behavior.

Emit affected phase arithmetic with:

```text
loop.stage = selectedKey.stageLane
partition  = selectedKey.partition
```

## M7: Post-ASP State.stage Diagnostic

After ASP emits stage and phase arithmetic, scan affected phase updates.

For each affected phase update in `selectedKey.stageLane`, inspect the
`acquireStage` value used by:

```text
1 << acquireStage
next >> acquireStage
```

Emit the diagnostic if `acquireStage` is produced in another `loop.stage`, or if
same-stage production cannot be proven:

```text
multi-stage phase split uses acquire stage value produced in another or unknown loop.stage;
pipeline scheduling may fail until stage computation is made stage-local
```

This diagnostic is non-fatal. Do not reject and do not rewrite IR for it.

## M8: Generic Behavior And Illustrative Examples

Implementation must be generic over:

```text
number of static loop stages
number of acquires per stage
semaphore depth
constant authored offsets
```

No implementation logic may special-case a concrete offset sequence, K/V shape,
or the examples below.

Generic accepted behavior:

```text
for each affected (partition, semaphore):
  compute slotClass for every acquire
  reject if any touched slotClass has acquires from more than one loopStage
  create one phase lane per static loopStage that acquires this key
  update only the selected stage lane for each acquire
```

Illustrative accepted example:

```text
D = 4
A = 4
G = gcd(4, 4) = 4
```

```text
event  loopStage  freshWrite  authoredOffset  advancePosition  classOffset  slotClass
A      0          yes         -1              1                0            0
B      0          yes         -1              2                1            1
C      1          yes         -1              3                2            2
D      1          yes         -1              4                3            3
```

Every touched slot class has one owner stage, so the split is legal.

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

Illustrative rejected example:

```text
D = 4
A = 4
G = gcd(4, 4) = 4
```

```text
event  loopStage  freshWrite  authoredOffset  advancePosition  classOffset  slotClass
A      0          yes         -1              1                0            0
B      0          yes         -2              2                0            0
C      0          yes         -1              3                2            2
D      1          yes          0              4                4            0
```

`slotClass 0` is touched by both stage 0 and stage 1, so the pass must fail for
this affected key:

```text
multi-stage phase split cannot prove disjoint stage-owned slots
```

## M9: Build And Lit Protocol

Implementation and validation order is mandatory:

1. Patch only the ASP implementation needed by this plan.
2. Build first:

   ```bash
   cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
   ninja triton triton-opt
   ```

3. Run all lit tests from the same build directory:

   ```bash
   /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test
   ```

4. The only acceptable lit failures are the known pre-existing failures:

   ```text
   TRITON :: Conversion/tritongpu_to_llvm_blackwell.mlir
   TRITON :: NVWS/assign_stage_phase_multi_stage_phase_v2.mlir
   TRITON :: TLX/tlx-verifier.mlir
   ```

5. After the lit run, stop and report. Do not run `run_nvws.sh`; the user will
   run the workload and provide further instructions.

Do not run pytest.

## Exit Criteria

1. Build succeeds.
2. Full lit run completes with no failures beyond the three known pre-existing
   failures listed in M9.
3. Stop and report after lit.
4. Unaffected keys keep current ASP behavior.
5. Affected keys use one phase vector per static `loop.stage`.
6. Affected acquires update only their selected stage lane.
7. The GCD proof rejects keys whose stages do not own disjoint slots.
