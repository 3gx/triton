# NVWS multi-stage phase implementation plan v2

Status: PLAN ONLY (18jun26).

Design source: `fable/multi-stage-phase-design-v2.md`.

Goal: implement multi-stage phase-vector lanes in
`--nvws-assign-stage-phase` for a `(partition, semaphore)` acquired in more
than one static `loop.stage`.

## Non-Goals

1. Do not add cursor lanes.
2. Do not change `State.stage`.
3. Do not change `tokenLogicalStage`, `tokToStagePosMap`, or
   `propagateStage` semantics.
4. Do not change InsertSemas, LowerAref, pending counts, or mbarrier
   allocation.
5. Do not add circular-buffer knowledge to AssignStagePhase.
6. Do not support single-phase scalar state for affected multi-stage keys.

## Source Touchpoints

Primary file:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp
```

Expected implementation areas:

```text
PhaseKey
AssignStagePhase constructor / setup analysis
computeSinglePhaseEligibility call site
getPhaseKey helpers
analyzeSemaphoreUseInBlockImpl
assignStateInForOp / assignStateInIfOp threading through existing key set
assignStateInBlock phase update section
```

Do not touch:

```text
InsertSemas*
LowerAref.cpp
NVWS op lowering
semaphore pending-count logic
```

## M1: Extend PhaseKey

Change `PhaseKey` from:

```text
(partitionId, semaphoreOrder, semaphore)
```

to:

```text
(partitionId, semaphoreOrder, semaphore, optional phaseLane)
```

Rules:

1. `phaseLane` is absent for current behavior.
2. `phaseLane` is present only for an affected multi-stage key.
3. `operator==` must compare `partitionId`, `semaphore`, and `phaseLane`.
4. `operator<` must order by `partitionId`, `semaphoreOrder`, and
   `phaseLane`, with absent lane ordered before present lanes.
5. Existing single-stage keys must keep the old shape.

Implementation note: use either `std::optional<int>` or a sentinel such as
`-1`. The code must make absent-lane and lane `0` distinct.

## M2: Detect Multi-Stage Phase Keys

Before assigning state for one semaphore group, collect:

```text
AcquireStages[(partition, semaphore)] = set of static loop.stage values
```

For each acquire:

```text
partition set = acquire.ttg.partition if present, otherwise all group partitions
stage         = acquire.loop.stage if statically present
```

Store:

```text
phaseLanes[(partition, semaphore)] = sorted static stage set
```

Activation rule:

```text
if all acquires for the key have static loop.stage
and size(phaseLanes[(partition, semaphore)]) > 1:
  key is affected
else:
  key uses current behavior
```

Missing-stage rule:

```text
if a key would otherwise be affected but one acquire lacks static loop.stage:
  emit a diagnostic
  do not split that key
  keep current behavior for that key
```

This preserves existing non-multi-stage behavior.

## M3: Force Multiphase For Affected Groups

Current ASP computes `useSinglePhaseForGroup` once per semaphore backing group.

New rule:

```text
if any key in the group is affected:
  useSinglePhaseForGroup = false
else:
  use existing computeSinglePhaseEligibility()
```

Rationale: affected keys always use multiphase phase vectors. Since current ASP
chooses one phase mode per group, the entire group must use multiphase when any
key splits.

Implementation watchpoint: use the final group phase mode for initial phase
constants too. Current code computes `singlePhaseEligible`, assigns
`useSinglePhaseForGroup`, and separately uses `singlePhaseEligible` to choose
initial phase constants. For affected groups, both must follow the forced
multiphase decision:

```text
finalUseSinglePhase = !hasAffectedKey && computeSinglePhaseEligibility()
useSinglePhaseForGroup = finalUseSinglePhase

if finalUseSinglePhase:
  isReleased=true  -> 0
  isReleased=false -> 1
else:
  isReleased=true  -> 0
  isReleased=false -> -1
```

Do not initialize an affected group with single-phase constants if its final
mode is multiphase.

## M4: Produce Phase Keys For Analysis

Replace direct calls that collect one key:

```text
getPhaseKey(pid, semaphore)
```

with a helper:

```text
getPhaseKeys(pid, semaphore)
```

For an unaffected key:

```text
{ PhaseKey(pid, semaphore) }
```

For an affected key with lanes `{0, 1}`:

```text
{
  PhaseKey(pid, semaphore, phaseLane=0),
  PhaseKey(pid, semaphore, phaseLane=1)
}
```

Use this helper in:

```text
analyzeSemaphoreUseInBlockImpl
merge paths through scf.for / scf.if via existing OrderedPhaseKeys
```

No separate loop/if threading mechanism is needed. Existing
`summary.acquiredPhaseKeys` already drives iter_args, if results, yields, and
partition outputs.

## M5: Duplicate Phase Updates In assignStateInBlock

Current phase update shape:

```text
for pid in acquire partitions:
  key = PhaseKey(pid, semaphore)
  phaseState = getPhase(state, key)
  phaseState' = update(phaseState, acquireStage)
  acquire.phase = phase bit from phaseState'
```

New shape:

```text
for pid in acquire partitions:
  keys = getPhaseKeys(pid, semaphore)

  for key in keys:
    phaseState = getPhase(state, key)
    phaseState' = update(phaseState, acquireStage)
    state.phases[key] = phaseState'

    if key is the acquire-selected key:
      acquire.phase = phase bit from phaseState'
```

Acquire-selected key:

```text
if key is unaffected:
  selected key = PhaseKey(pid, semaphore)
else:
  selected key = PhaseKey(pid, semaphore, phaseLane=acquire.loop.stage)
```

Affected keys must use the multiphase formula:

```text
phaseBit = 1 << acquireStage
nextVec  = phaseVec xor phaseBit
useBit   = (nextVec >> acquireStage) & 1
```

Existing single-phase formula remains only for groups with no affected keys.

## M6: Emit Lane-Stage Phase Arithmetic

For affected keys, duplicated phase arithmetic must be emitted with:

```text
loop.stage = phaseLane
ttg.partition = acquire partition for that PhaseKey
```

Changing only the key type is not enough.

Implementation rule:

```text
createIntoPhaseForKey(key, opTy, args...)
```

For unaffected keys:

```text
use current createIntoPhase behavior
```

For affected keys:

```text
use the same insertion point
use phase partition ids for the key
set loop.stage to key.phaseLane
preserve the acquire cluster when legal
otherwise choose a legal cluster in that lane or emit a diagnostic
```

No stage arithmetic is duplicated. `acquireStage` remains the value computed by
the existing `State.stage` logic.

## M7: Diagnose Cross-Stage acquireStage Use

The design explicitly allows phase splitting even when `acquireStage` is
produced in a different `loop.stage` than a duplicated phase update.

Run this as a post-ASP diagnostic scan after stage and phase arithmetic have
been materialized. Do not try to predict it during emission.

Reason: after ASP has finished, the IR already contains the actual
`acquireStage` SSA values, duplicated phase-lane updates, loop/if threading, and
`loop.stage` attributes. The diagnostic can be a direct IR check instead of a
second model of stage-update logic.

Implementation should emit a non-fatal diagnostic once per affected key or
function when the scan finds this shape:

```text
multi-stage phase split uses acquire stage value produced in another loop.stage;
pipeline scheduling may fail until stage computation is made lane-local
```

Detection rule for the first implementation, after ASP emission:

```text
for each duplicated phase-lane update op with loop.stage = phaseLane:
  inspect the acquireStage SSA value used as the shift amount in:
    1 << acquireStage
    phaseVec >> acquireStage

  if acquireStage is defined by an op with loop.stage != phaseLane:
    emit diagnostic

  if acquireStage is a block argument, scf.for result, or scf.if result and the
  source cannot be proven to be produced in phaseLane:
    emit diagnostic

  if acquireStage is a constant or is proven produced in phaseLane:
    no diagnostic
```

This should be conservative. If the scan cannot prove same-lane production,
warn.

This is intentionally a warning/remark path. Do not reject and do not undo the
phase split for this case.

## M8: Preserve Stage And Token Behavior

Do not change:

```text
state.stage update
acquire.stage assignment
tokenLogicalStage assignment
tokToStagePosMap mappings
propagateStage()
release/buffer stage propagation
```

The feature only changes acquire phase computation.

## Tests

Add a focused lit file:

```text
test/NVWS/assign_stage_phase_multi_stage_phase_v2.mlir
```

Required cases:

1. **Two-stage same-key acquire**
   - One `(partition, semaphore)` is acquired in `loop.stage = 0` and
     `loop.stage = 1`.
   - CHECK that the output uses multiphase arithmetic.
   - CHECK that two phase-vector loop-carried values are threaded.
   - CHECK that each acquire uses the phase bit from its matching lane.
   - CHECK that duplicated phase arithmetic is annotated with the lane
     `loop.stage`.

2. **Unaffected key in same group**
   - Include another semaphore/key acquired in one stage only.
   - CHECK it does not get a `phaseLane`-style duplicate unless group
     multiphase arithmetic naturally applies because the group is multiphase.

3. **Token propagation unchanged**
   - Include buffer/release users.
   - CHECK buffer/release stages still come from token propagation behavior and
     are not independently lane-selected.

4. **Missing static stage fallback**
   - Add a diagnostic test if reduced IR can express the shape.
   - Expected behavior: emit diagnostic and keep current behavior for that key.
   - If reduced IR cannot express this shape, leave a comment in the test file
     or plan follow-up explaining why it is not representable.

5. **Cross-stage acquireStage diagnostic**
   - Add a reduced test derived from the real issue dump:

     ```text
     logs/02-17jun26-fp8-1/nvws-fp8-1/passes/065-before-nvws-assign-stage-phase.mlir
     ```

   - Preserve the essential shape only:
     - one semaphore key acquired by the same partition in `loop.stage = 0`
       and `loop.stage = 1`;
     - stage arithmetic for one acquire materialized in `loop.stage = 1`;
     - duplicated phase update for the other lane requiring that stage value;
     - post-ASP diagnostic that pipelining may fail because stage computation is
       not lane-local.
   - CHECK both the transformed IR and the diagnostic text.
   - Do not copy the full FA IR into the lit test; reduce it to the minimal
     semaphore/acquire/store/use structure that preserves the failure mechanism.

Existing tests that should remain unchanged:

```text
test/NVWS/assign_stage_phase.mlir
test/NVWS/assign_stage_phase_circular.mlir
test/NVWS/lower_semaphore*.mlir
```

Do not update existing CHECKs unless the change is caused by an actual
multi-stage `(partition, semaphore)` case in that test.

## Validation

Build first:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Then run targeted lit:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  test/NVWS/assign_stage_phase.mlir \
  test/NVWS/assign_stage_phase_circular.mlir \
  test/NVWS/assign_stage_phase_multi_stage_phase_v2.mlir
```

Then run the AWS guard requested for this workstream:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  test/TritonGPU/automatic-warp-specialization.mlir
```

If targeted tests pass, optionally run broader NVWS lit:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test/NVWS
```

Do not run pytest.

## Exit Criteria

1. Existing no-multi-stage inputs produce identical IR.
2. A multi-stage `(partition, semaphore)` gets one phase vector per static
   acquire stage.
3. Affected groups use multiphase.
4. Duplicated phase arithmetic is annotated with the lane `loop.stage`.
5. Acquire uses the phase bit from the matching lane.
6. Stage/token propagation is unchanged.
7. Cross-stage stage-value use emits a diagnostic and does not block the pass.
8. Required targeted lit tests pass after a fresh build.
