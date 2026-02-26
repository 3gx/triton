# Phase 5 Spec: Pipeline Switch + Test Migration

**Depends on:** ALL phases 1-4
**Checkpoint:** Full pipeline works. All lit tests pass. User runs pytest.
**This is the ONLY phase that modifies existing files.**

## Prologue: Plan correction (2026-02-26)

During implementation we found a coverage gap: per-pass semaphore lit tests
did not catch a `PartitionLoops` failure mode (`ttg.partition`/`ttg.warp_specialize.tag`
invariant break) that appeared only in integrated pipelines.

To preserve existing AREF regression coverage and avoid churn in legacy tests,
we keep `test/NVWS/lower_aref.mlir` and `test/NVWS/assign_stage_phase.mlir`
unchanged. Instead, we add semaphore integration tests that cover the same
end-to-end path and explicitly include `-tritongpu-partition-loops`.

This prologue supersedes the earlier "rewrite existing AREF tests" requirement
in this document.

## What this phase does

1. Replace `createNVWSLowerAref` in `AutomaticWarpSpecialization.cpp` with the 3 new passes
2. Keep legacy AREF lit tests unchanged
3. Add semaphore end-to-end/integration tests (including `partition-loops`)

## 1. Pipeline change

Modify `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
line 73:

```cpp
// OLD:
pm.addPass(createNVWSLowerAref({numStages}));

// NEW:
pm.addPass(createNVWSLowerArefToSemaphore({numStages}));
pm.addPass(createNVWSAssignSemaphoreStagePhase());
pm.addPass(createNVWSLowerSemaphore());
```

No SCCP+CSE between the 3 passes. The existing SCCP+CSE at lines 71-72 runs BEFORE
and cleans up InsertAref/InsertTmemAref output. The observation-based arithmetic uses
constants where possible (SCCP-ready). Folding happens downstream or at PTXAS level.

## 2. Rewrite `test/NVWS/lower_aref.mlir` (6 CHECK-LABEL tests)

Change RUN line (note: existing RUN uses `-split-input-file --allow-unregistered-dialect`):
```
// OLD:
// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref | FileCheck %s

// NEW:
// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore | FileCheck %s
```

CHECK lines MUST be rewritten. The new 3-pass pipeline produces structurally different
output than the old single-pass:
- Stage: observation-based pattern (`andi was_obs, is_fresh`, `select should_advance`)
  which for the common case (unconditional obs + fresh write) contains extra constant
  ops that fold downstream
- Phase: MULTIPHASE (`shli + xori`) instead of aref-style (`xori + select`)
- Op ordering: stage arithmetic by pass 2, mbarrier ops by pass 3

**Approach:** For each of the 6 test functions:
1. Run the OLD pipeline: `triton-opt test.mlir --nvws-lower-aref` and capture output
2. Run the NEW pipeline: `triton-opt test.mlir --nvws-lower-aref-to-semaphore --nvws-assign-semaphore-stage-phase --nvws-lower-semaphore`
3. Compare: the mbarrier ops (WaitBarrierOp, ArriveBarrierOp, etc.) should be
   functionally equivalent. The arithmetic producing stage/phase values may differ.
4. Write new CHECK lines matching the new output.

The 6 test functions:
- `@two_consumers`
- `@three_consumers`
- `@reuse_argument`
- `@lower_aref_buffer`
- `@aref_not_in_loop`
- `@load_scale_mma_user`

## 3. Rewrite `test/NVWS/assign_stage_phase.mlir` (9 CHECK-LABEL tests)

Existing RUN line: `triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase -cse | FileCheck %s`

After the switch:
- Input IR must be converted from aref ops to semaphore ops
- RUN line changes to `--nvws-assign-semaphore-stage-phase` (keep `-split-input-file --allow-unregistered-dialect -cse`)
- CHECK lines verify observation-based stage arithmetic and MULTIPHASE phase

**Approach:** For each of the 9 test functions:
1. Take the current aref input IR
2. Manually convert aref ops to semaphore ops (put.enter→acquire+buffer, etc.)
3. Run `--nvws-assign-semaphore-stage-phase` and capture output
4. Write CHECK lines matching the new output

The 9 test functions:
- `@two_consumers`
- `@aref_lowering`
- `@warp_specialize_tma_matmul`
- `@matmul_tma_acc_with_unconditional_user`
- `@assign_stage_buffer`
- `@attention_forward`
- `@matmul_tma_acc_with_conditional_user`
- `@matmul_tma_persistent_ws_kernel`
- `@for_loop_control_operand_ppg`

## 4. New end-to-end test

Create `test/NVWS/aref_to_mbarrier_via_semaphore.mlir`:
```
// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect \
//   --nvws-lower-aref-to-semaphore \
//   --nvws-assign-semaphore-stage-phase \
//   --nvws-lower-semaphore | FileCheck %s
```

Take 2-3 representative inputs from `lower_aref.mlir` and verify the final mbarrier
output. This is the "golden" test proving the full pipeline works.

## 5. Upstream regression verification

```bash
BUILD=build/cmake.linux-x86_64-cpython-3.12
TOPT=$BUILD/bin/triton-opt

# These tests MUST still pass (passes run before our changes):
$TOPT test/NVWS/insert_aref.mlir -split-input-file --allow-unregistered-dialect --nvws-insert-aref | FileCheck test/NVWS/insert_aref.mlir
$TOPT test/NVWS/aref-tmem-insertion.mlir -split-input-file --allow-unregistered-dialect -nvws-insert-tmem-aref -cse | FileCheck test/NVWS/aref-tmem-insertion.mlir
$TOPT test/NVWS/hoist_tmem_store.mlir -split-input-file --allow-unregistered-dialect -nvws-hoist-tmem-store | FileCheck test/NVWS/hoist_tmem_store.mlir
```

## 6. New per-phase tests also still pass

```bash
$TOPT test/NVWS/semaphore_ops.mlir -split-input-file | FileCheck test/NVWS/semaphore_ops.mlir
$TOPT test/NVWS/lower_aref_to_semaphore.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore | FileCheck test/NVWS/lower_aref_to_semaphore.mlir
$TOPT test/NVWS/assign_semaphore_stage_phase.mlir -split-input-file --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck test/NVWS/assign_semaphore_stage_phase.mlir
$TOPT test/NVWS/lower_semaphore.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck test/NVWS/lower_semaphore.mlir
```

## 7. E2E validation (user runs manually)

After ALL lit tests pass, user runs:
```bash
python -m pytest python/test/unit/language/test_warp_specialization.py -v
```
All 10 tests must pass on Hopper/Blackwell. The assistant does NOT run pytest.

## 8. Files modified

| File | Change |
|------|--------|
| `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp` | Replace line 73 with 3 new passes |
| `test/NVWS/aref_to_mbarrier_via_semaphore.mlir` | Add end-to-end semaphore pipeline checks |
| `test/NVWS/semaphore_partition_loops.mlir` | Add integration regression for `partition-loops` failure mode |

## 9. Optional cleanup (future)

After Phase 5 is verified:
- Remove old `LowerAref.cpp` (now unused in pipeline)
- Remove old `AssignStagePhase.cpp` (now unused)
- Remove old pass registrations from `Passes.td`
- This is NOT part of Phase 5. It's future cleanup.
