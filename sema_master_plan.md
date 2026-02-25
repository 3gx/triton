# Master Plan: Lower Arefs to Semaphores

## Overview

Replace arefs with semaphores as intermediate representation. ALL NEW `.cpp`/`.h` files.
Existing `.cpp` files untouched until Phase 5. Phase 1 makes ADDITIVE changes to
existing `.td` files and `Ops.cpp` (new types/ops/impls — no existing code modified).
Each phase independently implementable and testable.

## Architecture

```
Current pipeline (UNCHANGED until Phase 5):
  InsertAref → InsertTmemAref → SCCP → CSE → LowerAref(AssignStagePhase + aref→mbarrier)

New passes (coexist with old):
  nvws-lower-aref-to-semaphore       (aref → sema)
  nvws-assign-semaphore-stage-phase  (observation-based stage/phase on sema ops)
  nvws-lower-semaphore               (sema → mbarrier)

Final pipeline (Phase 5 swap):
  InsertAref → InsertTmemAref → SCCP → CSE →
    LowerArefToSema → AssignSemaStagePhase → LowerSema →
    PartitionLoops → LowerWarpGroup → ScheduleLoops
```

## Implementation Order

Strict sequence. Each phase depends only on prior phases. Each has a build checkpoint
AND a lit test checkpoint. An independent agent implements each phase reading ONLY
this master plan + that phase's spec file.

| Phase | Spec file | Depends on | Build checkpoint | Lit test checkpoint |
|-------|-----------|-----------|-----------------|-------------------|
| 1 | `sema_phase1_spec.md` | Nothing | `ninja triton-opt` succeeds | `semaphore_ops.mlir` passes |
| 2 | `sema_phase2_spec.md` | Phase 1 | `ninja triton-opt` succeeds | `lower_aref_to_semaphore.mlir` passes |
| 3 | `sema_phase3_spec.md` | Phase 1 | `ninja triton-opt` succeeds | `assign_semaphore_stage_phase.mlir` passes |
| 4 | `sema_phase4_spec.md` | Phase 1 | `ninja triton-opt` succeeds | `lower_semaphore.mlir` passes |
| 5 | `sema_phase5_spec.md` | All 1-4 | `ninja triton-opt` succeeds | Full pipeline + rewritten tests pass |

**Phases 2, 3, 4 depend only on Phase 1** (semaphore ops must exist). They are
independent of each other and can be implemented in any order after Phase 1.
Phase 5 requires all prior phases.

**Note on Phase 4 lit tests:** Phase 4's input IR requires stage/phase already
assigned. For standalone testing, hand-write the stage/phase values in the test IR.
Phase 4 does NOT depend on Phase 3 at build time — only at integration time (Phase 5).

## Non-breakage guarantee

Phases 1-4: ALL existing lit tests pass at every checkpoint. No existing `.cpp` modified.
Phase 5: `AutomaticWarpSpecialization.cpp` modified + existing test CHECK lines rewritten.

## Key design decisions (context for all phases)

1. **Semaphore ops embed buffers** via `SemaphoreCreateOp(%buf)`. Buffer grouping in
   stage assignment uses this operand.

2. **Stage advancement rule:** Advance `%bufId` when `was_observed AND is_fresh_write`.
   Only `tc_gen5_mma(useD=true)` is NOT a fresh write. Everything else that writes is
   fresh. Code emits SCCP-ready constants; folding happens downstream (NOT in pipeline).

3. **Phase tracking:** Default MULTIPHASE (32-bit vector, flip per-stage bit at every
   acquire). Optimization to single-phase when A(s)=1 (proven in phase3 spec Appendix A).

4. **Shared stage counter:** All semaphores on same buffer share one `%bufId`.
   Proven necessary+sufficient by induction (phase3 spec Appendix B).

5. **`ThreadValue<T>`:** Generic template for threading values through scf.for/scf.if.
   Full implementation in phase3 spec.

6. **New pass names coexist with old:** `nvws-lower-aref` (old) and
   `nvws-lower-aref-to-semaphore` (new) are separate passes. `nvws-assign-stage-phase`
   (old) and `nvws-assign-semaphore-stage-phase` (new) are separate passes. Old passes
   untouched.

7. **No SCCP+CSE between new passes.** Observation-based arithmetic uses constants
   where possible (SCCP-ready). Folding happens downstream or at PTXAS level.

8. **Build command:** `ninja -C build/cmake.linux-x86_64-cpython-3.12/ triton-opt`

9. **No pytest by assistant.** Only lit tests via `triton-opt`. User runs pytest manually
   after Phase 5.

## Files created per phase

| Phase | New files |
|-------|-----------|
| 1 | Additive changes to `.td` files, `Ops.cpp`. 3 stub `.cpp` files + `CMakeLists.txt`. NEW: `test/NVWS/semaphore_ops.mlir` |
| 2 | Replace `LowerArefToSemaphore.cpp` stub with real impl. Add Python binding to `triton_nvidia.cc`. NEW: `test/NVWS/lower_aref_to_semaphore.mlir` |
| 3 | Replace `AssignSemaphoreStagePhase.cpp` stub with real impl. NEW: `SemaphoreUtilities.h`. Add Python binding. NEW: `test/NVWS/assign_semaphore_stage_phase.mlir` |
| 4 | Replace `LowerSemaphore.cpp` stub with real impl. Add Python binding. NEW: `test/NVWS/lower_semaphore.mlir` |
| 5 | `test/NVWS/aref_to_mbarrier_via_semaphore.mlir` + modified existing tests + `AutomaticWarpSpecialization.cpp` |

All `.cpp`/`.h` files under `third_party/nvidia/lib/Dialect/NVWS/Transforms/`.

## Spec file locations (in repo root)

- `sema_master_plan.md` — this file
- `sema_phase1_spec.md` — Phase 1: Semaphore ops/types (complete ODS, builders, traits)
- `sema_phase2_spec.md` — Phase 2: LowerArefToSemaphore (full rewrite pattern pseudocode)
- `sema_phase3_spec.md` — Phase 3: AssignSemaphoreStagePhase (includes full ThreadValue<T>
  implementation, observation algorithm, partition annotation contract, proofs in
  Appendices A and B)
- `sema_phase4_spec.md` — Phase 4: LowerSemaphore (mbarrier cleanup, fence detection
  logic, TMA handling, TMEM scales special case)
- `sema_phase5_spec.md` — Phase 5: Pipeline switch + test migration (15 test rewrites)
- `sema_appendix_a.md` — Proof: When is Multiphase Required?
- `sema_appendix_b.md` — Proof: Shared Stage Counter for N Semaphores (by induction)
