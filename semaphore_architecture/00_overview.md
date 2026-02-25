# Semaphore-Based Warp Specialization: Architecture Overview

## Document Index

| Document | Description |
|----------|-------------|
| [00_overview.md](00_overview.md) | This file. High-level rationale and pipeline comparison. |
| [01_aref_limitations.md](01_aref_limitations.md) | Why arefs cannot support TMEM and 3+ partition ownership. |
| [02_semaphore_ops.md](02_semaphore_ops.md) | The semaphore IR operations (dialect definition). |
| [03_insert_semaphore.md](03_insert_semaphore.md) | `InsertSemaphore` pass -- SMEM producer/consumer synchronization. |
| [04_insert_tmem_semaphore.md](04_insert_tmem_semaphore.md) | `InsertTmemSemaphore` pass -- TMEM ownership transfer. |
| [05_assign_stage_phase.md](05_assign_stage_phase.md) | `AssignSemaphoreStagePhase` -- the fresh-write rule algorithm. |
| [06_single_phase_optimization.md](06_single_phase_optimization.md) | Single-phase vs multiphase: eligibility analysis and proofs. |
| [07_lower_semaphore.md](07_lower_semaphore.md) | `LowerSemaphore` -- semaphore ops to mbarrier hardware primitives. |
| [08_proofs.md](08_proofs.md) | Formal proofs: multiphase necessity and shared stage counter. |

## What Changed

The warp specialization synchronization mechanism was redesigned from aref-based
 to semaphore-based. This is not a refactor -- it is an architectural change that removes 
limitations of the aref abstraction, particularly for tensor memory (TMEM) and
multi-partition (3+) ownership patterns.

## The Core Problem

Arefs model a two-party producer/consumer relationship. An aref wraps a memory
buffer and provides `put.enter`/`put.exit` (producer side) and `get.enter`/`get.exit`
(consumer side). This works for SMEM and TMEM data transfers between two partitions.

However, in general TMEM supports more general access patterns:

1. Any partition can both read and write (not fixed producer/consumer roles).
2. Ownership may rotate among >2 partitions
3. Buffer staging is governed by different predicates than unconditionally with arefs

Arefs embed buffer-staging into the put/get protocol itself -- `put.enter` returns
the next buffer stage, and stage is incremented unconditionally upon `put.enter`.

This coupling breaks correctness for TMEM patterns where
stage advancement and ownership transfer are independent.

See [01_aref_limitations.md](01_aref_limitations.md) for detailed examples from
the design rationale document.

## The Solution: Explicit Semaphores

Semaphores decouple three concerns that arefs entangle:

| Concern | Aref | Semaphore |
|---------|------|-----------|
| **Synchronization** | Embedded in put.enter/get.enter | Explicit acquire/release |
| **Buffer access** | Returned by put.enter/get.enter | Separate `semaphore.buffer` op |
| **Stage advancement** | Automatic at put.enter/get.enter | Computed by fresh write rule |

This decoupling enables:
- N-party ownership transfer (not just 2)
- Independent control of stage advancement and ownership
- Cleaner lowering to hardware mbarrier primitives

## Pipeline Comparison

### Aref Pipeline

```
  -> PartitionScheduling
  -> InsertAref                    // SMEM: create aref ops (put/get)
  -> InsertTmemAref                // TMEM: create aref ops (limited to 2 partitions)
  -> SCCP -> CSE
  -> LowerAref                    // Internally: combineArefs + multiBufferAref
     +-> AssignStagePhase          //   + AssignStagePhase (counter-based)
     +-> pattern rewrite           //   + aref ops -> mbarrier ops
  -> PartitionLoops
  -> LowerWarpGroup
  -> ScheduleLoops
```

### Semaphore Pipeline

```
  -> PartitionScheduling
  -> InsertSemaphore               // SMEM: create semaphore ops (acquire/release)
  -> InsertTmemSemaphore           // TMEM: create semaphore ops (ownership transfer)
  -> SCCP -> CSE
  -> LowerSemaphore                // Internally: combineSemaphores + multiBuffer
     +-> AssignSemaphoreStagePhase //   + fresh-write-based stage/phase assignment
     +-> pattern rewrite           //   + semaphore ops -> mbarrier ops
  -> PartitionLoops
  -> LowerWarpGroup
  -> ScheduleLoops
```

### Key Differences

| Aspect | Aref | Semaphore |
|--------|-----------|-----------------|
| Synchronization model | put.enter/put.exit, get.enter/get.exit | acquire/release |
| Buffer access | Embedded in enter ops + aref buffer op for TMEM | Separate `semaphore.buffer` op |
| Stage computation | Counter-based (increment per enter) | Advance when fresh-write occurs |
| Phase computation | Scalar flip on stage wrap (single-phase) | Bit-vector per stage (multiphase) or scalar flip (single-phase) |
| TMEM support | Limited to 2 partitions | N-partition ownership |
| Buffer staging | Automatic (embedded in protocol) | Explicit (decoupled from sync) |
| Multi-buffering | Deferred to LowerAref | Deferred to LowerSemaphore |
| Aref combination | combineArefs in LowerAref | combineSemaphores in LowerSemaphore |

## Semaphore Lifecycle

A semaphore goes through these stages:

```
[InsertSemaphore / InsertTmemSemaphore]
  1. SemaphoreCreateOp  -- allocate, set is_released, depth
  2. SemaphoreAcquireOp -- stage/phase
  3. SemaphoreBufferOp  -- stage
  4. SemaphoreReleaseOp -- stage, async_ops set

[LowerSemaphore -> combineSemaphores]
  5. Merge semaphore pairs feeding same consumer into combined pair

[LowerSemaphore -> multiBufferSemaphore]
  6. Expand SMEM depth -> depth=numStages

[LowerSemaphore -> AssignSemaphoreStagePhase]
  7. Assign stage (fresh-write rule) and phase (bit-vector or scalar) to all ops

[LowerSemaphore -> pattern rewrite]
  8. SemaphoreCreateOp  -> mbarrier alloc + InitBarrierOp per stage
  9. SemaphoreAcquireOp -> WaitBarrierOp(mbar[stage], phase_bit)
 10. SemaphoreBufferOp  -> MemDescIndexOp(buffer, stage)
 11. SemaphoreReleaseOp -> ArriveBarrierOp / TCGen5CommitOp / (nothing for TMA)
 12. Cleanup: InvalBarrierOp + LocalDeallocOp
```

## Source File Map

### Semaphore Passes

| File | Lines | Purpose |
|------|-------|---------|
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemaphore.cpp` | ~694 | Insert SMEM semaphore ops |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp` | ~815 | Insert TMEM semaphore ops |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignSemaphoreStagePhase.cpp` | ~1129 | Fresh-write-based stage/phase |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSemaphore.cpp` | ~1138 | Semaphore -> mbarrier |
| `third_party/nvidia/lib/Dialect/NVWS/IR/Ops.cpp` | | Semaphore op verifiers |
| `third_party/nvidia/lib/Dialect/NVWS/IR/SemaphorePendingCount.cpp` | ~98 | Pending count analysis |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/Utilities.cpp` | ~58 | Shared helpers |

### IR Definitions

| File | Purpose |
|------|---------|
| `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td` | Semaphore op definitions |
| `third_party/nvidia/include/Dialect/NVWS/IR/NVWSTypes.td` | SemaphoreType |
| `third_party/nvidia/include/Dialect/NVWS/IR/NVWSAttrDefs.td` | AsyncOp enum |
| `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td` | Pass definitions |

### Lit Tests

| File | Tests |
|------|-------|
| `test/NVWS/insert_semaphore.mlir` | 23 CHECK-LABELs |
| `test/NVWS/insert_tmem_semaphore.mlir` | 22 CHECK-LABELs |
| `test/NVWS/assign_semaphore_stage_phase.mlir` | 12 CHECK-LABELs |
| `test/NVWS/lower_semaphore.mlir` | 19 CHECK-LABELs |

### Aref Passes (removed)

| File | Purpose |
|------|---------|
| `InsertAref.cpp` | Insert aref ops (put/get) |
| `InsertTmemAref.cpp` | Insert TMEM aref ops (2-partition only) |
| `LowerAref.cpp` | combineArefs + multiBuffer + AssignStagePhase + lower to mbarrier |
| `AssignStagePhase.cpp` | Counter-based stage/phase assignment |
