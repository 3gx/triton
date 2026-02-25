# LowerSemaphore Pass

## Purpose

Convert high-level `nvws.semaphore.*` operations into concrete NVIDIA GPU
mbarrier-based synchronization primitives. This is the final lowering step
that produces hardware instructions.

**Source**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSemaphore.cpp` (~1138 lines)

## Pass Structure

`LowerSemaphore` operates in two modes depending on whether stage/phase
attributes have already been assigned:

### Phase 2 Mode (stage/phase absent)

When semaphore acquire/release/buffer ops lack stage/phase operands
(the typical case when invoked from `AutomaticWarpSpecialization`),
the pass runs a sub-pipeline before lowering:

```
1. combineSemaphores(loop)          -- merge semaphore pairs sharing consumers
2. multiBufferSemaphore(m, numStages) -- expand SMEM depth from 1 to numStages
3. AssignSemaphoreStagePhase()      -- invoke as sub-pass
4. Greedy pattern rewrite           -- semaphore ops -> mbarrier ops
```

The `requiresAssignSemaphoreStagePhase()` function checks if any semaphore
acquire/release/buffer ops are missing their stage or phase operands to
determine which mode to use.

### Direct Mode (stage/phase present)

When invoked standalone (e.g., in lit tests where stage/phase are already
assigned), skips steps 1-3 and goes directly to the pattern rewrite.

## Semaphore Combination (combineSemaphores)

Before lowering, semaphore pairs whose `SemaphoreAcquireOp`s share the same
**dominant consumer** (e.g., both TMA loads feed the same `tc_gen5_mma`) are
merged into a single combined semaphore pair.

This coalesces multiple mbarrier wait/arrive operations into one, reducing
synchronization overhead. The combined semaphore's buffer op returns all
buffers as a variadic result:

```mlir
// Before combining:
%buf_a = nvws.semaphore.buffer %sema_a, %tok_a -> !ttg.memdesc<128x64xf16>
%buf_b = nvws.semaphore.buffer %sema_b, %tok_b -> !ttg.memdesc<64x128xf16>

// After combining:
%buf_a, %buf_b = nvws.semaphore.buffer %combined, %tok -> (!ttg.memdesc<128x64xf16>, !ttg.memdesc<64x128xf16>)
```

**Pair matching** uses acquire-token lineage:
1. Find the `SemaphoreAcquireOp` user of `sem_cons`
2. Follow the token to its `SemaphoreReleaseOp` user
3. That release's `.getSemaphore()` is `sem_prod` (cross-release partner)

**TMEM exclusion**: Semaphores with `TMEMAllocOp` operands are excluded from
combining (they have their own depth management).

## Multi-Buffering (multiBufferSemaphore)

Expands SMEM semaphore allocations from depth=1 (set at insertion time) to
depth=`numStages` (pipeline depth, typically 3):

```mlir
// Before: depth=1
%buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem>
%sema = nvws.semaphore.create %buf : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16>], 1>

// After: depth=3
%buf = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem>
%sema = nvws.semaphore.create %buf : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16>], 3>
```

**TMEM skip**: TMEM allocations are skipped (their depth is already set
correctly at insertion time by `InsertTmemSemaphore`).

**Scales skip**: `TensorMemoryScalesEncodingAttr` buffers are always depth=1.

## Pending Count Computation

Each mbarrier is initialized with a **pending count** that determines how many
arrivals must occur before a waiting thread can proceed.

**Implementation**: `analyzeSemaphorePendingCount()` in
`third_party/nvidia/lib/Dialect/NVWS/IR/SemaphorePendingCount.cpp`

**Algorithm**:
1. For each `SemaphoreReleaseOp` user of the `SemaphoreCreateOp`:
   - Get the partition ID (must be exactly 1)
   - Count async ops in the release's `async_ops` array: each kind contributes +1
2. De-duplicate by partition: if the same partition appears multiple times,
   all its releases must contribute the same count
3. Sum contributions across distinct partitions
4. If sum is 0, default to 1

**Rule**: `pendingCount(sem) = sum over distinct releasing partitions P of |asyncOps(sem, P)|`

### Pending Count Examples

| Pattern | pendingCount |
|---------|-------------|
| 1 producer, 1 consumer, `[tma_load]` | 1 |
| 1 producer, 2 consumers, each `[none]` | 2 |
| 1 release with `[wgmma, none]` | 2 |
| 1 release with `[tmem_copy, none]` | 2 |
| 2 releases from different partitions, each `[none]` | 2 |

## Lowering Pattern: LowerSemaphoreCreate

The core rewrite pattern matches on `SemaphoreCreateOp` and rewrites the
entire constellation (create + all acquire/release/buffer users).

### Step 1: Create and Initialize mbarrier Array

```
mbar_alloc = local_alloc(i64, numStages)  // one i64 per stage
for i in 0..numStages:
    mbar_view = memdesc_index(mbar_alloc, i)
    init_barrier(mbar_view, pendingCount)
```

### Step 2: Handle TMA Loads (pre-collection)

**Before** erasing any semaphore ops, collect all TMA load information.
This is critical because the cross-release pattern means a release targeting
this semaphore may reference descriptor loads found via a *different*
semaphore's acquire token.

For each `SemaphoreReleaseOp` with `async_ops=[tma_load]`:
1. Follow the release's token back to the acquire -> buffer -> descriptor loads
2. Compute total `txCount` (sum of bytes across all loads)
3. Insert `BarrierExpectOp(mbar, txCount, pred)` before the first load
4. Replace each `nvws::DescriptorLoadOp` with `AsyncTMACopyGlobalToLocalOp(desc, indices, mbar, dst, pred)`
5. Replace each `nvws::DescriptorGatherOp` with `AsyncTMAGatherOp(..., mbar, dst, pred)`

TMA loads do **not** generate an arrive in `rewriteRelease` -- the TMA hardware
itself signals the mbarrier upon DMA completion.

### Step 3: Rewrite Acquire -> WaitBarrierOp

```
mbar_view = memdesc_index(mbar_alloc, stage)

// Phase bit extraction:
if single_phase:
    phase_bit = phase           // use directly
else:
    phase_bit = (phase >> stage) & 1   // extract per-stage bit

wait_barrier(mbar_view, phase_bit)
```

The **multiphase** extraction `(phase >> stage) & 1` selects the correct bit
from the 32-bit phase vector for the current stage. The **single-phase** mode
uses the scalar phase directly.

### Step 4: Rewrite Release -> Arrive/Commit

For each async kind in the release's `async_ops`:

| AsyncOp | Lowered To |
|---------|-----------|
| `none` | `arrive_barrier(mbar, 1)` |
| `wgmma` | `arrive_barrier(mbar, 1)` |
| `tc5mma` | `tc_gen5_commit(mbar)` |
| `tmem_copy` | `tc_gen5_commit(mbar)` |
| `tma_load` | *(nothing -- hardware signals)* |

### Step 5: Fence Detection

A `fence_async_shared` is inserted before an `arrive_barrier` when:
1. The release has `async_ops=[none]` (generic proxy)
2. The semaphore guards an **SMEM** buffer (not TMEM)
3. A **peer semaphore** sharing the same buffer has async releases (`tc5mma` or `tma_load`)

This ensures memory ordering between the generic proxy's synchronous writes
and the peer's asynchronous operations.

The peer check is pre-computed in `hasAsyncPeerBySema` **before** the greedy
rewrite begins, ensuring deterministic fence insertion regardless of rewrite order.

### Step 6: Rewrite Buffer -> MemDescIndexOp

```
for each buffer in semaphore.buffers:
    if buffer has TensorMemoryScalesEncoding:
        // Scales are always depth=1, no indexing needed
        replace with buffer directly
    else:
        // Strip leading dimension (numStages) by indexing
        view = memdesc_index(buffer, stage)  // shape: [N, M, K] -> [M, K]
        replace with view
```

The `replaceValueUsesAndPropagateType` helper recursively propagates type
changes through chains of `MemDescSubsliceOp`, `MemDescTransOp`, and
`MemDescReshapeOp`.

### Step 7: Cleanup

After all users are rewritten:

```
// Invalidate each mbarrier stage
for i in 0..numStages:
    mbar_view = memdesc_index(mbar_alloc, i)
    inval_barrier(mbar_view)

// Deallocate the mbarrier array
local_dealloc(mbar_alloc)
```

Token uses from `SemaphoreAcquireOp` are replaced with `ub::PoisonOp`.
All semaphore ops are erased in reverse topological order.

## Stage Cluster Annotations

**Every** created op (`WaitBarrierOp`, `ArriveBarrierOp`, `TCGen5CommitOp`,
`FenceAsyncSharedOp`, `MemDescIndexOp`, `BarrierExpectOp`) **must** be
annotated with `assignStageCluster(op, partitionIds, stageCluster, rewriter)`.

Without these annotations, downstream `PartitionLoops` and `ScheduleLoops`
passes cannot correctly place the operations in the pipeline schedule.

## Comparison with Old LowerAref

| Aspect | Old (LowerAref) | New (LowerSemaphore) |
|--------|-----------------|---------------------|
| Barrier model | Two arrays per aref (empty + full) | One array per semaphore |
| Phase extraction | Scalar: `(idx / depth) & 1` | Bit-vector: `(phase >> stage) & 1` or scalar |
| TMA handling | `insertArriveBarrier` with TMA case | Pre-collected `handleTMALoads` |
| Fence logic | `rewritePutExitOp` / `rewriteGetExitOp` | Unified `detectFenceNeeded` |
| Combination | `combineArefs` (merge arefs) | `combineSemaphores` (merge semaphore pairs) |
| Multi-buffer | `multiBufferAref` (expand allocs) | `multiBufferSemaphore` (expand allocs) |
| Stage assignment | Nested `AssignStagePhase` sub-pass | Nested `AssignSemaphoreStagePhase` sub-pass |

## Lit Test Coverage

19 CHECK-LABELs in `test/NVWS/lower_semaphore.mlir`:

| Test | Scenario |
|------|----------|
| `@basic` | Basic depth=2 create/acquire/release/buffer lowering |
| `@tma_load` | TMA load -> BarrierExpectOp + AsyncTMACopyGlobalToLocal |
| `@tma_gather` | TMA gather -> BarrierExpectOp + AsyncTMAGatherOp |
| `@tc5mma_commit` | tc5mma release -> TCGen5CommitOp |
| `@wgmma_pending_count` | `[wgmma, none]` -> pendingCount=2, two arrives |
| `@tmem_copy_pending_count` | `[tmem_copy, none]` -> pendingCount=2, commit+arrive |
| `@fence_needed` | Generic proxy + async peer -> FenceAsyncSharedOp |
| `@tmem_scales_passthrough` | Scales encoding -> no MemDescIndexOp |
| `@two_consumers` | 1 producer + 2 consumers -> EMPTY count=2, FULL count=1 |
| `@three_consumers` | 1 producer + 3 consumers -> EMPTY count=3 |
| `@cleanup_after_last_user` | InvalBarrierOp + LocalDeallocOp ordering |
| `@multi_buffer_trans_chain` | Multi-buffer with memdesc_trans chain |
| `@dual_tma_load_order` | Two TMA loads sharing one semaphore |
| `@phase2_multibuffer` | Full Phase2: depth=1 -> depth=3, complete pipeline |
| `@combine_two_tma_loads` | Two semaphore pairs combined into one |
| `@reuse_argument` | Fan-out with non-TMA producer |
| `@semaphore_not_in_loop` | Pre/post-loop TMEM semaphore ops |
| `@semaphore_buffer` | TMEM depth=2 with conditional consumer |
| `@semaphore_scale_mma_user` | Scaled MMA with TMEM ACC and scales |
