# Semaphore IR Operations

This document describes the MLIR operations that form the semaphore abstraction,
defined in `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td`.

## Operation Summary

| Operation | Purpose | Operands | Results |
|-----------|---------|----------|---------|
| `nvws.semaphore.create` | Create semaphore guarding a buffer | `buffers: Variadic<MemDesc>`, `is_released: I1` | `SemaphoreType` |
| `nvws.semaphore.acquire` | Wait for semaphore, gain ownership | `semaphore`, optional `stage`, `phase` | `AsyncToken` |
| `nvws.semaphore.release` | Signal semaphore, relinquish ownership | `semaphore`, `token`, optional `stage`, `async_ops` | (none) |
| `nvws.semaphore.buffer` | Get buffer view at current stage | `semaphore`, `token`, optional `stage` | `Variadic<MemDesc>` |

## SemaphoreType

Defined in `NVWSTypes.td`:

```
nvws.semaphore<[!ttg.memdesc<2x128x64xf16, ...>], 2>
                ^-- base types (buffer types)     ^-- numStages (depth)
```

- `baseType`: `TypeArrayAttr` -- the types of buffers this semaphore guards.
  A semaphore can guard multiple buffers (after semaphore combination).
- `numStages`: `int` (default 1) -- the pipeline depth. Each stage has its own
  mbarrier slot. At insertion time, depth=1 for SMEM; after multi-buffering in
  `LowerSemaphore`, depth increases to `numStages` (typically 3).

## SemaphoreCreateOp

```mlir
%empty = nvws.semaphore.create %buf true  : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, ...>], 1>
%full  = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, ...>], 1>
```

- `buffers`: The shared-memory allocation(s) this semaphore guards. A semaphore
  has **exclusive ownership** of its buffers -- the verifier enforces that buffers
  are only used by `SemaphoreCreateOp` or `LocalDeallocOp`.
- `is_released`: If `true`, the semaphore is initially in the released state (a
  waiting acquire can proceed immediately). If `false`, the first acquire will
  block until a release occurs.

**Convention**: For a producer/consumer pair on the same buffer:
- The "empty" semaphore (`is_released=true`) is acquired by the **producer**
  (buffer is free to write).
- The "full" semaphore (`is_released=false`) is acquired by the **consumer**
  (data is ready to read).

**Verifier** (`SemaphoreCreateOp::verify()` in `Ops.cpp`):
1. Exclusive buffer ownership -- buffers used only by this create or dealloc.
2. Shape consistency -- all buffers have non-empty shapes; leading dimensions match.
3. Depth consistency -- leading dimension equals `numStages`.
4. Release async-op deduplication -- no duplicate async kinds in any release user.
5. Pending-count analysis -- partition contributions are consistent across releases.

## SemaphoreAcquireOp

```mlir
%tok = nvws.semaphore.acquire %sema : !nvws.semaphore<...> -> !ttg.async.token

// After AssignSemaphoreStagePhase assigns stage/phase:
%tok = nvws.semaphore.acquire %sema[%stage, %phase] : !nvws.semaphore<...> -> !ttg.async.token
```

- `semaphore`: The semaphore to acquire.
- `stage` (optional): The pipeline stage index. Set by `AssignSemaphoreStagePhase`.
- `phase` (optional): The phase value (bit-vector or scalar). Set by `AssignSemaphoreStagePhase`.
- Returns an `AsyncToken` that proves ownership and is consumed by subsequent
  `semaphore.buffer` and `semaphore.release` ops.

Stage and phase start as absent (after insertion) and are filled in by the
`AssignSemaphoreStagePhase` pass.

Implements `SemaphoreStageInterface` for uniform `getStage()`/`setStage()`.

## SemaphoreReleaseOp

```mlir
nvws.semaphore.release %sema, %tok [tma_load] : !nvws.semaphore<...>, !ttg.async.token
nvws.semaphore.release %sema, %tok [tc5mma]   : !nvws.semaphore<...>, !ttg.async.token
nvws.semaphore.release %sema, %tok [none]     : !nvws.semaphore<...>, !ttg.async.token
```

- `semaphore`: The semaphore to release. **Cross-release pattern**: the producer
  acquires the "empty" semaphore but releases the "full" semaphore (and vice versa).
- `token`: The ownership token from the matching acquire.
- `stage` (optional): Set by `AssignSemaphoreStagePhase`.
- `async_ops`: Array of `AsyncOp` enum values describing what asynchronous
  operations must complete before the release takes effect.

**AsyncOp Kinds** (from `NVWSAttrDefs.td`):

| Kind | Arrival Mechanism | Description |
|------|-------------------|-------------|
| `none` | `ArriveBarrierOp(mbar, 1)` | Synchronous operation (generic proxy) |
| `tma_load` |  Hardware (no arrive needed) | TMA DMA transfer |
| `tc5mma` | `TCGen5CommitOp(mbar)` | Blackwell TC5 MMA |
| `tmem_copy` | `TCGen5CommitOp(mbar)` | TMEM copy operation |
| `cp_async` | (reserved) | CP.Async operation |
| `wgmma` | `ArriveBarrierOp(mbar, 1)` | Hopper WGMMA |

**Verifier**: Ensures no duplicate async kinds in the `async_ops` array.

## SemaphoreBufferOp

```mlir
%buf = nvws.semaphore.buffer %sema, %tok : !nvws.semaphore<...>, !ttg.async.token
    -> !ttg.memdesc<128x64xf16, ...>
```

- `semaphore`: The semaphore to access.
- `token`: Ownership proof from the acquire.
- `stage` (optional): Pipeline stage index for multi-buffered access.
- Returns a view of the underlying buffer at the given stage. The leading
  dimension (numStages) is stripped from the result type.

For multi-buffered semaphores (depth > 1), the buffer op returns a sub-view
at the given stage. For depth=1, it still strips the leading dimension of size 1.

For combined semaphores guarding multiple buffers, the result is variadic:
```mlir
%buf0, %buf1 = nvws.semaphore.buffer %combined, %tok
    : !nvws.semaphore<[!ttg.memdesc<3x128x64xf16>, !ttg.memdesc<3x64x128xf16>], 3>
    -> (!ttg.memdesc<128x64xf16>, !ttg.memdesc<64x128xf16>)
```

**Verifier**: Each result type must be a valid "slice" of the corresponding base
type (leading dimension removed, remaining dimensions match).

## Cross-Release Pattern

The semaphore protocol uses a **cross-release** pattern where the acquire and
release target **different** semaphores:

```
Producer iteration:
  acquire EMPTY  -> get buffer -> write data -> release FULL
                                                ^^^^^^^^^^^^
                                                releases the OTHER semaphore
Consumer iteration:
  acquire FULL   -> get buffer -> read data  -> release EMPTY
                                                ^^^^^^^^^^^^^
                                                releases the OTHER semaphore
```

This is essential for correctness: releasing the `full` semaphore signals the
consumer that data is ready; releasing the `empty` semaphore signals the producer
that the buffer is free for reuse.

The token from `acquire EMPTY` flows to `release FULL`, linking the producer's
write to the consumer's notification. The token from `acquire FULL` flows to
`release EMPTY`, linking the consumer's read to the producer's free notification.

## Partition Annotations

All semaphore ops carry partition annotations (`ttg.partition` attribute) that
specify which warp group executes the operation. These annotations are set during
insertion and are required for downstream `PartitionLoops` and `ScheduleLoops`
passes to correctly split the code among warp groups.

## Example: Complete SMEM Producer/Consumer

After `InsertSemaphore` runs on a TMA load consumed by MMA:

```mlir
// Before the warp-specialized loop:
%buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem>
%empty = nvws.semaphore.create %buf true  : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, ...>], 1>
%full  = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, ...>], 1>

// Inside the loop:
// Producer (partition 2):
%tok_e = nvws.semaphore.acquire %empty : ... -> !ttg.async.token  {ttg.partition = 2}
%pbuf  = nvws.semaphore.buffer %empty, %tok_e : ... -> !ttg.memdesc<128x64xf16>  {ttg.partition = 2}
nvws.descriptor_load %desc, %indices, %txCount, %pbuf  {ttg.partition = 2}
nvws.semaphore.release %full, %tok_e [tma_load]  {ttg.partition = 2}

// Consumer (partition 1):
%tok_f = nvws.semaphore.acquire %full : ... -> !ttg.async.token  {ttg.partition = 1}
%cbuf  = nvws.semaphore.buffer %full, %tok_f : ... -> !ttg.memdesc<128x64xf16>  {ttg.partition = 1}
// ... cbuf feeds tc_gen5_mma ...
nvws.semaphore.release %empty, %tok_f [tc5mma]  {ttg.partition = 1}
```
