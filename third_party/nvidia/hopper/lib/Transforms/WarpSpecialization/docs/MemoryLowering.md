# Memory Lowering

Memory lowering creates the actual async copy operations that transfer data
between partitions. While code partitioning (`WSCodePartition.cpp`) identifies
cross-partition data dependencies and creates abstract channels, memory
lowering materializes the copies — inserting producer-side store/copy
operations and consumer-side load operations through shared memory or tensor
memory.

## Files

| File | Scope |
|------|-------|
| `WSLowerMem.cpp` | Core memory lowering, including TT-to-NVWS descriptor conversion and TMA-load fusion |
| `WSTMAStoreLowering.cpp` | Native-Meta late TMA store/reduce materialization, wait placement, and legacy early lowering |
| `TMEMAlloc1D.cpp` | Special case: 1D tensor communication via TMEM |

## Entry Points

**File**: `WSLowerMem.cpp`

In the current pipeline there is no single `insertAsyncCopy`
dispatcher. Copies are materialized by two mechanisms:

- **`doConvertDescriptorLoadsToNVWS`** — runs before buffer allocation, after
  AutoWS has passed its final eligibility bailout. It converts every
  tensor-producing `tt.descriptor_load` into `nvws.descriptor_load`, whose
  destination is an explicit SMEM memdesc and whose `txCount` is the CTA-local
  transfer size.
- **`doConvertDescriptorStoresToNVWS`** — runs beside descriptor-load
  conversion, after partition scheduling and task propagation but before
  buffer allocation. It converts `tt.descriptor_store/reduce` into an empty
  staging `ttg.local_alloc`, a producer-owned `ttg.local_store`, and a
  resultless `nvws.descriptor_store/reduce` that explicitly reads the staging
  memdesc. No TTNG TMA token exists yet.
- **`optimizeTMALoads`** — for buffered NVWS descriptor-load producers. Called from
  `insertAsyncComm` (`WSCodePartition.cpp`) during the code-partition phase.
  Emits `barrier_expect` + `AsyncTMACopyGlobalToLocalOp`. The copy uses the
  planner-rewritten NVWS destination allocation and rebuilds its stage view
  with the fused barrier's buffer index, followed by a `wait_barrier` before
  the consumers. The NVWS operation is then erased; there is no tensor-result
  or `local_store` cleanup path. See below.
- **`createLocalAlloc`** — for non-TMA (register/plain-load) producers. Called
  from `createBuffer` during the buffer-allocation phase; it creates the SMEM (or
  1D-TMEM) buffer and inserts the producer-side `LocalStoreOp` + consumer-side
  `LocalLoadOp`.

### `createBufferView` — Multi-Buffer Indexing

A shared helper that creates `MemDescIndexOp` subviews into multi-buffered
allocations. Given an accumulation counter (`accumCnt`), it computes:

```
bufferIdx = accumCnt % numBuffers
```

and returns a view of the corresponding buffer slot.

## TMA Barrier Fusion (`optimizeTMALoads`)

**File**: `WSLowerMem.cpp`

When multiple TMA descriptor loads feed the same consumer (e.g., two operand
loads for the same MMA), they are fused onto a single barrier:

1. **Group by consumer**: Channels sharing the same dominant consumer are
   grouped together.
2. **Shared barrier**: A single pair of barriers (ready + empty) is allocated
   for the group.
3. **Combined expect**: One `BarrierExpectOp` is emitted with the sum of the
   NVWS loads' `txCount` attributes.
4. **Multiple copies, one wait**: Each `AsyncTMACopyGlobalToLocalOp` references
   the shared barrier. The consumer issues a single `WaitBarrierOp`.

See [Barrier Fusion](BarrierFusion.md) for more details.

## TMA Store/Reduce Lowering

**Files**: `WSLowerMem.cpp`, `WSTMAStoreLowering.cpp`

Native Meta keeps `tt.descriptor_store/reduce` intact through
`PartitionSchedulingMeta`. After task propagation,
`doConvertDescriptorStoresToNVWS` produces this canonical form:

```text
tt.descriptor_store/reduce %desc[%coords] %value
  -> %staging = ttg.local_alloc()
  -> ttg.local_store %value, %staging
  -> nvws.descriptor_store/reduce %desc[%coords] %staging
```

The empty allocation plus explicit store is deliberate. Leaving a sourceful
allocation for `doBufferAllocation` to normalize would expose one channel
during discovery and then create a second TMA staging buffer. The canonical
form makes one allocation serve both the producer-to-store channel and the TMA
source. A dead single-use `ttg.convert_layout` forwarder is bypassed so the
staging write retains the producer value and ownership used by the legacy
plan.

The NVWS operations are resultless and have no TMA completion token. Their
optional internal token/index or barrier/predicate operands carry only the
native-Meta channel release for the staging buffer.

### Planning, Subtiling, and Completion

`doMemoryPlanner` recognizes the explicit source through
`TMAStoreLikeOpInterface`, sets `buffer.tmaStaging`/`buffer.copy`, and hoists a
loop-invariant staging allocation before its outermost enclosing `scf.for` or
`scf.while`. `GenerateSubtiledRegion` keeps planned TMA staging outside the
producer tile region and passes it as a tile operand, preserving one physical
allocation between producer and consumer.

Rotation policy is annotated after subtile generation. This is the same
boundary used for the legacy TTNG wait path: if subtiling has hidden the source
behind a region argument, both forms conservatively retain an adjacent wait.

During code partitioning, the abstract NVWS store/reduce is the completion
endpoint. Deferred release token/index operands are attached to it, then
`doTokenLowering` resolves them to barrier/predicate operands. Subtiled regions
containing deferred operands are inlined before token lowering so the token
users are visible.

### Late TTNG Materialization

After loop scheduling and subtiled-region cleanup,
`doMaterializeAndPlaceTMAStoreWaits` unconditionally creates the hardware-facing
representation:

```text
nvws.descriptor_store
  -> %token = ttng.async_tma_copy_local_to_global
  -> ttng.async_tma_store_token_wait %token

nvws.descriptor_reduce
  -> %token = ttng.async_tma_reduce
  -> ttng.async_tma_store_token_wait %token
```

Resolved completion barriers/predicates move from the NVWS operation to the
generated wait. With TMA-store pipelining enabled, the existing rotation
algorithm may move that wait to the next safe staging overwrite. With it
disabled, materialization still runs and leaves the wait adjacent to the
issue. No abstract descriptor store/reduce may survive this step.

### Legacy Compatibility Path

`doTMAStoreLowering` remains available for non-Meta, Meta-to-NVWS, standalone
pass tests, and explicit `early_tma_store_lowering=true`. It creates the legacy
`LocalAllocOp` + TTNG issue/token/wait sequence before AutoWS. Native-Meta
eligibility bailout also invokes it before removing WS metadata, preserving
ordinary non-WS code generation.

### `TMAStoreTokenWaitLowering` Pass

A separate pass (`NVGPUTMAStoreTokenWaitLoweringPass`) lowers the abstract
`TMAStoreTokenWaitOp` into concrete operations:
- `TMAStoreWaitOp`: waits for the async TMA store to complete
- `ArriveBarrierOp`: signals the associated barrier that the SMEM buffer
  is now free

Before lowering, additional passes annotate and reorder the waits to
maximize overlap with computation. See
[TMA Store Wait Pipeline](TMAStoreWaitPipeline.md) for the full
annotation → validation → reorder → lowering sequence.

## 1D TMEM Allocation

**File**: `TMEMAlloc1D.cpp`

The `TMEM1DAllocator` handles the special case of 1D tensor values that need
to be communicated between partitions via TMEM. TMEM is inherently 2D (M × N
matrix), so 1D values require expansion.

### Algorithm

1. **Expand shape**: The 1D input `[K]` is expanded to 2D `[M, N]` where
   `M × N ≥ K`, choosing dimensions compatible with TMEM layout constraints.

2. **Allocate**: A 2D `TMEMAllocOp` is created with the expanded shape.

3. **Producer side** (`TMEMStore1D`):
   - `ExpandDimsOp`: reshape 1D → 2D
   - Optional `ConvertLayoutOp` for TMEM-compatible layout
   - `TMEMStoreOp`: write to TMEM

4. **Consumer side** (`TMEMLoad1D`):
   - `TMEMLoadOp`: read from TMEM
   - `ReshapeOp`: 2D → 1D
   - `ConvertLayoutOp`: convert to target encoding

### Entry Point

`generate1DAllocations()` walks the function for ops with `tmem.start`
attributes and creates the 1D TMEM channel infrastructure.

### TMEM Subslicing Utilities

`TMEMUtils.h` also provides utilities for carving sub-regions from TMEM
allocations:

- **`sliceAndReinterpretMDTMEM`**: Creates `TMEMSubSliceOp` +
  `MemDescReinterpretOp` to extract a sub-region with a different N dimension
  or element type.
- **`createTMEMDesc`**: Creates a `MemDescType` with
  `TensorMemoryEncodingAttr` for given M/N dimensions.
