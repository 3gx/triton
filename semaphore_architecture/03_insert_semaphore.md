# InsertSemaphore Pass -- SMEM Producer/Consumer Synchronization

## Purpose

The `InsertSemaphore` pass transforms cross-partition dataflow in warp-specialized
loops into explicit semaphore-based synchronization. When a value is produced in
one partition and consumed in a different partition, the data must travel through
shared memory (SMEM). This pass inserts the semaphore `acquire`/`release`
sequences and the SMEM loads/stores required to make that transfer safe.

The pass operates on `scf.for` loops that carry the `tt.warp_specialize` attribute
and have partition annotations (`ttg.partition`). It is the SMEM counterpart to
`InsertTmemSemaphore` (which handles tensor-memory ownership transfers).

**Source file**:
`third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemaphore.cpp` (694 lines).

**Registration**: `NVWSInsertSemaphoreBase` via
`nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc`, pass name
`--nvws-insert-semaphore`.

---

## Prerequisites: What You Need to Know

This document assumes familiarity with:

- **Partitions**: Each op in a warp-specialized loop is annotated with
  `ttg.partition = array<i32: ...>`, identifying which warp group(s) will execute it.
- **Arefs** (conceptually): The prior mechanism for cross-partition data transfer.
  Arefs bundled synchronization, buffer staging, and data movement into a single
  `put`/`get` protocol. Semaphores decouple these concerns (see
  [01_aref_limitations.md](01_aref_limitations.md)).
- **Semaphore ops**: `nvws.semaphore.create`, `nvws.semaphore.acquire`,
  `nvws.semaphore.release`, `nvws.semaphore.buffer` (see
  [02_semaphore_ops.md](02_semaphore_ops.md)).

---

## Algorithm Overview

The pass (`NVWSInsertSemaphore::runOnFunction`, line 628) walks over every
`scf.for` loop marked with `tt.warp_specialize` and processes it in three phases
plus a final cleanup step. The phases are ordered so that more specific patterns
are handled before more general ones, avoiding double-processing.

### Phase 1: Loop-Carried Values (iter_args)

```
loop.walk([&](scf::ForOp forOp) {
    for each iter_arg of type {RankedTensorType, FloatType, IntegerType}:
        determine producerPartition from partition.outputs
        insertSemaphores(builder, loop, forOp.getBody(), producedValue)
})
```

**What it catches**: `scf.for` iter_args whose producer partition (determined via
`ttg.partition.outputs`) differs from their consumer partition. These represent
loop-carried dependencies that cross partition boundaries. The canonical example
is a distance-one dependency where partition 0 yields a value that partition 1
reads on the next iteration.

(Lines 637--648)

### Phase 2: Memory Ops (descriptor_load + local_alloc patterns)

```
loop.walk([&](Operation *op) {
    if op produces a result matching:
        isDescLoadAndAlloc<LocalAllocOp>(result)   // TMA load -> local_alloc
        isa<LocalAllocOp>(op)                       // standalone local_alloc
    then: collect op into memoryOps
})
for each memoryOp:
    for each producedValue:
        insertSemaphores(...)
```

**What it catches**: The dominant pattern in TMA matmul -- a `tt.descriptor_load`
feeding into a `ttg.local_alloc`, where the producer partition (typically the
load warp group) differs from the consumer partition (typically the MMA warp
group). These are handled first because Phase 3 would otherwise see the
individual ops and potentially insert redundant semaphores.

(Lines 651--666)

### Phase 3: Remaining Cross-Partition Values

```
loop.walk([&](Operation *op) {
    skip: loop itself, MMAv5OpInterface, TMEMAllocOp, TMEMStoreOp
    for each producedValue:
        insertSemaphores(...)
})
```

**What it catches**: Register-to-register transfers, scalar values, and any other
cross-partition dataflow not covered by Phase 2. TMEM ops are excluded because
they are handled by the separate `InsertTmemSemaphore` pass.

(Lines 669--681)

---

## Cross-Partition Detection

The core detection logic lives in `insertSemaphores` (line 562). For a given
`ProducedValueInfo` (containing a result value and its producer partition set),
the function examines every SSA use:

```
For each use of producedValue.result:
    userPartitions = getPartitionIds(&use)
    remove all producer partition IDs from userPartitions
    for each remaining consumer partition id:
        resultsPerPartition[id].insert(result)
        usesPerPartition[id].push_back(&use)
```

If a use's partition set contains any ID not in the producer's partition set,
that use is a **cross-partition consumer** and needs a semaphore.

**Same-partition optimization**: If all uses are within the producer's partitions,
`resultsPerPartition` is empty and the function returns `false` without creating
any semaphore. This is tested by `@no_value_semaphore` (line 97 of the test
file) where producer and consumer are both in partition 0.

For `isDescLoadAndAlloc` patterns, the function also processes uses of the
intermediate `descriptor_load` result (the tensor value), since the `local_alloc`
result and the tensor result may have consumers in different partitions. This
is the "register and SMEM dual use" pattern tested by `@load_used_as_reg_and_smem`.

(Lines 562--617)

---

## Semaphore Pair Creation: `createSemaphores`

For each cross-partition value, a semaphore pair is created by `createSemaphores`
(line 323). The function:

1. **Determines the MemDescType** for the shared-memory buffer:
   - If the value is a `LocalAllocOp` result: use its `MemDescType` directly.
   - If the value is a `RankedTensorType`: compute a shared-memory descriptor
     via `getSharedEncoding` (preserving the tensor's encoding as shared layout).
   - If the value is a scalar (`FloatType`/`IntegerType`): splat to a `{1}`
     tensor using `getTensorTypeFromScalar`, then compute the shared-memory
     descriptor.

2. **Creates the allocation** with depth=1:
   ```
   allocBufType = getMultiBufferedType(memDescType, 1)
   alloc = nvws::createAlloc(builder, loc, allocBufType, Value())
   ```
   Multi-buffering (depth > 1) is deferred to the `LowerSemaphore` pass. At
   insertion time, the pipeline depth is always 1.

3. **Creates the semaphore pair**:
   ```
   semaTy = SemaphoreType<[allocBufType], depth>
   
   empty = SemaphoreCreateOp(semaTy, alloc, is_released=true)
   full  = SemaphoreCreateOp(semaTy, alloc, is_released=false)
   ```
   - **EMPTY** (`is_released=true`): The producer acquires this. Initially
     released means "buffer is free to write into."
   - **FULL** (`is_released=false`): The consumer acquires this. Initially
     not released means "no data available yet; wait for producer."

4. **Placement**: The allocation and both `semaphore.create` ops are placed
   immediately before the outermost warp-specialized loop:
   ```cpp
   auto wsLoop = getOuterWSLoop(loop);
   builder.setInsertionPoint(wsLoop);
   ```

(Lines 323--367)

---

## Producer Side: `createSemaphorePut`

The producer-side sequence (line 369) follows the pattern:

```
acquire EMPTY  ->  get buffer  ->  write data  ->  release FULL
```

In detail:

```mlir
%token   = nvws.semaphore.acquire %empty        // wait for buffer to be free
%dataBuf = nvws.semaphore.buffer  %empty, %token // get writable buffer view
// ... write data into %dataBuf ...
nvws.semaphore.release %full, %token [async_ops] // signal: data is ready
```

The write and async_ops depend on the value type:

### Case 1: TMA Descriptor Load + LocalAlloc (`isDescLoadAndAlloc`)

Detected when the produced value is a `LocalAllocOp` whose `src` is a
`triton::DescriptorOpInterface` (`tt.descriptor_load` or `tt.descriptor_gather`)
in the same partition.

```cpp
createNVWSDescriptorLoadOp(builder, descOp, dataBuf, producerPartitions, loc)
producerKind = AsyncOp::TMALoad
staleOps = {alloc, descOp}   // both removed after consumers are rewired
```

The `nvws.descriptor_load` is the semaphore-aware replacement for the original
`tt.descriptor_load`. It targets the semaphore buffer directly:

```mlir
nvws.descriptor_load %desc[%indices] txCount %dataBuf
```

The `txCount` is computed by `getTxCount` (line 97):
```
txCount = product(shapePerCTA) * elementBitWidth / 8
```
For a `tensor<128x64xf16>`: `128 * 64 * 16 / 8 = 16384` bytes.

The `async_ops=[tma_load]` annotation tells `LowerSemaphore` that the TMA
hardware will signal the mbarrier directly -- no software fence is needed.

### Case 2: LocalAllocOp with Source (non-TMA)

```cpp
LocalStoreOp(alloc.getSrc(), dataBuf)
producerKind = AsyncOp::NONE
staleOps = {alloc}
```

### Case 3: Register Tensor (RankedTensorType, no descriptor load)

```cpp
LocalStoreOp(result, dataBuf)
producerKind = AsyncOp::NONE
```

This includes the case where a bare `tt.descriptor_load` result (tensor in
registers) is used directly as a cross-partition value without going through
`local_alloc`. In that case, the descriptor load is replaced by an
`nvws.descriptor_load` targeting the semaphore buffer (same as Case 1) and
the `staleOps` list includes only the descriptor op.

### Case 4: Scalar (FloatType / IntegerType)

```cpp
%splat = tt.splat %scalar : scalar_type -> tensor<1xscalar_type, #blocked>
LocalStoreOp(%splat, dataBuf)
producerKind = AsyncOp::NONE
```

The scalar is first splatted to a `{1}` tensor, then stored. The `splat`
encoding uses a default blocked layout with `{1}` shape.

### Stale Op Cleanup

The `staleOps` returned by `createSemaphorePut` are erased after all consumers
have been rewired (line 612). This is important because the original ops may
still have uses being processed by the consumer-side logic.

(Lines 369--443)

---

## Consumer Side: `createSemaphoreGet`

The consumer-side sequence (line 445) follows the pattern:

```
acquire FULL  ->  get buffer  ->  read data  ->  release EMPTY
```

In detail:

```mlir
%token   = nvws.semaphore.acquire %full          // wait for data to be ready
%dataBuf = nvws.semaphore.buffer  %full, %token   // get readable buffer view
// ... read data from %dataBuf ...
nvws.semaphore.release %empty, %token [async_ops] // signal: buffer is free
```

The read depends on the original value type:

### Case A: LocalAllocOp Result (SMEM memdesc)

The buffer view is propagated directly -- no `local_load` is needed because the
consumer expects a `MemDescType`. The function calls `replaceUsesAndPropagateType`
to replace all consumer uses of the original `local_alloc` result with `dataBuf`,
also updating downstream view ops (like `memdesc_trans`) to have the mutable
memory type and correct partition annotations.

```cpp
replaceUsesAndPropagateType(builder, localAlloc, dataBuf, callback)
```

This is the path taken for the TMA matmul pattern where the MMA op directly
consumes a `!ttg.memdesc`.

### Case B: Tensor Result (RankedTensorType)

```cpp
%loaded = ttg.local_load %dataBuf : memdesc -> tensor
// replace all cross-partition uses of original result with %loaded
```

### Case C: Scalar Result

```cpp
%loaded = ttg.local_load %dataBuf : memdesc -> tensor<1xscalar_type>
%scalar = tt.unsplat %loaded : tensor<1xscalar_type> -> scalar_type
// replace all cross-partition uses with %scalar
```

### Consumer async_ops

The `async_ops` annotation on the consumer's `semaphore.release` tells
`LowerSemaphore` which hardware async operation's completion implies that the
buffer is no longer being read. The logic (in `getConsumerAsyncOpKinds`, line 204):

| Consumer Op Type | async_ops |
|-----------------|-----------|
| `MMAv5OpInterface` (e.g., `tc_gen5_mma`) | `tc5mma` |
| `WarpGroupDotOp` | `wgmma` |
| Everything else | `none` |

If multiple consumer ops exist for the same partition (e.g., both MMA and a
non-MMA use), the union of async op kinds is emitted.

### Release Placement

The consumer's `semaphore.release` is placed after the **last use** of the data
in the consumer partition. For `local_alloc` propagation (Case A), the last use
is found via `findNearestCommonPostDominator` over the transitive consumers. For
direct `local_load` (Cases B/C), it is placed after the `local_load` itself.

(Lines 445--560)

---

## Fan-Out: One Producer, N Consumers

When a value produced in partition P is consumed in partitions Q1, Q2, ..., QN,
the pass creates **one semaphore pair** but **N independent consumer sequences**:

```
Semaphore pair: (EMPTY, FULL) created once before the loop.

Producer:  acquire EMPTY -> write -> release FULL    (once)

Consumer Q1:  acquire FULL -> read -> release EMPTY  (independent)
Consumer Q2:  acquire FULL -> read -> release EMPTY  (independent)
...
Consumer QN:  acquire FULL -> read -> release EMPTY  (independent)
```

Each consumer partition independently acquires the FULL semaphore, gets its own
buffer view, loads the data, and releases the EMPTY semaphore. This is handled
by the loop at line 603:

```cpp
for (auto [consumerPartition, results] : resultsPerPartition) {
    createSemaphoreGet(builder, loop, sema, results, consumerPartition,
                       usesPerPartition[consumerPartition]);
}
```

The test `@two_consumers` (line 288) demonstrates this: `op_a` in partition 0
produces a value consumed by `op_b` in partition 1 and `op_c`/`op_d` in
partition 2. One semaphore pair is created, with two independent consumer
acquire/load/release sequences.

---

## TMA Descriptor Load Handling

TMA loads receive special treatment because the TMA hardware can signal an
mbarrier directly upon completion, avoiding a software fence.

### Detection

The `isDescLoadAndAlloc<LocalAllocOp>` template (line 79) checks whether a value
is defined by a `LocalAllocOp` whose `src` comes from a
`triton::DescriptorOpInterface` (`tt.descriptor_load` or `tt.descriptor_gather`)
in the same partition.

### Transformation

The original pair:
```mlir
%tensor = tt.descriptor_load %desc[%i, %j] : ... -> tensor<128x64xf16>
%alloc  = ttg.local_alloc %tensor : ... -> !ttg.memdesc<128x64xf16, ...>
```

Becomes:
```mlir
%token   = nvws.semaphore.acquire %empty
%dataBuf = nvws.semaphore.buffer %empty, %token
nvws.descriptor_load %desc[%i, %j] 16384 %dataBuf   // TMA writes directly to buf
nvws.semaphore.release %full, %token [#nvws.async_op<tma_load>]
```

Both the `tt.descriptor_load` and `ttg.local_alloc` are erased. The
`nvws.descriptor_load` targets the semaphore buffer directly.

### txCount Computation

`getTxCount` (line 97) computes the byte count the TMA will transfer:

```cpp
auto encoding = getEncodingFromDescriptor(descOp, tensorType, desc);
auto shapePerCTA = getShapePerCTA(encoding, tensorType.getShape());
return product(shapePerCTA) * getIntOrFloatOrPtrBitWidth(elementType) / 8;
```

Examples:
- `tensor<128x64xf16>`: `128 * 64 * 16 / 8 = 16384` bytes
- `tensor<128x64xf8E4M3FN>`: `128 * 64 * 8 / 8 = 8192` bytes
- `tensor<128x8xi8>` (scales): `128 * 8 * 8 / 8 = 1024` bytes

### async_ops Significance

The `[#nvws.async_op<tma_load>]` on the producer's release tells `LowerSemaphore`
to emit an `mbarrier.arrive.expect_tx` with the txCount, and the TMA hardware
will signal the mbarrier upon completion. Without this annotation, a software
`mbarrier.arrive` would be emitted instead.

---

## Stage/Phase Assignment

The `semaphore.acquire` and `semaphore.release` ops are created without
 `stage` or `phase` attributes. These are filled in later by the
 `AssignSemaphoreStagePhase` pass (see [05_assign_stage_phase.md](05_assign_stage_phase.md)).

The `stageCluster` values that are set on the inserted ops (from
`getStageClusterForProducer` and `getEnterAndExitStageClustersOfUses`) are
**scheduling metadata** (`loop.cluster` and `loop.stage`), not semaphore
stage/phase. These tell the pipeline scheduler where to place the ops within
the software-pipelined loop body.

---

## Comparison with the Old InsertAref Approach

The semaphore pass replaces the functionality that was previously handled by an
`InsertAref` pass. The conceptual flow is identical -- detect cross-partition
dataflow, insert synchronization -- but the mechanics differ:

| Aspect | Aref | Semaphore |
|--------|-----------|-----------------|
| **Cross-partition detection** | `getProducedValues`, `getTransitiveConsumers` | Same functions, same logic |
| **Producer ops** | `ArefPutEnterOp` / `ArefPutExitOp` | `SemaphoreAcquireOp` + `SemaphoreBufferOp` + write + `SemaphoreReleaseOp` |
| **Consumer ops** | `ArefGetEnterOp` / `ArefGetExitOp` | `SemaphoreAcquireOp` + `SemaphoreBufferOp` + read + `SemaphoreReleaseOp` |

---

## Helper Functions Reference

### `getProducedValues` (line 45)

For an operation with partition annotations, returns a list of
`ProducedValueInfo` structs, one per non-token result. Each contains the
result's partition set (from `getPartitionOutputs` or `getPartitionIds`) and
the result `Value`.

### `isDescLoadAndAlloc<AllocOp>` (line 79)

Template function that checks if a value is defined by an `AllocOp` whose `src`
is a `triton::DescriptorOpInterface`, both in the same partition. Returns the
pair `(alloc, descOp)` if matched.

### `getTensorTypeFromScalar` (line 87)

Creates a `RankedTensorType` with shape `{1}` and a default blocked encoding,
suitable for splatting a scalar into SMEM.

### `getTransitiveConsumers` (line 163)

Follows uses through `MemDescViewTrait` ops (like `memdesc_trans`) to find the
actual consuming operations in a given consumer partition. Returns the set of
leaf consumers.

### `getConsumerAsyncOpKinds` (line 204)

Maps consumer operations to their async op kind for the release annotation:
`MMAv5OpInterface` -> `tc5mma`, `WarpGroupDotOp` -> `wgmma`, else `none`.

### `getEarliestUserInBlock` (line 262)

Given a set of uses, finds the earliest user within a block. Used to determine
the insertion point for the consumer's acquire sequence.

### `getStageClusterForProducer` (line 138)

Traces through loop iter_args to find the scheduling cluster of the actual
producing operation. Handles the case where a `BlockArgument` is a loop-carried
value by following the yield chain.


---

## Lit Test Coverage

The test file at `test/NVWS/insert_semaphore.mlir` contains 21 `CHECK-LABEL`
directives covering the following scenarios:

| Test Function | What It Tests |
|--------------|---------------|
| `@warp_specialize_tma_matmul` | Basic TMA matmul: two descriptor loads (LHS/RHS) in partition 2, MMA consumer in partition 1. Two semaphore pairs, TMA async_ops. |
| `@specialize_load_only` | Single TMA load consumed as register tensor (via `local_load`). |
| `@no_value_semaphore` | Producer and consumer in the same partition -- no semaphore created (`CHECK-NOT`). |
| `@value_semaphore_multiple_producers` | Value in partition `{0,1}` consumed in partition 2 -- only the cross-partition use gets a semaphore. |
| `@load_used_as_reg_and_smem` | Same TMA load used as both register (partition 0, `local_load`) and SMEM (partition 1, direct buffer). |
| `@load_used_as_reg_and_smem_same_partition` | Both register and SMEM uses in the same consumer partition -- single semaphore pair. |
| `@matmul_scaled_rhs_scales_tma` | Scaled MMA with FP8 operands and scale tensors via TMA. |
| `@local_alloc_default_partition` | Three cross-partition values producing three semaphore pairs (6 creates). Non-TMA `local_alloc` with transpose. |
| `@two_consumers` | Fan-out: one producer in partition 0, two consumers in partitions 1 and 2. One semaphore pair, two consumer sequences. |
| `@distance_one` | Loop-carried dependency: iter_arg produced in partition 0, consumed in partition 1 (Phase 1 processing). |
| `@different_yield_partition` | Yielded value's partition output differs from the op that produces the yield operand. |
| `complex_case` (unnamed label) | Two cross-partition iter_args (`%k`, `%l`) with multiple consumers in partitions 1 and 2. |
| `@reuse_argument` | Two iter_args where `%l` (the second) has cross-partition consumers; `%k` does not. |
| `@multiplicity_branch` | Three iter_args, each consumed in a different partition. Three semaphore pairs. |
| `@multiplicity_branch2` | Three iter_args with each yielded from a different producer partition (0, 1, 2). Chain of partition-to-partition transfers. |
| `@self_recursion` | iter_arg used only within its own partition -- no semaphore needed. |
| `@self_recursion_and_use` | iter_arg used within its own partition AND in another partition. Semaphore only for the cross-partition use. |
| `@conditional_consumer` | Consumer inside `scf.if` -- semaphore acquire/release placed outside the conditional. |
| `@no_def_op` | Scalar iter_arg (`i32`) with cross-partition consumer: tests splat/unsplat path. |
| `@scalar_consumers` | Non-iter_arg scalar value crossing partitions: splat on producer side, unsplat on consumer side. |
| `cycle_in_partition` (2-partition) | Bidirectional cycle: partition 0 -> partition 1 -> partition 0. Two semaphore pairs. |
| `cycle_in_partition` (3-partition) | Three-way cycle: partition 0 -> 1 -> 2 -> 0. Three semaphore pairs. |
| `@inner_loop_fixed_operand` | Nested loops: outer loop produces LHS via TMA, inner loop produces RHS via TMA. Consumer in inner loop uses both. Four semaphore creates total. |
| `@semaphore_result_outside_scheduled_loop` | Cross-partition value used outside the `tt.scheduled_max_stage`-annotated inner loop. Tests stage cluster inference when the scheduled loop is not the direct parent. |

---

## End-to-End Example: TMA Matmul

To ground the abstractions, here is the complete transformation for the
`@warp_specialize_tma_matmul` test case.

**Before** (two cross-partition TMA loads feeding an MMA):

```mlir
// partition 2 (load warp group)
%lhs_tensor = tt.descriptor_load %descA[%i, %j] {ttg.partition = array<i32: 2>}
%rhs_tensor = tt.descriptor_load %descB[%i, %j] {ttg.partition = array<i32: 2>}
%lhs_alloc  = ttg.local_alloc %lhs_tensor       {ttg.partition = array<i32: 2>}
%rhs_alloc  = ttg.local_alloc %rhs_tensor       {ttg.partition = array<i32: 2>}

// partition 1 (MMA warp group)
%rhs_trans  = ttg.memdesc_trans %rhs_alloc       {ttg.partition = array<i32: 1>}
%token      = ttng.tc_gen5_mma %lhs_alloc, %rhs_trans, ...
                                                  {ttg.partition = array<i32: 1>}
```

**After**:

```mlir
// Before the loop:
%alloc1 = ttg.local_alloc : !ttg.memdesc<1x128x64xf16, #shared, #smem>
%empty1 = nvws.semaphore.create %alloc1 true
%full1  = nvws.semaphore.create %alloc1 false
%alloc2 = ttg.local_alloc : !ttg.memdesc<1x128x64xf16, #shared, #smem>
%empty2 = nvws.semaphore.create %alloc2 true
%full2  = nvws.semaphore.create %alloc2 false

// Inside loop body:

// --- Producer for LHS (partition 2) ---
%ptok1  = nvws.semaphore.acquire %empty1         {ttg.partition = array<i32: 2>}
%pbuf1  = nvws.semaphore.buffer  %empty1, %ptok1 {ttg.partition = array<i32: 2>}
nvws.descriptor_load %descA[%i, %j] 16384 %pbuf1 {ttg.partition = array<i32: 2>}
nvws.semaphore.release %full1, %ptok1 [tma_load] {ttg.partition = array<i32: 2>}

// --- Producer for RHS (partition 2) ---
%ptok2  = nvws.semaphore.acquire %empty2         {ttg.partition = array<i32: 2>}
%pbuf2  = nvws.semaphore.buffer  %empty2, %ptok2 {ttg.partition = array<i32: 2>}
nvws.descriptor_load %descB[%i, %j] 16384 %pbuf2 {ttg.partition = array<i32: 2>}
nvws.semaphore.release %full2, %ptok2 [tma_load] {ttg.partition = array<i32: 2>}

// --- Consumer for RHS (partition 1) ---
%gtok2  = nvws.semaphore.acquire %full2          {ttg.partition = array<i32: 1>}
%gbuf2  = nvws.semaphore.buffer  %full2, %gtok2  {ttg.partition = array<i32: 1>}
%rhs_t  = ttg.memdesc_trans %gbuf2               {ttg.partition = array<i32: 1>}

// --- Consumer for LHS (partition 1) ---
%gtok1  = nvws.semaphore.acquire %full1          {ttg.partition = array<i32: 1>}
%gbuf1  = nvws.semaphore.buffer  %full1, %gtok1  {ttg.partition = array<i32: 1>}
%mma    = ttng.tc_gen5_mma %gbuf1, %rhs_t, ...  {ttg.partition = array<i32: 1>}

// --- Consumer releases (partition 1) ---
nvws.semaphore.release %empty2, %gtok2 [tc5mma]  {ttg.partition = array<i32: 1>}
nvws.semaphore.release %empty1, %gtok1 [tc5mma]  {ttg.partition = array<i32: 1>}
```

