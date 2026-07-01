# Meta passes ported into NVWS-AWS

Meta-NVWS reuses Meta policy where NVWS needs the same decision before its own
code partitioning and semaphore pipeline. The Meta algorithm is the baseline;
NVWS adapters translate representation and materialize metadata.

## Data partition

The port keeps Meta's partition-dimension search, slice closure, operation
duplication, result reconstruction, and load reordering.

NVWS adds Blackwell representation coverage and legality checks for sliced
TMEM encodings, descriptor-gather coordinates, generic regionless operations,
and SMEM function-argument views. A failed NVWS legality check rejects that
candidate, so Meta's M-first/N-fallback search may select a later candidate;
the candidate order and partition search policy remain Meta's.

Sources:

- Meta: [`WSDataPartition.cpp`](../third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp)
- NVWS: [`WSDataPartition.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/WSDataPartition.cpp),
  `doDataPartition` and `NVWSWSDataPartition`

## Partition scheduling

The port keeps Meta's operation categories, initial partition layout,
propagation, schedule optimization, and data-partitioned `scf.if` splitting.
If Meta finds no schedulable load or MMA, NVWS also leaves the loop unchanged.

Meta hands an internal schedule to its code partitioner. NVWS instead closes the
selected schedule over structural and scalar/address operations, writes
`ttg.partition` and `ttg.partition.outputs`, assigns WS tags, and verifies the
metadata required by `PartitionLoops` and `InsertSemas`. The pass is
transactional: failure discards the cloned function.

Immediately afterward, Meta-NVWS strips `ttg.partition`,
`ttg.partition.outputs`, `ttg.partition.stages`, and the WS tag from operations
outside WS loops. It does not strip `loop.stage` or `loop.cluster`. Downstream
NVWS passes therefore consume partition metadata only in a retained WS region.

Sources:

- Meta: [`PartitionSchedulingMeta.cpp`](../third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PartitionSchedulingMeta.cpp)
- NVWS: [`PartitionSchedulingMeta.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/PartitionSchedulingMeta.cpp),
  `getInitialSchedule`, `finalizePartitionAnnotations`, and
  `NVWSPartitionSchedulingMeta::runOnFuncOp`

## Memory planning

`MemoryPlanner.cpp` keeps Meta's channel ordering, SMEM allocation, TMEM reuse,
copy-depth growth, and budget accounting. `MemoryPlannerNVWSAdapter` translates
NVWS IR into planner input and publishes the result. It does not add reuse
policy.

### Input model

A channel is a private planner record containing an allocation, its producer,
and its consumers. It bounds the allocation's lifetime; it is not a semaphore
and is not consumed by `InsertSemas`.

Meta builds channels from its `async_task_id` representation. NVWS rebuilds
them from the explicit memory operations produced by `InsertAllocas` and their
`ttg.partition` annotations.

| Input case | Meta | NVWS extension |
|---|---|---|
| Sourceful `ttng.tmem_alloc` | Skips the channel | Uses the allocation as producer when it has one producer partition and consumers |
| Same-partition TMEM producer/consumer | Skips the channel because no cross-partition communication exists | Keeps a synthetic channel so the planner sees the lifetime |
| Operand-D TMEM | Uses Meta's representation | Reconstructs the lifecycle from stores, MMA updates, loads, partitions, and transparent views |

The synthetic channels carry explicit source and destination operations. They
only expose liveness to the planner; `InsertSemas` derives synchronization from
the actual memory accesses independently.

### Modeled and unmodeled TMEM

The [overview](nvws-aws-overview.md#terminology) defines the terms:

```text
modeled   -> traversal finds live operations -> bounded lifetime
unmodeled -> traversal finds none            -> unknown lifetime
```

Unsupported TMEM users are errors, not unmodeled allocations. NVWS handles an
unknown or otherwise unassigned lifetime conservatively:

1. If allocation-time TMEM channel traversal yields no live operations, NVWS's
   `MemoryPlannerTmem::getLiveIntervals` uses the full-function interval. The
   corresponding Meta routine has no empty-traversal fallback.
2. If an allocation was not assigned by any innermost-loop allocation run,
   NVWS allocates it in the final run without a controlling loop. Meta has that
   final-run path but does not add these allocations to it.

### Output representation

After Meta policy selects IDs and depths, NVWS publishes:

```text
buffer.id
buffer.copy
buffer.offset
buffer.circular
buffer.start
```

The circular attributes describe reuse already selected by Meta policy; they
cannot create or resize a reuse group. SMEM is planned before TMEM, so TMEM IDs
follow the SMEM range.

Sources:

- Meta: [`WSMemoryPlanner.cpp`](../third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp)
- NVWS policy port: [`MemoryPlanner.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp),
  `doMemoryPlanner`
- Representation adapter:
  [`MemoryPlannerNVWSAdapter.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlannerNVWSAdapter.cpp)

## Not Meta ports

`InsertAllocas`, `InsertSemas`, `AssignStagePhase`, and `LowerSemaphore` are the
NVWS representation and synchronization pipeline. They consume the ported
decisions but are not copies of Meta code partitioning or token lowering.
