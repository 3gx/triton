# NVWS-AWS pipeline

## Two NVWS paths

On Blackwell, both paths enter `TritonGPUAutomaticWarpSpecialization`. The
difference is how data partitioning, partition scheduling, and memory planning
are selected.

| Step | Default NVWS | Meta-NVWS (`TRITON_NVWS_USE_META=1`) |
|---|---|---|
| Data partition | Hopper `WSDataPartition` pass | NVWS port of Meta `WSDataPartition` |
| Loop pre-schedule | Standard latency schedule | Meta-aware latency schedule |
| Partition schedule inside AWS | `TritonGPUPartitionScheduling` | NVWS port of `PartitionSchedulingMeta` |
| NVWS memory planner | Not run | NVWS port of Meta `WSMemoryPlanner` |
| Buffer materialization | `NVWSInsertAllocas` | `NVWSInsertAllocas` |
| Synchronization | `NVWSInsertSemas` | `NVWSInsertSemas` |
| Barrier lowering | `NVWSLowerSemaphore` | `NVWSLowerSemaphore` |

`TRITON_USE_META_WS=1` selects Meta AutoWS itself and takes precedence over
`TRITON_NVWS_USE_META`; it is not an NVWS-AWS path.

## Terminology

- **Backing**: a mutable SMEM or TMEM allocation guarded by a semaphore.
- **Backing group**: semaphores whose first backing is the same SSA value. They
  share one current-slot value across all participating partitions; there is
  not a separate current-slot value per partition.
- **Depth**: the number of buffered copies in a backing.
- **Slot**: an integer in `[0, depth)` selecting one backing copy and the
  mbarrier with the same index.
- **Current slot**: the slot most recently selected for the backing group. A
  fresh write advances it to the next slot modulo `depth`; a read leaves it
  unchanged.
- **Slot offset**: a signed integer that `InsertSemas` supplies for one
  semaphore operation. Because the current slot is shared across partitions,
  an operation may need an offset to select a different buffered copy. Its
  final slot is
  `(current slot + slot offset) modulo depth`.
- **Phase**: the mbarrier parity expected by an acquire's wait.
- **Static pipeline schedule**: `loop.stage` and `loop.cluster`, which determine
  when an operation executes. They do not select a backing copy or mbarrier.
- **Memory-planner channel**: a private liveness record connecting an allocation
  to its producer and consumers. It is not a semaphore.
- **Modeled TMEM allocation**: a TMEM allocation for which channel traversal
  finds producer/consumer operations and a bounded live interval.
- **Unmodeled TMEM allocation**: a TMEM allocation for which channel traversal
  finds no live operations, so its lifetime is unknown.

These documents reserve *slot* for the dynamic backing/mbarrier selector and
*stage* for static `loop.stage`. Current IR and C++ still name the slot value
`stage`, including the semaphore `stage` operand, `State::stage`, and
`getStage`/`setStage`. Renaming those symbols and `AssignStagePhase` is deferred.

For example, consider a depth-2 circular backing shared by logical buffers A
and B, with `buffer.start = 0` for A and `buffer.start = 1` for B. The current
slot starts at `depth - 1`, which is 1. Producers and consumers in every
partition use this same current-slot value:

| Operation | Current-slot action | Slot offset | Final slot |
|---|---|---:|---:|
| write A | advance `1 -> 0` | 0 | 0 |
| write B | advance `0 -> 1` | 0 | 1 |
| read A | keep 1 | -1 | 0 |
| read B | keep 1 | 0 | 1 |

The `-1` offset makes the read of A select slot 0 after the shared current slot
has advanced to 1 for B. Ordinary groups use the current slot directly.
`InsertSemas` currently authors slot offsets for circular reuse and exact-alias
multi-slot SMEM handoffs.

## Pass order

The relevant Blackwell pipeline is:

```text
data partition
-> assign latencies and schedule loops
-> automatic warp specialization
     -> partition scheduling             [default / Meta-NVWS variants]
     -> strip partition metadata outside WS loops [Meta-NVWS only]
     -> hoist loop-invariant TMEM stores
     -> InsertAllocas
     -> MemoryPlanner                    [Meta-NVWS only]
     -> InsertSemas
     -> LowerSemaphore
          -> AssignStagePhase
          -> lower semaphore IR to barriers
     -> partition loops
     -> lower warp groups
     -> schedule loops
     -> multi-buffer TMA descriptors
     -> clear internal WS metadata
-> software pipeline
```

Partition verification wraps the AWS passes from partition scheduling through
`LowerSemaphore`; it does not run after the terminal partition/lowering/
scheduling trio. The software pipeliner sees concrete partitions, barriers,
multi-buffered descriptors, and the finalized `loop.stage`/`loop.cluster`
schedule.

## Contracts between passes

```text
partition schedule
  ttg.partition, ttg.partition.outputs, WS tags, loop schedule

InsertAllocas
  explicit producer writes and consumer reads over mutable SMEM/TMEM buffers

MemoryPlanner (Meta-NVWS)
  buffer.id/copy/offset and optional circular metadata

InsertSemas
  ownership protocol and carrier threading,
  optional slot offsets because partitions sharing one backing also share one
  current-slot value; an offset can select another copy, for example when a
  circular-buffer read must select an earlier backing copy,
  pipeline-legal static loop.stage/loop.cluster annotations

AssignStagePhase
  final slot for each semaphore acquire/buffer/release and wait phase for acquire:
  slot selects one copy of the multi-buffered backing and its mbarrier;
  phase is the mbarrier parity expected by the wait

LowerSemaphore
  mbarrier allocation, wait, arrive/commit, and concrete buffer views
```

## Code map

- Backend selection: [`compiler.py`](../third_party/nvidia/backend/compiler.py),
  `make_ttgir`, Blackwell branch.
- AWS orchestration:
  [`AutomaticWarpSpecialization.cpp`](../lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp),
  `AutomaticWarpSpecialization::runOnOperation`.
- Pass definitions:
  [`Passes.td`](../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td).
