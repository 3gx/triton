# NVWS-AWS pipeline

## Two NVWS paths

On Blackwell, both paths enter `TritonGPUAutomaticWarpSpecialization`. The
difference is how data partitioning, partition scheduling, and memory planning
are selected.

| Step | Default NVWS | Meta-NVWS (`TRITON_NVWS_USE_META=1`) |
|---|---|---|
| Data partition | Hopper `WSDataPartition` pass | NVWS port of Meta `WSDataPartition` |
| Loop pre-schedule | Standard latency schedule (`scheduleKeyOpsUpstream`) | Meta-aware latency schedule (`scheduleKeyOpsMetaWS`) |
| Partition schedule inside AWS | `TritonGPUPartitionScheduling` | NVWS port of `PartitionSchedulingMeta` |
| NVWS memory planner | Not run | NVWS port of Meta `WSMemoryPlanner` |
| Buffer materialization | `NVWSInsertAllocas` | `NVWSInsertAllocas` |
| Synchronization | `NVWSInsertSemas` | `NVWSInsertSemas` |
| Barrier lowering | `NVWSLowerSemaphore` | `NVWSLowerSemaphore` |

`TRITON_USE_META_WS=1` selects Meta AutoWS itself and takes precedence over
`TRITON_NVWS_USE_META`; it is not an NVWS-AWS path.

Setting `TRITON_USE_MODULO_SCHEDULE` inserts a modulo-scheduling pass before
data partitioning on both NVWS paths. It writes `tt.autows` annotations on
MMA operations — a JSON string whose `stage` and `order` keys feed
`loop.stage` and `loop.cluster`. The pre-schedule (assign latencies +
schedule loops) is then skipped entirely — re-running it would override the
modulo schedule — and the annotations are honored by the terminal
schedule-loops run inside AWS (`scheduleKeyOpsAnnotation` in
`ScheduleLoops.cpp`) instead of either latency schedule in the table above.

## Terminology

Pipeline-wide terms, each with the code name it comes from. The InsertSemas
model objects (group, piece, node, owner, and so on) are defined in
the [InsertSemas overview](insert-semas/overview.md#core-objects). Every other
document uses these terms with exactly these meanings.

- **Backing**: a mutable SMEM or TMEM allocation guarded by a semaphore
  (`GroupDag::backing` in `InsertSemas.h`).
- **Depth**: the number of buffered copies of a backing. The memory planner
  records it as `buffer.copy`. Without a planner, depth is 1, except that
  `LowerSemaphore` widens TMA-load-fed SMEM backings to the owning WS loop's
  depth — its explicit `tt.num_stages` when present, else the `num-stages`
  pass option; the largest wins when several WS loops share the group — and
  the default path may double-buffer TMEM accumulators (see
  [SYNC-DAG, backing depth](insert-semas/sync-dag.md#backing-depth)). These
  documents use only this one word for that count.
- **Buffer stage**: an integer in `[0, depth)` selecting one backing copy and
  the mbarrier with the same index. In the IR it is the `stage` operand of the
  `nvws.semaphore.*` operations; `AssignStagePhase` tracks it as
  `State::stage`.
- **Current buffer stage**: the buffer stage most recently selected for a
  semaphore group. A fresh write advances it to the next copy modulo depth; a
  read leaves it unchanged. All partitions that use the group share this one
  value; there is no per-partition copy.
- **Fresh write**: an acquire whose first reachable buffer access writes the
  backing instead of reading it (`isFirstUseFreshWriteAfterAcquire` in
  `AssignStagePhase.cpp`). Only fresh writes advance the current buffer stage.
- **Stage offset**: a signed integer that `InsertSemas` may place in one
  semaphore operation's `stage` operand (`Node::stageOffset`). Because the
  current buffer stage is shared, an operation sometimes needs a different
  copy; its final buffer stage is
  `(current buffer stage + stage offset) mod depth`.
- **Phase**: the mbarrier parity an acquire's wait expects.
- **Pipeline stage**: the software pipeliner's static schedule, `loop.stage`
  together with `loop.cluster`. It determines when an operation executes; it
  never selects a backing copy or mbarrier.
- **Semaphore group**: the `nvws.semaphore.create` operations whose first
  buffer operand is the same allocation (`getSemaGroups` in `LowerAref.cpp`).
  A semaphore group shares one current buffer stage.
- **Channel**: a private memory-planner record connecting an allocation to its
  producer and consumers. It bounds the allocation's lifetime; it is not a
  semaphore, and `InsertSemas` never reads it. These documents use *channel*
  only in this sense.
- **Modeled / unmodeled TMEM allocation**: a TMEM allocation for which channel
  traversal does / does not find producer and consumer operations. An
  unmodeled allocation has an unknown lifetime.
- **Sourceful allocation**: a `ttg.local_alloc` or `ttng.tmem_alloc` with an
  initial-value operand.
- **WS tag**: the `ttg.warp_specialize.tag` integer identifying one
  warp-specialized loop scope.
- **Root**: defined with the InsertSemas model objects — see the
  [InsertSemas overview](insert-semas/overview.md#core-objects); printed as
  `root` in dumps.

The bare word *stage* is never used alone in these documents: it is always
*buffer stage* or *pipeline stage*. In the IR and C++ the semaphore operand,
`State::stage`, and `getStage`/`setStage` all say `stage` and mean the buffer
stage; `loop.stage` means the pipeline stage.

### Worked example

Consider a depth-2 circular backing shared by two allocations A and B — a
planner-selected reuse group, where `buffer.circular` marks the group and
`buffer.start` is each allocation's starting copy (see
[meta-ports](meta-ports.md#output-representation)) — with `buffer.start = 0`
for A and 1 for B. The current buffer stage starts at `depth - 1`, which is
1. Every partition uses this same shared
value:

| Operation | Current-buffer-stage action | Stage offset | Final buffer stage |
|---|---|---:|---:|
| write A | advance `1 -> 0` | 0 | 0 |
| write B | advance `0 -> 1` | 0 | 1 |
| read A | keep 1 | -1 | 0 |
| read B | keep 1 | 0 | 1 |

The `-1` offset lets the read of A select copy 0 after the shared current
buffer stage has advanced to 1 for B. Ordinary groups use the current buffer
stage directly; `InsertSemas` assigns stage offsets only for circular reuse
and for exact-alias SMEM handoffs across multiple copies.

### For readers coming from Meta AutoWS

| Meta AutoWS | NVWS |
|---|---|
| `Channel` (producer-to-consumer dependency record) | channel inside the ported memory planner (Meta-NVWS only) |
| token with `producer_acquire` / `producer_commit` / `consumer_wait` / `consumer_release` | one `nvws.semaphore` per ownership transfer; `acquire` returns a token, `release` consumes it |
| `bufferFull` / `bufferEmpty` mbarrier arrays | one mbarrier per buffer stage of each semaphore, allocated by `LowerSemaphore` |
| `accumCnt`, `bufferIdx = accumCnt % numBuffers` | buffer stage, tracked by `AssignStagePhase` |
| `phase = (accumCnt / numBuffers) & 1` | phase, computed by `AssignStagePhase` |
| `numBuffers` / multi-buffering | depth (`buffer.copy` or `num-stages`) |
| reuse group (allocations sharing `buffer.id`) | planner group sharing `buffer.id`; a semaphore group after lowering |
| `async_task_id` | `ttg.partition` plus the WS tag |

## Pass order

The relevant Blackwell pipeline is:

```text
modulo schedule                       [only with TRITON_USE_MODULO_SCHEDULE]
-> data partition
-> assign latencies and schedule loops [skipped with TRITON_USE_MODULO_SCHEDULE]
-> automatic warp specialization
     -> partition scheduling             [default / Meta-NVWS variants]
     -> strip partition metadata outside WS loops [Meta-NVWS only]
     -> hoist loop-invariant TMEM stores
     -> InsertAllocas
     -> MemoryPlanner                    [Meta-NVWS only]
     -> InsertSemas
     -> LowerSemaphore
          -> multi-buffer TMA-load-fed SMEM backings to depth num-stages
          -> AssignStagePhase
          -> lower semaphore IR to barriers
     -> partition loops
     -> lower warp groups
     -> schedule loops                   [default / Meta-NVWS variants]
     -> multi-buffer TMA descriptors
     -> clear internal WS metadata
-> software pipeline
```

`LowerSemaphore`'s first step (`multiBufferSemaphore` in `LowerAref.cpp`)
widens backings the planner did not size: a semaphore group whose release is
fed by a TMA load and whose SMEM backings carry no `buffer.copy` is rewritten
from one copy to the effective depth (`tt.num_stages` on the owning WS loop,
defaulting to the `num-stages` option) before `AssignStagePhase` computes
buffer stages. `InsertSemas` already analyzes such groups at that final depth (see
[SYNC-DAG, backing depth](insert-semas/sync-dag.md#backing-depth)).

The terminal `schedule loops` run also differs per path: the AWS driver passes
it a Meta flag (`useMetaWS` in `AutomaticWarpSpecialization.cpp`), selecting
the same standard/Meta-aware split as the pre-schedule.

A partition verifier (`VerifyWarpSpecializationPartitions`, which the AWS
driver runs after each wrapped pass) checks the passes from partition
scheduling through `LowerSemaphore`; it does not run after `partition loops`,
`lower warp groups`, or the terminal `schedule loops`. The software pipeliner sees
concrete partitions, barriers, multi-buffered descriptors, and the finalized
`loop.stage`/`loop.cluster` schedule.

## Contracts between passes

```text
partition schedule
  ttg.partition, ttg.partition.outputs, WS tags, loop schedule

InsertAllocas
  explicit producer writes and consumer reads over mutable SMEM/TMEM buffers

MemoryPlanner (Meta-NVWS)
  buffer.id/copy/offset and optional circular metadata; allocations sharing
  a buffer.id always connect through shared members into one component
  (reusers are stacked within their owner's columns) — InsertSemas asserts
  this and rejects a group whose pieces do not all connect (see ACCESS-DAG,
  "Pieces must connect")

InsertSemas
  semaphore create/acquire/buffer/release operations and token threading
  through structured control flow,
  optional stage offsets (partitions share one current buffer stage, so an
  operation may need an offset to select another copy, for example a
  circular-buffer read of an earlier copy),
  pipeline-legal loop.stage/loop.cluster annotations

LowerSemaphore
  first widens TMA-load-fed SMEM backings without buffer.copy to num-stages
  copies, then:

  AssignStagePhase
    final buffer stage for each acquire/buffer/release and wait phase for
    each acquire

  barrier lowering
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
