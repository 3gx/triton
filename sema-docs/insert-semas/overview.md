# InsertSemas overview

## Contract

`NVWSInsertSemas` consumes partitioned IR with explicit mutable SMEM/TMEM
accesses. Optional `buffer.*` attributes describe physical reuse and the
number of backing copies. It produces
`nvws.semaphore.create/acquire/buffer/release`, threads semaphore tokens
through structured control flow, and assigns pipeline-legal
`loop.stage`/`loop.cluster` annotations. Pipeline-wide terms are defined in
the [NVWS-AWS terminology](../nvws-aws-overview.md#terminology).

The pass models exclusive ownership for writes and shared ownership for
reads. For each physical piece of a buffer, synchronization construction
tracks two facts:

- `source`: the DAG node for the most recent write to the piece.
- `uses`: a table from partition to DAG node. After a write, it contains
  `[writing partition -> write]`. Each later read adds or replaces
  `[reading partition -> read]`.

The state is updated as follows:

```text
read in a partition already present in uses
  replace that partition's entry with this read

read in a partition not present in uses
  add source -> read
  add [this partition -> read] to uses

write
  for each other partition in uses not already ordered before this write:
    add uses[partition] -> write
  source = this write
  uses   = [writing partition -> this write]
```

After a write, `source` remains fixed at that write while reads update
`uses`. The first read in another partition is ordered after `source`; later
reads in the same partition only replace that partition's entry. Reads in
different partitions are therefore not ordered with respect to one another.
When another write is encountered, the required entries in `uses` are
ordered before it, and the new write resets both `source` and `uses`.

## One model, three steps

```text
ACCESS-DAG
  collect groups and overlap pieces; build accesses and owners;
  wrap region bodies with ENTER/EXIT boundaries
      |
SYNC-DAG
  build synchronization edges; choose backing copies; reduce and merge edges;
  choose semaphores, tokens, and semaphore copies;
  assign stage offsets, legalize clusters, and schedule acquires/releases
      |
EMIT-IR
  materialize the decided acquire/release/token/buffer protocol
```

All three steps use the same `Node` graph. Each step adds decisions to that
graph, so later steps do not reconstruct memory ownership from changed IR.

## Core objects

The shared graph types are in `InsertSemas.h`. The detailed construction is
covered by [ACCESS-DAG](access-dag.md) and [SYNC-DAG](sync-dag.md).

- **Group** (`GroupDag`): allocations analyzed together for ownership.
  Allocations of one memory kind with the same `buffer.id` normally form one
  group; an allocation without `buffer.id` gets a private synthetic group.
  Circular SMEM allocations and TMEM allocations with different
  `buffer.copy` values may use separate groups while sharing physical storage.
- **Member** (`Member`): one allocation in a group.
- **Piece** (`Piece`): a maximal address interval covered by one fixed set of
  members. A group's pieces must be connected through shared members. The
  group is one synchronization unit, while source and use state is tracked
  separately for each piece.
- **Node** (`Node`): one entry in the group's program-order graph. An
  `Access` is a real memory operation; `For` and `If` hold child chains;
  `ENTER` and `EXIT` mark region boundaries; `Acquire` and `Release` are added
  by synchronization construction. `Func` is the graph root.
- **Chain**: the node sequence for one region. A region node occupies one
  position in its parent chain and owns one child chain per region.
- **Owner** (`Owner`): the partition and WS tag that execute an access,
  acquire, or release: `(partition ID, WS tag)`.
- **Root**: an access with no owner. Root is distinct from partition 0 inside
  a WS scope.
- **Region boundary**: the per-piece owner and read/write effect recorded on a
  `For`, `If`, `ENTER`, or `EXIT` node. It presents a child chain as one event
  to its parent.
- **Touch** (`Touch`): one member access on an `Access` node, classified as a
  read (`R`) or write (`W`). A touch reaches every piece in that member's
  footprint.
- **Semaphore token**: the `!ttg.async.token` returned by
  `nvws.semaphore.acquire`. `nvws.semaphore.buffer` uses it to expose guarded
  memory, and `nvws.semaphore.release` consumes it. Tokens belong to a group,
  not to an individual piece.

For a supported memory access, owner resolution is:

```text
partition [p] with WS tag t  -> owner (p, t)
no partition or no tag       -> root
several partitions           -> unsupported for a memory access
```

Inside a tagged WS scope, a supported SMEM/TMEM access must execute in one
partition. A memory access carrying several partition IDs is unsupported.
Multi-partition metadata is supported on control operations, where it
describes which partitions execute the region rather than who owns a memory
access.

When root-held state enters a tagged WS loop, the first partition to touch it
may adopt it without a partition-to-partition handoff.

## Mutation boundary

ACCESS-DAG and per-group SYNC-DAG construction only build and annotate the
model. Schedule finalization may raise an existing `loop.cluster` when a
producer and consumer execute in the same pipelined iteration; it does not
change `loop.stage`. EMIT-IR then creates semaphore operations, threads their
tokens through `for` and `if`, materializes shared backing, and removes dead
token slots and replaced allocations.

## Step documents

- [ACCESS-DAG: accesses, owners, and boundaries](access-dag.md)
- [SYNC-DAG](sync-dag.md)
- [EMIT-IR](emit-ir.md)

## Code map

- Dispatcher:
  [`InsertSemas.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp)
- Shared model and traversal utilities:
  [`InsertSemas.h`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h)
- Pass options: [`Passes.td`](../../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td),
  `NVWSInsertSemas`
