# InsertSemas overview

## Contract

`NVWSInsertSemas` consumes partitioned IR with explicit mutable SMEM/TMEM
accesses. Optional `buffer.*` attributes describe physical reuse and depth.
It produces `nvws.semaphore.create/acquire/buffer/release`, threads semaphore
tokens through structured control flow, and assigns pipeline-legal
`loop.stage`/`loop.cluster` annotations. Pipeline-wide terms are defined in
the [NVWS-AWS terminology](../nvws-aws-overview.md#terminology).

The pass models exclusive ownership for writes and shared ownership for
reads. The *producer* of a piece of memory is the owner that last wrote it —
or, before any write, its first toucher; a *holder* is any owner currently
allowed to access it — the producer plus the readers that have joined since:

```text
new reader: take a token from the current producer, then join the holders
new writer: collect tokens from every other holder, then become exclusive
```

## One model, four steps

```text
ACCESS-DAG
  discover physical IDs, groups, overlap pieces, accesses, effects, regions
      |
OWNER-DAG
  add region boundary markers and assign an owner to each piece on each
  for/if node
      |
SYNC-DAG
  derive handoff edges, semaphores, token lifetimes, control-flow threading,
  buffer-stage schedule
      |
EMIT-IR
  materialize the already-decided protocol
```

All four steps use the same `Node` graph. Later steps extend it instead of
reconstructing ownership from mutated IR.

## Core objects

Model objects with their code names — all in `InsertSemas.h`, except that
Chain's builder `buildChainForBlock` lives in `InsertSemasAccessDag.cpp`. The
step documents use these terms with exactly these meanings.

- **Group** (`GroupDag`): the allocations analyzed together for ownership.
  Ordinarily, allocations of one memory kind with the same `buffer.id` form
  one group; an allocation without `buffer.id` gets a private synthetic
  group. Allocations sharing a physical `buffer.id` are analyzed as separate
  groups when they need independent synchronization: circular SMEM
  allocations, or TMEM allocations with different `buffer.copy` values.
- **Member** (`Member`): one allocation within a group.
- **Piece** (`Piece`): a maximal address interval with one fixed set of
  covering members.
- **Component** (`pieceComp`): connected pieces that share a member.
  Synchronization is per component — each component gets its own semaphores,
  token, crossings, and holds — while producer/holder state is tracked per
  piece.
- **Node** (`Node`): one entry of a group's program-order graph. Kinds:
  `Func` (the root of the graph), `Access` (a real operation touching group
  memory), `For`/`If` (a nested region), `Acquire`/`Release` (semaphore
  protocol added by SYNC-DAG), and the `ENTER`/`EXIT` boundary markers.
- **Chain**: the node sequence of one block, built in program order
  (`buildChainForBlock`); each region node holds child chains.
- **Owner** (`Owner`): the one partition that executes an access, acquire, or
  release inside a tagged WS scope, identified as `(partition ID, WS tag)`.
- **Root**: the owner of an access with no WS tag on itself or an enclosing
  `scf.for` (`Owner == std::nullopt`). A root access has no WS partition
  owner even if it carries `ttg.partition`; root is distinct from partition 0
  inside a WS scope.
- **Region partition set**: the partitions in which `PartitionLoops` will
  clone a `for` or `if`. It may contain several partitions and is separate
  from access ownership; the owner a region reports per piece is stored on
  the region node (`Node::pieceInfo`).
- **Touch** (`Touch`): one member access on an `Access` node, classified read
  (`R`) or write (`W`).
- **Semaphore token**: the `!ttg.async.token` returned by
  `nvws.semaphore.acquire`. `nvws.semaphore.buffer` uses it to expose the
  guarded memory, and `nvws.semaphore.release` consumes it.
- **Crossing** (`Crossing`): a record that a component's token is live
  inside a `for` or `if`; the crossing's hold decides whether the token
  actually passes through the region boundary.
- **Hold** (`Hold`): the span from an acquire through the last protected
  access to the release that closes it, during which one owner holds the
  component's token. SYNC-DAG's hold rule decides where holds are cut.

For supported accesses, owner resolution is:

```text
partition [p] with WS tag t -> owner (p, t)
no tag on the access or an enclosing scf.for -> root
```

Inside a tagged WS scope, every access to an SMEM or TMEM allocation handled
by `InsertSemas` must execute in one partition. Multi-partition accesses are
not supported: the pass does not diagnose them and silently treats such an
access as root-owned (`resolveOwner` in `InsertSemas.h`), which can drop the
intended handoff. Multi-partition metadata is supported on region and control
operations, where it describes execution rather than access ownership.

When root-held state enters a tagged WS loop, the first partition to touch it
may take ownership without an incoming partition-to-partition handoff.

## Mutation boundary

ACCESS, OWNER, and per-group SYNC construction are analysis-only. Global SYNC
schedule finalization may raise existing `loop.cluster` values so that
dependencies with no pipeline-stage slack become legal (see
[SYNC-DAG](sync-dag.md#buffer-stage-offsets-and-the-pipeline-schedule)); it
never changes `loop.stage`. EMIT-IR then renders the graph and performs only
representation-driven folding and cleanup.

## Step documents

- [ACCESS-DAG](access-dag.md)
- [OWNER-DAG](owner-dag.md)
- [SYNC-DAG](sync-dag.md)
- [EMIT-IR](emit-ir.md)

## Code map

- Dispatcher: [`InsertSemas.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp)
- Shared model and traversal utilities:
  [`InsertSemas.h`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h)
- Pass options: [`Passes.td`](../../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td),
  `NVWSInsertSemas`
