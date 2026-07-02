# InsertSemas overview

## Contract

`NVWSInsertSemas` consumes partitioned IR with explicit mutable SMEM/TMEM
accesses. Optional `buffer.*` attributes describe physical reuse and depth.
It produces `nvws.semaphore.create/acquire/buffer/release`, threads semaphore
tokens through structured control flow, and assigns pipeline-legal
`loop.stage`/`loop.cluster` annotations. Pipeline-wide terms are defined in
the [NVWS-AWS terminology](../nvws-aws-overview.md#terminology).

The pass models exclusive ownership for writes and shared ownership for
reads. The *producer* of a piece is the owner that last wrote it — or,
before any write, its first toucher. A *holder* is an owner whose latest
access to the current version a future writer must respect. After a write,
the writer is the sole holder and later readers join; a child chain instead
starts with its `ENTER` owner as the sole holder. The *version source* is the
concrete DAG node from which a new reader receives that version:

```text
write:       reset the version source and holders to the writer
reread:      move only that holder's latest node; keep the version source
new reader:  wait on the stable version source, then join the holders
new writer:  wait on every other holder's latest node, then become exclusive
```

The version source is the write node in the same chain, the first toucher
before any write, or a child `ENTER` node that represents a version established
outside the child. This keeps independent readers as a fan-out rather than a
reader-to-reader chain.

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
  derive handoff edges, semaphores, current and retained token lifetimes,
  control-flow threading, buffer-stage schedule
      |
EMIT-IR
  materialize the already-decided protocol
```

All four steps use the same `Node` graph. Later steps extend it instead of
reconstructing ownership from mutated IR.

## Core objects

Model objects with their code names — the shared graph types are in
`InsertSemas.h`, Chain's builder `buildChainForBlock` is in
`InsertSemasAccessDag.cpp`, and the walk's `VersionSource`, `PieceGame`, and
`HolderRec` are in `InsertSemasSyncDag.cpp`. The step documents use these
terms with exactly these meanings.

- **Group** (`GroupDag`): the allocations analyzed together for ownership.
  Ordinarily, allocations of one memory kind with the same `buffer.id` form
  one group; an allocation without `buffer.id` gets a private synthetic
  group. Allocations sharing a physical `buffer.id` are analyzed as separate
  groups when they need independent synchronization: circular SMEM
  allocations, or TMEM allocations with different `buffer.copy` values.
- **Member** (`Member`): one allocation within a group.
- **Piece** (`Piece`): a maximal address interval with one fixed set of
  covering members. A group's pieces must all connect through shared members
  — one connected component; `buildAccessDag` rejects the group otherwise
  (see [ACCESS-DAG](access-dag.md#pieces-must-connect)). The group is therefore the unit
  of synchronization — one set of semaphores, tokens scoped to the group
  (never to a piece), one set of crossings and holds — while
  version/holder state is tracked per piece.
- **Node** (`Node`): one entry of a group's program-order graph. Kinds:
  `Func` (the root of the graph), `Access` (a real operation touching group
  memory), `For`/`If` (a nested region), `Acquire`/`Release` (semaphore
  protocol added by SYNC-DAG), and the `ENTER`/`EXIT` boundary markers. An
  access or release may carry `Node::retainedTokenOwner`, SYNC-DAG's proof
  that the node can reuse an earlier token acquired by that owner.
- **Chain**: the node sequence of one block, built in program order
  (`buildChainForBlock`); each region node holds child chains.
- **Owner** (`Owner`): the one partition that executes an access, acquire, or
  release inside a tagged WS scope, identified as `(partition ID, WS tag)`.
- **Root**: the owner of an access with no WS tag on itself or an enclosing
  `scf.for` (`Owner == std::nullopt`). A root access has no WS partition
  owner even if it carries `ttg.partition`: a partition index names an
  executor only relative to a WS-tagged loop, so `ttg.partition` with no tag
  in scope is stray metadata, not ownership — the pass does not diagnose it
  and resolves the access to root (Meta-NVWS strips such metadata before
  this pass; see [meta-ports](../meta-ports.md#partition-scheduling)). Root
  is distinct from partition 0 inside a WS scope.
- **Region partition set**: the partitions in which `PartitionLoops` will
  clone a `for` or `if`. It may contain several partitions and is separate
  from access ownership; the owner a region reports per piece is stored on
  the region node (`Node::pieceInfo`).
- **Touch** (`Touch`): one member access on an `Access` node, classified read
  (`R`) or write (`W`).
- **Semaphore token**: the `!ttg.async.token` returned by
  `nvws.semaphore.acquire`. `nvws.semaphore.buffer` uses it to expose the
  guarded memory, and `nvws.semaphore.release` takes it as an operand. The
  value is not linear: a SYNC-DAG-marked same-owner buffer or release may
  reuse an earlier token even after a later acquire by another owner.
- **Retained token**: that explicitly proved earlier owner-local token.
  Retention is chain-local and is reset at region boundaries; EMIT-IR only
  renders `retainedTokenOwner`, never infers eligibility.
- **Crossing** (`Crossing`): a record that a `for` or `if` may need to return a
  token. An unused `if` crossing is removed; a surviving one returns a token.
  For a loop, the hold decides whether a token iter-arg and result remain.
- **Hold** (`Hold`): the token-placement decision for one loop crossing: keep a
  token iter-arg and result, move its acquire to the first protected access,
  or remove the outer loop's token slot when its final nested loop takes no
  token from the outer loop and returns none. Its materialized span runs from
  acquire through protected accesses to closing release. Retained owner-local
  tokens may coexist with that token.

One hold-placement safety check preserves the loop-carried form. For a plain
inner loop in a WS scope, if moving the acquire to the first body access also
needs an acquire after the loop, and those two acquires would have different
`loop.stage` values, SYNC-DAG keeps the loop-carried form with reason
`cross-stage-final-acquire`. Equal or unknown stages do not trigger it.

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
[SYNC-DAG](sync-dag-1.md#buffer-stage-offsets-and-the-pipeline-schedule)); it
never changes `loop.stage`. EMIT-IR then renders the graph and performs
representation-driven folding and cleanup; its one schedule exception is the
loop-scheduler workaround, which splits qualifying `scf.if` operations and
may copy a pipeline stage onto a release it moves (see
[EMIT-IR](emit-ir.md)).

## Step documents

- [ACCESS-DAG](access-dag.md)
- [OWNER-DAG](owner-dag.md)
- [SYNC-DAG](sync-dag-1.md)
- [EMIT-IR](emit-ir.md)

## Code map

- Dispatcher: [`InsertSemas.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp)
- Shared model and traversal utilities:
  [`InsertSemas.h`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h)
- Pass options: [`Passes.td`](../../third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td),
  `NVWSInsertSemas`
