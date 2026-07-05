# InsertSemas overview

## Contract

`NVWSInsertSemas` consumes partitioned IR with explicit mutable SMEM/TMEM
accesses. Optional `buffer.*` attributes describe physical reuse and the
number of backing copies.
It produces `nvws.semaphore.create/acquire/buffer/release`, threads semaphore
tokens through structured control flow, and assigns pipeline-legal
`loop.stage`/`loop.cluster` annotations. Pipeline-wide terms are defined in
the [NVWS-AWS terminology](../nvws-aws-overview.md#terminology).

The pass models exclusive ownership for writes and shared ownership for
reads. SYNC-DAG keeps two facts for each piece. Its *source* records the
logical producer of the current version and the concrete DAG node from which
a new reader receives that version. Its *uses* record each owner's latest
access to that version:

```text
first touch: source = this node; uses = [this owner -> this node]
write:       wait on uses owned by every other owner, unless already ordered;
             reset source and uses
reread:      move only this owner's use; keep the source
new reader:  wait on the stable source; add this owner's use
```

The source node is the latest write in the same chain, the first toucher
before any write, or a child `ENTER` node that represents a version established
outside the child. A child preserves the inherited logical producer while
using `ENTER` as its chain-local source. Keeping source and uses separate lets
independent readers fan out from the same source while a later writer still
waits for the node stored for every other owner in `uses`.

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
  derive required memory edges, token-supply edges, semaphores, token reuse,
  region token policy, buffer-stage schedule
      |
EMIT-IR
  materialize the already-decided protocol
```

All four steps use the same `Node` graph. Later steps extend it instead of
reconstructing ownership from mutated IR.

## Core objects

Model objects with their code names — the shared graph types are in
`InsertSemas.h`, the chain builder `buildChainForBlock` is in
`InsertSemasAccessDag.cpp`, and the walk's `VersionSource`, `PieceState`,
`ActiveUse`, and `Tokens` are in `InsertSemasSyncDag.cpp`. The step documents
use these terms with exactly these meanings.

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
  source/use state is tracked per piece.
- **Node** (`Node`): one entry of a group's program-order graph. Kinds:
  `Func` (the root of the graph), `Access` (a real operation touching group
  memory), `For`/`If` (a nested region), `Acquire`/`Release` (semaphore
  protocol added by SYNC-DAG), and the `ENTER`/`EXIT` boundary markers. An
  access or release may carry `Node::reuseTokenOwner`, SYNC-DAG's proof that
  the node can reuse that owner's earlier token.
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
  guarded memory, and `nvws.semaphore.release` takes it as an operand. Tokens
  are group-scoped, not piece-scoped. Within a chain, the pass keeps known
  owner tokens in deterministic order and uses the last token by default. A
  node marked with `reuseTokenOwner` instead uses that owner's earlier token.
  The region token policy resets this chain-local reuse state; EMIT-IR renders
  the mark and never infers reuse.
- **Crossing** (`Crossing`): a record that a `for` or `if` may need to return a
  token. Its `tokenOwner` is the owner of that boundary token. An unused `if`
  crossing is removed; a surviving one returns a token.
- **Hold** (`Hold`): the region token policy for one loop crossing. It either
  threads a token through the loop (`THREADED`), moves the acquire to the first
  protected access (`POINT_OF_USE`), or removes the outer token slot because
  the final child owns the protocol (`CHILD_OWNS`). Its materialized span runs
  from acquire through protected accesses to closing release. Other
  owner-indexed tokens may coexist inside the chain; the policy decides which
  single token, if any, crosses the region boundary.

One hold-placement safety check preserves the loop-carried form. For a plain
inner loop in a WS scope, if moving the acquire to the first body access also
needs an acquire after the loop, and those two acquires would have different
`loop.stage` values, SYNC-DAG keeps the loop-carried form and the dump prints
bare `gated`. Equal or unknown stages do not trigger this check.

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
schedule finalization may raise existing `loop.cluster` values when a
producer and consumer execute in the same pipelined iteration (see
[SYNC-DAG](sync-dag.md#pipeline-schedule)); it never changes `loop.stage`.
EMIT-IR then renders the graph and performs
representation-driven folding and cleanup; its one schedule exception is the
loop-scheduler workaround, which splits qualifying `scf.if` operations and
may copy a pipeline stage onto a release it moves (see
[EMIT-IR](emit-ir.md)).

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
