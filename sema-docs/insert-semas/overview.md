# InsertSemas overview

## Contract

`nvws-insert-semas` turns cross-partition accesses to managed local or tensor
memory into explicit semaphore synchronization. A managed allocation has a
logical group id: an authored `buffer.id` when present, otherwise a private
synthetic id. Every access is assigned an `Owner` from its partition metadata,
possibly `root` when no single warp-specialized partition owns it.

The pass must preserve three things at once:

1. memory versions: a read observes the intended write, and a later write does
   not overwrite a value while another partition may still use it;
2. physical storage: overlapping allocations and pipeline copies refer to the
   correct backing slot; and
3. scheduled execution: inserted operations respect authored pipeline stages
   and the ordering constraints required by the new handoffs.

The implementation separates those concerns into three models. ACCESS-DAG
describes memory and ownership. SYNC-DAG derives the required handoffs and
builds a complete, verified synchronization plan. EMIT-IR materializes that
plan without changing it.

This distinction is the central design rule:

> SYNC-DAG chooses every acquire, release, token producer, region result,
> semaphore channel, schedule, and copy offset. EMIT-IR only renders those
> choices.

## Running example

The documents build one example in stages. The input contains one local-memory
object, one writer partition, and one reader partition:

The excerpt is IR-shaped text; type details are abbreviated, but the
warp-specialization attributes that establish owners are shown:

```text
%m0 = ttg.local_alloc {buffer.id = 104} : !memdesc
scf.for ... {
  ttg.local_store %value, %m0 {ttg.partition = array<i32: 0>}
  %v = ttg.local_load %m0 {ttg.partition = array<i32: 1>}
  "use"(%v) {ttg.partition = array<i32: 1>}
} {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
   ttg.warp_specialize.tag = 0 : i32}
```

The notation used in the diagrams is deliberately shorter:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

ACCESS-DAG discovers that `m0` is one group member, its whole footprint is one
piece `P0`, and the loop first touches that piece as owner `{0}`:

```text
func root
`- for pieces{P0:W:{0}}
   |- ENTER pieces{P0:W:{0}}
   |- W m0 {0}
   |- R m0 {1}
   `- EXIT pieces{P0:W:{0}}
```

SYNC-DAG then derives two synchronization edges with stable names:

```text
e1: W m0 {0} -> R m0 {1}       current value becomes readable
e2: R m0 {1} -> EXIT {0}        piece P0; next iteration may overwrite it
```

After reduction, both remain. Point-of-use placement maps them to two
semaphore channels:

```text
EMPTY:  count=1, initially released, owner {0}
FULL:   count=1, initially blocked

for {
  a EMPTY {0}
  W m0 {0}
  r FULL  {0}
  a FULL  {1}
  R m0 {1}
  r EMPTY {1}
}
```

The next iteration's `a EMPTY` closes `e2`. The first iteration succeeds
because `EMPTY` starts released. A zero-trip loop performs none of the body
synchronization. Later sections derive every line rather than assuming this
diagram.

## One model, three steps

```text
input IR
  |
  v
ACCESS-DAG
  collect groups, members, pieces, accesses, owners, and region boundaries
  |
  v
SYNC-DAG
  derive and reduce memory edges
  choose backing copies
  directly place exact POU or FirstTouch synchronization
  form semaphore channels and validate the complete plan
  finalize schedules and copy offsets
  |
  v
EMIT-IR
  materialize backing, semaphores, tokens, views, and region signatures
  verify the emitted IR
  |
  v
output IR
```

ACCESS-DAG assigns owners and creates `ENTER`/`EXIT` boundaries while it
recursively builds each region; ownership is not a separate pipeline stage.

SYNC-DAG constructs the selected placement directly. A discarded candidate is
thrown away as a whole; nodes in an accepted candidate are not changed into
another placement policy later.

## Core objects

- **GroupDag**: one logical synchronization group, normally keyed by memory
  kind and group id. Circular local rules may split one physical `buffer.id`
  into several logical groups.
- **Member**: one allocation in the group. Members may overlap in physical
  address space.
- **PieceId**: one disjoint interval induced by all member endpoints. A
  member's `footprint` is the list of pieces it covers.
- **Owner**: `(ttg.partition, warp-specialization tag)`. An empty `Owner` is
  `root`, outside a warp-specialized partition.
- **Node**: one access, region, boundary marker, acquire, or release.
- **Chain**: the ordered nodes in a function, loop body, or `if` branch.
- **PieceInfo**: the owner and aggregate `R`/`W` effect of one piece at a node
  or region boundary.
- **RegionFlow**: the token carried through a `for` or `if`: one owner, one
  exact producer or pass-through marker per exit path, and a statically
  selected fallback channel. An existing incoming token keeps its channel.
- **Sema**: one logical semaphore channel, its pending count, optional entry
  owner, and eventually its emitted SSA value.

Several fields form the contract between planning and emission:

- `slotEffect` seals the aggregate memory effect of an access or region
  summary; `ENTER` and `EXIT` carry `pieceInfo` instead;
- `tokenSource` names the exact acquire or region producer consumed by an
  access or release;
- `producedTokenOwner` states the effective owner of a token-producing node;
- `sat` pairs a release with the acquire whose demand it satisfies;
- `scheduleAnchor` identifies the operation whose completion schedules a
  release;
- `recurrenceDistance` records a cross-iteration dependency explicitly; and
- `stageOffset` and `bufferStageOffset` select semaphore and backing copies.

Because token identity is exact, several same-owner tokens may remain live at
once. `tokenSource` distinguishes them, and EMIT-IR resolves that specific
producer.

## Source and use state

SYNC-DAG derives memory edges per piece. For each piece it tracks:

- the logical producer of the current memory version;
- the chain-local source node representing that version; and
- the latest use by each owner.

The logical producer and the chain-local source are not always the same. When
a child region receives a version from its parent, the child's `ENTER` is the
local source. This lets the child derive its own edges without inventing an
operation outside its chain.

Readers of one stable version may fan out. A later writer must wait for every
latest reader that is not already ordered before it. The full rules, including
region composition and edge reduction, are derived in
[SYNC-DAG](sync-dag.md).

## Placement policies

`placement-mode` accepts three values:

- `pou`: place loop recurrence acquires at the exact point of use; reject the
  pass if that placement cannot be proven valid;
- `first-touch`: use the canonical conservative carried recurrence for each
  eligible uniform-owner loop; and
- `auto`: try POU, then discard and rebuild the entire function candidate with
  the rejected `(group, loop)` forced to FirstTouch. Repeats may select
  different policies at different loop levels. If a rejection cannot be
  localized, the driver makes one all-FirstTouch attempt.

Both policies produce the same final graph schema. Scheduling and EMIT-IR do
not need to know which policy built it.

## Mutation boundary

Group collection, ACCESS-DAG construction, and each per-group SYNC-DAG
candidate are model-building steps. Schedule finalization may raise generated
or authored `loop.cluster` constraints, so Auto snapshots authored attributes
and restores them before rebuilding a rejected candidate.

EMIT-IR starts only after one complete candidate passes placement validation,
token-connectivity validation, structural verification, and schedule
finalization. It then rewrites SSA signatures and memory objects. It does not
move synchronization nodes across regions, choose another token, or split an
`scf.if` to change planning.

## Step documents

1. [ACCESS-DAG: accesses, owners, and boundaries](access-dag.md)
2. [SYNC-DAG: edges, placement, semaphores, and schedule](sync-dag.md)
3. [EMIT-IR: materializing the sealed plan](emit-ir.md)

## Code map

- Shared model: `InsertSemas.h`
- Groups, pieces, accesses, owners, and boundaries:
  `InsertSemasAccessDag.cpp`
- Memory edges, direct placement, semaphore formation, verification, and
  scheduling: `InsertSemasSyncDag.cpp`
- Symbolic dump and IR materialization: `InsertSemasEmitIR.cpp`
- Placement-mode retry driver: `InsertSemas.cpp`
