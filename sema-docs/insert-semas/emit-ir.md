# EMIT-IR

## Contract: materialize a sealed plan

EMIT-IR receives finalized `GroupDag`s. For every active group, SYNC-DAG has
already fixed:

- physical and semaphore copy counts;
- semaphore channels, entry state, and pending counts;
- exact acquire and release positions;
- every access and release `tokenSource`;
- every token producer's owner;
- every release/acquire pairing and completion anchor;
- every `RegionFlow` and exact path result;
- partition requirements, schedules, recurrence distances, and copy offsets.

The emitter allocates physical objects, rewrites structured-control-flow
signatures, renders those nodes, and verifies the result. It does not choose
POU versus FirstTouch, infer an owner token, or retry another placement. Its
one structural scheduling exception is the post-render loop-scheduler
workaround described below.

The central invariant is:

> Every Access, Release, and region input names an exact `tokenSource`.
> EMIT-IR looks up that producer directly; owner and lexical order are not
> routing inputs.

## Running example: semaphore DAG to MLIR

SYNC-DAG finished the running loop as:

```text
|- scf.for pieces{P0:W:{0}}
|  |- a EMPTY {0}
|  |- W m0 {0}
|  |- r FULL {0} [none]
|  |- a FULL {1}
|  |- R m0 {1}
|  |- r EMPTY {1} [none]

EMPTY: count=1, entry owner={0}
FULL:  count=1, initially blocked
BACKING: numCopies=1
```

The following snippets are schematic MLIR with long type signatures omitted.
The emitter first creates staged backing and the two semaphore objects beside
the original allocation. Rendering later redirects managed uses, and cleanup
removes dead originals:

```text
%base = ttg.local_alloc
  : !ttg.memdesc<1x1xi32, ...>

%empty = nvws.semaphore.create %base true
  {pending_count = 1}
%full = nvws.semaphore.create %base false
  {pending_count = 1}
```

`true` means the entry channel begins released. The body is then rendered in
the same order as the symbolic chain:

```text
scf.for ... {
  %tw = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>}
  %bw = nvws.semaphore.buffer %empty, %tw {ttg.partition = array<i32: 0>}
  ttg.local_store %value, %bw {ttg.partition = array<i32: 0>}
  nvws.semaphore.release %full, %tw [#nvws.async_op<none>]
    {arrive_count = 1, ttg.partition = array<i32: 0>}

  %tr = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>}
  %br = nvws.semaphore.buffer %full, %tr {ttg.partition = array<i32: 1>}
  %v = ttg.local_load %br {ttg.partition = array<i32: 1>}
  nvws.semaphore.release %empty, %tr [#nvws.async_op<none>]
    {arrive_count = 1, ttg.partition = array<i32: 1>}
}
```

Notice that the token may cross semaphore names:

```text
producer a EMPTY -> token %tw -> W m0 and r FULL
producer a FULL  -> token %tr -> R m0 and r EMPTY
```

The release destination is not the semaphore that produced the token. The
token proves access to the current buffer capability; the release opens the
next channel selected by SYNC-DAG.

## Emission order

`emitIR` performs these steps:

1. Create one function-level poison async token. It replaces uses of detached
   legacy token results and also serves as a temporary signature placeholder.
2. Select active groups: groups with no semaphores require no synchronization
   emission.
3. Clear legacy TMEM dependency operands, replace their old token-result uses
   with poison, and repeatedly erase dead pre-existing loop/`if` token slots.
4. Materialize all physical backing objects and semaphore creates.
5. Aggregate every group's requested `RegionFlow` slot and rewrite each
   affected `scf.for` or `scf.if` exactly once, outermost first.
6. Render each active group's finalized chain.
7. Apply the loop-scheduler workaround and remove newly dead token slots.
8. Erase dead alias operations and original allocations.
9. Erase the poison token when it is unused.
10. Verify emitted SSA, partition, locality, and lifetime contracts.

There is no separate “entry acquire” pass. An entry acquire is an ordinary
`Acquire` node rendered at its exact chain position.

## Loop-scheduler workaround

After protocol rendering, `workaroundLoopScheduler` splits qualifying
conditionals so the downstream loop scheduler sees the release and acquire at
separate structured boundaries:

```text
before                                 after
%t = scf.if %cond {                    scf.if %cond { release }
  release                              scf.if %cond { body }
  body                                 %t = scf.if %cond {
  %next = acquire                        %next = acquire
  yield %next                            yield %next
} else { yield %t0 }                   } else { yield %t0 }
```

The moved release and its exit `scf.if` preserve the release's
`loop.stage`/`loop.cluster`; if the release lacks them, it inherits the first
preceding MMA schedule. The moved acquire and its entry `scf.if` preserve the
acquire schedule. The middle `scf.if` keeps its authored schedule and receives
partition metadata derived from its remaining contents and live results.

## Physical backing and semaphore creates

### Ordinary groups

Each member's backing type reflects `numCopies`. For ordinary local or TMEM
memory, the copy dimension is added in front of the authored shape. TMEM scale
encodings retain their special shape convention.

Consider the ACCESS-DAG overlap example:

```text
m0=[0,256), footprint={P0,P1,P2}
m1=[64,192), footprint={P1}
```

Pieces guide synchronization, but every semaphore create carries one backing
value/type per group member. A covering member may physically serve contained
members when the memory kind, offsets, types, and authored planning metadata
permit it. TMEM containment uses `ttng.tmem_subslice` and, when needed,
`ttg.memdesc_reinterpret`.

The physical plan is complete before signature rewriting and chain rendering.
There is no later “fold the backing after emission” phase.

### Circular local groups

Circular groups retain separate logical SYNC-DAGs but share one physical
backing by `buffer.id`. All members must agree on backing type and be defined
in one block. The group with `buffer.start=0` supplies the authored backing
identity.

Compatible circular semaphore creates are shared by `(buffer.id, entry-state)`.
Their pending counts must agree. Copy and stage offsets were already assigned
by SYNC-DAG.

### Entry state

`Sema::entryOwner` controls the create's boolean entry flag:

```text
entryOwner present  -> semaphore.create ... true
entryOwner absent   -> semaphore.create ... false
```

Creates for entry channels are emitted before non-entry channels. The create's
`pending_count` is the uniform count already proved during channel formation.

## Exact token routing

`RenderState` stores records of this form:

```text
Token
  value       emitted async-token SSA value
  sema        emitted semaphore SSA value associated with the capability
  ref
    producer  exact symbolic producer node
    sema      symbolic render channel
    owner     effective owner
```

`recordToken` replaces only a record with the same producer. Tokens produced
by different nodes may coexist even when they have the same owner.

Before rendering an Access or Release, `renderChain` calls
`tokenForSource(node->tokenSource)`. Missing or owner-incompatible records are
errors; only the named producer can satisfy the consumer.

### Exact fan-out and reuse

The fan-out example from SYNC-DAG has three tokens live over time:

```text
a EMPTY(2) {0} -> producer token T0
  T0 -> W m0 {0}
  T0 -> r F1 {0}
  T0 -> r F2 {0}
  T0 -> later R m0 {0}

a F1 {1} -> T1 -> R m0 {1} -> r EMPTY {1}
a F2 {2} -> T2 -> R m0 {2} -> r EMPTY {2}
```

When `{0}` rereads the buffer, `T1` and `T2` may have been emitted more
recently. The access still names producer `T0`, so the emitter selects `T0`
exactly. Owner order cannot change that choice.

## Buffer views and accesses

An access needs both a token and a member view. `getView` emits:

```text
%m0_view, %m1_view, ... = nvws.semaphore.buffer %sema, %token
```

The result bundle includes a view for every group member. The current bundle
is reused only when all exact capability facts match:

- symbolic producer;
- symbolic channel;
- token SSA value;
- semaphore SSA value;
- owner;
- `bufferStageOffset`; and
- a `sameViewType`-compatible requested member type.

`sameViewType` compares shape, element type, encoding, memory space, and
mutability. It deliberately does not make allocation shape part of the cache
identity.

An owner-only cache would be unsound because the same owner may hold several
different producer capabilities.

### Alias replay

For a `Touch`, the emitter selects its member view and replays the alias path
recorded by ACCESS-DAG. Each cloned alias replaces the managed operand with the
current acquired view. Result types are re-inferred where possible so staged
allocation shapes propagate correctly.

A `memdesc_index` whose result is `sameViewType`-compatible with its source can
be elided. Other supported aliases are cloned in order:

```text
base semaphore view
  -> memdesc_index/subview/trans/reinterpret/reshape
  -> exact access view
```

### Access rewriting

Known accesses are rewritten as follows:

- general operations have the exact `Touch::accessValue` operand replaced;
- a sourceful local allocation becomes an explicit `ttg.local_store` through
  the acquired view;
- a scalar local source is splatted when the destination needs a tensor;
- managed allocation uses unrelated to semaphore creates and access nodes are
  redirected to the new view; and
- the rendered access returns its real completion anchor, including an
  ACCESS-DAG-selected descriptor store.

The returned anchor becomes `lastReal`, which determines the exact insertion
point for a following release.

## Synchronization-node mapping

| Symbolic node | Emitted action |
| --- | --- |
| `Acquire` | Insert before the next real node; otherwise before the containing region terminator, or after the last root-level real operation. Apply owner, schedule, and optional `stageOffset`; record the node as producer. |
| `Release` | Insert after the last rendered completion, or at the containing block start when no real node precedes it; use its exact source token, destination semaphore, payload array, count, schedule, and optional `stageOffset`; mark the source released for lifetime auditing. |
| `Access` | Resolve exact source, materialize/reuse the exact buffer bundle, replay aliases, and rewrite the operation. |
| `ENTER` / `EXIT` | Emit nothing; the parent region renderer wires path inputs/results. |
| `For` / `If` | Render the child chain and fill the preallocated token slot only when a `RegionFlow` exists. |

`chainBlock` locates the exact child block for a synchronization-only or otherwise
empty chain. Thus a branch containing only a release still receives that
release in its own block.

## Tokens through regions

### Plain POU loop

The running example has no `RegionFlow`, so its `scf.for` signature is
unchanged. Body-local acquires and releases render exactly where SYNC-DAG put
them.

`renderPlainLoop` still creates a nested render state. If the region has an
incoming exact token used only internally, it records that token under the
region producer while walking the body, then preserves the appropriate exact
record outside without adding a loop result.

### Carried FirstTouch loop

FirstTouch's symbolic graph contains:

```text
entry acquire -> loop tokenSource
loop RegionFlow.owner = {0}
loop RegionFlow.exits[0] = exact tail acquire
```

Signature rewriting first adds a poison placeholder:

```text
%result = scf.for ... iter_args(%carry = %poison) -> !ttg.async.token {
  ...
  scf.yield %poison
}
```

`renderCarriedLoop` then replaces the init with the exact entry token, records
the body iter-arg as the loop's token producer, renders the body, and replaces
the yield with the exact exit producer:

```text
%result = scf.for ... iter_args(%carry = %initial) -> !ttg.async.token {
  ...
  %next = nvws.semaphore.acquire %empty
  scf.yield %next
}
```

The loop result is recorded under:

- the loop node itself;
- the exact incoming producer alias when needed; and
- the exact exit producer alias.

Downstream `tokenSource` pointers can therefore resolve the result without an
owner-based search. On a zero-trip loop, MLIR naturally returns `%initial`.

### An `if` result

Suppose SYNC-DAG recorded:

```text
if thread{{4}}
  then ... EXIT yield{a Sback}
  else ... EXIT yield{pass}
```

Signature rewriting adds one result and one operand to each yield. During
rendering:

- the then path resolves the exact acquire named in `flow.exits[0]`;
- the else path sees `nullptr` and yields the exact incoming token; and
- the `scf.if` result becomes a new producer with owner `{4}`.

Schematic emitted IR:

```mlir
%out = scf.if %cond -> !ttg.async.token {
  ...
  %then_token = nvws.semaphore.acquire %Sback
  scf.yield %then_token
} else {
  scf.yield %incoming
}
```

Rendering normalizes a managed `if` with no authored else to an empty else. If
`RegionFlow` needs a result, that branch supplies the exact pass-through. The
emitter does not split the `if` or move synchronization outside it.

### Several groups threading tokens through one region

`rewriteSignatures` collects all groups' requested slots before touching an
operation. The `for` or `if` is rebuilt once with all extra token types. Each
group remembers its absolute slot index and later fills only that slot.

Operations are rewritten outermost first, and every graph node pointing at an
old region is retargeted to the replacement. This avoids repeated nested
signature surgery.

## Schedule, offsets, and partition metadata

Generated synchronization nodes receive the `stageCluster` already stored in
SYNC-DAG:

- acquire schedule from its demand/boundary;
- release schedule from its exact completion; and
- buffer view schedule from the access operation.

`stageOffset` is emitted as a signed `i32` operand on acquire/release and
selects a semaphore copy. `bufferStageOffset` is emitted on
`nvws.semaphore.buffer` and selects a backing copy. Neither alters `loop.stage`
or `loop.cluster`.

For the two-copy alias example:

```text
R m0 slot 0 -> release Shandoff with stageOffset=+1
W m1 slot 1 -> acquire Shandoff at its selected slot
```

The emitter transcribes `+1`; it does not replay the slot schedule.

For a structured operation that already carries partition metadata,
`requiredParts` extends its partition set when generated synchronization needs
additional owners inside it. Signature rewriting also extends
`ttg.partition.outputs` for every new token result and gives terminators the
region partition metadata when absent.

## Cleanup

After rendering all active groups, EMIT-IR:

- erases supported alias operations whose results are dead;
- erases original managed allocations that no longer have uses; and
- removes the temporary poison operation when no detached legacy use remains.

The initial dead-slot sweep may rebuild old `scf.for`/`scf.if` operations to
remove obsolete async-token results before new exact slots are added. This is
signature cleanup, not synchronization placement.

## Emitted-IR verification

The final verifier checks the materialized contract, not the planning policy.

### Exact cached-view reuse

If a cached buffer view is intentionally reused after a release, the verifier
requires that:

- the view came from `nvws.semaphore.buffer` with the recorded exact token;
- a release of that token exists in the same block; and
- the reused view has a witnessed use after that release.

This exception is admitted only for an exact reuse chosen in SYNC-DAG.

### Partition outputs

For every `scf.for` or `scf.if` carrying `ttg.partition.outputs`:

- the attribute arity must equal the operation's result count; and
- each yielded nonconstant producer with partition metadata must intersect the
  declared output partitions.

### Token and view locality

`verifyTokenLocality` traces every release/buffer token backward through:

- direct `nvws.semaphore.acquire` results;
- `scf.if` then/else yields; and
- `scf.for` iter-args, yields, and init values.

For partition-marked consumers and acquires, the partition sets must be equal.
An acquire without partition metadata is outside that comparison. For a
partition-marked semaphore buffer, every partition-marked view user must have
the same set; unpartitioned users are outside this check.

### Lifetime and loop slots

Within one block, an ordinary token may not create a new buffer view after it
has been released. A newly materialized exact-reuse buffer is recorded in
`exactReuseBufferOps` and exempted from that generic check. Reuse of an already
cached view records a `CachedReuseContract` and must pass the proof above.

For token slots that resolve directly, or through a bounded `scf.if` result
trace, to a semaphore acquire, a loop may not carry two slots for the create's
first physical backing. Downstream stage/phase assignment cannot represent
that state. The check excludes circular local backing, whose physical
synchronization is explicitly multi-channel.

Any failure here is a plan/emitter contract violation or malformed physical
plan. EMIT-IR does not respond by choosing another acquire placement.

## Output contract

After EMIT-IR succeeds:

- every managed memory consumer uses a view derived from its exact planned
  token producer;
- every release uses that same exact token and the planned destination
  semaphore;
- every structured path supplies its planned region result or exact
  pass-through;
- schedules, counts, stage offsets, and backing offsets match SYNC-DAG;
- partition-marked token and view consumers agree with their acquire or
  materialization partition sets; and
- old managed allocations and legacy token plumbing no longer define the
  active synchronization.

## Code map

- Symbolic dump: `SyncDagDumper`, `dumpSyncDags`
- Shared emission state: `EmitCtx`, `RenderState`
- Legacy-token cleanup: `nukeGroupTokens`, `eraseDeadTokenSlots`
- Backing and creates: `materializeLogicalBacking`,
  `materializeMixedDepth`, `materializeCircular`, `emitPhysicalIR`
- Aggregated region slots: `rewriteSignatures`
- Views and accesses: `getView`, `renderAccess`
- Structured control flow: `renderPlainLoop`, `renderCarriedLoop`,
  `renderRegion`
- Chain materialization: `renderChain`
- Locality proof: `verifyTokenLocality`
- Full emitted checks: `verifyEmittedIR`
- Entry point: `emitIR`
