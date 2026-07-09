# ACCESS-DAG: accesses, owners, and boundaries

## Purpose

ACCESS-DAG is the memory fact layer. It answers *who touches what, where, and
how*: physical overlap, program-order accesses, executing owners, region
effects, and the owners visible at region boundaries. It does not create
semaphores or tokens; [SYNC-DAG](sync-dag.md) uses these facts to do that.

The examples use a compact schematic form. `W m0 ttg.local_alloc {0}` means
that the `ttg.local_alloc` writes member `m0` in partition 0. A region summary
such as `pieces{P0:W:{0}}` says that the region writes piece P0 and presents
owner `{0}` at its boundary. Tree lines show block nesting and program order,
not synchronization edges. Model terms are defined in the
[InsertSemas overview](overview.md#core-objects).

## What is analyzed

Only mutable SMEM allocations (`ttg.local_alloc` with mutable memory) and
all TMEM allocations (`ttng.tmem_alloc`, no mutability filter): the on-chip
scratch that one partition writes and another reads
(the communication buffers `InsertAllocas` created, plus TMEM accumulators).
Global memory, registers, and constants are invisible to `InsertSemas`.

An immutable `local_alloc` can still communicate across partitions —

```text
%mem = ttg.local_alloc %src {ttg.partition = [1]}    // no `mutable` in the type
ttng.tc_gen5_mma ... %mem ... {ttg.partition = [2]}  // read in another partition
```

— and this pass skips `%mem`, so who synchronizes partition 2 with
partition 1's write? Nobody has to, because this shape never reaches
`InsertSemas`: [InsertAllocas](../insert-allocas.md) is expected to run
first and rewrite it to

```text
%buf = ttg.local_alloc : !ttg.memdesc<..., mutable>   // communication buffer
ttg.local_store %src, %buf {ttg.partition = [1]}      // explicit producer write
ttng.tc_gen5_mma ... %buf ... {ttg.partition = [2]}   // reads the mutable buffer
```

which `InsertSemas` synchronizes like any other mutable member (the store is
the `W`, the MMA operand the `R`). Hand-written IR that feeds the immutable
form directly into `InsertSemas` gets no synchronization and no diagnostic.

## Groups

These allocations are grouped by `buffer.id`, with SMEM and TMEM in separate
namespaces; an allocation without an ID receives a private synthetic group.

The analysis runs **once per group**. ACCESS-DAG builds each group's accesses
and boundaries, and SYNC-DAG walks each group's ownership separately (the
per-piece source/use state is described in [SYNC-DAG](sync-dag.md)). The later
steps do share some state: SYNC-DAG validates circular sibling groups
together, threads a function-wide TMEM
budget, and runs final schedule legalization over the whole function;
EMIT-IR sees all groups at once so it can reunify shared storage. Across
groups, however, synchronization remains independent. This is valid because
upstream guarantees that distinct `buffer.id`s occupy disjoint storage,
while split groups with one `buffer.id` are planned to take turns using the
shared storage. An operation that touches two groups appears in both analyses,
each seeing only its own group.

The memory planner can make several logical buffers take turns occupying
one physical allocation (announced through the `buffer.*` attributes): at
any moment each byte belongs to exactly one of the buffers, and over the
loop's iterations the same byte serves first one, then another. Buffers that
never hold a byte at the same time do not conflict, so in the two
arrangements below each logical buffer is analyzed as its own group — own
pieces, own chain, own semaphores — while EMIT-IR materializes one shared
physical backing before it emits their protocol:

- a circular local member requires `buffer.id`, `buffer.copy`, and
  `buffer.start`, forbids `buffer.offset`, uses logical offset zero, and forms
  a separate group over the shared multi-buffered backing;
- when every TMEM member of one `buffer.id` has `buffer.copy` and the values
  differ, each member forms a separate group over the physical allocation.

The second rule is TMEM-only. Non-circular SMEM members with one `buffer.id`
remain one group; if their `buffer.copy` values are inconsistent, SYNC-DAG
later rejects the group with a diagnostic when it computes the backing plan
(`getPlannedBufferCopy`).

## Pieces

Within a group, members may partially overlap. `buffer.offset` places a
member's start on the group's address line; missing `buffer.offset` is zero.
(The Meta planner emits the attribute only on TMEM allocations — see
[meta-ports](../meta-ports.md#memory-planning) — but the analysis honors it
on any member, and SMEM members with explicit offsets appear in hand-written
and lit-test IR.) The address line is measured in TMEM columns for TMEM
(extent = the full memdesc size) and in leading-dimension elements for local
memory (extent = the leading shape dimension). The algorithm cuts that address line
at every member start and end and merges adjacent intervals with identical
coverage. The resulting disjoint intervals are the **pieces**; because the
cuts happen at member boundaries, each member is *exactly* the union of a
subset of pieces — its footprint.

That gives two levels:

```text
member (m)     the handle an access uses to name memory — an IR label, nothing more
piece (P)      the unit at which overlap is resolved and accesses are tracked
```

Worked example — `test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces`, group `buffer.id=500`, the planner-realistic
containment shape: a spanning owner `m0` with the reuser `m1` nested
inside it. The reuser's boundaries cut the owner into before, overlap, and
after:

```text
members:    m0[0,256)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0, m1}   P2=[192,256){m0}
footprints: m0={P0, P1, P2}   m1={P1}
```

An access through `m0` touches *all* of `{P0, P1, P2}`; one through `m1`
touches only `P1`. The overlap between the two members lives only in this
table — the walk below never compares addresses; it meets the overlap when
two accesses land on the same piece.

### Pieces must connect

`buildAccessDag` requires all of a group's pieces to connect through shared
members. Because pieces are ordered intervals, `buildPieces` checks that each
adjacent pair touches and shares at least one member. The rest of the pass
relies on this: one group is one synchronization unit, with one group-scoped
semaphore/token protocol
([SYNC-DAG](sync-dag.md)). Draw members and pieces as a graph — an edge
wherever a member covers a piece — and the requirement is that the graph is
one island:

```text
          m0          m1
        / | \         |
      P0  P1  P2      |       one island: the check passes
           ^──────────┘
     covered by BOTH members
```

The spanning owner `m0` covers all three pieces, so it alone links the
graph into one island; the nested reuser `m1` adds a second edge into the
overlap piece P1. Contrast: had the two members not overlapped at all —
same `buffer.id`, disjoint ranges — no member would link two pieces:

```text
members:    m0[0,128)   m1[128,256)
pieces:     P0=[0,128){m0}   P1=[128,256){m1}

      m0        m1
       |         |
      P0        P1            two islands: the group is rejected
```

The pass rejects that group with a diagnostic instead of analyzing it
(`test/NVWS/insert_semas_multi_component_error.mlir` covers the reject):

```text
error: nvws-insert-semas: buffer.id group has disjoint pieces (more than one
connected component); InsertSemas requires one component per group
```

The physical reading: same `buffer.id` with disjoint ranges is *packing* —
neighbors in one allocation, no byte in common, nothing an access to one can
do to the other — really two independent buffers, each needing its own
protocol. The memory planner never produces this shape (reusers are stacked
within their owner's columns — see
[meta-ports](../meta-ports.md#output-representation)), so the check can fire
only on hand-written IR; there it fails loudly rather than synchronizing two
independent buffers as one. Overlapping ranges, by contrast, share bytes —
writing one member clobbers part of the other — and the piece table records
exactly that overlap.

## Recognizing accesses: the value-to-member map

The scan must answer, for every operation, "does this touch the group — and
through which member?" It does so with a map keyed on **SSA values**: it
starts with each member's allocation result (`%a -> m0`, `%b -> m1`), and
the function is then walked in instruction order, recursing into `scf.for`
and `scf.if` bodies:

```text
for each op, in program order:
  if op is a view op whose source is in the map:
      map[op.result] = member            # new name, same member; no access
      continue
  touches = classify(op):                # every value resolved by map lookup
    local_load / tmem_load               -> (R, member of the loaded operand)
    stores; descriptor load/gather       -> (W, member of the destination
                                             operand)
    sourceful allocation of this group   -> (W, member of its own result)
    MMA                                  -> (W, accumulator member) +
                                            (R, each other group operand)
    control-flow op (for/if/yield/return
    with a map-hit operand)              -> rejected with a diagnostic
                                            (see "What is rejected" below)
    any other op                         -> (W, member) per map-hit operand
  if touches: append ONE access node (touches, op, partition)
```

An operation yields at most one access node, carrying one touch per map-hit
value (normally one per member it reached). Membership is always the same map
lookup: classification only decides which values to look up and whether each
one is read or written. An operation whose values all miss the map contributes
nothing.
The one write with no memdesc operand is the sourceful allocation: it
initializes the memory its own result names (that result is already in the
map from seeding), producing the `W m0 ttg.local_alloc {0}` lines in the
figures. A descriptor load's destination is an ordinary memdesc operand —
the op returns nothing.

The recognized view names are `ttg.memdesc_index`, `ttg.memdesc_subview`,
`ttg.memdesc_trans`, `ttg.memdesc_reinterpret`, and
`ttg.memdesc_reshape`. They create another name for the same allocation
without touching memory. A view **adds** a map entry; the old name stays valid
because later operations may use both. Any other operation that consumes a
group memdesc and produces one memdesc is rejected with a diagnostic. A
memdesc produced among several results falls to the fallback-`W` rule instead,
and its result is not tracked as an alias.

Note what the map does and does not answer. It answers only *"which member
is this name?"* — always unambiguous, because an SSA value derives from
exactly one allocation through view ops, overlap or not. *"Which memory does
that touch, and who else touches it?"* is answered by the piece table: the
read `R m0 {1}` and the write `W m1 {2}` resolve through the map privately,
without knowing about each other, and their conflict is detected because
both fan out to the shared piece `P1`.

## The chain

Per group, the walk produces **one chain**: the program-order sequence of
nodes over all of the group's members, interleaved as they execute. `Access`
nodes carry (R/W, member, op, partition); a `for`/`if` becomes a node
holding child chains, and is kept only when its body touches the group.
Ignoring boundary nodes until the [regions section](#regions-and-boundaries),
the `@serialized_ring_reduces` group above has this access chain:

```text
|- scf.for (WS, tag=0) effects{P0:W, P1:W, P2:W}
|  |- W  m0  ttg.local_alloc {0}
|  |- R  m0  ttg.local_load {1}
|  |- W  m1  ttg.local_alloc {2}
|  |- R  m1  ttg.local_load {0}
```

Read through the footprints, the same chain is a per-piece access history —
the view the later steps actually consume:

```text
P0: W{0}, R{1}                    (only m0 reaches it)
P1: W{0}, R{1}, W{2}, R{0}        (both members reach it — the overlap)
P2: W{0}, R{1}                    (only m0 reaches it)
```

The loop node's `effects{P0:W, P1:W, P2:W}` annotation is the per-piece merge
of exactly these lists — `W` if any access in the body writes the piece,
otherwise `R`. The enclosing chain treats the whole loop as one event, and
`effects{}` is what that event looks like from outside.

Each access inside a tagged WS scope records its one executing partition as
`(ttg.partition, WS tag)` and keeps the view chain needed to rebuild its
memdesc during emission. An access outside a tagged WS scope records root.

Inside a tagged WS scope, every access to one of these buffers must execute
in one partition. Multi-partition accesses are not supported; as described
in the [overview](overview.md#core-objects), they are not diagnosed and are
silently treated as root-owned.

## Memory effects

- `R`: `local_load`, `tmem_load`, and non-accumulator MMA operands whose
  memdesc belongs to the group;
- `W`: SMEM/TMEM stores, sourceful SMEM/TMEM allocations, descriptor
  load/gather destinations, and MMA accumulators;
- fallback `W`: any other operation that directly consumes a group memdesc is
  conservatively recorded as `W` — unknown means "assume exclusive".

Fallback `W` means an exclusive synchronization touch, not necessarily a
literal memory store. If one operation reaches the same physical piece through
multiple views, its effects are combined; any `W` makes the combined effect
`W`.

## What is rejected, what is invisible

A group whose pieces do not all connect through shared members is rejected
before any scanning (see [Pieces must connect](#pieces-must-connect)).

Accesses inside `scf.for` and `scf.if` are supported. Passing a group's
memdesc through an `scf.for` argument/result, an `scf.yield`, or a function
return is rejected with a diagnostic, as is any unsupported view-shaped
operation (the closed view-op list under "Recognizing accesses"). Other
region-holding operations (for example `scf.while`) are not scanned: accesses
inside their bodies are invisible to the analysis, with no diagnostic. If
such an operation consumes a group memdesc, it is conservatively recorded as
a write — unless its sole result is a memdesc, in which case the
view-operation rule rejects it.

## When an access finishes

Normally an access completes at the memory operation itself. A `local_load`
from one of these SMEM buffers that feeds exactly one same-block descriptor
store, directly or through one `convert_layout`, remains physically live
until that store: TMA-store lowering turns that load/store pair into one
async copy issued directly out of the SMEM buffer, so the buffer must stay
protected until the store. The DAG records the store as `completionAnchor`;
later steps use it for release placement and scheduling without
re-discovering the pattern.

If no descriptor-store candidate exists, the ordinary completion point is
kept. Once a candidate exists, every extra direct load/convert user,
control-flow crossing, or owner mismatch fails the pass with a diagnostic.

## Regions and boundaries

A nested `for` or `if` occupies one position in its parent chain even though
its body may change owners several times. The region node therefore records,
for each piece its children touch:

- `R` if every child access only reads the piece, otherwise `W`;
- the owner used by that region's `ENTER` and `EXIT` boundaries.

That stored boundary owner normally contributes to the parent chain too. A
WS-tagged `for` is the exception: it stores the first owner inside the loop
for its children, but contributes root ownership to its enclosing chain.

Each child chain is enclosed by `ENTER` and `EXIT` nodes:

```text
For or If in the parent chain
  child chain
    ENTER pieces{piece:effect:owner}
    ... child accesses and nested regions ...
    EXIT  pieces{piece:effect:owner}
```

`ENTER` and `EXIT` are model nodes, not MLIR operations. They carry no
`loop.stage` or `loop.cluster`. They give SYNC-DAG concrete child-chain
endpoints for a value that enters or leaves a region. An `if` has one child
chain for each branch; a missing else body is represented by an empty chain.

### Choosing a boundary owner

Child regions are completed before their enclosing region, so a nested
`for` or `if` can contribute its already-known boundary owner.
`ENTER` and `EXIT` do not participate in selecting that owner.

For each piece touched by a `for`, the owner is the first owner that touches
the piece in the body. A direct access contributes its access owner. A nested
region contributes the owner visible on that nested region.

For each piece touched by an `if`, the preferred owner is the latest owner of
that piece before the `if` in the parent chain. If the parent has not touched
the piece, the first owner in the then chain is used, followed by the first
owner in the else chain.

A WS-tagged loop presents root ownership to its parent because its partition
owners are meaningful only inside that WS scope. The owner stored on the loop
for its own children is still the first owner inside the loop. An access with
its own WS tag resolves that tag directly even when no enclosing loop has it.

These rules determine ownership at the boundary. They do not imply that the
same owner holds the piece throughout the child, and they do not choose any
semaphore or token.

### Worked example: a loop can have a different owner per piece

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` has two staggered members:

```text
members: m0[0,128)  m1[64,192)
pieces:  P0={m0}    P1={m0,m1}    P2={m1}
```

This is a useful stress shape because staggering gives the pieces different
first owners. Planner-produced reuse normally nests the smaller member inside
the larger one, as in the [pieces example](#pieces).

Its access and boundary nodes are:

```text
|- scf.for (WS, tag=1) pieces{P0:W:{0}, P1:W:{0}, P2:W:{2}}
|  |- ENTER pieces{P0:W:{0}, P1:W:{0}, P2:W:{2}}
|  |- W  m0  ttg.local_alloc {0}
|  |- R  m0  ttg.local_load {1}
|  |- W  m1  ttg.local_alloc {2}
|  |- R  m1  ttg.local_load {0}
|  |- EXIT pieces{P0:W:{0}, P1:W:{0}, P2:W:{2}}
```

For P0 and P1, the first body access is `W m0 {0}` because m0 covers both
pieces. Their boundary owner is therefore `{0}`. No m0 access reaches P2, so
its first access is `W m1 {2}` and its boundary owner is `{2}`. Every piece is
written somewhere in the body, so every merged effect is `W`.

### Worked example: an `if` prefers the preceding owner

In `test/NVWS/insert_semas_raw_if_token.mlir`
`@raw_edge_token_carried_if`, the `if` contains only a read by partition
`{1}`, but its boundary owner is `{0}`:

```text
|  |- W  m0  ttng.tmem_store {0}
|  |- scf.if pieces{P0:R:{0}}
|  |  |- then
|  |  |  |- ENTER pieces{P0:R:{0}}
|  |  |  |- R  m0  ttng.tmem_load {1}
|  |  |  |- EXIT pieces{P0:R:{0}}
|  |  |- else
|  |  |  |- ENTER pieces{}
|  |  |  |- EXIT pieces{}
```

The latest access before the `if` is the store by `{0}`, so P0 enters the
region with boundary owner `{0}`. The merged effect is `R` because the branch
only reads P0. The read still has owner `{1}`. SYNC-DAG therefore derives the
`{0}` to `{1}` handoff inside the branch; matching `ENTER` and `EXIT` owners
do not erase ownership changes inside the child.

### What `ENTER` and `EXIT` record

For each child chain, `ENTER` and `EXIT` contain exactly the pieces that chain
touches. The two nodes have identical records: each piece has the child
chain's merged effect and the owner stored on the enclosing `for` or `if`. An
empty child has empty boundary records.

When SYNC-DAG enters a child, it preserves the logical producer known by the
parent but uses `ENTER` as the source node local to that child chain. New
readers in the child therefore receive the value from `ENTER`, not from a
previous child reader or directly from an outer node. At `EXIT`, the same
boundary owner is available for a handoff to a later parent access or to the
next loop iteration.

The boundary owner and the token returned by each child path also tell
SYNC-DAG whether a loop must carry a token or can acquire one immediately
before its first guarded buffer use.

## The algorithm

```text
1. group the mutable SMEM allocations and all TMEM allocations by
   buffer.id within each memory kind; no buffer.id -> private group
2. per group:
   a. cut member intervals -> pieces; check that every piece connects
      through shared members (reject the group otherwise)
   b. seed the map: each member's allocation result -> that member
   c. walk the function in instruction order, recursing into scf.for/scf.if:
        view op with a map-hit source -> add map[result] = member; no access
        otherwise classify the op     -> R/W touches on its map-hit values
                                         (see Recognizing accesses)
        any touches                   -> append one access node
                                         (touches, op, partition)
      keep a for/if node only when one of its child chains touched the group
   d. when returning from each kept for/if:
        merge child effects per piece
        choose each piece's boundary owner
        wrap every child chain in matching ENTER/EXIT nodes
        append the region node to its parent chain
```

## Output

The step produces, per group:

```text
members + pieces + view-chain map
one program-order chain of Access/For/If nodes
per-access owner/effect/completion facts
pieces, R/W effects, and boundary owners recorded on each for/if node
matching ENTER/EXIT nodes around each child chain
```

## Code map

[`InsertSemasAccessDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.cpp)
contains:

- `collectGroups`
- `buildPieces` (including the pieces-must-connect check)
- `collectTouches` and `deriveCompletionAnchor`
- `appendNode` and `buildChainForBlock`
- `buildAccessDag`
