# ACCESS-DAG

## Purpose

ACCESS-DAG is the memory fact layer: it answers *who touches what, where,
and how* — physical overlap, program-order accesses, executing partitions,
and region effects — without deciding any synchronization. Who *owns* memory
and when ownership must move is decided by the later steps
([OWNER-DAG](owner-dag.md), [SYNC-DAG](sync-dag-1.md)); nothing in this step
depends on those answers.

Figures use the pass's dump format (`NVWS_INSERT_SEMA_DUMP_DAG=1`, trimmed
excerpts): `W m0 ttg.local_alloc {0}` is a write of member `m0` by the op
`ttg.local_alloc` in partition 0, and region rows carry
`effects{piece:effect}`. Model terms are defined in the
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

The analysis runs **once per group**: ACCESS-DAG and OWNER-DAG are fully
independent per group, and SYNC-DAG walks each group's ownership
separately (the per-piece games — see [SYNC-DAG](sync-dag-1.md)). The later steps do share some state — SYNC-DAG validates
circular sibling groups together, threads a function-wide TMEM budget,
validates the mixed-depth TMEM aliases of one `buffer.id` as a pair (an
alternating two-partition lifecycle with pipeline-slack checks — see
[SYNC-DAG](sync-dag-1.md#mixed-depth-tmem-aliases-schedule-validation-across-split-groups)),
and its final schedule legalization runs over the whole
function; EMIT-IR
sees all groups at once so it can reunify shared storage — but across
distinct `buffer.id`s no step ever tracks a *conflict*. The license for
that is a contract from
upstream: distinct `buffer.id`s mean the planner placed the allocations in
disjoint storage (and `InsertAllocas` creates unrelated buffers as separate
groups), so those cross-group conflicts cannot exist by construction. Split
groups that share one `buffer.id` (circular, mixed-depth) instead rely on
validated time-sharing — the mixed-depth pair validation is the one step
that checks a hazard between two groups. The dump
prints one complete `GROUP` block per group; an op that touches two groups
appears in both analyses, each seeing only its own facet.

The memory planner can make several logical buffers take turns occupying
one physical allocation (announced through the `buffer.*` attributes): at
any moment each byte belongs to exactly one of the buffers, and over the
loop's iterations the same byte serves first one, then another. Buffers that
never hold a byte at the same time do not conflict, so in the two
arrangements below each logical buffer is analyzed as its own group — own
pieces, own chain, own semaphores — and EMIT-IR re-merges the shared storage
at the very end:

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
members — one connected component (`piecesSingleComponent`, a union-find
over the piece table). The rest of the pass relies on it: one group is one
synchronization unit, with one group-scoped semaphore/token protocol
([SYNC-DAG](sync-dag-1.md)). Draw members and pieces as a graph — an edge
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
exactly that entanglement.

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

An op yields at most one access node, carrying one touch per map-hit value
(normally one per member it reached); the ACCESS dump prints one row per
touch, and the SYNC dump joins them on a single row. Membership is always the same plain map lookup —
classification only decides which of the op's values to look up and with
what effect, and an op whose values all miss the map contributes nothing.
The one write with no memdesc operand is the sourceful allocation: it
initializes the memory its own result names (that result is already in the
map from seeding), producing the `W m0 ttg.local_alloc {0}` rows in the
figures. A descriptor load's destination is an ordinary memdesc operand —
the op returns nothing.

The view ops are a closed list — `memdesc_index`, `memdesc_trans`,
`memdesc_reinterpret`, `memdesc_reshape` — operations that create another
name for the same allocation without touching memory. (The code's whitelist
carries a fifth, dead name, `ttg.memdesc_subview`; no such op exists in this
repo.) A view **adds** a map entry; the old name stays valid, since later
ops may use both. Any other operation that consumes a group memdesc and
produces a single memdesc is rejected with a diagnostic — in particular
`ttg.memdesc_subslice` is rejected, not followed — so a single-result
memdesc alias can never silently escape the map (a memdesc produced among
multiple results falls to the fallback-`W` rule instead; its result is not
tracked).

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
The `@serialized_ring_reduces` group above:

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

The loop row's `effects{P0:W, P1:W, P2:W}` annotation is the per-piece merge
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
operation (the closed view-op list under "Recognizing accesses"). Other region-holding operations (for example
`scf.while`) are not scanned: accesses inside their bodies are invisible to
the analysis, with no diagnostic. If such an operation consumes a group
memdesc, it is conservatively recorded as a write — unless its sole result
is a memdesc, in which case the view-operation rule rejects it.

## Completion frontier

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

## Effects inside `for` and `if`

For every retained `for` and `if`, ACCESS-DAG records each piece accessed in
its body. The recorded effect is `W` if any access writes the piece, otherwise
`R` (the `effects{...}` row in the figure above). No owner is assigned yet,
and the `ENTER`/`EXIT` boundary markers are added by the next step,
OWNER-DAG.

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
      keep a for/if node (with its child chains) only when its body touched
      the group
3. per retained for/if: record the pieces its body touches and the merged
   R/W effect
```

## Output

The step produces, per group:

```text
members + pieces + view-chain map
one program-order chain of Access/For/If nodes
per-access owner/effect/completion facts
pieces and R/W effects recorded on each for/if node
```

## Code map

[`InsertSemasAccessDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.cpp):

- `collectGroups`
- `buildPieces` and `piecesSingleComponent` (the pieces-must-connect check)
- `collectTouches` and `deriveCompletionAnchor`
- `buildChainForBlock` and `computeEffectSummary`
- `buildAccessDag`
- the dump used in the figures: `NVWS_INSERT_SEMA_DUMP_DAG=1`
  (`dumpGroupAccessDag` prints the group header, then delegates the tree to
  `dumpDagTree` in `InsertSemasSyncDag.cpp`)
