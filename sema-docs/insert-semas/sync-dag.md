# SYNC-DAG

## Contents

- [What SYNC-DAG builds](#what-sync-dag-builds)
- [How to read the examples](#how-to-read-the-examples)
- [From accesses to raw edges](#from-accesses-to-raw-edges)
  - [Per-piece state](#per-piece-state)
  - [The per-access rules, in full](#the-per-access-rules-in-full)
  - [Memory edges and token supply](#memory-edges-and-token-supply)
  - [Example: one writer and one reader](#example-one-writer-and-one-reader)
  - [Example: independent readers and exact token reuse](#example-independent-readers-and-exact-token-reuse)
  - [Example: disjoint pieces stay independent](#example-disjoint-pieces-stay-independent)
  - [Composition: nested regions in the walk](#composition-nested-regions-in-the-walk)
  - [Example: counted recurrence at two region levels](#example-counted-recurrence-at-two-region-levels)
  - [Example: the boundary owner is unchanged](#example-the-boundary-owner-is-unchanged)
- [Reducing raw edges](#reducing-raw-edges)
  - [Straight-chain implication](#straight-chain-implication)
  - [Loop-close implication](#loop-close-implication)
  - [Async completion and release anchors](#async-completion-and-release-anchors)
  - [Example: one handoff preserves two completion kinds](#example-one-handoff-preserves-two-completion-kinds)
  - [Example: a preserved async edge stays direct](#example-a-preserved-async-edge-stays-direct)
  - [Example: repeated same-owner sources release after the later source](#example-repeated-same-owner-sources-release-after-the-later-source)
  - [Example: a direct edge is implied](#example-a-direct-edge-is-implied)
  - [Example: a loop-close edge is implied](#example-a-loop-close-edge-is-implied)
- [Direct protocol construction](#direct-protocol-construction)
  - [Exact token identity](#exact-token-identity)
  - [Accesses and straight-line handoffs](#accesses-and-straight-line-handoffs)
  - [Conditional paths](#conditional-paths)
  - [Loops](#loops)
  - [POU, FirstTouch, and Auto](#pou-firsttouch-and-auto)
  - [Example: the two loop placements](#example-the-two-loop-placements)
  - [Example: Auto discards and rebuilds](#example-auto-discards-and-rebuilds)
  - [Example: branch alternatives have count one](#example-branch-alternatives-have-count-one)
  - [Example: a POU plan can still carry a token](#example-a-pou-plan-can-still-carry-a-token)
  - [Example: nested POU without token carriers](#example-nested-pou-without-token-carriers)
  - [Example: a parent continuation needs a bridge](#example-a-parent-continuation-needs-a-bridge)
  - [Example: scheduled nested POU is placed directly](#example-scheduled-nested-pou-is-placed-directly)
  - [Example: branch completion keeps path-specific schedules](#example-branch-completion-keeps-path-specific-schedules)
- [Forming semaphore channels](#forming-semaphore-channels)
  - [Executed supplies and alternative supplies](#executed-supplies-and-alternative-supplies)
  - [Uniform pending counts](#uniform-pending-counts)
  - [Entry state](#entry-state)
- [Backing copies](#backing-copies)
  - [Physical copies](#physical-copies)
  - [Example: a TMEM accumulator gets two copies](#example-a-tmem-accumulator-gets-two-copies)
  - [Semaphore copies](#semaphore-copies)
  - [Example: a TMA load stages only semaphore state](#example-a-tma-load-stages-only-semaphore-state)
- [Finalizing the pipeline schedule](#finalizing-the-pipeline-schedule)
  - [Release/acquire constraints](#releaseacquire-constraints)
  - [Recurrence distance](#recurrence-distance)
  - [Circular and alias offsets](#circular-and-alias-offsets)
  - [Example: circular K and V select different copies](#example-circular-k-and-v-select-different-copies)
  - [Example: a non-circular alias advances the copy](#example-a-non-circular-alias-advances-the-copy)
  - [Example: one physical slot](#example-one-physical-slot)
- [Validation boundary](#validation-boundary)
- [Build order and code map](#build-order-and-code-map)

## What SYNC-DAG builds

ACCESS-DAG has already grouped memory, split overlapping members into pieces,
assigned an owner to every access, and built structured chains with `ENTER`
and `EXIT` nodes. SYNC-DAG turns those facts into one complete symbolic
synchronization plan.

The implementation runs these steps in this order:

```text
ACCESS-DAG chains
        |
        v
ChainWalker
  find where one partition must wait for another
        |
        v
reduceEdges
  remove only implied straight-chain and loop-close edges
        |
        v
computeBackingCopies
  choose the physical copy count from the reduced edge set
        |
        v
DirectBuilder
  place final acquires and releases
  assign every consumer an exact tokenSource
  build RegionFlow only when a token really crosses a region
        |
        v
formSemaphores
  union acquire sites into channels
  choose one pending count and optional entry owner per channel
        |
        v
computeRequiredParts / computeSemaphoreCopies
  seal region partition requirements and semaphore staging depth
        |
        v
validatePOUPlan / validateTokenConnectivity / verifySyncDag
        |
        v
finalizeSyncSchedule
  solve copy offsets, owner delays, and loop-cluster constraints
        |
        v
EMIT-IR
  materialize the already sealed plan
```

Edge reduction is therefore part of the POU design. It is common
preprocessing for every placement mode. POU versus FirstTouch is consulted
later, while `DirectBuilder` handles a loop.

In `placement-mode=auto`, the sequence above is transactional. The driver
first builds a complete candidate with POU enabled. If a particular loop
rejects POU, the driver throws away every group in that candidate, restores
schedule attributes, and rebuilds with that `(group, loop)` forced to
FirstTouch. Every accepted graph contains only the placements selected for
that successful candidate.

The principal boundary is:

> SYNC-DAG fixes protocol structure and exact token identity. EMIT-IR renders
> that sealed plan exactly.

## How to read the examples

Every worked protocol below names the lit function that produces it,
except the explicitly identified inline `@same_owner_nested`,
`@doc_preserved_async_edge`, and `@doc_repeated_same_owner_sources` inputs.
Those inputs are reproduced in high-level form and were run through the
current pass. For lit-backed cases, the edges, acquire sites, release
sites, counts, entry state, and token carriers are checked against source, the
live pass dump, and FileCheck. The inline inputs are checked against current
source and their live pass dumps, but are not claimed to be current lit
contracts.

The completed tree can be inspected with:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt input.mlir \
  -allow-unregistered-dialect --nvws-insert-semas
```

The pass assigns mechanical names `S0`, `S1`, and so on. The examples use
semantic aliases such as `EMPTY` and `FULL` so that one channel has the same
name throughout a derivation.

Notation:

```text
W m0 {0}              owner {0} writes member m0
R m0 {1}              owner {1} reads member m0
ENTER / EXIT           analysis boundaries of one structured region
e1: A -> B             synchronization edge: B must wait for A
a FULL(2) {2}          acquire FULL with pending_count=2 by owner {2}
r FULL(2) {3}          one release op contributing arrive_count=2
[tma_load]             release waits for that asynchronous completion
tokenSource=T          the consumer uses the exact token produced by T
```

An edge DAG and a semaphore DAG are different objects:

```text
edge DAG                    semaphore DAG

W m0 {0}                   W m0 [tw] {0}
   | e1                          | tw
   v                             v
R m0 {1}                   r FULL, tw {0}
                                | FULL
                                v
                         tr = a FULL {1}
                                | tr
                                v
                         R m0 [tr] {1}
```

An arrow labeled `walk` is program/control order, not a synchronization edge.
A line labeled with a semaphore connects one release
phase to one acquire phase. Mutually exclusive consumers are always drawn in
separate diagrams; a single rail never means that two acquire sites execute
together.

## From accesses to raw edges

<a id="the-walk-accesses-to-edges"></a>

`ChainWalker` visits each group in structured program order. Its job is to
record obligations, not to decide where semaphore operations will be placed.
It carries two kinds of state:

```text
per piece                       per chain

current version source         provisional logical token/source records
latest use by each owner        most recent usable source order
known order facts
```

The result is a vector of `EdgeRec` values. An edge stores concrete source and
destination nodes, source and destination owners, affected pieces,
asynchronous completion payloads, and whether its exact source carrying an
asynchronous completion must be preserved.

### Per-piece state

For each piece, `PieceState` contains a `VersionSource` and a map of
`ActiveUse` records:

```text
VersionSource
  producer       logical owner that produced this memory version
  sourceOwner    owner of the chain-local source node
  node           chain-local node from which a new reader receives the value
  payloads       completion required before that value is ready

uses
  owner -> latest node at which that owner uses the current version
           completion payloads at that node
           owners already proved to execute after that node
```

The logical producer and chain-local source can differ. A child region may
receive a value produced in its parent; inside the child, `ENTER` is the local
source while the producer identity still describes the inherited version.

A read updates only its owner's entry in `uses`. It does not move the version
source. Independent readers therefore fan out from the write or `ENTER`.
A write creates a new version and resets `uses` to that writer.

<a id="the-per-access-rules-in-full"></a>

### The per-access rules, in full

For one touched piece `P`:

```text
first touch of P
  establish this node as source and first use
  add no memory edge

read P by an owner already present in uses
  replace only that owner's latest-use node
  replace its completion payloads
  clear its old ordered-before facts
  add no memory edge

read P by a new owner
  if the chain-local source has another owner:
    add source -> read
    when the source use is exact, record that the source is ordered before
    this new owner
  add the reader to uses

write P
  for each written piece of a synchronous write in a multi-member group:
    if the operation's provisional token is reusable across every touched
    piece, that piece's current source has the same owner, and all of that
    piece's live uses have that owner:
      carry that source's unresolved completion payloads into this write
  for every latest use by another owner not already ordered before this write:
    add use -> write
  reset source and uses to this write

warp-specialized region adoption
  when a WS loop adopts an unpartitioned root version at its boundary,
  do not create a synthetic root-to-partition memory edge
```

When the source operation is asynchronous, a direct edge from that exact
operation carries its completion payload. Such an edge is marked `preserve`
when replacing it with a path through a synchronous access would lose the
operation whose completion matters.

At a region node the parent applies the same rules to the region summary. A
child chain starts from its own `ENTER` records and applies the same rules to
its real accesses. Parent and child edges remain distinct.

<a id="memory-edges-and-token-supply"></a>

### Memory edges and token supply

Memory order alone is not sufficient: every rewritten managed access will
eventually need a buffer view backed by one exact semaphore token.
`ChainWalker` maintains provisional logical token/source records so it can
identify candidate supplies and legal reuse before protocol nodes exist:

```text
owner, provisional source node, last completion node, payloads, closed flag
```

After applying the memory rules to an access:

- if the access has a memory edge, that edge is a candidate incoming supply;
  reduction may later replace it with a proved kept path and token reuse;
- if the owner already has a token valid for every touched piece, the access
  is marked reusable; or
- if neither is true and the current token belongs to another owner, a
  token-supply edge is added from that token's last node.

Owner equality is not the reuse proof. For a read, every touched piece must
already have a use by that owner. For a write, every other live use must
already be ordered before the writer. `DirectBuilder` later turns these facts
into explicit `tokenSource` pointers.

### Example: one writer and one reader

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` contains two
groups. For the per-iteration group `buffer.id=104`, the relevant input is:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

There is one piece, P0. The write creates the version; the new reader creates
`e1`. Because the loop may execute again, `EXIT` returns P0 to boundary owner
`{0}` and creates `e2` from the final foreign use:

```text
edge    source             destination       reason
e1      W m0(i) {0}       R m0(i) {1}       read after write
e2      R m0(i) {1}       EXIT(i) {0}        next iteration may overwrite P0
```

Neither edge is redundant:

```text
W m0(i) {0}
     | e1
     v
R m0(i) {1}
     | e2
     v
 EXIT(i) {0}
```

The accepted Auto candidate uses POU. `e1` becomes `FULL`; `e2` becomes an
initially released recurrence channel `EMPTY`:

```text
edge    channel    count    entry state
e1      FULL       1        blocked
e2      EMPTY      1        released for owner {0}
```

The completed semaphore DAG for one iteration is:

```text
ENTER(i) {0}
     | walk
     v
tw = a EMPTY {0}
     |
     v
W m0(i) [tw] {0}
     | walk
     v
r FULL, tw {0}                 e1
     | FULL
     v
tr = a FULL {1}
     |
     v
R m0(i) [tr] {1}
     | walk
     v
r EMPTY, tr {1}                e2
     | walk
     v
 EXIT(i) {0}
```

The next dynamic iteration executes the same static acquire:

```text
r EMPTY, tr(i) {1}
     | EMPTY phase
     v
tw = a EMPTY in iteration i+1 {0}
     |
     v
W m0(i+1) [tw] {0}
```

For iteration zero, the initially released `EMPTY` phase supplies `tw`. The
loop has no semaphore-token operand or result. A zero-trip loop executes none
of these protocol nodes and leaves `EMPTY` released.

### Example: independent readers and exact token reuse

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced` has this one-piece loop:

```text
for {
  W m0 {0}
  R m0 {1}
  R m0 {2}
  R m0 {0}
}
```

Both foreign readers receive the version from the write. The final owner-`{0}`
read reuses owner `{0}`'s original token and adds no edge. The loop close waits
for both foreign readers:

```text
edge    source          destination
f1      W m0 {0}       R m0 {1}
f2      W m0 {0}       R m0 {2}
f3      R m0 {1}       EXIT {0}
f4      R m0 {2}       EXIT {0}
```

```text
                     +-- f1 --> R m0 {1} -- f3 --+
W m0 {0} ------------+                           +--> EXIT {0}
                     +-- f2 --> R m0 {2} -- f4 --+

R m0 {0}: program order only; tokenSource is the write-phase acquire
```

Both loop-close edges supply the same owner-`{0}` recurrence acquire and
execute on every iteration. They therefore share one count-2 `EMPTY` channel.
The two outgoing read edges remain separate count-1 channels:

```text
edge      channel    count    entry state
f1        TO_R1      1        blocked
f2        TO_R2      1        blocked
f3,f4     EMPTY      2        released for owner {0}
```

Producer phase:

```text
t0 = a EMPTY(2) {0}
     |
     v
W m0 [t0] {0}
     |
     +--> r TO_R1, t0 {0} ----> t1 = a TO_R1 {1} ----> R m0 [t1] {1}
     |
     +--> r TO_R2, t0 {0} ----> t2 = a TO_R2 {2} ----> R m0 [t2] {2}
```

Reader completion and recurrence:

```text
R m0 [t1] {1} ----> r EMPTY, t1 {1} --+
                                      +--> EMPTY phase (2 arrivals)
R m0 [t2] {2} ----> r EMPTY, t2 {2} --+         |
                                                +--> next = a EMPTY(2) {0}
```

The later `R m0 {0}` names `t0`'s producer exactly. It does not accidentally
use whichever reader token was emitted most recently. No semaphore token is
returned by the loop.

### Example: disjoint pieces stay independent

`test/NVWS/insert_semas_tmem_container_subviews.mlir`
`@container_with_disjoint_subviews` is the concrete disjoint-piece example.
One container covers three pairwise
disjoint subviews:

```text
member    range
m0        [0,256)       container
m1        [0,128)       left subview
m2        [128,192)     middle subview
m3        [192,256)     right subview

piece     range          members
P0        [0,128)        m0,m1
P1        [128,192)      m0,m2
P2        [192,256)      m0,m3
```

The relevant access shape is:

```text
for {
  W m0 {0}              writes P0,P1,P2
  W m1 {1}              writes P0
  W m2 {2}              writes P1
  W m3 {3}              writes P2
  R m1 {1}
  R m2 {2}
  R m3 {3}
}
```

The reduced edge inventory is:

```text
edge    pieces    source          destination
e1      P0        W m0 {0}        W m1 {1}
e2      P1        W m0 {0}        W m2 {2}
e3      P2        W m0 {0}        W m3 {3}
e4      P0        R m1 {1}        EXIT {0}
e5      P1        R m2 {2}        EXIT {0}
e6      P2        R m3 {3}        EXIT {0}
```

```text
P0 path: W m0(i) {0} -- e1 --> W m1 {1} -- walk --> R m1 {1} -- e4 --> EXIT(i) {0}
P1 path: W m0(i) {0} -- e2 --> W m2 {2} -- walk --> R m2 {2} -- e5 --> EXIT(i) {0}
P2 path: W m0(i) {0} -- e3 --> W m3 {3} -- walk --> R m3 {3} -- e6 --> EXIT(i) {0}
```

The three rows repeat the same `W m0` and `EXIT` nodes so that each piece's
path is visible without a drifting three-column fork.

There is no edge between the three subview paths. The common container write
fans out, and all three paths must close before the next container write. The
channel assignment is:

```text
edge        channel    pending_count    initial state
e1          P0_FULL    1                blocked
e2          P1_FULL    1                blocked
e3          P2_FULL    1                blocked
e4,e5,e6    EMPTY      3                released for owner {0}
```

The exact POU plan uses one count-three acquire for the container and three
independent count-one handoffs:

```text
scf.for {
  whole = a EMPTY(3) {0}
  W m0 [whole] {0}
  r P0_FULL, whole {0}
  r P1_FULL, whole {0}
  r P2_FULL, whole {0}

  p0 = a P0_FULL {1}
  W m1 [p0] {1}
  p1 = a P1_FULL {2}
  W m2 [p1] {2}
  p2 = a P2_FULL {3}
  W m3 [p2] {3}

  R m1 [p0] {1}
  r EMPTY, p0 {1}
  R m2 [p1] {2}
  r EMPTY, p1 {2}
  R m3 [p2] {3}
  r EMPTY, p2 {3}
}
```

The symbolic order above matches the live dump. The dependency DAG is easier
to read with the independent arms separated. Each row below starts at the
same `W m0(i)` node; it is repeated only to make the three fan-out paths
explicit:

```text
whole = a EMPTY(3) {0} --> W m0(i) [whole] {0}

P0: W m0 [whole] --> r P0_FULL --> p0 = a P0_FULL --> W m1 --> R m1 --> r EMPTY, p0
P1: W m0 [whole] --> r P1_FULL --> p1 = a P1_FULL --> W m2 --> R m2 --> r EMPTY, p1
P2: W m0 [whole] --> r P2_FULL --> p2 = a P2_FULL --> W m3 --> R m3 --> r EMPTY, p2

fanin: {r EMPTY, p0 {1}, r EMPTY, p1 {2}, r EMPTY, p2 {3}}
       --> EMPTY phase with 3 arrivals --> next = a EMPTY(3) {0}
       --> W m0(i+1) [next] {0}
```

The backing has two physical copies, but that does not merge P0, P1, and P2
into one ordering chain. A zero-trip loop executes no protocol operation and
leaves `EMPTY` initially released.

<a id="composition-nested-regions-in-the-walk"></a>

### Composition: nested regions in the walk

A `for` or `if` occupies one node in its parent chain. Its `pieceInfo` gives
the parent one aggregate effect and one boundary owner per touched piece:

```text
parent chain: ... -> [region summary] -> ...

child chain:         ENTER -> real accesses -> EXIT
```

`ChainWalker::visitRegion` first applies the normal piece rule to the summary
node. It then creates independent child state seeded from the incoming
version, walks each child, and intersects order facts that are true on every
returning path. A possibly-zero-trip loop cannot establish an order fact only
proved by its body.

The parent never substitutes child access nodes for the summary. This is why
an outer edge can end at a loop node while a different set of edges exists
inside that loop.

### Example: counted recurrence at two region levels

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` is the first complete nested
example. Its relevant input is:

```text
outer for {
  W m0 {3}

  inner for {
    R m0 {2}
    R m0 {1}
    W m0 {1}
    R m0 {0}
  }
}
```

There is one piece P0. The inner loop first touches P0 with a read owned by
`{2}` and later writes it, so its parent summary is `P0:W:{2}`.

The two loop levels are analyzed separately. At the outer level, the inner
loop is one summary node. The complete parent edge inventory is:

```text
edge    source                  destination
p1      W m0 {3}               inner summary {2}
p2      inner summary {2}      EXIT outer {3}
```

```text
                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0(i) {3}
                                  | p1
                                  v
                   [inner summary P0:W:{2}]
                                  | p2
                                  v
                         EXIT outer(i) {3}
```

The child edge inventory is independent:

```text
edge    source             destination
c1      ENTER inner {2}    R m0 {1}
c2      R m0 {2}           W m0 {1}
c3      W m0 {1}           R m0 {0}
c4      W m0 {1}           EXIT inner {2}
c5      R m0 {0}           EXIT inner {2}
```

```text
                         ENTER inner(i,j) {2}
                         +---------+---------+
                    walk |                   | c1
                         v                   v
                     R m0 {2}            R m0 {1}
                      c2 |                   | walk
                         +---------+---------+
                                   v
                              W m0 {1}
                         +---------+---------+
                      c3 |                   | c4
                         v                   |
                     R m0 {0}                |
                      c5 |                   |
                         +---------+---------+
                                   v
                          EXIT inner(i,j) {2}
```

In pure ordering terms, `c3` followed by `c5` would imply `c4`. The reducer
nevertheless keeps `c4`: `reduceLoopCloses` deliberately preserves a close
whose destination is the chain's first access owner, because that close helps
open the next wave. Consequently the owner-`{1}` write and owner-`{0}` read
both contribute to the next owner-`{2}` acquire.

The parent and child levels together are therefore:

```text
parent row
  W m0(i) {3} -- p1 --> [inner summary P0:W:{2}] -- p2 --> EXIT outer(i) {3}

child DAG represented by that summary
  ENTER inner(i,j) {2} -- walk --> R m0 {2}
  ENTER inner(i,j) {2} -- c1 ----> R m0 {1}
  R m0 {2}             -- c2 ----> W m0 {1}
  R m0 {1}             -- walk --> W m0 {1}
  W m0 {1}             -- c3 ----> R m0 {0}
  W m0 {1}             -- c4 ----> EXIT inner(i,j) {2}
  R m0 {0}             -- c5 ----> EXIT inner(i,j) {2}
```

The parent arrows end at the summary, not at child `ENTER` or `EXIT`. The
child graph is the summary's internal proof. If the inner loop continues, its
`EXIT` returns to the next `ENTER`; if it completes, parent edge `p2` applies.
The semaphore alternatives below show those dynamic cases separately.

The resulting channel assignment is:

```text
edges       channel         pending_count    initial state
c1          R1_READY        1                blocked
c2          WRITE_READY     1                blocked
c3          R0_READY        1                blocked
p1,c4,c5    FULL            2                blocked
p2          OUTER_EMPTY     1                released for owner {3}
```

`p1`, `c4`, and `c5` share one channel but supply different dynamic cycles.
For the first inner acquire, or the zero-trip post-loop acquire, `p1` is one
scaled release with `arrive_count=2`. For a later recurrence or a nonempty
post-loop acquire, `c4` and `c5` coexecute and contribute one arrival each.
Every acquire of `FULL` therefore has `pending_count=2`.

The sealed symbolic plan, using semantic channel names, is:

```text
scf.for outer {
  outer = a OUTER_EMPTY {3}
  W m0 [outer] {3}
  r FULL(2), outer {3}                         p1

  scf.for inner {
    t2  = a FULL(2) {2}
    r R1_READY, t2 {2}                         c1
    R m0 [t2] {2}
    r WRITE_READY, t2 {2}                      c2
    t1r = a R1_READY {1}
    R m0 [t1r] {1}
    t1w = a WRITE_READY {1}
    W m0 [t1w] {1}
    r R0_READY, t1w {1}                        c3
    r FULL, t1w {1}                            c4
    t0 = a R0_READY {0}
    R m0 [t0] {0}
    r FULL, t0 {0}                             c5
  }

  done = a FULL(2) {2}
  r OUTER_EMPTY, done {2}                      p2
}
```

The complete semaphore DAG through one inner body is shown in parts. First,
the outer handoff opens the child. The phase and control rows are both
predecessors of the same acquire:

```text
token path:   outer = a OUTER_EMPTY {3} --> W m0(i) [outer] {3}
phase path:   W m0(i) --> r FULL(2), outer {3} p1 -- FULL --> t2 = a FULL(2) {2}
control path: W m0(i) --> ENTER inner(i,0) {2} ------ walk --> t2 = a FULL(2) {2}
```

The owner-`{1}` write needs both the owner-`{1}` program-order path and the
owner-`{2}` token path:

```text
ordering path: t2 --> r R1_READY --> t1r = a R1_READY --> R m0 [t1r] {1}
token path:    t2 --> R m0 [t2] --> r WRITE_READY --> t1w = a WRITE_READY {1}

join at write: {R m0 [t1r] has executed, exact token t1w is available}
                                      --> W m0 [t1w] {1}
```

The write then produces the two arrivals that close the owner-`{2}` phase:

```text
c4 path: W m0 [t1w] --> r FULL, t1w {1}
c5 path: W m0 [t1w] --> r R0_READY --> t0 = a R0_READY
                         --> R m0 [t0] {0} --> r FULL, t0 {0}

phase: {c4 arrival, c5 arrival} --> FULL phase with 2 arrivals
```

After those two arrivals, exactly one of two acquire sites executes. If the
inner loop continues, the next body consumes the phase:

```text
phase path:   {r FULL, t1w(j), r FULL, t0(j)} --> completed FULL phase
              --> next = a FULL(2) in body j+1 {2}
control path: ENTER inner(i,j+1) -- walk --> next
```

If it finishes, the post-loop acquire consumes the same phase and implements
the parent handoff:

```text
phase path:   {r FULL, t1w(last), r FULL, t0(last)} --> completed FULL phase
              --> done = a FULL(2) {2}
control path: EXIT inner(i,last) {2} -- walk --> done

done --> r OUTER_EMPTY, done {2} p2 -- OUTER_EMPTY --> next = a OUTER_EMPTY {3}
next --> W m0(i+1) [next] {3}
```

For a zero-trip inner loop, the scaled `p1` release feeds `done` directly:

```text
phase path:   r FULL(2), outer {3} --> completed FULL phase --> done = a FULL(2) {2}
control path: EXIT inner(zero trip) {2} -- walk -------------> done
done --> r OUTER_EMPTY, done {2} p2
```

Thus every executed `FULL` acquire receives exactly two arrivals. Neither
loop has a semaphore-token `iter_arg` in the emitted IR. The outer
`OUTER_EMPTY` acquire is placed at the owner-`{3}` write, and its initially
released phase starts outer iteration zero.

### Example: the boundary owner is unchanged

The preceding example changes owner from outer `{3}` to inner boundary `{2}`.
The inline `@same_owner_nested` input covers the other case: the outer
writer and inner boundary are both `{3}`.

No current lit function covers this exact same-boundary shape. The plan below
was generated by running the shown input through the pass;
the section is explicit about that weaker, live-dump validation status.

```text
outer for {
  W m0 {3}

  inner for {
    R m0 {3}
    R m0 {2}
    W m0 {1}
    R m0 {0}
  }
}
```

The inner loop first touches P0 with a read owned by `{3}` and later writes
P0, so its parent summary is `P0:W:{3}`. At the parent level every boundary
owner is `{3}`:

```text
node                              generated synchronization edge
ENTER outer(i) {3}                none
W m0 {3}                          none
[inner summary P0:W:{3}]          none
EXIT outer(i) {3}                 none
```

```text
                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0(i) {3}
                                  | walk
                                  v
                   [inner summary P0:W:{3}]
                                  | walk
                                  v
                         EXIT outer(i) {3}
```

No parent synchronization edge is needed. The child still has six raw
cross-owner obligations:

```text
edge    source                 destination
c1      ENTER inner {3}        R m0 {2}
c2      R m0 {3}               W m0 {1}
c3      R m0 {2}               W m0 {1}
c4      W m0 {1}               R m0 {0}
c5      W m0 {1}               EXIT inner {3}
c6      R m0 {0}               EXIT inner {3}
```

```text
                         ENTER inner(i,j) {3}
                         +---------+---------+
                    walk |                   | c1
                         v                   v
                     R m0 {3}            R m0 {2}
                      c2 |                   | c3
                         +---------+---------+
                                   v
                              W m0 {1}
                         +---------+---------+
                      c4 |                   | c5
                         v                   |
                     R m0 {0}                |
                      c6 |                   |
                         +---------+---------+
                                   v
                          EXIT inner(i,j) {3}
```

As in the previous example, the first-access-owner wave-opening guard keeps
both coexecuting closing edges `c5` and `c6`, even though the path through
`c4` and `c6` implies the ordering of `c5`. The resulting channels are:

```text
edges       channel         pending_count    initial state
c1          R2_READY        1                blocked
c2,c3       WRITE_READY     2                blocked
c4          R0_READY        1                blocked
c5,c6       READY           2                released for owner {3}
```

The key difference is that the initially acquired `READY` token can be used
by the outer write and then passed directly into the first inner iteration.
There is no parent release/acquire pair between them. The exact plan is:

```text
initial = a READY(2) root {3}

scf.for outer iter_args(outer = initial) {
  W m0 [outer] {3}

  result = scf.for inner iter_args(itok = outer) {
    r R2_READY, itok {3}                       c1
    R m0 [itok] {3}
    r WRITE_READY, itok {3}                    c2
    t2 = a R2_READY {2}
    R m0 [t2] {2}
    r WRITE_READY, t2 {2}                      c3
    t1 = a WRITE_READY(2) {1}
    W m0 [t1] {1}
    r R0_READY, t1 {1}                         c4
    r READY, t1 {1}                            c5
    t0 = a R0_READY {0}
    R m0 [t0] {0}
    r READY, t0 {0}                            c6
    next = a READY(2) {3}
    yield next
  }

  yield result
}
```

The complete semaphore DAG is split into fixed-width parts. The entry token
passes through both same-owner boundaries without a semaphore handoff:

```text
initial = a READY(2) {3}
             initial |
                     v
        ENTER outer(i) {3}
             initial |
                     v
        W m0(i) [initial] {3}
             initial |
                     v
       ENTER inner(i,0) {3}
               itok = initial
```

The two readers independently supply the owner-`{1}` write:

```text
c3 path: itok --> r R2_READY --> t2 = a R2_READY --> R m0 [t2] {2}
         --> r WRITE_READY, t2 {2}
c2 path: itok --> R m0 [itok] {3} --> r WRITE_READY, itok {3}

phase: {c2 arrival, c3 arrival} --> WRITE_READY phase with 2 arrivals
       --> t1 = a WRITE_READY(2) {1}
```

That write produces the two arrivals for the next owner-`{3}` token:

```text
c5 path: t1 --> W m0 [t1] {1} --> r READY, t1 {1}
c6 path: t1 --> W m0 [t1] {1} --> r R0_READY --> t0 = a R0_READY
         --> R m0 [t0] {0} --> r READY, t0 {0}

phase: {c5 arrival, c6 arrival} --> READY phase with 2 arrivals
       --> next = a READY(2) {3}
```

Finally, the exact `next` token follows the dynamic control path:

```text
inner continues: next --> EXIT inner(i,j) --> ENTER inner(i,j+1)
inner finishes:  next --> EXIT inner(i,last) --> result = next --> EXIT outer(i)
outer continues: result --> EXIT outer(i) --> ENTER outer(i+1)
outer finishes:  result --> EXIT outer(last) --> final token
```

The two `READY` releases always execute together, so the acquire count is
two. If the inner loop has zero trips, it returns `outer` unchanged. If the
outer loop has zero trips, it returns `initial` unchanged. These are token
pass-throughs, not synchronization edges.

## Reducing raw edges

Within each group candidate, reduction runs before backing-copy selection and
before DirectBuilder chooses a loop placement. The same reduced edge set is
therefore the input to POU and FirstTouch. An Auto retry rebuilds the candidate
and runs reduction again on the unchanged input.

`reduceEdges` first records `rawSources` for every key:

```text
(destination node, source owner, destination owner)
```

It records that information before deleting anything. If several original
sources represented one surviving handoff, direct placement can later keep
the release no earlier than the latest represented source that uses the same
exact token producer.

Reduction then processes each structured chain independently:

```text
reduceStraightEdges    implication within one chain execution
reduceLoopCloses       implication across the next loop iteration
recurse                analyze child chains with their own positions
```

An edge marked `preserve` is never removed by either implication reducer.

<a id="1-implied-ordering-reduceedges"></a>

### Straight-chain implication

`reduceStraightEdges` assigns a position to each node and scans destination
buckets in deterministic order. For each destination it considers later
source positions first. `KnownOrder` records what each destination owner is
already known to be behind.

An access edge may be dropped only when all of these are true:

1. the edge is not `preserve`;
2. already-kept edges prove that the destination owner is behind the edge's
   source position;
3. the current token owner is the destination owner; and
4. the destination is an access, not `ENTER`, `EXIT`, or a region.

Condition 3 is important. Ordering alone is insufficient if deleting the
edge would leave the destination without a usable token.

When an edge is dropped, its destination is inserted into `reusable`. That is
the bridge between reduction and exact token placement:

```text
dropped edge
  -> no new acquire for this obligation
  -> destination reuses the exact token already established by kept edges
```

Among eligible access-source edges, only kept edges update `KnownOrder`.
Edges without owners and edges whose source is not an access are skipped by
this reducer; a deleted edge can never justify another deletion.

### Loop-close implication

A raw loop-close edge has an access source and an `EXIT` destination. It
means that the next iteration's boundary owner must not reuse a piece until
the current source has completed.

`reduceLoopCloses` asks whether kept body edges already impose that wait at a
simulated next-iteration first-touch frontier:

1. collect the non-close order established by one body execution;
2. for each close, find the point at which the destination owner has first
   touched all affected pieces in a simulated next execution;
3. replay the kept non-close edges at second-iteration positions;
4. require an available token for the destination owner; and
5. drop the close only if the source is already covered and the destination
   owner is not the chain's first access owner.

The final guard preserves the wave-opening recurrence. The first owner still
needs a concrete way to begin the next iteration.

### Async completion and release anchors

Two source-level details survive reduction:

- `EdgeRec::preserve` retains an exact-source edge carrying an asynchronous
  completion when an implied path would lose that completion; and
- `EdgeRec::rawSources` retains the original source set for a surviving
  `(destination, source owner, destination owner)` handoff.

During `DirectBuilder::collectSupply`, a survivor may advance to a later raw
source only when both source nodes resolve to the same exact token producer
and the later source follows the survivor in the same chain. Payloads remain
the union required by the represented operations. Direct placement therefore
seals the final release anchor before channel formation.

These mechanisms are source contracts. The two implication examples below
use lit inputs whose emitted protocols directly pin the deleted edges. The
mixed-completion example first illustrates completion-payload propagation.

### Example: one handoff preserves two completion kinds

`test/NVWS/insert_semas_same_owner_mixed_completion.mlir`
`@same_owner_mixed_completion` pins same-owner completion propagation. It does
not exercise `rawSources` advancement; that separate reducer contract is
described above. Two exact-alias SMEM members are filled by owner `{0}` before
owner `{1}` reads either one:

```text
A: W m0 {0}      nvws.descriptor_load       completion [tma_load]
B: W m1 {0}      ttg.local_store            completion [none]
C: R m1 {1}
D: R m0 {1}
```

The asynchronous completion starts at A. Because B is a synchronous write by
the same owner using the same exact token, the per-access rule carries A's
unresolved `[tma_load]` completion into B and adds B's `[none]` completion.
The resulting reduced edge is:

```text
q: B sync fill {0} -> C first consumer {1}    [none,tma_load]
```

`D` reuses C's exact owner-`{1}` token, so it needs no second acquire. The
source and edge DAG is:

```text
                         A: W async m0 {0}
                                  | same-owner completion propagation
                                  v
                  B: W sync m1 {0} [none,tma_load]
                                  | q
                                  v
                         C: R m1 {1}
                                  | walk
                                  v
                         D: R m0 {1}
```

The channel assignment is:

```text
role                  channel    pending_count    initial state
entry / loop close    EMPTY      1                released for owner {0}
q                     FULL       2                blocked
```

One release operation carries two completion payloads. Its contribution is
`release.count=1 * payloads=2`, matching `FULL.pending_count=2`:

```text
                         producer = a EMPTY {0}
                                      producer |
                                               v
                           A: W async m0 {0}
                                      producer |
                                               v
                           B: W sync  m1 {0}
                                      producer |
                                               v
          r FULL, producer [none,tma_load] {0}
                                      FULL(2) |
                                              v
                              consumer = a FULL(2) {1}
                                           consumer |
                                                    v
                                      C: R m1 {1}
                                           consumer |
                                                    v
                                      D: R m0 {1}
                                           consumer |
                                                    v
                            r EMPTY, consumer [none] {1}
                                       EMPTY |
                                             v
                               next = a EMPTY {0}
```

Lowering therefore initializes the FULL barrier with count two, issues one
TMA expectation and one explicit arrival, and only then lets owner `{1}`
proceed. This example pins async completion preservation across a same-owner
wave. The distinct `rawSources` latest-source rule is implemented by
`reduceEdges` and `collectSupply`; this example does not pin it with a lit
contract.

### Example: a preserved async edge stays direct

The inline `@doc_preserved_async_edge` input isolates `EdgeRec::preserve`.
It is not a current lit contract; the edge and semaphore plans below come
from running this shape through the pass:

```text
for {
  A: W async m0 {0}    descriptor_load, completion [tma_load]
  B: R m0 {1}
  C: W m0 {2}
}
```

The walk records three forward obligations. Both exact-source edges out of A
carry the descriptor load's completion and are marked `preserve`:

```text
edge    source                destination    completion
a1      A: W async m0 {0}     B: R m0 {1}    [tma_load], preserve
a2      A: W async m0 {0}     C: W m0 {2}    [tma_load], preserve
a3      B: R m0 {1}           C: W m0 {2}    [none]
```

```text
a1 path: A: W async m0 {0} -- [tma_load], preserve -> B: R m0 {1}
a2 path: A: W async m0 {0} -- [tma_load], preserve -> C: W m0 {2}
a3 path: B: R m0 {1}       -- [none] --------------> C: W m0 {2}
```

There is no alternate path into B, so `a1` is not an implication candidate.
The path `a1 -> a3` does imply ordering from A to C. Nevertheless, the
implication reducer does not delete `a2`: its exact source carries an
asynchronous completion and the edge is marked `preserve`. The pass therefore
keeps two arrivals into C's acquire:

```text
edge / role       channel        pending_count    initial state
entry / close     EMPTY          1                released for owner {0}
a1                READ_READY     1                blocked
a2,a3             WRITE_READY    2                blocked
```

The sealed protocol from the live dump is:

```text
t0 = a EMPTY {0}
A: W async m0 [t0] {0}
r READ_READY,  t0 [tma_load] {0}                 a1
r WRITE_READY, t0 [tma_load] {0}                 a2

t1 = a READ_READY {1}
B: R m0 [t1] {1}
r WRITE_READY, t1 [none] {1}                     a3

t2 = a WRITE_READY(2) {2}
C: W m0 [t2] {2}
r EMPTY, t2 [none] {2}
```

The semaphore DAG makes the two required arrivals explicit:

```text
reader path: t0 --> A --> r READ_READY [tma_load]
             --> t1 = a READ_READY --> B --> r WRITE_READY [none]

direct path: t0 --> A --> r WRITE_READY [tma_load]

join: {direct-path arrival, reader-path arrival}
      --> WRITE_READY phase with 2 arrivals --> t2 = a WRITE_READY(2)
      --> C --> r EMPTY --> next = a EMPTY {0}
```

The direct and reader-path releases use different exact tokens, so InsertSemas
represents them as two arrivals rather than one merged release. This inline
case documents the symbolic InsertSemas plan only; no current lit contract
covers the dual-async-release shape through LowerSemaphore.

### Example: repeated same-owner sources release after the later source

The inline `@doc_repeated_same_owner_sources` input isolates the other release
anchor rule. `whole` spans P0 and P1; `part` aliases only P0:

```text
for {
  W whole(P0,P1) {0}
  R whole(P0,P1) {1}
  R part(P0)     {1}
}
```

The first owner-`{1}` read is the latest use of P1. The later read replaces
owner `{1}`'s latest use only for P0. The two loop-closing obligations are
therefore:

```text
m2: R whole {1} -> EXIT {0}    P1
m3: R part  {1} -> EXIT {0}    P0
```

```text
W whole {0} -- f1 --> R whole {1} -- walk --> R part {1}
                           |                         |
                           | m2                      | m3
                           +-------------------------+
                                                     v
                                                EXIT {0}
```

Both sources use the exact token produced by the same owner-`{1}` acquire.
`reduceEdges` records both nodes in the keyed handoff's `rawSources`, and
`collectSupply` combines same-producer supplies while selecting the later
source in chain order. The result is one count-1 release after `R part`, not
two arrivals and not an early release after `R whole`:

```text
edge / role       channel    pending_count    initial state
f1                FULL       1                blocked
entry / m2,m3     EMPTY      1                released for owner {0}
```

```text
t0 = a EMPTY {0}
W whole [t0] {0}
r FULL, t0 [none] {0}                              f1

t1 = a FULL {1}
R whole [t1] {1}
R part  [t1] {1}
r EMPTY, t1 [none] {1}                             merged m2,m3

next = a EMPTY {0}
```

```text
t0 --> W whole --> r FULL --> t1 = a FULL --> R whole --> R part
                                                               |
                                                               v
                                                     r EMPTY, t1
                                                               |
                                                               v
                                                    next = a EMPTY
```

The current live dump has exactly two count-1 channels and places the EMPTY
release after the second read. This example pins the observable late-release
result; the source-level `rawSources` list remains an internal construction
detail rather than a separate IR operation.

### Example: a direct edge is implied

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members and this access order:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

For the shared piece, the relevant raw subgraph is:

```text
s1: W m0 {0} -> R m0 {1}
s2: W m0 {0} -> W m1 {2}
s3: R m0 {1} -> W m1 {2}
```

Raw edge DAG:

```text
direct candidate
  W m0 {0} -- s2 --> W m1 {2}

kept path
  W m0 {0} -- s1 --> R m0 {1} -- s3 --> W m1 {2}
```

Keeping `s1` and `s3` already makes owner `{2}` wait for the owner-`{0}`
write, and the acquire for `s3` supplies owner `{2}`'s token. `s2` is removed.
The reducer also records `W m1` in `reusable`; the surviving `s3` still creates
its acquire in this example, while the reuse fact protects cases where the
remaining path already provides the exact live token.

Reduced edge DAG, including the following handoff:

```text
W m0 {0}
   | s1
   v
R m0 {1}
   | s3
   v
W m1 {2}
   | s4
   v
R m1 {0}
```

The full loop also has one owner-`{1}` close that opens the next owner-`{0}`
write. The current emitted POU protocol has four count-1 channels:

```text
EMPTY   initially released recurrence to owner {0}
F01     {0} -> {1}
F12     {1} -> {2}
F20     {2} -> {0}
```

```text
t0 = a EMPTY {0}
     |
     v
W m0 [t0] {0}
     |
     v
r F01, t0 {0} --> t1 = a F01 {1} --> R m0 [t1] {1}

recurrence path
  R m0 [t1] {1} -> r EMPTY, t1 {1}

forward path
  R m0 [t1] {1} -> r F12, t1 {1} -> t2 = a F12 {2}
                                            |
                                            v
                                      W m1 [t2] {2}
                                            |
                                            v
                                      r F20, t2 {2}
                                            |
                                            v
                                      t0b = a F20 {0}
                                            |
                                            v
                                      R m1 [t0b] {0}
```

There is no extra `{0}->{2}` semaphore. The test's FileCheck pins this
minimal serialized ring and the absence of an async-token loop argument.

### Example: a loop-close edge is implied

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` uses two partly overlapping members:

```text
m0 = [0,128)
m1 = [64,192)

P0 = [0,64)      covered by m0
P1 = [64,128)    covered by m0 and m1
P2 = [128,192)   covered by m1
```

The loop is:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

After duplicate-piece and straight-chain reduction, the relevant edges are:

```text
l1    W m0 {0} -> R m0 {1}
l2    R m0 {1} -> W m1 {2}
l3    W m1 {2} -> R m1 {0}
c0    R m0 {1} -> EXIT {0}       P0 recurrence
c2    R m1 {0} -> EXIT {2}       P2 recurrence candidate
```

To test `c2`, the reducer follows the next iteration's kept path to owner
`{2}`'s first P2 write:

```text
raw stored close
  R m1(i) {0} -- c2 --> EXIT(i) {2}
```

The semantic target of that close is owner {2}'s first P2 use in the next
iteration, W m1(i+1).

```text
kept next-iteration path
  R m1(i) {0} -- walk --> EXIT(i) -> ENTER(i+1) -> W m0(i+1) {0}
                                                               |
                                                               | l1
                                                               v
                                                          R m0(i+1) {1}
                                                               |
                                                               | l2
                                                               v
                                                          W m1(i+1) {2}
```

The kept `l1 -> l2` path already supplies owner `{2}` at that write, so `c2`
is deleted. `c0` is retained because owner `{0}` is the first access owner and
opens the next wave.

The emitted POU protocol therefore has exactly four count-1 semaphores:

```text
edge    channel    entry state
c0      EMPTY      released for owner {0}
l1      F01        blocked
l2      F12        blocked
l3      F20        blocked
```

```text
a EMPTY {0} -> W m0 {0} -> r F01 {0}
                               |
                               v
                         a F01 {1} -> R m0 {1}

recurrence path from that exact read token
  R m0 [t1] {1} -> r EMPTY, t1 {1}

forward path from that exact read token
  R m0 [t1] {1} -> r F12, t1 {1} -> a F12 {2}
                                             |
                                             v
                                        W m1 {2}
                                             |
                                             v
                                        r F20 {2}
                                             |
                                             v
                                        a F20 {0}
                                             |
                                             v
                                        R m1 {0}
```

The next iteration's `a EMPTY` waits for `c0`. Owner `{0}`'s current
`R m1` is ordered before that next write by its own partition program order,
so no additional P2 close is needed.

## Direct protocol construction

After reduction and physical-copy selection, `DirectBuilder` consumes every
surviving edge exactly once. It inserts protocol nodes directly at their final
symbolic sites. Channel IDs do not exist yet.

The construction order is:

```text
place chains recursively
  place Access handoffs
  compose If branch results or alternative releases
  select and construct each For topology
  preserve exact token producers through ENTER and EXIT

require every reduced edge and deferred conditional supply to be consumed

form semaphore channels from the completed placement
```

There is no provisional semaphore graph. Once `formSemaphores` starts, node
placement is already final.

### Exact token identity

The critical fields are pointers, not owner guesses:

```text
Access.tokenSource       exact acquire or region producer used by the access
Release.tokenSource      exact producer of the released token
Release.sat              exact acquire demand satisfied by the release
Node.scheduleAnchor      exact access/completion used by scheduling
RegionFlow.exits         exact producer per returning path; nullptr = pass input
```

Within one chain, `DirectBuilder::Tokens` keeps one current live record per
owner, and that record contains an exact producer pointer. Distinct branch
producers are represented by `RegionFlow` exits. EMIT-IR later resolves only
the selected `tokenSource`; lexical recency and owner equality are not
fallback routing rules.

### Accesses and straight-line handoffs

For an access, `placeAccess` combines:

- unhandled reduced edges whose destination is this access;
- inherited edges deferred through a surrounding region;
- any alternative `Supply` routed to this continuation; and
- the exact live-token/reuse facts established by the walk and reduction.

If no valid token exists, it creates an acquire at the access and materializes
the source releases. If a token is reusable, it assigns that exact producer to
the access. Every source release is anchored after the source token's last
completion and names the new acquire in `sat`.

### Conditional paths

Both `if` branches start with the same incoming token set. They are placed
independently.

At the join, `DirectBuilder` first handles the trivial boundary cases directly:

```text
no uniform boundary owner       retain the incoming token set
every branch passes the input   retain that exact input through the region
no later export is needed       discard the boundary-owner token
```

The remaining nontrivial cases use one of two representations:

```text
later code needs a completed token and every path can provide one
  RegionFlow.exits records each branch's exact producer
  nullptr means that branch passes the incoming token

branches only provide arrivals to a later acquire
  each branch contains its own releases
  appendAlternative requires equal arrival counts
  the continuation consumes whichever branch executed
```

Releases in different branches are alternatives and do not add their counts.
Releases that execute together within one branch do add their counts.

### Loops

`placeFor` first indexes two sets of obligations by destination owner:

```text
entry edges     parent-chain edges ending at the loop summary
close edges     child-chain edges ending at the loop EXIT
```

A mixed-owner loop has no single boundary token. `DirectBuilder` uses a
per-owner first-demand form for it in both placement modes.

For a uniform-owner loop, `placeUniformOwnerFor` collects facts before
selecting one outcome:

```text
incoming exact token, if any
first concrete demand for the boundary owner
child channel already established at that demand, if any
entry and close supplies
open body output token, if any
whether a continuation must preserve the input boundary token
```

Important cases are:

- if the body returns an open exact token and has no close supply, publish
  that token through `RegionFlow`;
- in FirstTouch with a recurrent close, establish or reuse the boundary token
  before entry, carry it, and acquire the next token before `EXIT`;
- in POU with a recurrent close, attach the close and entry supplies to the
  concrete acquire at the first demand, or bridge through an established
  child channel;
- if no demand and no supply exists, no synchronization token crosses; and
- if a needed POU recurrence or boundary proof cannot be represented, record
  a structured `POURejection`.

POU requirements apply to loops that actually need a recurrence placement.
A loop with no such demand can succeed without manufacturing an acquire.

### POU, FirstTouch, and Auto

The three pass modes select construction policy, not output schema:

```text
pou
  request direct point-of-use placement
  a structured rejection is a pass error

first-touch
  request canonical carried placement for eligible uniform-owner recurrences
  reuse an already-open input token when possible

auto
  request POU
  on rejection, discard the whole function candidate
  rebuild with that exact (group, loop) forced to FirstTouch
  repeat for other rejected loops
  if rejection cannot be localized, make one all-FirstTouch attempt
```

`validatePOUPlan` checks fixed stage compatibility and boundary preservation
for recorded POU sites. `validateTokenConnectivity` rejects a used symbolic
region producer that has neither materialized flow nor an incoming producer.
Both run on the completed symbolic candidate.

### Example: the two loop placements

`test/NVWS/insert_semas_placement_modes.mlir` uses the running loop:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

The reduced edges and channels are identical in both modes:

```text
W {0} --FULL--> R {1} --EMPTY recurrence--> next W {0}
```

POU, also selected by Auto for this input:

```text
scf.for ... {
  tw = a EMPTY {0}
  W m0 [tw] {0}
  r FULL, tw {0}

  tr = a FULL {1}
  R m0 [tr] {1}
  r EMPTY, tr {1}
}
```

There is no async-token `iter_arg`; the recurrence acquire belongs to the
next concrete write.

FirstTouch:

```text
initial = a EMPTY root             effective owner {0}

scf.for ... iter_args(carry = initial) -> token {
  W m0 [carry] {0}
  r FULL, carry {0}

  tr = a FULL {1}
  R m0 [tr] {1}
  r EMPTY, tr {1}

  next = a EMPTY {0}
  scf.yield next
}
```

If an exact compatible token is already open before a FirstTouch loop, the
loop can reuse it instead of creating the root acquire shown here.

### Example: Auto discards and rebuilds

`test/NVWS/insert_semas_placement_fallback.mlir`
`@completed_pou_fallback` has nested loops and fixed stage assignments:

```text
outer for {
  inner for {
    touch0 m0 {0} stage 0
    touch1 m0 {0} stage 1
    touch2 m0 {1} stage 1
  }
}
```

Raw-edge construction and reduction are policy-independent and run before
DirectBuilder consults the selected loop policy. DirectBuilder can build a
symbolic POU candidate here, but completed validation finds that the fixed
stages require a carried recurrence. The driver performs this sequence:

```text
attempt 1
  build all groups with POU
  build the completed symbolic protocol
  validate recorded POU sites
  reject the identified inner-loop site

rollback
  discard every GroupDag from attempt 1
  restore schedule attributes
  record (group, inner loop) as FirstTouch

attempt 2
  rebuild ACCESS/SYNC state from the unchanged input IR
  construct the selected carried topology
  validate and emit
```

The test requires Auto output to be byte-identical to explicit FirstTouch and
requires strict POU to report:

```text
fixed loop.stage constraints require a carried recurrence
```

The final topology carries the exact token through both relevant loop
boundaries:

```text
initial = a ENTRY root
outer iter_args(outerToken = initial) {
  inner iter_args(carry = outerToken) {
    touch0 [carry]
    touch1 [carry]
    r TO1, carry {0}

    t1 = a TO1 {1}
    touch2 [t1]
    r NEXT, t1 {1}

    next = a NEXT {0}
    yield next
  }
  yield inner-result
}
```

The emitted result contains no nodes from the discarded POU candidate.

### Example: branch alternatives have count one

`test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token` has a two-copy TMEM accumulator. The
relevant access shape is:

```text
for {
  W acc {1}                 MMA

  if cond {
    R acc {0}
  } else {
    no acc access
  }
}
```

At the loop level the `if` is a read-only summary with boundary owner `{1}`.
The then child contains the actual owner-`{0}` read; the else child is empty:

```text
outer loop view

ENTER loop {1} --> W MMA {1} --> [if summary P0:R:{1}] --> EXIT loop {1}

child views

then:  ENTER if {1} -- e1 --> R acc {0} -- e2 --> EXIT if {1}
else:  ENTER if {1} -----------------------------> EXIT if
```

Expanded across the branch, the synchronization obligations are:

```text
                                 W MMA(i) {1}
                                       |
                       +---------------+---------------+
                       | cond=true                     | cond=false
                       v                               v
              e1: R acc(i) {0}                 no buffer access
                       |                               |
                       v                               |
       e2-then: next W reusing slot             e2-else: next W reusing slot
```

`e1` is a real owner handoff. The recurrence supply represented by `e2` has
two path-specific producers:

- the owner-`{0}` read when the then branch executes; or
- the owner-`{1}` MMA completion when the else branch executes.

Those producers are mutually exclusive. They are alternative supplies for
one phase, so `EMPTY.pending_count=1`, not two. The channel assignment is:

```text
obligation                 channel    pending_count    initial state
entry / e2 alternatives    EMPTY      1                released for owner {1}
e1                         FULL       1                blocked
```

The exact symbolic plan contains protocol operations inside both paths:

```text
scf.for {
  tw = a EMPTY {1}
  W MMA [tw] {1}

  scf.if cond {
    r FULL, tw [tc5mma] {1}
    tr = a FULL {0}
    R acc [tr] {0}
    r EMPTY, tr [none] {0}
  } else {
    r EMPTY, tw [tc5mma] {1}
  }
}
```

The complete semaphore DAG is best read as two mutually exclusive dynamic
paths. On the then path:

```text
                         tw = a EMPTY(i) {1}
                                  tw |
                                     v
                           W MMA(i) [tw] {1}
                                  tw |
                                     v
                  r FULL, tw [tc5mma] {1}
                                FULL |
                                     v
                         tr = a FULL {0}
                                  tr |
                                     v
                           R acc(i) [tr] {0}
                                  tr |
                                     v
                    r EMPTY, tr [none] {0}
                               EMPTY |
                                     v
                next = a EMPTY at the next reuse {1}
```

On the else path:

```text
                         tw = a EMPTY(i) {1}
                                  tw |
                                     v
                           W MMA(i) [tw] {1}
                                  tw |
                                     v
                 r EMPTY, tw [tc5mma] {1}
                               EMPTY |
                                     v
                next = a EMPTY at the next reuse {1}
```

Only one `EMPTY` release executes for a given branch instance. The completed
`if` has no added semaphore-token result: each branch closes the live token
into its alternative supply, and the later point-of-use acquire consumes the
executed path's phase. The test checks that the then-only `FULL` protocol
remains inside the then branch and that the else release retains the MMA
completion payload.

### Example: a POU plan can still carry a token

A strict POU candidate may still carry an exact output token across a loop
boundary when an access in the current iteration and an access in the next
iteration genuinely share that token.

`test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout` has a two-copy TMEM group:

```text
for {
  W first {0}
  R reader1 {1}
  R reader2 {2}
  W final {0}
}
```

The reduced edges are:

```text
e1    W first {0}   -> R reader1 {1}
e2    W first {0}   -> R reader2 {2}
e3    R reader1 {1} -> W final {0}
e4    R reader2 {2} -> W final {0}
```

```text
e1/e3 path
  W first {0} -- e1 --> R reader1 {1} -- e3 --> W final {0}

e2/e4 path
  W first {0} -- e2 --> R reader2 {2} -- e4 --> W final {0}
```

The reader completions share a count-2 `EMPTY` channel. The acquire must occur
before `W final(i)`, so that token already exists at the end of iteration `i`.
The next iteration's `W first(i+1)` can reuse the same exact token:

```text
initial = a EMPTY(2) root

scf.for ... iter_args(carry = initial) -> token {
  W first [carry] {0}
    |
    +--> r TO_R1, carry {0} --> t1 = a TO_R1 {1} --> R reader1 [t1] {1}
    |                                                         |
    |                                                         v
    |                                                  r EMPTY, t1 {1}
    |
    +--> r TO_R2, carry {0} --> t2 = a TO_R2 {2} --> R reader2 [t2] {2}
                                                              |
                                                              v
                                                       r EMPTY, t2 {2}

  next = a EMPTY(2) {0}
  W final [next] {0}
  scf.yield next
}
```

The two reader releases feed the acquire before the final write:

```text
r EMPTY, t1 {1} ----+
                    +--> EMPTY phase (2 arrivals) --> next = a EMPTY(2) {0}
r EMPTY, t2 {2} ----+
```

Strict POU accepts this policy-independent carried output; explicit
FirstTouch produces the same topology. The acquire was placed by the direct
edge demand at `W final`, and its exact output token remains useful across
`EXIT` and `ENTER`. The FileCheck requires the root count-2 seed, token
`iter_args`, the count-2 acquire before the final store, and the yielded
`next` token. A zero-trip loop returns the root-acquired `initial` token
unchanged.

### Example: nested POU without token carriers

`test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop` has one outer WS loop and one inner loop:

```text
outer for {
  inner for {
    W acc {1}       tc_gen5_mma
    R acc {0}       tmem_load
  }
}
```

The outer level sees one summary with the same boundary owner `{1}` as its own
`ENTER` and `EXIT`, so it generates no parent synchronization edge:

```text
                         ENTER outer(i) {1}
                                  | walk
                                  v
                   [inner summary P0:W:{1}]
                                  | walk
                                  v
                         EXIT outer(i) {1}
```

The inner level has two reduced edges:

```text
edge    source                 destination
c1      W acc {1}              R acc {0}
c2      R acc {0}              EXIT inner {1}
```

```text
                         ENTER inner(i,j) {1}
                                  | walk
                                  v
                            W acc(i,j) {1}
                                  | c1
                                  v
                            R acc(i,j) {0}
                                  | c2
                                  v
                         EXIT inner(i,j) {1}
                                  | next iteration
                                  v
                       ENTER inner(i,j+1) {1}
```

`c1` becomes `FULL`. Edge `c2`, together with the initial phase, becomes
`EMPTY`:

```text
edge          channel    pending_count    initial state
c1            FULL       1                blocked
entry,c2      EMPTY      1                released for owner {1}
```

DirectBuilder places both acquires at concrete inner-body demands:

```text
outer scf.for {
  inner scf.for {
    tw = a EMPTY {1}
    W acc [tw] {1}
    r FULL, tw [tc5mma] {1}

    tr = a FULL {0}
    R acc [tr] {0}
    r EMPTY, tr [none] {0}
  }
}
```

The complete nested semaphore DAG for consecutive inner iterations is:

```text
                         ENTER outer(i) {1}
                                  | walk
                                  v
                         ENTER inner(i,0) {1}
                                  | walk
                                  v
                         tw = a EMPTY {1}
                                  tw |
                                     v
                           W acc(i,0) [tw] {1}
                                  tw |
                                     v
                  r FULL, tw [tc5mma] {1}
                                FULL |
                                     v
                         tr = a FULL {0}
                                  tr |
                                     v
                           R acc(i,0) [tr] {0}
                                  tr |
                                     v
                   r EMPTY, tr [none] {0}
                               EMPTY |
                                     v
                       next = a EMPTY {1}
                                next |
                                     v
                         W acc(i,1) [next] {1}
                                  ...
```

After the final inner iteration there is no following acquire in that inner
invocation; the completed `EMPTY` phase remains available for the next inner
invocation that actually executes, including one in a later outer iteration.
Neither loop has a semaphore-token operand or result, and no acquire or
release is hoisted to the root. The original TMEM dependency tokens are
removed by EMIT-IR because the exact semaphore protocol replaces them. A
zero-trip inner or outer loop executes no acquire or release and preserves the
available `EMPTY` phase.

### Example: a parent continuation needs a bridge

The next function in the same file,
`@nested_ws_inner_loop_parent_continuation`, adds a read after the inner loop:

```text
outer for {
  inner for {
    W acc {1}
    R acc {0}
  }

  R acc {0}          outer continuation
}
```

The child and parent edge inventories are separate:

```text
edge    source                       destination
c1      W inner {1}                  R inner {0}
c2      R inner {0}                  EXIT inner {1}
p1      inner summary P0:W:{1}       R outer {0}
p2      R outer {0}                  EXIT outer {1}
```

```text
parent: [inner summary P0:W:{1}] -- p1 --> R outer {0} -- p2 --> EXIT outer {1}

child:  W inner {1} -- c1 --> R inner {0} -- c2 --> EXIT inner {1}
```

The pass forms four count-1 channels:

```text
LOCAL_EMPTY    next inner write or post-inner done bridge; initially released
LOCAL_FULL     inner write -> inner read
OUTER_FULL     completed inner loop -> outer read
OUTER_EMPTY    outer read -> owner-{1} tail; initially released
```

The channel assignment is:

```text
edge / role       channel        pending_count    initial state
c1                LOCAL_FULL     1                blocked
entry,c2          LOCAL_EMPTY    1                released for owner {1}
p1                OUTER_FULL     1                blocked
entry,p2          OUTER_EMPTY    1                released for owner {1}
```

`OUTER_EMPTY`'s initial phase is drained once before the outer loop so that
the tail acquire cannot consume a stale permit. The complete symbolic plan is:

```text
drain = a OUTER_EMPTY root {1}

scf.for outer {
  scf.for inner {
    tw = a LOCAL_EMPTY {1}
    W inner [tw] {1}
    r LOCAL_FULL, tw [tc5mma] {1}

    tr = a LOCAL_FULL {0}
    R inner [tr] {0}
    r LOCAL_EMPTY, tr [none] {0}
  }

  done = a LOCAL_EMPTY {1}
  r OUTER_FULL, done [none] {1}
  to = a OUTER_FULL {0}
  R outer [to] {0}
  r OUTER_EMPTY, to [none] {0}
  tail = a OUTER_EMPTY {1}
  r LOCAL_EMPTY, tail [none] {1}
}
```

The integrated semaphore DAG starts with the unconditional drain and the
ordinary inner-body recurrence:

```text
drain = a OUTER_EMPTY {1}          drains the initial outer phase

ENTER outer(i) --> ENTER inner(i,0)
                         | walk
                         v
              tw = a LOCAL_EMPTY {1}
                         tw |
                            v
                  W inner [tw] {1}
                         tw |
                            v
      r LOCAL_FULL, tw [tc5mma] {1}
                  LOCAL_FULL |
                             v
               tr = a LOCAL_FULL {0}
                          tr |
                             v
                   R inner [tr] {0}
                          tr |
                             v
        r LOCAL_EMPTY, tr [none] {0}
```

If the inner loop continues, that phase feeds its next point of use:

```text
phase path:   r LOCAL_EMPTY, tr {0} -- LOCAL_EMPTY --> next = a LOCAL_EMPTY {1}
control path: ENTER inner(i,j+1) -- walk -----------> next
next --> W inner(i,j+1) {1}
```

If it finishes, the post-loop acquire bridges into the parent and the tail
refeeds the next outer invocation:

```text
phase path:   r LOCAL_EMPTY, tr {0} --> completed LOCAL_EMPTY phase
              --> done = a LOCAL_EMPTY {1}
control path: EXIT inner(i,last) -- walk --> done

done --> r OUTER_FULL, done {1} p1 -- OUTER_FULL --> to = a OUTER_FULL {0}
to --> R outer(i) [to] {0} --> r OUTER_EMPTY, to {0} p2
r OUTER_EMPTY -- OUTER_EMPTY --> tail = a OUTER_EMPTY {1}
tail --> r LOCAL_EMPTY, tail {1} -- LOCAL_EMPTY --> first executed consumer
                                                     in outer(i+1):
                                                     inner-body acquire or
                                                     zero-trip bridge
```

A pre-loop acquire drains the initially released `OUTER_EMPTY` phase, ensuring
that a later `tail` waits for the outer reader rather than reusing the initial
permit. The lit test pins the pre-loop drain, the post-inner bridge, the outer
reader handoff, the tail refeed, and the absence of token `iter_args` on both
loops.

If the inner loop is zero-trip, `done` consumes the pending `LOCAL_EMPTY`
phase: the initial phase on the first outer invocation, or the phase supplied
by the previous tail on a later one. If the outer loop is zero-trip, only the
unconditional pre-loop
`OUTER_EMPTY` drain executes; no body release or bridge executes.

### Example: scheduled nested POU is placed directly

`test/NVWS/insert_semas_nested_carrier.mlir`
`@scheduled_relocated_acquire_boundaries` keeps its historical name, but the
implementation does not relocate an acquire. DirectBuilder
constructs the recurrence acquire at its final point of use.

The relevant input is:

```text
outer for {
  W acc {0}                         tmem_alloc/store

  inner for {
    W acc {1} stage=0               tc_gen5_mma
    R acc {0} stage=1               tmem_load
  }

  R acc {0} stage=0                 post-inner load
}
```

The reduced edge DAG is:

```text
parent edges
  p1: W outer {0}             -> inner summary P0:W:{1}
  p2: inner summary P0:W:{1}  -> R post {0}

child edges
  c1: W MMA {1}               -> R inner {0}
  c2: R inner {0}             -> EXIT inner {1}
```

```text
parent: W outer {0} -- p1 --> [inner summary P0:W:{1}] -- p2 --> R post {0}

child:  W MMA {1} -- c1 --> R inner {0} -- c2 --> EXIT inner {1}
```

All four edges use count-one channels:

```text
edge / role       channel        initial state
entry,p2          OUTER_EMPTY    released for owner {0}
p1,c2             LOCAL_EMPTY    blocked; seeded by p1
c1                LOCAL_FULL     blocked
```

The current FileCheck pins these final schedule locations:

```text
operation                              owner    cluster    stage
a LOCAL_EMPTY at inner MMA             {1}      3          0
r LOCAL_FULL after MMA [tc5mma]         {1}      3          0
a LOCAL_FULL at inner read              {0}      2          1
r LOCAL_EMPTY after inner read          {0}      2          1
post-inner a LOCAL_EMPTY                {1}      owner boundary
r OUTER_EMPTY from bridge               {1}      3          0
a OUTER_EMPTY at post-inner read        {0}      4          0
```

The integrated semaphore DAG is:

```text
initial = a OUTER_EMPTY root {0}
                    initial |
                            v
                W outer(i) [initial] {0}
                    initial |
                            v
             r LOCAL_EMPTY, initial {0} p1
                 LOCAL_EMPTY |
                             v
        tw = a LOCAL_EMPTY {1} [cluster 3, stage 0]
                         tw |
                            v
              W MMA(i,0) [tw] {1} [cluster 3, stage 0]
                         tw |
                            v
       r LOCAL_FULL, tw [tc5mma] {1}
                  LOCAL_FULL |
                             v
        tr = a LOCAL_FULL {0} [cluster 2, stage 1]
                         tr |
                            v
                R inner [tr] {0}
                         tr |
                            v
       r LOCAL_EMPTY, tr [none] {0} c2
                 LOCAL_EMPTY |
                             +-----------------------------+
                             |                             |
                    next inner iteration              inner finishes
                             |                             |
                             v                             v
               a LOCAL_EMPTY at MMA         done = a LOCAL_EMPTY {1}
                                                           done |
                                                                v
                                         r OUTER_EMPTY, done {1} p2
                                                    OUTER_EMPTY |
                                                                v
                                            out = a OUTER_EMPTY {0}
                                                            out |
                                                                v
                                                  R post [out] {0}
                                                            out |
                                                                v
                                                  yield out to outer loop
```

The post-inner acquire is owned by `{1}` because it closes the child boundary.
Its release inherits the owner-`{1}` schedule boundary. The following
owner-`{0}` acquire uses the post-read's own schedule. The symbolic placement
and exact producer are final before EMIT-IR.

### Example: branch completion keeps path-specific schedules

The same test file's `@branch_completion_requires_carrier` is another nested
case whose historical name describes the old concern, not current behavior.
The pass emits no semaphore-token result from the `if`.

```text
outer for {
  W acc {0}

  inner for {
    W mma0 acc {1} stage=0

    if cond {
      W mma1 acc {1} stage=1
      R branch acc {0} stage=1
    } else {
      no acc access
    }

    R final acc {0} stage=1
  }

  R post acc {0}
}
```

The parent and inner edge DAGs are:

```text
parent
  p1: W outer {0}          -> inner summary P0:W:{1}
  p2: inner summary {1}    -> R post {0}

inner
  c1: W mma1 {1}           -> R branch {0}       (then only)
  b2: R branch {0}         -> EXIT if {1}         (then only)
  c2: if summary P0:W:{1}  -> R final {0}
  c3: R final {0}          -> EXIT inner {1}
```

```text
outer level

W outer {0} -- p1 --> [inner summary P0:W:{1}] -- p2 --> R post {0}

inner level

W mma0 {1} -- walk --> [if summary P0:W:{1}] -- c2 --> R final {0}
                                                               |
                                                               | c3
                                                               v
                                                        EXIT inner {1}

if children

then:  ENTER if {1} --> W mma1 {1} -- c1 --> R branch {0} -- b2 --> EXIT if
else:  ENTER if -----------------------------------------------> EXIT if
```

The child edge `b2` exports the then path's final owner-`{0}` supply. When
DirectBuilder handles parent edge `c2`, it composes that supply with the else
path, which passes through the incoming `mma0` token. The resulting `c2`
release is path-specific: the then path supplies it from `R branch` at stage
1, while the else path supplies it from `mma0` completion at stage 0. They are
alternatives, not two arrivals:

```text
edge / role       channel         pending_count    initial state
entry,p2          OUTER_EMPTY     1                released for owner {0}
p1,c3             LOCAL_EMPTY     1                blocked; seeded by p1
c1                BRANCH_FULL     1                blocked
c2 alternatives   CONVERGE        1                blocked
```

The common prefix and then path are:

```text
W outer [outer] {0}
      outer |
            v
r LOCAL_EMPTY, outer {0} p1
      LOCAL_EMPTY |
                  v
tw = a LOCAL_EMPTY {1} [stage 0]
               tw |
                  v
        W mma0 [tw] {1}
                  |
                  v
        if cond = true
                  |
                  v
        W mma1 [tw] {1} [stage 1]
               tw |
                  v
r BRANCH_FULL, tw [tc5mma] {1}
       BRANCH_FULL |
                   v
    tb = a BRANCH_FULL {0}
                tb |
                   v
        R branch [tb] {0}
                tb |
                   v
      r CONVERGE, tb [none] {0} [stage 1]
           CONVERGE |
                    v
       tf = a CONVERGE {0}
                 tf |
                    v
          R final [tf] {0}
```

The else path reaches the same acquire without inventing a common token:

```text
tw = a LOCAL_EMPTY {1} [stage 0]
               tw |
                  v
        W mma0 [tw] {1}
               tw |
                  v
        if cond = false
               tw |
                  v
r CONVERGE, tw [tc5mma] {1} [stage 0]
           CONVERGE |
                    v
       tf = a CONVERGE {0}
                 tf |
                    v
          R final [tf] {0}
```

Both paths then execute the same nested-loop close and parent bridge:

```text
R final [tf] {0}
          tf |
             v
r LOCAL_EMPTY, tf [none] {0} c3
     LOCAL_EMPTY |
                 +---------------------------+
                 |                           |
        next inner iteration            inner finishes
                 |                           |
                 v                           v
     a LOCAL_EMPTY at mma0      done = a LOCAL_EMPTY {1}
                                              done |
                                                   v
                              r OUTER_EMPTY, done {1} p2
                                         OUTER_EMPTY |
                                                     v
                                  out = a OUTER_EMPTY {0}
                                                 out |
                                                     v
                                       R post [out] {0}
```

The then release keeps its stage-1 reader completion; the else release keeps
the stage-0 MMA completion. The final acquire is shared because either path
provides exactly one `CONVERGE` phase. The `if` itself gains no semaphore-token
operand or result.

## Forming semaphore channels

Protocol nodes are placed before semaphore IDs exist. DirectBuilder tracks
temporary acquire equivalence with a disjoint-set relation. `uniteChannels`
is used when entry, recurrence, child, or tail acquire sites represent the
same logical phase.

`formSemaphores` then processes each acquire class:

1. find the maximum required arrival count among its sites;
2. reconcile every site's releases to that uniform count;
3. create one `Sema` with a new `S<n>` name;
4. assign that ID to all acquire sites and their paired releases; and
5. derive each `RegionFlow` result channel from a concrete exit producer, or
   from the exact incoming `tokenSource` when paths pass that input through.

No protocol node moves during or after this step.

### Executed supplies and alternative supplies

`DirectBuilder::Supply` has a release list and an `arrivals` count.

For releases that execute together:

```text
appendExecuted(A, B)
  arrivals = arrivals(A) + arrivals(B)
```

For mutually exclusive branches:

```text
appendAlternative(then, else)
  require arrivals(then) == arrivals(else)
  channel count = that common value
```

One release contributes:

```text
release.count * number of completion payloads
```

Most releases have count one and one payload. A release with two distinct
completion payloads contributes two arrivals because both completions execute.

This distinction explains the two earlier count examples:

```text
release-count loop
  c4 and c5 both execute on one path  -> FULL count 2

conditional accumulator
  then and else releases are exclusive -> EMPTY count 1
```

### Uniform pending counts

Every acquire site in one channel must use the same `pending_count`. If one
entry path has a single scalable release while the recurrent path has several
coexecuting releases, the pass can raise that release's `arrive_count`.

A release is scalable only when all of its completion payloads support a
counted arrival. The current implementation permits `[none]` and WGMMA-class
payloads. Scaling must divide the required count exactly.

The release-count example has:

```text
entry site       p1 by owner {3}: one [none] release
recurrent site   c4 by {1} plus c5 by {0}: two releases

uniform FULL count = 2
p1 arrive_count   = 2
c4 arrive_count   = 1
c5 arrive_count   = 1
```

If a site's releases cannot legally reach the class count, channel formation
fails instead of emitting a barrier with path-dependent semantics.

### Entry state

An acquire with no executed predecessor can be marked `seeded`. During
channel formation its owner becomes `Sema::entryOwner`:

```text
entryOwner present    semaphore.create ... true
entryOwner absent     semaphore.create ... false
```

The entry phase has the same pending count as every later phase. A count-2
entry semaphore therefore begins with a complete count-2 permit; it does not
begin half-released.

Entry state supplies only the first dynamic acquire of that phase. It does
not create a permanent owner token. Exact token ownership begins at the
acquire node that consumes the phase.

## Backing copies

SYNC-DAG distinguishes physical buffer copies from semaphore-state copies:

```text
numCopies             physical SMEM or TMEM copies
numSemaphoreCopies    staged phases for every semaphore channel
```

They are usually equal, but an unstaged local buffer fed by a TMA load can
use one physical copy and several semaphore phases.

### Physical copies

`computeBackingCopies` runs after raw-edge reduction and before direct
placement. It first validates authored `buffer.copy` values across every
member in the group.

```text
start with numCopies = 1

if the reduced edge set is empty
  keep numCopies = 1, even when buffer.copy was authored

otherwise, if buffer.copy is authored
  numCopies = authored value

otherwise, if this is an eligible TMEM group and the meta partitioner is off
  numCopies = 2
```

An eligible group need not have an MMA user. When MMA users are present,
automatic TMEM double buffering is rejected when accumulator
read/modify/write, unsupported MMA structure, explicit policy, scaled-MMA
shape, or the TMEM capacity budget makes it unsafe.

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` pins an eligible
two-copy accumulator. Its root initialization token is adopted as the exact
owner-`{1}` boundary token; no artificial root-to-partition handoff is added.
Strict POU rejects this loop because that exact boundary token must remain
available across it, so Auto selects the carried topology. The owner-`{2}` MMA
is a later handoff, not the owner-`{1}` boundary demand.

### Example: a TMEM accumulator gets two copies

The complete access shape of that test is:

```text
W acc root                    initial tmem_store

for {
  R acc {1}
  W acc {1}
  W acc {2}                  tc_gen5_mma
}

R acc root                    final tmem_load
```

The loop boundary owner is `{1}`. The root initialization token is adopted by
that boundary, so there is no artificial root-to-`{1}` edge. The parent and
child edge DAGs are:

```text
parent

W acc root -- walk, same exact token --> [loop summary P0:W:{1}]
                                           |
                                           | p1
                                           v
                                      R acc root

child

ENTER(i) {1} --> R acc {1} --> W acc {1} -- e1 --> W MMA {2}
                                                    |
                                                    | e2
                                                    v
                                               EXIT(i) {1}
```

The channels are:

```text
edge / role    channel    pending_count    initial state
entry,e2       EMPTY      1                released at root
e1             TO_MMA     1                blocked
p1             AFTER      1                blocked
```

Auto selects the carried topology because the exact boundary token must
remain available through the loop:

```text
root = a EMPTY root
W acc [root] root

result = scf.for iter_args(carry = root) {
  R acc [carry] {1}
  W acc [carry] {1}
  r TO_MMA, carry {1}                         e1

  mma = a TO_MMA {2}
  W acc [mma] {2}
  r EMPTY, mma [tc5mma] {2}                   e2

  next = a EMPTY {1}
  yield next
}

r AFTER, result {1}                           p1
out = a AFTER root
R acc [out] root
```

```text
                              root = a EMPTY
                                      root |
                                           v
                               W acc [root] root
                                      same token
                                           |
                                           v
                                  ENTER(0) {1}
                                           |
                                           v
                               R acc [root] {1}
                                           |
                                           v
                               W acc [root] {1}
                                      root |
                                           v
                           r TO_MMA, root {1}
                                    TO_MMA |
                                           v
                              mma = a TO_MMA {2}
                                       mma |
                                           v
                              W MMA [mma] {2}
                                       mma |
                                           v
                    r EMPTY, mma [tc5mma] {2}
                                      EMPTY |
                                            v
                               next = a EMPTY {1}
```

If the loop continues, `next` crosses `EXIT` and becomes the next iter-arg:

```text
next --> EXIT(i) {1} --> ENTER(i+1) {1} --> repeat body
```

If it finishes, the same value becomes the loop result and supplies `p1`:

```text
next --> EXIT(last) {1} --> result
                                result |
                                       v
                      r AFTER, result {1}
                               AFTER |
                                     v
                        out = a AFTER root
                                 out |
                                     v
                        R acc [out] root
```

For a zero-trip loop, `result` is the original root token and supplies `p1`.
The group has synchronization, no explicit `buffer.copy`, one eligible MMA,
and enough TMEM capacity, so the final storage contract is:

```text
input accumulator     memdesc<128x128xf32>
generated backing     memdesc<2x128x128xf32>
physical copies       2
semaphore copies      2
```

### Semaphore copies

After placement, `computeSemaphoreCopies` inspects final release payloads:

```text
SMEM/non-TMEM group
and no authored buffer.copy
and at least one release contains [tma_load]
  numSemaphoreCopies = max(1, InsertSemas num-stages)

otherwise
  numSemaphoreCopies = numCopies
```

For example, `test/NVWS/insert_semas.mlir` `@local_release_after_mma` has one
physical SMEM copy. Its producer is `nvws.descriptor_load`, so the FULL release
retains `[tma_load]`; lowering may stage its semaphore state even though the
backing remains single-copy.

```text
physical backing
  one SMEM slot

semaphore state
  phase 0, phase 1, ... according to InsertSemas num-stages
```

The InsertSemas `num-stages` option is the intended LowerSemaphore staging
depth and must be kept consistent with the later lowering option. The
distinction is recorded before EMIT-IR; rendering does not infer staging from
operation names.

### Example: a TMA load stages only semaphore state

The `buffer.id=102` group in `@local_release_after_mma` has this input:

```text
for {
  W m0 {0} stage=0       nvws.descriptor_load
  R m0 {1} stage=1       tc_gen5_mma operand
}
```

Its reduced edge DAG is the ordinary two-owner recurrence:

```text
edge    source                  destination
e1      W descriptor_load {0}   R MMA operand {1}
e2      R MMA operand {1}       EXIT {0}
```

```text
                         ENTER(i) {0}
                              | walk
                              v
                    W descriptor_load {0}
                              | e1
                              v
                       R MMA operand {1}
                              | e2
                              v
                          EXIT(i) {0}
```

`e1` becomes `FULL`; `e2` and the initial phase become `EMPTY`. Both channels
have `pending_count=1`:

```text
                         empty = a EMPTY {0}
                                  empty |
                                        v
                  W descriptor_load [empty] {0}
                                  empty |
                                        v
              r FULL, empty [tma_load] {0} e1
                                   FULL |
                                        v
                           full = a FULL {1}
                                   full |
                                        v
                      R MMA operand [full] {1}
                                   full |
                                        v
              r EMPTY, full [tc5mma] {1} e2
                                  EMPTY |
                                        v
                    next = a EMPTY(i+1) {0}
                                   next |
                                        v
                 W descriptor_load(i+1) {0}
```

The two completion payloads are different and both are required: `[tma_load]`
delays the consumer until the descriptor load has filled SMEM, while
`[tc5mma]` delays reuse until the MMA has finished reading it.

SMEM does not receive automatic TMEM double buffering, and the input has no
authored `buffer.copy`. Therefore:

```text
physical buffer copies     1
semaphore copies           max(1, InsertSemas num-stages)
```

The extra semaphore phases are a lowering contract, not extra SMEM storage.
There is no buffer use after the loop; the last `EMPTY` release has no later
consumer.

## Finalizing the pipeline schedule

Schedule finalization runs only after every group in one function has a
complete, verified symbolic protocol. It consumes final release/acquire pairs;
it does not alter placement policy.

The order is:

```text
assignCircularStageOffsets
assignAliasedHandoffStageOffsets
addSyncScheduleEdges
solveOwnerScheduleConstraints
legalizeLoopSchedule
assignSyncScheduleChain
```

Groups that share circular physical storage are considered together. Ordinary
groups form their own physical set.

### Release/acquire constraints

Every release has:

```text
tokenSource       exact token being released
scheduleAnchor    exact access or completion that produces the release
sat               exact acquire that consumes the phase
```

For a release/acquire pair directly in a scheduled loop body,
`addSyncScheduleEdges` finds the producer and consumer operations selected by
those anchors. It creates an `OwnerScheduleConstraint`:

```text
producer owner and operation
consumer owner and operation
producer loop.stage
consumer loop.stage
loop-carried distance
```

Its required owner delay is:

```text
producerStage - consumerStage - distance
```

`solveOwnerScheduleConstraints` uses Bellman-Ford-style relaxation. A positive
cycle means authored stage assignments demand an impossible semaphore cycle,
so the pass reports the concrete handoffs in that cycle. A zero-delay
constraint becomes a loop-cluster ordering edge. A nonzero-delay constraint
does so only when it is tight and a tight return path closes the owner cycle.
Existing SSA ordering suppresses a redundant edge.

`legalizeLoopSchedule` keeps every `loop.stage` fixed, incorporates same-stage
SSA constraints, adjusts and rebases `loop.cluster` values to satisfy the
required edges, and diagnoses a cyclic cluster constraint.

### Recurrence distance

Original input lexical order is not the recurrence proof. Direct placement
records a distance when it knows one; otherwise scheduling examines the final
symbolic release/acquire order:

```text
Acquire.recurrenceDistance = 1    next logical iteration
unset                             same-iteration or derived from slot replay
```

If a final release precedes its acquire in the same chain, the unset distance
is same-iteration. If it does not, schedule finalization replays physical slot
events. For multibuffered storage,
`computeLoopCarriedDistance` finds the smallest positive iteration distance at
which the consumer and producer refer to the same semaphore copy. A one-copy
group has distance one by definition.

When a relation requires inferred loop-carried distance, incomplete replay or
the absence of a matching slot orbit is an error.

### Circular and alias offsets

Circular local groups with one physical `buffer.id` retain separate logical
SYNC-DAGs. `assignCircularStageOffsets` orders all of their accesses in
function walk order, replays which write advances the shared slot, and assigns:

```text
Access.bufferStageOffset
Acquire.stageOffset
Release.stageOffset
```

It validates that authored `buffer.start` values agree with the replayed
producer order and that no consumer appears before a producer.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` pins the circular two-copy case. Separate K and
V logical groups share physical storage, but each access and protocol node
receives the offset of its own replayed slot.

For non-circular multibuffered aliases,
`assignAliasedHandoffStageOffsets` requires producer and consumer anchors in
the same direct loop body. It derives the release offset from the exact slot
pair and recurrence distance, then normalizes the other side of the handoff.

`test/NVWS/insert_semas_fused_alias_handoff.mlir` `@fused_alias_depth_two`
pins this representation. Its semaphores contain both group member backings,
and each `nvws.semaphore.buffer` returns both member views. The selected copy
offset is attached to acquire/release protocol nodes; it is not invented on a
buffer-view operation.

### Example: circular K and V select different copies

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` has two logical groups sharing one two-copy
physical circular allocation:

```text
K = local_alloc {buffer.id=301, buffer.copy=2,
                 buffer.circular, buffer.start=0}
V = local_alloc {buffer.id=301, buffer.copy=2,
                 buffer.circular, buffer.start=1}

for {
  W K {1}
  W V {1}
  R K {2}
  R V {2}
}
```

Each logical group has the same reduced edge shape:

```text
K edges                                  V edges

k1: W K {1} -> R K {2}                  v1: W V {1} -> R V {2}
k2: R K {2} -> EXIT {1}                 v2: R V {2} -> EXIT {1}
```

```text
      ENTER(i) {1}                             ENTER(i) {1}
           | walk                                   | walk
           v                                        v
       W K(i) {1}                                W V(i) {1}
           | k1                                     | v1
           v                                        v
       R K(i) {2}                                R V(i) {2}
           | k2                                     | v2
           v                                        v
       EXIT(i) {1}                               EXIT(i) {1}
```

Each GroupDag has count-one logical `FULL` and initially released `EMPTY`
channels. Because K and V are circular members of the same physical
`buffer.id`, EMIT-IR folds their logical sites into one physical `FULL` and
one physical `EMPTY` semaphore. DirectBuilder places recurrence acquires at
both writes; the loop gains no semaphore-token carrier for either group:

```text
K protocol                               V protocol

kt = a EMPTY {1}                         vt = a EMPTY {1}
W K [kt] {1}                             W V [vt] {1}
r FULL, kt {1}                           r FULL, vt {1}
kr = a FULL {2}                          vr = a FULL {2}
R K [kr] {2}                             R V [vr] {2}
r EMPTY, kr {2}                          r EMPTY, vr {2}
```

The physical access order advances one shared write number:

```text
event       current write number    required write number    offset
W K         -1 -> 0                 K producer = 0           0
W V          0 -> 1                 V producer = 1           0
R K          1                      K producer = 0          -1
R V          1                      V producer = 1           0
```

Consequently the logical GroupDag nodes use different endpoint offsets even
though they materialize operations on the same two physical semaphores:

```text
K logical sites on shared semaphores      V logical sites on shared semaphores

kt = a EMPTY offset=0 {1}                 vt = a EMPTY offset=0 {1}
W K [kt, buffer offset=0] {1}             W V [vt, buffer offset=0] {1}
r FULL offset=0, kt {1}                   r FULL offset=0, vt {1}
kr = a FULL offset=-1 {2}                 vr = a FULL offset=0 {2}
R K [kr, buffer offset=-1] {2}            R V [vr, buffer offset=0] {2}
r EMPTY offset=-1, kr {2}                 r EMPTY offset=0, vr {2}
```

The offsets resolve against the shared write cursor as follows:

```text
logical event       endpoint offset    resolved physical copy
W K / r FULL        0                  copy 0
W V / r FULL        0                  copy 1
a FULL / R K       -1                  copy 0
a FULL / R V        0                  copy 1
R K / r EMPTY      -1                  copy 0
R V / r EMPTY       0                  copy 1
```

The `-1` is relative to the current write number after V advanced it. It is
not a negative physical index; AssignStagePhase wraps it modulo two. The
current GroupDag dumps pin K's acquire and closing release at
`stage-offset=-1` and V's at zero; the lit test pins the folded pair of
physical semaphore creations.

### Example: a non-circular alias advances the copy

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@fused_alias_depth_two` uses two names for one two-copy SMEM backing:

```text
m0 = local_alloc {buffer.id=500, buffer.copy=2}
m1 = local_alloc {buffer.id=500, buffer.copy=2}

for {
  W m0 {4}
  R m0 {2}
  W m1 {4}
  R m1 {2}
}
```

The aliases overlap the same piece, so the reduced edge DAG is one chain:

```text
edge    source        destination
e1      W m0 {4}      R m0 {2}
e2      R m0 {2}      W m1 {4}
e3      W m1 {4}      R m1 {2}
e4      R m1 {2}      EXIT {4}
```

```text
                         ENTER(i) {4}
                              | walk
                              v
                          W m0(i) {4}
                              | e1
                              v
                          R m0(i) {2}
                              | e2
                              v
                          W m1(i) {4}
                              | e3
                              v
                          R m1(i) {2}
                              | e4
                              v
                          EXIT(i) {4}
```

Every channel has `pending_count=1`; `ENTRY` starts released:

```text
edge    channel
e1      M0_FULL
e2      M1_READY
e3      M1_FULL
e4      ENTRY
```

The POU semaphore DAG is:

```text
                     t0 = a ENTRY(i) {4}
                               t0 |
                                  v
                         W m0(i) [t0] {4}
                               t0 |
                                  v
                       r M0_FULL, t0 {4} e1
                            M0_FULL |
                                    v
                         t1 = a M0_FULL {2}
                                  t1 |
                                     v
                            R m0 [t1] {2}
                                  t1 |
                                     v
                       r M1_READY, t1 {2} e2
                           M1_READY |
                                    v
                        t2 = a M1_READY {4}
                                  t2 |
                                     v
                            W m1 [t2] {4}
                                  t2 |
                                     v
                        r M1_FULL, t2 {4} e3
                            M1_FULL |
                                    v
                         t3 = a M1_FULL {2}
                                  t3 |
                                     v
                            R m1 [t3] {2}
                                  t3 |
                                     v
                           r ENTRY, t3 {2} e4
                              ENTRY |
                                    v
                    next = a ENTRY(i+1) {4}
```

The first write/read pair uses copy `s`; the second uses `(s+1) mod 2`:

```text
handoff                    release offset    acquire offset
W m0 -> R m0               0                 0
R m0 -> W m1              +1                 0
W m1 -> R m1               0                 0
R m1 -> W m0(i+1)         +1                 0
```

Thus the current symbolic dump places `stage-offset=1` on the `M1_READY` and
`ENTRY` releases. Without those offsets, each acquire would wait on a
different physical semaphore phase from the release that is meant to satisfy
it. The offset is attached to the sealed protocol node before emission.

### Example: one physical slot

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` has one SMEM copy and this scheduled loop:

```text
W slot {3}          loop.stage 0
R first {1}         loop.stage 0
R last {1}          loop.stage 1
```

The memory cycle is:

```text
W(i) {3} --FULL--> R first(i) {1}
                         |
                         | same exact reader token
                         v
                    R last(i) {1}
                         |
                         | EMPTY, distance 1
                         v
                    W(i+1) {3}
```

The final read and next store can occupy the same pipelined wave even though
they belong to adjacent logical iterations. In the completed POU chain the
EMPTY acquire is before the store and its satisfying release is after the
final read. That final-chain order identifies a loop-carried relation; because
the group has one semaphore copy, `computeLoopCarriedDistance` returns
distance one by definition:

```text
r EMPTY after R last(i) {1}
     | completed release/sat pair; one-copy rule => distance 1
     v
a EMPTY before W(i+1) {3}
```

The channel assignment is:

```text
edge / role       channel    pending_count    initial state
W -> R first      FULL       1                blocked
entry / close     EMPTY      1                released for owner {3}
```

The complete generated semaphore DAG is:

```text
empty = a EMPTY {3}                 [cluster 3, stage 0]
             empty |
                   v
       W slot(i) [empty] {3}        [cluster 3, stage 0]
             empty |
                   v
      r FULL, empty [none] {3}      [cluster 3, stage 0]
              FULL |
                   v
        full = a FULL {1}           [cluster 3, stage 0]
              full |
                   v
      R first(i) [full] {1}         [cluster 3, stage 0]
              full |
                   v
      R last(i) [full] {1}          [cluster 2, stage 1]
              full |
                   v
      r EMPTY, full [none] {1}      [cluster 2, stage 1]
             EMPTY |
                   v
next = a EMPTY {3}                  [cluster 3, stage 0]
              next |
                   v
    W slot(i+1) [next] {3}          [cluster 3, stage 0]
```

There is no semaphore-token loop carrier: each iteration acquires `EMPTY` at
the store's point of use. `EMPTY`'s entry state supplies iteration zero; each
owner-`{1}` close supplies the next logical iteration.

The test requires cluster legalization to order the next store and its first
consumer after the final read. Its FileCheck observes the adjusted clusters
and the lowered wait/arrive order.

After constraints and offsets are solved, `assignSyncScheduleChain` annotates
each acquire and release with the schedule at its exact owner boundary.
Releases inherit the latest completion for their owner. Acquires use their
selected demand anchor, with special handling for a recurrence at `EXIT`.

## Validation boundary

The pass checks the symbolic protocol before EMIT-IR changes SSA or memory
objects.

Candidate-specific checks:

```text
validatePOUPlan
  reject a recorded POU site that loses a required boundary token
  reject one-copy fixed-stage recurrence mismatch

validateTokenConnectivity
  reject a used symbolic region producer that has neither RegionFlow
  nor a materializable incoming producer
```

`verifySyncDag` then checks the common final schema, regardless of placement
mode:

- each access has an exact, owner-compatible `tokenSource`;
- each release has an exact token, nonempty completion payload, schedule
  anchor, positive contribution, and one acquire in `sat`;
- release and acquire use the same semaphore channel;
- each acquire has a positive count equal to its `Sema` count and has either
  real releases or entry state;
- a repeated entry acquire inside a loop has a per-loop release;
- one channel is not acquired by two fixed partition owners;
- every recurrence distance is positive;
- each `RegionFlow` has one valid exit record per child path, a materialized
  input when required, and a channel; and
- exported path producers belong to the corresponding child and produce the
  required owner token.

DirectBuilder also has a completion check before channel formation: every
reduced `EdgeRec` must be marked handled, and every deferred conditional
`Supply` must be consumed. A missed obligation cannot disappear simply
because no semaphore was created for it.

EMIT-IR performs a second class of checks after materialization, including
exact SSA token lookup, region-slot locality, buffer-view lifetime, and
use-after-release. Those checks are described in [EMIT-IR](emit-ir.md).

## Build order and code map

The function driver in `InsertSemas.cpp` builds one candidate as follows:

```text
collectGroups
  -> buildAccessDag for every group
  -> buildSyncDag for every group
       ChainWalker
       reduceEdges
       computeBackingCopies
       DirectBuilder::run
       computeRequiredParts
       computeSemaphoreCopies
       validatePOUPlan
       validateTokenConnectivity
       verifySyncDag
  -> finalizeSyncSchedule across all groups
  -> emitIR
```

Auto retry owns the candidate transaction. It never asks an individual
GroupDag to undo a constructed protocol.

Source map:

| Responsibility | Implementation |
| --- | --- |
| Placement-mode parsing and retry | `InsertSemas.cpp` |
| Shared `Node`, `RegionFlow`, `Sema`, `GroupDag` model | `InsertSemas.h` |
| Groups, pieces, owners, accesses, region summaries | `InsertSemasAccessDag.cpp` |
| Raw edges | `ChainWalker`, `applyTouch` in `InsertSemasSyncDag.cpp` |
| Edge reduction | `reduceStraightEdges`, `reduceLoopCloses`, `reduceEdges` |
| Direct placement | `DirectBuilder` |
| Channel/count formation | `DirectBuilder::formSemaphores` |
| Candidate checks | `validatePOUPlan`, `validateTokenConnectivity`, `verifySyncDag` |
| Copies and schedule | `computeBackingCopies`, `computeSemaphoreCopies`, `finalizeSyncSchedule` |
| Symbolic dump and materialization | `InsertSemasEmitIR.cpp` |

The design can be summarized in one sentence:

> Derive and reduce obligations first; then construct one complete exact-token
> protocol directly; only after placement is final, name channels, validate,
> schedule, and emit.
