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
  - [Composition: nested regions in the walk](#composition-nested-regions-in-the-walk)
  - [Example: counted recurrence at two region levels](#example-counted-recurrence-at-two-region-levels)
- [Reducing raw edges](#reducing-raw-edges)
  - [Straight-chain implication](#straight-chain-implication)
  - [Loop-close implication](#loop-close-implication)
  - [Async completion and release anchors](#async-completion-and-release-anchors)
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
- [Forming semaphore channels](#forming-semaphore-channels)
  - [Executed supplies and alternative supplies](#executed-supplies-and-alternative-supplies)
  - [Uniform pending counts](#uniform-pending-counts)
  - [Entry state](#entry-state)
- [Backing copies](#backing-copies)
  - [Physical copies](#physical-copies)
  - [Semaphore copies](#semaphore-copies)
- [Finalizing the pipeline schedule](#finalizing-the-pipeline-schedule)
  - [Release/acquire constraints](#releaseacquire-constraints)
  - [Recurrence distance](#recurrence-distance)
  - [Circular and alias offsets](#circular-and-alias-offsets)
  - [Example: one physical slot](#example-one-physical-slot)
- [Validation boundary](#validation-boundary)
- [Build order and code map](#build-order-and-code-map)

## What SYNC-DAG builds

ACCESS-DAG has already grouped memory, split overlapping members into pieces,
assigned an owner to every access, and built structured chains with `ENTER`
and `EXIT` nodes. SYNC-DAG turns those facts into one complete symbolic
synchronization plan.

The current solid-01 implementation runs these steps in this order:

```text
ACCESS-DAG chains
        |
        v
ChainWalker
  derive raw memory and token-supply edges
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

Edge reduction is therefore part of the solid-01 POU design. It is common
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

Every worked protocol below names the solid-01 lit function that produces it.
Diagrams omit types and unrelated operations, but the edges, acquire sites,
release sites, counts, entry state, and token carriers are checked against the
current source, pass dump, and FileCheck expectations.

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
edge DAG                         semaphore DAG

W m0 {0}                        W m0 [tw] {0}
   | e1                            |
   v                               v
R m0 {1}                        r FULL, tw {0}
                                  | FULL
                                  v
                                tr = a FULL {1}
                                  |
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
asynchronous completion payloads, and whether its exact async source must be
preserved.

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

Both loop-close edges execute on every iteration and therefore share a
count-2 `EMPTY` acquire. The two outgoing read edges remain separate count-1
channels:

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
`@release_multiplicity_unified_fanin_regain` is a current solid-01 count
contract. Its relevant input is:

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

There is one piece P0. The inner loop first touches P0 as owner `{2}` and
writes it, so its parent summary is `P0:W:{2}`.

The parent edge inventory is:

```text
edge    source                  destination
p1      W m0 {3}               inner summary {2}
p2      inner summary {2}      EXIT outer {3}
```

```text
W m0 {3}
   | p1
   v
[inner summary P0:W:{2}]
   | p2
   v
EXIT outer {3}
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
incoming paths to the owner-{1} write

ENTER inner {2} -- c1 --> R m0 {1} -- walk ---+
       |                                      |
       | walk                                 v
       +------------> R m0 {2} -- c2 ----> W m0 {1}

paths closing the corrected version

  c4 path: W m0 {1} ------------------ c4 --> EXIT inner {2}
  c5 path: W m0 {1} -- c3 --> R m0 {0} -- c5 --> EXIT inner {2}
```

Both `c4` and `c5` survive solid-01 reduction. Owner `{1}`'s corrected write
and owner `{0}`'s later read both execute on the same path and contribute to
the next owner-`{2}` phase.
The resulting channels are:

```text
edges       channel         count    entry state
c1          R1_READY        1        blocked
c2          WRITE_READY     1        blocked
c3          R0_READY        1        blocked
p1,c4,c5    FULL            2        blocked
p2          EMPTY           1        released for owner {3}
```

Inside one nonempty inner iteration, the exact symbolic chain order is:

```text
t2  = a FULL(2) {2}
r R1_READY, t2 {2}                 c1
R m0 [t2] {2}
r WRITE_READY, t2 {2}              c2
t1r = a R1_READY {1}
R m0 [t1r] {1}
t1w = a WRITE_READY {1}
W m0 [t1w] {1}
r R0_READY, t1w {1}                c3
r FULL, t1w {1}                    c4
t0  = a R0_READY {0}
R m0 [t0] {0}
r FULL, t0 {0}                     c5
```

The two prerequisites of the owner-`{1}` write are distinct. Its prior
owner-`{1}` read reaches the write by program order; the owner-`{2}` read
opens the exact token used by the write:

```text
program-order prerequisite
  t2 -> r R1_READY -> t1r = a R1_READY -> R m0 [t1r]
                                              |
                                              | owner-{1} walk order
                                              v
                                         W m0 [t1w] {1}

token prerequisite
  t2 -> R m0 [t2] -> r WRITE_READY -> t1w = a WRITE_READY
                                              |
                                              | exact tokenSource
                                              v
                                         W m0 [t1w] {1}
```

After the write, one path supplies owner `{0}` and both final holders
contribute to FULL:

```text
                        +--> r FULL, t1w {1}                         c4
W m0 [t1w] {1} ---------+
                        +--> r R0_READY --> t0 = a R0_READY
                                                  |
                                                  v
                                            R m0 [t0] {0}
                                                  |
                                                  v
                                            r FULL, t0 {0}           c5
```

The FULL channel has two static acquire sites: one at the first owner-`{2}`
use inside the inner body and one immediately after the inner loop. They are
alternative consumers of a completed dynamic phase, not simultaneous
fan-out.

First inner iteration of a nonempty invocation:

```text
W m0 outer [to] {3}
     |
     v
r FULL(2), to {3}                p1: one release, arrive_count=2
     | FULL phase
     v
a FULL(2) at inner body entry {2}
```

Re-entry to another inner iteration:

```text
r FULL, t1w(j) {1} --+
                     +--> FULL phase --> a FULL(2) in body j+1 {2}
r FULL, t0(j)  {0} --+
```

Exit from a nonempty inner loop:

```text
r FULL, t1w(last) {1} --+
                        +--> FULL phase --> done = a FULL(2) after loop {2}
r FULL, t0(last)  {0} --+                           |
                                                    +--> r EMPTY, done {2}   p2
```

Zero-trip inner loop:

```text
r FULL(2), to {3} --> FULL phase --> done = a FULL(2) after loop {2}
                                             |
                                             +--> r EMPTY, done {2}          p2
```

Thus every execution of either acquire site receives exactly two arrivals.
Neither loop has a semaphore-token `iter_arg` in the current emitted IR. The
outer `EMPTY` acquire is at the next owner-`{3}` write; `EMPTY` starts released
for outer iteration zero.

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
the solid-01 bridge between reduction and exact token placement:

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

- `EdgeRec::preserve` retains an edge from the exact asynchronous operation
  when an implied synchronous path would not carry the same completion; and
- `EdgeRec::rawSources` retains the original source set for a surviving
  `(destination, source owner, destination owner)` handoff.

During `DirectBuilder::collectSupply`, a survivor may advance to a later raw
source only when both source nodes resolve to the same exact token producer
and the later source follows the survivor in the same chain. Payloads remain
the union required by the represented operations. Direct placement therefore
seals the final release anchor before channel formation.

These mechanisms are source contracts. The worked reduction examples below
use lit inputs whose emitted protocols directly pin the deleted edges.

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

At the join, solid-01 first handles the trivial boundary cases directly:

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

A mixed-owner loop has no single boundary token. Solid-01 uses a per-owner
first-demand form for it in both placement modes.

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

The next iteration's owner-`{1}` write must wait on either:

- the owner-`{0}` read when the then branch executes; or
- the owner-`{1}` MMA completion when the else branch executes.

Those sources are mutually exclusive. The channel count is one, not two.

Then branch:

```text
MMA [tw] {1}
   |
   v
r FULL, tw [tc5mma] {1}
   | FULL
   v
tr = a FULL {0}
   |
   v
R acc [tr] {0}
   |
   v
r EMPTY, tr [none] {0}
```

Else branch:

```text
MMA [tw] {1}
   |
   v
r EMPTY, tw [tc5mma] {1}
```

Join and next iteration:

```text
then path    r EMPTY, tr {0}  ----> one completed EMPTY phase
else path    r EMPTY, tw {1}  ----> one completed EMPTY phase

executed path's phase ----> next = a EMPTY {1}
```

The completed `if` has no added semaphore-token result: each branch closes
the live token into the alternative supply, and the next write acquires the
selected phase. The test checks that the then-only FULL operations remain in
the then branch and that the else release retains the MMA completion payload.

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

The inner reduced edge cycle is:

```text
W acc {1} --FULL--> R acc {0} --EMPTY--> next W acc {1}
```

The inner region is the only access to this group in the outer body, so the
outer chain adds no separate handoff. DirectBuilder establishes the local
channel at the concrete inner demand:

```text
outer scf.for ... {
  inner scf.for ... {
    tw = a EMPTY {1}
    W acc [tw] {1}
    r FULL, tw [tc5mma] {1}

    tr = a FULL {0}
    R acc [tr] {0}
    r EMPTY, tr [none] {0}
  }
}
```

`EMPTY` starts released. Neither loop has a semaphore-token operand or result,
and no acquire or release is hoisted to the root; semaphore creation still
occurs beside the backing allocation. The original TMEM dependency tokens are
removed later by EMIT-IR because the exact semaphore protocol replaces them.
A zero-trip inner or outer loop executes no acquire or release and leaves the
initial `EMPTY` phase available.

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

The child and parent obligations are separate:

```text
child chain
  W inner {1} -> R inner {0} -> EXIT inner {1}

parent chain
  inner summary {1} -> R outer {0} -> EXIT outer {1}
```

Solid-01 forms four count-1 channels:

```text
LOCAL_EMPTY    next inner write or post-inner done bridge; initially released
LOCAL_FULL     inner write -> inner read
OUTER_FULL     completed inner loop -> outer read
OUTER_EMPTY    outer read -> owner-{1} tail; initially released
```

The inner recurrence remains POU and token-free:

```text
tw = a LOCAL_EMPTY {1}
W inner [tw] {1}
r LOCAL_FULL, tw [tc5mma] {1}
     |
     v
tr = a LOCAL_FULL {0}
R inner [tr] {0}
r LOCAL_EMPTY, tr {0}
```

When the inner loop finishes, DirectBuilder consumes its final local phase
and hands the exact result to the parent continuation:

```text
done = a LOCAL_EMPTY after inner loop {1}
     |
     v
r OUTER_FULL, done {1}
     | OUTER_FULL
     v
to = a OUTER_FULL {0}
     |
     v
R outer [to] {0}
     |
     v
r OUTER_EMPTY, to {0}
```

The next outer invocation must re-open the child's local recurrence:

```text
r OUTER_EMPTY, to {0}
     | OUTER_EMPTY
     v
tail = a OUTER_EMPTY {1}
     |
     v
r LOCAL_EMPTY, tail {1}
     | LOCAL_EMPTY phase
     v
first consumer that executes in the next outer invocation:
  inner-body a LOCAL_EMPTY {1}, or the zero-trip post-inner done acquire
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
coexecuting releases, solid-01 can raise that release's `arrive_count`.

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

otherwise, if this is an eligible unspecialized TMEM accumulator
  numCopies = 2
```

Automatic TMEM double buffering is rejected when accumulator read/modify/write,
unsupported MMA structure, explicit policy, scaled-MMA shape, or the TMEM
capacity budget makes it unsafe.

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` pins an eligible
two-copy accumulator. Its root initialization token is adopted as the exact
owner-`{1}` boundary token; no artificial root-to-partition handoff is added.
Strict POU rejects this loop because that exact boundary token must remain
available across it, so Auto selects the carried topology. The owner-`{2}` MMA
is a later handoff, not the owner-`{1}` boundary demand.

### Semaphore copies

After placement, `computeSemaphoreCopies` inspects final release payloads:

```text
SMEM/non-TMEM group
and no authored buffer.copy
and at least one release contains [tma_load]
  numSemaphoreCopies = max(1, lower-semaphore num-stages)

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
  phase 0, phase 1, ... according to lower-semaphore num-stages
```

The distinction is recorded before EMIT-IR. Rendering does not infer staging
from operation names.

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
final read, so slot replay derives distance one:

```text
r EMPTY after R last(i) {1}
     | completed release/sat pair; slot replay => distance 1
     v
a EMPTY before W(i+1) {3}
```

The test requires cluster legalization to order the next store and its first
consumer after the final read. Its FileCheck observes the adjusted clusters
and the lowered wait/arrive order.

After constraints and offsets are solved, `assignSyncScheduleChain` annotates
each acquire and release with the schedule at its exact owner boundary.
Releases inherit the latest completion for their owner. Acquires use their
selected demand anchor, with special handling for a recurrence at `EXIT`.

## Validation boundary

Solid-01 checks the symbolic protocol before EMIT-IR changes SSA or memory
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

| Responsibility | Current solid-01 implementation |
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
