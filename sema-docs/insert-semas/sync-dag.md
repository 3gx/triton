# SYNC-DAG

## Contents

- [Purpose](#purpose)
- [Notation](#notation)
- [From accesses to synchronization edges](#from-accesses-to-synchronization-edges)
  - [Piece state](#piece-state)
  - [The access rules](#the-access-rules)
  - [What the initial edge set contains](#what-the-initial-edge-set-contains)
  - [Example: synchronization between two partitions](#example-synchronization-between-two-partitions)
  - [Example: several readers and token reuse](#example-several-readers-and-token-reuse)
  - [Example: disjoint pieces stay independent](#example-disjoint-pieces-stay-independent)
  - [Nested regions](#nested-regions)
  - [Example: the same rules at two region levels](#example-the-same-rules-at-two-region-levels)
- [Reducing synchronization edges](#reducing-synchronization-edges)
  - [Example: an edge within one region is redundant](#example-an-edge-within-one-region-is-redundant)
  - [Example: edge reduction lowers the pending count](#example-edge-reduction-lowers-the-pending-count)
  - [Example: an edge to `EXIT` is redundant](#example-an-edge-to-exit-is-redundant)
  - [Exact-source async edges](#exact-source-async-edges)
  - [Earliest release position (`releaseFloors`)](#earliest-release-position-releasefloors)
  - [Reduction does not choose placement](#reduction-does-not-choose-placement)
- [From reduced edges to semaphores](#from-reduced-edges-to-semaphores)
  - [Repeated edges from one sender](#repeated-edges-from-one-sender)
  - [Removing a release when another path imposes the same wait](#removing-a-release-when-another-path-imposes-the-same-wait)
  - [One destination, one semaphore](#one-destination-one-semaphore)
  - [Reading one semaphore](#reading-one-semaphore)
  - [Creating the first token](#creating-the-first-token)
  - [Entry and the next iteration use one semaphore](#entry-and-the-next-iteration-use-one-semaphore)
- [Tokens through `for` and `if`](#tokens-through-for-and-if)
  - [Region results](#region-results)
  - [Example: an `if` returns one owner's token](#example-an-if-returns-one-owners-token)
  - [Moving an acquire to its first use](#moving-an-acquire-to-its-first-use)
  - [When the loop keeps a token](#when-the-loop-keeps-a-token)
  - [A use after the loop](#a-use-after-the-loop)
  - [Nested loops](#nested-loops)
  - [Branch completion must agree](#branch-completion-must-agree)
- [Backing copies](#backing-copies)
- [Pipeline schedule](#pipeline-schedule)
  - [Minimal pipeline model](#minimal-pipeline-model)
  - [Example: one-copy synchronization between iterations](#example-one-copy-synchronization-between-iterations)
  - [Finalizing one release/acquire pair](#finalizing-one-releaseacquire-pair)
  - [Moving an acquire updates its schedule relation](#moving-an-acquire-updates-its-schedule-relation)
  - [Post-loop acquires use their owner's boundary](#post-loop-acquires-use-their-owners-boundary)
  - [Schedule and stage offset are separate](#schedule-and-stage-offset-are-separate)
- [Explicit buffer-stage offsets](#explicit-buffer-stage-offsets)
  - [Circular groups](#circular-groups)
  - [Non-circular aliases](#non-circular-aliases)
- [Build order and code map](#build-order-and-code-map)

## Purpose

SYNC-DAG turns buffer access order into a plan for
`nvws.semaphore.acquire`, `nvws.semaphore.release`, and token operations. It
also decides how tokens pass through `for` and `if`, how many buffer and
semaphore copies exist, and where the new semaphore operations belong in a
scheduled loop. EMIT-IR later renders this plan and creates each
`nvws.semaphore.buffer` at its access.

The construction order is important:

```text
all required synchronization edges
    -> choose physical backing copies
    -> remove safely redundant edges
    -> merge repeated edges from one sender
    -> group remaining edges by destination
    -> assign a semaphore and pending count
    -> create entry tokens and plan region tokens
    -> choose semaphore copies
    -> finalize stage offsets and schedules
```

The input [access DAG](access-dag.md#regions-and-boundaries) already contains:

- one node for each buffer access;
- an owner for each access;
- a disjoint piece table for overlapping allocations; and
- `ENTER` and `EXIT` nodes for each `for` and `if` path.

SYNC-DAG adds synchronization edges between those concrete nodes. Each edge
says that the destination owner must wait for the source owner. Semaphore
operations are the IR form of the reduced edges; they are not a second
ordering analysis.

## Notation

The examples use explanatory pseudo-IR and three kinds of diagrams. Each
diagram stays at one level:

```text
initial DAG    access, region, ENTER, and EXIT nodes joined by synchronization edges
reduced DAG    the initial DAG after redundant edges are removed
semaphore DAG  acquire, access, and release nodes joined by semaphores S0, S1, ...
```

`walk` marks relevant program order between nodes with the same owner when no
synchronization edge connects them. Diagrams may omit unrelated nodes between
them. An edge used by the next iteration is stored as `source -> EXIT(i)`. A
diagram that follows the edge across the loop boundary labels the two
iterations `i` and `i+1`.

Names used throughout:

```text
group          allocations analyzed together, ordinarily one buffer.id
backing        the physical SMEM or TMEM allocation used by the group
m0, m1         members: allocation names or views in the group
P0, P1         disjoint pieces of the backing
{0}, {1}       owners: partitions 0 and 1 of the enclosing WS loop
root           code with no partition owner
source         node that supplies the current value to a new reader
use            latest access to the current value by one owner
sender         source owner whose edges into one destination become one release
token          value returned by an acquire and used by releases and semaphore.buffer
```

Pseudo-IR omits types and unrelated attributes:

```text
%t = acquire S0 {1}
%b = semaphore.buffer S0, %t
R m0 [%b] {1}
release S1, %t {1}
for iter_args(%t = %entry) { ... yield %next }
```

An edge may record one or more completion kinds. `[none]` completes when the
release executes. `[tma_load]` and `[tc5mma]` complete with the named async
operation. After edges from one source owner are merged, each distinct
completion kind adds one to the acquire's `pending_count`.

## From accesses to synchronization edges

`ChainWalker` walks one group in program order. It keeps independent state for
each piece and a deterministic list of available owner tokens for each
region. The walk records every required synchronization edge, even when other
recorded reads or writes already impose the same wait.

### Piece state

For each piece, `PieceState` contains:

```text
source    DAG node for the most recent write to the piece
uses      [owner -> latest read or write by that owner since the source, ...]
```

The implementation also records completion kinds, the owner that established
the current contents, and which owners are already ordered after each use.

A read changes only that owner's entry in `uses`; the source remains the most
recent write. A write becomes the new source and removes the earlier entries
from `uses`. Before the first write, the first access or a child `ENTER`
provides the source node.

The region also keeps a deterministic list of available tokens:

```text
tokens = [{0} at W0, {1} at R1, ...]
```

An owner can reuse its earlier token when the recorded reads and writes for
every piece allow it. `Node::reuseTokenOwner` records that choice for EMIT-IR.

An `ENTER` whose pieces all have the same owner adds a token for that owner
without attaching the token to an access node. The owner may reuse it
immediately. This token alone cannot supply an additional edge to another
owner, but the normal new-reader rule may still add an edge from `ENTER`.

### The access rules

`applyTouch` applies these rules to every piece read or written by the access:

```text
first read or write
  synchronization edges: none
  state: source = this node; uses = [this owner -> this node]

write
  for every entry in uses with a different owner:
    add an edge from that entry's latest read or write to this write
  state: source = this write; uses = [this owner -> this write]

read whose owner is already in uses
  synchronization edges: none
  state: replace that owner's use with this read; source does not move

read by a new owner
  synchronization edge: source -> this read, except for the root case below
  state: add this owner -> this read to uses
```

For a write, no edge is added from a recorded use already known to occur
before this owner reaches the write. A warp-specialized `for` also adds no
edge from code with no owner when one of its partitions becomes the owner.

After applying these rules to every piece read or written by the access,
`visitAccess` needs an additional token decision only when the rules added no
synchronization edge:

```text
an earlier token with the same owner as this access is safe to reuse
  record that owner in reuseTokenOwner

otherwise, the last available token belongs to another owner
  add a synchronization edge from that token's node to this access
```

If one or more edges into this access remain after all edge removal and
grouping, the acquire created for their common destination returns the token
used by the access. If the latest token already has this access's owner, no
edge or reuse marker is needed.

A read can reuse a token only when its owner has a recorded use for every
piece that it reads. A write can reuse a token only when every recorded use by
another owner is already ordered before the write.

In a multi-member group, a later synchronous write may retain earlier async
completion kinds only when the owner can safely reuse its token. This keeps
the earlier descriptor load attached to the release that waits for it.

### What the initial edge set contains

The initial edge set records every wait required by the buffer reads and
writes. It also ensures that each access can obtain a token owned by its
partition.

Completeness is checked per piece. An access that spans two pieces can need
two synchronization edges even when both edges have the same endpoint nodes.
The edges remain separate while the pass visits accesses because the pieces
can differ later. They are combined only after reduction, when one release
from a source owner can satisfy the destination for both pieces.

The pass also records an edge when other recorded edges already impose the
same wait. Its source still determines how early a release may be placed. A
later step removes it only after checking that the other edges remain.

No synchronization edge is needed merely because two operations appear next
to each other. Reads and writes by the same owner normally use the current
token and follow program order. An edge is added when another owner must wait
before reading or writing, or when another owner must acquire a token.

Region boundaries do not introduce a different rule set. `ENTER` supplies
the buffer state at the start of a region path. Edges to `EXIT` make the next
iteration or later code wait for reads and writes inside that path. The same
source, uses, and token rules apply inside the region.

The initial set may contain more edges than the emitted releases and
acquires. The next steps remove redundant edges and merge remaining edges
that can use the same release or acquire.

### Example: synchronization between two partitions

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` stores in
owner `{0}` and loads in owner `{1}` on every iteration.

Initial edge set:

```text
nodes in walk order
  N0 = ENTER(i) {0}
  N1 = W store(i) {0}
  N2 = R load(i) {1}
  N3 = EXIT(i) {0}

generated synchronization edges
  e1: W store(i) {0} -> R load(i) {1}       read after write
  e2: R load(i) {1} -> EXIT(i) {0}          next write waits for this read
```

The initial DAG represents the wait before the next iteration as an edge to
`EXIT(i)`:

```text
                         ENTER(i) {0}
                              | walk
                              v
                         W store(i) {0}
                              | e1
                              v
                         R load(i) {1}
                              | e2
                              v
                          EXIT(i) {0}
```

After reduction, merging, grouping, and semaphore assignment, `e1` is
implemented with semaphore `FULL` and `e2` with initially released semaphore
`EMPTY`:

```text
%empty = acquire EMPTY {0}
W store [%empty] {0}
release FULL, %empty {0}

%full = acquire FULL {1}
R load [%full] {1}
release EMPTY, %full {1}
```

The acquire of `EMPTY` is placed at the first store inside the loop. Because
`EMPTY` is initially released, its acquire returns the token for iteration
zero. Each `release EMPTY` allows the acquire for the next iteration to
complete.

```text
semaphore DAG showing the boundary from iteration i to iteration i+1

                       acquire EMPTY(i) {0}
                                  | walk
                                  v
                           W store(i) {0}
                                  | walk
                                  v
                         release FULL(i) {0}
                                  | FULL
                                  v
                         acquire FULL(i) {1}
                                  | walk
                                  v
                           R load(i) {1}
                                  | walk
                                  v
                        release EMPTY(i) {1}
                                  | EMPTY
                                  v
                       acquire EMPTY(i+1) {0}
                                  | walk
                                  v
                         W store(i+1) {0}
```

### Example: several readers and token reuse

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced` writes in `{0}`, reads in `{1}` and `{2}`, then rereads
in `{0}`.

```text
walk node       generated synchronization edge     state after node
ENTER {0}       none                               source=ENTER; uses={0}:ENTER
W alloc {0}     none                               source=W; uses={0}:W
R load {1}      f1: W {0} -> R {1}                source=W; uses={0}:W,{1}:R1
R load {2}      f2: W {0} -> R {2}                source=W; uses={0}:W,{1}:R1,{2}:R2
R load {0}      none; reuse {0}'s token            source=W; uses={0}:R0,{1}:R1,{2}:R2
EXIT {0}        f3: R1 {1} -> EXIT {0}
                f4: R2 {2} -> EXIT {0}
```

Both `R {1}` and `R {2}` receive an edge from the same write, but no
synchronization edge orders the two reads with respect to each other. The
`{0}` reread also has no incoming synchronization edge; it appears only in
program order.

```text
                                  ENTER(i) {0}
                                       | walk
                                       v
                                  W alloc(i) {0}
                         +-------------+-------------+
                      f1 |        walk |             | f2
                         v             v             v
                  R load(i) {1} R load(i) {0} R load(i) {2}
                      f3 |        walk |             | f4
                         +-------------+-------------+
                                       v
                                  EXIT(i) {0}

semaphore DAG for the same synchronization edges

                     acquire EMPTY(i) pending_count=2 {0}
                                      ptok |
                                           v
                                     EXIT(i-1) {0}
                                           | next iteration
                                           v
                                      ENTER(i) {0}
                                           | walk
                                           v
                                  W alloc(i) [ptok] {0}
                    +----------------------+----------------------+
               walk |                 walk |                 walk |
                    v                      v                      v
    release F1, ptok {0}       R load(i) [ptok] {0}    release F2, ptok {0}
                 F1 |                 walk |                      | F2
                    v                      v                      v
   r1tok = acquire F1 {1}                  |          r2tok = acquire F2 {2}
               walk |                      |                 walk |
                    v                      |                      v
      R load(i) [r1tok] {1}                |         R load(i) [r2tok] {2}
               walk |                      |                 walk |
                    v                      |                      v
  release EMPTY, r1tok {1}                 |     release EMPTY, r2tok {2}
              EMPTY |                      |                     | EMPTY
                    +----------------------+---------------------+
                                           v
                      acquire EMPTY(i+1) pending_count=2 {0}
                                      next |
                                           v
                                      EXIT(i) {0}
                                           | next iteration
                                           v
                                    ENTER(i+1) {0}
```

The source nodes of `f3` and `f4` each release the same semaphore. Its acquire
has `pending_count=2`:

```text
release EMPTY, %reader1 {1}
release EMPTY, %reader2 {2}
%next = acquire EMPTY pending_count=2 {0}
```

`reuseTokenOwner` tells EMIT-IR to use the earlier `{0}` token for the final
`{0}` read.

### Example: disjoint pieces stay independent

This conceptual example uses `m0` for the first half of a buffer, `m1` for
the second half, and `m2` for the whole buffer:

```text
members:    m0[0,128)   m1[128,256)   m2[0,256)
pieces:     P0=[0,128){m0,m2}   P1=[128,256){m1,m2}
```

The DAG nodes and the edges that remain for semaphore construction are:

```text
DAG node         buffer pieces            synchronization edge ending here
ENTER(i) {0}     P0, P1                   none
W m2 {0}         P0, P1                   none
R m2 {1}         P0, P1                   e1: W m2 {0} -> R m2 {1}
W m0 {2}         P0                       e2: R m2 {1} -> W m0 {2}
R m0 {3}         P0                       e3: W m0 {2} -> R m0 {3}
W m1 {4}         P1                       e4: R m2 {1} -> W m1 {4}
R m1 {0}         P1                       e5: W m1 {4} -> R m1 {0}
EXIT(i) {0}      P0, P1                   e6: R m0 {3} -> EXIT(i) {0}
```

Before reduction, the pass records two edges from `W m2 {0}` to `R m2 {1}`:
one for P0 and one for P1. Because they have the same endpoints, one remaining
edge, `e1: W m2 {0} -> R m2 {1}`, is sufficient for that wait. The pass also
records additional direct edges whose waits are already imposed by the paths
shown below. Those additional edges are removed before semaphore construction,
as explained in
[Reducing synchronization edges](#reducing-synchronization-edges). The
remaining synchronization-edge DAG is:

```text
                                  ENTER(i) {0}
                                       | walk
                                       v
                                  W m2(i) {0}
                                       | e1
                                       v
                                  R m2(i) {1}
                         +-------------+-------------+
                  e2: P0 |                           | e4: P1
                         v                           v
                    W m0(i) {2}                 W m1(i) {4}
                      e3 |                           | e5
                         v                           v
                    R m0(i) {3}                 R m1(i) {0}
                      e6 |                           | walk
                         +-------------+-------------+
                                       v
                                  EXIT(i) {0}
```

`e1` becomes `FULL_BOTH`; `e2` through `e5` become the left and right
semaphores; and `e6` becomes initially released semaphore `EMPTY`:

```text
semaphore DAG

                     acquire EMPTY(i) pending_count=1 {0}
                                      btok |
                                           v
                                     EXIT(i-1) {0}
                                           | next iteration
                                           v
                                      ENTER(i) {0}
                                           | walk
                                           v
                                W m2(i) [btok] {0}
                                           | walk
                                           v
                             release FULL_BOTH, btok {0}
                                  FULL_BOTH |
                                            v
                               both = acquire FULL_BOTH {1}
                                            | walk
                                            v
                                 R m2(i) [both] {1}
                       +--------------------+--------------------+
                  walk |                                         | walk
                       v                                         v
       release LEFT_READY, both {1}             release RIGHT_READY, both {1}
            LEFT_READY |                              RIGHT_READY |
                       v                                          v
          left = acquire LEFT_READY {2}             right = acquire RIGHT_READY {4}
                  walk |                                         | walk
                       v                                         v
             W m0(i) [left] {2}                         W m1(i) [right] {4}
                  walk |                                         | walk
                       v                                         v
        release LEFT_FULL, left {2}              release RIGHT_FULL, right {4}
             LEFT_FULL |                               RIGHT_FULL |
                       v                                          v
         lread = acquire LEFT_FULL {3}             rread = acquire RIGHT_FULL {0}
                  walk |                                         | walk
                       v                                         v
            R m0(i) [lread] {3}                       R m1(i) [rread] {0}
                  walk |                                         | walk
                       v                                         |
          release EMPTY, lread {3}                                |
                 EMPTY |                                         |
                       +--------------------+--------------------+
                                            v
                       acquire EMPTY(i+1) pending_count=1 {0}
                                      next |
                                           v
                                      EXIT(i) {0}
                                           | next iteration
                                           v
                                    ENTER(i+1) {0}
```

After `R m2`, the P0 and P1 paths have no synchronization edge between them.
P0 releases `EMPTY` because its last reader is owner `{3}`. `R m1` and the
acquire of `EMPTY` both have owner `{0}`, and program order places the read
before the acquire.

### Nested regions

A region has one summary node in its parent node sequence and a separate node
sequence for each child path:

```text
parent nodes                     child nodes

... -> [for or if summary] ...   ENTER -> child nodes -> EXIT
```

The parent applies the normal read/write rules to the summary. Each child
path starts with separate state for every piece. A read by a new owner
receives an edge from `ENTER`, not directly from a parent access. `ENTER`
itself creates no acquire or release.

The child remembers which owner established the buffer contents before the
region. If that owner is also the `ENTER` owner, `ENTER` keeps the
asynchronous completion kinds; otherwise it uses `[none]`.

When the piece may be used in the next iteration or after the region, `EXIT`
receives an edge from the latest read or write by each other owner unless it
is already ordered before the `EXIT` owner. The parent resumes from the state
produced by its region summary; child uses never replace the parent's uses.

After a region, tokens recorded before the region are no longer considered
available. If the summary has one partition owner, the region node records a
token for that owner. Otherwise it records no partition token, so a later
access may need an edge solely to acquire one.

An ordering established inside an `if` is valid afterward only when every
path establishes it. A missing `else` leaves the incoming ordering unchanged.
For a loop that may execute zero times, ordering established only in its body
is not assumed after the loop.

### Example: the same rules at two region levels

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` contains an outer write by `{3}`
and an inner loop summarized as a write by `{2}`.

Parent synchronization edges:

```text
p1: W m0 {3} -> [inner-for summary P0:W:{2}]
p2: [inner-for summary P0:W:{2}] -> EXIT outer(i) {3}
```

```text
                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0 {3}
                                  | p1
                                  v
                   [inner-for summary P0:W:{2}]
                                  | p2
                                  v
                         EXIT outer(i) {3}
```

The child starts P0 at `ENTER inner(i) {2}` and applies exactly the same
rules:

```text
child synchronization edges
  c1: ENTER inner(i) {2} -> R m0 {1}
  c2: R m0 {2}          -> W m0 {1}
  c3: W m0 {1}          -> R m0 {0}
  c4: W m0 {1}          -> EXIT inner(i) {2}
  c5: R m0 {0}          -> EXIT inner(i) {2}
```

`R m0 {2}` rereads the buffer contents present at `ENTER` and updates only
`{2}`'s recorded use. The later write by `{1}` therefore waits for that read
through `c2`. `R m0 {1}` is not the source of `c2` because it has the same
owner as the write.

```text
                         ENTER inner(i) {2}
                                   |
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
                          EXIT inner(i) {2}
```

Only `c1` through `c5` are synchronization edges. The two `walk` lines are
program order.
The parent and child piece states remain separate. Semaphore sharing between
`p1` and the child edges to `EXIT` is explained in
[Entry and the next iteration use one semaphore](#entry-and-the-next-iteration-use-one-semaphore).

## Reducing synchronization edges

The walk produces a complete edge set. `reduceEdges` then applies two
functions:

- `reduceStraightEdges` handles edges between access nodes in one region.
- `reduceLoopCloses` handles edges from an access to `EXIT` when the other
  ordering crosses the loop boundary.

Both functions consider only program order and edges that have not been
removed. Only a kept edge updates the ordering and available-token
information used for later decisions.

### Example: an edge within one region is redundant

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members:

```text
members:    m0[0,256)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0,m1}   P2=[192,256){m0}
```

On P1, owner `{0}` writes, owner `{1}` reads, and owner `{2}` writes. The
second write finds recorded uses by both `{0}` and `{1}`.

```text
initial edge set
  e1: W m0 {0} -> R m0 {1}
  e2: W m0 {0} -> W m1 {2}       considered for removal
  e3: R m0 {1} -> W m1 {2}
```

```text
initial DAG

                            W m0 {0}
                       +---------+---------+
                    e1 |                   | e2
                       v                   |
                   R m0 {1}                |
                    e3 |                   |
                       +---------+---------+
                                 v
                            W m1 {2}
```

Following `e1` and then `e3` makes `W m1 {2}` wait for `W m0 {0}`, so `e2`
is unnecessary. `e3` also provides the token used by owner `{2}`.
`reduceStraightEdges` drops `e2`.

```text
reduced DAG

                            W m0 {0}
                                | e1
                                v
                            R m0 {1}
                                | e3
                                v
                            W m1 {2}
```

`reduceStraightEdges` does not consider root, region, `ENTER`, or `EXIT`
endpoints. It keeps an edge unless the other remaining edges together with
program order impose the same wait and provide a token for the destination
owner.

### Example: edge reduction lowers the pending count

The same example shows why reduction precedes semaphore creation. If all
three synchronization edges were kept, the write by `{2}` would wait for
releases from both `{0}` and `{1}`. After `e2` is removed, only `e3` ends at
that write:

```text
synchronization edges
  e1, e2, e3

after reduceStraightEdges
  e1, e3

after grouping
  e1 -> S0 pending_count=1 at R m0 {1}
  e3 -> S1 pending_count=1 at W m1 {2}
```

```text
semaphore DAG

                            W m0 {0}
                                | walk
                                v
                          release S0 {0}
                                | S0
                                v
                          acquire S0 {1}
                                | walk
                                v
                            R m0 {1}
                                | walk
                                v
                          release S1 {1}
                                | S1
                                v
                          acquire S1 {2}
                                | walk
                                v
                            W m1 {2}
```

`S1` has `pending_count=1` because only `e3` ends at `W m1 {2}`.

### Example: an edge to `EXIT` is redundant

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` uses two partly overlapping members:

```text
members:    m0[0,128)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,128){m0,m1}   P2=[128,192){m1}
```

The complete in-body synchronization edge set is:

```text
l1a: W m0 {0} -> R m0 {1}       P0
l1b: W m0 {0} -> R m0 {1}       P1
l2a: W m0 {0} -> W m1 {2}       P1, possible removal through {1}
l2b: R m0 {1} -> W m1 {2}       P1
l3a: W m1 {2} -> R m1 {0}       P1
l3b: W m1 {2} -> R m1 {0}       P2
```

The walk then records three synchronization edges to `EXIT`:

```text
c0: R m0(i) {1} -> EXIT(i) {0}   P0 returns to first owner {0}
c1: W m1(i) {2} -> EXIT(i) {0}   P1, possible removal through {0}
c2: R m1(i) {0} -> EXIT(i) {2}   P2 returns to first owner {2}
```

`c1` starts at `W m1`, not `R m0`, because that write replaces P1's contents.
The write is already ordered before `{0}`'s P1 read.

`reduceStraightEdges` keeps `l1a` and `l3a`, drops their same-endpoint
duplicates `l1b` and `l3b`, and drops `l2a` through `l1a -> l2b`.

For `reduceLoopCloses`, the stored `c2` edge ends at `EXIT(i)`. The diagram
follows owner `{2}` from `EXIT(i)` to its first P2 access, `W m1(i+1)`.

```text
path considered when testing c2, shown across two iterations

                         R m1(i) {0}
                         +-------------------------------+
                    walk |                            c2 |
                         v                               |
                      EXIT(i)                            |
                         | walk                          |
                         v                               |
                     ENTER(i+1)                          |
                         | walk                          |
                         v                               |
                    W m0(i+1) {0}                        |
                         | l1a                           |
                         v                               |
                    R m0(i+1) {1}                        |
                         | l2b                           |
                         v                               |
                    W m1(i+1) {2} <----------------------+
```

The kept path `l1a -> l2b` orders the next P2 write and makes `{2}`'s token
available there. `{2}` is not the owner of the region's first
partition-owned access, so `reduceLoopCloses` drops `c2`.

```text
after reduceLoopCloses, shown across two iterations

                         R m1(i) {0}
                              | walk
                              v
                           EXIT(i)
                              | walk
                              v
                          ENTER(i+1)
                              | walk
                              v
                         W m0(i+1) {0}
                              | l1a
                              v
                         R m0(i+1) {1}
                              | l2b
                              v
                         W m1(i+1) {2}
```

Both `c0` and `c1` target `{0}`, the first access owner, so
`reduceLoopCloses` retains them. `buildEdgesAndSemas` later removes `c1`:
`l3a` orders `{2}`'s P1 write before `{0}`'s P1 read, and `{0}`'s operation
order then reaches `EXIT`. `c0` is the only edge left for the next iteration.

The two functions use different information. `reduceLoopCloses` uses the
first access owner, the first read or write of each piece in the next
iteration, and token availability. The later removal requires remaining
edges from the source owner through another owner to the destination.

### Exact-source async edges

An edge starting at an asynchronous write also records that operation's
completion kind. `EdgeRec::preserve` prevents either edge reducer from
deleting it, but `buildEdgesAndSemas` may still remove its release when
another path provides the same wait.

```text
initial edge set
  a1: W async {0} -> R {1}       completion [tma_load]
  a2: W async {0} -> W {2}       exact source, completion [tma_load],
                                  possible removal through {1}
  a3: R {1}       -> W {2}
```

```text
initial DAG

                          W async {0}
                    +-----------+-----------+
     a1 [tma_load]  |                       | a2 [tma_load], through {1}
                    v                       |
                  R {1}                     |
               a3   |                       |
                    +-----------+-----------+
                                v
                              W {2}
```

The path `a1 -> a3` already orders the endpoints of `a2`. Because `a2` also
identifies the asynchronous write as its completion source, `preserve` keeps
it through both reducers.

`buildEdgesAndSemas` may still delete `a2`: `a1` waits for the TMA load before
`{1}` reads, and `a3` orders that read before `{2}` writes. If either edge
were removed, `a2` would remain. A later synchronous read is not the write
that produced the value and therefore does not receive `preserve`.

### Earliest release position (`releaseFloors`)

Before reduction, `releaseFloors` records the latest source node for every
`(destination, destination owner, source owner)` tuple. A remaining release
from that source owner cannot be placed before this node.

Consider two synchronization edges from `{1}` into the same destination:

```text
edge  endpoints
r1    A {1} -> D {2}
r2    B {1} -> D {2}      B follows A in the same node sequence
```

The earliest release position is `B`, even if reduction later removes `r2`.
When the remaining `{1}` edges are merged, their release is placed no earlier
than B:

```text
release and acquire placement after semaphore assignment

A {1} --walk--> B {1} --walk--> release S {1} --S--> acquire S {2}
                                                       |
                                                     walk
                                                       v
                                                     D {2}
```

This gives edge removal a placement rule:

- removing all edges from one source owner may remove that owner's release;
- reduction may remove a whole semaphore;
- a remaining release cannot move earlier.

The rule matters for asynchronous and warp-group work: a release after the
source owner's last write must not move before an earlier read merely because
one synchronization edge became redundant.

### Reduction does not choose placement

Reduction asks whether remaining edges provide the same wait and destination
token. Release placement asks how early a remaining release may execute. The
reducers remove edges but do not recompute `releaseFloors`.

Removing one per-piece edge may leave another edge from the same source owner
and retain its release. Removing every edge from one source owner removes that
owner's release. Removing every source owner removes the acquire and
semaphore. If a release remains, it stays at or after its recorded position.

Merging combines remaining edges from one source owner into one release and
combines their completion kinds. It does not prove an ordering path, move a
release earlier, or combine releases from different owners. Grouping places
releases with the same destination under one acquire and computes its pending
count.

## From reduced edges to semaphores

`buildEdgesAndSemas` processes the reduced edge set in this order:

```text
1. merge edges with the same destination, destination owner, and source owner
2. group merged edges by destination node and destination owner
3. remove a source-owner release when another remaining path provides the same wait
4. reconcile loop-entry and next-iteration pending counts
5. create one semaphore for each remaining destination group
6. insert one acquire at the destination and one release for each source owner
```

### Repeated edges from one sender

Two pieces can create edges from different nodes of the same owner into the
same `EXIT`. Use this conceptual loop:

```text
W m0 {0}        writes P0 and P1
R m0 {1}        latest {1} use for P0
R m1 {1}        later {1} use for P1
EXIT(i) {0}     returns both pieces to {0}
```

Initial edge set:

```text
m1a: W m0 {0} -> R m0 {1}       P0
m1b: W m0 {0} -> R m0 {1}       P1
m2: R m0 {1} -> EXIT(i) {0}      P0
m3: R m1 {1} -> EXIT(i) {0}      P1
```

```text
initial DAG

                            W m0 {0}
                         m1a,m1b |
                                v
                            R m0 {1}
                         +------+------+
                  m2(P0) |             | walk
                         |             v
                         |         R m1 {1}
                         |             | m3(P1)
                         +------+------+
                                v
                           EXIT(i) {0}
```

`reduceStraightEdges` keeps `m1a` and removes duplicate `m1b`.
`reduceLoopCloses` removes neither `m2` nor `m3`, so the reduced edge set is
`m1a`, `m2`, and `m3`.

`m2` and `m3` have the same sender and destination. The later source is
`R m1 {1}`, so one release after that node is sufficient for both edges.
Their completion kinds are combined.

```text
same-sender merge

destination       EXIT(i) {0}
sender owner      {1}
earliest release node  R m1 {1}
represented edges      m2, m3
merged edge             M1: R m1 {1} -> EXIT(i) {0}
```

With `[none]`, the release signals the semaphore once. With
`[none, tma_load]`, it signals twice, so the acquire has `pending_count=2`
even though there is one sender owner.

### Removing a release when another path imposes the same wait

While constructing synchronization edges, the pass may record that an edge
could later be removed through another owner. It does so only when the edge
starts at the current source node and an existing edge already orders that
node before the other owner's latest access.

After edges with the same destination node, destination owner, and source
owner are merged, the source owner's release can be removed only when every
edge represented by that release names such an intermediate owner. For each
edge, the remaining synchronization must provide both parts of the path:

1. from the original source owner to the intermediate owner; and
2. from the intermediate owner to the destination.

For the first part, a remaining release from the original source owner must
occur at or after the original source node. Its acquire must occur no later
than the node from which the intermediate owner releases to the final
destination. When the intermediate owner is also the destination owner, the
acquire must instead occur no later than the destination node, and that
owner's operation order provides the second part. A check may cross enclosing
`for` boundaries, but never from one `if` branch into another.

The pass records the edge before reduction and uses its source when computing
`releaseFloors`. If the edge is later removed, that earlier record still
prevents a remaining release from moving before its original position. If the
edge remains after reduction, it is merged with other edges from the same
source owner.

The inner loop in
`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` has this shape. Owner `{1}`
writes the buffer, owner `{0}` reads the written value, and the inner `EXIT`
has owner `{2}`.

```text
initial edge set
  k1: W correct {1} -> R corrected {0}
  k2: W correct {1} -> EXIT inner(i) {2}   possible removal through {0}
  k3: R corrected {0} -> EXIT inner(i) {2}
```

```text
initial DAG

                           W correct {1}
                       +---------+---------+
                    k1 |                   | k2 through {0}
                       v                   |
                 R corrected {0}           |
                    k3 |                   |
                       +---------+---------+
                                 v
                        EXIT inner(i) {2}
```

Both edges to `EXIT` remain after `reduceLoopCloses` because `{2}` is the
first access owner. After merging edges with the same destination node,
destination owner, and source owner, `buildEdgesAndSemas` checks whether the
path through `{0}` still exists:

- `k1` orders `{1}`'s write before `{0}`'s read;
- `k3` orders `{0}`'s read before the destination.

Therefore `k2` and `{1}`'s release into the `EXIT` semaphore can be removed.
The separate `{1}` release created for `k1` remains, as do `k3` and `{0}`'s
release into `EXIT`:

```text
reduced DAG after removing k2

                           W correct {1}
                                 | k1
                                 v
                         R corrected {0}
                                 | k3
                                 v
                        EXIT inner(i) {2}
```

After semaphore assignment, `k1` and `k3` become different semaphores:

```text
%writer = token already owned by {1}
W correct [%writer] {1}
release CORRECTED, %writer {1}
%corrected = acquire CORRECTED {0}
R corrected [%corrected] {0}
release READY, %corrected {0}
%regain = acquire READY pending_count=1 {2}
```

If any edge represented by a source owner's release has no recorded
intermediate owner, that release remains at the position recorded in
`releaseFloors`. This step never deletes only an earlier edge while retaining
another edge from the same source owner.

Each proposed removal is checked against the edges that remain. For example:

```text
initial edge set into D
  f1: A {0} -> B {1}
  f2: B {1} -> C {2}
  f3: A {0} -> D {3}     possible removal through {1}
  f4: B {1} -> D {3}     possible removal through {2}
  f5: C {2} -> D {3}     no alternate path
```

```text
initial DAG

                              A {0}
                       +---------+-----------------+
                    f1 |                           | f3 through {1}
                       v                           |
                     B {1}                         |
                       +---------+-----------------+
                    f2 |                           | f4 through {2}
                       v                           |
                     C {2}                         |
                  f5   |                           |
                       +------------+--------------+
                                    v
                                  D {3}
```

`f4` may be removed because `f2` orders `B` before `C` and `f5` orders `C`
before `D`. After `f4` is removed, that path still orders `B` before `D`, but
owner `{1}` no longer has its own release into `D`. The check for `f3`
requires that release, so `f3` must remain. The pass repeats the check until
every proposed removal still has both required path segments. It never uses
one `if` branch to prove ordering in another.

Finally, loop entry and the next iteration may share one semaphore. These removals
are optional, so the pass keeps the releases when removing them would make
the two sites unable to use one fixed pending count. The first two columns
below show the signal counts after the proposed removals but before any
`arrive_count` adjustment:

```text
entry signals    next-iteration signals   result
1 [none]         2                     entry release uses arrive_count=2
2                2                     both sites use pending_count=2
2                1 after removal       cancel all proposed removals at both sites;
                                       recheck counts
```

If all incoming edges to a destination are removed, no acquire or semaphore
is created for them.

### One destination, one semaphore

After reduction and merging, each `(destination node, destination owner)`
pair receives one semaphore and one acquire. Each remaining source owner
receives one release. The pending count is:

```text
sum over remaining senders of max(1, number of completion kinds)
```

This means:

- two `[none]` senders give pending count 2;
- one sender with `[none, tma_load]` gives pending count 2;
- two per-piece edges merged into one `[none]` source owner give pending count 1.

The release is placed immediately after its earliest allowed node, after any
earlier releases there. An asynchronous source uses its physical completion
point. The acquire is placed before the destination, except when it can be
safely placed at that owner's first read or write in the next iteration.

### Reading one semaphore

Start at an acquire and work backward. Its destination node and destination
owner identify the synchronization. Every remaining source owner has one
release of that semaphore. The pending count is the total number of times
those releases signal it. Releases from different owners remain separate.

Then work forward from the acquire. `semaphore.buffer` uses the semaphore and
the acquire token to produce the buffer operand used by the destination
access. The same token can feed later releases by the destination owner until
another acquire replaces it or the token is returned through `for` or `if`.

The semaphore is chosen from the reduced edges, not from the allocation
member that happened to create an edge. Overlapping members in one group can
therefore use the same semaphore when their reduced edges have the same
destination and owners. Disjoint paths keep separate semaphores because
their destination nodes or owners differ.

An async completion belongs on the release that represents its source. It
increases the pending count of the same acquire; it does not create another
acquire. A release with `[none, tma_load]` signals twice, while a plain
release signals once.

Entry and next-iteration acquires for the same buffer group may use one
semaphore. A loop-carried token requires both sites to use that semaphore and
the same pending count.

After this point, edges are not reduced or merged again. Token planning may
move or add acquires and add releases needed to pass a token across a region
boundary. Those operations use the established semaphore and pending count.
Schedule finalization may then change clusters or stage offsets without
recomputing the reduced edges.

### Creating the first token

`insertEntryAcquires` inserts an acquire before the first access or region
that needs a token. The acquire is outside the first `for` when that loop is
the first node. Its IR owner is `root`, while `entryTokenOwner` records the
first access owner for token checks and emission.

If a loop already uses a semaphore between iterations, the entry acquire uses
that semaphore and marks it initially released:

```text
%first = acquire READY root
for iter_args(%token = %first) {
  ...
  %next = acquire READY {1}
  yield %next
}
```

Otherwise, the pass creates a count-1 entry semaphore. Its acquire supplies
the first token, and its release follows the last access or region:

```text
%entry = acquire ENTRY root
... first access or region ...
... last access or region ...
release ENTRY, %last_token
```

If the only top-level access or region that needs a token is an `if`, and
exactly one branch contains an access or region that reads or writes the
group, the acquire can be placed in that branch. A synchronized group with no
access or region that reads or writes the group is an error.

### Entry and the next iteration use one semaphore

A token passed through `for iter_args` must come from the same semaphore on
entry and on the next iteration.

Using the two-level example above, the acquire created from parent edge `p1`
returns the first-iteration token. The acquire created from child edge `c5`
returns later-iteration tokens after `c4` is removed:

```text
acquire site                         reduced edges       semaphore   count
before the inner for                 p1                  READY       1
before EXIT inner(i) {2}             c5                  READY       1
before R m0 {1}                      c1                  S1          1
before W m0 {1}                      c2                  S2          1
before R m0 {0}                      c3                  S3          1
before EXIT outer(i) {3}             p2                  S4          1
```

```text
%first = acquire READY {2}       // parent p1
for iter_args(%token = %first) {
  %view = semaphore.buffer READY, %token
  ... use %view ...
  %next = acquire READY {2}      // child c5
  yield %next
}
```

When the next-iteration acquire has count 2 and entry has one `[none]` edge, the
single entry release uses `arrive_count=2`:

```text
release READY, %outer [none] arrive_count=2
%first = acquire READY pending_count=2

... inner body ...

release READY, %reader1 [none] arrive_count=1
release READY, %reader2 [none] arrive_count=1
%next = acquire READY pending_count=2
```

Increasing `arrive_count` is allowed only when the entry has exactly one
`[none]` release. Other differences are errors unless restoring removed
releases makes the two pending counts equal.

## Tokens through `for` and `if`

After semaphore placement, a region may receive a token at its entry and
produce a token on each exit path. `RegionFlow` records only what emission
needs:

```text
owner          owner of the token at the region boundary
exits          selected acquire or child region returning a token on each path;
               null means that path returns the input token unchanged
concreteSema   semaphore used when the result cannot use the input semaphore
```

Region results are planned from inner regions to outer regions. The parent
treats a finished child as one node that may return a token; it does not
inspect the child operations again.

For a loop, the pass chooses between two IR forms:

```text
carry the token
  %result = for iter_args(%token = %entry) { ... yield %next }

acquire at the first buffer use
  for { %token = acquire S; ... }
```

A final nested loop can also provide the needed token, so an outer loop may
need no token result of its own.

### Region results

`summarizeRegionFlow` selects the acquire or child whose token each path
returns. Every path that returns a token must return the same owner. A path
with no such node returns the input token when the region boundary has one
owner.

EMIT-IR selects one semaphore for the region result. A loop keeps its input
semaphore. An `if` normally does the same; when the input is an unpartitioned
entry acquire, the result can use a semaphore from a path that returns a
token. Without a usable input semaphore, the result uses `concreteSema`. This
allows:

- both branches to return tokens from acquires;
- one branch to return a token from an acquire while the other returns the
  input token; and
- an `if` without `else` to return the input token on its implicit path.

A path cannot return the input token when no input token exists.
`pruneDeadIfFlows` removes an `if` token result when no buffer access,
release, or child region with a token result appears before the next acquire,
provided the enclosing region does not retain that result.

A parent can use a child's returned token only when the child enters and
returns the same owner and no later acquire, release, group access, or child
region appears on any path.

### Example: an `if` returns one owner's token

`test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token` has a buffer owned by `{1}` around an
`if`. The then path transfers the token to `{0}` for the read and then returns
a token owned by `{1}`; the else path does nothing.

Input shape:

```text
%in = token owned by {1}
W m0 [%in] {1}
%out = if %cond {
  R m0 {0}
  yield token returned to {1}
} else {
  yield %in
}
release ..., %out {1}
```

The semaphore operations selected by SYNC-DAG are shown below. EMIT-IR can
subsequently split this shape into scheduler-safe release, body, and acquire
conditionals, as shown in
[Scheduler-safe conditional boundaries](emit-ir.md#scheduler-safe-conditional-boundaries).

```text
%out = if %cond -> token {
  release TO_READER, %in {1}
  %read = acquire TO_READER {0}
  %view = semaphore.buffer TO_READER, %read
  R m0 [%view] {0}
  release BACK, %read {0}
  %returned = acquire BACK {1}
  yield %returned
} else {
  yield %in
}
release NEXT, %out {1}
```

The region boundary owner is `{1}` on both paths. The then path returns the
token from the final acquire; the else path returns the input token.

### Moving an acquire to its first use

`planLoop` can remove a token from the loop operands and results when the
final token of iteration `i` is needed first at one buffer access in
iteration `i+1`. The acquire moves to that first access.

`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use` begins with this
loop-carried token form:

```text
%entry = acquire EMPTY root
for iter_args(%token = %entry) {
  W m0 [%token] {0}
  release FULL0, %token {0}

  %r = acquire FULL0 {1}
  R m0 [%r] {1}
  release FULL1, %r {1}

  %w = acquire FULL1 {2}
  W m0 [%w] {2}
  release EMPTY, %w {2}

  %next = acquire EMPTY {0}
  yield %next
}
```

Nothing uses `%next` after its acquire. The next operation that needs it is
the following iteration's `{0}` write, so the loop becomes:

```text
for {
  %token = acquire EMPTY {0}
  %view0 = semaphore.buffer EMPTY, %token
  W m0 [%view0] {0}
  release FULL0, %token {0}

  %r = acquire FULL0 {1}
  %view1 = semaphore.buffer FULL0, %r
  R m0 [%view1] {1}
  release FULL1, %r {1}

  %w = acquire FULL1 {2}
  %view2 = semaphore.buffer FULL1, %w
  W m0 [%view2] {2}
  release EMPTY, %w {2}
}
```

`EMPTY` is initially released, so the in-body acquire succeeds on iteration
zero. Every iteration releases it for the next one.

The move requires an input token, a returned token, and one first direct
buffer access before the next acquire. In particular:

- the search reaches a tagged WS `for` before crossing an enclosing `if`;
- the final node is an acquire or a child that returns the boundary owner;
- no later read, write, acquire, release, or child region touches the group;
- the incoming token has the same owner and remains available;
- no release occurs before the first direct access, and the required release
  count is one for a direct next-iteration acquire or zero when a nested child
  returns the next token;
- every member has explicit `buffer.copy = 1` when a buffer access before the
  loop retains the token;
- a TMEM allocation with a source operand blocks the move unless that loop
  itself has a WS tag; and
- any token needed after the loop has a compatible stage.

When these conditions do not hold, the loop keeps its token.

### When the loop keeps a token

`test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout` writes in `{0}`, reads in `{1}`
and `{2}`, then writes again in `{0}`.

```text
%entry = acquire EMPTY pending_count=2 root
for iter_args(%token = %entry) {
  W buf [%token] {0}
  release TO_R1, %token {0}
  release TO_R2, %token {0}

  %r1 = acquire TO_R1 {1}
  R buf [%r1] {1}
  release EMPTY, %r1 {1}

  %r2 = acquire TO_R2 {2}
  R buf [%r2] {2}
  release EMPTY, %r2 {2}

  %next = acquire EMPTY pending_count=2 {0}
  W buf [%next] {0}
  yield %next
}
```

`%next` protects both the final write in iteration `i` and the first write in
iteration `i+1`. Moving its acquire to the next first write would split one
owner's buffer use across the boundary and require an extra release/acquire
pair. The loop carries `%next` instead.

```text
emitted token path from iteration i to iteration i+1

                   %next = acquire EMPTY(i) {0}
                                | walk
                                v
                         W final(i) [%next] {0}
                                | walk
                                v
                         scf.yield %next
                                | loop backedge
                                v
                    %token = iter_arg(i+1)
                                | walk
                                v
                       W first(i+1) [%token] {0}
```

`EMPTY` has `pending_count=2` because owners `{1}` and `{2}` each release it.
The region boundary adds no semaphore.

### A use after the loop

A token needed after the loop does not always force a loop-carried token. The
pass can acquire the final released semaphore once after the loop:

```text
%entry = acquire READY {1}
W buf [%entry] {1}
release TO_READER, %entry {1}

for {
  %read = acquire TO_READER {2}
  R buf [%read] {2}
  release TO_WRITER, %read {2}

  %write = acquire TO_WRITER {1}
  W buf [%write] {1}
  release TO_READER, %write {1}
}

%final = acquire TO_READER {2}
release AFTER_LOOP, %final {2}
%consumer = acquire AFTER_LOOP {3}
R buf [%consumer] {3}
```

For a non-empty loop, `%final` waits for the final iteration. For a zero-trip
loop, it waits for the release before the loop. `postLoopAcquire` marks this
acquire inserted after the loop so schedule finalization uses the correct
owner boundary.

If the last child returns the loop token and that token is also used after the
loop, the token remains in the child result and the loop returns it.

### Nested loops

Inner loops are analyzed before outer loops. In
`test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop`, the inner loop acquires at its first MMA. The outer
loop does not use that token, so it needs no token argument or result:

```text
outer for {
  inner for {
    %mma = acquire MMA_READY {1}
    %acc = semaphore.buffer MMA_READY, %mma
    W acc [%acc] {1}
    release ACC_FULL, %mma [tc5mma] {1}

    %read = acquire ACC_FULL {0}
    %view = semaphore.buffer ACC_FULL, %read
    R acc [%view] {0}
    release MMA_READY, %read {0}
  }
}
```

`@nested_ws_inner_loop_parent_continuation` adds an outer read after the inner
loop. The inner loop still acquires at its first MMA. An acquire and release
after the inner loop provide a token for the outer read. A final acquire and
release in the outer body provide the token needed by the next inner
iteration. Neither loop needs a token argument or result:

```text
%outer_entry = acquire OUTER_EMPTY root

outer for {
  inner for {
    %mma = acquire LOCAL_EMPTY {1}
    ... MMA {1}, read {0} ...
    release LOCAL_EMPTY, %read {0}
  }

  %bridge = acquire LOCAL_EMPTY {1}
  release OUTER_FULL, %bridge [tc5mma] {1}
  %outer_read = acquire OUTER_FULL {0}
  %outer_view = semaphore.buffer OUTER_FULL, %outer_read
  R acc [%outer_view] {0}
  release OUTER_EMPTY, %outer_read {0}

  %tail = acquire OUTER_EMPTY {1}
  release LOCAL_EMPTY, %tail {1}
}
```

The root acquire provides the token for the first outer iteration without
becoming a loop argument. The outer analysis uses the token and completion
recorded for the inner region after its body is analyzed.

### Branch completion must agree

Both paths can return a token with the same owner while the latest operation
by that owner has a different schedule on each path. `completionAfterChain`
records whether a path keeps the schedule from before the region or uses the
schedule of a later operation. For a later operation it records `loop.stage`
and `loop.cluster`.

`test/NVWS/insert_semas_nested_carrier.mlir`
`@branch_completion_requires_carrier` has this inner loop:

```text
%result = for iter_args(%ready = %pre_loop) {
  %first = MMA acc[%ready] {1}              stage 0, cluster 1

  %branch = if %cond -> token {
    %second = MMA acc[%first] {1}           stage 1, cluster 2
    release BRANCH_FULL, %ready [tc5mma]
    %read = acquire BRANCH_FULL {0}         stage 1, cluster 3
    R acc [%read] {0}
    release BRANCH_BACK, %read {0}
    %returned = acquire BRANCH_BACK {1}     stage 1, cluster 2
    yield %returned
  } else {
    yield %ready                            incoming stage-0 completion
  }

  release FINAL_FULL, %branch [tc5mma] {1}
  %final = acquire FINAL_FULL {0}
  R acc [%final] {0}
  release READY, %final {0}
  %next = acquire READY {1}
  yield %next
}
```

The then path ends after an owner-`{1}` operation at stage 1. The else path
keeps the incoming stage-0 schedule. Because those schedules differ, the loop
keeps `%ready` as an iter-arg and yields `%next`. An absent `else` also keeps
the incoming schedule. A loop that executes zero times must account for both
the incoming schedule and the body result.

## Backing copies

`computeBackingCopies` chooses physical buffer copies. A synchronized group
with explicit `buffer.copy` uses that value. Otherwise it starts with one
copy.

A synchronized TMEM accumulator can use two copies when every MMA directly
inside the loop satisfies these checks:

- the loop does not read, modify, and write back an accumulator value;
- the MMA and loop support multiple accumulator copies;
- the enclosing WS loop does not disable them;
- two copies fit in the available TMEM blocks; and
- no scaled MMA uses block N of 256.

When `use-meta-partitioner` is set, the pass does not add this automatic TMEM
copy. An inconsistent or non-positive explicit `buffer.copy` in one group is
an error.

Semaphore copies are computed separately by `computeSemaphoreCopies`.
Usually they equal buffer copies. For a local buffer with no explicit
`buffer.copy`, a release after a TMA load uses at least the number of
semaphore stages requested by `LowerSemaphore`:

```text
numSemaphoreCopies = max(1, lowerSemaphoreNumStages)
```

This does not change the buffer copy count. Schedule and stage analysis use
the semaphore copy count that lowering will create.

For example, `@root_entry_accumulator_adopts_without_semaphore_handoff` in
`test/NVWS/insert_semas_root_entry_tmem.mlir` has one MMA satisfying the
checks above and enough TMEM for two accumulator copies.

By contrast, the `buffer.id = 102` group in
`test/NVWS/insert_semas.mlir` `@local_release_after_mma` keeps one buffer copy,
while its semaphore uses the lowering stage count because its release follows
a descriptor load:

```text
W m0  nvws.descriptor_load {0}
release FULL [tma_load] {0}

numCopies = 1
numSemaphoreCopies = max(1, lowerSemaphoreNumStages)
```

## Pipeline schedule

InsertSemas receives loops whose existing operations already have
`loop.stage` and `loop.cluster`. It assigns schedules to new acquire and
release nodes. EMIT-IR later copies each access schedule to the semaphore
buffer that serves it. For release/acquire relationships that execute in the
same expanded loop body, schedule finalization increases `loop.cluster` as
needed so the release executes first. It does not change `loop.stage`.

### Minimal pipeline model

`loop.stage` determines which logical iterations share an expanded loop body:

```text
before expansion

iteration i:       W(i)   stage 0  ...  R(i)   stage 1
iteration i+1:     W(i+1) stage 0  ...  R(i+1) stage 1

one expanded body

                   W(i+1) stage 0
                   R(i)   stage 1
```

Within one stage, lower `loop.cluster` executes first. Block order breaks ties
inside one cluster. A release/acquire pair between iterations may therefore
connect operations from different source iterations that execute in the same
expanded loop body.

For release/acquire relationships created from synchronization edges, and for
additional relationships created while moving tokens through regions, the
pass records a `ProtocolArc`. A dedicated entry acquire and its final release
need no `ProtocolArc`.

Each recorded relation contains:

```text
release    generated release node
acquire    generated acquire node
producer   source access or region
consumer   destination access or region
wait       acquire used when checking release-before-acquire schedule order
```

`producer` and `consumer` identify the source and destination access or
region. The record remains available when token placement moves an acquire.

### Example: one-copy synchronization between iterations

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` uses `EMPTY` to protect the next write and `FULL` to
protect the following read. `(s,c)` below is `(loop.stage, loop.cluster)`.

Copying schedules only from adjacent accesses gives:

```text
final-read(i)      owner {1}  (1,2)
release EMPTY(i)   owner {1}  (1,2)
acquire EMPTY(i+1) owner {3}  (0,1)
W(i+1)             owner {3}  (0,1)
release FULL(i+1)  owner {3}  (0,1)
acquire FULL(i+1)  owner {1}  (0,1)
first-read(i+1)    owner {1}  (0,1)
```

Grouped by owner in expanded execution order, both owners block before they
can execute the release needed by the other:

```text
wrong semaphore schedule

owner {3}
  acquire EMPTY(i+1) (0,1)   waits for release EMPTY(i)
  W(i+1)             (0,1)
  release FULL(i+1)  (0,1)

owner {1}
  acquire FULL(i+1)  (0,1)   waits for release FULL(i+1)
  first-read(i+1)    (0,1)
  final-read(i)      (1,2)
  release EMPTY(i)   (1,2)
```

There is one physical copy, so `final-read(i)` must precede `W(i+1)`.
Schedule finalization moves the stage-0 operations from cluster 1 to cluster
3:

```text
correct semaphore schedule

owner {1}
  final-read(i)      (1,2)
  release EMPTY(i)   (1,2)

owner {3}
  acquire EMPTY(i+1) (0,3)
  W(i+1)             (0,3)
  release FULL(i+1)  (0,3)

owner {1}
  acquire FULL(i+1)  (0,3)
  first-read(i+1)    (0,3)
```

`release EMPTY(i)` can now run and unblock `{3}`. Owner `{3}` writes and
releases `FULL`, which unblocks `{1}`.

The required loop distance depends on physical copy reuse.
`computeLoopCarriedDistance` follows the ordered reads and writes to determine
when a physical copy is reused:

```text
one copy:  W(i+1) first reuses the copy released in i      distance 1
two copies: W(i+2) first reuses that copy                  distance 2
```

With two copies, the copy released in iteration `i` is not reused until
iteration `i+2`, so no cluster change is needed.

### Finalizing one release/acquire pair

Consider the `EMPTY` release and acquire before their schedules are assigned:

```text
final-read(i)      owner {1}  (1,2)
release EMPTY(i)   owner {1}  (?,?)
acquire EMPTY(i+1) owner {3}  (?,?)
W(i+1)             owner {3}  (0,1)
```

The write/read analysis finds distance 1. After `legalizeLoopSchedule` moves
the destination operations to cluster 3, the release uses the source
completion schedule and the acquire uses the destination schedule:

```text
final-read(i)      owner {1}  (1,2)
release EMPTY(i)   owner {1}  (1,2)
acquire EMPTY(i+1) owner {3}  (0,3)
W(i+1)             owner {3}  (0,3)
```

For an async producer, the release copies the schedule of its physical
completion, not necessarily the access operation itself. A semaphore buffer
copies the schedule of the access it serves.

Owners execute independently. Let `offset[P]` be the whole-iteration delay
of owner P. A release by P at stage `before`, followed at loop distance
`distance` by an acquire owned by Q at stage `after`, requires:

```text
offset[Q] >= offset[P] + before - after - distance
```

`solveOwnerScheduleConstraints` solves all release/acquire relationships in
one scheduled loop together. The cycle total is the sum of
`before - after - distance` for the relationships around one owner cycle:

```text
cycle total < 0    the combined stage and loop-distance terms provide separation
cycle total = 0    the operations execute in one expanded loop body;
                   clusters must place the required releases first
cycle total > 0    every owner would need to run later than itself; error
```

A negative cycle is feasible because the combined stage and loop-distance
terms separate the operations. An individual zero-delay release/acquire pair
on that cycle can still need cluster ordering.

A positive delay on one edge is legal when the reverse path has enough
negative delay. For example, a `+1` edge and `-3` return edge form a legal
`-2` cycle. `test/NVWS/insert_semas_recurrence_owner_cycle.mlir` exercises
that shape.

`legalizeLoopSchedule` collects release/acquire orderings that must hold in
one expanded loop body, together with same-stage SSA orderings, and increases
clusters to satisfy them. A cycle in those required orderings is an error.
Stage values remain unchanged.

An acquire left at loop `EXIT` has no direct destination access in the same
region. It is placed after the last operation of its owner at that stage:

```text
owner {3}, stage 0

W main(i)             cluster 1
W other(i)            cluster 4
acquire EMPTY(i+1)    cluster 4
EXIT
```

Placing the acquire at cluster 1 could block owner `{3}` before `W other`.
Within cluster 4, block order keeps `W other` first.

### Moving an acquire updates its schedule relation

`planLoop` may move an acquire for the next iteration from the end of a loop
to the first buffer use. A release/acquire scheduling relation that still
names the moved acquire may no longer be valid at its new position.

`fixupProtocolArcs` handles every scheduling relation that waited on the moved
acquire:

```text
moved acquire remains in the same node sequence as the release, and either
the release precedes it or the semaphore is an entry semaphore
  keep wait = moved acquire

same-semaphore post-loop acquire follows the release in the same node sequence
  set wait = post-loop acquire

neither relation holds
  clear wait; this relation no longer adds a release-before-acquire schedule rule
```

`test/NVWS/insert_semas_nested_carrier.mlir`
`@scheduled_relocated_acquire_boundaries` has:

```text
inside the inner loop

%ready = acquire MMA_READY {1}      stage 0, cluster 1
MMA acc[%ready] {1}                 stage 0, cluster 1
release ACC_FULL, %ready [tc5mma]   stage 0, cluster 1

%full = acquire ACC_FULL {0}        stage 1, cluster 2
R acc [%full] {0}                   stage 1, cluster 2
release MMA_READY, %full {0}        stage 1, cluster 2
```

The acquire of `MMA_READY` is at the first MMA. The stage-1 release provides
the token for the next iteration. It must not constrain the moved acquire in
the current expanded loop body. Otherwise a false relation

```text
R acc(i) at (1,2) -> MMA acc(i+1) at (0,1)
```

would move the MMA operations to a later cluster even though the acquire is
already before the first MMA. The checked output keeps the MMA acquire,
buffer, MMA, and release at `(0,1)`.

`ProtocolArc::wait` affects only schedule construction. Clearing a stale wait
does not remove the semaphore release or acquire from IR.

### Post-loop acquires use their owner's boundary

An acquire inserted after a nested loop has no later read or write by the
same owner. `postLoopAcquire` prevents it from copying the schedule of an
unrelated later access.

The same test produces:

```text
last inner owner-{1} schedule
  MMA and release                    (stage 0, cluster 1)

after inner loop
  %bridge = acquire MMA_READY {1}    (stage 0, cluster 1)
  release OUTER_EMPTY, %bridge {1}   (stage 0, cluster 1)

  %outer = acquire OUTER_EMPTY {0}   (stage 0, cluster 4)
  R acc [%outer] {0}                 (stage 0, cluster 4)
```

`%bridge` uses owner `{1}`'s boundary schedule `(0,1)`, not owner `{0}`'s
`(0,4)` schedule. If no later operation provides a schedule,
`scheduleAtOwnerBoundary` uses the greatest `loop.cluster` assigned to any
operation with the same owner and stage.

Root entry acquires remain unscheduled.

### Schedule and stage offset are separate

`loop.stage` and `loop.cluster` determine when an acquire or release executes.
`stageOffset` and `bufferStageOffset` determine which semaphore stage and
buffer copy it uses. EMIT-IR places `bufferStageOffset` on the
`semaphore.buffer` created for an access.

Schedule finalization uses applicable `ProtocolArc` records whose release and
acquire still define a loop scheduling relationship. It uses loop distance
and owner order to decide whether a cluster must move. The result keeps the
release at the source completion and the acquire at the destination, but may
delay the destination cluster so the release can execute first.

Stage-offset assignment follows the ordered reads and writes to determine
which buffer copy or semaphore stage is required. For circular buffers it
records an offset on each read or write and on its corresponding acquire and
release. For non-circular aliases it records offsets directly on acquires and
releases. An offset does not move an operation; a cluster change does not
select another copy.

A release and acquire between iterations can be several iterations apart.
Their semaphore stage wraps within the semaphore copy count, while
`bufferStageOffset` selects a buffer copy. Loop-distance analysis determines
the iteration relationship; the offsets select the required stages.

`semaphore.buffer` receives the schedule of the access it serves. When an
access has a `bufferStageOffset`, EMIT-IR places that offset on its
`semaphore.buffer`. `AssignStagePhase` later uses it to select the physical
buffer copy. It does not add synchronization.

## Explicit buffer-stage offsets

`loop.stage` and `loop.cluster` place operations in the software pipeline. A
release or acquire `stageOffset` is a signed offset that `AssignStagePhase`
(ASP) adds to the current stage and wraps within the number of semaphore
copies. For a buffer with explicit `buffer.copy`, or for a circular buffer,
that count is `buffer.copy`. A non-circular alias can instead use
`numSemaphoreCopies`. An access `bufferStageOffset` is applied to its emitted
buffer in the same way:

```text
0     current copy
-1    preceding copy
+1    following copy
```

`finalizeSyncSchedule` scans reads and writes in order and assigns a write
number that is not reduced modulo the copy count. It remembers the latest
write number for each group. Every circular write advances the number. For a
non-circular group, a write advances it only when an acquire directly inside
the loop provides its token. A write that does not advance uses the current
number; a read uses the number recorded by the latest write.

```text
stage offset = required write number - current write number
```

`AssignStagePhase` wraps the result within the applicable copy count. The same
ordered read/write analysis handles circular groups and non-circular aliases;
their representation and attachment points differ.

### Circular groups

Circular members with one physical `buffer.id` are separate groups.
They must agree on type and `buffer.copy`, have unique valid `buffer.start`
values, and be written in the order required by those `buffer.start` values.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` uses K and V in a two-copy circular buffer
with starts 0 and 1:

```text
event       current write number   required write number   offset
store K     -1 -> 0                K = 0                   0
store V      0 -> 1                V = 1                   0
load K       1                     K = 0                  -1
load V       1                     V = 1                   0
```

K and V share one entry semaphore, `EMPTY`, and one non-entry semaphore,
`FULL`. `emitPhysicalIR` creates those two semaphores once for the circular
buffer. Each group places its own offsets on the acquire, `semaphore.buffer`,
and release operations that use those semaphores.

K writes copy 0 and V then writes copy 1, so K's read selects the preceding
copy. The acquire of `FULL` and the release of `EMPTY` receive offset `-1`:

```text
K semaphore operations

acquire EMPTY {1}               offset  0
W K {1}
release FULL {1}                offset  0
acquire FULL {2}                offset -1
R K {2}
release EMPTY {2}               offset -1
```

V stays on the current copy:

```text
V semaphore operations

acquire EMPTY {1}               offset 0
W V {1}
release FULL {1}                offset 0
acquire FULL {2}                offset 0
R V {2}
release EMPTY {2}               offset 0
```

The shared `EMPTY` semaphore is initially released so iteration zero's first
acquire succeeds and returns a token.

### Non-circular aliases

Non-circular members in one group use the same ordered read/write analysis.
An explicit `buffer.copy` makes every member with that `buffer.id` use the
same set of physical copies. Without an explicit copy count, this applies only
to exact aliases whose semaphore copy count exceeds their buffer copy count.
SMEM and TMEM use the same analysis.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@fused_alias_depth_two` has the same SMEM structure as a split dV epilogue
in backward attention: two allocation names use one two-copy backing.

```mlir
%dv0 = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}
%dv1 = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}

scf.for {
  ttg.local_store %v0, %dv0 {partition = 4}
  %r0 = ttg.local_load %dv0 {partition = 2}
  tt.descriptor_store ..., %r0 {partition = 2}

  ttg.local_store %v1, %dv1 {partition = 4}
  %r1 = ttg.local_load %dv1 {partition = 2}
  tt.descriptor_store ..., %r1 {partition = 2}
}
```

Both members read or write the same piece, so the access rules produce one
node sequence. The diagram shows semaphore operations after edge reduction
and merging, across the boundary from iteration `i` to iteration `i+1`.

```text
semaphore DAG across two iterations

                         acquire ENTRY(i) {4}
                                  | walk
                                  v
                            W dv0(i) {4}
                                  | walk
                                  v
                         release FULL0(i) {4}
                                  | FULL0
                                  v
                         acquire FULL0(i) {2}
                                  | walk
                                  v
                            R dv0(i) {2}
                                  | walk
                                  v
                      release DV1_READY(i) {2}
                                  | DV1_READY
                                  v
                      acquire DV1_READY(i) {4}
                                  | walk
                                  v
                            W dv1(i) {4}
                                  | walk
                                  v
                         release FULL1(i) {4}
                                  | FULL1
                                  v
                         acquire FULL1(i) {2}
                                  | walk
                                  v
                            R dv1(i) {2}
                                  | walk
                                  v
                         release ENTRY(i) {2}
                                  | ENTRY
                                  v
                       acquire ENTRY(i+1) {4}
                                  | walk
                                  v
                          W dv0(i+1) {4}
```

The emitted shape before ASP is:

```mlir
%base = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}

%entry   = nvws.semaphore.create %base true
%full0   = nvws.semaphore.create %base false
%dv1_ready = nvws.semaphore.create %base false
%full1   = nvws.semaphore.create %base false

scf.for {
  %t0 = nvws.semaphore.acquire %entry[0] {partition = 4}
  %b0 = nvws.semaphore.buffer %entry[0], %t0
  ttg.local_store %v0, %b0 {partition = 4}
  nvws.semaphore.release %full0[0], %t0 {partition = 4}

  %t1 = nvws.semaphore.acquire %full0[0] {partition = 2}
  %b1 = nvws.semaphore.buffer %full0[0], %t1
  %r0 = ttg.local_load %b1 {partition = 2}
  tt.descriptor_store ..., %r0 {partition = 2}
  nvws.semaphore.release %dv1_ready[1], %t1 {partition = 2}

  %t2 = nvws.semaphore.acquire %dv1_ready[0] {partition = 4}
  %b2 = nvws.semaphore.buffer %dv1_ready[0], %t2
  ttg.local_store %v1, %b2 {partition = 4}
  nvws.semaphore.release %full1[0], %t2 {partition = 4}

  %t3 = nvws.semaphore.acquire %full1[0] {partition = 2}
  %b3 = nvws.semaphore.buffer %full1[0], %t3
  %r1 = ttg.local_load %b3 {partition = 2}
  tt.descriptor_store ..., %r1 {partition = 2}
  nvws.semaphore.release %entry[1], %t3 {partition = 2}
}
```

The bracketed numbers are stage offsets, not final stage numbers. If the first
write/read pair uses physical stage `s`, the second pair uses `(s+1) mod 2`.

```text
release/acquire pairs that keep the same copy
  W dv0 at s     -> R dv0 at s
  W dv1 at s+1   -> R dv1 at s+1

release/acquire pairs that advance to the next copy
  R dv0 at s     -> W dv1 at s+1
  R dv1 at s+1   -> W dv0(i+1) at s
```

The two releases that advance to the next copy receive offset `+1`. Without
it, `%dv1_ready` would
release stage `s` while its acquire waits on stage `s+1`; the initially false
semaphore would never satisfy that acquire. The release/acquire pair between
iterations requires the same adjustment when the copy index wraps.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@tmem_fused_alias_depth_two` uses the same ordered read/write analysis for two
non-circular TMEM aliases with explicit `buffer.copy = 2`.

## Build order and code map

`buildSyncDag` processes one group in this order:

```text
ChainWalker                         add all required synchronization edges
computeBackingCopies               physical backing copies
buildEdgesAndSemas                 reduce and merge edges; group by destination;
                                   create semaphores, releases, and acquires
insertEntryAcquires                insert the acquire for the first token
buildRegionFlows                   summarize path results
planRegionFlows / planLoop         carry tokens or move acquires
pruneDeadIfFlows                   remove unused if token results
computeRequiredParts               record partitions needed by regions
computeSemaphoreCopies             semaphore-stage copies
```

After all groups are built, `finalizeSyncSchedule` assigns stage offsets,
adjusts loop clusters, and assigns schedules to acquire and release nodes.

Current implementation map in
[`InsertSemasSyncDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp):

- Access walk: `ActiveUse`, `VersionSource`, `PieceState`, `Tokens`,
  `applyTouch`, `raiseForeignUseEdges`, and `ChainWalker`.
- Edge reduction: `KnownOrder`, `reduceStraightEdges`,
  `reduceLoopCloses`, and `reduceEdges`.
- Semaphore construction: `buildEdgesAndSemas`, `arrivalContribution`,
  `reachesForward`, and `insertEntryAcquires`.
- Region tokens: `summarizeRegionFlow`, `buildRegionFlows`, `findFeed`,
  `matchDemand`, `planLoop`, `planRegionFlows`, `fixupProtocolArcs`, and
  `pruneDeadIfFlows`.
- Copies: `computeBackingCopies` and `computeSemaphoreCopies`.
- Stage offsets: `replaySlots`, `assignCircularStageOffsets`,
  `assignAliasedHandoffStageOffsets`, and `computeLoopCarriedDistance`.
- Pipeline placement: `solveOwnerScheduleConstraints`,
  `legalizeLoopSchedule`, `scheduleAtOwnerBoundary`, `assignSyncSchedules`,
  and `finalizeSyncSchedule`.
- Shared data types: `RegionFlow`, `CompletionSummary`, `ProtocolArc`, `Node`,
  `Sema`, and `GroupDag` in
  [`InsertSemas.h`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h).
