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
  - [Example: the outer owner starts the inner loop](#example-the-outer-owner-starts-the-inner-loop)
- [Reducing synchronization edges](#reducing-synchronization-edges)
  - [Example: an edge within one region is redundant](#example-an-edge-within-one-region-is-redundant)
  - [Example: an edge to `EXIT` is redundant](#example-an-edge-to-exit-is-redundant)
  - [Edges from an asynchronous write](#edges-from-an-asynchronous-write)
  - [A surviving release does not move earlier](#a-surviving-release-does-not-move-earlier)
- [From reduced edges to semaphores](#from-reduced-edges-to-semaphores)
  - [Repeated edges from one owner](#repeated-edges-from-one-owner)
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
- [Backing copies](#backing-copies)
  - [Example: a TMEM accumulator gets two copies](#example-a-tmem-accumulator-gets-two-copies)
  - [Example: a TMA load increases only the semaphore copies](#example-a-tma-load-increases-only-the-semaphore-copies)
- [Pipeline schedule](#pipeline-schedule)
  - [Minimal pipeline model](#minimal-pipeline-model)
  - [Example: one-copy synchronization between iterations](#example-one-copy-synchronization-between-iterations)
  - [Finalizing one release/acquire pair](#finalizing-one-releaseacquire-pair)
  - [Moving an acquire updates its schedule relation](#moving-an-acquire-updates-its-schedule-relation)
  - [Branch completion must agree](#branch-completion-must-agree)
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
    -> merge repeated edges from one source owner
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

When an acquire moves to the first access of the next iteration, the
semaphore arrow runs directly from the release to that acquire. `EXIT` and
the next `ENTER` remain on a separate control-flow spine; the semaphore arrow
does not pass through them.

Names used throughout:

```text
group          allocations analyzed together, ordinarily one buffer.id
backing        the physical SMEM or TMEM allocation used by the group
m0, m1         members: allocation names or views in the group
P0, P1         disjoint pieces of the backing
{0}, {1}       owners: partitions 0 and 1 of the enclosing WS loop
WS             warp-specialized
root           code with no partition owner
source         node that supplies the current value to a new reader
use            latest access to the current value by one owner
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

In the explanatory DAGs, `R m0 [t]` or `W m0 [t]` abbreviates creating a
`semaphore.buffer` with token `t` and using that buffer at the read or write.
Examples that show exact IR spell out the `semaphore.buffer` operation.

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
edge    semaphore    pending_count    initial state
e1      FULL         1                false
e2      EMPTY        1                initially released
```

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

                etok = acquire EMPTY(i) {0}
                               etok |
                                    v
                         W store(i) [etok] {0}
                                  | walk
                                  v
                       release FULL, etok {0}       e1
                                  | FULL
                                  v
                ftok = acquire FULL(i) {1}
                               ftok |
                                    v
                         R load(i) [ftok] {1}
                                  | walk
                                  v
                      release EMPTY, ftok {1}       e2
                                  | EMPTY
                                  v
                next = acquire EMPTY(i+1) {0}
                               next |
                                    v
                       W store(i+1) [next] {0}
```

There is no buffer use after the loop. After the final iteration, no later
acquire consumes its `EMPTY` release. A zero-trip loop executes none of the
shown operations and leaves `EMPTY` initially released.

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
```

```text
edge    semaphore    pending_count    initial state
f1      F1           1                false
f2      F2           1                false
f3, f4  EMPTY        2                initially released
```

For the first iteration, a root acquire consumes the initially released
`EMPTY` state and supplies owner `{0}`'s loop token:

```text
ptok = acquire EMPTY pending_count=2 root       supplies owner {0}
                      ptok |
                           v
                      ENTER(0) {0}
```

```text
semaphore DAG for the same synchronization edges

                                      ENTER(i) {0}
                                      ptok |
                                           v
                                  W alloc(i) [ptok] {0}
                    +----------------------+----------------------+
               walk |                 walk |                 walk |
                    v                      v                      v
    release F1, ptok {0}       R load(i) [ptok] {0}    release F2, ptok {0}
                 F1 |                 walk |                      | F2
                    v                      v                      v
   r1tok = acquire F1 {1}                  |          r2tok = acquire F2 {2}
              r1tok |                      |                r2tok |
                    v                      |                      v
      R load(i) [r1tok] {1}                |         R load(i) [r2tok] {2}
               walk |                      |                 walk |
                    v                      |                      v
  release EMPTY, r1tok {1}                 |     release EMPTY, r2tok {2}
              EMPTY |                      |                      | EMPTY
                    +----------------------+----------------------+
                                           v
               next = acquire EMPTY(i+1) pending_count=2 {0}
                                      next |
                                           v
                                      EXIT(i) {0}
                                      next | next iteration
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

After the final iteration, the loop returns its final `next` token. For a
zero-trip loop, it returns the root-acquired `ptok` unchanged.

### Example: disjoint pieces stay independent

This conceptual example uses `m0` for the first half of a buffer, `m1` for
the second half, and `m2` for the whole buffer:

```text
members:    m0[0,128)   m1[128,256)   m2[0,256)
pieces:     P0=[0,128){m0,m2}   P1=[128,256){m1,m2}
```

This example starts from the edges that remain after removal and merging
because its purpose is to show that P0 and P1 stay independent. The next
section derives edge removal from the complete initial set. The remaining
nodes and edges here are:

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

Before merging, P0 and P1 each produce an edge from `W m2 {0}` to
`R m2 {1}`. Their endpoints and owners match, so `e1` represents both.
Ordering-redundant direct edges are also absent from this view; their removal
is explained in [Reducing synchronization edges](#reducing-synchronization-edges).
The remaining synchronization-edge DAG is:

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
edge    semaphore       pending_count    initial state
e1      FULL_BOTH       1                false
e2      LEFT_READY      1                false
e3      LEFT_FULL       1                false
e4      RIGHT_READY     1                false
e5      RIGHT_FULL      1                false
e6      EMPTY           1                initially released
```

The diagram assumes a surrounding loop that must keep its boundary token, so
the `EMPTY` acquire remains before `EXIT`. In an eligible loop with no such
use, token planning may instead move that acquire to the first access of the
next iteration; the synchronization edges and semaphore mapping do not
change.

The carried-token form obtains its first token outside the loop:

```text
btok = acquire EMPTY pending_count=1 root       supplies owner {0}
                      btok |
                           v
                      ENTER(0) {0}
```

```text
semaphore DAG

                                      ENTER(i) {0}
                                      btok |
                                           v
                                W m2(i) [btok] {0}
                                           | walk
                                           v
                             release FULL_BOTH, btok {0}
                                  FULL_BOTH |
                                            v
                               both = acquire FULL_BOTH {1}
                                       both |
                                            v
                                 R m2(i) [both] {1}
                       +--------------------+--------------------+
                  walk |                                         | walk
                       v                                         v
       release LEFT_READY, both {1}            release RIGHT_READY, both {1}
            LEFT_READY |                             RIGHT_READY |
                       v                                         v
          left = acquire LEFT_READY {2}             right = acquire RIGHT_READY {4}
                  left |                                         | right
                       v                                         v
             W m0(i) [left] {2}                         W m1(i) [right] {4}
                  walk |                                         | walk
                       v                                         v
        release LEFT_FULL, left {2}             release RIGHT_FULL, right {4}
             LEFT_FULL |                              RIGHT_FULL |
                       v                                         v
         lread = acquire LEFT_FULL {3}             rread = acquire RIGHT_FULL {0}
                 lread |                                         | rread
                       v                                         v
            R m0(i) [lread] {3}                       R m1(i) [rread] {0}
                  walk |                                         | walk
                       v                                         |
          release EMPTY, lread {3}                                |
                 EMPTY |                                         |
                       +--------------------+--------------------+
                                            v
                next = acquire EMPTY(i+1) pending_count=1 {0}
                                      next |
                                           v
                                      EXIT(i) {0}
                                      next | next iteration
                                           v
                                    ENTER(i+1) {0}
```

After `R m2`, the P0 and P1 paths have no synchronization edge between them.
P0 releases `EMPTY` because its last reader is owner `{3}`. `R m1` and the
acquire of `EMPTY` both have owner `{0}`, and program order places the read
before the acquire.

After the final iteration, the loop returns `next`. A zero-trip loop returns
the root-acquired `btok` unchanged.

### Nested regions

A nested `for` or `if` contains many reads and writes, but occupies one
position among the surrounding nodes. The node at that position is called
the region summary because it records, for every buffer piece, whether the
child region reads or writes it and which owner appears at the child
boundary. ACCESS-DAG therefore gives SYNC-DAG two views of the region:

```text
parent view   ... -> [one region summary] -> ...
child view           ENTER -> reads and writes -> EXIT
```

For example, `P0:W:{2}` means that the region writes P0 and has boundary owner
`{2}` for that piece. The parent applies the ordinary read/write rules to
this one node. Each child path is analyzed separately between its `ENTER` and
`EXIT` nodes. ACCESS-DAG explains how the pass decides whether a region reads
or writes each piece and how it chooses boundary owners in
[Regions and boundaries](access-dag.md#regions-and-boundaries).

### Example: the same rules at two region levels

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` uses `m0`, which contains the
single buffer piece P0. The function has this high-level shape:

```text
outer for
  W m0 {3}

  inner for
    R m0 {2}
    R m0 {1}
    W m0 {1}
    R m0 {0}
```

The inner loop's first P0 access has owner `{2}`, so `{2}` is its boundary
owner. The inner loop also writes P0, so its parent summary is `P0:W:{2}`.
The parent therefore sees only these four nodes:

```text
DAG node                              synchronization edge ending here
ENTER outer(i) {3}                    none
W m0 {3}                              none
[inner for P0:W:{2}]                  p1: W m0 {3} -> inner for
EXIT outer(i) {3}                     p2: inner for -> EXIT outer(i) {3}
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

The inner loop uses its actual reads and write rather than the summary node:

```text
DAG node                 synchronization edge ending here
ENTER inner(i) {2}       none
R m0 {2}                 none
R m0 {1}                 c1: ENTER inner(i) {2} -> R m0 {1}
W m0 {1}                 c2: R m0 {2} -> W m0 {1}
R m0 {0}                 c3: W m0 {1} -> R m0 {0}
EXIT inner(i) {2}        c4: W m0 {1} -> EXIT inner(i) {2}
                         c5: R m0 {0} -> EXIT inner(i) {2}
```

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

`R m0 {2}` reads with the `ENTER` owner, so it needs no synchronization edge.
The other read of the incoming contents has owner `{1}`, so `c1` starts at
`ENTER`. The later write by `{1}` waits for `{2}`'s read through `c2`; it does
not wait for `{1}`'s read because that read has the same owner. Finally,
`c4` and `c5` make the next inner iteration wait for the write by `{1}` and
the read by `{0}`.

The path through `c3` and `c5` already makes `EXIT` wait for `W m0 {1}`, so
`c4` is removed. The remaining edges use four semaphores: `c1` uses `S1`,
`c2` uses `S2`, `c3` uses `S3`, and `c5` uses `READY`. Parent edge `p1`
releases `READY` for the first inner iteration; after that, `c5` releases it
for the next inner iteration. Every acquire has `pending_count=1`.

```text
edges      semaphore    pending_count
c1         S1           1
c2         S2           1
c3         S3           1
p1, c5     READY        1
```

The diagram begins after at least one inner iteration has completed so it can
show how `READY` crosses `EXIT` and `ENTER`. For the first inner iteration,
`p1` supplies `ptok` before the inner loop, so there is no preceding inner
`EXIT`.

```text
semaphore DAG for the inner loop

                         ptok = acquire READY(i) {2}
                                      ptok |
                                           v
                              EXIT inner(i-1) {2}
                                           | next iteration
                                           v
                               ENTER inner(i) {2}
                         +-----------------+-----------------+
                    walk |                                   | walk
                         v                                   v
             release S1, ptok {2}                  R m0(i) [ptok] {2}
                      S1 |                                   | walk
                         v                                   v
          r1tok = acquire S1 {1}                  release S2, ptok {2}
                   r1tok |                                S2 |
                         v                                   v
            R m0(i) [r1tok] {1}              wtok = acquire S2 {1}
                    walk |                              wtok |
                         +-----------------+-----------------+
                                           v
                                W m0(i) [wtok] {1}
                                           | walk
                                           v
                                release S3, wtok {1}
                                        S3 |
                                           v
                            r0tok = acquire S3 {0}
                                     r0tok |
                                           v
                              R m0(i) [r0tok] {0}
                                           | walk
                                           v
                            release READY, r0tok {0}
                                     READY |
                                           v
                        next = acquire READY(i+1) {2}
                                      next |
                                           v
                                EXIT inner(i) {2}
                                           | next iteration
                                           v
                              ENTER inner(i+1) {2}
```

The parent updates its state only from the summary node; the child reads and
writes do not replace the parent's recorded state. `ENTER` and `EXIT` are DAG
nodes, not acquire or release operations. The requirement that `p1` and `c5`
use the same `READY` semaphore is explained in
[Entry and the next iteration use one semaphore](#entry-and-the-next-iteration-use-one-semaphore).

Three boundary details matter in later sections:

- If buffer contents came from an asynchronous operation with the same owner
  as `ENTER`, its completion kinds remain attached to `ENTER`; otherwise
  `ENTER` uses `[none]`.
  [Edges from an asynchronous write](#edges-from-an-asynchronous-write)
  explains how those completion kinds affect a release.
- After a region, one partition token remains available only when the summary
  has one partition owner.
  [Tokens through `for` and `if`](#tokens-through-for-and-if) explains how
  regions return tokens.
- An ordering from an `if` is usable afterward only when every branch
  establishes it. A loop that may execute zero times cannot establish an
  ordering solely in its body. The later
  [Region results](#region-results) and
  [Branch completion must agree](#branch-completion-must-agree) examples show
  the corresponding token and schedule rules.

#### Both loops together

The complete example below uses `i` for the outer iteration and `j` for the
inner iteration. The box combines the two views without joining their DAGs:
`p1` ends at the summary node and `p2` starts there in the outer DAG, while
`c1`, `c2`, `c3`, and `c5` belong to the child DAG. The diagram shows the
edges that remain after `c4` is removed.

```text
synchronization-edge DAG with both region levels

                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0(i) {3}
                                  | p1
                                  v
 +-------------- [inner-for summary P0:W:{2}] ---------------+
 |                                                           |
 |                    ENTER inner(i,j) {2}                   |
 |                         +---------+---------+             |
 |                    walk |                   | c1          |
 |                         v                   v             |
 |                     R m0 {2}            R m0 {1}          |
 |                      c2 |                   | walk        |
 |                         +---------+---------+             |
 |                                   v                       |
 |                              W m0 {1}                     |
 |                                   | c3                    |
 |                                   v                       |
 |                              R m0 {0}                     |
 |                                   | c5                    |
 |                                   v                       |
 |                          EXIT inner(i,j) {2}              |
 |                                   | next inner iteration  |
 |                                   v                       |
 |                         ENTER inner(i,j+1) {2}            |
 |                                                           |
 +-----------------------------------------------------------+
                                  | p2
                                  v
                         EXIT outer(i) {3}
```

The arrow to `ENTER inner(i,j+1)` shows the case in which the inner loop runs
again. After its final iteration there is no following inner `ENTER`; the
child ends at `EXIT`, the summary completes, and `p2` applies.

The parent edges do not connect directly to the child `ENTER` or `EXIT`.
Their semaphore operations cross those boundaries when tokens are placed.
`READY` implements `p1` and `c5`; `S1`, `S2`, and `S3` implement `c1`,
`c2`, and `c3`; and initially released `EMPTY` implements `p2`. Every
semaphore has `pending_count=1`.

```text
semaphore DAG with both region levels

                         ENTER outer(i) {3}
                                  | walk
                                  v
               otok = acquire EMPTY(i) {3}
                              otok |
                                   v
                         W m0(i) [otok] {3}
                                  | walk
                                  v
                     release READY, otok {3}      p1
                             READY |
                                   v
                 ptok = acquire READY {2}
                              ptok |
                                   v
                      ENTER inner(i,0) {2}
                 +--------------+--------------+
            walk |                             | walk
                 v                             v
        release S1, ptok {2} c1       R m0(i,0) [ptok] {2}
              S1 |                             | walk
                 v                             v
  r1tok = acquire S1 {1}           release S2, ptok {2} c2
           r1tok |                          S2 |
                 v                             v
  R m0(i,0) [r1tok] {1}          wtok = acquire S2 {1}
            walk |                        wtok |
                 +--------------+--------------+
                                v
                     W m0(i,0) [wtok] {1}
                                | walk
                                v
                     release S3, wtok {1}       c3
                             S3 |
                                v
                 r0tok = acquire S3 {0}
                          r0tok |
                                v
                   R m0(i,0) [r0tok] {0}
                                | walk
                                v
                 release READY, r0tok {0}        c5
                          READY |
                                v
              next = acquire READY {2}
                           next |
                                v
                     EXIT inner(i,0) {2}
                       +--------+--------+
      another inner j  |                 | inner loop finishes
                  next |                 | next
                       v                 v
        ENTER inner(i,1) {2}       result = next
                       |                 |
             repeat the body             |
             through its final EXIT      |
                       |                 |
                       v                 |
             result = final next         |
                       +--------+--------+
                                v
                 release EMPTY, result {2}        p2
                    +-----------+-----------+
               walk |                       | EMPTY
                    v                       |
             EXIT outer(i) {3}              |
                    | next outer iteration  |
                    v                       |
            ENTER outer(i+1) {3}            |
               walk |                       |
                    +-----------+-----------+
                                v
           otok2 = acquire EMPTY(i+1) {3}
                           otok2 |
                                 v
                     W m0(i+1) [otok2] {3}
```

`result = next` and `result = final next` only name the token returned by the
inner loop; they are not operations.

For inner `j=0`, `p1` supplies `ptok` directly: there is no
`EXIT inner(i,-1)`. The acquire of `ptok` is outside the inner loop but still
inside outer iteration `i`. For `j>0`, `c5` supplies `next`; that acquire is
before the current inner `EXIT`, and the token passes through `EXIT` into the
next `ENTER`. On the final inner iteration, the final `next` becomes
`result`, which supplies the `p2` release.

`EMPTY` starts released, so outer iteration zero needs no preceding outer
`EXIT`. On later outer iterations, the `p2` release signals `EMPTY`, and its
acquire is placed immediately before the next outer write rather than being
carried by the outer loop. If the inner loop executes zero times, its input
`ptok` is its `result`.

### Example: the outer owner starts the inner loop

`test/NVWS/insert_semas.mlir` `@same_owner_nested` makes the outer write's
owner `{3}` the first inner reader. Its other reader has owner `{2}`, so the
later write by `{1}` must wait for two readers. As before, `m0` contains the
single buffer piece P0:

```text
outer for
  W m0 {3}

  inner for
    R m0 {3}
    R m0 {2}
    W m0 {1}
    R m0 {0}
```

The inner loop's first P0 access has owner `{3}`, so `{3}` is its boundary
owner. The inner loop writes P0, so its parent summary is `P0:W:{3}`. The
parent sees these four nodes:

```text
DAG node                              synchronization edge ending here
ENTER outer(i) {3}                    none
W m0 {3}                              none
[inner for P0:W:{3}]                  none
EXIT outer(i) {3}                     none
```

```text
                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0 {3}
                                  | walk
                                  v
                   [inner-for summary P0:W:{3}]
                                  | walk
                                  v
                         EXIT outer(i) {3}
```

The outer write, the inner boundary, and the outer boundary all have owner
`{3}`, so the parent needs no synchronization edge. In particular, this
example has no counterpart to `p1` or `p2` from the preceding example.

The child still needs synchronization between its four owners. Its complete
edge set is:

```text
DAG node                 synchronization edge ending here
ENTER inner(i) {3}       none
R m0 {3}                 none
R m0 {2}                 c1: ENTER inner(i) {3} -> R m0 {2}
W m0 {1}                 c2: R m0 {3} -> W m0 {1}
                         c3: R m0 {2} -> W m0 {1}
R m0 {0}                 c4: W m0 {1} -> R m0 {0}
EXIT inner(i) {3}        c5: W m0 {1} -> EXIT inner(i) {3}
                         c6: R m0 {0} -> EXIT inner(i) {3}
```

```text
                         ENTER inner(i) {3}
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
                          EXIT inner(i) {3}
```

`R m0 {3}` reads with the `ENTER` owner and needs no synchronization edge.
The other reader has owner `{2}`, so `c1` starts at `ENTER`; the two reads
can proceed independently. The later write has owner `{1}`, so it must wait
separately for both readers through `c2` and `c3`. Edge `c4` makes the final
read wait for that write. Edges `c5` and `c6` make the next inner iteration
wait for the write by `{1}` and the read by `{0}`.

The path through `c4` and `c6` already makes `EXIT` wait for `W m0 {1}`, so
`c5` is removed. The remaining edges become four semaphores:

```text
edge        semaphore    pending_count
c1          S1           1
c2, c3      S2           2
c4          S3           1
c6          READY        1
```

Because `c1` starts at `ENTER`, its `S1` release is immediately after `ENTER`,
before `R m0 {3}`. Both `c2` and `c3` release `S2` after their reads. Its one
acquire before `W m0 {1}` waits for both releases. `READY` is initially
released and also supplies the token for the first outer write.

For inner iteration zero, `itok` is the token already held by owner `{3}`
after the outer write. For later inner iterations, `itok` is `next` from the
preceding inner `EXIT`.

```text
semaphore DAG for the inner loop

                         ENTER inner(i,j) {3}
                                  | walk
                                  v
                     release S1, itok {3} c1
                     +---------------+---------------+
                walk |                               | S1
                     v                               v
       R m0(i,j) [itok] {3}         r2tok = acquire S1 {2}
                walk |                         r2tok |
                     v                               v
  release S2, itok {3} c2            R m0(i,j) [r2tok] {2}
                  S2 |                          walk |
                     |                               v
                     |              release S2, r2tok {2} c3
                     |                            S2 |
                     +---------------+---------------+
                                     v
           wtok = acquire S2 pending_count=2 {1}
                           wtok |
                                v
                     W m0(i,j) [wtok] {1}
                                | walk
                                v
                     release S3, wtok {1}       c4
                             S3 |
                                v
                 r0tok = acquire S3 {0}
                          r0tok |
                                v
                   R m0(i,j) [r0tok] {0}
                                | walk
                                v
                 release READY, r0tok {0}        c6
                          READY |
                                v
              next = acquire READY {3}
                           next |
                                v
                     EXIT inner(i,j) {3}
                                | next inner iteration
                                v
                   ENTER inner(i,j+1) {3}
```

There is no acquire between the outer write and the first inner `ENTER`.
Owner `{3}` uses the outer token for its read and for the releases into `S1`
and `S2`. On later iterations, the acquire of `READY` occurs before the
current `EXIT`, and `next` passes through that `EXIT` into the next `ENTER`.

#### Both loops together when the boundary owner is unchanged

The complete DAG after `c5` is removed keeps the parent and child levels
separate. The parent arrows are all program order; only `c1`, `c2`, `c3`,
`c4`, and `c6` are synchronization edges:

```text
synchronization-edge DAG with both region levels

                         ENTER outer(i) {3}
                                  | walk
                                  v
                              W m0(i) {3}
                                  | walk
                                  v
 +-------------- [inner-for summary P0:W:{3}] ---------------+
 |                                                           |
 |                    ENTER inner(i,j) {3}                   |
 |                         +---------+---------+             |
 |                    walk |                   | c1          |
 |                         v                   v             |
 |                     R m0 {3}            R m0 {2}          |
 |                      c2 |                   | c3          |
 |                         +---------+---------+             |
 |                                   v                       |
 |                              W m0 {1}                     |
 |                                   | c4                    |
 |                                   v                       |
 |                              R m0 {0}                     |
 |                                   | c6                    |
 |                                   v                       |
 |                          EXIT inner(i,j) {3}              |
 |                                   | next inner iteration  |
 |                                   v                       |
 |                         ENTER inner(i,j+1) {3}            |
 |                                                           |
 +-----------------------------------------------------------+
                                  | walk
                                  v
                         EXIT outer(i) {3}
```

The arrow to `ENTER inner(i,j+1)` shows the case in which the inner loop runs
again. After the final inner iteration, the child stops at `EXIT` and the
summary completes. No parent synchronization edge is added on either side
of the summary.

After semaphore and token placement, one `READY` semaphore supplies the
outer write, the next inner iteration, and the next outer iteration. `READY`
starts released, and every semaphore except `S2` has `pending_count=1`:

```text
semaphore DAG with both region levels

                         ENTER outer(i) {3}
                                  | walk
                                  v
               otok = acquire READY {3}
                              otok |
                                   v
                         W m0(i) [otok] {3}
                              same token |
                                         v
                              ENTER inner(i,0) {3}
                                      | walk
                                      v
                         release S1, otok {3} c1
                         +------------+------------+
                    walk |                         | S1
                         v                         v
           R m0(i,0) [otok] {3}   r2tok = acquire S1 {2}
                    walk |                   r2tok |
                         v                         v
      release S2, otok {3} c2       R m0(i,0) [r2tok] {2}
                      S2 |                    walk |
                         |                         v
                         |        release S2, r2tok {2} c3
                         |                      S2 |
                         +------------+------------+
                                      v
                wtok = acquire S2 pending_count=2 {1}
                                wtok |
                                     v
                          W m0(i,0) [wtok] {1}
                                     | walk
                                     v
                          release S3, wtok {1}       c4
                                  S3 |
                                     v
                      r0tok = acquire S3 {0}
                               r0tok |
                                     v
                        R m0(i,0) [r0tok] {0}
                                     | walk
                                     v
                      release READY, r0tok {0}        c6
                               READY |
                                     v
                   next = acquire READY {3}
                                next |
                                     v
                          EXIT inner(i,0) {3}
                            +--------+--------+
           another inner j  |                 | inner loop finishes
                       next |                 | next
                            v                 v
             ENTER inner(i,1) {3}       result = next
                            |                 |
                  repeat the body             |
                  through its final EXIT      |
                            |                 |
                            v                 |
                  result = final next         |
                            +--------+--------+
                                     v
                     release READY, result {3}
                         +-----------+-----------+
                    walk |                       | READY
                         v                       |
                  EXIT outer(i) {3}              |
                         | next outer iteration  |
                         v                       |
                 ENTER outer(i+1) {3}            |
                    walk |                       |
                         +-----------+-----------+
                                     v
                otok2 = acquire READY {3}
                                otok2 |
                                      v
                          W m0(i+1) [otok2] {3}
```

`result = next` and `result = final next` only name the token returned by the
inner loop; they are not operations.

For inner `j=0`, `otok` passes directly from the outer write to the inner
loop; no `EXIT inner(i,-1)` or separate inner acquire exists. For `j>0`, the
preceding inner iteration supplies `next`. The final `next` is the inner-loop
`result`.

The release after the inner loop is not a parent synchronization edge. It
returns owner `{3}`'s token to `READY` so the next outer write can acquire it.
The outer loop therefore does not carry a token. If the inner loop executes
zero times, `otok` is its `result` and is released to `READY` in the same
place.

## Reducing synchronization edges

The walk records every synchronization edge required by the read and write
rules. Some of those edges impose a wait that the other edges already impose.
Removing such an edge is safe only when the remaining edges and program order
still make the same owner wait and still leave an acquire for that owner.

Reduction considers edges in their fixed walk order. An edge that remains can
justify removing a later edge; an edge that has been removed cannot. Edges to
`EXIT` need an additional check because the replacement path may continue in
the next iteration.

### Example: an edge within one region is redundant

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` contains this sequence:

```text
W m0 {0}
R m0 {1}
W m1 {2}
```

The two members overlap:

```text
members:    m0[0,256)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0,m1}   P2=[192,256){m0}
```

Only P1 is relevant to this reduction. Owner `{0}` writes it, owner `{1}`
reads it, and owner `{2}` writes it. The write by `{2}` must initially wait
for the recorded uses by both earlier owners:

```text
DAG node       synchronization edge ending here
W m0 {0}       none
R m0 {1}       e1: W m0 {0} -> R m0 {1}
W m1 {2}       e2: W m0 {0} -> W m1 {2}
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
is unnecessary. The acquire created from `e3` will return the token used by
owner `{2}`.

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

The remaining edges map to two count-1 semaphores:

```text
edge    semaphore    acquire before    pending_count
e1      S0           R m0 {1}          1
e3      S1           W m1 {2}          1
```

Let `t0` be the token already held by owner `{0}`. The semaphore DAG is:

```text
semaphore DAG

                       W m0 [t0] {0}
                                | walk
                                v
                       release S0, t0 {0}       e1
                                | S0
                                v
                         t1 = acquire S0 {1}
                                t1 |
                                v
                       R m0 [t1] {1}
                                | walk
                                v
                       release S1, t1 {1}       e3
                                | S1
                                v
                         t2 = acquire S1 {2}
                                t2 |
                                v
                       W m1 [t2] {2}
```

If `e2` had remained, `W m1 {2}` would have waited for separate releases
from `{0}` and `{1}`. Removing it therefore changes that acquire from count 2
to count 1.

Implementation note: `reduceStraightEdges` performs this reduction for
access nodes in one region. It does not consider root, region, `ENTER`, or
`EXIT` endpoints.

### Example: an edge to `EXIT` is redundant

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` repeats this sequence in a loop:

```text
for
  W m0 {0}
  R m0 {1}
  W m1 {2}
  R m1 {0}
```

The two members partly overlap:

```text
members:    m0[0,128)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,128){m0,m1}   P2=[128,192){m1}
```

The complete edge set for one iteration is:

```text
DAG node     synchronization edge ending here
ENTER(i)     none
W m0 {0}     none
R m0 {1}     l1a: W m0 {0} -> R m0 {1}       P0
              l1b: W m0 {0} -> R m0 {1}       P1
W m1 {2}     l2a: W m0 {0} -> W m1 {2}       P1
              l2b: R m0 {1} -> W m1 {2}       P1
R m1 {0}     l3a: W m1 {2} -> R m1 {0}       P1
              l3b: W m1 {2} -> R m1 {0}       P2
EXIT(i)      c0: R m0 {1} -> EXIT(i)           destination owner {0}, P0
              c1: W m1 {2} -> EXIT(i)           destination owner {0}, P1
              c2: R m1 {0} -> EXIT(i)           destination owner {2}, P2
```

`c1` starts at `W m1`, not `R m0`, because that write replaces P1's contents.
The write is already ordered before `{0}`'s P1 read.

The initial synchronization-edge DAG is:

```text
                              W m0(i) {0}
                         +--------+------------------+
                l1a,l1b |                           | l2a
                         v                           |
                    R m0(i) {1}                      |
                    +----+----+                      |
                 c0 |         | l2b                  |
                    |         v                      |
                    |     W m1(i) {2} <--------------+
                    |      +-----+-----+
                    |   c1 |           | l3a,l3b
                    |      |           v
                    |      |      R m1(i) {0}
                    |      |           | c2
                    +------+-----+-----+
                                 v
                              EXIT(i)
```

The first reduction removes `l1b` and `l3b` because each duplicates an edge
with the same endpoints. It also removes `l2a` because `l1a -> l2b` imposes
the same wait.

To test `c2`, follow the loop boundary to owner `{2}`'s first P2 access in the
next iteration:

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

The path through `l1a` and `l2b` already ends at the next P2 write, where an
acquire will return `{2}`'s token. Therefore `c2` is removed. Edges `c0` and
`c1` end at the first access owner `{0}`, so this loop-boundary reduction
keeps them.

A later check removes `c1`: `l3a` makes `{0}` wait for `{2}`'s P1 write, and
`{0}`'s program order reaches `EXIT`. The final edge set is:

```text
edge    endpoints                         destination owner
l1a     W m0 {0} -> R m0 {1}             {1}
l2b     R m0 {1} -> W m1 {2}             {2}
l3a     W m1 {2} -> R m1 {0}             {0}
c0      R m0(i) {1} -> EXIT(i)            {0}
```

```text
reduced DAG

                         W m0(i) {0}
                              | l1a
                              v
                         R m0(i) {1}
                         +----+----------------+
                    l2b |                     | c0
                         v                     |
                    W m1(i) {2}                |
                         | l3a
                         v
                    R m1(i) {0}
                         | walk
                         +----------+----------+
                                    v
                                  EXIT(i)
                                    | walk
                                    v
                                ENTER(i+1)
```

The four remaining edges become four count-1 semaphores:

```text
edge    semaphore    acquire before                  pending_count
l1a     F01          R m0 {1}                        1
l2b     F12          W m1 {2}                        1
l3a     F20          R m1 {0}                        1
c0      EMPTY        W m0 in the next iteration      1
```

The edge `c0` is stored as an edge to `EXIT`, but its acquire is placed at the
next write that needs owner `{0}`'s token:

```text
semaphore DAG across two iterations

                    t0 = acquire EMPTY(i) {0}
                               t0 |
                                  v
                       W m0(i) [t0] {0}
                                  | walk
                                  v
                       release F01, t0 {0}       l1a
                                F01 |
                                    v
                         t1 = acquire F01 {1}
                                    t1 |
                                       v
                            R m0(i) [t1] {1}
                         +-------------+-------------+
                    walk |                           | walk
                         v                           v
              release F12, t1 {1} l2b    release EMPTY, t1 {1} c0
                         | F12                      EMPTY |
                         v                                |
              t2 = acquire F12 {2}                       |
                         t2 |                             |
                            v                             |
                 W m1(i) [t2] {2}                        |
                            | walk                        |
                            v                             |
                 release F20, t2 {2} l3a                  |
                         F20 |                             |
                             v                            |
                  t0b = acquire F20 {0}                   |
                         t0b |                             |
                              v                           |
                   R m1(i) [t0b] {0}                      |
                              | walk                      |
                              v                           |
                            EXIT(i)                       |
                              | walk                      |
                              v                           |
                          ENTER(i+1)                      |
                              | walk                      |
                              v                           |
              t0next = acquire EMPTY(i+1) {0} <-----------+
                         t0next |
                                v
                     W m0(i+1) [t0next] {0}
```

`EMPTY` starts released for iteration zero. On later iterations, the `c0`
release supplies its next phase.

Implementation note: `reduceLoopCloses` performs the loop-boundary check. It
uses the first read or write of each piece in the next iteration. The later
removal of `c1` happens while the remaining edges are converted to
semaphores.

### Edges from an asynchronous write

An edge from an asynchronous write records the operation that must finish.
Consider this sequence:

```text
W async m0 {0}    completion [tma_load]
R m0 {1}
W m0 {2}
```

The walk records:

```text
DAG node          synchronization edge ending here
W async m0 {0}    none
R m0 {1}          a1: W async m0 {0} -> R m0 {1}  [tma_load]
W m0 {2}          a2: W async m0 {0} -> W m0 {2}  [tma_load]
                   a3: R m0 {1} -> W m0 {2}        [none]
```

```text
initial DAG

                       W async m0 {0}
                    +---------+---------+
     a1 [tma_load]  |                   | a2 [tma_load]
                    v                   |
                 R m0 {1}               |
                  a3 |                   |
                    +---------+---------+
                              v
                           W m0 {2}
```

The early reduction keeps `a2` because it directly records the asynchronous
write whose completion it needs. When semaphores are built, `a2` may still be
removed if both `a1` and `a3` remain: `a1` waits for the TMA load before the
read, and `a3` orders that read before the final write.

```text
reduced DAG after removing a2

                       W async m0 {0}
                              | a1 [tma_load]
                              v
                           R m0 {1}
                              | a3
                              v
                           W m0 {2}
```

```text
edge    semaphore    completion    pending_count
a1      ASYNC_READY  [tma_load]    1
a3      WRITE_READY  [none]        1
```

```text
semaphore DAG

                   W async m0 [t0] {0}
                              | walk
                              v
       release ASYNC_READY, t0 [tma_load] {0}       a1
                    ASYNC_READY |
                                v
              t1 = acquire ASYNC_READY {1}
                                t1 |
                                   v
                        R m0 [t1] {1}
                                   | walk
                                   v
           release WRITE_READY, t1 [none] {1}       a3
                    WRITE_READY |
                                v
              t2 = acquire WRITE_READY {2}
                                t2 |
                                   v
                        W m0 [t2] {2}
```

If either `a1` or `a3` disappears, the direct `a2` release remains. A later
synchronous read is not the asynchronous operation that produced the buffer
contents, so an edge starting at that read does not carry `[tma_load]`.

Implementation note: `EdgeRec::preserve` keeps `a2` through the two early
edge reducers. The later whole-release check may remove it only after
validating the complete remaining path.

### A surviving release does not move earlier

Reduction decides which waits remain. It does not move a surviving release
before a source node that originally contributed to that release. Consider
four accesses to three pieces:

```text
A: W async a {1}      P0            completion [tma_load]
B: W b {1}            P1, P2        follows A
C: R b {0}            P1, P2
D: W all {2}          P0, P1, P2
```

The relevant edges are:

```text
DAG node    synchronization edge ending here
A {1}       none
B {1}       none
C {0}       q1: B {1} -> C {0}                    P1, P2
D {2}       q2: A {1} -> D {2} [tma_load]         P0
             q3: B {1} -> D {2}                    P1, P2
             q4: C {0} -> D {2}                    P1, P2
```

When P1 and P2 produce edges with the same endpoints, the table shows that
repeated pair once.

```text
initial DAG

                    A async {1}
                    +------+----------------------+
               walk |                             | q2 [tma_load]
                    v                             |
                  B {1}                           |
             +------+-------+                     |
          q1 |              | q3                  |
             v              |                     |
           C {0}            |                     |
          q4 |              |                     |
             +-------+------+---------------------+
                     v
                   D {2}
```

The path `q1 -> q4` makes `q3` unnecessary. Edge `q2` remains because it
directly records A's asynchronous completion; the early reducers preserve
such an edge rather than replacing it with a path through a later synchronous
access:

```text
reduced DAG

             A async {1} ----q2 [tma_load]---+
                  | walk                      |
                  v                           |
                B {1}                         |
                  | q1                        |
                  v                           |
                C {0}                         |
                  | q4                        |
                  +-------------+-------------+
                                v
                              D {2}
```

Edges `q2` and `q4` end at the same node and owner, so they share semaphore
`SD`. They have different source owners and therefore contribute two
releases. Edge `q1` uses `SC`:

```text
edge    semaphore    release owner    pending_count
q1      SC           {1}              1
q2,q4   SD           {1}, {0}         2
```

Although `q2` starts at A, owner `{1}`'s `SD` release stays after B because
the removed `q3` also contributed to the same destination before reduction:

```text
semaphore DAG

                        A [t1] {1}
                            | walk
                            v
                        B [t1] {1}
                     +------+------+
                walk |             | walk
                     v             v
          release SC, t1 {1}   release SD, t1 [tma_load] {1}
                  SC |             SD |                  q2, placed after B
                     v                |
         t0 = acquire SC {0}           |
                  t0 |                 |
                     v                 |
                 C [t0] {0}            |
                     | walk             |
                     v                  |
          release SD, t0 {0}       q4  |
                  SD |                  |
                     +---------+--------+
                               v
            t2 = acquire SD pending_count=2 {2}
                            t2 |
                               v
                           D [t2] {2}
```

This rule permits three outcomes: removing all edges from one source owner
may remove that owner's release; removing every source owner removes the
semaphore and acquire; but any release that survives stays at or after its
original latest source node. This is important when asynchronous or
warp-group work makes release placement performance-sensitive.

Implementation note: `releaseFloors` records that latest source before any
edge is removed. The reducers do not recompute it.

## From reduced edges to semaphores

The remaining synchronization edges are converted to semaphore, acquire, and
release nodes in this order:

```text
1. merge edges with the same destination and source owner
2. group the merged edges that end at the same node and owner
3. remove a source owner's release when another remaining path imposes the same wait
4. make loop entry and re-entry use one fixed pending count
5. assign one semaphore and acquire to each remaining destination
6. add one release for each remaining source owner
```

These are still plan nodes. EMIT-IR later creates the actual
`nvws.semaphore.acquire` and `nvws.semaphore.release` operations.

### Repeated edges from one owner

Two pieces can create edges from different nodes of the same owner into the
same `EXIT`. Use this conceptual loop:

```text
for
  W m0 {0}        writes P0 and P1
  R m0 {1}        reads P0 and P1
  R m1 {1}        later read of P1
```

At `EXIT`, both pieces return to owner `{0}`. Their initial edges are:

```text
DAG node          synchronization edge ending here
ENTER(i) {0}      none
W m0 {0}          none
R m0 {1}          m1a: W m0 {0} -> R m0 {1}       P0
                   m1b: W m0 {0} -> R m0 {1}       P1
R m1 {1}          none
EXIT(i) {0}       m2: R m0 {1} -> EXIT(i) {0}      P0
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

Edges `m1a` and `m1b` have the same endpoints, so only `m1a` remains. Edges
`m2` and `m3` have the same source owner and destination. One release after
the later read is sufficient for both pieces. Call the merged edge `M1`:

```text
after reduction and merging

                            W m0 {0}
                                | m1a
                                v
                            R m0 {1}
                                | walk
                                v
                            R m1 {1}
                                | M1 represents m2 and m3
                                v
                           EXIT(i) {0}
```

The two remaining edges map to count-1 semaphores:

```text
edge    semaphore    acquire before    pending_count    initial state
m1a     FULL         R m0 {1}          1                false
M1      READY        EXIT(i) {0}       1                initially released
```

Let `t0` be the token held by owner `{0}` at the start of the iteration:

This conceptual loop is shown in the carried-token form so the example can
focus on merging `m2` and `m3`. If token planning moves the `READY` acquire to
the next iteration's first access, the merged release and pending count stay
the same.

On the first iteration, a root acquire of initially released `READY` supplies
`t0`. In the carried-token form, the final `next` is the loop result; a
zero-trip loop returns the root-acquired `t0` unchanged.

```text
semaphore DAG

                         W m0(i) [t0] {0}
                                  | walk
                                  v
                       release FULL, t0 {0}       m1a
                              FULL |
                                   v
                    t1 = acquire FULL {1}
                              t1 |
                                 v
                      R m0(i) [t1] {1}
                                 | walk
                                 v
                      R m1(i) [t1] {1}
                                 | walk
                                 v
                      release READY, t1 {1}       M1
                             READY |
                                   v
                  next = acquire READY {0}
                            next |
                                 v
                         EXIT(i) {0}
                            next | next iteration
                                 v
                       ENTER(i+1) {0}
```

When all represented edges have completion `[none]`, the merged release
signals once. If their distinct completion kinds are `[none, tma_load]`, the
one release signals twice and the acquire has `pending_count=2`. Edges from
different source owners are never merged into one release.

### Removing a release when another path imposes the same wait

The inner loop in
`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` contains this part:

```text
inner for with boundary owner {2}
  W correct {1}
  R corrected {0}
```

Owner `{1}` writes the buffer, owner `{0}` reads the written value, and the
next inner iteration starts with owner `{2}`. The walk records:

```text
DAG node             synchronization edge ending here
W correct {1}        none
R corrected {0}      k1: W correct {1} -> R corrected {0}
EXIT inner(i) {2}    k2: W correct {1} -> EXIT inner(i) {2}
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

The early loop-boundary reduction keeps both edges to `EXIT` because `{2}` is
the inner loop's first access owner. Before creating releases, the pass checks
the remaining path:

- `k1` orders `{1}`'s write before `{0}`'s read;
- `k3` orders `{0}`'s read before the destination.

Therefore `k2` imposes no additional wait. Removing it removes owner `{1}`'s
release into the `EXIT` semaphore. Owner `{1}` still releases the different
semaphore created from `k1`:

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

The remaining edges become two count-1 semaphores:

```text
edge    semaphore    acquire before          pending_count
k1      CORRECTED    R corrected {0}         1
k3      READY        EXIT inner(i) {2}       1
```

Let `wtok` be the token already held by owner `{1}`:

```text
semaphore DAG

                    W correct [wtok] {1}
                              | walk
                              v
             release CORRECTED, wtok {1}       k1
                    CORRECTED |
                              v
            ctok = acquire CORRECTED {0}
                         ctok |
                              v
                 R corrected [ctok] {0}
                              | walk
                              v
                 release READY, ctok {0}       k3
                         READY |
                               v
                next = acquire READY {2}
                          next |
                               v
                    EXIT inner(i) {2}
                          next | next iteration
                               v
                   ENTER inner(i+1) {2}
```

The general check is deliberately stricter than finding any path in the DAG.
Every synchronization edge represented by one source owner's release must
have both of these remaining parts:

1. a semaphore transfers the buffer from the original source owner to
   another owner after the original source node; and
2. that owner still releases to the final destination, or that owner is the
   destination owner and its program order reaches the destination.

The first acquire must occur no later than the node from which the other
owner releases to the destination. A check may cross enclosing `for`
boundaries, but it never uses one `if` branch to prove ordering in another.

All edges represented by one release stay or disappear together. Proposed
removals are repeatedly checked against the releases that still remain. If
removing one release destroys the path needed to remove another, the second
release stays. The original source nodes still constrain placement as
described in [A surviving release does not move earlier](#a-surviving-release-does-not-move-earlier).

If every incoming release to a destination disappears, no semaphore or
acquire is created for that destination. Removals shared by loop entry and
re-entry are also optional: the pass restores them when removal would prevent
the two sites from using one fixed pending count.

Implementation note: `buildEdgesAndSemas` performs this whole-release check
after merging edges from each source owner. Candidate releases are removed
only after the candidate set stops changing.

### One destination, one semaphore

After reduction and merging, all remaining edges ending at the same node for
the same owner share one semaphore and one acquire. Each remaining source
owner contributes one release. The pending count is:

```text
sum over remaining source owners of max(1, number of completion kinds)
```

```text
remaining releases                           pending_count
two source owners, each with [none]          2
one source owner with [none, tma_load]        2
two per-piece edges from one owner, [none]    1
```

A release is placed after the latest source node recorded for that source
owner and destination. When that access has a `completionAnchor`, placement
uses the anchor described in
[When an access finishes](access-dag.md#when-an-access-finishes). If the
source starts asynchronous work, its completion kind stays on the release so
the signal waits for that work. The acquire is placed before the destination,
except when it can safely move to that owner's first read or write in the next
iteration.

### Reading one semaphore

To read a semaphore DAG, start at an acquire. The following access or boundary
is its destination. Working backward, each source owner has one release of
that semaphore. The pending count is the total number of times those releases
signal it; releases from different owners remain separate.

Working forward, the acquire returns a token. EMIT-IR uses the semaphore and
that token to create the `semaphore.buffer` operand for the destination
access. The destination owner can reuse the token for later accesses and
releases until another acquire replaces it or the token passes through a
`for` or `if` boundary.

The semaphore is chosen from the reduced edges, not from the allocation
member that happened to create one. Overlapping members can therefore share
a semaphore when their remaining edges have the same destination and owners.
Paths with different destinations or destination owners use different
semaphores.

An asynchronous completion stays on the release for its source. It increases
the pending count of the existing acquire rather than creating another
acquire. A release with `[none, tma_load]` signals twice; one with `[none]`
signals once.

Entry and next-iteration acquires for the same buffer group may use one
semaphore. Both sites must use the same pending count.

After this point, edges are not reduced or merged again. Token planning may
move or add acquires and add releases needed to pass a token across a region
boundary. Those operations use the established semaphore and pending count.
Schedule finalization may then change clusters or stage offsets without
recomputing the reduced edges.

### Creating the first token

The first owner needs a token even when no earlier access can release one. If
a loop already returns a token through a semaphore, that semaphore starts
released and supplies the first token too.

In [Repeated edges from one owner](#repeated-edges-from-one-owner), `READY`
returns owner `{0}`'s token at the end of each iteration. The same semaphore
supplies the first token too: `READY` starts released, and a root-owned
acquire before the loop returns the `t0` used at `W m0(0)`. On later
iterations, the `next` acquire shown in that example returns the token before
`EXIT`, and `EXIT` passes it into the next `ENTER`.

The acquire has IR owner `root` because it is outside the partitioned loop,
but it supplies the token used by owner `{0}`. If no existing semaphore can
return the first owner's token, the pass creates an initially released,
count-1 `ENTRY` semaphore. Its acquire supplies the first token, and its
release follows the last access or region that uses that token.

If the only top-level access or region that needs a token is an `if`, and
exactly one branch contains an access or region that reads or writes the
group, the acquire can be placed in that branch. A synchronized group with no
access or region that reads or writes the group is an error.

Implementation note: `insertEntryAcquires` creates this acquire.
`entryTokenOwner` records which partition uses the root-owned result.

### Entry and the next iteration use one semaphore

A token passed through a `for` must come from the same semaphore on first
entry and on every later entry. In
[Example: the same rules at two region levels](#example-the-same-rules-at-two-region-levels),
parent edge `p1` supplies the first inner iteration and child edge `c5`
supplies later inner iterations after `c4` is removed:

```text
edge    semaphore    token used at                         pending_count
p1      READY        ENTER inner(i,0) {2}                 1
c5      READY        EXIT inner(i,j), then ENTER(i,j+1)   1
c1      S1           R m0 {1}                             1
c2      S2           W m0 {1}                             1
c3      S3           R m0 {0}                             1
p2      EMPTY        W m0 in the next outer iteration     1
```

The two `READY` paths in the complete semaphore DAG are:

```text
first inner entry

             release READY, otok {3}       p1
                       READY |
                             v
              ptok = acquire READY {2}
                       ptok |
                            v
                  ENTER inner(i,0) {2}

later inner entry

             release READY, r0tok {0}      c5
                       READY |
                             v
              next = acquire READY {2}
                       next |
                            v
                   EXIT inner(i,j) {2}
                       next |
                            v
                 ENTER inner(i,j+1) {2}
```

The full parent and child flow appears in
[Both loops together](#both-loops-together). The first path has no preceding
inner `EXIT`; the second passes `next` through the current `EXIT` into the
next `ENTER`.

Both acquire sites must use one fixed pending count. If re-entry waits for
two signals but entry has one `[none]` release, that entry release can signal
twice through the literal IR attribute `arrive_count=2`:

```text
release READY, %outer [none] arrive_count=2
%first = acquire READY pending_count=2

... inner body ...

release READY, %reader1 [none] arrive_count=1
release READY, %reader2 [none] arrive_count=1
%next = acquire READY pending_count=2
```

The complete reconciliation rules are:

```text
entry signals    re-entry signals    result
1 [none]         2                   entry release uses arrive_count=2
2                2                   both acquires use pending_count=2
2                1 after removal     restore optional removals and recheck
```

Increasing `arrive_count` is allowed only when entry has exactly one `[none]`
release. Any remaining count mismatch is an error.

## Tokens through `for` and `if`

After semaphores are assigned, a region may receive a token before it starts
and return a token when it finishes. Each `if` path chooses separately:

```text
path performs an acquire     return the token from that acquire
path performs no acquire     return the token that entered the path
```

Every path must return a token with the same owner. A path cannot return its
input token when no input token exists.

A loop has another choice. It can carry a token from one iteration to the
next, or it can acquire the token immediately before the first buffer use
that needs it:

```text
carry the token
  %result = for iter_args(%token = %entry) { ... yield %next }

acquire at the first buffer use
  for { %token = acquire S; ... use buffer with %token ... }
```

The following examples derive these choices from the input accesses and
synchronization edges. Inner regions are handled first. Their parent then
uses the returned token without inspecting the child operations again.

### Region results

The token returned by a region must come from one semaphore. A loop normally
returns a token from the semaphore that supplied its input. An `if` normally
does the same, but an `if` entered with a root-owned token may instead use the
semaphore acquired on one of its paths. This permits:

- both branches to return tokens from acquires;
- one branch to return a token from an acquire while the other returns the
  input token; and
- an `if` without `else` to return the input token on its implicit path.

An `if` does not need to return a token when nothing uses that token before
the next acquire and its enclosing region does not need the result. A parent
uses a child's token only when the child enters and returns the same owner and
no later acquire, release, buffer use, or child region replaces it on any
path.

In the implementation, `RegionFlow` records the boundary owner, the acquire
or child selected on each path, and the semaphore used when the input
semaphore cannot be used. `summarizeRegionFlow` builds that record, and
`pruneDeadIfFlows` removes unused `if` results.

### Example: an `if` returns one owner's token

`test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token` has this relevant access pattern in one
loop iteration. `m0` contains one piece P0:

```text
W m0 {1}
if cond
  then: R m0 {0}
  else: no m0 access
```

The owner immediately before the `if` is `{1}`, so its summary is
`P0:R:{1}`. The parent needs no synchronization edge around that summary:

```text
DAG node                    synchronization edge ending here
W m0 {1}                    none
[if P0:R:{1}]               none
```

The then path changes owners twice. The empty else path changes nothing:

```text
then-path DAG node          synchronization edge ending here
ENTER if {1}                none
R m0 {0}                    e1: ENTER if {1} -> R m0 {0}
EXIT if {1}                 e2: R m0 {0} -> EXIT if {1}

else-path DAG node          synchronization edge ending here
ENTER if {1}                none
EXIT if {1}                 none
```

```text
synchronization-edge DAGs

parent                         then path                else path

W m0 {1}                      ENTER if {1}             ENTER if {1}
    | walk                         | e1                     | walk
    v                              v                        v
[if P0:R:{1}]                 R m0 {0}                  EXIT if {1}
                                   | e2
                                   v
                              EXIT if {1}
```

The parent summary and the two child paths are separate DAGs. Token planning
joins the two child results afterward; no synchronization edge connects the
summary directly to a child `ENTER` or `EXIT`.

The two edges use two semaphores. The semaphore used by `e2` also carries the
owner-`{1}` token into the loop and on to its next iteration:

```text
edge    semaphore    pending_count    initial state
e1      FULL         1                false
e2      EMPTY        1                initially released
```

`EMPTY` starts released. The then path returns a newly acquired owner-`{1}`
token. The else path returns the owner-`{1}` token that entered the `if`:

```text
semaphore DAG for one loop iteration

                         ENTER loop(i) {1}
                                   | walk
                                   v
                  itok = acquire EMPTY {1}
                              itok |
                                   v
                         W m0(i) [itok] {1}
                                   |
                         +---------+---------+
                  then   |                   | else
                         v                   v
                 ENTER if {1}             ENTER if {1}
                           |                 | walk
                           v                 v
       release FULL, itok [tc5mma] {1}     EXIT if {1}
                      FULL |                 |
                           v                 |
              rtok = acquire FULL {0}        |
                      rtok |                 |
                           v                 |
                  R m0(i) [rtok] {0}         |
                           |                 |
                release EMPTY, rtok {0}      |
                     EMPTY |                 |
                           v                 |
          returned = acquire EMPTY {1}       |
                  returned |                 |
                           v                 |
                   EXIT if {1}               |
                           |                 |
              out = returned        out = itok
                         +---------+---------+
                                   v
 release EMPTY, out {1} ---------------- EMPTY ----------------+
                 | walk                                      |
                 v                                           |
                         EXIT loop(i) {1}
                                   | next iteration
                                   v
                       ENTER loop(i+1) {1}
                                   | walk
                                   v
                    next = acquire EMPTY {1} <----------------+
                               next |
                                    v
                         W m0(i+1) [next] {1}
```

On iteration zero, the initially released `EMPTY` supplies `itok`. On later
iterations, the release after the preceding `if` supplies it. If the loop
executes zero times, none of these operations executes.

The `release EMPTY, out` after the `if` is not `e2`. Edge `e2` returns the
then path to owner `{1}`; the later release makes whichever owner-`{1}` token
the `if` returned available to the next loop iteration.

EMIT-IR can subsequently split the then-path release, body, and acquire into
scheduler-safe conditionals, as shown in
[Scheduler-safe conditional boundaries](emit-ir.md#scheduler-safe-conditional-boundaries).

### Moving an acquire to its first use

`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use` has this input. The
last operation directly consumes the memdesc through an unknown operation,
so ACCESS-DAG conservatively records it as an exclusive `W` access. It is not
a literal store:

```text
for
  W m0 {0}
  R m0 {1}
  exclusive use of m0 {2}    recorded as W
```

The boundary owner is `{0}`. The complete edge set is:

```text
DAG node                  synchronization edge ending here
ENTER(i) {0}              none
W m0 {0}                  none
R m0 {1}                  e1: W m0 {0} -> R m0 {1}
W m0 {2}                  e2: R m0 {1} -> W m0 {2}
EXIT(i) {0}               e3: W m0 {2} -> EXIT(i) {0}
```

```text
synchronization-edge DAG

                          ENTER(i) {0}
                               | walk
                               v
                           W m0 {0}
                               | e1
                               v
                           R m0 {1}
                               | e2
                               v
                           W m0 {2}
                               | e3
                               v
                           EXIT(i) {0}
```

The three edges use three semaphores:

```text
edge    semaphore    pending_count
e1      FULL0        1
e2      FULL1        1
e3      EMPTY        1
```

Immediately after semaphore placement, the acquire for `e3` is before
`EXIT(i)` and its token is returned to the next iteration:

```text
initial token path

release EMPTY, tok2 {2}
          EMPTY |
                v
next = acquire EMPTY {0}
           next |
                v
          EXIT(i) {0}
           next | next iteration
                v
        ENTER(i+1) {0}
           next |
                v
       W m0(i+1) [next] {0}
```

Nothing between that acquire and the next write uses `next`. The acquire can
therefore move directly before the write, removing the token from the loop
operands and results. The final semaphore DAG is:

```text
semaphore DAG after moving the acquire

                         ENTER(i) {0}
                              | walk
                              v
                tok0 = acquire EMPTY {0}
                         tok0 |
                              v
                    W m0(i) [tok0] {0}
                              | walk
                              v
                  release FULL0, tok0 {0}
                        FULL0 |
                              v
                 tok1 = acquire FULL0 {1}
                         tok1 |
                              v
                    R m0(i) [tok1] {1}
                              | walk
                              v
                  release FULL1, tok1 {1}
                        FULL1 |
                              v
                 tok2 = acquire FULL1 {2}
                         tok2 |
                              v
                    W m0(i) [tok2] {2}
                              | walk
                              v
        release EMPTY, tok2 {2} e3 -------------- EMPTY ---------------+
                                                                       |
                         EXIT(i) {0}                                   |
                              | next iteration                         |
                              v                                        |
                       ENTER(i+1) {0}                                  |
                              | walk                                   |
                              v                                        |
                next = acquire EMPTY {0} <-----------------------------+
                         next |
                              v
                  W m0(i+1) [next] {0}
```

`EMPTY` starts released, so the acquire succeeds on iteration zero. Every
iteration releases it for the next one. A zero-trip loop executes neither the
acquire nor the release.

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
`@tmem_single_producer_multi_consumer_fanout` has one piece P0 and this input:

```text
for
  W first {0}
  R reader1 {1}
  R reader2 {2}
  W final {0}
```

The boundary owner is `{0}`. The first write sends the value to both readers,
and the final write waits for both readers:

```text
DAG node                  synchronization edge ending here
ENTER(i) {0}              none
W first {0}               none
R reader1 {1}             e1: W first {0} -> R reader1 {1}
R reader2 {2}             e2: W first {0} -> R reader2 {2}
W final {0}               e3: R reader1 {1} -> W final {0}
                           e4: R reader2 {2} -> W final {0}
EXIT(i) {0}               none
```

```text
synchronization-edge DAG

                          ENTER(i) {0}
                               | walk
                               v
                          W first {0}
                         +-----+-----+
                      e1 |           | e2
                         v           v
                  R reader1 {1}  R reader2 {2}
                      e3 |           | e4
                         +-----+-----+
                               v
                          W final {0}
                               | walk
                               v
                          EXIT(i) {0}
```

The two edges ending at the final write share one count-2 semaphore:

```text
edge        semaphore    pending_count
e1          TO_R1        1
e2          TO_R2        1
e3, e4      EMPTY        2
```

The token acquired from `EMPTY` serves both the final write in iteration `i`
and the first write in iteration `i+1`:

```text
semaphore DAG

       entry = acquire EMPTY pending_count=2 root       supplies owner {0}
                              entry |
                                    v
                             ENTER(i) {0}
                                    | walk
                                    v
                       W first(i) [entry] {0}
                         +----------+----------+
                         |                     |
          release TO_R1, entry {0}   release TO_R2, entry {0}
                   TO_R1 |                TO_R2 |
                         v                     v
          r1tok = acquire TO_R1 {1}  r2tok = acquire TO_R2 {2}
                  r1tok |                     | r2tok
                         v                     v
             R reader1(i) [r1tok]    R reader2(i) [r2tok]
                         |                     |
      release EMPTY, r1tok {1}       release EMPTY, r2tok {2}
                   EMPTY |                     | EMPTY
                         +----------+----------+
                                    v
          next = acquire EMPTY pending_count=2 {0}
                               next |
                                    v
                       W final(i) [next] {0}
                                    | walk
                                    v
                              EXIT(i) {0}
                                    | loop backedge
                                    v
                            ENTER(i+1) {0}
                                    | walk
                                    v
                       W first(i+1) [next] {0}
```

Moving the count-2 acquire to the next first write would leave the final write
without a token and require another release/acquire pair. The loop therefore
carries `next`. The boundary, first write, and final write all have owner
`{0}`, so the boundary itself adds no synchronization edge. If the loop
executes zero times, `entry` is its result.

### A use after the loop

This deliberately simplified example shows why a use after a loop does not
always require a loop-carried token. It has one piece P0:

```text
W before {1}

for
  R body-read {2}
  W body-write {1}

R after {3}
```

The loop writes P0 and its first access has owner `{2}`, so its summary is
`P0:W:{2}`. The parent edges are:

```text
DAG node                     synchronization edge ending here
W before {1}                 none
[for P0:W:{2}]               p1: W before {1} -> for
R after {3}                  p2: for -> R after {3}
```

The child edges are:

```text
DAG node                     synchronization edge ending here
ENTER(i) {2}                 none
R body-read {2}              none
W body-write {1}             c1: R body-read {2} -> W body-write {1}
EXIT(i) {2}                  c2: W body-write {1} -> EXIT(i) {2}
```

```text
synchronization-edge DAGs

parent                              child

W before {1}                       ENTER(i) {2}
     | p1                               | walk
     v                                  v
[for P0:W:{2}]                    R body-read {2}
     | p2                               | c1
     v                                  v
R after {3}                       W body-write {1}
                                         | c2
                                         v
                                    EXIT(i) {2}
```

Parent edge `p1` and child edge `c2` use the same semaphore: `p1` supplies
the first iteration and `c2` supplies the next iteration. The other two edges
use their own semaphores:

```text
edges      semaphore    pending_count    initial state
p1, c2     TO_READER    1                initially released
c1         TO_WRITER    1                false
p2         AFTER_LOOP   1                false
```

The acquire after the loop consumes the same `TO_READER` release that would
have supplied another iteration. The initially released state of `TO_READER`
also supplies the root-owned token used by `W before`:

```text
semaphore DAG through one body iteration

        before = acquire TO_READER root       supplies owner {1}
                              before |
                                     v
                        W before [before] {1}
                                     | walk
                                     v
      release TO_READER, before {1} p1 -------- TO_READER --------+
                                                                 |
                             ENTER(0) {2}                        |
                                    | walk                       |
                                    v                            |
                read = acquire TO_READER {2} <-------------------+
                               read |
                                    v
                    R body-read(0) [read] {2}
                                    | walk
                                    v
                    release TO_WRITER, read {2}         c1
                          TO_WRITER |
                                    v
               write = acquire TO_WRITER {1}
                              write |
                                    v
                  W body-write(0) [write] {1}
                                    | walk
                                    v
                   release TO_READER, write {1}         c2

                              EXIT(0) {2}
```

If the loop continues, the next iteration acquires the release from `c2` at
its first read:

```text
release TO_READER, write(i) {1} c2 -------- TO_READER --------+
                                                              |
                    EXIT(i) {2}                              |
                         | next iteration                    |
                         v                                   |
                   ENTER(i+1) {2}                            |
                         | walk                              |
                         v                                   |
       next = acquire TO_READER {2} <------------------------+
                      next |
                           v
          R body-read(i+1) [next] {2}
```

If the loop finishes, the acquire after the loop consumes the release from
the final `c2`, then implements parent edge `p2`:

```text
release TO_READER, write(last) {1} c2 -------- TO_READER --------+
                                                                 |
                    EXIT(last) {2}                              |
                         | loop finishes                        |
                         v                                      |
       final = acquire TO_READER {2} <--------------------------+
                     final |
                           v
     release AFTER_LOOP, final {2}       p2
                AFTER_LOOP |
                           v
        out = acquire AFTER_LOOP {3}
                       out |
                           v
              R after [out] {3}
```

For a zero-trip loop, there is no child `ENTER` or `EXIT`; the post-loop
`final` acquire consumes the `p1` release after `W before`. The
implementation marks `final` with `postLoopAcquire` so schedule finalization
uses owner `{2}`'s boundary after the loop.

If the last child returns the loop token and that token is also used after the
loop, the token remains in the child result and the loop returns it.

### Nested loops

Inner loops are handled before their outer loops. The first example in
`test/NVWS/insert_semas_nested_ws_inner_loop.mlir`,
`@nested_ws_inner_loop`, has this input for one piece P0:

```text
outer for
  inner for
    W acc {1}    MMA
    R acc {0}
```

The inner summary is `P0:W:{1}`. It is the only P0 node in the outer body, so
the outer child has no synchronization edge:

```text
DAG node                         synchronization edge ending here
ENTER outer(i) {1}              none
[inner for P0:W:{1}]            none
EXIT outer(i) {1}               none
```

The inner child changes from `{1}` to `{0}` and back to boundary owner `{1}`:

```text
DAG node                         synchronization edge ending here
ENTER inner(i,j) {1}            none
W acc {1}                        none
R acc {0}                        e1: W acc {1} -> R acc {0}
EXIT inner(i,j) {1}             e2: R acc {0} -> EXIT inner(i,j) {1}
```

```text
synchronization-edge DAG with both region levels

                       ENTER outer(i) {1}
                                | walk
                                v
               +-- [inner-for summary P0:W:{1}] --+
               |                                   |
               |       ENTER inner(i,j) {1}        |
               |                | walk             |
               |                v                  |
               |            W acc {1}              |
               |                | e1               |
               |                v                  |
               |            R acc {0}              |
               |                | e2               |
               |                v                  |
               |       EXIT inner(i,j) {1}         |
               +-----------------------------------+
                                | walk
                                v
                        EXIT outer(i) {1}
```

`e1` uses `FULL`, and `e2` uses initially released `EMPTY`. Both have
`pending_count=1`:

```text
edge    semaphore    pending_count
e1      FULL         1
e2      EMPTY        1
```

The `EMPTY` acquire moves to the first MMA in each inner iteration:

```text
semaphore DAG for one executed inner iteration

                      ENTER inner(i,j) {1}
                                | walk
                                v
                   wtok = acquire EMPTY {1}
                            wtok |
                                 v
                    W acc(i,j) [wtok] {1}
                                 | walk
                                 v
          release FULL, wtok [tc5mma] {1}       e1
                           FULL |
                                v
                    rtok = acquire FULL {0}
                            rtok |
                                 v
                    R acc(i,j) [rtok] {0}
                                 | walk
                                 v
                    release EMPTY, rtok {0}       e2
```

The `EMPTY` semaphore connects that release to the first MMA in whichever
inner iteration executes next. The control-flow paths do not carry its token:

```text
release EMPTY, rtok {0} e2 ------------------- EMPTY -------------------+
                                                                         |
EXIT inner(i,j) {1}                                                      |
       +----------------------+----------------------+                    |
       | inner continues      | inner finishes       |                    |
       v                      v                                           |
ENTER inner(i,j+1) {1}   EXIT outer(i) {1}                               |
                              | next outer iteration                     |
                              v                                          |
                       ENTER outer(i+1) {1}                               |
                              | walk                                     |
                              v                                          |
                       ENTER inner(i+1,0) {1}                             |
       +----------------------+----------------------+                    |
                              v                                          |
            next = acquire EMPTY {1} <-----------------------------------+
                         next |
                              v
                 W acc [next] {1}
```

The initially released `EMPTY` supplies the first executed inner iteration.
Each read releases it for the next executed inner iteration, even when that
iteration belongs to the next outer iteration. If an inner loop executes zero
times, it performs no semaphore operation and leaves `EMPTY` available. The
pass adds no semaphore-token operand or result to either loop; the unrelated
async token already present in the test remains separate.

`@nested_ws_inner_loop_parent_continuation` adds one read after the inner
loop:

```text
outer for
  inner for
    W acc {1}    MMA
    R acc {0}

  R acc {0}      outer read
```

The inner edge set is unchanged. In the outer child, the inner summary has
owner `{1}` and the outer read has owner `{0}`:

```text
outer DAG node                   synchronization edge ending here
ENTER outer(i) {1}              none
[inner for P0:W:{1}]            none
R outer {0}                     p1: inner for -> R outer {0}
EXIT outer(i) {1}               p2: R outer {0} -> EXIT outer(i) {1}

inner DAG node                   synchronization edge ending here
ENTER inner(i,j) {1}            none
W acc {1}                        none
R inner {0}                     c1: W acc {1} -> R inner {0}
EXIT inner(i,j) {1}             c2: R inner {0} -> EXIT inner(i,j) {1}
```

```text
synchronization-edge DAG with outer continuation

                       ENTER outer(i) {1}
                                | walk
                                v
               +-- [inner-for summary P0:W:{1}] --+
               |                                   |
               |       ENTER inner(i,j) {1}        |
               |                | walk             |
               |                v                  |
               |            W acc {1}              |
               |                | c1               |
               |                v                  |
               |          R inner {0}              |
               |                | c2               |
               |                v                  |
               |       EXIT inner(i,j) {1}         |
               +-----------------------------------+
                                | p1
                                v
                         R outer {0}
                                | p2
                                v
                        EXIT outer(i) {1}
```

The four edges use four count-1 semaphores:

```text
edge    semaphore       pending_count    initial state
c1      LOCAL_FULL      1                false
c2      LOCAL_EMPTY     1                initially released
p1      OUTER_FULL      1                false
p2      OUTER_EMPTY     1                initially released
```

`LOCAL_EMPTY` and `OUTER_EMPTY` start released. The acquire of
`OUTER_EMPTY` before the outer loop consumes its initial value; it is not a
loop argument. This makes the later tail acquire wait for the outer read
rather than succeeding early.

```text
outer_entry = acquire OUTER_EMPTY root
```

```text
semaphore DAG for an inner iteration

                 wtok = acquire LOCAL_EMPTY {1}
                            wtok |
                                 v
                    W acc(i,j) [wtok] {1}
                                 | walk
                                 v
      release LOCAL_FULL, wtok [tc5mma] {1}       c1
                     LOCAL_FULL |
                                v
               rtok = acquire LOCAL_FULL {0}
                            rtok |
                                 v
                   R inner(i,j) [rtok] {0}
                                 | walk
                                 v
               release LOCAL_EMPTY, rtok {0}       c2
```

If the inner loop continues, that `LOCAL_EMPTY` release supplies the acquire
at the next MMA. If it finishes, `bridge` consumes the same release after the
last `EXIT`:

```text
inner loop continues

release LOCAL_EMPTY, rtok {0} c2 -------------- LOCAL_EMPTY --------------+
                                                                          |
EXIT inner(i,j) {1}                                                       |
       | next inner iteration                                             |
       v                                                                  |
ENTER inner(i,j+1) {1}                                                    |
       | walk                                                             |
       v                                                                  |
next = acquire LOCAL_EMPTY {1} <------------------------------------------+
       next |
            v
       next MMA

inner loop finishes

release LOCAL_EMPTY, rtok {0} c2 -------------- LOCAL_EMPTY --------------+
                                                                          |
EXIT inner(i,last) {1}                                                    |
       | loop finishes                                                    |
       v                                                                  |
bridge = acquire LOCAL_EMPTY {1} <----------------------------------------+
       bridge |
              v
release OUTER_FULL, bridge [tc5mma] {1} p1
       OUTER_FULL |
                  v
out = acquire OUTER_FULL {0}
       out |
           v
R outer(i) [out] {0}
       | walk
       v
release OUTER_EMPTY, out {0}       p2
       OUTER_EMPTY |
                   v
tail = acquire OUTER_EMPTY {1}
       tail |
            v
release LOCAL_EMPTY, tail {1}
```

The final release makes the token available in the next outer iteration. It
connects directly to either that iteration's first MMA acquire or its
post-inner `bridge` when the inner loop has zero iterations:

```text
release LOCAL_EMPTY, tail {1} ---------------- LOCAL_EMPTY ---------------+
                                                                          |
EXIT outer(i) {1}                                                         |
       | next outer iteration                                             |
       v                                                                  |
ENTER outer(i+1) {1}                                                      |
       +----------------------+----------------------+                     |
       | inner executes       | inner has zero trips |                     |
       v                      v                                            |
ENTER inner(i+1,0) {1}   bridge = acquire LOCAL_EMPTY {1} <----------------+
       | walk                                                             |
       v                                                                  |
first = acquire LOCAL_EMPTY {1} <------------------------------------------+
       first |
             v
        first MMA
```

The release from `tail` to `LOCAL_EMPTY` is not another synchronization
edge. It makes owner `{1}`'s token available to the first inner MMA of the
next outer iteration. If the inner loop executes zero times, `bridge` consumes
the initially released `LOCAL_EMPTY` on the first outer iteration or the
preceding `tail` release on later outer iterations. The outer read and tail
then proceed in the same way. Neither loop carries a semaphore token.

## Backing copies

After synchronization edges are known, the pass chooses how many physical
copies back each synchronized buffer group. A group with an explicit
`buffer.copy` uses that value. A group without the attribute starts with one
copy.

A synchronized TMEM accumulator can use two copies when every MMA directly
inside the loop satisfies these checks:

- the loop does not read, modify, and write back an accumulator value;
- the MMA and loop support multiple accumulator copies;
- the enclosing WS loop does not disable them;
- two copies fit in the available TMEM blocks; and
- no scaled MMA uses block N of 256.

When `use-meta-partitioner` is set, the pass does not add this automatic
second TMEM copy. An inconsistent or non-positive explicit `buffer.copy` in
one group is an error.

Semaphore copies are chosen separately and usually equal the buffer copies.
For a local buffer with no explicit `buffer.copy`, a release after a TMA load
uses at least the number of stages requested by later semaphore lowering:

```text
semaphore copies = max(1, requested lowering stages)
```

This does not change the buffer copy count. Schedule and stage analysis use
the semaphore copy count that lowering will create. The implementation calls
the lowering-stage source `LowerSemaphore`.

### Example: a TMEM accumulator gets two copies

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` has one TMEM piece
P0. Its relevant input is:

```text
W acc root                 initial store

for
  R acc {1}
  W acc {1}
  W acc {2}                MMA accumulator

R acc root                 final load
```

The loop boundary owner is `{1}`. The root token used by the initial store is
passed into the loop, so no root-to-`{1}` synchronization edge is needed. The
loop writes P0 and first accesses it in owner `{1}`, so its parent summary is
`P0:W:{1}`:

```text
parent DAG node                 synchronization edge ending here
W acc root                     none
[for P0:W:{1}]                 none
R acc root                     p1: for -> R acc root
```

```text
parent synchronization-edge DAG

                       W acc root
                           | walk, same token
                           v
                     [for P0:W:{1}]
                           | p1
                           v
                       R acc root
```

The child edge set is:

```text
DAG node                   synchronization edge ending here
ENTER(i) {1}               none
R acc {1}                  none
W acc {1}                  none
W acc {2}                  e1: W acc {1} -> W acc {2}
EXIT(i) {1}                e2: W acc {2} -> EXIT(i) {1}
```

```text
synchronization-edge DAG

                           ENTER(i) {1}
                                | walk
                                v
                            R acc {1}
                                | walk
                                v
                            W acc {1}
                                | e1
                                v
                       W acc {2}  MMA
                                | e2
                                v
                           EXIT(i) {1}
```

`e1` uses `TO_MMA`, `e2` uses initially released `EMPTY`, and parent edge
`p1` uses `AFTER`. All have `pending_count=1`:

```text
edge    semaphore    pending_count    initial state
e1      TO_MMA       1                false
e2      EMPTY        1                initially released
p1      AFTER        1                false
```

The same `EMPTY` token serves the root store and the first owner-`{1}`
access, so no release/acquire pair is inserted between them.

```text
semaphore DAG

                     root = acquire EMPTY
                              root |
                                   v
                      W acc [root] root
                              same token
                                   |
                                   v
                          ENTER(0) {1}
                                   | walk
                                   v
                         R acc(0) [root] {1}
                                   | walk
                                   v
                         W acc(0) [root] {1}
                                   |
                    release TO_MMA, root {1}      e1
                          TO_MMA |
                                 v
                    mma = acquire TO_MMA {2}
                             mma |
                                 v
                    W acc(0) [mma] {2}  MMA
                                 |
               release EMPTY, mma [tc5mma] {2}    e2
                           EMPTY |
                                 v
                   next = acquire EMPTY {1}
                            next |
                                 v
                           EXIT(0) {1}
                       +---------+---------+
       another iteration         |         | loop finishes
                       next       |         | next
                         v        |         v
                 ENTER(1) {1}     |  result = next
                         |        |         |
                    repeat body   |         v
                         |        |  release AFTER, result {1}       p1
                         |        |       AFTER |
                         +--------+             v
                                     out = acquire AFTER root
                                                   out |
                                                       v
                                          R acc [out] root
```

The release to `AFTER` implements parent edge `p1`; it is not another child
synchronization edge. For a zero-trip loop, the initial root token is the loop
result and supplies that release.

This group is synchronized, has no explicit `buffer.copy`, is in TMEM, and
has one qualifying MMA. Two copies fit, so the generated allocation and the
semaphores refer to a two-copy buffer:

```text
input TMEM buffer       memdesc<128x128xf32>
generated backing      memdesc<2x128x128xf32>
buffer copies           2
semaphore copies        2
```

### Example: a TMA load increases only the semaphore copies

The `buffer.id = 102` group in `test/NVWS/insert_semas.mlir`
`@local_release_after_mma` is an SMEM group with one piece P0 and no explicit
`buffer.copy`. Its input is:

```text
for
  W m0 {0}    descriptor_load
  R m0 {1}    MMA operand
```

The loop boundary owner is `{0}`. The descriptor load sends P0 to the MMA,
and the MMA must finish reading P0 before the next descriptor load:

```text
DAG node                  synchronization edge ending here
ENTER(i) {0}              none
W m0 {0}                  none
R m0 {1}                  e1: W m0 {0} -> R m0 {1}
EXIT(i) {0}               e2: R m0 {1} -> EXIT(i) {0}
```

```text
synchronization-edge DAG

                         ENTER(i) {0}
                              | walk
                              v
                          W m0(i) {0}
                              | e1
                              v
                          R m0(i) {1}
                              | e2
                              v
                          EXIT(i) {0}
```

`e1` uses `FULL`, and `e2` uses initially released `EMPTY`. Both have
`pending_count=1`:

```text
edge    semaphore    pending_count    initial state
e1      FULL         1                false
e2      EMPTY        1                initially released
```

The semaphore DAG shows the descriptor load's asynchronous completion on the
`FULL` release and the MMA completion on the `EMPTY` release:

```text
                    empty = acquire EMPTY(i) {0}
                                 empty |
                                       v
                    W m0(i) [empty] {0}       descriptor_load
                                       | walk
                                       v
        release FULL, empty [tma_load] {0}          e1
                                 FULL |
                                      v
                       full = acquire FULL {1}
                                  full |
                                       v
                    R m0(i) [full] {1}       MMA operand
                                       | walk
                                       v
         release EMPTY, full [tc5mma] {1}           e2
                                EMPTY |
                                      v
                  next = acquire EMPTY(i+1) {0}
                                 next |
                                      v
                       W m0(i+1) [next] {0}
```

SMEM does not receive the automatic second TMEM copy, so the buffer stays
single-copy. Because the `FULL` release carries `tma_load`, the semaphore
copy count for the whole group uses the lowering stage count:

```text
buffer copies       1
semaphore copies    max(1, requested lowering stages)
```

The initially released `EMPTY` supplies iteration zero. Each MMA release
supplies the next descriptor load. There is no buffer use after the loop, and
a zero-trip loop executes none of the shown operations.

In the implementation, `computeBackingCopies` chooses the buffer copies and
`computeSemaphoreCopies` applies the TMA-load rule.

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

For each release/acquire relationship created from a synchronization edge,
the pass records a schedule relation. Token movement through a region can
create another such relation. It remembers the source, destination, release,
and acquire so schedule finalization can check whether the release must
execute before that acquire in one expanded loop body. The relation remains
available when token placement moves an acquire. A dedicated entry acquire
and the release that replenishes it need no schedule relation.

### Example: one-copy synchronization between iterations

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` has one physical copy of P0. Its input has one write
to member `m0`, which covers P0, followed by two reads from `m0`. The write
has owner `{3}` and both reads have owner `{1}`. `(s,c)` is
`(loop.stage, loop.cluster)`:

```text
buffer.copy = 1

for i
  W m0(i) {3}             (0,1)
  R first m0(i) {1}       (0,1)
  R final m0(i) {1}       (1,2)
```

The two reads have the same owner. The final read therefore replaces that
owner's latest read without adding another synchronization edge:

```text
DAG node                  synchronization edge ending here
ENTER(i) {3}              none
W m0(i) {3}               none
R first m0(i) {1}         e1: W m0(i) {3} -> R first m0(i) {1}
R final m0(i) {1}         none
EXIT(i) {3}               e2: R final m0(i) {1} -> EXIT(i) {3}
```

```text
                          ENTER(i) {3}
                               | walk
                               v
                           W m0(i) {3}
                               | e1
                               v
                       R first m0(i) {1}
                               | walk
                               v
                       R final m0(i) {1}
                               | e2
                               v
                           EXIT(i) {3}
```

`e1` becomes `FULL`. `e2` becomes `EMPTY`, which is also initially released
to provide the first write token. Both semaphores have `pending_count=1`.
The semaphore DAG follows the same accesses across the iteration boundary:

```text
                    wtok = acquire EMPTY(i) {3}
                                wtok |
                                     v
                         W m0(i) [wtok] {3}
                                     | walk
                                     v
                     release FULL, wtok {3}       e1
                               FULL |
                                     v
                    rtok = acquire FULL(i) {1}
                                rtok |
                                     v
                   R first m0(i) [rtok] {1}
                                     | walk
                                     v
                   R final m0(i) [rtok] {1}
                                     | walk
                                     v
 release EMPTY, rtok {1} e2 --------------- EMPTY ---------------+
                                                                    |
                               EXIT(i) {3}
                                     | next iteration
                                     v
                             ENTER(i+1) {3}
                                     | walk
                                     v
                 next = acquire EMPTY(i+1) {3} <-------------------+
                                next |
                                     v
                       W m0(i+1) [next] {3}
```

The `EMPTY` release in iteration `i` satisfies the acquire in iteration
`i+1`. The acquire is at the next write rather than carried through the loop.
The initially released state supplies `wtok` when `i=0`. There is no
post-loop buffer use in this test; after the final iteration, no later acquire
consumes its `EMPTY` release. A zero-trip loop executes none of these
operations and leaves the initially released state untouched.

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

The required loop distance depends on physical copy reuse. The pass follows
the ordered reads and writes to determine when a physical copy is reused:

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

The write/read analysis finds distance 1. After schedule finalization moves
the destination operations to cluster 3, the release uses the source
completion schedule and the acquire uses the destination schedule:

```text
final-read(i)      owner {1}  (1,2)
release EMPTY(i)   owner {1}  (1,2)
acquire EMPTY(i+1) owner {3}  (0,3)
W(i+1)             owner {3}  (0,3)
```

When a source access has a `completionAnchor`, its release copies that
anchor's schedule rather than the access schedule. Asynchronous work instead
stays on the release as a completion kind. A semaphore buffer copies the
schedule of the access it serves.

Owners execute independently. Let `offset[P]` be the whole-iteration delay
of owner P. A release by P at stage `before`, followed at loop distance
`distance` by an acquire owned by Q at stage `after`, requires:

```text
offset[Q] >= offset[P] + before - after - distance
```

The pass solves all release/acquire schedule relations in one loop together.
The cycle total is the sum of
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

A positive delay in one schedule relation is legal when the reverse path has
enough negative delay. For example, a `+1` relation and a `-3` return
relation form a legal `-2` cycle.
`test/NVWS/insert_semas_recurrence_owner_cycle.mlir` exercises that shape.

Schedule finalization collects release/acquire orderings that must hold in one
expanded loop body, together with same-stage SSA orderings, and increases
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

`test/NVWS/insert_semas_nested_carrier.mlir`
`@scheduled_relocated_acquire_boundaries` has one outer loop and one inner
loop over a TMEM accumulator. At a high level its accesses are:

```text
outer for i
  W acc {0}

  inner for j
    W acc by MMA {1}      stage 0, cluster 1
    R acc {0}             stage 1, cluster 2

  R acc {0}               stage 0, cluster 4
```

The inner loop writes P0 and its first access has owner `{1}`, so its parent
summary is `P0:W:{1}`. The parent edges are:

```text
DAG node                         synchronization edge ending here
ENTER outer(i) {0}               none
W acc {0}                        none
[inner for P0:W:{1}]             p1: W acc {0} -> inner for
R acc after inner {0}            p2: inner for -> R acc after inner {0}
EXIT outer(i) {0}                none
```

```text
                      ENTER outer(i) {0}
                               | walk
                               v
                           W acc {0}
                               | p1
                               v
                    [inner-for P0:W:{1}]
                               | p2
                               v
                    R acc after inner {0}
                               | walk
                               v
                      EXIT outer(i) {0}
```

The child uses its actual write and read:

```text
DAG node                         synchronization edge ending here
ENTER inner(i,j) {1}             none
W acc by MMA {1}                 none
R acc {0}                        c1: W acc by MMA {1} -> R acc {0}
EXIT inner(i,j) {1}              c2: R acc {0} -> EXIT inner(i,j) {1}
```

```text
                       ENTER inner(i,j) {1}
                                | walk
                                v
                      W acc by MMA {1}
                                | c1
                                v
                           R acc {0}
                                | c2
                                v
                       EXIT inner(i,j) {1}
```

`p1` and `c2` use `MMA_READY`: `p1` supplies the first inner iteration and
`c2` supplies the next one. `c1` uses `ACC_FULL`, and `p2` uses
`OUTER_EMPTY`. Every semaphore has `pending_count=1`. Token placement gives:

```text
outer entry and one inner iteration

                   entry = acquire OUTER_EMPTY root
                               entry |
                                     v
                         ENTER outer(0) {0}
                               entry |
                                     v
                         W acc(0) [entry] {0}
                                     | walk
                                     v
 release MMA_READY, entry {0} p1 -------------- MMA_READY --------------+
                                                                         |
                         ENTER inner(0,0) {1}                            |
                                     | walk                              |
                                     v                                   |
                 ready = acquire MMA_READY {1} <-------------------------+
                               ready |
                                     v
            W acc by MMA(0,0) [ready] {1}       (0,1)
                                     | walk
                                     v
        release ACC_FULL, ready [tc5mma] {1}       c1
                           ACC_FULL |
                                     v
                    full = acquire ACC_FULL {0}
                                full |
                                     v
                       R acc(0,0) [full] {0}       (1,2)
                                     | walk
                                     v
                    release MMA_READY, full {0}       c2
```

After an executed inner iteration, the `c2` release supplies either the next
inner MMA or the post-loop `bridge` acquire:

```text
inner loop continues

release MMA_READY, full {0} c2 --------------- MMA_READY ---------------+
                                                                         |
EXIT inner(i,j) {1}                                                      |
       | next inner iteration                                            |
       v                                                                 |
ENTER inner(i,j+1) {1}                                                   |
       | walk                                                            |
       v                                                                 |
ready = acquire MMA_READY {1} <------------------------------------------+
       ready |
             v
       next MMA

inner loop finishes

release MMA_READY, full {0} c2 --------------- MMA_READY ---------------+
                                                                         |
EXIT inner(i,last) {1}                                                   |
       | loop finishes                                                   |
       v                                                                 |
bridge = acquire MMA_READY {1} <-----------------------------------------+
       bridge |
              v
release OUTER_EMPTY, bridge [tc5mma] {1} p2
       OUTER_EMPTY |
                   v
post = acquire OUTER_EMPTY {0}
       post |
            v
R acc after inner [post] {0}       (0,4)
       post |
            v
EXIT outer(i) {0}
       post | next outer iteration
            v
ENTER outer(i+1) {0}
       post |
            v
W acc(i+1) [post] {0}
```

For another inner iteration, the `MMA_READY` release supplies the acquire at
that iteration's first MMA. After the final inner iteration, `%bridge`
acquires the same released semaphore and supplies parent edge `p2`.
`entry` is the root acquire before outer iteration zero. The post-loop `{0}`
read keeps its token for each later outer iteration. If the inner loop
executes zero times, `%bridge` consumes the `MMA_READY` release from `p1`, so
parent edge `p2` is still implemented.

Moving the next-iteration acquire from the inner `EXIT` to the first MMA
changes where it appears, so every schedule relation that referred to the
old position is checked again:

```text
the release and moved acquire are in the same region path, and the release
precedes it or the semaphore is an entry semaphore
  use the moved acquire for the schedule comparison

a later acquire of the same semaphore follows the release in that region path
  use the later acquire for the schedule comparison

neither condition holds
  do not add a release-before-acquire cluster constraint
```

Inside the inner loop the resulting semaphore operations are:

```text
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

Discarding an obsolete schedule comparison does not remove the semaphore
release or acquire from IR.

### Branch completion must agree

A loop can carry one owner token through an `if` even when the two paths
finish that owner's work at different schedules. That difference matters
when deciding whether the loop's acquire can move to its first buffer use.

`test/NVWS/insert_semas_nested_carrier.mlir`
`@branch_completion_requires_carrier` has this relevant access pattern for
one TMEM piece P0:

```text
outer for
  W acc {0}

  inner for
    W first acc {1}                 stage 0, cluster 1

    if cond
      then:
        W branch acc {1}            stage 1, cluster 2
        R branch acc {0}            stage 1, cluster 3
      else:
        no acc access

    R final acc {0}                 stage 1, cluster 4

  R post acc {0}
```

The inner loop writes P0 and first accesses it in owner `{1}`, so its parent
summary is `P0:W:{1}`. The outer-level edges are:

```text
parent DAG node                    synchronization edge ending here
ENTER outer(i) {0}                 none
W acc {0}                          none
[inner for P0:W:{1}]               p0: W acc {0} -> inner for
R post acc {0}                     p3: inner for -> R post acc {0}
EXIT outer(i) {0}                  none
```

```text
parent synchronization-edge DAG

                       ENTER outer(i) {0}
                                | walk
                                v
                            W acc {0}
                                | p0
                                v
                     [inner for P0:W:{1}]
                                | p3
                                v
                         R post acc {0}
                                | walk
                                v
                       EXIT outer(i) {0}
```

The `if` writes P0 on its then path, and its first P0 access has owner `{1}`.
Its parent summary is therefore `P0:W:{1}`. At the inner-loop level, the
first write and the summary have the same owner. The final read changes to
owner `{0}`, and `EXIT` returns to boundary owner `{1}`:

```text
inner DAG node                    synchronization edge ending here
ENTER inner(j) {1}                none
W first acc {1}                   none
[if summary P0:W:{1}]             none
R final acc {0}                   e3: if summary -> R final acc {0}
EXIT inner(j) {1}                 e4: R final acc {0} -> EXIT inner(j) {1}
```

The then path changes from `{1}` to `{0}` and back to the `if` boundary.
The empty else path preserves the incoming owner and ordering:

```text
then-path DAG node                synchronization edge ending here
ENTER if {1}                      none
W branch acc {1}                  none
R branch acc {0}                  b1: W branch acc {1} -> R branch acc {0}
EXIT if {1}                       b2: R branch acc {0} -> EXIT if {1}

else-path DAG node                synchronization edge ending here
ENTER if {1}                      none
EXIT if {1}                       none
```

```text
synchronization-edge DAGs

inner-loop level                         if paths

ENTER inner(j) {1}                       ENTER if {1}
         | walk                           +-----+-----+
         v                           then |           | else
 W first acc {1}                         v           v
         | walk                W branch acc {1}   EXIT if {1}
         v                               | b1
[if summary P0:W:{1}]                    v
         | e3                  R branch acc {0}
         v                               | b2
 R final acc {0}                         v
         | e4                       EXIT if {1}
         v
 EXIT inner(j) {1}
```

Parent edge `p0` and child edge `e4` share `BRANCH_READY`: `p0` supplies the
first inner iteration and `e4` supplies later ones. The remaining edges use
four more count-1 semaphores:

```text
edges      semaphore       acquire before         pending_count    initial state
p0, e4     BRANCH_READY    ENTER/EXIT inner {1}   1                false
b1         BRANCH_FULL     R branch acc {0}       1                false
b2         BRANCH_BACK     EXIT if {1}            1                false
e3         FINAL_FULL      R final acc {0}        1                false
p3         BRANCH_EMPTY    R post acc {0}         1                initially released
```

Let `ready` be the owner-`{1}` token acquired from `BRANCH_READY` before the
inner loop. The then path acquires a replacement owner-`{1}` token; the else
path returns `ready` unchanged. The selected token then serves the final
read:

```text
outer and first inner entry

             outer = acquire BRANCH_EMPTY root
                            outer |
                                  v
                       ENTER outer(0) {0}
                            outer |
                                  v
                         W acc [outer] {0}
                                  | walk
                                  v
          release BRANCH_READY, outer {0}       p0
                     BRANCH_READY |
                                  v
               ready = acquire BRANCH_READY {1}
                            ready |
                                  v
                        ENTER inner(0) {1}
```

```text
semaphore DAG for one inner iteration

                  ENTER inner(j) {1}
                           ready |
                                 v
                W first acc [ready] {1}       (0,1)
                                 |
                       +---------+---------+
                then   |                   | else
                       v                   v
       W branch acc [ready] {1}       out = ready
                       |                   |
 release BRANCH_FULL, ready [tc5mma] {1} b1 |
            BRANCH_FULL |                  |
                        v                  |
       btok = acquire BRANCH_FULL {0}      |
                   btok |                  |
                        v                  |
          R branch acc [btok] {0}          |
                        |                  |
       release BRANCH_BACK, btok {0} b2    |
            BRANCH_BACK |                  |
                        v                  |
       returned = acquire BRANCH_BACK {1}  |
               returned |                  |
                        v                  |
           out = returned                  |
                       +---------+---------+
                                 v
        release FINAL_FULL, out [tc5mma] {1}        e3
                      FINAL_FULL |
                                 v
             final = acquire FINAL_FULL {0}
                            final |
                                  v
               R final acc [final] {0}       (1,4)
                                  |
               release BRANCH_READY, final {0}       e4
                     BRANCH_READY |
                                  v
              next = acquire BRANCH_READY {1}
                             next |
                                  v
                    EXIT inner(j) {1}
                             next | next inner iteration
                                  v
                  ENTER inner(j+1) {1}
```

The then path's owner-`{1}` result is `returned`, whose acquire is placed at
stage 1, cluster 2. The else path's result is `ready`, whose physical work in
this iteration is the first MMA at stage 0, cluster 1:

```text
path    returned token    owner-{1} completion schedule
then    returned          stage 1, cluster 2
else    ready             stage 0, cluster 1
```

Because the schedules differ, the inner loop keeps `ready` as an input token
and returns `next`. Moving the `BRANCH_READY` acquire to `W first acc` would
discard the distinct path result needed for scheduling. On the first inner
iteration, the outer write's release supplies `ready`; on later iterations,
`e4` supplies `next`. If the inner loop executes zero times, its input
`ready` is its result. A missing `else` likewise preserves the incoming
schedule on the path that does not enter the body.

After the inner loop, that result implements parent edge `p3`. The acquired
owner-`{0}` token then remains available through the next outer iteration:

```text
result = final next, or ready when the inner loop has zero trips
       result |
              v
release BRANCH_EMPTY, result [tc5mma] {1}       p3
       BRANCH_EMPTY |
                    v
post = acquire BRANCH_EMPTY {0}
       post |
            v
R post acc [post] {0}
       post |
            v
EXIT outer(i) {0}
       post | next outer iteration
            v
ENTER outer(i+1) {0}
       post |
            v
W acc(i+1) [post] {0}
```

Implementation note: `completionAfterChain` records, for each path, whether
the returned token keeps the incoming completion or uses a later owner's
schedule. The acquire moves only when all possible paths agree.

### Post-loop acquires use their owner's boundary

The `@scheduled_relocated_acquire_boundaries` example in
[Moving an acquire updates its schedule relation](#moving-an-acquire-updates-its-schedule-relation)
also shows why an acquire after a nested loop uses the schedule at its own
owner's boundary. `%bridge` has no later owner-`{1}` read or write from which
to copy a schedule. The relevant part of the semaphore DAG is:

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
`(0,4)` schedule. If no later operation provides a schedule, the acquire uses
the greatest `loop.cluster` assigned to any operation with the same owner and
stage.

Root entry acquires remain unscheduled.

### Schedule and stage offset are separate

`loop.stage` and `loop.cluster` determine when an acquire or release executes.
`stageOffset` and `bufferStageOffset` determine which semaphore stage and
buffer copy it uses. EMIT-IR places `bufferStageOffset` on the
`semaphore.buffer` created for an access.

Schedule finalization uses the schedule relations whose release and acquire
still define a loop ordering. It uses loop distance and owner order to decide
whether a cluster must move. The result keeps the release at the source
completion and the acquire at the destination, but may delay the destination
cluster so the release can execute first.

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

Circular members with one physical `buffer.id` are tracked separately for
access ordering. They must agree on type and `buffer.copy`, have unique valid
`buffer.start` values, and be written in the order required by those
`buffer.start` values.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` uses K and V in a two-copy circular buffer
with starts 0 and 1:

```text
K = local_alloc {buffer.id = 301, buffer.copy = 2,
                 buffer.circular, buffer.start = 0}
V = local_alloc {buffer.id = 301, buffer.copy = 2,
                 buffer.circular, buffer.start = 1}

for i
  W K(i) {1}
  W V(i) {1}
  R K(i) {2}
  R V(i) {2}
```

K and V each have one write-to-read edge and one edge that protects the next
write:

```text
member  DAG node          synchronization edge ending here
K       ENTER(i) {1}      none
K       W K(i) {1}        none
K       R K(i) {2}        k1: W K(i) {1} -> R K(i) {2}
K       EXIT(i) {1}       k2: R K(i) {2} -> EXIT(i) {1}

V       ENTER(i) {1}      none
V       W V(i) {1}        none
V       R V(i) {2}        v1: W V(i) {1} -> R V(i) {2}
V       EXIT(i) {1}       v2: R V(i) {2} -> EXIT(i) {1}
```

```text
K synchronization-edge DAG             V synchronization-edge DAG

ENTER(i) {1}                            ENTER(i) {1}
     | walk                                  | walk
     v                                       v
 W K(i) {1}                              W V(i) {1}
     | k1                                    | v1
     v                                       v
 R K(i) {2}                              R V(i) {2}
     | k2                                    | v2
     v                                       v
 EXIT(i) {1}                             EXIT(i) {1}
```

Emission creates one initially released `EMPTY` semaphore and one `FULL`
semaphore for the circular buffer. K and V use separate operations on those
two semaphores, each with `pending_count=1`. Before adding offsets, the
semaphore DAGs are:

```text
edge    semaphore    pending_count    initial state
k1      FULL         1                false
k2      EMPTY        1                initially released
v1      FULL         1                false
v2      EMPTY        1                initially released
```

```text
K semaphore DAG                          V semaphore DAG

ktok = acquire EMPTY {1}                 vtok = acquire EMPTY {1}
          ktok |                                    vtok |
               v                                         v
       W K [ktok] {1}                            W V [vtok] {1}
               | walk                                    | walk
               v                                         v
release FULL, ktok {1}                    release FULL, vtok {1}
          FULL |                                    FULL |
               v                                         v
kr = acquire FULL {2}                    vr = acquire FULL {2}
            kr |                                      vr |
               v                                         v
         R K [kr] {2}                              R V [vr] {2}
               | walk                                    | walk
               v                                         v
release EMPTY, kr {2}                    release EMPTY, vr {2}
               | next K write                            | next V write
               v                                         v
knext = acquire EMPTY {1}                 vnext = acquire EMPTY {1}
         knext |                                    vnext |
               v                                         v
 W K(i+1) [knext] {1}                       W V(i+1) [vnext] {1}
```

The accesses occur in the order shown by the input IR. Their write numbers
and offsets are:

```text
event       current write number   required write number   offset
store K     -1 -> 0                K = 0                   0
store V      0 -> 1                V = 1                   0
load K       1                     K = 0                  -1
load V       1                     V = 1                   0
```

K writes copy 0 and V then writes copy 1, so K's read selects the preceding
copy. The acquire of `FULL` and the release of `EMPTY` receive offset `-1`:

```text
K semaphore path with offsets

ktok = acquire EMPTY[0] {1}
W K [ktok, buffer 0] {1}
release FULL[0], ktok {1}

kr = acquire FULL[-1] {2}
R K [kr, buffer -1] {2}
release EMPTY[-1], kr {2}
```

V stays on the current copy:

```text
V semaphore path with offsets

vtok = acquire EMPTY[0] {1}
W V [vtok, buffer 0] {1}
release FULL[0], vtok {1}

vr = acquire FULL[0] {2}
R V [vr, buffer 0] {2}
release EMPTY[0], vr {2}
```

The shared `EMPTY` semaphore is initially released so iteration zero's first
acquire succeeds and returns a token. There is no buffer use after the loop,
so the final `EMPTY` releases have no later acquires. A zero-trip loop
executes none of these operations and leaves its stages initially released.

### Non-circular aliases

Non-circular members in one group use the same ordered read/write analysis.
An explicit `buffer.copy` makes every member with that `buffer.id` use the
same set of physical copies. Without an explicit copy count, this applies only
to exact aliases whose semaphore copy count exceeds their buffer copy count.
SMEM and TMEM use the same analysis.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@fused_alias_depth_two` has two allocation names for one two-copy SMEM
backing. The input test uses ordinary loads and consumers:

```mlir
%m0 = ttg.local_alloc {buffer.id = 500, buffer.copy = 2}
%m1 = ttg.local_alloc {buffer.id = 500, buffer.copy = 2}

scf.for {
  ttg.local_store %v0, %m0 {partition = 4}
  %r0 = ttg.local_load %m0 {partition = 2}
  "consume0"(%r0) {partition = 2}

  ttg.local_store %v1, %m1 {partition = 4}
  %r1 = ttg.local_load %m1 {partition = 2}
  "consume1"(%r1) {partition = 2}
}
```

Both members read or write the same piece, so the access rules produce one
ordered set of synchronization edges:

```text
DAG node             synchronization edge ending here
ENTER(i) {4}         none
W m0(i) {4}          none
R m0(i) {2}          e1: W m0(i) {4} -> R m0(i) {2}
W m1(i) {4}          e2: R m0(i) {2} -> W m1(i) {4}
R m1(i) {2}          e3: W m1(i) {4} -> R m1(i) {2}
EXIT(i) {4}          e4: R m1(i) {2} -> EXIT(i) {4}
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
                              | next iteration
                              v
                        ENTER(i+1) {4}
```

`e1` uses `FULL0`, `e2` uses `M1_READY`, `e3` uses `FULL1`, and `e4`
uses the initially released `ENTRY` semaphore. Every semaphore has
`pending_count=1`:

```text
edge    semaphore    pending_count    initial state
e1      FULL0        1                false
e2      M1_READY     1                false
e3      FULL1        1                false
e4      ENTRY        1                initially released
```

The semaphore DAG across two iterations is:

```text
semaphore DAG across two iterations

                         ENTER(i) {4}
                                  | walk
                                  v
                     t0 = acquire ENTRY(i) {4}
                               t0 |
                                  v
                         W m0(i) [t0] {4}
                                  | walk
                                  v
                       release FULL0, t0 {4}       e1
                                  | FULL0
                                  v
                    t1 = acquire FULL0(i) {2}
                               t1 |
                                  v
                         R m0(i) [t1] {2}
                                  | walk
                                  v
                    release M1_READY, t1 {2}       e2
                                  | M1_READY
                                  v
                    t2 = acquire M1_READY(i) {4}
                               t2 |
                                  v
                         W m1(i) [t2] {4}
                                  | walk
                                  v
                       release FULL1, t2 {4}       e3
                                  | FULL1
                                  v
                    t3 = acquire FULL1(i) {2}
                               t3 |
                                  v
                         R m1(i) [t3] {2}
                                  | walk
                                  v
                       release ENTRY, t3 {2}       e4
                              ENTRY |
                                    +-----------------------------------+
                                                                        |
                              EXIT(i) {4}
                                  | next iteration
                                  v
                            ENTER(i+1) {4}
                                  | walk
                                  v
                    next = acquire ENTRY(i+1) {4} <---------------------+
                             next |
                                  v
                       W m0(i+1) [next] {4}
```

The emitted shape before `AssignStagePhase` is:

```mlir
%base = ttg.local_alloc {buffer.id = 500, buffer.copy = 2}

%entry   = nvws.semaphore.create %base true
%full0   = nvws.semaphore.create %base false
%m1_ready = nvws.semaphore.create %base false
%full1   = nvws.semaphore.create %base false

scf.for {
  %t0 = nvws.semaphore.acquire %entry[0] {partition = 4}
  %b0 = nvws.semaphore.buffer %entry[0], %t0
  ttg.local_store %v0, %b0 {partition = 4}
  nvws.semaphore.release %full0[0], %t0 {partition = 4}

  %t1 = nvws.semaphore.acquire %full0[0] {partition = 2}
  %b1 = nvws.semaphore.buffer %full0[0], %t1
  %r0 = ttg.local_load %b1 {partition = 2}
  "consume0"(%r0) {partition = 2}
  nvws.semaphore.release %m1_ready[1], %t1 {partition = 2}

  %t2 = nvws.semaphore.acquire %m1_ready[0] {partition = 4}
  %b2 = nvws.semaphore.buffer %m1_ready[0], %t2
  ttg.local_store %v1, %b2 {partition = 4}
  nvws.semaphore.release %full1[0], %t2 {partition = 4}

  %t3 = nvws.semaphore.acquire %full1[0] {partition = 2}
  %b3 = nvws.semaphore.buffer %full1[0], %t3
  %r1 = ttg.local_load %b3 {partition = 2}
  "consume1"(%r1) {partition = 2}
  nvws.semaphore.release %entry[1], %t3 {partition = 2}
}
```

The bracketed numbers are stage offsets, not final stage numbers. If the first
write/read pair uses physical stage `s`, the second pair uses `(s+1) mod 2`.

```text
release/acquire pairs that keep the same copy
  W m0 at s     -> R m0 at s
  W m1 at s+1   -> R m1 at s+1

release/acquire pairs that advance to the next copy
  R m0 at s     -> W m1 at s+1
  R m1 at s+1   -> W m0(i+1) at s
```

The two releases that advance to the next copy receive offset `+1`. Without
it, `%m1_ready` would release stage `s` while its acquire waits on stage
`s+1`; the initially false semaphore would never satisfy that acquire. The
release/acquire pair between iterations requires the same adjustment when the
copy index wraps.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@tmem_fused_alias_depth_two` uses the same ordered read/write analysis for two
non-circular TMEM aliases with explicit `buffer.copy = 2`.

There is no buffer use after the loop. The final `ENTRY` release therefore
has no later acquire, and a zero-trip loop executes none of the shown
operations while leaving `ENTRY` initially released.

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
