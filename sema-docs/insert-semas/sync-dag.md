# SYNC-DAG

## Contents

- [Purpose](#purpose)
- [Notation](#notation)
- [From accesses to raw edges](#from-accesses-to-raw-edges)
  - [Piece state](#piece-state)
  - [The access rules](#the-access-rules)
  - [What a complete raw DAG contains](#what-a-complete-raw-dag-contains)
  - [Example: two-partition handoff](#example-two-partition-handoff)
  - [Example: fan-out and owner-token reuse](#example-fan-out-and-owner-token-reuse)
  - [Example: disjoint pieces stay independent](#example-disjoint-pieces-stay-independent)
  - [Nested regions](#nested-regions)
  - [Example: the same rules at two region levels](#example-the-same-rules-at-two-region-levels)
- [Reducing raw edges](#reducing-raw-edges)
  - [Example: a straight edge is redundant](#example-a-straight-edge-is-redundant)
  - [Example: edge reduction lowers the pending count](#example-edge-reduction-lowers-the-pending-count)
  - [Example: a loop-closing edge is redundant](#example-a-loop-closing-edge-is-redundant)
  - [Exact-source async edges](#exact-source-async-edges)
  - [Release floors](#release-floors)
  - [Reduction does not choose placement](#reduction-does-not-choose-placement)
- [From reduced edges to semaphores](#from-reduced-edges-to-semaphores)
  - [Repeated edges from one sender](#repeated-edges-from-one-sender)
  - [3. Covered senders (`buildEdgesAndSemas`)](#3-covered-senders-buildedgesandsemas)
  - [One destination, one semaphore](#one-destination-one-semaphore)
  - [Reading one semaphore handoff](#reading-one-semaphore-handoff)
  - [Seeding the first token](#seeding-the-first-token)
  - [Entry and recurrence use one semaphore](#entry-and-recurrence-use-one-semaphore)
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
  - [Example: one-copy loop-closing handoff](#example-one-copy-loop-closing-handoff)
  - [Finalizing one handoff](#finalizing-one-handoff)
  - [Moving an acquire repairs its schedule edge](#moving-an-acquire-repairs-its-schedule-edge)
  - [Post-loop acquires use their owner's boundary](#post-loop-acquires-use-their-owners-boundary)
  - [Schedule and stage offset are separate](#schedule-and-stage-offset-are-separate)
- [Authored buffer-stage offsets](#authored-buffer-stage-offsets)
  - [Circular groups](#circular-groups)
  - [Non-circular alias handoffs](#non-circular-alias-handoffs)
- [Build order and code map](#build-order-and-code-map)

## Purpose

SYNC-DAG turns buffer access order into a plan for
`nvws.semaphore.acquire`, `nvws.semaphore.release`, and token operations. It
also decides how tokens pass through `for` and `if`, how many buffer and
semaphore copies exist, and where the new protocol operations belong in a
scheduled loop. EMIT-IR later renders this plan and creates each
`nvws.semaphore.buffer` at its access.

The construction order is important:

```text
complete raw edges
    -> choose physical backing copies
    -> remove safely redundant edges
    -> merge repeated edges from one sender
    -> group remaining edges by destination
    -> assign a semaphore and pending count
    -> seed entry tokens and plan region tokens
    -> choose semaphore copies
    -> finalize stage offsets and schedules
```

A pending count never proves that a raw edge is redundant. The pending count
is calculated only after raw-edge reduction and merging.

The input [access DAG](access-dag.md#regions-and-boundaries) already contains:

- one node for each buffer access;
- an owner for each access;
- a disjoint piece table for overlapping allocations; and
- `ENTER` and `EXIT` nodes for each `for` and `if` path.

SYNC-DAG adds raw edges between those concrete nodes. A raw edge says that the
destination owner must wait for the source owner. Semaphore operations are
the IR form of the reduced edges; they are not a second ordering analysis.

## Notation

The examples use explanatory pseudo-IR and three kinds of diagrams. Each
diagram stays at one level:

```text
raw DAG       access, region, ENTER, and EXIT nodes joined by e1, e2, ...
reduced DAG   the same nodes after named raw edges have been removed
protocol DAG  acquire, access, and release nodes joined by S0, S1, ...
```

`walk` marks program order when no raw edge exists. A loop-closing raw edge is
stored as `source -> EXIT(i)`. When a diagram carries it to an access in the
next iteration, the diagram says that it is logically unrolled.

Names used throughout:

```text
group          allocations analyzed together, ordinarily one buffer.id
backing        the physical SMEM or TMEM allocation guarded by the group
m0, m1         members: allocation names or views in the group
P0, P1         disjoint pieces of the backing
{0}, {1}       owners: partitions 0 and 1 of the enclosing WS loop
root           code with no partition owner
source         node that supplies the current value to a new reader
use            latest access to the current value by one owner
token          value returned by an acquire and consumed by releases/buffers
```

Pseudo-IR omits types and unrelated attributes:

```text
%t = acquire S0 {1}
%b = semaphore.buffer S0, %t
R m0 [%b] {1}
release S1, %t {1}
for iter_args(%t = %entry) { ... yield %next }
```

An edge may carry one or more completion kinds. `[none]` completes when the
release executes. `[tma_load]` and `[tc5mma]` complete with the named async
operation. Each completion kind contributes one arrival. An edge with no
async kind contributes one arrival through `[none]`.

## From accesses to raw edges

`ChainWalker` walks one group in program order. It keeps independent state for
each piece and a deterministic list of available owner tokens for each chain.
The walk records every required raw edge; it does not suppress an edge merely
because another live use may cover its ordering.

### Piece state

For each piece, `PieceState` contains:

```text
source    producer, source owner, source node, and completion kinds
uses      [owner -> latest node for the current value, ...]
```

A read moves only that owner's use. It does not move the source. New readers
therefore fan out from the write, first touch, or `ENTER` that established the
value. A write starts a new value and resets `uses` to that writer.

The chain also keeps available tokens in deterministic handoff order:

```text
tokens = [{0} at W0, {1} at R1, ...]
```

The last source-bearing token can supply a token-only edge when the memory
rules add no edge. An owner can instead reuse its earlier token when every
touched piece proves that reuse is safe. `Node::reuseTokenOwner` records that
proof for emission.

A uniform `ENTER` seeds its owner's token without a source node. The owner may
reuse it immediately, but it cannot source a token-only edge until an access
records a concrete node.

### The access rules

`applyTouch` applies these rules to every touched piece:

```text
first touch
  raw edges: none
  state: source = this node; uses = [owner -> this node]

write by owner P
  raw edges: latest use -> this write for every other owner,
             unless that use is already known before P, or a WS region
             summary is adopting a root-held use
  state: source = this write; uses = [P -> this write]

read by P when P is already in uses
  raw edges: none
  state: replace P's use with this read; source does not move

read by a new owner P
  raw edge: source -> this read, unless a WS region adopts a root source
  state: add P -> this read to uses
```

After all pieces have advanced, `visitAccess` chooses the token:

```text
memory edge was added
  the destination acquire supplies the token

no memory edge and P can reuse an earlier token
  mark the access with reuseTokenOwner = P

no memory edge, no reusable P token, and another owner has the last token
  add a token-only raw edge from that token's node

otherwise
  the token already available to P supplies the access
```

For a read, reuse requires a live use for the owner on every touched piece.
For a write, every other live use on every touched piece must already be
ordered before the writer.

In a multi-member group, a later synchronous write may retain earlier async
completion kinds only when the owner can safely reuse its token. That keeps a
same-owner ownership wave intact and prevents the later write from hiding an
earlier descriptor load.

### What a complete raw DAG contains

The raw DAG is complete when every access can answer two questions from its
incoming state: which earlier owner makes the buffer value available, and
which token lets this owner name the guarded backing. Usually one memory edge
answers both questions. A token-only edge answers only the second.

Completeness is checked per piece. An access that spans two pieces can need
two raw edges even when both edges have the same endpoint nodes. Those facts
remain separate through the walk because the pieces can diverge later. They
are combined only after reduction, when one release from a sender can satisfy
the destination for both pieces.

The walk also records edges that are already ordered through another live
use. That is intentional. At this point the edge still records a real buffer
obligation, its source still constrains release placement, and a later
reduction still has to prove that the alternate path survives.

No raw edge is needed merely because two operations appear next to each
other. Same-owner accesses normally share their current token and follow
program order. A raw edge appears when ownership changes, when another owner
must finish using a piece before a write, or when token ownership must move
despite there being no new memory dependency.

Region boundaries do not introduce a different rule set. `ENTER` supplies
the incoming value inside a child path, and `EXIT` names an obligation that
must leave that path. The same source, live-use, and token decisions apply to
the concrete accesses between those nodes.

This complete set is deliberately larger than the emitted protocol. The
following sections first remove edges whose obligations are proved by kept
paths, then combine repeated facts into the acquire and releases that the IR
actually needs.

### Example: two-partition handoff

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` stores in
owner `{0}` and loads in owner `{1}` on every iteration.

Raw-edge inventory:

```text
nodes in walk order
  N0 = ENTER(i) {0}
  N1 = W store(i) {0}
  N2 = R load(i) {1}
  N3 = EXIT(i) {0}

generated raw edges
  e1: W store(i) {0} -> R load(i) {1}       read after write
  e2: R load(i) {1} -> EXIT(i) {0}          next write waits for this read
```

The raw DAG stores the close at `EXIT(i)`:

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

After reduction, merging, grouping, and semaphore assignment, `e1` becomes
`FULL` and `e2` becomes initially released `EMPTY`:

```text
%empty = acquire EMPTY {0}
W store [%empty] {0}
release FULL, %empty {0}

%full = acquire FULL {1}
R load [%full] {1}
release EMPTY, %full {1}
```

The acquire of `EMPTY` is placed at the first store inside the loop. The
initially released state supplies iteration zero; each `release EMPTY`
supplies the next iteration.

```text
protocol DAG, logically unrolled across one iteration boundary

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

### Example: fan-out and owner-token reuse

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced` writes in `{0}`, reads in `{1}` and `{2}`, then rereads
in `{0}`.

```text
walk node       generated raw edge                 state after node
ENTER {0}       none                               source=ENTER; uses={0}:ENTER
W alloc {0}     none                               source=W; uses={0}:W
R load {1}      f1: W {0} -> R {1}                source=W; uses={0}:W,{1}:R1
R load {2}      f2: W {0} -> R {2}                source=W; uses={0}:W,{1}:R1,{2}:R2
R load {0}      none; reuse {0}'s token            source=W; uses={0}:R0,{1}:R1,{2}:R2
EXIT {0}        f3: R1 {1} -> EXIT {0}
                f4: R2 {2} -> EXIT {0}
```

The raw DAG has real fan-out. `R {1}` and `R {2}` are not ordered with
respect to each other by a raw edge. The `{0}` reread also has no incoming
raw edge; it appears only in program order.

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
                                           | loop
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
                                           | loop
                                           v
                                    ENTER(i+1) {0}
```

`f3` and `f4` remain separate senders into one destination. They become two
releases to one count-2 semaphore:

```text
release EMPTY, %reader1 {1}
release EMPTY, %reader2 {2}
%next = acquire EMPTY pending_count=2 {0}
```

The final `{0}` buffer uses the earlier `{0}` token marked by
`reuseTokenOwner`; emission does not infer that reuse independently.

### Example: disjoint pieces stay independent

The following conceptual `spanning_split_parallel` shape uses `m0` for the
first half, `m1` for the second half, and `m2` for the full backing. It is a
compact teaching example, not the name of an in-tree test:

```text
members:    m0[0,128)   m1[128,256)   m2[0,256)
pieces:     P0=[0,128){m0,m2}   P1=[128,256){m1,m2}
footprints: m0={P0}   m1={P1}   m2={P0,P1}
```

The full-width write/read establishes the common prefix. `applyTouch` records
one raw edge per piece, so the two edges with the same full-width endpoints
are both present before raw-edge reduction:

```text
raw-edge inventory
  d1a: W m2 {0} -> R m2 {1}       P0
  d1b: W m2 {0} -> R m2 {1}       P1

  d2a: W m2 {0} -> W m0 {2}       P0, coveredVia {1}
  d2b: R m2 {1} -> W m0 {2}       P0
  d3:  W m0 {2} -> R m0 {3}       P0

  d4a: W m2 {0} -> W m1 {4}       P1, coveredVia {1}
  d4b: R m2 {1} -> W m1 {4}       P1
  d5:  W m1 {4} -> R m1 {0}       P1

  d6: W m0 {2} -> EXIT(i) {0}     P0, coveredVia {3}
  d7: R m0 {3} -> EXIT(i) {0}     P0
  d8: W m1 {4} -> EXIT(i) {0}     P1, coveredVia {0}
```

The two piece DAGs expose every candidate without drawing false dependency
between the half-width paths:

```text
raw DAG for P0

                             W m2 {0}
                        +--------+--------+
                     d2a|                 |d1a
                        |                 v
                        |             R m2 {1}
                        |                 |d2b
                        +--------+--------+
                                 v
                             W m0 {2}
                        +--------+--------+
                      d6|                 |d3
                        |                 v
                        |             R m0 {3}
                        |                 |d7
                        +--------+--------+
                                 v
                            EXIT(i) {0}

raw DAG for P1

                             W m2 {0}
                        +--------+--------+
                     d4a|                 |d1b
                        |                 v
                        |             R m2 {1}
                        |                 |d4b
                        +--------+--------+
                                 v
                             W m1 {4}
                        +--------+--------+
                      d8|                 |d5
                        |                 v
                        |             R m1 {0}
                        |                 |walk
                        +--------+--------+
                                 v
                            EXIT(i) {0}
```

Straight reduction keeps `d1a`, removes same-endpoint duplicate `d1b`,
removes `d2a` through `d1a -> d2b`, and removes `d4a` through
`d1a -> d4b`. The loop-close reducer retains `d6`, `d7`, and `d8` because
they return to the first owner `{0}`. Covered-sender pruning then removes
`d6` through `d3 -> d7` and removes `d8` through `d5` followed by
destination-owner `{0}`'s program order. Only `d7` supplies the recurrence
arrival.

The surviving `d1a` is the one full-width handoff. The P0 and P1 paths
otherwise remain separate. The P1-only branch after `R m2` neither acquires
nor releases recurrence semaphore `EMPTY`. `EMPTY` is group-scoped, although
its sole surviving close `d7` came from P0:

```text
reduced fact   protocol semaphore
d1a            FULL_BOTH
d2b            LEFT_READY
d3             LEFT_FULL
d4b            RIGHT_READY
d5             RIGHT_FULL
d7             EMPTY, initially released
```

```text
%both = acquire EMPTY pending_count=1 {0}
W m2 [%both] {0}
release FULL_BOTH, %both {0}

%read_both = acquire FULL_BOTH {1}
R m2 [%read_both] {1}
release LEFT_READY,  %read_both {1}
release RIGHT_READY, %read_both {1}

%left = acquire LEFT_READY {2}
W m0 [%left] {2}
release LEFT_FULL, %left {2}

%left_read = acquire LEFT_FULL {3}
R m0 [%left_read] {3}
release EMPTY, %left_read {3}

%right = acquire RIGHT_READY {4}
W m1 [%right] {4}
release RIGHT_FULL, %right {4}

%right_read = acquire RIGHT_FULL {0}
R m1 [%right_read] {0}
```

### Nested regions

A region is one summarized access in its parent chain and contains separate
child chains:

```text
parent chain                     child chain

... -> [for or if summary] ...   ENTER -> child nodes -> EXIT
```

The parent applies the ordinary read/write rules to the region summary. Each
child path then starts fresh piece state at its `ENTER` owner. A new child
reader receives its raw edge from `ENTER`, not from a concrete parent node.
`ENTER` itself emits no acquire or release.

The child's logical producer is inherited from the incoming value. When that
producer is also the `ENTER` owner, the child imports its async completion
kinds. Otherwise the `ENTER` source begins with `[none]`. This changes the
completion carried by a raw edge without changing its endpoints.

After the child walk, `EXIT` closes every foreign live use when the piece may
be needed by another loop iteration or after the region. The parent resumes
from the state produced by its region summary; child uses never replace the
parent's uses.

The region also resets the parent chain's available-token list. A summary
with one partition owner records that owner's token at the region node. A
mixed-owner or root-owned summary records no partition token, so a later
access may need a token-only edge even when it needs no memory edge.

Ordering learned inside an `if` is exported only when every path establishes
it. An absent `else` contributes the unchanged incoming ordering. A loop that
may execute zero times also contributes its bypass path, so its body alone
cannot establish ordering outside the loop.

### Example: the same rules at two region levels

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` contains an outer write by `{3}`
and an inner loop summarized as a write by `{2}`.

Parent raw-edge inventory:

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
child raw-edge inventory
  c1: ENTER inner(i) {2} -> R m0 {1}
  c2: R m0 {2}          -> W m0 {1}
  c3: W m0 {1}          -> R m0 {0}
  c4: W m0 {1}          -> EXIT inner(i) {2}
  c5: R m0 {0}          -> EXIT inner(i) {2}
```

`R m0 {2}` rereads the incoming value and moves only `{2}`'s use. Therefore
the later write by `{1}` waits for that reread through `c2`. `R m0 {1}` does
not source `c2`: the write is by the same owner `{1}`.

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

Only `c1` through `c5` are raw edges. The two `walk` lines are program order.
The parent and child piece states remain separate. Semaphore sharing between
`p1` and the child closes is explained in
[Entry and recurrence use one semaphore](#entry-and-recurrence-use-one-semaphore).

## Reducing raw edges

The walk produces a complete edge set. `reduceEdges` then applies two
different reductions:

- `reduceStraightEdges` handles edges between access nodes in one chain.
- `reduceLoopCloses` handles access-to-`EXIT` edges whose alternate path
  crosses the loop boundary.

Both reducers use only already-kept edges and program order. A dropped edge
does not update their known ordering or token state. A kept edge updates the
state immediately and is never reconsidered. Later decisions therefore
cannot invalidate an earlier proof.

### Example: a straight edge is redundant

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members:

```text
members:    m0[0,256)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0,m1}   P2=[192,256){m0}
```

On P1, owner `{0}` writes, owner `{1}` reads, and owner `{2}` writes. The
second write sees live uses for both `{0}` and `{1}`.

```text
raw-edge inventory
  e1: W m0 {0} -> R m0 {1}
  e2: W m0 {0} -> W m1 {2}       candidate
  e3: R m0 {1} -> W m1 {2}
```

```text
raw DAG

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

`e1 -> e3` orders the same endpoints as `e2`. Kept edge `e3` also supplies
owner `{2}`'s token at the destination. `reduceStraightEdges` drops `e2`.

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

The straight reducer does not consider root, region, `ENTER`, or `EXIT`
endpoints. It also keeps a candidate when the alternate path does not supply
the destination owner's token.

### Example: edge reduction lowers the pending count

The same example shows why reduction precedes semaphore creation. If all
three raw edges became arrivals, the write by `{2}` would wait for releases
from both `{0}` and `{1}`. After `e2` is removed, only `e3` enters that write:

```text
raw edges
  e1, e2, e3

after straight reduction
  e1, e3

after grouping
  e1 -> S0 pending_count=1 at R m0 {1}
  e3 -> S1 pending_count=1 at W m1 {2}
```

```text
protocol DAG

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

The pending count is one because one arrival remains. It is not a reason for
dropping `e2`.

### Example: a loop-closing edge is redundant

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` uses two partly overlapping members:

```text
members:    m0[0,128)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,128){m0,m1}   P2=[128,192){m1}
```

The complete in-body raw inventory is:

```text
l1a: W m0 {0} -> R m0 {1}       P0
l1b: W m0 {0} -> R m0 {1}       P1
l2a: W m0 {0} -> W m1 {2}       P1, coveredVia {1}
l2b: R m0 {1} -> W m1 {2}       P1
l3a: W m1 {2} -> R m1 {0}       P1
l3b: W m1 {2} -> R m1 {0}       P2
```

The walk then records three closes:

```text
c0: R m0(i) {1} -> EXIT(i) {0}   P0 returns to first owner {0}
c1: W m1(i) {2} -> EXIT(i) {0}   P1, coveredVia {0}
c2: R m1(i) {0} -> EXIT(i) {2}   P2 returns to first owner {2}
```

`c1` starts at `W m1`, not `R m0`: that write begins a new P1 version. Its
source use is already ordered before the P1 read by destination owner `{0}`.

Straight reduction keeps `l1a` and `l3a`, drops their same-endpoint
duplicates `l1b` and `l3b`, and drops `l2a` through `l1a -> l2b`.

For the loop-close proof, the reducer input below is logically unrolled. The
stored `c2` edge ends at `EXIT(i)`; the picture carries destination owner
`{2}` to its first P2 access, `W m1(i+1)`.

```text
loop-close reducer input after straight reduction, logically unrolled

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
available there. `{2}` is not the owner of the chain's first
partition-owned access, so `reduceLoopCloses` drops `c2`.

```text
after loop-close reduction, logically unrolled

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

Both `c0` and `c1` target `{0}`, the first access owner, so the loop-close
reducer retains them. Covered-sender pruning later removes `c1`: `l3a` hands
P1 from `{2}` to destination owner `{0}`, whose own program order carries it
to `EXIT`. `c0` remains as the one recurrence arrival.

The two deletion steps use different facts. The loop-close reducer needs the
first access owner, the first next-iteration touch of each piece, and token
availability across the boundary. Covered-sender pruning needs a surviving
two-leg handoff path into this destination.

### Exact-source async edges

An exact-source edge with an async completion does more than impose ordering.
Its release carries the completion of the operation that produced the value.
`EdgeRec::preserve` protects such an edge in both raw-edge reducers. It is not
a guard against later covered-sender pruning.

```text
raw-edge inventory
  a1: W async {0} -> R {1}       carries [tma_load]
  a2: W async {0} -> W {2}       exact source, carries [tma_load],
                                  coveredVia {1}
  a3: R {1}       -> W {2}
```

```text
raw DAG

                          W async {0}
                    +-----------+-----------+
     a1 [tma_load]  |                       | a2 [tma_load], coveredVia {1}
                    v                       |
                  R {1}                     |
               a3   |                       |
                    +-----------+-----------+
                                v
                              W {2}
```

The graph fact is that `a1 -> a3` orders the endpoints of `a2`. Because `a2`
also carries the exact producer's async completion, `preserve` keeps it
through `reduceStraightEdges` and `reduceLoopCloses`; the raw DAG is unchanged
by those reducers.

In this particular shape, `a2` is also `coveredVia {1}`. Covered-sender
pruning can remove its whole owner-`{0}` arrival because `a1` first carries
the TMA-load completion to `{1}`, and surviving `a3` carries the obligation
from `{1}` to `{2}`. If either leg did not survive, `a2` would remain as a
release. A later synchronous reread is not the exact version source and does
not receive `preserve` in the first place.

### Release floors

Before either reducer runs, `buildEdgesAndSemas` records the latest source for
every `(destination, destination owner, sender owner)` tuple. That node is the
sender's release floor.

Consider two raw edges from `{1}` into the same destination:

```text
edge  exact endpoints
r1    A {1} -> D {2}
r2    B {1} -> D {2}      B follows A in the same chain
```

The release floor is `B`, even if reduction later removes `r2`. When the
surviving `{1}` edges are merged, their release is placed no earlier than B:

```text
protocol placement after semaphore assignment

A {1} --walk--> B {1} --walk--> release S {1} --S--> acquire S {2}
                                                       |
                                                     walk
                                                       v
                                                     D {2}
```

This gives reduction a placement-invariance rule:

- reduction may remove a whole arrival;
- reduction may remove a whole semaphore;
- a surviving sender's release cannot move earlier.

The rule matters for async and warp-group work: a release after the sender's
last write must not move into the dependency path of an earlier read merely
because one raw edge became redundant.

### Reduction does not choose placement

Reduction and placement answer different questions. Reduction asks whether a
kept path supplies the ordering and destination token that a candidate edge
would have supplied. Placement asks where the remaining sender may release
that token without moving before any source fact recorded for the same
destination.

The complete raw set answers the placement question before either reducer
runs. `releaseFloors` keeps the latest source for each destination,
destination owner, and sender owner. The reducers may then remove facts from
the edge set, but they do not recompute that floor from only the survivors.

This separation is visible in three valid outcomes. A redundant piece edge
can disappear while another edge from the same sender keeps one release. All
edges from one sender can disappear and remove that arrival. If every arrival
to a destination disappears, the acquire and semaphore can disappear too.
Only the first outcome has a surviving release, and that release stays at or
after its floor.

Merging has a narrower purpose. It combines the surviving facts from one
sender into one arrival and unions their completion kinds. It does not prove
an ordering path, move a release earlier, or combine releases from different
owners. Grouping then collects those distinct arrivals at one destination and
derives the pending count.

Keeping these questions separate makes the protocol readable from the merged
handoffs: each merged sender is one real release, each destination is one
real acquire, and placement does not depend on which redundant piece fact was
chosen as the representative.

## From reduced edges to semaphores

`buildEdgesAndSemas` processes the reduced edge set in this order:

```text
1. merge edges with the same destination, destination owner, and sender owner
2. group the merged arrivals by destination node and destination owner
3. remove covered arrivals whose complete proof survives
4. reconcile loop-entry and recurrence pending counts
5. create one semaphore for each remaining destination group
6. insert one acquire at the destination and releases at sender floors
```

### Repeated edges from one sender

Two pieces can create edges from different nodes of the same owner into the
same `EXIT`. Use this conceptual loop:

```text
W m0 {0}        touches P0 and P1
R m0 {1}        latest {1} use for P0
R m1 {1}        later {1} use for P1
EXIT(i) {0}     returns both pieces to {0}
```

Raw-edge inventory:

```text
m1a: W m0 {0} -> R m0 {1}       P0
m1b: W m0 {0} -> R m0 {1}       P1
m2: R m0 {1} -> EXIT(i) {0}      P0
m3: R m1 {1} -> EXIT(i) {0}      P1
```

```text
raw DAG

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

Straight reduction keeps `m1a` and drops same-endpoint duplicate `m1b`.
Neither loop-close fact is removed. The reduced raw set is therefore `m1a`,
`m2`, and `m3`.

`m2` and `m3` have the same sender and destination. The later source is
`R m1 {1}`, so one merged arrival is sufficient. Completion kinds from both
edges are unioned.

```text
same-sender merge

destination       EXIT(i) {0}
sender owner      {1}
release floor     R m1 {1}
represented facts m2, m3
merged arrival    M1: R m1 {1} -> EXIT(i) {0}
```

One merged edge with `[none]` contributes one arrival. A merged edge with
`[none, tma_load]` contributes two arrivals, so its acquire has pending count
2 even though there is one sender owner.

### 3. Covered senders (`buildEdgesAndSemas`)

The access walk records a `coveredVia` owner when both of the following are
true:

1. a foreign live use is also the current version source;
2. an existing source edge already orders that use before another live use.

This annotation only creates a candidate. After edges are grouped by
destination, `buildEdgesAndSemas` separately proves that the later live use
still carries the obligation to this destination.

Every recorded cover must satisfy two legs in the surviving handoffs. For
leg 2, the covering owner must still release to this destination, unless it
is itself the destination owner and its program order carries the obligation.
For leg 1, a surviving handoff from the covered sender to the covering owner
must release at or after the covered source and be acquired no later than the
covering owner's contribution to this destination. Forward checks may cross
enclosing `for` boundaries, but never one `if` path into another.

The raw edge is still recorded. It participates in both reducers, sets its
sender's release floor, and joins same-sender merging.

The inner loop in
`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` has this closing shape. Owner
`{1}` corrects the buffer in place; owner `{0}` reads the corrected value;
owner `{2}` regains it at `EXIT`.

```text
raw-edge inventory
  k1: W correct {1} -> R corrected {0}
  k2: W correct {1} -> EXIT inner(i) {2}   coveredVia {0}
  k3: R corrected {0} -> EXIT inner(i) {2}
```

```text
raw DAG

                           W correct {1}
                       +---------+---------+
                    k1 |                   | k2 coveredVia {0}
                       v                   |
                 R corrected {0}           |
                    k3 |                   |
                       +---------+---------+
                                 v
                        EXIT inner(i) {2}
```

Both closes are protected from the loop-close reducer because `{2}` is the
first access owner. Covered-sender pruning works at arrival granularity after
grouping:

- leg 1: the surviving `{1} -> {0}` handoff `k1` is released after the
  correction and acquired before the `{0}` read;
- leg 2: `{0}` still arrives at this destination through `k3`.

The whole `{1}` arrival `k2` can therefore disappear. The `{0}` arrival
remains:

```text
reduced DAG after covered-arrival pruning

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
%corrected = acquire CORRECTED {0}
R corrected [%corrected] {0}
release READY, %corrected {0}
%regain = acquire READY pending_count=1 {2}
```

If a sender has any uncovered merged edge into the destination, the sender
stays. Its release remains at the pre-reduction floor. Covered pruning never
removes only an early edge of a surviving sender.

Cover proofs are checked against the surviving set. For example:

```text
raw-edge inventory into D
  f1: A {0} -> B {1}
  f2: B {1} -> C {2}
  f3: A {0} -> D {3}     coveredVia {1}
  f4: B {1} -> D {3}     coveredVia {2}
  f5: C {2} -> D {3}     uncovered
```

```text
raw DAG

                              A {0}
                       +---------+-----------------+
                    f1 |                           | f3 coveredVia {1}
                       v                           |
                     B {1}                         |
                       +---------+-----------------+
                    f2 |                           | f4 coveredVia {2}
                       v                           |
                     C {2}                         |
                  f5   |                           |
                       +------------+--------------+
                                    v
                                  D {3}
```

`f4` may be removed because `{2}` survives through `f5` and `f2` certifies
`B -> C`. `f3` cannot rely on `{1}`'s arrival after `f4` is removed, so `f3`
is retained. The candidate set is repeatedly shrunk until every remaining
candidate has both legs through surviving arrivals. An `if` branch cannot
certify another path; forward proof may climb only through enclosing `for`
regions that actually raised the obligation.

Finally, loop entry and recurrence may share one semaphore. Covered pruning
is optional, so pruning is undone when it would make the two sites unable to
use one fixed pending count. The useful cases are:

```text
entry arrivals   recurrence arrivals   result
1 generic        2                     entry release uses arrive_count=2
2                2                     both sites use pending_count=2
2                1 after pruning       restore all covered arrivals at both sites;
                                       recheck counts
```

A handoff with no surviving arrivals disappears entirely.

### One destination, one semaphore

After reduction and merging, every destination node and destination owner
receive one semaphore and one acquire. Every remaining sender receives one
release. The pending count is:

```text
sum over remaining senders of max(1, number of completion kinds)
```

This means:

- two `[none]` senders give pending count 2;
- one sender with `[none, tma_load]` gives pending count 2;
- two raw piece edges merged into one `[none]` sender give pending count 1.

The release is placed immediately after its source floor, after any earlier
releases there. An async source uses its physical completion point. The
acquire is placed before the destination, except for a safe loop-close
placement at that owner's first touch.

### Reading one semaphore handoff

Start at an acquire and work backward. Its destination node and destination
owner define the handoff. Every release of that semaphore is one surviving
sender arrival, and the pending count is the total arrival contribution of
those releases. Releases from different owners remain separate even when
they guard the same buffer value.

Then work forward from the acquire. Its token names the guarded backing for
the destination access. `semaphore.buffer` converts that token into the buffer
operand used by the access. The same token can feed later releases by the
destination owner until another acquire replaces it or region emission
returns it through `for` or `if`.

The semaphore name describes the handoff, not the allocation member that
happened to create an edge. Overlapping members in one group can therefore
use the same semaphore when their reduced edges have the same destination and
owners. Disjoint paths keep separate semaphores because their destination
nodes or owners differ.

An async completion belongs on the release that represents its source. It
adds an arrival to the same acquire; it does not create a second acquire. A
release with `[none, tma_load]` consequently arrives twice, while a plain
release arrives once.

At this construction point, entry and recurrence sharing can make two
acquire sites in one logical group name the same semaphore. The shared name
is required by the loop token channel, and both sites must agree on one fixed
pending count. Token planning can add another acquire of that semaphore, and
EMIT-IR can fold matching circular semaphores from separate logical groups.

There is no later arbitrary re-reduction or merge of these handoffs. Token
planning may detach or move an acquire, add a post-loop acquire, or add bridge
and close releases so a token can cross a region boundary. Those operations
use the established semaphore and pending count. Schedule finalization may
then change clusters or stage offsets without recomputing the reduced edges.

### Seeding the first token

`insertEntryAcquires` gives the group a token before its first placement node.
The acquire is outside the first `for` when that loop is the first node. Its
IR owner is `root`, while `entryTokenOwner` records the first access owner for
token checks and emission.

The pass scans placement nodes from last to first. When the latest placement
loop containing an acquire has a recurrence semaphore, the entry acquire
uses the loop body's last acquire semaphore and marks it initially released:

```text
%first = acquire READY root
for iter_args(%token = %first) {
  ...
  %next = acquire READY {1}
  yield %next
}
```

When no recurrence channel can seed the group, the pass creates a dedicated
count-1 entry semaphore. Its acquire guards the first placement, and its
release follows the last placement:

```text
%entry = acquire ENTRY root
... first access or region ...
... last access or region ...
release ENTRY, %last_token
```

If the only first placement is an `if` with one non-empty branch, placement
can descend into that branch. A synchronized group with no access or placement
node is an error.

### Entry and recurrence use one semaphore

A loop body has one fixed semaphore operand for a carried buffer token. Its
initial token and next-iteration token must therefore come from the same
semaphore.

Using the two-level example above, the parent edge `p1` supplies the first
inner iteration. Child edge `c5` supplies later iterations after covered
arrival `c4` has been removed:

```text
acquire site                         reduced raw edges   semaphore   count
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

When recurrence has count 2 and entry has one generic `[none]` edge, the
single entry release uses `arrive_count=2`:

```text
release READY, %outer [none] arrive_count=2
%first = acquire READY pending_count=2

... inner body ...

release READY, %reader1 [none] arrive_count=1
release READY, %reader2 [none] arrive_count=1
%next = acquire READY pending_count=2
```

Only a lone generic arrival can be scaled this way. Other mismatches are
errors unless restoring covered arrivals makes the shared count uniform.

## Tokens through `for` and `if`

After semaphore placement, a region may receive a token at its entry and
produce a token on each exit path. `RegionFlow` records only what emission
needs:

```text
owner          owner of the token at the region boundary
exits          final token-producing node on each path;
               null means that path passes the input token through
concreteSema   semaphore used when a result cannot inherit an input channel
```

Region results are planned from inner regions to outer regions. The parent
treats a finished child as one token-producing node; it does not inspect the
child operations again.

For a loop, the pass chooses between two observable IR shapes:

```text
carry the token
  %result = for iter_args(%token = %entry) { ... yield %next }

acquire at the first buffer use
  for { %token = acquire S; ... }
```

A final nested loop can also provide the needed token, so an outer loop may
need no token result of its own.

### Region results

`summarizeRegionFlow` finds the final acquire or token-producing child on
each path. All live paths must return the same owner. A path with no such node
passes the incoming token when the region has a uniform entry owner.

Emission selects one semaphore for the region result. A loop keeps its input
semaphore. An `if` normally does the same; when the input is an unpartitioned
entry acquire, a concrete semaphore from a live path can name the result.
Without a usable input semaphore, the concrete semaphore names it. This
allows:

- both branches to return fresh tokens;
- one branch to return a fresh token while the other passes the input; and
- an `if` without `else` to pass the input on its implicit path.

A pass-through path without an input token is not a valid token result.
`pruneDeadIfFlows` removes an `if` token result when no buffer, release, or
token-producing child uses it before the next acquire, provided the enclosing
region does not itself retain a token result.

A child is transparent to its parent only when it enters and returns the same
owner and no buffer access, release, or token-producing child follows its
final returned token on any path. Such a child can provide the parent's final
token without exposing its internal operations.

### Example: an `if` returns one owner's token

`test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token` has a buffer owned by `{1}` around an
`if`. The then path lends it to `{0}` and returns it; the else path does
nothing.

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

The SYNC-DAG token plan is shown below. EMIT-IR can subsequently split this
shape into scheduler-safe release, body, and acquire conditionals, as shown in
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

The region boundary owner is `{1}` on both paths. The then path returns a
fresh token; the else path returns the input token.

### Moving an acquire to its first use

`planLoop` can remove a loop token when the final token of iteration `i` is
needed first at one buffer access in iteration `i+1`. The acquire moves to
that first access.

`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use` begins with this
balanced carried protocol:

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

Nothing uses `%next` after its acquire. The first demand is the next
iteration's `{0}` write, so the loop becomes:

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

The move is allowed only when the pass can match one input token, one final
token, and one first demand without crossing another acquire, release, or
incompatible region. In particular:

- the loop is inside an eligible tagged WS scope and not hidden behind an
  enclosing `if` boundary;
- the final node is an acquire or a child that returns the boundary owner;
- no buffer access, release, or token-producing child follows it;
- the incoming token has the same owner and remains available;
- the first demand is unambiguous and has the needed closing release;
- a retained buffer use before the loop requires an authored single copy;
- a plain loop with a sourceful TMEM allocation does not use this move; and
- any required post-loop token has a compatible stage.

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
emitted token path, logically unrolled across the loop backedge

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

The count-2 `EMPTY` semaphore is the real fan-in from readers `{1}` and
`{2}`. No extra boundary semaphore is folded into it.

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
synthetic acquire so schedule finalization uses the correct owner boundary.

If a final child supplies the loop token and that token is also consumed
after the loop, the loop retains the token instead of taking it away from the
child result.

### Nested loops

Region planning runs from inner loops to outer loops. In
`test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop`, the inner loop acquires at its first MMA. The outer
loop only forwarded that token, so it needs no token slot:

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
loop. The inner loop still acquires at its first MMA. A post-loop acquire and
release connect the inner protocol to the outer read. A final acquire and
release in the outer body return the permit to the next inner loop. Neither
loop needs a semaphore token slot:

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

The root acquire seeds the outer cycle without becoming a loop argument. The
inner and outer decisions are independent, and the parent consumes only the
finished inner summary.

### Branch completion must agree

Token ownership can agree across an `if` while the completion schedule does
not. `completionAfterChain` records, for one owner, whether each path uses the
incoming completion or establishes a new completion and, when new, its
`(loop.stage, loop.cluster)`.

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

The then path establishes a stage-1 completion for `{1}`. The else path
passes the stage-0 completion into the `if`. There is no single completion
schedule that can be moved to the first stage-0 MMA. The loop therefore keeps
`%ready` as an iter-arg and yields `%next` for the next iteration.

An absent `else` is also a pass-through path. A loop that may execute zero
times likewise joins its body result with the incoming completion. This keeps
the decision valid for every control-flow path, not only the last path visited.

## Backing copies

`computeBackingCopies` chooses physical buffer copies. A synchronized group
with authored `buffer.copy` uses that value. Otherwise it starts with one
copy.

On the default NVWS path, a synchronized TMEM accumulator can use two copies
when every direct-loop MMA user passes these checks:

- the accumulator is not read-modify-written by the loop;
- accumulator multibuffering is structurally possible;
- the enclosing WS loop permits it;
- two copies fit in the available TMEM blocks; and
- a scaled MMA with block N of 256 is not used.

Meta-NVWS does not add this automatic TMEM copy. An inconsistent or
non-positive authored `buffer.copy` in one reuse group is an error.

Semaphore copies are computed separately by `computeSemaphoreCopies`.
Usually they equal buffer copies. A local backing with no authored
`buffer.copy` and a TMA-load release uses at least the number of semaphore
stages requested by `LowerSemaphore`:

```text
numSemaphoreCopies = max(1, lowerSemaphoreNumStages)
```

This does not change the backing's `numCopies`; it ensures schedule and stage
analysis see the semaphore copies that lowering will create.

For example, `@root_entry_accumulator_adopts_without_semaphore_handoff` in
`test/NVWS/insert_semas_root_entry_tmem.mlir` has one eligible MMA user and
fits two accumulator copies, so its synchronized TMEM backing uses two
copies.

By contrast, the `buffer.id = 102` group in
`test/NVWS/insert_semas.mlir` `@local_release_after_mma` keeps one local
backing copy while its descriptor-load release gives the semaphore the
lowering stage count:

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
buffer that serves it. When a semaphore handoff would be ordered backwards
after pipeline expansion, schedule finalization raises `loop.cluster` on
affected operations. It does not change `loop.stage`.

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
inside one cluster. A loop-closing semaphore may therefore connect operations
that appear in different source iterations but execute in the same expanded
body.

Every emitted handoff keeps a `ProtocolArc` with:

```text
release    generated release node
acquire    generated acquire node
producer   source access or region
consumer   destination access or region
wait       acquire that currently represents the wait for scheduling
```

This record lets schedule finalization reason from the physical producer to
the physical consumer even after token placement has moved an acquire.

### Example: one-copy loop-closing handoff

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
wrong protocol schedule

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
Schedule finalization raises the stage-0 chain from cluster 1 to cluster 3:

```text
correct protocol schedule

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
`computeLoopCarriedDistance` replays writes and reads of the physical set:

```text
one copy:  W(i+1) first reuses the copy released in i      distance 1
two copies: W(i+2) first reuses that copy                  distance 2
```

With two copies, the running example has iteration slack and needs no cluster
change.

### Finalizing one handoff

Consider only the `EMPTY` handoff before schedules are assigned to its
protocol nodes:

```text
final-read(i)      owner {1}  (1,2)
release EMPTY(i)   owner {1}  (?,?)
acquire EMPTY(i+1) owner {3}  (?,?)
W(i+1)             owner {3}  (0,1)
```

The physical-slot replay finds distance 1. After cluster legalization raises
the consumer chain, the release copies the producer completion schedule and
the acquire copies the finalized consumer schedule:

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

`solveOwnerScheduleConstraints` solves all handoffs in one scheduled loop
together. The useful cycle cases are:

```text
cycle total < 0    feasible cycle with iteration slack
cycle total = 0    same expanded wave; tight handoffs require cluster order
cycle total > 0    every owner would need to run later than itself; error
```

A negative cycle is feasible because it has iteration slack. An individual
zero-delay handoff on that cycle can still need cluster ordering.

A positive delay on one edge is legal when the reverse path has enough
negative delay. For example, a `+1` edge and `-3` return edge form a legal
`-2` cycle. `test/NVWS/insert_semas_recurrence_owner_cycle.mlir` covers that
shape.

`legalizeLoopSchedule` combines every zero-delay handoff, handoffs on tight
zero-delay cycles, and same-stage SSA edges. It raises clusters to satisfy
those constraints. A cycle in this same-wave order is an error. Stage values
remain unchanged.

An acquire left at loop `EXIT` has no direct destination access in the same
chain. It is placed after the last operation of its owner at that stage:

```text
owner {3}, stage 0

W main(i)             cluster 1
W other(i)            cluster 4
acquire EMPTY(i+1)    cluster 4
EXIT
```

Placing the acquire at cluster 1 could block owner `{3}` before `W other`.
Within cluster 4, block order keeps `W other` first.

### Moving an acquire repairs its schedule edge

`planLoop` may detach a recurrence acquire from the bottom of a loop and move
it before the first buffer use. A `ProtocolArc::wait` that still points at the
detached recurrence position would describe a nonexistent wait.

`fixupProtocolArcs` handles every arc that waited on the moved acquire:

```text
acquire remains linked in the release's chain, and either the release still
precedes it or the semaphore is an entry semaphore
  keep wait = moved acquire

same-semaphore post-loop acquire follows the release in the same chain
  set wait = post-loop acquire

neither relation holds
  clear wait; no live scheduling relation remains
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

The acquire of `MMA_READY` is at the first MMA. The stage-1 release occurs
after it and supplies the next dynamic iteration; it is not a same-wave wait
from the read back to this moved acquire. Its old scheduling relation is
cleared. Otherwise a false relation

```text
R acc(i) at (1,2) -> MMA acc(i+1) at (0,1)
```

would raise the MMA path even though token placement has already selected the
point-of-use shape. The checked output keeps the MMA acquire, buffer, MMA, and
release at `(0,1)`.

`ProtocolArc::wait` affects only schedule construction. Clearing a stale wait
does not remove the semaphore release or acquire from IR.

### Post-loop acquires use their owner's boundary

A synthetic acquire after a nested loop has no following buffer use of its
own owner. `postLoopAcquire` prevents an unrelated following access from
becoming its schedule anchor.

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

The bridge keeps owner `{1}`'s boundary schedule `(0,1)`; it does not copy
owner `{0}`'s `(0,4)` schedule. If no following schedule anchor exists,
`scheduleAtOwnerBoundary` scans the scheduled loop and places the acquire no
earlier than any operation of that same owner and stage.

Root entry acquires remain unscheduled.

### Schedule and stage offset are separate

The model carries two independent placement decisions. An acquire or release
has a pipeline schedule, `loop.stage` and `loop.cluster`, and can have a
`stageOffset` selecting a physical semaphore copy. An access can instead have
a `bufferStageOffset`; EMIT-IR places that value on the
`semaphore.buffer` created for the access.

Schedule finalization follows producer-to-consumer `ProtocolArc` records. It
uses loop distance and owner order to decide whether a cluster must move. The
result keeps the release at the producer completion and the acquire at the
consumer, but may delay the consumer's cluster so the release can execute
first.

Stage-offset replay follows writes and reads of a physical backing instead.
It asks which copy contains the required value. Circular replay records a
buffer offset on each access and transfers the adjacent values to acquire and
release offsets. Non-circular alias replay records the required protocol
offsets directly. Changing either offset does not move an operation in the
pipeline; changing a cluster does not select a different copy.

The distinction is especially important for a loop close. Its release and
next acquire can be one or more logical iterations apart, while the physical
copy can wrap modulo the semaphore depth. Loop-distance analysis handles the
first fact. The offset attached to the release or acquire handles the second.

`semaphore.buffer` receives the schedule of the access it serves. When that
access has a `bufferStageOffset`, the buffer receives that physical stage too.
It does not introduce another handoff. EMIT-IR uses these finalized choices
when it creates the concrete semaphore and buffer operations.

## Authored buffer-stage offsets

`loop.stage` and `loop.cluster` place operations in the software pipeline. A
release or acquire `stageOffset` instead selects a physical semaphore stage;
an access `bufferStageOffset` selects the stage used by its emitted buffer.
ASP applies a signed semaphore offset modulo the semaphore group's copy
depth. For authored or circular backing, that depth is `buffer.copy`. A
non-circular alias can instead have more semaphore copies than backing copies,
in which case the modulus is `numSemaphoreCopies`:

```text
0     current copy
-1    preceding copy
+1    following copy
```

`finalizeSyncSchedule` replays a fresh-write cursor for each physical set.
Every circular write begins a fresh slot and advances the cursor. For a
non-circular group, a write advances only when an incoming direct-loop
handoff selected it as a fresh write. Reads and other writes use the current
cursor and the latest value ordinal recorded for their group.

```text
stage offset = required value ordinal - current cursor ordinal
```

ASP applies this displacement modulo the copy count. The same replay handles
circular groups and non-circular aliases; their representation and attachment
points differ.

### Circular groups

Circular members with one physical `buffer.id` are separate logical groups.
They must agree on type and `buffer.copy`, have unique valid `buffer.start`
values, and appear in producer order.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` uses K and V in a two-copy ring with starts 0
and 1:

```text
event       cursor change   value ordinal   offset at event
store K     -1 -> 0         K = 0           0
store V      0 -> 1         V = 1           0
load K       unchanged      K = 0          -1
load V       unchanged      V = 1           0
```

K and V share one entry semaphore, `EMPTY`, and one non-entry semaphore,
`FULL`. `emitPhysicalIR` creates those two semaphores once for the circular
backing. Each logical group attaches its own offsets to the shared protocol.

The K consumer must address the copy produced before V advanced the ring.
Its acquire and closing release receive offset `-1`:

```text
K protocol

acquire EMPTY {1}               offset  0
W K {1}
release FULL {1}                offset  0
acquire FULL {2}                offset -1
R K {2}
release EMPTY {2}               offset -1
```

V stays on the current copy:

```text
V protocol

acquire EMPTY {1}               offset 0
W V {1}
release FULL {1}                offset 0
acquire FULL {2}                offset 0
R V {2}
release EMPTY {2}               offset 0
```

The shared `EMPTY` semaphore is initially released so iteration zero can
acquire its copy.

### Non-circular alias handoffs

Non-circular members in one group share the fresh-write replay. Authored
`buffer.copy` gives every member of the `buffer.id` group one physical stage
domain. Without authored copies, this rule applies only to exact aliases whose
semaphore copy count exceeds the backing copy count. SMEM and TMEM use the
same analysis.

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

Both members touch the same piece, so access rules produce one chain. This is
a protocol diagram: semaphore names are assigned after raw-edge reduction
and merging.

```text
protocol DAG, logically unrolled

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
                       release HANDOFF(i) {2}
                                  | HANDOFF
                                  v
                       acquire HANDOFF(i) {4}
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
%handoff = nvws.semaphore.create %base false
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
  nvws.semaphore.release %handoff[1], %t1 {partition = 2}

  %t2 = nvws.semaphore.acquire %handoff[0] {partition = 4}
  %b2 = nvws.semaphore.buffer %handoff[0], %t2
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
write/read use physical stage `s`, the second pair uses `(s+1) mod 2`.

```text
same-stage handoffs
  W dv0 at s     -> R dv0 at s
  W dv1 at s+1   -> R dv1 at s+1

crossing handoffs
  R dv0 at s     -> W dv1 at s+1
  R dv1 at s+1   -> W dv0(i+1) at s
```

The two crossing releases receive offset `+1`. Without it, `%handoff` would
release stage `s` while its acquire waits on stage `s+1`; the initially false
semaphore would never satisfy that acquire. The loop-close has the same issue
in the opposite direction after the two-copy wrap.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@tmem_fused_alias_depth_two` applies the same replay to two non-circular TMEM
aliases with authored `buffer.copy = 2`.

## Build order and code map

`buildSyncDag` processes one group in this order:

```text
ChainWalker                         complete raw edges
computeBackingCopies               physical backing copies
buildEdgesAndSemas                 reduce, merge, group, create protocol
insertEntryAcquires                seed the first token
buildRegionFlows                   summarize path results
planRegionFlows / planLoop         carry tokens or move acquires
pruneDeadIfFlows                   remove unused if token results
computeRequiredParts               record partitions needed by regions
computeSemaphoreCopies             semaphore-stage copies
```

After all groups are built, `finalizeSyncSchedule` assigns stage offsets,
repairs loop clusters, and assigns schedules to protocol nodes.

Current implementation map in
[`InsertSemasSyncDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp):

- Access walk: `ActiveUse`, `VersionSource`, `PieceState`, `Tokens`,
  `applyTouch`, `raiseForeignUseEdges`, and `ChainWalker`.
- Raw-edge reduction: `KnownOrder`, `reduceStraightEdges`,
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
