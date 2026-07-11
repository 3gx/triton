# SYNC-DAG

## Contents

- [Notation](#notation)
- [Purpose](#purpose)
- [How the pass works](#how-the-pass-works)
- [From buffer accesses to synchronization edges](#from-buffer-accesses-to-synchronization-edges)
  - [What the pass remembers for each piece](#what-the-pass-remembers-for-each-piece)
  - [Read and write rules](#read-and-write-rules)
  - [Ensuring every access has a token](#ensuring-every-access-has-a-token)
  - [Example: one writer and one reader](#example-one-writer-and-one-reader)
  - [Example: several readers and token reuse](#example-several-readers-and-token-reuse)
  - [Example: different pieces use separate waits](#example-different-pieces-use-separate-waits)
  - [Nested loops and branches](#nested-loops-and-branches)
  - [Example: nested loops with a count-2 semaphore](#example-nested-loops-with-a-count-2-semaphore)
  - [Example: the boundary owner is unchanged](#example-the-boundary-owner-is-unchanged)
- [Removing waits already covered by other waits](#removing-waits-already-covered-by-other-waits)
  - [Waits in one path](#waits-in-one-path)
  - [Waits between loop iterations](#waits-between-loop-iterations)
  - [Async operations and release positions](#async-operations-and-release-positions)
  - [Example: one release waits for two operations](#example-one-release-waits-for-two-operations)
  - [Example: an async writer keeps its direct edge](#example-an-async-writer-keeps-its-direct-edge)
  - [Example: one release after two reads](#example-one-release-after-two-reads)
  - [Example: other edges cover a direct edge](#example-other-edges-cover-a-direct-edge)
  - [Example: other edges cover a loop-exit edge](#example-other-edges-cover-a-loop-exit-edge)
- [Placing acquires and releases](#placing-acquires-and-releases)
  - [Straight-line code](#straight-line-code)
  - [`if` branches](#if-branches)
  - [Loops](#loops)
  - [Choosing POU, FirstTouch, or Auto](#choosing-pou-firsttouch-or-auto)
  - [Example: the two loop placements](#example-the-two-loop-placements)
  - [Example: Auto discards and rebuilds](#example-auto-discards-and-rebuilds)
  - [Example: `if` branches use count one](#example-if-branches-use-count-one)
  - [Example: POU can still carry a token](#example-pou-can-still-carry-a-token)
  - [Example: nested POU without loop-carried tokens](#example-nested-pou-without-loop-carried-tokens)
  - [Example: reading the buffer after the inner loop](#example-reading-the-buffer-after-the-inner-loop)
  - [Example: fixed stages in a nested POU loop](#example-fixed-stages-in-a-nested-pou-loop)
  - [Example: each branch keeps its own schedule](#example-each-branch-keeps-its-own-schedule)
- [Assigning semaphores and counts](#assigning-semaphores-and-counts)
  - [Releases on the same path or different paths](#releases-on-the-same-path-or-different-paths)
  - [One pending count per semaphore](#one-pending-count-per-semaphore)
  - [The first acquire](#the-first-acquire)
- [Buffer and semaphore copies](#buffer-and-semaphore-copies)
  - [Buffer copies](#buffer-copies)
  - [Example: a TMEM accumulator gets two copies](#example-a-tmem-accumulator-gets-two-copies)
  - [Semaphore copies](#semaphore-copies)
  - [Example: a TMA load stages only semaphore state](#example-a-tma-load-stages-only-semaphore-state)
- [Placing waits in a pipelined loop](#placing-waits-in-a-pipelined-loop)
  - [Release before acquire](#release-before-acquire)
  - [Waits between iterations](#waits-between-iterations)
  - [Selecting the matching copy](#selecting-the-matching-copy)
  - [Example: circular K and V select different copies](#example-circular-k-and-v-select-different-copies)
  - [Example: a non-circular alias advances the copy](#example-a-non-circular-alias-advances-the-copy)
  - [Example: one buffer copy](#example-one-buffer-copy)
- [Checks before changing IR](#checks-before-changing-ir)
- [Build order and code map](#build-order-and-code-map)

## Notation

The document uses the following terms throughout:

```text
group          allocations and views analyzed together, ordinarily one buffer.id
backing        physical SMEM or TMEM storage used by the group
m0, m1         members: allocation names or views in the group
P0, P1         non-overlapping pieces of the group storage
{0}, {1}       owners: partitions 0 and 1 with the enclosing loop's WS tag
root           code with no partition owner
source         latest write to the piece
uses           latest access to the piece by each owner since that write
token          value returned by an acquire and used by releases and semaphore.buffer
```

An owner contains both a partition number and a WS tag. Two operations have
the same owner only when both values match. Most diagrams show one loop at a
time, so `{0}` abbreviates `(partition 0, the WS tag of that loop)`. Nested
examples name the outer and inner loops when their tags differ.

The examples use this short form:

```text
W m0 {0}              owner {0} writes buffer member m0
R m0 {1}              owner {1} reads buffer member m0
ENTER / EXIT           start and end used to describe one loop or branch path
[region P0:W:{1}]      the region writes piece P0 and its boundary owner is {1}
e1: A -> B             synchronization edge: B waits for A
a FULL(2) {2}          owner {2} acquires FULL and waits for 2 arrivals
r FULL(2) {3}          owner {3} releases FULL with arrive_count=2
[tma_load]             the release waits for a TMA load
R m0 [t] {1}           the read uses the buffer selected by token t
```

`a` and `r` are short for acquire and release. `[none]` means the release
signals when it runs. `[tma_load]` means it waits for a TMA load. `[tc5mma]`
means it waits for an MMA.

The diagrams use three views:

```text
initial edge DAG    every wait found from the buffer accesses
reduced edge DAG    the waits left after removing waits already covered
semaphore DAG       the final acquires, releases, tokens, and buffer accesses
```

An arrow labeled `walk` shows program order, not a semaphore wait. An arrow
labeled with a semaphore connects a release to an acquire. Paths from opposite
`if` branches are shown separately because only one branch runs.

Placement modes used later:

```text
POU          point of use: acquire immediately before the access that needs it
FirstTouch   make the loop token available before entry and carry it through the loop
Auto         try POU; if one loop cannot use it, rebuild that loop with FirstTouch
```

## Purpose

SYNC-DAG decides where one partition must wait for another when they use the
same buffer. It chooses the semaphores, acquires, releases, and tokens for
those waits. It also chooses buffer and semaphore copies and places the new
operations in pipelined loops.

ACCESS-DAG has already found the buffer accesses and the loops and branches
that contain them. SYNC-DAG builds the final plan. EMIT-IR emits that plan
without changing the chosen waits or tokens.

## How the pass works

```text
find where one partition must wait for another
  -> remove waits already guaranteed by other waits
  -> choose physical buffer copies
  -> place acquires and releases and choose each access's token
  -> assign semaphores and counts
  -> choose semaphore copies and pipeline placement
  -> check the complete plan
  -> emit IR
```

SYNC-DAG decides all waits, semaphores, and tokens before EMIT-IR starts.

Every worked example names its test under `test/NVWS` except
`@same_owner_nested`, `@doc_preserved_async_edge`, and
`@doc_repeated_same_owner_sources`. Those three inputs were run through the
pass but are not test cases. The DAG dump can be printed with:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt input.mlir \
  -allow-unregistered-dialect --nvws-insert-semas
```

The dump names semaphores `S0`, `S1`, and so on. Examples use names such as
`EMPTY` and `FULL` to make their purpose easier to follow.

## From buffer accesses to synchronization edges

The pass visits buffer accesses in program order. For each buffer piece it
remembers where a new owner receives the current contents and the latest
access by each owner. It adds an edge whenever one owner must wait for another.

### What the pass remembers for each piece

```text
source    latest write to the piece
uses      latest access to the piece by each owner since that write
```

A read updates only that owner's entry in `uses`. It does not replace the
source, so readers that can run separately still start from the same source.
A write becomes the new source and removes the earlier entries from `uses`.

Inside a nested loop or branch, the child DAG uses `ENTER` to represent the
latest write from the parent. A write inside the child becomes the new source.

### Read and write rules

For one buffer piece `P`:

```text
first read or write of P
  remember this access as the source and first use
  add no edge

read P again by the same owner
  update that owner's latest use
  add no edge

read P by a new owner
  add source -> read
  remember the new reader

write P
  add latest use -> write for every other owner that is not already ordered
  remember the write as the new source and only use
```

A write does not need another edge when an existing path already makes the
other owner finish first. When a WS loop receives a buffer from code outside
a partition, its first partition does not need an extra edge from that code.

If an async operation wrote the buffer, a direct edge from that operation
also records the async work that must finish. The pass keeps that edge when a
different path would lose the async wait.

For a group with several members, a later non-async write by the same owner may
reuse the current token. In that case its later release also waits for any
unfinished async write covered by the same token.

The parent treats a nested loop or branch as one summary node. The child has
its own DAG from `ENTER`, through its buffer accesses, to `EXIT`. Parent and
child edges are never mixed.

### Ensuring every access has a token

Every buffer access handled by this pass needs a token:

- One or more synchronization edges that remain and end at an access provide
  one acquire for the new owner.
- If the owner already has a token valid for every piece touched by the
  access, the access reuses that token.
- Otherwise, the pass adds an edge from the last place that holds a usable
  token, so the new owner can acquire one.

For a read, reuse is valid only when that owner already has the current
contents for every piece it reads. For a write, reuse is valid only when all
other owners have already finished using the old contents.

### Example: one writer and one reader

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` contains two
groups. For the per-iteration group `buffer.id=104`, the relevant input is:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

There is one piece, P0. The write sets its contents, and the read by another
owner creates `e1`. Because the loop may run again, owner `{0}` must wait for
the read before overwriting P0. Edge `e2` records that wait at `EXIT`:

```text
edge    source             destination       reason
e1      W m0(i) {0}       R m0(i) {1}       read after write
e2      R m0(i) {1}       EXIT(i) {0}        next iteration may overwrite P0
```

Neither edge is unnecessary:

```text
W m0(i) {0}
     | e1
     v
R m0(i) {1}
     | e2
     v
 EXIT(i) {0}
```

Auto uses POU for this loop. `e1` uses semaphore `FULL`. `e2` uses semaphore
`EMPTY`, which starts released so that iteration zero can begin:

```text
edge    semaphore  pending_count    initial state
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

The next iteration runs the same acquire:

```text
r EMPTY, tr(i) {1}
     | EMPTY
     v
tw = a EMPTY in iteration i+1 {0}
     |
     v
W m0(i+1) [tw] {0}
```

For iteration zero, `EMPTY`'s initially released state lets `tw = a EMPTY`
finish. The loop carries no token in or out. A zero-trip loop runs none of
these semaphore operations and leaves `EMPTY` released.

### Example: several readers and token reuse

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

Both reads by other owners receive the contents written by owner `{0}`. The
final owner-`{0}` read reuses owner `{0}`'s original token and adds no edge.
The next iteration waits for both other readers:

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

R m0 {0}: program order only; it uses token t0 from the EMPTY acquire
```

Both edges to `EXIT` release the same `EMPTY` semaphore on every iteration.
The next owner-`{0}` acquire waits for both releases. The two edges from the
write use separate semaphores with pending count 1:

```text
edge      semaphore  pending_count    initial state
f1        TO_R1      1        blocked
f2        TO_R2      1        blocked
f3,f4     EMPTY      2        released for owner {0}
```

The writer releases both reader semaphores:

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

The readers release `EMPTY` for the next iteration:

```text
R m0 [t1] {1} ----> r EMPTY, t1 {1} --+
                                      +--> EMPTY receives 2 arrivals
R m0 [t2] {2} ----> r EMPTY, t2 {2} --+         |
                                                +--> next = a EMPTY(2) {0}
```

The later `R m0 {0}` uses `t0`, not either reader token. The loop returns no
token.

### Example: different pieces use separate waits

`test/NVWS/insert_semas_tmem_container_subviews.mlir`
`@container_with_disjoint_subviews` uses one large buffer and three smaller
views. The three smaller views do not overlap:

```text
member    range
m0        [0,256)       large buffer
m1        [0,128)       left view
m2        [128,192)     middle view
m3        [192,256)     right view

piece     range          members
P0        [0,128)        m0,m1
P1        [128,192)      m0,m2
P2        [192,256)      m0,m3
```

`[a,b)` means offsets starting at `a` and ending just before `b`.

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

The reduced edge list is:

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
path is easy to follow.

There is no edge between the three smaller-view paths. The large-buffer write
creates one edge for each piece, and the next large-buffer write waits for all
three paths. The semaphore assignment is:

```text
edge        semaphore  pending_count    initial state
e1          P0_FULL    1                blocked
e2          P1_FULL    1                blocked
e3          P2_FULL    1                blocked
e4,e5,e6    EMPTY      3                released for owner {0}
```

The POU plan uses one acquire with pending count 3 for the large buffer and
three separate acquires with pending count 1:

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

The pass emits the operations above in that order. The semaphore DAG is easier
to read with the three paths separated. Each row below starts at the
same `W m0(i)` node; it is repeated only to show all three paths:

```text
whole = a EMPTY(3) {0} --> W m0(i) [whole] {0}

P0: W m0 [whole] --> r P0_FULL --> p0 = a P0_FULL --> W m1 --> R m1 --> r EMPTY, p0
P1: W m0 [whole] --> r P1_FULL --> p1 = a P1_FULL --> W m2 --> R m2 --> r EMPTY, p1
P2: W m0 [whole] --> r P2_FULL --> p2 = a P2_FULL --> W m3 --> R m3 --> r EMPTY, p2

three releases: {r EMPTY, p0 {1}, r EMPTY, p1 {2}, r EMPTY, p2 {3}}
                --> EMPTY receives 3 arrivals --> next = a EMPTY(3) {0}
                --> W m0(i+1) [next] {0}
```

The buffer has two physical copies, but P0, P1, and P2 still have separate
waits. A zero-trip loop runs no semaphore operation and leaves `EMPTY`
initially released.

### Nested loops and branches

A loop or branch is one summary node in its parent's DAG. The summary says
which pieces the region reads or writes and gives each piece a boundary
owner:

```text
parent DAG: ... -> [region summary] -> ...

child DAG:         ENTER -> buffer accesses -> EXIT
```

The parent applies the same read and write rules to the summary. Each child
starts with the incoming buffer contents and builds its own edges. After an
`if`, the parent keeps only waits established by every returning branch. A
loop that may run zero times cannot establish a wait using only its body.

The parent uses only the summary node. It does not replace that node with the
child's accesses. An outer edge can therefore end at a loop while a separate
set of edges exists inside that loop.

### Example: nested loops with a count-2 semaphore

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
loop is one summary node. The complete parent edge list is:

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

The child has its own edge list:

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

The path through `c3` and `c5` already makes `EXIT` wait for the owner-`{1}`
write. Even so, the pass keeps every loop-exit edge to owner `{2}`, because
that owner starts the next iteration. Both `c4` and `c5` therefore remain,
and both release the semaphore acquired by owner `{2}`.

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

The parent edges end at the summary, not at the child's `ENTER` or `EXIT`.
The child edges stay inside the loop. If the inner loop continues, its `EXIT`
returns to the next `ENTER`. If it finishes, parent edge `p2` is used. The
semaphore diagrams below show those two cases separately.

The resulting semaphore assignment is:

```text
edges       semaphore       pending_count    initial state
c1          R1_READY        1                blocked
c2          WRITE_READY     1                blocked
c3          R0_READY        1                blocked
p1,c4,c5    FULL            2                blocked
p2          OUTER_EMPTY     1                released for owner {3}
```

`p1`, `c4`, and `c5` use the same semaphore at different times. Before the
first inner iteration, or when the inner loop has zero trips, `p1` releases
`FULL` with `arrive_count=2`. After a nonempty inner iteration, `c4` and `c5`
both run and contribute one arrival each. Every `FULL` acquire therefore has
`pending_count=2`.

The final plan, using descriptive semaphore names, is:

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
the outer write releases `FULL` for the child. The acquire runs only after the
program reaches it and `FULL` has received the required release:

```text
token path:   outer = a OUTER_EMPTY {3} --> W m0(i) [outer] {3}
semaphore:    W m0(i) --> r FULL(2), outer {3} p1 -- FULL --> t2 = a FULL(2) {2}
program:      W m0(i) --> ENTER inner(i,0) {2} ------ walk --> t2 = a FULL(2) {2}
```

Before the owner-`{1}` write runs, its earlier read must finish and token
`t1w` must be available:

```text
read path:  t2 --> r R1_READY --> t1r = a R1_READY --> R m0 [t1r] {1}
token path: t2 --> R m0 [t2] --> r WRITE_READY --> t1w = a WRITE_READY {1}

both required: {R m0 [t1r] has finished, token t1w is available}
                                       --> W m0 [t1w] {1}
```

The write and the final read each release `FULL` for owner `{2}`:

```text
c4 path: W m0 [t1w] --> r FULL, t1w {1}
c5 path: W m0 [t1w] --> r R0_READY --> t0 = a R0_READY
                         --> R m0 [t0] {0} --> r FULL, t0 {0}

result: {c4 arrival, c5 arrival} --> FULL receives 2 arrivals
```

After those two arrivals, exactly one of two acquires runs. If the inner loop
continues, the next iteration acquires `FULL`:

```text
semaphore: {r FULL, t1w(j), r FULL, t0(j)} --> FULL receives 2 arrivals
           --> next = a FULL(2) in body j+1 {2}
program:   ENTER inner(i,j+1) -- walk --> next
```

If the inner loop finishes, the acquire after the loop waits for the same two
releases. It then releases `OUTER_EMPTY` for the next outer iteration:

```text
semaphore: {r FULL, t1w(last), r FULL, t0(last)} --> FULL receives 2 arrivals
           --> done = a FULL(2) {2}
program:   EXIT inner(i,last) {2} -- walk --> done

done --> r OUTER_EMPTY, done {2} p2 -- OUTER_EMPTY --> next = a OUTER_EMPTY {3}
next --> W m0(i+1) [next] {3}
```

For a zero-trip inner loop, the `p1` release with count 2 lets `done` finish:

```text
semaphore: r FULL(2), outer {3} --> FULL receives 2 arrivals --> done = a FULL(2) {2}
program:   EXIT inner(zero trip) {2} -- walk ----------------> done
done --> r OUTER_EMPTY, done {2} p2
```

Thus every `FULL` acquire waits for exactly two arrivals. Neither loop has a
token `iter_arg` in the emitted IR. The `OUTER_EMPTY` acquire is placed at the
owner-`{3}` write, and its initially released state starts outer iteration
zero.

### Example: the boundary owner is unchanged

The preceding example changes owner from outer `{3}` to inner boundary `{2}`.
The inline `@same_owner_nested` input covers the other case: the outer
writer and inner boundary are both `{3}`.

No current test case covers this case. The plan below was generated by
running the shown input through the pass, so it is checked by the DAG dump but
not by a test case.

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

No parent synchronization edge is needed. The child still has six initial
edges between different owners:

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

The path through `c4` and `c6` already makes `EXIT` wait for the owner-`{1}`
write. Even so, the pass keeps every loop-exit edge to owner `{3}`, because
that owner starts the next iteration. Both `c5` and `c6` therefore remain and
run on every iteration. The resulting semaphores are:

```text
edges       semaphore       pending_count    initial state
c1          R2_READY        1                blocked
c2,c3       WRITE_READY     2                blocked
c4          R0_READY        1                blocked
c5,c6       READY           2                released for owner {3}
```

The key difference is that the initially acquired `READY` token can be used
by the outer write and then passed directly into the first inner iteration.
There is no parent release/acquire pair between them. The final plan is:

```text
initial = a READY(2) before outer loop    token for owner {3}

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

The entry token passes through both same-owner boundaries without another
release or acquire:

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

Both readers release `WRITE_READY` before the owner-`{1}` write can acquire
it:

```text
c3 path: itok --> r R2_READY --> t2 = a R2_READY --> R m0 [t2] {2}
         --> r WRITE_READY, t2 {2}
c2 path: itok --> R m0 [itok] {3} --> r WRITE_READY, itok {3}

result: {c2 arrival, c3 arrival} --> WRITE_READY receives 2 arrivals
        --> t1 = a WRITE_READY(2) {1}
```

That write produces the two arrivals for the next owner-`{3}` token:

```text
c5 path: t1 --> W m0 [t1] {1} --> r READY, t1 {1}
c6 path: t1 --> W m0 [t1] {1} --> r R0_READY --> t0 = a R0_READY
         --> R m0 [t0] {0} --> r READY, t0 {0}

result: {c5 arrival, c6 arrival} --> READY receives 2 arrivals
        --> next = a READY(2) {3}
```

Finally, the loops return the `next` token as follows:

```text
inner continues: next --> EXIT inner(i,j) --> ENTER inner(i,j+1)
inner finishes:  next --> EXIT inner(i,last) --> result = next --> EXIT outer(i)
outer continues: result --> EXIT outer(i) --> ENTER outer(i+1)
outer finishes:  result --> EXIT outer(last) --> final token
```

The two `READY` releases always execute together, so the acquire count is
two. If the inner loop has zero trips, it returns `outer` unchanged. If the
outer loop has zero trips, it returns `initial` unchanged. In both cases the
same token is returned; no synchronization edge is added.

## Removing waits already covered by other waits

The pass first records every required synchronization edge. It then removes an
edge only when the remaining edges do both of these jobs:

- they already make the destination owner wait for the source; and
- they give the destination owner a token it can use.

This step runs before buffer-copy selection and before POU or FirstTouch
placement. Both placement modes therefore use the same reduced edges. An Auto
retry starts again from the same initial edges.

If one remaining edge represents several initial edges, its release cannot be
placed before the latest represented source that uses the same token. The
pass records those sources before it removes any edge. It groups them by the
access that waits, the owner that releases, and the owner that acquires.

Each loop, branch path, and parent DAG is processed separately:

```text
one path              remove an edge covered by other edges in that path
loop boundary         check the kept path into the next iteration
nested region         process each child DAG separately
```

A direct edge that carries an async wait is kept when another path would lose
that wait.

### Waits in one path

An edge between two buffer accesses may be removed only when all of these are
true:

1. removing it does not lose a required async wait;
2. already-kept edges make the destination owner wait for its source;
3. those kept edges leave a usable token for the destination owner; and
4. the destination is a buffer access, not `ENTER`, `EXIT`, or a region.

The token check is important. A path that gives the right execution order is
not enough when the destination has no usable token.

When an edge is removed, its destination reuses the token established by the
kept edges:

```text
removed edge
  -> no new acquire
  -> destination reuses a token returned by an earlier acquire
```

Only kept edges can prove that another edge is unnecessary. A removed edge
can never be used to justify another removal. Edges without owners and edges
whose source is not a buffer access are left to the other steps.

### Waits between loop iterations

An edge from a buffer access to `EXIT` makes the next iteration wait. The next
iteration's boundary owner cannot reuse the piece until the source in the
current iteration has finished.

To decide whether that edge can be removed, the pass follows the kept edges
through a simulated next iteration:

1. record the waits established by one loop iteration;
2. find the destination owner's first access to all affected pieces in the
   next iteration;
3. follow the kept body edges to that access;
4. check that the destination owner has a usable token there; and
5. remove the edge only when the kept path already provides the wait and the
   destination owner is not the loop's first access owner.

The last rule keeps an edge needed to give the first owner a token for the
next iteration.

### Async operations and release positions

Two details must remain correct after edges are removed:

- A direct edge from an async operation is kept when another path would not
  wait for that operation.
- If one remaining edge represents several initial edges, its release stays
  after the latest represented source that uses the same token.

The second rule applies only when the sources use the same token and the later
source follows the earlier one in the same path. The release also waits for
every async operation represented by those initial edges.

The examples below show these rules in the emitted acquires and releases.

### Example: one release waits for two operations

`test/NVWS/insert_semas_same_owner_mixed_completion.mlir`
`@same_owner_mixed_completion` checks that a later write by the same owner does
not lose an earlier async wait. It does not test the separate rule that keeps
a release after the latest represented source. The two SMEM members name the
same buffer. Owner `{0}` writes both before owner `{1}` reads either one:

```text
A: W m0 {0}      nvws.descriptor_load       completion [tma_load]
B: W m1 {0}      ttg.local_store            completion [none]
C: R m1 {1}
D: R m0 {1}
```

A starts a TMA load. B is a non-async write by the same owner using the same
token. The release after B must therefore wait for both A's TMA load and B's
normal completion. The resulting reduced edge is:

```text
q: B sync fill {0} -> C first consumer {1}    [none,tma_load]
```

`D` reuses C's owner-`{1}` token, so it needs no second acquire. The access and
edge DAG is:

```text
                         A: W async m0 {0}
                                  | same token: keep A's TMA wait
                                  v
                  B: W sync m1 {0} [none,tma_load]
                                  | q
                                  v
                         C: R m1 {1}
                                  | walk
                                  v
                         D: R m0 {1}
```

The semaphore assignment is:

```text
role                    semaphore  pending_count    initial state
entry / next iteration  EMPTY      1                released for owner {0}
q                       FULL       2                blocked
```

One release waits for two completions, so `FULL` has `pending_count=2`:

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

The `FULL` semaphore therefore has `pending_count=2`. It waits once for the TMA load and
once for B's normal completion before owner `{1}` continues. This test does
not cover the separate rule for choosing the latest release position.

### Example: an async writer keeps its direct edge

The inline `@doc_preserved_async_edge` input shows the async rule above. It is
not a current test case; the edge and semaphore plans below come from running
this input through the pass:

```text
for {
  A: W async m0 {0}    descriptor_load, completion [tma_load]
  B: R m0 {1}
  C: W m0 {2}
}
```

The pass records three forward edges. Both direct edges from A carry the TMA
wait and must be kept:

```text
edge    source                destination    completion
a1      A: W async m0 {0}     B: R m0 {1}    [tma_load]
a2      A: W async m0 {0}     C: W m0 {2}    [tma_load]
a3      B: R m0 {1}           C: W m0 {2}    [none]
```

```text
a1 path: A: W async m0 {0} -- [tma_load] --> B: R m0 {1}
a2 path: A: W async m0 {0} -- [tma_load] --> C: W m0 {2}
a3 path: B: R m0 {1}       -- [none] ----> C: W m0 {2}
```

No other path makes B wait for A, so `a1` must remain. The path `a1 -> a3`
already puts C after A. The pass deliberately keeps C's direct edge from A,
so C's acquire explicitly waits for the TMA load as well as B's release. It
therefore waits for two arrivals:

```text
edge / role             semaphore      pending_count    initial state
entry / next iteration  EMPTY          1                released for owner {0}
a1                      READ_READY     1                blocked
a2,a3                   WRITE_READY    2                blocked
```

The final plan from the DAG dump is:

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

both releases: {direct-path arrival, reader-path arrival}
               --> WRITE_READY receives 2 arrivals --> t2 = a WRITE_READY(2)
               --> C --> r EMPTY --> next = a EMPTY {0}
```

The direct and reader-path releases use different tokens, so InsertSemas keeps
them as two arrivals rather than merging them into one release. This inline
case documents the InsertSemas plan only; no current test checks the
emitted IR for this two-release async case.

### Example: one release after two reads

The inline `@doc_repeated_same_owner_sources` input shows the other
release-position rule. `whole` spans P0 and P1; `part` is another name for P0:

```text
for {
  W whole(P0,P1) {0}
  R whole(P0,P1) {1}
  R part(P0)     {1}
}
```

The first owner-`{1}` read is the latest use of P1. The later read replaces
owner `{1}`'s latest use only for P0. The two edges to `EXIT` are therefore:

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

Both reads use the token returned by the same owner-`{1}` acquire. The pass
records both source nodes before removing either edge. It then uses the later
read as the release position. The result is one count-1 release after `R part`,
not two arrivals and not an early release after `R whole`:

```text
edge / role       semaphore  pending_count    initial state
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

The current DAG dump has exactly two semaphores with pending count 1 and
places the `EMPTY` release after the second read. The pass uses the list of
earlier reads only to choose that release position; it emits no extra IR
operation for the list.

### Example: other edges cover a direct edge

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members and this access order:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

For the shared piece, the relevant initial edges are:

```text
s1: W m0 {0} -> R m0 {1}
s2: W m0 {0} -> W m1 {2}
s3: R m0 {1} -> W m1 {2}
```

Initial edge DAG:

```text
direct edge
  W m0 {0} -- s2 --> W m1 {2}

kept path
  W m0 {0} -- s1 --> R m0 {1} -- s3 --> W m1 {2}
```

Keeping `s1` and `s3` already makes owner `{2}` wait for the owner-`{0}`
write, and the acquire for `s3` gives owner `{2}` a token. Edge `s2` is
removed. The pass records that `W m1` may use the token from the kept path.
In this example, `s3` still creates the acquire.

Reduced edge DAG, including the rest of the path:

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

The full loop also has one edge from owner `{1}` to the next owner-`{0}`
write. The emitted POU plan has four semaphores with pending count 1:

```text
EMPTY   initially released for owner {0}'s next iteration
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

next-iteration path
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

There is no extra `{0}->{2}` semaphore. The test checks this four-semaphore
cycle and checks that the loop has no extra token argument.

### Example: other edges cover a loop-exit edge

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

After duplicate edges and edges covered by another path are removed, the
relevant edges are:

```text
l1    W m0 {0} -> R m0 {1}
l2    R m0 {1} -> W m1 {2}
l3    W m1 {2} -> R m1 {0}
c0    R m0 {1} -> EXIT {0}       wait before next P0 use
c2    R m1 {0} -> EXIT {2}       possible wait before next P2 use
```

To test `c2`, the pass follows the next iteration's kept path to owner
`{2}`'s first P2 write:

```text
initial edge to EXIT
  R m1(i) {0} -- c2 --> EXIT(i) {2}
```

That edge makes owner `{2}`'s first P2 use in the next iteration wait. That
use is `W m1(i+1)`.

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

The kept `l1 -> l2` path already gives owner `{2}` a token at that write, so
`c2` is removed. Edge `c0` is kept because owner `{0}` is the first access
owner and needs a token to begin the next iteration.

The emitted POU plan therefore has exactly four semaphores with pending count
1:

```text
edge    semaphore  initial state
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

next-iteration path from that read token
  R m0 [t1] {1} -> r EMPTY, t1 {1}

forward path from that read token
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
so no additional P2 loop-exit edge is needed.

## Placing acquires and releases

After unnecessary edges are removed and physical copies are chosen, the pass
places every acquire and release. These locations are final before EMIT-IR
starts.

Every buffer access and every release records which acquire produced its
token. When a token crosses a loop or an `if`, the pass also records the same
token returned by that region. EMIT-IR follows those recorded links; it does
not guess from the owner or from the nearest token in the code.

The scheduled examples use `stage` and `cluster` as defined in
[Notation](#notation). The pass does not change an input `loop.stage`. It may
change `loop.cluster` so releases happen before their matching acquires.

### Straight-line code

At each buffer access, the pass does one of two things:

1. Reuse a token already held by that owner when it is still valid for every
   buffer piece touched by the access.
2. Otherwise, place an acquire immediately before the access.

The matching releases are placed after the earlier buffer accesses and after
any asynchronous work that they must wait for. Each release names both the
token it releases and the acquire that it enables.

An edge that enters or leaves a surrounding loop or `if` is handled the same
way. The acquire is normally placed at the buffer access that needs the token.
An acquire after an inner loop may instead connect the inner wait to later
outer code, and FirstTouch may carry a token through a loop.

### `if` branches

Both branches start with the same incoming tokens. The pass handles each
branch separately and then combines their results:

```text
both branches keep the same incoming token
  keep that token after the `if`

the branches end with different tokens and later code needs one
  each branch returns the token from the path that ran

the branches release a semaphore for a later acquire
  put the release in each branch
  the later acquire waits for the release from the branch that ran
```

Releases in different branches do not add their counts because only one
branch runs. Releases that execute together on one path do add their counts.

### Loops

A loop has two relevant kinds of edges:

```text
edge into the loop       earlier code -> first matching buffer use in the body
edge to the next turn    last body use -> EXIT -> first matching use next turn
```

If different pieces have different boundary owners, the pass places an
acquire at the first use for each owner. There is no single loop token to
carry.

When the pieces share one boundary owner, the pass chooses among these cases:

- If the body ends with a valid token and later code needs it, return that
  token from the loop.
- FirstTouch makes the boundary token available before entering the loop,
  carries it through the loop, and acquires the token for the next turn before
  `EXIT`.
- POU places the acquire at the first buffer use that needs the token. If an
  inner loop already uses the same semaphore, the acquire after that inner
  loop can connect the inner and outer waits.
- If the loop has no buffer use that needs a token, no token is added merely
  because a loop exists.

### Choosing POU, FirstTouch, or Auto

The pass has three placement modes:

```text
pou
  place each acquire at the buffer use that needs it
  report an error if fixed stages make that placement impossible

first-touch
  for a loop with one boundary owner, make its token available before entry
  and carry the next token through the loop
  reuse a valid incoming token when one already exists

auto
  try POU first
  if one loop cannot use POU, discard that attempted plan
  rebuild with FirstTouch for that loop
  repeat if another loop also needs FirstTouch
  if the failing loop cannot be identified, try FirstTouch once for every
  loop with one boundary owner
```

Before emitting IR, the pass checks that every used token comes from an
acquire or was passed into the loop or branch. It also checks that the chosen
placements respect fixed stages.

### Example: the two loop placements

`test/NVWS/insert_semas_placement_modes.mlir` uses the running loop:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

The remaining edges and semaphores are identical in both modes:

```text
W {0} --FULL--> R {1} --EMPTY--> next W {0}
```

Auto selects POU for this input:

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

The loop does not carry a semaphore token. `EMPTY` is acquired immediately
before the next write that needs it.

FirstTouch:

```text
initial = a EMPTY at root         token used by owner {0}

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

If owner `{0}` already has a valid token before a FirstTouch loop, the loop
can reuse it instead of creating the root acquire shown here.

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

The pass first tries POU for every loop. It can place all operations for this
input, but the final check finds that the fixed stages require the token to be
carried through the inner loop. The pass emits no IR from that attempt. It
does this instead:

```text
attempt 1
  place acquires and releases with POU
  check the complete plan
  find that the inner loop needs FirstTouch

discard attempt 1
  discard the complete plan for the function
  restore the input loop.stage and loop.cluster values
  select FirstTouch for this buffer group and inner loop

attempt 2
  rebuild from the unchanged input IR
  carry the token through the selected inner loop
  check and emit the new plan
```

For this input, Auto produces the same IR as explicit FirstTouch. Strict POU
reports:

```text
fixed loop.stage values require the loop to carry a token
```

The final plan carries the same token through both loops:

```text
initial = a ENTRY at root
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

The emitted result contains nothing from the discarded POU attempt.

### Example: `if` branches use count one

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

From the outer loop, the `if` reads the buffer and starts and ends with owner
`{1}`. The then branch contains the owner-`{0}` read; the else branch does not
access the buffer:

```text
outer loop view

ENTER loop {1} --> W MMA {1} --> [if summary P0:R:{1}] --> EXIT loop {1}

child views

then:  ENTER if {1} -- e1 --> R acc {0} -- e2 --> EXIT if {1}
else:  ENTER if {1} -----------------------------> EXIT if
```

Expanded across the branch, these are the waits:

```text
                                 W MMA(i) {1}
                                       |
                       +---------------+---------------+
                       | cond=true                     | cond=false
                       v                               v
              e1: R acc(i) {0}                 no buffer access
                       |                               |
                       v                               |
       e2-then: next W using same copy           e2-else: next W using same copy
```

For `e1`, owner `{0}` must wait for owner `{1}`. The release that allows the
next write comes from one of two places:

- the owner-`{0}` read when the then branch executes; or
- the owner-`{1}` MMA when the else branch executes.

Only one branch executes, so only one of these releases executes.
`EMPTY.pending_count` is therefore 1, not 2. The semaphore assignment is:

```text
wait                       semaphore  pending_count    initial state
entry / e2 alternatives    EMPTY      1                released for owner {1}
e1                         FULL       1                not released
```

The generated plan contains acquire and release operations inside both
branches:

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

Only one branch runs, so the semaphore DAG is shown as two separate paths.
On the then path:

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

Only one `EMPTY` release executes each time the `if` runs. The `if` returns no
semaphore token: each branch releases `EMPTY`, and the next buffer reuse
acquires it. The `FULL` acquire and release remain inside the then branch. The
else release still waits for the MMA completion through `[tc5mma]`.

### Example: POU can still carry a token

POU may still carry a token when the current iteration already acquired the
token needed by the next iteration.

`test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout` uses a TMEM buffer with two
physical copies:

```text
for {
  W first {0}
  R reader1 {1}
  R reader2 {2}
  W final {0}
}
```

The remaining edges are:

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

Both readers release the count-2 `EMPTY` semaphore. Its acquire must occur
before `W final(i)`, so that token already exists at the end of iteration `i`.
The next iteration's `W first(i+1)` can reuse the same token:

```text
initial = a EMPTY(2) at root       token used by owner {0}

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
                    +--> 2 releases --> next = a EMPTY(2) {0}
r EMPTY, t2 {2} ----+
```

POU and FirstTouch produce the same placement here. The count-2 acquire is
needed by `W final`, and its token remains valid for `W first` in the next
iteration. The loop therefore carries `next`. A zero-trip loop returns the
`initial` token, which was acquired before the loop, unchanged.

### Example: nested POU without loop-carried tokens

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

At the outer level, the inner loop starts and ends with owner `{1}`, the same
owner as the outer `ENTER` and `EXIT`. No parent-level semaphore is needed:

```text
                         ENTER outer(i) {1}
                                  | walk
                                  v
                   [inner summary P0:W:{1}]
                                  | walk
                                  v
                         EXIT outer(i) {1}
```

The inner level has two remaining edges:

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

`c1` becomes `FULL`. Edge `c2` and the initial ready state use `EMPTY`:

```text
edge          semaphore  pending_count    initial state
c1            FULL       1                not released
entry,c2      EMPTY      1                released for owner {1}
```

The pass places both acquires immediately before the inner buffer accesses
that need them:

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

After the final inner iteration, its `EMPTY` release has no following acquire
in that run of the inner loop. It remains ready for the next time the inner
loop executes, including in a later outer iteration. Neither loop carries a
semaphore token, and no acquire or release is moved outside both loops.
EMIT-IR removes the old tokens attached to the TMEM operations because the
semaphores now order the accesses. A zero-trip inner or outer loop executes no
acquire or release, so the ready `EMPTY` release remains available.

### Example: reading the buffer after the inner loop

The next function in the same file,
`@nested_ws_inner_loop_parent_continuation`, adds a read after the inner loop:

```text
outer for {
  inner for {
    W acc {1}
    R acc {0}
  }

  R acc {0}          outer read after the inner loop
}
```

The inner loop and outer loop have separate edge sets:

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

The pass forms four count-1 semaphores:

```text
LOCAL_EMPTY    next inner write or acquire after the inner loop; initially released
LOCAL_FULL     inner write -> inner read
OUTER_FULL     completed inner loop -> outer read
OUTER_EMPTY    outer read -> owner {1}; initially released
```

The semaphore assignment is:

```text
edge / role       semaphore      pending_count    initial state
c1                LOCAL_FULL     1                not released
entry,c2          LOCAL_EMPTY    1                released for owner {1}
p1                OUTER_FULL     1                not released
entry,p2          OUTER_EMPTY    1                released for owner {1}
```

An acquire before the outer loop consumes `OUTER_EMPTY`'s initially released
state. Without it, the later owner-`{1}` acquire could finish before the outer
read releases `OUTER_EMPTY`. The complete acquire/release plan is:

```text
initial = a OUTER_EMPTY at root    consumes the initially released state

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
  prepare = a OUTER_EMPTY {1}
  r LOCAL_EMPTY, prepare [none] {1}
}
```

The semaphore DAG starts by consuming the initial `OUTER_EMPTY` release, then
shows the wait between the inner write and read:

```text
initial = a OUTER_EMPTY {1}        consumes the initially released state

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

If the inner loop continues, the release feeds the next inner write:

```text
semaphore:    r LOCAL_EMPTY, tr {0} -- LOCAL_EMPTY --> next = a LOCAL_EMPTY {1}
program order: ENTER inner(i,j+1) -- walk ----------> next
next --> W inner(i,j+1) {1}
```

If the inner loop finishes, an acquire after the loop connects it to the outer
read. A final acquire and release prepare `LOCAL_EMPTY` for the next outer
iteration:

```text
semaphore:    r LOCAL_EMPTY, tr {0} --> LOCAL_EMPTY is ready
              --> done = a LOCAL_EMPTY {1}
program order: EXIT inner(i,last) -- walk --> done

done --> r OUTER_FULL, done {1} p1 -- OUTER_FULL --> to = a OUTER_FULL {0}
to --> R outer(i) [to] {0} --> r OUTER_EMPTY, to {0} p2
r OUTER_EMPTY -- OUTER_EMPTY --> prepare = a OUTER_EMPTY {1}
prepare --> r LOCAL_EMPTY, prepare {1} -- LOCAL_EMPTY --> first acquire
                                                           in outer(i+1)
```

The acquire before the loop consumes the initial `OUTER_EMPTY` release. The
later `prepare` acquire must therefore wait for the outer read's release.
Neither loop carries a semaphore token.

If the inner loop is zero-trip, `done` consumes `LOCAL_EMPTY`: its initially
released state on the first outer iteration, or the release from the previous
`prepare` on a later one. If the outer loop is zero-trip, only the
unconditional `OUTER_EMPTY` acquire before the loop executes.

### Example: fixed stages in a nested POU loop

`test/NVWS/insert_semas_nested_carrier.mlir`
`@scheduled_relocated_acquire_boundaries` has fixed stages on the inner write,
inner read, and read after the inner loop. The pass places each acquire at the
buffer access that needs it.

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

The remaining edge DAG is:

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

All four edges use count-one semaphores:

```text
edge / role       semaphore      initial state
entry,p2          OUTER_EMPTY    released for owner {0}
p1,c2             LOCAL_EMPTY    not released; first released by p1
c1                LOCAL_FULL     not released
```

The generated schedule locations are:

```text
operation                              owner    cluster    stage
a LOCAL_EMPTY at inner MMA             {1}      3          0
r LOCAL_FULL after MMA [tc5mma]         {1}      3          0
a LOCAL_FULL at inner read              {0}      2          1
r LOCAL_EMPTY after inner read          {0}      2          1
post-inner a LOCAL_EMPTY                {1}      inner boundary
r OUTER_EMPTY after inner loop          {1}      3          0
a OUTER_EMPTY at post-inner read        {0}      4          0
```

The integrated semaphore DAG is:

```text
initial = a OUTER_EMPTY at root    token used by owner {0}
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

After the inner loop, owner `{1}` acquires `LOCAL_EMPTY` and releases
`OUTER_EMPTY`. That release uses owner `{1}`'s schedule. The following
owner-`{0}` acquire uses the outer read's schedule. These locations are final
before EMIT-IR starts.

### Example: each branch keeps its own schedule

The same test file's `@branch_completion_requires_carrier` has different
fixed stages in the two `if` paths. The `if` does not return a token.

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

On the then path, `R branch` holds the token used to release `CONVERGE` at
stage 1. On the else path, the release uses the token from `mma0` and waits
for its completion at stage 0. Only one path runs, so the two releases do not
add their counts:

```text
edge / role       semaphore       pending_count    initial state
entry,p2          OUTER_EMPTY     1                released for owner {0}
p1,c3             LOCAL_EMPTY     1                not released; first released by p1
c1                BRANCH_FULL     1                not released
c2 alternatives   CONVERGE        1                not released
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

The else path reaches the same acquire with the token from `mma0`:

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

Both paths then execute the same release to the next inner iteration and the
same acquire/release sequence after the inner loop:

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

The then release follows the stage-1 read. The else release waits for the
stage-0 MMA. The final acquire is shared because either path provides exactly
one `CONVERGE` release. The `if` itself does not take or return a token.

## Assigning semaphores and counts

The pass first decides where every acquire and release goes. It then gives
matching acquires and releases the same semaphore. An acquire at the start of
a loop, an acquire in the next iteration, and an acquire after the loop may
use the same semaphore when they wait for different endings of the same work.

For each semaphore, the pass:

1. counts how many releases each acquire must wait for;
2. makes that count the same at every acquire of the semaphore;
3. assigns the semaphore to those acquires and their releases; and
4. connects tokens that leave a `for` or `if` to the acquire that uses them.

The acquire and release locations do not change after this step.

### Releases on the same path or different paths

When two releases always run, one acquire waits for both:

```text
one release contribution = arrive_count * number of completions
pending_count = sum of all release contributions on the path
```

When releases are in opposite `if` branches, only one runs. Both branches
must contribute the same count, and the acquire waits for that count:

```text
then path contributes N arrivals
else path contributes N arrivals
acquire pending_count = N
```

Most releases have `arrive_count=1` and one completion, so they contribute one
arrival. A release with two completions contributes two arrivals because both
operations must finish.

This explains the two earlier count examples:

```text
release-count loop
  c4 and c5 both execute on one path  -> FULL count 2

conditional accumulator
  then and else releases are exclusive -> EMPTY count 1
```

### One pending count per semaphore

Every acquire of one semaphore uses the same `pending_count`. Sometimes the
first acquire has one release before it, while later loop iterations have
several releases before the same acquire. The single release can use a larger
`arrive_count` so that both cases provide the same total.

A path can be scaled only when it has exactly one release. Every completion
on that release must be `[none]` or WGMMA; TMA and `[tc5mma]` cannot be scaled.
The semaphore's `pending_count` must be evenly divisible by the number of
completions. The pass then sets:

```text
arrive_count = pending_count / number of completions
```

The release-count example has:

```text
first acquire    p1 by owner {3}: one [none] release
later iteration  c4 by {1} plus c5 by {0}: two releases

FULL pending_count = 2
p1 arrive_count   = 2
c4 arrive_count   = 1
c5 arrive_count   = 1
```

If the releases cannot provide the same count at every acquire, the pass
reports an error instead of creating a semaphore with inconsistent counts.

### The first acquire

Some semaphores start released for one owner. This lets the first acquire run
before any release has executed:

```text
starts released       semaphore.create ... true
starts blocked        semaphore.create ... false
```

The first acquire has the same `pending_count` as every later acquire of that
semaphore. A semaphore with `pending_count=2` therefore starts with both
arrivals ready; it does not start half-released.

This initial state allows only the first acquire to run. Later acquires wait
for real releases. The token belongs to the owner that performs the acquire.

## Buffer and semaphore copies

The number of physical buffer copies and semaphore copies can differ:

```text
buffer copies       physical SMEM or TMEM copies
semaphore copies    copies of each semaphore
```

They are usually equal. A local buffer filled by a TMA load can instead use
one physical buffer copy and several semaphore copies.

### Buffer copies

The pass chooses the number of buffer copies after it removes unnecessary
wait edges. All names and views for the same buffer must specify compatible
`buffer.copy` values.

```text
start with one buffer copy

if no wait edges remain
  keep one buffer copy, even when buffer.copy was specified

otherwise, if buffer.copy is specified
  use that many buffer copies

otherwise, if this is a TMEM buffer in the normal NVWS pipeline
and the two-copy checks pass
  use two buffer copies
```

The NVWS Meta pipeline is selected by `TRITON_NVWS_USE_META=1`. Its memory
planner runs before InsertSemas and writes `buffer.copy`. Automatic warp
specialization then calls InsertSemas with `use-meta-partitioner=true`.
InsertSemas still uses any `buffer.copy` value, but it does not guess two TMEM
copies when that value is absent. A missing value therefore means one copy in
this pipeline.

In the normal NVWS pipeline, a synchronized TMEM buffer without `buffer.copy`
can be given two copies automatically. It does not need an MMA user. When it
does have an MMA user, it stays single-copy if any of these are true:

- the MMA reads the old accumulator while writing the new value;
- that MMA and loop do not support two accumulator copies;
- the surrounding WS loop disables accumulator copies;
- two copies would make total TMEM use exceed `128 * 512`; or
- this is a scaled MMA whose accumulator's N dimension is 256.

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` checks a two-copy
accumulator. The acquire before the loop creates a token that partition `{1}`
uses directly, so no extra semaphore is needed between root code and that
partition. The loop must carry this token from one iteration to the next.
Strict POU rejects that placement, so Auto rebuilds this loop with FirstTouch.
The later MMA in partition `{2}` still needs its own semaphore.

### Example: a TMEM accumulator gets two copies

The complete access shape of that accumulator is:

```text
W acc root                    initial tmem_store

for {
  R acc {1}
  W acc {1}
  W acc {2}                  tc_gen5_mma
}

R acc root                    final tmem_load
```

Partition `{1}` is the first partition to use the buffer in the loop. It uses
the token created by the acquire before the loop, so no wait edge is needed
from root code to partition `{1}`. The outer and inner edge DAGs are:

```text
parent

W acc root -- walk, same token --> [loop summary P0:W:{1}]
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

The semaphores are:

```text
edge / role    semaphore  pending_count    initial state
entry,e2       EMPTY      1                released at root
e1             TO_MMA     1                blocked
p1             AFTER      1                blocked
```

Auto uses FirstTouch for this loop because the token created before the loop
must remain available through the loop. The loop carries that token:

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

If the loop continues, `next` crosses `EXIT` and becomes the next `iter_arg`:

```text
next --> EXIT(i) {1} --> ENTER(i+1) {1} --> repeat body
```

If the loop finishes, the same token becomes the loop result and is released
on `AFTER` for edge `p1`:

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

For a zero-trip loop, `result` is the original root token and is released on
`AFTER`. This buffer has wait edges, no explicit `buffer.copy`, an MMA shape
that permits two copies, and enough TMEM capacity. The result is:

```text
input accumulator     memdesc<128x128xf32>
generated buffer      memdesc<2x128x128xf32>
physical copies       2
semaphore copies      2
```

### Semaphore copies

After placing the acquires and releases, the pass chooses how many copies of
each semaphore are needed:

```text
SMEM buffer
and no input buffer.copy
and at least one release waits for [tma_load]
  semaphore copies = max(1, num-stages)

otherwise
  semaphore copies = buffer copies
```

For example, `test/NVWS/insert_semas.mlir` `@local_release_after_mma` has one
physical SMEM copy. The buffer is filled by `nvws.descriptor_load`, so the
`FULL` release waits for `[tma_load]`. Its semaphores may use several copies
even though the buffer has one copy.

```text
physical buffer
  one SMEM buffer copy

semaphore copies
  copy 0, copy 1, ... according to num-stages
```

The `num-stages` setting on `--nvws-insert-semas` must match the setting on
`--nvws-lower-semaphore`. The copy count is decided here, before the acquire
and release operations are emitted.

### Example: a TMA load stages only semaphore state

The buffer with `buffer.id=102` in `@local_release_after_mma` has this input:

```text
for {
  W m0 {0} stage=0       nvws.descriptor_load
  R m0 {1} stage=1       tc_gen5_mma operand
}
```

The same two waits repeat on every iteration:

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

`e1` uses `FULL`. `e2` uses `EMPTY`, which starts released for iteration zero.
Both semaphores have `pending_count=1`:

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

Both waits are required. `[tma_load]` keeps the `FULL` release waiting until
the descriptor load has filled SMEM. `[tc5mma]` keeps the `EMPTY` release
waiting until the MMA has finished reading that buffer copy.

SMEM does not receive automatic TMEM double buffering, and the input has no
`buffer.copy`. Therefore:

```text
physical buffer copies     1
semaphore copies           max(1, num-stages)
```

The extra semaphore copies do not create extra SMEM copies. There is no
buffer use after the loop, so no acquire waits for the final `EMPTY` release.

## Placing waits in a pipelined loop

The pass schedules acquires and releases only after their locations and
semaphores are fixed. Scheduling does not move them to different buffer
accesses.

A scheduled loop uses three values:

```text
loop.stage       pipeline stage written on the input operation
loop.cluster     order of operations within and across those stages
stage-offset     which buffer and semaphore copy an operation uses
```

The pass keeps every `loop.stage` fixed. It may change `loop.cluster` and
`stage-offset` so that each release happens before the matching acquire and
both operations select the same copy.

### Release before acquire

For every release and acquire pair, the pass knows:

```text
the operation that must finish before the release
the acquire that waits for the release
the loop.stage of both operations
whether the wait crosses a loop iteration
```

The required delay between their owners is:

```text
release operation stage - acquire operation stage - iteration distance
```

The pass keeps the input stages and adjusts clusters to satisfy these waits.
If the required waits form a cycle that cannot execute, it reports the cycle
as an error. The pass does not add another ordering rule when the existing
token use already guarantees the order.

### Waits between iterations

A release can wake an acquire in the same iteration or in a later iteration.
When this is already known, the wait records that positive iteration distance.
Otherwise the pass follows the physical buffer and semaphore copies used by
successive iterations.

For a buffer with several copies, it finds the first later iteration in which
the release and acquire use the same copy. A buffer with one copy always
returns to that copy on the next iteration, so its distance is one.

If the pass cannot find a later iteration that uses the matching copy, it
reports an error.

### Selecting the matching copy

Circular buffers can have several names with one `buffer.id` and one shared
set of physical copies. The pass visits their reads and writes in program
order, tracks which write advances the shared copy number, and assigns an
offset to each buffer access, acquire, and release.

It also checks that every input `buffer.start` agrees with this write order
and that no read selects a copy before a write has produced it.

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` checks the circular two-copy case. K and V are
analyzed separately but share physical storage. Each access, acquire, and
release receives the offset of the copy it uses.

A non-circular alias is another name or view of the same physical buffer. If
that buffer has several copies, the operations on both sides of a wait must
be directly inside the same loop body. The pass selects a release offset that
uses the same copy as the acquire, including for a wait between iterations.

`test/NVWS/insert_semas_fused_alias_handoff.mlir` `@fused_alias_depth_two`
checks this case. Its semaphores contain both buffer names, and each
`nvws.semaphore.buffer` returns both views. The copy offset belongs to the
acquire or release; the buffer view does not choose a different copy.

### Example: circular K and V select different copies

`test/NVWS/insert_semas_circular_smem.mlir`
`@circular_tutorial_1_1_to_2_2` has K and V sharing one circular buffer with
two physical copies:

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

K and V have the same wait-edge shape:

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

K and V each need a `FULL` semaphore with `pending_count=1` and an `EMPTY`
semaphore that starts released. Because they share the same physical
`buffer.id`, the emitted IR uses one physical `FULL` semaphore and one physical
`EMPTY` semaphore for both. Each write acquires `EMPTY` immediately before it,
so the loop does not carry a token for either buffer:

```text
K semaphore DAG                          V semaphore DAG

kt = a EMPTY {1}                         vt = a EMPTY {1}
W K [kt] {1}                             W V [vt] {1}
r FULL, kt {1}                           r FULL, vt {1}
kr = a FULL {2}                          vr = a FULL {2}
R K [kr] {2}                             R V [vr] {2}
r EMPTY, kr {2}                          r EMPTY, vr {2}
```

The two buffers share one write number. Each write advances it:

```text
event       current write number    required write number    offset
W K         -1 -> 0                 K producer = 0           0
W V          0 -> 1                 V producer = 1           0
R K          1                      K producer = 0          -1
R V          1                      V producer = 1           0
```

The K and V operations therefore use different offsets on the same two
physical semaphores:

```text
K operations on shared semaphores         V operations on shared semaphores

kt = a EMPTY offset=0 {1}                 vt = a EMPTY offset=0 {1}
W K [kt, buffer offset=0] {1}             W V [vt, buffer offset=0] {1}
r FULL offset=0, kt {1}                   r FULL offset=0, vt {1}
kr = a FULL offset=-1 {2}                 vr = a FULL offset=0 {2}
R K [kr, buffer offset=-1] {2}            R V [vr, buffer offset=0] {2}
r EMPTY offset=-1, kr {2}                 r EMPTY offset=0, vr {2}
```

The offsets select these physical copies relative to the latest shared write:

```text
operation           stage offset       physical copy
W K / r FULL        0                  copy 0
W V / r FULL        0                  copy 1
a FULL / R K       -1                  copy 0
a FULL / R V        0                  copy 1
R K / r EMPTY      -1                  copy 0
R V / r EMPTY       0                  copy 1
```

The `-1` means the previous copy after V advanced the shared write number. It
is not a negative physical index. With two copies, wrapping `-1` selects copy
0. The generated DAG gives K's acquire and closing release `stage-offset=-1`
and V's offset zero. The test also checks that the IR contains the one shared
pair of physical semaphores.

### Example: a non-circular alias advances the copy

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@fused_alias_depth_two` uses two names for one two-copy SMEM buffer:

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

Both names refer to the same bytes, so the wait-edge DAG is one chain:

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

Every semaphore has `pending_count=1`; `ENTRY` starts released:

```text
edge    semaphore
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
wait                       release offset    acquire offset
W m0 -> R m0               0                 0
R m0 -> W m1              +1                 0
W m1 -> R m1               0                 0
R m1 -> W m0(i+1)         +1                 0
```

The generated DAG therefore places `stage-offset=1` on the `M1_READY` and
`ENTRY` releases. Without these offsets, an acquire and its release would use
different physical semaphore copies. The pass assigns the offset to each
release before it emits IR.

### Example: one buffer copy

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` has one SMEM copy and this scheduled loop:

```text
W buffer {3}        loop.stage 0
R first {1}         loop.stage 0
R last {1}          loop.stage 1
```

The memory cycle is:

```text
W(i) {3} --FULL--> R first(i) {1}
                         |
                         | same reader token
                         v
                    R last(i) {1}
                         |
                         | EMPTY, distance 1
                         v
                    W(i+1) {3}
```

The schedule can overlap the final read of iteration `i` with work from
iteration `i+1`. The `EMPTY` acquire is immediately before the next store,
and the matching `EMPTY` release is after the final read. This wait therefore
crosses from iteration `i` to iteration `i+1`. Because there is one semaphore
copy, the distance is one iteration:

```text
r EMPTY after R last(i) {1}
     | matching release/acquire; one copy => distance 1
     v
a EMPTY before W(i+1) {3}
```

The semaphore assignment is:

```text
edge / role             semaphore  pending_count    initial state
W -> R first            FULL       1                blocked
entry / next iteration  EMPTY      1                released for owner {3}
```

The complete generated semaphore DAG is:

```text
empty = a EMPTY {3}                 [cluster 3, stage 0]
             empty |
                   v
       W buffer(i) [empty] {3}      [cluster 3, stage 0]
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
    W buffer(i+1) [next] {3}        [cluster 3, stage 0]
```

The loop does not carry a semaphore token. Each iteration acquires `EMPTY`
immediately before the store. `EMPTY` starts released for iteration zero; the
release from partition `{1}` lets the next iteration acquire it.

The pass adjusts `loop.cluster` so the next store and its first reader are
ordered after the final read. The test checks those clusters and the final
acquire and release operations.

After the ordering and copy offsets are fixed, each acquire and release gets
the schedule of the operation where it was placed. A release follows the
latest asynchronous work that it must wait for. An acquire uses the schedule
of the operation that next needs its token. For an acquire used by the next
iteration, the pass also accounts for the loop boundary.

## Checks before changing IR

The pass checks the complete acquire/release DAG before it changes the input
IR.

For a POU attempt, it checks that:

- moving an acquire to a buffer access does not lose a token needed after a
  `for` or `if`;
- a one-copy wait between iterations agrees with the fixed pipeline stages;
  and
- every token leaving a `for` or `if` can be traced to an incoming token or an
  acquire inside that region.

For every accepted placement, it checks that:

- every buffer access has a token owned by the correct owner;
- every release has a token, the work it must wait for, a positive arrival
  count, and exactly one matching acquire;
- each matching release and acquire use the same semaphore;
- every acquire has the semaphore's positive `pending_count` and either has a
  release before it or uses a semaphore that starts released;
- an acquire repeated inside a loop has a release that lets the next
  iteration continue;
- one semaphore is not acquired by two different owners;
- every wait between iterations has a positive distance;
- every path through a `for` or `if` returns the token needed after that
  path; and
- every wait edge became part of an acquire/release pair.

After emitting IR, the pass checks that every token and buffer view is used
inside the region where it exists and that no token is used after its
release. See [EMIT-IR](emit-ir.md) for those checks.

## Build order and code map

This final section names C++ functions and types for readers changing the
implementation. The earlier sections do not require these names.

The function driver in `InsertSemas.cpp` builds one attempt as follows:

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

If POU fails in Auto mode, the driver discards that attempt and rebuilds with
FirstTouch for the failing loop. It does not try to undo only part of the
failed attempt.

Source map:

| Responsibility | Implementation |
| --- | --- |
| Placement-mode parsing and retry | `InsertSemas.cpp` |
| Shared `Node`, `RegionFlow`, `Sema`, `GroupDag` model | `InsertSemas.h` |
| Groups, pieces, owners, accesses, region summaries | `InsertSemasAccessDag.cpp` |
| Initial wait edges | `ChainWalker`, `applyTouch` in `InsertSemasSyncDag.cpp` |
| Remove unnecessary wait edges | `reduceStraightEdges`, `reduceLoopCloses`, `reduceEdges` |
| Place acquires and releases | `DirectBuilder` |
| Assign semaphores and counts | `DirectBuilder::formSemaphores` |
| Checks before changing IR | `validatePOUPlan`, `validateTokenConnectivity`, `verifySyncDag` |
| Copies and schedule | `computeBackingCopies`, `computeSemaphoreCopies`, `finalizeSyncSchedule` |
| DAG dump and IR emission | `InsertSemasEmitIR.cpp` |

The design can be summarized in one sentence:

> Find the waits, remove waits already guaranteed by other paths, place the
> acquires and releases, assign semaphores, check the result, then emit IR.
