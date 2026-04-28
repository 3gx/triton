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
- [Removing waits already guaranteed by other waits](#removing-waits-already-guaranteed-by-other-waits)
  - [Waits in one path](#waits-in-one-path)
  - [Waits between loop iterations](#waits-between-loop-iterations)
  - [Async operations and release positions](#async-operations-and-release-positions)
  - [Example: one release waits for two operations](#example-one-release-waits-for-two-operations)
  - [Example: an async writer keeps its direct edge](#example-an-async-writer-keeps-its-direct-edge)
  - [Example: one release after two reads](#example-one-release-after-two-reads)
  - [Example: a direct edge is unnecessary](#example-a-direct-edge-is-unnecessary)
  - [Example: a loop-exit edge is unnecessary](#example-a-loop-exit-edge-is-unnecessary)
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
  - [Example: a TMA load uses the lowering stage count](#example-a-tma-load-uses-the-lowering-stage-count)
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
reduced edge DAG    the waits left after removing waits already guaranteed
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
first write of P
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
unfinished async write associated with the same token.

The parent treats a nested loop or branch as one summary node. The child has
its own DAG from `ENTER`, through its buffer accesses, to `EXIT`. Parent and
child edges are never mixed.

### Ensuring every access has a token

Once a group needs synchronization, every buffer access in that group needs a
token:

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

No edge is removed. The two edges have different destinations, so no edges
are merged or grouped together:

```text
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

Auto uses POU for this loop. `e1` uses semaphore `FULL`. `e2` uses semaphore
`EMPTY`, which starts released so that iteration zero can begin:

```text
edge    semaphore    release owner    pending_count    initial state
e1      FULL         {0}              1                blocked
e2      EMPTY        {1}              1                released for owner {0}
```

The semaphore DAG uses the same vertical layout as the edge DAG. The `EMPTY`
semaphore arrow bypasses the loop boundary and reaches the acquire in the next
iteration:

```text
                 ENTER(i) {0}
                      | walk
                      v
          tw = acquire EMPTY(i) {0}
                     tw |
                        v
              W m0(i) [tw] {0}
                      | walk
                      v
           release FULL, tw {0} e1
                   FULL |
                        v
            tr = acquire FULL {1}
                     tr |
                        v
              R m0(i) [tr] {1}
                      | walk
                      v
          release EMPTY, tr {1} e2 ---------------- EMPTY ----------------+
                                                                          |
                 EXIT(i) {0}                                              |
                      | next iteration                                    |
                      v                                                   |
               ENTER(i+1) {0}                                             |
                      | walk                                              |
                      v                                                   |
       tw2 = acquire EMPTY(i+1) {0} <-------------------------------------+
                    tw2 |
                        v
            W m0(i+1) [tw2] {0}
```

For iteration zero, `EMPTY`'s initially released state lets `tw = a EMPTY`
finish without a preceding `e2` release. Re-entry uses the `e2` release from
the preceding iteration. After the final iteration, no later acquire consumes
its `EMPTY` release. The loop carries no token in or out. A zero-trip loop runs
none of these semaphore operations and leaves `EMPTY` released.

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
f3      R m0(i) {1}    EXIT(i) {0}
f4      R m0(i) {2}    EXIT(i) {0}
```

```text
                                  ENTER(i) {0}
                                       | walk
                                       v
                                  W m0 {0}
                         +-----------+-----------+
                      f1 |      walk |           | f2
                         v           v           v
                   R m0 {1}     R m0 {0}     R m0 {2}
                      f3 |      walk |           | f4
                         +-----------+-----------+
                                     v
                                 EXIT {0}
```

No edge is removed: `f1` and `f2` start two independent reader paths, and
`f3` and `f4` both make the next iteration wait. No edges have both the same
source owner and destination, so no edges are merged. Edges `f3` and `f4`
have the same destination and destination owner, so they share one semaphore
and acquire while keeping two releases. The two edges from the write use
separate semaphores:

```text
edge      semaphore    release owner    pending_count    initial state
f1        TO_R1        {0}              1                blocked
f2        TO_R2        {0}              1                blocked
f3        EMPTY        {1}              2                released for owner {0}
f4        EMPTY        {2}              2                released for owner {0}
```

The semaphore DAG keeps owner `{0}` in the middle and the two readers on the
outside, as in the edge DAG. Both owner-`{0}` releases remain on one
program-order path. The two reader releases join at the next iteration's
`EMPTY` acquire:

```text
                                            ENTER(i) {0}
                                                  | walk
                                                  v
                                      t0 = acquire EMPTY(2) {0}
                                               t0 |
                                                  v
                                          W m0(i) [t0] {0}
                                                  | walk
                                                  v
                                      release TO_R1, t0 {0} f1
                +---------------------------------+
          TO_R1 |                                 | walk
                v                                 v
     t1 = acquire TO_R1 {1}           release TO_R2, t0 {0} f2
             t1 |                                 +---------------------------------+
                v                            walk |                                 | TO_R2
        R m0(i) [t1] {1}                          v                                 v
                | walk                    R m0(i) [t0] {0}               t2 = acquire TO_R2 {2}
                v                                 | walk                         t2 |
    release EMPTY, t1 {1} f3                      v                                 v
          EMPTY |                            EXIT(i) {0}                    R m0(i) [t2] {2}
                |                                 | next iteration                  | walk
                |                                 v                                 v
                |                            ENTER(i+1) {0}              release EMPTY, t2 {2} f4
                |                                 | walk                      EMPTY |
                |                                 v                                 |
                +------------------> next = acquire EMPTY(2) {0} <------------------+
                                             next |
                                                  v
                                        W m0(i+1) [next] {0}
```

For iteration zero, `EMPTY` starts released with count 2 and supplies `t0`.
For re-entry, `f3` and `f4` supply `next`. The later `R m0 {0}` uses `t0`,
not either reader token. After the final iteration, no later acquire consumes
the two `EMPTY` releases. The loop returns no token. A zero-trip loop executes
no acquire or release and leaves `EMPTY` released.

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

`[a,b)` means offsets starting at `a` and ending just before `b`. The relevant
input is:

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

The exact edge inventory is:

```text
DAG node          pieces       synchronization edge ending here
ENTER(i) {0}      P0,P1,P2     none
W m0(i) {0}       P0,P1,P2     none
W m1(i) {1}       P0           e1: W m0(i) {0} -> W m1(i) {1}
W m2(i) {2}       P1           e2: W m0(i) {0} -> W m2(i) {2}
W m3(i) {3}       P2           e3: W m0(i) {0} -> W m3(i) {3}
R m1(i) {1}       P0           none; same-owner program order
R m2(i) {2}       P1           none; same-owner program order
R m3(i) {3}       P2           none; same-owner program order
EXIT(i) {0}       P0           e4: R m1(i) {1} -> EXIT(i) {0}
                   P1           e5: R m2(i) {2} -> EXIT(i) {0}
                   P2           e6: R m3(i) {3} -> EXIT(i) {0}
```

```text
                                            ENTER(i) {0}
                                                  | walk
                                                  v
                                             W m0(i) {0}
                    +-----------------------------+-----------------------------+
                 e1 |                          e2 |                             | e3
                    v                             v                             v
               W m1(i) {1}                   W m2(i) {2}                   W m3(i) {3}
                    | walk                        | walk                        | walk
                    v                             v                             v
               R m1(i) {1}                   R m2(i) {2}                   R m3(i) {3}
                 e4 |                          e5 |                          e6 |
                    +-----------------------------+-----------------------------+
                                                  v
                                             EXIT(i) {0}
```

No edge is removed or merged. Edges `e1`, `e2`, and `e3` have different
destinations. Edges `e4`, `e5`, and `e6` have the same destination and
destination owner, so they share one semaphore and acquire while retaining
three releases. The semaphore assignment is:

```text
edge    semaphore    release owner    pending_count    initial state
e1      P0_FULL      {0}              1                blocked
e2      P1_FULL      {0}              1                blocked
e3      P2_FULL      {0}              1                blocked
e4      EMPTY        {1}              3                released for owner {0}
e5      EMPTY        {2}              3                released for owner {0}
e6      EMPTY        {3}              3                released for owner {0}
```

POU places the count-3 acquire at the large write. The owner-`{0}` releases
stay in program order. Each other owner can start as soon as its matching
release runs. All three `EMPTY` releases then join at the next acquire:

```text
                                                                                          ENTER(i) {0}
                                                                                                | walk
                                                                                                v
                                                                                  whole = acquire EMPTY(3) {0}
                                                                                          whole |
                                                                                                v
                                                                                        W m0(i) [whole] {0}
                                                                                                | walk
                                                                                                v
                                                                                  release P0_FULL, whole {0} e1
                    +---------------------------------------------------------------------------+
            P0_FULL |                                                                           | walk
                    v                                                                           v
        p0 = acquire P0_FULL {1}                                                  release P1_FULL, whole {0} e2
                 p0 |                                     +-------------------------------------+
                    v                             P1_FULL |                                     | walk
            W m1(i) [p0] {1}                              v                                     v
                    | walk                    p1 = acquire P1_FULL {2}            release P2_FULL, whole {0} e3
                    v                                  p1 |                                     +-------------------------------------+
            R m1(i) [p0] {1}                              v                                walk |                                     | P2_FULL
                    | walk                        W m2(i) [p1] {2}                              v                                     v
                    v                                     | walk                           EXIT(i) {0}                    p2 = acquire P2_FULL {3}
        release EMPTY, p0 {1} e4                          v                                     | next iteration                   p2 |
              EMPTY |                             R m2(i) [p1] {2}                              v                             W m3(i) [p2] {3}
                    |                                     | walk                         ENTER(i+1) {0}                               | walk
                    |                                     v                                     | walk                                v
                    |                         release EMPTY, p1 {2} e5                          v                             R m3(i) [p2] {3}
                    |                               EMPTY |                                     |                                     | walk
                    |                                     |                                     |                                     v
                    |                                     |                                     |                         release EMPTY, p2 {3} e6
                    |                                     |                                     |                               EMPTY |
                    +-------------------------------------+----------------------> next = acquire EMPTY(3) {0} <----------------------+
                                                                                           next |
                                                                                                v
                                                                                      W m0(i+1) [next] {0}
```

The buffer has two physical copies, but P0, P1, and P2 still have separate
waits. `EMPTY`'s initially released count-3 state supplies `whole` on the
first iteration. The three releases supply `next` on re-entry. After the final
iteration, no later acquire consumes them. A zero-trip loop runs no semaphore
operation and leaves `EMPTY` initially released.

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
p2      inner summary {2}      EXIT outer(i) {3}
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
c4      W m0(i,j) {1}      EXIT inner(i,j) {2}
c5      R m0(i,j) {0}      EXIT inner(i,j) {2}
```

```text
              ENTER inner(i,j) {2}
                    +---------------------- c1 > ----------------------+
                    | walk                                                v
                    v                                                  R m0 {1}
                R m0 {2}                                                  | walk
                    | c2                                                   |
                    +---------------------- c2 > --------------------------+
                    |                                                      v
                    |                                                  W m0 {1}
                    |                                                      |
                    |                                                      v
                    +---------------------- < c4 --------------------------+---------------------- c3 > ----------------------+
                    |                                                                                                           v
                    |                                                                                                       R m0 {0}
                    |                                                                                                           | c5
                    +--------------------------------------------------- < c5 ---------------------------------------------------+
                    v
              EXIT inner(i,j) {2}
```

The path through `c3` and `c5` already makes `EXIT` wait for the owner-`{1}`
write. Even so, the pass keeps every loop-exit edge to owner `{2}`, because
that owner starts the next iteration. Both `c4` and `c5` therefore remain,
and both release the semaphore acquired by owner `{2}`.

The parent edges end at the summary, not at the child's `ENTER` or `EXIT`.
The child edges stay inside the loop. If the inner loop continues, its `EXIT`
returns to the next `ENTER`. If it finishes, parent edge `p2` is used. The
semaphore diagrams below show those two cases separately.

No edge is removed or merged. In the child, `c4` and `c5` have the same
destination and destination owner, so they share one `FULL` acquire while
keeping releases from owners `{1}` and `{0}`. Parent edge `p1` uses that same
semaphore for the first child entry. Its single release signals twice so every
`FULL` acquire waits for the same count:

```text
edge    semaphore      release owner / count    pending_count    initial state
c1      R1_READY       {2} x1                   1                blocked
c2      WRITE_READY    {2} x1                   1                blocked
c3      R0_READY       {1} x1                   1                blocked
p1      FULL           {3} x2                   2                blocked
c4      FULL           {1} x1                   2                blocked
c5      FULL           {0} x1                   2                blocked
p2      OUTER_EMPTY    {2} x1                   1                released for owner {3}
```

`p1`, `c4`, and `c5` use the same semaphore at different times. Before the
first inner iteration, or when the inner loop has zero trips, `p1` releases
`FULL` with `arrive_count=2`. After a nonempty inner iteration, `c4` and `c5`
both run and contribute one arrival each. Every `FULL` acquire therefore has
`pending_count=2`.

The diagrams keep each control path continuous and draw semaphore signals on
separate side paths. First, `p1` supplies the first inner acquire when the
loop runs:

```text
           ENTER outer(i) {3}
                    | walk
                    v
     outer = acquire OUTER_EMPTY {3}
              outer |
                    v
           W m0(i) [outer] {3}
                    | walk
                    v
release FULL(2), outer {3} p1 ----------- FULL(2) > -----------+
                    | enter inner                                |
                    v                                            |
          ENTER inner(i,0) {2}                                   |
                    | walk                                       |
                    v                                            |
       t2 = acquire FULL(2) {2} <--------------------------------+
                 t2 |
```

If the inner loop has zero trips, the same real `p1` release instead supplies
the post-loop `done` acquire:

```text
release FULL(2), outer {3} p1 ------- FULL(2) > -------+
                    | enter inner                       |
                    v                                   |
        inner scf.for executes zero trips              |
                    | loop finishes                     |
                    v                                   |
       done = acquire FULL(2) {2} <--------------------+
                 done |
```

For an executed inner iteration, the semaphore DAG uses the same owner lanes
and branch structure as the child edge DAG. Owner `{2}` stays on the left
control path. Releases `c4` and `c5` feed a separate `FULL(2)` path that
bypasses `EXIT` and the next `ENTER`, then ends directly at the next POU
acquire:

```text
              ENTER inner(i,j) {2}
                        | walk
                        v
            t2 = acquire FULL(2) {2}
                     t2 |
                        v
           release R1_READY, t2 {2} c1
                        +---------------------------->----------------------------+
                        | walk                                           R1_READY |
                        v                                                         v
              R m0(i,j) [t2] {2}                                     t1r = acquire R1_READY {1}
                        | walk                                                t1r |
                        v                                                         v
         release WRITE_READY, t2 {2} c2                                 R m0(i,j) [t1r] {1}
                        |                                                         | walk
                        +---------------------------->----------------------------+
                        | walk                                                    v
                        v                                           t1w = acquire WRITE_READY {1}
                        |                                                     t1w |
                        |                                                         v
                        |                                               W m0(i,j) [t1w] {1}
                        |                                                         | walk
                        |                                                         v
                        |                                           release R0_READY, t1w {1} c3
                        |                                                         +----------------------->-----------------------+
                        |                                                         | walk                                 R0_READY |
                        |                                                         v                                               v
                        |                                             release FULL, t1w {1} c4                        t0 = acquire R0_READY {0}
                        |                       +----------------<----------------+                                            t0 |
                        |                  FULL |                                                                                 v
                        |                       |                                                                       R m0(i,j) [t0] {0}
                        |                       |                                                                                 | walk
                        |                       |                                                                                 v
                        |                       |                                                                      release FULL, t0 {0} c5
                        |                       +----------------------------------------<----------------------------------------+
                        |               FULL(2) |
                        v                       |
               EXIT inner(i,j) {2}              |
                        | next iteration        |
                        v                       |
             ENTER inner(i,j+1) {2}             |
                        | walk                  |
                        v                       |
                        +-----------<-----------+
                        v
           next = acquire FULL(2) {2}
                   next |
```

After the final inner iteration, the same two releases instead supply
`done`. This is the same executed body, but the control path finishes the
loop instead of entering another iteration:

```text
              ENTER inner(i,j) {2}
                        | walk
                        v
            t2 = acquire FULL(2) {2}
                     t2 |
                        v
           release R1_READY, t2 {2} c1
                        +---------------------------->----------------------------+
                        | walk                                           R1_READY |
                        v                                                         v
              R m0(i,j) [t2] {2}                                     t1r = acquire R1_READY {1}
                        | walk                                                t1r |
                        v                                                         v
         release WRITE_READY, t2 {2} c2                                 R m0(i,j) [t1r] {1}
                        |                                                         | walk
                        +---------------------------->----------------------------+
                        | walk                                                    v
                        v                                           t1w = acquire WRITE_READY {1}
                        |                                                     t1w |
                        |                                                         v
                        |                                               W m0(i,j) [t1w] {1}
                        |                                                         | walk
                        |                                                         v
                        |                                           release R0_READY, t1w {1} c3
                        |                                                         +----------------------->-----------------------+
                        |                                                         | walk                                 R0_READY |
                        |                                                         v                                               v
                        |                                             release FULL, t1w {1} c4                        t0 = acquire R0_READY {0}
                        |                       +----------------<----------------+                                            t0 |
                        |                  FULL |                                                                                 v
                        |                       |                                                                       R m0(i,j) [t0] {0}
                        |                       |                                                                                 | walk
                        |                       |                                                                                 v
                        |                       |                                                                      release FULL, t0 {0} c5
                        |                       +----------------------------------------<----------------------------------------+
                        |               FULL(2) |
                        v                       |
            EXIT inner(i,last) {2}              |
                        | loop finishes         |
                        v                       |
                        +-----------<-----------+
                        v
           done = acquire FULL(2) {2}
                   done |
```

Either real `done` acquire above implements `p2`. The control path continues
through the outer boundary, while `OUTER_EMPTY` bypasses that boundary and
ends directly at the next outer acquire:

```text
           done = acquire FULL(2) {2}
                   done |
                        v
release OUTER_EMPTY, done {2} p2 ---------------- OUTER_EMPTY > ----------------+
                        | finish outer body                                     |
                        v                                                       |
              EXIT outer(i) {3}                                                 |
                        | next iteration                                        |
                        v                                                       |
            ENTER outer(i+1) {3}                                                |
                        | walk                                                  |
                        v                                                       |
nextOuter = acquire OUTER_EMPTY {3} <-------------------------------------------+
              nextOuter |
                        v
          W m0(i+1) [nextOuter] {3}
```

Thus every `FULL` acquire waits for exactly two arrivals. Neither loop has a
token `iter_arg` in the emitted IR. The `OUTER_EMPTY` acquire is placed at the
owner-`{3}` write. Its initially released state starts outer iteration zero;
later outer iterations use the preceding `p2` release. After the final outer
iteration, no later acquire consumes `p2`. A zero-trip outer loop executes no
semaphore operation and leaves `OUTER_EMPTY` released.

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
c5      W m0(i,j) {1}          EXIT inner(i,j) {3}
c6      R m0(i,j) {0}          EXIT inner(i,j) {3}
```

```text
              ENTER inner(i,j) {3}
                        +------------------- c1 > --------------------+
                        | walk                                        v
                    R m0 {3}                                      R m0 {2}
                        | c2                                          | c3
                        +------------------- c2 > --------------------+------------------- c3 > --------------------+
                        |                                                                                           v
                        |                                                                                       W m0 {1}
                        |                                                                                           |
                        |                                                                                           v
                        +------------------------------------------ < c5 -------------------------------------------+------------------- c4 > --------------------+
                        |                                                                                                                                         v
                        |                                                                                                                                     R m0 {0}
                        |                                                                                                                                         | c6
                        +----------------------------------------------------------------- < c6 ------------------------------------------------------------------+
                        v
               EXIT inner(i,j) {3}
```

The path through `c4` and `c6` already makes `EXIT` wait for the owner-`{1}`
write. Even so, the pass keeps every loop-exit edge to owner `{3}`, because
that owner starts the next iteration. Both `c5` and `c6` therefore remain and
run on every iteration. No edges are removed or merged. Edges `c2` and `c3`
share a destination and destination owner, so they share one count-2 acquire.
Edges `c5` and `c6` do the same at `EXIT` and share `READY`:

```text
edge    semaphore      release owner    pending_count    initial state
c1      R2_READY       {3}              1                blocked
c2      WRITE_READY    {3}              2                blocked
c3      WRITE_READY    {2}              2                blocked
c4      R0_READY       {1}              1                blocked
c5      READY          {1}              2                released for owner {3}
c6      READY          {0}              2                released for owner {3}
```

The key difference is that the initially acquired `READY` token can be used
by the outer write and then passed directly into the first inner iteration.
There is no parent release/acquire pair between them. The entry token passes
through both same-owner boundaries:

```text
     initial = acquire READY(2) {3}
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

An executed inner iteration keeps those same four owner lanes. Owner `{3}`
stays on the uninterrupted left path. Releases `c2` and `c3` meet at the
count-2 write acquire. Releases `c5` and `c6` meet at the count-2 `READY`
acquire on the owner-`{3}` path. The resulting `next` token then crosses
`EXIT` and the next `ENTER`:

```text
              ENTER inner(i,j) {3}
                   itok |
                        v
          release R2_READY, itok {3} c1
                        +---------------- R2_READY > -----------------+
                        | walk                               R2_READY |
                        v                                             v
              R m0(i,j) [itok] {3}                        t2 = acquire R2_READY {2}
                   itok |                                          t2 |
                        v                                             v
        release WRITE_READY, itok {3} c2                     R m0(i,j) [t2] {2}
                        |                                             | walk
                        |                                             v
                        |                              release WRITE_READY, t2 {2} c3
                        +-------------- WRITE_READY > ----------------+-------------- WRITE_READY > ----------------+
                        | walk                                                                       WRITE_READY(2) |
                        |                                                                                           v
                        |                                                                            t1 = acquire WRITE_READY(2) {1}
                        |                                                                                        t1 |
                        |                                                                                           v
                        |                                                                                  W m0(i,j) [t1] {1}
                        |                                                                                           | walk
                        |                                                                                           v
                        |                                                                              release R0_READY, t1 {1} c4
                        |                                                                                           +---------------- R0_READY > -----------------+
                        |                                                                                           | walk                               R0_READY |
                        |                                                                                           v                                             v
                        |                                                                               release READY, t1 {1} c5                      t0 = acquire R0_READY {0}
                        |                      +----------------------------- < READY ------------------------------+
                        |                READY |                                                                                                               t0 |
                        |                      |                                                                                                                  v
                        |                      |                                                                                                         R m0(i,j) [t0] {0}
                        |                      |                                                                                                                  | walk
                        |                      |                                                                                                                  v
                        |                      |                                                                                                      release READY, t0 {0} c6
                        |                      +---------------------------------------------------- < READY -----------------------------------------------------+
                        |             READY(2) |
                        |                      |
                        +--------- < ----------+
                        v
           next = acquire READY(2) {3}
                   next |
                        v
               EXIT inner(i,j) {3}
                   next | next inner iteration
                        v
             ENTER inner(i,j+1) {3}
            itok = next |
```

If a nonempty inner loop finishes, that same token becomes its result:

```text
       next = acquire READY(2) {3}
               next |
                    v
       EXIT inner(i,last) {3}
               next |
                    v
          result = next
```

If the inner loop has zero trips, it returns its incoming outer token instead:

```text
                  outer
              outer | inner loop has zero trips
                    v
             result = outer
```

When the outer loop continues, `result` crosses its boundary and supplies the
next outer write:

```text
                 result
             result |
                    v
            EXIT outer(i) {3}
                    | next outer iteration
                    v
          ENTER outer(i+1) {3}
             result |
                    v
         W m0(i+1) [result] {3}
```

When the outer loop finishes, it returns `result`:

```text
                 result
             result |
                    v
          EXIT outer(last) {3}
                    |
                    v
             final = result
```

A zero-trip outer loop returns the original acquired token:

```text
     initial = acquire READY(2) {3}
            initial | outer loop has zero trips
                    v
             final = initial
```

The two `READY` releases always execute together, so the acquire count is
two. If the inner loop has zero trips, it returns `outer` unchanged. If the
outer loop has zero trips, it returns `initial` unchanged. In both cases the
same token is returned; no synchronization edge is added.

## Removing waits already guaranteed by other waits

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
destination node, the owner that releases, and the owner that acquires.

Each loop, branch path, and parent DAG is processed separately:

```text
one path              remove an edge already guaranteed by that path
loop boundary         check the kept path into the next iteration
nested region         process each child DAG separately
```

A direct edge from the current source that carries an async wait is always
kept. The pass does not remove it even when another path reaches the same
destination.

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

- A direct edge from the current source that carries an async wait is always
  kept.
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
normal completion. Because `m0` and `m1` are exact aliases, they contain one
piece. The exact edge inventory is:

```text
DAG node             synchronization edge ending here
ENTER(i) {0}         none
A: W async m0 {0}    none
B: W sync m1 {0}     none
C: R m1 {1}          q1: B {0} -> C {1}       [none,tma_load]
D: R m0 {1}          none; reuse C's token
EXIT(i) {0}           q2: D {1} -> EXIT(i) {0} [none]
```

No edge is removed. `q1` starts at B because B is the latest write, but its
completion list also includes A's TMA load. `D` replaces C as owner `{1}`'s
latest access, so `q2` starts at D. The synchronization-edge DAG is:

```text
                                                              ENTER(i) {0}
                                                                    | walk
                                                                    v
                                                          A: W async m0(i) {0}
                                                                    | walk
                                                                    v
                                                           B: W sync m1(i) {0}
                        +-------------------------------------------+
                     q1 |                                           | walk
                        v                                           v
                 C: R m1(i) {1}                                     |
                        | walk                                      |
                        v                                           |
                 D: R m0(i) {1}                                     |
                     q2 |                                           |
                        +-------------------------------------------+
                                                                    v
                                                               EXIT(i) {0}
```

The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released for owner {0}
q2             EMPTY        {1}              1                same semaphore
q1             FULL         {0}              2                blocked
```

One release waits for two completions, so `FULL` has `pending_count=2`.
Owner `{0}` continues directly toward `EXIT` while owner `{1}` executes the
reader chain:

```text
                                                              ENTER(i) {0}
                                                                    | walk
                                                                    v
                                                     producer = acquire EMPTY(i) {0}
                                                           producer |
                                                                    v
                                                   A: W async m0(i) [producer] {0}
                                                                    | walk
                                                                    v
                                                    B: W sync m1(i) [producer] {0}
                                                                    | walk
                                                                    v
                                              release FULL, producer [none,tma_load] {0} q1
                        +-------------------------------------------+
                FULL(2) |                                           | walk
                        v                                           v
         consumer = acquire FULL(2) {1}                        EXIT(i) {0}
               consumer |                                           | next iteration
                        v                                           v
             C: R m1(i) [consumer] {1}                       ENTER(i+1) {0}
                        | walk                                      | walk
                        v                                           v
             D: R m0(i) [consumer] {1}                              |
                        | walk                                      |
                        v                                           |
      release EMPTY, consumer [none] {1} q2                         |
                  EMPTY |                                           |
                        +---------------------------> next = acquire EMPTY(i+1) {0}
                                                               next |
                                                                    v
                                                 A: W async m0(i+1) [next] {0}
```

The `FULL` semaphore therefore has `pending_count=2`. It waits once for the
TMA load and once for B's normal completion before owner `{1}` continues.
This test does not cover the separate rule for choosing the latest release
position. On the final iteration, no later acquire consumes `q2`. A zero-trip
loop executes no acquire or release and leaves `EMPTY` released.

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

The pass records three forward edges and one edge to the next iteration. Both
direct edges from A carry the TMA wait and must be kept:

```text
DAG node              synchronization edge ending here
ENTER(i) {0}          none
A: W async m0 {0}     none
B: R m0 {1}           a1: A {0} -> B {1}       [tma_load]
C: W m0 {2}           a2: A {0} -> C {2}       [tma_load]
                       a3: B {1} -> C {2}       [none]
EXIT(i) {0}            a4: C {2} -> EXIT(i) {0} [none]
```

```text
                  ENTER(i) {0}
                        | walk
                        v
              A: W async m0(i) {0}
                        +-------------------->----------------------+-------------------->----------------------+
                        | walk                                   a1 |                                           | a2
                        v                                           v                                           |
                        |                                    B: R m0(i) {1}                                     |
                        |                                        a3 |                                           |
                        |                                           +-------------------->----------------------+
                        |                                                                                       v
                        |                                                                                C: W m0(i) {2}
                        |                                                                                    a4 |
                        +------------------------------------------<--------------------------------------------+
                        v
                   EXIT(i) {0}
```

No other path makes B wait for A, so `a1` must remain. The path `a1 -> a3`
already puts C after A. The pass deliberately keeps C's direct edge from A,
so C's acquire explicitly waits for the TMA load as well as B's release. It
therefore waits for two arrivals. No edge is removed in this example. Edges
`a2` and `a3` share one destination and semaphore, but their source owners are
different, so they remain two releases:

```text
edge / role    semaphore      release owner    pending_count    initial state
entry          EMPTY          -                1                released for owner {0}
a4             EMPTY          {2}              1                same semaphore
a1             READ_READY     {0}              1                blocked
a2,a3          WRITE_READY    {0}, {1}         2                blocked
```

The semaphore DAG keeps owner `{0}` on one uninterrupted spine. Owner `{1}`
branches from `a1`; its `a3` release and owner `{0}`'s later `a2` release
provide the two arrivals required by owner `{2}`:

```text
                  ENTER(i) {0}
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
       A: W async m0(i) [t0] {0}
                        | walk
                        v
    release READ_READY, t0 [tma_load] {0} a1
                        +-------------------->----------------------+
                        | walk                           READ_READY |
                        v                                           v
                        |                              t1 = acquire READ_READY {1}
                        |                                        t1 |
                        |                                           v
                        |                                B: R m0(i) [t1] {1}
                        |                                           | walk
                        |                                           v
                        |                         release WRITE_READY, t1 [none] {1} a3
                        |                                           +-------------------->----------------------+
                        |                                                                           WRITE_READY |
                        |                                                                                       v
    release WRITE_READY, t0 [tma_load] {0} a2                                                                   |
                        +------------------------------------------->-------------------------------------------+
                   walk |                                                                        WRITE_READY(2) |
                        v                                                                                       v
                   EXIT(i) {0}                                                                   t2 = acquire WRITE_READY(2) {2}
                        | next iteration                                                                     t2 |
                        v                                                                                       v
                 ENTER(i+1) {0}                                                                      C: W m0(i) [t2] {2}
                        | walk                                                                                  | walk
                        v                                                                                       v
                        |                                                                        release EMPTY, t2 [none] {2} a4
                        |                                                                                 EMPTY |
                        +------------------------------------------ < ------------------------------------------+
                        v
          next = acquire EMPTY(i+1) {0}
                   next |
                        v
     A: W async m0(i+1) [next] {0}
```

The direct and reader-path releases use different tokens, so InsertSemas keeps
them as two arrivals rather than merging them into one release. This inline
case documents the InsertSemas plan only; no current test checks the emitted
IR for this two-release async case. For iteration zero, `EMPTY` starts
released. On later iterations, `a4` supplies the acquire before A, as shown in
the same diagram.

The final `a4` release has no later acquire. A zero-trip loop executes none of
the shown operations and leaves `EMPTY` released.

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
owner `{1}`'s latest use only for P0. The complete edge inventory is:

```text
DAG node           synchronization edge ending here
ENTER(i) {0}       none
W whole {0}        none
R whole {1}        f1a: W whole {0} -> R whole {1}    P0
                   f1b: W whole {0} -> R whole {1}    P1
R part {1}         none; reuse R whole's token
EXIT(i) {0}        m2: R whole {1} -> EXIT(i) {0}     P1
                   m3: R part  {1} -> EXIT(i) {0}     P0
```

```text
              ENTER(i) {0}
                    | walk
                    v
               W whole {0}
                    +-------------------->--------------------+
                    | walk                                    | f1a,f1b
                    v                                         v
                    |                                    R whole {1}
                    |                             +-----<-----+
                    |                             | m2   walk |
                    |                             v           v
                    |                             |      R part {1}
                    |                             |           | m3
                    +--------------<--------------+-----<-----+
                    v
               EXIT(i) {0}
```

Both reads use the token returned by the same owner-`{1}` acquire. The pass
keeps `f1a` first. That edge already gives owner `{1}` the token used by
`R whole`, so the duplicate edge `f1b` is removed:

```text
after removing f1b

              ENTER(i) {0}
                    | walk
                    v
               W whole {0}
                    +-------------------->--------------------+
                    | walk                                    | f1a
                    v                                         v
                    |                                    R whole {1}
                    |                             +-----<-----+
                    |                             | m2   walk |
                    |                             v           v
                    |                             |      R part {1}
                    |                             |           | m3
                    +--------------<--------------+-----<-----+
                    v
               EXIT(i) {0}
```

No other edge is removed. The pass records both exit-edge source nodes before
forming releases. Edges `m2` and `m3` have the same destination and source
owner, so they become one release. Because both reads use the same token, that
release is placed after the later source, `R part`. The result is one count-1
release after `R part`, not two arrivals and not an early release after
`R whole`:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released for owner {0}
m2,m3          EMPTY        {1}              1                same semaphore
f1a            FULL         {0}              1                blocked
```

```text
              ENTER(i) {0}
                    | walk
                    v
         t0 = acquire EMPTY {0}
                 t0 |
                    v
           W whole(i) [t0] {0}
                 t0 |
                    v
     release FULL, t0 [none] {0} f1a
                    +-------------------->--------------------+
                    | walk                                    | FULL
                    v                                         v
                    |                               t1 = acquire FULL {1}
                    |                                      t1 |
                    |                                         v
                    |                                R whole(i) [t1] {1}
                    |                                      t1 | walk
                    |                                         v
                    |                                R part(i) [t1] {1}
                    |                                      t1 |
                    |                                         v
                    |                        release EMPTY, t1 [none] {1} m2,m3
                    |                           +------<------+
               EXIT(i) {0}                      | EMPTY
                    | next iteration            |
                    v                           |
             ENTER(i+1) {0}                     |
                    | walk                      |
                    v                           |
                    +------------ < ------------+
                    v
        next = acquire EMPTY {0}
               next |
                    v
         W whole(i+1) [next] {0}
```

The current DAG dump has exactly two semaphores with pending count 1 and
places the `EMPTY` release after the second read. The pass uses the list of
earlier reads only to choose that release position; it emits no extra IR
operation for the list. The same diagram shows `EMPTY` bypassing `EXIT` and
`ENTER` to reach the next owner-`{0}` acquire.

On the final iteration, no later acquire consumes the release. A zero-trip
loop executes no acquire or release and leaves `EMPTY` released.

### Example: a direct edge is unnecessary

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` has overlapping members:

```text
m0 = [0,256)
m1 = [64,192)

P0 = [0,64)       m0 only
P1 = [64,192)     m0 and m1
P2 = [192,256)    m0 only
```

The loop access order is:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

The complete initial edge inventory is:

```text
DAG node       synchronization edge ending here
ENTER(i) {0}   none
W m0 {0}       none
R m0 {1}       s1a: W m0 {0} -> R m0 {1}       P0
                s1b: W m0 {0} -> R m0 {1}       P1
                s1c: W m0 {0} -> R m0 {1}       P2
W m1 {2}       s2:  W m0 {0} -> W m1 {2}       P1
                s3:  R m0 {1} -> W m1 {2}       P1
R m1 {0}       s4:  W m1 {2} -> R m1 {0}       P1
EXIT(i) {0}    c0a: R m0 {1} -> EXIT(i) {0}    P0
                c0b: R m0 {1} -> EXIT(i) {0}    P2
```

The complete initial DAG keeps every operation as one node. Labels on the
same arrow name the separate piece edges with those endpoints:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +------------------------------------------------- s1a,s1b,s1c > ---------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0a,c0b
                        +------------------------- s2 > --------------------------+------------------------- < s3 --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                         | s4                                                      |
                        +------------------------- < s4 --------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +--------------------------------------------------- < c0a,c0b -----------------------------------------------------+
                        v
                   EXIT(i) {0}
```

There is no edge from `W m1` to `EXIT` for P1. Edge `s4` already makes owner
`{0}` wait for that write at `R m1`, and `R m1` precedes the same owner's
`EXIT`.

At `R m0`, the pass keeps `s1a` first. That wait gives owner `{1}` a token for
the whole group, so `s1b` and `s1c` are unnecessary. The path through `s1a`
and `s3` then makes owner `{2}` wait for the owner-`{0}` write, so `s2` is
unnecessary:

```text
removed edge    wait already provided by
s1b             s1a
s1c             s1a
s2              s1a followed by s3
```

Edges `c0a` and `c0b` survive reduction. They have the same destination and
source owner, so they become one release when semaphores are formed.

The resulting synchronization-edge DAG is:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +----------------------------------------------------- s1a > -------------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0a,c0b
                        |                                                         +------------------------- < s3 --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                         | s4                                                      |
                        +------------------------- < s4 --------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +--------------------------------------------------- < c0a,c0b -----------------------------------------------------+
                        v
                   EXIT(i) {0}
```

The four remaining waits become four count-1 semaphores. The entry row is the
initial state of `EMPTY`; it is not another synchronization edge:

```text
edges          semaphore    release owner    pending_count    initial state
s1a            F01          {0}              1                blocked
s3             F12          {1}              1                blocked
s4             F20          {2}              1                blocked
entry           EMPTY        none             1                released
c0a,c0b         EMPTY        {1}              1                same semaphore
```

The semaphore DAG uses the same lane order as the edge DAG: owner `{0}` on
the left, owner `{2}` in the middle, and owner `{1}` on the right. After
`R m0`, owner `{1}` releases `F12` and then immediately releases `EMPTY` on
one vertical path. `F12` branches left to owner `{2}`; `EMPTY` branches right
to an outside path that bypasses the other owners and the loop boundary:

```text
                  ENTER(i) {0}
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
                W m0(i) [t0] {0}
                     t0 |
                        v
             release F01, t0 {0} s1a
                        +------------------------------------------------------- F01 > ---------------------------------------------------------+
                        | walk                                                                                                              F01 |
                        |                                                                                                                       v
                        |                                                                                                             t1 = acquire F01 {1}
                        |                                                                                                                    t1 |
                        |                                                                                                                       v
                        |                                                                                                               R m0(i) [t1] {1}
                        |                                                                                                                    t1 | walk
                        |                                                                                                                       v
                        |                                                                                                            release F12, t1 {1} s3
                        |                                                           +------------------------- F12 < ---------------------------+
                        |                                                           |                                                      walk |
                        |                                                           |                                                           v
                        |                                                           |                                             release EMPTY, t1 {1} c0a,c0b
                        |                                                           |                                                           +--------- EMPTY > -----------+
                        |                                                           v                                                                                         |
                        |                                                 t2 = acquire F12 {2}                                                                                |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                   W m1(i) [t2] {2}                                                                                  |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                release F20, t2 {2} s4                                                                               |
                        +------------------------- F20 < ---------------------------+                                                                                         |
                        v                                                                                                                                                     |
              t0b = acquire F20 {0}                                                                                                                                           |
                    t0b |                                                                                                                                                     |
                        v                                                                                                                                                     |
                R m1(i) [t0b] {0}                                                                                                                                             |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
                   EXIT(i) {0}                                                                                                                                                |
                        | next iteration                                                                                                                                      |
                        v                                                                                                                                                     |
                 ENTER(i+1) {0}                                                                                                                                               |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
          next = acquire EMPTY(i+1) {0} -------------------------------------------------------------- < EMPTY ---------------------------------------------------------------+
                   next |
                        v
              W m0(i+1) [next] {0}
```

There is no extra `{0}->{2}` semaphore. The test checks these four semaphores
and checks that the loop has no token argument. On iteration zero, the
initially released `EMPTY` supplies `t0`. Re-entry uses the preceding
iteration's `c0a,c0b` release. On the final iteration, no later acquire
consumes that release. A zero-trip loop executes no semaphore operation and
leaves `EMPTY` released.

### Example: a loop-exit edge is unnecessary

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` uses two partly overlapping members:

```text
m0 = [0,128)
m1 = [64,192)

P0 = [0,64)      in m0
P1 = [64,128)    in m0 and m1
P2 = [128,192)   in m1
```

The loop access order is:

```text
W m0 {0}
R m0 {1}
W m1 {2}
R m1 {0}
```

The complete initial edge inventory is:

```text
DAG node       synchronization edge ending here
ENTER(i)       none
W m0 {0}       none
R m0 {1}       l1a: W m0 {0} -> R m0 {1}       P0
                l1b: W m0 {0} -> R m0 {1}       P1
W m1 {2}       l2a: W m0 {0} -> W m1 {2}       P1
                l2b: R m0 {1} -> W m1 {2}       P1
R m1 {0}       l3a: W m1 {2} -> R m1 {0}       P1
                l3b: W m1 {2} -> R m1 {0}       P2
EXIT(i)        c0:  R m0 {1} -> EXIT(i) {0}    P0
                c2:  R m1 {0} -> EXIT(i) {2}    P2
```

There is no edge from `W m1` to `EXIT` for P1. Edges `l3a` and `l3b` already
make owner `{0}` wait for that write at `R m1`, and `R m1` precedes the same
owner's `EXIT`.

The complete initial DAG keeps one node for each operation. Labels on one
arrow retain every piece edge:

```text
                  ENTER(i)
                        | walk
                        v
                   W m0(i) {0}
                        +--------------------------------------------------- l1a,l1b > -----------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0
                        +------------------------ l2a > --------------------------+------------------------ < l2b --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                     | l3a,l3b                                                      |
                        +---------------------- < l3a,l3b -----------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | c2                                                                                                                |
                        +------------------------------------------------------ < c0 -------------------------------------------------------+
                        v
                    EXIT(i)
```

The pass removes four edges. For `c2`, the kept path starts with owner `{0}`'s
program order from `R m1(i)` to `W m0(i+1)`, then uses `l1a` and `l2b` to
reach owner `{2}` at `W m1(i+1)`:

```text
removed edge    wait already provided by
l1b             l1a
l2a             l1a followed by l2b
l3b             l3a
c2              same-owner next-iteration order, then l1a followed by l2b
```

Edge `c0` remains because owner `{0}` needs a token for P0 before the first
access of the next iteration. No other edge is removed.

The final synchronization-edge DAG is:

```text
                  ENTER(i)
                        | walk
                        v
                   W m0(i) {0}
                        +----------------------------------------------------- l1a > -------------------------------------------------------+
                        | walk                                                                                                              v
                        |                                                                                                              R m0(i) {1}
                        |                                                                                                                   | c0
                        |                                                         +------------------------ < l2b --------------------------+
                        |                                                         v                                                         |
                        |                                                    W m1(i) {2}                                                    |
                        |                                                        | l3a                                                       |
                        +------------------------ < l3a -------------------------+                                                         |
                        v                                                                                                                   |
                   R m1(i) {0}                                                                                                              |
                        | walk                                                                                                              |
                        +------------------------------------------------------ < c0 -------------------------------------------------------+
                        v
                    EXIT(i)
```

The emitted POU plan therefore has exactly four semaphores with pending count
1. The entry row records `EMPTY`'s initial state; it is not an edge:

```text
edges        semaphore    release owner    pending_count    initial state
entry         EMPTY        none             1                released
c0            EMPTY        {1}              1                same semaphore
l1a          F01          {0}              1                blocked
l2b          F12          {1}              1                blocked
l3a          F20          {2}              1                blocked
```

As in the previous example, the edge and semaphore DAGs use lanes `{0}`,
`{2}`, `{1}` from left to right. Owner `{1}` releases `F12` and then
immediately releases `EMPTY` on one vertical path. `F12` branches left to
owner `{2}`, while `EMPTY` branches right to the outside recurrence path:

```text
                  ENTER(i)
                        | walk
                        v
            t0 = acquire EMPTY(i) {0}
                     t0 |
                        v
                W m0(i) [t0] {0}
                     t0 |
                        v
             release F01, t0 {0} l1a
                        +------------------------------------------------------- F01 > ---------------------------------------------------------+
                        | walk                                                                                                              F01 |
                        |                                                                                                                       v
                        |                                                                                                             t1 = acquire F01 {1}
                        |                                                                                                                    t1 |
                        |                                                                                                                       v
                        |                                                                                                               R m0(i) [t1] {1}
                        |                                                                                                                    t1 | walk
                        |                                                                                                                       v
                        |                                                                                                           release F12, t1 {1} l2b
                        |                                                           +------------------------- F12 < ---------------------------+
                        |                                                           |                                                      walk |
                        |                                                           |                                                           v
                        |                                                           |                                               release EMPTY, t1 {1} c0
                        |                                                           |                                                           +--------- EMPTY > -----------+
                        |                                                           v                                                                                         |
                        |                                                 t2 = acquire F12 {2}                                                                                |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                   W m1(i) [t2] {2}                                                                                  |
                        |                                                        t2 |                                                                                         |
                        |                                                           v                                                                                         |
                        |                                                release F20, t2 {2} l3a                                                                              |
                        +------------------------- F20 < ---------------------------+                                                                                         |
                        v                                                                                                                                                     |
              t0b = acquire F20 {0}                                                                                                                                           |
                    t0b |                                                                                                                                                     |
                        v                                                                                                                                                     |
                R m1(i) [t0b] {0}                                                                                                                                             |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
                   EXIT(i)                                                                                                                                                    |
                        | next iteration                                                                                                                                      |
                        v                                                                                                                                                     |
                 ENTER(i+1)                                                                                                                                                   |
                        | walk                                                                                                                                                |
                        v                                                                                                                                                     |
          next = acquire EMPTY(i+1) {0} -------------------------------------------------------------- < EMPTY ---------------------------------------------------------------+
                   next |
                        v
              W m0(i+1) [next] {0}
```

The kept path from `R m1(i)` through the next `W m0`, `l1a`, and `l2b`
already reaches owner `{2}` at `W m1(i+1)`, so `c2` is unnecessary. The loop
carries no semaphore token. Iteration zero uses `EMPTY`'s initial state;
re-entry uses `c0`. The final `EMPTY` release has no later acquire. A
zero-trip loop executes no semaphore operation and leaves `EMPTY` released.

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

`test/NVWS/insert_semas_placement_modes.mlir` `@placement_mode` uses the
running loop:

```text
for {
  W m0 {0}
  R m0 {1}
}
```

The complete edge inventory is identical in both modes:

```text
DAG node       synchronization edge ending here
ENTER(i) {0}   none
W m0 {0}       none
R m0 {1}       e1: W m0 {0} -> R m0 {1}
EXIT(i) {0}    e2: R m0 {1} -> EXIT(i) {0}
```

No edge is removed or merged. The synchronization-edge DAG is:

```text
                  ENTER(i) {0}
                        | walk
                        v
                   W m0(i) {0}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                    R m0(i) {1}
                        +----------------- < e2 ------------------+
                        v
                   EXIT(i) {0}
```

Both placement modes use the same semaphore assignment:

```text
edge / role    semaphore    release owner    pending_count    initial state
e1             FULL         {0}              1                blocked
entry           EMPTY        none             1                released
e2              EMPTY        {1}              1                same semaphore
```

Auto selects POU for this input:

```text
                  ENTER(i) {0}
                        | walk
                        v
            tw = acquire EMPTY(i) {0}
                     tw |
                        v
                W m0(i) [tw] {0}
                     tw | walk
                        v
             release FULL, tw {0} e1
                        +---------------- FULL > -----------------+
                   walk |                                         v
                        |                               tr = acquire FULL {1}
                        |                                      tr |
                        |                                         v
                        |                                 R m0(i) [tr] {1}
                        |                                      tr | walk
                        |                                         v
                        |                             release EMPTY, tr {1} e2
                        |                                         |
                   EXIT(i) {0}                                    |
                        | next iteration                          |
                        v                                         |
                 ENTER(i+1) {0}                                   |
                        | walk                                    |
                        v                                         |
                        +--------------- < EMPTY -----------------+
                        v
          tw2 = acquire EMPTY(i+1) {0}
                    tw2 |
                        v
               W m0(i+1) [tw2] {0}
```

The loop does not carry a semaphore token. `EMPTY` is acquired immediately
before each write that needs it.

FirstTouch:

```text
         initial = acquire EMPTY at root
                initial |
                        v
         scf.for iter_arg carry=initial
                  carry |
                        v
                  ENTER(i) {0}
                  carry |
                        v
               W m0(i) [carry] {0}
                  carry | walk
                        v
           release FULL, carry {0} e1
                        +---------------- FULL > -----------------+
                   walk |                                         v
                        |                               tr = acquire FULL {1}
                        |                                      tr |
                        |                                         v
                        |                                 R m0(i) [tr] {1}
                        |                                      tr | walk
                        |                                         v
                        |                             release EMPTY, tr {1} e2
                        +--------------- < EMPTY -----------------+
                        v
            next = acquire EMPTY {0}
```

If the loop continues, that `next` token becomes the next iteration's
`carry`:

```text
            next = acquire EMPTY {0}
                   next |
                        v
               EXIT(i) yields next
                        | next iteration
                        v
                 ENTER(i+1) {0}
             carry=next |
                        v
              W m0(i+1) [carry] {0}
```

If the executed iteration is final, the same token becomes the loop result:

```text
            next = acquire EMPTY {0}
                   next |
                        v
            EXIT(last) yields next
                        | loop finishes
                        v
                  result=next
```

On a zero-trip loop, the body does not execute and the root token is the
result:

```text
         initial = acquire EMPTY at root
                initial |
                        v
      scf.for executes zero trips
                initial |
                        v
               result=initial
```

If owner `{0}` already has a valid token before a FirstTouch loop, the loop
can reuse it instead of creating the root acquire shown here. POU consumes
the initially released `EMPTY` at the first executed write. FirstTouch
consumes it before the loop. On the final iteration, FirstTouch returns its
last `next` token; POU leaves its last release unconsumed. A zero-trip
FirstTouch loop returns `initial`, while a zero-trip POU loop executes no
semaphore operation and leaves `EMPTY` released.

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

The pass treats each touch as a write, so the inner edge inventory is:

```text
DAG node            synchronization edge ending here
ENTER inner {0}     none
touch0 m0 {0}       none
touch1 m0 {0}       none; same owner as touch0
touch2 m0 {1}       e1: touch1 {0} -> touch2 {1}
EXIT inner {0}      e2: touch2 {1} -> EXIT inner {0}
```

There is one piece, P0. The outer DAG contains only the inner summary between
owner-`{0}` boundaries, so it has no synchronization edge:

```text
                  ENTER outer {0}
                        | walk
                        v
         [inner summary P0:W:{0}]
                        | walk
                        v
                   EXIT outer {0}
```

No inner edge is removed or merged. Its synchronization-edge DAG is:

```text
                  ENTER inner {0}
                        | walk
                        v
                 touch0 m0 {0}
                        | walk
                        v
                 touch1 m0 {0}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                touch2 m0 {1}
                        +----------------- < e2 ------------------+
                        v
                   EXIT inner {0}
```

The completed FirstTouch plan uses three count-1 semaphores:

```text
edge / role    semaphore    release owner    pending_count    initial state
root entry     ENTRY        none             1                released
e1             TO1          {0}              1                blocked
e2             NEXT         {1}              1                blocked
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
point-of-use placement is unavailable for this loop:
fixed loop.stage constraints require a carried recurrence
```

The final plan carries the same token through both loops. The common body
keeps owner `{0}` on the left and owner `{1}` on the right. The two
same-owner touches use `carry` in program order:

```text
     initial = acquire ENTRY at root
            initial |
                    v
           ENTER outer(i) {0}
         outerToken = initial
         outerToken |
                    v
          ENTER inner(i,j) {0}
             carry = outerToken
              carry |
                    v
          touch0 m0 [carry] {0}
              carry | walk
                    v
          touch1 m0 [carry] {0}
              carry |
                    v
        release TO1, carry {0} e1
                    +------------------- > -------------------+
                    | walk                                    | TO1
                    v                                         v
                    |                               t1 = acquire TO1 {1}
                    |                                      t1 |
                    |                                         v
                    |                                touch2 m0 [t1] {1}
                    |                                      t1 | walk
                    |                                         v
                    |                              release NEXT, t1 {1} e2
                    +------------------- < -------------------+
                    v
         next = acquire NEXT {0}
               next |
```

If the inner loop continues, `next` becomes its next `carry`:

```text
         next = acquire NEXT {0}
               next |
                    v
       EXIT inner(i,j) yields next
               next | next inner iteration
                    v
         ENTER inner(i,j+1) {0}
             carry = next
              carry |
                    v
          touch0 m0 [carry] {0}
```

If the inner loop finishes and the outer loop continues, `next` becomes the
inner result and then the next outer token:

```text
         next = acquire NEXT {0}
               next |
                    v
     EXIT inner(i,last) yields next
                    |
                    v
           innerResult = next
        innerResult |
                    v
    EXIT outer(i) yields innerResult
                    | next outer iteration
                    v
          ENTER outer(i+1) {0}
        outerToken = innerResult
         outerToken |
                    v
         ENTER inner(i+1,0) {0}
             carry = outerToken
              carry |
                    v
          touch0 m0 [carry] {0}
```

A zero-trip inner loop returns its incoming outer token:

```text
               outerToken
          outerToken | inner loop has zero trips
                    v
        innerResult = outerToken
```

When the outer loop finishes, it returns the inner result:

```text
               innerResult
        innerResult |
                    v
   EXIT outer(last) yields innerResult
                    |
                    v
        outerResult = innerResult
```

A zero-trip outer loop returns the root token unchanged:

```text
     initial = acquire ENTRY at root
            initial | outer loop has zero trips
                    v
          outerResult = initial
```

If the inner loop continues, `next` is its next `carry`. If it finishes,
`next` becomes the inner result and then the outer loop's carried token. The
emitted result contains nothing from the discarded POU attempt. A zero-trip
inner loop returns `outerToken` unchanged. A zero-trip outer loop returns
`initial` unchanged. After the final executed inner iteration, its `next`
token is returned through both loops.

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
access the buffer. The exact edge inventory is:

```text
parent DAG node                  synchronization edge ending here
ENTER loop {1}                  none
W MMA {1}                       none
[if summary P0:R:{1}]           none
EXIT loop {1}                   none

then DAG node                   synchronization edge ending here
ENTER if {1}                    none
R acc {0}                       e1: ENTER if {1} -> R acc {0}
EXIT if {1}                     e2: R acc {0} -> EXIT if {1}

else DAG node                   synchronization edge ending here
ENTER if {1}                    none
EXIT if {1}                     none
```

The parent and child DAGs are separate:

```text
parent

                  ENTER loop {1}
                        | walk
                        v
                   W MMA {1}
                        | walk
                        v
         [if summary P0:R:{1}]
                        | walk
                        v
                   EXIT loop {1}

then child

                  ENTER if {1}
                        +----------------- e1 > ------------------+
                   walk |                                         v
                        |                                    R acc {0}
                        +----------------- < e2 ------------------+
                        v
                   EXIT if {1}

else child

                  ENTER if {1}
                        | walk
                        v
                   EXIT if {1}
```

The parent has no synchronization edge around the `if`. The two then-path
edges remain unchanged, and the else path has no synchronization edge. No edge
is removed or merged.

For `e1`, owner `{0}` must wait for owner `{1}`. The release that allows the
next write comes from one of two places:

- the owner-`{0}` read when the then branch executes; or
- the owner-`{1}` MMA when the else branch executes.

Only one branch executes, so only one of these releases executes.
`EMPTY.pending_count` is therefore 1, not 2. The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
e1             FULL         {1}              1                blocked
entry           EMPTY        none             1                released
e2              EMPTY        {0}              1                same semaphore (then)
else supply     EMPTY        {1}              1                same semaphore (else)
```

The else release is not a synchronization edge. It supplies the same next
`EMPTY` phase that `e2` supplies when the then branch runs.

The two complete paths below repeat the common acquire and MMA so that each
path can be followed without jumping between diagrams. They are exclusive
control paths, not parallel lanes. Owner `{1}` stays on the left and owner
`{0}` stays on the right.

On the then path:

```text
                    ENTER loop(i) {1}
                            | walk
                            v
                tw = acquire EMPTY(i) {1}
                         tw |
                            v
                    W MMA(i) [tw] {1}
                            | walk
                            v
                       scf.if cond
                            | then
                            v
                      ENTER if {1}
                         tw | walk
                            v
            release FULL, tw [tc5mma] {1} e1
                            +------------------- FULL > --------------------+
                       walk |                                               v
                            |                                     tr = acquire FULL {0}
                            |                                            tr |
                            |                                               v
                            |                                       R acc(i) [tr] {0}
                            |                                            tr | walk
                            |                                               v
                            |                                release EMPTY, tr [none] {0} e2
                            |                                               +-------------- EMPTY > ----------------+
                            v                                                                                       |
                   EXIT if (then) {1}                                                                               |
                            | branch completes                                                                      |
                            v                                                                                       |
                    EXIT loop(i) {1}                                                                                |
                            | next iteration                                                                        |
                            v                                                                                       |
                   ENTER loop(i+1) {1}                                                                              |
                            | walk                                                                                  |
                            v                                                                                       |
              next = acquire EMPTY(i+1) {1} ------------------------------------<-----------------------------------+
                       next |
                            v
                  W MMA(i+1) [next] {1}
```

On the else path, owner `{0}` does not execute:

```text
                    ENTER loop(i) {1}
                            | walk
                            v
                tw = acquire EMPTY(i) {1}
                         tw |
                            v
                    W MMA(i) [tw] {1}
                            | walk
                            v
                       scf.if cond
                            | else
                            v
                      ENTER if {1}
                         tw | walk
                            v
             release EMPTY, tw [tc5mma] {1}
                            +------------------ EMPTY > --------------------+
                       walk |                                               |
                            v                                               |
                   EXIT if (else) {1}                                       |
                            | branch completes                              |
                            v                                               |
                    EXIT loop(i) {1}                                        |
                            | next iteration                                |
                            v                                               |
                   ENTER loop(i+1) {1}                                      |
                            | walk                                          |
                            v                                               |
              next = acquire EMPTY(i+1) {1} ----------------<---------------+
                       next |
                            v
                  W MMA(i+1) [next] {1}
```

Whichever branch runs, its one `EMPTY` release supplies the next reuse. The
two diagrams are alternatives, not two inputs to one acquire.

Only one `EMPTY` release executes each time the `if` runs. Neither the `if`
nor the loop returns a semaphore token: each branch releases `EMPTY`, and the
next buffer reuse acquires it. The `FULL` acquire and release remain inside
the then branch. The else release still waits for the MMA completion through
`[tc5mma]`.
On the final iteration, no later acquire consumes the selected `EMPTY`
release. A zero-trip loop executes neither branch and leaves `EMPTY` released.

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

There is one piece, P0. The complete edge inventory is:

```text
DAG node         synchronization edge ending here
ENTER(i) {0}     none
W first {0}      none
R reader1 {1}    e1: W first {0} -> R reader1 {1}
R reader2 {2}    e2: W first {0} -> R reader2 {2}
W final {0}      e3: R reader1 {1} -> W final {0}
                 e4: R reader2 {2} -> W final {0}
EXIT(i) {0}      none
```

No edge is removed or merged. The two reader paths are independent and join
at the final write:

```text
                                                        ENTER(i) {0}
                                                              | walk
                                                              v
                                                       W first(i) {0}
                    +----------------- < e1 ------------------+----------------- e2 > ------------------+
                    v                                    walk |                                         v
            R reader1(i) {1}                                  |                                 R reader2(i) {2}
                    +----------------- e3 > ------------------+----------------- < e4 ------------------+
                                                              v
                                                       W final(i) {0}
                                                              | walk
                                                              v
                                                         EXIT(i) {0}
```

Edges `e3` and `e4` share a destination and destination owner, so they use
one semaphore and acquire. Their source owners differ, so they remain two
releases. The entry row is the initial `EMPTY` state:

```text
edges        semaphore    release owner    pending_count    initial state
e1           TO_R1        {0}              1                blocked
e2           TO_R2        {0}              1                blocked
entry         EMPTY        none             2                released
e3            EMPTY        {1}              2                same semaphore
e4            EMPTY        {2}              2                same semaphore
```

Both readers release the count-2 `EMPTY` semaphore. Its acquire must occur
before `W final(i)`, so that token already exists at the end of iteration `i`.
The next iteration's `W first(i+1)` can reuse the same token. The complete
semaphore DAG keeps the two owner-`{0}` releases on one ordered spine and the
reader paths separate:

```text
                                             initial = acquire EMPTY(2) at root
                                                      initial |
                                                              v
                                               scf.for iter_arg carry=initial
                                                              +---------------------------------- > ------------------------------------+
                                                     executes |                                                               zero trip |
                                                              v                                                                         v
                                                        ENTER(i) {0}                                                             result=initial
                                                        carry |
                                                              v
                                                   W first(i) [carry] {0}
                                                        carry | walk
                                                              v
                                                 release TO_R1, carry {0} e1
                    +--------------- < TO_R1 -----------------+
                    v                                    walk |
         t1 = acquire TO_R1 {1}                  release TO_R2, carry {0} e2
                 t1 |                                         +--------------- TO_R2 > -----------------+
                    v                                         |                                         v
          R reader1(i) [t1] {1}                               |                              t2 = acquire TO_R2 {2}
                 t1 | walk                                    |                                      t2 |
                    v                                         |                                         v
        release EMPTY, t1 {1} e3                              |                               R reader2(i) [t2] {2}
                    |                                         |                                      t2 | walk
                    |                                         |                                         v
                    |                                         |                             release EMPTY, t2 {2} e4
                    +--------------- EMPTY > -----------------+--------------- < EMPTY -----------------+
                                                              v
                                                 next = acquire EMPTY(2) {0}
                                                         next |
                                                              v
                                                    W final(i) [next] {0}
                                                         next | walk
                                                              v
                                                     EXIT(i) yields next
                                                              +---------------------------------- > ------------------------------------+
                                               next iteration |                                                           loop finishes |
                                                              v                                                                         v
                                                       ENTER(i+1) {0}                                                              result=next
                                                   carry=next |
                                                              v
                                                  W first(i+1) [carry] {0}
```

POU and FirstTouch produce the same placement here. The count-2 acquire is
needed by `W final`, and its token remains valid for `W first` in the next
iteration. The loop therefore carries `next`. A zero-trip loop returns the
`initial` token, which was acquired before the loop, unchanged.

After the final iteration, the loop returns `next` as its result.

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
owner as the outer `ENTER` and `EXIT`. The parent inventory has no
synchronization edge:

```text
parent DAG node                  synchronization edge ending here
ENTER outer(i) {1}              none
[inner summary P0:W:{1}]        none
EXIT outer(i) {1}               none
```

The parent synchronization-edge DAG is:

```text
                         ENTER outer(i) {1}
                                  | walk
                                  v
                   [inner summary P0:W:{1}]
                                  | walk
                                  v
                         EXIT outer(i) {1}
```

The child inventory has two edges:

```text
DAG node                 synchronization edge ending here
ENTER inner(i,j) {1}     none
W acc(i,j) {1}           none
R acc(i,j) {0}           c1: W acc(i,j) {1} -> R acc(i,j) {0}
EXIT inner(i,j) {1}      c2: R acc(i,j) {0} -> EXIT inner(i,j) {1}
```

```text
                  ENTER inner(i,j) {1}
                            | walk
                            v
                     W acc(i,j) {1}
                            +-------------------c1 >--------------------+
                            | walk                                      v
                            |                                    R acc(i,j) {0}
                            +-------------------< c2--------------------+
                            v
                   EXIT inner(i,j) {1}
```

No edge is removed or merged.

`c1` becomes `FULL`. Edge `c2` and the initial ready state use `EMPTY`:

```text
edge / role    semaphore    release owner    pending_count    initial state
c1             FULL         {1}              1                blocked
entry           EMPTY        none             1                released
c2              EMPTY        {0}              1                same semaphore
```

The pass places both acquires immediately before the inner buffer accesses
that need them. The parent DAG contains the child as one summary. Each
complete alternative below includes the body so that its `c2` release has a
visible source. If the next inner turn is in the same outer iteration,
the owner-`{1}` control path crosses the loop boundary while the `EMPTY` rail
bypasses it and ends at the next acquire:

```text
                  ENTER inner(i,j) {1}
                            | walk
                            v
                 tw = acquire EMPTY {1}
                         tw |
                            v
                   W acc(i,j) [tw] {1}
                            | walk
                            v
            release FULL, tw [tc5mma] {1} c1
                            +----------------- FULL > ------------------+
                       walk |                                           v
                            |                                 tr = acquire FULL {0}
                            |                                        tr |
                            |                                           v
                            |                                  R acc(i,j) [tr] {0}
                            |                                        tr | walk
                            |                                           v
                            |                            release EMPTY, tr [none] {0} c2
                            |                                           +-------------- EMPTY > ----------------+
                            v                                                                                   |
                   EXIT inner(i,j) {1}                                                                          |
                            | next inner iteration                                                              |
                            v                                                                                   |
                 ENTER inner(i,j+1) {1}                                                                         |
                            | walk                                                                              |
                            v                                                                                   |
                next = acquire EMPTY {1} -----------------------------------<-----------------------------------+
                       next |
                            v
                 W acc(i,j+1) [next] {1}
```

If the next executed inner turn is in a later outer iteration, the same
release stays ready while control crosses both loop boundaries. It ends at
the first acquire in that later inner-loop execution:

```text
                 ENTER inner(i,last) {1}
                            | walk
                            v
                 tw = acquire EMPTY {1}
                         tw |
                            v
                 W acc(i,last) [tw] {1}
                            | walk
                            v
            release FULL, tw [tc5mma] {1} c1
                            +----------------- FULL > ------------------+
                       walk |                                           v
                            |                                 tr = acquire FULL {0}
                            |                                        tr |
                            |                                           v
                            |                                R acc(i,last) [tr] {0}
                            |                                        tr | walk
                            |                                           v
                            |                            release EMPTY, tr [none] {0} c2
                            |                                           +-------------- EMPTY > ----------------+
                            v                                                                                   |
                 EXIT inner(i,last) {1}                                                                         |
                            | inner finishes                                                                    |
                            v                                                                                   |
                    EXIT outer(i) {1}                                                                           |
                            | later outer iteration                                                             |
                            v                                                                                   |
                   ENTER outer(k) {1}                                                                           |
                            | walk                                                                              |
                            v                                                                                   |
                  ENTER inner(k,0) {1}                                                                          |
                            | walk                                                                              |
                            v                                                                                   |
                first = acquire EMPTY {1} -----------------------------------<----------------------------------+
                      first |
                            v
                 W acc(k,0) [first] {1}
```

The two alternatives are exclusive. Neither loop carries a semaphore token.

After the final inner iteration, its `EMPTY` release has no following acquire
in that run of the inner loop. It remains ready for the next time the inner
loop executes, including in a later outer iteration. Neither loop carries a
semaphore token, and no acquire or release is moved outside both loops.
For the first executed inner iteration, `EMPTY`'s initially released state
supplies `tw`. EMIT-IR removes the old tokens attached to the TMEM operations
because the semaphores now order the accesses. A zero-trip inner or outer loop
executes no acquire or release, so the ready `EMPTY` state remains available.

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
parent synchronization-edge DAG

                  ENTER outer(i) {1}
                        | walk
                        v
          [inner summary P0:W:{1}]
                        +----------------- p1 > ------------------+
                   walk |                                         v
                        |                                 R outer(i) {0}
                        +----------------- < p2 ------------------+
                        v
                  EXIT outer(i) {1}

child synchronization-edge DAG

                  ENTER inner(i,j) {1}
                        | walk
                        v
                  W inner(i,j) {1}
                        +----------------- c1 > ------------------+
                   walk |                                         v
                        |                                 R inner(i,j) {0}
                        +----------------- < c2 ------------------+
                        v
                  EXIT inner(i,j) {1}
```

No edge is removed or merged. Parent and child edges remain separate.

The pass forms four count-1 semaphores:

```text
LOCAL_EMPTY    next inner write or acquire after the inner loop; initially released
LOCAL_FULL     inner write -> inner read
OUTER_FULL     completed inner loop -> outer read
OUTER_EMPTY    outer read -> owner {1}; initially released
```

The semaphore assignment is:

```text
edge / role        semaphore      release owner       pending_count    initial state
c1                 LOCAL_FULL     {1}                 1                blocked
entry               LOCAL_EMPTY    none                1                released
c2                  LOCAL_EMPTY    {0}                 1                same semaphore
prepare             LOCAL_EMPTY    {1}                 1                same semaphore
p1                 OUTER_FULL     {1}                 1                blocked
entry               OUTER_EMPTY    none                1                released
p2                  OUTER_EMPTY    {0}                 1                same semaphore
```

An acquire before the outer loop drains `OUTER_EMPTY`'s initially released
state. Its token is not used by a buffer access:

```text
initial = acquire OUTER_EMPTY at root {1}    token unused
```

When the inner loop continues, `c2` supplies the next inner acquire. Owner
`{1}` stays on the left, owner `{0}` stays on the right, and the
`LOCAL_EMPTY` rail bypasses the loop boundary:

```text
              ENTER inner(i,j) {1}
                        | walk
                        v
         wtok = acquire LOCAL_EMPTY {1}
                   wtok |
                        v
             W inner(i,j) [wtok] {1}
                        | walk
                        v
    release LOCAL_FULL, wtok [tc5mma] {1} c1
                        +------------- LOCAL_FULL > --------------+
                   walk |                                         v
                        |                           rtok = acquire LOCAL_FULL {0}
                        |                                    rtok |
                        |                                         v
                        |                              R inner(i,j) [rtok] {0}
                        |                                    rtok | walk
                        |                                         v
                        |                      release LOCAL_EMPTY, rtok [none] {0} c2
                        |                                         +------------- LOCAL_EMPTY > ---------------+
                        v                                                                                     |
               EXIT inner(i,j) {1}                                                                            |
                        | next inner iteration                                                                |
                        v                                                                                     |
             ENTER inner(i,j+1) {1}                                                                           |
                        | walk                                                                                |
                        v                                                                                     |
         next = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   next |
                        v
            W inner(i,j+1) [next] {1}
```

If an executed inner loop finishes, `done` consumes `c2`. For a zero-trip
inner loop, `done` instead consumes `LOCAL_EMPTY`'s initial state on the first
outer iteration or the previous `prepare` release on a later iteration. The
complete executed-inner path and the post-loop handoff are:

```text
             ENTER inner(i,last) {1}
                        | walk
                        v
         wtok = acquire LOCAL_EMPTY {1}
                   wtok |
                        v
           W inner(i,last) [wtok] {1}
                        | walk
                        v
    release LOCAL_FULL, wtok [tc5mma] {1} c1
                        +------------- LOCAL_FULL > --------------+
                   walk |                                         v
                        |                           rtok = acquire LOCAL_FULL {0}
                        |                                    rtok |
                        |                                         v
                        |                            R inner(i,last) [rtok] {0}
                        |                                    rtok | walk
                        |                                         v
                        |                      release LOCAL_EMPTY, rtok [none] {0} c2
                        |                                         +------------- LOCAL_EMPTY > ---------------+
                        v                                                                                     |
             EXIT inner(i,last) {1}                                                                           |
                        | loop finishes                                                                       |
                        v                                                                                     |
         done = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   done |
                        v
     release OUTER_FULL, done [none] {1} p1
                        +------------- OUTER_FULL > --------------+
                   walk |                                         v
                        |                            to = acquire OUTER_FULL {0}
                        |                                      to |
                        |                                         v
                        |                                R outer(i) [to] {0}
                        |                                      to | walk
                        |                                         v
                        |                       release OUTER_EMPTY, to [none] {0} p2
                        |                                         |
        prepare = acquire OUTER_EMPTY {1} -----< OUTER_EMPTY -----+
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
```

The `prepare` release has two exclusive consumers in the next outer
iteration. When the inner loop executes, it supplies the first inner acquire:

```text
        prepare = acquire OUTER_EMPTY {1}
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
                        +---------------------------------- LOCAL_EMPTY > ------------------------------------+
                   walk |                                                                                     |
                        v
                EXIT outer(i) {1}                                                                             |
                        | next outer iteration                                                                |
                        v
              ENTER outer(i+1) {1}                                                                            |
                        | walk                                                                                |
                        v
             ENTER inner(i+1,0) {1}                                                                           |
                        | walk                                                                                |
                        v                                                                                     |
         first = acquire LOCAL_EMPTY {1} ----------------------------------<----------------------------------+
                  first |
                        v
           W inner(i+1,0) [first] {1}
```

When that inner loop has zero trips, the same `prepare` release supplies the
post-loop `done` acquire instead:

```text
        prepare = acquire OUTER_EMPTY {1}
                prepare |
                        v
     release LOCAL_EMPTY, prepare [none] {1}
                        +---------------------------------- LOCAL_EMPTY > ------------------------------------+
                   walk |                                                                                     |
                        v
                EXIT outer(i) {1}                                                                             |
                        | next outer iteration                                                                |
                        v
              ENTER outer(i+1) {1}                                                                            |
                        | inner scf.for executes zero trips                                                   |
                        v                                                                                     |
         done = acquire LOCAL_EMPTY {1} ----------------------------------<-----------------------------------+
                   done |
                        v
```

The root drain makes the later `prepare` acquire wait for the outer read's
`p2` release. Neither loop carries a semaphore token. If the outer loop is
zero-trip, only the unconditional root drain executes.

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
parent synchronization-edge DAG

                  ENTER outer(i) {0}
                        | walk
                        v
                   W outer(i) {0}
                        +----------------- p1 > ------------------+
                   walk |                                         v
                        |                       [inner summary P0:W:{1}]
                        +----------------- < p2 ------------------+
                        v
                   R post(i) {0}
                        | walk
                        v
                  EXIT outer(i) {0}

child synchronization-edge DAG

                                                        ENTER inner(i,j) {1}
                                                                  | walk
                                                                  v
                                                           W MMA(i,j) {1}
                        +----------------- < c1 ------------------+
                        v                                    walk |
                R inner(i,j) {0}                                  |
                        +----------------- c2 > ------------------+
                                                                  v
                                                         EXIT inner(i,j) {1}
```

No edge is removed or merged.

All four edges use count-one semaphores:

```text
edge / role    semaphore      release owner    pending_count    initial state
entry,p2       OUTER_EMPTY    {1}              1                released
p1,c2          LOCAL_EMPTY    {0}              1                blocked
c1             LOCAL_FULL     {1}              1                blocked
```

The generated schedule locations are:

```text
operation                                    owner    cluster    stage
acquire LOCAL_EMPTY at inner MMA             {1}      3          0
release LOCAL_FULL after MMA [tc5mma]         {1}      3          0
acquire LOCAL_FULL at inner read              {0}      2          1
release LOCAL_EMPTY after inner read          {0}      2          1
post-inner acquire LOCAL_EMPTY                {1}      inner boundary
release OUTER_EMPTY after inner loop          {1}      3          0
acquire OUTER_EMPTY at post-inner read        {0}      4          0
```

The root acquire and outer token stay in owner `{0}`. The inner loop has no
token `iter_arg`. The diagrams use `[cN,sM]` for cluster `N`, stage `M` and
keep owner `{0}` on the left and owner `{1}` on the right.

When the inner loop executes and continues, `p1` supplies its first acquire
and `c2` supplies the next one. Both semaphore rails end at the acquires;
neither rail flows through an `ENTER` or `EXIT`:

```text
            initial = acquire OUTER_EMPTY at root
                      initial |
                              v
                scf.for iter_arg out=initial
                              | executes
                              v
                     ENTER outer(i) {0}
                          out |
                              v
                    W outer(i) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
                              +----------------- enter inner > -------------------+
                  LOCAL_EMPTY |                                                   v
                              |                                         ENTER inner(i,0) {1}
                              |                                                   | walk
                              |                                                   v
                              +-------- LOCAL_EMPTY > ---------first = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                            first |
                                                                                  v
                                                                   W MMA(i,0) [first] {1} [c3,s0]
                                                                            first | walk
                                                                                  v
                                                          release LOCAL_FULL, first [tc5mma] {1} c1 [c3,s0]
                              +------------------ LOCAL_FULL < -------------------+
                              v                                              walk |
             tr = acquire LOCAL_FULL {0} [c2,s1]                                  |
                           tr |                                                   |
                              v                                                   |
                R inner(i,0) [tr] {0} [c2,s1]                                     |
                           tr | walk                                              |
                              v                                                   |
        release LOCAL_EMPTY, tr [none] {0} c2 [c2,s1]                             |
                  LOCAL_EMPTY |                                                   v
                              |                                          EXIT inner(i,0) {1}
                              |                                                   | next inner iteration
                              |                                                   v
                              |                                         ENTER inner(i,1) {1}
                              |                                                   | walk
                              |                                                   v
                              +-------- LOCAL_EMPTY > ---------next = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                             next |
                                                                                  v
                                                                    W MMA(i,1) [next] {1} [c3,s0]
```

On the final executed inner iteration, `c2` bypasses `EXIT inner` and ends at
the unstamped post-loop acquire. The following `p2` release uses `[c3,s0]`;
the post-inner read acquires `OUTER_EMPTY` at `[c4,s0]`. That token is the
outer loop's carried `out` token:

```text
                                                                       ENTER inner(i,last) {1}
                                                                                  | walk
                                                                                  v
                                                               wtok = acquire LOCAL_EMPTY {1} [c3,s0]
                                                                             wtok |
                                                                                  v
                                                                  W MMA(i,last) [wtok] {1} [c3,s0]
                                                                             wtok | walk
                                                                                  v
                                                          release LOCAL_FULL, wtok [tc5mma] {1} c1 [c3,s0]
                              +------------------ LOCAL_FULL < -------------------+
                              v                                              walk |
             tr = acquire LOCAL_FULL {0} [c2,s1]                                  |
                           tr |                                                   |
                              v                                                   |
              R inner(i,last) [tr] {0} [c2,s1]                                    |
                           tr | walk                                              |
                              v                                                   |
        release LOCAL_EMPTY, tr [none] {0} c2 [c2,s1]                             |
                  LOCAL_EMPTY |                                                   v
                              |                                        EXIT inner(i,last) {1}
                              |                                                   | loop finishes
                              |                                                   v
                              +------- LOCAL_EMPTY > --------done = acquire LOCAL_EMPTY {1} [unstamped]
                                                                             done |
                                                                                  v
                                                           release OUTER_EMPTY, done [none] {1} p2 [c3,s0]
                                                                                  |
            out = acquire OUTER_EMPTY {0} [c4,s0] -------- < OUTER_EMPTY ---------+
                          out |
                              v
                 R post(i) [out] {0} [c4,s0]
                          out | walk
                              v
                  EXIT outer(i) yields out
                              +------------------------------ loop finishes > --------------------------------+
                              | next outer iteration                                                          v
                              v                                                                          result=out
                ENTER outer(i+1) receives out
                          out |
                              v
                   W outer(i+1) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
```

When the inner loop has zero trips, the same real `p1` release supplies the
same unstamped `done` acquire. The control path completes the zero-trip loop;
the `LOCAL_EMPTY` rail remains separate until `done`:

```text
                     ENTER outer(i) {0}
                          out |
                              v
                    W outer(i) [out] {0}
                          out | walk
                              v
               release LOCAL_EMPTY, out {0} p1
                              +----------------- enter inner > -------------------+
                  LOCAL_EMPTY |                                                   v
                              |                                   inner scf.for executes zero trips
                              |                                                   | loop finishes
                              |                                                   v
                              +------- LOCAL_EMPTY > --------done = acquire LOCAL_EMPTY {1} [unstamped]
                                                                             done |
                                                                                  v
                                                           release OUTER_EMPTY, done [none] {1} p2 [c3,s0]
```

Thus `p1` chooses first-inner versus zero-trip `done`, and `c2` chooses
next-inner versus final `done`; none of those alternatives execute together.
If the outer loop is zero-trip, it returns `initial`; after the final executed
outer iteration, it returns `out`.

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
parent synchronization-edge DAG

                   ENTER outer(i) {0}
                            | walk
                            v
                     W outer(i) {0}
                            +-------------------p1 >--------------------+
                            | walk                                      v
                            |                               [inner summary P0:W:{1}]
                            +-------------------< p2--------------------+
                            v
                      R post(i) {0}
                            | walk
                            v
                    EXIT outer(i) {0}
```

```text
inner synchronization-edge DAG

                                                              ENTER inner(i,j) {1}
                                                                        | walk
                                                                        v
                                                                 W mma0(i,j) {1}
                                                                        | walk
                                                                        v
                                                              [if summary P0:W:{1}]
                            +-------------------< c2--------------------+
                            v
                    R final(i,j) {0}
                            +-------------------c3 >--------------------+
                                                                        v
                                                               EXIT inner(i,j) {1}
```

```text
then-child synchronization-edge DAG

                                                                  ENTER if {1}
                                                                        | walk
                                                                        v
                                                                   W mma1 {1}
                            +-------------------< c1--------------------+
                            v
                      R branch {0}
                            +-------------------b2 >--------------------+
                                                                        v
                                                                   EXIT if {1}
```

```text
else-child synchronization-edge DAG

                                                                  ENTER if {1}
                                                                        | walk
                                                                        v
                                                                   EXIT if {1}
```

No edge is removed or merged. The parent, inner, and `if` child DAGs remain
separate.

On the then path, `R branch` holds the token used to release `CONVERGE` at
stage 1. On the else path, the release uses the token from `mma0` and waits
for its completion at stage 0. Only one path runs, so the two releases do not
add their counts:

```text
edge / role        semaphore       release owner       pending_count    initial state
entry,p2           OUTER_EMPTY     {1}                 1                released
p1,c3              LOCAL_EMPTY     {0}                 1                blocked
c1                 BRANCH_FULL     {1}                 1                blocked
b2,c2 alternatives CONVERGE        then:{0}; else:{1}  1                blocked
```

All semaphore views use owner `{0}` on the left and owner `{1}` on the right.
`[cN,sM]` means cluster `N`, stage `M`. The two `if` paths are mutually
exclusive, so each complete view below executes exactly one count-1
`CONVERGE` release.

This complete then-path view shows inner-loop continuation. The parent `p1`
rail ends at `tw`; the then-path `CONVERGE` release passes `EXIT if` and ends
at `tf`; and `c3` bypasses the inner-loop boundary and ends at `next`:

```text
                initial = acquire OUTER_EMPTY at root
                          initial |
                                  v
                    scf.for iter_arg out=initial
                                  | executes
                                  v
                         ENTER outer(i) {0}
                              out |
                                  v
                        W outer(i) [out] {0}
                              out | walk
                                  v
                   release LOCAL_EMPTY, out {0} p1
                                  +-------------------- enter inner > ----------------------+
                      LOCAL_EMPTY |                                                         v
                                  |                                               ENTER inner(i,j) {1}
                                  |                                                         | walk
                                  |                                                         v
                                  +----------- LOCAL_EMPTY > -------------tw = acquire LOCAL_EMPTY {1} [c5,s0]
                                                                                         tw |
                                                                                            v
                                                                              W mma0(i,j) [tw] {1} [c5,s0]
                                                                                         tw | walk
                                                                                            v
                                                                                       scf.if cond
                                                                                            | then
                                                                                            v
                                                                                   ENTER if (then) {1}
                                                                                         tw | walk
                                                                                            v
                                                                                 W mma1 [tw] {1} [c2,s1]
                                                                                         tw | walk
                                                                                            v
                                                                     release BRANCH_FULL, tw [tc5mma] {1} c1 [c2,s1]
                                  +-------------------- BRANCH_FULL < ----------------------+
                                  v                                                    walk |
                tb = acquire BRANCH_FULL {0} [c3,s1]                                        |
                               tb |                                                         |
                                  v                                                         |
                      R branch [tb] {0} [c3,s1]                                             |
                               tb | walk                                                    |
                                  v                                                         |
            release CONVERGE, tb [none] {0} b2,c2 [c3,s1]                                   |
                         CONVERGE |                                                         v
                                  |                                                EXIT if (then) {1}
                                  |                                                         | branch completes
                                  |                                                         v
                  tf = acquire CONVERGE {0} [c4,s1]                                         |
                               tf |                                                         |
                                  v                                                         |
                      R final [tf] {0} [c4,s1]                                              |
                               tf | walk                                                    |
                                  v                                                         |
            release LOCAL_EMPTY, tf [none] {0} c3 [c4,s1]                                   |
                      LOCAL_EMPTY |                                                         v
                                  |                                                EXIT inner(i,j) {1}
                                  |                                                         | next inner iteration
                                  |                                                         v
                                  |                                              ENTER inner(i,j+1) {1}
                                  |                                                         | walk
                                  |                                                         v
                                  +----------- LOCAL_EMPTY > ------------next = acquire LOCAL_EMPTY {1} [c5,s0]
                                                                                       next |
                                                                                            v
                                                                            W mma0(i,j+1) [next] {1} [c5,s0]
```

This complete else-path view shows inner-loop finish. Its one `CONVERGE`
release waits for `mma0` at `[c5,s0]`; after `EXIT if`, `tf` and the final
read retain `[c4,s1]`. The `c3` rail ends at `done`, after which `p2` supplies
the outer read and owner `{0}` carries `out`:

```text
                initial = acquire OUTER_EMPTY at root
                          initial |
                                  v
                    scf.for iter_arg out=initial
                                  | executes
                                  v
                         ENTER outer(i) {0}
                              out |
                                  v
                        W outer(i) [out] {0}
                              out | walk
                                  v
                   release LOCAL_EMPTY, out {0} p1
                                  +-------------------- enter inner > ----------------------+
                      LOCAL_EMPTY |                                                         v
                                  |                                               ENTER inner(i,j) {1}
                                  |                                                         | walk
                                  |                                                         v
                                  +----------- LOCAL_EMPTY > -------------tw = acquire LOCAL_EMPTY {1} [c5,s0]
                                                                                         tw |
                                                                                            v
                                                                              W mma0(i,j) [tw] {1} [c5,s0]
                                                                                         tw | walk
                                                                                            v
                                                                                       scf.if cond
                                                                                            | else
                                                                                            v
                                                                                   ENTER if (else) {1}
                                                                                         tw | walk
                                                                                            v
                                                                    release CONVERGE, tw [tc5mma] {1} c2 else [c5,s0]
                                  +---------------------- CONVERGE < -----------------------+
                         CONVERGE |                                                         v
                                  |                                                EXIT if (else) {1}
                                  |                                                         | branch completes
                                  |                                                         v
                  tf = acquire CONVERGE {0} [c4,s1]                                         |
                               tf |                                                         |
                                  v                                                         |
                      R final [tf] {0} [c4,s1]                                              |
                               tf | walk                                                    |
                                  v                                                         |
            release LOCAL_EMPTY, tf [none] {0} c3 [c4,s1]                                   |
                      LOCAL_EMPTY |                                                         v
                                  |                                              EXIT inner(i,last) {1}
                                  |                                                         | loop finishes
                                  |                                                         v
                                  +---------- LOCAL_EMPTY > -----------done = acquire LOCAL_EMPTY {1} [unstamped]
                                                                                       done |
                                                                                            v
                                                                     release OUTER_EMPTY, done [none] {1} p2 [c2,s1]
                                                                                            |
                    out = acquire OUTER_EMPTY {0} ------------- < OUTER_EMPTY --------------+
                              out |
                                  v
                         R post(i) [out] {0}
                              out | walk
                                  v
                      EXIT outer(i) yields out
                                  +-------------------------------------- loop finishes > ----------------------------------------+
                                  | next outer iteration                                                                          v
                                  v                                                                                          result=out
                    ENTER outer(i+1) receives out
                              out |
                                  v
                       W outer(i+1) [out] {0}
                              out | walk
                                  v
                   release LOCAL_EMPTY, out {0} p1
```

The continuation and finish tails do not depend on which `if` path ran: the
then path may finish and the else path may continue with the same respective
tails. They are shown once each to avoid implying that the two branch
releases execute together. The `if` itself takes and returns no semaphore
token. If the inner loop is zero-trip, `p1` supplies the same unstamped
`done` acquire before the finish view's `OUTER_EMPTY` release. If the outer
loop is zero-trip, it returns `initial`; after the final executed outer
iteration, it returns `out`.

## Assigning semaphores and counts

The pass first decides where every acquire and release goes. It then gives
matching acquires and releases the same semaphore. An acquire at the start of
a loop, an acquire in the next iteration, and an acquire after the loop may
use the same semaphore when they wait for different endings of the same work.

For each semaphore, the pass:

1. counts how many releases each acquire must wait for;
2. makes that count the same at every acquire of the semaphore;
3. assigns the semaphore to those acquires and their releases; and
4. records which semaphore carries a token through each `for` or `if`.

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

Every acquire has the semaphore's `pending_count`. A semaphore with
`pending_count=2` therefore starts with both arrivals ready; it does not start
half-released.

With one physical semaphore copy, this state lets the first acquire run. With
several copies, it lets the first acquire of each copy run. A later acquire of
the same copy waits for a real release. The token belongs to the owner that
performs the acquire.

## Buffer and semaphore copies

InsertSemas records a buffer-copy count and a semaphore-copy count. They can
differ while this pass builds and schedules the DAG:

```text
buffer copies       physical SMEM or TMEM copies
semaphore copies    copies of each semaphore
```

They are usually equal. For a local buffer filled by a TMA load, InsertSemas
can record one buffer copy while checking the schedule with several semaphore
copies. LowerSemaphore later creates the staged SMEM buffer and semaphore
storage. InsertSemas's `num-stages` option must match the stage count that
LowerSemaphore will actually use.

### Buffer copies

The pass chooses the number of buffer copies after it removes unnecessary
synchronization edges. All names and views for the same buffer must either
omit `buffer.copy`, or all specify the same positive value.

```text
start with one buffer copy

if no synchronization edges remain
  leave the buffer unchanged; this pass creates no backing

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
the token created by the acquire before the loop, so no synchronization edge
is needed from root code to partition `{1}`. The exact edges are:

```text
parent DAG node                 synchronization edge ending here
W acc root                     none
[loop summary P0:W:{1}]        none; it uses the root token
R acc root                     p1: loop summary -> R acc root

child DAG node                 synchronization edge ending here
ENTER(i) {1}                   none
R acc {1}                      none
W acc {1}                      none
W MMA {2}                      e1: W acc {1} -> W MMA {2}
EXIT(i) {1}                    e2: W MMA {2} -> EXIT(i) {1}
```

The parent and child are separate DAGs. The parent has no synchronization
edge into the loop; only `p1` leaves the loop summary. The exact root-token
adoption is shown later in the semaphore/token view:

```text
parent synchronization-edge DAG

                         W acc root
                              | walk
                              v
                    [loop summary P0:W:{1}]
                              | p1
                              v
                         R acc root
```

```text
child synchronization-edge DAG

                                                      ENTER(i) {1}
                                                            | walk
                                                            v
                                                      R acc(i) {1}
                                                            | walk
                                                            v
                                                      W acc(i) {1}
                                                            +-----------------e1 >------------------+
                                                            | walk                                  v
                                                            |                                 W MMA(i) {2}
                                                            +-----------------< e2------------------+
                                                            v
                                                       EXIT(i) {1}
```

The semaphores are:

```text
edge / role    semaphore    release owner    pending_count    initial state
entry          EMPTY        -                1                released at root
e2             EMPTY        {2}              1                same semaphore
e1             TO_MMA       {1}              1                blocked
p1             AFTER        {1}              1                blocked
```

Auto uses FirstTouch for this loop because the token created before the loop
must remain available through the loop. The entry view shows token adoption,
the two semaphore handoffs, and the owner-`{1}` acquire carried through
`EXIT`:

```text
          root = acquire EMPTY
               root |
                    v
              W acc [root] root
                    +-------------root token >--------------+
                                                            v
                                                      ENTER(0) {1}
                                                       root |
                                                            v
                                                   R acc(0) [root] {1}
                                                            | walk
                                                            v
                                                   W acc(0) [root] {1}
                                                            | walk
                                                            v
                                               release TO_MMA, root {1} e1
                                                            +---------------TO_MMA >----------------+
                                                            | walk                                  v
                                                            |                           mma = acquire TO_MMA {2}
                                                            |                                   mma |
                                                            |                                       v
                                                            |                              W MMA(0) [mma] {2}
                                                            |                                       | walk
                                                            |                                       v
                                                            |                      release EMPTY, mma [tc5mma] {2} e2
                                                            +----------------< EMPTY----------------+
                                                            v
                                                next = acquire EMPTY {1}
                                                       next |
                                                            v
                                                   EXIT(0) yields next
```

If the loop continues, the same owner-`{1}` token crosses the boundary and is
used by the next iteration's first read and write:

```text
                                                          next
                                                       next |
                                                            v
                                                   EXIT(i) yields next
                                                            | next iteration
                                                            v
                                                ENTER(i+1) receives next
                                                       next |
                                                            v
                                                 R acc(i+1) [next] {1}
                                                            | walk
                                                            v
                                                 W acc(i+1) [next] {1}
                                                            | walk
                                                            v
                                               release TO_MMA, next {1} e1
```

If the loop finishes, `next` becomes `result`. The release to `AFTER`
implements parent edge `p1`:

```text
                                                          next
                                                       next |
                                                            v
                                                EXIT(last) yields result
                                                     result |
                                                            v
                                              release AFTER, result {1} p1
                    +----------------< AFTER----------------+
                    v
        out = acquire AFTER root
                out |
                    v
              R acc [out] root
```

For a zero-trip loop, the continuation view is skipped: `result` is the
original root token and enters the same owner-`{1}` `AFTER` release. This
buffer has synchronization edges, no explicit `buffer.copy`, an MMA shape
that permits two copies, and enough TMEM capacity. The result is:

```text
input accumulator     memdesc<128x128xf32>
generated buffer      memdesc<2x128x128xf32>
physical copies       2
semaphore copies      2
```

### Semaphore copies

After placing the acquires and releases, the pass chooses how many semaphore
copies to assume while it checks schedules and offsets:

```text
SMEM buffer
and no input buffer.copy
and at least one release waits for [tma_load]
  semaphore copies = max(1, num-stages)

otherwise
  semaphore copies = buffer copies
```

For example, `test/NVWS/insert_semas.mlir` `@local_release_after_mma` has one
buffer copy in the InsertSemas DAG. The buffer is filled by
`nvws.descriptor_load`, so the `FULL` release waits for `[tma_load]`.
InsertSemas checks its schedule using `max(1, num-stages)`, where `num-stages`
is the option on `--nvws-insert-semas`.

```text
InsertSemas DAG
  buffer copies = 1

schedule model
  semaphore copies = max(1, num-stages)
```

LowerSemaphore uses the owning WS loop's `tt.num_stages` when that attribute
is present. Otherwise it uses the option on `--nvws-lower-semaphore`. If one
physical semaphore group is shared by several TMA-producing WS loops, it uses
the largest of their stage counts. The option on `--nvws-insert-semas` must
match that effective count. LowerSemaphore then creates the staged semaphore
storage and replaces the eligible one-copy SMEM allocation with a matching
staged allocation.

### Example: a TMA load uses the lowering stage count

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
                W descriptor_load(i) {0}
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                 R MMA operand(i) {1}
                            +-------------------< e2--------------------+
                            v
                       EXIT(i) {0}
```

`e1` uses `FULL`. `e2` uses `EMPTY`, which starts released for iteration zero.
Both semaphores have `pending_count=1`:

```text
edge    semaphore    release owner    pending_count    initial state
e1      FULL         {0}              1                blocked
e2      EMPTY        {1}              1                released for owner {0}
```

The two fixed lanes keep each owner's operations vertical. `[cN,sM]` means
cluster `N`, stage `M`. The `EMPTY` rail remains separate from owner `{0}`'s
control spine while control crosses `EXIT` and `ENTER`:

```text
                      ENTER(i) {0}
                            | walk
                            v
          empty = acquire EMPTY(i) {0} [c1,s0]
                      empty |
                            v
       W descriptor_load(i) [empty] {0} [c1,s0]
                            | walk
                            v
      release FULL, empty [tma_load] {0} e1 [c1,s0]
                            +------------------FULL >-------------------+
                            | walk                                      v
                            |                            full = acquire FULL {1} [c0,s1]
                            |                                      full |
                            |                                           v
                            |                          R MMA operand(i) [full] {1} [c0,s1]
                            |                                           | walk
                            |                                           v
                            |                      release EMPTY, full [tc5mma] {1} e2 [c0,s1]
                            |                                     EMPTY |
                            v                                           |
                       EXIT(i) {0}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {0}                                     |
                            | walk                                      |
                            v                                           |
          next = acquire EMPTY(i+1) {0} [c1,s0] --------< EMPTY --------+
                       next |
                            v
     W descriptor_load(i+1) [next] {0} [c1,s0]
```

Both waits are required. `[tma_load]` keeps the `FULL` release waiting until
the descriptor load has filled SMEM. `[tc5mma]` keeps the `EMPTY` release
waiting until the MMA has finished reading that buffer copy.

SMEM does not receive automatic TMEM double buffering, and the input has no
`buffer.copy`. Therefore the InsertSemas DAG records:

```text
buffer copies              1
semaphore copies used
for schedule checks        max(1, num-stages)
```

LowerSemaphore later creates the matching staged SMEM and semaphore storage.
There is no buffer use after the loop, so no acquire waits for the final
`EMPTY` release. A zero-trip loop executes none of the shown operations and
leaves every initially released semaphore copy ready for its first acquire.

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

The pass keeps every `loop.stage` fixed. It changes `loop.cluster` when a
release must execute before its acquire. It changes `stage-offset` when the
release, acquire, and buffer access must select a different copy.

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
as an error. The pass does not add another ordering rule when existing data
flow already orders the operations.

### Waits between iterations

A release can wake an acquire in the same iteration or in a later iteration.
When this is already known, the wait records that positive iteration distance.
Otherwise the pass follows the physical buffer and semaphore copies used by
successive iterations.

The pass replays the ordered buffer accesses and uses the semaphore-copy
count. With one semaphore copy, the distance is one iteration. With several,
it finds the first later iteration whose acquire returns to the release's
semaphore copy.

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

K and V are two logical groups during synchronization analysis. Their edge
DAGs are separate views, not parallel execution. The exact edges are:

```text
group  DAG node          synchronization edge ending here
K      ENTER(i) {1}      none
K      W K(i) {1}        none
K      R K(i) {2}        k1: W K(i) {1} -> R K(i) {2}
K      EXIT(i) {1}       k2: R K(i) {2} -> EXIT(i) {1}

V      ENTER(i) {1}      none
V      W V(i) {1}        none
V      R V(i) {2}        v1: W V(i) {1} -> R V(i) {2}
V      EXIT(i) {1}       v2: R V(i) {2} -> EXIT(i) {1}
```

```text
K synchronization-edge DAG

                      ENTER(i) {1}
                            | walk
                            v
                       W K(i) {1}
                            +-------------------k1 >--------------------+
                            | walk                                      v
                            |                                      R K(i) {2}
                            +-------------------< k2--------------------+
                            v
                       EXIT(i) {1}

V synchronization-edge DAG

                      ENTER(i) {1}
                            | walk
                            v
                       W V(i) {1}
                            +-------------------v1 >--------------------+
                            | walk                                      v
                            |                                      R V(i) {2}
                            +-------------------< v2--------------------+
                            v
                       EXIT(i) {1}
```

No synchronization edge is removed or merged within either logical group.
The loop still executes K before V in each owner. The pass folds the two
logical groups onto one physical `FULL` semaphore and one physical `EMPTY`
semaphore because they share one circular `buffer.id`:

```text
edge    semaphore    release owner    pending_count    initial state
k1      FULL         {1}              1                blocked
k2      EMPTY        {2}              1                released for owner {1}
v1      FULL         {1}              1                blocked
v2      EMPTY        {2}              1                released for owner {1}
```

The loop still executes the accesses in input order: K before V in each owner.
The synchronization views stay separate, as they do in the edge DAGs above.
Both views use the same physical `FULL`/`EMPTY` pair. Their two external
`EMPTY` rails select different stages of that pair; they are not different
semaphores.

```text
K semaphore view

                      ENTER(i) {1}
                            | walk
                            v
                 kt = acquire EMPTY {1}
                         kt |
                            v
                     W K(i) [kt] {1}
                            | walk
                            v
                 release FULL, kt {1} k1
                            +------------------ FULL > -----------------+
                       walk |                                           v
                            |                                kr = acquire FULL {2}
                            |                                         kr |
                            |                                            v
                            |                                    R K(i) [kr] {2}
                            |                                            | walk
                            |                                            v
                            |                          release EMPTY, kr {2} k2 -------- EMPTY (K stage) > ------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
                nextK = acquire EMPTY {1} -------------------- EMPTY (K stage) < --------------------------------------+
                      nextK |
                            v
                   W K(i+1) [nextK] {1}

V semaphore view

                      ENTER(i) {1}
                            | walk
                            v
                 vt = acquire EMPTY {1}
                         vt |
                            v
                     W V(i) [vt] {1}
                            | walk
                            v
                 release FULL, vt {1} v1
                            +------------------ FULL > -----------------+
                       walk |                                           v
                            |                                vr = acquire FULL {2}
                            |                                         vr |
                            |                                            v
                            |                                    R V(i) [vr] {2}
                            |                                            | walk
                            |                                            v
                            |                          release EMPTY, vr {2} v2 -------- EMPTY (V stage) > ------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
                nextV = acquire EMPTY {1} -------------------- EMPTY (V stage) < --------------------------------------+
                      nextV |
                            v
                   W V(i+1) [nextV] {1}
```

Each write therefore acquires `EMPTY` immediately before it, and the loop
does not carry a semaphore token.

The two buffers share one write number. Each write advances it:

```text
event       current write number    required write number    offset
W K         -1 -> 0                 K producer = 0           0
W V          0 -> 1                 V producer = 1           0
R K          1                      K producer = 0          -1
R V          1                      V producer = 1           0
```

Adding offsets to the same two-view scaffold gives:

```text
K semaphore view with offsets

                      ENTER(i) {1}
                            | walk
                            v
            kt = acquire EMPTY {1} [offset 0]
                         kt |
                            v
                 W K(i) [kt] {1} [copy 0]
                            | walk
                            v
           release FULL, kt {1} k1 [offset 0]
                            +-------------- FULL (copy 0) > ------------+
                       walk |                                           v
                            |                     kr = acquire FULL {2} [offset -1]
                            |                                         kr |
                            |                                            v
                            |                            R K(i) [kr] {2} [copy 0]
                            |                                            | walk
                            |                                            v
                            |        release EMPTY, kr {2} k2 [offset -1] -------- EMPTY (copy 0) > -------------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
     nextK = acquire EMPTY {1} [offset 0] -------------------- EMPTY (copy 0) < ---------------------------------------+
                      nextK |
                            v
            W K(i+1) [nextK] {1} [copy 0]

V semaphore view with offsets

                      ENTER(i) {1}
                            | walk
                            v
            vt = acquire EMPTY {1} [offset 0]
                         vt |
                            v
                 W V(i) [vt] {1} [copy 1]
                            | walk
                            v
           release FULL, vt {1} v1 [offset 0]
                            +-------------- FULL (copy 1) > ------------+
                       walk |                                           v
                            |                      vr = acquire FULL {2} [offset 0]
                            |                                         vr |
                            |                                            v
                            |                            R V(i) [vr] {2} [copy 1]
                            |                                            | walk
                            |                                            v
                            |         release EMPTY, vr {2} v2 [offset 0] -------- EMPTY (copy 1) > -------------------+
                            |                                                                                          |
                            v                                                                                          |
                       EXIT(i) {1}                                                                                     |
                            | next iteration                                                                           |
                            v                                                                                          |
                     ENTER(i+1) {1}                                                                                    |
                            | walk                                                                                     |
                            v                                                                                          |
     nextV = acquire EMPTY {1} [offset 0] -------------------- EMPTY (copy 1) < ---------------------------------------+
                      nextV |
                            v
            W V(i+1) [nextV] {1} [copy 1]
```

The offsets select these physical copies relative to the latest shared write:

```text
operation                 stage offset       physical copy
W K / release FULL        0                  copy 0
W V / release FULL        0                  copy 1
acquire FULL / R K       -1                  copy 0
acquire FULL / R V        0                  copy 1
R K / release EMPTY      -1                  copy 0
R V / release EMPTY       0                  copy 1
```

The `-1` means the previous copy after V advanced the shared write number. It
is not a negative physical index. With two copies, wrapping `-1` selects copy
0. The generated DAG gives K's acquire and closing release `stage-offset=-1`
and V's offset zero. The test also checks that the IR contains the one shared
pair of physical semaphores.

For iteration zero, the initially released state of each physical `EMPTY`
copy supplies `kt` and `vt`. On re-entry, `k2` supplies the next K acquire and
`v2` supplies the next V acquire. After the final iteration, their releases
have no later acquires. A zero-trip loop executes none of these operations and
leaves both physical `EMPTY` copies initially released.

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

Both names refer to the same bytes, so the synchronization-edge DAG is one
ordered path:

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
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                      R m0(i) {2}
                            +-------------------< e2--------------------+
                            v
                       W m1(i) {4}
                            +-------------------e3 >--------------------+
                            | walk                                      v
                            |                                      R m1(i) {2}
                            +-------------------< e4--------------------+
                            v
                       EXIT(i) {4}
```

No edge is removed or merged.

Every semaphore has `pending_count=1`; `ENTRY` starts released:

```text
edge    semaphore    release owner    pending_count    initial state
e1      M0_FULL      {4}              1                blocked
e2      M1_READY     {2}              1                blocked
e3      M1_FULL      {4}              1                blocked
e4      ENTRY        {2}              1                released for owner {4}
```

The POU semaphore DAG for iteration `i` is:

```text
                      ENTER(i) {4}
                            | walk
                            v
                t0 = acquire ENTRY(i) {4}
                         t0 |
                            v
                    W m0(i) [t0] {4}
                            | walk
                            v
               release M0_FULL, t0 {4} e1
                            +-----------------M0_FULL >-----------------+
                            | walk                                      v
                            |                               t1 = acquire M0_FULL {2}
                            |                                        t1 |
                            |                                           v
                            |                                   R m0(i) [t1] {2}
                            |                                           | walk
                            |                                           v
                            |                              release M1_READY, t1 {2} e2
                            +----------------< M1_READY-----------------+
                            v
                t2 = acquire M1_READY {4}
                         t2 |
                            v
                    W m1(i) [t2] {4}
                            | walk
                            v
               release M1_FULL, t2 {4} e3
                            +-----------------M1_FULL >-----------------+
                            | walk                                      v
                            |                               t3 = acquire M1_FULL {2}
                            |                                        t3 |
                            |                                           v
                            |                                   R m1(i) [t3] {2}
                            |                                           | walk
                            |                                           v
                            |                               release ENTRY, t3 {2} e4
                            |                                     ENTRY |
                            v                                           |
                       EXIT(i) {4}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {4}                                     |
                            | walk                                      |
                            v                                           |
              next = acquire ENTRY(i+1) {4} ----------< ENTRY ----------+
                       next |
                            v
                  W m0(i+1) [next] {4}
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
`ENTRY` releases. The full offset overlay uses the same two-lane scaffold:

```text
                      ENTER(i) {4}
                            | walk
                            v
            t0 = acquire ENTRY {4} [offset 0]
                         t0 |
                            v
                    W m0(i) [t0] {4} [copy s]
                            | walk
                            v
          release M0_FULL, t0 {4} e1 [offset 0]
                            +-----------------M0_FULL >-----------------+
                            | walk                                      v
                            |                          t1 = acquire M0_FULL {2} [offset 0]
                            |                                        t1 |
                            |                                           v
                            |                                   R m0(i) [t1] {2} [copy s]
                            |                                           | walk
                            |                                           v
                            |                        release M1_READY, t1 {2} e2 [offset +1]
                            +----------------< M1_READY-----------------+
                            v
          t2 = acquire M1_READY {4} [offset 0]
                         t2 |
                            v
                    W m1(i) [t2] {4} [copy s+1]
                            | walk
                            v
          release M1_FULL, t2 {4} e3 [offset 0]
                            +-----------------M1_FULL >-----------------+
                            | walk                                      v
                            |                          t3 = acquire M1_FULL {2} [offset 0]
                            |                                        t3 |
                            |                                           v
                            |                                   R m1(i) [t3] {2} [copy s+1]
                            |                                           | walk
                            |                                           v
                            |                         release ENTRY, t3 {2} e4 [offset +1]
                            |                                     ENTRY |
                            v                                           |
                       EXIT(i) {4}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {4}                                     |
                            | walk                                      |
                            v                                           |
           next = acquire ENTRY {4} [offset 0] --------< ENTRY ---------+
                       next |
                            v
                  W m0(i+1) [next] {4} [copy s]
```

Without the two `+1` release offsets, the following acquire would wait on a
different physical semaphore copy. For iteration zero, `ENTRY` is initially
released and supplies `t0`. On re-entry, `e4` supplies `next`. After the final
iteration, no later acquire consumes the final `ENTRY` release. A zero-trip
loop executes none of these operations and leaves `ENTRY` initially released.

### Example: one buffer copy

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` has one SMEM copy and this scheduled loop:

```text
buffer.copy = 1

for {
  W buffer {3}      loop.stage 0
  R first {1}       loop.stage 0
  R last {1}        loop.stage 1
}
```

The two reads have the same owner. The final read replaces that owner's latest
access without adding another synchronization edge. The exact edges are:

```text
DAG node                  synchronization edge ending here
ENTER(i) {3}              none
W buffer(i) {3}           none
R first(i) {1}            e1: W buffer(i) {3} -> R first(i) {1}
R last(i) {1}             none; same-owner program order
EXIT(i) {3}               e2: R last(i) {1} -> EXIT(i) {3}
```

```text
                      ENTER(i) {3}
                            | walk
                            v
                     W buffer(i) {3}
                            +-------------------e1 >--------------------+
                            | walk                                      v
                            |                                    R first(i) {1}
                            |                                           | walk
                            |                                           v
                            |                                     R last(i) {1}
                            +-------------------< e2--------------------+
                            v
                       EXIT(i) {3}
```

No edge is removed or merged.

The schedule can overlap the final read of iteration `i` with work from
iteration `i+1`. The `EMPTY` acquire is immediately before the next store,
and the matching `EMPTY` release is after the final read. Although edge `e2`
ends at `EXIT(i)`, POU places its acquire at `W buffer(i+1)`. With one
semaphore copy, that wait has distance one iteration.

The semaphore assignment is:

```text
edge / role    semaphore    release owner    pending_count    initial state
e1             FULL         {3}              1                blocked
entry          EMPTY        -                1                released for owner {3}
e2             EMPTY        {1}              1                same semaphore
```

The generated semaphore DAG, including the wait into iteration `i+1`, is
below. `[cN,sM]` means cluster `N`, stage `M`:

```text
                      ENTER(i) {3}
                            | walk
                            v
          empty = acquire EMPTY(i) {3} [c3,s0]
                      empty |
                            v
                W buffer(i) [empty] {3} [c3,s0]
                            | walk
                            v
        release FULL, empty [none] {3} e1 [c3,s0]
                            +------------------FULL >-------------------+
                            | walk                                      v
                            |                            full = acquire FULL {1} [c3,s0]
                            |                                      full |
                            |                                           v
                            |                                R first(i) [full] {1} [c3,s0]
                            |                                           | walk
                            |                                           v
                            |                                 R last(i) [full] {1} [c2,s1]
                            |                                           | walk
                            |                                           v
                            |                       release EMPTY, full [none] {1} e2 [c2,s1]
                            |                                     EMPTY |
                            v                                           |
                       EXIT(i) {3}                                      |
                            | next iteration                            |
                            v                                           |
                     ENTER(i+1) {3}                                     |
                            | walk                                      |
                            v                                           |
          next = acquire EMPTY(i+1) {3} [c3,s0] --------< EMPTY --------+
                       next |
                            v
              W buffer(i+1) [next] {3} [c3,s0]
```

The loop does not carry a semaphore token. Each iteration acquires `EMPTY`
immediately before the store. `EMPTY` starts released for iteration zero; the
release from partition `{1}` lets the next iteration acquire it. After the
final iteration, no later acquire consumes its `EMPTY` release. A zero-trip
loop executes none of these operations and leaves `EMPTY` initially released.

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
- every acquire has the semaphore's positive `pending_count` and either is
  supplied by at least one release or uses a semaphore that starts released;
- an acquire repeated inside a loop has a release that lets the next
  iteration continue;
- one semaphore is not acquired by two different owners;
- every wait between iterations has a positive distance;
- every path through a `for` or `if` returns the token needed after that
  path; and
- every synchronization edge became part of an acquire/release pair.

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
| Initial synchronization edges | `ChainWalker`, `applyTouch` in `InsertSemasSyncDag.cpp` |
| Remove unnecessary synchronization edges | `reduceStraightEdges`, `reduceLoopCloses`, `reduceEdges` |
| Place acquires and releases | `DirectBuilder` |
| Assign semaphores and counts | `DirectBuilder::formSemaphores` |
| Checks before changing IR | `validatePOUPlan`, `validateTokenConnectivity`, `verifySyncDag` |
| Copies and schedule | `computeBackingCopies`, `computeSemaphoreCopies`, `finalizeSyncSchedule` |
| DAG dump and IR emission | `InsertSemasEmitIR.cpp` |

The design can be summarized in one sentence:

> Find the waits, remove waits already guaranteed by other paths, place the
> acquires and releases, assign semaphores, check the result, then emit IR.
