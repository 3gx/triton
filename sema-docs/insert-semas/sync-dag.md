# SYNC-DAG

## Contents

- [Purpose](#purpose)
- [Notation](#notation)
- [The walk: accesses to edges](#the-walk-accesses-to-edges)
  - [Memory edges and token supply](#memory-edges-and-token-supply)
  - [Example: two-partition handoff](#example-two-partition-handoff)
  - [Example: edge reduction lowers the pending count](#example-edge-reduction-lowers-the-pending-count)
  - [Example: fan-out and owner-token reuse](#example-fan-out-and-owner-token-reuse)
  - [Example: disjoint submember protocols run independently](#example-disjoint-submember-protocols-run-independently)
  - [The per-access rules, in full](#the-per-access-rules-in-full)
  - [Composition: nested regions in the walk](#composition-nested-regions-in-the-walk)
    - [Worked example: apply the same rules at both levels](#worked-example-apply-the-same-rules-at-both-levels)
- [Edges to semaphores](#edges-to-semaphores)
  - [Example: a redundant edge is dropped](#example-a-redundant-edge-is-dropped)
  - [Example: a loop-closing edge is dropped](#example-a-loop-closing-edge-is-dropped)
  - [The deletion conditions, in full](#the-deletion-conditions-in-full)
    - [Implied ordering (`reduceEdges`)](#1-implied-ordering-reduceedges)
    - [Repeats from one sender (`buildEdgesAndSemas`)](#2-repeats-from-one-sender-buildedgesandsemas)
    - [Covered senders (`buildEdgesAndSemas`)](#3-covered-senders-buildedgesandsemas)
  - [One destination node, one semaphore](#one-destination-node-one-semaphore)
  - [Composition: why loop entry and loop recurrence share one semaphore](#composition-why-loop-entry-and-loop-recurrence-share-one-semaphore)
- [Region flows](#region-flows)
  - [Why this is needed](#why-this-is-needed)
  - [Region summaries](#region-summaries)
  - [The loop decision](#the-loop-decision)
  - [Point of use](#point-of-use)
  - [Trailing use](#trailing-use)
  - [Use after the loop](#use-after-the-loop)
  - [Nested loops](#nested-loops)
  - [Dump labels](#dump-labels)
- [Backing copies](#backing-copies)
- [Pipeline schedule](#pipeline-schedule)
  - [Minimal pipeliner model](#minimal-pipeliner-model)
  - [Example: one-copy loop-closing handoff](#example-one-copy-loop-closing-handoff)
  - [Finalizing one handoff](#finalizing-one-handoff)
- [Authored buffer-stage offsets](#authored-buffer-stage-offsets)
  - [Circular groups](#circular-groups)
  - [Non-circular alias handoffs](#non-circular-alias-handoffs)
- [Code map](#code-map)

## Purpose

SYNC-DAG converts OWNER-DAG ownership changes into a balanced semaphore
protocol: who waits for whom, through which semaphore, with how many backing
copies, and where in the pipeline schedule.

The design has five moving parts:

1. a **node** is one event in a chain;
2. each piece has a **source** plus a **uses** map from owner to node;
3. each chain has an ordered list of known **tokens**;
4. an **edge** records one required wait or token handoff between nodes;
5. a **region flow** summarizes the token returned by a `for` or `if` and
   records the loop token decision.

`Owner`, `Piece`, and `Effect` are attributes of those objects. Acquires,
releases, and semaphores are the generated representation of edges, not a
second correctness model.

```text
input IR ─► ACCESS-DAG ─► OWNER-DAG ─► SYNC-DAG ─► EMIT-IR ─► output IR
            memory facts   owners      edges, semaphores,   render
                                       region flows, schedule
```

The whole step is four moves; this page is those moves in order, each with
a worked example:

```text
1. walk the accesses in program order; every "this must wait for that"
   becomes an edge between two concrete DAG nodes
2. delete the redundant edges
3. the edges converging on one destination become one acquire; each incoming
   edge's tail becomes a release. Each completion kind carried by an edge
   counts as one release; an edge with no completion kind also counts as one.
   The acquire's pending count is the total number of those releases
4. summarize `for`/`if` boundaries, select carried, point-of-use, or
   child-owned loop handling, choose the number of backing copies, then
   extend the pipeline schedule with the semaphore dependencies
```

## Notation

The `|-` listings are trimmed excerpts of actual pass dumps. State tables and
DAG sketches reconstruct the same nodes and generated edges to explain the
walk. Most examples come from the listed lit tests; the section explicitly
labeled synthetic came from a temporary input run through the same pass.
Unrelated groups, `parts{...}` fields, and some `ENTER` lines are elided;
`; ...` and `<-` annotations are added. The command is:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt <input.mlir> -split-input-file \
    -allow-unregistered-dialect --nvws-insert-semas
```

Object shorthand (full definitions in the
[InsertSemas overview](overview.md#core-objects)):

```text
group         the allocations analyzed together (ordinarily one buffer.id)
backing       the SMEM/TMEM allocation the group guards (mutable SMEM;
              any TMEM)
m0, m1        member: one allocation of the group; m0[0,64) = its interval
P0, P1        piece: a disjoint interval of the backing; {m0, m1} = members
              covering it (an access through a member touches ALL of that
              member's pieces)
{0}, {1}      owner: partition 0, 1 of the enclosing WS loop; root = no
              partition
{@0.1}        tag-qualified owner: partition 1 under WS tag 0
producer      the owner that last wrote a piece (before any write: its
              first toucher)
version source
              the concrete DAG node and owner from which a new reader receives
              that version: the latest write in the same chain, the first
              toucher before any write, or the child chain's ENTER node when
              the version was established outside
use           one `owner -> node` entry for the current value; `node` is
              that owner's most recent access. A later access by that owner
              replaces `node`; a write resets the whole map
tokens        the chain's known owner tokens in deterministic order; the last
              source-bearing token supplies a handoff when no memory edge or
              owner-specific token reuse supplies one
token reuse   a node-level proof (`reuseTokenOwner`) that an access or release
              may use its owner's earlier token without changing token order
hold          acquire -> protected operations -> closing release for one
              owner; an explicitly marked owner token may be reused later
regain        the acquire near the end of a loop body that supplies the
              next iteration
region flow   the token returned by each `for`/`if` path and, for a loop,
              whether it carries the token, acquires at the first use, or
              leaves token handling to its final child
chain         the group's node sequence of one block, in program order; a
              region node holds child chains
```

Dump notation:

```text
|- scf.for (WS, tag=0)      loop node; (WS, tag=N) = warp-specialized loop
|- W m0  ttg.local_store {1}   access node: W(rite)/R(ead), member, op, owner
|- a  S1  {0}               acquire of semaphore S1 by partition 0
|- a  S0(2)  {0}            acquire with pending count 2; waits for 2 releases
|- r  S0  {1} [tma_load]    release when the TMA load completes
|- r  S0(2)  {1}            release that counts twice toward pending count
|- a  S3  root  ; entry     unpartitioned entry acquire, spliced before the
                            group's first placement node — a top-level node of
                            the group's chain that involves the group: an
                            access, or a region containing one (immediately
                            above the loop when the loop is that first node)
pieces{P0:W:{0}}            region node: per-piece merged effect and owner
thread{{0}}                 region node: a RegionFlow summary exists for owner
                            {0}; only CARRIED loops necessarily add a token
                            iter-arg and result
holdrule{...}               loop node: selected mode and any placement
                            details, explained in "Region flows"
yield{X}                    EXIT node: what this chain returns for its
                            region flow — a S<n> = that acquire's token;
                            native = no token crosses (protocol lives
                            inside); drop = this loop has no token result
                            because its final nested loop returns none; pass =
                            this branch has no acquire or nested region flow,
                            so it returns the token available before
                            the `if`;
                            scf.for/scf.if =
                            the actual yield operand is the token result of
                            that nested scf.for/scf.if
stage-offset=N              stage offset assigned to this protocol node
S<n> / E<n>                 semaphore names; E<n> = dedicated entry semaphore
SEMAS: S1{count=1 entry inherit={@0.0}}   per-semaphore summary; entry =
                            created initially released; inherit={...} = the
                            owner recorded on that initially released state
                            (its `entryTokenOwner`) — read while finding the
                            loop's input acquire and by token verification;
                            EMIT-IR also uses it as
                            the owner of an unpartitioned entry acquire when
                            recording the acquired token — which is one
                            of three: the owner of the group's
                            first real access (`insertEntryAcquires`, both of
                            its paths), the retargeted EXIT-handoff acquire's
                            owner (`buildEdgesAndSemas`), or the owner selected
                            by point-of-use lowering
BACKING: numCopies=N        copies chosen for the backing allocation
```

Pseudo-IR strips types and attributes: `%t = a S0 {1}` / `r S0 %t {1}` are
the token-producing acquire and token-using release, `W m0 [%t]` is an
access made using token `%t`, and `for iter_args(%t = %t0)` —
abbreviated `for (%t = %t0)` in the figures below — is a loop carrying the
token.

Do not confuse the two layers: `P0` and edges are analysis labels that never
appear in the IR. The semaphore *token* is an IR value — each emitted
acquire returns a fresh one, and a release takes that token as an operand.
One token can feed several releases. A later buffer or release can also use
that token when SYNC-DAG explicitly marks the node for owner-specific reuse;
EMIT-IR never infers that exception on its own.

## The walk: accesses to edges

The walk runs once per group, in program order over its chains
(`ChainWalker::run`). At each access it first
applies two memory rules:

1. **Read after write (RAW)** — a new reader waits for the piece's version
   source.
2. **Write after uses (WAR)** — a write replaces the data, so it waits for the
   node stored for every other owner, unless that node is already ordered
   before the write.

Every wait becomes an edge between two concrete DAG nodes (`EdgeRec` stores
the two node pointers). The complete walk state is:

```text
source   per piece   logical producer plus the chain-local node that supplies
                     the current version to a new reader
uses     per piece   map `[owner -> node, ...]`; `node` is that owner's most
                     recent access to the current value
tokens   per chain   known owner tokens in deterministic order, with the node
                     and completion payload when an access has established a
                     handoff source
```

Source and uses decide memory edges. Tokens answer the separate supply
question only after those rules: if no memory edge supplies the access and its
owner cannot reuse an earlier token, the walk adds an edge from the last
token's node. That deterministic list order preserves which legal handoff is
emitted; it is not another memory-dependency rule. A uniform `ENTER` token is
reusable immediately, but cannot supply a token-supply edge until an access has
recorded its source node.

```text
R  m0  ttg.local_load  {1}
└──────────┬────────┘  └┬┘
     the ACCESS           the OWNER
  (one event: this op     (the partition executing
   touching the memory)    this access: partition 1)
```

For each piece, `PieceState` stores a stable `VersionSource` and a `uses` map.
The implementation represents each map entry with `ActiveUse`. A reread
replaces only that owner's node in `uses`; it does not move the source.
Independent readers therefore fan out from the write, first toucher, or ENTER
rather than forming a reader-to-reader chain. A write waits on the node stored
for every other owner unless that node is already ordered before the write,
then resets both source and uses to itself.

### Memory edges and token supply

RAW/WAR rules answer whether an access needs a memory edge. Independently, an
access rewritten under a group's semaphore protocol must use a token-backed
buffer view. After applying the memory rules, there are two cases:

1. The access's owner has a token valid for the access, so the access uses it.
2. The access's owner has no valid token, so the pass creates an acquire that
   produces one.

An owner may reuse an earlier token only if every touched piece passes
`canReuseTokenForPiece`: a read requires an entry for that owner in every
piece's `uses` map; a write requires the node stored for every other owner on
every piece already to be ordered before the write.

Together, these rules produce one fan-out/fan-in access-order sketch:

```text
                 ┌── R {0} ──┐
W {0} ───────────├── R {1} ──┼────────── W {3}
                 └── R {2} ──┘
```

This is not an edge-only DAG: the `R {0}` branch shows an independent access
by the source owner, with no generated edge into it. The new-reader branches
are the generated RAW edges.

The new readers `{1}` and `{2}` receive RAW edges from `W {0}`. The returning
`R {0}` needs no memory edge because `uses` already contains `{0}`; it reuses
`{0}`'s earlier token and is not ordered behind either new reader. The later
`W {3}` receives WAR edges from the nodes stored for all three owners.

Implementation correspondence: `Tokens` stores the known owner tokens and
their deterministic order. `Node::reuseTokenOwner` marks case 1 after the
per-piece checks pass. This is token-supply bookkeeping, not an additional
memory-dependency rule.

At a region's `EXIT`, each piece returns to its `EXIT` owner. If the piece
will be used again — during the next loop iteration or after the region —
the `EXIT` owner waits for the node stored for every other owner in `uses`.
No new edge is added if an earlier edge already makes the `EXIT` owner wait
for that node. EXIT then keeps only the boundary owner's entry in `uses`;
it does not move the version source.

The worked examples show the common shapes; the complete checklist follows
them.

Each DAG sketch in the access-walk examples below unrolls one loop boundary.
An `Sx` label marks a release/acquire handoff through semaphore `Sx`;
unlabeled lines show the surrounding program and loop order.

In state tables, `tokens=[...]` lists known owner tokens from oldest to newest
(last at right). `:no-source` means the incoming owner token is reusable but
cannot yet supply a token-supply edge. `uses=[{0}->W, {1}->R]` displays the
complete uses map, not merely its owner keys.

### Example: two-partition handoff

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` — partition
0 stores, partition 1 loads, every iteration. Post-state is shown for each
access; the `EXIT` line shows the recurrence edge:

```text
walk            edge                                    state AFTER the node
ENTER {0}       —                                        source=ENTER@{0} uses=[{0}->ENTER@{0}] tokens=[{0}:no-source]
W store {0}     — (same-owner write)                    source=store@{0} uses=[{0}->store@{0}] tokens=[{0}]
R load  {1}     e1: store@{0} -> load@{1}    (RAW)      source=store@{0} uses=[{0}->store@{0}, {1}->load@{1}] tokens=[{0},{1}]
EXIT            e2: load@{1} -> store@{0}@next (WAR)
```

Two edges: the load waits for the store (`e1`), and the next iteration's
store waits for this load (`e2`, raised at `EXIT`). What the
default conversion of those two edges would emit — carried, full protocol,
the untransformed shape every loop starts from (pseudo-IR):

```text
%t0 = a S1 root                  ; entry, seeds iteration 0; S1 is created
for (%t = %t0) {                 ;   initially released
  W m0 [%t]  ttg.local_store {0}
  r  S0 %t  {0}                  ; e1: data ready
  %t1 = a S0 {1}
  R m0 [%t1] ttg.local_load {1}
  r  S1 %t1 {1}                  ; e2: buffer free for the next iteration
  %t2 = a S1 {0}                 ; recurrence acquire for the NEXT iteration
  yield %t2                      ; carried out through the iter-arg
}
```

The pass then selects POINT_OF_USE — the pre-loop acquire is gone and the
recurrence acquire sits at the store instead (the node's
`holdrule{pointofuse->...}` label; the decision itself is explained in
[Region flows](#region-flows)). The resulting SYNC-DAG:

```text
|- scf.for (WS, tag=0) pieces{P0:W:{0}} thread{{0}} holdrule{pointofuse->ttg.local_store}
|  |- a  S1  {0}                ; e2 satisfied: buffer free for this iteration
|  |- W m0  ttg.local_store {0}
|  |- r  S0  {0} [none]         ; e1: data ready
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]         ; e2: buffer free for the next iteration
|  |- EXIT pieces{P0:W:{0}} yield{native}
SEMAS: S0{count=1} S1{count=1 entry inherit={@0.0}}
```

```text
                          ENTER(i)
                              |
                          a S1 {0}
                              |
                          W m0 {0}
                              | S0
                              v
                          R m0 {1}
                              | S1
                              v
                          EXIT(i)
                              |
                              v
                         ENTER(i+1)
                              |
                          a S1 {0}
                              |
                              v
                          W m0 {0}
```

`S1` is created initially released so iteration zero's `a S1` succeeds
before any release has run.

### Example: edge reduction lowers the pending count

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` — members `m0` and `m1` overlap on piece P1.
The table follows that piece as `{0}` writes, `{1}` reads, and `{2}` writes:

```text
walk          edge                                       P1 state AFTER the node
ENTER {0}     —                                           source=ENTER@{0} uses=[{0}->ENTER@{0}]
W m0 {0}      — (same-owner write)                       source=W@{0} uses=[{0}->W@{0}]
R m0 {1}      e1: W@{0} -> R@{1}             (RAW)      source=W@{0} uses=[{0}->W@{0}, {1}->R@{1}]
W m1 {2}      e2: {0} -> {2}, e3: {1} -> {2} (WAR)      source=W@{2} uses=[{2}->W@{2}]
```

The write raises two candidates because `uses` has entries for both `{0}` and
`{1}`.
Reduction drops `e2`: kept edges `e1` and `e3` already order
`{0} -> {1} -> {2}`. Only `e3` remains incoming to `{2}`'s write, so its
acquire has pending count 1:

```text
W m0 {0}
r S0 {0}                    ; e1: {0} releases to {1}
a S0 {1}
R m0 {1}
r S1 {1}                    ; e3: {1} releases to {2}
a S1 {2}                    ; one incoming edge -> pending count 1
W m1 {2}
```

```text
                            ENTER(i)
                                |
                            a S3 {0}
                                |
                            W m0 {0}
                                | S0
                                v
                            R m0 {1}  (also r S3 for next iteration)
                                | S1
                                v
                            W m1 {2}
                                | S2
                                v
                            R m1 {0}
                                |
                                v
                             EXIT(i)
                                |
                                v
                           ENTER(i+1)
                                |
                            a S3 {0}
                                |
                                v
                            W m0 {0}
```

The complete loop and the reduction proof appear in
[Example: a redundant edge is dropped](#example-a-redundant-edge-is-dropped).

### Example: fan-out and owner-token reuse

`test/NVWS/insert_semas_transitive_reduction.mlir` `@fanout_not_reduced` —
partition 0 writes, partitions 1 and 2 read, then partition 0 *re-reads*.
Post-state is shown for each access; the `EXIT` lines show recurrence edges:

```text
walk            edge                                    state AFTER the node
ENTER {0}       —                                        source=ENTER@{0} uses=[{0}->ENTER@{0}] tokens=[{0}:no-source]
W alloc {0}     — (same-owner write)                    source=alloc@{0} uses=[{0}->alloc@{0}] tokens=[{0}]
R load  {1}     e1: alloc@{0} -> load@{1}    (RAW)      source=alloc@{0} uses=[{0}->alloc@{0}, {1}->load@{1}] tokens=[{0},{1}]
R load  {2}     e2: alloc@{0} -> load@{2}    (RAW)      source=alloc@{0} uses=[{0}->alloc@{0}, {1}->load@{1}, {2}->load@{2}] tokens=[{0},{1},{2}]
R load  {0}     — (reuse {0}'s token)                   source=alloc@{0} uses=[{0}->load@{0}, {1}->load@{1}, {2}->load@{2}] tokens=[{0},{1},{2}]
EXIT            e3: load@{1} -> alloc@{0}@next (WAR)
(recurrence)    e4: load@{2} -> alloc@{0}@next (WAR)
```

Both new readers take their memory edge from the unchanged version source,
so neither reader is serialized behind the other. The fourth access is a reread
by an owner already present in `uses`. It needs no memory edge, and `{0}` still has the
token acquired before its write, so SYNC-DAG marks the node to reuse that
token. Reusing a token that is not last does not reorder the token list, so
`{2}` remains the deterministic source for any later token-supply edge.

At `EXIT`, the next iteration's write waits for the nodes stored under `{1}`
and `{2}` in `uses`. Those edges converge on `S2`, whose pending count is 2:

```text
|- a  S2(2)  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated}
|  |- W m0  ttg.local_alloc {0}
|  |- r  S0  {0} [none]
|  |- r  S1  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S2  {1} [none]        ; e3
|  |- a  S1  {2}
|  |- R m0  ttg.local_load {2}
|  |- r  S2  {2} [none]        ; e4
|  |- R m0  ttg.local_load {0} ; reuses {0}'s S2 token
|  |- a  S2(2)  {0}            ; waits for readers {1} and {2}
|  |- EXIT ... yield{a S2}
SEMAS: S0{count=1} S1{count=1} S2{count=2 entry inherit={@0.0}}
```

```text
                           a S2(2) root
                                 |
                             ENTER(0)
                                 |
                             W m0 {0}
                    +------------+------------+
                 S0 |            |            | S1
                    v            v            v
                R m0 {1}     R m0 {0}     R m0 {2}
                 S2 |            |            | S2
                    +------------+------------+
                                 |
                           a S2(2) {0}
                                 |
                              EXIT(0)
                                 |
                              ENTER(1)
                                 |
                             W m0 {0}
```

The token-reuse mark is not printed in this dump. EMIT-IR renders it by
building `{0}`'s final buffer view from the carried `S2` token, even though
that same token already fed the two producer releases. Only nodes carrying
this explicit SYNC-DAG proof receive that exception.

### Example: disjoint submember protocols run independently

A minimal synthetic function, `@spanning_split_parallel`, uses two disjoint
half-width members and one full-width member. It first writes and reads the
full-width member, then writes and reads each half. Running it through
`InsertSemas` prints:

```text
members:    m0[0,128)   m1[128,256)   m2[0,256)
pieces:     P0=[0,128){m0,m2}   P1=[128,256){m1,m2}
footprints: m0={P0}   m1={P1}   m2={P0,P1}

ACCESS-DAG
|- W m2  ttg.local_store {0}
|- R m2  ttg.local_load  {1}
|- W m0  ttg.local_store {2}
|- R m0  ttg.local_load  {3}
|- W m1  ttg.local_store {4}
|- R m1  ttg.local_load  {0}
```

The generated SYNC-DAG is:

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S5(2)  {0}
|  |- W m2  ttg.local_store {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- R m2  ttg.local_load {1}
|  |- r  S1  {1} [none]
|  |- r  S3  {1} [none]
|  |- a  S1  {2}
|  |- W m0  ttg.local_store {2}
|  |- r  S2  {2} [none]
|  |- r  S5  {2} [none]
|  |- a  S2  {3}
|  |- R m0  ttg.local_load {3}
|  |- r  S5  {3} [none]
|  |- a  S3  {4}
|  |- W m1  ttg.local_store {4}
|  |- r  S4  {4} [none]
|  |- a  S4  {0}
|  |- R m1  ttg.local_load {0}
|  |- EXIT ... yield{native}
SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=1} S4{count=1}
       S5{count=2 entry inherit={@0.0}}
```

The full-width `W m2 {0}` releases `S0`, which `R m2 {1}` acquires. That read
then starts two independent paths:

```text
                              ENTER(i)
                                  |
                            a S5(2) {0}
                                  |
                              W m2 {0}
                                  | S0
                                  v
                              R m2 {1}
                     +------------+------------+
                  S1 |                         | S3
                     v                         v
                 W m0 {2}                 W m1 {4}
                +----+----+                    | S4
             S2 |         | S5                 v
                v         |                R m1 {0}
            R m0 {3}      |                    |
             S5 |         |                    v
                |         |                 EXIT(i)
                |         |                    |
                |         |                    v
                |         |               ENTER(i+1)
                |         |                    |
                +---------+---------------> a S5(2) {0}
                                                |
                                                v
                                            W m2 {0}
```

`S5` has two incoming edges because P0's `uses` map has two entries at `EXIT`. `W m0
{2}` resets its uses to `{2}`, then `R m0 {3}` waits for that write and adds
its use, leaving `{2,3}`. P0 returns to owner `{0}` for the next iteration,
so the `EXIT` rule adds one close from `{2}` to `{0}` and another from `{3}`
to `{0}`. The `S2` edge records that `{2}` is ordered before `{3}`, not that
either stored node is already ordered before `{0}`, so neither close is
skipped. These are memory edges from the per-piece uses; token supply adds
nothing here.

EMIT-IR materializes those same two paths (unrelated operations and types
omitted):

```text
for {
  %t0 = a S5(2) {0}
  W m2 [%t0] {0}
  r S0 %t0 {0}

  %t1 = a S0 {1}
  R m2 [%t1] {1}
  r S1 %t1 {1}
  r S3 %t1 {1}

  %t2 = a S1 {2}
  W m0 [%t2] {2}
  r S2 %t2 {2}
  r S5 %t2 {2}
  %t3 = a S2 {3}
  R m0 [%t3] {3}
  r S5 %t3 {3}

  %t4 = a S3 {4}
  W m1 [%t4] {4}
  r S4 %t4 {4}
  %t5 = a S4 {0}
  R m1 [%t5] {0}
}
```

### The per-access rules, in full

`applyTouch` advances one piece's source/use state:

```text
first touch of a piece:
  edges:  none
  state:  source = this node; uses = [toucher -> this node]

write by owner P:
  edges:  for each entry Q -> node in uses where Q != P,
            add node -> this write unless node is already ordered before it
  state:  source = this write; uses = [P -> this write]

read by an owner already in uses:
  edges:  none
  state:  replace that owner's entry with owner -> this read;
          the source does not move

read by a new owner:
  edges:  source -> this read
  state:  add reader -> this read to uses
```

After all touched pieces have run through `applyTouch`,
`ChainWalker::visitAccess` chooses how the access obtains a token:

```text
one or more memory edges were added
  their acquire supplies the token; record the access owner's token last

no memory edge, and the owner has an eligible earlier token
  mark reuseTokenOwner; if that owner's token is not last, preserve the list
  order; otherwise update its source node to this access

no memory edge, no reusable owner token, and the last token has another owner
  add last token's node -> this access only to supply a token; record the
  access owner's token last

otherwise
  no extra edge; the token already available supplies the access, and a
  partition-owned access records its token last
```

### Composition: nested regions in the walk

Nested `for` and `if` regions use the same read, write, token-supply, and
`EXIT` rules already defined above. The walk applies those rules at two DAG
levels:

```text
parent DAG                           child DAG

... -> [region node] -> ...          ENTER -> child nodes -> EXIT
```

Read those two levels in this order.

1. **Walk the region node in the parent DAG.** OWNER-DAG has already put a
   per-piece summary on that node. For example, `pieces{P0:W:{2}}` says that,
   from the parent DAG, the whole region is one write of P0 by `{2}`. Apply the
   ordinary write rule to that node. This determines the edges into the region
   and the parent state after it.

2. **Start a separate child DAG at `ENTER`.** Each piece on `ENTER` names that
   piece's owner at the child boundary:

```text
ENTER pieces{P0:W:{2}, P1:W:{3}}

P0 starts as: source = ENTER (owner {2}), uses = [{2} -> ENTER]
P1 starts as: source = ENTER (owner {3}), uses = [{3} -> ENTER]
```

   `ENTER` is the child DAG's node for the version that arrived from the
   parent. A new child reader therefore gets its RAW edge from `ENTER`, not
   from a concrete node in the parent DAG.

3. **`ENTER` adds no token operation.** Piece owners and the token owner are
   separate facts:

```text
%t = a S {2}

ENTER pieces{P0:W:{2}, P1:W:{3}}   // no acquire or release here

W m0 [%t] {2}                       // m0 touches P0; uses %t
```

   `ENTER` emits no acquire or release. A child starts fresh token state; when
   all `ENTER` pieces have one partition owner, `Tokens::remember` records
   that incoming owner token without a source node. The owner may reuse it,
   but it cannot supply a token-supply edge until an access records its node.
   With the mixed owners above, no owner token is recorded. P1 owner `{3}`
   does not retag `%t`; a later `{3}` access follows the ordinary memory and
   token-supply rules.

4. **Walk the child nodes with the rules already learned.** Reads and writes
   use [The per-access rules, in full](#the-per-access-rules-in-full). At
   `EXIT`, if a piece is needed by another iteration or after the region, the
   ordinary `EXIT` rule closes the node stored for each owner other than that
   piece's `ENTER` owner.

5. **Resume the parent DAG from the region node.** Child `PieceState`s remain
   separate from the parent `PieceState`; entries in the child's `uses` map do
   not replace entries in the parent's map. The parent continues from the state produced when it
   applied the region node's summary in step 1.

A `for` has one child DAG. An `if` applies steps 2–4 separately to its `then`
and `else` DAGs. The example below uses a plain inner `for`.

#### Worked example: apply the same rules at both levels

`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain` has an outer `W m0 {3}` and a
nested plain `for`. That region node carries `pieces{P0:W:{2}}`: from the
parent, the whole child is one write of P0 by `{2}`. The listing and DAG below
label that node `[for summary P0:W:{2}]`. For one outer iteration, the parent
DAG contains these four nodes:

```text
parent DAG node              generated edge
ENTER outer(i) {3}           —
W m0 {3}                     —
[for summary P0:W:{2}]       e1: W m0 {3} -> [for summary P0:W:{2}]
EXIT outer(i) {3}            e2: [for summary P0:W:{2}] -> EXIT outer(i) {3}

state after node             source                             uses                              tokens (last at right)
ENTER outer(i) {3}           ENTER outer(i) {3}                [{3}->ENTER outer(i) {3}]          [{3}:no-source]
W m0 {3}                     W m0 {3}                          [{3}->W m0 {3}]                    [{3}]
[for summary P0:W:{2}]       [for summary P0:W:{2}]            [{2}->for summary P0:W:{2}]        [{2}]
EXIT outer(i) {3}            [for summary P0:W:{2}]            [{3}->EXIT outer(i) {3}]           [{2}]
```

These are the ordinary rules:

- `ENTER outer(i) {3}` starts P0 with source `ENTER` and
  `uses = [{3} -> ENTER]`.
- `W m0 {3}` writes as the existing owner, so it creates no edge and becomes
  the new source; `uses` becomes `[{3} -> W m0 {3}]`.
- `[for summary P0:W:{2}]` is a write by `{2}`. The write rule makes it wait
  for the node stored under `{3}` (`e1`), then sets the source to the region
  node and `uses` to `[{2} -> region node]`.
- `EXIT outer(i) {3}` returns P0 to owner `{3}`. The only entry in `uses` is
  owned by `{2}`, so the `EXIT` rule creates `e2`.

The same parent walk, with generated edges overlaid, is:

```text
                     ENTER outer(i) {3}
                              | walk order; no generated edge
                              v
                          W m0 {3}
                              | e1
                              v
                 [for summary P0:W:{2}]
                              | e2
                              v
                     EXIT outer(i) {3}
```

Before applying `[for summary P0:W:{2}]`, SYNC-DAG saves that `{3}` produced
the incoming version. P0's owner at `ENTER inner(i) {2}` is `{2}`, so the
child starts with:

```text
producer                  = {3}
source                    = ENTER inner(i) {2}
uses                      = [{2} -> ENTER inner(i) {2}]
tokens                    = [{2}:no-source]
```

Now apply the same rules to the child nodes:

```text
child DAG node              generated edge
ENTER inner(i) {2}          —
R m0 {2}                    —
R m0 {1}                    e3: ENTER inner(i) {2} -> R m0 {1}
W m0 {1}                    e4: R m0 {2} -> W m0 {1}
R m0 {0}                    e5: W m0 {1} -> R m0 {0}
EXIT inner(i) {2}           e6: W m0 {1} -> EXIT inner(i) {2}
                             e7: R m0 {0} -> EXIT inner(i) {2}

state after node             source                    uses                                      tokens (last at right)
ENTER inner(i) {2}          ENTER inner(i) {2}        [{2}->ENTER inner(i) {2}]                  [{2}:no-source]
R m0 {2}                    ENTER inner(i) {2}        [{2}->R m0 {2}]                            [{2}]
R m0 {1}                    ENTER inner(i) {2}        [{2}->R m0 {2}, {1}->R m0 {1}]             [{2},{1}]
W m0 {1}                    W m0 {1}                  [{1}->W m0 {1}]                            [{2},{1}]
R m0 {0}                    W m0 {1}                  [{1}->W m0 {1}, {0}->R m0 {0}]             [{2},{1},{0}]
EXIT inner(i) {2}           W m0 {1}                  [{2}->EXIT inner(i) {2}]                   [{2},{1},{0}]
```

Read the table mechanically:

- `ENTER inner(i) {2}` starts P0 with source `ENTER` and
  `uses = [{2} -> ENTER]`.
- `R m0 {2}` is a reread by an owner already in `uses`. It creates no memory
  edge, changes the entry to `[{2} -> R m0 {2}]`, and reuses `{2}`'s token.
- `R m0 {1}` is a new reader, so `e3` comes from the unchanged version source,
  `ENTER inner(i) {2}`. `uses` gains `{1} -> R m0 {1}`.
- `W m0 {1}` waits for every other owner's node in `uses`. The node stored
  under `{2}` is now `R m0 {2}`, so that read is the source of `e4`. No edge
  from `R m0 {1}` is needed because that entry is owned by `{1}`. The write
  sets both the source and `uses` to `W m0 {1}`.
- `R m0 {0}` is a new reader of that version, so `e5` comes from `W m0 {1}`
  and `uses` gains `{0} -> R m0 {0}`.
- `EXIT inner(i) {2}` returns P0 to owner `{2}`. The entries in `uses` are
  owned by `{1}` and `{0}`, so the ordinary `EXIT` rule creates `e6` and `e7`.

`e6` leaves from `W m0 {1}`, not from `R m0 {1}`, because the write makes
the `{1}` entry in `uses` point to `W m0 {1}`. The child walk with its generated
dependency edges overlaid is below. `walk` means walk order with no generated
edge:

```text
                         ENTER inner(i) {2}
                                   |
                         +---------+---------+
                    walk |                   | e3
                         v                   v
                     R m0 {2}            R m0 {1}
                      e4 |                   | walk
                         +---------+---------+
                                   |
                                   v
                              W m0 {1}
                                   |
                         +---------+---------+
                      e5 |                   | e6
                         v                   |
                     R m0 {0}                |
                      e7 |                   |
                         +---------+---------+
                                   v
                          EXIT inner(i) {2}
```

Only `e3`–`e7` are generated dependency edges. In the overlaid walk,
`ENTER -> R m0 {2}` and `R m0 {1} -> W m0 {1}` are program order only; they
are not DAG edges. The generated `e5`/`e6` split leaves `W m0 {1}`, and `e7`
joins `R m0 {0}` at the same EXIT.

At this walk stage, `EXIT inner(i) {2}` is the destination of `e6` and `e7`.
During edge-to-semaphore conversion, their `{2}` acquire is placed immediately
before `EXIT inner(i) {2}`. That EXIT yields the token to inner iteration
`i+1`, or returns it to the parent when the inner loop finishes.

The child and parent have separate `PieceState`s, so child uses `{1,0}` never
replace the parent's uses. In this loop, `e6` and `e7` separately close those
child uses to `EXIT inner(i) {2}`. The parent continues from the state already
established at `[for summary P0:W:{2}]`: source at the region node and one use
for `{2}`. The region also clears the pre-region token list. A uniform
partition-owned summary records only its owner's token, using the region node
as the handoff source; a mixed-owner or root-owned summary records no
partition token.

This reset feeds directly into the token-supply rule learned above. For
example, after `W m0 {0}` then `[region summary P0:R:{1}]`, `{0}` still has an
entry in `uses`, but its pre-region token is no longer recorded. A later `R m0
{0}` needs no RAW/WAR edge; it has no reusable owner token, so the ordinary
token-supply rule adds the `{1} -> {0}` handoff. No new dependency rule is
involved.

The logical producer is remembered state, not another child DAG node or edge
source. Child RAW edges originate at `ENTER inner(i) {2}`. The producer
decides whether this ENTER imports the outer completion kinds and is then
preserved for deeper children. Its completion kind is `[none]` when its owner
is not that producer. Here the producer is `{3}` and the ENTER owner is `{2}`,
so this distinction changes the completion kind carried by `e3`, but does not
change the topology of edges `e1`–`e7`.

At this point the walk has produced only edges `e1`–`e7`; it has not assigned
semaphore names, pending counts, or release multiplicities. Their conversion
is shown later in
[Composition: why loop entry and loop recurrence share one semaphore](#composition-why-loop-entry-and-loop-recurrence-share-one-semaphore).

## Edges to semaphores

After redundant edges are deleted, `buildEdgesAndSemas` groups edges with the
same destination and destination owner. Within one group, repeats from the
same source owner are merged at that owner's later source node and their
completion kinds are unioned. Each resulting incoming edge's tail becomes one
release carrying those completion kinds. Each carried
completion kind counts as one release; an edge carrying no completion kind
counts as one release. The destination becomes one acquire whose pending count
is the total number of releases from its incoming edges. An edge entering a
loop node can instead reuse the loop's recurrence semaphore and inherit its count —
the composition case below.

Placement needs no extra information, because the walk never produced a
placeless fact: every edge already points at two concrete nodes.

```text
release  ->  spliced immediately AFTER the edge's source node
             (after the completionAnchor instead, when ACCESS-DAG
             recorded one)
acquire  ->  spliced immediately BEFORE the shared destination node
create   ->  before the WS loop; its position carries no synchronization
```

Token reuse remains an explicit SYNC-DAG fact during this conversion. If an
edge leaves an access marked with `reuseTokenOwner`, its inserted release is
marked for the same owner, so both nodes use that owner's earlier token.
`verifySyncDag` checks the resulting token and structural invariants; EMIT-IR
checks that the named owner still has a live token. EMIT-IR only renders these
marks; it does not decide that another node is eligible.

Before conversion, `buildEdgesAndSemas` first deletes ordering already
implied by other edges (`reduceEdges`), then groups each destination and
coalesces repeats from one sender while building that handoff. Reduction is a
monotonic trace over permanently kept edges;
the complete conditions follow the two worked examples as fine print.

### Example: a redundant edge is dropped

`test/NVWS/insert_semas_transitive_reduction.mlir` `@serialized_ring_reduces`
— member `m0` covers the whole backing; member `m1` is a view nested inside it:

```text
members:    m0[0,256)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0, m1}   P2=[192,256){m0}
footprints: an access through m0 touches all three pieces; through m1, P1
```

Four accesses per iteration; the walk on the shared piece P1 raises all
four in-body edges (P0 and P2 repeat e1), and P0/P2 raise the one
recurrence edge at `EXIT`:

```text
walk             P1 state before       rule                      raw edge
ENTER {0}        —                     seed source ENTER and     —
                                       use {0}
W m0 {0}         source ENTER          same-owner write; reset   —
                 uses [{0}->ENTER]     source and uses to W
R m0 {1}         source W@{0}          read by new owner         e1: {0} -> {1}
                 uses [{0}->W m0 {0}]
W m1 {2}         uses [{0}->W m0 {0},  write: edge from EVERY    e2: {0} -> {2}
                 {1}->R m0 {1}]        other owner's stored node e3: {1} -> {2}
R m1 {0}         source W@{2}          read by new owner         e4: {2} -> {0}
                 uses [{2}->W m1 {2}]
EXIT             every piece carries   P0/P2: use {1} closes;    e5: {1} -> {0}@next
(recurrence)     owner {0}             P1: skip, e4 already
                                       orders the node stored for {2}
                                       before {0}
```

The raw DAG makes `e2`'s redundant branch and `e5`'s `EXIT` close visible:

```text
                             ENTER(i)
                                 |
                             W m0 {0}
                    +------------+------------+
                 e1 |                         | e2 (redundant)
                    v                         |
                R m0 {1}                      |
           +--------+--------+                |
        e5 |              e3 |                |
           |                  +----------------+
           |                                   |
           |                                   v
           |                               W m1 {2}
           |                                   | e4
           |                                   v
           |                               R m1 {0}
           |                                   |
           +------------------+----------------+
                              |
                              v
                         EXIT(i) {0}
                              |
                              v
                         ENTER(i+1)
                              |
                              v
                          W m0 {0}
```

The diagram contains two edges that look redundant if we check ordering only:

- `e2` has the alternate path `e1 -> e3` to the same `W m1 {2}`.
- `e5` has the alternate path `e3 -> e4`, followed by `{0}`'s own order to
  the next iteration's `W m0 {0}`.

Deleting an edge requires more than alternate ordering: the destination must
still receive a token in time.

- **Drop `e2`.** Kept `e3` enters `W m1 {2}`, so it supplies `{2}`'s acquire
  and token as well as completing the alternate ordering path.
- **Keep `e5`.** `W m0 {0}` is the next iteration's first partition-owned
  access. No earlier access in that iteration can supply its token, and the
  token acquired by `e4` in the current iteration is not the loop-entry token
  for the next iteration. `e5` supplies that loop-entry token, so the
  wrap-around reducer protects it as the close into first-access owner `{0}`.

In the raw DAG, `e5` ends at `EXIT(i) {0}` and represents the handoff to the
next iteration. Region-flow placement puts its acquire immediately before
`W m0 {0}`.

The result is:

```text
dropped:  e2
kept:     e1  e3  e4  e5
```

Reduction finishes before semaphore creation. The four kept edges have four
different destinations: `e1` enters `R m0 {1}`, `e3` enters `W m1 {2}`, `e4`
enters the current iteration's `R m1 {0}`, and `e5` enters `EXIT(i) {0}` for
the next iteration. They therefore become four count-1 semaphores. In
particular, `e5` becomes its own S3 with pending count 1; it does not increase
another semaphore's pending count. The emitted body is:

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_alloc}
|  |- a  S3  {0}                ; e5's acquire, at the top of the body
|  |- W m0  ttg.local_alloc {0}
|  |- r  S0  {0} [none]         ; e1 tail: {0} releases to {1}
|  |- a  S0  {1}                ; e1 head
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]         ; e3 tail: {1} releases to {2}
|  |- r  S3  {1} [none]         ; e5 tail: releases the next iteration's {0}
|  |- a  S1  {2}                ; e3 head
|  |- W m1  ttg.local_alloc {2}
|  |- r  S2  {2} [none]         ; e4 tail: {2} releases to {0}
|  |- a  S2  {0}                ; e4 head
|  |- R m1  ttg.local_load {0}
|  |- EXIT ... yield{native}
SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=1 entry inherit={@0.0}}
```

After reduction and semaphore placement, `e2` is gone. S0/S1/S2 form the
in-body ring, while S3 carries `e5` to the next iteration:

```text
                             ENTER(i)
                                 |
                             a S3 {0}
                                 |
                             W m0 {0}
                                 | S0
                                 v
                             R m0 {1}
                        +--------+--------+
                     S1 |                 | S3
                        v                 |
                    W m1 {2}              |
                        | S2              |
                        v                 |
                    R m1 {0}              |
                        |                 |
                        v                 |
                     EXIT(i)              |
                        |                 |
                        v                 |
                   ENTER(i+1)             |
                        |                 |
                    a S3 {0} <------------+
                        |
                        v
                    W m0 {0}
```

`S3`'s acquire sits at the top of the body rather than above the loop —
that placement is what the node's `holdrule{pointofuse->...}` label stands
for, decided in [Region flows](#region-flows) below.
Contrast `@fanout_not_reduced` above, `holdrule{gated}`: its `S2`
stays a pre-loop root entry and the loop uses CARRIED mode.

And contrast the semaphore count: `@fanout_not_reduced` has two RAW fan-out
edges plus two recurrence edges sharing one destination. Its four kept
edges therefore become three semaphores, one with pending count 2. Same
rule both times: one destination node, one semaphore.

### Example: a loop-closing edge is dropped

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` — the same four accesses and owners as
`@serialized_ring_reduces`, but `m1` extends *past* `m0` instead of
nesting inside it, so piece P2 belongs to `m1` alone, its first toucher in
the body is `{2}`, and one recurrence edge now targets an owner other than
the chain's first:

```text
members:    m0[0,128)   m1[64,192)
pieces:     P0=[0,64){m0}   P1=[64,128){m0, m1}   P2=[128,192){m1}
footprints: an access through m0 touches P0 and P1; through m1, P1 and P2
```

The walk raises the same in-body ring as before — kept `e1 {0}->{1}`,
`e2 {1}->{2}`, `e3 {2}->{0}` (a raw `{0}->{2}` drops exactly the way the
previous example's `e2` did; and `R m1 {0}` raises its edge through P1 and
P2 at once, counted once when `buildEdgesAndSemas` groups the handoff). At
`EXIT` two closes are
raised:

```text
e4: R m1@{0} -> {2}@next   (piece P2 — {2}'s write replaces data {0} holds)
e5: R m0@{1} -> {0}@next   (pieces P0/P1 — same for {0}'s write)
```

The diagram below is after the ordinary in-body reduction, which already
removed the direct `{0}->{2}` edge, but before the wrap-around reduction.
The stored `e4` and `e5` edges enter `EXIT`; this equivalent unrolled picture
carries each boundary owner to its next-iteration access. `walk` marks walk
order with no generated edge. The long path from `R m1 {0}` through iteration
`i+1` reaches the same `W m1 {2}` as `e4`:

```text
                                  ENTER(i)
                                      | walk
                                      v
                                  W m0 {0}
                                      | e1
                                      v
                                  R m0 {1}
                         +------------+------------+
                      e5 |                         | e2
                         |                         v
                         |                     W m1 {2}
                         |                         | e3
                         |                         v
                         |                     R m1 {0}
                         |                         +----------------+
                         |                         | walk        e4 |
                         |                         v                |
                         |                      EXIT(i)             |
                         |                         | walk           |
                         |                         v                |
                         |                    ENTER(i+1)            |
                         |                         | walk           |
                         |                         v                |
                         +------------------> W m0 {0}              |
                                                   | e1             |
                                                   v                |
                                               R m0 {1}             |
                                                   | e2             |
                                                   v                |
                                               W m1 {2} <-----------+
```

The diagram shows all three facts: the alternate path implies `e4`; kept `e2`
has made `{2}`'s token available by P2's first touch; and `{2}` is not the
owner of the chain's first partition-owned access. Therefore `e4` drops. `e5` targets
`{0}`, the owner of the chain's first partition-owned access, so it is kept
and becomes S3. The emitted body has no second loop close into `{2}`:

```text
|- scf.for (WS, tag=1) ... holdrule{pointofuse->ttg.local_alloc}
|  |- a  S3  {0}                ; e5, the kept close
|  |- W m0  ttg.local_alloc {0}
|  |- r  S0  {0} [none]         ; e1 tail
|  |- a  S0  {1}                ; e1 head
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]         ; e2 tail
|  |- r  S3  {1} [none]         ; e5 tail
|  |- a  S1  {2}                ; e2 head — the only acquire {2}'s hold
|  |- W m1  ttg.local_alloc {2} ;   needs: e4 is gone
|  |- r  S2  {2} [none]         ; e3 tail
|  |- a  S2  {0}                ; e3 head
|  |- R m1  ttg.local_load {0}
|  |- EXIT ... yield{native}
SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=1 entry inherit={@1.0}}
```

The final DAG makes the deletion visible. S3 closes into the next iteration's
`{0}` access; there is no cross-iteration edge into `W m1 {2}` in iteration
`i+1`:

```text
                                  ENTER(i)
                                      |
                                      v
                                  W m0 {0}
                                      | S0
                                      v
                                  R m0 {1}
                             +--------+--------+
                          S1 |                 | S3
                             v                 |
                         W m1 {2}              |
                             | S2              |
                             v                 |
                         R m1 {0}              |
                             | walk            |
                             v                 |
                          EXIT(i)              |
                             | walk            |
                             v                 |
                        ENTER(i+1)             |
                             |                 |
                             +------------> W m0 {0}
                                               | S0
                                               v
                                           R m0 {1}
                                               | S1
                                               v
                                           W m1 {2}
```

### The deletion conditions, in full

#### 1. Implied ordering (`reduceEdges`)

The straight-line sweep considers only edges between partition-owned access
nodes. It does not reduce root, `for`, `if`, `ENTER`, or `EXIT` endpoints. A
separate wrap-around pass below considers access-to-`EXIT` loop closes.

For a straight-line edge, the complete deletion shape is:

```text
                           W m0 {0}
                      +---------+---------+
                   e1 |                   | e2 candidate
                      v                   |
                  R m0 {1}                |
                   e3 |                   |
                      +---------+---------+
                                v
                           W m1 {2}
```

`e2` can be deleted because the kept `e1 -> e3` path already orders its
endpoints, and the most recently applied kept handoff, `e3`, makes `{2}` the
last recorded token owner. The result is:

```text
                           W m0 {0}
                                | e1
                                v
                           R m0 {1}
                                | e3
                                v
                           W m1 {2}
```

If either fact is missing — no alternate kept path, or the destination owner
is not the last recorded token owner after `ENTER` and the kept handoffs — the
candidate stays.

A loop-closing candidate uses the same idea, but the alternate path crosses
the iteration boundary. `reduceLoopCloses` simulates that wrap-around;
the equivalent unrolled dependency picture is:

```text
                  R m1 {0} (i) ----- e4 close to {2}@next ---------+
                       | walk                                      |
                       v                                           |
                    EXIT(i)                                        |
                       | walk                                      |
                       v                                           |
                  ENTER(i+1)                                       |
                       | walk                                      |
                       v                                           |
                  W m0 {0}  first partition-owned access           |
                       | e1                                        |
                       v                                           |
                  R m0 {1}                                         |
                       | e2  kept handoff into {2}                 |
                       v                                           v
                  W m1 {2}  first P2 touch <-----------------------+
```

The picture contains all three requirements for dropping `e4`: the long kept
path implies its ordering; the kept handoff `e2` has made `{2}`'s token
available when `{2}` first touches P2; and destination owner `{2}` is not the
owner of the chain's first partition-owned access.

The reduction is proof-by-construction. A dropped edge never updates closure
or token state, while a kept edge is applied immediately and never
reconsidered. Therefore each deletion is proved using only program order and
edges already committed to the kept set; a later deletion cannot invalidate
that proof. `reduceStraightEdges` decides ordinary candidates first, then
`reduceLoopCloses` sees those decisions as permanent and applies each kept
close before considering the next one. `verifySyncDag` still checks the
resulting token, region-flow, semaphore-count, and structural invariants;
it does not replay the reducer.

#### 2. Repeats from one sender (`buildEdgesAndSemas`)

`reduceEdges` has finished. While `buildEdgesAndSemas` groups a destination,
it handles two kept edges that have the same sending owner and destination but
leave from different nodes.

Use this minimal conceptual loop; no lit test dumps this exact shape. `m0`
touches P0 and P1, while `m1` touches only P1:

```text
node             pieces / role
W m0 {0}         touches P0,P1; first partition-owned access
R m0 {1}         touches P0,P1
R m1 {1}         touches P1
EXIT(i) {0}      returns P0,P1 to {0}

at EXIT:
  P0's latest {1} node is R m0 {1} -> e2
  P1's latest {1} node is R m1 {1} -> e3
```

Both closes survive `reduceEdges` because they return to `{0}`, owner of the
chain's first partition-owned access. The raw DAG entering handoff grouping is:

```text
                            W m0 {0}
                                | e1
                                v
                            R m0 {1}
                         +------+------+
                  e2(P0) |             | walk
                         |             v
                         |         R m1 {1}
                         |             | e3(P1)
                         +------+------+
                                v
                           EXIT(i) {0}
```

`e2` and `e3` have the same sending owner `{1}` and the same destination
`EXIT(i) {0}`. `R m1 {1}` follows `R m0 {1}` in the same owner's walk, so one
release after `R m1 {1}` is also after `R m0 {1}`.
`buildEdgesAndSemas` combines the two closes at that later node and unions
their completion payloads. Piece
identities have already served reduction and are not propagated into the
semaphore record; `P0,P1` below is explanatory:

```text
                            W m0 {0}
                                | e1
                                v
                            R m0 {1}
                                | walk
                                v
                            R m1 {1}
                                | e2+e3  merged close: P0,P1
                                v
                           EXIT(i) {0}
```

The merged edge carries one completion kind, so its release counts once and
conversion creates one semaphore with pending count 1. If the merged edge
carries `[none, tma_load]`, its release counts twice: once when the synchronous
operation completes and once when the TMA load completes. The destination
acquire therefore has pending count 2. In the unrolled edge-only protocol DAG,
S0 represents `e1` and S1 represents merged `e2+e3`; the initial entry is
omitted:

```text
                            ENTER(i)
                                |
                            W m0 {0}
                                | S0
                                v
                            R m0 {1}
                                | walk
                                v
                            R m1 {1}
                         +------+------+
                    walk |             | S1
                         v             |
                      EXIT(i)          |
                         | walk        |
                         v             |
                    ENTER(i+1)         |
                         +------+------+
                                v
                            W m0 {0}
```

#### 3. Covered senders (`buildEdgesAndSemas`)

The walk annotates rather than deletes: when a WAR/WAW edge's source use is
the piece's version source and another live use is already ordered after it
(the source's `orderedBefore` names that use's owner), the edge is recorded
with `coveredVia` = that owner. The fact is still true and the edge still
participates in reduction and in the same-sender merge above — a covered
edge of a surviving sender anchors that sender's release exactly as before.

After grouping and merging, a sender whose EVERY merged edge is covered
contributes no ordering of its own: each covered source is ordered before
the coverer's acquire, and the coverer raises its own arrival into the same
destination. Such a sender's arrival is deleted whole, and the semaphore's
pending count shrinks by its contribution; a handoff whose every sender is
covered dissolves entirely. The deletion validates that the covering path
survives: the coverer must still arrive at this destination (leg 2), and a
surviving release of the deleted sender — acquired by the coverer no later
than the coverer's own release into this handoff — must certify the covered
source (leg 1). A candidate may only rely on paths that themselves survive,
so the candidate set shrinks to a fixpoint before it is applied.

Granularity is the point of this rule. Deleting a covered edge whose sender
keeps other edges into the destination saves nothing — the merge emits one
release either way — but re-anchors that release at an earlier surviving
source. A release lowers to a warp-group rendezvous plus one arrive, and
re-anchoring it from after the sender's last write to just after a long
TMEM read exposed the rendezvous on the read's dependency shadow, stalling
all warps of the bottleneck partition on the slowest read every iteration
(~7% FP16 flash-attention loss on B300). Handoff-granularity deletion can
only remove whole arrivals or whole semaphores, never move a surviving
release earlier, so redundancy elimination is placement-invariant by
construction.

### One destination node, one semaphore

What remains is grouped by destination node and destination owner: one
destination, one semaphore, and one acquire. Each incoming edge produces a
release. A release counts once for each completion kind carried by its edge,
or once when the edge carries no completion kind. Their total is the acquire's
pending count. The next section shows the case in which a loop-entry edge
reuses a recurrence semaphore whose pending count came from a larger fan-in.

### Composition: why loop entry and loop recurrence share one semaphore

This section explains why `e1`, `e6`, and `e7` use the same semaphore in the
following example, and how its pending count affects `e1`. `e1` feeds
iteration 0; `e6` and `e7` feed later iterations. The `e6`/`e7` fan-in gives
`S0` pending count 2. Because `e1` is the only incoming edge at this site, its
release is given count 2: `r S0(2)`.

The loop body uses one fixed semaphore for its token iter-arg. The other edges
use the ordinary conversion rules.

The nested walk in
[Composition: nested regions in the walk](#composition-nested-regions-in-the-walk)
already produced the three raw edges involved here:

```text
edge  source                         destination
e1    W m0 {3}                       [for summary P0:W:{2}]
e6    W m0 {1}                       EXIT inner(i) {2}
e7    R m0 {0}                       EXIT inner(i) {2}
```

No raw edge is added or changed here; semaphore sharing happens only after the
raw DAG is complete. The three edges belong to two separate DAG levels. In
the parent DAG, `e1` requires the inner `for` node owned by `{2}` to wait for
`W m0 {3}`:

```text
parent DAG

                     ENTER outer(i) {3}
                              | walk
                              v
                          W m0 {3}
                              | e1
                              v
                 [for summary P0:W:{2}]
                              | e2
                              v
                     EXIT outer(i) {3}
```

In the child DAG, `e6` and `e7` close the nodes stored under `{1}` and `{0}`
to `{2}` at the end of an inner iteration:

```text
child DAG (closing part)

                              W m0 {1}
                         +---------+---------+
                      e5 |                   | e6
                         v                   |
                     R m0 {0}                |
                      e7 |                   |
                         +---------+---------+
                                   v
                          EXIT inner(i) {2}
```

After conversion, the two acquire sites feed the same token iter-arg: `e1`
supplies iteration 0, and `e6`/`e7` supply the next iteration:

```text
%first = a S0(2) {2}                 ; fed by e1
scf.for ... iter_args(%token = %first) {
  %view = nvws.semaphore.buffer S0, %token
  ... use %view ...
  %next = a S0(2) {2}                ; fed by e6 and e7
  yield %next
}
```

The loop body is emitted once, so its `nvws.semaphore.buffer` has one fixed
semaphore operand. Only `%token` is carried; there is no second loop argument
that selects a different semaphore on later iterations. Therefore `%first`
and `%next` must both be tokens from `S0`. The conversion is mechanical:

1. `e6` and `e7` have the same destination, `EXIT inner(i) {2}`. They create
   `S0` with pending count 2. Their two source nodes each emit one `r S0`.
2. `e1` enters a `for` node with acquiring owner `{2}`. Conversion finds the
   last child destination group for the same owner — here the `e6`/`e7` group
   — and reuses its `S0`, rather than creating a different semaphore for
   `e1`.
3. Every `a S0(2)` waits for two releases. `e1` has one source edge, so its
   release has count 2: `r S0(2) {3}`.

The `2` on `r S0(2)` is its release count. It does not mean that the walk
created another raw edge. The complete raw-edge-to-semaphore mapping is:

```text
acquire site                  raw edges  semaphore  pending count
before the inner for          e1         S0         2
before EXIT inner(i) {2}      e6,e7      S0         2
before R m0 {1}               e3         S1         1
before W m0 {1}               e4         S2         1
before R m0 {0}               e5         S3         1
before EXIT outer(i) {3}      e2         S4         1
```

The first two entries are two acquire sites that reuse `S0`; they are not one
three-edge fan-in.

The full converted protocol is:

```text
|- a  S4  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated}
|  |- W m0  ttg.local_store {3}
|  |- r  S0(2)  {3} [none]        ; e1 tail: one edge, release count 2
|  |- a  S0(2)  {2}               ; e1 head: initial inner-loop token
|  |- scf.for ... holdrule{gated}
|  |  |- r  S1  {2} [none]        ; e3 tail: ENTER has no op, so this starts the child chain
|  |  |- R m0  ttg.local_load {2}
|  |  |- r  S2  {2} [none]        ; e4 tail
|  |  |- a  S1  {1}               ; e3 head
|  |  |- R m0  ttg.local_load {1}
|  |  |- a  S2  {1}               ; e4 head
|  |  |- W m0  ttg.local_store {1}
|  |  |- r  S3  {1} [none]        ; e5 tail
|  |  |- r  S0  {1} [none]        ; e6 tail: release count 1
|  |  |- a  S3  {0}               ; e5 head
|  |  |- R m0  ttg.local_load {0}
|  |  |- r  S0  {0} [none]        ; e7 tail: release count 1
|  |  |- a  S0(2)  {2}            ; e6/e7 head: next inner-loop token
|  |  |- EXIT ... yield{a S0}
|  |- r  S4  {2} [none]           ; e2 tail
|  |- a  S4  {3}                  ; e2 head
|  |- EXIT ... yield{a S4}
SEMAS: S0{count=2} S1{count=1} S2{count=1} S3{count=1} S4{count=1 entry inherit={@0.3}}
```

The parent and child remain separate DAG levels, but both levels use `S0`.
These diagrams include the release and acquire nodes added during conversion.
In the parent DAG, the single `r S0(2) {3}` supplies both releases required by
the initial `a S0(2) {2}`:

```text
parent protocol DAG

                               a S4 root
                                   | walk
                            ENTER outer(0)
                                   | walk
                               W m0 {3}
                                   | walk
                                   v
                            r S0(2) {3}
                                   | S0
                                   v
                             a S0(2) {2}
                                   | walk
                                   v
                 [for summary P0:W:{2}]
                                   | walk
                                   v
                               r S4 {2}
                                   | S4
                                   v
                               a S4 {3}
                                   | walk
                                   v
                            EXIT outer(0)
```

In the child DAG, `r S0 {1}` and `r S0 {0}` each supply one release to the
bottom `a S0(2) {2}`. `EXIT inner(i)` yields that token to
`ENTER inner(i+1)`:

```text
child protocol DAG

                              ENTER inner(i) {2}
                                       | walk
                                       v
                                  r S1 {2}
                             +---------+---------+
                        walk |                   | S1
                             v                   v
                       R m0 {2}             a S1 {1}
                             | walk              | walk
                             v                   v
                        r S2 {2}             R m0 {1}
                          S2 |                   | walk
                             +---------+---------+
                                       v
                                  a S2 {1}
                                       | walk
                                       v
                                  W m0 {1}
                                       | walk
                                       v
                                  r S3 {1}
                             +---------+---------+
                        walk |                   | S3
                             v                   v
                        r S0 {1}             a S3 {0}
                          S0 |                   | walk
                             |                   v
                             |               R m0 {0}
                             |                   | walk
                             |                   v
                             |               r S0 {0}
                             |                   | S0
                             +----------+--------+
                                        v
                                  a S0(2) {2}
                                        | walk
                                        v
                               EXIT inner(i) {2}
                                        | carried token
                                        v
                              ENTER inner(i+1) {2}
```

## Region flows

### Why this is needed

After edges become acquires and releases, a `for` or `if` may have a token at
its entry and a token on each exit. `RegionFlow` summarizes those tokens so the
parent can handle the region as one node. The parent does not walk the child's
operations again.

For a loop, the same summary also records one of three decisions:

```text
mode           dump label                    planned loop token
CARRIED        gated(...) or gated           iter-arg and result
POINT_OF_USE   pointofuse->op                none
CHILD_OWNS     passthrough-drop              none; final child owns the protocol
```

Later dead-token cleanup can remove a CARRIED slot if both its iter-arg and
result become dead. `thread{{...}}` only says the region has a `RegionFlow`;
it does not by itself mean that a loop carries a token.

A loop with no internal acquire and no child flow needs none of these
decisions. One hold can cover the whole loop:

```text
%t = a S0 {1}
for {
  W acc [%t] {1}
}
r S1 %t {1}
```

### Region summaries

`summarizeRegionFlow` runs from inner regions to outer regions. It records one
boundary owner, each path's returned token or input pass, the semaphore to use
when no input token exists, and the combined schedule needed by the later
stage check. For a loop it also records the decision from the table above.

If the region has an input token, EMIT-IR uses its semaphore for the result.
Without an input token, it uses the semaphore recorded for a fresh result.
This supports an `if` whose branches return fresh tokens from different
semaphores, and an `if` whose one branch returns a fresh token while the other
passes the input. A pass-through path without an input token is invalid.

Consider `test/NVWS/insert_semas_conditional_multi_result.mlir`
`@conditional_multi_result_if_token`. Its trimmed OWNER-DAG is:

```text
|- scf.for pieces{P0:W:{1}}
|  |- ENTER pieces{P0:W:{1}}
|  |- W m0  ttng.tc_gen5_mma {1}
|  |- scf.if pieces{P0:R:{1}}
|  |  |- then
|  |  |  |- ENTER pieces{P0:R:{1}}
|  |  |  |- R m0  ttng.tmem_load {0}
|  |  |  |- EXIT pieces{P0:R:{1}}
|  |  |- else
|  |  |  |- ENTER
|  |  |  |- EXIT
|  |- EXIT pieces{P0:W:{1}}
```

The `if` is owned by `{1}` at its boundary even though its then branch hands
the buffer to `{0}`. The matching SYNC-DAG shows the handoff back to `{1}`:

```text
|- a S1 {1}
|- W m0  ttng.tc_gen5_mma {1}
|- scf.if pieces{P0:R:{1}} thread{{1}}
|  |- then
|  |  |- ENTER pieces{P0:R:{1}}
|  |  |- r S0 {1}
|  |  |- a S0 {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r S1 {0}
|  |  |- a S1 {1}
|  |  |- EXIT pieces{P0:R:{1}} yield{a S1}
|  |- else
|  |  |- ENTER
|  |  |- EXIT yield{pass}
|- r S1 {1}
```

The parent sees one `if` node returning owner `{1}`: a fresh token on the then
path and the input token on the else path. The child owns the
`{1}->{0}->{1}` handoff. The parent may still emit its ordinary closing
release after the `if`, as above.

A region is `transparent` only when the boundary owner is restored and no
group access or protocol node follows the final returned token on any path.
That lets a parent use the child as its final recurrence node. Different
returned owners, a non-empty final node that cannot return a token, or an
invalid transfer are hard verification errors. An empty path may pass the
input token. These cases are not changed to CARRIED. The recorded schedule is
consulted only by the stage check described below.

<a id="the-decision-per-region"></a>

### The loop decision

`planLoopFlow` makes the decision from the loop's `RegionFlow` and its direct
chain. It does not inspect child operations.

1. The loop must be in an eligible tagged WS scope, and its enclosing regions
   must allow the token slot to be removed.
2. The final token must come from an acquire or a transparent child region. If
   the final child loop already owns the protocol, choose CHILD_OWNS.
3. If a group access, release, or child flow follows that final token, choose
   CARRIED and print `gated(trailing-use)`.
4. The pass must find an input acquire with the same owner. If operations
   before the loop still use that token, the plan must have one copy.
5. The input and body semaphores must match, or the pass must be able to add
   the required release.
6. The pass must find the first body use. If the final token comes from an
   acquire, its closing release must exist; if it comes from a child, the pass
   adds one.
7. When a plain inner loop also needs an acquire after the loop, its known body
   and exit stages must agree.
8. Otherwise the pass places the acquire before the first use, removes the
   loop token, and adds any required acquire or release immediately outside
   the loop.

A failed check keeps CARRIED. `trailing-use` and the nested-exit
`result-consumed` case are named in the dump; other failures print bare
`gated`. Invalid returned tokens or owners fail verification instead.

### Point of use

`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use` starts with this
carried protocol:

```text
%t0 = a S2 root
for (%t = %t0) {
  W m0 [%t]  ttg.local_store {0}
  r S0 %t {0}
  %t1 = a S0 {1}
  R m0 [%t1] ttg.local_load {1}
  r S1 %t1 {1}
  %t2 = a S1 {2}
  W m0 [%t2] use_smem {2}
  r S2 %t2 {2}
  %next = a S2 {0}
  yield %next
}
```

Nothing uses `%next` at the end of this iteration. Its first use is the store
at the head of the next iteration. The pass moves `a S2` there and removes the
loop token:

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_store}
|  |- a S2 {0}
|  |- W m0  ttg.local_store {0}
|  |- r S0 {0} [none]
|  |- a S0 {1}
|  |- R m0  ttg.local_load {1}
|  |- r S1 {1} [none]
|  |- a S1 {2}
|  |- W m0  use_smem {2}
|  |- r S2 {2} [none]
|  |- EXIT ... yield{native}
SEMAS: S0{count=1} S1{count=1} S2{count=1 entry inherit={@0.0}}
```

`S2` is initially released for iteration zero. Each iteration's `r S2`
supplies the next iteration's in-body acquire.

### Trailing use

In `test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout`, owner `{0}` writes, owners
`{1}` and `{2}` read, and `{0}` then writes again:

```text
%t0 = a S2(2) root
for (%t = %t0) {
  W buf [%t] {0}
  r S0 %t {0}
  r S1 %t {0}

  %r1 = a S0 {1}
  R buf [%r1] {1}
  r S2 %r1 {1}

  %r2 = a S1 {2}
  R buf [%r2] {2}
  r S2 %r2 {2}

  %next = a S2(2) {0}
  W buf [%next] {0}
  yield %next
}
```

`%next` protects both the last `{0}` write in iteration `i` and the first
`{0}` write in iteration `i+1`. Splitting the hold at the loop boundary would
add a same-owner release/acquire pair:

```text
What removing the loop token would add. Only the end of iteration i and start
of iteration i+1 are shown. V is illustrative and is not created.

                    %next = a S2(2) {0} at i
                                  | walk
                                  v
                        W buf [%next] {0}
                                  | walk
                                  v
                         r V %next {0}
                           +------+------------------+
                      walk |                         | V
                           v                         |
                        EXIT(i)                      |
                           | next iteration           |
                           v                         |
                       ENTER(i+1)                    |
                      walk |                         |
                           +-----------+-------------+
                                       v
                                  %t = a V {0}
                                       | walk
                                       v
                                W buf [%t] {0}
```

Carrying `%next` removes that pair:

```text
What the pass emits for the same two operations.

                    %next = a S2(2) {0} at i
                                  | walk
                                  v
                        W buf [%next] {0}
                                  | walk
                                  v
                       EXIT(i) yield %next
                                  | next iteration
                                  | carried token
                                  v
                  ENTER(i+1) iter-arg %t = %next
                                  | walk
                                  v
                           W buf [%t] {0}
```

The pass therefore emits CARRIED directly:

```text
|- a S2(2) root ; entry
|- scf.for (WS, tag=0) ... holdrule{gated(trailing-use)}
|  |- W m0  ttng.tmem_store {0}
|  |- r S0 {0}
|  |- r S1 {0}
|  |- a S0 {1}
|  |- R m0  ttng.tmem_load {1}
|  |- r S2 {1}
|  |- a S1 {2}
|  |- R m0  ttng.tmem_load {2}
|  |- r S2 {2}
|  |- a S2(2) {0}
|  |- W m0  ttng.tmem_store {0}
|  |- EXIT ... yield{a S2}
```

`hasTrailingCompUse` checks for any group access, release, or child flow after
the final token, so `gated(trailing-use)` names the check; it does not by itself
prove the same-owner shape above.

`S2` is the real count-two fan-in from the two readers. The illustrative `V`
must not be folded into `S2`; doing so would change the pending count.

### Use after the loop

Using the loop result after the loop does not always require CARRIED. The pass
can add one acquire after a POINT_OF_USE loop:

```text
%entry = a S0 {1}
W buf [%entry] {1}
r S1 %entry {1}

for {
  %t = a S1 {2}
  R buf [%t] {2}
  r S2 %t {2}

  %u = a S2 {1}
  W buf [%u] {1}
  r S1 %u {1}
}

%final = a S1 {2}              ; postLoopAcquire
r S3 %final {2}                ; once, after the loop
%v = a S3 {3}
R buf [%v] {3}
```

For a non-empty loop, `%final` waits for the last iteration's `r S1`. For a
zero-trip loop, it waits for the `r S1` placed before the loop.

When the final token comes from a child and the same semaphore is also needed
after the loop, the pass keeps the loop token and prints
`gated(result-consumed):nestedExit`. Other failed checks print bare `gated`.

### Nested loops

Region flows are decided from inner to outer. After an inner loop is decided,
the parent sees only its updated summary.

In `test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop`, the inner loop becomes POINT_OF_USE. The outer loop
had only forwarded that token, so it chooses CHILD_OWNS and drops its slot:

```text
|- scf.for (WS, tag=0) ... holdrule{passthrough-drop:nestedExit}
|  |- scf.for ... holdrule{pointofuse->ttng.tc_gen5_mma}
|  |  |- a S1 {1}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r S0 {1} [tc5mma]
|  |  |- a S0 {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r S1 {0} [none]
|  |  |- EXIT ... yield{native}
|  |- EXIT ... yield{drop}
```

`@nested_ws_inner_loop_parent_continuation` adds a read after the inner loop.
The inner acquire stays at the MMA. The post-loop `a S1` starts the handoff to
the outer read; `a S2` supplies the token used by that read. The final `r S1`
supplies the next outer iteration:

```text
|- func @nested_ws_inner_loop_parent_continuation
|  |- a S3 root ; entry
|  |- scf.for (WS, tag=1) ... holdrule{gated(trailing-use)}
|  |  |- scf.for ... holdrule{pointofuse->ttng.tc_gen5_mma:postLoopAcquire:entryBridge}
|  |  |  |- a S1 {1}
|  |  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |  |- r S0 {1} [tc5mma]
|  |  |  |- a S0 {0}
|  |  |  |- R m0  ttng.tmem_load {0}
|  |  |  |- r S1 {0} [none]
|  |  |  |- EXIT ... yield{native}
|  |  |- a S1 {1}                 ; postLoopAcquire
|  |  |- r S2 {1} [tc5mma]
|  |  |- a S2 {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r S3 {0} [none]
|  |  |- a S3 {1}
|  |  |- r S1 {1} [none]          ; entryBridge
|  |  |- EXIT ... yield{a S3}
```

The inner and outer decisions are independent: the inner loop is
POINT_OF_USE; the outer loop keeps its carried token because it has a trailing
use.

### Dump labels

`holdrule` prints the loop decision:

```text
pointofuse->op
    acquire moved to the first use; no loop token

passthrough-drop
    CHILD_OWNS; the parent returns no token because its final child returns none

gated(trailing-use)
    CARRIED; an access, release, or child flow follows the final token

gated(result-consumed)
    CARRIED; the nested final node needs an acquire after the loop

gated
    CARRIED; another check failed
```

Suffixes add details to any mode:

```text
:nestedExit          final token comes from a child region
:postLoopAcquire     an acquire was added after the loop
:entryBridge         a release was added to supply the first body acquire
```

## Backing copies

`computeBackingPlan` chooses the number of copies. `buffer.copy`, when
present, is authoritative: `@fused_alias_depth_two` (worked below in
[Alias handoffs](#non-circular-alias-handoffs))
carries it on its allocation —

```text
%m0 = ttg.local_alloc {buffer.copy = 2, buffer.id = 500}
```

— and its dump closes with that number taken straight from it:

```text
BACKING: numCopies=2
```

Without `buffer.copy`, `numCopies` starts at 1 and nothing lowers it. Two
rules can change what the later analyses see.

First, synchronized TMEM may receive two copies, only on the default NVWS
path (see [Two NVWS paths](../nvws-aws-overview.md#two-nvws-paths));
Meta-NVWS adds no TMEM copies of its own. The decision is a trace over the
group's MMA users, run here on the
`@root_entry_accumulator_adopts_without_semaphore_handoff` group in
`test/NVWS/insert_semas_root_entry_tmem.mlir`.
Follow its one MMA user, `ttng.tc_gen5_mma {2}`: it sits directly in the
WS `scf.for`; the `128x128xf32` accumulator is not read-modify-written in
that loop; accumulator multibuffering is structurally possible; the loop
carries no disallow-multibuffer flag; and two copies
(`2 * 128 * 128 = 32768` cells) fit TMEM's 128x512 cells alongside the
TMEM blocks already planned — every check passes, and the group's dump
closes with the line shown in its figure:

```text
BACKING: numCopies=2
```

The checklist that trace ran (`isMultiBufferedGroup`), for every MMA user of
the group's allocations whose immediate parent is an `scf.for`: the
accumulator is not read-modify-written in that loop
(`hasAccReadModifyWrite`), accumulator multibuffering is structurally
possible (`isAccMultibufferingPossible`), the enclosing WS loop does not
carry the disallow-multibuffer flag, and `canDoubleBufferAcc` — two copies
of the `blockM x blockN` accumulator still fit TMEM's 128x512 cells
alongside the TMEM blocks already planned, and the op is not a scaled MMA
with `blockN` 256.

Second, a local backing written by a TMA load records the number of semaphore
copies that `LowerSemaphore` will give it (`g.numSemaphoreCopies = max(1,
lowerSemaphoreNumStages)`; see the
[pass order](../nvws-aws-overview.md#pass-order)), so the loop-carried
dependency analysis below sees the copies that will actually exist. This is a
separate field next to `numCopies`, and the dump does not print it —
`test/NVWS/insert_semas.mlir` `@local_release_after_mma` (its `buffer.id
= 102` group) is this shape, and its `BACKING` line still reads
`numCopies=1`:

```text
|- W m0  nvws.descriptor_load {0}
|- r  S0  {0} [tma_load]        <- the TMA-load release that triggers the rule
   ...
BACKING: numCopies=1            <- numCopies is unchanged;
                                   numSemaphoreCopies is not a dump field
```

## Pipeline schedule

InsertSemas runs after the loop schedule is chosen and before software
pipeline expansion. It adds acquire, buffer, and release operations to an
already scheduled loop. Before EMIT-IR, SYNC-DAG assigns schedules to the new
operations and, when required, raises `loop.cluster` on existing operations so
every handoff remains ordered after expansion. It never changes `loop.stage`.

### Minimal pipeliner model

`loop.stage` controls which logical loop iterations overlap. For example, a
stage-1 operation from iteration `i` meets a stage-0 operation from iteration
`i+1` in one expanded loop body:

```text
before expansion

iteration i:       W(i) [loop.stage 0]  ...  R(i) [loop.stage 1]
iteration i+1:     W(i+1) [loop.stage 0] ... R(i+1) [loop.stage 1]

after expansion: one loop body contains

                   W(i+1) [loop.stage 0]
                   R(i)   [loop.stage 1]
```

The schedule builder orders lower `loop.cluster` values first, then preserves
block order within one cluster, before passing the result to PipelineExpander.
A loop-carried dependency's `distance` is the number of logical iterations
from its producer to its consumer. Those three facts are the complete
pipeliner model needed here.

The preexisting schedule has no SSA use from `R(i)` to `W(i+1)`. SYNC-DAG may
nevertheless require that ordering because `W(i+1)` reuses the buffer copy
that `R(i)` is finishing:

```text
required handoff:  R(i) -> r S  ...  a S -> W(i+1)

local inheritance can produce this expanded order:

cluster 1:  a S -> W(i+1)
cluster 2:  R(i) -> r S          <- the required producer is later
```

Copying the schedule of the access immediately after an acquire to that
acquire, and the schedule of the access immediately before a release to that
release, is not enough.

### Example: one-copy loop-closing handoff

`test/NVWS/insert_semas_recurrence_schedule.mlir`
`@one_slot_recurrence` makes the failure concrete. `EMPTY` protects the next
write, and `FULL` protects the following read:

```text
Each (s, c) is (loop.stage, loop.cluster).

final-read(i)      {1}  (1, 2)
r EMPTY(i)         {1}  (1, 2)
a EMPTY(i+1)       {3}  (0, 1)
W(i+1)             {3}  (0, 1)
r FULL(i+1)        {3}  (0, 1)
a FULL(i+1)        {1}  (0, 1)
first-read(i+1)    {1}  (0, 1)
```

Here `first-read(i+1)` is the first read of the backing copy in logical
iteration `i+1`; `final-read(i)` is the last read of that same copy in the
preceding logical iteration `i`. Pipeline expansion puts the stage-0 accesses
`W(i+1)` and `first-read(i+1)` in the same loop body as the stage-1 access
`final-read(i)`. Copying the adjacent access schedules produces the following
top-to-bottom sequence for each owner:

```text
WRONG

owner {3}, top to bottom:

a EMPTY(i+1)       {3}  (0, 1)  BLOCKED: waits for r EMPTY(i) {1}
W(i+1)             {3}  (0, 1)  not reached
r FULL(i+1)        {3}  (0, 1)  not reached

owner {1}, top to bottom:

a FULL(i+1)        {1}  (0, 1)  BLOCKED: waits for r FULL(i+1) {3}
first-read(i+1)    {1}  (0, 1)  not reached
final-read(i)      {1}  (1, 2)  not reached
r EMPTY(i)         {1}  (1, 2)  not reached
```

Owner `{3}` blocks at `a EMPTY(i+1)`, waiting for `r EMPTY(i)`. Owner `{1}`
blocks at `a FULL(i+1)`, waiting for `r FULL(i+1)`. Neither owner can reach the
release needed by the other.

SYNC-DAG must put `final-read(i) -> r EMPTY(i)` before the next write. It keeps
the `loop.stage` values fixed and raises `W(i+1)`, `first-read(i+1)`, and their
same-body SSA users from `loop.cluster 1` to `loop.cluster 3`. The adjacent
acquires and releases receive those corrected schedules:

```text
CORRECT

Each (s, c) is (loop.stage, loop.cluster).

final-read(i)      {1}  (1, 2)
r EMPTY(i)         {1}  (1, 2)
a EMPTY(i+1)       {3}  (0, 3)  <- changed from (0, 1)
W(i+1)             {3}  (0, 3)  <- changed from (0, 1)
r FULL(i+1)        {3}  (0, 3)  <- changed from (0, 1)
a FULL(i+1)        {1}  (0, 3)  <- changed from (0, 1)
first-read(i+1)    {1}  (0, 3)  <- changed from (0, 1)
```

The same corrected schedule grouped by owner:

```text
CORRECT — by owner, top to bottom

owner {3}:

a EMPTY(i+1)       {3}  (0, 3)  waits for r EMPTY(i) {1}
W(i+1)             {3}  (0, 3)
r FULL(i+1)        {3}  (0, 3)  releases owner {1}

owner {1}:

final-read(i)      {1}  (1, 2)
r EMPTY(i)         {1}  (1, 2)  releases owner {3}
a FULL(i+1)        {1}  (0, 3)  waits for r FULL(i+1) {3}
first-read(i+1)    {1}  (0, 3)
```

Now `r EMPTY(i)` unblocks owner `{3}`, which writes and executes `r FULL(i+1)`;
that release then unblocks owner `{1}`.

With one physical copy, changing `loop.cluster` from 1 to 3 preserves the
original sequential reuse order, `final-read(i) -> W(i+1)`, after owner
partitioning and pipeline expansion. It serializes reuse of this backing copy,
not the entire loop or all work performed by the owners.

This repair is conditional on the number of physical copies, not
unconditional. `computeLoopCarriedDistance` uses `numSemaphoreCopies` — one
here, from `buffer.copy = 1` — to find the first future iteration that reuses
the released copy. With one copy, `W(i+1)` reuses it and the distance is 1, so
the cluster repair above is required. With `buffer.copy = 2`, `W(i+1)` uses
the other copy and `W(i+2)` is the first reuse; the distance is 2, the existing
schedule is valid, and no cluster changes.

### Finalizing one handoff

Protocol construction and region-flow placement have already decided where the
release and acquire go. An ordinary handoff now has one of these two shapes:

```text
source access completion -> r ... a -> destination access
source access completion -> r ... a -> scheduled-loop EXIT
```

In both shapes, the release copies the source completion's schedule. The
acquire copies the finalized destination-access schedule or the destination
owner's schedule at `EXIT`. SYNC-DAG determines that destination schedule
before filling in the release and acquire schedules.

The example above contains an `EMPTY` handoff and a `FULL` handoff. Keep only
`EMPTY`. Here the acquire was placed immediately before `W(i+1)`. The two
access schedules are known; the release and acquire schedules are not yet
assigned:

```text
Each (s, c) is (loop.stage, loop.cluster).

final-read(i)      {1}  (1, 2)
r EMPTY(i)         {1}  (?, ?)
a EMPTY(i+1)       {3}  (?, ?)
W(i+1)             {3}  (0, 1)
```

First determine when the released physical copy is reused. With one copy,
`W(i+1)` is the first reuse, so the distance is 1. At that distance,
stage-1 `final-read(i)` and stage-0 `W(i+1)` land in the same expanded loop
body. Their clusters are backwards: cluster 1 places the write before the
cluster-2 final read.

SYNC-DAG therefore raises the stage-0 chain to cluster 3. Only after that
repair does it fill in the two unknown schedules: the release copies
`final-read(i)`'s physical-completion schedule, and the acquire copies the
finalized schedule of `W(i+1)`:

```text
final-read(i)      {1}  (1, 2)
r EMPTY(i)         {1}  (1, 2)
a EMPTY(i+1)       {3}  (0, 3)
W(i+1)             {3}  (0, 3)
```

When the acquire is placed before an access, an in-iteration handoff has
distance 0. For a loop-closing handoff, `computeLoopCarriedDistance` finds the
first future iteration in which that access reuses the physical copy released
by the source access.

A handoff cannot be classified from those two stages alone. Its owners execute
independently, so an acquire that appears early may block while its producer
continues to the release. Let `offset[P]` be the whole-iteration delay of owner
`P`. For a release by `P` at stage `before`, followed at loop distance
`distance` by an acquire owned by `Q` at stage `after`, correctness requires:

```text
offset[Q] >= offset[P] + before - after - distance
```

The term on the right after `offset[P]` is the handoff's required owner delay.
A positive delay on one handoff is legal backpressure. The owner offsets are
infeasible only when the complete owner graph contains a cycle whose required
delays sum to a positive value: every owner in that cycle would have to run
later than itself. SYNC-DAG solves all handoff constraints for the scheduled
loop together and rejects that owner-offset failure. A feasible zero-delay
cycle still proceeds to cluster legalization, which rejects it if its
same-wave operation order is itself cyclic.

Per-partition stage normalization does not change this proof. Subtracting a
constant stage from every operation of one owner is the same as adding that
constant to its `offset`; owner offsets cancel around every cycle.

After solving the offsets, a handoff either has iteration slack or is tight:

```text
adjusted delay < 0
    an earlier iteration supplies the token; normally no cluster change

adjusted delay = 0 and the edge lies on a tight owner cycle
    the handoff is in the same expanded wave; repair loop.cluster

positive-delay owner cycle
    no owner offsets can satisfy the schedule; compilation fails
```

`legalizeLoopSchedule` orders every tight-cycle handoff, together with its
same-body SSA users. This includes the direct stage equality in the running
example and a retimed equality such as delays `+1` and `-1` around the same
owner cycle. A directly zero-delay handoff also retains the existing local
cluster repair when it is not part of a cycle, including when another owner
constraint gives that edge adjusted iteration slack. The repair never changes
`loop.stage`. For an asynchronous access, the release uses the schedule of its
physical completion. A semaphore buffer uses the schedule of the access it
serves. EMIT-IR only transcribes these decisions.

The representative cycles are:

```text
two-copy running example: EMPTY -1, FULL 0, cycle -1
    -> iteration slack; unchanged

one-copy running example: EMPTY 0, FULL 0, cycle 0
    -> tight cycle; repair loop.cluster

errors twin: EMPTY +1, FULL 0, cycle +1
    -> impossible owner delays; compilation fails

four-slot/two-advance ring: EMPTY +1, FULL -3, cycle -2
    -> legal cross-partition backpressure; unchanged
```

The last case is the shape in
`test/NVWS/insert_semas_recurrence_owner_cycle.mlir`: the positive `EMPTY`
edge is not itself an error because the reverse `FULL` edge leaves two
iterations of credit in the complete cycle.

If the acquire remains at the bottom of iteration `i`, its destination is the
scheduled loop's `EXIT`, not `W(i+1)`. Its result is already carried to the next
iteration as a loop iter-arg, so the access-distance comparison above does not
apply. The acquire instead uses the last cluster at which its owner performs
work at the acquire's stage.

For example, add a later access from another buffer group owned by `{3}`:

```text
Each (s, c) is (loop.stage, loop.cluster).

owner {1}:

final-read(i)          {1}  (1, 2)
r EMPTY(i)             {1}  (1, 2)

owner {3}, top to bottom:

W(i)                   {3}  (0, 1)
W other(i)             {3}  (0, 4)
a EMPTY(i+1)           {3}  (0, 4)
EXIT
```

Cluster 4 is the smallest valid `EXIT` position for owner `{3}` at stage 0.
Using cluster 1 would move the blocking acquire before `W other(i)`, preventing
the same owner from reaching that operation if the acquire waits. Within
cluster 4, block order keeps `W other(i)` before the acquire. The release's
cluster need not precede the acquire's cluster; the semaphore supplies that
cross-owner ordering. The `EXIT` position only has to keep the acquire after
all work of its own owner at that stage. Copy count is irrelevant here: it has
already selected which physical copy the acquire addresses.

A synthetic acquire immediately after a nested loop has no destination access
and is not inside the child loop at its `EXIT`. It uses the last schedule
recorded for its owner on the parent chain, not the schedule of an unrelated
access that happens to follow it:

```text
last parent schedule   {3}  (0, 4)
nested loop
a EMPTY(i+1)           {3}  (0, 4)
next access            {0}  (...)   unrelated
```

Root entry acquires remain unscheduled.

## Authored buffer-stage offsets

`loop.stage` and `loop.cluster` determine when the software pipeliner executes
an operation. A semaphore node's `stage-offset` instead specifies a signed
shift from the current stage of its backing buffer. The shift is applied modulo
`buffer.copy`: `0` selects the current stage, `-1` the preceding stage, and
`+1` the following stage.

`analyzeSyncSchedule` runs one physical-stage analysis for circular
groups and non-circular aliased backings. It replays the fresh-write cursor
that ASP will use: a write records the cursor ordinal as the group's current
value, and a read uses the latest ordinal recorded for its group. The analysis
does not distinguish SMEM from TMEM. Circular metadata and alias grouping only
determine how the allocations are represented and where the computed offsets
are attached.

Circular members are separate groups, while non-circular aliases are names in
one group. For an access or semaphore node:

```text
stage-offset = required value ordinal - current cursor ordinal
```

ASP applies that displacement modulo the backing's copy count. The two cases
below use the same analysis and differ only in how their offsets are attached
to the generated protocol.

### Circular groups

For circular groups sharing one physical `buffer.id`, SYNC-DAG first validates
the circular metadata: common type and `buffer.copy`, unique `buffer.start`
values, and producer order. It then applies the shared analysis above. The
resulting stage offset is stored on the access node and its adjacent
acquire/release nodes before any IR is emitted.

`test/NVWS/insert_semas_circular_smem.mlir` `@circular_tutorial_1_1_to_2_2`
— K and V share one two-copy ring (`buffer.start` 0 and 1); each circular
member is its own group, and the contract with [EMIT-IR](emit-ir.md) is that
it folds these groups onto the one physical ring:

```text
store K:   counter 0 -> 1    K produced at ordinal 0    offset  0
store V:   counter 1 -> 2    V produced at ordinal 1    offset  0
load  K:   counter stays 2   K's latest = 1 ago         offset -1
load  V:   counter stays 2   V's latest = current       offset  0

K group (the access nodes carry no stage-offset):

|- scf.for (WS, tag=1) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S1  {1}  stage-offset=0
|  |- W m0  ttg.local_store {1}
|  |- r  S0  {1} [none]  stage-offset=0
|  |- a  S0  {2}  stage-offset=-1         <- K's consumer, bracketing the load
|  |- R m0  ttg.local_load {2}
|  |- r  S1  {2} [none]  stage-offset=-1  <- K's consumer, bracketing the load
|  |- EXIT ... yield{native}

V group:

|- scf.for (WS, tag=1) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S1  {1}  stage-offset=0
|  |- W m0  ttg.local_store {1}
|  |- r  S0  {1} [none]  stage-offset=0
|  |- a  S0  {2}  stage-offset=0
|  |- R m0  ttg.local_load {2}
|  |- r  S1  {2} [none]  stage-offset=0
|  |- EXIT ... yield{native}
```

`S1` is created initially released, so iteration zero's `a S1 {1}` succeeds
before any release has run (`S1{count=1 entry inherit={@1.1}}`). K's consumer
must address the copy produced *before* V advanced the ring — the `-1` on
exactly K's consumer nodes (`a S0 {2}` and `r S1 {2}`).

### Non-circular alias handoffs

Non-circular alias handoffs reuse the fresh-write stage replay described above.
Their release shifts are derived from that replay. Planner-authored
`buffer.copy` supplies one shared physical stage domain for every member of a
`buffer.id` group; the member memdescs may be different views of that backing.
Separately staged semaphores without a planner-authored multi-copy backing keep
the narrower exact-alias requirement. This handling applies uniformly to SMEM
and TMEM.

The motivating example happens to use SMEM: a split epilogue such as the `dV`
store in backward attention. Two logical allocations have the same
`buffer.id`, the same shape, and no distinct `buffer.offset`, so both name the
full extent of one two-copy backing:

```mlir
%dv0_smem = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}
    : memdesc<128x32xf16>
%dv1_smem = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}
    : memdesc<128x32xf16>

scf.for {
  ttg.local_store %dv0, %dv0_smem {partition = 4}
  %dv0_read = ttg.local_load %dv0_smem {partition = 2}
  tt.descriptor_store ..., %dv0_read {partition = 2}

  ttg.local_store %dv1, %dv1_smem {partition = 4}
  %dv1_read = ttg.local_load %dv1_smem {partition = 2}
  tt.descriptor_store ..., %dv1_read {partition = 2}
}
```

Because both allocation names touch the same piece, the ordinary read and
write rules produce one chain:

```text
Sentry initially released
          |
          v
   W dv0(i) {4}
          | Sfull0
          v
   R dv0(i) {2}
          | Shandoff
          v
   W dv1(i) {4}
          | Sfull1
          v
   R dv1(i) {2}
          | Sentry
          v
 W dv0(i+1) {4}
```

`buildEdgesAndSemas` creates a semaphore for each destination node in this
chain. It does not fold `Sentry` with `Shandoff`, or `Sfull0` with `Sfull1`.
The reduced IR after InsertSemas and before ASP is therefore:

```mlir
%base = ttg.local_alloc {buffer.id = 5, buffer.copy = 2}
    : memdesc<2x128x32xf16>

%Sentry   = nvws.semaphore.create %base, %base true
%Sfull0   = nvws.semaphore.create %base, %base false
%Shandoff = nvws.semaphore.create %base, %base false
%Sfull1   = nvws.semaphore.create %base, %base false

scf.for {
  %t0 = nvws.semaphore.acquire %Sentry[0] {partition = 4}
  %b0:2 = nvws.semaphore.buffer %Sentry[0], %t0
  ttg.local_store %dv0, %b0#0 {partition = 4}
  nvws.semaphore.release %Sfull0[0], %t0 {partition = 4}

  %t1 = nvws.semaphore.acquire %Sfull0[0] {partition = 2}
  %b1:2 = nvws.semaphore.buffer %Sfull0[0], %t1
  %dv0_read = ttg.local_load %b1#0 {partition = 2}
  tt.descriptor_store ..., %dv0_read {partition = 2}
  nvws.semaphore.release %Shandoff[1], %t1 {partition = 2} // release next buffer stage

  %t2 = nvws.semaphore.acquire %Shandoff[0] {partition = 4}
  %b2:2 = nvws.semaphore.buffer %Shandoff[0], %t2
  ttg.local_store %dv1, %b2#1 {partition = 4}
  nvws.semaphore.release %Sfull1[0], %t2 {partition = 4}

  %t3 = nvws.semaphore.acquire %Sfull1[0] {partition = 2}
  %b3:2 = nvws.semaphore.buffer %Sfull1[0], %t3
  %dv1_read = ttg.local_load %b3#1 {partition = 2}
  tt.descriptor_store ..., %dv1_read {partition = 2}
  nvws.semaphore.release %Sentry[1], %t3 {partition = 2} // release next buffer stage
}
```

Here the bracketed values are stage offsets, not final stage numbers. ASP
assigns the first store and read a stage `s`. The second store is another
fresh write, so ASP advances it and its read to `(s + 1) mod 2`.

The same-stage edges need offset zero:

```text
W dv0 at s       -> R dv0 at s
W dv1 at s + 1   -> R dv1 at s + 1
```

The other two edges cross stages:

```text
R dv0 at s       -> W dv1 at s + 1
R dv1 at s + 1   -> W dv0(i+1) at s
```

Without an authored offset, a release uses its source access's stage. The
first crossing would therefore release `Shandoff[s]` while the acquire before
`W dv1` waits on `Shandoff[s + 1]`. `Shandoff` was created `false`, so that
acquire would never see the release. The loop-closing edge has the same
problem in the opposite direction after the two-copy wrap.

The shared physical-stage replay records `stage-offset=1` on those two
releases. ASP then materializes:

```text
release Shandoff[(s + 1) mod 2] -> acquire Shandoff[(s + 1) mod 2]
release Sentry[s]               -> acquire Sentry[s] in iteration i+1
```

Thus the offsets are required by the current separate-semaphore protocol:
each release must address the stage used by the acquire that the SYNC-DAG
paired with it. This case has no `buffer.circular`; `buffer.copy = 2` alone
supplies the two stages.

`test/NVWS/insert_semas_fused_alias_handoff.mlir`
`@tmem_fused_alias_depth_two` applies the same calculation to two
non-circular TMEM aliases with planner-authored `buffer.copy = 2`. The two
crossing releases likewise receive `stage-offset=1`.

## Code map

[`InsertSemasSyncDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp):

- `ChainWalker::run`, `applyTouch`, `VersionSource`, `ActiveUse`,
  `PieceState`, `Tokens`, and `canReuseTokenForPiece`
- `collectEdges`, `reduceStraightEdges`, `reduceLoopCloses`, `reduceChain`,
  `reduceEdges`, and `buildEdgesAndSemas`
- `insertEntryAcquires`
- `summarizeChainBoundary`, `summarizeRegionFlow`, `buildRegionFlows`, and
  `pruneDeadIfFlows`
- `findInputAcquire`, `matchDemandPrefix`, `planLoopFlow`,
  `lowerPointOfUse`, `planRegionFlows`, and `verifyPointOfUseFlow`
- `CapabilityRef`, `SemaTransfer`, and `RegionFlow` are defined in
  `InsertSemas.h`
- `computeBackingPlan`
- `assignCircularStageOffsets`, `assignAliasedHandoffStageOffsets`,
  `PhysicalSchedules`, `replaySlots`, `computeSlotSchedule`,
  `computeLoopCarriedDistance`,
  `addSyncScheduleEdges`, `legalizeLoopSchedule`, `analyzeSyncSchedule`,
  `scheduleAtOwnerBoundary`, `assignSyncScheduleChain`, and
  `finalizeSyncSchedule`
- `buildSyncDag`
- the DAG dump used throughout: `NVWS_INSERT_SEMA_DUMP_DAG=1`
  (`dumpDagTree`)
