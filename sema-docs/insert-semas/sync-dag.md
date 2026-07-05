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
    - [Repeats from one sender (`mergeEdges`)](#2-repeats-from-one-sender-mergeedges)
  - [One destination node, one semaphore](#one-destination-node-one-semaphore)
  - [Composition: why loop entry and loop recurrence share one semaphore](#composition-why-loop-entry-and-loop-recurrence-share-one-semaphore)
- [Crossings and holds](#crossings-and-holds)
  - [Why this machinery exists](#why-this-machinery-exists)
  - [The decision, per region](#the-decision-per-region)
  - [Future investigation: replace a consumed loop result](#future-investigation-replace-a-consumed-loop-result)
  - [Composition: apply the hold rule from inner to outer](#composition-apply-the-hold-rule-from-inner-to-outer)
- [Backing copies](#backing-copies)
- [Pipeline schedule](#pipeline-schedule)
  - [Minimal pipeliner model](#minimal-pipeliner-model)
  - [Example: one-copy loop-closing handoff](#example-one-copy-loop-closing-handoff)
  - [Finalizing one handoff](#finalizing-one-handoff)
- [Authored buffer-stage offsets](#authored-buffer-stage-offsets)
  - [Circular groups](#circular-groups)
  - [Non-circular exact-alias handoffs](#non-circular-exact-alias-handoffs)
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
5. the **region token policy** decides how one token crosses a `for` or `if`.

`Owner`, `Piece`, and `Effect` are attributes of those objects. Acquires,
releases, and semaphores are the generated representation of edges, not a
second correctness model.

```text
input IR ─► ACCESS-DAG ─► OWNER-DAG ─► SYNC-DAG ─► EMIT-IR ─► output IR
            memory facts   owners      edges, semaphores,   render
                                       token holds, schedule
```

The whole step is four moves; this page is those moves in order, each with
a worked example:

```text
1. walk the accesses in program order; every "this must wait for that"
   becomes an edge between two concrete DAG nodes
2. delete the redundant edges
3. the edges converging on one DAG node become one acquire, pending count =
   their number; each edge's tail becomes a release
4. loops: seed iteration zero, decide how tokens cross for/if boundaries
   (holds), choose the number of backing copies, then extend the pipeline
   schedule with the semaphore dependencies
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
hold          acquire -> accesses -> closing release, all by one owner: the
              ordinary span protected by that token; read spans may overlap,
              and an owner token may be reused later when SYNC-DAG proves it
regain        the acquire at the bottom of a loop body through which the
              body re-acquires the group's semaphore for the next iteration
              (`Hold::regain`)
chain         the group's node sequence of one block, in program order; a
              region node holds child chains
```

Dump notation:

```text
|- scf.for (WS, tag=0)      loop node; (WS, tag=N) = warp-specialized loop
|- W m0  ttg.local_store {1}   access node: W(rite)/R(ead), member, op, owner
|- a  S1  {0}               acquire of semaphore S1 by partition 0
|- a  S0(2)  {0}            acquire with pending count 2 (waits for 2 releases)
|- r  S0  {1} [tma_load]    release by partition 1; [..] = completion kind,
                            which selects the release's lowering (arrive,
                            MMA commit, or TMA completion path)
|- r  S0(2)  {1}            release standing in for 2 waits (arrive_count 2)
|- a  S3  root  ; entry     unpartitioned entry acquire, spliced before the
                            group's first placement node — a top-level node of
                            the group's chain that involves the group: an
                            access, or a region containing one (immediately
                            above the loop when the loop is that first node)
pieces{P0:W:{0}}            region node: per-piece merged effect and owner
thread{{0}}                 region node: this region has a crossing; a surviving
                            `if` crossing returns a token, while a loop's hold
                            decides whether it has a token iter-arg and result
holdrule{...}               loop node: the crossing's hold outcome (its values
                            are explained in "Crossings and holds")
yield{X}                    EXIT node: what this chain returns for the
                            crossing — a S<n> = that acquire's token;
                            native = no token crosses (protocol lives
                            inside); drop = this loop has no token result
                            because its final nested loop returns none; pass =
                            this branch has no acquire or nested region with a
                            crossing, so it returns the token available before
                            the `if`;
                            scf.for/scf.if =
                            the actual yield operand is the token result of
                            that nested scf.for/scf.if
stage-offset=N              stage offset assigned to this protocol node
S<n> / E<n>                 semaphore names; E<n> = dedicated entry semaphore
SEMAS: S1{count=1 entry inherit={@0.0}}   per-semaphore summary; entry =
                            created initially released; inherit={...} = the
                            owner recorded on that initially released state
                            (its `entryTokenOwner`) — read by the hold decision
                            (as an entry feed's owner, buildUniformHold) and
                            by the verifier that checks token reuse;
                            EMIT-IR also uses it as
                            the owner of an unpartitioned entry acquire when
                            recording the acquired token — which is one
                            of three: the owner of the group's
                            first real access (insertEntryAcquires, both of
                            its paths), the retargeted EXIT-handoff
                            acquire's owner (buildEdgesAndSemas), or the
                            hold owner (applyHoldRulePlacement)
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
(`walkChain`). At each access it first applies two memory rules:

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
  %t2 = a S1 {0}                 ; regain: acquires the NEXT iteration's token
  yield %t2                      ; carried out through the iter-arg
}
```

The pass then decided this loop out of the carried shape — the pre-loop
acquire is gone and the regain sits at the store instead (the node's
`holdrule{pointofuse->...}` label; the decision itself is
[Crossings and holds](#crossings-and-holds)). The resulting SYNC-DAG:

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

After all touched pieces have run through `applyTouch`, `walkChain` chooses
how the access obtains a token:

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

The model, in one line: *after redundant edges are deleted, the edges
converging on one node become one acquire with pending count n — n being
their number — and each edge's tail becomes a release.* That is the
newly-created-semaphore case; a handoff into a loop node can instead
reuse the loop's regain semaphore and inherit its count — the
composition case below.

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
`verifyTokenLocality` checks that the named token exists and applies the same
region token resets. EMIT-IR only renders these marks; it does not decide that
another node is eligible.

Before conversion, redundant edges are deleted — in two forms, in this
order (`buildEdgesAndSemas`): ordering already implied by the other edges
(`reduceEdges`), then repeats from one sender (`mergeEdges`). The deletion
decision is a trace over the kept edges; it runs on a real loop first,
and the complete condition lists follow the two worked examples as fine
print.

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
next iteration. Hold placement puts its acquire immediately before
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
for, decided in [Crossings and holds](#crossings-and-holds) below.
Contrast `@fanout_not_reduced` above, `holdrule{gated}`: its `S2`
stays a pre-loop root entry, the carried default of the same section.

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
P2 at once, counted once by `mergeEdges`). At `EXIT` two closes are
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
the iteration boundary. `sweepTraversalClosure` simulates that wrap-around;
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

Every deletion is then re-proved against the kept DAG alone. A failed
straight-line proof reports `transitive-reduction closure violation`; a
failed wrap-around proof reports `traversal-closure violation: dropped close
not implied`.

#### 2. Repeats from one sender (`mergeEdges`)

`reduceEdges` has finished. `mergeEdges` now handles two kept edges that have
the same sending owner and destination but leave from different nodes.

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
chain's first partition-owned access. The raw DAG entering `mergeEdges` is:

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
release after `R m1 {1}` is also after `R m0 {1}`. `mergeEdges` combines the
two closes at that later node and unions their pieces and completion payloads:

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

Only one edge now enters the destination, so conversion creates one semaphore
with pending count 1. In the unrolled edge-only protocol DAG, S0 represents
`e1` and S1 represents merged `e2+e3`; the initial entry is omitted:

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

### One destination node, one semaphore

What remains is grouped by destination node and destination owner: one
destination, one semaphore, one acquire; its source owners are the
releases, and the pending count is how many there are —
`@fanout_not_reduced`'s `a S2(2)` above, where `e3` and `e4` converge. A
release's `arrive_count` is raised above one only when one release must
stand in for several, keeping the total equal to the pending count. The next
section shows the case that requires it: a loop-entry edge reuses a recurrence
semaphore whose pending count came from a larger fan-in.

### Composition: why loop entry and loop recurrence share one semaphore

This section explains why `e1`, `e6`, and `e7` use the same semaphore in the
following example, and how its pending count affects `e1`. `e1` feeds
iteration 0; `e6` and `e7` feed later iterations. The `e6`/`e7` fan-in gives
`S0` pending count 2. Because `e1` reuses `S0`, its one release supplies two
arrivals: `r S0(2)`.

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
3. Every `a S0(2)` needs two arrivals. `e1` has only one source node, so its
   one release supplies both arrivals: `r S0(2) {3}`.

The `2` on `r S0(2)` is an arrival count. It does not mean that the walk
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
|  |- r  S0(2)  {3} [none]        ; e1 tail: one release, two arrivals
|  |- a  S0(2)  {2}               ; e1 head: initial inner-loop token
|  |- scf.for ... holdrule{gated(result-consumed)}
|  |  |- r  S1  {2} [none]        ; e3 tail: ENTER has no op, so this starts the child chain
|  |  |- R m0  ttg.local_load {2}
|  |  |- r  S2  {2} [none]        ; e4 tail
|  |  |- a  S1  {1}               ; e3 head
|  |  |- R m0  ttg.local_load {1}
|  |  |- a  S2  {1}               ; e4 head
|  |  |- W m0  ttg.local_store {1}
|  |  |- r  S3  {1} [none]        ; e5 tail
|  |  |- r  S0  {1} [none]        ; e6 tail: one arrival
|  |  |- a  S3  {0}               ; e5 head
|  |  |- R m0  ttg.local_load {0}
|  |  |- r  S0  {0} [none]        ; e7 tail: one arrival
|  |  |- a  S0(2)  {2}            ; e6/e7 head: next inner-loop token
|  |  |- EXIT ... yield{a S0}
|  |- r  S4  {2} [none]           ; e2 tail
|  |- a  S4  {3}                  ; e2 head
|  |- EXIT ... yield{a S4}
SEMAS: S0{count=2} S1{count=1} S2{count=1} S3{count=1} S4{count=1 entry inherit={@0.3}}
```

The parent and child remain separate DAG levels, but both levels use `S0`.
These diagrams include the release and acquire nodes added during conversion.
In the parent DAG, the single `r S0(2) {3}` supplies both arrivals needed by
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

In the child DAG, `r S0 {1}` and `r S0 {0}` each supply one arrival to the
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

## Crossings and holds

### Why this machinery exists

When a token must cross a loop, the unimproved way through is the carried
shape — this loop, complete:

```text
%t0 = a S root                    ; entry, seeds iteration 0
%tN = for (%t = %t0) {            ; token in through the iter-arg
  W buf [%t] {0}                  ; the body works under the carried token
  r  S %t {0}
  %t1 = a S {0}                   ; regain: acquires the NEXT iteration's token
  yield %t1                       ; carries it out
}                                 ; %tN unused below
```

Always correct—but potentially suboptimal, because the token may be held
across more of the loop than necessary. We therefore try to shorten that
hold by moving the acquire that produces `%t1` to the next iteration's first
access. For this to be legal, `%t1` must be used only to reach that access.
We verify this by tracing every use of `%t1`, including its continuation as
the loop result `%tN`:

```text
W buf [%t] {0}       no  — runs under %t, the previous token
r  S %t {0}          no  — consumes %t
nodes after the regain, before the yield:   none
yield %t1            yes — but only to deliver it to the NEXT
                           iteration's first access
%tN, after the loop:                        unused
```

`%t1` has exactly one real customer: the next iteration's first access. A
token whose only job is to be delivered somewhere can be acquired *there*
instead — that is the move:

```text
for ... {                         ; no token iter-arg
  %t = a S {0}                    ; the same acquire, at the first access; S is
                                  ;   created initially released, seeds iter 0
  W buf [%t] {0}
  r  S %t {0}                     ; released inside
}
```

Two important token uses block the move. The first is an in-body use of
`%t1`:

```text
%t1 = a S {0}
R m0 [%t1] {0}                  ; still needs %t1 here: cannot move it
yield %t1
```

Using the loop result also blocks it:

```text
%tN = for ... {
  ...
  %t1 = a S {0}
  yield %t1
}
R m0 [%tN] root                 ; still needs the returned token
```

Direct POINT_OF_USE requires the acquire feeding loop entry and the acquire at
loop re-entry to use the same semaphore. In one supported nested-composition
shape, a post-loop acquire and bridge first normalize a different incoming
semaphore, so the outside code no longer uses `%tN`.

`buildUniformHold` records the result before emission:

```text
move succeeded                           holdrule{pointofuse->op}
trailing use                             holdrule{gated(trailing-use)}
same-semaphore result used afterward     holdrule{gated(result-consumed)}
other eligibility failure                holdrule{gated}
```

A hold is one owner's accesses covered by one acquired token:

```text
%t = a S {0}                    ; acquire before the first access
W m0 [%t] {0}                   ; first access
R m1 [%t] {0}                   ; last access
r S %t {0}                      ; release after the last access
```

### The decision, per region

**Before the searches: does a token have to cross this region at all?**
If the body contains no acquire of the group and no nested region that
already has a crossing, no token passes through the boundary — there is no
crossing, and nothing below applies (a body holding only releases records
none).
One hold simply covers the whole loop: acquired before it, every
iteration's accesses run under that one token (its value dominates the
body), released after it. The dump line for such a loop carries no
`thread{}` and no `holdrule{}`. The accumulator loop of a plain matmul is
this shape (pseudo-IR; no lit test covers it):

```text
%t = a S0 {0}                   ; acquire, before the loop; S0 is created
for k {                         ;   initially released, seeding this wait
  W buf [%t] {0}                ; every iteration, under the same token
}                               ;   single owner, no protocol inside
r  S1 %t {0} [tc5mma]           ; release, after the loop
%u = a S1 root                  ; epilogue acquire, satisfied by that release
R  buf [%u] root                ; epilogue read
```

Otherwise a *crossing* is recorded: the region may need to return a token.

**For an `if`, only use after the `if` matters.** If nothing afterward uses
the token returned by the `if`, and no enclosing region needs it, the crossing
is removed and the `if` returns no token (`pruneDeadIfCrossings`). Otherwise,
the crossing remains. In `test/NVWS/insert_semas_raw_if_token.mlir`
`@raw_edge_token_carried_if`, the store after the `if` uses the returned token:

```text
|  |- scf.if pieces{P0:R:{0}} parts{0, 1} thread{{0}}
|  |  |- then
|  |  |  |- ENTER pieces{P0:R:{0}}
|  |  |  |- r  S0  {0} [none]
|  |  |  |- a  S0  {1}
|  |  |  |- R m0  ttng.tmem_load {1}
|  |  |  |- r  S1  {1} [none]
|  |  |  |- a  S1  {0}
|  |  |  |- EXIT pieces{P0:R:{0}} yield{a S1}
|  |  |- else
|  |  |  |- ENTER
|  |  |  |- EXIT yield{pass}      <- this branch returns the token that was
|  |  |                              available before the if
|  |- W m0  ttng.tmem_store {0}   <- the consumer: this is what keeps the
                                     crossing alive
```

`yield{pass}` means that this branch contains no acquire or nested region with
a crossing, so it returns the token already available before the `if`.

**For a loop, the decision checks two kinds of use of the regain's token**
(`buildUniformHold`). Structural checks can stop the decision earlier, and an
inside use stops it before the outside check. At a single level — judging this
loop's own nodes — the trace has two outcomes; nested regions are handled by
composition in the next subsection.

**Both searches empty → POINT_OF_USE.**
`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use`. What the default
would emit — carried, full protocol:

```text
%t0 = a S2 root                      ; entry, seeds iteration 0
for (%t = %t0) {
  W m0 [%t]  ttg.local_store {0}
  r  S0 %t  {0}
  %t1 = a S0 {1}
  R m0 [%t1] ttg.local_load {1}
  r  S1 %t1 {1}
  %t2 = a S1 {2}
  W m0 [%t2] use_smem {2}
  r  S2 %t2 {2}
  %t3 = a S2 {0}                     ; regain at the bottom
  yield %t3
}
```

Trace the regain's token `%t3`: between the regain and the yield —
nothing; the loop's result — unused. Both searches are empty and the move's
safety checks pass. The move deletes the regain, the yield's token,
and the iter-arg, and acquires directly before the first access instead;
the pass emitted:

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S2  {0}                 ; the moved acquire, at the first access
|  |- W m0  ttg.local_store {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]
|  |- a  S1  {2}
|  |- W m0  use_smem {2}
|  |- r  S2  {2} [none]          ; closing release, in the body
|  |- EXIT ... yield{native}     ; no token crosses the boundary
SEMAS: S0{count=1} S1{count=1} S2{count=1 entry inherit={@0.0}}
```

`S2` is created initially released so iteration zero's in-body acquire
succeeds.

**An inside use keeps the token iter-arg and result, as does any use of the
result outside the loop.** In one supported shape, the pass inserts an acquire
after the loop and the outside code uses that token instead. The loop result
then has no outside use. The two searches are:

- *inside the body*, between the regain and the yield: an access to the
  group's memory, a release of one of the group's semaphores, or a nested
  `for`/`if` with a crossing (`hasTrailingCompUse`). An access or release still
  uses the token at the bottom; any nested crossing is conservatively treated
  as a trailing use, so the dump prints `gated(trailing-use)`;
- *outside the loop*, through the result: something after the loop
  consumes the token the loop returns (`regionResultConsumedAfter`). If the
  feeding and regain semaphores are the same, the dump prints
  `gated(result-consumed)`. If they differ, the pass may insert the replacement
  acquire after the loop; if it cannot, the loop stays gated.

**Inside search fails — `trailing-use`.**
`test/NVWS/insert_semas_per_edge_tmem.mlir`
`@tmem_single_producer_multi_consumer_fanout` has one writer and two readers:

```text
%t0 = a S2(2) root                ; S2 is initially released
for (%t = %t0) {
  W buf [%t] {0}
  r  S0 %t {0}
  r  S1 %t {0}

  %r1 = a S0 {1}
  R buf [%r1] {1}
  r  S2 %r1 {1}

  %r2 = a S1 {2}
  R buf [%r2] {2}
  r  S2 %r2 {2}

  %t1 = a S2(2) {0}              ; waits for BOTH readers
  W buf [%t1] {0}                 ; trailing use
  yield %t1
}
```

Here `%t1` covers both the trailing `{0}` write and the first `{0}` write of
the next iteration. Removing the token iter-arg and result would require
keeping `a S2(2)` before the trailing write, then adding a release after that
write and another acquire at the next iteration's first write. Carrying `%t1`
across the loop is precisely what eliminates that same-owner release/acquire
pair. The POINT_OF_USE optimization does not add the pair back, so it cannot
remove the token carry in this shape.

**Outside search fails — `result-consumed`.**
The failure is easiest to see when the loop result hands the completed cycle
to a third owner:

```text
%e = a S0 {1}                    ; S0 is initially released
W buf [%e] {1}
r  S1 %e {1}
%t0 = a S1 {2}

%tN = for (%t = %t0) {
  R buf [%t] {2}
  r  S2 %t {2}

  %u = a S2 {1}
  W buf [%u] {1}
  r  S1 %u {1}

  %t1 = a S1 {2}                 ; waits for {1}'s write
  yield %t1
}

r  S3 %tN {2}                    ; hands the completed cycle to {3}
%v = a S3 {3}
R buf [%v] {3}
```

On every iteration, the bottom `a S1 {2}` waits for `{1}`'s write. Its yielded
token supplies the next iteration. After the final iteration, the returned
`%tN` is instead consumed by `r S3`: `{3}` cannot pass `a S3` until `{1}`'s
final write has completed.

This outside use is why the loop reports `gated(result-consumed)` and keeps
its token iter-arg and result. If that check were ignored, POINT_OF_USE would
remove the iter-arg and result and move the bottom `a S1` to the next
iteration's first `{2}` read. `r S3` would then have no token: a token acquired
inside the loop body cannot be used after the loop unless the loop returns it.
Moving `r S3` into the body would not preserve the program either. It would
release `S3` on every iteration, so `{3}` could consume the first release and
read `buf` while later iterations were still writing it. Adding a new
`a S1 {2}` before `r S3` could provide one token after the loop, but that would
be a different same-semaphore transformation; the `result-consumed` case
described here does not perform it.

### Future investigation: replace a consumed loop result

The same-semaphore `result-consumed` case may be able to use POINT_OF_USE if
the pass creates one acquire after the loop to replace the token it no longer
returns:

```text
for {
  %t = a S1 {2}                  ; point-of-use acquire
  R buf [%t] {2}
  r  S2 %t {2}

  %u = a S2 {1}
  W buf [%u] {1}
  r  S1 %u {1}
}

%tok = a S1 {2}                 ; waits for the final iteration's r S1
r  S3 %tok {2}                  ; the once-after-loop handoff still has a token
%v = a S3 {3}
R buf [%v] {3}
```

The post-loop acquire would consume the final iteration's `S1` release; for a
zero-trip loop, it would consume the release that originally fed `%t0`. This
would preserve the single handoff to `{3}` while allowing the loop to have no
token iter-arg or result.

Before adopting this transformation, future work must prove correct semaphore
phase and count behavior, including the zero-trip case; teach SYNC-DAG to
represent the pre-loop release feeding either the first in-loop acquire or the
post-loop acquire; verify that the final handoff to `{3}` remains correctly
ordered; and measure the performance impact. Until those checks are complete,
the pass keeps `gated(result-consumed)`.

### Composition: apply the hold rule from inner to outer

Nested loops use the same hold rule already described for one loop. Apply it
from the innermost loop outward:

```text
1. Decide the inner loop.
2. Represent the decided inner loop as one node in the outer loop:
     POINT_OF_USE -> no token enters or leaves the node
     gated        -> a token enters and leaves the node
3. Apply the same hold rule to the outer loop.
4. Repeat for each enclosing loop.
```

**Example 1: the inner loop becomes POINT_OF_USE.**
`test/NVWS/insert_semas_nested_ws_inner_loop.mlir` `@nested_ws_inner_loop`.
Before applying POINT_OF_USE, both loops carry the token:

```text
%t0 = a S1 root                      ; entry, seeds iteration 0
for outer (%t = %t0) {
  %ti = for inner (%u = %t) {
    W m0 [%u]  ttng.tc_gen5_mma {1}
    r  S0 %u  {1} [tc5mma]
    %u1 = a S0  {0}
    R m0 [%u1] ttng.tmem_load {0}
    r  S1 %u1 {0}
    %u2 = a S1  {1}                  ; inner regain
    yield %u2
  }
  yield %ti                          ; the outer only forwards the inner's token
}
```

Decide the inner loop first. No access or release uses `%u2` after its
acquire. `%ti` is passed through the outer loop, but nothing after the outer
loop consumes the resulting token. The single-level rule therefore moves
`a S1 {1}` to the inner loop's first access and removes the inner loop's token
iter-arg and result.

The decided inner loop is now one node in the outer loop:

```text
inner loop node
  token in:  none
  token out: none
```

Now apply the same rule to the outer loop. For this buffer, its body contains
only that inner-loop node. Because the node no longer takes or returns a
token, the outer loop has no token left to carry. Its token iter-arg and result
are removed too. The emitted dump is:

```text
|- scf.for (WS, tag=0) ... holdrule{passthrough-drop}
|  |- scf.for ... holdrule{pointofuse->ttng.tc_gen5_mma}
|  |  |- a  S1  {1}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S0  {1} [tc5mma]
|  |  |- a  S0  {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r  S1  {0} [none]
|  |  |- EXIT ... yield{native}
|  |- EXIT ... yield{drop}          <- the outer loop has no token result
```

`passthrough-drop` is not another hold rule. It means that the outer loop
previously carried a token only for the inner loop, and the decided inner loop
no longer takes or returns that token.

**Example 2: the inner loop remains gated, but the enclosing loop becomes
POINT_OF_USE.**
`test/NVWS/insert_semas_uniform_hold_transparency.mlir`
`@uniform_hold_s2_owner_change_cut`.

First decide the inner loop. Its result `%innerN` is used by the following
`{2}` read:

```text
%inner0 = a S0 {2}
%innerN = for inner (%v = %inner0) {
  R buf [%v] {2}
  r  S1 %v {2}
  %w = a S1 {1}
  W buf [%w] {1}
  r  S0 %w {1}
  %innerNext = a S0 {2}
  yield %innerNext
}
R buf [%innerN] {2}                  ; consumes the inner result
```

The inner loop therefore remains `gated(result-consumed)`. At the enclosing
level it is now one decided node:

```text
inner-loop node
  token in:  %inner0
  token out: %innerN
```

Now place that node in the enclosing loop. Before applying POINT_OF_USE to the
enclosing loop, its shape is:

```text
%outer0 = a S2 root                  ; seeds the first enclosing iteration
%outerN = for outer (%t = %outer0) {
  W buf [%t] {1}
  r  S0 %t {1}
  %inner0 = a S0 {2}

  %innerN = for inner (%v = %inner0) ...
             ; one gated node: token in=%inner0, token out=%innerN

  R buf [%innerN] {2}
  r  S2 %innerN {2}
  %next = a S2 {1}
  yield %next
}
```

Nothing uses `%next` after its acquire, and no access or release consumes
`%outerN` after the loop. Apply the same single-level move: put `a S2 {1}` at
the next iteration's first `{1}` store and remove `%outer0`, `%outerN`, and the
outer token iter-arg and yield. The already-decided inner node does not change:

```text
for outer {                           ; pointofuse->ttg.local_store
  %t = a S2 {1}
  W buf [%t] {1}
  r  S0 %t {1}
  %inner0 = a S0 {2}

  %innerN = for inner (%v = %inner0) ...
             ; same gated node: token in=%inner0, token out=%innerN

  R buf [%innerN] {2}
  r  S2 %innerN {2}
}
```

A gated inner loop therefore does not force its enclosing loop to remain
gated. It tells the enclosing loop only that this one node takes and returns a
token; the enclosing loop still makes its own hold decision.

**Example 3: both loops remain gated.**
`test/NVWS/insert_semas_nested_carrier.mlir`
`@outer_sourceful_alloc_inner_loop_reentry`.

Again, decide the inner loop first. The node immediately after it is
`r S2 {1}`, which uses the token returned by the inner loop:

```text
|- scf.for ... holdrule{gated(result-consumed)}
|  |- W m0  ttng.tc_gen5_mma {1}
|  |- r  S1  {1} [tc5mma]
|  |- a  S1  {0}
|  |- R m0  ttng.tmem_load {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- EXIT ... yield{a S0}
|- r  S2  {1} [tc5mma]              <- uses the inner loop's result
```

The inner result is consumed, so the single-level rule produces
`gated(result-consumed)`. The decided inner loop is therefore one node with a
token entering and leaving it:

```text
inner loop node
  token in:  yes
  token out: yes
```

Place that node in the outer loop and apply the same single-level rule. The
outer loop's bottom `a S2 {0}` is followed by `R m0 {0}`, so that token still
has an in-loop use. This is the already-described `trailing-use` case, and the
outer loop remains gated too:

```text
|- a  S2  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated(trailing-use)}
|  |- W m0  ttng.tmem_alloc {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- scf.for ... holdrule{gated(result-consumed)}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r  S0  {0} [none]
|  |  |- a  S0  {1}
|  |  |- EXIT ... yield{a S0}      <- the regain's token IS the loop result
|  |- r  S2  {1} [tc5mma]          <- the parent CONSUMES that result: the
                                      inner loop cannot stop yielding, so
                                      gated(result-consumed)
|  |- a  S2  {0}                   <- the outer loop's regain
|  |- R m0  ttng.tmem_load {0}     <- an access AFTER it: the regain still
                                      does work this iteration, so
                                      gated(trailing-use)
|  |- EXIT ... yield{a S2}
```

Examples 2 and 3 both have a gated inner loop. In Example 2, the enclosing
loop needs its next token only at the next iteration's first access, so it
becomes POINT_OF_USE. In Example 3, an access still uses the enclosing loop's
bottom token in the current iteration, so it remains gated. That is the
complete composition rule: decide the inner loop, represent it as one node
with its decided token input and result, and apply the same hold rule to the
next enclosing loop.

If POINT_OF_USE cannot be proven safe, the loop keeps its token iter-arg and
result. The dump prints `gated(trailing-use)` or `gated(result-consumed)` for
the two token-use blockers above; every other eligibility failure is printed
as bare `gated`. All three use the same carried-token fallback.

## Backing copies

`computeBackingPlan` chooses the number of copies. `buffer.copy`, when
present, is authoritative: `@fused_alias_depth_two` (worked below in
[Exact-alias handoffs](#exact-alias-handoffs-the-release-signals-the-successor-copy))
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

Protocol construction and hold placement have already decided where the
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
by the source access. In the comparison below, `before` is the source access
and `after` is the access following the acquire:

```text
before loop.stage < distance + after loop.stage
    already ordered; no cluster change

before loop.stage = distance + after loop.stage
    same expanded body; repair loop.cluster if needed

before loop.stage > distance + after loop.stage
    the existing loop.stage assignments cannot represent the handoff;
    SYNC-DAG does not change stages, so compilation fails
```

The handoff is never discarded. In the equality case,
`legalizeLoopSchedule` raises the access after the acquire and any same-body
SSA users needed to preserve their order; it never changes `loop.stage`. For
an asynchronous access, the release uses the schedule of its physical
completion. A semaphore buffer uses the schedule of the access it serves.
EMIT-IR only transcribes these decisions.

The three outcomes for the running example are:

```text
two copies: distance 2, before loop.stage 1 < 2 + after loop.stage 0
            -> unchanged
one copy:   distance 1, before loop.stage 1 = 1 + after loop.stage 0
            -> repair loop.cluster
errors twin: before loop.stage 2 > 1 + after loop.stage 0
             -> compilation fails
```

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

`assignBufferStageOffsets` runs one physical-stage analysis for circular
groups and non-circular exact-alias backings. It replays the fresh-write cursor
that ASP will use: a write records the cursor ordinal as the group's current
value, and a read uses the latest ordinal recorded for its group. The analysis
does not distinguish SMEM from TMEM. Circular metadata and exact aliasing only
determine how the allocations are represented and where the computed offsets
are attached.

Circular members are separate groups, while exact aliases are names in one
group. For an access or semaphore node:

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

### Non-circular exact-alias handoffs

Non-circular exact-alias handoffs reuse the fresh-write stage replay described
above. Their release shifts are derived from that replay. This handling applies
uniformly to SMEM and TMEM.

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

- `walkChain`, `applyTouch`, `VersionSource`, `ActiveUse`, `PieceState`,
  `Tokens`, and `canReuseTokenForPiece`
- `reduceEdges` and `buildEdgesAndSemas`
- `insertEntryAcquires` and `computeCrossings`
- `computeHoldRules`, `buildUniformHold`, `analyzeHoldPrefix`, and
  `applyHoldRulePlacement` (the token-reuse helpers `markTokenReuse`,
  `nodeReusesToken`, and `verifyTokenLocality`, and the hold records
  `HoldFeed`/`HoldPrefix`, are file-local here)
- `ownerCompletionScheduleAtLoopExit`
- `computeBackingPlan`
- `assignBufferStageOffsets`, `computeSlotSchedule`,
  `computeLoopCarriedDistance`, `addSyncScheduleEdges`,
  `legalizeLoopSchedule`, `assignSyncScheduleChain`, and
  `finalizeSyncSchedule`
- `buildSyncDag`
- the DAG dump used throughout: `NVWS_INSERT_SEMA_DUMP_DAG=1`
  (`dumpDagTree`)
