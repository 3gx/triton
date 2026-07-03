# SYNC-DAG

## Purpose

SYNC-DAG converts OWNER-DAG ownership changes into a balanced semaphore
protocol: who waits for whom, through which semaphore, at what backing
depth, and where in the pipeline schedule.

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
   (holds), choose the backing depth, then project the protocol onto the
   pipeline schedule
```

## Notation

Figures are trimmed excerpts of actual pass dumps, not hand-derived diagrams.
Most come from the listed lit tests; the section explicitly labeled synthetic
came from a temporary input run through the same pass. Unrelated groups,
`parts{...}` fields, and some `ENTER` rows are elided; `; ...` and `<-`
annotations are added. The command is:

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
holder        an owner whose latest access a future writer must respect;
              after a write: the writer plus readers that later join;
              at child entry: the ENTER owner
retained token
              an earlier token acquired by the same owner that SYNC-DAG has
              explicitly proved a later access may reuse
wave          the owner and node associated with the token the walk considers
              current;
              it starts unset at ENTER, and a retained-token return across
              another owner's active wave does not move it
hold          acquire -> accesses -> closing release, all by one owner: the
              ordinary span protected by that token; read spans may overlap,
              and an explicitly retained token may be reused later
regain        the acquire at the bottom of a loop body through which the
              body re-acquires the group's semaphore for the next iteration
              (`Hold::regain`)
chain         the group's node sequence of one block, in program order; a
              region node holds child chains
```

Dump rows:

```text
|- scf.for (WS, tag=0)      loop node; (WS, tag=N) = warp-specialized loop
|- W m0  ttg.local_store {1}   access row: W(rite)/R(ead), member, op, owner
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
pieces{P0:W:{0}}            region row: per-piece merged effect and owner
thread{{0}}                 region row: this region has a crossing; a surviving
                            `if` crossing returns a token, while a loop's hold
                            decides whether it has a token iter-arg and result
holdrule{...}               loop row: the crossing's hold outcome (its values
                            are decoded in "Reading a loop row at sight";
                            until then read them as opaque outcome labels)
yield{X}                    EXIT row: what this chain returns for the
                            crossing — a S<n> = that acquire's token;
                            native = no token crosses (protocol lives
                            inside); drop = this loop has no token result
                            because its final nested loop returns none; pass =
                            this branch has no acquire or nested region with a
                            crossing, so it returns the token available before
                            the `if`;
                            scf.for/scf.if =
                            the token source is a nested region node, decoded
                            in "Reading a loop row at sight"
stage-offset=N              stage offset assigned to this protocol row
S<n> / E<n>                 semaphore names; E<n> = dedicated entry semaphore
SEMAS: S1{count=1 entry inherit={@0.0}}   per-semaphore summary; entry =
                            created initially released; inherit={...} = the
                            owner recorded on that initially released state
                            (its inheritStamp) — read by the hold decision
                            (as an entry feed's owner, buildUniformHold) and
                            by the verifier that checks retained-token use;
                            EMIT-IR also uses it as
                            the owner of an unpartitioned entry acquire when
                            recording current/retained tokens — which is one
                            of three: the owner of the group's
                            first real access (insertEntryAcquires, both of
                            its paths), the retargeted EXIT-handoff
                            acquire's owner (buildEdgesAndSemas), or the
                            hold owner (applyHoldRulePlacement)
BACKING: numStages=N        depth chosen for the backing allocation
```

Pseudo-IR strips types and attributes: `%t = a S0 {1}` / `r S0 %t {1}` are
the token-producing acquire and token-using release, `W m0 [%t]` is an
access made while holding token `%t`, and `for iter_args(%t = %t0)` —
abbreviated `for (%t = %t0)` in the figures below — is a loop carrying the
token.

Do not confuse the two layers: `P0` and edges are analysis labels that never
appear in the IR. The semaphore *token* is an IR value — each emitted
acquire returns a fresh one, and a release takes that token as an operand.
One token can feed several releases. A later buffer or release can also use
that token when SYNC-DAG explicitly marks the node for same-owner retention;
EMIT-IR never infers that exception on its own.

## The walk: accesses to edges

The walk runs once per group, in program order over its chains
(`walkChain`). At each access it first asks whether the data needs an edge.
There are only two data rules:

1. **Read after write (RAW)** — a new reader waits for the piece's version
   source.
2. **Write after other holders (WAR)** — a write replaces the data, so it waits
   for every holder with a different owner that is not already ordered
   before the writer.

Every wait becomes an edge between two concrete DAG nodes (`EdgeRec` stores
the two node pointers). The walk tracks three kinds of state:

```text
version source   per piece   latest write, first toucher, or child ENTER proxy
holders          per piece   owners whose latest nodes a writer must respect
wave             per chain   source node and owner of the token considered
                             current
```

The first two decide data edges. The wave is used only after those rules: when
an access needs a token but neither a data edge nor an earlier owner-local
token supplies one, the walk adds an edge from the wave's node.

```text
R  m0  ttg.local_load  {1}
└──────────┬────────┘  └┬┘
     the ACCESS           the OWNER
  (one event: this op     (the partition executing
   touching the memory)    this access: partition 1)
```

For each piece, `PieceGame` stores a stable `VersionSource` and a holder
record for each owner. A read moves only that reader's holder record; it
does not move the version source. Independent readers therefore fan out
from the write, first toucher, or ENTER rather than forming a reader-to-reader
chain. A write waits on the latest node of every other holder, then resets
both the version source and the holder set to itself.

### Data edges and token supply

RAW/WAR rules answer whether an access must wait for data. Independently, an
access rewritten under a group's semaphore protocol must use a token-backed
buffer view. After applying the data rules, there are two cases:

1. The access's owner has a token valid for the access, so the access uses it.
2. The access's owner has no valid token, so the pass creates an acquire that
   produces one.

An earlier token remains valid for a read only while its owner is a holder. It
remains valid for a write only after every other holder is already ordered
before the writer.

Together, these rules produce one fan-out/fan-in access shape:

```text
                 ┌── R {0} ──┐
W {0} ───────────├── R {1} ──┼────────── W {3}
                 └── R {2} ──┘
```

The new readers `{1}` and `{2}` receive RAW edges from `W {0}`. The returning
`R {0}` needs no data edge because `{0}` is already a holder; it reuses `{0}`'s
earlier token and is not ordered behind either new reader. The later `W {3}`
receives WAR edges from the latest accesses of all three holders.

Implementation correspondence: `WaveSt` records the owner and source node of
the token the walk considers current. `hadWave` records owners that may still
reuse an earlier token, and `Node::retainedTokenOwner` marks that reuse in case
1 after the per-piece checks pass. These are bookkeeping for token supply, not
additional data-dependency rules.

At a region's `EXIT`, each piece returns to its `EXIT` owner. If the piece
will be used again — during the next loop iteration or after the region —
the `EXIT` owner waits for every holder with a different owner.
No new edge is added if an earlier edge already makes the `EXIT` owner wait
for that holder. EXIT then keeps only the carried owner in the holder set;
it does not move the version source.

The worked examples show the common shapes; the complete checklist follows
them.

### Example: two-partition handoff

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` — partition
0 stores, partition 1 loads, every iteration. State shown after each row,
with the rule that fired:

```text
walk            edge                                    state AFTER the row
W store {0}     — (first touch)                         version=store@{0} holders={0}    wave={0}
R load  {1}     e1: store@{0} -> load@{1}    (RAW)      version=store@{0} holders={0,1}  wave={1}
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
acquire is gone and the regain sits at the store instead (the row's
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

`S1` is created initially released so iteration zero's `a S1` succeeds
before any release has run — see
[Loops and the first acquire](#loops-and-the-first-acquire).

### Example: edge reduction lowers the pending count

`test/NVWS/insert_semas_transitive_reduction.mlir`
`@serialized_ring_reduces` — members `m0` and `m1` overlap on piece P1.
The table follows that piece as `{0}` writes, `{1}` reads, and `{2}` writes:

```text
walk          edge                                       P1 state AFTER the row
W m0 {0}      — (first touch)                            version=W@{0} holders={0}
R m0 {1}      e1: W@{0} -> R@{1}             (RAW)      version=W@{0} holders={0,1}
W m1 {2}      e2: {0} -> {2}, e3: {1} -> {2} (WAR)      version=W@{2} holders={2}
```

The write raises two candidates because `{0}` and `{1}` are both holders.
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

The complete loop and the reduction proof appear in
[Example: a redundant edge is dropped](#example-a-redundant-edge-is-dropped).

### Example: fan-out and a retained owner token

`test/NVWS/insert_semas_transitive_reduction.mlir` `@fanout_not_reduced` —
partition 0 writes, partitions 1 and 2 read, then partition 0 *re-reads*.
State shown after each row:

```text
walk            edge                                    state AFTER the row
W alloc {0}     — (first touch)                         version=alloc@{0} holders={0}     wave={0}
R load  {1}     e1: alloc@{0} -> load@{1}    (RAW)      version=alloc@{0} holders={0,1}   wave={1}
R load  {2}     e2: alloc@{0} -> load@{2}    (RAW)      version=alloc@{0} holders={0,1,2} wave={2}
R load  {0}     — (reuse {0}'s retained token)          version=alloc@{0} holders={0,1,2} wave={2}
EXIT            e3: load@{1} -> alloc@{0}@next (WAR)
(recurrence)    e4: load@{2} -> alloc@{0}@next (WAR)
```

Both new readers take their data edge from the unchanged version source,
so neither reader is serialized behind the other. The fourth row is also a
read of that version by an existing holder. It needs no data edge, and `{0}`
still has the token acquired before its write, so SYNC-DAG marks the node
to reuse that token. Because this access uses the earlier token instead of the
one tracked by the wave, the wave remains at `{2}`.

At `EXIT`, the next iteration's write waits for the two *other* holders,
`{1}` and `{2}`. Those edges converge on `S2`, whose pending count is 2:

```text
|- a  S2(2)  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated(rel-count)}
|  |- W m0  ttg.local_alloc {0}
|  |- r  S0  {0} [none]
|  |- r  S1  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S2  {1} [none]        ; e3
|  |- a  S1  {2}
|  |- R m0  ttg.local_load {2}
|  |- r  S2  {2} [none]        ; e4
|  |- R m0  ttg.local_load {0} ; uses {0}'s retained S2 token
|  |- a  S2(2)  {0}            ; waits for both foreign readers
|  |- EXIT ... yield{a S2}
SEMAS: S0{count=1} S1{count=1} S2{count=2 entry inherit={@0.0}}
```

The retained-token mark is not printed in this dump. EMIT-IR renders it by
building `{0}`'s final buffer view from the carried `S2` token, even though
that same token already fed the two producer releases. Only nodes carrying
this explicit SYNC-DAG proof receive that exception.

### The per-access rules, in full

`applyTouch` advances the per-piece data game:

```text
first touch of a piece:
  edges:  none
  state:  version source = this node; holders = {toucher}

write by owner P:
  edges:  H's last access -> P, for every OTHER holder H
            skipping an H whose last access an earlier edge already
            orders before P
  state:  version source = this write; holders = {P}

read by an owner already holding the piece:
  edges:  none
  state:  that holder's last node moves to this read;
          the version source does not move

read by a new owner:
  edges:  version source -> this read
  state:  the reader joins holders
```

After all touched pieces have run through `applyTouch`, `walkChain` chooses
how the access obtains a token:

```text
one or more data edges were added
  their acquire supplies the token; for a partition-owned access, move the
  wave to this access

no data edge, and the owner has an eligible earlier token
  mark retainedTokenOwner; if the wave belongs to another owner, leave the
  wave unchanged; otherwise establish or advance the wave at this access

no data edge, no retained token, and the wave belongs to another owner
  add wave node -> this access only to supply a token; move the wave here

otherwise
  no extra edge; the token already available supplies the access, and a
  partition-owned access establishes the wave
```

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
|  |- a  S5  {0}
|  |- W m2  ttg.local_store {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- R m2  ttg.local_load {1}
|  |- r  S1  {1} [none]
|  |- r  S3  {1} [none]
|  |- a  S1  {2}
|  |- W m0  ttg.local_store {2}
|  |- r  S2  {2} [none]
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
       S5{count=1 entry inherit={@0.0}}
```

The full-width `W m2 {0}` releases `S0`, which `R m2 {1}` acquires. That read
then starts two independent paths. This diagram unrolls one iteration
boundary; `A --Sx--> B` means that `A` releases `Sx` and `B` acquires it:

```text
                              ENTER(i)
                                  |
                              a S5 {0}
                                  |
                              W m2 {0}
                                  | S0
                                  v
                              R m2 {1}
                     +------------+------------+
                  S1 |                         | S3
                     v                         v
                 W m0 {2}                 W m1 {4}
                     | S2                     | S4
                     v                        v
                 R m0 {3}                 R m1 {0}
                     | S5                     |
                     |                        v
                     |                     EXIT(i)
                     |                        |
                     |                        v
                     |                    ENTER(i+1)
                     |                        |
                     +-------------------> a S5 {0}
                                              |
                                              v
                                          W m2 {0}
```

EMIT-IR materializes those same two paths (unrelated operations and types
omitted):

```text
for {
  %t0 = a S5 {0}
  W m2 [%t0] {0}
  r S0 %t0 {0}

  %t1 = a S0 {1}
  R m2 [%t1] {1}
  r S1 %t1 {1}
  r S3 %t1 {1}

  %t2 = a S1 {2}
  W m0 [%t2] {2}
  r S2 %t2 {2}
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

Although the dump and IR print the `m0` path first, the `m1` path waits only on
`S3`, released by `R m2 {1}` before either path starts. It does not wait on any
release from `{2}` or `{3}`. Likewise, the `m0` path does not wait on `{4}` or
the final `{0}` read. The right path already ends at `{0}`, the owner that starts
the next iteration, so it needs no loop-closing release. `S5` is created
initially released, so iteration zero's `a S5 {0}` succeeds. On the left path,
`R m0 {3}` already waits for `W m0 {2}` through `S2`. Its loop-closing release
therefore also covers the writer, whose separate close is deleted. The sole
`r S5` feeds the count-1 acquire on the next iteration; it does not order the
two paths within the current iteration.

### Composition: nested regions in the walk

A nested `for` or `if` is one node in its parent chain. Its per-piece
owner/effect summary is applied through the same two data rules. Unlike a
plain access, the region node neither reuses an earlier token nor receives an
extra edge solely so it can acquire one. Each child chain then starts a
separate game at `ENTER`:

```text
logical producer   imported from the parent before the region summary
version source     the child's ENTER node and its boundary owner
holders            that boundary owner alone
wave               unset
```

If all ENTER pieces have the same partition owner, that owner may reuse its
earlier token. Otherwise no owner can reuse one from outside the child.

The ENTER node is the concrete proxy for the version established outside the
child. New child readers therefore fan out from ENTER, never from the
previous child reader or directly from an outer node. The logical producer
is carried so deeper children can import it. It also decides whether ENTER
can import the outer version's completion kinds: it does so when the
boundary owner is that producer, and uses `[none]` otherwise.

After all children are walked, the parent discards the owners whose earlier
tokens might have been reused. A uniform partition-owned region summary
becomes the new wave and its owner is the only one remembered. A mixed-owner
or root-owned summary leaves the wave unset. This reset prevents a
retained-token proof from crossing a region boundary.

#### Why a holder can still need an acquire after a child

```text
W {0}
  │
  ▼
┌─────────────────────────┐
│ child region            │
│                         │
│   R {1}                 │
│   R {2}                 │
│   ...                   │
│   return {1}'s token    │
└────────────┬────────────┘
             │
      release by {1}
      acquire by {0}
             │
             ▼
           R {0}
```

The child only reads, so `{0}` remains a holder and the final `R {0}` needs
no RAW/WAR data edge. But the child ends with `{1}`'s token, and `{0}`'s
earlier token is not retained across the child boundary. `{1}` therefore
releases after the child and `{0}` acquires before `R {0}`. They are needed
only so `R {0}` has a token it can use.

A WS-tagged loop may take over root-held memory without an edge (the
adoption shown in
[Example: root store adopted](#example-root-store-adopted-entry-acquire-seeds-iteration-zero)
below).

Both levels, worked on one real test —
`test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain`: partition `{3}` stores at the
outer level, then a nested plain `for` (summary `pieces{P0:W:{2}}`) runs
readers `{2}`, `{1}`, `{0}`, `{3}`, with `{1}` also writing. The parent chain
walks the nested loop as one write by `{2}`:

```text
parent walk       edge                                        state AFTER the row
W store {3}       — (first touch)                             version=store@{3} holders={3}  wave={3}
for [P0:W:{2}]    e1: store@{3} -> for@{2}      (WAR)         version=for@{2} holders={2}    wave={2}
EXIT              e2: for@{2} -> store@{3}@next (WAR)
```

The child imports logical producer `{3}`, but its concrete version source
is `ENTER@{2}`. Its first read moves holder `{2}`'s latest node without
moving that source. The next reader still receives the version from ENTER;
because `{2}`'s holder node has moved to its read, `{1}`'s later write must
wait for that read as a separate WAR dependency:

```text
child walk        edge                                        state AFTER the row
R load {2}        — (reuse ENTER token)                       version=ENTER@{2} holders={2}        wave={2}
R load {1}        e3: ENTER@{2} -> load@{1}  (RAW)            version=ENTER@{2} holders={2,1}      wave={1}
W store {1}       e4: load@{2} -> store@{1}  (WAR)            version=store@{1} holders={1}        wave={1}
R load {0}        e5: store@{1} -> load@{0}  (RAW)            version=store@{1} holders={1,0}      wave={0}
R load {3}        e6: store@{1} -> load@{3}  (RAW)            version=store@{1} holders={1,0,3}    wave={3}
EXIT              raw: store@{1} -> {2}@next (later deleted)
(recurrence)      e7: load@{0} -> {2}@next   (WAR)
                  e8: load@{3} -> {2}@next   (WAR)
```

The walk proposes three recurrence closes because the corrected version is
still held by its writer `{1}` and readers `{0}` and `{3}`. Both readers
already wait for the writer through `e5` or `e6`, so either surviving reader
close also covers the writer. Reduction deletes the writer's direct close.
The two reader closes both remain because neither reader waits for the other.

As emitted — `e1` lands on the same semaphore `S0` that `e7`/`e8` converge
on; why a handoff into a loop node shares the loop's own semaphore, and why
its release arrives with count 2, is the loop-node special case at the end
of [Edges to semaphores](#edges-to-semaphores):

```text
|- a  S5  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated(rel-count)}
|  |- W m0  ttg.local_store {3}
|  |- r  S0(2)  {3} [none]        ; e1 tail
|  |- a  S0(2)  {2}               ; e1 head, spliced before the loop node
|  |- scf.for ... holdrule{gated(result-consumed)}
|  |  |- r  S1  {2} [none]        ; e3 tail is ENTER, placed before the read
|  |  |- R m0  ttg.local_load {2}
|  |  |- r  S2  {2} [none]        ; e4 tail
|  |  |- a  S1  {1}               ; e3 head
|  |  |- R m0  ttg.local_load {1}
|  |  |- a  S2  {1}               ; e4 head
|  |  |- W m0  ttg.local_store {1}
|  |  |- r  S3  {1} [none]        ; e5 tail
|  |  |- r  S4  {1} [none]        ; e6 tail
|  |  |- a  S3  {0}               ; e5 head
|  |  |- R m0  ttg.local_load {0}
|  |  |- r  S0  {0} [none]        ; e7 tail
|  |  |- a  S4  {3}               ; e6 head
|  |  |- R m0  ttg.local_load {3}
|  |  |- r  S0  {3} [none]        ; e8 tail
|  |  |- a  S0(2)  {2}            ; e7/e8 head: the child loop's regain
|  |  |- EXIT ... yield{a S0}
|  |- r  S5  {2} [none]           ; e2 tail
|  |- a  S5  {3}                  ; e2 head: the parent loop's regain
|  |- EXIT ... yield{a S5}
SEMAS: S0{count=2} S1{count=1} S2{count=1} S3{count=1} S4{count=1}
       S5{count=1 entry inherit={@0.3}}
```

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

Retention remains an explicit SYNC-DAG fact during this conversion. If an
edge leaves an access marked for retained-token use, its inserted release
is marked for the same owner, so both the access and release select that
owner's earlier token. A verifier checks that the token is still available
and clears retained state at region boundaries. EMIT-IR only renders these
marks; it does not decide that another node is eligible.

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
walk             P1 game before        rule                      raw edge
W m0 {0}         seeded, holder {0}    write; writer is the      —
                                       sole holder
R m0 {1}         version W@{0}         read by new owner         e1: {0} -> {1}
W m1 {2}         holders {0},{1}       write: edge from EVERY    e2: {0} -> {2}
                                       other holder              e3: {1} -> {2}
R m1 {0}         sole holder {2}       read by new owner         e4: {2} -> {0}
EXIT             every piece carries   P0/P2: holder {1}         e5: {1} -> {0}@next
(recurrence)     owner {0}             closes; P1: skip, e4
                                       already ordered {2}'s
                                       last access before {0}
```

Reduction keeps four of the five raw edges and drops one:

```text
kept:     e1 {0}->{1}   e3 {1}->{2}   e4 {2}->{0}   e5 {1}->{0}@next

dropped:  e2 {0}->{2}   {2} already waits on {1} (e3), and {1} already
                        waited on {0} (e1) — a direct wait on {0} adds
                        nothing; and e3, the kept handoff into {2}, already
                        gives {2}'s hold its acquire
```

The decision is that trace — follow `e2`. Fact one, its ordering is
already implied: `{2}` waits on `{1}` through kept `e3`, `{1}` waited on
`{0}` through kept `e1`, and stitching those two through each partition's
own program order forces `{0}`'s store before `{2}`'s write with no direct
edge. Fact two, `e2` is not the first kept edge into `{2}`'s hold: `e3` is
the kept handoff into `{2}`, so the hold keeps its acquire and its token.
Both facts hold, so `e2` is deleted. No other in-body edge has fact one.
`e5` closes the loop, which the straight-line trace cannot judge — closes
are judged by the wrap-around pass in the fine print below; this one is
kept.

Every kept edge has its own destination node, so four edges become four
count-1 semaphores — the serialized ring the test is named after:
`{0} -> {1} -> {2} -> {0}` (S0/S1/S2), plus the loop-closing handoff
`{1} -> next iteration's {0}` (S3, the kept `e5`). The read-to-next-write leg
on `{0}` itself needs no semaphore — it is `{0}`'s own program order. The
emitted body, each kept edge's acquire/release pair marked:

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

`S3`'s acquire sits at the top of the body rather than above the loop —
that placement is what the row's `holdrule{pointofuse->...}` label stands
for, decided in [Crossings and holds](#crossings-and-holds) below.
Contrast `@fanout_not_reduced` above, `holdrule{gated(...)}`: its `S2`
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

`e4`'s three facts, traced in wrap-around order:

```text
1. implied      {0}'s read precedes {0}'s own next-iteration alloc (program
                order across the boundary), which e1 orders before {1}'s
                read, which e2 orders before {2}'s alloc — the close adds
                nothing
2. handoff      in wrap-around order, {2} first touches P2 at its alloc,
   applied      and e2 — the kept handoff into {2} — lands exactly there:
   in time      {2}'s hold has its acquire and token at that point
3. not first    the chain's first wave owner is {0}, not {2}
-> e4 dropped
```

`e5` targets the chain's first wave owner, `{0}`. It is the only close feeding
that next-iteration acquire, so it must remain and becomes `S3`. The emitted
body, with `{2}`'s hold opening on the in-body `S1` alone:

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

### The deletion conditions, in full

1. **Implied ordering** (`reduceEdges`). An edge is a candidate only if both
   its endpoints are partition-owned accesses — edges with a root or region
   endpoint are never reduced. A candidate is deleted only when two
   facts hold. First, its ordering is already implied: following kept
   edges, stitched through each partition's own program order, the source
   is already forced before the destination. Second, it is not the first
   kept edge into the destination's hold: the group's most recent kept
   handoff already targets this same destination owner, so that owner's
   hold already has its acquire — the first kept edge into a hold is what
   creates that acquire and hands the owner a token, and deleting it would
   leave the hold with no acquire at all, however implied its ordering is.
   Edges that close the loop — from this iteration's nodes to the next
   iteration's — cannot be proven by the straight-line sweep, because the
   implication path wraps around the loop boundary; a second, wrap-around
   traversal (`sweepTraversalClosure`) handles them in two cases.

   Multiple closes feeding the same next-iteration acquire of the chain's
   first wave owner are reduced within that fan-in. A later close can replace
   an earlier one only when a kept direct edge from the earlier source to the
   later source carries every completion required by the earlier close, and
   the later close will not itself be merged away. At least one close remains;
   closes on independent paths remain. The `W m0 {2}` close removed behind
   `R m0 {3}` in
   [Example: disjoint submember protocols run independently](#example-disjoint-submember-protocols-run-independently)
   is this case.

   A close to any other owner is dropped only when three facts hold: its
   ordering is implied by the kept edges; a kept handoff into that owner has
   already been applied by the point, in wrap-around order, where the
   destination owner first touches the edge's pieces (the relaxed form of the
   second condition — *some* kept handoff into that owner, evaluated at that
   first touch, not the first kept edge); and the destination owner is not the
   chain's first wave owner. Finally, deletions can lean on each other's
   proofs, so every deleted edge is re-proven against the kept set alone rather
   than emit a missing wait. Straight-line drops are re-proven by `sweepChain`
   and fail with `transitive-reduction closure violation`; dropped closes are
   re-verified by `sweepTraversalClosure` (which `sweepChain` skips) and fail
   with `traversal-closure violation: dropped close not implied`.
2. **Repeats from one sender** (`mergeEdges`). The same handoff raised
   through several pieces counts once (same source node, destination node,
   source owner) — the example above shows this form: `R m1 {0}` raises
   its edge through P1 and P2 at once, counted once. And several edges
   from one owner into one destination collapse to the owner's *last*
   access — if that owner is past its last access, it is past the earlier
   ones by its own program order, so one release there covers them all.
   The collapse, complete (pseudo-walk; no lit test dumps this shape):
   member `m0` covers pieces P0 and P1, member `m1` covers P1 alone; in a
   WS loop body, `{0}` stores, `{1}` reads through both members, and the
   recurrence pays `{0}`'s next-iteration store:

   ```text
   walk          edge                                   state AFTER the row
   W m0 {0}      — (first touch of P0 and P1)           version=W@{0} holders={0}   wave={0}
   R m0 {1}      e1: store@{0} -> load@{1}   (RAW)      holders={0,1}, both pieces wave={1}
   R m1 {1}      — (same holder; retain {1}'s token)    P1: {1}'s last node moves  wave={1}
   EXIT          WAR closes, one per piece:
   (recurrence)    e2: R m0@{1} -> {0}@next   (P0)
                   e3: R m1@{1} -> {0}@next   (P1)
   ```

   Two raw closes, one source owner, one destination — `mergeEdges` keeps
   the later source, the `R m1` read, and one release there covers both;
   the next iteration's acquire keeps pending count 1:

   ```text
   |- a  S1  root  ; entry            seeds iteration zero
   |- scf.for (WS, tag=0) ...
   |  |- W m0  {0}
   |  |- r  S0  {0}                   ; e1 tail
   |  |- a  S0  {1}                   ; e1 head
   |  |- R m0  {1}
   |  |- R m1  {1}
   |  |- r  S1  {1}                   ; the ONE release, after {1}'s last access
   |  |- a  S1  {0}                   ; the regain both closes converged on:
   |  |- EXIT                         ;   pending count 1, not 2
   ```

### One destination node, one semaphore

What remains is grouped by destination node and destination owner: one
destination, one semaphore, one acquire; its source owners are the
releases, and the pending count is how many there are —
`@fanout_not_reduced`'s `a S2(2)` above, where `e3` and `e4` converge. A
release's `arrive_count` is raised above one only when one release must
stand in for several, keeping the total equal to the pending count; in the
code that raise is reachable only through the semaphore sharing of the
composition case below, which shows it.

### Composition: a handoff into a loop node

One special case: a handoff into a loop node reuses the semaphore of that
loop's regain (`Hold::regain`) when the acquiring owner matches.

Both raises are real in `test/NVWS/insert_semas_release_count.mlir`
`@release_multiplicity_unified_fanin_regain`, the nested loop walked in
[Composition: nested regions in the walk](#composition-nested-regions-in-the-walk)
above. Its parent handoff (`store@{3} -> for@{2}`) points at a loop node,
and its acquiring owner `{2}` matches the owner of that loop's regain —
the bottom re-acquire on which the two surviving reader closes from `{0}`
and `{3}` converge — so instead of a new semaphore the handoff reuses `S0`,
and its acquire is spliced before the loop node. The writer's direct close
was removed because both readers already wait for it. `S0`'s pending count
is 2, but the handoff has one source, so its single release stands in for
both waits — `arrive_count` 2, dumped as `r S0(2)`:

```text
|- W m0  ttg.local_store {3}
|- r  S0(2)  {3} [none]     ; one release, arrive_count 2: both waits at once
|- a  S0(2)  {2}            ; the handoff's acquire, before the loop node,
|- scf.for ...              ;   through the loop's own regain semaphore
|  |- ...
|  |- a  S0(2)  {2}         ; the regain the handoff reused
|  |- EXIT ... yield{a S0}
```

## Loops and the first acquire

An edge that closes the loop is one waiting requirement that recurs every
iteration. Unrolled, its instances are plain; rolled back into the IR, the
same semaphore must appear at two textual places:

```text
unrolled reality:                      rolled IR:

wait -> iter 0's accesses              %t0 = a S3(2) root  ; instance #0,
        ... r S3 {1} ... r S3 {2}      for iter_args(%t = %t0) {   before the loop
wait -> iter 1's accesses                ...
        ... r S3 {1} ... r S3 {2}        ... r S3 {1} ... r S3 {2}
wait -> iter 2's accesses                %t1 = a S3(2) {0} ; instances #1..#N
        ...                              yield %t1         ; carried to next iter
                                       }
```

Instances 1..N are the bottom-of-body re-acquire — each iteration acquires
the token *for the next one* and yields it into the iter_args. Instance 0
has no previous iteration to release for it, so it must be written above
the loop — same semaphore, seeding the carried chain. The pre-loop acquire
is not an edge's head: it comes from separate seeding logic
(`insertEntryAcquires`), which runs after `buildEdgesAndSemas` has
converted the edges, and the semaphore is created initially
released so the ledger balances. Over N iterations of the example above:

```text
credits:  initial release (1) + N in-loop release pairs        = N+1 satisfactions
waits:    1 pre-loop acquire + N in-body acquires               = N+1   ✓ balanced
without the initial release: the pre-loop acquire starves -> DEADLOCK
```

The seeding rule in full — every synchronized group needs its semaphore to
be satisfiable once before the first acquire runs. The seeding is
implemented across three passes, which run in this order:
`buildEdgesAndSemas`, then `insertEntryAcquires`, then
`applyHoldRulePlacement`. The bullets below are ordered by the group's
shape, not by pass order; each bullet's bold label names the pass that does
the placement. One outcome label they use ahead of its section:
POINT_OF_USE is the hold outcome that moves a loop's regain to its first
in-body access; the trace that decides it is in
[Crossings and holds](#crossings-and-holds).

- **`insertEntryAcquires`** — the loop body re-acquires each iteration
  through a regain: an unpartitioned acquire of that same semaphore, spliced
  before the group's first placement node (immediately above the loop when the
  loop is that first node), seeds iteration zero (the example above);
- **`buildEdgesAndSemas`** — the `EXIT` handoff targets
  an owner that is not the first owner with a partition-owned access in the
  loop body: the acquire is placed at that owner's first in-body access and
  the semaphore is created initially released. That retargeted semaphore gets
  no pre-loop instance of its own; the group's regain semaphore still receives
  one from `insertEntryAcquires`;
- **`applyHoldRulePlacement`** (which edits the acquire `insertEntryAcquires`
  placed) — a POINT_OF_USE hold fed by the same semaphore as
  its regain drops its pre-loop acquire; iteration zero is seeded solely by
  the initially released state (the `@local_loop_carried_and_result` figure
  above shows this: `S1` is entry, and its acquire sits in the body);
- **`applyHoldRulePlacement`** — a POINT_OF_USE hold fed by a *different*
  semaphore keeps its feeding acquire — the acquire above the loop that
  supplies the first iteration's token, defined in
  [The structural conditions](#the-structural-conditions-and-every-gated-reason)
  — above the loop and marks the loop's own regain semaphore initially
  released; the worked figure sits there too, beside the final-permission
  construction the same shape requires;
- **`insertEntryAcquires`** — otherwise a dedicated entry semaphore (`E<n>`)
  is created, acquired once before the group's first node and released after
  the group's last top-level node, as in `test/NVWS/insert_semas.mlir`
  `@warp_specialize_tma_matmul`:

```text
|- a  E1  root  ; entry              <- acquired before the group's first node
|- W m0  ttng.tmem_store root
|- scf.for (WS, tag=0) ...            <- no thread{}, no holdrule{}: no crossing
|  |- W m0  ttng.tc_gen5_mma {1}
|  |- EXIT pieces{P0:W:{1}}           <- plain EXIT, the no-crossing signature
|- r  S0  {@0.1} [tc5mma]
|- a  S0  root
|- R m0  ttng.tmem_load root
|- r  E1  root [none]                <- released after the group's last node
SEMAS: S0{count=1} E1{count=1 entry inherit=root}
```

The owner of the group's first real access is recorded as the entry
semaphore's `inheritStamp`; the entry acquire itself stays root-owned.

Composition note — descent through `if`s: while the only top-level node
involving the group is an `if` with exactly one involved child chain,
`insertEntryAcquires` descends into that chain and applies the same
placement there; it never descends into a loop. Complete, minimal
(pseudo-IR; no lit test keeps a descended acquire — in the tests that
exercise the descent, `@branch_local_initial_acquire_stays_with_create`
and `@guarded_tokenless_if_deferred_initial_acquire`, the loop goes
POINT_OF_USE and the acquire is later deleted):

```text
if %c {                         ; the group's only top-level node
  %t0 = a S root                ; the entry acquire, placed INSIDE the
  %tN = for (%t = %t0) {        ;   branch, immediately above the loop —
    W buf [%t] {0}              ;   not above the if
    r  S %t {0}
    %t1 = a S {0}               ; regain; S is created initially released
    yield %t1
  }
  r  S2 %tN {0}                 ; the loop result is consumed inside the
  %u = a S2 root                ;   branch, so the loop keeps the carried
  R buf [%u] root               ;   shape and the entry acquire survives
}
```

The `buildEdgesAndSemas` seeding — the retargeted `EXIT` handoff —
complete (pseudo-IR; no lit test dumps this shape). Two members, `m0`
first touched by `{1}` — the chain's first owner — and `m1` first touched
by `{2}`, so `m1`'s recurrence handoff targets `{2}` and its acquire moves
to `{2}`'s first in-body access:

```text
%t0 = a S2 root                 ; entry — the group's regain S2 still gets
                                ;   its pre-loop instance
W m0 [%t0] root                 ; root init, adopted by {1} without an edge
%tN = for (%t = %t0) {          ; the carried default (the root init above
  W m0 [%t] {1}                 ;   keeps this loop out of POINT_OF_USE)
  r  S1 %t {1}
  %u = a S3 {2}                 ; <- the retargeted handoff acquire, at {2}'s
                                ;    first in-body access; S3 is created
                                ;    initially released, NO pre-loop instance
  W m1 [%u] {2}
  r  S0 %u {2}
  %v = a S0 {1}
  R m1 [%v] {1}
  r  S3 %v {1}                  ; the handoff's tail: releases the NEXT
  %w = a S1 {2}                 ;   iteration's {2}
  R m0 [%w] {2}
  r  S2 %w {2}
  %t1 = a S2 {1}                ; the regain, seeded by %t0's semaphore
  yield %t1
}
```

### Example: root store adopted, entry acquire seeds iteration zero

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` — a root
`tmem_store` initializes the accumulator before the WS loop; the loop's
first toucher `{1}` adopts it without a root-to-partition handoff, and the
regain semaphore `S1` gets one unpartitioned pre-loop acquire:

```text
|- a  S1  root  ; entry              <- seeds iteration zero of the regain
|- W m0  ttng.tmem_store root        <- adopted: no root->partition semaphore
|- scf.for (WS, tag=0) ... holdrule{gated(entry-consumed)}
|  |- R m0  ttng.tmem_load {1}
|  |- W m0  ttng.tmem_store {1}
|  |- r  S0  {1} [none]
|  |- a  S0  {2}
|  |- W m0  ttng.tc_gen5_mma {2}
|  |- r  S1  {2} [tc5mma]
|  |- a  S1  {1}                     <- regain, yielded through the iter-arg
|  |- EXIT ... yield{a S1}
|- r  S2  {@0.1} [none]              <- post-loop handoff of the final value
|- a  S2  root                          back to the root reader
|- R m0  ttng.tmem_load root
SEMAS: S0{count=1} S1{count=1 entry inherit=root} S2{count=1}
BACKING: numStages=2
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

Always correct — and can be suboptimal. Whether anything better is
possible is decided by following one value: the regain's token `%t1`.
Who uses it?

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

Any *other* user of `%t1` blocks the move — with one rescue documented
under the structural conditions: a final-permission acquire with a proven
bridge can stand in for an outside consumer. That is the core of the
decision; the checklist below is its systematic form plus the structural
placement conditions, run per loop (`buildUniformHold`). The decision runs on the DAG before anything is
emitted, so every dump figure in this section shows a loop *after* its
decision; the carried shape survives only where the decision fell through
— the `gated(...)` loops. Where the resulting acquires and releases land
is the hold rule:

```text
cut token ownership where execution context changes
hold = the maximal run of accesses between cuts
acquire before the hold's first access
release after the hold's last access
```

### The decision, per region

**Before the searches: does a token have to cross this region at all?**
If the body contains no acquire of the group and no nested region that
already has a crossing, no token passes through the boundary — there is no
crossing, and nothing below applies (a body holding only releases records
none).
One hold simply covers the whole loop: acquired before it, every
iteration's accesses run under that one token (its value dominates the
body), released after it. The dump row of such a loop carries no
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
The last acquire or nested region that can produce that token is recorded as
the child chain's final node (`computeCrossings`, `chainFinalForComp`) — group
events may still trail it (the `trailing-use` shape of the gated reasons
below). The hold decision later determines whether a loop takes a token as an
iter-arg and returns one, or has no token iter-arg or result.
The decision considers the `%t1` trace from the opening in two directions. An
`if` considers only the outside direction (`pruneDeadIfCrossings`). Loop hold
analysis (`buildUniformHold`) considers the inside direction first and reaches
the outside direction only if earlier checks pass:

```text
inside    after the node that produces the boundary token: does anything
          still use the token there?
outside   beyond the boundary: does anything consume the token?
```

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
nothing; the loop's result — unused. Both searches empty, the structural
conditions below hold. The move deletes the regain, the yield's token,
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

**An inside use keeps the token iter-arg and result. An outside use normally
does too.** The final-permission construction described below is the one case
that can satisfy an outside use without keeping the loop's token result. The
two searches are:

- *inside the body*, between the regain and the yield: an access to the
  group's memory, a release of one of the group's semaphores, or a nested
  `for`/`if` with a crossing (`hasTrailingCompUse`). An access or release still
  uses the token at the bottom; any nested crossing is conservatively treated
  as a trailing use → reason `trailing-use`;
- *outside the loop*, through the result: something after the loop
  consumes the token the loop returns (`regionResultConsumedAfter`). If the
  feeding and regain semaphores are the same, this produces reason
  `result-consumed`. If they differ, the final-permission construction can
  instead provide the needed token after the loop; if that construction fails,
  a structural reason below keeps the token iter-arg and result.

(In code these two searches sit inside a longer checklist run in a fixed
order; the complete reason list below is that order.)

Each failure, as a complete minimal loop (entry acquire seeds iteration
zero; the semaphore is created initially released, so waits and releases
balance):

```text
inside search fails — trailing-use:

%t0 = a S root                    ; entry
for (%t = %t0) {
  W buf [%t] {0}
  r  S %t {0}
  %t1 = a S {0}                   ; deletion target
  R buf [%t1] {0}                 ; still needs the token, inside
  yield %t1
}

outside search fails — result-consumed:

%t0 = a S root                    ; entry
%tN = for (%t = %t0) {
  W buf [%t] {1}
  r  S %t {1}
  %t1 = a S {1}                   ; deletion target
  yield %t1                       ; feeds TWO customers
}
r  S2 %tN {1}                     ; the second customer: outside,
%u = a S2 root                    ;   unreachable by the move
R  buf [%u] root
```

**Keeping a token iter-arg and result is the fallback** — the inside search
finds another use, an outside use cannot use the final-permission construction,
or a structural condition fails (below). The loop keeps the default shape, and
the dump prints the *first failed check* as
`gated(reason)`. The label changes nothing in the emitted protocol — it exists
so that a human asking "why didn't this loop get the point-of-use shape?"
reads the answer instead of repeating the search the pass already did.

### Composition: nested regions

Nesting adds no new rules. Hold decisions run innermost-first:
`computeHoldRules` visits regions in post order (`forEachRegionPostOrder`
— children before their parent). Once a nested loop has been decided, its
parent sees the loop as one node and needs only its boundary behavior:

```text
holdrule{pointofuse->...} or holdrule{passthrough-drop}
    the parent passes no token into the loop and receives no token result

holdrule{gated(reason)}
    the loop takes a token as an iter-arg and returns a token as its result
```

If the final child loop passes no token, the parent removes its own token slot
without running the searches and prints `holdrule{passthrough-drop}`. Otherwise,
the parent runs the same single-level searches against the child region node.

**Child takes no token from its parent and returns none.**
`test/NVWS/insert_semas_nested_ws_inner_loop.mlir` `@nested_ws_inner_loop`.
What the default would emit — both loops with token iter-args and results:

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

Trace, innermost first. The inner regain `%u2`: between it and the inner
yield — nothing (`hasTrailingCompUse`). Is the inner result consumed?
`regionResultConsumedAfter` walks the nodes after the inner loop: there
are none, so it recurses into the parent — the inner loop is the outer
crossing's final (the outer's bottom re-acquire position) — and
asks the same question after the outer loop:
nothing involving the group follows it either. Not consumed. So the
inner goes POINT_OF_USE and stops yielding. Now the outer: `%t` entered only to
feed the inner loop — which no longer takes a token in — and `%ti` no longer
exists to forward. The outer therefore removes its token iter-arg and result
without running the searches; the dump prints `yield{drop}`. As emitted:

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

The difference from the no-crossing case: there, no acquire occurs inside;
the body uses one token acquired before the loop. Here the acquires and
releases remain inside the nested loop, and the outer loop has no token
iter-arg or result.

**Child takes a token from its parent and returns one.** Its region node
provides that token at the parent level, and the parent's searches run on it
like on any other node. The consumption test is mechanical:
`regionResultConsumedAfter` reports
consumed the moment the walk after the region meets a release, a group
access, or a nested `for`/`if` with a crossing, and not-consumed
at an acquire. Both failure reasons of the
single-level trace, in one real test — the node right after the inner
loop is a release, `r S2 {1}`, hence consumed —
test `@outer_sourceful_alloc_inner_loop_reentry`; the rows that prove each
check are marked:

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

**When the yield's token source is itself a nested region** — a `for` or
`if` rather than a plain acquire — that region is checked branch-by-branch
before the inside-use check. It blocks the move unless each branch's final
node returns the hold owner's token and no group event follows it
(`isHoldTransparentRegion`); a region that passes is transparent to the hold,
one that fails gates the loop (`region-not-transparent`).

### Reading a loop row at sight

```text
holdrule{pointofuse->op}     EXIT: yield{native}   acquire at op; no token iter-arg/result
holdrule{gated(reason)}      EXIT: yield{...}      token enters as iter-arg and leaves as result
holdrule{passthrough-drop}   EXIT: yield{drop}     no outer token slot; final nested loop returns none
no holdrule, no thread{}     plain EXIT            token acquired before loop remains valid in body
```

One qualifier on the gated row: when the yield's token source is itself a
nested region — the case just above — its `EXIT` reads `yield{scf.for}`
rather than `yield{a S<n>}`. Real in
the test `@three_level_reentry_without_post_access`, whose middle loop is gated and
yields its inner loop's result:

```text
|  |  |- scf.for ... holdrule{gated(entry-sema-mismatch)}
|  |  |  |- scf.for ... holdrule{gated(entry-sema-mismatch)}
|  |  |  |  |- ...                     ; the inner protocol
|  |  |  |- EXIT ... yield{scf.for}    <- the token source is the region node
```

### The structural conditions, and every gated reason

Two definitions the conditions use: the *feeding acquire* is the acquire
above the loop that supplies the token the first iteration consumes
(`HoldFeed`); the *hold prefix* is the run of body nodes from the top of
the body down to the group's first acquire, evaluated before the move
(`HoldPrefix`) — the accesses the moved acquire will protect.

The complete list of `gated(reason)` values, in the order the checks run
(`buildUniformHold`; the first failure is the one printed). The prefix
reasons come from one scan (`analyzeHoldPrefix`): `region-crossing` and the
prefix form of `region-not-transparent` abort that scan the moment they are
seen, so they preempt `no-buf`/`rel-count`/`rel-before-buf`, which are
evaluated only after the scan completes. `prefix-not-buffer-view` is checked
once the prefix returns clean; the stage check is last:

```text
non-ws-scope           the loop is not inside a WS-tagged nest. A policy
                       gate: the move is only attempted where partitions
                       pipeline against each other; elsewhere the default
                       shape is kept without analysis
if-encloser            an enclosing `if` cannot drop the token
                       (allEnclosersCanDrop)
no-final               the crossing has no bottom re-acquire node at all
final-permission       the bottom node is a final-permission acquire (below)
region-not-transparent the bottom node is a nested region the hold cannot
                       pass over unchanged (isHoldTransparentRegion) — or
                       not an acquire at all; also reported when such a
                       region sits in the prefix
trailing-use           the token has a user inside, between the regain and
                       the yield (access, release, or nested region)
entry-consumed         no usable feeding acquire: scanning upward from the
                       loop, an access to the group is met before any
                       acquire
region-feed            the same upward scan meets a region crossing first
release-feed           the same upward scan meets a release first
no-entry-acquire       the same upward scan runs out of nodes
entry-sema-mismatch    either a transparent region tail does not resolve to
                       one common returned semaphore, or the feeding
                       semaphore differs from the regain semaphore and the
                       final-permission rescue's four conjuncts do not all
                       hold (below) — a region tail, no consumer after the
                       loop, a feed owner that is not the hold owner, or no
                       bridge acquire found after the loop
result-consumed        feeding and regain use the same semaphore, but the
                       move still needs an acquire after the loop (normally
                       because its result is consumed outside); different
                       semaphores take the final-permission path below
region-crossing        a token-threading region sits in the prefix before
                       any access (the prefix form of region-not-transparent
                       aborts the scan here too, ahead of the three below)
no-buf                 the prefix contains no access at all
rel-count              the prefix's releases do not stand in for exactly
                       one wait — a release with arrive_count n counts n
                       times (analyzeHoldPrefix adds max(1, count)), so
                       the parent loop of
                       @release_multiplicity_unified_fanin_regain gates
                       here on its single release node, r S0(2) (zero
                       expected when the prefix ends in a nested region)
rel-before-buf         a prefix release precedes the first access
prefix-not-buffer-view the first prefix node is not a single buffer view —
                       a node whose op only views the backing, writing no
                       value of its own (prefixRowIsSingleBufferView) —
                       e.g. a sourceful TMEM allocation in an inner loop
                       that does not itself carry the WS tag
cross-stage-final-acquire
                       moving the acquire to the first access would also
                       require a same-owner acquire after the loop, but the
                       two acquires have different loop.stage values
```

One construction can save a loop whose feeding semaphore differs from its
regain semaphore: a *final-permission acquire*
(`Node::finalPermissionAcquire`) — a single acquire after the loop that
stands in for the token the loop no longer returns. It applies only when all
four conjuncts hold: (a) the hold is not a *region tail* — a hold whose
bottom node is a nested region standing in for the plain regain acquire, the
transparent-region case above (such a hold dumps as
`holdrule{pointofuse->...:regionTail}`, e.g.
`@if_split_yield_routing_three_partitions` in
`test/NVWS/insert_semas_if_split_metadata.mlir`); (b) something after
the loop consumes the result (the same fact that would otherwise print
`result-consumed`); (c) the feeding acquire's owner — or, for an entry feed,
its semaphore's `inheritStamp` — is the hold owner; and (d) an acquire of the
feeding semaphore by the hold owner is found after the loop with no other
group event trailing it (`findBridgeAcquireAfter`). That found acquire is the
bridge; the matching release of the loop's own semaphore is then created
after it by `materializeFinalPermissionAcquires`
(`Hold::bridgeAcquire`/`bridgeRelease`; a hold that records a
final-permission acquire dumps as `:finalAcquire`, and the bridge as
`:entryBridge`). If any conjunct fails, `entry-sema-mismatch`.

Those four conjuncts establish only that the final-permission construction
is structurally possible. A later schedule check can still require a token
iter-arg and result. It runs for a plain inner loop inside a WS scope when point-of-use
needs an acquire after the loop and the first access has the hold owner. If
both schedules are known and the first access's `loop.stage` differs from
that owner's completion stage at loop exit, the result is
`gated(cross-stage-final-acquire)`. The moved in-loop acquire and the
post-loop acquire would otherwise be two same-semaphore, same-partition
acquires in different stages, a shape `AssignStagePhase` cannot treat as
one direct loop-body protocol. Only `loop.stage` is compared; a cluster
difference or an unknown schedule does not trigger this fallback.

The construction, and the different-semaphore `applyHoldRulePlacement`
seeding of [Loops and the first acquire](#loops-and-the-first-acquire),
real in one figure — `test/NVWS/insert_semas_nested_ws_inner_loop.mlir`
`@nested_ws_inner_loop_parent_continuation`. The inner loop's regain is
`S1`, but the acquire feeding it from above is the outer entry `a S3 root`
— a different semaphore — so the feeding acquire is kept and `S1` is
marked initially released instead (`inherit={@1.1}` is the hold owner, the
third `inheritStamp` source of the notation); the `a S1 {1}` after the
inner loop is the final-permission acquire, and the `a S3 {1}` with the
`r S1 {1}` after it is the bridge:

```text
|- a  S3  root  ; entry               <- the feeding acquire, KEPT above the loop
|- scf.for (WS, tag=1) ... holdrule{gated(trailing-use)}
|  |- scf.for ... holdrule{pointofuse->ttng.tc_gen5_mma:finalAcquire:entryBridge}
|  |  |- a  S1  {1}                   <- the regain, moved to the first access;
|  |  |- W m0  ttng.tc_gen5_mma {1}      S1 initially released seeds inner
|  |  |- r  S0  {1} [tc5mma]             iteration zero
|  |  |- a  S0  {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r  S1  {0} [none]
|  |  |- EXIT ... yield{native}
|  |- a  S1  {1}                      <- final-permission acquire: the token for
|  |- r  S2  {1} [tc5mma]                the nodes below; this release is the
|  |- a  S2  {0}                         result consumer of conjunct (b)
|  |- R m0  ttng.tmem_load {0}
|  |- r  S3  {0} [none]
|  |- a  S3  {1}                      <- the bridge acquire (of the feeding S3),
|  |- r  S1  {1} [none]               <- and the bridge release of S1 after it,
|  |- EXIT ... yield{a S3}               re-releasing S1 for the next outer
                                         iteration's inner loop
SEMAS: S0{count=1} S1{count=1 entry inherit={@1.1}} S2{count=1} S3{count=1 entry inherit={@1.1}}
```

## Backing depth

`computeBackingPlan` chooses the depth. `buffer.copy`, when present, is
authoritative: `@fused_alias_depth_two` (worked below in
[Exact-alias handoffs](#exact-alias-handoffs-the-release-signals-the-successor-copy))
carries it on its allocation —

```text
%m0 = ttg.local_alloc {buffer.copy = 2, buffer.id = 500}
```

— and its dump closes with the depth taken straight from it:

```text
BACKING: numStages=2
```

Without `buffer.copy`, the default depth is 1 — `numStages` starts there
and nothing lowers it. Two rules can change what the later analyses see.

First, synchronized TMEM may be double-buffered, only on the default NVWS
path (see [Two NVWS paths](../nvws-aws-overview.md#two-nvws-paths));
Meta-NVWS adds no TMEM depth of its own. The decision is a trace over the
group's MMA users, run here on the
`@root_entry_accumulator_adopts_without_semaphore_handoff` group above.
Follow its one MMA user, `ttng.tc_gen5_mma {2}`: it sits directly in the
WS `scf.for`; the `128x128xf32` accumulator is not read-modify-written in
that loop; accumulator multibuffering is structurally possible; the loop
carries no disallow-multibuffer flag; and two copies
(`2 * 128 * 128 = 32768` cells) fit TMEM's 128x512 cells alongside the
TMEM blocks already planned — every check passes, and the group's dump
closes with the row shown in its figure:

```text
BACKING: numStages=2
```

The checklist that trace ran (`isMultiStagedGroup`), for every MMA user of
the group's allocations whose immediate parent is an `scf.for`: the
accumulator is not read-modify-written in that loop
(`hasAccReadModifyWrite`), accumulator multibuffering is structurally
possible (`isAccMultibufferingPossible`), the enclosing WS loop does not
carry the disallow-multibuffer flag, and `canDoubleBufferAcc` — two copies
of the `blockM x blockN` accumulator still fit TMEM's 128x512 cells
alongside the TMEM blocks already planned, and the op is not a scaled MMA
with `blockN` 256.

Second, a local backing written by a TMA load records the depth that
`LowerSemaphore` will give it (`g.semaphoreDepth = max(1,
lowerSemaphoreNumStages)`; see the
[pass order](../nvws-aws-overview.md#pass-order)), so the recurrence
analysis below sees the copies that will actually exist. This is a
separate field next to `numStages`, and the dump does not print it —
`test/NVWS/insert_semas.mlir` `@local_release_after_mma` (its `buffer.id
= 102` group) is this shape, and its `BACKING` row still reads
`numStages=1`:

```text
|- W m0  nvws.descriptor_load {0}
|- r  S0  {0} [tma_load]        <- the TMA-load release that triggers the rule
   ...
BACKING: numStages=1            <- numStages untouched; the recorded
                                   semaphoreDepth is not a dump field
```

## Buffer-stage offsets and the pipeline schedule

Semaphore edges are not SSA dependencies, so they must be projected onto
the existing loop schedule. Two facts anchor the projection; the rest of
the algorithm is the slack check the example below traces, restated as
steps 3–5 after it:

1. Source anchors use the last physical completion operation
   (`completionAnchor` when one was recorded); destination anchors use the
   destination owner's first real operation after its acquire — the start
   of the hold that acquire opens.
2. For an edge that closes a loop, reconstruct which buffer stage the
   release addresses and find the first future iteration whose acquire
   addresses the same buffer stage; the iteration count between them is
   the recurrence distance `d` (`computeRecurrenceDistance`).

### Example: recurrence distance, slack, and the cluster raise

`test/NVWS/insert_semas_recurrence_schedule.mlir` `@one_slot_recurrence`
(its `_errors.mlir` twin is the rejected case). One depth-1 backing
(`buffer.copy = 1`), producer at `loop.stage 0`, last consumer at
`loop.stage 1`; the loop-closing edge runs from the `loop.stage 1` read
back to a later iteration's `loop.stage 0` store. Iteration `i` writes
buffer stage `i mod depth`:

```text
depth 1:  iter0 -> s0 | iter1 -> s0 | ...   every iteration reuses buffer stage 0    d = 1
depth 2:  iter0 -> s0 | iter1 -> s1 | iter2 -> s0 (the buffer stage iter0 released)  d = 2

slack = d + Sv - Su, with Su = 1 (the releasing read), Sv = 0 (the store):

PASSING    depth 2:  slack = 2 + 0 - 1 = 1 > 0    headroom; nothing changes
ZERO-SLACK depth 1:  slack = 1 + 0 - 1 = 0        iter i's read and iter i+1's
                                                  store execute in the same
                                                  steady-state iteration of the
                                                  pipelined loop -> ordering
                                                  constraint: read before store
REJECTED   Su = 2 (errors twin):
                     slack = 1 + 0 - 2 = -1 < 0
  error: nvws-insert-semas: fixed loop.stage assignment cannot satisfy
         semaphore handoff (source stage 2, destination stage 0,
         recurrence distance 1)
```

The loop's scheduled rows, with the `loop.stage`/`loop.cluster` they start
from (read off the `.mlir`; the labels are the trace's names):

```text
row            owner  starting loop.stage, loop.cluster   op
store          {3}    loop.stage 0, cluster 1              ttg.local_store
first          {1}    loop.stage 0, cluster 1              ttg.local_load (%first)
consume_first  {1}    loop.stage 0, cluster 1              "consume_first"(%first)
last           {1}    loop.stage 1, cluster 2              ttg.local_load (%last)
consume_last   {1}    loop.stage 1, cluster 2              "consume_last"(%last)
```

The zero-slack case records the constraint `last-read -> store` and joins
it with the SSA edges of that same steady-state iteration. The fixed point
(`legalizeLoopSchedule`, driven by `finalizeSyncSchedule`) raises only
destination `loop.cluster` values
(`cluster(dst) >= cluster(src) + sep`, `sep = 1` when the source sits after
the destination in the block, else `0`):

```text
start:   store 1   first 1   consume_first 1   last 2   consume_last 2
pass 1:  {last -> store}, last is AFTER store  -> store := 2+1 = 3
pass 2:  {store -> first}                      -> first := 3
         {first -> consume_first}              -> consume_first := 3
pass 3:  no change -> converged

result:  store/first/consume_first  loop.stage 0, cluster 3
         last/consume_last          loop.stage 1, cluster 2   (loop.stage untouched)
```

The steps the trace just performed, as the checklist (continuing 1–2
above):

3. With source `loop.stage` `Su` and destination `loop.stage` `Sv`, require
   `d + Sv - Su >= 0`.
4. Negative: rejected. Positive: nothing changes. Zero: add a
   source-before-destination ordering constraint.
5. Solve the zero-slack semaphore and SSA constraints together by raising
   destination `loop.cluster` values to a fixed point; `loop.stage` never
   changes.

### Circular groups: the production counter assigns stage offsets

For circular local groups sharing one physical `buffer.id`, SYNC-DAG
validates the group set (`assignCircularStageOffsets`: common type and depth,
unique `buffer.start` values, producer order), then walks accesses in
program order: writes advance a
global production counter, and each read refers to its group's latest
produced value. The resulting stage offset is stored on the access node and
its adjacent acquire/release nodes before any IR is emitted.

`test/NVWS/insert_semas_circular_smem.mlir` `@circular_tutorial_1_1_to_2_2`
— K and V share one depth-2 ring (`buffer.start` 0 and 1); each circular
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

### Exact-alias handoffs: the release signals the successor copy

For SMEM groups whose copies are handed off as exact aliases of one
allocation, SYNC-DAG likewise assigns stage offsets
(`assignAliasedHandoffStageOffsets`) so each release signals the buffer stage
addressed by the acquire it satisfies; unsupported or path-dependent
buffer-stage schedules fail with `cannot derive staged exact-alias handoff
slots` rather than being guessed (no lit test covers the failure).
`test/NVWS/insert_semas_fused_alias_handoff.mlir` `@fused_alias_depth_two` —
two members cover the same interval, the exact-alias shape:

```text
members:    m0[0,128)   m1[0,128)
pieces:     P0=[0,128){m0, m1}
footprints: an access through either member touches P0
```

The read-to-next-write release carries `stage-offset=1` because it must
signal the copy the *next* acquire will address, not the copy the read used.
The rule the rows below trace: the ring advances at each store — with two
stores per iteration on a depth-2 ring, the `m0` store fills one copy (call
it A) and the `m1` store the other (B) — and each release's offset is the
distance from its own copy to the copy addressed by the acquire it
satisfies. The full body, each release's derivation beside its row (`S3` is
created initially released, seeding iteration zero — `S3{count=1 entry
inherit={@0.4}}`):

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S3  {4}  stage-offset=0
|  |- W m0  ttg.local_store {4}            ; fills copy A of the ring
|  |- r  S0  {4} [none]  stage-offset=0    <- satisfies a S0 {2} below, same
|  |- a  S0  {2}  stage-offset=0           ;    iteration: the read uses copy A,
|  |- R m0  ttg.local_load {2}             ;    just filled -> offset 0
|  |- r  S1  {2} [none]  stage-offset=1    <- satisfies a S1 {4} below: the m1
|  |- a  S1  {4}  stage-offset=0           ;    store fills the ring's NEXT copy,
|  |- W m1  ttg.local_store {4}            ;    B, one past the read's A -> +1
|  |- r  S2  {4} [none]  stage-offset=0    <- satisfies a S2 {2} below: the read
|  |- a  S2  {2}  stage-offset=0           ;    uses copy B, just filled -> 0
|  |- R m1  ttg.local_load {2}
|  |- r  S3  {2} [none]  stage-offset=1    <- satisfies a S3 {4} of the NEXT
|  |- EXIT ... yield{native}               ;    iteration: its m0 store fills the
                                           ;    copy after B, wrapping the ring
                                           ;    -> +1
SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=1 entry inherit={@0.4}}
```

### Mixed-depth TMEM aliases: schedule validation across split groups

The different-`buffer.copy` TMEM aliases that ACCESS-DAG split into
separate groups (see [ACCESS-DAG](access-dag.md#groups)) get one
cross-group step — an exception to per-group independence.
`addMixedDepthAliasScheduleEdges` (run by `finalizeSyncSchedule`)
validates, per shared `buffer.id`:

```text
exactly two split groups, each a single-member TMEM group (the
  diagnostics name these "logical channels")
a unique physical owner by extent-times-depth containment and element
  width (canOwnMixedDepthTmem) — owner and reuser name the two split
  groups here (the owner group's interval contains the reuser group's),
  not partition owners
distinct logical copy depths (owner-group and reuser-group numStages
  differ)
one writer and one reader per split group, run by exactly two partitions
  crosswise: the owner group's writer partition is the reuser group's
  reader partition and vice versa
all four accesses direct in ONE scheduled loop body, ordered
  owner-group write -> reuser-group read and owner-group read ->
  reuser-group write, each with fixed loop.stage/loop.cluster
same-iteration slack:  reuser-group write pipeline stage - owner-group
                       read pipeline stage >= 0 (zero adds an ordering
                       edge to the legalization)
backedge slack:        1 + owner-group write pipeline stage -
                       reuser-group read pipeline stage > 0
```

Each violated line is a loud diagnostic (each naming `mixed-depth TMEM`);
the passing case contributes at most one zero-slack ordering edge to the
same cluster-raising fixed point the recurrence example above feeds. This
step leaves no dump field and has no worked example in the corpus.

### Protocol schedules

Each ordinary owner-qualified acquire inherits the pipeline stage of its
schedule anchor — the destination owner's first real operation after that
acquire (`scheduleAnchor`, the start of the hold it opens), the same anchor
step 1 above uses for destinations; root entry acquires stay unscheduled.
An ordinary acquire with no real operation after it in its chain — a gated
loop's bottom regain immediately before `EXIT` — instead uses the schedule
cached for its owner. A final-permission acquire is a separate case: it
always uses the owner's cached schedule, even when real operations follow
it. Each release also uses its owner's cached schedule
(`assignSyncScheduleChain`). The schedules are
not dump fields — the dump prints only `stage-offset` on protocol rows;
they land as `loop.stage`/`loop.cluster` attributes on the emitted acquire
and release ops. `@one_slot_recurrence` above, its protocol rows annotated
with the schedules these two rules assign:

```text
|- a  S1  {3}                    <- loop.stage 0, cluster 3: inherits the
|- W m0  ttg.local_store {3}        store's hold anchor, the store (raised
   ...                              to cluster 3 in the fixed point above)
|- r  S1  {1} [none]             <- loop.stage 1, cluster 2: inherits {1}'s
                                    last completion, the %last read
                                    (stage 1, cluster 2)
```

The lit test checks those attributes on the emitted ops
(`test/NVWS/insert_semas_recurrence_schedule.mlir`):

```text
CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>}
CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<none>] {arrive_count = 1 : i32, loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
```

A final-permission acquire uses the schedule cached for its owner at that
point (`assignSyncScheduleChain`), even though the acquire itself sits after
the loop. The cache follows the same control-flow rules used by the earlier
point-of-use safety check: a plain loop propagates schedules found in its
body, a WS-tagged loop does not leak its body schedule outward, and an `if`
uses the then-branch value when present, with the else branch filling only a
missing owner. `ownerCompletionScheduleAtLoopExit` mirrors those rules when
it predicts the post-loop acquire's stage during hold selection.

This is the schedule distinction behind
`gated(cross-stage-final-acquire)`. In
`test/NVWS/insert_semas_meta_fa_fwd.mlir`, the `buffer.id=4` path keeps
POINT_OF_USE because its first access and loop-exit owner completion are both
at stage 0. The `buffer.id=5` path keeps the token passing through the loop
because its first access is at stage 0 while its loop-exit owner completion
is at stage 1. A tail
adjustment can later raise a final acquire's `loop.cluster` within its
already chosen `loop.stage`
(`placeFinalAcquireAtLaneExit`).

The `a S1 {1}` row after the inner loop in the
`@nested_ws_inner_loop_parent_continuation` figure
([The structural conditions](#the-structural-conditions-and-every-gated-reason))
is a final-permission acquire. EMIT-IR copies these schedule and offset
facts; its one remaining schedule exception is the release-schedule
fallback of the loop-scheduler workaround (see [EMIT-IR](emit-ir.md)).

## Code map

[`InsertSemasSyncDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp):

- `walkChain`, `applyTouch`, `VersionSource`, and `retentionEligible`
- `reduceEdges` and `buildEdgesAndSemas`
- `insertEntryAcquires` and `computeCrossings`
- `computeHoldRules`, `buildUniformHold`, `analyzeHoldPrefix`, and
  `applyHoldRulePlacement` (the walk state `PieceGame`/`WaveSt`/`hadWave`
  and the hold records `HoldFeed`/`HoldPrefix` are file-local here)
- `ownerCompletionScheduleAtLoopExit`
- `computeBackingPlan`
- `assignCircularStageOffsets`, `assignAliasedHandoffStageOffsets`,
  `addMixedDepthAliasScheduleEdges`, and `finalizeSyncSchedule`
- `buildSyncDag`
- the DAG dump used throughout: `NVWS_INSERT_SEMA_DUMP_DAG=1`
  (`dumpDagTree`)
