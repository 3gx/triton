# SYNC-DAG

## Purpose

SYNC-DAG converts OWNER-DAG ownership changes into a balanced semaphore
protocol. It also decides whether semaphore tokens cross region boundaries,
the backing depth, stage offsets, and the pipeline-stage placement required by
the software pipeliner:

```text
input IR ─► ACCESS-DAG ─► OWNER-DAG ─► SYNC-DAG ─► EMIT-IR ─► output IR
            memory facts   owners      edges, semaphores,   render
                                       token holds, schedule
```

This document is self-contained: the notation section below redefines what it
uses. Full definitions live in the
[InsertSemas overview](overview.md#core-objects) (the model objects) and the
[NVWS-AWS terminology](../nvws-aws-overview.md#terminology) (buffer stage,
stage offset, pipeline stage).

## Notation

Every diagram in this document uses the pass's own dump format. Figures are
trimmed excerpts — unrelated groups, `parts{...}` fields, and some `ENTER`
rows are elided — of the dump produced by running the listed lit test under
`NVWS_INSERT_SEMA_DUMP_DAG=1`:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt <test.mlir> -split-input-file \
    -allow-unregistered-dialect --nvws-insert-semas
```

Object shorthand (full definitions in the overview):

```text
group         the allocations analyzed together (ordinarily those sharing one buffer.id)
backing       the mutable SMEM/TMEM allocation the group guards
m0, m1        member: one allocation of the group; m0[0,64) = its interval, in
              leading-dimension elements (local memory) or TMEM columns
P0, P1        piece: a disjoint interval of the backing; {m0, m1} = members covering it
              (an access through a member touches ALL of that member's pieces;
              a group's pieces always connect through shared members, so the
              group is the unit of synchronization — see ACCESS-DAG,
              "Pieces must connect")
{0}, {1}      owner: partition 0, 1 of the enclosing WS loop; root = no partition
{@0.1}        tag-qualified owner: partition 1 under WS tag 0 (printed where the
              anchor sits outside that loop, e.g. on SEMAS lines)
producer      the owner that last wrote a piece (before any write: its first toucher)
holder        an owner currently allowed to access a piece (producer + joined readers)
hold          acquire -> accesses -> closing release, all by one owner: the
              interval during which that owner holds the group's token
```

Dump rows (each row is one node of the group's program-order chain):

```text
|- scf.for (WS, tag=0)      loop node; (WS, tag=N) marks a warp-specialized loop
|- ENTER / |- EXIT          region boundary markers around each child chain
|- W m0  ttg.local_store {1}   access node: W(rite)/R(ead), member, op, owner
|- a  S1  {0}               acquire of semaphore S1 by partition 0
|- a  S0(2)  {0}            acquire with pending count 2 (waits for 2 releases)
|- r  S0  {1} [tma_load]    release by partition 1; [..] = completion kind
|- a  S3  root  ; entry     unpartitioned entry acquire, placed before the loop
pieces{P0:W:{0}}            region row: per-piece merged effect and owner
parts{0, 1}                 region row: partitions that execute this region
thread{{0}}                 region row: the group's token is live here (a
                            crossing); the hold decides whether it actually
                            passes the boundary
holdrule{...}               loop row: the crossing's hold outcome (see Crossings)
yield{X}                    EXIT row: what this chain returns for the group's
                            crossing
stage-offset=N              stage offset assigned to this protocol row
S<n> / E<n>                 semaphore names; E<n> marks a dedicated entry semaphore
SEMAS: S1{count=1 entry inherit={@0.0}}   the group's semaphore summary:
                            pending count; entry = initially released;
                            inherit = the owner emission stamps on the
                            entry acquire (normally the group's first
                            accessor; for a mid-body entry, the acquiring
                            owner)
BACKING: numStages=N        depth chosen for the backing allocation
```

Pseudo-IR in this document strips types and attributes; `%t = a S0 {1}` /
`r S0 %t {1}` are the token-producing acquire and token-consuming release,
`W m0 [%t]` is an access performed while holding token `%t`, and
`for iter_args(%t = %t0)` is a loop carrying the token.

Do not confuse the two layers: `P0` and edges are analysis labels that
never appear in the IR. The semaphore *token* is an IR value — each emitted
acquire returns a fresh one, covering one hold of its group, and the
matching release consumes it.

## Ownership walk

The walk runs once per group, in program order over its chains (`walkChain`).
At each access it answers one question: *must this access wait for an
earlier access — and for which one?* Every "yes" becomes an edge. Edges are
raw material, not yet semaphores: the next section drops the redundant ones
and merges the survivors into acquire sites, so several edges can end up as
one semaphore, or as none. An access must wait in three situations:

1. **Read after write** — a reader waits for the write that produced the
   data it consumes.
2. **Write over held data** — a write replaces the data, so it waits for
   every owner that has the current data. All of them: anyone skipped could
   still be reading when the write lands.
3. **Re-entry** — an owner accesses, other owners run after it, then it
   accesses again. The second access waits for the owner that ran last.
   Unlike rules 1 and 2 this is not a data hazard — a re-read touches data
   nobody changed. It is an obligation of the emitted protocol: every
   access runs under a live token, inside a hold, and this owner's hold is
   already closed — its release is exactly what let the others run. So the
   returning access needs a fresh acquire, and the release that feeds it
   comes from the owner that ran last.

The walk carries one fact per rule:

```text
producer   per piece   the owner that last wrote it             -> rule 1
holders    per piece   the owners that have the current data:   -> rule 2
                       the producer, plus every reader since
                       the last write
wave       per group   the group's most recent partition-owned  -> rule 3
                       access, and the owner that ran it
```

Producer and holders together are the piece's *game* (`PieceGame`),
matching the contract in the [overview](overview.md#contract); the wave
(`WaveSt`) is a single slot for the whole group. Two properties keep the
traces below readable:

- Updating this state is bookkeeping, not synchronization. Each access
  first consults the state — that is what may emit an edge — and then
  updates it for the accesses that follow. A reader joining `holders`
  emits nothing: it is a note that a future write owes that reader a wait.
- Between writes, `holders` only grows; readers join and nobody leaves. A
  write resets the set to the writer alone — after paying rule 2 — and the
  number of *other* holders just before the write is that handoff's fan-in.

How the wave updates, and how it is read:

```text
after every partition-owned access   wave = (that access, its owner)
                                     reads move it too

after a nested for/if node           one partition owned all its accesses
                                       -> wave = that partition
                                     mixed owners inside
                                       -> wave = unknown

at an access by owner Q              "the wave has moved" = wave is not Q:
                                     someone else ran since Q's last access
```

Unknown counts as moved: with mixed owners inside a region, someone ran but
no single owner can be named, so the walk assumes the worst.

Each touch of a piece advances that piece's game (`applyTouch`). These are
the precise forms of the three rules, with the edge-source and dedup
details the summary above compresses:

```text
first touch of a piece:
  edges:  none
  state:  producer = the toucher; holders = {toucher}

write by owner P:
  edges:  H's last access -> P, for every OTHER holder H       (rule 2)
            skipping an H whose last access an earlier edge already
            orders before P — redundant — unless the wave has moved,
            which voids that shortcut
          if that produced no edge and the wave has moved, the write
          is a re-entry: wave's last access -> P                (rule 3)
            (P was the piece's only holder — overwriting its own
            data — while others were active on the group)
  state:  producer = P; holders = {P}

read by an owner already holding the piece:
  edges:  none — unless the wave has moved, a re-entry:         (rule 3)
            producer's last access -> this read; when the re-reader IS
            the producer (or the producer has no access in this chain),
            the source is the wave's last access instead
  state:  the holder's last access moves to this read

read by a new owner:
  edges:  producer's last access -> this read                   (rule 1)
            (when the producer has no access in this chain, the seed
            holder is the source; see the nested-region seeding below)
  state:  the reader joins holders
```

An edge records the source access's completion kind — `none`, TMA, MMA, and
so on (`EdgeRec::payloads`); it selects how the release is lowered (plain
arrive, MMA commit, or TMA completion path — the `[none]`/`[tc5mma]` tags
on the dump's release rows).

### Example: two-partition handoff

`test/NVWS/insert_semas.mlir` `@local_loop_carried_and_result` (group
`buffer.id=104`) — partition 0 stores, partition 1 loads, every iteration:

```mlir
%buf = ttg.local_alloc {buffer.id = 104}
scf.for ... {
  ttg.local_store %arg, %buf {loop.stage = 0, ttg.partition = [0]}
  %l = ttg.local_load %buf   {loop.stage = 1, ttg.partition = [1]}
  ...
} {tt.warp_specialize, ttg.warp_specialize.tag = 0}
```

Walk trace — one row per event; middle column names the state fact that
fired the rule, right column the emitted edge (`@next` = next-iteration
instance, from the loop recurrence handled at `EXIT`):

```text
walk                     state fact                    edge (rule)
ENTER {P0:W:{0}}         seed: game live, holder {0}   —
W m0 {0}                 write; sole holder = seed {0} —   (producer={0}, holders={0})
R m0 {1}                 read by new owner             e1: store@{0} -> load@{1}      (rule 1)
EXIT (recurrence)        holder {1} != next            e2: load@{1} -> store@{0}@next (rule 2)
                         iteration's writer {0}
```

Edge `e1` becomes semaphore `S0` ("data ready"), `e2` becomes `S1` ("buffer
free again"). The resulting SYNC-DAG (dump):

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

`S1` is created initially released (`entry`) so iteration zero's `a S1`
succeeds before any release has run — see
[First acquire and initial release](#first-acquire-and-initial-release).

### Example: the wave moves, a reread re-acquires

partition 0 writes, partitions 1 and 2 read, then partition 0 *re-reads* the
piece it still holds:

```text
walk            edge (rule)                            state AFTER the row
W alloc {0}     — (first touch)                        producer={0} holders={0}     wave={0}
R load  {1}     e1: alloc@{0} -> load@{1}   (rule 1)   producer={0} holders={0,1}   wave={1}
R load  {2}     e2: alloc@{0} -> load@{2}   (rule 1)   producer={0} holders={0,1,2} wave={2}
R load  {0}     e3: load@{2} -> load@{0}    (rule 3)   producer={0} holders={0,1,2} wave={0}
EXIT            e4: load@{1} -> alloc@{0}@next (rule 2)
(recurrence)    e5: load@{2} -> alloc@{0}@next (rule 2)
```

The fourth row is the one only the wave can catch. `{0}` is already a
holder — it has had the data since its write, so rules 1 and 2 have
nothing to say and the game does not change. But the wave says `{2}` ran
last: two owners ran since `{0}`'s write, so `{0}` re-enters
behind the most recent one (the re-reader is itself the producer, so the
edge's source is the wave's last access, not the producer's). At `EXIT`,
rule 2 pays off the bookkeeping: the next iteration's write collects one
edge from each *other* holder — which is why `holders` had to remember
`{1}` and `{2}` all along.

In the dump, `e1`/`e2` are `S0`/`S1`, the reread edge `e3` is `S2`
(`r S2 {2}` then `a S2 {0}`), and the two recurrence edges `e4`/`e5` are
grouped into one fan-in semaphore `S3` with pending count 2:

```text
|- a  S3(2)  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{gated(rel-count)}
|  |- W m0  ttg.local_alloc {0}
|  |- r  S0  {0} [none]
|  |- r  S1  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S3  {1} [none]        ; e4
|  |- a  S1  {2}
|  |- R m0  ttg.local_load {2}
|  |- r  S2  {2} [none]        ; e3
|  |- r  S3  {2} [none]        ; e5
|  |- a  S2  {0}
|  |- R m0  ttg.local_load {0}
|  |- a  S3(2)  {0}            ; fan-in: waits for both readers
|  |- EXIT ... yield{a S3}
SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=2 entry inherit={@0.0}}
```

(dump of `test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced`)

e3 is the edge to pause on. No data hazard demands it — all three reads
touch data nobody changes, and in this example a protocol feeding `{0}`'s
re-acquire from `{0}`'s own post-write release would have let them run in
parallel, correctly. The wait on `{2}` is rule 3's policy: the re-entering
access needs a live token, and the walk always feeds it from the most
recent access, keeping each group one provable chain rather than a set of
interleavings to reason about case by case. Edge reduction (next section)
does not revisit that choice — it deletes an edge only when the kept edges
already enforce the same ordering, and nothing else orders anything before
`{0}`'s re-read; dropping e3 would leave the re-read with no acquire at
all, not with a parallel one. So e3 survives as `S2`.

Removing edges like it would therefore be a walk-and-emission change, not
a reduction change — and the trigger is already computed: the reread rule
fires exactly when the re-toucher is still a holder (no write intervened),
which is precisely when the ordering is optional. The re-read would still
need its token another way: fed by the re-toucher's own previous release
(a count-1 self-loop, no cross-partition wait), or by keeping the hold
open across the readers' holds — which touches emission's contract that no
buffer view follows its token's release. Today the uniform
chain is the chosen shape. The serialized-ring
example below shows reduction's actual target: an edge the kept set
already implies.

The parent chain sees a nested `for` or `if` as one node and applies its
per-piece owner/effect summary like a single access. Each child chain is then
walked as its own sequence: every piece's game is seeded at `ENTER` with the
owner recorded there as its first holder (the *seed holder*), importing the
parent's pre-region producer so that redundancy across the boundary can
still be proven. A WS-tagged loop may take over root-held memory
without an edge. `EXIT` adds a handoff edge from the other holders to the
region owner only when the next loop iteration or a later access needs it,
skipping any holder that an earlier edge already ordered behind the region
owner (the same synchronized-behind skip as the write rule); it never adds
an edge whose only effect would be to end ownership.

## Edge reduction and semaphore formation

Reduction runs on the raw edges before any semaphore exists; only the
surviving edges become acquires and releases, so a deleted edge never
reaches the IR.

A kept edge makes the destination owner wait for the source; the edges
sharing a destination later merge into that destination's one acquire.
Several edges can therefore land in one hold: when a write waits for two
holders, the writer gets one acquire — one hold — with two edges in. The
first kept edge into the hold is what creates that acquire and hands the
owner a token; the other edges only add waits to a hold that already
exists. Reduction may delete one of the others; it must never delete the
first, however implied its ordering is. An edge is removed only when:

1. its ordering is already implied: following kept edges, stitched through
   each partition's own program order, the source is already forced before
   the destination;
2. it is not the first kept edge into the hold: the group's most recent
   kept handoff already targets this same destination owner, so that
   owner's hold already has its acquire.

Edges that close the loop — from this iteration's access to the next
iteration's — cannot be proven by the straight-line sweep: the implication
path wraps around the loop boundary. A second, wrap-around traversal
handles them; across the boundary "most recent" loses meaning, so
condition 2 relaxes to *some kept handoff into that owner exists*.

Deletions can lean on each other's proofs — edge A proven redundant using
edge B while B is itself deleted. A final pass therefore re-proves every
deleted edge against the kept set alone; if a proof no longer holds, the
pass fails with a diagnostic (`transitive-reduction closure violation`)
rather than emit an under-synchronized protocol.

### Example: a redundant fan-in arm is dropped

`test/NVWS/insert_semas_transitive_reduction.mlir` `@serialized_ring_reduces`
— the spanning owner `m0` with the reuser `m1` nested inside; the overlap
piece has two holders by the time a third partition writes it:

```text
members:    m0[0,256)   m1[64,192)              overlap: [64,192)
pieces:     P0=[0,64){m0}   P1=[64,192){m0, m1}   P2=[192,256){m0}
footprints: an access through m0 touches all three pieces; through m1, P1
```

The body has four accesses in program order — `W m0 {0}`, `R m0 {1}`,
`W m1 {2}`, `R m1 {0}`. The walk on the overlap piece P1 raises all four
in-body edges (the owner-only pieces P0 and P2 repeat e1). The one
recurrence edge comes from P0 and P2 at `EXIT`; P1 itself closes nothing —
its carried owner is `{0}`, and e4 already ordered holder `{2}`'s last
access before `{0}` (the same synchronized-behind skip as the write rule):

```text
walk             P1 game before        rule                      raw edge
W m0 {0}         seeded, holder {0}    write; writer is the      —
                                       sole holder
R m0 {1}         producer {0}          read by new owner         e1: {0} -> {1}
W m1 {2}         holders {0},{1}       write: edge from EVERY    e2: {0} -> {2}
                                       other holder              e3: {1} -> {2}
R m1 {0}         sole holder {2}       read by new owner         e4: {2} -> {0}
EXIT             every piece carries   P0/P2 (owner-only,        e5: {1} -> {0}@next  (P0, P2)
(recurrence)     owner {0}             holders {0},{1}):
                                       holder {1} closes;
                                       P1: skip via e4
```

Reduction keeps four of the five raw edges and drops one:

```text
kept:     e1 {0}->{1}   e3 {1}->{2}   e4 {2}->{0}   e5 {1}->{0}@next

dropped:  e2 {0}->{2}        {2} already waits on {1} (e3), and {1} already
                             waited on {0} (e1) — a direct wait on {0} adds
                             nothing, and {2} still receives its handoff
                             through the kept e3
```

Each kept edge lands in its own destination group, so this function gets four
semaphores, all with pending count 1:

```text
S0 <- e1   m0 data ready for {1}
S1 <- e3   the overlap piece is free for {2}'s write (covers the dropped e2)
S2 <- e4   m1 data ready for {0}
S3 <- e5   buffer free for the next iteration; created initially released,
           its acquire moved into the body by the POINT_OF_USE hold

SEMAS: S0{count=1} S1{count=1} S2{count=1} S3{count=1 entry inherit={@0.0}}
```

The result is the fully serialized ring the test is named after —
`{0} -> {1} -> {2} -> {0} -> next iteration` — one count-1 semaphore per hop.

Remaining edges are:

1. deduplicated by source, destination, and source owner;
2. collapsed to the latest source node for one destination and source owner;
3. grouped by destination node and destination owner.

One destination group becomes one acquire, and its distinct source owners
become releases. A semaphore is therefore one *acquire site* — never one per
raw edge and never one per piece: identical edges raised through several
pieces of one group deduplicate first (step 1), and edges sharing a
destination merge into one acquire whose pending count is the fan-in. In the
`@fanout_not_reduced` example above, the two recurrence edges `e4`/`e5`
share the destination (`alloc@{0}@next`) and become one acquire `a S3(2)`
with two releases `r S3 {1}` / `r S3 {2}`; in the serialized ring, every
destination group holds a single kept edge, so four edges became four
count-1 semaphores. The acquire's pending count is the number of releases it
waits for. A release's
`arrive_count` is raised above one only when one release site must stand in
for several releases, so that the total still matches the pending count. A
handoff into a loop node reuses that loop's regain semaphore — the semaphore
its body re-acquires each iteration (`Hold::regain`) — when the acquiring
owner matches.

## Crossings and holds

A crossing records that the group's token is live inside a `for` or `if`
and says whether the region must return it. An `if` crossing is removed
when nothing after the `if` uses that token.

For each loop crossing, hold analysis (`buildUniformHold`) picks one of the
three `Hold::Outcome` values:

- **CARRIER**: pass the token through the loop iter-arg and result.
- **POINT_OF_USE**: move the per-iteration acquire immediately before the
  first access, and release the token after the hold's last access, inside
  the body.
- **CHILD_OWNS**: the last nested `for` already holds the token; the outer
  loop adds nothing of its own.

In the dump these print as `holdrule{gated(reason)}`,
`holdrule{pointofuse-><firstToucherOp>}`, and
`holdrule{passthrough-drop}`.

### CARRIER: the token rides the iter-arg

`test/NVWS/insert_semas_nested_carrier.mlir`
`@outer_sourceful_alloc_inner_loop_reentry`. The inner loop's token result is
consumed by the parent (`gated(result-consumed)`), and the outer loop has a
trailing use after its hold (`gated(trailing-use)`) — both must carry:

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
|  |  |- EXIT ... yield{a S0}      <- the bottom re-acquire IS what yield returns
|  |- r  S2  {1} [tc5mma]          <- consumes the INNER loop's result token
|  |- a  S2  {0}
|  |- R m0  ttng.tmem_load {0}
|  |- EXIT ... yield{a S2}
```

The emitted outer loop, as a standalone sketch (not row-aligned with the
dump):

```text
%t0 = a S2 root                      ; entry acquire
%tN = for iter_args(%t = %t0) {
  W m0 [%t] {0}                      ; the carried token is consumed at the
  ...                                ;   first access
  %ti = for iter_args(...) { ... }   ; inner loop carries its own token
  r S2 %ti {1} [tc5mma]              ; consumes the inner loop's result token
  %t' = a S2 {0}                     ; regain
  R m0 [%t'] {0}
  yield %t'
}
```

### POINT_OF_USE: the acquire moves to the first access

`test/NVWS/insert_semas.mlir` `@local_reg_and_smem_use`. The loop has no
iter-args before the pass and gains none: `holdrule{pointofuse->
ttg.local_store}` re-materializes the per-iteration acquire directly before
the first access, and the closing release sits after the hold's last access:

```text
   CARRIER shape                          POINT_OF_USE shape
   ...                                    ...
   r  S2 {2}                              r  S2 {2}
   a  S2 {0}   <- regain at BOTTOM  ==>   (deleted here)
   yield %t'                              }                 no token iter-arg
   }                                      a  S2 {0}  <- moved UP to the first
                                          W m0 ... {0}         access (top of body)
```

```text
|- scf.for (WS, tag=0) ... holdrule{pointofuse->ttg.local_store}
|  |- a  S2  {0}                 ; per-iteration acquire, in the body
|  |- W m0  ttg.local_store {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]
|  |- a  S1  {2}
|  |- W m0  use_smem {2}
|  |- r  S2  {2} [none]          ; closing release, in the body
|  |- EXIT ... yield{native}
SEMAS: S0{count=1} S1{count=1} S2{count=1 entry inherit={@0.0}}
```

`yield{native}` means no token crosses the loop boundary; `S2` is
initially released so iteration zero's in-body acquire succeeds.

### CHILD_OWNS: a nested loop already holds it

`test/NVWS/insert_semas_nested_ws_inner_loop.mlir` `@nested_ws_inner_loop` —
both accesses are confined to the inner loop, so the outer loop passes
nothing (`yield{drop}`; neither loop grows an iter-arg):

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
|  |- EXIT ... yield{drop}
```

### Acceptance conditions

Two more definitions are used below. The *feeding acquire* is the acquire
above the loop that supplies the token the loop's first iteration consumes
(`HoldFeed`); its semaphore is the *feeding semaphore*. The *hold prefix* is
the run of body nodes from the top of the body up to the group's first
acquire, evaluated before the point-of-use move — the accesses the moved
acquire will protect (`HoldPrefix`; when the prefix ends in a nested region,
it runs up to that region instead).

POINT_OF_USE and CHILD_OWNS are considered only for loops inside a WS-tagged
loop nest; any other loop keeps CARRIER (reject reason `non-ws-scope`). Past
that gate, POINT_OF_USE is accepted only when the hold prefix is structurally
uniform: a usable feeding acquire, no disqualifying enclosing `if`
(`allEnclosersCanDrop`), no use of the group's memory after the regain — the
crossing's final, bottom re-acquire (reject reason `trailing-use`; the other
owners' protocol between the prefix and the regain is expected) — a
hold-transparent nested region if one is present (`isHoldTransparentRegion`),
exactly one release in the prefix (none when it ends in a nested region),
and — for an inner loop that does not itself carry the WS tag — a first
prefix node that is not a sourceful TMEM allocation (reject reason
`prefix-not-buffer-view`). Prefix-owner uniformity is not gated here; it is
verified after placement, where a violation fails the pass.

The `gated(reason)` label names the first failed check. In the figures of
this document, `rel-count` fired because the hold prefix contains two
releases where the rule expects exactly one, and `entry-consumed` because a
pre-loop access to the group sits between the entry acquire and the loop
— the upward scan for a feeding acquire meets that access first, so the loop
has no usable feeding acquire (in the root-adoption example of the next
section, that access is the root `tmem_store`).

Two cases stay CARRIER even past those checks. If the loop's token result is
consumed by the parent and the feeding semaphore is the same as the regain
semaphore, the token must travel through the iter-arg (that is the
`result-consumed` gate in the CARRIER example above). And a
*final-permission acquire* (`Node::finalPermissionAcquire`) — a single
acquire after the loop that stands in for the token the loop no longer
returns — is legal only when the feeding semaphore differs from the regain
semaphore and a *bridge* is proven: an acquire of the outer feeding
semaphore followed by a release of the loop's own semaphore, placed after
the loop, handing the feeding semaphore's permission over to the loop's
protocol (`Hold::bridgeAcquire`/`bridgeRelease`, dumped as `:entryBridge`).
Otherwise the token stays in the iter-arg.

This realizes the hold rule:

```text
cut token ownership where execution context changes
hold = the maximal run of accesses between cuts
acquire before the hold's first access
release after the hold's last access
```

## First acquire and initial release

Every synchronized group needs its semaphore to be available once before
the first acquire can succeed at run time. One of five placements applies:

- if the loop body re-acquires the group's token each iteration through a
  regain, an unpartitioned acquire of that same semaphore before the loop
  seeds iteration zero;
- if the handoff added at `EXIT` for the next iteration targets an owner that
  is not the first owner with a partition-owned access in the loop body —
  the scan runs over the group's whole body chain —
  the acquire is placed at that owner's first in-body access,
  and the semaphore is created initially released (`Sema::isEntry`) so that
  iteration zero's mid-body acquire succeeds before any release has run; no
  acquire is placed before the loop;
- a POINT_OF_USE hold fed by the same semaphore as its regain loses its
  pre-loop acquire; iteration zero is seeded solely by the initially
  released state;
- a POINT_OF_USE hold fed by a different semaphore keeps its feeding acquire
  above the loop and marks the loop's own regain semaphore initially
  released;
- otherwise a dedicated entry semaphore is created, acquired once before the
  group's first node, and released after the group's last top-level
  node.

Placement may descend through a chain of `if`s: while the only node involving
the group is an `if` with exactly one involved child chain, placement
descends into that chain. It never descends into a loop. The owner of the
group's first real access is recorded as the entry semaphore's
`inheritStamp`; the entry acquire itself remains root-owned.

### Example: root store adopted, entry acquire seeds iteration zero

`test/NVWS/insert_semas_root_entry_tmem.mlir`
`@root_entry_accumulator_adopts_without_semaphore_handoff` — a root
`tmem_store` initializes the accumulator before the WS loop; the loop's first
toucher `{1}` adopts it without a root-to-partition handoff, and the regain
semaphore `S1` gets one unpartitioned pre-loop acquire:

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
|  |- a  S1  {1}                     <- regain, yielded by the carrier
|  |- EXIT ... yield{a S1}
|- r  S2  {@0.1} [none]              <- post-loop handoff of the final value
|- a  S2  root                          back to the root reader
|- R m0  ttng.tmem_load root
SEMAS: S0{count=1} S1{count=1 entry inherit=root} S2{count=1}
```

Why `S1` must be created initially released — the count ledger over a run of
N iterations:

```text
credits:  initial release (1) + N in-loop releases (r S1 {2})   = N+1
waits:    1 root entry acquire + N regain acquires (a S1 {1})   = N+1   ✓ balanced
without the initial release:                                    = N+1 waits, N credits
                                                                -> the entry acquire
                                                                   starves -> DEADLOCK
```

## Backing depth

`buffer.copy`, when present, is authoritative. Without it, synchronized TMEM
may be double-buffered only on the default NVWS path: direct MMA users
immediately inside an `scf.for` must pass the accumulator multibuffering and
capacity checks (`canDoubleBufferAcc`), while other MMA users do not veto the
choice. Meta-NVWS adds no TMEM depth of its own. A local backing written by
a TMA load records the depth that `LowerSemaphore`
will give it (see the [pass order](../nvws-aws-overview.md#pass-order)), so
buffer-stage recurrence is analyzed against the copies that will actually
exist.

## Buffer-stage offsets and the pipeline schedule

Semaphore edges are not SSA dependencies, so they must be projected onto the
existing loop schedule.

1. Source anchors use the last physical completion operation
   (`completionAnchor` when one was recorded); destination anchors use the
   destination owner's first real operation after its acquire — the start
   of the hold that acquire opens.
2. For an edge that closes a loop, reconstruct which buffer stage the release
   addresses, and find the first future iteration whose acquire addresses the
   same buffer stage. The iteration count between them is the recurrence
   distance `d` (`computeRecurrenceDistance`).
3. With source `loop.stage` `Su` and destination `loop.stage` `Sv`, require
   `d + Sv - Su >= 0`.
4. A negative value is rejected. A positive value needs no change. Zero adds
   a source-before-destination ordering constraint.
5. Solve the zero-slack semaphore and SSA constraints together by raising
   destination `loop.cluster` values to a fixed point. `loop.stage` never
   changes.

### Worked example: recurrence distance, slack, and the cluster raise

`test/NVWS/insert_semas_recurrence_schedule.mlir` `@one_slot_recurrence` (its
`_errors.mlir` twin is the rejected case). One depth-1 backing
(`buffer.copy = 1`), producer at `loop.stage 0`, last consumer at
`loop.stage 1`:

```mlir
%buf = ttg.local_alloc {buffer.copy = 1, buffer.id = 420}
scf.for ... {
  %v = "producer"()            {loop.stage = 0, loop.cluster = 1, ttg.partition = [3]}
  ttg.local_store %v, %buf     {loop.stage = 0, loop.cluster = 1, ttg.partition = [3]}
  %first = ttg.local_load %buf {loop.stage = 0, loop.cluster = 1, ttg.partition = [1]}
  "consume_first"(%first)      {loop.stage = 0, loop.cluster = 1, ttg.partition = [1]}
  %last  = ttg.local_load %buf {loop.stage = 1, loop.cluster = 2, ttg.partition = [1]}
  "consume_last"(%last)        {loop.stage = 1, loop.cluster = 2, ttg.partition = [1]}
} {tt.scheduled_max_stage = 1, tt.warp_specialize, ...}
```

The loop-closing edge is `r S1` (source anchor = the `loop.stage 1` last
read) back to `a S1` (destination anchor = the `loop.stage 0` store of a
later iteration).
Which iteration re-addresses the released buffer stage — iteration `i`
writes buffer stage `i mod depth` (`s0` = buffer stage 0):

```text
depth 1:  iter0 -> s0 | iter1 -> s0 | ...   every iteration reuses buffer stage 0   d = 1
depth 2:  iter0 -> s0 | iter1 -> s1 | iter2 -> s0 (the buffer stage iter0 released) d = 2
```

Slack = `d + Sv - Su`, with source `loop.stage` `Su = 1` (the releasing
read) and destination `loop.stage` `Sv = 0` (the re-acquiring store):

```text
PASSING    depth 2:  slack = 2 + 0 - 1 = 1 > 0    headroom; nothing changes
ZERO-SLACK depth 1:  slack = 1 + 0 - 1 = 0        iter i's loop.stage-1 read and
                                                  iter i+1's loop.stage-0 store
                                                  execute in the SAME steady-state
                                                  iteration of the pipelined loop
                                                  -> ordering constraint: read
                                                  before store
REJECTED   Su = 2 (errors twin moves the last read to loop.stage 2):
                     slack = 1 + 0 - 2 = -1 < 0
  error: nvws-insert-semas: fixed loop.stage assignment cannot satisfy
         semaphore handoff (source stage 2, destination stage 0,
         recurrence distance 1)
```

The zero-slack case records the constraint `last-read -> store` and joins it
with the SSA edges of that same steady-state iteration. The fixed point then raises only destination
`loop.cluster` values (`cluster(dst) >= cluster(src) + sep`, `sep = 1` when
the source sits after the destination in the block, else `0`):

```text
start:   store 1   first 1   consume_first 1   last 2   consume_last 2
pass 1:  {last -> store}, last is AFTER store  -> store   := 2+1 = 3
pass 2:  {store -> first}                      -> first   := 3
         {first -> consume_first}              -> consume_first := 3
pass 3:  no change -> converged

result:  store/first/consume_first  loop.stage 0, cluster 3
         last/consume_last          loop.stage 1, cluster 2   (loop.stage untouched)
```

Within each steady-state iteration of the pipelined loop, the next
iteration's producer is now ordered after the previous iteration's final
read — matching the test's CHECK lines.

### Circular groups: the production counter assigns stage offsets

For circular local groups sharing one physical `buffer.id`, SYNC-DAG
validates the common type and depth, unique `buffer.start` values, and
producer order. It then walks accesses in program order: writes advance a
global production counter, and each read refers to its group's latest
produced value. The resulting stage offset is stored on the access node and
its adjacent acquire/release nodes before any IR is emitted.

`test/NVWS/insert_semas_circular_smem.mlir` `@circular_tutorial_1_1_to_2_2` —
K and V share one depth-2 ring (`buffer.id=301`, `buffer.copy=2`,
`buffer.start` 0 and 1). Each circular member is analyzed as its own group —
hence the "K group" / "V group" dumps below — and EMIT-IR later folds them
onto the one physical ring. Counter trace:

```text
store K:   counter 0 -> 1    K produced at ordinal 0    offset  0
store V:   counter 1 -> 2    V produced at ordinal 1    offset  0
load  K:   counter stays 2   K's latest = 1 ago         offset -1
load  V:   counter stays 2   V's latest = current       offset  0
```

K's consumer must address the copy produced *before* V advanced the ring —
visible in the dump as `stage-offset=-1` on exactly K's consumer rows:

```text
K group:  a  S0  {2}  stage-offset=-1         V group:  a  S0  {2}  stage-offset=0
          r  S1  {2} [none]  stage-offset=-1            r  S1  {2} [none]  stage-offset=0
```

### Exact-alias handoffs: the release signals the successor copy

For SMEM groups whose copies are handed off as exact aliases of one
allocation, SYNC-DAG likewise assigns stage offsets so each release signals
the buffer stage addressed by the acquire it satisfies. Unsupported or
path-dependent buffer-stage schedules are rejected rather than guessed.

`test/NVWS/insert_semas_fused_alias_handoff.mlir` `@fused_alias_depth_two` —
two members covering the same range collapse into one piece over a depth-2
backing; the writer-side rows keep `stage-offset=0`, but each
read-to-next-write release carries `stage-offset=1`, because it must signal
the copy the *next* acquire will address, not the copy the read used:

```text
|- a  S0  {2}  stage-offset=0
|- R m0  ttg.local_load {2}
|- r  S1  {2} [none]  stage-offset=1   <- signals the successor copy (the one m1 occupies)
|- a  S1  {4}  stage-offset=0
|- W m1  ttg.local_store {4}
```

### Protocol schedules

Finally, each owner-qualified acquire inherits the pipeline stage of the next
ownership anchor; root entry acquires stay unscheduled. Each release inherits
its owner's last completion schedule — in the worked example above, `a S1`
copies the store's schedule (`loop.stage 0`, cluster 3) and `r S1` copies
the last read's (`loop.stage 1`, cluster 2). A final-permission acquire is placed after its
owner's last operation in the body (`placeFinalAcquireAtLaneExit`). EMIT-IR
copies these schedule and offset facts; its one remaining schedule exception
is the release-schedule fallback of the loop-scheduler workaround (see
[EMIT-IR](emit-ir.md)).

## Code map

[`InsertSemasSyncDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp):

- `walkChain` and `applyTouch`
- `reduceEdges` and `buildEdgesAndSemas`
- `insertEntryAcquires` and `computeCrossings`
- `computeHoldRules`, `buildUniformHold`, `analyzeHoldPrefix`, and
  `applyHoldRulePlacement` (the walk state `PieceGame`/`WaveSt` and the hold
  records `HoldFeed`/`HoldPrefix` are file-local types here)
- `computeBackingPlan`
- `assignCircularStageOffsets`, `assignAliasedHandoffStageOffsets`, and
  `finalizeSyncSchedule`
- `buildSyncDag`
- the DAG dump used throughout this document: set
  `NVWS_INSERT_SEMA_DUMP_DAG=1` (`dumpDagTree`)
