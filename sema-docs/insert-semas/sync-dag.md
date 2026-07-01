# SYNC-DAG

## Purpose

SYNC-DAG converts OWNER-DAG ownership changes into a balanced semaphore
protocol. It also decides whether semaphore tokens cross region boundaries,
the backing depth, stage offsets, and the pipeline-stage placement required by
the software pipeliner:

```text
input IR ──► ACCESS-DAG ──owners──► OWNER-DAG ──edges+semas+schedule──► SYNC-DAG ──render──► output IR
```

This document is self-contained: the notation section below redefines what it
uses. Full definitions live in the
[InsertSemas overview](overview.md#core-objects) (the model objects) and the
[NVWS-AWS terminology](../nvws-aws-overview.md#terminology) (buffer stage,
stage offset, pipeline stage).

## Notation

Every diagram in this document uses the pass's own dump format, so every
figure can be reproduced by running a listed lit test under
`NVWS_INSERT_SEMA_DUMP_DAG=1`:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1 triton-opt <test.mlir> --nvws-insert-semas
```

Object shorthand (full definitions in the overview):

```text
group         the allocations analyzed together (ordinarily those sharing one buffer.id)
backing       the mutable SMEM/TMEM allocation the group guards
m0, m1        member: one allocation of the group; m0[0,64) = bytes/columns it covers
P0, P1        piece: a disjoint interval of the backing; {m0,m1} = members covering it
c0            component: connected pieces; synchronization is per component
{0}, {1}      owner: partition 0, 1 of the enclosing WS loop; root = no partition
{@0.1}        tag-qualified owner: partition 1 under WS tag 0 (printed where the
              anchor sits outside that loop, e.g. on SEMAS lines)
producer      the owner that last wrote a piece (before any write: its first toucher)
holder        an owner currently allowed to access a piece (producer + joined readers)
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
thread{c0:{0}}              region row: component c0's token crosses here
holdrule{c0:...}            loop row: the crossing's hold outcome (see Crossings)
yield{c0: X}                EXIT row: what this chain returns for c0's crossing
stage-offset=N              stage offset assigned to this protocol row
SEMAS c0: S1{count=1 entry inherit={@0.0}}   per-component semaphore summary:
                            pending count; entry = initially released;
                            inherit = owner of the component's first real
                            access, recorded on the semaphore
BACKING: numStages=N        depth chosen for the backing allocation
```

Pseudo-IR in this document strips types and attributes; `%t = a S0 {1}` /
`r S0 %t {1}` are the token-producing acquire and token-consuming release,
`W m0 [%t]` is an access performed while holding token `%t`, and
`for iter_args(%t = %t0)` is a loop carrying the token.

## Ownership walk

The walk runs once per group, in program order over its chains (`walkChain`).
It keeps two kinds of state:

- per piece, the *game* (`PieceGame`): the piece's producer and holders, as
  defined in the [overview](overview.md#contract);
- per component, the *wave* (`WaveSt`): the owner of the component's most
  recent partition-owned access. An access by a different owner moves the
  wave to that owner — reads included. A nested region moves the wave to the
  region's owner when that owner is unique for the component and
  partition-owned, and invalidates the wave otherwise.

```text
per piece P0:   game { producer: {0}, holders: [{0}, {1}, ...] }
per comp  c0:   wave { owner of the LAST partition-owned access, any piece }
```

The wave detects interleaving: when an access's owner differs from the
current wave owner, another owner has touched the component since this
owner's previous access. The rules below call that "the wave has moved".

Each touch of a piece advances that piece's game (`applyTouch`):

```text
first touch of a piece:
  record the toucher as producer and sole holder; add no edge

write by owner P:
  add an edge H -> P for every other holder H of the piece, except when an
  earlier edge already orders H's last access before P — the wave having
  moved re-enables those skipped edges
  if no holder edge was added and the wave has moved, add an edge from the
  wave's last access instead
  P becomes the piece's producer and sole holder

read by an owner already holding the piece:
  if the wave has moved, the holder must re-acquire: add an edge from the
  producer's last access — or from the wave's last access when the
  re-reader is itself the producer or the producer holds no access in
  this chain
  update the holder's last access

read by a new owner:
  add an edge: producer -> reader
  (when the producer holds no access in this chain, the seed holder is the
  source instead; see the nested-region seeding below)
  the reader joins the holders
```

An edge records the source access's completion kind — `none`, TMA, MMA, and
so on (`EdgeRec::payloads`) — because a release may need to wait for
asynchronous completion.

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
walk                     state fact                    edge
ENTER {P0:W:{0}}         seed: game live, holder {0}   —
W m0 {0}                 write; sole holder = seed {0} —   (producer={0}, holders={0})
R m0 {1}                 read by new owner             e1: store@{0} -> load@{1}
EXIT (recurrence)        holder {1} != next            e2: load@{1} -> store@{0}@next
                         iteration's writer {0}
```

Edge `e1` becomes semaphore `S0` ("data ready"), `e2` becomes `S1` ("buffer
free again"). The resulting SYNC-DAG (dump):

```text
|- scf.for (WS, tag=0) pieces{P0:W:{0}} thread{c0:{0}} holdrule{c0:pointofuse->ttg.local_store}
|  |- a  S1  {0}                ; e2 satisfied: buffer free for this iteration
|  |- W m0  ttg.local_store {0}
|  |- r  S0  {0} [none]         ; e1: data ready
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]         ; e2: buffer free for the next iteration
|  |- EXIT pieces{P0:W:{0}} yield{c0: native}
SEMAS c0: S0{count=1} S1{count=1 entry inherit={@0.0}}
```

`S1` is created initially released (`entry`) so iteration zero's `a S1`
succeeds before any release has run — see
[First acquire and initial release](#first-acquire-and-initial-release).

### Example: the wave moves, a reread re-acquires

partition 0 writes, partitions 1 and 2 read, then partition 0 *re-reads* the
piece it still holds:

```text
walk                game (P0)                     wave (c0)      edge
W alloc {0}         write; sole holder = seed     -> {0}         —
R load  {1}         new holder; holders {0,1}     {0} -> {1}     e1: alloc@{0} -> load@{1}
R load  {2}         new holder; holders {0,1,2}   {1} -> {2}     e2: alloc@{0} -> load@{2}
R load  {0}         reread by producer {0};       {2} -> {0}     e3: load@{2} -> load@{0}
                    wave has moved -> re-acquire
                    (re-reader IS the producer ->
                    source = wave's last access)
EXIT (recurrence)   holders {1},{2} vs writer     —              e4: load@{1} -> alloc@{0}@next
                    {0}@next                                     e5: load@{2} -> alloc@{0}@next
```

In the dump, `e1`/`e2` are `S0`/`S1`, the reread edge `e3` is `S2`
(`r S2 {2}` then `a S2 {0}`), and the two recurrence edges `e4`/`e5` are
grouped into one fan-in semaphore `S3` with pending count 2:

```text
|  |- a  S3(2)  root  ; entry
|  |- scf.for (WS, tag=0) ... holdrule{c0:gated(rel-count)}
|  |  |- W m0  ttg.local_alloc {0}
|  |  |- r  S0  {0} [none]
|  |  |- r  S1  {0} [none]
|  |  |- a  S0  {1}
|  |  |- R m0  ttg.local_load {1}
|  |  |- r  S3  {1} [none]        ; e4
|  |  |- a  S1  {2}
|  |  |- R m0  ttg.local_load {2}
|  |  |- r  S2  {2} [none]        ; e3
|  |  |- r  S3  {2} [none]        ; e5
|  |  |- a  S2  {0}
|  |  |- R m0  ttg.local_load {0}
|  |  |- a  S3(2)  {0}            ; fan-in: waits for both readers
|  |  |- EXIT ... yield{c0: a S3}
SEMAS c0: S0{count=1} S1{count=1} S2{count=1} S3{count=2 entry inherit={@0.0}}
```

(dump of `test/NVWS/insert_semas_transitive_reduction.mlir`
`@fanout_not_reduced`)

The parent chain sees a nested `for` or `if` as one node and applies its
per-piece owner/effect summary like a single access. Each child chain is then
walked as its own sequence: every piece's game is seeded at `ENTER` with the
owner recorded there as its first holder (the *seed holder*), importing the
parent's pre-region producer so that redundancy across the boundary can
still be proven. A WS-tagged loop may take over a root-held component
without an edge. `EXIT` adds a handoff edge from the other holders to the
region owner only when the next loop iteration or a later access needs it; it
never adds an edge whose only effect would be to end ownership.

## Edge reduction and semaphore formation

An edge is removed only when two facts already hold at its destination: the
waits established by kept edges, followed through per-partition program
order, order the edge's source before its destination; and the component's
most recent kept handoff already targets the same destination owner, so the
candidate edge would not be the one that starts that owner's ownership span.
Acquires and releases are formed afterward, from the kept edges. A second
traversal proves the removals for edges that close a loop; there the second
condition is keyed by destination owner alone — any kept handoff into that
owner, regardless of component, satisfies it. Every removed edge is
rechecked against the closure of the kept edges.

### Example: a redundant fan-in arm is dropped

`test/NVWS/insert_semas_transitive_reduction.mlir` `@serialized_ring_reduces`
— two overlapping members create a shared piece with two holders when a
third partition writes it:

```text
members: m0[0,128) m1[64,192)          pieces: P0=[0,64){m0}  P1=[64,128){m0,m1}  P2=[128,192){m1}

     W m0 {0}                       raw edges at "W m1 {2}" (P1 holders = {0},{1}):
        |                              {0} -> {2}   producer arm
     R m0 {1}                          {1} -> {2}   reader arm
        |
     W m1 {2}    <- two raw arms    kept: {0}->{1} (S0), {1}->{2} (S1)
        |                           dropped: {0}->{2} — already implied through
     R m1 {0}                                {0}->{1}->{2}, and {2}'s span was
                                             already opened by the kept {1}->{2}
```

The dump shows the write of `m1` gated by a *single* acquire `a S1 {2}` with
pending count 1 — the producer arm never becomes a semaphore. The final
`R m1 {0}` likewise emits no loop-closing release back to `{2}`: the second
traversal proves that close implied as well.

Remaining edges are:

1. deduplicated by source, destination, source owner, and component;
2. collapsed to the latest source node for one source owner, destination, and
   component;
3. grouped by destination node, destination owner, and component.

One destination group becomes one acquire, and its distinct source owners
become releases — in the `@fanout_not_reduced` example above, the two
recurrence edges `e4`/`e5` share the destination (`alloc@{0}@next`) and
become one acquire `a S3(2)` with two releases `r S3 {1}` / `r S3 {2}`. The
acquire's pending count is the number of releases it waits for. A release's
`arrive_count` is raised above one only when one release site must stand in
for several releases, so that the total still matches the pending count. A
handoff into a loop node reuses that loop's regain semaphore — the semaphore
its body re-acquires each iteration (`Hold::regain`) — when the component
and the acquiring owner match.

## Crossings and holds

A crossing says which component's semaphore token a `for` or `if` must
return. An `if` crossing is removed when nothing after the `if` uses that
token.

For each loop crossing, hold analysis (`buildUniformHold`) picks one of the
three `Hold::Outcome` values:

- **CARRIER**: pass the token through the loop iter-arg and result.
- **POINT_OF_USE**: move the per-iteration acquire immediately before the
  first access, and release the token after the hold's last access, inside
  the body.
- **CHILD_OWNS**: the last nested `for` already holds the token; the outer
  loop adds nothing of its own.

In the dump these print as `holdrule{c0: gated(reason)}`,
`holdrule{c0: pointofuse-><firstToucherOp>}`, and
`holdrule{c0: passthrough-drop}`.

### CARRIER: the token rides the iter-arg

`test/NVWS/insert_semas_nested_carrier.mlir`
`@outer_sourceful_alloc_inner_loop_reentry`. The inner loop's token result is
consumed by the parent (`gated(result-consumed)`), and the outer loop has a
trailing use after its hold (`gated(trailing-use)`) — both must carry:

```text
|- a  S2  root  ; entry
|- scf.for (WS, tag=0) ... holdrule{c0:gated(trailing-use)}
|  |- W m0  ttng.tmem_alloc {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- scf.for ... holdrule{c0:gated(result-consumed)}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S1  {1} [tc5mma]
|  |  |- a  S1  {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r  S0  {0} [none]
|  |  |- a  S0  {1}
|  |  |- EXIT ... yield{c0: a S0}      <- the bottom re-acquire IS what yield returns
|  |- r  S2  {1} [tc5mma]              <- consumes the INNER loop's result token
|  |- a  S2  {0}
|  |- R m0  ttng.tmem_load {0}
|  |- EXIT ... yield{c0: a S2}
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
iter-args before the pass and gains none: `holdrule{c0:pointofuse->
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
|- scf.for (WS, tag=0) ... holdrule{c0:pointofuse->ttg.local_store}
|  |- a  S2  {0}                 ; per-iteration acquire, in the body
|  |- W m0  ttg.local_store {0}
|  |- r  S0  {0} [none]
|  |- a  S0  {1}
|  |- R m0  ttg.local_load {1}
|  |- r  S1  {1} [none]
|  |- a  S1  {2}
|  |- W m0  use_smem {2}
|  |- r  S2  {2} [none]          ; closing release, in the body
|  |- EXIT ... yield{c0: native}
SEMAS c0: ... S2{count=1 entry inherit={@0.0}}
```

`yield{c0: native}` means no token crosses the loop boundary; `S2` is
initially released so iteration zero's in-body acquire succeeds.

### CHILD_OWNS: a nested loop already holds it

`test/NVWS/insert_semas_nested_ws_inner_loop.mlir` `@nested_ws_inner_loop` —
both accesses are confined to the inner loop, so the outer loop passes
nothing (`yield{c0: drop}`; neither loop grows an iter-arg):

```text
|- scf.for (WS, tag=0) ... holdrule{c0:passthrough-drop}
|  |- scf.for ... holdrule{c0:pointofuse->ttng.tc_gen5_mma}
|  |  |- a  S1  {1}
|  |  |- W m0  ttng.tc_gen5_mma {1}
|  |  |- r  S0  {1} [tc5mma]
|  |  |- a  S0  {0}
|  |  |- R m0  ttng.tmem_load {0}
|  |  |- r  S1  {0} [none]
|  |  |- EXIT ... yield{c0: native}
|  |- EXIT ... yield{c0: drop}
```

### Acceptance conditions

Two more definitions are used below. The *feeding acquire* is the acquire
above the loop that supplies the token the loop's first iteration consumes
(`HoldFeed`); its semaphore is the *feeding semaphore*. The *hold prefix* is
the run of body nodes from the top of the body up to the component's first
acquire, evaluated before the point-of-use move — the accesses the moved
acquire will protect (`HoldPrefix`; when the prefix ends in a nested region,
it runs up to that region instead).

POINT_OF_USE and CHILD_OWNS are considered only for loops inside a WS-tagged
loop nest; any other loop keeps CARRIER (reject reason `non-ws-scope`). Past
that gate, POINT_OF_USE is accepted only when the hold prefix is structurally
uniform: a usable feeding acquire, no disqualifying enclosing `if`
(`allEnclosersCanDrop`), no use of the component after the prefix, a
hold-transparent nested region if one is present (`isHoldTransparentRegion`),
exactly one release in the prefix (none when it ends in a nested region),
and — for an inner loop that does not itself carry the WS tag — a first
prefix node that is not a sourceful TMEM allocation (reject reason
`prefix-not-buffer-view`). Prefix-owner uniformity is not gated here; it is
verified after placement, where a violation fails the pass.

The `gated(reason)` label names the first failed check. In the figures of
this document, `rel-count` fired because the hold prefix contains two
releases where the rule expects exactly one, and `entry-consumed` because
the loop's carrier consumes the pre-loop entry acquire's token, which pins
that acquire outside the body.

Two cases stay CARRIER even past those checks. If the loop's token result is
consumed by the parent and the feeding semaphore is the same as the regain
semaphore, the token must travel through the iter-arg (that is the
`result-consumed` gate in the CARRIER example above). And a
*final-permission acquire* (`Node::finalPermissionAcquire`) — a single
acquire after the loop that stands in for the token the loop no longer
returns — is legal only when the feeding semaphore differs from the regain
semaphore and a *bridge* is proven: a release/acquire pair placed after the
loop that hands the outer feeding semaphore's permission over to the loop's
own semaphore (`Hold::bridgeAcquire`/`bridgeRelease`, dumped as
`:entryBridge`). Otherwise the token stays in the iter-arg.

This realizes the hold rule:

```text
cut token ownership where execution context changes
hold = the maximal run of accesses between cuts
acquire before the hold's first access
release after the hold's last access
```

## First acquire and initial release

Every synchronized component needs its semaphore to be available once before
the first acquire can succeed at run time. One of five placements applies:

- if the loop body re-acquires the component each iteration through a regain,
  an unpartitioned acquire of that same semaphore before the loop seeds
  iteration zero;
- if the handoff added at `EXIT` for the next iteration targets an owner that
  is not the first owner with a partition-owned access in the loop body —
  this scan runs over the group's whole body chain, not only the handoff's
  component — the acquire is placed at that owner's first in-body access,
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
  component's first node, and released after the component's last top-level
  node.

Placement may descend through a chain of `if`s: while the only node involving
the component is an `if` with exactly one involved child chain, placement
descends into that chain. It never descends into a loop. The owner of the
component's first real access is recorded as the entry semaphore's
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
|- scf.for (WS, tag=0) ... holdrule{c0:gated(entry-consumed)}
|  |- R m0  ttng.tmem_load {1}
|  |- W m0  ttng.tmem_store {1}
|  |- r  S0  {1} [none]
|  |- a  S0  {2}
|  |- W m0  ttng.tc_gen5_mma {2}
|  |- r  S1  {2} [tc5mma]
|  |- a  S1  {1}                     <- regain, yielded by the carrier
|  |- EXIT ... yield{c0: a S1}
|- r  S2  {@0.1} [none]              <- post-loop handoff of the final value
|- a  S2  root                          back to the root reader
|- R m0  ttng.tmem_load root
SEMAS c0: S0{count=1} S1{count=1 entry inherit=root} S2{count=1}
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
choice. Meta-NVWS adds no TMEM depth of its own. A local backing whose
producer is a TMA load records the `num-stages` depth that `LowerSemaphore`
will give it (see the [pass order](../nvws-aws-overview.md#pass-order)), so
buffer-stage recurrence is analyzed against the copies that will actually
exist.

## Buffer-stage offsets and the pipeline schedule

Semaphore edges are not SSA dependencies, so they must be projected onto the
existing loop schedule.

1. Source anchors use the last physical completion operation
   (`completionAnchor` when one was recorded); destination anchors use the
   first real operation of the destination owner's wave.
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

The loop-closing edge is `r S1` (source anchor = the stage-1 last read) back
to `a S1` (destination anchor = the stage-0 store of a later iteration).
Which iteration re-addresses the released buffer stage — iteration `i`
writes buffer stage `i mod depth` (`s0` = buffer stage 0):

```text
depth 1:  iter0 -> s0 | iter1 -> s0 | ...   every iteration reuses buffer stage 0   d = 1
depth 2:  iter0 -> s0 | iter1 -> s1 | iter2 -> s0 (the stage iter0 released)        d = 2
```

Slack = `d + Sv - Su`, with source stage `Su = 1` (the releasing read) and
destination stage `Sv = 0` (the re-acquiring store):

```text
PASSING    depth 2:  slack = 2 + 0 - 1 = 1 > 0    headroom; nothing changes
ZERO-SLACK depth 1:  slack = 1 + 0 - 1 = 0        iter i's stage-1 read and iter
                                                  i+1's stage-0 store land in the
                                                  SAME pipeline step -> ordering
                                                  constraint: read before store
REJECTED   Su = 2 (errors twin moves the last read to loop.stage 2):
                     slack = 1 + 0 - 2 = -1 < 0
  error: nvws-insert-semas: fixed loop.stage assignment cannot satisfy
         semaphore handoff (source stage 2, destination stage 0,
         recurrence distance 1)
```

The zero-slack case records the constraint `last-read -> store` and joins it
with the same-step SSA edges. The fixed point then raises only destination
`loop.cluster` values (`cluster(dst) >= cluster(src) + sep`, `sep = 1` when
the source sits after the destination in the block, else `0`):

```text
start:   store 1   first 1   consume_first 1   last 2   consume_last 2
pass 1:  {last -> store}, last is AFTER store  -> store   := 2+1 = 3
pass 2:  {store -> first}                      -> first   := 3
         {first -> consume_first}              -> consume_first := 3
pass 3:  no change -> converged

result:  store/first/consume_first  stage 0, cluster 3
         last/consume_last          stage 1, cluster 2   (loop.stage untouched)
```

Within each pipeline step, the next iteration's producer is now ordered
after the previous iteration's final read — matching the test's CHECK lines.

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
K group:  a S0 {2}  stage-offset=-1     V group:  a S0 {2}  stage-offset=0
          r S1 {2}  stage-offset=-1               r S1 {2}  stage-offset=0
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
copies the store's schedule (stage 0, cluster 3) and `r S1` copies the last
read's (stage 1, cluster 2). A final-permission acquire is placed after its
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
