# NVWS circular local buffer design

Status: DOWNSTREAM METADATA DESIGN (16jun26), planner policy superseded
29jun26 by `fable/nvws-memory-planner-meta-parity.md`.

For algorithm 1, MemoryPlanner may mark a circular group only when Meta-AWS
phase 4 selects the corresponding two-record reuse group. For algorithm 0,
it marks the compatible innermost shared-id pool already selected by Meta and
assigns one start per member. `buffer.circular` and `buffer.start` only expose
those decisions to NVWS downstream passes; they do not authorize independent
all-pairs coalescing, post-budget depth growth, or any other divergence from
Meta's planner.

Scope: NVWS local/SMEM circular reuse groups produced by
`--nvws-memory-planner` and consumed by `--nvws-insert-semas`.

This document replaces the older design where circular K/V channels were
modeled as independent logical semaphore streams that remained distinct until
LowerAref. That old design made LowerAref understand physical semaphore aliasing
and required new mbarrier allocation semantics. This version keeps LowerAref
unchanged.

## Problem

The memory planner can make multiple logical local buffers share one physical
SMEM ring:

```mlir
%k = ttg.local_alloc ... {
  buffer.id = N : i32,
  buffer.copy = D : i32,
  buffer.circular,
  buffer.start = 0 : i32
}

%v = ttg.local_alloc ... {
  buffer.id = N : i32,
  buffer.copy = D : i32,
  buffer.circular,
  buffer.start = 1 : i32
}
```

K and V share the same physical ring, but they must not use the same slot in
the same logical iteration. With `D = 2`, the intended steady state is:

```text
K -> slot 0
V -> slot 1
next K -> slot 0 only after slot 0 is released
next V -> slot 1 only after slot 1 is released
```

With `D = 3`, the intended stream is:

```text
K0, V1, K2, V0, K1, V2, ...
```

Current InsertSemas sees K/V through the owner-DAG. If K and V are produced in
the same partition and consumed in the same partition:

```text
for {1} {
  ENTRY {1}
  ld k  {1}
  ld v  {1}
  use k {2}
  use v {2}
  EXIT {1}
}
```

there is no owner-DAG reason to force an extra synchronization edge between
`ld k` and `ld v`. Therefore circular support cannot be modeled by only
coalescing K/V into one existing semaphore-buffer tuple. InsertSemas first needs
logical per-buffer semaphore streams, then it must fold those streams into one
physical empty semaphore and one physical full semaphore.

## Planner Metadata

The memory planner marks circular local reuse explicitly:

```mlir
ttg.local_alloc {
  buffer.id = N : i32,
  buffer.copy = D : i32,
  buffer.circular,
  buffer.start = S : i32
}
```

Definitions:

- `buffer.id`: physical SMEM ring identity.
- `buffer.copy`: ring depth.
- `buffer.circular`: this local alloc is part of a circular SMEM reuse group.
- `buffer.start`: static channel order seed in the circular group.

`buffer.start` is not a byte offset, element offset, spatial view offset, or
direct semaphore stage operand. It identifies the logical channel's position in
the planned circular group. InsertSemas computes per-event stage offsets from
the actual circular event order.

Program order dictates `buffer.start`. The memory planner assigns
`buffer.start` in first-producer program order inside the circular group. For
the common K/V case, if K is produced before V, K gets start `0` and V gets
start `1`. InsertSemas validates that the first-producer order it observes
matches ascending `buffer.start`; mismatch is malformed IR for this design.

Planner invariants:

1. Circular reuse is explicit. Same `buffer.id` alone does not imply circular.
2. Every circular local alloc has `buffer.id`, `buffer.copy`, and
   `buffer.start`.
3. `buffer.copy > 0`.
4. `0 <= buffer.start < buffer.copy`.
5. All circular allocs with the same `buffer.id` have the same `buffer.copy`.
6. All circular allocs with the same `buffer.id` have identical logical local
   size/type unless a later design adds an explicit view contract.
7. `buffer.start` values are distinct within one circular group.
8. `buffer.offset` and `buffer.circular` are mutually exclusive for local
   allocs.

Invalid hand-written IR or planner output must be diagnosed before lowering.

## InsertSemas Design

InsertSemas uses two representations:

1. Logical representation while building the owner-DAG synchronization.
2. Folded representation before exiting the pass.

### Logical Representation

For each circular logical buffer, InsertSemas creates normal logical empty/full
semaphores. For K/V:

```text
K_empty, K_full
V_empty, V_full
```

This lets the owner-DAG placement remain natural. K and V can have separate
producer/consumer token streams without inventing artificial dependencies.

InsertSemas annotates these logical semaphore creates with a fold key:

```text
K_empty.semaphore.id = E
V_empty.semaphore.id = E

K_full.semaphore.id = F
V_full.semaphore.id = F
```

`E` and `F` are different. Empty and full semaphores are different physical
mbarrier arrays because they have different initial state.

`semaphore.id` is an InsertSemas fold key in this design. It is not a LowerAref
physical-aliasing contract.

### Folded Representation

Before InsertSemas finishes, it folds logical semaphore creates with the same
`semaphore.id` into one actual `nvws.semaphore.create`.

This fold is an explicit post-render IR rewrite inside InsertSemas. The normal
emitter first emits the logical SyncDag result: independent logical K/V
acquire/buffer/release streams, logical semaphore creates, and logical backing
allocations/views. The normal emitter must not directly emit the folded physical
semaphore creates or the final shared circular backing. The circular
post-processing rewrite runs after the normal render walk and before InsertSemas
exits; it rewrites logical creates/backings/uses into the folded physical form
and authors the circular stage offsets on the already-emitted semaphore ops.

The same post-render rewrite must preserve token semantics. Circular semaphore
tokens carry permission only. Event-local acquire tokens and loop-carried
permission tokens may remain distinct after folding, and they are not
interchangeable merely because their semaphores fold onto one physical backing.

The current `verifySingleCarrierPerGroup` rule is therefore a non-circular
implementation limitation, not the target circular invariant. For folded
circular groups, multiple carrier token slots for one folded physical semaphore
are legal. AssignStagePhase must not derive circular stages from token lineage;
it threads the shared circular cursor through control flow and computes every
circular acquire/buffer/release stage from that op's authored offset plus the
current shared cursor in program order.

For K/V:

```text
K_empty, V_empty -> empty
K_full,  V_full  -> full
```

After folding there is one empty semaphore and one full semaphore for the
circular group:

```mlir
%empty = nvws.semaphore.create %base true
%full  = nvws.semaphore.create %base false
```

The folded IR keeps per-event circular placement by writing the existing
optional semaphore `stage` operand before AssignStagePhase runs. Before
AssignStagePhase, that operand is a signed circular offset, not a final
physical stage. For the straight-line order `ld k`, `ld v`, `use k`, `use v`,
the offsets are:

```text
acq empty[ 0] ; ld k ; rel full [ 0]
acq empty[ 0] ; ld v ; rel full [ 0]

acq full [-1] ; use k ; rel empty[-1]
acq full [ 0] ; use v ; rel empty[ 0]
```

This design does not add a new `stage.offset` attribute. The same `stage`
operand has two pass-local meanings:

```text
after InsertSemas, before AssignStagePhase:
  stage operand = signed circular offset

after AssignStagePhase:
  stage operand = final physical stage, normalized modulo buffer.copy
```

The offset is an event property. AssignStagePhase consumes the offset exactly
once on each circular semaphore op and overwrites that op's stage operand with
the final physical stage. For circular ops, `nvws.semaphore.buffer` and
`nvws.semaphore.release` do not get their stage from the acquire token. Their
token operand is only the permission proof.

`nvws.semaphore.acquire` must support a stage-only form before
AssignStagePhase:

```mlir
nvws.semaphore.acquire %sem
nvws.semaphore.acquire %sem[%stage]
nvws.semaphore.acquire %sem[%stage, %phase]
```

Before AssignStagePhase, circular acquire uses the stage-only form and the
stage value is the signed circular offset. After AssignStagePhase, acquire uses
the stage+phase form, with final physical stage and computed phase.

### Pending Count

`pending_count` is authored and validated before or during InsertSemas folding.
After that, it is ground truth.

For each logical semaphore stream, InsertSemas computes the pending count while
the logical stream is still distinct. When logical semaphores are folded by
`semaphore.id`, all logical semaphores with that id must agree on the authored
pending count:

```text
K_full.semaphore.id = V_full.semaphore.id = F
K_full.pending_count = 1
V_full.pending_count = 1

folded full.pending_count = 1
```

This agreement validation is by `semaphore.id`, before the physical streams are
folded into one `nvws.semaphore.create`.

`SemaphoreCreateOp::verify()` and LowerAref / lower-semaphore must use the
authored `pending_count` attribute verbatim for folded circular semaphores. They
must not re-derive or re-validate pending count from the folded physical IR,
because folded circular IR intentionally interleaves releases to different
physical stages under one semaphore. A stage-blind verifier over that folded IR
can overcount arrivals; the validation point is the producing pass before
folding, not the generic verifier and not LowerAref.

## Examples

All examples use the same circular event order:

```text
ld k
ld v
use k
use v
```

`ld` means the producer operation that fills the local circular buffer. Because
the event order is the same in all four cases, the circular offsets are also the
same:

```text
ld k  -> offset  0
ld v  -> offset  0
use k -> offset -1
use v -> offset  0
```

The partition assignment changes token ownership and carrier threading; it does
not change the offset math.

### Example 1

Both producers run in partition `{1}` and both consumers run in partition `{2}`:

```text
for {1} {
  ENTRY {1}
  ld k  {1}
  ld v  {1}
  use k {2}
  use v {2}
  EXIT {1}
}
```

Logical InsertSemas placement:

```text
acq K_empty {1} ; ld k  {1} ; rel K_full  {1}
acq V_empty {1} ; ld v  {1} ; rel V_full  {1}
acq K_full  {2} ; use k {2} ; rel K_empty {2}
acq V_full  {2} ; use v {2} ; rel V_empty {2}
```

After post-render folding:

```text
acq empty[ 0] {1} ; ld k  {1} ; rel full [ 0] {1}
acq empty[ 0] {1} ; ld v  {1} ; rel full [ 0] {1}
acq full [-1] {2} ; use k {2} ; rel empty[-1] {2}
acq full [ 0] {2} ; use v {2} ; rel empty[ 0] {2}
```

This is the easiest ownership case because K/V producers share one partition
and K/V consumers share one partition. It still does not prove that one carried
token is enough: if loop-carried K and V tokens are both live at the next
iteration boundary, both permission proofs must remain available.

### Example 2

Every event runs in a different partition:

```text
for {1} {
  ENTRY {1}
  ld k  {1}
  ld v  {2}
  use k {3}
  use v {4}
  EXIT {1}
}
```

Logical InsertSemas placement:

```text
acq K_empty {1} ; ld k  {1} ; rel K_full  {1}
acq V_empty {2} ; ld v  {2} ; rel V_full  {2}
acq K_full  {3} ; use k {3} ; rel K_empty {3}
acq V_full  {4} ; use v {4} ; rel V_empty {4}
```

After post-render folding:

```text
acq empty[ 0] {1} ; ld k  {1} ; rel full [ 0] {1}
acq empty[ 0] {2} ; ld v  {2} ; rel full [ 0] {2}
acq full [-1] {3} ; use k {3} ; rel empty[-1] {3}
acq full [ 0] {4} ; use v {4} ; rel empty[ 0] {4}
```

This case exposes why arbitrary carrier-token coalescing is unsound. K and V
permission tokens can be owned and consumed by different partitions. A single
surviving token cannot stand in for both if both are live.

### Example 3

Both producers run in partition `{1}`, but consumers run in different
partitions:

```text
for {1} {
  ENTRY {1}
  ld k  {1}
  ld v  {1}
  use k {2}
  use v {3}
  EXIT {1}
}
```

Logical InsertSemas placement:

```text
acq K_empty {1} ; ld k  {1} ; rel K_full  {1}
acq V_empty {1} ; ld v  {1} ; rel V_full  {1}
acq K_full  {2} ; use k {2} ; rel K_empty {2}
acq V_full  {3} ; use v {3} ; rel V_empty {3}
```

After post-render folding:

```text
acq empty[ 0] {1} ; ld k  {1} ; rel full [ 0] {1}
acq empty[ 0] {1} ; ld v  {1} ; rel full [ 0] {1}
acq full [-1] {2} ; use k {2} ; rel empty[-1] {2}
acq full [ 0] {3} ; use v {3} ; rel empty[ 0] {3}
```

This case shows that shared producer-side ownership does not make the
consumer-side permission tokens interchangeable.

### Example 4

Producers run in different partitions, but both consumers run in partition `{3}`:

```text
for {1} {
  ENTRY {1}
  ld k  {1}
  ld v  {2}
  use k {3}
  use v {3}
  EXIT {1}
}
```

Logical InsertSemas placement:

```text
acq K_empty {1} ; ld k  {1} ; rel K_full  {1}
acq V_empty {2} ; ld v  {2} ; rel V_full  {2}
acq K_full  {3} ; use k {3} ; rel K_empty {3}
acq V_full  {3} ; use v {3} ; rel V_empty {3}
```

After post-render folding:

```text
acq empty[ 0] {1} ; ld k  {1} ; rel full [ 0] {1}
acq empty[ 0] {2} ; ld v  {2} ; rel full [ 0] {2}
acq full [-1] {3} ; use k {3} ; rel empty[-1] {3}
acq full [ 0] {3} ; use v {3} ; rel empty[ 0] {3}
```

This case shows the symmetric producer-side issue: shared consumer ownership
does not make producer-side permission tokens interchangeable.

### Consequence

The four examples all support the same folded semaphore/backing structure and
the same circular offsets. They do not support a general rule that folded
circular K/V must have one loop-carried token. The correct design requirement is
permission tokens passed as-is, plus one shared circular cursor for stage
assignment. Circular stages are computed from the current shared cursor and each
op's own authored offset.

After AssignStagePhase with depth 2, all four examples use the same physical
slot sequence:

```text
acq empty [0] ; ld k  ; rel full  [0]
acq empty [1] ; ld v  ; rel full  [1]
acq full  [0] ; use k ; rel empty [0]
acq full  [1] ; use v ; rel empty [1]
```

For depth 3, the next logical iteration continues:

```text
acq empty [2] ; ld k  ; rel full  [2]
acq empty [0] ; ld v  ; rel full  [0]
```

The wait on `empty[0]` naturally prevents V from overwriting slot 0 until the
previous K consumer releases `empty[0]`.

## Offset Computation

InsertSemas computes offsets from the circular event order, not directly from
`buffer.start`.

For each folded circular group:

```text
cursor = current physical producer slot
rank[X] = physical slot reserved by the latest producer of logical buffer X
```

For a fresh producer event:

```text
cursor = (cursor + 1) % depth
rank[X] = cursor
offset = 0
```

For a consumer event:

```text
offset = rank[X] - cursor
```

For:

```text
ld k
ld v
use k
use v
```

the derivation is:

```text
ld k:
  cursor = 0, rank[K] = 0, offset = 0

ld v:
  cursor = 1, rank[V] = 1, offset = 0

use k:
  cursor = 1, rank[K] = 0, offset = -1

use v:
  cursor = 1, rank[V] = 1, offset = 0
```

## AssignStagePhase Contract

AssignStagePhase continues to compute the shared cursor for a folded semaphore
group. The required extension is that a preexisting `stage` operand on a
circular semaphore event is interpreted as a signed offset before assignment,
then overwritten with the final physical stage.

Definitions:

```text
authoredStage = optional stage operand present before AssignStagePhase
offset        = authoredStage if present on a circular event, otherwise 0
baseStage     = stage computed by the existing stage state machine
eventStage    = (baseStage + offset) mod depth
```

For every circular semaphore op in program order:

```text
if op is acquire:
  baseStage = state.stage

  if acquire is a fresh write:
    baseStage = (baseStage + 1) % depth

  offset = signed value from acquire.stage if present, otherwise 0
  eventStage = (baseStage + offset) % depth

  state.stage = baseStage
  acquire.stage = eventStage
  phase arithmetic uses eventStage

if op is buffer or release:
  baseStage = state.stage
  offset = signed value from op.stage if present, otherwise 0
  eventStage = (baseStage + offset) % depth
  op.stage = eventStage
```

AssignStagePhase must not derive circular buffer/release stages from token
lineage. Token propagation remains the non-circular mechanism only.

This preserves the existing rule that LowerAref indexes both mbarriers and data
buffers using the stage operands present on semaphore ops. There is no separate
logical mbarrier stage and data-buffer stage in this design.

If a folded circular event has no authored stage operand, that is an InsertSemas
bug unless the implementation explicitly treats absence as offset `0` for that
event class. The simpler verifier rule is: circular events must author the
stage operand, even when the offset is zero. If a non-circular event has no
authored stage operand, AssignStagePhase behaves as it does today with offset
`0`.

## LowerAref Contract

LowerAref remains unchanged.

By the time LowerAref runs:

1. Logical circular semaphores have already been folded.
2. There is one actual empty `nvws.semaphore.create` and one actual full
   `nvws.semaphore.create` for the circular group.
3. `nvws.semaphore.acquire`, `nvws.semaphore.release`, and
   `nvws.semaphore.buffer` already carry final physical stage operands after
   AssignStagePhase.

LowerAref keeps doing what it does today:

```text
acquire stage -> wait mbarrier index
release stage -> arrive mbarrier index
buffer stage  -> memdesc_index data slot
```

It does not need to inspect `buffer.circular`, `buffer.start`, or
`semaphore.id`.

## SCF Soundness Rule

Static offsets are valid only when the circular event order is path-uniform in
the relevant SCF region.

For a straight-line SCF block:

```text
ld k
ld v
use k
use v
```

the offsets are static and valid.

For:

```text
ld k
scf.if %cond {
  ld v
}
use k
```

there is no single correct static offset for `use k`:

```text
condition false: K is the most recent circular event, offset 0
condition true:  V advanced the circular cursor, offset -1
```

The transform must reject circular folding for that group, or avoid marking the
group circular in the planner. This design does not introduce dynamic circular
cursor SSA values.

## Tests

Add targeted lit coverage in this order:

1. Memory planner emits `buffer.circular` and `buffer.start` for K/V local
   circular reuse, with same `buffer.id`, same `buffer.copy`, and starts `0/1`.
2. Memory planner rejects or avoids circular grouping for unequal local sizes.
3. InsertSemas initially builds separate logical K/V semaphore streams for a
   circular group.
4. InsertSemas assigns the same fold `semaphore.id` to K/V empty semaphores and
   a different shared fold id to K/V full semaphores.
5. InsertSemas folds same-id logical semaphores into one empty and one full
   `nvws.semaphore.create`.
6. InsertSemas authors static circular offsets in the existing stage operands
   on folded acquire/release/buffer events.
7. AssignStagePhase consumes authored stage operands as offsets and overwrites
   them with final physical stages.
8. LowerAref output uses final physical stage operands directly and requires no
   circular special case.
9. A negative SCF `if` test rejects path-nonuniform circular folding.
10. The PFA/FA reproducer no longer aliases K and V into the same physical SMEM
    slot after lowering.

## Non-goals

- Do not make same `buffer.id` imply circular semantics.
- Do not use `buffer.offset` for local circular starts.
- Do not make LowerAref allocate mbarrier arrays by `semaphore.id`.
- Do not preserve distinct K/V logical semaphore creates until LowerAref.
- Do not introduce dynamic per-event offset computation for path-nonuniform
  circular order. AssignStagePhase may still thread its normal shared
  stage/cursor SSA through loops and `scf.if`.
