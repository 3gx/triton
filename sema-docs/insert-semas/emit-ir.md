# EMIT-IR

EMIT-IR turns the finalized synchronization plan into MLIR. The semaphores,
pending counts, acquire and release placement, token paths, backing copies,
and schedules are already fixed. Emission materializes those choices and
checks the resulting IR.

## A handoff in emitted IR

Consider one buffer written by partition `{0}` and then read by partition
`{1}`. The finalized plan has an initially released `EMPTY` semaphore and a
`FULL` semaphore, both with pending count 1. This is the protocol order,
after edge reduction and semaphore assignment:

```text
{0}: acquire EMPTY -> write -> release FULL
{1}: acquire FULL  -> read  -> release EMPTY
```

The emitted MLIR has the following shape. Types and unrelated attributes are
omitted here so the token path is easy to follow:

```mlir
%backing = ttg.local_alloc ...
%empty = nvws.semaphore.create %backing true  {pending_count = 1}
%full  = nvws.semaphore.create %backing false {pending_count = 1}

%empty_token = nvws.semaphore.acquire %empty
%write_buffer = nvws.semaphore.buffer %empty, %empty_token
ttg.local_store %value, %write_buffer
nvws.semaphore.release %full, %empty_token [#nvws.async_op<none>]
    {arrive_count = 1}

%full_token = nvws.semaphore.acquire %full
%read_buffer = nvws.semaphore.buffer %full, %full_token
%value = ttg.local_load %read_buffer
nvws.semaphore.release %empty, %full_token [#nvws.async_op<none>]
    {arrive_count = 1}
```

An acquire returns a token for one copy of the backing. A buffer operation
uses the semaphore and that token to select the corresponding buffer view. A
release uses the same token to make that copy available through the next
semaphore. This is why a token acquired from `EMPTY` can be passed to a
release of `FULL`.

The same shape applies to TMEM. A TMEM buffer is passed to
`ttng.tmem_store`, `ttng.tmem_load`, or MMA in place of the original TMEM
allocation. If the original allocation had an initial value, emission writes
that value through the semaphore buffer before retargeting its other uses.

## Emission order

The order is important because each step prepares concrete SSA values for
the next one:

1. Remove the original TMEM dependency-token operands and results for groups
   that use semaphores. Remove dead token slots from `scf.for` and `scf.if`.
2. Materialize physical backing allocations and all semaphore creates,
   including backing shared by several groups.
3. Emit entry acquires. These provide tokens that are live before the first
   access or region that needs them.
4. Add all required token slots to `scf.for` and `scf.if` signatures. Each
   region is rebuilt once, even when several groups need slots on it.
5. Render every group's acquires, releases, buffers, accesses, and region
   yields. When the last group renders an `scf.if`, finish its scheduler-safe
   shape if the release and acquire pattern requires it.
6. Remove token slots, alias operations, original allocations, and temporary
   poison values that became dead.
7. Verify token use, loop token slots, and partition outputs.

Temporary poison tokens let signatures be rebuilt before every group has
rendered its exact initial value and yields. Rendering replaces the live
placeholders; cleanup removes any slot that remained unused.

## Physical backing and semaphore creates

The physical backing is created before protocol operations. This makes every
`nvws.semaphore.create` refer directly to the final allocation or view.

### Ordinary groups

For an ordinary multi-buffered group, the added leading dimension is the
number of copies. A TMEM scales encoding already represents its physical
copies and keeps its existing shape:

```mlir
// One logical memdesc, two physical copies.
%backing = ttg.local_alloc
    : () -> !ttg.memdesc<2x128x64xf16, ...>
%empty = nvws.semaphore.create %backing true {pending_count = 1 : i32}
    : <[!ttg.memdesc<2x128x64xf16, ...>]>
```

When one member fully covers another member, both may use one backing. TMEM
members use a subslice when their offsets or types differ. Local-memory
members marked as copies share only when their offset and backing type agree.

### Shared physical backing

Groups that name the same physical buffer may still have separate
synchronization plans:

- Circular local-memory groups with the same `buffer.id` and backing type
  use one allocation. Their matching entry or non-entry semaphore creates
  are also shared, and their pending counts must agree.
- Mixed-depth TMEM groups use the larger physical allocation. The other
  group receives a checked TMEM subslice followed by an explicit
  reinterpretation to its backing type.

This sharing is complete before entry acquires or group rendering begin.
There is no later allocation-folding step.

Each semaphore create records two facts from the finalized plan:

- `true` or `false` selects whether the semaphore is initially released;
- `pending_count` is the number of arrivals required by each acquire cycle.

## Rendering acquires, releases, and buffers

Emission tracks the live semaphore tokens while walking one group's nodes in
program order.

### Acquires and releases

An acquire is inserted immediately before its next access or region when
there is one. Its result becomes the current token, and cached buffer views
are cleared because the new token may select another backing copy. It
replaces an older token for the same owner; tokens for other owners remain
available only where the finalized plan explicitly reuses them.

A release is inserted after the operation selected by the finalized plan. It
receives:

- the semaphore to release;
- the exact live token selected for that release;
- the async completion kind, such as `tc5mma`, `tma_load`, or `none`;
- `arrive_count`;
- its finalized partition and pipeline schedule;
- an optional stage-offset operand.

The current token is used by default. A node may use an earlier token owned
by the same partition only when that reuse was explicitly selected before
emission. EMIT-IR does not infer another token path.

### Buffers and accesses

The first access to a member for a given token and owner emits a buffer:

```mlir
%token = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>}
%buffer = nvws.semaphore.buffer %full, %token
    {ttg.partition = array<i32: 1>}
%value = ttg.local_load %buffer {ttg.partition = array<i32: 1>}
```

Later accesses may reuse that buffer while the same token and owner remain
active. For an access through `memdesc_index`, `memdesc_subview`,
`memdesc_trans`, `memdesc_reinterpret`, or `memdesc_reshape`, emission clones
the alias chain on top of the semaphore buffer and retargets the access to
the cloned result. A type-identical `memdesc_index` step is elided because it
would not change the view.

A `ttg.local_alloc` or `ttng.tmem_alloc` with an initial value is rendered
as a store into its semaphore buffer. The original allocation result is then
replaced at the group's accesses. The original allocation remains until all
groups have rendered; cleanup erases it once it is unused.

## Tokens through regions

Region boundaries do not emit standalone operations. They are represented by
token operands, results, and `scf.yield` values.

### A loop-carried token

When a loop must preserve a token across iterations, the emitter adds one
`iter_arg` and one result for that group:

```mlir
%entry_token = nvws.semaphore.acquire %empty

%loop_token = scf.for %i = %lb to %ub step %step
    iter_args(%carried = %entry_token) -> (!ttg.async.token) {
  %buffer = nvws.semaphore.buffer %empty, %carried
  ttg.local_store %value, %buffer
  nvws.semaphore.release %full, %carried [#nvws.async_op<none>]

  %full_token = nvws.semaphore.acquire %full
  %read_buffer = nvws.semaphore.buffer %full, %full_token
  %value = ttg.local_load %read_buffer
  nvws.semaphore.release %empty, %full_token [#nvws.async_op<none>]

  %next = nvws.semaphore.acquire %empty
  scf.yield %next : !ttg.async.token
}
```

The loop input is the exact token needed by the zero-trip path. The body
yields the exact token selected for the next iteration. The loop result is
then the live token after the loop.

If a loop does not need to carry a token, its signature is unchanged. Any
acquire and release assigned to the loop body are rendered directly at their
planned locations.

### A token returned by `scf.if`

An `if` may return a new token from one branch and pass its input token
through the other:

```mlir
%result = scf.if %condition -> (!ttg.async.token) {
  %fresh = nvws.semaphore.acquire %empty
  scf.yield %fresh : !ttg.async.token
} else {
  scf.yield %input : !ttg.async.token
}
```

Each branch yield is filled from that branch's exact final token. When
several buffer groups use the same `if`, all token results are added in one
signature rewrite and each group fills only its own slot. Partition-output
metadata is extended at the same time. One semaphore is selected for each
added result slot, and both paths must supply the token that reaches that
boundary for the selected owner. The region result becomes the active token;
an `if` with no added token result leaves the outer token unchanged.

## Scheduler-safe conditional boundaries

A branch can begin with a release and end with the acquire whose token is
returned by the `if`. Keeping both protocol operations inside the body makes
the branch boundary difficult to schedule. When this exact shape is present,
`finishIfRender` splits the protocol boundary into three `if` operations.

The input shape is:

```mlir
%token = scf.if %condition -> (!ttg.async.token) {
  nvws.semaphore.release %full, %input [#nvws.async_op<tc5mma>]
  // Branch work.
  %next = nvws.semaphore.acquire %empty
  scf.yield %next : !ttg.async.token
} else {
  scf.yield %input : !ttg.async.token
}
```

The final shape below omits the dead token slot on the body `if`:

```mlir
// Release boundary.
scf.if %condition {
  nvws.semaphore.release %full, %input [#nvws.async_op<tc5mma>]
}

// Original branch work, without the boundary release and acquire.
scf.if %condition {
  // Branch work.
}

// Acquire boundary and the live token result.
%token = scf.if %condition -> (!ttg.async.token) {
  %next = nvws.semaphore.acquire %empty
  scf.yield %next : !ttg.async.token
} else {
  scf.yield %input : !ttg.async.token
}
```

The same transformation works when the protocol operations are in the else
branch. A TMEM branch that acquires another token later may need only the
release `if`; in that case the body retains its later acquire and no acquire
boundary is added. Another accepted shape has the release immediately before
the original `if` and an acquire at the start of the then branch; the
same release/body/acquire boundaries are produced.

The split is performed only after every group using the original `if` has
rendered it. This preserves all token slots and lets the three resulting
operations receive partition metadata from their actual contents:

- when both boundaries are present, the release and acquire `if` operations
  cover the union of the release and acquire partitions; the release `if`
  has no result, while the acquire `if` returns the token;
- a release-only `if` follows the release partition;
- the body `if` covers its remaining operations and results.

If a moved release has no pipeline schedule, it uses the schedule of the
nearest preceding MMA in the block. Other generated operations retain the
schedule already selected for them.

## Schedule preservation

Generated acquires and releases receive their stored `loop.stage`,
`loop.cluster`, and partition. A buffer receives the schedule of the access
it guards. These placements are transcribed; emission does not choose a new
placement.

Stage offsets are separate SSA operands:

```mlir
%stage = arith.constant -1 : i32
%token = nvws.semaphore.acquire %full[%stage] ...
%buffer = nvws.semaphore.buffer %full[%stage], %token ...
nvws.semaphore.release %empty[%stage], %token ...
```

An acquire or release uses its protocol stage offset. A buffer uses the
offset selected for that access. These values choose a backing copy later;
they are not `loop.stage` attributes.

## Cleanup and output checks

After rendering, cleanup repeatedly removes dead token results and loop
arguments. It then removes unused alias operations, original allocations,
and the function-level poison token when nothing refers to it.

The final verifier checks three properties:

1. When `ttg.partition.outputs` is present, it has one entry per `scf.for` or
   `scf.if` result, and each yielded partitioned value is compatible with its
   output entry.
2. Within one block, a token has no ordinary semaphore buffer use after a
   release using that token. A buffer marked for the explicitly selected
   same-owner reuse is the only exception.
3. A loop carries at most one semaphore token for one physical backing.
   Circular local-memory backing is excluded because its groups intentionally
   share that allocation.

These checks ensure that the emitted token and buffer paths are complete
before semaphore lowering assigns physical stages and phases.

## Code map

[`InsertSemasEmitIR.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp):

- `emitIR`: the emission order and cleanup.
- `nukeGroupTokens`, `eraseDeadTokenSlots`: original-token and dead-slot
  cleanup.
- `emitPhysicalIR`, `materializeLogicalBacking`, `materializeCircular`,
  `materializeMixedDepth`: backing and semaphore creation.
- `emitEntryAcquires`, `rewriteSignatures`: entry tokens and region slots.
- `RenderState`, `renderChain`, `renderAccess`, `getView`, `renderRegion`:
  tokens, buffers, accesses, loops, and conditionals.
- `finishIfRender`: scheduler-safe conditional boundaries.
- `verifyEmittedIR`: final token, loop-slot, and partition checks.
