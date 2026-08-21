# TMA Store Wait Pipeline

**File**: `WSTMAStoreLowering.cpp`, `WSMemoryPlanner.cpp`

Native Meta keeps `tt.descriptor_store/reduce` through partition scheduling,
then `doConvertDescriptorStoresToNVWS` converts each operation to an empty
staging `LocalAllocOp`, a producer-owned `LocalStoreOp`, and a resultless
`nvws.descriptor_store/reduce`. The NVWS operation remains tokenless through
memory planning, code partitioning, subtile handling, channel-token lowering,
and loop scheduling. Only the post-schedule wait-placement phase creates the
TTNG issue, TMA completion token, and `TMAStoreTokenWaitOp` (see
[Memory Lowering](MemoryLowering.md#tma-storereduce-lowering)).

The legacy `doTMAStoreLowering` route remains for non-Meta, Meta-to-NVWS,
explicit compatibility mode, and native-Meta bailout. Both representations use
the same planner metadata and post-subtile rotation-policy boundary.

## Memory Planner: `buffer.tmaStaging` Handling

**File**: `WSMemoryPlanner.cpp` (within `allocateSmemBuffers`)

The explicit `local_alloc` staging is visible to the memory planner in both
native-Meta and legacy forms. Each WSBuffer is classified through the common
store-like interface, with the concrete reduce forms retaining their distinct
tag (Phase 1):

```cpp
for (auto user : alloc->getUsers()) {
    if (isa<ttng::AsyncTMAReduceOp,
            nvws::DescriptorReduceOp>(user)) {
        buf.tmaStaging = 2;   // TMA atomic-add reduce staging (dq, ...)
        break;
    }
    if (isTiledTMAStoreLike(user))
        buf.tmaStaging = 1;   // regular TMA store staging (dk, dv, c, o, ...)
}
```

`buf.tmaStaging` is a classification tag (0 = not TMA staging, 1 = store,
2 = reduce), not a count. It is propagated to the resulting `local_alloc`
op as the `buffer.tmaStaging` attribute and consumed by later passes that
need to distinguish reduce staging from regular store staging.

The flag drives a special path through three phases:

### Phase 3.5: TMA Staging Fusion

TMA staging WSBuffers are merged by descriptor and originating load (via
`fuseEpilogueWSBuffers`). If the originating load cannot be traced, as with
Hopper register accumulators, the producer task is used instead. For example,
the 4 same-task subtile stagings of `desc_dq` (under `EPILOGUE_SUBTILE=4`) all
share one `bufferId`, as do the dk and dv subtile stagings (each per their own
descriptor). Different data-partition tasks targeting the same descriptor stay
in distinct buffers.

The shared `bufferId` is honored by `doCodePartition` downstream: the
subtile allocs are physically merged into one `local_alloc` of shape
`numCopies x original_shape` (e.g., `1x128x32xf32` for `numCopies=1`).
All subtile stores then index into the same physical SMEM region via
`memdesc_index`. As a result `computeTotalSmem`'s cost model
(`max(size) × copies` per `bufferId`) matches the actual post-merge
physical footprint — there is no per-entry multiplier to add.

### Phase 3.6: Inter-Buffer SMEM Reuse via `allocation.reuseTarget`

When the post-Phase-3.5 SMEM total still exceeds `smemBudget`,
the planner runs an inter-buffer reuse pass (`Phase 3.6: Reuse
allocated buffers when base total exceeds budget` in
`WSMemoryPlanner.cpp::allocateSmemBuffers`). It looks for a host
buffer whose physical SMEM region can be overlapped with a staging
buffer's region without a liveness conflict (the host's last consumer
finishes before the staging's first store), and annotates the staging
alloc with `allocation.reuseTarget = <host bufferId>`. The annotated
WSBuffer's footprint is then counted against the host's region in the
planner's `computeTotalSmem` accounting (i.e. it costs 0 extra bytes
as long as `max(host, staging × copies)` fits inside the host's
existing allocation).

Concrete example for FA-bwd: with `_BWD_DOT_ATTRS_TMEM` and
`BLOCK_M1=128`, the planner annotates

```mlir
%desc_dv_staging = ttg.local_alloc {
  allocation.reuseTarget = 3 : i32,   // v's buffer.id
  buffer.copy = 2, buffer.id = 22, buffer.tmaStaging = 1
} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>

%desc_dk_staging = ttg.local_alloc {
  allocation.reuseTarget = 4 : i32,   // do's buffer.id
  buffer.copy = 2, buffer.id = 24, buffer.tmaStaging = 1
} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
```

so that the planner can keep both staging groups at `numCopies=2`
inside the 220 KB budget by treating them as "free" (they share v's
and do's 32 KB regions respectively).

**Important:** the planner sets the attribute and accounts for the
reuse, but the SMEM layout pass (`AllocateSharedMemoryNv`) does *not*
read `allocation.reuseTarget` directly — it sees the staging alloc as
a request for its own region. Realizing the reuse therefore requires
an extra step in `doCodePartition` that rewrites the staging alloc
into a reinterpret view of the host alloc (see
[CodePartition: Realizing `allocation.reuseTarget`](#code-partition-realizing-allocationreuetarget)
below).

### Phase 4.5: Epilogue Group Copy Increase

Each fused P2_Other group (including TMA staging groups) is treated as
an epilogue group. `increaseFusedEpilogueCopies` iteratively bumps
`numCopies` from the current value up to `numBuffers`, accepting each
bump as long as `computeTotalSmem ≤ smemBudget`.

#### `K | S` cap for same-partition (wait_group-drained) staging

A staging group holds **S** subtiles (`S = indices.size()`, the fused
`EPILOGUE_SUBTILE` / `DQ_SUBTILE` stores) that rotate through **K =
numCopies** slots. How the buffer is drained — and therefore which K values
are *correct* — depends on the channel's producer/consumer topology:

- **Same-partition staging** (producer and consumer in the same warp task,
  e.g. bwd `dk`/`dv`/`dq`): drained only by `cp.async.bulk.wait_group(K-1)`,
  with the per-tile slot `theIdx % K` (see `getStaggeredAccumCnt`). The fixed
  in-flight-count wait is correct only if same-slot stores are exactly K apart
  across the tile boundary, i.e. **K must divide S**. A non-dividing K (e.g.
  `K=3, S=4`, or `K > S`) makes a store clobber a slot before it drains →
  wrong gradients, with nothing downstream to catch it.
- **Cross-partition staging** (producer task ≠ consumer task, e.g. fwd
  `desc_o`): the slot/phase come from a continuous `accumCnt` producer/consumer
  mbarrier rotation and tolerate **any** K.

Phase 4.5 therefore detects same-partition staging via
`getAsyncTaskIds(ch->getSrcOp()) == getAsyncTaskIds(ch->getDstOp())` and, for
those groups, only accepts a `tryCopies` that divides S (others are skipped).
This is a **defensive backstop**: in all shipped configs `num_stages = 2` and
`S ∈ {2,4}`, so `K | S` already holds and the cap is a no-op — it exists to
keep a future `num_stages`/subtile combination from silently violating the
rotation invariant. The cross-stage floor (`WSBuffer::minCopies`) is a hard
correctness floor and is never reduced; if it itself violates `K | S` for a
wait_group ring the pass emits an `LDBG` warning rather than dropping below it.

`computeTotalSmem` accounts for the active `buffer.id` groups using
`max(size) × copies` per ID. For TMA store staging groups this matches the
planner's reuse model, but the downstream physical allocation still leaves
each staging alloc as a separate op.

### Combined SMEM Cost

If the planner needs an exact hardware-SMEM accounting pass for TMA store
staging, the cost needs per-entry accounting because the allocs are NOT merged
into one physical alloc downstream:

```
tmaStoreSmem = numEntries * size * copies
```

### Phase 6: Hoist Before Outermost Loop

All TMA staging allocs are moved before the outermost enclosing
`scf.for` or `scf.while` loop when every allocation operand is invariant with
respect to that loop. This keeps persistent-loop iterations on one planned
staging allocation and makes it available to the rotation mechanism. A
non-invariant allocation is left in place rather than hoisted illegally.

## Wait Annotation and Reordering Pipeline

Within the AutoWS monolithic pass (`WarpSpecialization.cpp`), abstract and
legacy forms converge at the post-subtile annotation and post-schedule
materialization boundaries:

```
doMemoryPlanner
  → doGenerateSubtiledRegion
  → doAnnotateAbstractTMAStores + validation
  → doAnnotateTMAStoreWaits + validation  ← legacy compatibility form
  → doCodePartition
  → doLowerSubtiledRegionsWithNVWSOps
  → doTokenLowering             ← resolve completion tokens to barriers
  → scheduleLoops                 ← SWP assigns pipeline stages
  → cleanupWarpSpecializedLoops   ← prune dead loop-carried values
  → doMaterializeAndPlaceTMAStoreWaits
      ├─ materialize NVWS → TTNG issue/token/wait unconditionally
      └─ optionally move waits using the SWP schedule
```

This sequence is controlled by the `nvgpu-warp-specialization` pass option
`tma-store-pipelining`, which defaults to `true`. When set to `false`, AutoWS
skips annotation, validation, and post-schedule reordering. It does not skip
late materialization: each abstract operation still becomes its TTNG
issue/token/wait, with the wait left adjacent to the issue. The independent
`early_tma_store_lowering=true` option selects the legacy representation before
warp specialization.

Each function is also available as a standalone MLIR pass for use outside
the monolithic pipeline.

## Step 1: Annotate Rotation Policy

**Test pass**: `nvgpu-test-annotate-tma-store-waits` (`NVGPUTestAnnotateTMAStoreWaitsPass`)

At the post-subtile boundary, `doAnnotateAbstractTMAStores` inspects
`nvws.descriptor_store/reduce`; `doAnnotateTMAStoreWaits` handles legacy
`TMAStoreTokenWaitOp`. For a legacy wait, it first traces the token back to the
defining TTNG store/reduce. Both paths then inspect the SMEM source:

1. Get the `LocalAllocOp` that produces the buffer.
2. Read the `buffer.copy` attribute (set earlier by the memory planner),
   which records how many physical copies of this buffer exist.
3. If `buffer.copy = K`, set `can_rotate_by_buffer_count = K` on the abstract
   operation or legacy wait.

The attribute means: "K buffer copies exist, so the eventual wait can be delayed
until the K-th subsequent TMA store to the same buffer — at that point
the buffer slot is about to be overwritten and the earlier store must
have finished reading."

Annotation intentionally runs after subtile generation. A subtiled source may
already be a region argument at this boundary; if the underlying planned alloc
cannot be recovered, no rotation attribute is attached. Applying the same rule
to abstract and legacy forms prevents the representation change from enabling
a rotation that the legacy pipeline did not perform.

### Token Tracing

`getDefiningTMAStoreOp` handles two cases:

| Case | Pattern |
|------|---------|
| **Direct** | Token is the direct SSA result of `AsyncTMACopyLocalToGlobalOp` or `AsyncTMAReduceOp` |
| **Loop-carried** | Token is a block argument of the `scf.for` body; the function follows the corresponding yield operand back to its `AsyncTMACopyLocalToGlobalOp` or `AsyncTMAReduceOp` |

## Step 2: Validate Rotation Policy

This is a safety pass that runs immediately after annotation. It
re-checks every annotated abstract operation and legacy wait. The abstract form
uses `doValidateAbstractTMAStoreAnnotations`; the legacy form uses
`doValidateTMAStoreAnnotations`. Either strips
`can_rotate_by_buffer_count` when the underlying `LocalAllocOp` can no longer
be resolved.

## Completion/Release Metadata

The public NVWS form has no TMA completion-token result, but the operation can
temporarily carry paired internal operands:

- deferred `nvws_tokens` and indices attached by `doCodePartition`; or
- resolved completion barriers and predicates produced by `doTokenLowering`.

The abstract store/reduce itself is the final staging-buffer consumer during
code partitioning. For a subtiled operation, the inner operation receives the
captured token/index and the containing region is inlined before token lowering
examines token users. Token lowering replaces the deferred pairs with the same
barrier/predicate pairs that the legacy path attaches to
`TMAStoreTokenWaitOp`.

## Step 3: Materialize and Reorder

**Test pass**: `nvgpu-test-tma-store-token-wait-reorder` (`NVGPUTestTMAStoreTokenWaitReorderPass`)

`doMaterializeAndPlaceTMAStoreWaits` runs **after** `scheduleLoops` has assigned
pipeline stages and clusters. It first replaces every abstract operation:

```text
nvws.descriptor_store  -> ttng.async_tma_copy_local_to_global + token_wait
nvws.descriptor_reduce -> ttng.async_tma_reduce + token_wait
```

The generated wait receives any resolved completion barriers/predicates and
the rotation attribute. The conversion fails if deferred NVWS tokens remain,
and verifies that no abstract operation survives. Materialization is
unconditional; only the following call to `doTMAStoreWaitReorder` depends on
`tma-store-pipelining`.

When enabled, reordering uses the SWP `CoarseSchedule` to move waits forward in
the linearized pipeline order.

### Algorithm

For each annotated `TMAStoreTokenWaitOp` with `can_rotate_by_buffer_count = K`:

1. **Deserialize the schedule** from the `scf.for` loop. If no schedule
   exists, create a trivial single-stage schedule so the logic can still
   proceed.

2. **Linearize from the defining TMA store**: use
   `schedule.linearized(forOp, tmaStore)` to get an iterator that walks
   ops in pipeline-unrolled order (wrapping across stages up to
   `numStages + K`). Note: That we may only increase by 1 stage (we move
   by K TMA stores, not necessarily K pipeline stages).

3. **Count K stores**: walk the linearized schedule, counting
   `AsyncTMACopyLocalToGlobalOp` and `AsyncTMAReduceOp` ops. Stop at the K-th
   store-like op — its source slot is the one about to be reused.

4. **Find the overwrite boundary**: locate the scheduled `ttg.local_store`
   that writes the target issue's source memdesc and insert before it. If that
   writer belongs to another partition and is not directly visible, insert
   before the scheduled `WaitBarrierOp` guarding the target issue. If neither
   target can be proven, leave the wait in its adjacent fallback position.

5. **Update the schedule**: split the cluster at the insertion target and
   create a new cluster for the wait op, assigned to the target's pipeline
   stage. Serialize the modified schedule back to the loop.

6. **Remove the annotation**: strip `can_rotate_by_buffer_count` from the
   wait op.

### Example

With `buffer.copy = 2` (double-buffered) and a 3-stage pipeline:

```
Stage 0: AsyncTMACopyLocalToGlobal (store to buffer[0])
         TMAStoreTokenWait          ← originally placed here
Stage 1: AsyncTMACopyLocalToGlobal (store to buffer[1])
Stage 2: LocalStore (next value to buffer[0])
         AsyncTMACopyLocalToGlobal (next store from buffer[0])
```

After reordering with K=2, the wait moves forward to just before the writer for
the second subsequent issue, which would overwrite buffer[0]:

```
Stage 0: AsyncTMACopyLocalToGlobal (store to buffer[0])
Stage 1: AsyncTMACopyLocalToGlobal (store to buffer[1])
Stage 2: TMAStoreTokenWait          ← moved here
         LocalStore (next value to buffer[0])
         AsyncTMACopyLocalToGlobal (next store from buffer[0])
```

This allows the compute in stage 1 to overlap with the asynchronous TMA
store instead of stalling.

## Final Lowering: `NVGPUTMAStoreTokenWaitLoweringPass`

**Pass**: `nvgpu-tma-store-token-wait-lowering`

After reordering, a separate pass lowers each `TMAStoreTokenWaitOp` into
concrete hardware operations:

1. **Compute pendings**: count TMA store-like ops
   (`AsyncTMACopyLocalToGlobalOp` or `AsyncTMAReduceOp`) between the
   defining store and the wait (in program order). For loop-carried
   tokens, this wraps around the loop body boundary.
2. **Emit `TMAStoreWaitOp`**: waits until at most `pendings` TMA stores
   remain in flight.
3. **Emit `ArriveBarrierOp`**: for each barrier attached to the wait,
   signals that the SMEM buffer is now free for reuse.
4. **Erase** the original `TMAStoreTokenWaitOp`.

See also [Memory Lowering](MemoryLowering.md) for the broader context of
how TMA stores fit into the WS memory lowering pipeline.

## Code Partition: Realizing `allocation.reuseTarget`

**File**: `WSCodePartition.cpp::doCodePartition`

The planner's `allocation.reuseTarget = N` annotation is a SMEM-accounting
hint. To actually realize the SMEM overlap in the emitted kernel, two
coordinated steps are required inside `doCodePartition` before the layout
pass (`AllocateSharedMemoryNv`) runs:

### Step 1: Insert the Cross-Tile Reuse WAR Barrier (Step 7.5)

Because the staging buffer aliases the host operand SMEM, the **next tile's
operand load** must not overwrite that SMEM until the **previous tile's staging
TMA store** has finished reading it. Step 7.5 emits a dedicated, single-buffered
cross-partition token for this write-after-read edge:

- the **load task** does a `producer_acquire` at the **top of the persistent
  outer loop** (before the operand loads), and
- the **staging task** does a `consumer_release` at the **bottom of the outer
  loop** (after the staging stores, which have already drained).

Both are loop-carried (phase derived from the outer-loop induction variable via
`getBufferIdxAndPhase`), so the edge serializes `load(tile N+1)` after
`staging-store(tile N)`. The load and staging genuinely alias the same SMEM, so
this serialization removes no legitimate overlap.

```mlir
scf.for %tile = ... {
  nvws.producer_acquire %reuse_token, %idx, %phase   // load task, top
  ... operand loads (v/do) ...
  ... MMAs, epilogue, dv/dk staging stores + async_tma_store_token_wait ...
  nvws.consumer_release %reuse_token, %idx           // staging task, bottom
  scf.yield
}
```

**Why a dedicated token (not the host operand's own barrier).** The host
operand buffer's empty barrier is released by its MMA consumer, which finishes
*before* the staging store — too early. It also cannot carry an extra release
from the staging task: the staging task and the MMA task have different warp
counts, which trips the `consumerWarps == nWarps` assert in `WSLowerToken`. A
separate token has a uniform consumer (the staging task), avoiding the conflict.

**Earlier (degenerate) design — fixed.** Step 7.5 previously emitted a
`producer_acquire` on the *host* token before the staging store, with constant
`bufferIdx = 0` / `phase = 0`. That gated the wrong direction (the staging
*write* after the host read, which the dv/dk accumulator-complete wait already
covers) and, with a constant phase, never alternated across the persistent loop
— a no-op. It happened to be harmless on the non-persistent (single-tile) path
but left the persistent path with a cross-tile SMEM race (non-deterministic
wrong dv/dk gradients). The cross-tile WAR token above replaces it. Regression
coverage: `test_bwd_tmem_dsT_reuse_3group_persistent`.

**Important ordering constraint**: this insertion **must** run **before**
`insertAsyncComm`, whose cleanup sweep (`removeTokenfNotUsed`) erases any token
alloc currently lacking users — the freshly-created reuse token only gains uses
once the acquire/release are inserted. This is enforced as "Step 7.5" — running
between `createToken` (Step 7) and `insertAsyncComm` (Step 8). See commit
`c67893c25` for the related ordering bug-fix.

### Step 2: Reinterpret Staging Alloc into Host Alloc

After Step 7.5 makes the cross-partition timing safe, the staging
`local_alloc` is **replaced** by a `memdesc_reinterpret` view of the
host alloc. All uses of the staging memdesc (TMA copy/reduce ops,
`memdesc_index` for rotation, `local_store`, etc.) are rewritten to
reference the reinterpret view, and the original staging alloc is
erased.

This is analogous to the TMEM path in the existing `replaceBufferReuse`
(which uses `sliceAndReinterpretMDTMEM` to overlay multiple TMEM
allocs into one region — see
[ReuseGroups.md §3 Buffer Replacement](ReuseGroups.md#3-buffer-replacement)),
extended to SMEM:

```mlir
# Before:
%v             = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared>
%dv_staging    = ttg.local_alloc {allocation.reuseTarget = 3}
                                 : () -> !ttg.memdesc<2x128x64xf16, #shared>
... %dv_staging uses ...

# After:
%v             = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared>
%dv_staging.view = ttg.memdesc_reinterpret %v
                  : !ttg.memdesc<128x128xf16> -> !ttg.memdesc<2x128x64xf16>
... %dv_staging.view uses ...
# (original %dv_staging is erased)
```

The reinterpret is sound because the host alloc's storage covers at
least `staging.size × staging.numCopies` bytes (this is exactly the
condition `findReuseCandidate` checks when picking a target).

### Why both steps are necessary

| Step omitted | Symptom |
|---|---|
| Step 1 (cross-partition barrier) | Race: staging writer can clobber the host region while the host's consumer is still reading. |
| Step 2 (reinterpret merge) | `AllocateSharedMemoryNv` places staging at a fresh offset (it has no awareness of `allocation.reuseTarget`), so the planner's "free" accounting is wrong: actual SMEM usage = planner total + Σ(staging × copies). For FA-bwd idx=2 / idx=0 this gap is ~64 KB and pushes the kernel over B200's 232 KB SMEM ceiling at codegen time. |
