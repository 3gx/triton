# NVWS local_alloc memory planner port plan

Status: PLAN ONLY (16jun26).

Scope: `triton-solid-01.git`, NVWS path only. This plan ports the remaining
SMEM/local allocation planning functionality from the original Meta
warp-specialization planner into the NVWS memory planner, then teaches the
semaphore passes to respect the planner's `buffer.copy` depth.

## Goal

Close the gap between:

- Meta/Hopper WS memory planning:
  `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization.cpp`
  calls `doMemoryPlanner(...)`, and
  `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`
  plans both `ttg.local_alloc` and `ttng.tmem_alloc`.
- NVWS memory planning:
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp`
  currently runs only the TMEM allocator from `doTmemMemoryPlanning(...)`.

End state:

1. `--nvws-memory-planner` assigns `buffer.id` and `buffer.copy` to planned
   `ttg.local_alloc` operations, not only to `ttng.tmem_alloc`.
2. `InsertSemas` uses `buffer.copy` as the authoritative backing-buffer depth
   when it is present. This overrides the depth that would otherwise be inferred
   from IR/numStages.
3. `InsertSemas` handles planned local reuse groups, identified by same
   `buffer.id` plus consistent `buffer.copy`, using the same physical-reuse
   model as TMEM: one backing allocation plus the needed
   index/reinterpret/slice view for each logical alloc.
4. `LowerSemaphore` does not rewrite the depth of a local semaphore backing
   buffer when that backing comes from an alloc with `buffer.copy`.

## Current source facts to preserve

- `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td` declares
  `NVWSMemoryPlanner` as a TMEM planner and says the `smem-*` options are
  accepted only for compatibility and ignored.
- `NVWSMemoryPlanner::runOnOperation()` currently calls only
  `doTmemMemoryPlanning(funcOp, numBuffers)`.
- Current standalone NVWS `--nvws-memory-planner` documents `num-buffers` as
  the starting TMEM buffer id, and `test/NVWS/MetaAutoWS/memory_planner.mlir`
  pins `num-buffers=7` to TMEM `buffer.id` values 7 and 8. This is a
  TMEM-only shortcut, not the target contract for the combined planner.
- The NVWS automatic warp-specialization pipeline already passes `numStages`
  into `NVWSMemoryPlannerOptions.numBuffers`, which matches Meta's
  "WS buffering depth" meaning rather than "first TMEM id".
- `InsertSemasSyncDag.cpp::computeBackingPlan` sets
  `g.backingPlan.numStages = 1` by default, and only changes it for TMEM
  accumulator groups.
- `InsertSemasEmitIR.cpp::backingType` prepends
  `g.backingPlan.numStages` to the member type shape for non-scales encodings.
- `InsertSemasEmitIR.cpp::emitBackingsAndCreates` already preserves
  `buffer.id`, `buffer.offset`, and `buffer.copy` attributes onto the new
  backing allocs, but the type depth is still computed from
  `g.backingPlan.numStages`.
- `LowerAref.cpp::multiBufferSemaphore` rewrites eligible non-TMEM semaphore
  buffers to `numStages` whenever the semaphore group has a producer load. It
  does not currently check `buffer.copy`.
- `LowerAref.cpp` already has TMEM duplicate-buffer coalescing by `buffer.id`
  and `buffer.copy`, with sub-slice plus reinterpret view creation.

## Non-goals

- Do not change semaphore ordering, hold rules, acquire/release placement, or
  partition scheduling.
- Do not introduce an env flag to select old/new behavior.
- Do not hand-edit partition metadata in lit tests.
- Do not update broad goldens before the targeted planner/semaphore tests prove
  the new contract.

## Milestones

### M0 - Baseline and direct coverage audit

No behavior change.

1. Record the existing NVWS memory planner entry points:
   - `Passes.td` option descriptions.
   - `MemoryPlanner.cpp::doTmemMemoryPlanning`.
   - `NVWSMemoryPlanner::runOnOperation`.
2. Record the existing Meta planner entry points:
   - `WarpSpecialization.cpp::doMemoryPlanner(...)` call.
   - `WSMemoryPlanner.cpp` local/SMEM planning path.
   - `WSMemoryPlanner.cpp` TMEM planning path.
3. Record the existing lit coverage:
   - `test/NVWS/MetaAutoWS/*memory_planner*.mlir` invokes
     `--nvws-memory-planner`.
   - Existing NVWS memory-planner tests currently do not require
     `ttg.local_alloc {buffer.copy = ..., buffer.id = ...}`.
   - Hopper `test/Hopper/WarpSpecialization/ws_memory_planner*.mlir` does have
     those local_alloc checks.

Exit gate: source-only audit committed in the implementation notes or PR
description; no code changes required in M0.

### M1 - Add NVWS local/SMEM channel discovery

Implement a local channel model beside the existing TMEM channel model in
`third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp`.

Required work:

1. Add a local/SMEM channel type analogous to `TmemDataChannelPost`.
2. Collect channels for `ttg::LocalAllocOp`.
3. Recognize local producers that appear in NVWS IR:
   - `nvws::DescriptorLoadOp` and any sibling descriptor/gather op that writes
     directly into a local memdesc.
   - `ttg::LocalStoreOp`.
   - Sourceful `ttg.local_alloc` if still present in a direct lit input.
4. Recognize local consumers:
   - `ttng::MMAv5OpInterface` users.
   - `ttg::LocalLoadOp`.
   - Consumers reached through memdesc view ops such as `ttg.memdesc_index`,
     `ttg.memdesc_trans`, and `ttg.memdesc_reinterpret`.
5. Use the existing NVWS task-id helper path, not the Meta task-id helper path.

Exit gate:

- M1 must have an observable proof. If channel discovery emits no IR-visible
  metadata, a normal before/after MLIR dump is not sufficient.
- Either add a deterministic planner/channel dump that can be checked in a lit
  test, or treat M1 as an internal implementation step whose first gate is the
  IR-visible allocator output in M2.
- The proof must cover the FA shape: Q, K, V, and output staging local allocs.

### M2 - Port local/SMEM allocation planning

Run local/SMEM planning before TMEM planning, matching the Meta planner order.

Required work:

1. Add a `SmemAllocator` or equivalent local allocator in NVWS
   `MemoryPlanner.cpp`.
2. Port the Meta local allocation behavior from
   `WSMemoryPlanner.cpp`, including:
   - liveness over local producers and consumers;
   - cross-scope liveness propagation;
   - assignment of `buffer.id`;
   - assignment of `buffer.copy`;
   - minimum-copy behavior matching Meta for the selected SMEM algorithm and
     reuse class. Do not apply one unconditional rule to every same-`buffer.id`
     group: cyclic/data-partition reuse may require distinct rotating slots,
     while budgeted epilogue fusion can validly keep same-`buffer.id` buffers at
     `buffer.copy = 1` when liveness and budget allow it;
   - epilogue/output-buffer fusion when Meta would fuse them;
   - annotation handling for `smem` annotations.
3. Wire existing `smem-alloc-algo`, `smem-budget`, and
   `smem-circular-reuse` options to real behavior instead of ignoring them.
   The default path must match the current Meta default. The `smem-alloc-algo=1`
   path must match the Meta WSBuffer path where that option is tested.
4. Match Meta's per-WS-loop SMEM override path. Pass options provide defaults,
   then `tt.smem_alloc_algo`, `tt.smem_budget`, and
   `tt.smem_circular_reuse` on the WS loop or enclosing loop chain override
   those defaults, with the innermost loop taking priority.
5. Align `num-buffers` with Meta semantics. In the combined NVWS planner,
   `num-buffers` means WS buffering depth:
   - local/SMEM planning consumes it when computing `buffer.copy`;
   - local/SMEM planning returns the next free `buffer.id`;
   - TMEM planning starts from that returned id.
   The old standalone TMEM-only interpretation, where `num-buffers` is the
   first TMEM id, must be removed from the target contract.
6. Change the planner driver from TMEM-only to combined planning:
   - collect and allocate local/SMEM first;
   - return the next free `buffer.id`;
   - start TMEM planning from that id so local and TMEM ids do not collide.
7. Preserve existing TMEM behavior unless local id reservation requires the
   expected id offset.

Exit gate:

- New or updated NVWS memory-planner lit checks prove local alloc metadata:
  - FA forward: output staging copy=1; Q copy=1; K/V cross-stage copy equals
    num-buffers and share the Meta-equivalent `buffer.id` where applicable.
  - Backward/persistent cases that already exist in Hopper local planner tests.
  - Epilogue fusion and `smem-budget` behavior.
  - Annotation-driven SMEM pinning.
- Update the old TMEM-only standalone lit contract. In particular,
  `test/NVWS/MetaAutoWS/memory_planner.mlir` must no longer assert that
  `num-buffers=7` means first TMEM id 7 unless a separate explicit
  first-TMEM-id option is deliberately added.
- Existing TMEM reuse checks still pass.

### M3 - Make InsertSemas use `buffer.copy` as backing depth

Update `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas*`.

Required contract:

1. If a member alloc has `buffer.copy`, that value is the backing depth for the
   semaphore backing buffer.
2. `buffer.copy` takes precedence over:
   - local default depth `1`;
   - TMEM depth inferred from `isMultiStagedGroup`;
   - `numStages` passed through the pipeline.
3. For a `buffer.id` group, non-null `buffer.copy` values must be consistent.
   If they are inconsistent, emit a hard diagnostic instead of silently choosing
   one.
4. `BackingPlan` must store the effective planned depth, not just the legacy
   IR-derived `numStages`.
5. `backingType(...)` must use that effective planned depth.
6. Preserve `buffer.copy` on the backing alloc only after the backing type has
   been created with the same depth.

Local reuse-view requirement:

1. When two or more local allocs share the same `buffer.id` and have planned
   physical-reuse metadata, InsertSemas must not materialize independent
   physical buffers for them. The intended trigger is planned reuse, not
   same-`buffer.id` alone; in practice this means `buffer.copy` is present and
   consistent for the local reuse group.
2. It must create the correct logical view from the shared backing:
   - index by stage/copy where needed;
   - apply offset/slice semantics if `buffer.offset` is present;
   - create `ttg.memdesc_reinterpret` or the existing local memdesc view op
     needed to match the original alloc's memdesc type.
3. The resulting access replacement must have the same logical memdesc type as
   the original local alloc access site.
4. This is the local analogue of the existing TMEM same-`buffer.id` reuse view
   in `LowerAref.cpp`.

Exit gate:

- A direct InsertSemas lit test with `ttg.local_alloc {buffer.copy = N}` shows
  a backing type with leading depth `N`.
- A direct InsertSemas lit test with two local allocs sharing one `buffer.id`
  and planned reuse metadata shows shared backing plus the required
  view/reinterpret, not two independent physical buffers.
- Existing no-`buffer.copy` local tests keep their current depth and physical
  backing behavior. In particular, current local same-`buffer.id` semantic
  tests such as `test/NVWS/insert_semas_local_buffer_reuse.mlir` and
  `test/NVWS/insert_semas_transitive_reduction.mlir` should remain unchanged
  unless they are intentionally converted into planned-reuse test cases with
  `buffer.copy`.

### M4 - Make LowerSemaphore preserve planned local depth

Update `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp`.

Required contract:

1. `multiBufferSemaphore` makes a whole semaphore-group decision, not a
   per-operand decision.
2. The legacy `numStages` rewrite is allowed only when every semaphore operand
   is a direct `ttg.local_alloc` result and none of those allocs has
   `buffer.copy`.
3. If any semaphore operand is not a direct `ttg.local_alloc` result, leave the
   whole group unchanged. This includes `ttg.memdesc_index`,
   `ttg.memdesc_subview`, `ttg.memdesc_trans`, `ttg.memdesc_reinterpret`, or
   any other view derived from a planned local alloc.
4. If any direct local alloc operand has `buffer.copy`, leave the whole group
   unchanged.
5. LowerSemaphore must not remove `buffer.copy` from local allocs. Existing TMEM
   attr stripping behavior remains separate.

Exit gate:

- A LowerSemaphore lit test proves a planned local backing with
  `buffer.copy = 1` is not rewritten to `numStages = 2/3`.
- A LowerSemaphore lit test proves a planned local backing with
  `buffer.copy = N` keeps depth `N`.
- A LowerSemaphore lit test proves a semaphore operand that is a local
  view/reinterpret derived from a planned alloc is not rewritten.
- Existing unplanned local semaphore behavior remains unchanged.

### M5 - End-to-end gates

Build first:

```sh
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Then run targeted lit tests from the build directory:

```sh
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  test/NVWS/MetaAutoWS \
  test/NVWS/lower_semaphore.mlir \
  test/NVWS/insert_semas*.mlir \
  test/TritonGPU/automatic-warp-specialization.mlir
```

Then run the reproducer pair with fresh dumps:

```sh
MLIR_ENABLE_DUMP=1 sh run_nvws.sh
PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/ \
MLIR_ENABLE_DUMP=1 sh run_meta.sh
```

Required end-to-end result:

- `run_nvws.sh` no longer fails with shared-memory OOR caused by local_alloc
  over-depth.
- Before allocate-shared-memory, NVWS local alloc depths match the planned
  `buffer.copy` values.
- LowerSemaphore does not inflate planned local buffers after InsertSemas.
- The shared-memory total is below the hardware limit for the repro.

## Blocker protocol

Stop and report if any of these are observed:

1. NVWS local channel discovery cannot identify a producer/consumer relation
   that the Meta planner can identify for the same logical buffer.
2. The planner emits inconsistent `buffer.copy` for one `buffer.id` group.
3. InsertSemas needs a local view/slice op that does not exist or cannot
   preserve the original memdesc type.
4. LowerSemaphore still changes the depth of an alloc that has `buffer.copy`.
5. A Hopper-equivalent local planner test cannot be expressed in NVWS IR without
   changing the intended semantics.

For each blocker, report:

- exact file:line evidence;
- exact before/after IR around the affected alloc;
- whether the failure is in planning, InsertSemas materialization, or
  LowerSemaphore preservation.

## Implementation order

1. Add local channel discovery, with one direct lit test that only proves
   discovery/emission once the allocator lands.
2. Add local allocator and wire combined local-then-TMEM planning.
3. Add NVWS memory-planner lit checks for local `buffer.id`/`buffer.copy`.
4. Update InsertSemas depth selection and local same-`buffer.id` views.
5. Add InsertSemas local depth/reuse tests.
6. Update LowerSemaphore planned-depth skip.
7. Add LowerSemaphore preservation tests.
8. Build, run targeted lit, then rerun `run_nvws.sh`/`run_meta.sh` dumps.

## Chapter 2: circular buffer

Status: DESIGN ONLY extension to Chapter 1.

Design reference: `fable/circular-buffer-disign.md`.

Chapter 1 ports local allocation planning and makes `buffer.copy` authoritative
for planned local depth. Chapter 2 adds the missing circular-local-reuse
contract. Without this chapter, two local allocs such as K and V can share one
physical backing and depth, but NVWS still has no IR fact saying which logical
channel uses which slot in the circular ring.

### Goal

Support local circular reuse groups without aliasing distinct logical buffers
into the same physical SMEM slot.

End state:

1. The memory planner marks circular local reuse explicitly with
   `buffer.circular = true` and `buffer.start`.
2. `buffer.start` is a stable channel-order seed in the circular group. It is
   assigned from first-producer program order and is not copied directly into
   semaphore stage operands.
3. Circular local members with the same `buffer.id` are independent logical
   semaphore channels but share one physical SMEM data backing.
4. InsertSemas builds independent logical semaphore streams for circular
   members, then folds streams with the same physical `semaphore.id`.
5. After InsertSemas and before AssignStagePhase, the existing
   `nvws.semaphore.acquire` / `release` / `buffer` stage operand carries a
   signed circular offset, e.g. `-1`.
6. AssignStagePhase consumes that stage operand as an offset, computes the final
   physical stage modulo `buffer.copy`, and overwrites the stage operand before
   LowerAref.
7. `pending_count` is authored and validated before or during InsertSemas
   folding. After that it is ground truth.
8. LowerAref remains unchanged for circular stage lowering and sees only final
   physical stage operands.
9. `SemaphoreCreateOp::verify()` and LowerAref / lower-semaphore use authored
   `pending_count` verbatim for folded circular semaphores and do not re-derive
   it from folded physical IR.

### Metadata contract

For a circular local reuse group, every member alloc has:

```mlir
buffer.id = N : i32
buffer.copy = D : i32
buffer.circular = true
buffer.start = S : i32
```

Required invariants:

1. The marker is valid only on `ttg.local_alloc`.
2. `buffer.copy > 0`.
3. `0 <= buffer.start < buffer.copy`.
4. All circular allocs with one `buffer.id` have identical `buffer.copy`.
5. All circular allocs with one `buffer.id` have identical logical local buffer
   size. The implementation should enforce identical memdesc type unless a
   later design adds a local reinterpret/slice contract.
6. `buffer.start` values in one circular group are distinct.
7. `buffer.offset` and `buffer.circular` are mutually exclusive.

`buffer.offset` remains the TMEM spatial-packing marker. `buffer.circular` plus
`buffer.start` is the local/SMEM circular-channel marker.

### C0 - Memory planner emits circular markers

Required work:

1. When SMEM circular reuse chooses a local reuse group, assign the same
   `buffer.id` and `buffer.copy` to all members.
2. Add `buffer.circular = true` to those members.
3. Add stable `buffer.start` values in first-producer program order. For the
   K/V two-channel case, if K is produced before V, K gets start `0` and V gets
   start `1`.
4. Do not add `buffer.offset` to circular local groups.
5. Assert or diagnose if the group members do not have identical local
   size/type.
6. Do not mark non-circular same-`buffer.id` cases as circular. Epilogue
   liveness fusion and TMEM spatial packing keep their existing metadata.

Exit gate:

- A memory-planner lit test proves the K/V shape has one shared `buffer.id`,
  shared `buffer.copy`, `buffer.circular = true`, and starts `0/1`.
- The same test or a paired InsertSemas test proves `buffer.start` matches
  first-producer program order.
- A negative or avoidance test proves unequal local sizes are not circularized.

### C1 - InsertSemas models circular members as logical channels

Implementation shape:

- The normal InsertSemas emitter must first render the logical semaphore IR
  that comes out of SyncDag. It should emit independent logical K/V
  acquire/buffer/release streams and their logical semaphore creates.
- Circular physical sharing is a required post-render IR rewrite inside
  InsertSemas, after the normal render walk has emitted logical IR and before
  InsertSemas exits.
- That post-render circular rewrite is where the pass folds logical
  `nvws.semaphore.create` operations with the same fold key, rewrites their
  uses to the folded physical create, replaces/coalesces logical local
  backings with one shared circular backing, and authors circular stage
  offsets on the already-emitted acquire/release/buffer operations.
- The same post-render rewrite must preserve token semantics. Event-local
  acquire tokens may remain distinct, and live loop-carried permission tokens
  for logical streams that fold onto one physical backing are not
  interchangeable. Do not arbitrarily coalesce several live carried tokens to
  one survivor. For circular semaphores, tokens carry permission only; they do
  not carry stage.
- Do not implement circular folding by making the normal emitter directly emit
  folded physical semaphore creates or special shared circular backings. The
  normal emitter remains a transcription of the logical SyncDag; circular
  folding is a separate post-processing step, analogous in placement to the
  existing post-emit backing rewrites.

Required work:

1. Validate circular invariants during group construction.
2. Do not treat same-`buffer.id` circular members as overlapping logical pieces.
   They are independent logical channels separated by `buffer.start`.
3. Do not emit one variadic `nvws.semaphore.buffer` for multiple circular
   members. Current NVWS has one stage operand per buffer op, not one per
   result.
4. Emit separate logical acquire/buffer/release rows for circular members.
5. In the post-render circular rewrite, materialize one physical local backing
   for the circular group, with leading depth equal to `buffer.copy`, and
   rewrite the logical local backings/views to use it.
6. Assign fold keys so K/V empty logical semaphores fold to one physical empty
   semaphore and K/V full logical semaphores fold to one separate physical full
   semaphore:

   ```text
   K_empty.semaphore.id = V_empty.semaphore.id = E
   K_full.semaphore.id  = V_full.semaphore.id  = F
   E != F
   ```

7. Compute and author `pending_count` while logical semaphore streams are still
   distinct.
8. In the post-render circular rewrite, fold logical semaphores with the same
   `semaphore.id`. Validate that all logical semaphores for that id have the
   same authored `pending_count`. The folded `nvws.semaphore.create` receives
   that value. Do not recompute the folded count from the post-fold physical
   release stream.
9. After folding logical semaphores/backings, preserve all live loop-carried
   permission tokens as-is. Multiple carrier token slots for one folded circular
   physical semaphore are legal. Do not coalesce them. Do not require per-token
   stage threading. Circular stage assignment is driven by the shared cursor plus
   authored per-op offsets.
10. Relax or bypass `verifySingleCarrierPerGroup` for folded circular groups.
    The existing one-carrier-slot verifier remains valid for non-circular
    groups only.
11. Update `NVWS_SemaphoreAcquireOp` assembly/parser/printer so acquire supports
   all three forms:

   ```mlir
   nvws.semaphore.acquire %sem
   nvws.semaphore.acquire %sem[%stage]
   nvws.semaphore.acquire %sem[%stage, %phase]
   ```

   Before AssignStagePhase, circular acquire uses the stage-only form and the
   stage value is a signed offset. After AssignStagePhase, acquire uses the
   stage+phase form.
12. Compute a signed circular offset for each folded semaphore event from the
   event order, not directly from `buffer.start`.
13. Author that offset in the existing stage operand on folded
   `nvws.semaphore.acquire`, `nvws.semaphore.release`, and
   `nvws.semaphore.buffer` operations. A negative offset such as `-1` is valid
   in this pre-AssignStagePhase IR.
14. Validate that first-producer program order matches ascending
    `buffer.start`; mismatch is malformed circular IR and must not be silently
    ignored.
15. Do not add a `stage.offset` attribute.
16. Preserve `buffer.circular`, `buffer.start`, `buffer.id`, and `buffer.copy`
   where downstream passes can inspect them.

Exit gate:

- An InsertSemas lit test shows circular K/V as separate logical semaphore
  rows, not one two-result `nvws.semaphore.buffer`.
- The same test shows one physical local backing for the circular group.
- The same test shows folded circular semaphore ops carrying stage operands
  that are offsets before AssignStagePhase, including the `-1` K-consumer case
  for `ld k; ld v; use k; use v`.
- The same test shows folded `nvws.semaphore.create` carries the authored
  `pending_count` for its `semaphore.id`.
- Loop-carried circular K/V tests cover the four partition-assignment examples
  from `fable/circular-buffer-disign.md` and prove multiple live loop-carried
  permission tokens for one folded circular semaphore are passed unchanged.
- A negative test rejects folding if logical semaphores with the same
  `semaphore.id` disagree on authored `pending_count`.
- A folded circular test proves `verifySingleCarrierPerGroup` is relaxed or
  bypassed for folded circular groups only, while the non-circular verifier
  remains enforced.

### C2 - AssignStagePhase consumes stage operands as offsets

Required work:

1. For circular semaphore groups, read any preexisting semaphore stage operand
   before overwriting it. In pre-AssignStagePhase circular IR, that value is a
   signed offset.
2. Require circular events to author a stage operand, even when the offset is
   zero, unless the implementation deliberately accepts absence as offset `0`.
   The stricter verifier is preferred because it catches malformed circular IR.
3. Visit circular `nvws.semaphore.acquire`, `nvws.semaphore.buffer`, and
   `nvws.semaphore.release` operations in program order for the folded group.
4. Keep the existing fresh-write stage update for circular acquires:

   ```text
   baseStage = state.stage
   if acquire is a fresh write:
     baseStage = (baseStage + 1) % depth
   ```

5. For circular `buffer` and `release`, `baseStage = state.stage`; these ops do
   not advance the shared cursor.
6. Compute each circular op's event stage from that op's authored offset:

   ```text
   offset = signed value from op.stage before assignment
   eventStage = positive_mod(baseStage + offset, depth)
   ```

7. Store `state.stage = baseStage`, not `eventStage`, so the shared circular
   cursor remains the unshifted producer cursor.
8. Overwrite `op.stage = eventStage`.
9. Compute acquire phase from `eventStage`, because LowerAref indexes the
   mbarrier using the final stage operand.
10. Single-phase eligibility for circular groups must account for authored
   offsets when forming the virtual-stage key, or conservatively mark circular
   groups multiphase. The conservative multiphase choice is correct for the
   first implementation.
11. For circular groups, do not propagate stage through the acquire token.
   Circular `buffer` and `release` stages are computed from their own authored
   stage operands and the current shared cursor.
12. Loop/if state threading for circular groups must thread the shared cursor
    independent of any one carrier token.
13. Keep token-based stage propagation for non-circular groups only.
14. Do not maintain separate logical-stage and data-stage operands in this
    design.

Exit gate:

- An AssignStagePhase lit test starts with circular semaphore ops that already
  have offset stage operands, including `-1`.
- The same test proves AssignStagePhase overwrites those operands with final
  non-negative physical stages modulo depth.
- The same test proves acquire, buffer, and release for one event receive the
  same final physical stage.
- The same test proves phase is computed from the final physical stage.
- A multi-token folded circular test proves two live loop-carried tokens for one
  folded circular semaphore are passed unchanged, while buffer/release stages
  are computed from their own authored offsets and the shared cursor, not from
  token lineage.

### C3 - Generic verifier and LowerAref pending-count contract

Required work:

1. No LowerAref semantic change is required for circular local buffer stage
   lowering.
2. By the time LowerAref runs, AssignStagePhase has replaced every circular
   offset stage operand with a final physical stage.
3. `rewriteAcquire` and `rewriteRelease` already use the final stage for
   mbarrier wait/arrive.
4. `rewriteBuffer` already uses the final stage on `nvws.semaphore.buffer` for
   data `ttg.memdesc_index`.
5. LowerAref must not see negative circular offsets. If it does, that is an
   upstream AssignStagePhase verifier failure.
6. Update `third_party/nvidia/lib/Dialect/NVWS/IR/Ops.cpp` so
   `SemaphoreCreateOp::verify()` does not exact-rederive `pending_count` from
   folded physical IR for folded circular semaphores. For this path, authored
   `pending_count` is ground truth because InsertSemas validated it before
   folding by `semaphore.id`.
7. Make the folded-circular predicate concrete in code. The verifier bypass
   must be gated by a reliable marker, preferably by tracing the semaphore
   backing to local alloc metadata with `buffer.circular`, and must not apply
   to ordinary non-circular semaphores.
8. Update LowerAref / lower-semaphore so it also uses the authored
   `pending_count` attribute verbatim. Remove or bypass the current exact
   re-derivation from folded IR in this path.
9. LowerAref must not infer circular behavior from duplicate operands, inspect
   `buffer.start`, or implement per-result circular offsets. If a circular
   group reaches LowerAref as one variadic buffer op that needs per-result
   offsets, that is an upstream InsertSemas or AssignStagePhase verifier
   failure.

Required final-stage shape for one K/V depth-2 iteration:

```text
acq empty[0] ; ld k  ; rel full [0]
acq empty[1] ; ld v  ; rel full [1]
acq full [0] ; use k ; rel empty[0]
acq full [1] ; use v ; rel empty[1]
```

Exit gate:

- Source inspection or an existing lowering lit test confirms LowerAref still
  lowers acquire/release stages to mbarriers and buffer stages to data
  `ttg.memdesc_index`.
- A verifier lit test proves `SemaphoreCreateOp::verify()` accepts folded
  circular IR whose authored `pending_count` was validated before folding and
  would be overcounted by stage-blind post-fold analysis.
- A LowerAref/lower-semaphore lit test proves authored `pending_count` is used
  verbatim for mbarrier initialization and is not rejected by stage-blind
  reanalysis of folded circular IR.
- The circular AssignStagePhase lit test is sufficient to prove LowerAref will
  index K/V into different physical slots without a circular-specific rewrite.

### C4 - End-to-end circular gates

Build first:

```sh
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Then run targeted lit tests from the build directory:

```sh
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  test/NVWS/MetaAutoWS \
  test/NVWS/insert_semas*.mlir \
  test/NVWS/lower_semaphore*.mlir
```

Then rerun the repro with dumps:

```sh
MLIR_ENABLE_DUMP=1 sh run_nvws.sh
```

Required end-to-end result:

- K/V planned circular local reuse carries `buffer.circular` and
  `buffer.start`.
- InsertSemas emits separate logical circular channels.
- InsertSemas folds logical channels by physical `semaphore.id` and authors
  circular offsets in existing stage operands.
- InsertSemas validates and authors one `pending_count` per folded
  `semaphore.id`.
- AssignStagePhase consumes those operands as offsets and overwrites them with
  final physical stages.
- LowerAref remains unchanged and indexes K/V into different physical SMEM
  slots because AssignStagePhase assigned final physical stages.
- `SemaphoreCreateOp::verify()` and LowerAref / lower-semaphore use authored
  `pending_count` without stage-blind reanalysis of the folded circular stream.
- `run_nvws.sh` does not fail shared-memory OOR and does not alias K/V into the
  same circular slot.
