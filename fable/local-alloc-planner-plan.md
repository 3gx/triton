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
