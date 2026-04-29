# Phase 1 Plan: Port Meta DP, PSMeta, and TMEM Memory Planner to NVWS

## Goal

Port the Meta warp-specialization flow needed for Blackwell FA-style kernels into
the NVWS pipeline, initially focused on TMEM correctness and eventually enabling
a safe fallback to the existing generic NVWS partition scheduler.

The first phase-1 objective is narrower than full pipeline integration:

- port the new DP, PSMeta, and TMEM memory-planner passes into the NVWS
  transform directory;
- make each new pass build;
- make each new pass independently runnable through `triton-opt`;
- keep existing NVWS passes untouched except for one narrow
  `NVWSHoistTmemStore` fallback-owner change;
- update `NVWSHoistTmemStore` fallback ownership to
  derive the MMA partition when possible while preserving the old hardcoded
  fallback behavior when derivation is not possible;
- keep existing lit tests passing;
- leave the full `automatic-warp-specialization` integration path out of scope;
  it is understood that manually trying the new path there is not expected to
  work until a later integration phase.

The eventual target pipeline is:

```text
NVWSDataPartition

partition scheduling choice:
  try NVWSPartitionSchedulingMeta
  if PSMeta fails:
    run current generic scheduler: createTritonGPUPartitionScheduling

NVWSHoistTmemStore
NVWSMemoryPlannerTmem
NVWSInsertSemaphore
NVWSInsertTmemSemaphore
NVWSLowerSemaphore
PartitionLoops
NVWSLowerWarpGroup
ScheduleLoops
```

The intended ordering is important:

- Data partitioning runs before partition scheduling.
- The scheduler choice is explicit: try PSMeta first; if it fails without
  mutating its pass-entry IR, run the current generic scheduler
  `createTritonGPUPartitionScheduling`.
- PSMeta or current generic fallback partition scheduling runs before TMEM
  hoisting.
- TMEM hoisting runs before the TMEM memory planner.
- TMEM memory planner runs before semaphore insertion.

## Non-Goals For Phase 1

- Do not port the full SMEM planner unless it is required as scaffolding for
the TMEM planner.
- Do not replace all generic NVWS partition scheduling behavior.
- Do not optimize partitioning quality beyond preserving the PSMeta semantic
FA layout when PSMeta succeeds.
- Do not make `doHoistLoopInvariantTMEMStore` part of the initial port unless
a new test shows that `NVWSHoistTmemStore` plus the TMEM planner requires it.
- Do not modify existing NVWS or TritonGPU pass implementation files or
  implementation headers in phase 1, except for the narrow
  `HoistTmemStore.cpp` fallback-owner change.
- Do not modify existing lit test files in phase 1.
- Do not wire the new passes into the default automatic warp-specialization
  pipeline in phase 1.

## Source Components

All newly ported implementation files for this phase must live under:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/
```

Do not add the new DP, PSMeta, or memory-planner ports under the Hopper
transform directory. Hopper files are sources to port from, not the ownership
location for the NVWS implementation.

Port from Hopper/Meta:

- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSDataPartition.cpp`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/PartitionSchedulingMeta.cpp`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`

Existing NVWS context, not phase-1 edit targets:

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemaphore.cpp`
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp`
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp`
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp`
- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp`

These files explain the downstream contract and current pipeline behavior. Do
not modify the existing NVWS or TritonGPU pass implementation files above in
phase 1. If a future fix needs changes to existing pass implementation logic, it
belongs in a later phase or a separate explicitly approved task.

Narrow phase-1 exception:

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/HoistTmemStore.cpp` may be
  changed only to replace the hardcoded fallback TMEM alloc owner partition
  with a derived single partition from the unique associated MMA, preserving
  the old `{1}` fallback when derivation fails.

Allowed existing-file edits for phase 1:

- pass registration TableGen files such as
  `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td`;
- pass registration headers such as
  `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h`;
- build integration files such as
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt`;
- new NVWS-owned private utility source/header files under
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/` when needed by the new
  passes;
- new lit test files copied/ported for the new passes.

Disallowed existing-file edits for phase 1:

- existing NVWS pass implementation `.cpp` files, except the narrow
  `HoistTmemStore.cpp` fallback-owner change described above;
- existing TritonGPU warp-specialization pass implementation `.cpp` files;
- existing pass implementation/helper headers whose behavior is consumed by
  current passes;
- existing lit test files.

## Current Baseline

The current automatic NVWS pipeline is:

```text
createTritonGPUPartitionScheduling
createNVWSHoistTmemStore
createNVWSInsertSemaphore
createNVWSInsertTmemSemaphore
createNVWSLowerSemaphore
createTritonGPUPartitionLoops
createNVWSLowerWarpGroup
createTritonGPUScheduleLoops
```

`createTritonGPUPartitionScheduling` has the robust generic serialization
contract:

- every op under a `tt.warp_specialize` loop gets `ttg.partition`;
- `scf.yield` terminators are annotated;
- `ttg.partition.outputs` is produced for `scf.for`, `scf.if`, and
`tt.reduce`;
- region ops carry the union of their child/result partitions;
- `ttg.warp_specialize.tag` is emitted on WS loops.

PSMeta has the semantic FA partitioning we want, but it currently lacks the full
generic serialization contract and is not atomic.

## Required IR Contract After Partition Scheduling

After `NVWSPartitionSchedulingMeta`, every scheduled WS loop must
satisfy the same downstream contract that generic NVWSPS satisfies.

For each `scf.for` with `tt.warp_specialize`:

- the loop has `ttg.warp_specialize.tag`;
- the loop has `ttg.partition.stages`;
- the loop has `ttg.partition.types` when PSMeta is used;
- the loop has `ttg.partition`;
- every child op except allowed poison/dead ops has `ttg.partition`;
- `scf.yield` terminators have `ttg.partition`;
- every result-bearing `scf.for`, `scf.if`, and `tt.reduce` has
`ttg.partition.outputs`;
- every `ttg.partition.outputs` entry is a subset of the parent op's
`ttg.partition`;
- if an op outside the lexical WS loop is partitioned as part of that loop's
schedule, it also has enough WS identity to disambiguate its partition.

The last rule matters because `ttg.partition = 0` is ambiguous outside the loop:
partition IDs are local to a WS loop. Outside a loop region, a partitioned op
must be associated with a specific `ttg.warp_specialize.tag`.

Example:

```mlir
%r = scf.for ... -> (...) {
  ...
} {tt.warp_specialize,
   ttg.warp_specialize.tag = 0 : i32,
   ttg.partition = array<i32: 0, 1>}

%v, %tok = ttng.tmem_load %acc[%r#3]
  {ttg.partition = array<i32: 0>,
   ttg.warp_specialize.tag = 0 : i32}
```

Without the tag on the post-loop `ttng.tmem_load`, NVWS TMEM semaphore insertion
cannot know whether this is partition 0 of WS loop 0, partition 0 of another WS
loop, or unrelated partition metadata.

## Phase 1 Work Items

### 1. Phase-1 Standalone Scope

This is the first concrete objective. Add the new pass declarations,
implementations, and build wiring described in the following work items so each
new pass can be invoked directly with `triton-opt`.

Required properties:

- each pass is registered and visible in `triton-opt --help`;
- each pass can run alone on a focused MLIR input;
- pass failures are diagnostics, not assertions;
- the new pass source files live in
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/`;
- existing NVWS and TritonGPU pass implementation source/header files are not
  modified, except for the narrow `HoistTmemStore.cpp` fallback-owner change;
- existing lit test files are not modified;
- existing lit tests continue to pass.

Acceptance:

- `ninja triton triton-opt` succeeds.
- `triton-opt` can run each new pass by name.
- Existing lit tests pass after the new pass files and narrow HoistTmemStore
  change are added.
- New or ported lit tests validate only the new standalone passes, the narrow
  HoistTmemStore fallback-owner change, and explicit `triton-opt` pass
  pipelines.
- No existing lit test is edited, skipped, or marked expected failure.

### 2. Add NVWS Pass Declarations And Build Wiring

Add NVWS pass entries for:

- `nvws-ws-data-partition`
- `nvws-partition-scheduling-meta`
- `nvws-memory-planner`

The pass names can differ if local naming conventions require it, but the
pipeline should make the NVWS ownership explicit rather than reusing Hopper pass
names.

Expected files:

- `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td`
- `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.h`
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt`
- new implementation files under
`third_party/nvidia/lib/Dialect/NVWS/Transforms/`
- optional new private utility files under
`third_party/nvidia/lib/Dialect/NVWS/Transforms/`

The `.td`, `Passes.h`, and `CMakeLists.txt` edits are expected registration and
build wiring changes. They do not count as touching existing pass
implementation logic.

Required implementation location:

- `NVWSDataPartition` implementation:
`third_party/nvidia/lib/Dialect/NVWS/Transforms/`
- `NVWSPartitionSchedulingMeta` implementation:
`third_party/nvidia/lib/Dialect/NVWS/Transforms/`
- `NVWSMemoryPlanner` implementation:
`third_party/nvidia/lib/Dialect/NVWS/Transforms/`

Acceptance:

- `triton-opt --help` lists the new NVWS passes.
- The new passes can be run as no-op stubs before porting internals.

### 3. Port WSDataPartition To NVWS

Create an NVWS-owned data partition pass based on Hopper
`WSDataPartition.cpp`.

The new implementation must be added under:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/
```

Initial behavior:

- operate on loops with `tt.warp_specialize`;
- respect existing `tt.data_partition_factor`;
- preserve the physical slicing behavior needed by FA;
- preserve or adapt `async_task_id` propagation only where needed for the port;
- avoid depending on Hopper-only task partitioning as a prerequisite.

Important constraints:

- DP must run before PSMeta/fallback partition scheduling.
- DP should not create `ttg.partition` attrs. Partitioning metadata belongs to
the scheduler.
- DP should not require memory planner state.

Initial tests:

- Existing Hopper DP tests should be copied or mirrored into an NVWS test area.
- Add a focused test for FA-like softmax split showing that a
`tt.data_partition_factor = 2` WS loop is physically split into the expected
two sub-flows before scheduling.
- Add a nested `scf.if` or epilogue-subtile case because PSMeta previously
missed scalar ops after this shape.

Acceptance:

- `NVWSDataPartition` transforms the target FA IR without introducing
partition attrs.
- Running generic NVWSPS after `NVWSDataPartition` still verifies, even if the
resulting partitioning is not the desired semantic PSMeta layout.
- The pass is runnable directly through `triton-opt --nvws-ws-data-partition`.

### 4. Port PSMeta To NVWS As A No-Fallback Scheduler

Create an NVWS PSMeta scheduler based on Hopper
`PartitionSchedulingMeta.cpp`.

The new implementation must be added under:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/
```

This pass should not contain fallback logic. Its contract is:

- on success, it commits the PSMeta schedule and emits verifier-clean partition
metadata;
- on failure, it returns pass failure without modifying the input IR;
- fallback to generic `createTritonGPUPartitionScheduling` is future
  orchestration owned by `AutomaticWarpSpecialization.cpp`, not by this pass.

In the eventual automatic pipeline, the PSMeta pass input may already include
the effects of `NVWSDataPartition`. The no-mutation-on-failure contract is
relative to PSMeta pass entry: failure must preserve the post-DP IR exactly so
generic partition scheduling can retry on that same DP-transformed IR.

This no-mutation-on-failure contract is not true for current Hopper PSMeta.
Current PSMeta mutates IR before it knows the full schedule is valid:

- it calls `setPartition`;
- it can clone memdesc/view ops;
- `optimizeSchedule` clones and rewrites uses;
- `splitDataPartitionedIfOps` creates new `scf.if`, replaces uses, and erases
old `scf.if` ops.

Therefore the NVWS PSMeta pass must be made transactional internally, but the
transaction only protects its own success/failure behavior. It must not run the
generic fallback itself.

Required phase-1 transaction design inside the PSMeta pass:

```text
tryMetaTransaction(funcOp)
  clone exactly one triton::FuncOp into scratch IR
  run PSMeta analysis/rewrites/finalizer on the cloned function
  run strict PSMeta checks on the cloned function
  if success:
    replace the original function with the cloned function
  if failure:
    discard clone
    return failure with pass-entry IR unchanged
```

This pass must run at `triton::FuncOp` granularity in phase 1. Do not use
module-level clone/commit or region-level partial commit for PSMeta. Each
function is an independent transaction: a failure in one function leaves that
function's pass-entry IR unchanged and reports pass failure for that pass
invocation.

Implementation note:

- The clone should be built in scratch IR or a temporary container so the
  original module never contains two live functions with the same symbol name
  during analysis.
- On success, preserve the original function symbol identity and replace the
  original `triton::FuncOp` with the verified clone in one commit step.
- On failure, erase/discard the clone and leave the original `triton::FuncOp`
  unchanged.
- If the pass is orchestrated from a module pipeline later, use a nested
  `triton::FuncOp` pass manager or equivalent per-function invocation. Do not
  make the PSMeta pass itself a module transaction.

Longer-term alternative:

```text
pure planning refactor
  compute a side-table schedule without mutating IR
  only commit attrs/rewrites after all checks pass
```

Phase 1 uses the function-level clone-and-commit transaction above. Do not add a
module-level fallback transaction in phase 1; if function replacement proves
awkward, fix the function transaction helper rather than weakening the
no-mutation-on-failure contract.

PSMeta pass failure criteria:

- no matching MMAs/loads for PSMeta;
- ambiguous multi-WS ownership for post-loop partitioned ops;
- missing partition coverage after finalization;
- invalid `partition.outputs`;
- verifier failure;
- unsupported op category or shape;
- any PSMeta internal failure.

Acceptance:

- If PSMeta succeeds, it emits a complete schedule and returns success.
- If PSMeta fails, the PSMeta pass-entry IR is unchanged.
- The pass does not invoke generic NVWSPS.
- The failure mode is deterministic and diagnostic, not an assertion.
- The pass is runnable directly through
  `triton-opt --nvws-partition-scheduling-meta`.

### 4a. PSMeta Commit Requirements

PSMeta may commit its cloned IR back to the original function only after all of
the following checks succeed on the clone:

- PSMeta semantic scheduling succeeds for every targeted WS loop in that
  function;
- the finalizer has completed the generic NVWS partition metadata contract;
- the PSMeta-owned strict verifier passes;
- partitioned pre/post-loop ops satisfy the post-loop WS tag rule;
- partitioned pre/post-loop ops with SSA results satisfy the use-closure or
  existing loop-result plumbing rule;
- failures are reported as diagnostics, not assertions.

If any item fails, discard the clone and return pass failure with the pass-entry
function IR unchanged. Sections 5, 5b, and 6 describe the finalizer, strict
verifier, and post-loop checks that make up this commit contract.

### 5. Add PSMeta Finalization For The Generic NVWS Contract

Add a finalizer that runs after PSMeta has chosen semantic partitions and before
the PSMeta attempt is committed. This finalizer is one of the mandatory PSMeta
commit gates from Section 4a.

The finalizer should mirror the useful serialization behavior from generic
`PartitionScheduling.cpp`, without changing the semantic PSMeta partition
layout.

Required finalizer behavior:

- assign missing `ttg.partition` to all ops under each WS loop;
- annotate `scf.yield` terminators;
- compute `ttg.partition.outputs` for:
  - `scf.for` results from region iter args/yield values;
  - `scf.if` results from then/else yielded values and result users;
  - `tt.reduce` results from `tt.reduce.return` and result users;
- set region op `ttg.partition` to the union of child partitions and output
partitions;
- write `ttg.partition.stages`;
- preserve/write `ttg.partition.types`;
- set `ttg.warp_specialize.tag`;
- attach or derive WS tag information for partitioned pre/post-loop ops owned by
the schedule.

HoistTmemStore compatibility rule:

- Do not require PSMeta to preserve any numeric convention such as "partition
  1 is gemm/MMA." That would make the PSMeta layout depend on an implementation
  detail of an existing follow-on pass.
- Instead, narrowly extend `NVWSHoistTmemStore` so its non-hoist fallback derives
  the fallback owner from the unique associated MMA partition when possible.
- The extension must be backward compatible: if the unique MMA or a single MMA
  partition cannot be derived, keep the old hardcoded `{1}` assignment.
- The PSMeta finalizer only needs to produce correct partition metadata on the
  MMA and TMEM ops. It must not fail merely because gemm/MMA is not partition
  index `1`.

Partition completion rules:

- For pure scalar/index glue ops, assign the union of all consumer partitions.
  This is the primary rule because scalar SSA values must be available in every
  consumer partition that uses them.
- A scalar/index op with consumers in multiple partitions should normally have a
  multi-element `ttg.partition` set. That is valid metadata; it is not a reason
  to introduce synchronization.
- Do not plan to insert semaphores for scalar SSA dependencies. Semaphores are
  for memory/TMEM/async ordering, not for making scalar/index glue visible to
  multiple consumer partitions.
- If no consumer partition exists, use operand/def partitions. If that is still
  insufficient, use the parent region union or fail PSMeta rather than assigning
  an arbitrary single partition.
- Constants and simple scalar index arithmetic may be placed in the union of
  consumers, or in the parent region union if there is no better owner.

Special post-loop rule:

- A partitioned op outside the WS loop must have a single associated WS tag.
- If that ownership can be inferred from an operand defined by a loop result,
attach the tag.
- If ownership is ambiguous across multiple WS tags, fail PSMeta without
  modifying IR so a later automatic orchestration phase can choose generic
  scheduling fallback.

### 5a. Reuse Pre-14031 Partition Propagation Ideas

The parent of commit `14031d221ba1401dc49a73565d3ff933c70a2d55` is:

```text
952acbd509abaa4932495c00ad1ee3e1da687058
```

That version of:

```text
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionScheduling.cpp
```

contains propagation/finalization logic that is directly applicable to the
NVWS PSMeta finalizer. Do not port the old initial scheduler wholesale; PSMeta
still owns the semantic FA partition choices. Instead, reuse the completion
ideas that make partition metadata available on every required op.

Applicable old logic:

- `getUseOps` follows value uses through `scf.for` region iter args,
  `scf.yield`, and `scf.if` results to find the effective consumer ops.
- `assignMissingPartitions` seeds a map from existing op partitions and then
  repeatedly assigns each unpartitioned leaf op the union of the partitions of
  its effective consumers until convergence.
- That consumer-driven fixed point is the right model for scalar/index glue:
  if a scalar feeds partitions `0` and `2`, the scalar gets
  `ttg.partition = array<i32: 0, 2>`.
- The old second pass uses operand/def partitions only for remaining
  scalar/zero-result ops with no useful consumer information. This is a
  fallback, not the primary rule.
- The old region-output logic computes partitions for yielded values and then
  derives `ttg.partition.outputs` plus the parent region op's partition as the
  union of body/output partitions.
- The old parent-inheritance workaround for zero-result side-effect-like ops
  inside region bodies is useful for cases such as scalar assumptions or other
  glue ops that do not produce a value but must still be assigned to the
  enclosing partition set.

Required NVWS adaptation:

- Implement this logic inside the new NVWS PSMeta finalizer, not by modifying
  existing `PartitionScheduling.cpp` or existing NVWS pass implementations.
- Preserve multi-element `ttg.partition` sets for scalar/index ops. Do not
  clone scalar glue merely because it has consumers in multiple partitions.
- Do not introduce scalar semaphores. Scalar SSA availability is represented by
  assigning the scalar op to the union of consumer partitions.
- Replace old assertions with diagnostics and PSMeta failure. Because the pass
  is transactional, a finalizer failure must discard the clone and leave the
  PSMeta pass-entry IR unchanged.
- Extend the old control-flow handling where needed. The old code had a TODO
  for fully mutually-recursive nested control flow; the NVWS finalizer must
  cover nested `scf.if`, `scf.for`, and `tt.reduce` cases needed by FA and DP
  epilogue-subtile shapes.
- The old scheduler skipped direct `scf.yield` annotation in verification; the
  NVWS finalizer must still annotate `scf.yield` terminators because the
  current generic NVWS serialization contract requires it.
- Keep TMEM-specific repair local and conservative. Old TMEM alloc/load/store
  completion can inform the finalizer, but it must not override PSMeta's chosen
  FA partition layout.

Additional focused tests:

- scalar/index op consumed by two different partitions gets a multi-element
  `ttg.partition`;
- scalar/index value forwarded through `scf.if` yield or `scf.for` iter arg
  inherits the union of effective consumer partitions;
- scalar/zero-result glue with no useful consumers falls back to operand/def or
  parent-region partitioning;
- nested `scf.if` plus DP epilogue-subtile shape has no unpartitioned scalar
  ops after finalization;
- finalizer failure on unsupported nested control flow leaves the PSMeta
  pass-entry IR unchanged.

### 5b. Add A Strict PSMeta Contract Verifier

Do not rely solely on the existing `verifyPartitionedLoop` helper as the
soundness gate for PSMeta output. The existing verifier validates malformed
`ttg.partition.outputs` when the attribute exists, but it does not require every
result-bearing `scf.for`, `scf.if`, or `tt.reduce` to have that attribute. It
also does not check the post-loop WS tag contract.

Add a PSMeta-owned strict verifier helper in the new NVWS PSMeta implementation
or private utility code. This must not modify existing `Partition.cpp`.

Mandatory strict checks:

- every scheduled WS loop has `ttg.warp_specialize.tag`,
  `ttg.partition.stages`, `ttg.partition`, and, for PSMeta, valid
  `ttg.partition.types`;
- partition IDs on every op are sorted, non-empty, unique, and within the
  `ttg.partition.stages` range;
- every non-poison op under a scheduled WS loop has `ttg.partition`;
- every `scf.yield` terminator under a scheduled WS loop has `ttg.partition`;
- every result-bearing `scf.for`, `scf.if`, and `tt.reduce` under a scheduled
  WS loop has `ttg.partition.outputs` with one entry per result;
- every `ttg.partition.outputs` entry is non-empty and is a subset of the
  parent op's `ttg.partition`;
- every region op's `ttg.partition` contains the union of child partitions and
  output partitions;
- every partitioned pre/post-loop op owned by a WS schedule has exactly one
  unambiguous `ttg.warp_specialize.tag`;
- no strict verifier check may require a particular numeric partition ID for
  gemm/MMA; ownership-sensitive follow-on passes must derive ownership from IR
  metadata instead.

Failure of any strict check is a PSMeta failure. Because PSMeta is
transactional, strict-verifier failure must leave PSMeta pass-entry IR
unchanged.

The strict verifier is a PSMeta-output pre-commit invariant. Later passes such
as `NVWSHoistTmemStore` may mutate partition metadata, and the post-hoist IR is
not required to satisfy this verifier unchanged unless a later phase explicitly
adds that requirement.

Acceptance:

- The PSMeta output passes both the new strict verifier and
  `verifyPartitionedLoop`.
- `PartitionLoops` does not crash due to missing `partition.outputs`.
- `NVWSHoistTmemStore`, `NVWSInsertSemaphore`, and
  `NVWSInsertTmemSemaphore` do not need to guess missing partition metadata.

### 6. Handle Post-Loop WS Tags Without Modifying Semaphore Passes

Current `NVWSInsertTmemSemaphore` starts WS-tag lookup at the op itself and then
climbs lexical parent `scf.for` ops until it finds `ttg.warp_specialize.tag`.
That means a partitioned post-loop op is acceptable if the PSMeta finalizer
attaches `ttg.warp_specialize.tag` directly to that op.

Phase-1 rule:

- do not modify `InsertTmemSemaphore.cpp`;
- do not modify existing NVWS semaphore pass implementation files;
- instead, require PSMeta finalization to attach an unambiguous
  `ttg.warp_specialize.tag` to partitioned pre/post-loop ops;
- if the finalizer cannot determine a single WS tag, PSMeta must fail without
  modifying IR. A later automatic integration phase can decide how to fall back.

PartitionLoops use-closure rule:

- A direct post-loop WS tag is sufficient for current
  `NVWSInsertTmemSemaphore` lookup, but it is not by itself sufficient for
  `PartitionLoops` soundness.
- `PartitionLoops` clones and erases tagged partitioned ops in the WS loop's
  block and only has special replacement/plumbing for loop results. Therefore,
  a partitioned pre/post-loop op with SSA results is phase-1 safe only if its
  result uses are closed within the same tagged partitioned op set and no value
  needs to remain live after those ops are erased, or if existing loop-result
  plumbing already carries the value.
- If a partitioned pre/post-loop op has result users outside that closed set and
  the value is not already represented by loop-result plumbing, PSMeta must fail
  for phase 1 rather than producing IR that would later break `PartitionLoops`.
- Adding new post-loop result plumbing is a later integration task unless a
  focused phase-1 test proves the needed case is already handled by existing
  loop-result replacement logic.
- This use-closure/plumbing check is a PSMeta commit gate in addition to the
  strict verifier from Section 5b.

Future robustness, outside phase 1:

- convert the existing assertion in `InsertTmemSemaphore.cpp` into a diagnostic;
- derive missing tags in the semaphore pass if needed.

Acceptance:

- A post-loop `ttng.tmem_load` with both `ttg.partition` and
  `ttg.warp_specialize.tag` is accepted only when it also satisfies the
  use-closure/plumbing rule.
- PSMeta finalization rejects a post-loop partitioned TMEM op if it cannot attach
  a single tag.
- Existing NVWS semaphore pass files are unchanged.

### 6a. Extend HoistTmemStore Fallback Ownership

`NVWSHoistTmemStore` currently has a non-hoist fallback that assigns some
remaining MMA-owned TMEM allocs to `ttg.partition = array<i32: 1>`. That is safe
for the current generic scheduling layout but is the wrong contract for PSMeta:
PSMeta should be free to choose semantic partition numbering.

Phase-1 change:

- narrowly update `HoistTmemStore.cpp` in the fallback path only;
- use the existing unique alloc-to-MMA path to find the associated MMA;
- if the associated MMA has exactly one `ttg.partition`, assign the TMEM alloc
  to that same partition;
- if the associated MMA cannot be found or does not have exactly one partition,
  keep the old hardcoded `{1}` assignment;
- do not make HoistTmemStore depend on `ttg.partition.types` or PSMeta-specific
  partition names;
- do not change the successful hoisting path, token threading, loop-result
  plumbing, or semaphore insertion passes.

This preserves existing behavior for current layouts where the MMA is already
partition `1`, while allowing PSMeta layouts where the MMA partition has a
different numeric ID.

Acceptance:

- existing generic NVWS tests remain unchanged and pass;
- a focused test with an MMA in a non-`1` partition shows the non-hoist fallback
  assigns the alloc to the MMA partition;
- a focused test where the MMA partition cannot be derived still gets the old
  `{1}` assignment.

### 7. Port TMEM-Only Memory Planner To NVWS

Create an NVWS memory planner pass from the TMEM-relevant parts of Hopper
`WSMemoryPlanner.cpp`.

The new implementation must be added under:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/
```

Phase-1 scope:

- TMEM allocations and TMEM channels only;
- enough channel collection to support `ttng.tmem_alloc`,
`ttng.tmem_store`, `ttng.tmem_load`, and `ttng.tc_gen5_mma`;
- no full SMEM buffer planning unless required by shared utilities;
- no broad performance tuning.

Required NVWS-owned utility scope:

- Hopper `WSMemoryPlanner.cpp` depends on Hopper `CodePartitionUtility.*` for
  `Channel`, `DataChannelKind`, `TmemDataChannelPost`, channel collection, and
  TMEM channel query helpers. The NVWS port must not include or depend on the
  Hopper private utility header directly.
- Add a minimal NVWS-owned private utility implementation under
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/` for the TMEM-only planner.
  This can copy/adapt the relevant Hopper utility code, but ownership and build
  wiring must be NVWS-local.
- The minimal utility should include the TMEM-post channel representation,
  `collectPostChannels` or a TMEM-only equivalent, operand-D channel handling,
  channel attribute helpers, and any `getSrcOp`/`getDstOp`/`getDstOps` helpers
  the planner needs.
- SMEM channel types/helpers should be omitted or stubbed only when necessary
  for shared interfaces. They must not expand phase-1 scope into full SMEM
  memory planning.

Pipeline position:

```text
NVWSHoistTmemStore
NVWSMemoryPlannerTmem
NVWSInsertSemaphore
NVWSInsertTmemSemaphore
```

Important semantic requirement:

- Treat `ttng.tmem_alloc %zero` used as operand D initialization as an
initialized allocation, not as a separate producer channel.

Example:

```mlir
%acc = ttng.tmem_alloc %zero : (tensor<...xf32>) -> !ttg.memdesc<..., #tmem>
%tok = ttng.tc_gen5_mma %a, %b, %acc[%dep], ...
```

This must not create an extra producer channel from the alloc source `%zero` to
the MMA. It should model the allocation as already initialized. This preserves
the Hopper behavior in this path and avoids over-synchronizing or inventing a
spurious producer.

TMEM channel expectations:

- MMA producing into TMEM and later `tmem_load` should form a producer-consumer
channel.
- `tmem_store` feeding MMA through the same allocation should form the needed
producer-consumer relation.
- If a `tmem_load` and a later `tmem_store` on the same allocation require
ordering, preserve the existing sibling/back-edge logic from Hopper if it is
needed by the ported IR.
- Do not create a producer channel for the initialization source of
`tmem_alloc %zero`.

Acceptance:

- The planner can run after `NVWSHoistTmemStore` on the FA fixture.
- It emits or updates the TMEM metadata/channel attributes expected by the
  existing NVWS semaphore passes.
- Phase-1 required acceptance for planner/semaphore interaction is structural:
  the planner must produce the expected metadata without requiring semaphore pass
  edits. Explicit `triton-opt` pipelines may be extended through the existing
  semaphore passes only as opportunistic smoke tests, not as required phase-1
  acceptance, unless they pass unchanged.
- The pass is runnable directly through `triton-opt --nvws-memory-planner`.

### 8. Keep Automatic Pipeline Integration Out Of Scope

Do not modify
`lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
in the first phase-1 objective. The new passes should be validated through
direct `triton-opt` pipelines first.

Do not add, edit, skip, or mark any automatic-warp-specialization lit test for
the new DP/PSMeta/MP path in phase 1. It is understood that the new automatic
path is not wired and does not work yet.

The eventual automatic orchestration should be:

```text
runPipeline(createNVWSDataPartition());

for each triton::FuncOp:
  if (succeeded(runPipeline(createNVWSPartitionSchedulingMeta()))) {
    runPipeline(createVerifyWarpSpecializationPartitionsPass());
  } else {
    // PSMeta must have left that function's post-DP IR unchanged on failure.
    runPipeline(createTritonGPUPartitionScheduling());
    runPipeline(createVerifyWarpSpecializationPartitionsPass());
  }

runPipeline(createNVWSHoistTmemStore());
runPipeline(createVerifyWarpSpecializationPartitionsPass());
runPipeline(createNVWSMemoryPlanner({tmemOnly = true}));
runPipeline(createVerifyWarpSpecializationPartitionsPass());
runPipeline(createNVWSInsertSemaphore());
runPipeline(createVerifyWarpSpecializationPartitionsPass());
runPipeline(createNVWSInsertTmemSemaphore());
runPipeline(createVerifyWarpSpecializationPartitionsPass());
runPipeline(createNVWSLowerSemaphore({numStages}));
runPipeline(createVerifyWarpSpecializationPartitionsPass());
runPipeline(createTritonGPUPartitionLoops());
runPipeline(createNVWSLowerWarpGroup());
runPipeline(createTritonGPUScheduleLoops());
```

The exact future code can still group adjacent passes into `OpPassManager`s, but
the PSMeta attempt must be observed at `triton::FuncOp` granularity. A failed
PSMeta run for one function should not force a module-level retry over functions
where PSMeta already committed; it should trigger the current generic scheduler
retry for that same function whose post-DP IR PSMeta left unchanged.

Phase-1 test policy:

- existing automatic warp-specialization lit tests must continue to pass;
- do not touch existing automatic warp-specialization lit tests;
- do not add new automatic-warp-specialization lit coverage for the new
  DP/PSMeta/MP path in phase 1;
- validate new behavior only through new or ported standalone `triton-opt` lit
  tests covering the new passes and the narrow HoistTmemStore fallback-owner
  change.

In a later integration phase, if the new path needs to be staged behind an
option, add a conservative option such as:

```text
use-nvws-meta-scheduler
```

or:

```text
enable-nvws-dp-mp
```

Default policy can be decided separately during that later integration phase.

Acceptance:

- Existing generic NVWS tests continue to use the old scheduler path.
- The new DP/PSMeta/MP passes are exercised through explicit `triton-opt`
  pipelines, not through the default automatic pipeline.
- No existing lit test is modified.
- No automatic-warp-specialization test is added for the new path in phase 1.

## Verification Plan

Follow repository instructions for full validation after implementation:

```text
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test
```

Do not run pytest unless explicitly requested.

Phase-1 focused lit tests:

Only add new lit test files or copied/ported lit test files for the new passes
and the narrow HoistTmemStore fallback-owner change. Do not edit any existing
lit test file.

1. `NVWSDataPartition` transforms FA-like DP IR and leaves no partition attrs;
   an explicit `NVWSDataPartition -> generic NVWS partition scheduling` pipeline
   verifies on the focused DP fixture.
2. `NVWSPartitionSchedulingMeta` succeeds on FA forward and emits:
  - `ttg.partition.types`;
  - `ttg.partition.stages`;
  - complete `ttg.partition`;
  - complete `ttg.partition.outputs`;
  - post-loop WS tags where needed.
3. Strict PSMeta verifier negative tests:
  - missing `ttg.partition.outputs` on a result-bearing region op fails;
  - missing/ambiguous post-loop WS tag fails;
  - gemm/MMA in a non-`1` partition is allowed when the IR metadata is
    otherwise valid.
4. PSMeta failure/no-mutation test:
  - feed an unsupported WS loop;
  - verify `NVWSPartitionSchedulingMeta` returns failure without modifying its
    pass-entry IR;
  - verify no partial PSMeta attrs/clones remain.
  - add this test after the transaction shell is in place, since no-mutation is
    a property of the transaction wrapper.
5. Post-loop TMEM tag and use-closure test:
  - partitioned post-loop `ttng.tmem_load` with tag is accepted;
  - missing/ambiguous tag makes PSMeta fail before existing NVWS semaphore
    passes run;
  - partitioned post-loop op with live-out users not covered by existing
    loop-result plumbing makes PSMeta fail in phase 1.
6. TMEM alloc initialization test:
  - `ttng.tmem_alloc %zero` does not create an extra producer channel.
7. TMEM planner utility/channel-collection test:
  - the NVWS-owned TMEM utility collects the same relevant TMEM-post channels
    as the Hopper source fixture for the phase-1 TMEM cases.
8. HoistTmemStore fallback-owner test:
  - non-hoist fallback derives TMEM alloc ownership from the unique MMA
    partition when available;
  - fallback keeps the old `{1}` assignment when the unique MMA partition cannot
    be derived.
9. Standalone composition smoke:
  - `DP -> PSMeta -> HoistTmemStore -> MemoryPlannerTmem`;
  - this smoke depends on the narrow HoistTmemStore fallback-owner change when
    PSMeta places the associated MMA in a partition other than `1`;
  - extending the explicit `triton-opt` pipeline to existing semaphore passes is
    optional phase-1 smoke coverage only, and must not require semaphore pass
    implementation changes.

Known useful fixtures:

- `test/Hopper/WarpSpecialization/blackwell_fa_code_partition.mlir`
- `test/Hopper/WarpSpecialization/partition-scheduling-meta-fa-forward.mlir`
- `test/Hopper/WarpSpecialization/ws_data_partition_epilogue_subtile.mlir`

## Risks And Mitigations

### Risk: PSMeta mutates IR before returning failure

Mitigation:

- make the PSMeta pass transactional internally;
- use the `triton::FuncOp`-level clone-and-commit transaction inside PSMeta for
  phase 1;
- document future fallback orchestration in `AutomaticWarpSpecialization.cpp`
  without implementing it in the first objective;
- add a regression test that proves PSMeta failure leaves no residue.

### Risk: Post-loop partition attrs are ambiguous

Mitigation:

- require `ttg.warp_specialize.tag` on partitioned post-loop ops;
- use the existing direct-op tag lookup behavior in NVWS TMEM semaphore code;
- fail on multi-tag ambiguity.

### Risk: Finalizer invents incorrect partitions for scalar glue

Mitigation:

- assign scalar/index glue to the union of consumer partitions so each consumer
  partition can use the value directly;
- allow multi-element `ttg.partition` on scalar/index ops;
- never treat scalar SSA dependencies as a reason to insert semaphores;
- only use def-driven fallback when there is no consumer information;
- verify with nested `scf.if`, DP epilogue-subtile, and scalar-used-by-multiple
  partitions cases.

### Risk: TMEM planner over-synchronizes initialized allocs

Mitigation:

- explicitly model `ttng.tmem_alloc %zero` as initialized storage;
- add a lit test that checks no extra producer channel is emitted.

### Risk: TMEM planner is not standalone without Hopper utilities

Mitigation:

- add NVWS-owned private TMEM utility code for the minimal channel and
  `TmemDataChannelPost` functionality needed by the phase-1 planner;
- do not include Hopper `CodePartitionUtility.h` from the NVWS pass;
- keep SMEM utility code out of scope unless a shared interface requires a
  narrow stub.

### Risk: Existing partition verifier is weaker than the PSMeta contract

Mitigation:

- add a PSMeta-owned strict verifier that requires every contract item needed
  by downstream NVWS passes;
- keep the existing `verifyPartitionedLoop` call as an additional sanity check,
  not as the sole soundness gate;
- add negative lit tests for missing `ttg.partition.outputs`, missing post-loop
  WS tags, and malformed partition metadata.

### Risk: Post-loop partitioned ops break PartitionLoops live-out handling

Mitigation:

- require a use-closure/plumbing check for every partitioned pre/post-loop op
  with SSA results;
- fail PSMeta in phase 1 when a live-out value is not already carried by
  existing loop-result plumbing;
- defer new post-loop result plumbing to a later integration task.

### Risk: HoistTmemStore hardcodes partition 1 for some TMEM allocs

Mitigation:

- narrowly extend the fallback path to derive the owner from the unique
  associated MMA partition when possible;
- preserve the old `{1}` assignment when derivation fails, so existing behavior
  remains unchanged;
- do not impose a numeric partition convention on PSMeta;
- add tests for both derived-owner and old-fallback behavior.

### Risk: Automatic integration is not ready in phase 1

Mitigation:

- do not modify `AutomaticWarpSpecialization.cpp` in the first objective;
- validate new passes through explicit `triton-opt` pipelines;
- do not add or modify automatic-warp-specialization lit coverage for the new
  path in phase 1;
- keep all pre-existing lit tests passing.

## Suggested Implementation Order

1. Add NVWS pass declarations and no-op stubs.
2. Build `triton` and `triton-opt`; confirm the new pass names appear in
   `triton-opt --help`.
3. Port `WSDataPartition.cpp` to `NVWSDataPartition`.
4. Add focused lit tests for standalone `triton-opt --nvws-ws-data-partition`.
5. Add an empty transactional `NVWSPartitionSchedulingMeta` shell, with no
   internal fallback and `triton::FuncOp`-level clone/discard/commit behavior.
6. Port PSMeta internals inside the transaction shell.
7. Add the PSMeta finalizer, strict verifier checks, scalar union propagation,
   and post-loop use-closure checks.
8. Add focused lit tests for standalone
   `triton-opt --nvws-partition-scheduling-meta`.
9. Ensure post-loop WS tags are emitted by PSMeta finalization; do not modify
   existing NVWS semaphore passes.
10. Narrowly update `NVWSHoistTmemStore` fallback ownership to derive from the
    unique associated MMA partition when possible and preserve old `{1}`
    fallback behavior otherwise.
11. Add the minimal NVWS-owned TMEM utility needed by the memory planner.
12. Port TMEM-only memory planner.
13. Add focused lit tests for standalone `triton-opt --nvws-memory-planner`.
14. Add an explicit `triton-opt` smoke pipeline for
   `DP -> PSMeta -> HoistTmemStore -> MemoryPlannerTmem`.
15. Leave automatic-warp-specialization integration out of scope; do not add or
   edit lit tests for that path.
16. Build, then run lit tests per repository instructions.

All implementation steps above that create ported pass source files must place
those files in `third_party/nvidia/lib/Dialect/NVWS/Transforms/`.
Do not modify existing NVWS or TritonGPU pass implementation files or
implementation/helper headers in this phase, except for the narrow
`HoistTmemStore.cpp` fallback-owner change. Registration `.td`/`Passes.h` and
`CMakeLists.txt` changes are allowed and expected.

## Phase 1 Exit Criteria

Phase 1 is complete when:

- `triton-opt --help` lists the new NVWS DP, PSMeta, and memory-planner passes;
- each new pass can be run directly through `triton-opt`;
- the standalone NVWS pass pipeline can run DP before PSMeta scheduling;
- PSMeta succeeds on the FA target shape and emits the full generic partition
contract;
- unsupported shapes make PSMeta return failure without residual IR mutation;
- PSMeta output passes the new strict contract verifier before it is committed;
- `AutomaticWarpSpecialization.cpp` is not modified for the new path;
- `NVWSHoistTmemStore` runs after scheduling and before memory planning;
- `NVWSHoistTmemStore` fallback owner derivation supports MMAv5/TMEM layouts
  where the associated MMA is not partition `1`, while preserving old `{1}`
  fallback behavior when derivation fails;
- the TMEM-only memory planner runs before semaphore insertion;
- `ttng.tmem_alloc %zero` is handled as initialized storage;
- partitioned post-loop ops carry or derive an unambiguous WS tag;
- partitioned post-loop ops with SSA results satisfy the use-closure/plumbing
  rule or make PSMeta fail;
- TMEM planner dependencies on Hopper channel utilities are replaced by
  NVWS-owned private utility code;
- focused lit tests cover success, PSMeta no-mutation failure, post-loop
tagging, HoistTmemStore fallback-owner behavior, and TMEM initialized alloc
behavior;
- no existing lit test file is modified;
- all pre-existing lit tests pass.

## Appendix: Phase 2 Automatic Integration Notes

Mixed scheduler choice across functions is acceptable in the eventual automatic
pipeline. A module may contain one `triton::FuncOp` where
`NVWSPartitionSchedulingMeta` succeeds and another `triton::FuncOp` where PSMeta
fails and the current generic scheduler is used instead.

Phase 2 must make that mixed state explicit. The automatic integration should
record, annotate, or otherwise track which partition scheduler was used for each
function or WS loop. The exact mechanism is left to phase 2, but the
requirements are:

- PSMeta success for one function must not force PSMeta success for every
  function in the module.
- PSMeta failure for one function must not require reverting already committed
  PSMeta output in another function.
- Generic fallback must run only on the function whose PSMeta transaction failed
  and whose pass-entry post-DP IR was preserved.
- Downstream passes must be able to consume a module containing both PSMeta and
  generic scheduler output, as long as each scheduled function satisfies the
  common NVWS partition metadata contract.
- Diagnostics and optional debug attributes should make it clear which scheduler
  produced each function's partition metadata.

This appendix is documentation only for phase 1. Do not implement automatic
integration or mixed-scheduler annotations in phase 1.
