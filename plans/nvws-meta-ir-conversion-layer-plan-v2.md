# NVWS Meta IR conversion layer: plan v2

## Hard scope boundary

The protected code is canonical Meta AWS, not the integration code around it.

**Absolute taboo:** do not modify, delete, or add files anywhere under:

```text
third_party/nvidia/hopper/**
```

Those files are the canonical Meta implementation. The design must invoke
their existing passes verbatim and adapt their existing output from outside
that tree. No Meta breadcrumbs, `planOnly` mode, pass options, annotations,
early exits, cleanup changes, heuristic changes, or test-only hooks may be
added there.

## Explicitly authorized source files

The following integration files are expected and authorized to change:

```text
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp
third_party/nvidia/backend/compiler.py
third_party/nvidia/triton_nvidia.cc
third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td
third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt
```

The following NVWS implementation files are authorized to be created or
modified:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/MetaToNVWSConvert.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/PackEpilogueSlices.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAllocas.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignStagePhase.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp
```

## Absolute InsertSemas taboo

`InsertSemas` is an unchanged consumer of the converted IR. Do not modify any
of these files:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.h
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp
test/NVWS/insert_semas*.mlir
test/NVWS/insert_semaphore*.mlir
test/NVWS/tmem-buffer-reuse-semas.mlir
```

This prohibition includes refactoring, access classification, alias or token
handling, synchronization changes, diagnostics, and FileCheck updates.
`MetaToNVWSConvert` must produce IR that satisfies the existing InsertSemas
contract and passes every existing InsertSemas test unchanged. If that appears
impossible, treat it as a converter bug and stop; it is not authorization to
edit InsertSemas. Any other LIT file whose `RUN` line directly invokes
`--nvws-insert-semaphore` is equally read-only, including such files under
`test/NVWS/MetaAutoWS/`.

These copied NVWS policy-port files are authorized to be removed once their
replacement path is proven:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/WSDataPartition.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/PartitionSchedulingMeta.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlannerNVWSAdapter.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlannerNVWSAdapter.h
third_party/nvidia/lib/Dialect/NVWS/Transforms/WSUtility.cpp
third_party/nvidia/lib/Dialect/NVWS/Transforms/WSUtility.h
```

Their pass declarations, CMake entries, Python bindings, and compiler pipeline
calls may be removed from the explicitly authorized integration files above.
If a factory name must remain because another path still calls it, retain a
small NVWS-local entry shell until the caller is migrated; do not edit Meta to
make deletion easier.

## Authorized tests and documentation

The copied policy tests under this exact directory may be deleted or migrated
to conversion-layer coverage, except for files whose `RUN` line invokes
`--nvws-insert-semaphore`; the InsertSemas taboo takes precedence:

```text
test/NVWS/MetaAutoWS/**
```

The following conversion tests may be created or updated:

```text
test/NVWS/meta_to_nvws_convert_partitions.mlir
test/NVWS/meta_to_nvws_convert_descriptor.mlir
test/NVWS/meta_to_nvws_convert_buffer_plan.mlir
test/NVWS/meta_to_nvws_convert_errors.mlir
test/NVWS/pack_epilogue_slices.mlir
```

The existing descriptor-allocation test below is also authorized to be
migrated to converter coverage while its implementation is moved:

```text
test/NVWS/insert_allocas.mlir
```

All direct NVWS mechanism tests under `test/NVWS/insert_semas*.mlir` are
read-only and must pass with their existing CHECK lines unchanged.

Runtime tutorial/test files and non-NVWS LIT files may be changed when removing
an obsolete pass, binding, flag use, or stale expectation actually requires
it. Before editing one, add its exact path and the reason to this allowlist.
There is no wildcard authorization to rewrite tests merely to make a failure
green.

The following gate test remains explicitly read-only because the user already
required it to pass unchanged:

```text
test/TritonGPU/automatic-warp-specialization.mlir
```

Authorized documentation files are:

```text
plans/nvws-meta-ir-conversion-layer-plan-v2.md
plans/agents_gates.md
sema-docs/meta-ports.md
sema-docs/nvws-aws-overview.md
```

## Stop rule for any additional file

If implementation appears to require a source file not explicitly listed
above:

1. Stop before editing it.
2. Spawn an independent, read-only subagent to verify the blocker.
3. Collect the exact failing command, first failing pass and diagnostic,
   minimal IR around the failure, and source lines establishing the cause.
4. Check whether the problem can be solved in the authorized NVWS files or
   integration files.
5. Report the root cause and the exact additional file/change requested.
6. Wait for explicit authorization.

The subagent must not edit the candidate file. This stop rule never permits an
edit under `third_party/nvidia/hopper/**`; a Hopper requirement is reported as
a design blocker.

## Target flow

```text
compiler.py: canonical standalone Meta preparation passes, unchanged
  DataPartition
  Meta-aware loop scheduling
  TMA-store lowering
  SinkBroadcast

AutomaticWarpSpecialization.cpp: canonical Meta planning pass objects
  PartitionSchedulingMeta
  TaskIdPropagation
  redundant-TMEM-zero normalization
  buffer allocation
  loop-invariant TMEM-store hoisting
  NVWS epilogue-slice packing
  MemoryPlanner
  TMA-store annotation and validation
             |
             v
MetaToNVWSConvert                  // first verifier-enabled outer pass
  - consume final async_task_id into ttg.partition
  - complete partition.outputs and WS tags
  - translate descriptor_load/gather + local_store to nvws descriptor ops
  - translate the final Meta buffer plan to NVWS representation
  - reject incomplete or ambiguous input with a precise diagnostic
             |
             v
NVWS mechanisms
  InsertSemas -> LowerSemaphore -> PartitionLoops -> LowerWarpGroup
             |
             v
normal compiler suffix
```

There is exactly one conversion pass. No capture pass and no duplicated Meta
heuristic analysis are allowed.

`AutomaticWarpSpecialization.cpp` owns and invokes the Meta planning prefix as
an ordered pipeline of pass objects. Because Meta's intermediate ownership IR
is intentionally verifier-incomplete between partition scheduling and task-ID
propagation, automatic verification is disabled within that planning prefix.
After the prefix completes, `MetaToNVWSConvert` is the first pass in the outer
pipeline and is added with `addPassWithPartitionVerifier`. MLIR therefore
verifies the IR after conversion, then the explicit partition verifier runs,
before any NVWS mechanism pass consumes the result. No verifier observes the
intermediate Meta representation.

`MetaToNVWSConvert.cpp` must not include Meta pass declarations, invoke a Meta
pass or helper, construct a Meta pass manager, or expose options that cause
Meta policy to run. It accepts already planned IR and performs conversion only.

The NVWS epilogue-slice packing pass is a scheduling-only compatibility
optimization. It runs after canonical Meta buffer allocation has materialized
explicit `local_store` operations and after TMEM-store hoisting, but before
memory planning, so the planner observes the final shortened live ranges. It
does not choose partitions, buffers, synchronization, or conversion
representation.

## Existing-port audit and reuse rule

The existing NVWS ports are the tested source for the small representation
differences that the converter needs. Reuse or refactor that code instead of
reimplementing it, but do not move Meta policy into the converter.

Every consumed source annotation is removed after its NVWS replacement has
been materialized. Only annotations the converter does not consume remain in
the IR.

| Existing logic | Converter action |
| --- | --- |
| `PartitionSchedulingMeta.cpp` scheduling, propagation, splitting, and partition choice | None. Meta has already made these decisions. |
| Final `async_task_id`/result/tag serialization in `PartitionSchedulingMeta.cpp` and `WSUtility.cpp` | Materialize the equivalent NVWS partition attributes without changing the chosen partitions. |
| Direct descriptor construction in `InsertAllocas.cpp:500-539,627-652` | Refactor into `MetaToNVWSConvert` and reuse Meta's exact planned buffer. |
| Buffer annotation emission in `MemoryPlannerNVWSAdapter.cpp:697-839` | Refactor only the final annotation translation; do not copy planner/channel/lifetime logic. |
| `WSDataPartition.cpp` slicing and scheduling enhancements | None. These are Meta heuristic work, not IR conversion. |
| Memory-planner channel, liveness, copy-count, and reuse decisions | None. Preserve the completed Meta plan. |
| `InsertSemas` access analysis, aliases, tokens, synchronization, and tests | Absolute taboo. No source or CHECK changes; the converter must satisfy its current input contract. |

### Partition annotation materialization

This is attribute materialization, not a scheduling algorithm. The converter
does only the following:

- consume each final Meta `async_task_id` into sorted, unique
  `ttg.partition`, then remove `async_task_id`;
- materialize `ttg.partition.outputs` from the already-partitioned structured
  result/yield relationships, reusing the existing per-result helper;
- materialize or copy the WS tag required by NVWS for an already-partitioned
  operation; and
- verify that every operation for which NVWS requires a partition was already
  assigned one by Meta.

It must not choose a partition, add a partition to fix a scheduling decision,
propagate ownership through consumers, or reschedule anything. Preserve
`loop.stage`, `loop.cluster`, partition type/stage metadata, planner attributes,
and tags.

Migrate only the annotation-output CHECKs from the existing partition
scheduler tests into `meta_to_nvws_convert_partitions.mlir`. Meta heuristic
CHECKs remain Meta coverage and are not converter tests.

### Descriptor representation

Convert Meta's already-planned:

```text
tt.descriptor_load/gather + ttg.local_store to a planned ttg.local_alloc
```

to direct `nvws.descriptor_load/gather` using that same allocation. Reuse the
existing transaction-count and op-building code, preserve coordinates and
attributes, and insert at the `local_store` position so the destination
dominates the new op. The converter does not create a buffer or redo gather
slicing.

Migrate the corresponding direct-load/direct-gather cases from
`test/NVWS/insert_allocas.mlir` into
`meta_to_nvws_convert_descriptor.mlir`.

### Buffer-plan annotation translation

The converter does not plan memory. It mechanically translates the final Meta
annotations to the representation expected by NVWS:

- Allocations that already share the same `buffer.id` and do not carry
  `allocation.reuseTarget` form the circular group. Keep that ID, add
  `buffer.circular`, and assign `buffer.start` in allocation order.
- An allocation with `allocation.reuseTarget = X` is ordinary backing-buffer
  reuse, not circular allocation. Set its `buffer.id = X` and
  `buffer.offset = 0`; do not add `buffer.circular`, then remove the consumed
  `allocation.reuseTarget` annotation.
- Preserve Meta's `buffer.copy` and every other completed plan attribute. Do
  not select an algorithm, copy count, lifetime, reuse target, or new buffer.

Reuse only the annotation-emission/validation portion of the existing NVWS
memory-planner adapter. Cover both mapping rules in
`meta_to_nvws_convert_buffer_plan.mlir`. This is not a separate
`reuseTarget` feature and requires no InsertSemas change.

### InsertSemas boundary

The converter emits no semaphore and performs no effect, lifetime, completion,
alias, or general token analysis. `InsertSemas` continues to own all of that
unchanged. Neither its source nor its tests may change. The converted IR must
work with the current pass and satisfy every existing CHECK exactly.

## Integration responsibilities

### `compiler.py`

Keep `TRITON_NVWS_USE_META` unchanged. Under that flag, schedule the existing
canonical standalone preparation passes that precede core AWS. Core AWS owns
the remaining Meta planning pass pipeline, conversion, and NVWS mechanism
path. Remove the old copied-NVWS-policy calls after the conversion path is
wired. Do not change unrelated flags or pipelines.

### `triton_nvidia.cc`

Expose the existing canonical Meta pass factories needed to assemble the
planning prefix, and expose the NVWS conversion entry if the Python pipeline
needs it. This is binding/wiring only; the Meta implementations remain
unchanged.

### `AutomaticWarpSpecialization.cpp`

For `useMetaPartitioner=true`, run the canonical Meta planning prefix as actual
pass objects in production order, stopping before Meta code partition begins.
Run `MetaToNVWSConvert` as the first verifier-enabled outer-pipeline pass after
that prefix, then enter the unchanged NVWS mechanism boundary. Do not call
copied NVWS partitioning, allocation, or planning. Preserve the current
non-Meta NVWS path.

### `MetaToNVWSConvert.cpp`

Perform representation conversion only. It may reconstruct structural facts
explicitly listed above for NVWS consumers, but it must not choose partitions,
schedules, buffer counts, allocation algorithms, reuse policy, or
synchronization policy. It must not run or directly call any Meta heuristic.
Missing policy information is a diagnostic, not permission to modify Meta.

## Removal sequence

1. Capture the existing correctness/performance baseline.
2. Build the single NVWS converter by refactoring the audited representation
   logic, and migrate its focused LIT tests before deleting their old homes.
3. Wire standalone Meta preparation through compiler integration and assemble
   the remaining canonical Meta prefix as pass objects in core AWS.
4. Make core AWS run the converter after the Meta pass prefix and enter the
   NVWS mechanism after conversion, while the non-Meta path is unchanged.
5. Pass build, mandatory LIT, runtime correctness, and performance gates.
6. Remove copied NVWS policy implementations after their representation CHECKs
   have moved to converter tests; do not move Meta heuristic CHECKs into the
   converter.
7. Remove the corresponding declarations, bindings, CMake entries, compiler
   calls, and copied-policy tests only with their implementation.
8. Rerun the full gates from `plans/agents_gates.md`.

## Scope audit before every build

Before each build and before final handoff, inspect every changed path. Apart
from pre-existing user changes, every edited source must appear explicitly in
this plan. If it does not, remove the edit and follow the stop rule.
