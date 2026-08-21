# NVWS descriptor store/reduce implementation plan

## Decision summary

The native Meta stack has two buffer-taking, abstractly synchronous
operations:

```mlir
nvws.descriptor_store %desc[%coords] %src
nvws.descriptor_reduce add, %desc[%coords] %src
```

Both operations are resultless and have no TMA completion token. They make the
shared-memory source explicit while hiding the eventual asynchronous TTNG
issue/wait representation.

The native Meta pipeline creates these operations after task propagation and
keeps them through buffer allocation, memory planning, subtile formation, code
partitioning, and loop scheduling. At the existing post-schedule TMA-store
wait-placement phase, each operation becomes the exact TTNG sequence used by
the compatibility path:

```mlir
%token = ttng.async_tma_copy_local_to_global ... -> !ttg.async.token
ttng.async_tma_store_token_wait %token

%token = ttng.async_tma_reduce ... -> !ttg.async.token
ttng.async_tma_store_token_wait %token
```

The generated wait is placed before the next overwrite of the same physical
SMEM staging slot using the existing `buffer.copy = K` algorithm in
`doTMAStoreWaitReorder`.

The hard compatibility requirement remains that the IR after late
materialization and all final TTGIR/LLVM/PTX code are identical to the current
early-lowered path. Completion/release metadata carried by the legacy early
wait is now transferred to the late-generated wait. General Meta-to-NVWS
`InsertSemas` support is a separate follow-up and is not part of the
native-Meta implementation.

### Implementation status (2026-08-21)

The native-Meta implementation described by this plan is present. In
particular:

- post-PSM conversion creates the canonical empty `ttg.local_alloc` plus
  `ttg.local_store`, followed by tokenless `nvws.descriptor_store/reduce`;
- memory planning recognizes the abstract operations and hoists TMA staging
  through enclosing `scf.for` and `scf.while` loops when its operands are loop
  invariant;
- rotation policy is annotated after subtile generation, at the same boundary
  for abstract and legacy representations;
- code partition and token lowering carry deferred completion/release metadata
  on the abstract operation and resolve it to barrier/predicate operands; and
- post-schedule materialization transfers those operands to the generated
  `ttng.async_tma_store_token_wait` before applying the existing rotation
  algorithm.

General `InsertSemas` consumption of the new operations remains deferred.

Final verification on 2026-08-21 rebuilt `triton` and `triton-opt`, then ran
the affected lit suites (211 passed, 9 expected failures) and the authorized
six-file GPU pytest gate (1942 passed, 931 skipped in 141.58 seconds).

## Scope

### In scope

- Define `nvws.descriptor_store` and `nvws.descriptor_reduce`.
- Convert native-Meta `tt.descriptor_store` and `tt.descriptor_reduce` to the
  new buffer-taking operations after Meta partition scheduling and task-ID
  propagation.
- Preserve the current staging allocation, memory plan, partition ownership,
  subtile structure, loop schedule, TMA issue, token, wait placement, final
  pending count, and PTX.
- Create the TMA completion token only at the post-schedule wait-placement
  phase.
- Continue supporting pre-existing TTNG TMA store/reduce operations and the
  standalone early-lowering pass for pipelines outside native Meta.

### Initially out of scope

- Changing Python language semantics or the `tt.descriptor_*` operations.
- Adding descriptor scatter support.
- Replacing the Meta-to-NVWS bridge's current early TTNG store/reduce handling.
- Teaching the general NVWS `InsertSemas` pipeline to consume the new
  operations. That work is explicitly deferred.
- Changing store/reduce scheduling, SMEM depth, barrier topology, or generated
  instructions as an optimization.

## Legacy and implemented native-Meta pipelines

### Legacy early-lowered pipeline

```text
tt.descriptor_store/reduce
  -> early local_alloc + TTNG issue token + token_wait
  -> PartitionSchedulingMeta
  -> doTaskIdPropagate
  -> doBufferAllocation
  -> doMemoryPlanner
  -> doAnnotateTMAStoreWaits
  -> doCodePartition
  -> doTokenLowering
  -> scheduleLoops
  -> doTMAStoreWaitReorder
  -> generic pipeliner
  -> TMAStoreTokenWaitLowering
  -> LLVM/PTX
```

### Implemented native-Meta pipeline

```text
tt.descriptor_store/reduce
  -> PartitionSchedulingMeta
  -> doTaskIdPropagate
  -> empty local_alloc + local_store + tokenless nvws.descriptor_store/reduce
  -> doBufferAllocation
  -> doMemoryPlanner
  -> doGenerateSubtiledRegion
  -> annotate the abstract op with buffer.copy = K policy
  -> doCodePartition
  -> lower subtiled regions carrying deferred NVWS completion tokens
  -> doTokenLowering
  -> scheduleLoops
  -> materialize TTNG issue token + token_wait
  -> doTMAStoreWaitReorder
  -> generic pipeliner
  -> TMAStoreTokenWaitLowering
  -> LLVM/PTX
```

`PartitionSchedulingMeta` already recognizes raw `tt.descriptor_store` and
`tt.descriptor_reduce`, so production conversion must remain after that pass.
No new PSM representation is needed.

## Operation definitions

### Common semantic contract

Both new operations:

- are resultless;
- return no `!ttg.async.token`;
- are abstractly synchronous: at this abstraction level the descriptor update
  and the read of the source SMEM buffer are complete when the operation is
  complete;
- take the source as `!ttg.memdesc`, making the SMEM read explicit;
- model the descriptor as global-memory read/write and the source as
  shared-memory read;
- preserve the descriptor coordinates exactly;
- carry no load-style `txCount`, because store completion does not use a TMA
  load transaction-count barrier;
- carry no cache modifier;
- initially lower with `EvictionPolicy::NORMAL`, exactly as
  `doTMAStoreLowering` does today;
- carry no public buffer-count or wait-placement operand. `buffer.copy` remains
  planner metadata on the staging allocation, with an internal
  `can_rotate_by_buffer_count` policy attribute copied to the abstract op after
  planning.

The production form may temporarily carry native-Meta channel-completion
operands added by code partitioning: deferred `nvws_tokens` plus indices, or
resolved barriers plus predicates. These are operands used to return ownership
of the source buffer; they are not a TMA completion-token result. Source
conversion creates all four segments empty, token lowering resolves deferred
tokens, and late lowering transfers the resulting barriers/predicates to
`TMAStoreTokenWaitOp`.

### Store-like interface

Use the existing `ttng::TMAStoreLikeOpInterface` for both new operations. Its
contract is exactly the representation-neutral API needed here: `getDesc`,
`getSrc`, and `getSrcMutable`; it has no token method or token-result
requirement. Broaden its documentation from "asynchronous hardware operation"
to a TMA operation with an explicit shared-memory source, because the NVWS form
is abstractly synchronous and the TTNG form is asynchronous.

Using the existing interface has two important benefits:

- core `GenerateSubtiledRegion.cpp` can recognize the NVWS operations without
  depending on concrete third-party NVWS classes; and
- `ScheduleLoops.cpp` already recognizes `TMAOpInterface`, inherited by
  `TMAStoreLikeOpInterface`, as a latency/scheduling operation.

Add the TritonNvidiaGPU op-interface TableGen include to `NVWSOps.td` and link
`NVWSIR` against `TritonNvidiaGPUIR` in
`third_party/nvidia/lib/Dialect/NVWS/IR/CMakeLists.txt`.

Proxy-fence and sanitizer passes also recognize this interface. That is safe
because the NVWS operations model the same TMA SMEM read and must be eliminated
before those later passes in the production pipeline. The post-materialization
survivor check makes that ordering explicit.

### `nvws.descriptor_store`

Proposed ODS shape in
`third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td`:

```tablegen
def NVWS_DescriptorStoreOp
    : NVWS_Op<"descriptor_store", [TMAStoreLikeOpInterface]> {
  let summary = "Store from shared memory through a tensor descriptor";
  let description = [{
    Abstractly synchronous descriptor store whose source is an explicit shared
    memory buffer. The operation is lowered to an asynchronous TTNG TMA issue
    and a completion wait after Meta loop scheduling.
  }];
  let arguments = (ins
    Arg<TT_TensorDescType, "",
        [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$desc,
    Variadic<I32>:$indices,
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src
  );
  let assemblyFormat =
      "$desc `[` $indices `]` $src attr-dict `:` type(operands)";
  let hasVerifier = 1;
}
```

Example:

```mlir
nvws.descriptor_store %desc[%m, %n] %staging
  : !tt.tensordesc<128x64xf16, #shared>, i32, i32,
    !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
```

The operation does not carry the legacy `tt.descriptor_store.reduce_kind`.
Only `reduce_kind = none` maps to this operation.

### `nvws.descriptor_reduce`

Proposed ODS shape:

```tablegen
def NVWS_DescriptorReduceOp
    : NVWS_Op<"descriptor_reduce", [TMAStoreLikeOpInterface]> {
  let summary =
      "Reducing store from shared memory through a tensor descriptor";
  let description = [{
    Abstractly synchronous descriptor reduction whose source is an explicit
    shared memory buffer. Atomicity and supported kinds match
    ttng.async_tma_reduce.
  }];
  let arguments = (ins
    TT_DescriptorReduceKindAttr:$kind,
    Arg<TT_TensorDescType, "",
        [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$desc,
    Variadic<I32>:$indices,
    Arg<TTG_MemDescType, "", [MemRead<SharedMemory>]>:$src
  );
  let assemblyFormat =
      "$kind `,` $desc `[` $indices `]` $src attr-dict `:` type(operands)";
  let hasVerifier = 1;
}
```

Example:

```mlir
nvws.descriptor_reduce add, %desc[%m, %n] %staging
  : !tt.tensordesc<128x64xf32, #shared>, i32, i32,
    !ttg.memdesc<128x64xf32, #shared, #smem, mutable>
```

### Verification

Implement the verifiers in
`third_party/nvidia/lib/Dialect/NVWS/IR/Ops.cpp`:

1. Reuse `verifyDescriptorLoadStoreOp` to require matching descriptor/source
   element types and element counts.
2. Require tiled-store coordinate counts accepted by the eventual TTNG op.
3. Require a shared-memory `MemDescType` source.
4. Reject `DescriptorReduceKind::NONE` on `nvws.descriptor_reduce`.
5. Either share the TTNG reduce-kind/type predicate or leave target-specific
   legality to the generated `ttng.async_tma_reduce` verifier. Do not maintain
   two divergent legality tables.

### Internal completion-carrier fields

Both operations contain the same paired internal completion fields carried by
`TMAStoreTokenWaitOp`:

```tablegen
AttrSizedOperandSegments,
Variadic<TTG_MemDescType>:$barriers,
Variadic<I1>:$barrier_preds,
Variadic<AnyType>:$nvws_tokens,
Variadic<I32>:$nvws_token_indices
```

`addBarrier(Value, Value)` and `addToken(Value, Value)` helpers reuse the
existing paired-segment verification rules. The ordinary parsed form has empty
segments and remains resultless.

## Phase 0: Establish the codegen oracle

Before changing the production pipeline, capture current native-Meta output for
at least these cases:

- ordinary descriptor store, `buffer.copy = 1`;
- rotated descriptor store, `buffer.copy = 2`;
- descriptor reduce;
- mixed store/reduce ordering;
- same-task subtiled epilogue;
- separate-task epilogue with a completion/release barrier;
- `tmaStorePipelining = false`.

Save normalized dumps at these boundaries:

1. after `PartitionSchedulingMeta`;
2. after `doMemoryPlanner`;
3. after `doCodePartition` and `doTokenLowering`;
4. after `doTMAStoreWaitReorder`;
5. after the generic pipeliner;
6. final TTGIR, LLVM IR, and PTX.

Normalization may remove locations and unstable symbol hashes only. Operation
order, attributes, partition IDs, loop stages/clusters, buffer metadata,
barrier operands, wait `pendings`, and instructions are part of the oracle.
Write these dumps under `/tmp/nvws-descriptor-store-reduce/` or the repository
root's `.agent-artefacts/` directory. Do not create dump or reproducer artifacts
inside source-tree subdirectories.

## Phase 1: Convert TT operations to tokenless NVWS operations

### Keep TT operations through PSM

Stop running the standalone early TMA-store lowering before
`PartitionSchedulingMeta` only for native Meta. Audit both native-Meta compiler
paths in `third_party/nvidia/backend/compiler.py`:

- in the SM80/SM90 branch, make the current `add_tma_store_lowering` call run
  only when `knobs.nvidia.use_meta_ws` is false; and
- in the SM100+ branch, retain the Meta-to-NVWS call in the `use_nvws_meta`
  path, but remove/skip the separate call in the native `use_meta_ws` path.

Keep early lowering for:

- non-Meta compilation;
- the Meta-to-NVWS bridge while it still expects early TTNG operations;
- standalone pass tests and any explicit compatibility mode.

Do not globally remove the `ttg.early_tma_store_lowering` option or the existing
`doTMAStoreLowering` implementation.

### Add conversion beside descriptor load/gather conversion

In `WSLowerMem.cpp`, add a native-Meta conversion routine next to
`doConvertDescriptorLoadsToNVWS`, for example:

```cpp
LogicalResult doConvertDescriptorStoresToNVWS(triton::FuncOp funcOp);
```

Call it in `WarpSpecialization.cpp` immediately after task-ID propagation and
the existing descriptor load/gather conversion, before `doBufferAllocation`.

For a plain store:

```text
tt.descriptor_store(desc, indices, tensor_src)
  -> ttg.local_alloc() : memdesc
  -> ttg.local_store(tensor_src, memdesc)
  -> nvws.descriptor_store(desc, indices, memdesc)
```

For a reduction:

```text
tt.descriptor_reduce(kind, desc, indices, tensor_src)
  -> ttg.local_alloc() : memdesc
  -> ttg.local_store(tensor_src, memdesc)
  -> nvws.descriptor_reduce(kind, desc, indices, memdesc)
```

Conversion requirements:

- Reuse `getEncodingFromDescriptor` and the exact mutable shared `MemDescType`
  construction from `doTMAStoreLowering`.
- Preserve the current staging `NameLoc` convention so allocation diagnostics
  and deterministic grouping do not change.
- Preserve split ownership exactly. The empty staging `ttg.local_alloc` and
  its `ttg.local_store` belong to the tensor source producer's task (for
  example computation task 0), while `nvws.descriptor_store/reduce` belongs to
  the original descriptor operation's epilogue/reduction task (for example
  task 2). Copy the staging task ID from `src.getDefiningOp()` when available,
  with an explicitly tested fallback for block arguments; do not stamp the
  descriptor operation's task ID onto every operation.
- Give the staging allocation and local store the original descriptor
  operation's loop stage/cluster metadata while keeping the source producer's
  task ownership.
- Give the NVWS operation the original descriptor operation's
  `async_task_id`, `ttg.partition`, `loop.stage`, `loop.cluster`, WS tag, and
  other scheduling metadata.
- Preserve descriptor, indices, source shape/type, reduction kind, and
  location exactly.
- Bypass and erase dead single-use `ttg.convert_layout` forwarders before
  staging so the local store keeps the logical producer value and ownership,
  matching the legacy channel plan.
- Create no TTNG issue, TMA completion token, or token wait.
- After conversion, diagnose any supported plain store/reduce left behind so
  NVWS operations cannot leak unpredictably later.

Treat legacy `tt.descriptor_store reduce_kind != none` explicitly. The narrow
path is:

1. establish its current intended lowering with a baseline test;
2. canonicalize it to `tt.descriptor_reduce` when semantically valid; otherwise
3. reject it from the new native-Meta path with a clear diagnostic.

Do not silently drop `reduce_kind` or map it to an ordinary store.

## Phase 2: Teach Meta analyses about the abstract operations

The abstract operations must be treated exactly like the corresponding TTNG
issue for source-buffer ownership, planning, and scheduling, but not as if a
hardware TMA issue or completion token already exists.

### Buffer and channel analysis

Update the semantic store-like classifications in:

- `CodePartitionUtility.cpp`
- `WSCodePartition.cpp`
- `WSBuffer.cpp`

Use `TMAStoreLikeOpInterface` or a small common helper instead of adding more
scattered concrete `isa` lists. Required behavior:

- the NVWS operation is a read/consumer of its source staging allocation;
- its descriptor and source appear in debug/graph descriptions;
- local-load-to-descriptor-store forwarding and actual-consumer discovery see
  the NVWS form;
- store and reduce remain distinguishable where reduction ownership matters;
- partition and channel construction produce the same result as the current
  TTNG issue/wait pair, except for the deferred completion endpoint discussed
  below.

Audit all concrete checks for `DescriptorStoreOp`, `DescriptorReduceOp`,
`AsyncTMACopyLocalToGlobalOp`, and `AsyncTMAReduceOp` in those files. Retain a
concrete cast only where opcode-specific fields such as `kind` are required.

### Memory planner

Update `WSMemoryPlanner.cpp` so both new operations participate in every place
that currently recognizes TTNG store/reduce staging:

- staging detection (`buffer.tmaStaging = 1` for store and `2` for reduce);
- descriptor/original-producer grouping for fused epilogues;
- search-model `BufferKind::Staging` classification;
- multi-store fallback counting;
- staging priority and copy-depth selection;
- allocation hoisting through memdesc views; and
- hoisting loop-invariant TMA staging before the outermost enclosing
  `scf.for` or `scf.while`.

The required invariant is identical `buffer.id`, `buffer.copy`,
`buffer.tmaStaging`, `allocation.shareGroup`, `allocation.reuseTarget`, SMEM
offsets, and total SMEM usage for the baseline kernels.

### Subtiled regions

Refactor `GenerateSubtiledRegion.cpp` to use `TMAStoreLikeOpInterface` for
TMA-store-source discovery. A tokenless NVWS operation must still
be pulled into the same per-tile chain as its same-task `ttg.local_store`:

```text
local_store_t -> nvws.descriptor_store/reduce_t
```

There is intentionally no token wait to pull into that chain yet. The subtiled
regions are lowered before late TMA materialization, after which the generated
wait is placed using the flattened schedule. Preserve the existing same-task
interleaving and distinct-buffer safety checks. Planned TMA staging
allocations remain outside producer tile regions and are passed as tile
operands, so producer `local_store` and consumer descriptor operations continue
to reference the same physical allocation.

### Loop scheduling and latency

The new operations survive through `scheduleLoops`. Their
`TMAStoreLikeOpInterface` makes the core schedule treat them as TMA latency
operations, but still audit the concrete store-like classifications in the
NVIDIA latency and modulo-scheduling utilities, including `NVLatencyModel.cpp`
and `ModuloSchedulePass.cpp`. Give an abstract store/reduce the same issue
latency and scheduling category as its current TTNG counterpart. Copy the final
serialized `loop.stage` and `loop.cluster` onto the generated TTNG issue during
late lowering.

The generic Triton pipeliner does not need to understand the NVWS operations:
late materialization happens before `passes.ttgpuir.add_pipeline`.

## Phase 3: Record the rotation policy without a token

Generalize `doAnnotateTMAStoreWaits` and its validator in
`WSTMAStoreLowering.cpp` to recognize both:

- legacy/pre-existing `ttng.async_tma_store_token_wait`; and
- tokenless `nvws.descriptor_store/reduce`.

For the NVWS form, read `buffer.copy = K` from the source staging allocation
after `doGenerateSubtiledRegion` and put the internal
`can_rotate_by_buffer_count = K` attribute on the abstract operation. This is
planner policy, not a completion token or a wait location.

This post-subtile boundary is intentional. A subtiled source is already a
region argument there, so both the abstract and legacy representations take
the same conservative fallback when the underlying allocation can no longer
be proven. Annotating before subtile formation would rotate only the new form
and change code generation.

Recording `K` here avoids trying to rediscover the base allocation after code
partitioning has introduced stage views, subtiled arguments, or other aliases.
Validation must remove the attribute if the source cannot be proven to belong
to the planned staging allocation, matching the current conservative fallback.

Keep legacy wait annotation working for TTNG operations that enter the pass
from compatibility paths.

## Phase 4: Materialize the TTNG issue/token/wait and place the wait

Refactor the existing post-schedule wait phase in
`WSTMAStoreLowering.cpp` into an unconditional materialization step followed by
conditional rotation, for example:

```cpp
LogicalResult doMaterializeAndPlaceTMAStoreWaits(
    triton::FuncOp funcOp, bool enableRotation);
```

Call it after `scheduleLoops`, subtiled-region lowering, and
`cleanupWarpSpecializedLoops`, at the current `doTMAStoreWaitReorder` location.
The call itself must be unconditional. `tmaStorePipelining` may disable wait
movement, but it must never prevent NVWS-to-TTNG lowering.

### Late lowering mapping

For `nvws.descriptor_store`:

```mlir
%token = ttng.async_tma_copy_local_to_global
    %desc[%coords] %src evictionPolicy = normal -> !ttg.async.token
ttng.async_tma_store_token_wait %token
```

For `nvws.descriptor_reduce`:

```mlir
%token = ttng.async_tma_reduce kind,
    %desc[%coords] %src evictionPolicy = normal -> !ttg.async.token
ttng.async_tma_store_token_wait %token
```

Requirements:

- Create an explicit `!ttg.async.token` result. Do not use the existing
  no-token TTNG builder overload.
- The token initially has exactly one completion-wait user.
- Copy location, owner/partition, loop stage, loop cluster, predicates if later
  introduced, and all valid WS scheduling metadata to the TTNG issue.
- Copy `can_rotate_by_buffer_count` to the generated wait.
- Replace the abstract operation in the serialized `CoarseSchedule` with the
  TTNG issue at exactly the same stage/cluster.
- Copy the abstract operation's final stage/cluster to both the TTNG issue and
  the initial adjacent wait before attempting to rotate it. This reproduces the
  current disabled-pipelining behavior and gives the reordering code the same
  fallback schedule.
- Erase every abstract store/reduce and diagnose any survivor before leaving
  the pass.

### Wait-placement decision

Reuse the current algorithm rather than reimplementing it:

1. Read `K = can_rotate_by_buffer_count`.
2. Walk the linearized schedule after the generated issue.
3. Find the K-th subsequent TMA store-like issue.
4. Find the `ttg.local_store` that will overwrite that issue's source slot.
5. Insert the generated token wait immediately before that writer.
6. If another partition owns the writer, insert the wait before the guarding
   `wait_barrier` instead.
7. If no safe target is proven, leave the wait adjacent to its issue.

This is a write/reuse boundary, not an arbitrary later SMEM access. Additional
reads do not require completion; overwriting the physical source slot does.

### `tmaStorePipelining = false`

Still lower every NVWS operation and create its specific token wait. Leave the
wait adjacent to the issue and do not attach/consume a rotation attribute. Add
a regression check that no `nvws.descriptor_store/reduce` survives this mode.

### Downstream token handling

No new token-threading implementation is needed after late materialization:

- the generic pipeliner runs next and can make the newly created token
  loop-carried exactly as it does today;
- `TMAStoreTokenWaitLowering` continues tracing the token through direct,
  `scf.if`, loop-result, and loop-carried forms;
- `computePendings` continues counting intervening TTNG store/reduce issues;
- final lowering continues emitting `ttng.async_tma_store_wait {pendings = N}`
  followed by any barrier arrivals;
- LLVM lowering continues emitting the same TMA commit and
  `cp.async.bulk.wait_group N` instructions.

The token is therefore created late enough to avoid polluting Meta's abstract
store representation, but early enough for all existing token threading and
final pending-count logic.

## Phase 4.5: Carry completion/release metadata

This phase is implemented separately from operation/token placement and is
part of the production native-Meta pipeline.

Today `doCodePartition` follows the early TTNG issue token to
`TMAStoreTokenWaitOp`, chooses that wait as the final SMEM consumer, and attaches
deferred consumer-release token/index operands to it. `doTokenLowering` resolves
those operands into real barrier/predicate operands before
`doTMAStoreWaitReorder` moves the wait. A late-generated wait must receive the
same operands.

The implemented codegen-preserving flow is:

1. Treat `nvws.descriptor_store/reduce` as the abstract completion endpoint
   during code partitioning.
2. Give the new operations the internal completion-carrier operand segments and
   `addToken`/`addBarrier` helpers described above. These are channel
   synchronization operands, not a TMA completion-token result; the public
   source form remains resultless.
3. In the actual-consumer analysis in `WSCodePartition.cpp`, keep the abstract
   store/reduce itself as the final consumer; there is no issue token to follow
   yet.
4. At consumer-release insertion, attach the deferred release token/index to
   the abstract operation with `addToken` instead of emitting an early
   standalone release.
5. For `SubtiledRegionOp`, attach the captured tile token/index to the inner
   abstract store/reduce. Extend
   `doLowerSubtiledRegionsWithNVWSOps` so a region containing such deferred
   operands is inlined before token lowering can inspect the token use.
6. Have `WSLowerToken.cpp` count the abstract operation as a consumer-token user
   and resolve its deferred token/index operands to the same real
   barriers/predicates currently added to `TMAStoreTokenWaitOp`. Keep the
   abstract operation alive with those realized operands.
7. During late materialization, transfer the realized barriers/predicates to
   the generated `TMAStoreTokenWaitOp` and clear them from the erased abstract
   operation.
8. Let the existing wait reordering move completion and release together.

Do not move the release independently of the wait, and do not insert an arrival
beside the issue. Either change would return the staging buffer before the TMA
engine has finished reading it.

The general NVWS `InsertSemasAccessDag.cpp` transition can later use the same
abstract completion contract, replacing its current early-TTNG completion
anchor workaround. That bridge work should be a separate plan/series so the
native-Meta codegen-preserving change remains reviewable.

## Phase 5: Switch native Meta and retain compatibility paths

The production switch uses the following rules:

1. Disable the standalone early TTNG store lowering at every native-Meta
   compiler insertion point.
2. Enable post-PSM TT-to-NVWS store/reduce conversion.
3. Keep early lowering unchanged for non-Meta and Meta-to-NVWS paths.
4. Keep legacy TTNG annotation/reorder support so standalone tests and imported
   TTGIR remain valid.
5. Add a final verifier immediately after late materialization requiring no
   `nvws.descriptor_store/reduce` in executable IR.

Do not combine this switch with planner tuning, semaphore optimizations, wait
coalescing, or unrelated cleanup. Any final-code difference must be treated as
a regression, not an expected consequence of the abstraction change.

## Critical implementation sites

Line numbers are from the branch at plan creation and may shift during the
series; function names are the stable anchors.

| Area | Current site | Implemented behavior |
|---|---|---|
| Native-Meta pipeline | `third_party/nvidia/backend/compiler.py` | Skip early lowering only at the two native-Meta sites; retain non-Meta and Meta-to-NVWS behavior. |
| Abstract conversion | `WSLowerMem.cpp`; `WarpSpecialization.cpp` | Create a source-owned empty staging alloc + local store plus descriptor-op-owned NVWS store/reduce beside load/gather conversion. |
| Planner annotation | `WarpSpecialization.cpp`; `WSTMAStoreLowering.cpp` | After subtile generation, put `can_rotate_by_buffer_count` on the abstract op and retain legacy wait support at the same boundary. |
| Subtile recognition | `GenerateSubtiledRegion.cpp:30-40,500-529,1101-1113,1262-1288,1315-1348` | Recognize `TMAStoreLikeOpInterface` without requiring a result token. |
| Staging planning | `WSMemoryPlanner.cpp` | Treat NVWS store/reduce like concrete staging consumers and hoist loop-invariant staging through both `scf.for` and `scf.while`. |
| Buffer/channel classification | `WSBuffer.cpp:290-295`; `CodePartitionUtility.cpp:3370-3420`; `WSCodePartition.cpp:2672-2714,3480-3526` | Preserve the same source channel, actual consumer, and store/reduce classification. |
| Completion attachment | `WSCodePartition.cpp` | Attach deferred release metadata to the abstract completion carrier, including subtiled cases. |
| Channel-token lowering | `WSLowerToken.cpp` | Resolve abstract-op deferred tokens to real barrier/predicate operands. |
| Late materialization | `WarpSpecialization.cpp`; `WSTMAStoreLowering.cpp` | Unconditionally create TTNG issue/token/wait, transfer completion barriers, then conditionally run the existing rotation decision. |
| Final wait lowering | `WSTMAStoreLowering.cpp:707-823`; `compiler.py:1191` | Leave unchanged; use it as an equivalence boundary. |

`PartitionSchedulingMeta.cpp` and the pre-PSM data-partition classifiers should
not require new NVWS cases in the production path because they continue to see
the original TT operations.

## Tests

Prefer extending existing test files rather than creating new ones.

### Dialect syntax and verification

- Extend `test/NVWS/ops.mlir` with store and reduce parse/print cases and assert
  that neither operation has a result/token.
- Extend `test/NVWS/invalid.mlir` with descriptor/source element and element
  count mismatches, non-SMEM source rejection, coordinate errors, and reduce
  kind `none` rejection.

### TT-to-NVWS conversion

- Extend or repurpose
  `test/Hopper/WarpSpecialization/ws_tma_store_lowering.mlir` to check:
  - ordinary store conversion;
  - reduce conversion;
  - a reduce-only function;
  - exact descriptor, coordinates, kind, source memdesc, task, stage, and
    cluster preservation;
  - no early TTNG issue/token/wait in the new conversion mode;
  - explicit handling of legacy store `reduce_kind != none`.

Keep the existing standalone early-lowering checks under their compatibility
mode rather than deleting coverage.

### Memory planning and subtile handling

- Extend `ws_memory_planner_tma_store_staging_cap.mlir` with abstract store and
  reduce sources, checking identical `buffer.tmaStaging`, `buffer.copy`, and
  `buffer.id`.
- Retain the larger backward-memory regressions in
  `ws_memory_planner_bwd_buffer_reuse.mlir` and
  `ws_code_partition_bwd_staging_slot_rotation.mlir`.
- Extend existing generate-subtiled-region tests with tokenless store/reduce
  chains and verify same-task interleaving remains
  `local_store_t -> nvws.descriptor_*_t` before late lowering.

### Late lowering and wait placement

- Extend `ws_tma_store_token_wait_reorder.mlir` with abstract stores covering
  `K = 1`, `K = 2`, missing `buffer.copy`, outside-loop fallback, local-writer
  placement, and cross-partition guarding-barrier placement.
- Extend `ws_tma_store_token_wait_reorder_reduce_xfail.mlir` (despite its name,
  it currently runs as an active test) with abstract reduce and mixed
  store/reduce ordering.
- Extend `ws_tma_store_pipelining_option.mlir` to prove both enabled and disabled
  modes eliminate all abstract operations.
- Add an integration RUN to `ws_tma_store_token_wait_pendings.mlir` that starts
  from the abstract operations, performs late materialization/reordering, then
  checks the same final `pendings` values.

### Completion/release equivalence

- Preserve and update the existing full-pipeline checks that fuse a
  consumer-release barrier into `TMAStoreTokenWaitOp`, including the relevant
  checks in `blackwell_ws_matmul_tma.mlir` and
  `ws_code_partition_bwd_staging_slot_rotation.mlir`.
- Cover both ordinary store and reduce, with one and multiple release barriers.
- Assert that the barrier is attached to the generated wait and moves with it;
  assert that no standalone early arrival remains beside the issue.

### End-to-end equivalence

For every Phase 0 oracle case, compare old and new output:

- exact normalized IR immediately after `doTMAStoreWaitReorder`;
- exact normalized IR after the generic pipeliner;
- exact final TTGIR and LLVM IR;
- exact normalized PTX instruction stream.

At minimum, final PTX must preserve:

- TMA store/reduce issue count and order;
- commit count and order;
- `cp.async.bulk.wait_group` values and positions;
- barrier arrive count, predicates, and positions;
- shared-memory size and offsets;
- warp-group/partition ownership.

The user explicitly requires the six warp/AutoWS pytest files listed below as
one post-change gate with a 240-second timeout covering the complete command.
No other pytest suite is authorized by this plan.

## Build and lit-test commands

The order is mandatory: first build both `triton` and `triton-opt`, and only
after that succeeds run lit tests. Run every command from this branch's build
directory, replacing the `triton-01.git` directory named in `../AGENTS.md` with
the current `triton-meta-01.git` branch directory:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-meta-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

After that build succeeds, remain in the same directory and run the focused lit
tests with the exact `llvm-lit` executable required by `../AGENTS.md`:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/NVWS/ops.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/NVWS/invalid.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_tma_store_lowering.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_tma_store_token_wait_reorder.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_tma_store_token_wait_reorder_reduce_xfail.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_tma_store_token_wait_pendings.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_tma_store_pipelining_option.mlir
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization/ws_memory_planner_tma_store_staging_cap.mlir
```

Then, still from that same build directory, run the complete affected lit
suites:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/NVWS
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization
```

Only after the build and lit tests pass, return to the source root and run all
six explicitly authorized pytest files in one command. The one `timeout 240s`
wraps the entire six-file suite, not each file independently:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-meta-01.git/
timeout --signal=TERM --kill-after=10s 240s \
  env PYTHONPATH="$PWD/python" \
  pytest -n24 -s --tb=short \
    python/test/unit/language/test_amd_warp_pipeline.py \
    python/test/unit/language/test_tutorial09_warp_specialization.py \
    python/test/unit/language/test_tlx_warp_specialization.py \
    python/test/unit/language/test_warp_specialization.py \
    python/test/unit/language/test_autows_addmm.py \
    python/test/unit/language/test_autows_quantized_matmul.py
```

Verified pre-change baseline on 2026-08-21:

```text
1942 passed, 931 skipped, 18 warnings in 127.14s
```

Use the exact environment/configuration from the Phase 0 baseline for old/new
IR and PTX comparisons. Store generated dumps only in `/tmp` or
`.agent-artefacts` at the source-tree root.

## Commit sequence

Keep the implementation reviewable and do not switch production behavior in
the first commit:

1. **Dialect contract**: NVWS operation definitions using the existing TTNG
   store-like interface, required NVWSIR linkage, verifiers, and syntax/invalid
   tests.
2. **Abstract conversion**: TT-to-NVWS conversion plus focused conversion tests;
   leave the production compiler on early TTNG lowering.
3. **Meta analysis support**: buffer/channel, planner, subtile, and scheduler
   recognition with focused lit coverage.
4. **Late token materialization**: unconditional NVWS-to-TTNG issue/token/wait
   generation and existing wait-placement reuse, including the disabled option.
5. **Completion/release transfer**: carry deferred synchronization through the
   abstract op and transfer it to the generated wait.
6. **Native-Meta switch**: disable early lowering only in native Meta, update
   docs, and land exact IR/PTX equivalence evidence.

Store and reduce must be covered in every layer before proceeding to the next
commit. In particular, include a reduce-only test so the current early
`storeOps.empty()` return cannot mask missing reduce handling.

## Documentation updates

The production switch updates these documents:

- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/MemoryLowering.md`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/TMAStoreWaitPipeline.md`
- `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/docs/Overview.md`
- `sema-docs/insert-semas/access-dag.md`

The semaphore document should say that the operations now exist but that
general `InsertSemas` consumption remains deferred until its follow-up series.

## Acceptance criteria

The native-Meta transition is complete only when all of the following hold:

- Both NVWS operations have explicit, verified, tokenless public semantics.
- Raw TT store/reduce operations survive PSM and convert after task propagation.
- Planner output is identical for all baseline cases.
- Subtiled and non-subtiled schedules are identical.
- Every abstract operation is unconditionally lowered before the generic
  pipeliner, including when TMA-store pipelining is disabled.
- The generated TMA token is threaded and lowered exclusively by the existing
  downstream token machinery.
- Wait placement protects the next overwrite of the physical SMEM slot.
- Consumer-release barriers are attached to and move with the generated wait.
- Normalized IR after late materialization and final TTGIR/LLVM/PTX match the
  baseline.
- The required focused and complete affected lit suites pass after a successful
  `ninja triton triton-opt` build.
- The six explicitly authorized warp/AutoWS pytest files pass together under
  one 240-second total timeout.
- Non-Meta and Meta-to-NVWS compatibility paths retain their existing early
  lowering until separately migrated.
