# New Semaphore Pipeline v4

This plan is only for staging the semaphore-pass replacement. It is not an
implementation patch.

The first deliverable is a TMEM-only `nvws-insert-semas` pass that can replace
`nvws-insert-tmem-semaphore` without changing observable IR behavior. Local/SMEM
support and `insert-allocs` are later work after the TMEM path is proven.

## Scope

In scope:

- Add a new `nvws-insert-semas` pass.
- First make it a wiring clone of `nvws-insert-tmem-semaphore`.
- Then replace token-derived scheduling with an alloc/access analysis for TMEM.
- Treat every TMEM alloc as belonging to a logical `buffer.id` group.
- If an alloc has no `buffer.id`, synthesize a unique id in pass-side analysis.
- Verify that old and new passes produce equivalent output for existing lit
  tests and for the saved fused-attention dump.
- Switch automatic WS from `createNVWSInsertTmemSemaphore()` to
  `createNVWSInsertSemas()` only after the TMEM-only no-token path passes.

Out of scope for the first deliverable:

- Local/SMEM support.
- `insert-allocs`.
- MemoryPlanner redesign.
- Deleting old passes.
- Changing existing lit tests.
- Running pytest.

## Non-Negotiable Rules

1. Existing lit tests must not be modified unless the user explicitly
   authorizes the change after a root-cause report.

2. New lit tests may be added.

3. Per `AGENTS.md`, build first, then run lit tests:

   ```
   cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
   ninja triton triton-opt
   ```

4. Run lit from the build directory, but use absolute source test paths:

   ```
   /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
     /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/insert_tmem_semaphore.mlir
   ```

5. Do not run pytest unless the user explicitly asks for pytest.

6. Known current lit baseline failures are:

   - `TRITON :: Conversion/tritongpu_to_llvm_blackwell.mlir`
   - `TRITON :: NVWS/MetaAutoWS/blackwell_ws_data_partition.mlir`
   - `TRITON :: TLX/tlx-verifier.mlir`

   The new work must not add failures beyond this baseline.

## Existing Implementation Facts

Current TMEM pass:

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp`
- pass option: `--nvws-insert-tmem-semaphore`
- pass definition:
  `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td`

The current pass is hybrid:

- It groups allocs by `buffer.id`.
- If `buffer.id` is missing, it already synthesizes a unique per-alloc id in
  pass-side analysis.
- Groups with more than one member use `BufferAccessDag`.
- Singleton groups still use `TmemAccessDag`, `SingleTMEMSemaphore`, and
  `insertTmemSemaphoreSingle`.
- `collectSlots` derives loop/if token slot information from `TmemAccessDag`.

The new pass must remove this split. For the new design, every alloc is handled
through the same logical-buffer-id group path. A group can have one member or
many members, but the analysis and emission path is the same.

Important token distinction:

- Original input TMEM async-token chains from `ttng.tmem_alloc`,
  `ttng.tmem_load`, `ttng.tmem_store`, MMAv5 ops, `scf.for`, and `scf.if`
  must not drive ownership or semaphore scheduling.
- Newly inserted `nvws.semaphore.acquire` tokens are real SSA values. They may
  need to be threaded through `scf.for` iter_args/results/yields or `scf.if`
  results so `nvws.semaphore.buffer` and later release/acquire operations can
  use the correct token.
- Buffers must not be threaded through loop iter_args. Recreate buffer views
  from the current semaphore token using `nvws.semaphore.buffer`.

Actual semaphore op spelling is:

- `nvws.semaphore.create`
- `nvws.semaphore.acquire`
- `nvws.semaphore.buffer`
- `nvws.semaphore.release`

## Important Dump Boundaries

Use these saved dumps for complex-kernel checks:

- pre-current-TMEM-pass input:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/064-NVWSInsertTmemSemaphore.mlir`
- sanity reference for current output:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/065-anonymous-VerifyWarpSpecializationPartitions.mlir`
- pre-`NVWSInsertSemaphore` subpipeline input:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/062-NVWSInsertSemaphore.mlir`

The stored `065` file is a sanity reference, not the oracle. Equivalence should
compare freshly generated old-vs-new outputs from the same input.

## Stage 0: Baseline Guardrail

Before behavior changes, build and run the current relevant tests.

Build:

```
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Targeted lit:

```
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/insert_tmem_semaphore.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/TritonGPU/automatic-warp-specialization.mlir
```

Acceptance:

- The current baseline is recorded.
- Any failure outside the known baseline is understood before proceeding.

## Stage 1: Wiring Clone Only

Create `nvws-insert-semas` as a wiring clone of
`nvws-insert-tmem-semaphore`.

This stage is allowed to use the old token-based implementation because its
only purpose is to validate pass plumbing and test coverage. Stage 1 does not
satisfy the no-token scheduling invariant.

Implementation steps:

1. Add:

   ```
   third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp
   ```

   Initially copy `InsertTmemSemaphore.cpp`.

2. Add a pass definition in:

   ```
   third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td
   ```

   Proposed option:

   ```
   --nvws-insert-semas
   ```

3. Add registration and build wiring in the same style as
   `NVWSInsertTmemSemaphore`.

4. Do not modify:

   ```
   lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp
   ```

   in this stage.

5. Add copied lit coverage:

   ```
   test/NVWS/insert_tmem_semaphore.mlir
   -> test/NVWS/insert_semas.mlir
   ```

   and:

   ```
   test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir
   -> test/NVWS/tmem-buffer-reuse-semas.mlir
   ```

   Only change the RUN lines from `--nvws-insert-tmem-semaphore` to
   `--nvws-insert-semas`.

Acceptance:

- Build passes.
- Existing tests pass unchanged:

  ```
  test/NVWS/insert_tmem_semaphore.mlir
  test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir
  ```

- New copied tests pass:

  ```
  test/NVWS/insert_semas.mlir
  test/NVWS/tmem-buffer-reuse-semas.mlir
  ```

- No existing lit test is modified.

## Stage 2: Fresh Old-Vs-New Equivalence

Compare freshly generated outputs from the old and new pass on the same inputs.
Do not use the stored `065` dump as the oracle.

Example isolated TMEM check from the build directory:

```
./bin/triton-opt \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/meta-aws-logs/run-12may26-nvws-tmem/passes/064-NVWSInsertTmemSemaphore.mlir \
  -allow-unregistered-dialect --nvws-insert-tmem-semaphore -cse \
  > /tmp/nvws-old-tmem.mlir

./bin/triton-opt \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/meta-aws-logs/run-12may26-nvws-tmem/passes/064-NVWSInsertTmemSemaphore.mlir \
  -allow-unregistered-dialect --nvws-insert-semas -cse \
  > /tmp/nvws-new-semas.mlir

diff -u /tmp/nvws-old-tmem.mlir /tmp/nvws-new-semas.mlir
```

Example full subpipeline check:

```
./bin/triton-opt \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/meta-aws-logs/run-12may26-nvws-tmem/passes/062-NVWSInsertSemaphore.mlir \
  -allow-unregistered-dialect --nvws-insert-semaphore --nvws-insert-tmem-semaphore -cse \
  > /tmp/nvws-old-subpipeline.mlir

./bin/triton-opt \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/meta-aws-logs/run-12may26-nvws-tmem/passes/062-NVWSInsertSemaphore.mlir \
  -allow-unregistered-dialect --nvws-insert-semaphore --nvws-insert-semas -cse \
  > /tmp/nvws-new-subpipeline.mlir

diff -u /tmp/nvws-old-subpipeline.mlir /tmp/nvws-new-subpipeline.mlir
```

Acceptance:

- Old and new outputs are equivalent for `064`.
- Old and new outputs are equivalent for the `062` subpipeline.
- Existing SMEM and SSA-TMEM behavior from `NVWSInsertSemaphore` is unchanged.

## Stage 3: Replace Token-Derived Scheduling

Replace the scheduling analysis in `InsertSemas.cpp` with one uniform
alloc/access DAG path for all TMEM allocs.

This is the stage where the no-token scheduling invariant becomes mandatory.

### Logical Buffer Id Model

Every TMEM alloc has a logical buffer id:

- If the IR has `buffer.id`, use it.
- If the IR does not have `buffer.id`, assign a unique synthetic id in pass-side
  analysis.

This means every alloc is processed through:

```
logical buffer id -> group of one or more alloc members -> access DAG
```

There is no singleton scheduling special case.

### Required Deletions From The New Pass

The no-token implementation must remove these from the scheduling path:

- `TmemAccessDag` as the source of semaphore schedule.
- `SingleTMEMSemaphore`.
- `insertTmemSemaphoreSingle`.
- `collectSlots` as a scheduling dependency.
- loop/if slot maps derived from token positions.

This does not mean "no semaphore token SSA." It means original TMEM async-token
use-def chains do not decide the schedule. New semaphore tokens are created by
the access-DAG-driven state machine and are wired normally where SSA requires
them.

### Access Collection

For each logical buffer-id group:

1. Seed an alias map from each original alloc result:

   ```
   Value -> {bufferId, memberIdx, viewChain}
   ```

   Initial `viewChain` is empty.

2. Assign `memberIdx` by deterministic group order.

3. Propagate aliases through concrete pure memdesc view/forwarding ops used by
   the IR. The initial supported set should include:

   - `ttg.memdesc_trans`
   - `ttg.memdesc_reinterpret`
   - `ttg.memdesc_index`
   - `ttg.memdesc_subslice`
   - `ttg.memdesc_reshape`

4. Record enough data in `viewChain` to rebuild each supported forwarding op
   from an `nvws.semaphore.buffer` result:

   - op kind
   - exact result type
   - attributes
   - location if useful for diagnostics
   - all non-source operands needed to recreate the op, such as index operands
     for `ttg.memdesc_index`
   - static offsets or shape metadata needed by subslice, reshape, or
     reinterpret operations

5. If a TMEM memdesc value with tracked alias metadata flows through an
   unsupported memdesc forwarding op before reaching a terminal access, emit a
   hard diagnostic. Do not silently drop the alias and miss the access.

6. Recognize terminal TMEM access events:

   - `ttng.tmem_load`
   - `ttng.tmem_store`
   - sourceful `ttng.tmem_alloc` as an implicit store event
   - all TMEM operands of MMAv5 ops

The MMAv5 rule is intentionally all TMEM operands, not only accumulator
operands. Accumulator operands may still need special handling for async
payload or cleanup, but access discovery must scan all operands.

### Access Event Sequence

For each logical buffer-id group, build a deterministic ordered access-event
sequence from program order and structured control flow:

- function/block sequence
- `scf.for`
- `scf.if`

For each access event, record:

- terminal operation
- event kind:
  - load
  - store
  - MMAv5 operand use
  - sourceful alloc implicit store
- effective owner domain:
  - `root/external`
  - `(wsTag, partitionId)`
- member index or indexes
- alias/view chain needed to rewrite the operand
- stage/cluster at the access site
- async payload kind needed for `nvws.semaphore.release`

This access-event sequence replaces `TmemAccessDag::collectPartitionsVec()` as
the source for:

- ownership transitions
- producer/consumer pattern recognition
- root/external ownership detection
- loop/if reconciliation decisions

### General Ownership Model

Effective owner domains are:

- `root/external`: sourceful implicit stores or terminal accesses outside a
  warp-specialized partitioned region, or accesses without a partition owner.
- `(wsTag, partitionId)`: terminal accesses owned by a warp-specialized
  partition.

`root/external` is distinct from every `(wsTag, partitionId)` owner.

The new access-DAG path must not preserve the old two-owner limitation. The
semaphore schedule is derived by walking the ordered access-event sequence.
Whenever adjacent accesses for the same logical buffer group have different
owner domains, insert an ownership transfer:

```text
owner A access
release by owner A
acquire by owner B
owner B access
```

This applies to any number of owner domains. For example:

```text
A -> B -> C -> A
```

is handled as three ownership transfers:

```text
A releases, B acquires
B releases, C acquires
C releases, A acquires
```

Do not keep the old `partitions.size() <= 2` restriction. The implementation
must support N owner domains as long as the access sequence is representable in
structured program order.

Also remove residual two-owner reconciliation helpers from the new scheduling
path. Do not carry over logic that assumes "some other partition" is enough,
including:

- `pickOtherPartition`
- two-owner `closeStateBefore` / `closeStateAfter` behavior
- bounded phase reconciliation that only toggles to an arbitrary non-current
  owner

Replace this with access-event-state-driven transitions to the exact required
target owner and phase. For loop/if reconciliation, the target is the owner and
phase required by the next access-event state, not an arbitrary owner different
from the current one. If the exact target owner/phase cannot be represented
safely through structured placement, emit a hard diagnostic. Do not fall back to
token-chain scheduling or arbitrary two-owner selection.

### Semaphore State Model

Use one logical semaphore state machine per logical buffer-id group, not one
hard-coded two-owner path.

The state machine tracks:

- current owner domain
- current semaphore phase/token
- current backing buffer view cache
- async payload for the current owner
- stage/cluster metadata for each owner as needed

On ownership transition:

1. Emit `nvws.semaphore.release` in the previous owner domain after the
   previous owner's last access.

2. Emit `nvws.semaphore.acquire` in the new owner domain before the new owner's
   first access.

3. Recreate `nvws.semaphore.buffer` views from the new acquire token.

4. Retarget the new owner's access to the appropriate buffer/view.

### Rewriting Strategy

First compute the effective owners of each logical buffer-id group.

No-op rule:

- If a live logical buffer group has exactly one effective owner domain, leave
  it unchanged and emit no `nvws.semaphore.*`.
- If a group has both `root/external` and any partition-owned access, it has
  more than one effective owner and requires semaphore insertion.
- If a group has accesses in multiple partition owner domains, it requires
  semaphore insertion.
- Dead or zero-access TMEM allocs should follow the existing unused-alloc
  cleanup behavior, not semaphore insertion.
- For no-op groups, do not erase original allocs, do not rewrite access
  operands, and leave any existing structural async-token values unchanged.

This preserves cases such as `same_owner_alias_no_semaphore` in
`test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir`.

For each ownership transition:

1. Emit `nvws.semaphore.release` for the previous owner when needed.

2. Emit `nvws.semaphore.acquire` for the new owner.

3. Emit `nvws.semaphore.buffer`.

4. Pick the semaphore-buffer result for each tracked member/op operand in the
   event. `ttng.tmem_load` and `ttng.tmem_store` usually retarget one member;
   MMAv5 events may touch multiple members from the same logical buffer group;
   sourceful alloc implicit-store events remain single-member.

5. Rebuild the recorded `viewChain` for each retargeted operand from the
   selected semaphore-buffer result.

6. Retarget every tracked terminal op operand in the event to the rebuilt view
   or direct buffer.

7. Preserve correct:

   - `ttg.partition`
   - `loop.stage`
   - `loop.cluster`
   - async payload metadata

8. Clean up old token operands/results only after the access-driven schedule has
   been emitted.

### TMEM Semaphore Backing Buffer Policy

Stage 3 must preserve the existing `NVWSInsertTmemSemaphore` 1x-vs-2x backing
buffer behavior.

The new pass must not use `TmemAccessDag` or original TMEM async-token use-def
chains to derive ownership or producer/consumer order. Instead, derive the
ordered access-event sequence from the alloc/access DAG.

Once that access-event sequence is available, preserve the existing non-token
backing-buffer policy:

- derive producer/consumer multistage eligibility from the access-event
  sequence
- preserve the accumulator checks:
  - no accumulator read-modify-write
  - `isAccMultibufferingPossible`
  - no `tt.disallow_acc_multi_buffer`
  - `canDoubleBufferAcc`
- preserve `numTmemBlocks` accounting
- preserve `getSemaphoreMultiBufferedType(..., numStages)`

The only dependency being removed is `TmemAccessDag`; the observable 1x/2x
behavior must remain.

Example:

- A producer/consumer loop that currently emits
  `!ttg.memdesc<2x128x128xf32, ...>` must still emit `2x...` under
  `nvws-insert-semas` if the access-DAG-derived events satisfy the same
  eligibility checks.
- A case with `tt.disallow_acc_multi_buffer` that currently emits
  `!ttg.memdesc<1x128x128xf32, ...>` must still emit `1x...`.

Fresh old-vs-new comparison must not allow 1x-vs-2x differences. Such a
difference is a bug in access-DAG producer/consumer classification or in the
preserved multibuffering policy.

For new N-owner patterns that do not match the old producer/consumer
multistage-eligibility pattern, use conservative single-stage backing buffers
unless and until an access-DAG-derived multistage policy is added for them.

### Original Token Cleanup

After the access-DAG-driven semaphore schedule has been emitted for a group
with more than one effective owner, remove obsolete dependencies on original
TMEM async tokens:

- Clear old token operands/dependencies on rewritten `ttng.*` and MMAv5 ops
  when the new semaphore ordering replaces them.
- Replace old token results with `ub.poison : !ttg.async.token` when they are
  only structural leftovers needed to keep result arity/type valid.
- Rewrite `scf.for` operands, results, and yields that carried old TMEM tokens
  so the IR remains type-valid.
- Rewrite `scf.if` results/yields carrying old TMEM tokens for the same reason.
- Keep newly inserted semaphore tokens distinct from old TMEM tokens. New
  semaphore tokens may be loop-carried or if-carried when required by SSA and
  lowering.

### Control Flow

For `scf.for` and `scf.if`, process ownership state through the access DAG, not
through input TMEM token chains.

For `scf.for`:

- determine body entry owner from first access
- determine body exit owner from last access
- reconcile loop-body exit state with the next-iteration entry state when the
  backedge needs a different owner for the next iteration
- do not thread buffers through iter_args
- thread newly inserted semaphore tokens through iter_args/results/yields when
  needed so later `nvws.semaphore.buffer` or `nvws.semaphore.release` ops use
  the token from the correct iteration
- do not derive this threading from original input TMEM token chains; derive it
  from the access-DAG ownership state

For `scf.if`:

- process both branches through the same state machine
- reconcile branch exits to a common post-if state
- allow newly inserted semaphore tokens to be returned by the `scf.if` when the
  post-if state needs a token produced in a branch
- do not thread buffers through the `scf.if`
- preserve stage/cluster placement for inserted ops in each branch

If an ownership pattern cannot be represented safely through structured control
flow, emit a hard diagnostic. Do not fall back to token-chain scheduling.

Acceptance:

- Existing lit tests pass unchanged.
- New copied lit tests pass.
- Existing old-compatible cases still match `nvws-insert-tmem-semaphore`,
  including existing 1x/2x TMEM semaphore backing-buffer behavior.
- Fresh old-vs-new comparisons must not allow 1x-vs-2x differences for
  old-compatible two-owner cases.
- Missing-`buffer.id` singleton cases from `insert_tmem_semaphore.mlir` are
  still handled by synthetic unique logical ids.
- Grouped `buffer.id` cases from
  `tmem-buffer-reuse-semaphore-insertion.mlir` still pass.
- Same-owner groups still emit no `nvws.semaphore.*`, do not erase original
  allocs, and are left unchanged.
- Obsolete original TMEM async-token dependencies are removed or poisoned where
  needed, while new semaphore tokens are wired independently.
- New tests cover at least one N-owner ownership sequence, for example:
  partition 0 writes, partition 1 reads, partition 2 writes, partition 0 reads.
- The N-owner test checks that `nvws.semaphore.release` /
  `nvws.semaphore.acquire` are inserted at every ownership change.

## Stage 4: Switch Automatic WS

Only after Stage 3 passes, update:

```
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp
```

Replace:

```
createNVWSInsertTmemSemaphore()
```

with:

```
createNVWSInsertSemas()
```

Do not delete the old pass yet.

Required tests:

```
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/TritonGPU/automatic-warp-specialization.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/insert_tmem_semaphore.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/insert_semas.mlir \
  /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/test/NVWS/tmem-buffer-reuse-semas.mlir
```

Acceptance:

- Build passes.
- `automatic-warp-specialization.mlir` passes.
- Existing lit tests pass unchanged.
- New lit tests pass.

If `automatic-warp-specialization.mlir` fails:

1. Report the exact failing check or verifier error.
2. Root-cause the implementation behavior.
3. Explain whether the new IR is intended and correct.
4. If an existing lit edit is needed, stop and ask the user for authorization
   before changing the test.

## Stage 5: Later Work: `insert-allocs` And Local/SMEM

Only after the TMEM-only replacement is proven in automatic WS:

1. Split current `NVWSInsertSemaphore` into alloc materialization and semaphore
   insertion.

2. Add `insert-allocs` to materialize backing allocs for SSA communication
   values.

3. Run memory planning after `insert-allocs`.

4. Extend `insert-semas` to local/SMEM allocs by adding local alloc/access
   recognition to the same logical-buffer-id analysis.

5. Keep rank-1 floating-point SSA-TMEM channels for alpha/m/l under the
   existing `NVWS_USE_SSA_TMEM=1` policy.

This stage must not change the already-proven TMEM behavior.

## Stage 6: Later Work: Retire Old Standalone Passes

Only after the unified path is proven:

1. Decide whether old standalone passes still have debugging value.

2. If not, delete or retire:

   - `NVWSInsertTmemSemaphore`
   - `NVWSInsertSemaphore`

3. Make deletion a separate cleanup commit.

4. Add focused final lit coverage for:

   - TMEM-only replacement behavior
   - local/SMEM semaphore insertion
   - SSA-TMEM alpha/m/l rank-1 floating-point channels
   - `loop.stage` / `loop.cluster` preservation
   - `scf.for` ownership reconciliation
   - `scf.if` branch reconciliation

## Existing Lit Test Change Policy

Allowed without additional authorization:

- Add `test/NVWS/insert_semas.mlir`.
- Add `test/NVWS/tmem-buffer-reuse-semas.mlir`.
- Add new focused lit tests for new behavior.

Not allowed without explicit user authorization:

- Editing existing lit checks.
- Weakening existing FileCheck patterns.
- Deleting existing lit tests.
- Marking existing tests unsupported.
- Marking existing tests expected-fail.

If an existing lit test appears to require a change, the report must include:

1. Exact failing test.
2. Exact failing check or verifier error.
3. Implementation root cause.
4. Why the new IR is correct.
5. Why the old check is invalid.
6. Minimal proposed test edit.

## Recommended Commit Staging

1. Add `NVWSInsertSemas` as a wiring clone plus copied lit tests.

2. Add documented old-vs-new equivalence commands for the fused-attention dump.

3. Replace token-derived scheduling with the uniform logical-buffer-id
   alloc/access DAG while staying TMEM-only.

4. Switch automatic WS from `createNVWSInsertTmemSemaphore()` to
   `createNVWSInsertSemas()`.

5. Later: add `insert-allocs` and extend `insert-semas` to local/SMEM.

6. Later: retire old standalone passes.
