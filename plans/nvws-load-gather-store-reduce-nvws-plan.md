# Meta+NVWS descriptor operations plan

## Goal

Make `TRITON_NVWS_USE_META=1`, with
`TRITON_NVWS_USE_META_NVWS_ALLOCAS=0`, use all four NVWS descriptor operations
through `NVWSInsertSemas`:

```text
nvws.descriptor_load
nvws.descriptor_gather
nvws.descriptor_store
nvws.descriptor_reduce
```

The work has three independently authorized stages:

0. create dedicated conversion/materialization passes that wrap existing Meta
   helpers;
1. wire and verify load/gather;
2. add store/reduce.

## Mandatory workflow

For each stage:

1. The user explicitly asks to implement that stage.
2. The agent implements only that stage, builds, and runs its tests.
3. The agent reports the uncommitted diff and exact results, then stops.
4. The user tests it and requests same-stage fixes if needed.
5. The agent commits only after a separate explicit commit request.
6. After committing, the agent stops.
7. The next stage starts only after another explicit request naming it.

Implementation authorization is not commit authorization. Commit
authorization is not authorization for the next stage. The user has already
authorized the exact runtime gates below for Stages 1 and 2; no other pytest
scope is authorized. Existing untracked artifacts are never staged.

## No-overreach and verified-blocker protocol

The active stage defines the complete mutation scope. The agent must not fix,
refactor, optimize, or clean up anything outside that scope merely because it
was discovered while implementing or testing the stage.

Unless the active stage explicitly names a removal or replacement, all work is
additive:

- do not delete, move, rename, merge, or rewrite existing tests;
- do not remove existing `RUN` lines, checks, negative cases, or coverage;
- do not weaken checks, add an XFAIL, or change expected behavior to make a new
  test pass;
- do not remove or repurpose existing test-only dispatches, pass wrappers,
  compatibility paths, flags, or production call sites;
- adding a new pass or API does not authorize migrating existing callers or
  cleaning up the older path in the same stage.

New pass coverage must be added alongside existing coverage. If testing the
new feature appears to require restructuring or deleting existing coverage,
report the conflict and stop for explicit authorization instead.

If a suspected out-of-scope issue appears to block the active stage:

1. Do not modify the suspected out-of-scope code.
2. Reproduce the failure with the smallest read-only diagnostic or test run.
3. Assign an independent subagent to verify the reproduction and root cause.
4. Treat it as a blocker only when both the primary investigation and the
   subagent confirm that it is real, reproducible, and prevents the current
   stage's stated acceptance criteria.
5. An assumption, guess, possible future risk, unrelated failure, cleanup
   opportunity, or desirable optimization is not a blocker.
6. If verified, report the evidence, root cause, affected scope, and a minimal
   proposed solution. Then stop.
7. Do not implement the proposed blocker solution, expand the current stage,
   start another stage, or commit anything without explicit user authorization.

If the issue cannot be independently verified, do not present it as a blocker
and do not make speculative changes for it.

## Verified starting point

- `doConvertDescriptorLoadsToNVWS` and
  `doConvertDescriptorStoresToNVWS` are helper functions, not passes.
- The Meta buffer-allocation wrapper calls load conversion before allocation.
- The Meta memory-planner wrapper calls load conversion again idempotently.
- `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` already creates
  `nvws.descriptor_load/gather` inside `NVWSInsertAllocas`.
- ACCESS-DAG, `AsyncOp::TMALoad`, `AssignStagePhase`, and LowerSemaphore already
  support NVWS load/gather. These are not reimplemented by this plan.
- Meta+NVWS still early-lowers store/reduce to TTNG before automatic warp
  specialization.
- Commit `1ce1b72ec0` already defines and verifies NVWS store/reduce and provides
  `doConvertDescriptorStoresToNVWS`.

The optional `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` route is out of scope for
this plan and remains unchanged.

---

## Stage 0: Dedicated Meta helper passes

### Deliverable

Create three standalone production passes:

```text
nvgpu-convert-descriptor-loads-to-nvws
nvgpu-convert-descriptor-stores-to-nvws
nvgpu-materialize-nvws-descriptor-stores
```

The two conversion passes each:

- walk each `tt.func`;
- call its existing conversion helper;
- propagate failure through the pass manager;
- rely on the helper's survivor check to reject unconverted supported TT ops.

The materialization pass walks each `tt.func` and calls the existing Meta
helper:

```cpp
doMaterializeAndPlaceTMAStoreWaits(func, /*enableRotation=*/false)
```

It converts:

```text
nvws.descriptor_store/reduce
-> TTNG issue/token
-> TMAStoreTokenWait
```

It does not compute the final TMA wait `pendings` value.

The passes do not perform buffer allocation, memory planning, partitioning,
InsertSemas, or LowerSemaphore work.

### Implementation

- Add pass definitions and dependent dialects in Hopper `Passes.td`.
- Add thin conversion wrappers beside the helpers in `WSLowerMem.cpp`.
- Add a thin materialization wrapper beside the existing Meta store-wait
  materialization code in `WSTMAStoreLowering.cpp`.
- Expose normal pass factories so `AutomaticWarpSpecialization` can add them to
  its nested Meta pass manager.
- Keep `ttg.test_nvws_tma_store_conversion`, its existing wrapper behavior,
  and every existing conversion test unchanged.
- Add separate focused tests for the three new pass entry points; do not move or
  delete existing test cases.
- Do not change any production pipeline in Stage 0.

### Tests

Direct pass tests must cover:

- load;
- gather, including i16-offset extension;
- ordinary store;
- descriptor reduce;
- preservation of task, partition, stage, cluster, descriptor coordinates,
  reduction kind, and source/destination buffer type;
- explicit rejection of legacy `tt.descriptor_store reduce_kind != none`;
- no raw load/gather surviving the load pass;
- no raw store/reduce surviving the store pass;
- no allocation/planner side effects beyond those intentionally created by
  the conversion helper.
- direct NVWS store and reduce materialization to TTNG issue plus
  `TMAStoreTokenWait`;
- rotation remaining disabled in the materialization pass;
- no final `ttng.async_tma_store_wait` yet, proving final `pendings` lowering is
  still owned by the existing later pass.

Add separate focused pass tests. Existing conversion test files, including
`ws_tma_store_lowering.mlir`, remain unchanged; input patterns may be copied
into the new focused tests, but existing cases are not moved.

Agent gate:

- build `triton` and `triton-opt`;
- run the focused conversion tests;
- run affected Hopper WarpSpecialization lit tests;
- run `git diff --check`.

Stage 0 does not run the runtime gate because neither new pass is wired into a
production pipeline yet. The mandatory runtime gate begins in Stage 1.

### Stage 0 stop

Report the three pass names, direct input/output IR, tests, and uncommitted diff.
Wait for user testing and a separate Stage 0 commit request. After committing,
stop. Stage 1 requires a new explicit request.

---

## Stage 1: Wire and verify NVWS load/gather

### Prerequisite

Stage 0 is explicitly committed and the user separately requests Stage 1.

### Deliverable

Meta+NVWS explicitly invokes the Stage 0 load/gather conversion pass on the
default Meta allocation route. Each arrow below names exactly one pass:

```text
tt.descriptor_load/gather
  -- nvgpu-convert-descriptor-loads-to-nvws -->
nvws.descriptor_load/gather
  -- nvgpu-test-ws-buffer-allocation -->
planned nvws.descriptor_load/gather
  -- nvws-insert-semas -->
nvws.descriptor_load/gather + semaphore.release [tma_load]
  -- nvws-lower-semaphore -->
TTNG async TMA load/gather
```

### Implementation

- In the default Meta allocation branch of
  `AutomaticWarpSpecialization.cpp`, add
  `nvgpu-convert-descriptor-loads-to-nvws` immediately before
  `NVGPUTestWSBufferAllocation`.
- Keep the existing conversion calls inside `NVGPUTestWSBufferAllocationPass`
  and `NVGPUTestWSMemoryPlannerPass` unchanged. They are idempotent after the
  dedicated pass; removing or refactoring them is outside Stage 1.
- Keep all standalone buffer-allocation/planner test pipelines unchanged.
- Do not modify or test the optional `NVWSInsertAllocas` allocation route in
  this stage.
- Do not change ACCESS-DAG, `AsyncOp::TMALoad`, AssignStagePhase, or
  LowerSemaphore in Stage 1. If integration exposes a failure there, follow
  the verified-blocker protocol and stop pending explicit authorization.
- Do not change store/reduce behavior.

### Tests and acceptance

Required matrix:

| Allocation route | Load | Gather |
|---|---:|---:|
| Dedicated pass + default Meta buffer allocation | pass | pass |

Tests must show:

```text
after allocation:      nvws.descriptor_load/gather exists
after InsertSemas:     release [tma_load] exists
after LowerSemaphore:  TTNG TMA load/gather exists; NVWS op is gone
```

Run focused coverage in existing files for conversion, InsertAllocas,
MetaToNVWS ownership, InsertSemas, LowerSemaphore, and
`meta_nvws_automatic_warp_specialization.mlir`, then complete `test/NVWS` and
`git diff --check`. After lit passes, run the complete authorized runtime gate
defined below: the six-file command and all three fused-attention commands.

### Stage 1 stop

Report the two matrix results, the three IR boundaries, tests, and uncommitted
diff. Wait for user testing and a separate Stage 1 commit request. After
committing, stop. Stage 2 requires a new explicit request.

---

## Stage 2: Add NVWS store/reduce

### Prerequisite

Stage 1 is explicitly committed and the user separately requests Stage 2.

### Deliverable

The default Meta+NVWS allocation route implements the following pipeline. Each
arrow names exactly one pass, and every Stage 0 pass is referenced by its exact
name:

```text
tt.descriptor_store/reduce
  -- nvgpu-convert-descriptor-stores-to-nvws -->
nvws.descriptor_store/reduce
  -- nvgpu-test-ws-buffer-allocation -->
planned nvws.descriptor_store/reduce
  -- nvws-insert-semas -->
nvws.descriptor_store/reduce + semaphore.release [none]
  -- nvgpu-materialize-nvws-descriptor-stores -->
TTNG store/reduce issue + TMAStoreTokenWait + semaphore.release [none]
  -- nvws-lower-semaphore -->
TTNG store/reduce issue + TMAStoreTokenWait + EMPTY ArriveBarrier
  -- nvgpu-tma-store-token-wait-lowering -->
TTNG store/reduce issue + TMAStoreWait {pendings=N} + EMPTY ArriveBarrier
```

All four descriptor operation kinds are NVWS operations at the InsertSemas
boundary. Explicit `early_tma_store_lowering=True` remains the legacy A/B
route.

### Implementation

1. Add `nvgpu-convert-descriptor-stores-to-nvws` immediately after
   `nvgpu-convert-descriptor-loads-to-nvws`, before the default Meta buffer
   allocation pass.
2. During development, use explicit `early_tma_store_lowering=False`; before
   handoff, make the default Meta+NVWS route use NVWS store/reduce and retain
   explicit `True` for legacy behavior.
3. Verify that default `doBufferAllocation` preserves the already-supported
   canonical source-free staging shape unchanged. Do not modify the optional
   InsertAllocas route.
4. Extend MetaToNVWS root/external preconverted-descriptor recognition from
   load/gather to store/reduce so their partition and WS metadata is preserved.
   Reuse the abstract annotation already invoked by the Meta+NVWS prefix, add
   `doValidateAbstractTMAStoreAnnotations` to its validation wrapper, and teach
   ordinary-store epilogue packing about the abstract form. Keep legacy
   workarounds for explicit compatibility.
5. Classify store/reduce source as an ACCESS-DAG read and classify both as reads
   in AssignStagePhase. Keep the existing generic `AsyncOp::NONE` release
   protocol; do not add `AsyncOp::TMAStore`.
6. Invoke `nvgpu-materialize-nvws-descriptor-stores` immediately after
   `nvws-insert-semas` and before `nvws-lower-semaphore`; its wrapper calls the
   Meta helper with rotation disabled.
7. Reuse LowerSemaphore's existing generic `[none]` release lowering unchanged.
   Do not add store-token association, barrier attachment, wait movement, or
   store-specific lowering to LowerSemaphore.
8. Reuse the existing global `nvgpu-tma-store-token-wait-lowering` pass
   unchanged. It computes TMA wait `pendings=N` from issue/token/wait ordering
   and replaces the token wait with
   `ttng.async_tma_store_wait {pendings=N}`.
9. Keep semaphore `pending_count`/`arrive_count` independent from TMA wait
   `pendings`. Do not copy or derive a value between those count domains.
10. Assert no NVWS store/reduce survives the materialization pass and update the
    relevant `sema-docs` documents.

All downstream edits in Stage 2 are limited to recognizing
`nvws.descriptor_store` and `nvws.descriptor_reduce` as reads and inserting the
existing materialization pass. Existing load/gather, generic semaphore,
LowerSemaphore, final token-wait lowering, MMA, TMEM, compatibility, and
unrelated planner behavior must remain unchanged.

Do not improvise a custom store-token protocol, `AsyncOp::TMAStore`, custom
LowerSemaphore store lowering, or a new pending-count algorithm. If the fixed
flow above fails an acceptance test, follow the verified-blocker protocol:
independently verify the failure and root cause, report minimal options, and
stop for explicit authorization.

Running the existing Meta materialization and token-wait-lowering logic
verbatim is the Stage 2 contract. Stage 2 does not add a new scheduler-ordering
analysis, verifier, wait-placement algorithm, or compensating synchronization.

### Tests and acceptance

Required matrix:

| Allocation route | Store | Reduce |
|---|---:|---:|
| Default Meta allocation | pass | pass |

Required cases:

- single store and single reduce;
- alias/subview staging and packed epilogue;
- root/external ownership;
- semaphore `pending_count`/`arrive_count` unchanged from the generic `[none]`
  protocol;
- no extra LowerSemaphore-specific logic or proxy fence;
- explicit legacy compatibility.

Required boundaries:

```text
after conversion:          nvws.descriptor_store/reduce
after InsertSemas:         nvws store/reduce followed by release [none]
after materialization:     TTNG issue + TMAStoreTokenWait; no NVWS survivor
after token-wait lowering: TMAStoreWait {pendings=N}; no token-wait survivor
final AWS output:          no TT/NVWS store/reduce survivor
```

Semaphore counts and TMA wait `pendings` remain separate existing mechanisms;
Stage 2 does not copy or derive either value from the other.

Agent gate:

- build first;
- run focused conversion, allocation, MetaToNVWS, InsertSemas,
  materialization, AssignStagePhase, LowerSemaphore, final token-wait lowering,
  packing, and diagnostic lit tests;
- run complete `test/NVWS` and `test/Hopper/WarpSpecialization`;
- run relevant TritonNvidiaGPU subtile tests;
- run existing explicit-legacy compatibility coverage without adding a new
  wait-scheduling analysis or replacement algorithm;
- run the complete authorized runtime gate defined below: the six-file command
  and all three fused-attention commands;
- run `git diff --check`.

### Stage 2 stop

Report the two matrix results, required IR boundaries, tests, and the complete
uncommitted diff. The user tests the default allocation route and runtime
kernels. Commit only after a separate Stage 2 commit request. After
committing, stop. Removing legacy compatibility requires another explicit
request and is outside this plan.

---

## Post-plan TODO: LowerSemaphore-owned materialization

This is recorded for posterity only. It is not part of Stage 0, 1, or 2 and is
not authorized by this plan.

Consider removing the future production dependency on
`nvgpu-materialize-nvws-descriptor-stores` by moving NVWS descriptor
store/reduce issue and `TMAStoreTokenWait` creation into LowerSemaphore. That
alternative would still reuse the existing Meta
`NVGPUTMAStoreTokenWaitLowering` pass unchanged for final `pendings=N`
calculation and token-wait lowering.

This alternative requires a separate investigation, plan, user authorization,
implementation, tests, and commit. No Stage 2 failure authorizes switching to
it automatically; the verified-blocker protocol applies.

---

## Build and lit commands

Build before testing every stage:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-meta-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Run lit from that directory:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/<path>.mlir
```

Complete suites:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/NVWS
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test/Hopper/WarpSpecialization
```

## Complete authorized runtime gate for Stages 1 and 2

Run all six files in one command under one 240-second timeout. Use a fresh
cache directory for every run:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-meta-01.git/
timeout --signal=TERM --kill-after=10s 240s \
  env PYTHONPATH="$PWD/python" \
      TRITON_CACHE_DIR=/tmp/nvws-descriptor-runtime-$$ \
  pytest -n24 -s --tb=short \
    python/test/unit/language/test_amd_warp_pipeline.py \
    python/test/unit/language/test_tutorial09_warp_specialization.py \
    python/test/unit/language/test_tlx_warp_specialization.py \
    python/test/unit/language/test_warp_specialization.py \
    python/test/unit/language/test_autows_addmm.py \
    python/test/unit/language/test_autows_quantized_matmul.py
```

The timeout covers the complete six-file command, not each file separately.

Also run these three exact fused-attention runtime commands in both Stage 1 and
Stage 2:

```bash
TRITON_META_WS_USE_CHANNEL_SMEM=0 TRITON_USE_META_PARTITION=1 TRITON_ALWAYS_COMPILE=1 TRITON_USE_META_WS=1 pytest -n16 python/tutorials/fused-attention-ws-device-tma.py
```

```bash
MLIR_ENABLE_DIAGNOSTICS=warnings TRITON_FP8_PROMOTE_TO_TMEM=0 NVWS_USE_SSA_TMEM=1 TRITON_ALWAYS_COMPILE=1 TRITON_NVWS_USE_META=1 pytest -n16 python/tutorials/fused-attention-ws-device-tma.py
```

```bash
MLIR_ENABLE_DIAGNOSTICS=warnings  TRITON_FP8_PROMOTE_TO_TMEM=1 NVWS_USE_SSA_TMEM=1 TRITON_ALWAYS_COMPILE=1 TRITON_NVWS_USE_META=1  python python/tutorials/fused-attention-ws-device-tma.py
```

The first command is the native-Meta reference. The second exercises the
Meta+NVWS pytest path with FP8 promotion disabled. The third directly runs the
Meta+NVWS tutorial with FP8 promotion enabled. No additional pytest or runtime
commands are authorized by this plan.

Generated dumps go under `/tmp` or `.agent-artefacts/` at repository root.
