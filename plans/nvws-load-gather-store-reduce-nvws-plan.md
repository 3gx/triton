# Meta+NVWS descriptor operations plan

## Goal

Make `TRITON_NVWS_USE_META=1` use all four NVWS descriptor operations through
`NVWSInsertSemas` and lower them in `NVWSLowerSemaphore`:

```text
nvws.descriptor_load
nvws.descriptor_gather
nvws.descriptor_store
nvws.descriptor_reduce
```

The work has three independently authorized stages:

0. create dedicated TT-to-NVWS descriptor conversion passes;
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

---

## Stage 0: Dedicated conversion passes

### Deliverable

Create two standalone production passes:

```text
nvgpu-convert-descriptor-loads-to-nvws
nvgpu-convert-descriptor-stores-to-nvws
```

Each pass does one thing only:

- walk each `tt.func`;
- call its existing conversion helper;
- propagate failure through the pass manager;
- rely on the helper's survivor check to reject unconverted supported TT ops.

The passes do not perform buffer allocation, memory planning, partitioning,
InsertSemas, or LowerSemaphore work.

### Implementation

- Add pass definitions and dependent dialects in Hopper `Passes.td`.
- Add thin wrappers beside the helpers in `WSLowerMem.cpp`.
- Expose normal pass factories so `AutomaticWarpSpecialization` can add them to
  its nested Meta pass manager.
- Replace the `ttg.test_nvws_tma_store_conversion` test-only dispatch with the
  dedicated store/reduce pass in its conversion tests.
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

Use existing conversion test files where possible, especially Hopper
`ws_tma_store_lowering.mlir` and existing descriptor-load conversion coverage.

Agent gate:

- build `triton` and `triton-opt`;
- run the focused conversion tests;
- run affected Hopper WarpSpecialization lit tests;
- run `git diff --check`.

Stage 0 does not run the runtime gate because neither new pass is wired into a
production pipeline yet. The mandatory runtime gate begins in Stage 1.

### Stage 0 stop

Report the two pass names, direct input/output IR, tests, and uncommitted diff.
Wait for user testing and a separate Stage 0 commit request. After committing,
stop. Stage 1 requires a new explicit request.

---

## Stage 1: Wire and verify NVWS load/gather

### Prerequisite

Stage 0 is explicitly committed and the user separately requests Stage 1.

### Deliverable

Meta+NVWS has an explicit, non-duplicated load/gather conversion boundary:

```text
default Meta allocation:
  dedicated load/gather conversion pass
  -> WSBufferAllocation

TRITON_NVWS_USE_META_NVWS_ALLOCAS=1:
  existing MetaToNVWSConvert
  -> existing NVWSInsertAllocas load/gather conversion
```

Both routes then use the already-implemented path:

```text
NVWS load/gather
-> InsertSemas release [tma_load]
-> LowerSemaphore createTMALoad/createTMAGather
-> TTNG TMA load/gather
```

### Implementation

- In the default Meta allocation branch of
  `AutomaticWarpSpecialization.cpp`, add the Stage 0 load/gather pass
  immediately before `NVGPUTestWSBufferAllocation`.
- Make `NVGPUTestWSBufferAllocationPass` perform buffer allocation only.
- Make `NVGPUTestWSMemoryPlannerPass` perform memory planning only; remove its
  repeated conversion call.
- Update standalone buffer-allocation/planner test pipelines that contain raw
  TT load/gather to run the dedicated conversion pass explicitly.
- Leave the optional `NVWSInsertAllocas` conversion implementation unchanged.
- Do not change ACCESS-DAG, `AsyncOp::TMALoad`, AssignStagePhase, or
  LowerSemaphore in Stage 1. If integration exposes a failure there, follow
  the verified-blocker protocol and stop pending explicit authorization.
- Do not change store/reduce behavior.

### Tests and acceptance

Required matrix:

| Allocation route | Load | Gather |
|---|---:|---:|
| Dedicated pass + Meta buffer allocation | pass | pass |
| Existing NVWSInsertAllocas route | pass | pass |

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

Report the four matrix results, the three IR boundaries, tests, and uncommitted
diff. Wait for user testing and a separate Stage 1 commit request. After
committing, stop. Stage 2 requires a new explicit request.

---

## Stage 2: Add NVWS store/reduce

### Prerequisite

Stage 1 is explicitly committed and the user separately requests Stage 2.

### Deliverable

Both Meta+NVWS allocation routes implement:

```text
tt.descriptor_store/reduce
-> dedicated Stage 0 store/reduce conversion pass
-> nvws.descriptor_store/reduce
-> InsertSemas release [tma_store]
-> LowerSemaphore TTNG store/reduce issue
-> TMAStoreTokenWait
-> exactly one EMPTY-barrier arrival after completion
```

All four descriptor operation kinds are NVWS operations at the InsertSemas
boundary. Explicit `early_tma_store_lowering=True` remains the legacy A/B
route.

### Implementation

1. Add the Stage 0 store/reduce pass after Meta task propagation and before the
   allocation branch.
2. During development, use explicit `early_tma_store_lowering=False`; before
   handoff, make the default Meta+NVWS route use NVWS store/reduce and retain
   explicit `True` for legacy behavior.
3. Verify that default `doBufferAllocation` preserves the already-supported
   canonical source-free staging shape unchanged. In the optional InsertAllocas
   route, add only the handling required to prevent a second staging buffer.
4. Extend MetaToNVWS root/external preconverted-descriptor recognition from
   load/gather to store/reduce so their partition and WS metadata is preserved.
   Reuse the abstract annotation already invoked by the Meta+NVWS prefix, add
   `doValidateAbstractTMAStoreAnnotations` to its validation wrapper, and teach
   ordinary-store epilogue packing about the abstract form. Keep legacy
   workarounds for explicit compatibility.
5. Add `AsyncOp::TMAStore`; classify store/reduce source as an ACCESS-DAG read;
   emit `release [tma_store]`; count one pending arrival; classify both as reads
   in AssignStagePhase.
6. Add `createTMAStore` and `createTMAReduce` in LowerSemaphore.
7. Make LowerSemaphore associate each `tma_store` release with every descriptor
   operation whose SMEM-read completion it represents. Derive the association
   and token strategy from the actual emitted semaphore IR; do not assume one
   token or one CFG shape without a focused test.
8. Emit the TTNG issues and ensure EMPTY is released exactly once, only after
   every relevant SMEM read is complete. Suppress any duplicate ordinary
   arrival.
9. Audit whether real integration IR combines several store/reduce operations
   under one release. If it does, support and test that exact shape before the
   default cutover; otherwise diagnose it explicitly rather than inventing an
   unverified combined-token algorithm.
10. Assert no NVWS store/reduce survives LowerSemaphore and update the relevant
    `sema-docs` documents.

Do not add a LowerSemaphore proxy fence unless codegen comparison proves one is
missing; the later NVIDIA fence pass already handles TTNG store/reduce issues.

### Tests and acceptance

Required matrix:

| Allocation route | Store | Reduce |
|---|---:|---:|
| Default Meta allocation | pass | pass |
| `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` | pass | pass |

Required cases:

- single store and single reduce;
- alias/subview staging and packed epilogue;
- root/external ownership;
- pending-count and `arrive_count` behavior;
- any combined-release shape observed in the integrated test IR;
- no duplicate staging allocation;
- no extra LowerSemaphore-specific proxy fence, and final fence placement/count
  matching the explicit legacy route;
- explicit legacy compatibility.

Required boundaries:

```text
after conversion:          nvws.descriptor_store/reduce
after InsertSemas:         release [tma_store]
after LowerSemaphore:      TTNG issue + barrier-carrying token wait
after token-wait lowering: single-op case has one store wait + one arrival
final AWS output:          no TT/NVWS store/reduce survivor
```

If integration emits a combined release, its test must instead prove that all
associated SMEM reads complete before exactly one EMPTY release; it must not
assume the single-op wait/token shape.

Agent gate:

- build first;
- run focused conversion, allocation, MetaToNVWS, InsertSemas, pending-count,
  AssignStagePhase, LowerSemaphore, packing, observed association-shape, and
  diagnostic lit tests;
- run complete `test/NVWS` and `test/Hopper/WarpSpecialization`;
- run relevant TritonNvidiaGPU subtile tests;
- compare new default with explicit legacy at MemoryPlanner, LowerSemaphore,
  final TTGIR, waits, arrivals, and normalized PTX;
- run the complete authorized runtime gate defined below: the six-file command
  and all three fused-attention commands;
- run `git diff --check`.

### Stage 2 stop

Report the four matrix results, all IR boundaries, equivalence evidence, tests,
and the complete uncommitted diff. The user tests both allocation modes and
runtime kernels. Commit only after a separate Stage 2 commit request. After
committing, stop. Removing legacy compatibility requires another explicit
request and is outside this plan.

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
