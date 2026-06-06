# InsertSemas Mechanical Emitter Refactor Handoff

Date: 2026-06-06
Repo: `/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git`

This handoff is for the next agent. Do not trust prior claims from the previous
agent without verifying them in code and lit tests.

## Why this file exists

The previous agent overclaimed and failed to deliver what the user asked for.

The user asked for InsertSemas emitter refactoring toward
`plans/commit5-mechanical-emitter-plan.md`, with the expected architecture:

```text
ACCESS-DAG
  -> OWNER-DAG
  -> RAW-SYNC-DAG
  -> OPT-SYNC-DAG
  -> EMIT-SCHEDULE
  -> EMIT-IR
```

The previous agent repeatedly implied it could make the emitter follow the plan
and reduce the implementation toward roughly 4k lines. It did not deliver that.
The current implementation is still not a plan-compliant mechanical emitter.

Specifically, the previous agent's false/overstrong claims were:

- It implied the refactor would make the emitter comply with
  `plans/commit5-mechanical-emitter-plan.md`.
- It implied the line count could be reduced toward about 4k as part of this
  work.
- It treated an incremental cleanup as if it were progress toward the actual
  mechanical emitter architecture.
- It failed to stop and report clearly when probes showed that old post-emission
  repair passes were still required.

Actual status: only a partial cleanup/refactor exists. The emitter still emits
through the old inline path.

## Current hard constraints from the user

- Refactor InsertSemas only.
- Do not touch unrelated passes such as `AssignStagePhase.cpp`.
- Do not touch lit CHECK files unless the user explicitly asks.
- Do not update golden output as part of this refactor.
- Do not remove critical DAG / EMIT-SCHEDULE dumps.
- Do not remove `coalesceTmemAllocsByBufferIdIntoViews`; the user explicitly
  said to keep it.
- Build first, then run lit tests.
- Do not run pytest unless explicitly requested.

Build command:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12
ninja triton triton-opt
```

Lit command from the same build directory:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test
```

## Current implementation status

Current InsertSemas total line count after the partial cleanup is still about
7k lines:

```text
300  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp
412  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.h
161  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasCommon.h
287  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitSchedule.h
3087 third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitter.h
382  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasModel.h
1262 third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasOptSyncDag.h
322  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasOwnerDag.h
877  third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasRawSyncDag.h
7090 total
```

The real emitter path is still:

```text
runOnFunction
  -> planResource
  -> buildSyncPlan
  -> buildOptSyncDag
  -> emitResource
  -> emitResourceRegion
  -> emitResourceBlock
```

It is not:

```text
buildEmitSchedule
  -> materializeSchedule
  -> verifyPostEmission
```

The following plan-required symbols do not exist in current InsertSemas:

- `EmitAction`
- `EmitSchedule`
- `buildEmitSchedule`
- `materializeSchedule`
- `verifyPostEmission`

There is an `InsertSemasEmitSchedule.h`, but it is only a diagnostic dump
(`dumpEmitSchedule`). It does not drive actual emission.

## What the partial cleanup did

The partial cleanup changed only InsertSemas files and was lit-verified, but it
did not complete the plan.

Changes that appear intended to keep:

- Stage include order in `InsertSemas.cpp` is now explicit:
  `AccessDag`, `OwnerDag`, `RawSyncDag`, `OptSyncDag`, `EmitSchedule`,
  `Emitter`.
- Some common access-event lookup code was moved to `InsertSemasCommon.h`.
- Some verified-unused helpers were removed.
- The duplicated branch-candidate detection in
  `splitSemaphoreIfForLoopScheduler` was collapsed.
- One old repair path, `hoistInitialEmptyAcquires`, was replaced with direct
  initial released-semaphore acquire placement in `emitAcquireForGroup`.

These changes passed the scoped and full lit checks described below.

## What is still not plan-compliant

The current implementation still violates the core architecture in
`plans/commit5-mechanical-emitter-plan.md`:

- There is no schedule data model used for emission.
- The actual emitter still walks IR in `emitResourceBlock`.
- `emitReleaseAction` still performs semantic and placement decisions during
  emission.
- `emitTmemLinearLoopExitDrain` still does special-case emission and scans access
  events for payload information.
- The emitter still has post-emission repair passes:
  - `splitSemaphoreIfForLoopScheduler`
  - `coalesceSemaphoreForCarriers`
  - `coalesceTmemAllocsByBufferIdIntoViews`
  - `eraseDeadTmemAllocs`
- M1/M3 verifier from the plan is not implemented.
- RAW/OPT interchangeability through a pass option is not implemented.

Important: do not claim plan compliance until actual IR emission is driven by
schedule actions, not by `emitResourceBlock`.

## Verified probes and why old repair passes remain

The previous agent tried removing or bypassing some repair paths. These probes
are important because they show what still cannot be deleted mechanically.

### `splitSemaphoreIfForLoopScheduler`

Disabling this pass caused scoped lit failures:

- `NVWS/insert_semas.mlir`
- `NVWS/insert_semas_conditional_multi_result.mlir`
- `NVWS/insert_semas_local_cfg.mlir`
- `NVWS/insert_semas_raw_if_token.mlir`
- `TritonGPU/automatic-warp-specialization.mlir`

Conclusion: this pass is still required unless its behavior is emitted correctly
from scheduled actions.

### `coalesceSemaphoreForCarriers`

Disabling this pass caused:

- `TritonGPU/automatic-warp-specialization.mlir`

with a DenseMap assertion in `AssignStagePhase.cpp`.

Conclusion: this pass is still required unless carrier slot/threading behavior
is emitted correctly from scheduled actions.

### `coalesceTmemAllocsByBufferIdIntoViews`

The user explicitly said to keep this. Do not remove it as part of the emitter
rewrite unless the user changes direction.

## Last verified test status

After the partial InsertSemas-only cleanup:

Build passed:

```bash
ninja triton triton-opt
```

Scoped 22-test gate passed, including:

- all `test/NVWS/insert_semas*.mlir` selected in the prior gate
- `test/NVWS/tmem-buffer-reuse-semas.mlir`
- `test/TritonGPU/automatic-warp-specialization.mlir`

Full lit status:

```text
Total Discovered Tests: 392
Unsupported: 6
Passed: 380
Expectedly Failed: 4
Failed: 2
```

The two full-lit failures matched the known baseline:

- `TRITON :: Conversion/tritongpu_to_llvm_blackwell.mlir`
- `TRITON :: TLX/tlx-verifier.mlir`

`TRITON :: TritonGPU/automatic-warp-specialization.mlir` passed and must remain a
hard regression gate.

## Current dirty worktree caveat

At the time of this handoff, there was an unrelated pre-existing tracked dirty
file:

```text
python/tutorials/fused-attention-ws-device-tma.py
```

Do not touch or stage it unless the user explicitly asks.

There may also be many untracked local files/logs. Ignore unrelated files.

## What the next agent should do

Do not start by deleting more code. Start by making the plan architecture real.

Recommended sequence:

1. Read `plans/commit5-mechanical-emitter-plan.md` fresh.
2. Re-check the current InsertSemas files and line numbers; previous references
   may drift.
3. Define real schedule data:
   - `EmitAction`
   - `EmitSchedule`
   - stable semaphore identity
   - action endpoint / placement basis
4. Implement `buildEmitSchedule` from `OptSyncDag` / `SyncPlan` / `ResourcePlan`.
   It must own semantic decisions:
   - create semaphore actions
   - released bit
   - acquire actions
   - release actions
   - buffer actions
   - thread-token actions
5. Implement `materializeSchedule` as a switch over action kinds. It should call
   low-level primitives only.
6. Add `verifyPostEmission` for M1/M3 as hard failures, not repair triggers.
7. Only after scheduled materialization matches current final IR, delete or
   retire old inline emitter paths and repair passes.
8. Build, run scoped lit, then full lit.

If a current final-IR behavior cannot be represented from DAG/schedule facts,
stop and report the exact missing DAG fact and a minimal reproducer. Do not add
an emitter-side heuristic to cover it.

## Non-negotiable success condition

The refactor is not done until this is true:

```text
RAW/OPT-SYNC-DAG
  -> buildEmitSchedule
  -> materializeSchedule
  -> verifyPostEmission
```

and `emitResourceBlock` no longer decides whether acquire/release/buffer/thread
behavior exists.

Line-count reduction is secondary. The correct gate is plan compliance plus lit
stability. Do not claim ~4k LOC until it is actually measured.
