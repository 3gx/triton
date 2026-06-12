# Plan: port insert-semas to the nvws-semaphore baseline (perf-isolation experiment)

Date: 11jun26 (rev 2 — simplified, pass list verified against the
actual trees). Status: PLAN — awaiting user approval.

## Repos

| repo | branch | HEAD | role |
|---|---|---|---|
| `triton-01.git` | `egx/nvws-semaphore` | `34245cc5ef` | baseline perf reference — untouched |
| `triton-03.git` | `egx/nvws-semaphore-insert-semas` | `34245cc5ef` (same commit) | port workspace — all edits here |
| `triton-solid-01.git` | current | — | source of ported files; perf reference — untouched |

## Build & test protocol (from `triton-03.git/AGENTS.md`)

- build:
  `cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-03.git/build/cmake.linux-x86_64-cpython-3.12/ && ninja triton triton-opt`
- lit (run from that same build folder, **always build first**):
  `/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test`
  — for the single gate test:
  `/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test/TritonGPU/automatic-warp-specialization.mlir`
- "FIRST BUILD, *THEN* RUN lit-tests" — every lit invocation in this
  plan is preceded by a `ninja triton triton-opt`.
- "do not run pytest unless explicitly told by user" — the 4 runtime
  gates below ARE the user-mandated pytest set; nothing beyond them
  gets run without asking.

## The change (verified against 03.git `AutomaticWarpSpecialization.cpp:104-106`)

Replace:

```cpp
  addPassWithPartitionVerifier(createNVWSInsertSemaphore());
  addPassWithPartitionVerifier(createNVWSInsertTmemSemaphore());
  addPassWithPartitionVerifier(createNVWSLowerSemaphore({numStages}));
```

with:

```cpp
  addPassWithPartitionVerifier(createNVWSInsertAllocas());
  addPassWithPartitionVerifier(createNVWSInsertSemas(insertSemasOptions));
  addPassWithPartitionVerifier(createNVWSLowerSemaphore({numStages}));
```

That is all. Everything before (partition-scheduling, hoist-tmem-store)
and after (partition-loops, lower-warp-group, schedule-loops, …) stays.

## Files

Remove in 03.git (`third_party/nvidia/lib/Dialect/NVWS/Transforms/`):
`InsertSemaphore.cpp`, `InsertTmemSemaphore.cpp`; replace
`LowerAref.cpp` with solid's (same factory name
`createNVWSLowerSemaphore`; it schedules `AssignStagePhase` internally
in both versions).

Copy from solid-01 (same dir): `InsertAllocas.cpp`, `InsertSemas.{cpp,h}`,
`InsertSemasAccessDag.{cpp,h}`, `InsertSemasOwnerDag.{cpp,h}`,
`InsertSemasSyncDag.{cpp,h}`, `InsertSemasEmitIR.{cpp,h}`,
`LowerAref.cpp`.

Dialect update (this branch is the ancestor of solid's NVWS dialect —
semaphore ops already exist; bring them to solid's form):
`NVWSOps.td` (`pending_count` on create, `arrive_count` on release,
builders), `IR/Ops.cpp` (verifiers), add
`IR/SemaphorePendingCount.cpp` + `include/.../SemaphorePendingCount.h`.
Registration: `Passes.td` entries for `nvws-insert-allocas` /
`nvws-insert-semas`, CMakeLists for both dirs.

`AssignStagePhase.cpp`: kept as-is (same lineage as solid's). If the
pipeline gate shows it needs solid's newer version, take solid's —
it is part of the same sema family.

Compile fixes during the port: expected (the ops/dialect get updated
as part of the port); each fix logged below.

## Steps

1. **M0**: clear the `UU AGENTS.md` index state in 03.git (file
   content already conflict-free — verified; just `git add AGENTS.md`);
   rebuild stock branch per the protocol above; leg-0 sanity: 06-fa.py
   on stock 03.git reproduces 01.git baseline perf (same commit —
   validates the build).
2. **M1**: dialect update + copy files + registration + pipeline edit;
   build.
3. **M2**: cheap IR parity check — `MLIR_ENABLE_DUMP` on 06-fa.py,
   compare after insert-semas / lower-semaphore / assign-stage-phase
   against solid's dumps
   (`logs/fa-11jun26-v1/passes-solid-01/{056,057,058}`): same 15
   semaphores, same edges, 25 mbarriers.
4. **Gates** (in order, lit FIRST):
   - **Gate 1 (lit)**: `test/TritonGPU/automatic-warp-specialization.mlir`
     — CHECKs regenerated for the new pass output; must pass before
     any runtime test is run.
   - **Gates 2–5 (runtime)**, `python/test/unit/language/test_warp_specialization.py`:
     - `test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`
     - `test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]`
     - `test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]`
     - `test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024]`
   - **Gate 6 (runtime, mxfp4)** — added 12jun26 (was in the
     insert-semas battery, initially omitted here):
     `python/triton_kernels/tests/test_matmul.py::test_op[True-False-False-False-None-16-768-512-1024-ragged-bfloat16-mxfloat4_e2m1-10-1-False-True-None-False-False-False-True-None]`
     (solid-01 ID); baseline-parametrization equivalent on 03.git:
     `test_op[None-False-False-False-False-None-16-768-512-1024-ragged-bfloat16-mxfloat4_e2m1-None-10-1-False-False-False-None-False-False-False-True-None]`
     (the persistent-flag sibling auto-skips at the baseline:
     "persistent kernel is required" opt_flags conflict).
5. **Measurement** (after all gates green): 06-fa.py, three legs,
   isolated `TRITON_CACHE_DIR` per leg, back-to-back, per-leg artifact
   fingerprint before reading TFLOPS:
   - 01.git (baseline) — expect ~670–680
   - 03.git (ported)
   - solid-01 (reference) — expect ~606

## Decision table

| 03.git ported result | conclusion |
|---|---|
| ≈ 01.git baseline | insert-semas machinery exonerated; the solid-01 gap is backend divergence |
| ≈ solid-01 | the machinery / its emitted sync IR carries the regression |
| in between | measured split |

No commits unless the user asks. 01.git and solid-01 are not edited.

## Log (11jun26 — EXECUTED, experiment CONCLUSIVE)

**M0**: `UU AGENTS.md` index cleared (content was already conflict-free);
build was current; leg-0 sanity on stock 03.git = **682.8 TFLOPS** ✓
(matches 01.git baseline).

**M1 — mechanical fixes** (all inside ported files or inert plumbing;
no downstream pass logic touched):
1. `NVWSDialect.td`: dropped `usePropertiesForAttributes = 1` — option
   removed in the baseline's newer MLIR (properties are default-on).
2. `include/triton/Analysis/BufferRegion.h`: fwd-decl + declaration of
   `getMemDescSize` (solid's own header pattern); `BufferRegion.cpp`:
   exported one-line wrapper — the baseline implementation existed but
   was in an anonymous namespace.
3. `Partition.h`: added `setPartition(Operation*, ArrayRef<int>)`
   declaration — the implementation ALREADY existed in the baseline's
   `Partition.cpp:404`, it just wasn't declared.
4. `GetEnv.hpp`: whitelisted `NVWS_USE_SSA_TMEM` (ported pass reads it;
   baseline asserts on unrecognized names).
5. `LowerAref.cpp`: `TCGen5CommitOp::create` — solid's `two_ctas=false`
   arg → baseline's `descs=ValueRange()` (op grew a Variadic instead of
   a UnitAttr).
6. `LowerAref.cpp` + `InsertAllocas.cpp`:
   `TensorMemoryEncodingAttr::get` — solid's `CTASplitM/CTASplitN`
   params → baseline's `CGALayout` (copy `encoding.getCGALayout()` /
   identity `CGAEncodingAttr::get1CTALayout(ctx, 2)`).
7. `triton_nvidia.cc`: swapped the two dead pass bindings for
   `add_insert_allocas`/`add_insert_semas` (no python callers existed).
8. `LowerAref.cpp`: `AsyncTMACopyGlobalToLocalOp::create` reordered to
   the baseline signature — solid's leading `multicastTargets` operand
   doesn't exist at the baseline (and it has an extra `offsets`
   variadic); the misordered call type-matched silently and crashed
   with a null desc at runtime.

**M2 — IR parity**: ported insert-semas output vs solid's dumps —
semaphore creates identical (15: 8 false + 7 true, all
`pending_count = 1`); the full normalized edge trace
(`sema_trace.py`) is **byte-identical** to `trace-solid-057.txt`.
The port emits exactly solid's sync IR.

**Gates**:
- Gate 1 lit `automatic-warp-specialization.mlir`: **PASSED**
  unmodified (no CHECK regeneration needed).
- Runtime gates (baseline parametrization differs from solid's plan
  IDs; nearest equivalents used):
  `tma_matmul[True-True-False-4-2-64-128-128-2048-2048-512]` ✓
  (the literal `False-False-...` variant skips: "requires at least one
  TMA load"; baseline has no 8192 shape for this test),
  `tma_matmul_persistent[False-False-False-True-8-2-128-128-128-32-32-32]` ✓
  (+ all-False variant ✓),
  `attention_forward[False-4-True-3-128-128-1024-1024]` ✓ (verbatim),
  `attention_persistent_forward[True-8-True-3-128-128-1024-1024]` ✓
  (verbatim). **4/4 green.**

**Three-leg measurement** (isolated `TRITON_CACHE_DIR`, back-to-back,
fingerprinted):

| leg | TFLOPS | fingerprint |
|---|---|---|
| 01.git baseline | **682.1** | cubin c6cc57ac4c |
| 03.git ported (baseline + solid sema stack ONLY) | **606.8** | cubin 6e6da105bb |
| solid-01 reference | **605.5** | cubin ee3d4801ad |

## VERDICT

The ported leg lands exactly on solid-01 (606.8 vs 605.5, within
noise), **75 TFLOPS (−11%) below the baseline**, on a tree where
everything except the sema stack is identical to the baseline.
Decision table row 2: **the insert-semas machinery — specifically the
sync IR it emits — carries the entire FA fwd WS regression.**

Corollary (reconciles §10 of the perf study): the backend-era
differences (f32x2 ~24 TFLOPS, reduce-lowering structure) are real but
NOT the cause of the solid-vs-baseline gap — the ported leg HAS the
baseline's modern backend (f32x2 packing included) and still runs at
solid's level. Under the new sync IR the softmax partition is
stall-bound, not issue-bound, so the issue-count savings stop
mattering. Which property of the emitted sync IR does it (the
uniform bottom-of-loop re-acquire placement of §11 is the prime
mechanistic candidate, P-buffer ping-pong specifically; the extra acc
semaphore is epilogue-only) is the NEXT experiment — e.g., emit
point-of-use acquire for P only.

## Follow-up experiments on the branch (11jun26, later)

**Semaphore combine**: verified inert for this kernel in BOTH stacks
(no multi-buffer creates form at dump 050: 14/15 single-buffer creates;
perf identical with it commented out: baseline 682.9, ported 608.5).
Disabled on the branch (commit `c50c886658`) — this also removes the
only two stamped-acquire assumptions in LowerAref
(`analyzeCombinedSemaphoreGroup:791`, `combineSemaphores:943`),
unblocking the parked ROOT-OUTSIDE feature
(fable/attr-less-acquire-release-handoff.md).

**Suspect #1 measured — ROOT-OUTSIDE entry/exit placement** (handoff
doc §3's two edits applied to the ported `InsertSemasEmitIR.cpp`:
attr-less outside-WS-loop emission for p0/root, entry acquires always
attr-less): gate 1 lit PASSED unmodified, 4/4 runtime gates green;
IR verified — all 7 entry acquires attr-less, landing in the ROOT
block at dump 053 (v2-stamped had them in partition prologues);
non-zero post-loop consumer keeps its stamp per rule 2.2.

| config | TFLOPS |
|---|---|
| ported, stamped entry acquires (v2) | 606.3–608.5 |
| ported, ROOT-OUTSIDE (v3) | **619.7–623.9** |
| baseline | 680.6–682.9 |

**Suspect #1 is worth ~14–17 TFLOPS (~2.5%).** Remaining gap to the
baseline: ~60 TFLOPS — by elimination now isolated to suspect #2
(in-loop placement: bottom-of-loop empty-permit re-acquires + wait
order within gates). Dumps: `logs/fa-11jun26-v3-root/` (passes-03/,
trace-03root-sema.txt).

**Gate 6 (mxfp4) results** (12jun26): 03.git ported tree (with both
placement fixes) — baseline-equivalent ID **passed** (12.2s); solid-01
final tree (all three ported commits) — original ID **passed** (6.1s),
recorded with the 5-gate battery in the study §14.5.
