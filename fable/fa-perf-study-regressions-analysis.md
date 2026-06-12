# FA fwd perf study: solid-01 vs 01.git regression analysis (11 jun 26)

## Question

`06-fa.py` (FA-2 forward, BATCH=4, H=32, N_CTX=65536, HEAD_DIM=128,
causal=False, warp_specialize=True, fp16) runs ~7.5% slower on this repo
(solid-01, new `nvws-insert-semas` stack) than on the 01.git baseline
(old `nvws-insert-semaphore` + `nvws-insert-tmem-semaphore` stack):

| run | solid-01 | 01.git |
|---|---|---|
| user's measurement | 604.4 TFLOPS | 665.6 TFLOPS |
| this study (under `MLIR_ENABLE_DUMP`) | 619.7 TFLOPS | 670.1 TFLOPS |

Is the regression caused by the new semaphore-insertion / auto-WS stack
(numStages? assign-stage-phase? mbarrier structure?) — or by something else?

## Verdict (TLDR)

**SUPERSEDED by §13 (port experiment, 11jun26): the insert-semas-emitted
sync IR causes the ENTIRE gap.** Porting solid's sema stack onto the
baseline tree (everything else identical — backend, reduce lowering,
LLVM, Membar) reproduces solid's perf exactly: baseline 682.1 / ported
606.8 / solid-01 605.5 TFLOPS. The backend-era differences analyzed in
§§5–10 and §12 are real, measured code differences but are NOT the cause
of the solid-vs-baseline FA gap. The v2 dump comparison (§13.3) narrows
the cause to exactly two placement differences in the emitted sync IR
(§13.4); everything else — including per-partition pipelining — is
verified identical.

Measured decomposition — **complete, gap fully recovered (§14)**:
semaphore combine — inert, ruled out; **suspect #1 (pre/post-loop
placement → ROOT-OUTSIDE emission) = ~15 TFLOPS** (606–608 → 620–624);
**suspect #2 (in-loop placement → point-of-use re-acquires) = ~47
TFLOPS** (620–624 → 667–671). With both emitter placement fixes the
ported stack reaches **parity with the baseline** (interleaved A/B:
671.6 vs 670.5, 659.2 vs 660.8 — tracking run-for-run). The FA
regression is fully explained by two insert-semas emitter placement
policies; pipelining was never affected (§13.3), so the in-loop
mechanism is runtime stalling — the §11.4 analysis (softmax blocking on
the PV-MMA commit before the next iteration's math) measured out after
all, on this kernel shape.

The original verdict below is kept for the record of how the
investigation evolved; read it with §13 in mind.

ORIGINAL (superseded): the auto-WS/insert-semas stack is exonerated;
both repos converge to count-identical synchronization in the hot loop;
the gap lives in the **backend codegen of the softmax warpgroup** —
FA's two per-iteration row reductions (max, sum) — where 01.git is
generations ahead (upstream reduce-lowering rewrite #9219/#9220/#9897
plus a newer pinned LLVM).

**Causally measured decomposition (§10, gated A/B on 11jun26):**

| component | measured worth |
|---|---|
| `add.f32x2` packing of the row-sum adds | **~24 TFLOPS (~3.5%)** — 01.git drops 682.0 → 658.3 with packing disabled at both producers; SASS confirms the only change is FADD2→FADD |
| everything else | **~52 TFLOPS (~8%)** — fully-scalar 01.git (658.3) still beats solid-01 (605.8). Primary candidate: the old SMEM+barrier reduce structure (+17 BAR.SYNC, ~+300 uniform-datapath ops, +208 instructions in solid). A second IR difference — the emitter's uniform bottom-of-loop re-acquire placement (§11) — was ruled perf-neutral by the user based on past A/B experiments on other workloads (deliberate design; caveat in §11.5) |

Note the f32x2 packing has **two independent producers** in 01.git: the
MLIR-level reduce vectorization (#9220, `packVectorized`) and **LLVM's
SLPVectorizerPass**, which re-creates the packing even when the MLIR-level
one is disabled (verified by per-pass dumps, §10). solid-01's older pinned
LLVM (`a992f294` vs `87717bf9`) does not SLP-pack v2f32 for sm_100.

Fix direction: port the upstream reduce-lowering rewrite into solid-01 and
consider the LLVM bump (nontrivial — solid-01's `ReduceOpToLLVM.cpp`
carries local multi-CTA reduction changes interleaved into the same file).
See §8.

---

## 1. Method and artifacts

Both repos compiled the same driver from the same cwd, differing only in
`PYTHONPATH`:

```
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 \
  PYTHONPATH=<repo>/python python 06-fa.py 2> dump-<repo>.log
```

All artifacts live in `logs/fa-11jun26-v1/`:

| artifact | contents |
|---|---|
| `passes-solid-01/` (111 files), `passes-01/` (99 files) | one `.mlir` per `IR Dump Before <pass>`, numbered in pipeline order; `INDEX.txt` lists order |
| `dump-solid-01.log`, `dump-01.log` | raw unsplit stderr captures (13 MB / 19 MB) |
| `run-*.out` | benchmark stdout (TFLOPS) |
| `ptx-solid-01.ptx`, `ptx-01.ptx` | final PTX (from triton cache) |
| `llir-solid-01.ll`, `llir-01.ll` | final LLVM IR |
| `sass-solid-01.txt`, `sass-01.txt` | `nvdisasm -c` of both cubins |
| `summary-*-postpipeline.txt` | per-partition sync-op sequences after `tritongpu-pipeline` |
| `histo-*.txt` | per-partition loop-body op histograms, final pre-LLVM TTGIR |
| `split_passes.py`, `summarize_partitions.py`, `loop_histo.py`, `sass_loops.py` | the analysis scripts |

Since every dump is *before* a pass, "after pass X" = the next-numbered file.
Key indices:

| stage | solid-01 | 01.git |
|---|---|---|
| input to auto-WS | `047` | `040` |
| after partition-scheduling | `049` | `042` |
| input to sema insertion | `054` (insert-semas) | `045` (insert-semaphore) |
| after sema insertion | `056` | `049` |
| after lower-semaphore (no-op here, §4.3) | `057` | `050` |
| after assign-stage-phase (mbarriers appear) | `058` | `051` |
| after partition-loops + pipeline | `063` | `056` |
| final pre-LLVM TTGIR | `092` | `080` |

Pipeline note: the WS-relevant segment is aligned between repos
(AWS → partition-scheduling → hoist-tmem-store → sema passes →
lower-semaphore → assign-stage-phase → partition-loops → lower-warp-group →
schedule-loops → pipeline). solid-01 additionally has passes 01.git lacks
(`nvgpu-ws-data-partition`, `tritongpu-prefetch`,
`tma-store-buffer-reuse`, `nvgpu-multi-cta-reduction`,
`prune-unused-barriers`, `global-scratch-memory-allocation`,
`tma-store-token-wait-lowering`) — none implicated below, but they show the
repos have diverged well beyond the sema stack.

## 2. Suspect checklist — verdicts

| suspect | verdict | evidence |
|---|---|---|
| (a) numStages wrong vs 01.git | **cleared** | `tt.scheduled_max_stage = 4` both; `ttg.partition.stages = [0,0,1,0]` both; K/V SMEM deepened 1→3 by the pipeliner in both; QK TMEM 2-deep, acc/P 1-deep in both |
| (b) assign-stage-phase mis-advancing stages | **cleared** | `loop.stage` histograms match (18@s0, 1@s1, 46@s2); in-loop sync is count-identical (§4.4); only placement (rotation) differs |
| (c) mbarrier level not kosher | **cleared** | 25 vs 24 `init_barrier`, all `count = 1` in both; the +1 is an epilogue-only acc handoff sema (§4.2); per-iteration wait/arrive/commit identical |
| (d) input to auto-WS very different | **cleared (almost)** | op histograms over the AWS input are *identical*; only two attribute deltas, both pre-AWS repo drift, neither material downstream (§3) |
| (e) partition-scheduling drastically different | **cleared** | identical `ttg.partition` stamp histogram: 20×{0}, 15×{1}, 5×{3}, 3×{2}, 2×{0,1,2,3} |
| actual cause | **backend `tt.reduce` lowering** | §§5–7 |

## 3. Input to auto-WS (047 vs 040)

Structurally the same kernel: the op-name histograms of the two files are
**identical** (verified by sorted uniq-count diff). Exactly two deltas, both
created by passes upstream of AWS (repo drift, not the sema stack):

1. `tt.self_latency = 1` (solid-01) vs `= 0` (01.git) on both
   `ttng.tc_gen5_mma` ops (QK and PV). `tt.latency = 2` on the K/V
   `tt.descriptor_load`s and the QK MMA is the same in both.
   Downstream effect: **none observed** — buffering depths, peeled prologue
   (2 QK MMAs), and steady-state schedules came out the same (§4, §5).
2. PV MMA use-acc operand: loop-carried `%arg29` (solid-01, from
   optimize-accumulator-init) vs constant `%true` (01.git). First-iteration
   semantics only; final loop histograms differ by one `select`.

`loop.cluster`/`loop.stage` stamps on the MMAs are identical (QK: c2/s2,
PV: c0/s4).

## 4. The sync structure is equivalent end-to-end

### 4.1 Buffers and semaphores (after sema insertion, 056 vs 049)

Same six resources, same shapes, same depths:

| resource | type | depth (both repos) |
|---|---|---|
| qk | TMEM 128x128 f32 | 2 |
| acc | TMEM 128x128 f32 | 1 |
| P (f16 probs; SSA-named `offsetv_y` in solid, `acc_33` in 01.git) | TMEM 128x128 f16 | 1 |
| k, v | SMEM 128x128 f16 | 1 (→3 after pipeliner) |
| m_i, m_ij | SMEM 128 f32 | 1 |

- 01.git: 14 `nvws.semaphore.create` = 7 empty/full pairs.
- solid-01: 15 creates — the same 7 pairs **plus a third semaphore on
  `%acc`** (one `true` + two `false`). The extra one closes the
  correction→epilogue acc handoff; its wait/arrive run **once per kernel**,
  not in the loop.
- All solid-01 creates carry `pending_count = 1`; all releases
  `arrive_count = 1` (no multiplicity in this kernel).

### 4.2 Lowering to mbarriers

In *both* repos the dump after `nvws-lower-semaphore` still contains all
semaphore ops (68 and 66 respectively). CORRECTION (11jun26, §11): the
pass is NOT a no-op — in both repos it performs the K/V multibuffering
(allocs deepened 1→3, identically); it just doesn't lower semaphores
here. The semaphores are consumed and mbarriers materialized inside
`nvws-assign-stage-phase` (dumps 058/051): solid-01 → **25**
`init_barrier`, 01.git → **24**, every one with count 1.

### 4.3 Per-iteration sync (after assign-stage-phase, 058 vs 051)

Whole-module op counts differ (22w/8a/5c vs 18w/10a/6c) but the difference
is entirely **prologue/epilogue**. Inside the hot N_CTX loop (512
iterations) both repos execute per iteration:

```
14 wait_barrier · 7 arrive_barrier · 5 tc_gen5_commit
 2 tc_gen5_mma  · 2 tmem_load     · 2 tmem_store
```

The only difference is *placement*: solid-01 sinks seven stage-advance
waits into a contiguous block at the loop bottom (computed-phase waits),
01.git interleaves them at point of next use. This is the emitter's
uniform rotated entry-acquire shape — documented in detail in §11,
where it is ruled a deliberate, known perf-neutral design (user, from
past A/B experiments on other workloads). solid-01 also carries ~17
more stage-4 index/phase arithmetic ops at this stage; they fold away
by final TTGIR.

**Stage-advance verification** (user follow-up, confirmed in IR): both
repos emit the identical canonical advance per semaphore —

```
idx'   = (idx + 1 == DEPTH) ? 0 : idx + 1     // addi, cmpi eq DEPTH, select
phase' = wrapped ? phase ^ 1 : phase           // xori, cmpi eq 0, select
```

— and the per-resource DEPTH constants match exactly:

| resource | solid-01 wrap | 01.git wrap |
|---|---|---|
| qk | 2 | 2 |
| k | **3** | **3** |
| v | **3** | **3** |
| m_i, m_ij, acc, P | 1 | 1 |

Note assign-stage-phase already plans k/v at depth 3 (3-deep
`memdesc<3x1xi64>` mbarrier arrays, mod-3 index advance) in *both* repos,
anticipating the pipeliner's deepening of the data buffers from
`tt.latency = 2`; the 25/24 `init_barrier` counts are exactly the slot
sums (k 3+3, v 3+3, qk 2+2, m_i/m_ij/P 1+1 each, acc 1+1 (+1 extra in
solid-01)). Loop-carried sync counters: 21 (solid-01) vs 20 (01.git) i32s —
the +1 is the extra acc semaphore's; additionally solid-01 advances a
dedicated index for the P buffer where 01.git shares/elides the redundant
depth-1 counter. Depth-1 indices always wrap, so phases toggle every
iteration in both — identical runtime behavior, and the redundant
counters fold by final TTGIR (§4.4 histograms within ±1 op).

### 4.4 Final per-partition loops (092 vs 080)

`tritongpu-pipeline` deepened K/V to 3 buffers and peeled 2 QK MMAs in
both. Final loop-body op histograms per partition:

| region | solid-01 | 01.git | delta |
|---|---|---|---|
| default (softmax, 4 warps) | 41 ops | 42 | 01.git +1 `xori` |
| partition0 (correction, 4 warps) | 32 | 31 | solid +1 `ttg.convert_layout` |
| partition1 (MMA, 1 warp) | 45 | 46 | 01.git +1 `select` |
| partition2 (TMA load, 1 warp) | 27 | 27 | none |

Same wait/arrive/commit/MMA/TMA counts in every region; same
`num_warps(4,1,1,2,4)`; `ttg.total-num-warps = 12` both.

### 4.5 Resource footprint

| | solid-01 | 01.git |
|---|---|---|
| `ttg.shared` | 231104 B | 231056 B |
| `ttg.tensor_memory_size` | 512 | 512 |
| module `maxnreg` | 168 | 168 |
| spills (PTX `.local`, SASS LDL/STL) | none | none |
| WS registers actual | [328, 152, 24, 24, 24] | [392, 88, 24, 24, 24] |
| WS registers requested | [152, 24, 24, 16] | [88, 24, 24, 16] |

The register split differs (solid-01's optimize-partition-warps estimates
152 for the correction partition where 01.git estimates 88), but the big
default warpgroup **clamps to the 256 hardware cap in both** (PTX
`setmaxnreg.inc 256`; SASS `USETMAXREG` 0x100=256, 0x98=152, 0x58=88,
0x78=120, 0x18=24), so this redistribudes only headroom, not the hot
partition's budget. Not the cause.

## 5. Root cause: `tt.reduce` lowering divergence

The evidence chain, top down:

1. **SASS instruction histograms** (whole kernel): identical fp work where
   expected — FMUL 519 = 519, FFMA 141 = 141, MUFU 131 = 131,
   F2FP 128 = 128, UTCHMMA 32 = 32 — but:

   | opcode | solid-01 | 01.git |
   |---|---|---|
   | FADD (scalar) | **132** | ~0 |
   | FADD2 (packed f32x2) | 0 | **63** |
   | FMNMX/FMNMX3 | 63 (FMNMX3) | 44 + 42 |
   | BAR | 68 | 51 |
   | SYNCS | 97 | 90 |
   | UMOV | 184 | 63 |
   | total static | 2573 | 2302 |

2. **Address-range localization**: the 132 scalar FADDs co-locate exactly
   with the MUFU.EX2 region — i.e. the **softmax (default) warpgroup**.
   Inside that region: solid-01 BAR=57, UMOV=178, STS=10, LDS=6 versus
   01.git BAR=39, UMOV=52, STS=4, LDS=4. The old lowering does an
   SMEM-roundtrip cross-warp combine with extra `bar.sync`s and uniform
   address moves; the new one keeps the tree reduce in registers.

3. **PTX**: 01.git emits **63× `add.f32x2`** (plus 4 scalar `add.f32`);
   solid-01 emits **0** packed and 130 scalar `add.f32`. So the packing is
   decided *before* ptxas.

4. **LLVM IR**: 01.git has 64× `fadd <2 x float>` (with the
   `insertelement <2 x float>` pairing pattern); solid-01 has 131× scalar
   `fadd float`. CORRECTION (11jun26, second pass — per-pass dump
   bisection, §10): the packing has **two independent producers** in
   01.git — the MLIR-level reduce vectorization (#9220,
   `ReduceOpToLLVM::packVectorized`, gated by
   `ReduceOpHelper::getInThreadVectorizeOpKind`) **and LLVM's
   `SLPVectorizerPass`** (enabled at `python/src/llvm.cc:697`), which
   re-creates the identical packing from scalar input. Disabling only the
   MLIR-level one changes nothing observable in PTX. solid-01 pins an
   older LLVM (`a992f294` vs 01.git's `87717bf9`) whose SLP does not form
   v2f32 for this pattern, and its NVPTX packs at most a trickle at ISel
   (16 in a probe kernel, 0 in the FA kernel).

5. **Source provenance** (`git log -- lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`):
   - 01.git tip history: `483327f033` **#9219** "Improve and simplify
     ReduceOp's lowering", `b5e3800aec` **#9220** "Perform tree reductions
     on in-thread values" (the `<2 x float>` packing), `bb75a87080` #9221
     cross-CTA reduce, `bc791297eb` **#9897** ternary FMNMX3 grouping —
     a LinearLayout-based rewrite of the whole lowering.
   - solid-01 tip history: local commits `053b50f75e` (#1100
     bitwise-consistent reductions prototype) and `e4c43d957d` (#1102
     multi-CTA reduction) on an **older upstream base** (~#6026/#6323 era).

FA's softmax executes two `tt.reduce` per iteration (row-max for
rescaling, row-sum for `l_i`) over the 128x128 `p` tile, 128 elements per
thread in the 4-warp default WG. The old lowering spends ~127 scalar FADDs
plus an SMEM/barrier combine on what the new lowering does in ~64 packed
FADD2s with fewer barriers.

## 6. Magnitude (measured, supersedes the original estimate)

The default (softmax) warpgroup is the critical path: its steady-state body
is ~1100–1200 issue slots/iteration (FMUL 519, FFMA 141, MUFU 131,
F2FP 128 + adds + uniform/control), while the single-warp MMA partition
needs ≈1024 tcgen05 cycles for the two 128³ MMAs — the partitions are
nearly balanced, so softmax issue count directly moves the bottom line.

The gated A/B (§10) decomposes the gap causally:

| config | TFLOPS | SASS |
|---|---|---|
| 01.git stock (f32x2 on) | 682.0 | FADD2 63, BAR 51, UMOV 64, total 2168 |
| 01.git f32x2 OFF (both producers gated) | 658.3 | FADD 132, BAR 51, UMOV 64, total 2232 |
| solid-01 | 605.8 | FADD 132, BAR 68, UMOV 185, total 2440 |

- ON→OFF differs **only** in FADD2→FADD (+64 instructions, all else
  identical): packing alone = **~24 TFLOPS (~3.5%)**.
- OFF→solid (both fully scalar): the remaining **~52 TFLOPS (~8%)**
  correlates with solid's +17 BAR.SYNC, ~+300 uniform-datapath ops
  (UMOV/ULEA/ULOP3/USHF/R2UR), −71 UIADD3, +208 total instructions —
  the old SMEM+barrier reduce structure and older-LLVM codegen, all
  localized in the softmax region (§5.2). This part is correlational
  (no single switch can A/B the lowering structure); a causal split
  would require the reverse port.

## 7. Control experiment: warp_specialize=False

Same benchmark with WS off (`/tmp/06-fa-nows.py`):

| | solid-01 | 01.git |
|---|---|---|
| TFLOPS | **303.5** | 208.4 |

The sign *flips* — solid-01 is 45% faster on the non-WS software-pipelined
path. This doesn't isolate the reduce effect (the non-WS path exercises a
different pipeliner where the repos diverge in many other ways), but it
confirms the two trees differ broadly outside the auto-WS stack, and that
solid-01 is not generically slower.

## 8. Recommendation

Port the upstream reduce-lowering rewrite into solid-01:

- minimum: **#9219 + #9220** (LinearLayout-based lowering + in-thread tree
  reductions with f32x2 packing) — this is where the FADD2s and the
  barrier reduction come from;
- nice-to-have: **#9897** (ternary FMNMX3 grouping for max/min — also
  helps FA's row-max).

Complication: solid-01's `ReduceOpToLLVM.cpp` interleaves local features
into the same code paths — the multi-CTA reduction support (#1102,
feeding the `nvgpu-multi-cta-reduction` pass: 1-element cross-CTA DSM
exchange cases) and the bitwise-consistent-reduction prototype (#1100).
These must be re-grafted onto the rewritten lowering, so this is a port,
not a cherry-pick.

Measured expectation on this benchmark (§6): the structural part of the
rewrite is worth ~8%; the f32x2 packing another ~3.5% — but note the
packing needs either #9220's MLIR-level vectorization or a newer LLVM
whose SLP forms v2f32 (solid-01's pinned `a992f294` does not), so an LLVM
bump may be required to collect that last part.

## 9. Residual deltas attributable to the new stack (all benign)

For completeness, everything the insert-semas stack does differently from
the old passes on this kernel — none in the steady-state loop:

1. **+1 semaphore on `%acc`** (15 vs 14 creates; 25 vs 24 mbarriers):
   correction→epilogue handoff, once per kernel.
2. **3 extra pre-loop waits** in the default WG (phase-1 waits on
   initially-released semaphores), once per kernel.
3. **Stage-advance wait placement**: contiguous block at loop bottom vs
   distributed. This is the emitter's uniform rotated entry-acquire
   shape (solid applies it to all buffers; 01.git only to QK/ACC) —
   detailed in §11. USER RULING: deliberate design, known perf-neutral
   from past A/B experiments on other workloads (caveat in §11.5).
4. **~17 extra phase/index arithmetic ops** after assign-stage-phase —
   folded by final TTGIR (histograms within ±1 op).
5. **+1 `ttg.convert_layout`** in the correction partition's loop
   (final TTGIR) — single 128-wide f32 vector relayout, 4-warp partition,
   not on the critical path.

Items 1, 2, 4, 5 cannot account for a measurable fraction of the gap.
Item 3's original dismissal reasoning ("if anything favors solid-01")
was wrong as stated; the corrected analysis is in §11, where the
difference is documented in full and ruled perf-neutral by the user
(past A/B experiments on other workloads).

## 10. Causal A/B: disabling f32x2 in 01.git (11jun26, second pass)

User-requested experiment: temporarily disable the f32x2 packing in 01.git
and measure whether it explains the gap.

### Temporary gates (in 01.git — REVERT when done)

| file | gate |
|---|---|
| `lib/Analysis/Utility.cpp:141` (`getInThreadVectorizeOpKind`) | env `TRITON_DISABLE_REDUCE_VEC` → `InThreadVectorizeOpKind::None` (kills the MLIR-level #9220 packing) |
| `python/src/llvm.cc:697` | env `TRITON_DISABLE_SLP_VEC` → `tuningOptions.SLPVectorization = false` (kills LLVM's re-packing) |

Build: `cd build/cmake.linux-x86_64-cpython-3.12 && ninja triton triton-opt`.
With both envs unset the build behaves exactly like stock (verified:
probe and FA artifacts identical to pre-edit).

### Finding the second producer

With only the MLIR-level gate, PTX still contained `add.f32x2` — initially
misread as a dead knob. Per-stage dump bisection on a minimal `tl.sum`
probe kernel resolved it:

- `MLIR_ENABLE_DUMP=1`: **zero** `vector<2xf32>` across all 91 MLIR dumps —
  the MLIR gate works; the LLVM-dialect module is fully scalar.
- `LLVM_IR_ENABLE_DUMP=1`: the first dump containing `fadd <2 x float>` is
  **"IR Dump After SLPVectorizerPass"** — LLVM re-vectorizes the scalar
  adds.

Probe matrix (fresh `TRITON_CACHE_DIR` per run; llir vector-fadd / ptx
f32x2): stock 64/48; reduce-vec off 64/48; SLP off 64/32; **both off 0/0**.

### Results (isolated caches, same binary for ON/OFF, back-to-back)

| leg | TFLOPS | FA PTX f32x2 |
|---|---|---|
| 01.git stock | 682.0 | 63 |
| 01.git both gates | 658.3 | 0 |
| solid-01 reference | 605.8 | 0 |

SASS confirms the OFF leg's only change vs ON is FADD2→FADD (totals
2168→2232; BAR/UMOV/branches identical) — no SLP collateral on this
kernel, so the 24-TFLOPS delta is attributable to the packing alone.

### Verdict

f32x2 packing = **~3.5%** of the gap, causally measured. The remaining
**~8%** persists with both kernels fully scalar and correlates with the
reduce-lowering *structure* (solid: +17 BAR.SYNC, ~+300 uniform ops,
+208 instructions, all in the softmax region) — see §6.

### Methodology trap (for the record)

A first version of this A/B produced a bogus "packing ≈ 1%" result: with
the shared `~/.triton/cache`, the inspected "OFF" artifacts were actually
a solid-01 entry (env vars are not part of the cache key, and
`MLIR_ENABLE_DUMP` IS part of it, which scrambled entry attribution).
Every number above was re-derived with per-leg `TRITON_CACHE_DIR` and
artifact fingerprinting (tensordesc print-syntax differs per repo;
f32x2 count; cubin md5). An intermediate "solid and 01-vecOFF cubins are
bit-identical" observation was an artifact of the same mixup (solid
compared against itself) and is retracted.

## 11. Addendum (11jun26): post-lower-semaphore comparison — the acquire-placement difference

Follow-up to the remaining ~50-TFLOP gap (§6): a careful, verified
comparison of the IR state after `nvws-lower-semaphore`
(`passes-solid-01/057` vs `passes-01/050`), requested as "verified
answers, no assumptions". Two of the report's original claims are
corrected by this section (§4.2 "no-op", §4.3/§9.3 "rotation, neutral").

### 11.1 What nvws-lower-semaphore actually does (verified by exact diff)

Full-text diff of before/after dumps (056→057 solid, 049→050 01.git):
in BOTH repos the pass performs exactly one transformation — **K and V
SMEM allocs are deepened 1→3** (`memdesc<1x128x128xf16>` →
`memdesc<3x128x128xf16>`, with type updates on every referencing
semaphore op). No semaphore is lowered at this position. The only other
textual deltas: solid's ops carry the first-class `pending_count` /
`arrive_count` attrs, and 01.git's `semaphore.buffer` result types print
an allocShape suffix (`..., 1x128x128>`). Identical transformation,
identical depths.

### 11.2 The semaphore edge graph is identical (verified by normalized trace)

Method: `sema_trace.py` maps every semaphore to a stable identity
(buffer role × init flag × ordinal; the P buffer matched by type across
its differing SSA names — `%offsetv_y` in solid, `%acc_33` in 01.git)
and prints every acquire/release/buffer plus key compute ops in program
order with region (PRE/LOOP/POST), partition, async kind, and
stage/cluster. Outputs: `trace-solid-057.txt`, `trace-01-050.txt`
(68 vs 67 lines).

Result: the edge graph matches **edge for edge** — same 7 resources,
same producer→consumer pairs, same `[tc5mma]`/`[tma_load]`/`[none]`
async kinds, same `ttg.partition` and `loop.stage/cluster` stamps on
every corresponding op. Immaterial deltas: solid's extra `ACC-b`
epilogue semaphore (§9.1); a permutation inside the PV-MMA's three
tc5mma releases (V,P,ACC vs P,ACC,V); slightly different epilogue QK
handling; one missing stage/cluster stamp on 01.git's p1 M_I- acquire.

### 11.3 THE difference: acquire placement (the one real divergence)

**solid-01 entry-acquires ALL SEVEN empty-side semaphores before the
loop and re-acquires them as a contiguous block at the loop BOTTOM**,
carrying 7 `!ttg.async.token` iter_args (loop has 32 results). **01.git
acquires at POINT OF USE inside the loop body**, carrying only 2 tokens
(QK+ and ACC+ — whose trailing acquires exist in both repos; loop has
25 results).

solid's trailing block (with consumer partition/stage):
`QK+ p2/s2 · ACC+ p1/s4 · P+ p0/s4 · M_I+ p0/s4 · K+ p3/s0 · V+ p3/s2 ·
M_IJ+ p0/s4`. The five beyond QK+/ACC+ are the solid-only ones.

This placement survives **verbatim to the final pre-LLVM IR** (dump 092
vs 080). Softmax (default) partition loop body, side by side:

| solid-01 (092) | 01.git (080) |
|---|---|
| arrive m_i-full | **wait m_i-empty** ← point of use |
| wait qk-full → tmem_load qk | arrive m_i-full |
| arrive m_ij-full | wait qk-full → tmem_load qk |
| **exp2 / softmax math block** | **wait m_ij-empty** ← after qk load |
| tmem_store p → arrive P-full | arrive m_ij-full |
| **wait P-empty** ← loop bottom | **exp2 / softmax math block** |
| **wait m_i-empty** ← loop bottom | **wait P-empty** ← after ALL the math |
| **wait m_ij-empty** ← loop bottom | tmem_store p → arrive P-full |
| yield | yield |

### 11.4 Why it matters (dependency mechanics, readable off the IR)

P is 1-deep. `P-empty` is released by the **PV MMA's tc5mma commit**
(partition 2), and the PV MMA can only start after softmax stores p
(plus the correction partition's acc store). Therefore:

- **solid-01**: softmax stores p(i), releases P-full, then *immediately*
  stalls at the loop bottom until PV MMA(i) has run AND committed —
  before iteration i+1's qk load and exp2 math even start. The PV-MMA
  latency (MMA + commit + barrier round-trip) is **fully exposed, every
  iteration**, in the kernel's critical partition.
- **01.git**: iteration i+1's entire math body executes before the
  P-empty wait; the PV-MMA latency is hidden behind ~the whole softmax
  body. The wait is in practice already satisfied when reached.

Severity by semaphore: `P-empty` is the big one (hoisted above the
entire body). `m_ij-empty` (released by the correction partition's read)
is hoisted above the qk load + max math — smaller but real.
`m_i-empty` is positionally equivalent (bottom-of-i ≈ top-of-i+1, no
work in between). The TMA partition's K+/V+ bottom-acquires are
equivalent too (its body has no work between bottom and point of use),
and QK+/ACC+ trail in both repos — so the materially affected partition
is exactly the softmax one, where the whole measured gap lives.

### 11.5 Attribution and status (USER RULING, 11jun26)

Precise characterization (verified in the traces): solid-01 applies the
rotated shape (entry acquire + bottom re-acquire, token through
iter_args) **uniformly to all seven buffers**; 01.git applies it only
to **QK and ACC**, using point-of-use acquires for the four SMEM
buffers AND for P — i.e. P, although TMEM, follows the local-alloc
shape in 01.git.

USER RULING: this difference is **expected** — the rotated
entry-acquire shape is the emitter's deliberate design — and past A/B
experiments on other workloads showed it does **not** cause perf
regressions. The perf attribution speculated above is therefore ruled
out as the explanation of the remaining gap; the IR-level facts in
§11.1–11.4 stand as a record of the difference itself.

Caveat kept for the record: the prior experiments were on other
workloads; the shape that distinguishes this kernel is the 1-deep
softmax↔PV-MMA ping-pong on P, where the bottom-vs-point-of-use
distance spans the entire softmax math body (§11.4). If the remaining
gap survives the reduce-lowering port (§8), the cheap causal test is:
emit the in-loop re-acquire at point of first use for P only, rebuild,
re-run the §10 three-leg benchmark.

### 11.6 Artifacts

`logs/fa-11jun26-v1/`: `sema_trace.py`, `trace-solid-057.txt`,
`trace-01-050.txt`; the exact diffs are reproducible via
`diff passes-solid-01/056-*.mlir passes-solid-01/057-*.mlir` (and
049/050 for 01.git).

## 12. Addendum (11jun26): PTX comparison, scalar-vs-scalar (measured facts only)

User-requested: compare the PTX of 01.git-without-f32x2 (§10 OFF leg,
658.3 TFLOPS) against solid-01 (605.8) — both fully scalar. Artifacts:
`ptx-ab2-{on,off,solid}.ptx` (2722/2717/2803 lines). Everything below
is measured; no attribution of the remaining ~52 TFLOPS to any
individual delta has been established.

### 12.1 Whole-kernel PTX opcode deltas (solid vs 01off)

| op | solid | 01off |
|---|---|---|
| `bar.sync` | 53 | 36 |
| `elect.sync` | 10 | 61 |
| `and.pred` | 12 | 58 |
| `and.b32` | 62 | 11 |
| `or.b64` | 56 | 7 |
| `cvt.u64` | 56 | 7 |
| `add.s64` | 0 | 49 |
| `add.s32` | 179 | 128 |
| `max.f32` | 65 | 86 |
| total lines | 2803 | 2717 |

SASS-level counters from the same kernel pair (§6 table, nvdisasm):
solid +17 BAR, +121 UMOV, +49 ULEA, +56 ULOP3, +41 USHF, +31 R2UR,
−71 UIADD3, +208 total instructions; FMNMX3/FMNMX 63/2 (solid) vs
42/44 (01off).

### 12.2 bar.sync localization (measured)

Per-loop extraction (backward-branch spans): the softmax warpgroup's
steady-state loop contains **8 bar.syncs in solid vs 5 in 01.git**;
the other warpgroup region totals 13 vs 14. Of the softmax-loop
barriers, five appear in both repos at the same kind of site
(store/load → bar.sync → mbarrier arrive: m_i store, qk tmem-load,
m_ij store, P tcgen05.st with a double bar). solid's three additional
sites, read directly from the PTX:

1. after the `arrive` on m_i-full, before the qk-full wait loop
   (`ptx-ab2-solid.ptx:359`);
2. before the m_ij `stmatrix` store — solid has barriers on both sides
   of this store, 01.git only after (`:455`);
3. after the `arrive` on P-full, before the loop-bottom wait block
   (`:938`).

All three sit between an mbarrier arrive and the next SMEM-touching
operation. The corresponding positions in 01.git's loop have no
barrier.

What was checked about their origin: solid's
`ttng.wait_barrier` lowering emits no bar.sync
(`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp:214`
read in full), and the final TTGIR dumps contain almost no explicit
barrier ops (solid 1, 01.git 3) — so these barriers are materialized
during the TTGIR→LLVM conversion. **Which component inserts them has
NOT been determined.** Related fact: `lib/Analysis/Membar.cpp` differs
between the repos; 01.git's history on that file includes #9281
(`canSkipBarSync`) + reverts, #9456, #10035, while solid-01's last
touches are #8755/#8798.

### 12.3 Not established / next measurement

No data point yet links any §12.1/§12.2 delta to the ~52 TFLOPS.
The direct measurement available: remove the three solid-only
`bar.sync`s from solid's generated PTX (sites listed in §12.2) and
benchmark the patched kernel — this would price the barriers exactly,
with no compiler change. (Superseded by §13, which attributes the gap
wholesale before this measurement was needed.)

## 13. The port experiment (11jun26): the emitted sync IR causes the entire gap

The decisive single-variable experiment (user-directed; plan + full
execution log in `fable/port-insert-seams-to-main.md`): port solid's
sema machinery (InsertAllocas + InsertSemas + its LowerAref, plus the
dialect deltas) onto the **baseline tree** — `triton-03.git`, branch
`egx/nvws-semaphore-insert-semas`, same commit `34245cc5ef` as
01.git's `egx/nvws-semaphore` — replacing only
`InsertSemaphore + InsertTmemSemaphore + LowerSemaphore(old)`.
Everything else (partition-scheduling, hoist-tmem-store,
assign-stage-phase, partition-loops, lower-warp-group, per-partition
pipeliner, Membar, reduce lowering incl. f32x2 packing, LLVM pin) is
the baseline's, byte for byte. Committed as `1ff94f70c8` in 03.git.

### 13.1 Result

Port fidelity verified before measuring: the ported pass's emitted
sync IR is **byte-identical** to solid-01's normalized semaphore trace
(`sema_trace.py`). Gates: `automatic-warp-specialization.mlir` lit
passed unmodified; 4 warp-spec runtime gates green (baseline-
equivalent parametrizations).

| leg | TFLOPS |
|---|---|
| 01.git baseline (old sema stack) | **682.1** |
| 03.git ported (baseline + solid sync IR only) | **606.8** |
| solid-01 reference | **605.5** |

The ported leg lands exactly on solid-01: **the sync IR emitted by
insert-semas accounts for the full −75 TFLOPS (−11%)**.

### 13.2 Reconciliation with §§5–12

The backend-era findings remain true as code differences but are not
the cause of the solid-vs-baseline gap: the ported leg HAS the modern
backend (FADD2 packing included) and still runs at solid's level.
The §10 f32x2 measurement (−24 TFLOPS within 01.git) is context-
dependent: with the OLD sync IR the softmax partition is issue-bound,
so instruction savings pay; under the NEW sync IR the partition is
stall-bound and the same savings stop mattering. Effects are
non-additive; the sync IR is the binding constraint.

### 13.3 v2 dump comparison: what is verified IDENTICAL

Full per-pass dumps of both legs in `logs/fa-11jun26-v2/`
(`passes-01/`, `passes-03/`, indices align 1:1; summaries
`sum-*-postpart.txt`, `sum-*-postpipe.txt`, `histo-*-056.txt`,
`trace-{01,03}-sema.txt`). Verified identical between baseline and
ported:

- partition assignment and `partition.stages`;
- stage/cluster maps fed to the per-partition pipeliner (dump 055):
  p1 ≈ {s0/c5 ~20, s2/c2 ~115, s4/c0 78} both; default/p0 unpipelined
  both;
- **pipelining itself**: 2 peeled QK MMAs in p1's prologue, K/V
  multibuffered to 3 (18 = 18 type refs), same steady-state MMA loop
  (5 waits / 2 MMAs / 5 commits) — the MMA loop is pipelined correctly
  in both;
- per-partition loop-body op histograms after the pipeliner (056):
  default 65 vs 64, p0 45 = 45, p1 56 vs 55, p2 27 = 27 (±1 = phase
  arithmetic folding);
- consumer-side (full-permit) critical sections: `acquire → buffer →
  op → release` adjacent in BOTH trees, for every consumer (correction
  reads of m_i/m_ij, MMA consumption of K/V/P/acc, softmax read of qk);
- releases adjacent to their protected op in both trees.

### 13.4 The two remaining suspects (everything else eliminated)

1. **Pre/post-loop acquire/release placement** (once per kernel).
   Entry: baseline has 2 unstamped acquires landing in ROOT before
   `warp_specialize` (053 lines 109/139); ported has 7, six
   partition-stamped, landing in partition prologues (3 in default:
   053:243–269, 1 in p1: 053:458, 2 in p2: 053:572/585). Exit:
   baseline drains QK post-loop (commit p2 + acquire/release p0);
   ported uses the extra third acc semaphore (`ACC-b`) for the
   epilogue handoff — the one entity-level difference, entirely in
   this bucket. Once-per-kernel, so to matter it must act indirectly
   (what each warpgroup executes at entry / captures after
   lower-warp-group — not yet examined below dump 053).
2. **In-loop placement** (per iteration, ×512). Empty-permit acquires
   at point-of-use (baseline) vs bottom-of-loop block (ported) —
   softmax's P/m_i/m_ij, p2's K/V, p1's QK; plus wait ORDER within
   gates: PV-MMA gate `v, acc, P` (baseline) vs `acc, P, v` (ported),
   and the qk-empty wait rotation in p1. USER PRIOR: bottom-vs-top
   rotation alone was perf-neutral in past A/B experiments on other
   workloads — and §13.3 confirms it does NOT disturb pipelining here,
   so if this bucket is the cause the mechanism is runtime stalling
   (e.g., the 1-deep P ping-pong of §11.4), not schedule damage.

The entire 682 → 606 must live in these two buckets (or their
interaction with lower-warp-group/codegen). Direct A/B of either is
now cheap on 03.git since the experiment branch builds and gates
green.

## 14. Suspect decomposition on the experiment branch (11jun26, evening)

All on `triton-03.git` `egx/nvws-semaphore-insert-semas`; per-step
commits make the branch a perf bisection log:

```
601ea4ceba  ROOT-OUTSIDE placement enabled   -> 620-624 TFLOPS
c50c886658  semaphore combine disabled       -> no change (inert)
1ff94f70c8  insert-semas stack ported        -> 606-608 TFLOPS
34245cc5ef  shared baseline                  -> 681-683 TFLOPS
```

### 14.1 Semaphore combine: verified inert (ruled out)

User hypothesis check: does `combineSemaphores` (LowerAref) fire here?
IR evidence: every `semaphore.create` at dump 050 is single-buffer in
both stacks (baseline 14, ported 15 — exactly the insertion counts;
combined semaphores would be multi-buffer). Causal check: with the
combine call commented out in BOTH trees, perf is unchanged —
baseline 682.9 (vs 682.1 ON), ported 608.5 (vs 606.8 ON). Combine's
grouping needs ≥2 non-TMEM full-semaphore consumers sharing one
dominant consumer op in one partition; FA's K/V feed different MMAs
and m_i/m_ij feed different loads, so no group forms.

Side effect that mattered: the only two stamped-acquire assumptions in
the ported LowerAref (`analyzeCombinedSemaphoreGroup:791`,
`combineSemaphores:943` — the exact blockers recorded in
`fable/attr-less-acquire-release-handoff.md` §4) are inside combine.
Disabling it (commit `c50c886658`) unblocked the parked ROOT-OUTSIDE
feature with no other LowerAref change (full audit: every remaining
`getPartitionIds` site is `hasPartition`-guarded).

### 14.2 Suspect #1 measured: ROOT-OUTSIDE placement = ~15 TFLOPS

The handoff doc §3's two emitter edits applied to the ported
`InsertSemasEmitIR.cpp` (commit `601ea4ceba`): entry acquires emitted
attr-less (root block), p0-owned outside-loop ops attr-less, non-zero
outside-loop consumers keep `{P}` + tag — i.e. the old pass's
placement convention.

Verified: gate-1 lit unmodified + 4/4 runtime gates green; at the
semaphore level all 7 entry acquires are attr-less
(`logs/fa-11jun26-v3-root/trace-03root-sema.txt`), and at dump 053 the
7 entry waits sit in the ROOT block before `warp_specialize` (the
stamped version had them in the partition prologues: 3 in default,
1 in p1, 2 in p2).

| config | TFLOPS |
|---|---|
| ported, stamped (v2) | 606.3–608.5 |
| ported, ROOT-OUTSIDE (v3) | **619.7–623.9** |
| baseline | 680.6–682.9 |

Notable: a once-per-kernel placement difference was worth ~15 TFLOPS —
the mechanism is presumably not the one-time waits themselves but what
their partition stamps drag into the warp-group regions (token
captures / phase chains at region entry); not root-caused further
since the measurement stands on its own.

### 14.3 Suspect #2 measured: point-of-use in-loop acquires = ~47 TFLOPS — GAP CLOSED

Implementation (`rewriteCarriedTokensToPointOfUse` in the ported
`InsertSemasEmitIR.cpp`, post-emission fixup): every loop-carried token
matching the rotated protocol (entry acquire → iter_arg → in-loop
buffer/release uses → bottom re-acquire → yield) is rewritten to the
old pass's point-of-use shape — the in-loop acquire moves to
immediately before the token's first user, the uses take its token
directly, the entry acquire is deleted (iteration 0 consumes the
initial permit from the initially-released create), and the dead slot
is poisoned on both init and yield. Semaphores whose entry token has
uses beyond the loop init are left rotated — in FA that's exactly
`acc` (its entry token feeds the pre-loop init store), matching the
old pass, which also keeps qk/acc rotated.

Implementation lesson (cost one lit-gate iteration): the converted
token must **die inside the body**. Yielding the moved acquire's token
as a dead carry re-enters `AssignStagePhase`'s stage-propagation walk,
which only knows tokens it threaded itself
(`tokToStagePosMap.at` → assert). Poisoning the yield slot fixes it;
the partition-outputs verifier exempts poison producers.

Verified (v4 dumps): the in-loop trace is the old pass's shape —
`acquire → buffer → op → release` adjacent for M_I, M_IJ, K, V, QK, P;
gate-1 lit unmodified; 4/4 runtime gates green.

| config | TFLOPS |
|---|---|
| ported, rotated in-loop (v3) | 619.7–623.9 |
| ported, point-of-use in-loop (v4) | **667.0–671.6** |
| baseline, interleaved same session | 660.8–672.2 |

Interleaved A/B (ported/baseline alternating): 671.6 / 670.5 / 659.2 /
660.8 — the legs track each other run-for-run (ambient drift moves
both). **Parity.**

### 14.4 Final decomposition of the −75 TFLOPS

| cause (emitter placement policy) | worth |
|---|---|
| in-loop rotated re-acquires (vs point-of-use) | **~47 TFLOPS** |
| partition-stamped entry/exit acquires (vs root/attr-less) | **~15 TFLOPS** |
| residual | within noise |

Both causes are insert-semas EMISSION placement policies; the DAG, the
edges, the counts, and the per-partition pipelining were correct
throughout (§13.3). The earlier prior that bottom-vs-top rotation is
perf-neutral (true on other workloads) does not hold for this kernel's
1-deep producer-consumer ping-pongs: with the rotation, the softmax
partition stalls on the PV-MMA commit (and the correction partition's
reads) before starting the next iteration's math (§11.4); point-of-use
hides those latencies behind the body. Pipelining was unaffected
either way — the cost was pure runtime stalling.

Fix path for solid-01: port the two emitter changes back
(ROOT-OUTSIDE emission — already written as the parked feature,
`fable/attr-less-acquire-release-handoff.md` — plus the point-of-use
rewrite or a native point-of-use emission mode), with the combine
interaction resolved (combine's stamped-acquire assumptions; on the
experiment branch combine is disabled as inert for FA, but solid needs
the §5.1 tolerance guards from the handoff doc if combine stays).

Artifacts: `logs/fa-11jun26-v3-root/`, `logs/fa-11jun26-v4-inloop/`
(full per-pass dumps, traces, run outputs).

### 14.5 Back-port to solid-01 (12jun26) — REGRESSION FIXED

The three experiment-branch commits ported to solid-01 (branch
`egx/meta/sema10a-meta-new-sema-fresh-v5-fable-perf-1`), per user
ruling combine is disabled verbatim (not guarded):

| commit | content | test fallout handled |
|---|---|---|
| `83fb5f2b58` | combine disabled in lower-semaphore | 3 combine-pinning lit cases rewritten to the per-pair uncombined lowering (`warp_specialize_tma_matmul`, `ws_multibuffer_trans_chain`, `combine_two_tma_loads`) |
| `b6b717917c` | ROOT-OUTSIDE placement (un-parks the handoff-doc feature) | 16 insert_semas goldens regenerated (`gen_insert_semas_checks.py --apply`); manual CHECK-NOT + META blocks restored |
| `00095cd04a` | point-of-use in-loop acquires | 14 goldens regenerated again; same manual blocks restored |

Gates: NVWS lit suite **90/90** after each commit; the 5 runtime gates
(4 warp-spec pytest + mxfp4) all green on the final tree.

Result (06-fa.py, interleaved with the 01.git baseline, isolated
caches): **solid 668.4 / 657.5 vs baseline 669.1 / 664.6 TFLOPS** —
parity within ambient drift, up from solid's 604–622 before the port.
The FA fwd WS regression is fixed in solid-01. Notably solid reaches
baseline parity despite its older backend (no FADD2 packing): with the
stalls removed, the §10 backend deltas do not measurably bite at these
operating points.

### 14.6 Correctness regression in the point-of-use rewrite, and the fix (12jun26)

The loose convertibility guard of the point-of-use rewrite was unsound
for **mid-loop protocol acquires**. The moe mxfp4 matmuls (52 failing
block-128 cases in `fail-moe.log`, plus a machine-dependent
`run_nvws_1.sh` hang) exposed the acc pattern: the carried token covers
the init store + full-release; a MID-LOOP acquire then waits for the
consumer's tc5mma commit and its token guards the epilogue read of the
MMA result before being yielded. The rewrite took that acquire for the
bottom re-acquire (it defines the yield operand) and moved it above the
init store — stripping the `wait_barrier` off the `tmem_load` (verified
in the lowered IR: rotated form has the wait directly before the load;
converted form has none). Unsynchronized accumulator read ⇒ wrong
results or hang depending on timing. The block-16 mxfp4 gate missed it;
the failures are all block-128 shapes.

Fix (`e51ec62358`): convert only when (a) the yielded acquire's token
has no use besides the yield, (b) the iter_arg's users are exactly one
buffer followed by one release, and (c) the loop's token result is
dead. Verified: 52/52 previously-failing cases pass; FA fwd WS retained
at 662 TFLOPS (FA's slots all convert under the tightened guard); NVWS
lit 90/90 (5 goldens re-regenerated); gate battery green.

Methodology note (second occurrence this study): env-var knobs are not
part of the triton cache key — any knob A/B that shares
`~/.triton/cache` silently reuses the other leg's kernel. All
conclusive runs here used per-leg `TRITON_CACHE_DIR`.

### 14.7 Token retention re-evaluated under the new placement (12jun26) — re-parked

With the placement fixes in, the parked TOKEN RETENTION elision
(semas-report3.md Addendum A, impl `844bf8fa63`) was re-ported and
re-measured on the theory that the old ~5% regression was an artifact
of the stall-bound regime. Outcome: (a) it deadlocked `run_nvws_1.sh`
via an interaction with the point-of-use rewrite — retention's rides
depend on the rotated bottom re-acquire's program-order position, which
point-of-use removes; root-caused at IR level and fixed with the
component-wide ride guard (kept in-tree, commit below); (b) with the
guard in place and everything green (NVWS 91/91 at the time, 53 mxfp4,
gates, no hang), perf STILL regressed (user-measured) — the
pacing-point value of the "redundant" semaphore is real even in the
stall-free regime. User ruling: retention reverted (re-parked), ride
guard retained. Full record in semas-report3.md Addendum A.
