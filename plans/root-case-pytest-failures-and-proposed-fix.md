# Root Cause of the NVWS-AWS Pytest Failures and Verified Structural Fixes

Date: 2026-06-29
Base revision investigated: `373438659ba0a462978831891ea742deff2c1241`
Final verification: implementation worktree on 2026-06-29

## 1. Executive conclusion

The original eight failures were specific to the NVWS-AWS path and split into
three defects. Fixing the third exposed a fourth NVWS synchronization defect
that had previously been hidden by the compile-time rejection:

| Config | Failing cases | Observed failure | Proven NVWS defect | Structural fix area |
|---|---:|---|---|---|
| 0, H128, persistent | 2 | `dV` corruption, 67.3% mismatched elements in the representative case | InsertSemas releases the epilogue SMEM channel after `local_load`; TMA lowering then reuses that SMEM as the asynchronous store source, so the empty arrival occurs before the TMA read and wait | Extend the ACCESS-DAG ownership endpoint to the descriptor-store completion anchor |
| 1, H128, persistent | 2 | 248408 bytes required vs. 232448 bytes available | NVWS MemoryPlanner turns eight logical members from two four-member epilogue alias groups into eight circular slots, after checking the budget at a smaller depth | Transcribe Meta's planning phases and account by physical `buffer.id` groups |
| 2, H64, persistent and nonpersistent | 4 | InsertSemas rejects inconsistent `buffer.copy` values in one reuse group | InsertSemas assumes one logical ring per physical TMEM id, but Meta deliberately assigns qkT copy 1 and ppT copy 2 to one physical id | Preserve independent logical rings and coalesce only the physical backing |
| 2, H64, after the preceding fixes | hang | All warp groups block on a barrier cycle | A fused two-member SMEM epilogue group advances slots `0,0,1,1`, but releases signal `0,0,1,1` instead of successor handoff slots `0,1,1,0`; ASP also selected scalar single-phase parity for two advances at depth 2 | Author release-slot offsets in SYNC-DAG and force multiphase when the stage orbit does not cover the ring |

The original baseline and final verified result are:

```text
NVWS-AWS: 8 failed, 20 passed, 20 skipped
Meta-AWS: 28 passed, 20 skipped

final NVWS-AWS: 28 passed, 20 skipped
final Meta-AWS: 28 passed, 20 skipped
```

Evidence:

- NVWS summary: `logs/06-29jun26-recurrence-regression/full-current/stdout.log:762-772`.
- Meta summary: `logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/meta/full-stdout.log:49-66`.
- Final NVWS summary:
  `logs/06-29jun26-recurrence-regression/structural-fused-handoff-fix/nvws-full/stdout.log`.
- Final Meta summary:
  `logs/06-29jun26-recurrence-regression/structural-fused-handoff-fix/meta-full/stdout.log`.
- Both paths subsequently execute the same generic TMA lowering in
  `third_party/nvidia/backend/compiler.py:450-458`; the path selection differs
  earlier at `third_party/nvidia/backend/compiler.py:400-442`.

Therefore the report does not propose a generic TMA-lowering change. The
defects are in NVWS ownership analysis, NVWS memory planning, and the NVWS
SYNC-DAG/ASP stage protocol.

## 2. Reproduction and controls

The full NVWS run used:

```bash
env \
  -u TRITON_META_WS_USE_CHANNEL_SMEM \
  -u TRITON_USE_META_PARTITION \
  -u TRITON_USE_META_WS \
  PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python \
  TRITON_FP8_PROMOTE_TO_TMEM=0 \
  NVWS_USE_SSA_TMEM=1 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_NVWS_USE_META=1 \
  pytest -v --tb=short python/tutorials/fused-attention-ws-device-tma.py
```

The full Meta control used:

```bash
env \
  -u TRITON_NVWS_USE_META \
  -u NVWS_USE_SSA_TMEM \
  -u TRITON_FP8_PROMOTE_TO_TMEM \
  PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python \
  TRITON_META_WS_USE_CHANNEL_SMEM=0 \
  TRITON_USE_META_PARTITION=1 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_USE_META_WS=1 \
  pytest -v --tb=short python/tutorials/fused-attention-ws-device-tma.py
```

Per-configuration dumps were captured with `MLIR_ENABLE_DUMP=1` and
`MLIR_DUMP_PATH=<...>/mlir-dump.log`. `MLIR_ENABLE_DUMP` is the actual flag;
it is read in `python/src/ir.cc:1951-1954`.

The three tutorial configurations are defined at
`python/tutorials/fused-attention-ws-device-tma.py:892-928`:

- Config 0 uses explicit channel annotations from lines 666-672.
- Config 1 supplies only stage/order and delegates memory decisions to the
  planner, lines 684-690 and 906-914.
- Config 2 is the BM64 explicit-channel case, lines 674-682 and 915-927.

All three execute the same backward computation at lines 718-755. In
particular, line 736 consumes `ppT` as operand A of the dV MMA; it is not the
dV accumulator.

## 3. Failure inventory

The exact NVWS failures are:

```text
config 0, SUBTILING=False, H128, ws_persistent-bwd  -> dV mismatch
config 0, SUBTILING=True,  H128, ws_persistent-bwd  -> dV mismatch

config 1, SUBTILING=False, H128, ws_persistent-bwd  -> SMEM OOR
config 1, SUBTILING=True,  H128, ws_persistent-bwd  -> SMEM OOR

config 2, SUBTILING=False, H64, ws_persistent-bwd   -> inconsistent buffer.copy
config 2, SUBTILING=False, H64, ws-bwd              -> inconsistent buffer.copy
config 2, SUBTILING=True,  H64, ws_persistent-bwd   -> inconsistent buffer.copy
config 2, SUBTILING=True,  H64, ws-bwd              -> inconsistent buffer.copy
```

The full list is in
`logs/06-29jun26-recurrence-regression/full-current/stdout.log:762-770`.

## 4. Config 0: channel ownership ends before the TMA store is finished

### 4.1 Runtime symptom

The representative test fails at the `dV` comparison:

```text
Mismatched elements: 11287900 / 16777216 (67.3%)
Greatest absolute difference: 2.75
```

See
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/nvws/stdout.log:14-21`.

### 4.2 Before InsertSemas

NVWS data partitioning creates a cross-partition SMEM channel between the
partition-4 TMEM epilogue and the partition-2 descriptor store:

```mlir
// mlir-dump.log:259818-259822
%11 = arith.truncf %dvs_60 ... {ttg.partition = [4]}
ttg.local_store %11, %3       {ttg.partition = [4]}
%12 = ttg.local_load %3       {ttg.partition = [2]}
%13 = ttg.convert_layout %12  {ttg.partition = [2]}
tt.descriptor_store %desc_dv[...], %13 {ttg.partition = [2]}
```

Source:
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/nvws/mlir-dump.log:259808-259837`.

This channel is required because the producer and descriptor-store task are
in different NVWS partitions. The defect is not the existence of the channel;
it is where ownership is returned.

### 4.3 After InsertSemas

InsertSemas brackets the direct `local_load`, then releases the channel before
the layout conversion and descriptor store:

```mlir
// mlir-dump.log:260403-260408
%24 = nvws.semaphore.acquire %5 {ttg.partition = [2]}
%25:4 = nvws.semaphore.buffer %5, %24 ...
%26 = ttg.local_load %25#0
nvws.semaphore.release %6, %24 [#nvws.async_op<none>]
%27 = ttg.convert_layout %26
tt.descriptor_store %desc_dv[...], %27
```

Source:
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/nvws/mlir-dump.log:260399-260441`.

This output followed the base implementation named at the top of this report:

- `InsertSemasAccessDag.cpp:285-286` records `local_load` as the channel read.
- The fallback at lines 317-325 only sees operations with a tracked memdesc
  operand. `descriptor_store` consumes a tensor SSA value, not the channel
  memdesc, so it creates no channel touch.
- `InsertSemasEmitIR.cpp:881-903` emits a Release at its SYNC-DAG chain
  position, immediately after the source row.

The emitter was behaving mechanically. The missing fact was in stage 1: the
ACCESS-DAG did not represent the downstream operation through which the
channel's physical lifetime is extended.

The fixed implementation records that separate frontier in
`InsertSemasAccessDag.cpp:357-429` and makes the rendered Access row end there
in `InsertSemasEmitIR.cpp:660-734`.

### 4.4 The unsafe order is explicit after TMA lowering

Before generic TMA lowering, the order is already:

```mlir
ttng.wait_barrier ...
%45 = ttg.local_load %channel
ttng.arrive_barrier %empty, 1
tt.descriptor_store ..., %45
```

See
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/nvws/mlir-dump.log:304830-304836`.

Generic TMA lowering reuses the channel allocation as the asynchronous TMA
source and preserves the empty arrival before that use:

```mlir
// mlir-dump.log:306099-306106
ttng.wait_barrier %43, %42
%44 = ttg.memdesc_index %arg85[%39]       // channel slot
ttng.arrive_barrier %45, 1               // producer may overwrite now
ttng.fence_async_shared
ttng.async_tma_copy_local_to_global ... %44
ttng.async_tma_store_wait {pendings = 0}
```

The same order repeats for all four dV subtiles at lines 306113-306148 and
for dK starting at line 306149.

This is a concrete ownership violation: the producer can observe the empty
arrival and overwrite `%44` while the TMA engine still reads `%44`.

### 4.5 Why Meta-AWS passes

Meta keeps the TMEM loads, register conversions, and descriptor stores in one
async task. It does not insert the NVWS cross-partition epilogue channel:

```mlir
// Meta mlir-dump.log:269867-269894
%dv_203 = ttng.tmem_load ...
...
%4 = arith.truncf %dv_204
...
tt.descriptor_store ..., %4
```

After the same generic TMA lowering, Meta creates a private source allocation
for each store:

```mlir
// Meta mlir-dump.log:270878-270893
%7 = ttg.local_alloc %4
ttng.fence_async_shared
ttng.async_tma_copy_local_to_global ... %7
ttng.async_tma_store_wait {pendings = 0}
```

No independently scheduled producer can overwrite `%7` before the wait.
Sources:

- `logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/meta/mlir-dump.log:269867-269894`
- `logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/meta/mlir-dump.log:270854-270893`

### 4.6 Structural fix

The fix belongs in the NVWS synchronization model, before emission:

1. During ACCESS-DAG discovery, detect a managed `local_load` whose tensor
   result reaches `tt.descriptor_store` through only ownership-preserving
   register operations such as `convert_layout`.
2. Keep `local_load` as the real memory touch and retargeting point.
3. Record a separate `completionAnchor` on that Access row, or a lifetime-only
   row with `retarget = false`, anchored on the terminal descriptor store.
4. Build the release edge from that completion anchor. EMIT-IR still renders
   the SYNC-DAG mechanically; it does not search for or move the release.
5. After TMA lowering, require the concrete order:

```text
wait full -> TMA copy from channel -> TMA store wait -> arrive empty
```

Do not classify every tensor user as a channel touch. The trace must be a
closed, explicit set of forwarding operations ending in a descriptor store;
unsupported fan-out or control-flow escape should diagnose rather than guess.

This requires a plan/spec amendment before implementation. The current spec
says:

- one Access node per terminal memory access,
  `fable/semas-report3.md:172-178`;
- `local_load` is a synchronous reader with `<none>` completion,
  `fable/semas-report3.md:72-80`;
- releases are placed immediately after their source row and may not be moved
  post-emission, `fable/semas-report3.md:918-931,963-967`.

The amendment must distinguish the direct memory touch from the physical
ownership-completion anchor. A post-hoc emitter move would violate the plan
and should not be used.

### 4.7 Required regression test

Add a synthetic NVWS lit test with this exact shape:

```text
P4: local_store channel
P2: local_load channel -> convert_layout -> descriptor_store
```

Check both boundaries:

- After InsertSemas, the consumer-side release is after `descriptor_store`.
- After LowerSemaphore plus TMA lowering, `arrive empty` is after
  `async_tma_store_wait`.

Then run both config-0 H128 persistent tests with `SUBTILING=False/True`.

## 5. Config 1: eight aliases become an unbudgeted eight-slot ring

### 5.1 Runtime symptom

The representative test fails before launch:

```text
Required: 248408, Hardware limit: 232448
```

See
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config1/nvws/stdout.log:30-39`.

### 5.2 Final NVWS plan

After NVWS MemoryPlanner, all eight epilogue subtiles have one `buffer.id`,
eight copies, and distinct circular starts:

```mlir
// mlir-dump.log:259233-259240
%3  = ttg.local_alloc {buffer.circular, buffer.copy = 8,
                       buffer.id = 5, buffer.start = 0} ...
%4  = ttg.local_alloc {buffer.circular, buffer.copy = 8,
                       buffer.id = 5, buffer.start = 1} ...
...
%10 = ttg.local_alloc {buffer.circular, buffer.copy = 8,
                       buffer.id = 5, buffer.start = 7} ...
```

The physical allocation confirms the depth:

```mlir
// mlir-dump.log:337520
%3 = ttg.local_alloc ... -> !ttg.memdesc<8x128x32xf16, ...>
```

Each slot is `128 * 32 * 2 = 8192` bytes. The ring therefore consumes:

```text
8 * 8192 = 65536 bytes
```

Sources:

- `logs/06-29jun26-recurrence-regression/meta-vs-nvws-config1/nvws/mlir-dump.log:259228-259240`
- `logs/06-29jun26-recurrence-regression/meta-vs-nvws-config1/nvws/mlir-dump.log:337510-337520`

### 5.3 Eight logical members represent two physical epilogue values

With config 0's explicit channel annotations, the same epilogue has two
four-member exact-alias groups:

```mlir
// dV subtiles: four exact aliases
%3 = ttg.local_alloc {buffer.copy = 1, buffer.id = 9} ...
%4 = ttg.local_alloc {buffer.copy = 1, buffer.id = 9} ...
%5 = ttg.local_alloc {buffer.copy = 1, buffer.id = 9} ...
%6 = ttg.local_alloc {buffer.copy = 1, buffer.id = 9} ...

// dK subtiles: four exact aliases
%7  = ttg.local_alloc {buffer.copy = 1, buffer.id = 13} ...
%8  = ttg.local_alloc {buffer.copy = 1, buffer.id = 13} ...
%9  = ttg.local_alloc {buffer.copy = 1, buffer.id = 13} ...
%10 = ttg.local_alloc {buffer.copy = 1, buffer.id = 13} ...
```

See
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config0/nvws/mlir-dump.log:259233-259240`.

That representation costs two physical slots, `2 * 8192 = 16384` bytes,
not eight slots. The difference is 49152 bytes.

### 5.4 Source-level mechanism

At the base revision, the defect was the interaction of two planner phases in
`third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp`:

1. `fuseEpilogueBuffers()` at lines 1387-1407 correctly gives disjoint
   subtiles from the same original load the same `buffer.id`.
2. `coalesceCircularReuseCandidates()` at lines 1410-1435 iterates every
   logical `LocalBuffer`, including members already fused into physical alias
   groups.
3. Its budget check at line 1428 occurs while the prospective group still has
   the old copy depth.
4. `assignCircularStarts()` later sets
   `requiredCopies = group.size()` at lines 1439-1461. Because the group has
   eight logical members, it raises every member to copy 8 and assigns eight
   distinct starts.
5. There is no budget check after that expansion; `emitAttrs()` immediately
   writes the result at lines 1465-1481.

The planner therefore checks one cost and emits a larger cost.

### 5.5 Meta-AWS reference behavior

Meta's planner preserves epilogue exact aliases in phase 3.5. P2 buffers are
excluded from the P0/P1 circular phase, then each fused P2 group may grow
uniformly in phase 4.5 while checking the final physical-group cost:

- Exact-alias fusion:
  `third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp:1364-1368`.
- Only P0 and P1 participate in phase-4 circular grouping:
  `WSMemoryPlanner.cpp:1370-1526`.
- Fused P2 groups grow uniformly with each attempted depth budgeted and
  reverted on overflow: `WSMemoryPlanner.cpp:1200-1260` and
  `WSMemoryPlanner.cpp:1528-1532`.

The passing Meta config-1 module reports `ttg.shared = 222632` and kernel
allocation offset 222304, both below the 232448-byte limit:
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config1/meta/mlir-dump.log:348935-348938`.

### 5.6 Structural fix

The implemented NVWS MemoryPlanner now follows Meta's phase order:

1. Create one record per local allocation and apply pinned annotations.
2. Apply the budgeted cross-stage minimum and classify unpinned records using
   Meta's P0/P1/P2 predicates, including the two-non-trivial-dimension test.
3. Fuse compatible P2 epilogue records by original TMEM-load provenance.
4. Run circular grouping only for an exactly two-record P0 or P1 priority;
   Meta's odd result remains one NVWS circular group and its even result splits.
5. Grow each fused P2 physical group uniformly, checking the physical-group
   cost at every attempted final depth.
6. Validate common depth and circular starts before emitting NVWS downstream
   attributes.

The phase transcription is in `MemoryPlanner.cpp:1329-1478`, P2 fusion and
growth are at `MemoryPlanner.cpp:1568-1620`, and the downstream group
postconditions are checked at `MemoryPlanner.cpp:1645-1695`.

For the historical failure, preserving two physical epilogue groups instead
of one eight-member circular ring removes the unbudgeted 65536-byte shape.
Their final copy depth is whatever Meta phase 4.5 accepts; it is not assumed to
remain one.

### 5.7 Required regression tests

Add a planner lit test with:

- four disjoint subtiles derived from one original load;
- four disjoint subtiles derived from a second original load;
- circular reuse enabled;
- a budget that exposes any post-check copy-depth expansion.

Check that each four-member alias set remains one physical group, no
eight-member circular ring is emitted, and every optional final depth is
accepted using Meta's physical-group budget equation. Then run both config-1
H128 persistent tests.

## 6. Config 2: one physical TMEM id contains two logical ring depths

Correction: the earlier diagnosis that `ppT` was incorrectly grown because it
is MMA operand A is revoked. Meta-AWS intentionally produces the same mixed
depth. The defect was NVWS InsertSemas treating a physical `buffer.id` as if it
implied one logical semaphore ring.

### 6.1 Compile-time symptom

InsertSemas reports:

```text
nvws-insert-semas: allocs in one planned reuse group have inconsistent
buffer.copy values
```

The diagnostic points to `tl.dot(ppT, do, ...)` at tutorial line 736. See
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config2/nvws/stderr.log:1572-1581`.

The diagnostic reflected the old one-logical-ring-per-`buffer.id` assumption.
That assumption was the defect. Uniform depth remains required for local groups
and ordinary same-depth TMEM groups, but a proved mixed-depth TMEM alias must be
split into independent logical DAGs before the uniform-depth check.

### 6.2 The mixed depths are intentional

The qk accumulator is copy 1 at physical `buffer.id = 2`:

```mlir
// mlir-dump.log:186914
%qkT, %qkT_tok = ttng.tmem_alloc {
  buffer.copy = 1, buffer.id = 2, buffer.offset = 0
} ... !ttg.memdesc<128x64xf32, ...>
```

The derived probability tile `ppT` is converted to a sourceful TMEM alloc. It
is operand A of the dV MMA and receives two logical FP16 slots while reusing
the same physical columns:

```mlir
// mlir-dump.log:186941-186947
%ppT = arith.truncf %pT_81 ...
%dv_82 = ttng.tmem_alloc %ppT {
  buffer.copy = 2, buffer.id = 2, buffer.offset = 0
} ... !ttg.memdesc<128x64xf16, ...>
...
%dv_87 = ttng.tc_gen5_mma %dv_82, %do, %dv[...] ...
//                           A        B    accumulator D
```

Source:
`logs/06-29jun26-recurrence-regression/meta-vs-nvws-config2/nvws/mlir-dump.log:186914-186947`.

This is not invalid growth. One FP32 qkT slot occupies the same columns as two
FP16 ppT slots. The logical channels require different ring depths even though
their physical storage aliases.

### 6.3 Source-level mechanism

Meta's round-robin TMEM policy deliberately increases the sourceful ppT
channel when it feeds the loop-carried dV MMA. NVWS MemoryPlanner must preserve
that decision.

The failure occurred later in `collectGroups()`: all allocations with one TMEM
`buffer.id` were placed into one `GroupDag`, and `getPlannedBufferCopy()` then
required every member to have the same depth. That requirement is valid for a
single logical channel, but not for Meta's qkT/ppT physical packing.

### 6.4 Meta-AWS reference behavior

Meta keeps the physical QK allocation at one FP32 copy and marks it as sharing
group 0:

```mlir
// Meta mlir-dump.log:221517
%qkT_97 = ttng.tmem_alloc {
  allocation.shareGroup = 0, buffer.copy = 1, buffer.id = 2
} ... !ttg.memdesc<1x128x64xf32, ...>
```

InsertAllocas then reinterprets that one FP32 allocation as a two-slot FP16
view. The computation partition indexes the view and stores `ppT`; the GEMM
partition indexes the same view and consumes it as dV operand A:

```mlir
// Meta mlir-dump.log:229644-229652
%qkT_178 = ttg.memdesc_reinterpret %qkT_177
  : !ttg.memdesc<1x128x32xf32, ...>
 -> !ttg.memdesc<2x128x64xf16, ...>
%ppT_179 = ttg.memdesc_index %qkT_178[%c0_i32_155] ...
ttng.tc_gen5_mma %ppT_179, %do_170, %dv_180, ...

// Meta mlir-dump.log:229893-229898
%qkT_197 = ttg.memdesc_reinterpret %qkT_196
  : !ttg.memdesc<1x128x32xf32, ...>
 -> !ttg.memdesc<2x128x64xf16, ...>
%ppT_198 = ttg.memdesc_index %qkT_197[%c0_i32_157] ...
ttng.tmem_store %ppT_195, %ppT_198, %curr_m
```

This is the same physical shape represented in NVWS planning as `qkT copy=1`,
`ppT copy=2`: one FP32 allocation becomes two FP16 logical slots. Meta's
generated IR therefore proves that the mixed logical depth is reference
behavior, not evidence of a planner error.

### 6.5 Structural fix

For one physical TMEM id with distinct authored depths:

1. Build one logical `GroupDag`, ownership protocol, semaphore ring, and
   backing depth per allocation.
2. Require the exact two-channel alternating ownership proof: each channel has
   one writer and one reader; the owners cross; and the direct scheduled-loop
   order is `A.write -> B.read` and `A.read -> B.write`.
3. Keep the logical semaphore creates independent.
4. After rendering, select the unique physical owner by span and element-width
   containment and replace the reuser backing with a checked
   subslice/reinterpretation of that owner.

Local-memory groups still require one common depth. TMEM mixed depths are
accepted only by this proved two-channel adapter; they are never normalized to
the maximum depth.

The implementation splits mixed-depth buckets at
`InsertSemasAccessDag.cpp:100-130`, proves the alternating lifecycle at
`InsertSemasSyncDag.cpp:2554-2682`, and coalesces only the physical backing at
`InsertSemasEmitIR.cpp:990-1082`.

### 6.6 Required regression tests

Check that qkT remains a one-slot logical ring, ppT remains a two-slot logical
ring, both lower to one physical TMEM allocation, and their alternating
handoffs remain pipeline-legal. Then run all four config-2 H64 tests, covering
persistent/nonpersistent and both SUBTILING values.

## 7. Final config-2 hang: fused SMEM releases addressed the wrong slots

### 7.1 Exact ownership shape

After the earlier fixes, the dV epilogue contains two exact-alias local
members at depth 2. Its Access and SYNC DAGs are:

```text
W m0 P4 -> R m0 P2 -> W m1 P4 -> R m1 P2

a S3 -> W m0 -> r S0
a S0 -> R m0 -> r S1
a S1 -> W m1 -> r S2
a S2 -> R m1 -> r S3
```

This is captured at
`logs/06-29jun26-recurrence-regression/post-meta-parity-fix/config2-point-of-use-dump/mlir-dump.log:221784-221822`.

ASP advances the shared data cursor only at fresh writes. The access slots in
one outer iteration are therefore:

```text
W m0 = 0, R m0 = 0, W m1 = 1, R m1 = 1
```

### 7.2 Old incorrect lowering

Before the fix, every release inherited its carrier token's data slot. The
old post-ASP IR shows the first reader waiting S0 at `%48`, then arriving S1
at the same `%48`; the next writer waits S1 at `%66 = (%48 + 1) mod 2`:

- old `R m0` wait and `r S1[%48]`:
  `logs/06-29jun26-recurrence-regression/post-meta-parity-fix/config2-point-of-use-dump/mlir-dump.log:224146-224153`;
- old `W m1` computes `%66` and waits S1 at `%66`:
  `logs/06-29jun26-recurrence-regression/post-meta-parity-fix/config2-point-of-use-dump/mlir-dump.log:224155-224169`;
- old `R m1` arrives entry S3 at `%66`, while next `W m0` returns to `%48`:
  `logs/06-29jun26-recurrence-regression/post-meta-parity-fix/config2-point-of-use-dump/mlir-dump.log:224172-224184`.

The old physical protocol was therefore:

```text
access slots:   0, 0, 1, 1
release slots:  0, 0, 1, 1
needed slots:   0, 1, 1, 0
```

CUDA-GDB observed the resulting cycle: computation waited on the S3 slot-0
barrier while the epilogue waited on the S2 slot-1 barrier. The P, DO, and DQ
waits were downstream cycle participants, not the primary mismatched handoff.

ASP also selected scalar single-phase parity. This group advances twice per
outer iteration at depth 2, so `gcd(depth, advances) = 2`; each transition
semaphore remains in one slot class. A semaphore fixed at slot 1 never reaches
the slot-0 event where scalar single-phase toggles parity.

### 7.3 Structural correction

SYNC-DAG now computes the access ordinal and authors a signed stage offset on
every release of an exact-alias, multistage local group:

1. Same-iteration handoffs target the satisfied acquire's ordinal.
2. A loop-closing release keeps its source slot when that slot occurs in the
   future acquire orbit.
3. If the source slot is outside that orbit, it targets the next iteration's
   destination slot.
4. EMIT-IR only materializes the authored node field. It does not infer or
   move synchronization.

ASP consumes those offsets and uses multiphase parity. Independently, its
single-phase eligibility now rejects `gcd(depth, advances) != 1`.

The slot analysis is `InsertSemasSyncDag.cpp:2844-2942`; ASP's independent
parity-orbit guard is `AssignStagePhase.cpp:195-257`. EMIT-IR only transcribes
the authored node field at `InsertSemasEmitIR.cpp:867-900`.

The corrected pre-ASP releases use offsets `0,1,0,1` at
`logs/06-29jun26-recurrence-regression/structural-fused-handoff-fix/config2-selected/mlir-dump.log:219961-219989`.
Post-ASP, the first reader arrives S1 at `(%48 + 1) mod 2` and the second
writer waits S1 at the next slot `%73`; the final reader similarly arrives S3
at `(%73 + 1) mod 2`. See the same log at lines 220940-221015.

### 7.4 Pipeline and runtime verification

The post-pipeline IR still contains qkT, dpT, dV, dQ, and dK MMA operations,
including pipelined prologue/body copies at lines 226844-227009 of the fresh
dump. The runtime pass is therefore not caused by suppressing MMA pipelining.

The reduced regression is
`test/NVWS/insert_semas_fused_alias_handoff.mlir`. It checks authored
pre-ASP offsets, post-ASP successor slots, and multiphase bit arithmetic.

## 8. Implemented fix sequence

The implemented NVWS-only sequence is:

1. Port Meta's SMEM planning phase order, physical-group budget accounting,
   epilogue fusion, and TMEM copy policy into NVWS MemoryPlanner while retaining
   NVWS downstream attributes.
2. Preserve Meta's mixed-depth qkT/ppT TMEM plan with independent logical DAGs
   and one checked physical backing.
3. Extend local-load ownership through descriptor-store completion.
4. Author exact-alias SMEM handoff slot offsets in SYNC-DAG and make ASP parity
   orbit-aware.

No generic TMA lowering or tutorial source was changed by this implementation.

## 9. Verification results

For each source change, follow `AGENTS.md`: build first, then lit.

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12
ninja triton triton-opt
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v <targeted-tests>
```

Verified on 2026-06-29:

1. `ninja triton triton-opt` passed.
2. All 103 tests under build-tree `test/NVWS` passed.
3. The previously hanging config-2 selector passed in 14.46 seconds with a
   fresh 43 MiB MLIR dump.
4. Final full NVWS tutorial rerun: `28 passed, 20 skipped` in 30.23 seconds.
5. Final full Meta-AWS control rerun: `28 passed, 20 skipped` in 23.19 seconds.
6. Fresh post-pipeline config-2 IR retains all five MMA families.

## 10. Rejected fixes

The evidence rules out these approaches:

- Do not change generic TMA lowering for this matrix. Both paths use it, and
  Meta supplies a valid lifetime while NVWS supplies an invalid one.
- Do not move releases in an InsertSemas post-pass or in EMIT-IR. Protocol
  decisions must be represented in the ACCESS/OWNER/SYNC DAGs.
- Do not increase the hardware SMEM limit, reduce the kernel shape, or silently
  accept a planner plan over budget.
- Do not assign each epilogue alias a separate circular slot.
- Do not repair mixed-depth TMEM groups by setting every member to the maximum
  copy depth. That destroys Meta's distinct qkT/ppT logical rings.
- Do not weaken the uniform-depth diagnostic for local or ordinary TMEM
  groups. Mixed-depth TMEM aliases must first pass the explicit alternating
  reuse proof and split into independent logical groups.

## 11. Final root-cause statement

The complete failure set came from four NVWS invariants that were missing:

1. A shared-memory source is not returned to its producer until its final
   asynchronous consumer has completed.
2. Memory budget and circular depth are computed over physical reuse units,
   not logical alias count, and are checked at the emitted depth.
3. One physical TMEM id may contain independent logical channels with distinct
   copy depths; synchronization depth is not inferred from physical identity.
4. A release signals the physical slot consumed by its satisfied acquire, and
   parity tracks every slot orbit that can be reused.

Restoring those invariants makes the full NVWS and Meta matrices pass without
changing the tutorial, generic TMA lowering, or Meta-AWS.
