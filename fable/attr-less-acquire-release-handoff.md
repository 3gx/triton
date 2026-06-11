# Handoff: attr-less (root-outside) semaphore acquire/release emission

Status: **PARKED** (user ruling 10jun26). Implemented and verified at the
insert-semas level; blocked by partition-attr assumptions in the shared
downstream lowering (`LowerAref.cpp`). This doc carries the feature
definition, examples, the exact emitter changes, and the issues that must
be solved before enabling.

## 1. Feature: the ROOT-OUTSIDE rule

Stamping rules for `nvws.semaphore.*` ops emitted by `nvws-insert-semas`:

- **Inside the tt.ws loop** — unchanged, mandatory: every sync op carries
  exactly its wave owner's `ttg.partition` (partition-loops routes child
  ops by it; wave locality guarantees acquire/buffer/release on one token
  are one partition).
- **Outside the tt.ws loop — the default is ROOT (attr-less).**
  Annotation `{P}` + `ttg.warp_specialize.tag` is emitted ONLY when P is
  non-zero, i.e. the op consumes a token/phase produced by a non-zero
  partition's chain and must be routed into that warp-group region so its
  stage/phase SSA chain stays local.
- **Entry acquires are always attr-less.** They are phase SOURCES: the
  one-time wait passes off the initial permit (create released), there is
  no incoming phase to communicate, and the token reaches the partitioned
  loop clones by plain SSA capture (partition-loops leaves unannotated
  ops in the root block; `AssignStagePhase` buckets them as `pid=-1` by
  design).
- **Partition 0 and root are one cost domain** (the default warpgroup
  executes the root block), so a `{0}`-owned op outside the loop emits
  attr-less; stamping it `{p0, tag}` distinguishes nothing.

Rationale (the cost mechanism): after partition-loops, stage/phase values
are SSA chains living inside the owning warp-group region. Annotating an
outside-loop phase-CONSUMER (release/buffer with a non-zero-partition
token) keeps it inside that region; unannotated it would force the phase
value to be exported across the warp_group boundary into partition 0 —
real cross-warpgroup communication that exists only for bookkeeping.
Phase SOURCES (entry acquires) have nothing to export, ever.

## 2. Examples

### 2.1 Entry acquire (operand buffer, first toucher {2})
```mlir
// before (current behavior):
%t = nvws.semaphore.acquire %S {ttg.partition = array<i32: 2>,
                                ttg.warp_specialize.tag = 0 : i32}
// after (this feature):
%t = nvws.semaphore.acquire %S          // attr-less; stays in root block
scf.for ... iter_args(%a = %t) ...      // token captured into p2's clone
```
The DAG already says `a S8 root ; entry` — the feature makes the emitted
IR match the DAG (the dump-faithfulness gap that motivated the ruling).
`inheritStamp` (the component's first access owner) remains a recorded
analysis fact in the SEMAS dump; it is no longer an emission stamp.

### 2.2 Post-loop release, token from an in-loop acquire {1} (NON-zero)
```mlir
// before AND after — unchanged; this is what annotation is FOR:
%r:2 = scf.for ...                       // carrier yielded by p1's chain
nvws.semaphore.release %S, %r#1 [#nvws.async_op<none>]
    {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
```
Partition-loops moves it into p1's warp-group region where S's phase
chain (threaded out of p1's cloned loop results) already lives.

### 2.3 Post-loop release, token from an in-loop acquire {0} (partition ZERO)
```mlir
// before:
nvws.semaphore.release %S, %r#1 [#nvws.async_op<none>]
    {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
// after:
nvws.semaphore.release %S, %r#1 [#nvws.async_op<none>]   // root
```
p0 == root cost domain; the stamp distinguished nothing.

### 2.4 Post-WS root-read tail (D-shape) — unchanged, already attr-less
```mlir
%t2 = nvws.semaphore.acquire %S_ready
%v  = nvws.semaphore.buffer %S_ready, %t2
%x  = ttng.tmem_load %v
nvws.semaphore.release %S_free, %t2 [#nvws.async_op<none>]
```
Root-owned rows (the WS scope barrier makes all outside access rows
root) already emit attr-less today; the gates prove this path works,
including in `LowerAref` (the conditional-acc precedent).

### 2.5 Outside-loop buffer from a non-zero token — unchanged, annotated
```mlir
%v = nvws.semaphore.buffer %S, %tok {ttg.partition = array<i32: 2>,
                                     ttg.warp_specialize.tag = 0 : i32}
```
Same reasoning as 2.2: the view's stage index is part of p2's chain.

## 3. The implementation (exact, for re-enabling)

Both edits in `InsertSemasEmitIR.h`; the working tree at handoff time has
them REVERTED. The verifiers need no change: the post-emit token/view
locality subpass already exempts attr-less acquires as the sanctioned
seed (root-produced tokens consumed by root ops pass; annotated consumers
must match their token's acquire partition).

1. `emitInto` (the createInto wrapper), outside-WS-loop branch:
```cpp
if (!forOp) {                       // op landed outside any WS-tagged loop
  if (owner->first == 0)
    op->removeAttr(gpu::kPartitionAttrName);   // p0 == root: attr-less
  else
    gpu::setWarpSpecializeTag(op, owner->second); // non-zero: keep {P}+tag
}
```
2. `emitEntryAcquires`: pass `Owner()` (attr-less) instead of
   `s.inheritStamp` to `emitInto` for the pre-loop entry instances.

Verified with the feature ON: corpus 23/23 emission verifier-clean and
deterministic; stage-3 wave-locality + closure verifiers green; post-emit
token/view locality green; `AssignStagePhase` (`pid=-1` bucket,
`AssignStagePhase.cpp:126-134`) and `TritonGPUPartitionLoops` (attr-less
ops stay in the root block, `PartitionLoops.cpp:459`) handle the shape by
design.

## 4. The blocker: LowerAref assumes stamped acquires

With the feature ON, the AWS sub-pipeline crashes inside
`NVWSLowerSemaphore`/`LowerAref` with
`dyn_cast on a non-existent value (DenseArrayAttr)`:

- **`LowerAref.cpp:752`**
  `producerPartitionIds.push_back(getPartitionIds(prodAcquire).front());`
  — unguarded, executed for every `SemaphoreAcquireOp` user of an
  empty-side semaphore. An attr-less producer acquire (our new entry
  form) has no `ttg.partition` attr → assertion crash.
- **`LowerAref.cpp:905`** `assert(hasPartition(acquireOp));` in the
  acquire-combining step — same class.

Gate signature with the feature ON: gate 1
(`automatic-warp-specialization.mlir`) FAILS; gate-2 `tma_matmul`,
`tma_matmul_persistent`, `attention_persistent_forward` FAIL;
`attention_forward` passes; `run_nvws.sh` FAILS, `run_nvws_1.sh` passes —
i.e. the crash fires exactly where the producer-acquire walk meets an
attr-less entry, and shapes that avoid that walk are unaffected.

## 5. What must be solved before enabling

1. **Teach `LowerAref` the attr-less convention** (attr-less acquire ⇒
   root/default warpgroup), consistent with `AssignStagePhase`'s
   `pid=-1`. Concretely: at `:752` either skip attr-less acquires in the
   producer-partition walk or contribute partition 0; at `:905` skip
   attr-less acquires from the combining groups (it is an optimization
   pass — exclusion is always sound). Audit the remaining
   `getPartitionIds` call sites in the file (`:77`, `:391` is guarded,
   `:500`) for the same assumption. These are 2-3-line tolerance guards,
   additive for the existing producers (`InsertSemaphore`,
   `InsertAllocas` always stamp), but `LowerAref` is a SHARED pass —
   the change needs its own review and the full gate battery.
2. **Arrive predication audit for root releases** (example 2.3): a
   release left in the root block executes on all warps of the CTA; the
   existing root-owned D-shape releases pass the gates, so election
   demonstrably works for those shapes, but the predication path in the
   lowering has not been audited for the general case.
3. **Re-run everything**: corpus + determinism, the three lits
   (`transitive_reduction`, `release_count`, gate 1), the four gate-2
   pytest cases, both gate-3 scripts.
4. Spec/plan: the ROOT-OUTSIDE sections in `fable/semas-report3.md`
   (stamping rules + the entry-acquire inherit paragraph) are written and
   marked PARKED; flip the marker when enabling.

## 6. Pointers

- Ruling conversation artifacts: entry acquires root
  ("semaphore can be used in root outside tt.ws; once it is in p1 it is
  always in p1"); annotation-for-cost ("if release uses tok produced
  outside partition 0, we want release outside tt.ws annotated, otherwise
  safe in root — stage/phase would need to be computed in partition 0").
- M3 acquirer-class rule already allows {root} ∪ {one partition} per
  semaphore — designed for exactly this split.
- Downstream behavior references: `PartitionLoops.cpp:457-470` (stamped
  outside-loop ops are MOVED into the warp-group partition region;
  attr-less ops stay in root; partition-without-tag asserts),
  `AssignStagePhase.cpp:126-134` (`pid=-1`), `LowerAref.cpp:752/:905`
  (the blockers).
