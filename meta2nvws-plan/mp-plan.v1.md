# NVWS TMEM Memory Planner: Virtual Hoisted Storage Semantics

## Objective

Enhance the NVWS TMEM memory planner so it can consume unhoisted NVWS IR and
make the same TMEM allocation decisions that the Meta/Hopper memory planner
makes after TMEM alloc/store hoisting and splitting.

This is a planner-only semantic normalization. The implementation must not
actually move, hoist, split, clone, or erase `ttng.tmem_alloc` / `ttng.tmem_store`
IR. It only changes how `NVWSMemoryPlanner` models TMEM storage and TMEM value
lifetimes before assigning:

```text
buffer.id
buffer.copy
buffer.offset
```

Primary implementation file:

```text
third_party/nvidia/lib/Dialect/NVWS/Transforms/MemoryPlanner.cpp
```

## Due Diligence Summary

The Hopper/Meta path relevant to this plan is:

```text
doHoistLoopInvariantTMEMStore
doMemoryPlanner
```

The Hopper memory planner first collects producer/consumer channels with
`collectPostChannels`, then `MemoryPlannerTmem` computes TMEM liveness from
channel users. The important detail is that Meta's useful TMEM value liveness
comes from producer/consumer relationships, not from the physical position of
the hoisted `ttng.tmem_alloc` op.

In other words, Meta can have storage allocated near the top:

```mlir
%buf = ttng.tmem_alloc
```

while the useful value becomes live later:

```mlir
ttng.tmem_store %src, %buf
...
consumer(%buf)
```

The current NVWS planner does not fully model that split. It builds one
`TmemBuffer` from the real `ttng.tmem_alloc` op position, and for sourceful
allocs with no explicit producer channel it can treat the alloc as channel-less.
That allows overly permissive reuse through paths such as `!hasBothChannels`.

The desired NVWS behavior is therefore:

```text
all TMEM storage is planned as if alloc storage were hoisted;
TMEM value liveness is planned from actual/virtual producers to consumers;
sourceful tmem_alloc %src is treated as a virtual tmem_store %src, %alloc.
```

## Concrete Motivation

Current NVWS after-memory-planner output for the FA case can contain:

```mlir
%acc_101 = ttng.tmem_alloc %p_99 {
  buffer.id = 5 : i32,
  buffer.offset = 0 : i32
} : (tensor<128x128xf16, #linear>)
    -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>

%acc_102 = ttng.tmem_alloc %p_100 {
  buffer.id = 5 : i32,
  buffer.offset = 64 : i32
} : (tensor<128x128xf16, #linear>)
    -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory>
```

This is not the Meta allocation decision.

The corresponding Meta after-memory-planner dump has hoisted storage plus
explicit stores:

```mlir
%acc_0 = ttng.tmem_alloc {
  buffer.id = 8 : i32,
  buffer.offset = 0 : i32
} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>

%acc_1 = ttng.tmem_alloc {
  buffer.id = 7 : i32,
  buffer.offset = 0 : i32
} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>

ttng.tmem_store %acc_152, %acc_1, %true
ttng.tmem_store %acc_153, %acc_0, %true
```

NVWS should keep its input IR shape:

```mlir
%acc_101 = ttng.tmem_alloc %p_99
%acc_102 = ttng.tmem_alloc %p_100
```

but plan storage as if it had the Meta split/hoisted shape. For the current FA
dump, that means:

```text
partition-5 sourceful f16 operand maps to the qk_0 reuse group
partition-4 sourceful f16 operand maps to the qk_1 reuse group
the two sourceful f16 operands remain distinct from each other
both sourceful f16 operands have offset 0
```

## Scope

This plan is for TMEM memory-planner semantics in NVWS, independent of where the
real `ttng.tmem_alloc` appears in the IR.

The semantic model applies to:

- sourceful `ttng.tmem_alloc %src`;
- sourceless `ttng.tmem_alloc`;
- explicit `ttng.tmem_store` producers;
- MMA overwrite producers, such as operand-D `useAccumulator = false`;
- MMA accumulator update flows that both consume and produce operand-D values;
- allocs inside or outside `tt.warp_specialize` loops.

The initial validation focus remains the FA/NVWS WS input, but the plan must not
encode a WS-only or sourceful-only semantic model. Hoisting should become
unnecessary for TMEM memory-planner decisions.

## Non-Goals

This plan does not:

- move or hoist `ttng.tmem_alloc` ops;
- split `ttng.tmem_alloc %src` into real `ttng.tmem_alloc` plus
  `ttng.tmem_store` IR;
- change `NVWSHoistTmemStore`;
- change `NVWSInsertTmemSemaphore`;
- require byte-for-byte MLIR equivalence with Meta output;
- require creating Meta-only scratch TMEM allocations that do not exist in NVWS
  input IR;
- create new IR-level `tmem.start` / `tmem.end` annotations solely to satisfy
  memory-planner liveness.

## Required Semantic Model

### 1. TMEM Storage Is Virtual-Hoisted

Every `ttng.tmem_alloc` has a planner-only storage object:

```text
TmemStorage {
  realAllocOp
  virtualStorageAnchor
  size
  final buffer.id/copy/offset
}
```

`virtualStorageAnchor` is a planner ordering concept only. It represents the
storage allocation as if it were hoisted to the canonical Meta allocation point.
The real alloc op is not moved and still receives the final buffer attributes.

The storage anchor must not be used as value liveness.

### 2. TMEM Values Are Producer-Boundary Live Ranges

Each storage object has one or more value live ranges:

```text
TmemValueRange {
  storage
  producer
  consumers
  overwriteEnd
  live interval from producer through all consumers before overwrite
}
```

Value liveness starts at the producer, not at the virtual storage anchor.

Value liveness ends after all consumers of that produced value, or at the first
full overwrite if the previous value is no longer read.

The planner must support multiple value ranges for one storage object. A single
min/max interval per alloc is not sufficient for full Meta-equivalent semantics
when there are holes between producer/consumer phases.

### 3. Sourceful Alloc Is A Virtual Store

For:

```mlir
%t = ttng.tmem_alloc %src
```

the planner must internally model:

```mlir
%t_storage = virtual ttng.tmem_alloc
virtual ttng.tmem_store %src, %t_storage
```

The sourceful alloc op is the producer of the initial TMEM value for planner
liveness and producer/consumer reasoning.

The implicit init-store task is taken from the sourceful alloc op's
`ttg.partition` annotation, and that annotation must contain exactly one
partition. A sourceful alloc with multiple partitions is invalid for this model:
one implicit store cannot be assigned to multiple writer partitions without
changing the communication contract.

This applies regardless of whether the sourceful alloc is inside a
`tt.warp_specialize` loop. The only reason WS appears in tests is that the
current target FA input is WS-shaped.

### 4. Sourceless Alloc Uses Explicit Producers

For:

```mlir
%t = ttng.tmem_alloc
ttng.tmem_store %src, %t
```

the storage is still modeled as virtual-hoisted, but the value producer is the
explicit `ttng.tmem_store`.

For a sourceless alloc, the alloc op's `ttg.partition` annotation must not be
used as producer/task evidence. The producer task comes from the explicit store,
MMA overwrite, or other real producer. This keeps storage placement separate
from value production.

For MMA operand-D:

- `useAccumulator = false` is a full overwrite producer;
- `useAccumulator = true` consumes the previous D value and produces the next D
  value;
- if `useAccumulator = true` has no previous producer, the planner must fail.

Existing operand-D channel handling should be reused where it already describes
the producer/consumer chain.

### 5. Planner Records Are Separate From Semaphore Channels

Current NVWS `TmemDataChannelPost` serves two roles:

- memory-planner producer/consumer/liveness record;
- optional semaphore-visible channel source/end annotation.

The enhanced planner needs a planner-only record that can exist even when no
semaphore channel is needed.

Required behavior:

```text
getSrcOp() = producer op for this TMEM value range
getDstOps() = all consumers before first full overwrite
getDstOp() = last consumer in that set, for existing single-destination APIs
producer/consumer task ids are derived from the real producer and consumers
```

Task-id rules:

- sourceful `ttng.tmem_alloc %src`: producer task id is the alloc op's single
  `ttg.partition` entry; multiple entries are a hard failure;
- sourceless `ttng.tmem_alloc`: the alloc op's partition is not a producer task
  and must not be used for same-partition or dependency checks;
- explicit `ttng.tmem_store`: producer task id comes from the store op;
- MMA overwrite/update producer: producer task id comes from the MMA op;
- consumers use their own consumer op partitions.

If a real `TmemDataChannelPost` already exists, keep it as the authoritative
record and augment only the missing storage/liveness metadata. Do not create a
competing virtual channel for the same value range.

If no real channel exists because the value is same-partition or sourceful
implicit-init, create a planner-only virtual record. It must participate in
greedy and backtracking allocation decisions, but it must not emit new
`tmem.start` / `tmem.end` attributes unless semaphore insertion explicitly
requires them.

This closes the current bug where channel-less allocs can become automatically
compatible via logic such as `!hasBothChannels`.

## Allocation Equivalence Contract

"Same allocation decision as Meta" means:

- corresponding TMEM tensors have the same alias/reuse grouping;
- corresponding tensors have the same offset-vs-owner relation;
- corresponding loop-carried TMEM buffers have equivalent `buffer.copy`;
- exact numeric `buffer.id` is not required unless deterministic traversal
  naturally produces it;
- exact SSA names do not matter.

For the current FA sourceful f16 operands, the required relation is:

```text
partition-5 sourceful f16 id == qk_0 id
partition-4 sourceful f16 id == qk_1 id
partition-5 sourceful f16 id != partition-4 sourceful f16 id
partition-5 sourceful f16 offset == 0
partition-4 sourceful f16 offset == 0
```

The wrong relation is:

```text
partition-5 sourceful f16 id == partition-4 sourceful f16 id
offsets 0 and 64
```

## Implementation Plan

### 1. Introduce Planner Storage And Value-Range State

Add planner-only state around each `ttng.tmem_alloc`:

```text
allocToStorage
allocToValueRanges
allocToPlannerRecords
```

The exact C++ structure can differ, but it must represent:

- the real alloc op receiving final attrs;
- the virtual storage anchor;
- one or more value producer/consumer ranges;
- optional real channel pointer;
- optional planner-only virtual record.

### 2. Build TMEM Value Ranges

Walk each TMEM alloc's users, following `ttg.memdesc_index`,
`ttg.memdesc_reinterpret`, and other existing transparent memdesc users.

Classify:

- sourceful `ttng.tmem_alloc %src` as an initial producer;
- `ttng.tmem_store` as a producer for the destination region it writes;
- MMA `useAccumulator = false` on operand-D as an overwrite producer for the
  D region it writes;
- MMA `useAccumulator = true` on operand-D as both consumer and producer;
- `ttng.tmem_load` and non-D MMA uses as consumers.

Only a store or MMA write that covers the whole modeled TMEM value region is a
full overwrite. If the destination is a `ttg.memdesc_index`,
`ttg.memdesc_reinterpret`, or other view, the planner must either compute the
exact written region and model per-region value ranges, or fail with a clear
diagnostic. It must not end the previous value range for the whole storage when
only a slice was overwritten.

For each producer, collect all consumers until the next full overwrite.
Multiple consumers before overwrite are required to be represented as part of
the same value range.

If the producer/consumer order is ambiguous, cross-block ordering cannot be
lifted, or a view/region relation cannot be modeled safely, fail with a clear
diagnostic. Do not silently fall back to first-consumer liveness.

### 3. Build Planner Records

For each value range:

- use the existing real `TmemDataChannelPost` if one already represents that
  producer/consumer relation;
- otherwise create a planner-only virtual record.

These records are the input to:

```text
livenessForTmemAlloc / successor replacement
samePartition
alongDependencyChain
hasPotentialReuse
findReuseChannel
tryAllocateBacktracking
allocateTmemAllocsBacktracking
allocateTmemAllocs
```

They must not by themselves create semaphore-visible IR attributes.

### 4. Route Annotated Preassignment Through The Same Semantics

The existing annotated-preassignment path must not assign a shared `buffer.id`
and nonzero offsets solely because multiple allocs carry the same traced or
annotated buffer id.

`preAssignAnnotatedAllocs` and any equivalent preallocation path must treat
annotations as constraints or hints, then pass each proposed group through the
same storage/value-range/planner-record compatibility checks as normal
allocation:

- if an annotated group is Meta-compatible, keep the intended grouping;
- if the annotation is only physical packing guidance, split the group into
  separate semantic allocation groups;
- if a required annotation contradicts the value-range model, fail rather than
  producing a shared id with unsafe offsets.

This gate must run before both greedy and backtracking allocation decisions, so
neither path can bypass the virtual-store semantics.

### 5. Replace Single-Interval Reasoning Where Needed

The current NVWS allocator has one interval per alloc. That is insufficient for
general Meta-equivalent TMEM semantics.

Replace or augment:

```text
DenseMap<Operation *, Interval<size_t>> allocToIntervals
```

with either:

```text
DenseMap<Operation *, SmallVector<Interval<size_t>>> allocToValueIntervals
```

or an equivalent `LiveRangeSet`.

Two storages interfere if any relevant value intervals overlap and their column
ranges overlap.

Do not include the virtual storage anchor in these value intervals.

### 6. Replace One-Loop-Per-Alloc Bucketing

The current allocator buckets each alloc into one innermost control region.
That is not sufficient once one virtual-hoisted storage object can have several
producer-bound value ranges in different loops or regions.

Update the loop/control-region logic so allocation scheduling is derived from
value ranges and planner records, not from a single alloc-op location:

- `handledAllocs`-style one-shot assignment by alloc must not decide the whole
  storage lifetime;
- `sameLoop` and dependency checks must compare the relevant value-range
  producer/consumer records;
- `tt.tmem_alloc_algo` choices must account for all ranges of a storage, or
  conservatively fail if a storage spans unsupported control regions;
- `buffer.copy` decisions must follow loop-carried TMEM value/token ranges, not
  merely the alloc op's syntactic parent.

### 7. Preserve Existing Real-Channel Behavior

Existing explicit `ttng.tmem_store` and operand-D paths must continue to use
their real channel records when present.

The new planner-only records fill gaps where Meta-equivalent producer semantics
exist but no real channel is available, especially:

- sourceful `ttng.tmem_alloc %src`;
- same-partition producer/consumer cases where no semaphore channel is needed;
- unhoisted storage where the alloc op placement should not drive liveness.

### 8. Update Allocation Reuse Decisions

Update both greedy and backtracking allocation paths to use:

```text
storage metadata for size / final attrs
value-range metadata for liveness overlap
planner records for partition/dependency compatibility
```

The allocator must not treat a candidate as freely reusable because either side
lacks a real semaphore channel.

If planner records cannot prove Meta-equivalent compatibility, allocate new
space or fail. Do not produce a same-ID/nonzero-offset packing decision that the
Meta storage model would not produce.

### 9. Validate Against Meta Allocation Decisions

Regenerate:

```text
meta-aws-logs/nvws-after-memory-planner-v3.mlir
```

from:

```text
meta-aws-logs/nvws-afeter-hoist.mlir
```

using:

```bash
triton-opt meta-aws-logs/nvws-afeter-hoist.mlir \
  --nvws-memory-planner=num-buffers=3 \
  -allow-unregistered-dialect \
  --mlir-print-debuginfo \
  --mlir-use-nameloc-as-prefix
```

Compare against:

```text
meta-aws-logs/meta-after-memory-planner.mlir
```

The checker should compare mapped allocation groups and offsets, not raw SSA
names.

For the current Meta dump, skip the six Meta-only scratch TMEM allocations that
do not exist in the NVWS input:

```text
%alpha_1
%alpha_0
%m_ij_1
%l_i0_0
%m_ij_0
%l_i0_1
```

## Required Lit Tests

Add focused tests under:

```text
test/NVWS/MetaAutoWS/
```

### 1. FA Sourceful F16 Reuse Group

Use the existing FA-derived NVWS memory-planner input.

Expected:

```text
partition-5 sourceful f16 id == qk_0 id
partition-4 sourceful f16 id == qk_1 id
partition-5 sourceful f16 id != partition-4 sourceful f16 id
partition-5 sourceful f16 offset == 0
partition-4 sourceful f16 offset == 0
```

### 2. Sourceful Alloc With QK-Like Reuse Pattern

Construct a small focused test with real qk-like producer/consumer ops:

```mlir
%qk0 = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1, 5>}
%qk1 = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1, 4>}

%qk0_tok = ttng.tc_gen5_mma ..., %qk0[...] {ttg.partition = array<i32: 1>}
%qk1_tok = ttng.tc_gen5_mma ..., %qk1[...] {ttg.partition = array<i32: 1>}
%qk0_val, %qk0_load_tok = ttng.tmem_load %qk0[%qk0_tok]
    {ttg.partition = array<i32: 5>}
%qk1_val, %qk1_load_tok = ttng.tmem_load %qk1[%qk1_tok]
    {ttg.partition = array<i32: 4>}

%a = ttng.tmem_alloc %src0 {ttg.partition = array<i32: 5>}
%b = ttng.tmem_alloc %src1 {ttg.partition = array<i32: 4>}
%a_tok = ttng.tc_gen5_mma %a, ... {ttg.partition = array<i32: 1>}
%b_tok = ttng.tc_gen5_mma %b, ... {ttg.partition = array<i32: 1>}
```

Expected:

```text
%a id == %qk0 id
%b id == %qk1 id
%a id != %b id
%a offset == 0
%b offset == 0
```

### 3. Sourceful Alloc Task-ID Validation

Create one valid sourceful alloc with exactly one partition and one invalid
sourceful alloc with multiple partitions:

```mlir
%ok = ttng.tmem_alloc %src0 {ttg.partition = array<i32: 5>}
%bad = ttng.tmem_alloc %src1 {ttg.partition = array<i32: 4, 5>}
```

Expected:

```text
%ok uses partition 5 as the implicit init-store producer task;
%bad fails with a clear diagnostic.
```

Also include a sourceless alloc carrying a partition annotation and prove that
annotation is not used as producer evidence:

```mlir
%storage = ttng.tmem_alloc {ttg.partition = array<i32: 4, 5>}
ttng.tmem_store %src, %storage {ttg.partition = array<i32: 1>}
consumer(%storage) {ttg.partition = array<i32: 5>}
```

Expected:

```text
producer task is partition 1 from the store, not partitions 4/5 from the
sourceless alloc.
```

### 4. Sourceful Alloc Outside WS

Create a sourceful `ttng.tmem_alloc %src` outside any `tt.warp_specialize`
ancestor.

Expected:

```text
planner still treats it as virtual hoisted storage plus virtual init producer
for allocation decisions;
the real IR op remains in place.
```

### 5. Sourceless Alloc With Explicit Store

Input:

```mlir
%a = ttng.tmem_alloc
ttng.tmem_store %src, %a
consumer(%a)
```

Expected:

```text
storage is planned independently of the alloc op placement;
value liveness starts at the explicit tmem_store, not the alloc op.
```

### 6. Multiple Consumers Before Overwrite

Input:

```mlir
%a = ttng.tmem_alloc %src
consumer0(%a)
intervening_tmem_alloc_or_store()
consumer1(%a)
```

Expected:

```text
the initialized value remains live through both consumers;
the intervening allocation cannot reuse %a as if liveness ended at consumer0.
```

### 7. Overwrite Boundary

Input:

```mlir
%a = ttng.tmem_alloc %src0
consumer(%a)
ttng.tmem_store %src1, %a
consumer_after_overwrite(%a)
```

Expected:

```text
the %src0 value range ends at the overwrite;
the %src1 value range starts at the explicit store;
reuse decisions are based on the two value ranges, not one storage-anchor range.
```

### 8. Partial View Store Is Not A Full Overwrite

Input:

```mlir
%a = ttng.tmem_alloc %src0
%slice = ttg.memdesc_index %a[...]
ttng.tmem_store %src1, %slice
consumer_of_unsliced_or_other_slice(%a)
```

Expected:

```text
the slice store does not end the whole-storage %src0 value range;
the planner either models the written region precisely or fails clearly.
```

### 9. Existing Real Channel Is Not Duplicated

Use an operand-D/sourceful case that already creates a real
`TmemDataChannelPost`.

Expected:

```text
the real channel remains authoritative;
no competing virtual channel is created;
storage/liveness metadata is augmented only as needed.
```

### 10. Annotated Preassignment Compatibility

Create annotated allocs that would previously be grouped solely by annotated
buffer id.

Expected:

```text
shared annotated id is preserved only when the value-range/planner-record gate
proves Meta-compatible reuse;
otherwise the group is split or the pass fails with a diagnostic.
```

### 11. Multi-Region Storage Bucketing

Create one storage with producer/consumer ranges in two different loops or
control regions.

Expected:

```text
allocation decisions use the relevant value ranges and planner records;
the storage is not assigned to one loop solely because of the alloc op parent.
```

### 12. Existing Explicit Store Regression

Existing explicit `ttng.tmem_store` tests should continue to pass and should not
gain extra semaphore-visible `tmem.start` / `tmem.end` attrs solely due to the
planner-only records.

## Build And Test

Follow repo instructions:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test/NVWS
```

Then run full lit only after the build and focused NVWS tests pass:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test
```

Do not run pytest.

## Exit Criteria

This task is complete when:

1. `NVWSMemoryPlanner` does not rewrite or hoist TMEM IR.
2. All `ttng.tmem_alloc` storage is modeled as virtual-hoisted for allocation
   decisions, regardless of real IR placement.
3. `ttng.tmem_alloc %src` is modeled as virtual init store for producer/liveness
   decisions.
4. A sourceful alloc's implicit init-store producer task comes from exactly one
   alloc-op partition; multiple partitions fail.
5. Sourceless alloc partitions are not used as producer/task evidence.
6. Explicit `ttng.tmem_store` and MMA overwrite producers define value-range
   boundaries only for the regions they actually write.
7. Value liveness is producer-boundary based and covers all consumers until
   overwrite.
8. Greedy and backtracking allocation paths use planner records and value-range
   liveness, not real-channel presence alone.
9. Annotated preassignment goes through the same compatibility gate as normal
   allocation and cannot force unsafe same-id/nonzero-offset packing.
10. Loop/control-region bucketing is range-based and does not collapse a
    storage to one alloc-op parent.
11. The FA sourceful f16 operands match Meta reuse groups:
   partition-5 f16 with qk_0, partition-4 f16 with qk_1, distinct from each
   other, both offset 0.
12. The generated NVWS after-memory-planner dump matches Meta allocation
   decisions for mapped TMEM tensors, ignoring Meta-only scratch allocations.
13. Existing explicit-store and operand-D behavior is preserved.
14. Focused NVWS lit tests cover sourceful, sourceless explicit-store,
    multi-consumer, overwrite-boundary, partial-view, annotated-preassignment,
    multi-region, and real-channel cases.
15. `ninja triton triton-opt` passes.
16. `llvm-lit -v test/NVWS` passes.
17. Full `llvm-lit -v test` has no new failures beyond known unrelated baseline
    failures.
