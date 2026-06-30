# NVWS MemoryPlanner Meta-AWS parity contract

Status: IMPLEMENTED AND VERIFIED (29jun26).

The root-cause evidence, final IR correlations, and executed validation matrix
are recorded in `plans/root-case-pytest-failures-and-proposed-fix.md`.

This document supersedes the independent NVWS planning policy in
`fable/local-alloc-planner-plan.md` and `fable/circular-buffer-disign.md`.
Those documents still define the downstream metadata representation, but they
do not authorize a different allocation policy from Meta-AWS.

## 1. Goal

`third_party/nvidia/hopper/lib/Transforms/WarpSpecialization/WSMemoryPlanner.cpp`
is the behavioral source of truth for NVWS memory planning. The NVWS planner
must be a direct transcription of the Meta planner's phase ordering, candidate
classification, reuse decisions, copy-depth growth, and budget accounting.

The NVWS implementation may differ only where its input IR or downstream
passes require a mechanical adapter. Every such difference is listed in
section 3. An unlisted policy difference is a bug.

## 2. Meta policy copied into NVWS

### 2.1 Driver order

For each function:

1. Collect communication channels.
2. Apply the effective per-loop SMEM options.
3. Plan local/SMEM buffers first and return the next free `buffer.id`.
4. Plan TMEM starting at that id.

### 2.2 SMEM algorithm 0

Match Meta's original `MemoryPlanner` behavior:

1. Innermost-loop buffers with at least two non-trivial dimensions share an
   element-type-compatible id and start at `numBuffers` copies.
2. Other buffers start with a unique id and one copy.
3. A shared id has at least as many copies as logical members assigned to it.
4. Compatible epilogue buffers from one original TMEM load may share an id
   only when all of their liveness intervals are pairwise disjoint.

### 2.3 SMEM algorithm 1

Transcribe Meta's `allocateSmemBuffers` phases in the same order:

1. Create one buffer record per local allocation, with a unique id and one
   copy. Apply explicit annotations and pin those records.
2. Raise a cross-stage unpinned record to two copies only if `numBuffers >= 2`
   and the resulting total remains within `smemBudget`.
3. Classify unpinned records as `P0_InnermostTMA`,
   `P1_InnermostNonTMA`, or `P2_Other` using Meta's predicates.
4. Fuse compatible P2 epilogue records derived from one original TMEM load.
5. Process P0 and then P1. A circular reuse candidate exists only when that
   priority contains exactly two records. Use Meta's group start, incremental
   budget checks, and odd/even finalization exactly.
6. Increase copies of fused P2 groups uniformly using Meta's phase 4.5 and a
   budget check at every attempted final depth.
7. Emit the resulting `buffer.id` and `buffer.copy`.

Planner arithmetic is over physical `buffer.id` groups:

```text
group cost = max(member size) * max(member copy depth)
total cost = sum(group cost)
```

Logical aliases inside one physical group never become additional circular
slots. `smemBudget` controls optional heuristic growth exactly as in Meta; it
is not a hard cap on the copy-1 baseline or on pinned annotations. The later
physical shared-memory allocator remains the hard hardware-capacity check.

### 2.4 TMEM

Keep the Meta TMEM allocation order and reuse algorithms. The final
round-robin copy increase uses Meta's direct-user test exactly. In particular,
a sourceful TMEM allocation used as MMA operand A is eligible when that MMA's
accumulator token is loop-carried. This is how Meta assigns two logical FP16 P
slots to the same physical columns occupied by one FP32 QK slot.

`buffer.copy` is a logical-channel depth, not a property that must be uniform
across every allocation with one physical `buffer.id`. Meta's backward H64
plan deliberately contains:

```text
qkT  buffer.id=2  buffer.copy=1  f32
ppT  buffer.id=2  buffer.copy=2  f16  buffer.offset=0
```

The two logical channels have independent synchronization rings. Code
partitioning later reinterprets the one-copy FP32 QK allocation as two FP16 P
slots without allocating more TMEM.

## 3. Permitted NVWS adaptations

These are the complete allowed differences from Meta planning policy.

### 3.1 Channel adapter

NVWS uses `LocalDataChannelPost` and `TmemDataChannelPost` plus NVWS async task
ids instead of Meta's `Channel` hierarchy. This may change lookup mechanics,
not planner decisions.

### 3.2 NVWS descriptor producers

SMEM TMA classification recognizes the NVWS operations that directly write a
local memdesc: `nvws.descriptor_load` and `nvws.descriptor_gather`. This is the
NVWS representation of Meta's TMA producer, not a new priority policy.

### 3.3 NVWS scaled-epilogue provenance

NVWS materializes the backward dK scale multiply before the split subtiles are
stored into cross-partition SMEM channels. For epilogue grouping only, an
`arith.mulf` may be traced when all of its operands that reach TMEM identify
one unique `ttng.tmem_load`; zero or two distinct TMEM roots do not match.
This adapter preserves the Meta grouping decision for the NVWS-specific IR
shape and does not classify the multiply as a generally transparent value op.

### 3.4 Downstream allocation annotations

NVWS continues to emit the attributes consumed by `nvws-insert-semas` and the
allocation lowering:

```text
buffer.id
buffer.copy
buffer.offset
buffer.circular
buffer.start
```

`buffer.circular` and `buffer.start` describe a two-record reuse group that
Meta's phase 4 already selected. They must not create, broaden, or resize a
reuse group. Both members retain Meta's final `buffer.copy`; starts are `0`
and `1` in producer program order.

P2 epilogue fusion is not circular and receives no `buffer.circular` or
`buffer.start`.

Algorithm 0's Meta policy deliberately assigns every compatible innermost
record to one shared-id pool and then raises `buffer.copy` to at least the
number of records. NVWS represents that already-selected pool as circular:
all records receive `buffer.circular`, and `buffer.start` is their distinct
zero-based position in planner program order. This metadata does not change
the Meta id assignment or copy depth; it exposes Meta's logical pool to
InsertSemas and allocation lowering.

### 3.5 Mixed-depth TMEM reuse groups

InsertSemas normally builds one ownership DAG for a physical `buffer.id`.
That representation is invalid for Meta's mixed-depth TMEM plan because it
would collapse QK's one-slot ring and P's two-slot ring into one protocol.

For a TMEM id whose members have different authored `buffer.copy` values,
NVWS must instead:

1. Build one logical ownership DAG, semaphore ring, and backing shape per
   allocation, preserving each member's own copy depth.
2. Require the exact two-channel alternating reuse proof used by the Meta
   plan. In one scheduled loop, channel A's writer and channel B's reader must
   have one owner, channel A's reader and channel B's writer must have the
   other owner, and the in-block order must be:

```text
A.write -> B.read
A.read  -> B.write
```

   The first order closes across the loop backedge as
   `B.read(i) -> A.write(i+1)`; the second is a same-iteration handoff.
   Unsupported member counts, control flow, owners, or ordering diagnose.
3. Keep the logical semaphore creates independent. Only after all accesses
   are rendered, replace the non-owner physical backing with a checked TMEM
   subslice/reinterpretation of the owner backing, using physical span,
   `buffer.offset`, and element widths exactly as Meta's
   `sliceAndReinterpretMDTMEM` does. NVWS writes `buffer.offset = 0` on both
   owners and zero-offset reusers, so owner selection must be inferred from a
   unique legal span/element-width containment, not attribute presence.

This is an NVWS synchronization adapter for Meta's planner output. It neither
changes planner copy depths nor allocates an extra QK buffer.

### 3.6 Hard postconditions

Before returning from NVWS MemoryPlanner:

1. Local-memory members of one physical `buffer.id` group have one consistent
   `buffer.copy` value. TMEM members may differ only when the mixed-depth
   alternating-reuse proof in section 3.5 succeeds.
2. Every algorithm-1 circular group has exactly two physical records, common
   depth and compatible local types, with distinct starts `0` and `1`.
   Every algorithm-0 circular group has one start per member in
   `[0, member-count)`, and its common depth is at least the member count.
3. No phase after a budgeted Meta growth decision changes that decision's
   physical copy depth. In particular, downstream annotation assignment never
   increases `buffer.copy`.

Violation is a planner diagnostic. InsertSemas keeps its equivalent checks as
defense in depth.

## 4. InsertSemas completion frontier

The NVWS epilogue channel creates this valid pre-lowering SSA shape:

```text
local_load managed_smem -> [convert_layout] -> descriptor_store
```

Generic TMA lowering can reuse `managed_smem` as the asynchronous TMA source.
Therefore the physical read lifetime ends at descriptor-store completion, not
at the synchronous `local_load` instruction.

### 4.1 Stage-1 fact

An Access node has two anchors:

```text
op                direct managed-memory touch and retargeting point
completionAnchor  last operation that must complete before ownership returns
```

`completionAnchor` defaults to `op`.

For a managed `ttg.local_load`, stage 1 recognizes only this closed forwarding
shape:

```text
local_load -> descriptor_store
local_load -> convert_layout -> descriptor_store
```

The descriptor store must be the unique matching terminal use, must follow the
load in the same block, and must have the same effective owner. Ambiguous
fan-out or a matching store across control flow is a hard diagnostic.

The descriptor store is a lifetime anchor, not another memory touch: the
Access row retains the local-load R touch and its `<none>` payload.

### 4.2 Stage 3 and emission

Release schedule reasoning uses the source Access node's
`completionAnchor`. EMIT-IR retargets `op` and then reports the already-planned
`completionAnchor` as the row's placement endpoint. The following Release row
is consequently emitted after the descriptor store without searching or
moving emitted synchronization.

After LowerSemaphore and TMA lowering, the required order is:

```text
wait full
async_tma_copy_local_to_global managed_smem
async_tma_store_wait
arrive empty
```

This is an extension of the Access-DAG fact, not a post-emission repair.

## 5. Exact-alias staged SMEM handoffs

A non-circular local group can contain several logical members that have the
same offset, extent, and memdesc type. When that exact-alias group has depth
greater than one, one ASP cursor selects the physical slot for all members.
The cursor advances once for each fresh write, not once for every ownership
handoff.

For example, two fused epilogue members at depth two have this ownership order:

```text
access:          W m0 P4 -> R m0 P2 -> W m1 P4 -> R m1 P2
access ordinal:      0          0          1          1
release offset:      0          1          0          1
release target:      0          1          1          0
```

The release target is the slot that the satisfied acquire will address:

1. For a forward, same-iteration handoff, SYNC-DAG authors
   `(destinationOrdinal - sourceOrdinal) mod depth`.
2. For a loop-closing handoff, it keeps offset zero when a future destination
   acquire reaches the source slot within one complete cursor orbit.
3. If that orbit never reaches the source slot, it authors the offset from the
   source slot to the next iteration's destination slot.

The rule applies only when all accesses are direct children of one ownership
loop and the slot schedule is complete. Unsupported control-flow shapes are a
diagnostic. It does not apply to circular groups or TMEM groups.

SYNC-DAG records the result in each Release node's `stageOffset`. If any release
is shifted, it also records explicit zero offsets on the group's Acquire nodes.
If every release offset is zero, it discards those provisional offsets and
keeps the group on ordinary token-stage propagation.
EMIT-IR transcribes those fields to semaphore operations; it does not infer the
slot protocol. The authored acquire offsets force AssignStagePhase to use
multiphase parity. Independently, single-phase is eligible only when
`gcd(depth, advancesPerIteration) == 1`, because otherwise some fixed
semaphore sites never visit slot zero and scalar parity cannot toggle for every
reused mbarrier.

## 6. Required tests

### 6.1 MemoryPlanner

1. A Meta/NVWS phase-4 parity test for exactly two circular candidates.
2. An eight-member epilogue test proving two four-member fused groups do not
   become an eight-slot ring and the final budget is respected.
3. A TMEM planner test with a sourceful operand A feeding a loop-carried MMA;
   it receives Meta's copy increase while a one-copy allocation sharing its
   physical id remains one-copy.
4. An InsertSemas test for the exact QK/P mixed-depth group: independent
   one-slot and two-slot semaphore rings, one coalesced physical TMEM backing,
   and legal pre/post-pipeline order.
5. Existing MetaAutoWS planner tests remain green.

### 6.2 InsertSemas

1. A direct local-load descriptor-store case checks the consumer Release after
   the descriptor store.
2. A converted local-load case checks the same placement.
3. A cross-pass check confirms the lowered `arrive empty` follows
   `async_tma_store_wait`.
4. An exact-alias depth-two case checks release offsets `0,1,0,1`, successor
   slots `0,1,1,0`, and multiphase parity after AssignStagePhase.

## 7. Acceptance

1. Build `triton` and `triton-opt` before lit.
2. Run targeted planner and InsertSemas lit tests.
3. Run the complete NVWS/MetaAutoWS and InsertSemas lit directories plus
   `automatic-warp-specialization.mlir`.
4. Run all eight previously failing tutorial selectors.
5. Run the complete `python/tutorials/fused-attention-ws-device-tma.py`
   pytest matrix under NVWS-AWS and Meta-AWS.
6. Capture fresh config-0/1/2 IR and verify the four structural invariants,
   not only the pytest result.
7. For config 2, verify post-pipeline qkT, dpT, dV, dQ, and dK MMA operations
   remain present, so a runtime pass cannot be attributed to disabled MMA
   pipelining.
