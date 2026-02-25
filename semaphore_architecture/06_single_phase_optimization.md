# Single-Phase Optimization in AssignSemaphoreStagePhase

## Purpose

The `AssignSemaphoreStagePhase` pass must compute, for each `SemaphoreAcquireOp`,
the mbarrier phase that the acquire should wait on. There are two strategies for
tracking this phase:

- **Multiphase**: a per-stage bit vector that is always correct but costs more arithmetic.
- **Single-phase**: a single scalar that is cheaper but only correct under certain conditions.

This document describes the analysis that determines, per buffer group, whether
single-phase mode is safe. The analysis is implemented in `computeSinglePhaseEligibility`
in `AssignSemaphoreStagePhase.cpp`.

**Source file**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignSemaphoreStagePhase.cpp`

---

## Background: Why Phase Matters

Each mbarrier stage has an internal phase parity** (0 or 1). The hardware toggles
this parity every time an acquire-release cycle completes on that stage:

- After initialization, phase = 0.
- Each completed acquire-release cycle on a given stage flips the phase for that stage.

Concretely, the k-th acquire of `mbar[s]` requires phase:

```
phi(k) = phi_0 XOR ((k - 1) mod 2)
```

where `phi_0` is the initial phase (0 for `is_released=true`, 1 for `is_released=false`).

Getting the phase wrong causes deadlock or undefined behavior. If software
computes a phase value that does not match the hardware's internal parity, the
`mbarrier.try_wait` will either:

- Wait forever (deadlock) if it thinks the barrier has not yet been signaled, or
- Return immediately on a stale signal (data race / undefined behavior).

The phase computation must therefore be proven correct for every acquire in every
possible execution path.

---

## Multiphase vs Single-Phase

### Multiphase

Multiphase tracking uses a per-stage bit vector stored as an integer, where bit
position `s` tracks the phase parity for stage `s`. At every acquire of stage `s`,
the phase is updated as:

```
phase = phase XOR (1 << acquireStage)
```

This is always correct by definition: each stage bit independently tracks that
stage's parity. The cost is a `shl` + `xor` pair at every acquire, plus the initial
phase must encode all bits (0x00000000 for released, 0xFFFFFFFF for unreleased).

### Single-phase

Single-phase tracking uses a single scalar (0 or 1). It flips only when the
stage counter wraps around from `depth-1` back to 0:

```
if (acquireStage == 0)
    phase = phase XOR 1
else
    phase = phase    // no change
```

This is simpler: one `cmpi` + `select` + `xori`, with no `shl`. The initial phase
is 0 or 1 (not a bit vector). The control flow is also simpler because the phase
is a plain boolean rather than a packed integer.

### Trade-off

Single-phase generates less arithmetic, but it is only
correct when A(s) = 1 for all stages s. The metric A(s) is defined in the next
section.

---

## The A(s) Metric

**Definition**: A(s) is the number of acquires of for stage s per cycle, where one
cycle is a single pass through all D stages (one full iteration of the steady-state
loop).

### When A(s) = 1

Each stage is acquired exactly once per cycle. The acquire sequence visits stages
in round-robin order: s=0, s=1, ..., s=D-1, s=0, s=1, .... In this pattern,
stage `s` is acquired on iterations `n` where `n mod D == s`. The phase parity
for each acquire depends only on which "lap" (W(n) = floor(n/D)) we are on, and
single-phase correctly tracks this with a flip at wrap.

### When A(s) > 1 for any s

Some stage `s` is acquired more than once per cycle. Two acquires within the same
cycle see the same `acquireStage` value, so single-phase computes the same phase
for both. But the hardware flipped the phase after the first acquire-release, so the
second acquire needs the opposite phase. Single-phase is therefore incorrect.

A(s) counts acquires (mbarrier waits), NOT individual load/store ops within a
tenure. Everything between an acquire and the matching release is one tenure, one
mbarrier wait. Multiple loads, stores, conditionals, or RMW operations within a
single tenure do not affect A(s).

---

## Proof of Correctness (Summary)

Full formal proofs are in [08_proofs.md](08_proofs.md). Here we state the key results.

### Sufficiency: A(s) = 1 implies single-phase correct

When A(s) = 1 for all s, the cumulative acquire count for stage `s` up to the n-th
global acquire is:

```
K(s, n) = floor(n / D)
```

and the "lap number" (number of full cycles completed) is:

```
W(n) = floor(n / D)
```

Therefore `K(s, n) = W(n)` for all s and n, which means the phase parity at every
acquire matches the single-phase formula `phi_0 XOR (W(n) mod 2)`.

### Necessity: A(s) > 1 implies single-phase fails

When A(s*) > 1 for some stage s*, there exist two acquires in the same cycle that
target the same stage. They have the same W(n) value (same cycle), so single-phase
assigns the same phase. But the first acquire has K(s*, n1) and the second has
K(s*, n2) = K(s, n1) + 1, so their required phases differ by a flip. Single-phase
fails.

---

## Eligibility Analysis Algorithm

The analysis is implemented as two methods on `struct AssignSemaphoreStagePhase`:

- `computeSinglePhaseEligibility()` -- entry point, returns `bool`
- `walkBlockForEligibility(Block *, int &virtualStage, DenseSet<...> &seen)` -- recursive walk

### computeSinglePhaseEligibility

```
bool computeSinglePhaseEligibility():
    1. If depth == 1:
         return true   // one stage, nothing to cycle, trivially correct

    2. Find the warp-specialized for-loop containing group acquires:
         Walk groupSemaphoresList -> users -> SemaphoreAcquireOp -> parent ops
         Look for scf::ForOp with kWarpSpecializeAttrName
         If not found: return false  // conservative, no loop -> multiphase

    3. Walk loop body recursively:
         DenseSet<(Value, int, int)> seen = {}
         int virtualStage = 0
         eligible = walkBlockForEligibility(wsLoop.getBody(), virtualStage, seen)
         If !eligible: return false  // duplicate found

    4. If virtualStage == 0:
         return false   // no advances in loop -> multiphase

    5. return true
```

### walkBlockForEligibility

The recursive walk visits every operation in a block:

```
bool walkBlockForEligibility(Block *block, int &virtualStage,
                             DenseSet<(Value, int, int)> &seen):
    for each op in block:
        if op is SemaphoreAcquireOp on a group semaphore:
            if isFirstUseFreshWriteAfterAcquire(op):
                virtualStage++          // this acquire advances the stage
            pid = getPartitionId(op) or 0
            key = (semaphore, pid, virtualStage)
            if key already in seen:
                return false            // DUPLICATE -> multiphase required
            add key to seen

        else if op is scf::ForOp:
            recurse into body

        else if op is scf::IfOp:
            save (virtualStage, seen)
            walk then-block
            save then-result
            restore to pre-if state
            walk else-block (if present)
            merge: virtualStage = max(then_vs, else_vs)
                   seen = union(then_seen, else_seen)

    return true  // no duplicates found
```

Key details of the `scf.if` handling: both branches are explored independently from
the pre-if state. After both branches, the algorithm takes the conservative merge --
max of virtualStage (pessimistic about how far the pipeline has advanced) and union
of seen keys (any acquire that happens in either branch is counted).

---

## All-or-Nothing Per Buffer Group

A buffer group is the set of semaphores that share the same backing buffer(s). The
release-acquire ring (P releases C, C releases P) forces A_P(s) = A_C(s) for all s:

- For C to be acquired at stage s, there must have been a prior `release C` at stage s,
  which requires a prior `acquire P` at stage s. So A_C(s) <= A_P(s).
- Symmetrically, A_P(s) <= A_C(s).
- Therefore A_P(s) = A_C(s). This generalizes to N semaphores via ring transitivity.

Since A(s) is the same for all semaphores in the group, the single-phase eligibility
decision is the same for all of them.

**Implementation**: `computeSinglePhaseEligibility` returns a single `bool`, and the
`run()` method applies it to every `SemaphoreCreateOp` in the group:

```cpp
bool eligible = analyzer.computeSinglePhaseEligibility();
for (auto semaOp : semaOps) {
    semaOp->setAttr(kUseSinglePhaseAttrName,
                    BoolAttr::get(semaOp.getContext(), eligible));
    useSinglePhaseBySemaphore[semaOp.getResult()] = eligible;
}
```

---

## Examples

These examples are critical for building intuition. Each shows the loop body,
the virtual-stage trace, and the eligibility result.

### Example 1: Standard Producer-Consumer -- SINGLE-PHASE

```
for {
    acquire P @1   // advance (store is FreshWrite)
    store buf  @1
    release C  @1

    acquire C @2   // observation (load)
    load buf   @2
    release P  @2
}
```

**Trace**:
- `acquire P`: advance, vs=1. Key (P, pid, 1) -- inserted.
- `acquire C`: no advance. Key (C, pid, 1) -- inserted.

No duplicates. virtualStage=1 > 0. Single-phase.

P has 1 acquire per iteration, C has 1 acquire per iteration. A(s)=1 for all s.

### Example 2: Two Acquires Per Semaphore, All Advance -- SINGLE-PHASE

```
for {
    acquire P @1   // advance (store)
    store      @1
    release C  @1

    acquire C @2   // observation (load)
    load       @2
    release P  @2

    acquire P @1   // advance (store)
    store      @1
    release C  @1

    acquire C @2   // observation (load)
    load       @2
    release P  @2
}
```

**Trace**:

- `acquire P`: advance, vs=1. Key (P, pid, 1) -- inserted.
- `acquire C`: no advance. Key (C, pid, 1) -- inserted.
- `acquire P`: advance, vs=2. Key (P, pid, 2) -- inserted.
- `acquire C`: no advance. Key (C, pid, 2) -- inserted.

No duplicates (P@1 != P@2, C@1 != C@2). virtualStage=2 > 0. Single-phase.**

Each P acquire advances the stage, so the two P acquires see different stages:
2 observations + 2 advances means A(s)=1
even though each semaphore is acquired twice per iteration.

### Example 3: Two Acquires, One Does Not Advance -- MULTIPHASE

```
for {
    acquire P @1   // advance (mma useD=false -> FreshWrite)
    mma F      @1
    release C  @1

    acquire C @2   // observation (tmem_load)
    tmem_load  @2
    release P  @2

    acquire P @1   // NO advance (mma useD=true -> NOT FreshWrite)
    mma T      @1
    release C  @1

    acquire C @2   // observation (tmem_load)
    tmem_load  @2
    release P  @2
}
```

**Trace**:
- `acquire P`: advance, vs=1. Key (P, pid, 1) -- inserted.
- `acquire C`: no advance. Key (C, pid, 1) -- inserted.
- `acquire P`: NO advance. Key (P, pid, 1) -- DUPLICATE!

**Multiphase.** The second P acquire does not advance (MMA with `useD=true` reads
the accumulator, which is not a FreshWrite), so both P acquires land at the same
virtual stage. A(s)=2 for the P semaphore.

### Example 4: Three Semaphores, Fan-Out -- SINGLE-PHASE

```
for {
    acquire P @1   // advance (store)
    store buf  @1
    release C  @1
    release D  @1

    acquire C @2   // observation
    load buf   @2
    release P  @2

    acquire D @3   // observation
    load buf   @3
    release P  @3
}
```

**Trace**:
- `acquire P`: advance, vs=1. Key (P, pid, 1) -- inserted.
- `acquire C`: no advance. Key (C, pid, 1) -- inserted.
- `acquire D`: no advance. Key (D, pid, 1) -- inserted.

No duplicates. virtualStage=1 > 0. Single-phase.

P has 1 acquire, C has 1 acquire, D has 1 acquire. A(s)=1 for all s.

---

## Attribute Tagging

Each `SemaphoreCreateOp` is tagged with the `nvws.use_single_phase` `BoolAttr`
after the eligibility analysis runs:

```mlir
%empty = nvws.semaphore.create %buf true  {nvws.use_single_phase = true}
%full  = nvws.semaphore.create %buf false {nvws.use_single_phase = true}
```

This attribute is consumed in two places:

1. **`assignStateInBlock`** (same file): At each `SemaphoreAcquireOp`, the method
   checks `shouldUseSinglePhase(semaphore)`. If true, the phase update is:
   ```cpp
   // Single-phase: flip on wrap
   auto wrapped = createOp(phasePids, arith::CmpIOp{},
                            arith::CmpIPredicate::eq, acquireStage, c0);
   lanePhase = createOp(phasePids, arith::SelectOp{}, wrapped,
                         nextPhase, lanePhase);
   ```
   If false (multiphase), the phase update is:
   ```cpp
   // Multiphase: per-stage bit flip
   auto phaseBit = createOp(phasePids, arith::ShLIOp{}, c1, acquireStage);
   lanePhase = createOp(phasePids, arith::XOrIOp{}, lanePhase, phaseBit);
   ```

2. **`LowerSemaphore.cpp`**: Reads the `kUseSinglePhaseAttrName` attribute from
   the `SemaphoreCreateOp` to determine initial phase values and lowering strategy
   for the mbarrier hardware primitives.

The initial phase values differ between modes:

| Mode | `is_released=true` (empty) | `is_released=false` (full) |
|------|---------------------------|---------------------------|
| Single-phase | `0x00000000` (0) | `0x00000001` (1) |
| Multiphase | `0x00000000` (0) | `0xFFFFFFFF` (-1, all bits set) |

---

## Lit Test Correlation

Tests in `test/NVWS/assign_semaphore_stage_phase.mlir` exercise both single-phase
and multiphase code paths. Each test's CHECK lines verify the `nvws.use_single_phase`
attribute on the `SemaphoreCreateOp` and the resulting arithmetic (select+xor for
single-phase, shli+xor for multiphase).

### Tests that produce SINGLE-PHASE

| Test | Reason |
|------|--------|
| `@warp_specialize_tma_matmul` | depth=1, single-phase trivially (one stage) |
| `@matmul_tma_acc_with_unconditional_user` | depth=2, acquires have unique keys, single-phase |
| `@attention_forward` | Mixed depths across buffer groups; all groups eligible |
| `@for_loop_control_operand_ppg` | depth=1, single-phase trivially |

### Tests that produce MULTIPHASE

| Test | Reason |
|------|--------|
| `@assign_stage_basic` | depth=2, no advance in loop (first use is load, vs=0) |
| `@assign_stage_buffer` | depth=2, multiphase (first use is conditional/observation) |
| `@matmul_tma_acc_with_conditional_user` | depth=2, multiphase for acc group |

### Tests with mixed groups

| Test | Detail |
|------|--------|
| `@matmul_tma_persistent_ws_kernel` | AB group (depth=3): single-phase. ACC group (depth=2): multiphase. Separate buffer groups get independent decisions. |

The CHECK lines in these tests verify the attribute value directly:
```mlir
// CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true {nvws.use_single_phase = true}
// CHECK: [[FULL:%.*]] = nvws.semaphore.create %{{.*}} false {nvws.use_single_phase = true}
```

And for multiphase, the arithmetic pattern includes `shli`:
```mlir
// CHECK: [[EMPTY:%.*]] = nvws.semaphore.create %{{.*}} true {nvws.use_single_phase = false}
// CHECK: arith.shli
// CHECK: arith.xori
```

---

## Summary of Decision Tree

```
computeSinglePhaseEligibility():
    |
    +-- depth == 1? ---------> SINGLE-PHASE (trivial)
    |
    +-- No warp-specialized loop? -> MULTIPHASE (conservative)
    |
    +-- Walk loop body:
    |     |
    |     +-- Duplicate (sema, pid, vs) key? -> MULTIPHASE
    |     |
    |     +-- virtualStage == 0 after walk? --> MULTIPHASE (no advances)
    |     |
    |     +-- Otherwise ----------------------> SINGLE-PHASE
```
