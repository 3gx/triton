# AssignSemaphoreStagePhase -- The Fresh-Write Rule Algorithm

**Source**: `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignSemaphoreStagePhase.cpp` (~1129 lines)

**Test**: `test/NVWS/assign_semaphore_stage_phase.mlir` (12 CHECK-LABELs)

## 1. Purpose

After insertion passes create semaphore ops with absent `stage` and `phase`
operands, `AssignSemaphoreStagePhase` fills them in. It assigns concrete SSA
values to:

- **stage**: which slot of the multi-buffered shared memory (or TMEM) the
  acquire/release/buffer op refers to.
- **phase**: the mbarrier phase parity that ensures a wait does not succeed on
  a stale signal from a prior iteration.

This pass is the heart of the semaphore design. It replaces the aref
counter-based `AssignStagePhase` (which incremented a running counter at every
`put.enter`/`get.enter`) with the following approach: stage advances
only when the first use after an acquire is a **fresh write** to the buffer.

## 2. Position in the Pipeline

```
  -> InsertSemaphore / InsertTmemSemaphore
  -> SCCP -> CSE
  -> LowerSemaphore
       +-> combineSemaphores
       +-> multiBufferSemaphore
       +-> AssignSemaphoreStagePhase   <-- THIS PASS
       +-> pattern rewrite (-> mbarrier)
  -> PartitionLoops
```

At this point in the pipeline, semaphore ops exist with correct partition
annotations and multi-buffered depths, but all `stage` and `phase` operands
are `Optional<I32>` set to absent.

## 3. Single-Traversal Architecture

The pass uses a single forward walk that assigns stage and
phase together, maintaining a `State` struct threaded through the walk:

```
Single-traversal architecture:
  One walk assigns stage and phase together.
  Stage remains a shared buffer property.
  Phase keeps per-(partition, semaphore) lanes to preserve existing behavior.
```

This eliminates an entire class of bugs where stage and phase state could
diverge due to separate traversals making different control-flow decisions.

## 4. Buffer Groups

Semaphores are grouped by their backing buffer -- the first buffer operand
of `SemaphoreCreateOp`. All semaphores that share the same first buffer form a
buffer group and share a single stage counter.

```
Grouping in assignSemaphoreStagePhase():

  MapVector<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
  for (auto semaOp : semaOps) {
    semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
  }
```

For example, a producer/consumer pair for SMEM:

```mlir
%empty = nvws.semaphore.create %buf true  : !nvws.semaphore<[...], 3>
%full  = nvws.semaphore.create %buf false : !nvws.semaphore<[...], 3>
```

Both `%empty` and `%full` share `%buf` as their first buffer, so they form a
single buffer group. They share the same `stage` counter: when the producer
advances stage, the consumer sees the same stage value.

This is proven necessary and sufficient in Appendix B of
[08_proofs.md](08_proofs.md): semaphores guarding the same physical buffer
must agree on which buffer slot is in use.

## 5. The State Data Structure

Each buffer group maintains a `State` during the forward walk:

```cpp
struct State {
  Value stage;                   // shared stage index (i32)
  SmallVector<Value> basePhases; // initial phase per semaphore index
  MapVector<LaneKey, Value> lanes; // phase per (partitionId, semaphoreIndex)
  Value token;                   // SSA token for stage propagation
};
```

### Fields

| Field | Type | Scope | Description |
|-------|------|-------|-------------|
| `stage` | `Value` (i32) | Shared across entire buffer group | Which buffer slot is currently active. All semaphores in the group see the same stage. |
| `basePhases` | `SmallVector<Value>` | Per semaphore index | Initial phase for each semaphore. `0x00000000` for `is_released=true`, `0xFFFFFFFF` (multiphase) or `0x00000001` (single-phase) for `is_released=false`. |
| `lanes` | `MapVector<LaneKey, Value>` | Per (partition, semaphore) | The current phase value for a specific partition accessing a specific semaphore. A lane is lazily initialized from `basePhases[semaIdx]` on first access. |
| `token` | `Value` | Per buffer group | The SSA `AsyncToken` from the most recent acquire. Used to propagate stage values to downstream release/buffer ops via `propagateStage`. |

Where `LaneKey = std::pair<int, int>` is `(partitionId, semaphoreIndex)`.

### Initialization

```cpp
State initState;
initState.stage = arith.constant(depth - 1, i32);
// Per-semaphore initial phases:
for (auto semaOp : semaOps) {
  if (singlePhase)
    initPhase = semaOp.isReleased ? 0x00000000 : 0x00000001;
  else
    initPhase = semaOp.isReleased ? 0x00000000 : 0xFFFFFFFF;
  initState.basePhases.push_back(arith.constant(initPhase, i32));
}
```

The initial stage is `depth - 1`. This is deliberately set so that the first
 acquire, which performs a FreshWrite, wraps the stage to 0. 

## 6. The Fresh-Write Rule (Stage Computation)

### AccessKind Classification

Every operation is classified into one of four access kinds relative to a
buffer group:

```cpp
enum class AccessKind { None, Observation, FreshWrite, FreshWriteMMA };
```

| AccessKind | Operations | Meaning |
|------------|-----------|---------|
| `None` | Everything not touching this group's buffer | Irrelevant |
| `Observation` | `LocalLoadOp`, `TMEMLoadOp`, MMA reading A/B operand | Reads existing data without overwriting |
| `FreshWrite` | `LocalStoreOp`, `DescriptorLoadOp`, `DescriptorGatherOp`, `TMEMStoreOp` | Writes new data, overwriting the buffer contents |
| `FreshWriteMMA` | MMA writing to accumulator operand | Writes MMA result into the buffer |

The classification is determined by `classifyAccess(op)` which checks if the
operation's source/destination buffer is a view of this group's buffer (via
`isGroupView`).

### First-Use Analysis: `isFirstUseFreshWriteAfterAcquire`

For each acquire, the pass asks: what is the first thing the code does with
the buffer obtained from this acquire?**

The answer determines whether stage advances:

```
isFirstUseFreshWriteAfterAcquire(acquireOp):
  1. Start scanning forward from the acquire's position in the block.
  2. Follow the acquire's token chain.
  3. Skip SemaphoreBufferOps (they create views, not accesses).
  4. If a scf.for takes the token as init_arg, follow it into the loop body.
  5. First non-view buffer access found:
     - FreshWrite or FreshWriteMMA -> return true  (stage advances)
     - Otherwise                   -> return false (stage stays)
  6. If end of block reached in a for-loop body, check if the token wraps
     back to the top via yield -> iter_arg, and scan from the top.
```

Pseudocode:

```
function isFirstUseFreshWriteAfterAcquire(acquireOp):
    token = acquireOp.getToken()
    for op in block[after acquireOp]:
        if op creates a view of token's buffer:
            continue
        if op is scf.for and token flows into its init_args:
            iterArgToken = forOp.regionIterArg[pos]
            return isFirstUseFreshWriteInBlock(forOp.body, iterArgToken)
        access = classifyAccessForToken(op, token)
        if access in {FreshWrite, FreshWriteMMA}:
            return true
        if access != None:
            return false     // Observation first
        if op uses a view of this token's buffer:
            return false     // Unknown use, conservatively Observation
    // Reached end of for-loop body; check wrap-around
    if block is a for-loop body and token yields back:
        iterArgToken = forOp.regionIterArg[yieldPos]
        return isFirstUseFreshWriteInBlock(block, iterArgToken)
    return false
```

### Stage Advancement

At each acquire, if `isFirstUseFreshWriteAfterAcquire` returns true, stage
advances:

```
// Pseudocode
if advanceStage:
    acquireStage = (stage + 1) == depth ? 0 : stage + 1
else:
    acquireStage = stage   // no change
state.stage = acquireStage
```

This emits the following arithmetic:

```mlir
%c1 = arith.constant 1 : i32
%c0 = arith.constant 0 : i32
%cDepth = arith.constant <depth> : i32
%next = arith.addi %stage, %c1 : i32
%wrapped = arith.cmpi eq, %next, %cDepth : i32
%acquireStage = arith.select %wrapped, %c0, %next : i32
```

### Why the Fresh-Write Rule Works

Consider a depth-2 double-buffer with initial stage = 1:

```
Iteration 0:
  Producer acquire EMPTY:
    first use = DescriptorLoad (FreshWrite) -> advance: stage = (1+1)%2 = 0
    acquire EMPTY at stage 0, write to buffer[0]
    release FULL at stage 0

  Consumer acquire FULL:
    first use = LocalLoad (Observation) -> no advance: stage stays 0
    acquire FULL at stage 0, read buffer[0]
    release EMPTY at stage 0

Iteration 1:
  Producer acquire EMPTY:
    first use = DescriptorLoad -> advance: stage = (0+1)%2 = 1
    acquire EMPTY at stage 1, write to buffer[1]
    release FULL at stage 1

  Consumer acquire FULL:
    first use = LocalLoad -> no advance: stage stays 1
    acquire FULL at stage 1, read buffer[1]
    release EMPTY at stage 1
```

The stage index correctly cycles through the buffer slots. The producer always
advances (it writes fresh data). The consumer never advances (it reads what the
producer wrote).

## 7. Comparison with Aref AssignStagePhase (Counter-Based)

| Aspect | Aref Counter-Based | Semaphore Fresh-Write Rule |
|--------|--------------------|-----------------------|
| **Stage rule** | Increment at every `put.enter` / `get.enter` | Advance only at FreshWrite first-use |
| **Stage tracking** | Per aref per partition | Per buffer group (shared across all semaphores) |
| **Phase representation** | Scalar that flips on wrap (single-phase) | Bit-vector (multiphase) or scalar (single-phase) |
| **TMEM support** | Limited to 2-partition only | N-partition support |

The counter-based approach increments stage at every enter operation, treating
producers and consumers symmetrically. The fresh-write rule is asymmetric: only
fresh-write acquires advance stage. This eliminates the need for separate
tracking per partition and per aref direction, since all semaphores in a buffer
group share a single stage counter.

## 8. Phase Computation -- Multiphase Mode

### Concept

In multiphase mode, phase is a 32-bit integer** where bit `i` tracks the
phase parity for stage `i`. This allows each stage to independently cycle
between odd and even phases, supporting up to 32 pipeline stages.

### Phase Update

At each acquire:

```
phaseBit = 1 << acquireStage
lanePhase = lanePhase XOR phaseBit
```

This flips exactly the bit corresponding to the current stage. The acquire then
uses the updated `lanePhase` as its phase argument.

Generated IR:

```mlir
%c1 = arith.constant 1 : i32
%phaseBit = arith.shli %c1, %acquireStage : i32
%newPhase = arith.xori %lanePhase, %phaseBit : i32
nvws.semaphore.acquire %sema[%acquireStage, %newPhase]
```

### Initial Values

| `is_released` | Initial phase | Meaning |
|---------------|---------------|---------|
| `true` | `0x00000000` | All bits zero: first XOR at any stage flips to 1, matching the mbarrier's expected phase after its initial release |
| `false` | `0xFFFFFFFF` | All bits one: first XOR at any stage flips to 0, matching the mbarrier's expected phase before any release |

### Why Bit-Vector?

Consider depth=3. The producer cycles stage 0, 1, 2, 0, 1, 2, ...
At each stage, the mbarrier alternates between phase 0 and phase 1. But stage 0
might be on its 5th use (phase 1) while stage 2 is only on its 3rd use (phase 1
also, by coincidence). The bit-vector tracks each stage independently, ensuring
the acquire always uses the correct phase for the specific stage being accessed.

## 9. Phase Computation -- Single-Phase Mode

### Concept

When the pass determines that single-phase is safe (see
[06_single_phase_optimization.md](06_single_phase_optimization.md)), phase
collapses to a **single scalar** (0 or 1) instead of a bit-vector.

### Phase Update

At each acquire:

```
if acquireStage == 0:
    lanePhase = lanePhase XOR 1     // flip on wrap
else:
    lanePhase = lanePhase            // no change
```

Generated IR:

```mlir
%c1 = arith.constant 1 : i32
%nextPhase = arith.xori %lanePhase, %c1 : i32
%c0 = arith.constant 0 : i32
%wrapped = arith.cmpi eq, %acquireStage, %c0 : i32
%newPhase = arith.select %wrapped, %nextPhase, %lanePhase : i32
nvws.semaphore.acquire %sema[%acquireStage, %newPhase]
```

### Initial Values

| `is_released` | Initial phase | Meaning |
|---------------|---------------|---------|
| `true` | `0` | First wrap-flip goes to 1 |
| `false` | `1` | First wrap-flip goes to 0 |

### Eligibility

Single-phase mode is safe when `A(s) = 1` for all stages `s` -- where `A(s)=1` means that  stage `s` is
acquired at most once per loop iteration. The pass computes this via
`computeSinglePhaseEligibility()`:

1. If `depth == 1`, always single-phase (one stage, nothing to cycle).
2. Find the warp-specialized for-loop containing group acquires.
3. Walk the loop body, tracking a `virtualStage` counter. Increment on
   FreshWrite acquires. Track `(semaphore, partitionId, virtualStage)` tuples.
4. If any duplicate tuple is found (same semaphore acquired at same virtual
   stage by same partition), fall back to multiphase.
5. If `virtualStage == 0` after the walk (no advances at all), fall back to
   multiphase.

This is an all-or-nothing decision per buffer group using in semaphores. The result is stored as
the `nvws.use_single_phase` attribute on each `SemaphoreCreateOp`.



## 10. Complete Walk Example

Consider a simple depth-2 SMEM double buffer with producer (partition 0) and
consumer (partition 1):

```
Initial state:
  stage = 1     (depth - 1)
  basePhases = [0x00000000 (empty, is_released=true),
                0xFFFFFFFF (full,  is_released=false)]

scf.for {
  (1) Acquire EMPTY {partition=0}:
      first use = DescriptorLoad -> FreshWrite -> advance stage
      acquireStage = (1+1)%2 = 0
      phase_empty_p0 = 0x00000000 XOR (1<<0) = 0x00000001
      -> acquire %empty[0, 0x00000001]
      state.stage = 0, state.token = %tok_e

  (2) SemaphoreBuffer, DescriptorLoad, Release FULL {partition=0}:
      propagateStage assigns stage=0 to buffer and release

  (3) Acquire FULL {partition=1}:
      first use = LocalLoad -> Observation -> NO advance
      acquireStage = 0  (unchanged)
      phase_full_p1 = 0xFFFFFFFF XOR (1<<0) = 0xFFFFFFFE
      -> acquire %full[0, 0xFFFFFFFE]
      state.stage = 0

  (4) SemaphoreBuffer, LocalLoad, Release EMPTY {partition=1}:
      propagateStage assigns stage=0 to buffer and release

  scf.yield stage=0, phase_empty_p0=0x1, phase_full_p1=0xFFFFFFFE
}
```

## 11. Lit Test Coverage

The test file `test/NVWS/assign_semaphore_stage_phase.mlir` contains 12
CHECK-LABELs:

| Test | Description | Key properties |
|------|-------------|----------------|
| `@assign_stage_basic` | Depth=2, single semaphore, one acquire/release per iteration | Observation (LocalLoad precedes LocalStore) -> no stage advance, multiphase XOR |
| `@shared_stage_two_semaphores` | Two semaphores on same buffer, depth=2 | Both share single stage counter, first-use is FreshWrite -> advance stage |
| `@if_observation` | Semaphore use inside scf.if (no acquire in if) | Stage/phase threading not needed for if (no acquires inside); buffer used inside if gets correct stage |
| `@two_consumers` | Depth=3, producer (p0) + 2 consumers (p1, p2) | Producer advances stage; consumers do not; 4 iter_args (stage + 3 phase lanes); per-consumer phase lanes |
| `@semaphore_lowering` | Two independent semaphore groups, one with conditional use | Each group has independent stage counter; scf.if threads stage+phases for group1 when acquire is in else branch |
| `@warp_specialize_tma_matmul` | TMEM single-phase, depth=1 | Single-phase flip: `if stage==0 then phase XOR 1 else phase`; stage always wraps to 0 at depth=1 |
| `@matmul_tma_acc_with_unconditional_user` | TMEM double-buffer, single-phase, depth=2 | `nvws.use_single_phase = true`; consumer reads accumulator (Observation), producer writes (FreshWriteMMA via tmem_store) |
| `@assign_stage_buffer` | TMEM double-buffer, multiphase, depth=2 | `nvws.use_single_phase = false`; scf.if threads token/stage/phase through both branches |
| `@attention_forward` | 3 TMEM groups, mixed depths (2,1,1), 4 partitions | Independent stage counters per group; all single-phase; complex iter_arg threading with 13 extra values |
| `@matmul_tma_acc_with_conditional_user` | TMEM depth=2, multiphase, acquire in conditional | scf.if adds extra results for stage+phase; else branch passes through unchanged state |
| `@matmul_tma_persistent_ws_kernel` | Persistent kernel with nested loops, 2 buffer groups | Outer loop threads AB and ACC states; inner loop threads AB state; AB is single-phase (depth=3), ACC is multiphase (depth=2) |
| `@for_loop_control_operand_ppg` | Nested for-loops, token as control operand | Token crosses for-loop boundary as init_arg; `propagateStage` follows token through nested for-loop to set stage on release/buffer ops |

## 12. Algorithmic Summary

```
AssignSemaphoreStagePhase::run(semaOps):

  1. GROUPING
     Group semaphores by first buffer operand.
     Compute depth = max(numStages) across group.
     Collect all partition IDs and wsTag from acquires.

  2. SINGLE-PHASE ELIGIBILITY
     Create temporary analyzer, call computeSinglePhaseEligibility().
     Tag each SemaphoreCreateOp with nvws.use_single_phase = true/false.

  3. INITIALIZATION
     stage = arith.constant(depth - 1)
     basePhases[i] = arith.constant(initial phase for semaphore i)
     Set wsTag on init constants.

  4. FORWARD WALK (assignStateInBlock)
     For each op in program order:
       - SemaphoreAcquireOp:
           Classify first-use -> advance stage or not
           Emit stage arithmetic (add/cmp/select for wrap)
           Emit phase arithmetic (XOR + shift for multiphase, XOR + select for single-phase)
           Update state.stage, state.lanes[key], state.token
           Set acquire's stage and phase operands

       - scf.ForOp:
           Add extra iter_args for stage + used phase lanes
           Walk body recursively
           Yield updated state
           Record tokToStagePosMap entries

       - scf.IfOp:
           Add extra result types
           Walk both branches with copies of incoming state
           Yield from both, merge state via if results

  5. STAGE PROPAGATION (propagateStage)
     For each acquire in the group:
       Follow token chain to release/buffer ops
       Set their stage operands via propagateStage()

  6. BACKWARD-SLICE ANNOTATION
     For each ws-loop result of integer type:
       If used by unpartitioned ops, annotate backward slice with partition=0

  7. VERIFICATION
     Assert: every acquire has stage and phase assigned
     Assert: every release/buffer has stage assigned
```



## 13. Invariants and Guarantees

After the pass completes:

1. **Every `SemaphoreAcquireOp**` in the group has non-null `stage` and `phase`
   operands.
2. Every `SemaphoreReleaseOp` and `SemaphoreBufferOp` in the group has a
   non-null `stage` operand (set by `propagateStage`).
3. All stage/phase arithmetic carries correct `ttg.partition` annotations.
4. Stage values are shared across all semaphores in the buffer group.
5. Phase values are per-(partition, semaphore) and correctly track the mbarrier
   phase parity.
6. scf.for and scf.if ops that contain group acquires have been widened with
   extra iter_args/results to thread state.
7. `nvws.use_single_phase` is set as a boolean attribute on each
   `SemaphoreCreateOp`.
