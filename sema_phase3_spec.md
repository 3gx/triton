# Phase 3 Spec: AssignSemaphoreStagePhase

**Depends on:** Phase 1 (semaphore ops must exist)
**Checkpoint:** `ninja triton-opt` builds + `test/NVWS/assign_semaphore_stage_phase.mlir` passes
**Non-breakage:** All existing lit tests pass. `AssignStagePhase.cpp` is NOT modified.

## What this phase does

Replace the Phase 1 stub in `AssignSemaphoreStagePhase.cpp` with real implementation.
NEW helper file `SemaphoreUtilities.h`. Pass: `--nvws-assign-semaphore-stage-phase`.
- Input: IR with semaphore ops (stage/phase absent)
- Output: IR with semaphore ops (stage/phase assigned)

The pass uses the **observation-based rule** to determine when to advance the stage.
It does NOT know producer/consumer roles. It classifies operations on the buffer.

## 1. New file: `SemaphoreUtilities.h`

### ThreadValue<T> template

This is a generic utility for threading a value through `scf.for` iter_args and
`scf.if` results. It is parameterized on an operation type `T` that triggers
value updates. Full implementation (from PoC commit `4bcad78`):

```cpp
#ifndef NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H
#define NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::nvws {

// ThreadValue<T> threads a value through control flow (scf.for, scf.if),
// updating it at every occurrence of operation type T.
//
// Usage:
//   ThreadValue<SemaphoreAcquireOp>::run(wgOp, initValue, updateValue);
//
// Where:
//   initValue(builder, defOp) -> Value: creates the initial value from the
//     defining op (e.g., SemaphoreCreateOp)
//   updateValue(builder, currentValue, triggerOp) -> Value: computes the next
//     value at each trigger op occurrence
//
// The template:
// 1. Walks all regions of wgOp to find uses of T's first operand (the key)
// 2. Initializes a ValueMap mapping each key to its initial value
// 3. Walks blocks: at each T op, calls updateValue; at scf.for/scf.if,
//    threads the value through iter_args/results

template <class T> struct ThreadValue {
  std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)> updateValue;
  using ValueMap = llvm::MapVector<Value /*key*/, Value /*value*/>;
  using UseSet = llvm::SetVector<Value /*key*/>;

  // Find all keys (first operand of T) used in a block and nested regions
  UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(op.getOperand(0)); // key = semaphore SSA value
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        useSet = analyzeUseInBlock(forOp.getBody(), useSet);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        useSet = analyzeUseInBlock(ifOp.thenBlock(), useSet);
        if (ifOp.elseBlock())
          useSet = analyzeUseInBlock(ifOp.elseBlock(), useSet);
      }
    }
    return useSet;
  }

  // Thread value through a scf.for loop
  void assignValueInForOp(scf::ForOp forOp, ValueMap &valueMap) {
    auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
    if (useInBlock.empty()) return;

    // Add extra iter_args for each key used in the loop body
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIterArgs.push_back(valueMap.lookup(key));
      valueRefs.push_back(&valueMap[key]);
    }

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // Update valueMap with the new iter_args inside the loop body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // Recursively assign values in the loop body
    auto valueMapInBlock = assignValueInBlock(forOp.getBody(), valueMap);

    // Append updated values to the yield op
    SmallVector<Value> extraYieldArgs;
    for (auto key : useInBlock)
      extraYieldArgs.push_back(valueMapInBlock[key]);
    appendToForOpYield(forOp, extraYieldArgs);

    // Update valueMap with loop results
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getResult(idx);
  }

  // Thread value through a scf.if op
  void assignValueInIfOp(scf::IfOp ifOp, ValueMap &valueMap) {
    auto useInBlock = analyzeUseInBlock(ifOp.thenBlock(), {});
    if (useInBlock.empty()) return;

    useInBlock = ifOp.elseBlock()
                     ? analyzeUseInBlock(ifOp.elseBlock(), useInBlock)
                     : useInBlock;

    // Add extra results to the if op
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIfResults.push_back(valueMap.lookup(key).getType());
      valueRefs.push_back(&valueMap[key]);
    }

    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // Assign in then-block and else-block
    auto thenMap = assignValueInBlock(newIfOp.thenBlock(), valueMap);
    auto elseMap = ifOp.elseBlock()
                       ? assignValueInBlock(newIfOp.elseBlock(), valueMap)
                       : valueMap;

    // Append values to yields
    auto thenYield = newIfOp.thenYield();
    auto elseYield = newIfOp.elseYield();
    for (auto key : useInBlock) {
      thenYield->insertOperands(thenYield.getNumOperands(), thenMap[key]);
      elseYield->insertOperands(elseYield.getNumOperands(), elseMap[key]);
    }
    ifOp.erase();

    // Update valueMap with if results
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *valueRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  // Walk a block, updating values at each T op and threading through control flow
  ValueMap assignValueInBlock(Block *block, ValueMap valueMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        ImplicitLocOpBuilder b(op.getLoc(), &op);
        b.setInsertionPointAfter(&op);
        auto value = valueMap.lookup(op.getOperand(0));
        valueMap[op.getOperand(0)] = updateValue(b, value, &op);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignValueInForOp(forOp, valueMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignValueInIfOp(ifOp, valueMap);
      }
    }
    return valueMap;
  }

  // Entry point: run on a WarpGroupOp
  static void run(
      WarpGroupOp wgOp,
      std::function<Value(ImplicitLocOpBuilder &, Operation *)> initValue,
      std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)> updateValue) {
    ThreadValue<T> value{updateValue};
    UseSet useSet;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      useSet = value.analyzeUseInBlock(block, useSet);
    }

    // Initialize values
    ValueMap valueMap;
    for (auto key : useSet) {
      ImplicitLocOpBuilder b(key.getLoc(), key.getDefiningOp());
      b.setInsertionPointAfter(key.getDefiningOp());
      valueMap[key] = initValue(b, key.getDefiningOp());
    }

    // Assign in all regions
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      value.assignValueInBlock(block, valueMap);
    }
  }
};

} // namespace mlir::triton::nvws
#endif // NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H
```

## 2. New file: `AssignSemaphoreStagePhase.cpp`

### 2.1 Pass structure

**Required boilerplate at top of file** (replaces the Phase 1 stub):
```cpp
#define GEN_PASS_DEF_NVWSASSIGNSEMAPHORESTAGEPHASE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
```

```cpp
class NVWSAssignSemaphoreStagePhase
    : public impl::NVWSAssignSemaphoreStagePhaseBase<NVWSAssignSemaphoreStagePhase> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    m.walk([&](FuncOp funcOp) {
      if (failed(assignSemaphoreStagePhase(funcOp)))
        signalPassFailure();
    });
  }
};
```

### 2.2 Main function: `assignSemaphoreStagePhase(FuncOp)`

```cpp
LogicalResult assignSemaphoreStagePhase(FuncOp funcOp) {
    // Step 1: Group semaphores by buffer
    DenseMap<Value, SmallVector<SemaphoreCreateOp>> bufferGroups;
    funcOp.walk([&](SemaphoreCreateOp op) {
        bufferGroups[op.getBuffers()[0]].push_back(op);
    });

    // Step 2: For each buffer group, assign stage
    for (auto &[buffer, semaphores] : bufferGroups) {
        assignStageForBufferGroup(funcOp, buffer, semaphores);
    }

    // Step 3: For each semaphore, assign phase (MULTIPHASE)
    // Uses ThreadValue<SemaphoreAcquireOp>
    // (see section 2.4)

    // Step 4: Propagate stage to release/buffer ops via token chain
    // (see section 2.5)

    // Step 5: Preserve backward-slice partition logic
    // (see section 2.6)

    return success();
}
```

### 2.3 Step 2: assignStageForBufferGroup

Determine depth from buffer type. Walk IR in program order. At each op using a
buffer view from this group, classify and emit stage arithmetic.

**Classification (by operation type only — no def-use analysis):**

| Operation | Classification | Rationale |
|-----------|---------------|-----------|
| `ttg.local_load %view` | OBSERVATION | Reads buffer |
| `ttng.tmem_load %view` | OBSERVATION | Reads buffer |
| `tc_gen5_mma ..., %view, ...` (SMEM operand, not accumulator) | OBSERVATION | MMA reads SMEM input |
| `ttg.local_store %val, %view` | FRESH WRITE | Writes new data |
| `nvws.descriptor_load ..., %view` | FRESH WRITE | TMA writes |
| `nvws.descriptor_gather ..., %view` | FRESH WRITE | TMA gather writes |
| `ttng.tmem_store %val, %view` | FRESH WRITE | Writes new data |
| `tc_gen5_mma ..., %view, ..., useD=false` (accumulator) | FRESH WRITE | Discards old |
| `tc_gen5_mma ..., %view, ..., useD=true` (accumulator) | REUSE | Accumulates on old |

**How to find buffer views from a semaphore group:**
1. From `SemaphoreCreateOp`, follow users to `SemaphoreAcquireOp`
2. From acquire's token, follow users to `SemaphoreBufferOp`
3. From buffer op's results, follow users — those are the data operations

**State:** Two iter_args per buffer group: `%bufId` (i32, init=0) and
`%was_observed` (i1, init=false).

**At each OBSERVATION op:** Set `%was_observed = true`.
If inside `scf.if`: then-branch yields `true`, else-branch yields current `%was_observed`.

**At each FRESH WRITE op:** Emit:
```mlir
// %is_fresh: constant true for all except tc_gen5_mma
// For tc_gen5_mma: %is_fresh = arith.xori %useD, %c1
%should_advance = arith.andi %was_observed, %is_fresh
%next = arith.addi %bufId, %c1
%wrap = arith.cmpi eq, %next, %cDepth
%wrapped = arith.select %wrap, %c0, %next
%bufId_new = arith.select %should_advance, %wrapped, %bufId
%was_observed_new = arith.select %should_advance, %cFalse, %was_observed
```

**At REUSE ops:** No change to `%bufId` or `%was_observed`.

**Threading through control flow:** Use `addIterArgsToLoop` (existing utility from
`triton/Dialect/TritonGPU/Transforms/Utility.h`) for `scf.for` and
`replaceIfOpWithNewSignature` for `scf.if`. Same utilities used by current
`AssignStagePhase.cpp` at lines 118-203 (for) and 205-287 (if).

**Assign stage to semaphore ops:** After threading, for each acquire/buffer/release:
```cpp
op.getStageMutable().assign(bufId);  // bufId is clean i32, no encoding
```

### 2.4 Step 3: Assign phase (MULTIPHASE)

Use `ThreadValue<SemaphoreAcquireOp>` for each semaphore. `ThreadValue::run()`
takes a `WarpGroupOp` — find it by walking the function:

```cpp
SmallVector<WarpGroupOp> wgOps;
funcOp.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

for (auto wgOp : wgOps) {
    // ThreadValue discovers ALL semaphore keys used in the wgOp regions.
    // Run once per semaphore — but ThreadValue::run() handles ALL keys at once.
    // So call it ONCE, not per-semaphore:
    auto initPhase = [](ImplicitLocOpBuilder &b, Operation *op) -> Value {
        auto sema = cast<SemaphoreCreateOp>(op);
        return b.create<arith::ConstantIntOp>(
            sema.getIsReleased() ? 0xFFFFFFFF : 0x00000000, 32);
    };
    auto updatePhase = [](ImplicitLocOpBuilder &b, Value phase,
                          Operation *op) -> Value {
        auto acquireOp = cast<SemaphoreAcquireOp>(op);
        acquireOp.getPhaseMutable().assign(phase);
        auto phaseBit = b.create<arith::ShLIOp>(
            b.create<arith::ConstantIntOp>(1, 32), acquireOp.getStage());
        return b.create<arith::XOrIOp>(phase, phaseBit);
    };
    // One call handles ALL semaphores — ThreadValue internally keys by operand(0)
    ThreadValue<SemaphoreAcquireOp>::run(wgOp, initPhase, updatePhase);
}
```

**Note:** `ThreadValue<SemaphoreAcquireOp>::run()` internally calls
`analyzeUseInBlock()` which collects ALL semaphore SSA values used in the wgOp
regions. It creates a `ValueMap` with one entry per semaphore, each with its own
phase. The `updateValue` callback is called for each acquire, and the phase is
looked up/updated by the semaphore's SSA value (operand 0 of the acquire). So
ONE call to `run()` per `WarpGroupOp` handles ALL semaphores.

**Single-phase optimization:** When A(s)=1 for all s (see Appendix A), rewrite to:
```mlir
%flipped = arith.xori %phase, %c1
%phase_new = arith.select %wrap, %flipped, %phase
```
Detection: statically count observations vs advances per stage tenure in the buffer
group. This optimization is OPTIONAL for initial implementation.

### 2.5 Step 4: Propagate stage to release/buffer ops

After stage is assigned to acquire ops, follow the token chain to set stage on
release and buffer ops:

```cpp
void propagateStage(Value token, Value stage, DenseSet<Operation *> &visited) {
    for (auto &use : token.getUses()) {
        auto owner = use.getOwner();
        if (visited.contains(owner)) continue;
        visited.insert(owner);

        if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(owner))
            releaseOp.setStage(stage);
        if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(owner))
            bufferOp.setStage(stage);

        // Token flows through scf.for: token is an init_arg → becomes iter_arg
        if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
            auto pos = use.getOperandNumber() - forOp.getNumControlOperands();
            auto iterTok = forOp.getRegionIterArg(pos);
            // Stage was also threaded through iter_args (from step 2).
            // Find the stage iter_arg at the corresponding position.
            // The stage was added by addIterArgsToLoop — its position is tracked
            // by the stage threading logic in step 2. Use a tokToStagePosMap
            // (same pattern as AssignStagePhase.cpp:165-167, 372-373).
            auto stagePos = tokToStagePosMap.at({forOp, iterTok});
            propagateStage(iterTok, forOp.getRegionIterArgs()[stagePos], visited);
        }
        // Token flows through scf.yield → parent op result
        if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
            auto pos = use.getOperandNumber();
            auto parentOp = yieldOp->getParentOp();
            auto stagePos = tokToStagePosMap.at({yieldOp, token});
            propagateStage(parentOp->getResult(pos),
                           parentOp->getResult(stagePos), visited);
        }
    }
}
```

This mirrors `AssignStagePhase.cpp:340-382`. The `tokToStagePosMap` maps
`(Operation*, Value token)` → stage iter_arg position. Populated during step 2
when adding `%bufId` as iter_args to for/if ops (same as
`AssignStagePhase.cpp:165-167`).

### 2.6 Step 5: Backward-slice partition logic

Copy `visitBackwardSlice` helper function from `AssignStagePhase.cpp:449-492`.
This function operates on loop results and partition annotations — NOT on aref ops.
It can be copied verbatim.

The CALLING code (`AssignStagePhase.cpp:513-540`) walks `scf::ForOp` with
`kWarpSpecializeAttrName` and calls `visitBackwardSlice` on scalar results.
This code also does NOT reference aref ops — it references `scf::ForOp` results
and their users. Copy verbatim.

**What cannot be copied:** The function `assignStagePhase()` at line 494 walks
`ArefCreateOp` and calls `AssignStagePhase<ArefPutEnterOp>::run`. This is replaced
by the new `assignSemaphoreStagePhase()` function (section 2.2).

### 2.7 Partition annotations

All arithmetic ops created by the pass must carry partition annotations matching
the semaphore ops they're associated with. Use `triton::gpu::createInto<>()`
(existing utility) to propagate partition IDs.

- `%bufId` ops: annotate with ALL partition IDs from all semaphores in the group
- `%was_observed` ops: same as `%bufId`
- `%phase` ops: annotate with partition IDs from that specific semaphore's acquire ops

## 3. Files to create

| File | Change |
|------|--------|
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/AssignSemaphoreStagePhase.cpp` | REPLACE Phase 1 stub with real implementation |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/SemaphoreUtilities.h` | NEW (ThreadValue<T> template) |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt` | Already has entry from Phase 1 — no change |
| `third_party/nvidia/triton_nvidia.cc` | ADD: `ADD_PASS_WRAPPER_0("add_assign_semaphore_stage_phase", mlir::triton::createNVWSAssignSemaphoreStagePhase);` |

## 4. Lit test

Create `test/NVWS/assign_semaphore_stage_phase.mlir` with hand-written semaphore IR.

Test cases:
1. **SMEM TMA + MMA consumer (depth=2):** descriptor_load = FRESH WRITE,
   tc_gen5_mma reading SMEM = OBSERVATION. Both unconditional → `%was_observed`
   and `%is_fresh` are constants. Verify stage cycles 0,1,0,1.

2. **TMEM MMA + conditional load (depth=2):** mma with runtime useD = conditional
   FRESH WRITE, tmem_load inside `scf.if` = conditional OBSERVATION. Verify
   `%should_advance = %was_observed AND NOT %useD`.

3. **Multiple consumers (depth=2):** Two semaphores, two acquires per iteration.
   Verify both get same stage value.

4. **Nested loop:** Inner loop with mma(useD transitions). Verify `%bufId` and
   `%was_observed` threaded through inner loop iter_args.

## 5. Verification

```bash
BUILD=build/cmake.linux-x86_64-cpython-3.12
TOPT=$BUILD/bin/triton-opt

ninja -C $BUILD triton-opt
$TOPT test/NVWS/assign_semaphore_stage_phase.mlir -split-input-file \
    --allow-unregistered-dialect --nvws-assign-semaphore-stage-phase | FileCheck ...
# ALL existing tests still pass (AssignStagePhase.cpp untouched)
```

---

## Appendix A: When is Multiphase Required?

Full proof in `sema_appendix_a.md`. Summary:

**Theorem.** Single-phase is correct iff A(s) = 1 for all stages s, where A(s)
is the number of acquire-release round-trips on stage s between consecutive
stage advances through s.

**IR condition:** multiphase required iff two observations on same stage without
intervening advance. Pigeonhole: O > V implies multiphase.

## Appendix B: Shared Stage Counter for N Semaphores

Full proof (by induction) in `sema_appendix_b.md`. Summary:

**Theorem.** For N >= 2 semaphores sharing buf, all must use same %bufId.
Proved by induction: base case (N=2) + transitivity along ownership ring.
Six counterexample attempts examined — none found.
