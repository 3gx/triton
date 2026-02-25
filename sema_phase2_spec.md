# Phase 2 Spec: LowerArefToSemaphore

**Depends on:** Phase 1 (semaphore ops must exist)
**Checkpoint:** `ninja triton-opt` builds + `test/NVWS/lower_aref_to_semaphore.mlir` passes
**Non-breakage:** All existing lit tests pass. `LowerAref.cpp` is NOT modified.

## What this phase does

Replace the Phase 1 stub in `LowerArefToSemaphore.cpp` with real implementation of
pass `--nvws-lower-aref-to-semaphore`. The file already exists from Phase 1 (stub).
- Input: IR with aref ops (after InsertAref + InsertTmemAref + SCCP + CSE)
- Output: IR with semaphore ops (no arefs left). Stage/phase are Optional ABSENT.

Also: add Python binding and CMakeLists.txt entry.

## 1. New file: `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerArefToSemaphore.cpp`

### 1.1 Structure (modeled after LowerAref.cpp)

The pass `NVWSLowerArefToSemaphore::runOnOperation()` does:

```cpp
void runOnOperation() override {
    ModuleOp m = getOperation();

    // Step 1: combineArefs (copy from LowerAref.cpp lines 827-902 verbatim)
    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) {
        if (loop->hasAttr(kWarpSpecializeAttrName))
            loop->walk([&](scf::ForOp op) { loops.push_back(op); });
    });
    for (auto loop : loops) combineArefs(loop);

    // Step 2: multiBufferAref (copy from LowerAref.cpp lines 931-939 verbatim)
    SmallVector<ArefCreateOp> arefOps;
    m.walk([&](ArefCreateOp arefOp) {
        if (isProducerLoad(arefOp)) arefOps.push_back(arefOp);
    });
    multiBufferAref(arefOps, numStages);

    // Step 3: Rewrite arefs to semaphores (NEW — see section 1.3)
    RewritePatternSet patterns(context);
    patterns.add<LowerArefToSemaCreate>(context, numStages);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    applyPatternsGreedily(m, std::move(patterns), config);

    // Step 4: Hoist poison ops (copy from LowerAref.cpp lines 905-909)
    m.walk([&](FuncOp f) { hoistPoissonOps(f); });
}
```

### 1.2 Copied functions (from LowerAref.cpp)

Copy ALL helper functions from `LowerAref.cpp` EXCEPT the mbarrier-specific ones.
The easiest approach: copy everything and remove only what's not needed.

**Copy these (and their transitive dependencies):**

| Function | Lines | Transitive deps |
|----------|-------|-----------------|
| `combineArefs()` | 818-902 | `findSharedMemorySinkOps()`, `getDominantConsumer()`, `createCombinedArefOps()`, `createArefCreateOp()` (from Utilities.h) |
| `multiBufferAref()` | at 931-939 | `isProducerLoad()` 664-692, `getBufferViewType()`, `getMultiBufferedType()`, `createAlloc()` (from Utilities.h) |
| `getAsyncMMAv5Consumers()` | 582-608 | `getTopLevelUsersInLoop()` |
| `setIsAsync()` | 108-130 | `getNumStagesOrDefault()`, `isOperandPipelineable()`, `areScalesPipelineable()` |
| `propagateMutability()` | 427-435 | `getAsMutable()` 421-425 |
| `hoistPoissonOps()` | 905-909 | (none) |
| `castAsyncOpAttrs()` | 164-170 | (none) |
| `getPartitionWsTagIds()` | 73-83 | (none) |
| `assignStageCluster()` | 86-96 | (none) |

**DO NOT copy:**
- `createAndInitMbar()`, `createBarriers()`, `getEmptyBarrier()`, `getFullBarrier()`
- `insertWaitOp()`, `insertArriveBarrier()`
- `lowerTMALoad()`, `createTMALoad()`, `createTMAGather()`
- `getSubViews()` (semaphore.buffer replaces this)
- `rewritePutEnterOp()`, `rewriteGetEnterOp()`, `rewritePutExitOp()`,
  `rewriteGetExitOp()`, `rewriteArefBufferOp()` (replaced by new rewrite functions)
- `LowerArefCreate` class (replaced by `LowerArefToSemaCreate`)
- `getArrivalCount()` / `BarrierCount` (mbarrier-specific)

### 1.2a Required includes and boilerplate

```cpp
#include "Utilities.h"
// ... (copy #include list from LowerAref.cpp lines 1-57, remove mbarrier-specific)
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
// ... add semaphore op headers

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERAREFTOSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {
// ... helper functions, rewrite pattern, pass class
} // namespace

} // namespace triton
} // namespace mlir
```

### 1.3 New rewrite pattern: `LowerArefToSemaCreate`

This replaces `LowerArefCreate` from LowerAref.cpp:610-662. Instead of creating
mbarriers, it creates semaphore ops.

```cpp
struct SemaValue {
    Value sem0;   // SemaphoreCreateOp with is_released=true
    Value sem1;   // SemaphoreCreateOp with is_released=false
    SmallVector<Value> buffers;  // from arefOp.getOperands()
};

SemaValue createSemaphores(ArefCreateOp op, PatternRewriter &rewriter) {
    auto arefTy = op.getType();
    auto arefBufTypes = llvm::to_vector(llvm::map_range(
        arefTy.getBaseType(), [](Type t) { return cast<MemDescType>(t); }));
    auto depth = getArefDepth(arefBufTypes[0]);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto semaTy = SemaphoreType::get(b.getContext(), arefTy.getBaseType(), depth);
    auto sem0 = b.create<SemaphoreCreateOp>(semaTy, op.getOperands(), true);
    auto sem1 = b.create<SemaphoreCreateOp>(semaTy, op.getOperands(), false);
    return SemaValue{sem0, sem1, SmallVector<Value>(op.getOperands())};
}
```

### 1.4 Per-user rewrite logic — ORDERING IS CRITICAL

The rewrite MUST follow this order:
1. **Collect** all info from aref ops BEFORE any replacement
2. **Create** semaphore ops
3. **Replace** uses (old → new)
4. **Erase** old aref ops

This is because finding the matching `ArefPutExitOp` requires the ORIGINAL aref
token (not the new semaphore token). Same as current `LowerAref.cpp:621-630`.

**The correct `matchAndRewrite` body:**

```cpp
LogicalResult matchAndRewrite(ArefCreateOp op, PatternRewriter &rewriter) {
    // ── STEP 1: COLLECT all info from aref ops BEFORE any modification ──

    auto sema = createSemaphores(op, rewriter);
    auto mmav5Ops = getAsyncMMAv5Consumers(op.getResult());

    // Pre-collect: for each PutEnterOp, find its matching PutExitOp via token
    DenseMap<ArefPutEnterOp, ArefPutExitOp> putPairs;
    for (auto user : op->getUsers()) {
        if (auto putEnter = dyn_cast<ArefPutEnterOp>(user)) {
            for (auto tokUser : putEnter.getToken().getUsers()) {
                if (auto putExit = dyn_cast<ArefPutExitOp>(tokUser)) {
                    putPairs[putEnter] = putExit;
                    break;
                }
            }
        }
    }

    // Pre-collect: for each ArefBufferOp, determine which semaphore
    // by checking if its token comes from a PutEnter (→ sem0) or GetEnter (→ sem1)
    DenseMap<ArefBufferOp, Value> bufferSemMap;
    for (auto user : op->getUsers()) {
        if (auto bufOp = dyn_cast<ArefBufferOp>(user)) {
            Value tok = bufOp.getToken();
            // Trace back: if token def is PutEnter → sem0, GetEnter → sem1
            if (tok.getDefiningOp<ArefPutEnterOp>())
                bufferSemMap[bufOp] = sema.sem0;
            else if (tok.getDefiningOp<ArefGetEnterOp>())
                bufferSemMap[bufOp] = sema.sem1;
            else
                bufferSemMap[bufOp] = sema.sem0; // default for loop-carried tokens
        }
    }

    // ── STEP 2: CREATE semaphore ops and REPLACE uses ──

    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    for (auto userOp : op->getUsers()) {
        opToDelete.insert(userOp);
        if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
            rewriter.setInsertionPointAfter(user);
            auto tok = rewriter.create<SemaphoreAcquireOp>(
                user.getLoc(), rewriter.getType<AsyncTokenType>(), sema.sem0);
            // Buffer view
            SmallVector<Type> viewTypes;
            for (auto t : cast<SemaphoreType>(sema.sem0.getType()).getBaseType())
                viewTypes.push_back(getArefViewBufferType(cast<MemDescType>(t)));
            auto bufOp = rewriter.create<SemaphoreBufferOp>(
                user.getLoc(), viewTypes, sema.sem0, tok);
            for (auto [old, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers()))
                old.replaceAllUsesWith(view);
            user.getToken().replaceAllUsesWith(tok);

            // setIsAsync using pre-collected exitOp
            if (auto exitOp = putPairs.lookup(user)) {
                auto kinds = castAsyncOpAttrs(exitOp.getAsyncOps());
                if (llvm::any_of(kinds, [](AsyncOp k) {
                        return k == AsyncOp::TMALoad || k == AsyncOp::CpAsync; }))
                    for (auto mma : mmav5Ops) setIsAsync(mma, numStages);
            }

        } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
            rewriter.setInsertionPointAfter(user);
            auto tok = rewriter.create<SemaphoreAcquireOp>(
                user.getLoc(), rewriter.getType<AsyncTokenType>(), sema.sem1);
            SmallVector<Type> viewTypes;
            for (auto t : cast<SemaphoreType>(sema.sem1.getType()).getBaseType())
                viewTypes.push_back(getArefViewBufferType(cast<MemDescType>(t)));
            auto bufOp = rewriter.create<SemaphoreBufferOp>(
                user.getLoc(), viewTypes, sema.sem1, tok);
            for (auto [old, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers())) {
                old.replaceAllUsesWith(view);
                propagateMutability(view);
            }
            user.getToken().replaceAllUsesWith(tok);

        } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
            // Cross-release: put.exit releases sem1
            rewriter.setInsertionPointAfter(user);
            rewriter.create<SemaphoreReleaseOp>(
                user.getLoc(), sema.sem1, user.getToken(), user.getAsyncOps());

        } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
            // Cross-release: get.exit releases sem0
            rewriter.setInsertionPointAfter(user);
            rewriter.create<SemaphoreReleaseOp>(
                user.getLoc(), sema.sem0, user.getToken(), user.getAsyncOps());

        } else if (auto user = dyn_cast<ArefBufferOp>(userOp)) {
            rewriter.setInsertionPointAfter(user);
            Value sem = bufferSemMap.lookup(user);
            auto bufOp = rewriter.create<SemaphoreBufferOp>(
                user.getLoc(), user.getBuffers().getTypes(), sem, user.getToken());
            for (auto [old, view] : llvm::zip(user.getBuffers(), bufOp.getBuffers()))
                old.replaceAllUsesWith(view);

        } else {
            llvm_unreachable("unexpected aref user");
        }
    }

    // ── STEP 3: CLEANUP — replace remaining aref tokens with poison, erase ──

    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken = ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    for (auto op : sorted) {
        if (auto e = dyn_cast<ArefPutEnterOp>(op))
            e.getToken().replaceAllUsesWith(replToken);
        else if (auto e = dyn_cast<ArefGetEnterOp>(op))
            e.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
        rewriter.eraseOp(*it);

    return success();
}
```

All per-user rewrite logic is INLINED in the `matchAndRewrite` body above.
No separate helper functions needed — the logic is self-contained.

### 1.5 What is NOT done in this pass

- **No stage/phase assignment.** Semaphore ops have Optional stage/phase absent.
- **No `descriptor_load`/`descriptor_gather` lowering.** These ops stay in the IR.
- **No mbarrier creation.** That's Phase 4.

## 2. Files to create/modify

| File | Change |
|------|--------|
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerArefToSemaphore.cpp` | REPLACE Phase 1 stub with real implementation |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt` | Already has entry from Phase 1 — no change |
| `third_party/nvidia/triton_nvidia.cc` | ADD: `ADD_PASS_WRAPPER_0("add_lower_aref_to_semaphore", mlir::triton::createNVWSLowerArefToSemaphore);` after line 89 |

## 3. Lit test

Create `test/NVWS/lower_aref_to_semaphore.mlir`.

Input: IR with aref ops (hand-crafted or taken from InsertAref output — NOT from
`lower_aref.mlir` which has post-stage-assignment IR with `[%stage, %phase]` args).

Test cases:
- Basic: aref with put.enter/put.exit + get.enter/get.exit → 2 semaphores, acquire/release
- TMA load: descriptor_load preserved (not lowered)
- ArefBufferOp: mapped to SemaphoreBufferOp
- Two consumers: 1 aref → 2 semaphores, 2 get.enter → 2 acquire
- setIsAsync: verify MMAv5 marked async when producer has tma_load

CHECK for:
```
// CHECK: %[[SEM0:.*]] = nvws.semaphore.create{{.*}}true
// CHECK: %[[SEM1:.*]] = nvws.semaphore.create{{.*}}false
// CHECK: %[[TOK0:.*]] = nvws.semaphore.acquire %[[SEM0]]
// CHECK-NOT: [%
// CHECK: nvws.semaphore.buffer %[[SEM0]], %[[TOK0]]
// CHECK-NOT: [%
// CHECK: nvws.semaphore.release %[[SEM1]], %[[TOK0]] [#nvws.async_op<tma_load>]
// CHECK: nvws.descriptor_load
```

## 4. Verification

```bash
BUILD=build/cmake.linux-x86_64-cpython-3.12
TOPT=$BUILD/bin/triton-opt

ninja -C $BUILD triton-opt
$TOPT test/NVWS/lower_aref_to_semaphore.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore | FileCheck test/NVWS/lower_aref_to_semaphore.mlir
# ALL existing tests still pass (LowerAref.cpp untouched)
```
