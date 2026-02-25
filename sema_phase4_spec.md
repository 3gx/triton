# Phase 4 Spec: LowerSemaphore

**Depends on:** Phase 1 (semaphore ops must exist)
**Checkpoint:** `ninja triton-opt` builds + `test/NVWS/lower_semaphore.mlir` passes
**Non-breakage:** All existing lit tests pass. `LowerAref.cpp` NOT modified.

## What this phase does

Replace the Phase 1 stub in `LowerSemaphore.cpp` with real implementation of
pass `--nvws-lower-semaphore`. The file already exists from Phase 1 (stub).
- Input: IR with semaphore ops WITH stage/phase assigned
- Output: IR with mbarrier ops (WaitBarrierOp, ArriveBarrierOp, etc.)

## 1. New file: `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSemaphore.cpp`

### 1.1 Pass structure

**Required boilerplate at top of file** (replaces the Phase 1 stub):
```cpp
#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
```

```cpp
class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LowerSemaphoreCreate>(context);
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
};
```

### 1.2 LowerSemaphoreCreate pattern

Single pattern that processes each `SemaphoreCreateOp` and rewrites all its users.
Mirrors `LowerArefCreate` in `LowerAref.cpp:610-662`.

**CRITICAL: Two-pass approach for TMA loads.** TMA loads arrive on the RELEASED
semaphore's mbarrier, not the acquired one. But in the cross-release pattern,
the release targets a DIFFERENT `SemaphoreCreateOp`. The single-pattern greedy
rewrite may erase that other create before we can look up its mbarrier.

**Solution:** Pre-collect ALL TMA info BEFORE erasing anything.

```cpp
class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto mbars = createAndInitMbar(op, rewriter);
    auto buffers = SmallVector<Value>(op.getBuffers());

    // ── PRE-COLLECT TMA info before any modification ──
    // For each acquire whose matching release has async_ops=[tma_load],
    // collect the acquire, the descriptor loads, and the RELEASE semaphore's
    // mbar (which must be created first).
    // NOTE: TMA arrives on the RELEASED semaphore's mbar. For this semaphore,
    // the BarrierExpectOp goes on THIS mbar (because the release targeting
    // this semaphore has async_ops=[tma_load] from the other partition).
    handleTMALoads(op, rewriter, mbars);

    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    // ── FIND cleanup insertion point BEFORE erasing ──
    SetVector<Operation *> allUsers;
    for (auto u : op->getUsers()) allUsers.insert(u);
    auto sortedUsers = topologicalSort(allUsers);
    auto lastUser = op->getBlock()->findAncestorOpInBlock(*sortedUsers.back());

    for (auto userOp : op->getUsers()) {
      opToDelete.insert(userOp);
      if (auto user = dyn_cast<SemaphoreAcquireOp>(userOp))
        rewriteAcquire(user, rewriter, mbars);
      else if (auto user = dyn_cast<SemaphoreReleaseOp>(userOp))
        rewriteRelease(op, user, rewriter, mbars);
      else if (auto user = dyn_cast<SemaphoreBufferOp>(userOp))
        rewriteBuffer(user, rewriter, buffers);
      else
        llvm_unreachable("unexpected semaphore user");
    }

    // ── CLEANUP ──
    // 1. Invalidate and deallocate mbarriers BEFORE erasing ops
    //    (lastUser is still alive at this point)
    {
      ImplicitLocOpBuilder b2(op.getLoc(), rewriter);
      b2.setInsertionPointAfter(lastUser);
      int numStages = op.getType().getNumStages();
      for (int i = 0; i < numStages; i++) {
        auto view = createSingleBufferView(b2, mbars, i);
        InvalBarrierOp::create(b2, op.getLoc(), view);
      }
      LocalDeallocOp::create(b2, op.getLoc(), mbars);
    }

    // 2. Replace tokens with poison, THEN erase ops
    auto sorted = topologicalSort(opToDelete);
    OpBuilder b(op);
    auto replToken = ub::PoisonOp::create(b, op.getLoc(), b.getType<AsyncTokenType>());
    for (auto o : sorted) {
      if (auto acq = dyn_cast<SemaphoreAcquireOp>(o))
        acq.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }
};
```

**`assignStageCluster` requirement:** Every created op (WaitBarrierOp, ArriveBarrierOp,
TCGen5CommitOp, FenceOp, MemDescIndexOp, BarrierExpectOp) MUST be annotated with
`assignStageCluster(createdOp, getPartitionWsTagIds(semOp), getStageCluster(semOp), rewriter)`.
Copy helpers `getPartitionWsTagIds()` and `assignStageCluster()` from LowerAref.cpp
lines 73-96. Without these, downstream `PartitionLoops` and `ScheduleLoops` break.

### 1.3 createAndInitMbar

```cpp
Value createAndInitMbar(SemaphoreCreateOp op, PatternRewriter &rewriter) {
    int numStages = op.getType().getNumStages();
    int pendingCount = getPendingCount(op);

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto mbars = createScalarAlloc(b, rewriter.getI64Type(), numStages);
    for (int i = 0; i < numStages; i++) {
        auto view = createSingleBufferView(rewriter, mbars, i);
        InitBarrierOp::create(rewriter, op.getLoc(), view, pendingCount);
    }
    return mbars;
}
```

### 1.4 getPendingCount

De-duplicate by partition group (same as `getArrivalCount()` in `LowerAref.cpp:172-225`):

```cpp
int getPendingCount(SemaphoreCreateOp op) {
    SetVector<int> releaseGroups;
    int count = 0;

    for (auto user : op->getUsers()) {
        if (!hasPartition(user)) continue;
        auto partitionIds = getPartitionIds(user);
        assert(partitionIds.size() == 1);

        if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user)) {
            if (releaseGroups.count(partitionIds.front())) continue;
            releaseGroups.insert(partitionIds.front());
            for (auto kind : castAsyncOpAttrs(releaseOp.getAsyncOps())) {
                switch (kind) {
                case AsyncOp::TC5MMA:
                case AsyncOp::TMALoad:
                case AsyncOp::NONE:
                    count += 1;
                    break;
                default:
                    llvm_unreachable("unsupported async op");
                }
            }
        }
    }
    // If the semaphore is not used within a warp-specialized loop,
    // the pending count will be 0. Fall back to 1. (Same as LowerAref.cpp:217-222)
    if (count == 0)
        count = 1;
    return count;
}
```

### 1.5 rewriteAcquire (SemaphoreAcquireOp → WaitBarrierOp)

```cpp
void rewriteAcquire(SemaphoreAcquireOp op, PatternRewriter &rewriter, Value mbars) {
    auto loc = op.getLoc();
    rewriter.setInsertionPointAfter(op);
    auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());

    // Extract phase bit from MULTIPHASE bit-vector
    Value phaseBit = arith::ShRUIOp::create(rewriter, loc, op.getPhase(), op.getStage());
    phaseBit = arith::AndIOp::create(rewriter, loc, phaseBit,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32));

    WaitBarrierOp::create(rewriter, loc, mbar, phaseBit);
}
```

### 1.6 rewriteRelease (SemaphoreReleaseOp → arrive/commit)

Includes fence logic. Mirrors `rewritePutExitOp` (LowerAref.cpp:497-542) and
`rewriteGetExitOp` (LowerAref.cpp:544-580).

```cpp
void rewriteRelease(SemaphoreCreateOp semaOp, SemaphoreReleaseOp op,
                    PatternRewriter &rewriter, Value mbars) {
    auto loc = op.getLoc();
    auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
    rewriter.setInsertionPointAfter(op);

    // Fence detection: needed when this release has async_ops=[none] (generic proxy)
    // and the OTHER semaphore in the same buffer group has a mismatched consumer
    bool needFence = detectFenceNeeded(semaOp, op, asyncKinds);
    if (needFence) {
        FenceAsyncSharedOp::create(rewriter, loc, /*bCluster=*/false);
    }

    auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());

    for (auto asyncOpEnum : asyncKinds) {
        switch (asyncOpEnum) {
        case AsyncOp::NONE:
        case AsyncOp::WGMMA:
            ArriveBarrierOp::create(rewriter, loc, mbar, 1);
            break;
        case AsyncOp::TC5MMA:
        case AsyncOp::TMEMCopy:
            TCGen5CommitOp::create(rewriter, loc, mbar, Value(), ValueRange{});
            break;
        case AsyncOp::TMALoad:
            break; // handled at acquire site via BarrierExpectOp
        case AsyncOp::CpAsync:
        default:
            llvm_unreachable("unknown async op");
        }
    }
}
```

### 1.7 Fence detection logic

Mirrors `LowerAref.cpp:504-529` (putExit fence) and `551-568` (getExit fence).
The fence is needed when a generic proxy (async_ops=NONE) needs to synchronize
with an async consumer/producer on the other semaphore.

```cpp
bool detectFenceNeeded(SemaphoreCreateOp semaOp, SemaphoreReleaseOp releaseOp,
                       ArrayRef<AsyncOp> asyncKinds) {
    bool isGenericProxy = llvm::any_of(asyncKinds,
        [](AsyncOp k) { return k == AsyncOp::NONE; });
    if (!isGenericProxy) return false;

    // Check if buffer is TMEM — TMEM doesn't need fence
    auto semaType = cast<SemaphoreType>(semaOp.getType());
    auto bufType = cast<MemDescType>(semaType.getBaseType()[0]);
    if (bufType.getMemorySpace() == TensorMemorySpaceAttr::get(semaOp.getContext()))
        return false;

    // Find the OTHER semaphore(s) sharing the same buffer
    Value buffer = semaOp.getBuffers()[0];
    for (auto user : buffer.getUsers()) {
        auto otherSema = dyn_cast<SemaphoreCreateOp>(user);
        if (!otherSema || otherSema == semaOp) continue;

        // Check if the other semaphore's releases have async ops (TC5MMA or TMALoad)
        for (auto otherUser : otherSema->getUsers()) {
            if (auto otherRelease = dyn_cast<SemaphoreReleaseOp>(otherUser)) {
                auto otherKinds = castAsyncOpAttrs(otherRelease.getAsyncOps());
                bool hasAsyncConsumer = llvm::any_of(otherKinds,
                    [](AsyncOp k) { return k == AsyncOp::TC5MMA; });
                bool hasAsyncProducer = llvm::any_of(otherKinds,
                    [](AsyncOp k) { return k == AsyncOp::TMALoad; });
                if (hasAsyncConsumer || hasAsyncProducer) return true;
            }
        }
    }
    return false;
}
```

### 1.8 rewriteBuffer (SemaphoreBufferOp → MemDescIndexOp)

```cpp
void rewriteBuffer(SemaphoreBufferOp op, PatternRewriter &rewriter,
                   SmallVectorImpl<Value> &buffers) {
    auto loc = op.getLoc();
    rewriter.setInsertionPointAfter(op);

    for (auto [i, buffer] : llvm::enumerate(buffers)) {
        auto memDesc = cast<MemDescType>(buffer.getType());

        // TMEM scales don't support multi-buffering (LowerAref.cpp:274-277)
        if (isa<TensorMemoryScalesEncodingAttr>(memDesc.getEncoding())) {
            op.getBuffers()[i].replaceAllUsesWith(buffer);
            continue;
        }

        // Create sub-view: strip first dimension (the multi-buffer axis)
        auto shape = memDesc.getShape();
        SmallVector<int64_t> viewShape(shape.begin() + 1, shape.end());
        auto viewType = MemDescType::get(viewShape, memDesc.getElementType(),
            memDesc.getEncoding(), memDesc.getMemorySpace(), /*mutable=*/true);
        auto view = MemDescIndexOp::create(rewriter, loc, viewType, buffer, op.getStage());
        op.getBuffers()[i].replaceAllUsesWith(view);
    }
}
```

### 1.9 TMA load handling

**Key insight:** TMA hardware arrives on the mbarrier of the semaphore being
RELEASED, not acquired. In the cross-release pattern, @1 acquires sem0 and releases
sem1. TMA hardware arrives on sem1's mbar. So `BarrierExpectOp` goes on THIS
semaphore's mbar when THIS semaphore receives a release with `async_ops=[tma_load]`.

This is handled in `handleTMALoads(SemaphoreCreateOp op, ..., Value mbars)` which
is called from `matchAndRewrite` BEFORE any user is erased. It looks at releases
targeting THIS semaphore and finds the descriptor loads via the release's token chain:

```cpp
void handleTMALoads(SemaphoreCreateOp op, PatternRewriter &rewriter,
                    Value mbars) {
    // Look at releases targeting THIS semaphore with async_ops=[tma_load]
    for (auto user : op->getUsers()) {
        auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
        if (!releaseOp) continue;
        auto kinds = castAsyncOpAttrs(releaseOp.getAsyncOps());
        if (!llvm::any_of(kinds, [](AsyncOp k) { return k == AsyncOp::TMALoad; }))
            continue;

        // The release's token traces back to an acquire on a DIFFERENT semaphore.
        // That acquire's buffer ops have the descriptor loads.
        int txCount = 0;
        SmallVector<Operation *> loadOps;
        Value tok = releaseOp.getToken();
        for (auto tokUser : tok.getUsers()) {
            if (auto bufOp = dyn_cast<SemaphoreBufferOp>(tokUser)) {
                for (auto buf : bufOp.getBuffers()) {
                    for (auto viewUser : buf.getUsers()) {
                        if (auto descLoad = dyn_cast<DescriptorLoadOpInterface>(viewUser)) {
                            loadOps.push_back(descLoad);
                            txCount += descLoad.getTxCount();
                        }
                    }
                }
            }
        }
        if (loadOps.empty()) continue;

        // BarrierExpectOp on THIS semaphore's mbar (TMA arrives here)
        auto mbar = createSingleBufferView(rewriter, mbars, releaseOp.getStage());
        auto pred = arith::ConstantIntOp::create(rewriter, op.getLoc(), 1, 1);
        assignStageCluster(pred, getPartitionWsTagIds(releaseOp),
                           getStageCluster(releaseOp), rewriter);
        auto expectOp = BarrierExpectOp::create(rewriter, op.getLoc(), mbar, txCount, pred);
        assignStageCluster(expectOp, getPartitionWsTagIds(releaseOp),
                           getStageCluster(releaseOp), rewriter);

        // Rewrite descriptor loads to use THIS mbar
        for (auto loadOp : loadOps) {
            rewriter.setInsertionPoint(loadOp);
            if (auto descLoad = dyn_cast<DescriptorLoadOp>(loadOp)) {
                auto newOp = AsyncTMACopyGlobalToLocalOp::create(rewriter,
                    loadOp->getLoc(), descLoad.getDesc(), descLoad.getIndices(),
                    mbar, descLoad.getResult(), pred);
                assignStageCluster(newOp, getPartitionWsTagIds(loadOp),
                                   getStageCluster(loadOp), rewriter);
            } else if (auto descGather = dyn_cast<DescriptorGatherOp>(loadOp)) {
                auto newOp = AsyncTMAGatherOp::create(rewriter, loadOp->getLoc(),
                    descGather.getDesc(), descGather.getXOffsets(),
                    descGather.getYOffset(), mbar, descGather.getResult(), pred);
                assignStageCluster(newOp, getPartitionWsTagIds(loadOp),
                                   getStageCluster(loadOp), rewriter);
            }
            loadOp->erase();
        }
    }
}
```

## 2. Files to create/modify

| File | Change |
|------|--------|
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerSemaphore.cpp` | REPLACE Phase 1 stub with real implementation |
| `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt` | Already has entry from Phase 1 — no change |
| `third_party/nvidia/triton_nvidia.cc` | ADD: `ADD_PASS_WRAPPER_0("add_lower_semaphore", mlir::triton::createNVWSLowerSemaphore);` |

## 3. Lit test

Create `test/NVWS/lower_semaphore.mlir` with hand-written semaphore IR that
HAS stage/phase assigned.

Test cases:
1. **Basic:** Single acquire/release pair → WaitBarrierOp + ArriveBarrierOp
2. **Multi-stage (depth=2):** Two stages, verify mbar array has 2 entries
3. **TMA load:** `async_ops=[tma_load]` → BarrierExpectOp + AsyncTMACopyGlobalToLocal
4. **TMA gather:** `async_ops=[tma_load]` + descriptor_gather → AsyncTMAGatherOp
5. **TC5MMA:** `async_ops=[tc5mma]` → TCGen5CommitOp
6. **Fence:** SMEM buffer, one semaphore with `async_ops=[none]`, other with
   `async_ops=[tc5mma]` → FenceAsyncSharedOp before wait
7. **TMEM scales:** Buffer with TensorMemoryScalesEncodingAttr → no MemDescIndexOp
8. **Cleanup:** Verify InvalBarrierOp + LocalDeallocOp after all users

## 4. Verification

```bash
BUILD=build/cmake.linux-x86_64-cpython-3.12
TOPT=$BUILD/bin/triton-opt

ninja -C $BUILD triton-opt
$TOPT test/NVWS/lower_semaphore.mlir -split-input-file \
    --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck test/NVWS/lower_semaphore.mlir
# ALL existing tests still pass (LowerAref.cpp untouched)
```
