# InsertTmemSemaphore: TMEM Ownership Transfer Synchronization

This document describes the `NVWSInsertTmemSemaphore` pass, which inserts
semaphore signaling for TMEM ownership transfers between
warp-specialized partitions. The audience is someone familiar with arefs and the
partition model but new to the semaphore-based synchronization that replaces
them.

**Source file:**
`third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp`
(~815 lines)

**Lit tests:**
`test/NVWS/insert_tmem_semaphore.mlir` (22 CHECK-LABELs)

---

## 1. Purpose

TMEM is a special on-chip memory on Blackwell (sm_100) GPUs,
accessible by both the MMA tensor core pipeline (via `tc_gen5_mma`) and the
thread execution pipeline (via `tmem_store` / `tmem_load`). Unlike shared
memory (SMEM), where the producer always writes and the consumer always reads,
TMEM buffers support an ownership transfer model: any partition can read, update or
write the buffer, but only one partition may own TMEM it at a time if there are stores/updates.
Ownership passes between partitions, and the current owner can perform any operation.

This pass analyzes each `TMEMAllocOp` to determine which partitions access the
TMEM buffer and in what order, then inserts `nvws.semaphore.create`,
`nvws.semaphore.acquire`, `nvws.semaphore.release`, and
`nvws.semaphore.buffer` operations at every ownership transition point.

---

## 2. TMEM vs. SMEM: Why Ownership Transfer?

For **SMEM** buffers, roles are usually fixed:
- Partition A is always the producer (e.g., TMA load into shared memory).
- Partition B is always the consumer (e.g., read the loaded data).
- A single semaphore pair suffices: "empty" guards the write side, "full" guards
  the read side.

For **TMEM** buffers, roles are not fixed. Consider a matmul accumulator:

1. **Partition 1** (MMA partition): writes via `tc_gen5_mma` (tensor core write).
2. **Partition 0** (thread partition): reads via `tmem_load`, processes results,
   then may write a new accumulator via `tmem_store`.
3. **Partition 1**: updates the new accumulator via `tc_gen5_mma` again.

Both partitions can read and write. The synchronization must track who
currently owns the buffer and enforce handoffs via acquire/release pairs.

---

## 3. TmemAccessDag: The Token-Chain DAG

The core data structure is `TmemAccessDag`, a linked-list DAG that represents
the chain of operations accessing a single TMEM allocation.

### 3.1 Construction

Built by `TmemAccessDag::build(TMEMAllocOp)`:

```
TmemAccessDag::build(allocOp):
    root = Node(allocOp, partitionId=allocOp's partition or nullopt)
    tok = allocOp.getToken()         // SSA token output
    follow tok's single use -> addOp(use, root)
    return dag rooted at root
```

The DAG follows the token SSA chain: each TMEM operation consumes a token
and produces a new one. The pass follows uses of each token to find the next
operation.

### 3.2 Node Structure

```cpp
struct Node {
    Operation *op;               // The MLIR operation
    OpOperand *tokOperand;       // Which operand consumes the token
    optional<PartitionId> partitionId;  // (partition_index, ws_tag)
    Node *parent;                // Previous node in the chain
    Node *parentDag;             // Parent in the DAG hierarchy (for sub-DAGs)
    optional<int> tokPos;        // Token position in scf.for/scf.if results
    unique_ptr<Node> user;       // Next node in the chain
    SmallVector<unique_ptr<Node>> subDags;  // For loop bodies / if branches
};
```

### 3.3 Supported Node Types

| Node type | Handling |
|-----------|----------|
| `TMEMLoadOp` | Leaf op; produces a new token via `getToken()` |
| `TMEMStoreOp` | Leaf op; produces a new token via `getToken()` |
| `MMAv5OpInterface` | Leaf op; covers `tc_gen5_mma` and `tc_gen5_mma_scaled` |
| `scf.ForOp` | Creates a sub-DAG for the loop body; token enters via `init_arg`, exits via `yield` |
| `scf.IfOp` | Creates two sub-DAGs (then-branch, else-branch); detected when a token has exactly 2 uses |

### 3.4 Token Threading Through Loops

When the DAG encounters an `scf.ForOp`:

```
addForOp(tokOperand, forOpNode):
    forOp = cast<ForOp>(tokOperand.getOwner())
    tokPos = tokOperand.getOperandNumber() - 3   // skip lb, ub, step
    tokArg = forOp.getRegionIterArg(tokPos)       // block arg inside body
    subDag = new Node(placeholder)
    addOp(tokArg's single use, subDag)            // build body sub-DAG
    finalNode = walk to end of subDag
    finalNode.user = Node(yieldOp, yieldOp.operand[tokPos])
    forOpNode.subDags.push_back(subDag)
    return forOp.getResult(tokPos)                // token continues after loop
```

The token enters the loop body as an iter_arg at position `tokPos`, is threaded
through all TMEM operations in the body, and exits via the `scf.yield` operand
at the same position.

### 3.5 Token Threading Through Conditionals

When a token has exactly two uses (one in each branch of an `scf.if`):

```
addIfOp(tok, node):
    uses = tok.getUses()    // exactly 2
    sort so useThen is in thenBlock, useElse is in elseBlock
    thenDag = addOp(useThen, ...)
    elseDag = addOp(useElse, ...)
    tokPos = findValuePos(thenYield.operands, thenTok)
    ifOpNode.subDags = [thenDag, elseDag]
    return ifOp.getResult(tokPos)
```

---

## 4. TMEMSemaphore State Machine

The `TMEMSemaphore` struct tracks the two-semaphore pair and the current
ownership state as the pass walks the DAG.

### 4.1 Semaphore Pair

```cpp
struct TMEMSemaphore {
    enum Kind { PING, PONG };

    Value empty;        // semaphore with is_released=true  (producer starts)
    Value full;         // semaphore with is_released=false (consumer waits)
    Value allocBuf;     // underlying TMEM buffer allocation
    Value origBuffer;   // original TMEMAllocOp result (for rewriting)
    Value replToken;    // poison token replacing original dep tokens

    Value buffer;       // current SemaphoreBufferOp result (lazily created)
    Value token;        // current acquire token
    Kind kind;          // PING or PONG -- toggles on release
    ...
};
```

### 4.2 State Machine Transitions

The naming uses PING/PONG instead of follow exising PUT/GET `InsertTmemAref.cpp` model:

| Current Kind | acquire() waits on | release() signals | After release, kind becomes |
|-------------|-------------------|------------------|---------------------------|
| PING | `empty` semaphore | `full` semaphore | PONG |
| PONG | `full` semaphore | `empty` semaphore | PING |

This is the cross-release pattern: when you are the PING side, you acquire
the "empty" semaphore (the buffer is free to write) and release the "full"
semaphore (signaling data is ready). This toggles you to PONG for the next
transition.

This model can be ultimately be extended to N-party ownership.

### 4.3 Initial State

The state machine starts at `kind = PING`:
1. Acquire `empty` (buffer is free).
2. Producer writes (e.g., `tmem_store` for initial accumulator zero-fill).
3. On ownership transition, release `full` (data ready), kind becomes `PONG`.
4. Consumer acquires `full`, reads, releases `empty`, kind becomes `PING`.
5. And so on.

### 4.4 Buffer Access

`getBuffer()` lazily creates a `SemaphoreBufferOp` to obtain the actual TMEM
buffer view from the semaphore:

```cpp
Value getBuffer(OpBuilder &b, optional<PartitionId> pid, Operation *op) {
    if (!buffer) {
        Value sem = (kind == PING) ? empty : full;
        buffer = SemaphoreBufferOp::create(b, loc, {pid, stageCluster},
                                           sem, {dataBufType}, token);
    }
    return buffer;
}
```

The buffer view is invalidated (set to `{}`) whenever the token changes (after
acquire, after loop exit), forcing a new `SemaphoreBufferOp` on next access.

### 4.5 AsyncOp Tracking

Each release must declare what asynchronous operation produced the data, so
downstream lowering can emit the correct fence/wait:

| TMEM operation | AsyncOp value |
|----------------|---------------|
| `MMAv5OpInterface` (`tc_gen5_mma`, `tc_gen5_mma_scaled`) | `AsyncOp::TC5MMA` |
| `TMEMLoadOp`, `TMEMStoreOp` | `AsyncOp::NONE` |

This is set in `insertTmemSemaphoreImpl`:

```cpp
if (isa<MMAv5OpInterface>(node->op))
    state.asyncOp[node->partitionId] = AsyncOp::TC5MMA;
else if (isa<TMEMLoadOp, TMEMStoreOp>(node->op))
    state.asyncOp[node->partitionId] = AsyncOp::NONE;
```

---

## 5. Ownership Transition Detection: `insertTmemSemaphoreImpl`

This is the core recursive function that walks the DAG and inserts semaphore
operations at every ownership transition.

### 5.1 Algorithm

```
insertTmemSemaphoreImpl(node, curPartitionId, state):
    // 1. Detect ownership transition
    if curPartitionId is set AND node.partitionId != curPartitionId:
        // Insert release AFTER the previous node
        state.release(b, prevOp.loc)          // old owner relinquishes

        // Insert acquire BEFORE the current node
        state.acquire(b, curOp.loc, {node.partitionId, stageCluster})

    // 2. Recurse into sub-DAGs (loop bodies, if branches)
    for each subDag in node.subDags:
        subdagState = state (copy)
        if node is ForOp:
            subdagState.token = forOp.getRegionIterArg(tokPos)
            subdagState.buffer = {}            // invalidate
        insertTmemSemaphoreImpl(subDag, node.partitionId, subdagState)
        state.asyncOp = subdagState.asyncOp    // propagate back
        state.partitionId = subdagState.partitionId

    // 3. Track async op for this node
    if node is MMAv5OpInterface:   state.asyncOp = TC5MMA
    if node is TMEMLoad/Store:     state.asyncOp = NONE

    // 4. Rewrite the TMEM operation to use semaphore buffer
    if TMEMLoadOp:
        tmemLoadOp.src = state.getBuffer(...)     // replace raw alloc
        tmemLoadOp.dep.clear()                    // remove old dep token
        tmemLoadOp.token.replaceAllUsesWith(replToken)  // poison old token

    if TMEMStoreOp:
        tmemStoreOp.dst = state.getBuffer(...)
        tmemStoreOp.dep.clear()
        tmemStoreOp.token.replaceAllUsesWith(replToken)

    if MMAv5OpInterface:
        mmaOp.accDep.clear()
        mmaOp.token.replaceAllUsesWith(replToken)
        for each operand == origBuffer:
            operand.set(state.getBuffer(...))

    if scf.YieldOp:
        yieldOp.setOperand(tokPos, state.token)   // carry semaphore token

    if scf.ForOp / scf.IfOp:
        state.token = op.getResult(tokPos)
        state.buffer = {}                         // invalidate after loop/if

    // 5. Continue to next node in chain
    if node.user:
        return insertTmemSemaphoreImpl(node.user, node.partitionId, state)
```

### 5.2 Rewriting Operations

The key rewrite for each TMEM operation is:
1. **Replace the buffer operand**: The original `%result` from `TMEMAllocOp` is
   replaced with the buffer obtained from `SemaphoreBufferOp`.
2. **Remove the dependency token**: The `[%token]` dependency on the original
   alloc token is cleared, since synchronization is now handled by the
   semaphore.
3. **Poison the output token**: The output token of each TMEM op is replaced
   with a poison value (`ub.poison`), since the semaphore token is now the
   synchronization mechanism.

---

## 6. Top-Level Driver: `insertTmemSemaphore`

The function `insertTmemSemaphore(TmemAccessDag&, int numTmemBlocks)` is called
once per TMEM allocation that has multiple owners.

### 6.1 Double-Buffering Decision

```
isMultiStaged = hasProducerConsumerPartitioning(accessDag)
if isMultiStaged:
    for each MMA user of allocOp:
        accIsMultiBuffered =
            !hasAccReadModifyWrite(mmaOp, loop) &&
            isAccMultibufferingPossible(mmaOp, loop) &&
            !getDisallowAccMultiBuffer(wsLoop) &&
            canDoubleBufferAcc(mmaOp, numTmemBlocks)
        isMultiStaged = isMultiStaged && accIsMultiBuffered

numStages = 1 + isMultiStaged    // 1 = single-buffered, 2 = double-buffered
```

The TMEM allocation shape is extended at insertion time (not deferred):
- Single-buffered: `<128x128xf32>` becomes `<1x128x128xf32>`
- Double-buffered: `<128x128xf32>` becomes `<2x128x128xf32>`

### 6.2 `canDoubleBufferAcc`

```cpp
bool canDoubleBufferAcc(MMAv5OpInterface mmaOp, int numTmemBlocks) {
    blockM = accum.shape[0];  blockN = accum.shape[1];
    // Check TMEM capacity: 128 rows x 512 columns
    if (numTmemBlocks + blockM * blockN * 2 > 128 * 512)
        return false;
    // Scaled MMA with blockN=256 cannot double-buffer
    if (isa<TCGen5MMAScaledOp>(mmaOp) && blockN == 256)
        return false;
    return true;
}
```

### 6.3 Semaphore Creation and Initial Acquire

```
// Hoist alloc before the outermost warp-specialize loop if present
semAlloc = TMEMAllocOp(semBufType)    // e.g., <2x128x128xf32>

// Create semaphore pair
emptySem = SemaphoreCreateOp(semAlloc, is_released=true)
fullSem  = SemaphoreCreateOp(semAlloc, is_released=false)

// Initial acquire on empty side (buffer is free to write)
state = TMEMSemaphore(emptySem, fullSem, ...)
state.acquire(empty)

// If allocOp has initial data (tmem_alloc %src), store it
if allocOp.getSrc():
    buffer = state.getBuffer(...)
    TMEMStoreOp(buffer, src)
```

### 6.4 Post-Loop Cleanup

After `insertTmemSemaphoreImpl` returns, a final release is inserted after the
last node. If the state ends in `PONG` mode (meaning the last release was on the
"full" side, toggling to PONG), an additional acquire/release pair is inserted
for the other partition to balance the protocol:

```
state.release(b, lastNode.loc)     // final release

if state.kind == PONG:
    // Other partition needs a balancing acquire/release
    state.acquire(b, lastNode.loc, {otherPartitionId, {}})
    state.release(b, lastNode.loc)
```

This ensures both semaphores end in a consistent state.

---

## 7. The 2-Partition Limitation

The driver enforces:

```cpp
auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
assert(partitions.size() <= 2 && "expecting at most 2 partitions");
```

This matches the limitation of the old `InsertTmemAref` pass. The semaphore
abstraction itself supports N partitions (a semaphore is just an mbarrier-based
signal), but the current insertion logic assumes a two-partition ping-pong
pattern. The guard is a 1:1 compatibility constraint enabling future extension.

The `totalOwners` check (`hasRootPartition + partitions.size() > 1`) determines
whether any synchronization is needed at all. If a TMEM buffer is only accessed
by one partition, no semaphore is inserted.

---

## 8. `hasProducerConsumerPartitioning`

Validates that the access pattern strictly alternates between "producer" and
"consumer" roles with exactly 2 group changes per loop iteration:

```cpp
bool hasProducerConsumerPartitioning(TmemAccessDag &accessDag) {
    auto [hasRootPartition, partitions] = accessDag.collectPartitionsVec();
    bool expectProducer = true;
    int changeGroup = 0;

    for (i = 0; i < partitions.size() - 1; ++i) {
        auto op = partitions[i].second;
        if (isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op)) {
            valid = valid && (expectProducer
                ? isa<TMEMStoreOp, MMAv5OpInterface>(op)   // producer ops
                : isa<TMEMLoadOp>(op));                     // consumer ops
        }
        if (partitions[i].first != partitions[i+1].first) {
            expectProducer = !expectProducer;
            ++changeGroup;
        }
    }
    return valid && changeGroup == 2;
}
```

This checks:
- Producer operations (`TMEMStoreOp`, `MMAv5OpInterface`) appear when a
  producer is expected.
- Consumer operations (`TMEMLoadOp`) appear when a consumer is expected.
- Exactly 2 partition transitions occur per iteration (producer->consumer,
  consumer->producer), which is the requirement for double-buffering the
  accumulator.

---

## 9. `workaroundForLoopScheduler`

A post-processing fixup that restructures `scf.if` blocks to enable the
downstream loop scheduler to independently schedule release/acquire operations.

### 9.1 Pattern Detected

In the then-block of an `scf.if`:

```mlir
scf.if %cond -> (!ttg.async.token) {
    nvws.semaphore.release %full, %tok [tc5mma]    // first op
    ...body ops (acquire full, buffer, load, release empty, use)...
    %new_tok = nvws.semaphore.acquire %empty        // last before yield
    scf.yield %new_tok
} else {
    scf.yield %tok
}
```

### 9.2 Transformation

The pass splits this into three `scf.if` operations:

```mlir
// 1. EXIT if: release only (consumer partition annotation)
scf.if %cond {
    nvws.semaphore.release %full, %tok [tc5mma]
} {ttg.partition = array<i32: 1>}

// 2. BODY if: the remaining MMA/load body (producer partition annotation)
%tok_or_poison = scf.if %cond -> (...) {
    ...body ops...
    scf.yield %poison_token    // token invalidated
} else {
    scf.yield %poison_token
} {ttg.partition = array<i32: 0>}

// 3. ENTER if: acquire only (consumer partition annotation)
%new_tok = scf.if %cond -> (!ttg.async.token) {
    %t = nvws.semaphore.acquire %empty
    scf.yield %t
} else {
    scf.yield %else_tok
} {ttg.partition = array<i32: 1>}
```

### 9.3 Partition Annotations

```cpp
SetVector<int> enterExitIds, middleIds;
enterExitIds.insert(1);    // consumer partition
middleIds.insert(0);       // producer partition
setPartition(enterIf, enterExitIds);
setPartition(exitIf, enterExitIds);
setPartition(ifOp, middleIds);
```

The enter/exit `scf.if` operations get the consumer partition ID (1), while the
body `scf.if` gets the producer partition ID (0). This allows the loop scheduler
to move the release earlier and the acquire later relative to the MMA/load body.

### 9.4 Stage/Cluster Propagation

```cpp
enterIf->setAttrs(ifOp->getAttrs());
exitIf->setAttrs(ifOp->getAttrs());
assignStage(b, enterIf, getStageCluster(acquireOp));
assignStage(b, exitIf, getStageCluster(releaseOp));
```

---

## 10. Token Threading Through Loops (Detail)

When the DAG walker enters a `scf.for` body:

1. **Entry**: The semaphore token enters the `scf.for` as an `init_arg`. Inside
   the body, `forOp.getRegionIterArg(tokPos)` provides the block argument that
   carries the token.

2. **Body**: The token is consumed and produced by successive TMEM operations.
   At each ownership transition, `insertTmemSemaphoreImpl` inserts
   release/acquire pairs.

3. **Exit**: The `scf.yield` operand at `tokPos` is updated to carry the
   current semaphore token (`yieldOp.setOperand(tokPos, state.token)`).

4. **After loop**: The for-op result at `tokPos` becomes the new token.
   The buffer view is invalidated (`state.buffer = {}`) because the buffer
   pointer from inside the loop is no longer valid -- a new
   `SemaphoreBufferOp` must be created after re-acquiring.

For the init_arg threading in the for-op node itself:

```cpp
if (isa<scf::ForOp>(node->op)) {
    node->op->setOperand(tokOperand->getOperandNumber(), state.token);
    state.token = node->op->getResult(*node->tokPos);
    state.buffer = {};
}
```

---

## 11. Comparison with `InsertTmemAref`

| Aspect | InsertTmemAref | InsertTmemSemaphore |
|--------|---------------------|--------------------------|
| **DAG structure** | Same `TmemAccessDag` | Same `TmemAccessDag` |
| **2-partition limit** | `partitions.size() <= 2` | `partitions.size() <= 2` |
| **Sync mechanism** | Binary aref (put/get) | Semaphore pair (empty/full) with cross-release |

---

## 12. Lit Test Coverage

The file `test/NVWS/insert_tmem_semaphore.mlir` contains 22 CHECK-LABELs:

| Test | Scenario | Buffer depth |
|------|----------|-------------|
| `@warp_specialize_tma_matmul` | Basic matmul: alloc + store + loop(MMA) + load | 1x (single) |
| `@matmul_tma_acc_with_unconditional_user` | MMA + unconditional load + store in loop | 2x (double) |
| `@matmul_tma_acc_with_conditional_user` | MMA + conditional load/use in `scf.if` | 2x (double) |
| `@matmul_tma_acc_with_conditional_def` | MMA + unconditional load + conditional store | 2x (double) |
| `@matmul_tma_acc_with_conditional_def_and_use` | MMA + conditional load + conditional store | 2x (double) |
| `@matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag` | Same but `tt.disallow_acc_multi_buffer` set on loop | 1x (single) |
| `@matmul_scaled_rhs_scales_tma` | Scaled MMA with separate RHS scales buffer (dual semaphore) | 1x ACC, scales |
| `@user_partition_has_cycle` | MMA -> load -> yield with cyclic partition pattern | 2x (double) |
| `@matmul_tma_acc_with_conditional_def_and_use_flag` | `use_d` flag variant of conditional def+use | 2x (double) |
| `@specialize_mma_only` | MMA-only specialization (no load in loop) | 1x (single) |
| `@load_scale_mma_user` | Scaled MMA with both LHS/RHS scales as separate TMEM | 1x ACC, scales |
| `@store_mma_load` | `tmem_store` -> MMA -> `tmem_load` (reversed roles) | 1x (single) |
| `@local_alloc_into_mma` | `local_alloc` feeding directly into MMA | 1x (single) |
| `@shmem_sink_iterator_invalidation` | Two TMEM allocs (ACC + LHS operand); dual semaphore pairs | 1x both |
| `@attention_forward` | 3 TMEM buffers (S, O, P), 4 partitions, complex dataflow | S=2x, O=1x, P=1x |
| `@hoisted_alloc` | Alloc with `%src` hoisted outside nested loops | 1x (single) |
| `@if_split_workaround` | Tests `workaroundForLoopScheduler` if-splitting | 1x (single) |
| `@nested_loop_yes_double_buffer` | Nested loop, inner store in producer partition | 2x (double) |
| `@nested_loop_no_double_buffer` | Nested loop, inner store in consumer partition | 1x (single) |
| `@nested_loop_yes_double_buffer_scaled` | Nested loop, scaled MMA, small enough for double | 2x (double) |
| `@nested_loop_no_double_buffer_scaled` | Nested loop, scaled MMA, 128x256 too large | 1x (single) |
| `@test_tmem_no_ws` | No warp specialization -- pass is a no-op, no semaphores | N/A (no-op) |

---

## 13. Worked Example: Basic Matmul (`@warp_specialize_tma_matmul`)

**Input** (simplified):

```mlir
%result, %token = ttng.tmem_alloc : () -> (memdesc<128x128xf32>, async.token)
%tok0 = ttng.tmem_store %zeros, %result[%token], %true   // P1: init acc
%tok1 = scf.for ... iter_args(%tok = %tok0) {
    %tok2 = ttng.tc_gen5_mma %A, %B, %result[%tok]       // P1: MMA
    scf.yield %tok2
}
ttng.tmem_load %result[%tok1]                             // P0: read result
```

**Output** (simplified):

```mlir
// Alloc with depth=1 (single-buffered, only 1 partition in loop body)
%abuf = ttng.tmem_alloc : () -> memdesc<1x128x128xf32>
%empty = nvws.semaphore.create %abuf true     // initially released
%full  = nvws.semaphore.create %abuf false    // initially blocked

// Initial acquire + store (state: kind=PING, sem=empty)
%atok = nvws.semaphore.acquire %empty
%buf  = nvws.semaphore.buffer %empty, %atok
ttng.tmem_store %zeros, %buf

// Loop (no ownership change inside -- MMA stays in P1)
%tok_out = scf.for ... iter_args(%tok = %atok) {
    %buf2 = nvws.semaphore.buffer %empty, %tok   // same sem (still PING)
    %tok2 = ttng.tc_gen5_mma %A, %B, %buf2[]
    scf.yield %tok2
}

// After loop: ownership transition P1 -> P0
nvws.semaphore.release %full, %tok_out [tc5mma]   // release FULL (cross)
// kind toggles: PING -> PONG

%ctok = nvws.semaphore.acquire %full               // P0 acquires FULL
%buf3 = nvws.semaphore.buffer %full, %ctok
ttng.tmem_load %buf3[]
nvws.semaphore.release %empty, %ctok [none]        // release EMPTY (cross)
// kind toggles: PONG -> PING
```

---

## 14. Pass Entry Point

```cpp
class NVWSInsertTmemSemaphore
    : public NVWSInsertTmemSemaphoreBase<NVWSInsertTmemSemaphore> {
public:
    void runOnOperation() override {
        getOperation().walk([&](triton::FuncOp funcOp) {
            if (failed(runOnFunction(funcOp)))
                return WalkResult::interrupt();
            return WalkResult::advance();
        });
    }
};
```

`runOnFunction`:
1. Check if any `scf.ForOp` has `kWarpSpecializeAttrName`. If not, return
   success (no-op).
2. Walk all `TMEMAllocOp` and build a `TmemAccessDag` for each.
3. For each DAG with multiple owners (`totalOwners > 1`), call
   `insertTmemSemaphore`.
4. Run `workaroundForLoopScheduler` as a post-pass fixup.

```cpp
LogicalResult runOnFunction(triton::FuncOp funcOp) {
    // Early exit if no warp-specialize loops
    if (!walkResult.wasInterrupted())
        return success();

    SmallVector<TmemAccessDag> tmemDags;
    funcOp.walk([&](TMEMAllocOp allocOp) {
        tmemDags.push_back(TmemAccessDag::build(allocOp));
    });

    int numTmemBlocks = 0;
    for (auto &accessDag : tmemDags) {
        auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
        assert(partitions.size() <= 2);
        auto totalOwners = hasRootPartition + partitions.size();
        if (totalOwners > 1)
            numTmemBlocks = insertTmemSemaphore(accessDag, numTmemBlocks);
    }

    workaroundForLoopScheduler(funcOp);
    return success();
}
```

Note: `numTmemBlocks` is accumulated across allocations. It is passed to
`canDoubleBufferAcc` to track total TMEM consumption and prevent
over-allocation.
