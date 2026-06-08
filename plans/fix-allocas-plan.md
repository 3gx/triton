# Plan - Fix NVWS single-value communication buffers

## Context

`NVWSInsertAllocas` currently runs with `InsertCommunicationOptions::createSemaphores = false`, but the local/shared-memory path still calls `getMultiBufferedType(memDescType, 1)`. That adds a leading dimension to ordinary no-semaphore local communication buffers, for example:

```mlir
ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, ..., mutable>
ttg.memdesc_index %alloc[%c0] : !ttg.memdesc<1x128x64xf16, ...> -> !ttg.memdesc<128x64xf16, ...>
```

This is mechanically correct enough for later passes because the `ttg.memdesc_index [0]` recovers the useful view, and `InsertSemas` can retarget managed accesses to `nvws.semaphore.buffer` views. It is still the wrong IR shape for `InsertAllocas`: no-semaphore communication should allocate only the smallest valid memory shape needed to communicate one value, subject to the memory-space representation requirements.

For SMEM/local memory, if `1x128x128` and `128x128` can both represent one communicated value, use `128x128`. For TMEM, rank-1 tensor values cannot be represented as `128xf32` because TMEM communicates 2D tiles, so use the smallest valid TMEM tile, `128x1xf32`. Do not introduce `1x128x1xf32` in `InsertAllocas`; that leading depth belongs only to semaphore backing created by `InsertSemas`.

## Goal

Make `--nvws-insert-allocas` allocate the smallest valid per-value communication buffer shape when semaphores are not being created. Then make `InsertSemas` create the staged local/shared-memory backing allocation used by `nvws.semaphore.create`, just as it already does for TMEM. The `1x` on that new semaphore backing allocation is real staging depth.

Direct local `1x... + ttg.memdesc_index` input IR is no longer an implementation constraint for this fix. The real runtime pipeline should not generate that shape once `InsertAllocas` is fixed. Lit tests that directly feed that shape into `--nvws-insert-semas` may fail and will be patched later after runtime gates pass.

## Hard constraints

1. Do not change TMEM rank expansion. `tensor<128xf32>` communicated through SSA TMEM still becomes a `128x1` TMEM tile because that is the smallest valid TMEM representation for one communicated value.
2. Do not preserve direct local `1x... + ttg.memdesc_index` input compatibility as part of this implementation. That shape is lit-only fallout for this pass, not a runtime gate requirement.
3. `InsertSemas` must derive the local semaphore member shape from the input local alloc/access shape it sees in the real pipeline, then add the real semaphore depth dimension only to the local alloc created as semaphore backing. For new `InsertAllocas` output `128x64`, the local backing shape is `1x128x64`.
4. The only local/shared allocs that may gain a staging dimension are the allocs created as semaphore backing buffers. The leading `1x` on those backing allocs is a real depth-one staging buffer; later passes may change that depth.
5. `InsertSemas` cleanup must delete stale original alloc/view chains at the end when they are dead after access retargeting. Newly created semaphore backing allocs are marked and kept.
6. Do not change `getMultiBufferedType`. The fix changes which callers use it and what member shapes they pass to it; the helper itself remains unchanged.
7. Keep the change mechanical: no semaphore scheduling rewrite, no ownership/sync-DAG behavior changes, no lit test rewrites during the first implementation pass.
8. The first line of quality testing is the runtime gates. Some lit tests are expected to fail until the implementation is reviewed and the lit goldens/direct-input tests are patched.

## Implementation plan

### 1. Allocate the smallest valid per-value shape

In `InsertAllocas.cpp`, make `createCommunicationBuffer` choose the allocation type based on both memory space and `createSemaphores`.

Current behavior:

```c++
if (isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace())) {
  allocBufType = options.createSemaphores
                     ? getSemaphoreMultiBufferedType(memDescType, 1)
                     : memDescType;
} else {
  allocBufType = getMultiBufferedType(memDescType, 1);
}
```

Required behavior:

- TMEM + semaphores: keep `getSemaphoreMultiBufferedType(memDescType, depth)`.
- TMEM + no semaphores: keep `memDescType`.
- SMEM/local + semaphores: keep `getMultiBufferedType(memDescType, depth)`.
- SMEM/local + no semaphores: use `memDescType`, not `getMultiBufferedType(..., 1)`.

For `InsertAllocas` no-semaphore output, `memDescType` must already be the smallest valid shape for one communicated value. For SMEM/local this means no leading depth dimension when the underlying tensor shape itself is valid. For rank-1 TMEM this means `Mx1`, not `1xMx1`.

This plan does not change semaphore depth policy. Paths that already create semaphores keep their existing depth choice.

### 2. Make no-semaphore buffer views identity-compatible

Update `createStage0BufferView` so it does not assume every local allocation has a leading depth dimension.

Required behavior:

- If the allocation type already has the exact requested view type, return the allocation directly.
- If the allocation is a real staged backing type created for semaphores, keep the existing `ttg.memdesc_index` path.
- Preserve mutable/non-mutable result typing. If a no-semaphore consumer needs an immutable view of a mutable alloc, use an existing memdesc view operation such as `ttg.memdesc_reinterpret` only if needed and only as a type view, not as a shape-changing staging depth.

This is the main place where the old leading-dimension assumption leaks into access materialization.

### 3. Make `InsertSemas` create local semaphore backing

Do not reuse input local/shared-memory allocs as semaphore backing. `InsertSemas` input IR has ordinary local buffers and access views; it does not have semaphore backing buffers. The pass must insert separate staged backing allocs for local/shared memory, matching the existing TMEM model.

Required behavior:

- TMEM backing: keep the existing separate `ttng.tmem_alloc` path using `getSemaphoreMultiBufferedType(member.type, depth)`.
- Local/shared backing: always create a `ttg.local_alloc` for `nvws.semaphore.create`; this is the only local alloc class that gets a staging dimension.
- The local/shared backing shape must be the semaphore stage dimension followed by the input member shape. For new `InsertAllocas` output `128x64`, the backing is `1x128x64`. The leading dimension added by `InsertSemas` is real semaphore staging depth.
- The local/shared backing type must use the existing `getMultiBufferedType(member.type, depth)` path for real-pipeline local member shapes. Do not modify `getMultiBufferedType`, and do not add extra machinery just to support direct lit-only `1x... + ttg.memdesc_index` inputs.
- Mark only the new backing alloc with `nvws.semaphore.backing`. Do not mark original input allocs as backing.
- `nvws.semaphore.buffer` should produce the member view type required by real-pipeline accesses after `InsertAllocas` stops emitting the unnecessary local leading dimension.

Keep the existing alias handling needed by real runtime inputs:

- `ttg.memdesc_index` must remain a supported alias op.
- `materializeAliasForBuffer` must continue skipping redundant `ttg.memdesc_index` clones when the `nvws.semaphore.buffer` result already has the needed access type.
- Cleanup at the end must erase dead original alloc/view chains after retargeting. It must not erase the newly created semaphore backing allocs.

The fix is not "delete memdesc_index support"; the fix is "stop `InsertAllocas` from generating local `memdesc_index [0]` solely to undo an allocation shape that was larger than needed for one communicated value." Direct local `1x... + ttg.memdesc_index` tests can be patched after runtime gates pass.

### 4. Patch lit tests only after runtime gates

Expected lit fallout:

- `test/NVWS/insert_allocas.mlir` currently expects `ttg.local_alloc : () -> !ttg.memdesc<1x...>` in many SMEM checks.
- Those checks should eventually move to `!ttg.memdesc<...>` without the leading depth for `--nvws-insert-allocas`.
- Existing `insert_semas` tests that directly provide `1x... + ttg.memdesc_index` input are expected lit fallout. Real runtime gates do not depend on that direct input shape.

Do not update lit goldens before the runtime gates pass and review approves the code direction.

## Runtime verification gates

Run from the repository root unless noted. Set:

```bash
export PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/
```

Build first:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Then run the required runtime gates:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git
PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/ \
  timeout 60s pytest -v -n16 python/test/unit/language/test_warp_specialization.py

PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/ \
  timeout 60s sh run_nvws.sh

PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/ \
  timeout 60s sh run_nvws_1.sh
```

### Runtime hang protocol

Each runtime gate has a hard 60s timeout. If any runtime gate does not finish within 60s, treat that as a hang introduced by the change.

On a timeout:

1. Do not rerun the timed-out runtime gate.
2. Do not try nearby runtime variants to "confirm" the hang.
3. Investigate from the timed-out command's output, the local code diff, and static IR/code inspection.
4. If the cause is found, make the smallest targeted fix and then restart verification from build first.
5. If the cause cannot be found without rerunning or guessing, stop and report the exact timed-out command, the available evidence, and the unresolved question to the user.

Do not overreach, theorize, or assume a timeout is environmental. For this plan, a 60s runtime timeout means the change created a hang until proven otherwise by concrete evidence.

## Review checkpoint

After the runtime gates pass:

1. Report the exact code diff and summarize the intentional `InsertSemas` behavior change for local semaphore backing.
2. Show before/after IR for one representative local allocation from `insert_allocas.mlir`.
3. Show representative IR after `InsertAllocas` followed by `InsertSemas`: local single-value alloc shape first, then staged local semaphore backing in `InsertSemas`.
4. Stop for review before editing lit goldens.

## Stop conditions

- The fix requires changing ownership, release/acquire placement, or sync-DAG logic.
- TMEM rank-1 SSA communication stops using `Mx1` tiles.
- Runtime IR generated by the real pipeline still contains direct local `1x... + ttg.memdesc_index` input to `InsertSemas`.
- A runtime gate fails and the cause cannot be traced to a concrete in-scope implementation bug.
- Any runtime gate times out and the hang cause cannot be identified without rerunning or guessing.
