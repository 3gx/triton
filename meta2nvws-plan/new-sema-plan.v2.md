# New Semaphore Pipeline v2

This plan stages the replacement of `NVWSInsertTmemSemaphore` and eventually
`NVWSInsertSemaphore` with an alloc/access-driven `NVWSInsertSemas` pass.

The first milestone is intentionally narrow: implement `insert-semas` for TMEM
only, with no token-based ordering analysis, and prove it can reproduce the IR
that today's `insert-tmem-semaphore` produces on the existing complex kernels.

## Non-negotiable rules

1. `insert-semas` operates on allocated memory objects and their memory
   accesses. It must not infer the semaphore schedule from async token use-def
   chains.

2. Stage 1 supports TMEM only. Local/SMEM support comes later, after the TMEM
   replacement is proven.

3. The initial TMEM-only `insert-semas` must produce the same semantic IR as
   `insert-tmem-semaphore` for the same input. Exact textual equality is a good
   debugging target, but the acceptance condition should be structural IR
   equivalence after deterministic cleanup such as `-cse`, because SSA names
   and incidental op ordering can differ.

4. Do not change existing lit tests to make this pass.

5. If an existing lit test appears to require a change, stop and provide the
   root cause. The user must authorize the test change before it is made.

6. The known current lit baseline has these failures:

   - `TRITON :: Conversion/tritongpu_to_llvm_blackwell.mlir`
   - `TRITON :: NVWS/MetaAutoWS/blackwell_ws_data_partition.mlir`
   - `TRITON :: TLX/tlx-verifier.mlir`

   The new work must not introduce additional failures.

7. Per `AGENTS.md`: build first, then run lit tests. Do not run pytest unless
   the user explicitly asks for pytest.

## Reference files and pipeline points

Current TMEM pass:

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp`
- pass option: `--nvws-insert-tmem-semaphore`
- pass definition: `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td`

New pass:

- proposed implementation file:
  `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp`
- proposed pass option: `--nvws-insert-semas`
- proposed test file:
  `test/NVWS/insert_semas.mlir`

Current pipeline replacement point:

- `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp`
- replace `createNVWSInsertTmemSemaphore()` only after the standalone
  `insert-semas` lit coverage passes.

Important dumps:

- isolated `insert-tmem-semaphore` input:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/064-NVWSInsertTmemSemaphore.mlir`
- effective current output after `insert-tmem-semaphore`:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/065-anonymous-VerifyWarpSpecializationPartitions.mlir`
- full subpipeline input before `insert-semaphore`:
  `meta-aws-logs/run-12may26-nvws-tmem/passes/062-NVWSInsertSemaphore.mlir`

Use `064 -> 065` to validate the isolated TMEM-only pass. Use `062` to validate
the combined subpipeline by comparing:

```
062 + --nvws-insert-semaphore + --nvws-insert-tmem-semaphore
```

against:

```
062 + --nvws-insert-semaphore + --nvws-insert-semas
```

## Stage 0: establish the guardrail baseline

Before editing behavior:

1. Build:

   ```
   cd build/cmake.linux-x86_64-cpython-3.12/
   ninja triton triton-opt
   ```

2. Run targeted lit around the current pass:

   ```
   /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v \
     test/NVWS/insert_tmem_semaphore.mlir \
     test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir \
     test/TritonGPU/automatic-warp-specialization.mlir
   ```

3. If running the larger suite, the only tolerated failures are the three known
   baseline failures listed above. Any other failure is a regression.

Deliverable:

- A recorded baseline result before `insert-semas` behavior diverges from the
  existing implementation.

## Stage 1: scaffold `insert-semas` as a TMEM-only clone

Create a new pass that initially behaves exactly like
`NVWSInsertTmemSemaphore`.

Implementation steps:

1. Add `InsertSemas.cpp` by copying the current TMEM implementation.

2. Add a new pass definition in `Passes.td`, for example:

   ```
   def NVWSInsertSemas: Pass<"nvws-insert-semas", "mlir::ModuleOp">
   ```

3. Add the pass registration and CMake wiring in the same style as
   `NVWSInsertTmemSemaphore`.

4. Do not modify `AutomaticWarpSpecialization.cpp` yet.

5. Copy:

   ```
   test/NVWS/insert_tmem_semaphore.mlir
   ```

   to:

   ```
   test/NVWS/insert_semas.mlir
   ```

   Change only the RUN line from `--nvws-insert-tmem-semaphore` to
   `--nvws-insert-semas`.

Acceptance:

- `test/NVWS/insert_tmem_semaphore.mlir` still passes unchanged.
- `test/NVWS/insert_semas.mlir` passes.
- No existing lit test is modified.

This stage proves the new pass is wired correctly before changing the analysis.

## Stage 2: add explicit equivalence checks for the complex kernel

Use the saved fused-attention dumps as a non-trivial equivalence target.

Isolated TMEM-only check:

1. Run current pass on:

   ```
   meta-aws-logs/run-12may26-nvws-tmem/passes/064-NVWSInsertTmemSemaphore.mlir
   ```

2. Run new pass on the same file.

3. Compare against the current output shape represented by:

   ```
   meta-aws-logs/run-12may26-nvws-tmem/passes/065-anonymous-VerifyWarpSpecializationPartitions.mlir
   ```

4. Normalize through the same cleanup pipeline, for example `-cse`, before
   comparing if exact SSA names differ.

Full subpipeline check:

1. Run current flow:

   ```
   062 + --nvws-insert-semaphore + --nvws-insert-tmem-semaphore
   ```

2. Run new flow:

   ```
   062 + --nvws-insert-semaphore + --nvws-insert-semas
   ```

3. The result must be structurally equivalent.

Acceptance:

- The new pass reproduces TMEM semaphore insertion for the complex kernel.
- Existing SMEM and SSA-TMEM behavior from `NVWSInsertSemaphore` is unchanged.

## Stage 3: remove token reliance from `insert-semas`

After Stage 1 and Stage 2 pass, replace the token-DAG analysis with an
alloc/access analysis while still supporting only TMEM.

The new analysis must:

1. Collect `ttng.tmem_alloc` operations that carry `buffer.id`.

2. Group TMEM allocs by `buffer.id`.

3. Forward through memdesc view operations such as subview, transpose,
   reinterpret, and other pure memdesc forwarding operations used by the
   existing pass.

4. Recognize terminal TMEM accesses:

   - `ttng.tmem_load`
   - `ttng.tmem_store`
   - TCGEN5 MMA accumulator uses
   - sourceful `tmem_alloc` as an implicit store event

5. Walk program order through blocks, `scf.for`, and `scf.if`.

6. Infer ownership transitions from access partitions, not from async token
   users.

7. Insert the same semaphore backing allocs, `nvws.sem.create`,
   `nvws.sem.acquire`, `nvws.sem.buffer`, and `nvws.sem.release` operations as
   the current pass.

8. Retarget TMEM access operands to the `nvws.sem.buffer` views.

9. Preserve correct `loop.stage`, `loop.cluster`, partition, and async payload
   annotations on inserted operations.

10. Clean up old token results only after ordering has been derived from memory
    accesses. Replacing dead token uses with poison is allowed as cleanup, but
    token use-def must not drive the schedule.

Acceptance:

- `test/NVWS/insert_semas.mlir` still passes.
- The old `test/NVWS/insert_tmem_semaphore.mlir` still passes unchanged.
- The complex `064 -> 065` equivalence check still passes.
- The full `062` subpipeline equivalence check still passes.

## Stage 4: replace `InsertTmemSemaphore` in automatic WS

Only after Stage 3 passes, update:

```
lib/Dialect/TritonGPU/Transforms/WarpSpecialization/AutomaticWarpSpecialization.cpp
```

Replace:

```
createNVWSInsertTmemSemaphore()
```

with:

```
createNVWSInsertSemas()
```

Do not remove the old pass yet.

Acceptance:

- Build passes.
- `test/TritonGPU/automatic-warp-specialization.mlir` must pass.
- `test/NVWS/insert_semas.mlir` must pass.
- `test/NVWS/insert_tmem_semaphore.mlir` must still pass unchanged.
- No existing lit test is modified.

If `automatic-warp-specialization.mlir` fails:

1. Root-cause whether the failure is an implementation bug, a deliberate IR
   shape difference, or an existing test assumption.

2. If the test expectation needs a change, stop and ask the user for
   authorization before modifying the lit test.

3. If the failure is an implementation bug, fix the implementation.

## Stage 5: prepare for `insert-allocs` plus unified `insert-semas`

After the TMEM-only replacement is proven:

1. Split the current `NVWSInsertSemaphore` responsibility into alloc
   materialization and semaphore insertion.

2. Introduce `insert-allocs` to materialize local or TMEM backing allocs for
   SSA communication values.

3. Run memory planning after `insert-allocs`.

4. Extend `insert-semas` from TMEM-only to both:

   - `ttng.tmem_alloc`
   - `ttg.local_alloc`

5. Use the same ownership and access analysis for both memory spaces.

6. Keep rank-1 floating-point SSA-TMEM channels for alpha/m/l under the
   existing `NVWS_USE_SSA_TMEM=1` policy.

Acceptance:

- TMEM behavior remains identical to the Stage 4 proven implementation.
- Local/SMEM support is added by extending alloc/access recognition, not by
  adding a separate token or SSA-value-driven semaphore scheduler.
- Existing lit tests still do not change without explicit authorization.

## Stage 6: retire old passes

Only after the unified path is proven:

1. Remove `NVWSInsertTmemSemaphore` from the automatic WS pipeline.

2. Remove `NVWSInsertSemaphore` from the automatic WS pipeline.

3. Keep or delete the old pass files based on whether there is still standalone
   debugging value. Deletion should be a separate, reviewable cleanup commit.

4. Add final lit coverage for:

   - TMEM-only replacement behavior
   - local/SMEM semaphore insertion
   - SSA-TMEM alpha/m/l rank-1 floating-point channels
   - loop `stage` / `cluster` preservation across inserted reshapes and
     semaphore operations
   - `scf.for` ownership reconciliation
   - `scf.if` branch reconciliation

## Test modification policy

Allowed without extra authorization:

- Add `test/NVWS/insert_semas.mlir` as a copy of
  `test/NVWS/insert_tmem_semaphore.mlir` with the pass name changed.
- Add new lit tests for new behavior.

Not allowed without explicit user authorization:

- Editing existing lit checks.
- Weakening existing FileCheck patterns.
- Deleting existing lit tests.
- Marking existing tests unsupported or expected-fail.

If an existing lit test has to change, the required report must include:

1. The exact failing test.
2. The exact failing check or verifier error.
3. The implementation root cause.
4. Why the new IR is correct.
5. Why the old check is no longer valid.
6. The minimal proposed test edit.

## Commit staging

Recommended commits:

1. Add `NVWSInsertSemas` as a TMEM-only clone plus
   `test/NVWS/insert_semas.mlir`.

2. Add complex-kernel equivalence coverage or documented repro commands.

3. Replace token-based ordering in `InsertSemas.cpp` with alloc/access-driven
   ordering while keeping TMEM-only behavior.

4. Switch `AutomaticWarpSpecialization.cpp` from
   `createNVWSInsertTmemSemaphore()` to `createNVWSInsertSemas()`.

5. Later: add `insert-allocs` and extend `insert-semas` to local/SMEM.

6. Later: delete or retire the old passes.
