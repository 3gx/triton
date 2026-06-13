# Plan: Compositional Holding Rule For Nested Regions

Status: implementation plan.

This plan implements the design in
`fable/codex-compositional-holding-rule-design.md`.

The main correction is conceptual: this is not a special nested-loop feature.
It is a uniform nested-region SYNC-DAG construction. Every child region chain is
handled with the same ENTRY/EXIT holding-rule protocol, whether the child region
is an `scf.for` body, an `scf.if` then region, or an `scf.if` else region.

## Scope

In scope:

- Make SYNC-DAG construction explicitly compositional for nested `For` and `If`
  regions.
- Preserve OWNER-DAG policy.
- Preserve single-level behavior.
- Preserve inner-loop holding-rule performance.
- Keep EmitIR mechanical: render consistent SYNC-DAG facts.
- Add targeted nested-region tests that show composition through `for`, `if`,
  and mixed nesting.

Out of scope:

- Changing PartitionLoops.
- Changing the OWNER-DAG owner policy.
- Adding ad hoc EmitIR scheduling policy.
- Running pytest. Per repository instructions, do not run pytest unless the user
  explicitly asks.

## Acceptance Criteria

1. All existing single-loop-level `InsertSemas` lit tests remain unchanged.
2. Nested-region lit tests either remain unchanged or change only where the new
   design deliberately fixes nested handling.
3. Inner-loop point-of-use/native holding-rule placement remains available.
4. A nested `For` or `If` can be owned by one partition for a component while
   being required in every partition that executes body work or semaphore
   protocol.
5. `If` pass-through and liveness pruning still work.
6. Release counts for local fan-out remain correct.
7. EmitIR receives an internally consistent SYNC-DAG and does not need to infer
   synchronization policy.
8. Build passes before lit tests are run.

If any single-level test output changes, stop and explain the exact diff before
continuing.

## Files Expected To Change

Primary implementation file:

- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp`

Possible supporting files:

- `third_party/nvidia/include/Dialect/NVWS/Transforms/InsertSemas.h`
  - Only if existing crossing/placement fields cannot express the
    compositional placement.
- `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp`
  - Only for mechanical rendering or verifier changes.
  - Do not add semantic placement decisions here.
- `test/NVWS/insert_semas*.mlir`
  - Add focused nested-region coverage.
  - Update goldens only after build and after confirming the change is expected.
- `fable/*.md`
  - Keep design and plan documentation aligned.

## Current Code Shape To Preserve

`InsertSemasSyncDag.cpp` already has useful structure:

- `walkChain` handles `Node::For` and `Node::If` in the same case.
- Region body traversal uses fresh local games seeded from parent facts.
- Nested regions contribute to `requiredParts`.
- The verifier traverses nested `For` and `If`.

The weak point is that the current holding-rule placement is still too
For-specific. The implementation should make placement decisions per child
region chain, not by special-casing only nested loop rows.

`InsertSemasEmitIR.cpp` already has useful structure:

- Signature rewrites are outside-in for `For` and `If`.
- `For` loop-carried tokens/views are added as iter args.
- `If` crossings are added as results.
- Region rendering recurses.
- Partition outputs are restamped from `requiredParts`.
- Token/view locality verifiers traverse nested control ops.

The plan assumes EmitIR remains mostly unchanged.

## Milestone M0: Align Terms And Static Invariants

Goal: make the code and docs use the same mental model before changing
behavior.

Tasks:

1. Add this design and plan document.
2. Re-read:
   - `fable/semas-report3.md`
   - `fable/hold-rule-implementation-plan.md`
   - `fable/new-insert-semas-plan-2.md`
   - `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasSyncDag.cpp`
   - `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasEmitIR.cpp`
   - `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp`
3. Fix stale comments only if they contradict the design:
   - `if` owner comments must say incoming-owner/Rule A, not arbitrary branch
     first-touch unless no incoming owner exists.
   - nested `For` comments must not imply that a child body directly mutates
     parent state.
4. Confirm the dump format exposes enough facts to audit:
   - region owner,
   - region required parts,
   - crossings,
   - final owner/payload route,
   - hold placement,
   - release counts.

Exit criteria:

- Documentation is internally consistent.
- No code behavior changes yet.

## Milestone M1: Compute Compositional Placement Side-By-Side

Goal: add an analysis-only computation that can be compared against the current
placement without changing emitted IR.

Tasks:

1. Define a helper around the existing chain solver, conceptually:

   ```c++
   RegionSummary solveRegionChain(Chain &child,
                                  Owner incomingOwner,
                                  ProducerSnapshot incomingProducer,
                                  RegionKind kind);
   ```

   The helper does not need this exact signature. The important part is that it
   returns a summary for one child chain and does not inspect arbitrary
   grandchildren except through their summaries.

2. For every `Node::For` and `Node::If` child region:
   - seed local state at ENTRY,
   - run the holding-rule cut/hold calculation inside that chain,
   - record local acquire/release placement,
   - record EXIT or crossing final state,
   - compute recursive `requiredParts`.

3. Print or dump the side-band result under the existing debug/dump mechanism.

4. Compare side-band compositional placement to current placement on the nested
   corpus:
   - `logs/nested-12jun26-v1`
   - `test/NVWS/insert_semas*.mlir`
   - `test/NVWS/tmem-buffer-reuse-semas.mlir`

Exit criteria:

- Side-band analysis gives the same result for single-level cases.
- Nested cases show per-region local placement instead of requiring parent
  flattening.
- No IR output changes yet.

## Milestone M2: Switch SYNC-DAG To Compositional Placement

Goal: make SYNC-DAG insertion use the compositional placement.

Tasks:

1. In `InsertSemasSyncDag.cpp`, make `For` and `If` region handling share the
   same child-chain protocol:
   - ENTRY seeds from parent producer snapshot.
   - local game is solved inside child chain.
   - EXIT returns the carried owner/payload when load-bearing.
   - route facts are stored on the parent control op.

2. Keep the control-specific routing outside the local child solver:
   - `For`: recurrence and loop-carried crossing routing.
   - `If`: then/else routing, pass-through, and liveness pruning.

3. Relax any placement rule that forces a nested final to boundary-device only
   because it is nested. Boundary-device placement should be selected because
   the holding rule requires it, not merely because the producer is inside a
   nested region.

4. Preserve point-of-use/native placement for ungated inner holds.

5. Ensure release-count computation remains local to the region chain that owns
   the fan-out.

6. Recompute `requiredParts` recursively after acquire/release insertion:
   - access owners,
   - acquire/release owners,
   - nested region required parts,
   - control condition/bounds/yield users.

Exit criteria:

- SYNC-DAG dump shows each child region solved independently.
- Parent rows only consume child summaries.
- Existing single-level dumps are unchanged except for harmless debug wording.

## Milestone M3: Keep EmitIR Mechanical

Goal: make only the mechanical EmitIR changes needed to render the new SYNC-DAG.

Tasks:

1. Verify `rewriteSignatures` still handles every new crossing fact
   outside-in.

2. Verify `renderRegion` can render:
   - child-local acquires/releases,
   - loop-carried crossing finals,
   - if branch crossing finals,
   - branch pass-through.

3. If a new placement enum or route fact is introduced, teach EmitIR to render
   it directly. Do not make EmitIR choose placement.

4. Strengthen verifiers if needed:
   - partition outputs contain every required part,
   - condition/bounds values are available in every required partition,
   - semaphore token/view locality is maintained,
   - branch-local semaphores do not escape dominance,
   - crossing finals have a producer or explicit pass-through.

Exit criteria:

- EmitIR remains a transcription layer.
- Any EmitIR code change is tied to a concrete SYNC-DAG fact.

## Milestone M4: Add Focused Tests

Goal: cover the composition cases that motivated this design.

Add or update lit tests under `test/NVWS/`.

Required test shapes:

1. Inner `for` with owner `{1}` and body parts `{1, 2}`:

   ```mlir
   for {tt.ws} {
     op1 {1}
     for {1} {
       op2 {1}
       op3 {2}
     }
   }
   ```

   Expected: inner loop has its own semaphore set; parent sees a region summary.

2. Outer store, inner cross-partition store, outer load:

   ```mlir
   for {tt.ws} {
     store A {1}
     for {1} {
       load A {1}
       store A {2}
     }
     load A {1}
   }
   ```

   Expected: handoff to `{2}` and handback to `{1}` happen inside the inner
   region; bottom load is ordered after inner store.

3. Region owner differs from only real body executor:

   ```mlir
   for {tt.ws} {
     store A {1}
     for {1} {
       load A {2}
     }
   }
   ```

   Expected: partition `{1}` may run the inner loop only for semaphore bracket
   protocol; partition `{2}` runs the load. Correctness matters more than local
   efficiency for this case.

4. `if` branch with different body executor:

   ```mlir
   for {tt.ws} {
     op1 {1}
     if {1} {
       op2 {2}
     }
   }
   ```

   Expected: then branch uses the same local ENTRY/EXIT protocol as a child
   region; else path passes through incoming `{1}` owner/payload.

5. Mixed `for`/`if` nesting:

   ```mlir
   for {tt.ws} {
     if {0} {
       for {1} {
         if {1} {
           op1 {1}
           op2 {2}
         }
       }
     }
   }
   ```

   Expected: each level consumes only immediate child summaries.

6. Nested release-count fan-out:

   ```mlir
   for {tt.ws} {
     producer {3}
     for {3} {
       read0 {2}
       read1 {1}
     }
   }
   ```

   Expected: inner release count reflects local fan-out and is not recomputed by
   the parent.

7. Single-level unchanged sentinels:
   - one single-level `for` hold-rule point-of-use case,
   - one single-level `if` pass-through case.

Exit criteria:

- Tests prove nested composition.
- Tests prove single-level behavior did not move.

## Milestone M5: Validation Commands

Per repository instructions, build first, then run lit tests.

Build:

```sh
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Targeted lit:

```sh
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test/NVWS/insert_semas*.mlir test/NVWS/tmem-buffer-reuse-semas.mlir
```

If the implementation touches partitioning-visible control-flow behavior, also
run the narrow warp-specialization lit tests that cover partition metadata.

Do not run pytest unless explicitly requested.

## Stop Conditions

Stop and report before continuing if any of these happen:

1. A single-level lit test changes.
2. `requiredParts` grows to include a partition where a condition, bound, token,
   view, or yielded value cannot be made available.
3. An `if` branch-local semaphore would need to be hoisted across a dominance
   boundary.
4. A release count changes without a matching local fan-out explanation.
5. EmitIR needs to decide synchronization placement instead of rendering a
   SYNC-DAG fact.
6. PartitionLoops or a later verifier rejects the required partition set.
7. A nested case can only be fixed by changing OWNER-DAG policy.

## Implementation Notes

Use the following mental model while editing:

```text
for each parent chain:
  for each node in chain:
    if Access:
      process local access

    if Acquire/Release:
      process local semaphore op

    if RegionOp:
      for each child region chain:
        childSummary = solve child chain with fresh local state
      process RegionOp in parent from child summaries
```

Do not flatten grandchildren into the parent game. The parent consumes summaries
only. The child remains responsible for its own local acquire/release placement.

For `If`:

```text
thenSummary = solveRegionChain(then)
elseSummary = solveRegionChain(else) or pass-through
ifSummary = route dynamic branch result
prune dead crossings if no later consumer needs them
```

For `For`:

```text
bodySummary = solveRegionChain(body)
loopSummary = route recurrence through bodySummary
use boundary-device placement only when the holding rule requires it
```

## Expected Result

After this plan, the design should be simpler:

- OWNER-DAG remains the single source of owner assignment.
- SYNC-DAG uniformly handles nested region composition.
- EmitIR renders the SYNC-DAG without policy decisions.
- Inner holding-rule performance is preserved.
- Arbitrary nested `for`/`if` correctness follows from single-region
  correctness by composition.
