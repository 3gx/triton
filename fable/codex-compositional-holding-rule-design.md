# Compositional Holding Rule For Nested Regions

Status: design extension and clarification.

This document reframes the "nested loop extension" as a more general rule:
SYNC-DAG should handle every nested region boundary compositionally. The same
ENTRY/EXIT region protocol applies to an `scf.for` body, an `scf.if` then
region, and an `scf.if` else region. The surrounding control op still has its
own routing semantics, but the internal semaphore game for each child region is
uniform.

Relevant background documents:

- `fable/semas-report3.md`
  - Owner assignment rules for loops and ifs.
  - Fresh local state for region bodies.
  - Region super-node rows that are uniform for `For` and `If`.
  - Addendum B holding rule.
- `fable/hold-rule-implementation-plan.md`
  - Native hold-rule emission.
  - Boundary-device versus point-of-use placement.
  - Current acceptance gates for preserving performance.
- `fable/new-insert-semas-plan-2.md`
  - `crossings`, `requiredParts`, and EmitIR transcription rules.
  - Outside-in rewriting of `For` and `If` signatures.
- `fable/extend-design-to-nested-loops-plan-v2.md`
  - Earlier nested-loop-specific framing. This design generalizes that idea to
    every nested region, including `if` branches.

## Core Claim

We do not need a special design only for nested loops. We need the SYNC-DAG to
respect a uniform nested-region composition model.

For any nested region chain:

1. The parent SYNC-DAG sees the nested `For` or `If` operation as one
   owner/effect super-node.
2. The child region starts a fresh local semaphore game at ENTRY.
3. The child local game is seeded from the parent producer snapshot and carried
   owner for that memory component.
4. The holding rule is applied inside that child region.
5. EXIT returns the carried owner, or records the crossing final needed by the
   parent.
6. EmitIR renders the resulting SYNC-DAG facts mechanically.

By applying this recursively, arbitrary `for`/`if` nesting is correct by
composition. The single-region holding rule is the base case; each nested
control op becomes a summarized node in its parent.

## Non-Goals

This design does not change OWNER-DAG policy.

This design does not change PartitionLoops. A `for` or `if` op still executes in
all partitions that its body needs. Separately, for ownership/semaphore
purposes, each memory piece has one owner at the region boundary.

This design does not ask EmitIR to invent synchronization policy. EmitIR should
continue to render internally consistent SYNC-DAG facts: `Acquire`, `Release`,
`Access`, `Region`, `crossings`, `requiredParts`, and route facts.

This design must not change any single-loop-level behavior. If a lit test with
only one region level changes, that is a regression unless the change is proven
to be unrelated formatting.

## Owner Is Not The Same As Execution Partitions

For a memory component, OWNER-DAG assigns a single owner to each op row. That is
the partition responsible for the component's semaphore ordering at that point.

The op can still execute in multiple partitions.

Example:

```mlir
for {tt.ws} {
  op1 {1}
  if {1} {
    op2 {2}
  }
}
```

The `if` op must execute in both partitions `{1, 2}` because partition `{2}`
contains `op2`, and partition `{1}` owns the branch boundary for the component.
For semaphore ownership, the `if` row has owner `{1}` for that component. For
PartitionLoops, the op has partition set `{1, 2}`.

That distinction is essential:

- Owner controls semaphore handoff.
- Required parts control where the control op and its operands/results must be
  available after partitioning.

## Uniform Region Boundary Model

For every child region chain, independent of whether the parent op is `For` or
`If`, SYNC-DAG uses the same conceptual shape:

```text
parent chain
  ...
  RegionOp row     # owner/effect summary in parent
  ...

child chain
  ENTER
    local Access/Acquire/Release/Region rows
  EXIT
```

The child chain is not allowed to mutate the parent's in-region state directly.
Instead, it imports the relevant parent producer snapshot at ENTER and exports
the final owner/payload state through EXIT or crossing facts.

Uniform handling means:

```text
solveRegion(childChain, incomingOwner, incomingProducer):
  seed local state at ENTER
  run the same holding-rule scheduler inside childChain
  close the child at EXIT when the result is load-bearing
  report final owner/payload back to the control-op wrapper
```

The wrapper around the region then applies control-specific routing:

- `For` routes loop-carried values through recurrence.
- `If` chooses one dynamic branch and uses pass-through for untaken or empty
  paths.
- Dead `If` crossings can still be liveness-pruned when no later consumer needs
  them.

Those are routing differences, not different semaphore-region algorithms.

## What Remains Different Between For And If

The internal ENTRY/EXIT protocol is uniform. The surrounding control op still
has different control-flow behavior.

For `scf.for`:

- The body may run multiple iterations.
- A component can be loop-carried across iterations.
- The holding rule decides whether a crossing needs boundary-device placement or
  can remain native point-of-use.
- Recurrence means a post-body owner can become the next iteration's incoming
  owner.

For `scf.if`:

- Only one branch executes dynamically.
- Each branch has its own child region chain.
- A missing branch-side producer is represented as pass-through from the
  incoming owner/payload.
- Crossings that are not consumed after the if can be pruned.

The design is still uniform because each branch/body region is solved by the
same region-chain algorithm.

## Soundness By Composition

The soundness argument is structural.

Base case: a single region chain is correct when the holding rule is correct
inside that chain.

For a component in one region chain, the holding rule cuts at owner or predicate
context changes. Each hold has:

- an acquire before the first access needing the new owner,
- the relevant accesses under that owner,
- a release after the last access needed by the next owner.

That gives the local chain sequential consistency for the component.

Induction step: assume every child region chain is locally correct and exports a
correct summary. The parent only observes the child as a RegionOp super-node with
one owner/effect row for the component. Replacing the child with that summary
preserves the parent's ordering because:

- all internal child accesses are ordered by the child local game,
- the child ENTER import is ordered with the parent producer snapshot,
- the child EXIT or crossing final is ordered before the next parent consumer,
- the parent never relies on unmodeled internal child state.

Therefore, if every immediate child region is correct, the parent region is
correct. Recursing from leaves to root gives correctness for arbitrary nesting of
`for` and `if`.

## Performance Preservation

The existing holding rule already improves inner-loop performance by keeping
ungated components at point-of-use instead of forcing boundary-device
acquire/release placement.

This design preserves that property.

The parent does not flatten the child and blanket-guard the whole nested body
unless the holding rule says that the parent-level crossing itself is gated. The
child still runs its own local hold game. If the inner loop can keep a handoff
native, it remains native inside the inner loop, even when the whole inner loop
is nested under an outer region.

In other words:

- Parent level: synchronize the RegionOp summary when the parent needs it.
- Child level: synchronize the child accesses with the existing holding rule.

The composition adds correctness at nested boundaries without giving up the
performance already obtained at inner levels.

## Example 1: Inner For Is Solved The Same With Or Without Outer For

Input:

```mlir
for {tt.ws} {
  op1 {1}
  for {1} {
    op2 {1}
    op3 {2}
  }
}
```

Parent view:

```mlir
for {tt.ws} {
  op1 {1}
  INNER_FOR {owner = 1, parts = {1, 2}}
}
```

The inner loop has its own local semaphore set:

```mlir
for {1} {
  op2 {1}
  release S0 {1}

  acquire S0 {2}
  op3 {2}
  release S1 {2}

  acquire S1 {1}
}
```

After partitioning, conceptually:

```mlir
// partition 1
for {
  op2
  release S0
  acquire S1
}

// partition 2
for {
  acquire S0
  op3
  release S1
}
```

If there is no outer loop, this is the complete local game. If there is an
outer loop, the outer loop only sees the inner loop as one summary row owned by
`{1}` and required in `{1, 2}`. The inner loop's local protocol is unchanged.

## Example 2: Outer Store, Inner Cross-Partition Store, Outer Load

Input:

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

The outer region sees:

```mlir
for {tt.ws} {
  store A {1}
  INNER_FOR {owner = 1, parts = {1, 2}}
  load A {1}
}
```

The inner region is responsible for the handoff to `{2}` and the handback to
`{1}`:

```mlir
for {1} {
  load A {1}
  release S0 {1}

  acquire S0 {2}
  store A {2}
  release S1 {2}

  acquire S1 {1}
}
```

The bottom `load A {1}` in the outer loop is ordered after the inner
`store A {2}` because the inner region exits with ownership returned to `{1}`.
The parent then continues from the RegionOp summary to the bottom load.

Partition view:

```mlir
// partition 1
for {
  store A
  for {
    load A
    release S0
    acquire S1
  }
  load A
}

// partition 2
for {
  for {
    acquire S0
    store A
    release S1
  }
}
```

There is no race between the bottom outer load and the inner store. The ordering
comes from the child region handback plus the parent summary ordering.

## Example 3: Read-Only Inner For With Different Executor

Input:

```mlir
for {tt.ws} {
  store A {1}
  for {2} {
    load A {2}
  }
  load A {1}
}
```

One legal local protocol is:

```mlir
for {tt.ws} {
  store A {1}
  release S0 {1}

  acquire S0 {2}
  for {2} {
    load A {2}
  }
  release S1 {2}

  acquire S1 {1}
  load A {1}
}
```

The important point is not the exact placement spelling. The important point is
that the `{2}` read is ordered after the `{1}` store, and `{1}` regains the
component before the later `{1}` load or before the next iteration's `{1}`
producer.

If the holding rule can place this natively at point-of-use, it should do so. If
the loop recurrence forces boundary-device placement, the boundary-device form
is also correct. The compositional rule allows either placement only when it is
already a valid holding-rule placement.

## Example 4: Owner Of Region Op Versus Body Partitions

Input:

```mlir
for {tt.ws} {
  store A {1}
  for {1} {
    load A {2}
  }
}
```

Here the inner `for` is owned by `{1}` for the component, but it must execute in
both `{1}` and `{2}` because its body contains a `{2}` access.

Pre-partition local protocol:

```mlir
for {tt.ws} {
  store A {1}
  for {1} {
    release S0 {1}

    acquire S0 {2}
    load A {2}
    release S1 {2}

    acquire S1 {1}
  }
}
```

Partition view:

```mlir
// partition 1
for {
  store A
  for {
    release S0
    acquire S1
  }
}

// partition 2
for {
  for {
    acquire S0
    load A
    release S1
  }
}
```

This can be inefficient because partition `{1}` runs the inner loop only to
bracket the semaphore protocol. It is still functionally correct. That is the
expected consequence of keeping owner and execution partition separate.

If OWNER-DAG would normally derive the inner loop owner from first touch, then a
body whose only real access is `{2}` should normally become owned by `{2}`. The
example assumes `{1}` is the carried owner chosen by OWNER-DAG for that region
row.

## Example 5: If Region Uses The Same ENTRY/EXIT Protocol

Input:

```mlir
for {tt.ws} {
  op1 {1}
  if {1} {
    op2 {2}
  }
}
```

Parent view:

```mlir
for {tt.ws} {
  op1 {1}
  IF {owner = 1, parts = {1, 2}}
}
```

The then region is just a child region chain:

```mlir
if {
  release S0 {1}

  acquire S0 {2}
  op2 {2}
  release S1 {2}

  acquire S1 {1}
} else {
  // pass-through: incoming owner/payload remains {1}
}
```

Partition view:

```mlir
// partition 1
for {
  op1
  if {
    release S0
    acquire S1
  } else {
    // pass-through
  }
}

// partition 2
for {
  if {
    acquire S0
    op2
    release S1
  } else {
    // no real work for this component
  }
}
```

The then region is handled exactly like a nested `for` body from the semaphore
region perspective. The only `if`-specific behavior is dynamic branch routing:
the else path passes through the incoming `{1}` owner.

## Example 6: Nested For/If/For/If

Input shape:

```mlir
for {tt.ws} {
  op0 {0}
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

The recursive summaries are:

```text
root for body
  op0 {0}
  IF0 summary

IF0 then child
  INNER_FOR summary

INNER_FOR body child
  IF1 summary

IF1 then child
  op1 {1}
  op2 {2}
```

SYNC-DAG solves from the innermost child outward:

1. Solve `IF1` then region as a local chain and route else pass-through.
2. Summarize `IF1` to the inner-for body.
3. Solve the inner-for body recurrence.
4. Summarize the inner for to the `IF0` then branch.
5. Solve `IF0` then/else routing.
6. Summarize `IF0` to the root loop body.
7. Solve the root loop body.

No level needs to inspect arbitrary grandchildren as special cases. Each level
only consumes the summary of its immediate child regions.

## Example 7: Release Count And Fan-In Stay Local

Nested fan-in is also compositional.

Shape:

```mlir
for {tt.ws} {
  producer {3}
  for {3} {
    read0 {2}
    read1 {1}
    store {1}
    load {0}
  }
}
```

The inner region has a local fan-out from `{3}` to `{2}` and `{1}`. Its release
count is computed inside the inner local game. If both `{2}` and `{1}` acquire
from the same producer, the release count belongs to the inner semaphore node,
not to an outer flattened protocol.

The parent sees the inner loop summary. It does not recompute the inner fan-in
or fan-out by scanning the inner body as if it were inline parent state.

## Required Parts

After the child region local game is built, the control op's `requiredParts`
must include every partition that needs the op after partitioning:

- partitions that execute real accesses in the child,
- partitions that own acquires/releases in the child,
- partitions needed by nested child regions,
- partitions needed for control operands, bounds, conditions, and yielded
  values.

This requirement is recursive. A parent `For` or `If` must include the parts
required by nested regions, even if the parent row has a single semaphore owner
for a given component.

## Conditions, Bounds, And Dominance

Uniform region handling must preserve control dependencies.

If adding semaphore protocol expands `requiredParts`, then EmitIR must be able
to make these values available in every required partition:

- `scf.if` condition,
- `scf.for` bounds and step,
- loop-carried values and yielded semaphore tokens/views,
- branch-local semaphore state.

Branch-local semaphore creation or branch-local first acquire must not be
hoisted to a location that violates dominance. If a semaphore is only created on
one branch, its protocol must remain inside the branch or be rewritten with a
valid outer producer. The uniform region model does not permit dominance
shortcuts.

## Relation To Current EmitIR

EmitIR already has the right high-level shape:

- rewrite `For` and `If` signatures outside-in,
- add loop iter args for loop crossings,
- add if results for branch crossings,
- render regions recursively,
- restamp partition outputs from `requiredParts`,
- verify partition-output and token/view locality.

The design intent is that the implementation change should be localized to
SYNC-DAG construction and placement. EmitIR should only need to render a more
complete and internally consistent SYNC-DAG.

If EmitIR needs a change, it should be mechanical, such as:

- accepting a new crossing placement fact,
- asserting an invariant that SYNC-DAG now guarantees,
- exposing a verifier error when `requiredParts` cannot be satisfied.

It should not decide whether a nested loop or branch should be synchronized.

## Design Invariants

The implementation must preserve these invariants:

1. OWNER-DAG policy is unchanged.
2. A child region is solved from a fresh local state seeded at ENTRY.
3. Parent state observes only the child RegionOp summary.
4. A `For` or `If` op can be owned by one partition for a component and still be
   required in multiple execution partitions.
5. Inner hold-rule placement remains native/point-of-use whenever the current
   holding rule permits it.
6. Boundary-device placement is used only when required by the holding rule.
7. `If` pass-through remains explicit for untaken or empty paths.
8. Dead `If` crossings remain liveness-prunable.
9. Release counts are computed at the local region level that owns the fan-out.
10. Single-level tests do not change.

## Practical Reading Of The Design

When debugging a nested case, read each level independently:

1. What is the owner/effect summary of the nested `For` or `If` row in the
   parent?
2. What partitions are required by the nested child protocol?
3. What is the child region's incoming owner and producer snapshot?
4. What local acquire/release game does the holding rule create inside that
   child?
5. What final owner/payload does EXIT return to the parent?

If every level answers those questions consistently, the whole nested program is
correct by composition.
