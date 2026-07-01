# OWNER-DAG and region boundaries

## Why regions need explicit boundaries

A nested `for` or `if` occupies one position in its parent chain even though
its body may change owners several times. OWNER-DAG records the pieces touched
inside, their `R`/`W` effects, and the owner that the parent sees for each
piece. Model terms are defined in the
[InsertSemas overview](overview.md#core-objects).

OWNER-DAG represents this with:

- a `For` or `If` node at its position in the enclosing chain;
- `ENTER` and `EXIT` marker nodes around each child chain (the body chain of a
  `for`; the then and else chains of an `if`);
- an empty else chain when `scf.if` has no else region.

`ENTER` and `EXIT` are analysis markers, not MLIR operations. They never
receive `loop.stage` or `loop.cluster`; they record the owner and `R`/`W`
effect for each piece at a chain boundary. SYNC-DAG later maps required
ownership changes to real acquire/release nodes.

## Assigning owners to `for` and `if`

Nested `for` and `if` nodes are processed before their enclosing node.

For each piece accessed by the node:

1. **Loop**: scan the loop body from the beginning and stop at the first direct
   access to the piece or nested `for`/`if` whose body accesses the piece. For a
   direct access, use its owner. For a nested `for` or `if`, use the owner
   previously assigned to the piece on that nested `for` or `if` node itself.
2. **If**: scan backward from the operation immediately before the `if` and stop
   at the first direct access to the piece or nested `for`/`if` whose body
   accesses the piece. For a direct access, use its owner. For a nested `for` or
   `if`, use the owner previously assigned to the piece on that nested `for` or
   `if` node itself. If no earlier operation accesses the piece, scan the then
   chain from the beginning, followed by the else chain, and apply the same
   rule.
3. **WS scope boundary**: a WS-tagged loop reports root to its parent, because
   partition ownership is meaningful only inside the WS scope that defines it.
   An access with its own WS tag keeps its resolved owner even when it is not
   enclosed by a WS loop.

`ENTER` and `EXIT` are ignored while searching for these owners. These are
fixed ownership rules, not profitability choices.

## `ENTER` and `EXIT` records

For each child chain, `ENTER` and `EXIT` list exactly the pieces accessed in
that chain. Each piece uses the owner recorded on the enclosing `for` or `if`;
its effect is `W` if any access in the chain writes it, otherwise `R`. A chain
that accesses no pieces has empty `ENTER` and `EXIT` records.

The verifier checks:

```text
ENTER and EXIT contain the same pieces, owners, and effects
the recorded pieces are exactly those accessed in the chain
each owner matches the owner recorded on the enclosing for or if
```

## Why this matters downstream

SYNC-DAG treats each `for` or `if` node as one access in the parent chain. At
`ENTER`, it starts analyzing the child chain with the owner recorded for each
piece. At `EXIT`, that owner is the target of any handoff needed by the next
loop iteration or by a later access. The later hold analysis decides whether a
semaphore token must pass through the boundary; OWNER-DAG does not create or
thread tokens.

## Code map

[`InsertSemasOwnerDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasOwnerDag.cpp):

- `spliceEnterExit`
- `toucherContribution`
- `assignOwners`
- `verifyOwnerDag`
- `buildOwnerDag`
