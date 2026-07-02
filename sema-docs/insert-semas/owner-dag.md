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

### Worked example: the loop rule

`test/NVWS/insert_semas_local_buffer_reuse.mlir`
`@local_n_owner_aliased_buffers` — two staggered members, `m0[0,128)` and
`m1[64,192)`, so pieces P0={m0}, P1={m0, m1}, P2={m1}. (A hand-written
stress shape: planner output nests a reuser inside its owner — see
[ACCESS-DAG](access-dag.md#pieces) — but the staggering is what gives the
pieces different first touchers.) The OWNER-DAG dump:

```text
|- scf.for (WS, tag=1) pieces{P0:W:{0},P1:W:{0},P2:W:{2}}
|  |- ENTER pieces{P0:W:{0},P1:W:{0},P2:W:{2}}
|  |- W  m0  ttg.local_alloc {0}
|  |- R  m0  ttg.local_load {1}
|  |- W  m1  ttg.local_alloc {2}
|  |- R  m1  ttg.local_load {0}
|  |- EXIT pieces{P0:W:{0},P1:W:{0},P2:W:{2}}
```

Per piece, the scan stops at the first in-body access whose footprint
reaches it. P0 and P1: the first access is `W m0 {0}` (m0's footprint is
{P0, P1}) — owner `{0}`. P2: no m0 access reaches it; the first access
through m1 is `W m1 {2}` — owner `{2}`. Each piece's effect is the body
merge (every piece is written somewhere in the body, so all are `W`), and
`ENTER`/`EXIT` mirror the loop node.

### Worked example: the if rule

`test/NVWS/insert_semas_raw_if_token.mlir` `@raw_edge_token_carried_if` —
the `if`'s only access is a read by partition `{1}`, yet the `if` node
reports owner `{0}`:

```text
|  |- W  m0  ttng.tmem_store {0}
|  |- scf.if pieces{P0:R:{0}}          <- owner {0}, not the {1} inside
|  |  |- then
|  |  |  |- ENTER pieces{P0:R:{0}}
|  |  |  |- R  m0  ttng.tmem_load {1}
|  |  |  |- EXIT pieces{P0:R:{0}}
|  |  |- else
```

The backward scan from the operation before the `if` stops at the
`tmem_store {0}`, so the piece enters the region under `{0}`'s ownership;
the effect is `R` because nothing in the body writes P0. The read inside
still resolves its own owner `{1}` — the region boundary records who owns
the piece *at the boundary*, and SYNC-DAG's child walk then derives the
`{0}` to `{1}` handoff inside.

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
piece. The child keeps the logical producer known before the region, but uses
`ENTER` and its boundary owner as the concrete version source. New child
readers therefore fan out from `ENTER`, not from a previous child reader or
directly from the outer version's source node. At `EXIT`, the recorded owner
is the target of any handoff needed by the next loop iteration or by a later
access. The later
hold analysis decides whether a semaphore token must pass through the
boundary; OWNER-DAG does not create or thread tokens.

## Code map

[`InsertSemasOwnerDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasOwnerDag.cpp):

- `spliceEnterExit`
- `toucherContribution`
- `assignOwners`
- `verifyOwnerDag`
- `buildOwnerDag`
