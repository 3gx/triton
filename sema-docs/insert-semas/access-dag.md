# ACCESS-DAG

## Purpose

ACCESS-DAG is the memory fact layer. It records physical overlap,
program-order accesses, execution owners, and region effects without deciding
synchronization. Model terms are defined in the
[InsertSemas overview](overview.md#core-objects).

## Group and piece discovery

Mutable local allocations and all TMEM allocations are grouped by `buffer.id`
in separate memory-kind namespaces; an allocation without an ID receives a
private synthetic group.

Members may partially overlap. Missing `buffer.offset` is zero. The address
line is measured in TMEM columns for TMEM (extent = the full memdesc size)
and in leading-dimension elements for local memory (extent = the leading
shape dimension). The
algorithm cuts that address line at every member start and end, merges adjacent
intervals with identical coverage, and builds each member's piece footprint.
Pieces connected through a common member form one component.

Two planned layouts are analyzed as separate groups even though they share one
physical allocation; the shared storage is reunified later, by EMIT-IR:

- a circular local member requires `buffer.id`, `buffer.copy`, and
  `buffer.start`, forbids `buffer.offset`, uses logical offset zero, and forms
  a separate group over the shared multi-buffered backing;
- when every TMEM member of one `buffer.id` has `buffer.copy` and the values
  differ, each member forms a separate group over the physical allocation.

The second rule is TMEM-only. Non-circular SMEM members with one `buffer.id`
remain one group; if their `buffer.copy` values are inconsistent, SYNC-DAG
later rejects the group with a diagnostic when it computes the backing plan
(`getPlannedBufferCopy`).

## Access discovery

The function is scanned in program order. For each group, the pass follows its
memdescs through `memdesc_index`, `memdesc_trans`, `memdesc_reinterpret`, and
`memdesc_reshape`; these operations create another view of the same allocation
and are not memory accesses. If an operation consumes a group memdesc and its
sole result is another memdesc, it must be one of these view operations or the
pass rejects it with a diagnostic — in particular `ttg.memdesc_subslice` is
rejected, not followed.

Accesses inside `scf.for` and `scf.if` are supported. Passing a group's
memdesc through an `scf.for` argument/result, an `scf.yield`, or a function
return is rejected with a diagnostic. Other region-holding operations (for
example `scf.while`) are not scanned: accesses inside their bodies are
invisible to the analysis, with no diagnostic. If such an operation consumes
a group memdesc, it is conservatively recorded as a write — unless its sole
result is a memdesc, in which case the view-operation rule above rejects it.
A `for` or `if` node
is kept in ACCESS-DAG only when its body touches the group.

Memory effects are classified as follows:

- `R`: `local_load`, `tmem_load`, and non-accumulator MMA operands whose
  memdesc belongs to the group;
- `W`: SMEM/TMEM stores, sourceful SMEM/TMEM allocations, descriptor
  load/gather destinations, and MMA accumulators;
- fallback `W`: any other operation that directly consumes a group memdesc is
  conservatively recorded as `W`.

Fallback `W` means an exclusive synchronization touch, not necessarily a
literal memory store. If one operation reaches the same physical piece through
multiple views, its effects are combined; any `W` makes the combined effect
`W`.

Each access inside a tagged WS scope records its one executing partition as
`(ttg.partition, WS tag)` and keeps the view chain needed to rebuild its
memdesc during emission. An access outside a tagged WS scope records root.

Inside a tagged WS scope, every access to an SMEM or TMEM allocation handled
by `InsertSemas` must execute in one partition. Multi-partition accesses are
not supported; as described in the [overview](overview.md#core-objects), they
are not diagnosed and are silently treated as root-owned.

## Completion frontier

Normally an access completes at the memory operation itself. A `local_load`
from an SMEM allocation handled by `InsertSemas` that feeds exactly one
same-block descriptor store, directly or through one `convert_layout`, remains
physically live until that store. The DAG records the store as
`completionAnchor`; later steps use it for release placement and scheduling
without re-discovering the pattern.

If no descriptor-store candidate exists, the ordinary completion point is
kept. Once a candidate exists, every extra direct load/convert user,
control-flow crossing, or owner mismatch is rejected.

## Effects inside `for` and `if`

For every retained `for` and `if`, ACCESS-DAG records each piece accessed in
its body. The recorded effect is `W` if any access writes the piece, otherwise
`R`. No owner is assigned yet, and the `ENTER`/`EXIT` boundary markers are
added by the next step, OWNER-DAG.

## Output

The step produces, per group:

```text
members + pieces + components + view-chain map
program-order Access/For/If node chains
per-access owner/effect/completion facts
pieces and R/W effects recorded on each for/if node
```

## Code map

[`InsertSemasAccessDag.cpp`](../../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemasAccessDag.cpp):

- `collectGroups`
- `buildPieces`
- `collectTouches` and `deriveCompletionAnchor`
- `buildChainForBlock` and `computeEffectSummary`
- `buildAccessDag`
