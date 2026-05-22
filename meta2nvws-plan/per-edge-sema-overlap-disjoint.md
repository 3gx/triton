# Container Overlap vs. Sub-View Disjointness — Resource-Key Model

## Problem

A `buffer.id` group may contain a *container* member (covering the full
physical slot) plus several *sub-view* members that tile or partially
cover the container. The container overlaps every sub-view, so any
access to a sub-view physically conflicts with any access to the
container. But the sub-views themselves are typically disjoint with
each other, so cross-sub-view accesses *do not* physically conflict and
should not require synchronization.

The current `assignTmemResourceKeys` implementation uses union-find on
pairwise overlap, which is **transitive**: if `m0` overlaps `m1` and
`m0` overlaps `m2`, then `m1` and `m2` end up in the same equivalence
class even when they are physically disjoint. That over-approximates
the conflict set and forces unnecessary synchronization between
sub-views.

The v4 plan §Physical Conflict Key states:

> If two touches are in the same logical buffer-id group but are
> physically non-overlapping, they should have different `resourceKey`s
> so the planner does not over-synchronize them.

A single `resourceKey` per equivalence class violates the spirit of
that rule whenever a container exists.

## Examples

### Example 1 — Disjoint sub-views inside a container

`test/NVWS/insert_semas_tmem_container_subviews.mlir::container_with_disjoint_subviews`

```
m0 = [0, 256)   container
m1 = [0, 128)   sub-view
m2 = [128, 192) sub-view
m3 = [192, 256) sub-view
```

Pairwise conflict matrix (1 = overlap, 0 = disjoint):

```
      m0  m1  m2  m3
  m0   -   1   1   1
  m1   1   -   0   0
  m2   1   0   -   0
  m3   1   0   0   -
```

**Today**: union-find unites everything through `m0` → all members
share `resourceKey = 0`. A store on `m1` and a store on `m2` running
in different partitions will synchronize through one semaphore, even
though they touch disjoint physical bytes.

**Optimal**: three atomic sub-ranges — `A = [0,128)`, `B = [128,192)`,
`C = [192,256)`. Touch sets:

```
m0 covers {A, B, C}
m1 covers {A}
m2 covers {B}
m3 covers {C}
```

Two touches conflict iff their atom sets intersect:

- `m1` ∩ `m2` = ∅ → no synchronization between them
- `m1` ∩ `m0` = {A} → synchronize
- `m2` ∩ `m0` = {B} → synchronize
- `m3` ∩ `m0` = {C} → synchronize
- `m2` ∩ `m3` = ∅ → no synchronization between them

The OWNERSHIP-DAG would be printed as three independent trees keyed by
atom, with each touch contributing rows to the trees of every atom it
covers (`m0` appears in all three; `m1`/`m2`/`m3` each in one).

### Example 2 — Meta flash-attention `buffer.id = 4`

`test/NVWS/insert_semas_meta_fa_fwd.mlir`

```
m3 = [0, 128)   container (qk_0, type 128x128 f32)
m0 = [64, 65)   sub-view  (alpha,        type 128x1 f32)
m1 = [66, 67)   sub-view  (offsetkv_y_2, type 128x1 f32)
m2 = [65, 66)   sub-view  (offsetkv_y_1, type 128x1 f32)
m4 = [0, 128)   coextensive with m3 but different element type (f16)
```

Pairwise conflict (ignoring `m4`'s element-type-based separation):

```
      m0  m1  m2  m3
  m0   -   0   0   1
  m1   0   -   0   1
  m2   0   0   -   1
  m3   1   1   1   -
```

`m0`, `m1`, `m2` are pairwise disjoint single-column sub-views;
`m3` is the full container.

**Today**: all four members share `resourceKey = 2`. Touches on the
three single-column sub-views will all synchronize through the same
semaphore, even though their column addresses don't overlap.

**Optimal**: atomic sub-ranges within `[0, 128)` are
`[0,64) [64,65) [65,66) [66,67) [67,128)`. Touch sets:

```
m3 covers all five
m0 covers {[64,65)}
m2 covers {[65,66)}
m1 covers {[66,67)}
```

Cross-sub-view pairs `m0`/`m1`/`m2` have empty atom intersection → no
synchronization between them. Each sub-view still synchronizes with
`m3` via its single atom.

### Example 3 — Overlapping sub-views inside a container

`test/NVWS/insert_semas_tmem_container_subviews.mlir::container_with_overlapping_subviews`

```
m0 = [0, 256)   container
m1 = [0, 128)   sub-view
m2 = [64, 192)  sub-view (partially overlaps m1)
```

Pairwise conflict:

```
      m0  m1  m2
  m0   -   1   1
  m1   1   -   1
  m2   1   1   -
```

All three pairs conflict.

**Today**: union-find merges all → `resourceKey = 0`. Correct here
(the over-approximation matches the actual conflict graph).

**Optimal**: atoms are `[0,64) [64,128) [128,192) [192,256)`. Touch
sets:

```
m0 covers all four
m1 covers {[0,64), [64,128)}
m2 covers {[64,128), [128,192)}
```

All three pairs intersect through `[64,128)` (m1∩m2) or via `m0`'s
universal coverage. Same conflict graph as today, but the model is
expressed correctly rather than as a coincidence.

### Example 4 — Pure disjoint pair, no container

Hypothetical: `m0 = [0, 128)`, `m1 = [128, 128)` — slot is 256
columns, two halves with no member spanning both.

Pairwise conflict:

```
      m0  m1
  m0   -   0
  m1   0   -
```

**Today**: union-find leaves them in separate classes →
`m0.resourceKey = 0`, `m1.resourceKey = 1`. Correct.

**Optimal**: two atoms `[0,128)` and `[128,128)`. `m0` covers the
first, `m1` covers the second. No intersection → no synchronization.
Same answer as today.

This is the only case where union-find gets the right answer
"naturally" — when no container exists, transitive closure is the
identity relation.

### Example 5 — Full alias

```
m0 = [0, 128)
m1 = [0, 128)
```

Pairwise conflict:

```
      m0  m1
  m0   -   1
  m1   1   -
```

**Today**: one equivalence class. Correct.
**Optimal**: one atom `[0,128)`, both members cover it. Same.

## Summary of behavior gap

| Pattern                         | Today (union-find) | Optimal (atoms) | Today over-syncs? |
|---------------------------------|--------------------|-----------------|-------------------|
| Container + disjoint sub-views  | 1 resource         | k atoms         | **yes**           |
| Meta-FA single-column sub-views | 1 resource         | k atoms         | **yes**           |
| Container + overlapping subs    | 1 resource         | k atoms         | no                |
| Disjoint no container           | 2 resources        | 2 atoms         | no                |
| Full alias                      | 1 resource         | 1 atom          | no                |
| Single member                   | 1 resource         | 1 atom          | no                |

The over-sync cases are exactly the ones where a container is present
*and* the sub-views are pairwise disjoint (or partially disjoint
between some pairs). This is the common Triton attention pattern —
a wide accumulator (the container) plus a handful of single-column
"scalar" allocations (alpha, m_i, sum, …) that don't physically
collide with each other but each lives inside the accumulator's
column range.

## What the refactor would look like

`BufferMember` switches from:

```cpp
struct BufferMember {
  // ...
  int64_t offset;
  int64_t extent;
  int64_t resourceKey;  // single equivalence-class id
};
```

to something like:

```cpp
struct BufferMember {
  // ...
  int64_t offset;
  int64_t extent;
  SmallVector<int64_t> atomKeys;  // atoms this member covers
};
```

The atom set is computed once per `BufferGroup` by sweeping all member
endpoints and partitioning `[min_offset, max_offset+max_extent)` into
maximal non-overlapping intervals.

Conflict between two touches becomes:

```cpp
bool conflicts(const AccessTouch &a, const AccessTouch &b) {
  // Set intersection on a.atomKeys / b.atomKeys (small, sorted).
  return any common atom;
}
```

Downstream consequences:

- The OWNERSHIP-DAG prints one tree per atom (replacing one tree per
  `resourceKey`). A member touching multiple atoms appears in each of
  its atoms' trees.
- RAW-SYNC edges (commit 3) are keyed by atom; cross-owner
  dependencies are derived per atom.
- The fanout/fanin combiner (commit 4) operates on per-atom edges.
- IR emission (commit 5) materializes one semaphore per atom rather
  than per equivalence class. For the container-only-overlap case
  this multiplies semaphore count by *k* atoms but eliminates all
  spurious cross-sub-view synchronization.

## When to act

The current union-find behavior is **safe** — it only causes extra
semaphore acquire/release traffic; it never breaks correctness. For
commits 1–4 (dump-only and DAG construction) the over-approximation
is invisible: the dumps and the planner produce more conservative
state than necessary but nothing observable changes in IR.

The refactor becomes meaningful at commit 5 (EMIT), where extra
semaphores translate into extra runtime cost. The natural place to
land this is between RAW-SYNC and OPT-SYNC, before the emitter
materializes any IR. Until then, the limitation is documented here
and union-find stays.

## Verified Scope Analysis — Defer-Now vs. Refactor-Now

Performed by direct code inspection of
`third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp` to
determine whether the atom-based refactor is a small localized change
(deferrable) or a structural one (block on it now).

### Sites touching `resourceKey` (16 occurrences in `InsertSemas.cpp`)

| Role                                            | Refactor impact                                              |
|-------------------------------------------------|--------------------------------------------------------------|
| `BufferMember.resourceKey: int64_t`             | type change to `SmallVector<int64_t> atomKeys`               |
| `AccessTouch.resourceKey: int64_t`              | same                                                         |
| `addTouch` copy member → touch                  | trivial (copy vector)                                        |
| `assignTmemResourceKeys` union-find             | rewrite to endpoint-sweep + per-member atom enumeration      |
| `makeLocalGroup` single-member init             | trivial                                                      |
| `dumpBackingGroupHeader` format string          | trivial format change                                        |
| `ResourceId = pair<groupId, key>` typedef       | unchanged — `key` becomes "atom" but still `int64_t`         |
| `planResource(group, key)` filter (2 sites)     | `member.resourceKey == key` → `contains(atomKeys, key)`      |
| `runOnFunction` per-resource loop               | same loop, fed from union of atom sets across members        |
| Comments and dump-format strings (4 sites)      | text only                                                    |

### What changes

- 2 struct field types: `int64_t → SmallVector<int64_t>`.
- 1 function body rewrite: `assignTmemResourceKeys` →
  `assignTmemAtoms` (sweep endpoints, partition into maximal
  non-overlapping intervals, list per member the atoms it covers).
- 1 filter primitive at 2 sites: equality (`==`) → set-membership.
- Element-type-distinct guard (currently inside union-find) becomes:
  members with different element types occupy disjoint atom-id
  namespaces. Same complexity.

### What does NOT change

- `ResourcePlan` struct shape — still per-`(groupId, key)`.
- `Planner` algorithm (`firstEventOwnerIn`, `nextEventOwnerAfter`,
  `planRegion`, in-scope filtering via `isEventInScopeForRegion`,
  WS-tagged-`scf.for` scope barrier) — operates on
  `useOwner`/`useTagSource` per resource, unchanged.
- Region-ownership walk, OWNERSHIP-DAG dump structure (still one tree
  per `(groupId, key)`).
- `ResourceId` typedef and `runOnFunction`'s per-resource loop.
- Pass entry point, gating, event collection, alias chain.

### Verified LoC estimate

~60–80 lines of changes in `InsertSemas.cpp`. No new files, no new
dependencies, no architectural shifts.

### Impact on commits 3–5

Verified against the v4 plan text:

- v4 §Physical Conflict Key already treats `resourceKey` as an opaque
  key carried alongside `(logicalGroupId, versionId)`. Plan text does
  NOT require equivalence-class semantics.
- Per-edge sync graph (commit 3), fanout/fanin combiner (commit 4),
  and emitter (commit 5) iterate over keys without caring how the keys
  were derived. The same code path works for atoms.
- The only observable difference downstream at commit 5: emit one
  `nvws.semaphore.create` per atom instead of per equivalence class.
  Emitter shape is unchanged.

### Currently-passing lit tests pinned to per-equivalence-class behavior

Zero. The `nvws.semaphore.create`/`acquire`/`release` CHECK lines in
existing tests are already failing during commits 1–4 (per the v4
plan's dump-only contract). They only start matching at commit 5, so
any model change folds into the normal commit-5 lit-test work anyway.

### Unverified risk (honest)

Commits 3–5 are not yet written. I cannot confirm by code inspection
that they will not introduce a data structure that *assumes* the key
is a single union-find label. I can confirm:

- The v4 plan text doesn't require it.
- The current code doesn't require it.
- The opaqueness of `resourceKey` in commit-1/commit-2 code paths is
  consistent with atom-keyed semantics.

Whether the implementation drifts when written is a question only the
implementation will answer.

### Verified recommendation

**Defer.** The refactor is local (~60–80 LoC, no architectural
change), the v4 plan text already supports atom-keyed semantics, no
currently-passing test pins the old behavior, and the natural place to
land it is between RAW-SYNC (commit 3) and OPT-SYNC (commit 4) — or
co-landed with commit 5 — where the extra semaphore count first
becomes observable. Doing it now versus later costs the same number
of edited lines; the difference is whether the test-output churn is
absorbed now or as part of commit 5's normal lit-test pass.
