# Per-Edge Semaphore Plan v4

## Executive Summary

```text
┌─────────────────────────────────────────────────────────────────┐
│             NVWS Insert Semaphores — v4 Pipeline                │
└─────────────────────────────────────────────────────────────────┘

      ┌────────────────────────────┐
      │  Discover backing buffers  │   (TMEM + Local, uniform)
      └─────────────┬──────────────┘
                    │
                    ▼
      ┌────────────────────────────┐
      │   Build ACCESS DAG         │   per-member touches in CFG order
      └─────────────┬──────────────┘
                    │
                    ▼
      ┌────────────────────────────┐
      │   Build OWNERSHIP DAG      │   who owns each resource in each
      └─────────────┬──────────────┘   region (function / if / for body)
                    │
                    ▼
      ┌────────────────────────────┐
      │   Build RAW-SYNC DAG       │   one edge per cross-owner
      └─────────────┬──────────────┘   transition (ready / done / handoff)
                    │
                    ▼
      ┌────────────────────────────┐
      │   Build OPT-SYNC DAG       │   combine fanout / fanin / linear
      └─────────────┬──────────────┘   chain
                    │
                    ▼
      ┌────────────────────────────┐
      │     EMIT semaphore IR      │   one uniform walk over OPT-SYNC
      │     (driven by OPT-SYNC)   │   DAG; carrier threading falls out
      └────────────────────────────┘   of OWNERSHIP DAG boundaries
```

Same flow for every backing buffer, every memory space, every CFG shape.
The five stages are sequential; each consumes the previous stage's output.
Emit is purely a renderer of the OPT-SYNC DAG. No per-pattern code.

## Implementation Plan

Stages are landed as separate commits. Between commits 1 and 4 the pass
is **dump-only** — it does not mutate IR, it prints the constructed DAG
to stderr so the user can verify each stage by eye. lit tests fail
during commits 1–4 and that is expected. At commit 5 the pass becomes a
real transform and lit takes over.

After every commit the verification protocol is:

1. The agent reads the pass output (stderr dump) for every test in the
   verification set, compares it against the corresponding `.mlir` input
   IR, and confirms the dump correctly reflects what's in the input. The
   agent reports findings before proceeding.

2. Once the agent is satisfied the output is correct, the agent runs the
   pass on every file in the verification set and saves the **verbatim
   stderr** to
   `logs/per-edge-plan/commit<N>/<basename-of-mlir>.txt`. The user can
   then open each `.mlir` input and the matching `.txt` log side-by-side
   for independent review.

The verification set:

- every file matching `test/NVWS/insert_semas*.mlir`
- plus `test/NVWS/tmem-buffer-reuse-semas.mlir`

The verbatim-capture command per file:

```bash
cd build/cmake.linux-x86_64-cpython-3.12/
ninja triton-opt

bin/triton-opt \
    test/NVWS/<file>.mlir \
    -split-input-file -allow-unregistered-dialect \
    --nvws-insert-semas 2>logs/per-edge-plan/commit<N>/<file>.txt \
    >/dev/null
```

`2>...` captures stderr (the dump); `>/dev/null` discards stdout (the
transformed IR, which equals the input during commits 0–4).

### Commit 0 — Empty pass

- Nuke the entire existing `InsertSemas.cpp` implementation.
- Keep only the pass registration boilerplate: the `NVWSInsertSemas`
  class with a `runOnOperation` that does nothing.
- Pass prints nothing, mutates nothing.
- Lit tests fail. That is expected.
- This is the clean slate the rest of the plan builds on.

### Commit 1 — Discovery + ACCESS DAG

- Add data structs, discovery (uniform over TMEM + Local), access-event
  collection, alias machinery, `assignTmemResourceKeys`.
- Pass prints: backing-buffer list, ACCESS DAG per buffer.
- No IR mutation.

### Commit 2 — + OWNERSHIP DAG

- Build per-resource `RegionOwnership` via the pure planners.
- Pass prints: ACCESS DAG + OWNERSHIP DAG per backing resource.
- No IR mutation.

### Commit 3 — + RAW-SYNC DAG

- Derive `SyncEdgeInfo` from ownership transitions.
- Pass prints: + RAW-SYNC DAG per resource (r / a markers inline).
- No IR mutation.

### Commit 4 — + OPT-SYNC DAG

- Build `SyncGroupInfo` via fanout / fanin / linear-chain combines.
- Pass prints: + OPT-SYNC DAG per resource.
- No IR mutation.

### Commit 5 — + EMIT

- One uniform recursive walk over the CFG-shaped ownership tree renders
  `nvws.semaphore.*` IR from the OPT-SYNC DAG.
- Carrier-token threading falls out of region-boundary owner mismatches.
- lit tests start running and should pass at 72/72.

After commit 5 the verification command becomes:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit \
    -v test/NVWS/
```

## Goal

Redesign `nvws-insert-semas` so semaphore insertion is based on explicit
access-DAG dependency edges, not on the current TMEM ping-pong state machine.

The pass should:

1. Build an ordered, CFG-shaped access DAG for each logical buffer group.
2. Assign structured region ownership for each backing resource used by that DAG.
3. Derive one raw synchronization edge per cross-owner dependency implied by the
   ownership plan and access DAG.
4. Optimize the synchronization graph at DAG/planner level to recover compact
   fanout/fanin semaphore shapes.
5. Emit final `nvws.semaphore.*` IR once, from the optimized sync graph.
6. Preserve existing fanout lit-test behavior where one producer feeds multiple
   consumers.

This plan applies to `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas.cpp`.

## Current Problem

The current TMEM path still has a ping-pong semaphore model:

```cpp
struct TMEMSemaphore {
  enum Kind { PING, PONG };
  ...
  // Cross-release: PING releases pong, PONG releases ping.
}
```

That model creates two semaphores for a whole logical buffer group and toggles
between them. It is not expressive enough when one physical buffer-id group has
more than two ownership dependencies.

The observed problematic pattern is not a two-owner ping-pong:

```text
@p1 write qk_0
      |
@p5 read qk_0
@p5 store alpha
@p5 store p/acc
     /              \
@p0 read alpha      @p1 mma p/acc
     \              /
      join before @p1 next qk_0 overwrite
```

If the pass reuses one ping-pong semaphore for unrelated `@p1` transitions, the
lowered pending count can make a later acquire wait for a release that depends on
that same acquire. That is a real deadlock.

## Target Model

### Uniform Access-DAG Builder

There must be one generic access-DAG construction algorithm for both TMEM and
local/SMEM.

The design split is:

```text
TMEM collector       \
                     -> normalized AccessEvent list
local/SMEM collector /

normalized AccessEvent list
  -> generic access-DAG builder
  -> structured region ownership planner
  -> generic dependency-edge scheduler over the ownership plan
  -> raw SyncEdge graph
  -> optimized SyncEdge graph
  -> final semaphore IR emission

TMEM rewrite adapter       \
                            -> retarget terminal ops/views
local/SMEM rewrite adapter /
```

Do not implement separate scheduling DAGs such as "TMEM DAG" and "local DAG".
Memory-space-specific code may only:

- collect terminal memory events into the normalized `AccessEvent` form
- track supported alias/view chains
- choose or validate backing buffer/view types
- rebuild memory-space-specific memdesc views from `nvws.semaphore.buffer`
- retarget terminal ops
- classify async payloads when the classification depends on the terminal op

All owner-domain comparison, structured region ownership assignment,
produced-version construction, read-phase grouping, dependency-edge
construction, and fanout/fanin combination must be uniform and independent of
whether the original allocation was `ttng.tmem_alloc` or `ttg.local_alloc`.

### Access Events

For each logical buffer-id group, build a deterministic event sequence/DAG.

An `AccessEvent` records the terminal operation and one or more per-member
touches. The event itself is not classified as a single read or write, because
one terminal op may touch different members with different effects.

An `AccessEvent` records:

- terminal operation
- memory space:
  - TMEM
  - local/SMEM
- logical group id
- effective owner domain:
  - `root/external`
  - `(wsTag, partitionId)`
- stage/cluster at the event site
- touches:
  - member index
  - physical conflict key / overlap class
  - effect: `read`, `write`, or `readwrite`
  - access value
  - alias/view chain needed to rebuild the memdesc from
    `nvws.semaphore.buffer`
  - async payload required by a release for this touch

Produced-version construction and dependency-edge derivation operate on touches
and their physical conflict keys, not on a single event-wide kind.

Touch illustration:

```text
@p5 writes p
@p1 mma uses p and updates acc
@p0 reads acc
```

The MMAv5 terminal op produces one `AccessEvent` with separate touches:

```text
touch p   effect=read
touch acc effect=write/readwrite
```

The `p` touch creates a dependency from `@p5` to the MMA, while the `acc` touch
creates a produced-version dependency from the MMA to later readers. The debug
dump still prints only `R` and `W`: any touch with a write effect is printed as
`W`, and read-only touches are printed as `R`.

### Physical Conflict Key

Member index is not enough to decide overwrite/reuse dependencies.

Each touch must carry a physical conflict key, also called an overlap class. This
key identifies the physical backing region whose produced version is protected by
semaphores.

The key is derived from memory-space-specific metadata:

- TMEM: allocation metadata such as `buffer.id`, `buffer.offset`,
  `buffer.copy`, shape/type, and memory-planner reuse metadata.
- local/SMEM: backing `ttg.local_alloc`, view/index metadata, shape/type, and
  memory-planner reuse metadata.

Rules:

- If two touches can overlap or reuse the same physical storage slot, they must
  have the same `resourceKey`.
- If two touches are in the same logical buffer-id group but are physically
  non-overlapping, they should have different `resourceKey`s so the planner does
  not over-synchronize them.
- Produced versions are keyed by `(logicalGroupId, resourceKey, versionId)`.
- `done` edges, `handoff` edges, and initial writable permits are per
  `resourceKey`, not blindly per member and not blindly per whole buffer-id
  group.
- If the planner cannot prove whether two touches overlap, it must use a
  conservative shared `resourceKey` or emit a hard diagnostic. It must not
  silently assume non-overlap.

Physical-key illustration:

```text
A and B are different members but reuse the same physical slot.

@p0 write A
@p1 read A
@p2 write B
```

This requires a `done` edge from `@p1` to `@p2` because `B` overwrites/reuses the
same physical resource even though the member changes:

```text
@p0 write A    resourceKey=slot0
@p1 read  A    resourceKey=slot0
@p2 write B    resourceKey=slot0
```

If `A` and `B` are provably non-overlapping members in the same `buffer.id`
group, they must use different resource keys and should not synchronize through
each other:

```text
@p0 write A    resourceKey=slot0
@p1 read  A    resourceKey=slot0
@p2 write B    resourceKey=slot1
```

The pass must not derive ordering from original TMEM async-token chains.
Generated semaphore tokens may still be threaded through loops/ifs when the
structured ownership plan requires state to cross a CFG boundary.

### Structured Region Ownership

Before deriving raw `SyncEdge`s or emitting any `nvws.semaphore.*` IR, assign
ownership for each backing resource to every structured region where that
resource is used, directly or through nested regions. This is the stage that
decides whether a handoff is outside a control-flow op or only inside a taken
region.

The planning unit is the physical backing resource:

```text
resource = (logicalGroupId, resourceKey)
```

For every resource, compute a pure in-memory ownership record over the
CFG-shaped access tree. This is a side data structure keyed by the access-tree
node / `Operation *` / region plus `(logicalGroupId, resourceKey)`. Do not write
ownership markers into IR as the source of truth.

```text
RegionOwnership:
  function region, then/else region, or loop body region
  resource
  entry owner/state
  exit owner/state
  direct use owners
  nested region owners
  required internal transitions
  required carried token slot, if ownership state crosses the region boundary
```

This ownership record deliberately separates two decisions:

- Which owner/state holds the backing resource at region entry and region exit?
- Which owner holds the backing resource for each direct or nested memory use
  inside that region?

Those answers are related by data flow, but they are not identical. Choosing a
region entry/exit owner is an optimization policy. Straight-line state between
direct uses is derived from the previous use or the previous region exit in
program order. Once region owners and use owners are fixed, semaphore placement
follows mechanically from owner changes.

Ownership assignment follows the ordered accesses for each resource:

- Preserve the incoming owner at a region boundary when only a conditional path
  needs a different owner and the continuation returns to the incoming owner.
- Move ownership before the region when all paths, the hot path, or the
  continuation after the region requires the new owner.
- For `scf.if`, assign then-region and else-region owners independently, then
  reconcile branch exit owners to the chosen post-if owner/state.
- For `scf.for`, choose the loop-body carried owner/state from the first access,
  last access, next-iteration access, and post-loop continuation.
- Different `resourceKey`s in the same logical group are assigned independently.

Planning functions should have narrow responsibilities:

```text
planRegion(region, entryOwnership) -> exitOwnership + region ownership records
planIf(ifOp, entryOwnership) -> joinOwnership + branch ownership records
planFor(forOp, entryOwnership) -> loopExitOwnership + loop ownership records
reconcileRegion(controlOp, childExitOwnerships) -> chosen exitOwnership or diagnostic
```

This planner is pure. It must not create `nvws.semaphore.create`,
`nvws.semaphore.acquire`, `nvws.semaphore.buffer`, or
`nvws.semaphore.release`, and it must not mutate IR just to record ownership.
The authoritative ownership state remains the side plan.

### Debug DAG Dumps

Add an environment-gated debug dump:

```text
NVWS_INSERT_SEMA_DUMP_DAG=1
```

The dump must be one section per logical backing buffer group. It must preserve
the old `InsertTmemSemaphore.cpp::printDag()` style: a control-flow tree in
program order, with `scf.for` / `scf.if` nesting visible. Do not dump only a
flat event list; flat lists are unreadable for real attention loops.

For each backing buffer group, print four tree views:

1. `ACCESS-DAG`: access events only, nested under the real control-flow shape.
2. `OWNERSHIP-DAG`: one tree per backing resource, with partition owner/state
   printed on each function/then/else/body region, yield, and direct memory use.
3. `RAW-SYNC-DAG`: one tree per backing resource, using the same region tree,
   with raw per-edge semaphore edges printed inline.
4. `OPT-SYNC-DAG`: one tree per backing resource, using the same region tree,
   after DAG-level fanout/fanin optimization.

The format should be compact and visually close to the old dump. Keep vertical
tree lines through loop bodies, and print owner/partition on every access,
release, and acquire. The full authoritative shapes are the end-to-end examples
below. The key ownership rule is that the ownership view is printed from the
side `RegionOwnership` plan, not from IR markers. It prints one tree per backing
resource. Each line prints the partition that owns that resource at the
corresponding function/then/else/body region, yield, or direct memory use:

```text
OWNERSHIP-DAG backing=qk_0
|- scf.for %offsetkv_y        structural
|  |- body region             entry {1} exit {1} carried
|  |  |- W qk_0  tc_gen5_mma  use {1}
|  |  |- R qk_0  tmem_load    use {5}
|  |  |- scf.if %cond         structural
|  |  |  |- then region       entry {1} exit {1}
|  |  |  |  |- R qk_0         use {5}
|  |  |  |- else region       entry {1} exit {1}
|  |  |- scf.yield            owner {1}
```

Dump requirements:

- preserve `scf.for` and `scf.if` nesting
- keep vertical tree lines through nested region bodies
- the function region is never annotated. A `func region @<name>` row
  carries no entry/exit owner — partition numbers are scoped to a
  warp-specialized for-loop and the function sits outside any such loop.
- print ownership partitions on region/use rows as `{<partition>}` or
  `root`. A row anchored inside a warp-specialized for-loop prints the
  partition alone; a row anchored outside any warp-specialized for-loop
  must include both the warp specialization tag and the partition, e.g.
  `{@0.1}`. Partition-only display outside any warp-specialized loop is
  invalid: the partition number is unanchored without the tag identifying
  which warp-specialized loop defined it.
- a regioned op (`scf.if`, `scf.for`) is included in `OWNERSHIP-DAG` only
  when at least one access to the current backing resource exists somewhere
  in its subtree — directly inside its region or at any nested depth. If no
  access exists anywhere in subtree, the regioned op and its region rows are
  omitted entirely. The same rule applies to `RAW-SYNC-DAG` and
  `OPT-SYNC-DAG`.
- owner propagation respects the warp-specialized for-loop as a scope
  barrier. An access whose `wsTag` is supplied by an enclosing
  warp-specialized `scf.for` (extrinsic) only propagates to region
  annotations *inside the body of that same `scf.for`*. It never escapes
  to ancestor regions. Two narrow exceptions:
  - **Root events** (op carries no partition annotation at all): propagate
    freely across all regions, displayed as `root`.
  - **Intrinsic-tag events** (op carries `ttg.partition` AND
    `ttg.warp_specialize.tag` directly on itself, as in
    `strip_partition_attrs_outside_ws.mlir`): the op self-names which
    WS-loop's partition system it belongs to, so its owner propagates to
    any region. Displayed with tagged form `{@<tag>.<partition>}` on rows
    outside that WS-loop, untagged on rows inside it.
- in `OWNERSHIP-DAG`, print one tree per backing resource
- in each ownership tree, print the partition owner/state on every
  then/else/body region, yield, and direct memory use line that is
  printed (subject to the no-access-no-row rule and scope-barrier rule
  above). Regions whose subtree has transitive access but no in-scope
  events display `entry root exit root`.
- print release/acquire owner partitions directly on the sync marker line
- use only `R` and `W` for memory access rows; MMAv5 accumulator updates are `W`
- use `r` for semaphore release/signal rows and `a` for semaphore acquire/wait
  rows
- do not print `ready`, `done`, `sync`, or `RW` labels
- indent semaphore marker lines as children of the nearest access/control-flow
  line, with the semaphore kind column and semaphore-name column aligned to the
  access kind/member columns: `|  |- W  qk_0 ...` followed by
  `|  |  r  S_qk ...`
- enforce the alignment mechanically in the printer:
  - access rows use the fixed prefix `|  |- <kind><two spaces><member>`
  - semaphore rows use the fixed prefix
    `|  |  <r/a><two spaces><semaphore-name>`
  - the `r`/`a` semaphore kind must be in the same column as the `R`/`W` access
    kind, and the first character of the semaphore name must be in the same
    column as the first character of the member name on access rows
  - memory member names are lowercase, while semaphore names start with an
    uppercase letter, e.g. `qk_0` vs `S_qk`
  - the operation column must be aligned across memory and semaphore rows, e.g.
    `ttng.tc_gen5_mma`, `ttng.tmem_load`, `release`, and `acquire` start in the
    same column inside a tree depth
  - do not hand-pad each row independently in a way that can drift by one
    character
- print member name/index and access kind
- print stage/cluster when present
- print sync markers exactly where the planner intends release/acquire
  placement
- keep access, ownership, raw, and optimized views in the same control-flow
  shape so differences are easy to compare

### Effective Owner

`root/external` is distinct from `(wsTag = 0, partition = 0)`.

Cross-owner means:

```text
owner(A) != owner(B)
```

Only cross-owner DAG dependencies need semaphore edges. Same-owner dependencies
are ordered by local program order or by the existing generated token in that
owner domain.

### Dependency Edges

The access DAG creates dependency facts, not partition-pair channels. After
structured region ownership is assigned, those facts become raw `SyncEdge`s for
the required owner transitions.

Required edge classes:

- ready edge: a produced version becomes readable/usable by another owner
- done edge: a reader/user completes so a later writer may overwrite/reuse the
  physical buffer
- handoff edge: a write in one owner is followed by a write in another owner
  without an intervening read fanout

The implementation should coalesce same-owner read events of the same
`(logicalGroupId, resourceKey, producedVersion)` into one reader phase before
emitting semaphore edges. This keeps one release per owner per phase and matches
pending-count lowering semantics.

## Control Flow Rules

Control flow is resolved by structured region ownership, not by the combine
subpass and not by ad hoc token threading. For each backing resource
`(logicalGroupId, resourceKey)`, assign owner records to regions and direct uses
first. Then derive semaphore transitions from differences between region
entry/exit owners and direct-use owners.

`scf.if` and `scf.for` ops are structural rows in debug dumps. They do not own a
backing resource. Their regions own the resource:

- `scf.if`: then region and else region
- `scf.for`: body region, including loop-carried entry/exit owner
- direct memory uses, printed as `use {partition}`

The placement rule is:

```text
for each backing resource, every path must reach the planned owner/state
at the next use or overwrite of that same (logicalGroupId, resourceKey)
```

A later write/readwrite to a different `resourceKey` in the same logical group
must not force reconciliation. If two members overlap or reuse the same physical
slot, they share the same `resourceKey`, and this rule applies to the shared
backing resource.

Once the `OWNERSHIP-DAG` is known, semaphore insertion is mechanical:

- a region whose entry owner differs from the previous use/region-exit owner in
  program order gets a handoff before entering that region
- a direct use whose owner differs from the previous use/region-exit owner in
  program order gets a handoff immediately before that use or use phase
- a region whose exit owner differs from its internal current owner gets a
  handoff before leaving the region
- if a planned owner/state crosses an `scf.if` or `scf.for` boundary, thread only
  the generated semaphore carrier token needed to represent that state

`AssignStagePhase.cpp` is useful only as a mechanical reference for adding
`scf.for` iter args, widening `scf.if` results, appending yields, and preserving
partition attrs. It is not the correctness authority for ownership assignment or
semaphore placement.

If a control-flow pattern cannot be represented safely, emit a hard diagnostic
before IR mutation. Do not fall back to ping-pong or original token-chain
scheduling.

## End-to-End Examples

Each example below uses the same required format:

1. input IR shape
2. one `OWNERSHIP-DAG` per backing resource
3. raw sync DAG
4. optimized sync DAG

### Example 1: Simple Fanout/Fanin

Input IR shape:

```mlir
func.func @fanout(%v0: tensor, %v1: tensor) {
  %alloc = ttg.local_alloc ... : !ttg.memdesc<...>
  ttg.local_store %v0, %alloc        // owner {1}
  %a = ttg.local_load %alloc         // owner {5}
  %b = ttg.local_load %alloc         // owner {6}
  ttg.local_store %v1, %alloc        // owner {7}
  return
}
```

Ownership DAG:

```text
OWNERSHIP-DAG backing=%alloc
|- func region @fanout                  entry {1} exit {7}
|  |- %alloc = ttg.local_alloc          backing
|  |- W %alloc  ttg.local_store         use {1}
|  |- R %alloc  ttg.local_load          use {5}
|  |- R %alloc  ttg.local_load          use {6}
|  |- W %alloc  ttg.local_store         use {7}
|  |- return
```

Raw-sync DAG:

```text
RAW-SYNC-DAG backing=%alloc
|- a S_init acquire EMPTY                {1}
|- W %alloc  ttg.local_store            {1}
|  |- r R_1_5 release FULL              {1} -> {5}
|  |- r R_1_6 release FULL              {1} -> {6}
|- a R_1_5 acquire FULL                 {5}
|- R %alloc  ttg.local_load             {5}
|  |- r D_5_7 release EMPTY             {5} -> {7}
|- a R_1_6 acquire FULL                 {6}
|- R %alloc  ttg.local_load             {6}
|  |- r D_6_7 release EMPTY             {6} -> {7}
|- a D_5_7 acquire EMPTY                {7}
|- a D_6_7 acquire EMPTY                {7}
|- W %alloc  ttg.local_store            {7}
```

Optimized-sync DAG:

```text
OPT-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store            {1}
|  |- r S_full release FULL             {1} -> {{5},{6}}
|- a S_full acquire FULL                {5}
|- R %alloc  ttg.local_load             {5}
|  |- r S_empty release EMPTY           {5} -> {7}
|- a S_full acquire FULL                {6}
|- R %alloc  ttg.local_load             {6}
|  |- r S_empty release EMPTY           {6} -> {7}
|- a S_empty acquire EMPTY pending={{5},{6}} {7}
|- W %alloc  ttg.local_store            {7}
```

### Example 2: Conditional-Only If Consumption

Input IR shape:

```mlir
func.func @if_cond_only(%cond: i1, %v0: tensor, %v1: tensor) {
  %alloc = ttg.local_alloc ... : !ttg.memdesc<...>
  ttg.local_store %v0, %alloc        // owner {1}
  scf.if %cond {
    %x = ttg.local_load %alloc       // owner {2}
  }
  ttg.local_store %v1, %alloc        // owner {1}
  return
}
```

Ownership DAG:

```text
OWNERSHIP-DAG backing=%alloc
|- func region @if_cond_only             entry {1} exit {1}
|  |- %alloc = ttg.local_alloc           backing
|  |- W %alloc  ttg.local_store          use {1}
|  |- scf.if %cond                       structural
|  |  |- then region                     entry {1} exit {1}
|  |  |  |- R %alloc  ttg.local_load     use {2}
|  |  |- else region                     entry {1} exit {1}
|  |- W %alloc  ttg.local_store          use {1}
|  |- return
```

Raw-sync DAG:

```text
RAW-SYNC-DAG backing=%alloc
|- a S_init acquire EMPTY                {1}
|- W %alloc  ttg.local_store             {1}
|- scf.if %cond                          structural
|  |- then region                        entry {1} exit {1}
|  |  |- r S_then_full release FULL      {1} -> {2}
|  |  |- a S_then_full acquire FULL      {2}
|  |  |- R %alloc  ttg.local_load        {2}
|  |  |- r S_then_empty release EMPTY    {2} -> {1}
|  |  |- a S_then_empty acquire EMPTY    {1}
|  |- else region                        entry {1} exit {1}
|- W %alloc  ttg.local_store             {1}
```

Optimized-sync DAG:

```text
OPT-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
|- scf.if %cond                          structural
|  |- then region                        entry {1} exit {1}
|  |  |- r S_then_full release FULL      {1} -> {2}
|  |  |- a S_then_full acquire FULL      {2}
|  |  |- R %alloc  ttg.local_load        {2}
|  |  |- r S_then_empty release EMPTY    {2} -> {1}
|  |  |- a S_then_empty acquire EMPTY    {1}
|  |- else region                        entry {1} exit {1}
|- W %alloc  ttg.local_store             {1}
```

### Example 3: If Consumption Continues After Join

Input IR shape:

```mlir
func.func @if_post_consume(%cond: i1, %v0: tensor, %v1: tensor) {
  %alloc = ttg.local_alloc ... : !ttg.memdesc<...>
  ttg.local_store %v0, %alloc        // owner {1}
  scf.if %cond {
    %x = ttg.local_load %alloc       // owner {2}
  }
  %y = ttg.local_load %alloc         // owner {2}
  ttg.local_store %v1, %alloc        // owner {1}
  return
}
```

Ownership DAG:

```text
OWNERSHIP-DAG backing=%alloc
|- func region @if_post_consume          entry {1} exit {1}
|  |- %alloc = ttg.local_alloc           backing
|  |- W %alloc  ttg.local_store          use {1}
|  |- scf.if %cond                       structural
|  |  |- then region                     entry {2} exit {2}
|  |  |  |- R %alloc  ttg.local_load     use {2}
|  |  |- else region                     entry {2} exit {2}
|  |- R %alloc  ttg.local_load           use {2}
|  |- W %alloc  ttg.local_store          use {1}
|  |- return
```

Raw-sync DAG:

```text
RAW-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
|  |- r S_full release FULL              {1} -> {2}
|- a S_full acquire FULL                 {2}
|- scf.if %cond                          structural
|  |- then region                        entry {2} exit {2}
|  |  |- R %alloc  ttg.local_load        {2}
|  |- else region                        entry {2} exit {2}
|- R %alloc  ttg.local_load              {2}
|  |- r S_empty release EMPTY            {2} -> {1}
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
```

Optimized-sync DAG:

```text
OPT-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
|  |- r S_full release FULL              {1} -> {2}
|- a S_full acquire FULL                 {2}
|- scf.if %cond                          structural
|  |- then region                        entry {2} exit {2}
|  |  |- R %alloc  ttg.local_load        {2}
|  |- else region                        entry {2} exit {2}
|- R %alloc  ttg.local_load              {2}
|  |- r S_empty release EMPTY            {2} -> {1}
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
```

### Example 4: For Body With Nested Conditional Use

Input IR shape:

```mlir
func.func @loop_if(%cond: i1, %n: index, %v0: tensor, %v1: tensor) {
  %alloc = ttg.local_alloc ... : !ttg.memdesc<...>
  ttg.local_store %v0, %alloc        // owner {1}
  scf.for %i = %c0 to %n step %c1 {
    ttg.local_store %v0, %alloc      // owner {1}
    scf.if %cond {
      %x = ttg.local_load %alloc     // owner {2}
    }
    ttg.local_store %v1, %alloc      // owner {1}
  }
  ttg.local_store %v1, %alloc        // owner {1}
  return
}
```

Ownership DAG:

```text
OWNERSHIP-DAG backing=%alloc
|- func region @loop_if                  entry {1} exit {1}
|  |- %alloc = ttg.local_alloc           backing
|  |- W %alloc  ttg.local_store          use {1}
|  |- scf.for %i = %c0 to %n step %c1    structural
|  |  |- body region                     entry {1} exit {1} carried
|  |  |  |- W %alloc  ttg.local_store    use {1}
|  |  |  |- scf.if %cond                 structural
|  |  |  |  |- then region               entry {1} exit {1}
|  |  |  |  |  |- R %alloc               use {2}
|  |  |  |  |- else region               entry {1} exit {1}
|  |  |  |- W %alloc  ttg.local_store    use {1}
|  |  |  |- scf.yield                    owner {1}
|  |- W %alloc  ttg.local_store          use {1}
|  |- return
```

Raw-sync DAG:

```text
RAW-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {1} exit {1} carried
|  |  |- W %alloc  ttg.local_store       {1}
|  |  |- scf.if %cond                    structural
|  |  |  |- then region                  entry {1} exit {1}
|  |  |  |  |- r S_then_full release FULL   {1} -> {2}
|  |  |  |  |- a S_then_full acquire FULL   {2}
|  |  |  |  |- R %alloc  ttg.local_load     {2}
|  |  |  |  |- r S_then_empty release EMPTY {2} -> {1}
|  |  |  |  |- a S_then_empty acquire EMPTY {1}
|  |  |  |- else region                  entry {1} exit {1}
|  |  |- W %alloc  ttg.local_store       {1}
|  |  |- scf.yield                       owner {1}
|- W %alloc  ttg.local_store             {1}
```

Optimized-sync DAG:

```text
OPT-SYNC-DAG backing=%alloc
|- a S_empty acquire EMPTY               {1}
|- W %alloc  ttg.local_store             {1}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {1} exit {1} carried
|  |  |- W %alloc  ttg.local_store       {1}
|  |  |- scf.if %cond                    structural
|  |  |  |- then region                  entry {1} exit {1}
|  |  |  |  |- r S_then_full release FULL   {1} -> {2}
|  |  |  |  |- a S_then_full acquire FULL   {2}
|  |  |  |  |- R %alloc  ttg.local_load     {2}
|  |  |  |  |- r S_then_empty release EMPTY {2} -> {1}
|  |  |  |  |- a S_then_empty acquire EMPTY {1}
|  |  |  |- else region                  entry {1} exit {1}
|  |  |- W %alloc  ttg.local_store       {1}
|  |  |- scf.yield                       owner {1}
|- W %alloc  ttg.local_store             {1}
```

### Example 5: qk/alpha/pacc Shared TMEM Group

This is the important non-ping-pong case from the attention-style buffer reuse
pattern. The example uses two backing resources inside the same logical group:

- `qk_alpha_slot`: `qk_0` and `alpha_0` reuse the same physical slot
- `acc_slot`: `acc_0` / `pacc_0` is a different physical slot

Input IR shape:

```mlir
func.func @qk_alpha_acc(%n: index) {
  scf.for %i = %c0 to %n step %c1 {
    ttng.tc_gen5_mma ... qk_0        // writes qk_0, owner {1}
    %q = ttng.tmem_load qk_0         // owner {5}
    ttng.tmem_store %a, alpha_0      // owner {5}
    ttng.tmem_store %p, acc_0        // owner {5}
    %alpha = ttng.tmem_load alpha_0  // owner {0}
    ttng.tc_gen5_mma ... acc_0       // reads/updates acc_0, owner {1}
    scf.yield
  }
  return
}
```

Ownership DAGs:

```text
OWNERSHIP-DAG backing=qk_alpha_slot
|- func region @qk_alpha_acc             entry {1} exit {1}
|  |- scf.for %i = %c0 to %n step %c1    structural
|  |  |- body region                     entry {1} exit {1} carried
|  |  |  |- W qk_0     tc_gen5_mma       use {1}
|  |  |  |- R qk_0     tmem_load         use {5}
|  |  |  |- W alpha_0  tmem_store        use {5}
|  |  |  |- R alpha_0  tmem_load         use {0}
|  |  |  |- scf.yield                    owner {1}

OWNERSHIP-DAG backing=acc_slot
|- func region @qk_alpha_acc             entry {5} exit {5}
|  |- scf.for %i = %c0 to %n step %c1    structural
|  |  |- body region                     entry {5} exit {5} carried
|  |  |  |- W acc_0    tmem_store        use {5}
|  |  |  |- W acc_0    tc_gen5_mma       use {1}
|  |  |  |- scf.yield                    owner {5}
```

Raw-sync DAG:

```text
RAW-SYNC-DAG backing=qk_alpha_slot
|- a S_alpha_done acquire EMPTY          {1}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {1} exit {1} carried
|  |  |- W qk_0      tc_gen5_mma         {1}
|  |  |  |- r S_qk_ready release FULL    {1} -> {5}
|  |  |- a S_qk_ready acquire FULL       {5}
|  |  |- R qk_0      tmem_load           {5}
|  |  |- W alpha_0   tmem_store          {5}
|  |  |  |- r S_alpha_ready release FULL {5} -> {0}
|  |  |- a S_alpha_ready acquire FULL    {0}
|  |  |- R alpha_0   tmem_load           {0}
|  |  |  |- r S_alpha_done release EMPTY {0} -> {1}
|  |  |- a S_alpha_done acquire EMPTY    {1}
|  |  |- scf.yield                       owner {1}

RAW-SYNC-DAG backing=acc_slot
|- a S_acc_empty acquire EMPTY           {5}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {5} exit {5} carried
|  |  |- W acc_0     tmem_store          {5}
|  |  |  |- r S_acc_ready release FULL   {5} -> {1}
|  |  |- a S_acc_ready acquire FULL      {1}
|  |  |- W acc_0     tc_gen5_mma         {1}
|  |  |  |- r S_acc_empty release EMPTY  {1} -> {5}
|  |  |- a S_acc_empty acquire EMPTY     {5}
|  |  |- scf.yield                       owner {5}
```

Optimized-sync DAG:

```text
OPT-SYNC-DAG backing=qk_alpha_slot
|- a S_alpha_done acquire EMPTY          {1}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {1} exit {1} carried
|  |  |- W qk_0      tc_gen5_mma         {1}
|  |  |  |- r S_qk_ready release FULL    {1} -> {5}
|  |  |- a S_qk_ready acquire FULL       {5}
|  |  |- R qk_0      tmem_load           {5}
|  |  |- W alpha_0   tmem_store          {5}
|  |  |  |- r S_alpha_ready release FULL {5} -> {0}
|  |  |- a S_alpha_ready acquire FULL    {0}
|  |  |- R alpha_0   tmem_load           {0}
|  |  |  |- r S_alpha_done release EMPTY {0} -> {1}
|  |  |- a S_alpha_done acquire EMPTY    {1}
|  |  |- scf.yield                       owner {1}

OPT-SYNC-DAG backing=acc_slot
|- a S_acc_empty acquire EMPTY           {5}
|- scf.for %i = %c0 to %n step %c1       structural
|  |- body region                        entry {5} exit {5} carried
|  |  |- W acc_0     tmem_store          {5}
|  |  |  |- r S_acc_ready release FULL   {5} -> {1}
|  |  |- a S_acc_ready acquire FULL      {1}
|  |  |- W acc_0     tc_gen5_mma         {1}
|  |  |  |- r S_acc_empty release EMPTY  {1} -> {5}
|  |  |- a S_acc_empty acquire EMPTY     {5}
|  |  |- scf.yield                       owner {5}
```

`S_acc_ready` and `S_alpha_done` both target partition `{1}`, but they protect
different backing resources and guard different operation sites. The optimized
graph must keep them separate. A combine rule keyed only by target partition is
invalid.

## Raw Per-Edge Sync Planning

### Backing Data Buffers

Per-edge sync groups must not allocate a new data buffer per edge.

For each logical buffer group:

- create or reuse the same semaphore backing data allocation policy already used
  by `insert-semas`
- all raw edge semaphores for that group reference that same backing buffer set
- the combine subpass may reduce the number of `nvws.semaphore.create` ops, but
  it must not create another data backing allocation

For TMEM, preserve the existing approved backing-buffer type policy. This plan
only replaces the ownership scheduling model; it does not authorize unrelated
changes to 1x/2x backing-buffer behavior.

For local/SMEM, `insert-semas` wraps the allocation produced by
`insert-allocas`; it must not reapply `getMultiBufferedType` or allocate a
second local backing buffer.

### Initial Writable Permit

Every semaphore-managed physical conflict class needs an initial writable/empty
permit. This is a planned `SyncGroup` whose semaphore is initially released. It
gives the first write or sourceful implicit write for a
`(logicalGroupId, resourceKey)` a carrier token for `nvws.semaphore.buffer` even
when that write has no incoming cross-owner edge.

Minimal shape:

```text
@p1 store V
@p5 read V
```

The first writer still needs this shape:

```text
@p1 acquire EMPTY
@p1 semaphore.buffer EMPTY, token
@p1 store V
@p1 release FULL

@p5 acquire FULL
@p5 semaphore.buffer FULL, token
@p5 read V
```

The initial permit is not a special IR shortcut. It is part of the same planned
sync graph as other writable/empty states:

- it is created once per semaphore-managed `(logicalGroupId, resourceKey)`
- it is initially released
- the first write/sourceful write consumes it to get a carrier token
- in loops, the next-iteration write may consume the writable/empty state
  released by prior readers instead of the original initial state
- generated permit tokens may be threaded through `scf.for` / `scf.if` when the
  structured ownership plan requires state to cross the boundary

### Carrier Token

Every rewritten memory access needs one `nvws.semaphore.buffer` token.

If an event has multiple incoming guard edges, insert all required acquires
before the event, then designate one acquired token as the carrier token for the
`nvws.semaphore.buffer` operation. The other acquire tokens are ordering guards.

After fanin combine, the common case becomes one acquire and therefore one
carrier token.

Before combine shape:

```text
@p7 acquire D_p5_p7
@p7 acquire D_p6_p7
@p7 semaphore.buffer D_p6_p7, token_from_D_p6_p7
@p7 store V2
```

After combine shape:

```text
@p7 acquire EMPTY
@p7 semaphore.buffer EMPTY, token_from_EMPTY
@p7 store V2
```

### Planned SyncEdgeInfo

After structured region ownership is assigned, record in-memory `SyncEdgeInfo` /
`SyncGroupInfo` objects for the owner transitions required by the access DAG. Do
not create or store actual `nvws.semaphore.*` ops before optimization. The final
IR is emitted once from the optimized `SyncGroup` graph.

Each planned `SyncEdgeInfo` should include:

- logical buffer group id
- resource key / overlap class
- memory space
- produced version key, including `resourceKey`
- edge kind: `ready`, `done`, or `handoff`
- source event id and target event id
- source owner and target owner
- source and target region ownership records
- pre-edge and post-edge ownership state
- planned release anchor
- planned acquire anchor
- whether the anchors are outside the control-flow op or inside a specific
  branch/body according to the ownership plan
- carrier-token choice, if this edge/group provides the carrier token
- stage/cluster metadata
- async payload
- tracked touches, including member indexes and effects
- backing data buffers

The fanout/fanin optimizer consumes planned `SyncEdgeInfo` and produces planned
`SyncGroupInfo`. Only the final `SyncGroupInfo` objects create
`nvws.semaphore.create`, `nvws.semaphore.acquire`, `nvws.semaphore.buffer`, and
`nvws.semaphore.release` ops. Debug metadata should stay in side data or debug
dump state, not in IR.

## Final Combine Subpass

Add a final DAG-level optimization step inside `InsertSemas.cpp`, before
creating `nvws.semaphore.*` ops:

```text
build normalized access tree / access DAG
assign structured region ownership
derive raw SyncEdge graph from access dependencies and ownership transitions
optimizeSyncEdgesForFanoutFanin(...)
emit final semaphore IR, including mechanical CFG token threading
run legacy IR cleanup/workarounds only:
  splitLocalConditionalTransfers(...)
  workaroundForLoopScheduler(...)
eraseUnusedTmemAllocs(...)
run post-emission ownership/CFG verifier
```

Do not rely on `LowerAref::combineSemaphores` for this. That existing combine is
later in the pipeline, skips TMEM, and combines a different shape grouped by
dominant consumer. The primary optimization must happen on planned `SyncEdge`
data before IR emission, not by rewriting already-created semaphore IR.

All scheduling, semaphore placement, structured-region ownership reconciliation,
and token threading needed for correctness must be represented in the ownership
plan and planned `SyncGroup` graph before final IR emission.
`splitLocalConditionalTransfers(...)` and
`workaroundForLoopScheduler(...)` may remain temporarily as legacy cleanup or
IR-shape workarounds only. They must not decide where semaphores go, introduce
new ownership transitions, repair an invalid `SyncGroup` graph, or otherwise
make scheduling/placement decisions after final semaphore IR has been emitted.

The final emitter may materialize:

- backing allocations
- semaphore creates
- planned acquires
- planned semaphore buffers
- planned releases
- planned CFG token threading
- memory-space-specific retargeting of terminal ops/views

The emitter must not introduce new dependency edges, move sync anchors, change
resource ownership, or make new combine decisions.

### Combine A: Ready Fanout

The snippets in the combine subsections are local edge-combine patterns only.
They are not standalone examples; each pattern inherits ownership from the
verified `OWNERSHIP-DAG` and raw sync DAG.

Merge multiple ready edges from the same producer event and same produced
version key into one readable/full semaphore.

Raw edge pattern:

```text
@p0 release R_p0_p1
@p0 release R_p0_p2

@p1 acquire R_p0_p1
@p2 acquire R_p0_p2
```

Combined edge-group pattern:

```text
@p0 release FULL

@p1 acquire FULL
@p2 acquire FULL
```

Required safety checks:

- same logical buffer group
- same resource key
- same produced version key
- same source event
- same source owner
- all target phases are read-only consumers of that produced version key
- release async payloads are identical or provably equivalent
- same backing data buffers and view types
- release placement is the same anchor or a safe common anchor immediately after
  the producer event
- no movement across `scf.if`/`scf.for` unless the structured ownership plan
  proves every path has the same resource owner/state

### Combine B: Done Fanin

Merge multiple done edges from reader phases to the same next write/overwrite
guard into one writable/empty semaphore.

Raw edge pattern:

```text
@p1 release D_p1_p0
@p2 release D_p2_p0

@p0 acquire D_p1_p0
@p0 acquire D_p2_p0
@p0 store V2
```

Combined edge-group pattern:

```text
@p1 release EMPTY
@p2 release EMPTY

@p0 acquire EMPTY     // pending count = 2
@p0 store V2
```

Required safety checks:

- same logical buffer group
- same resource key
- same produced version key being retired
- same target write/overwrite event
- same target owner
- same acquire/guard site
- each source owner contributes at most one release for this phase
- async payloads remain on the per-source release ops
- same backing data buffers and view types
- no merge across unrelated target events even if the target partition is the
  same

### Combine C: Linear Handoff Chain Preservation

For a single live physical resource with a linear ownership sequence, preserve
the compact one-`EMPTY` / one-`FULL` semaphore shape when it is safe.

This is required for compatibility with existing linear reuse tests such as
`@n_owner_alias_sequence` in `test/NVWS/tmem-buffer-reuse-semas.mlir`.

This is not the old ping-pong scheduler. The raw `SyncEdge`s are still derived
from exact access-DAG dependencies first. The combine only recognizes that the
edges form a linear chain for one `(logicalGroupId, resourceKey)` and emits the
compact semaphore shape as an optimization over those exact edges.

Raw chain shape:

```text
@p0 write A
@p0 release E0 -> @p1
@p1 acquire E0
@p1 read/write A
@p1 release E1 -> @p2
@p2 acquire E1
@p2 read/write A
@p2 release E2 -> @p0
@p0 acquire E2
@p0 write A.next
```

Compact emitted shape:

```text
@p0 acquire EMPTY
@p0 write A
@p0 release FULL

@p1 acquire FULL
@p1 read/write A
@p1 release EMPTY

@p2 acquire EMPTY
@p2 read/write A
@p2 release FULL

@p0 acquire FULL
@p0 write A.next
```

Required safety checks:

- all edges are for the same logical buffer group
- all edges are for the same `resourceKey`
- there is exactly one live produced version at a time for that resource
- the access sequence is linear in structured program order, with no fanout or
  fanin that requires pending-count semantics
- each handoff has one source owner and one target owner
- no edge in the chain crosses an unsupported `scf.if` / `scf.for`
  reconciliation point
- backing data buffers and view types are identical for the chain
- async payloads remain attached to the release generated for the corresponding
  source access

If any safety check fails, keep the exact per-edge sync groups or emit a hard
diagnostic if the uncombined per-edge form cannot be represented. Do not fall
back to target-partition-only ping-pong state.

## Plan Verification and Enablement

### Ownership Plan Verification

Before emitting semaphore IR, verify the planned ownership graph:

- every memory access that will be retargeted has exactly one owner record
- every cross-owner dependency has a planned ready/done/handoff edge or a proven
  combined sync group
- every `scf.if` join has a known owner/state for each live resource
- every `scf.for` has a known loop-carried owner/state for each live resource
- every planned CFG-threaded token has a producer on all paths
- no branch-local token is planned for use outside the branch unless it is
  yielded through the CFG op
- ready fanout groups share the same produced version and source event
- done fanin groups share the same retired produced version and target event
- no combine is keyed only by target partition
- no acquire is planned to wait on a release that is control-dependent on that
  same acquire

Failure in this verifier is a pass diagnostic, not an assertion-only condition.

### Post-Emission Verification

After emission, walk the IR and compare it back to the ownership plan:

- each planned acquire exists at the planned anchor
- each planned release exists at the planned anchor
- each planned memory access uses the planned `nvws.semaphore.buffer` result
- each `nvws.semaphore.buffer` token traces to the planned acquire
- each release token traces to the planned carrier acquire lineage
- each planned `scf.if` result / `scf.for` iter arg carries the planned resource
  owner/state
- no generated token crosses a CFG boundary except through a planned result or
  iter arg
- no branch-local generated token has an outside-region use

Run this verifier unconditionally while the new planner is being brought up. It
can later be kept behind a debug or expensive-check option if needed.

### Staged Enablement

Enable structured region ownership in small, test-backed slices:

1. Straight-line region bodies only; diagnose unsupported `scf.if` /
   loop-carried token needs.
2. Conditional-only consumption before overwrite.
3. Consumption that continues after an `if`.
4. Then-only, else-only, and branch-symmetric conditional consumption.
5. Simple loop-carried carrier token for one resource.
6. Multiple independent `resourceKey`s in the same logical group.
7. Fanout/fanin combine with ownership verifier coverage.

Do not enable a new CFG shape by falling back to ping-pong or original token
chain scheduling. Each new shape must be accepted by ownership planning,
pre-emission verification, post-emission verification, and lit coverage.

## Implementation Stages

### Stage 0: Add Regression Inputs First

Add new lit tests only. Do not modify existing lit tests.

Required new coverage:

- compact TMEM qk/alpha/pacc shared-buffer reproducer
- simple `store -> two readers -> overwrite` fanout/fanin case
- N-owner sequence:

  ```text
  @p0 write
  @p1 read
  @p2 write
  @p0 read
  ```

- conditional-only local/SMEM consumption
- post-if local/SMEM consumption
- branch-symmetric local/SMEM conditional consumption before an overwrite. Add
  both then-only and else-only variants and verify every path reaches the same
  semaphore state at the join before the next write:

  ```text
  @p0 write V
  if %cond {
    @p1 read V
  } else {
  }
  @p0 write V2
  ```

  ```text
  @p0 write V
  if %cond {
  } else {
    @p1 read V
  }
  @p0 write V2
  ```

Existing tests that must remain valid:

- `test/NVWS/insert_semaphore.mlir`
- `test/NVWS/insert_semas.mlir`
- `test/NVWS/tmem-buffer-reuse-semaphore-insertion.mlir`
- `test/NVWS/lower_semaphore.mlir`

### Stage 1: Build Shared Access-Event Model

Refactor event collection so TMEM and local/SMEM both expose the same normalized
`AccessEvent` form, then build the access DAG with one shared algorithm.

Memory-space-specific collectors remain responsible only for:

- identifying terminal access ops
- normalizing terminal ops into per-member touches, including mixed-effect ops
  such as MMAv5
- deriving each touch's physical conflict key / overlap class from allocation
  and memory-planner metadata
- tracking alias/view chains
- selecting backing data buffer types
- rebuilding memdesc views from `nvws.semaphore.buffer`
- classifying async payloads

The shared planner owns:

- generic access-DAG construction
- owner-domain comparison
- produced-version construction per physical conflict key
- read-phase coalescing
- CFG-aware access tree construction used by structured region ownership
- raw dependency facts per physical conflict key, consumed after ownership
  assignment to create `SyncEdge`s

There must not be separate TMEM-vs-local DAG construction logic after event
normalization. If the generic builder needs more information, add it to
`AccessEvent`; do not fork the DAG algorithm by memory space.

### Stage 2: Assign Structured Region Ownership

For each backing resource, assign ownership to every function region,
then/else region, and loop body region where the resource is used directly or
through nested regions.

This stage consumes the normalized access tree and produces:

- entry/exit owner-state records for each live resource at each function region,
  then/else region, and loop body region
- direct-use owners for each terminal access
- nested region owners for branch and loop bodies
- required internal owner transitions
- CFG token-threading requirements for ownership state that crosses a boundary

This is where the pass decides whether a handoff happens outside a control-flow
op or only inside the taken region. The later semaphore emitter must follow this
ownership plan exactly.

### Stage 3: Derive Raw SyncEdges and Replace TMEM Ping-Pong

Remove the `TMEMSemaphore::Kind { PING, PONG }` scheduling model.

Replace it with:

- one backing data buffer set per logical group
- raw `SyncEdge`s derived from the access DAG and structured ownership plan
- planned raw edge semaphores referencing the shared backing data buffer set
- planned acquire/release/buffer actions driven by dependency edges
- planned loop/if token threading for generated semaphore tokens required by the
  structured ownership plan

The old token-derived singleton path must not remain as a final behavior.

### Stage 4: Add Fanout/Fanin Combine Subpass

Implement `optimizeSyncEdgesForFanoutFanin` inside `InsertSemas.cpp`.

It consumes the raw `SyncEdge` graph and produces an optimized `SyncGroup` graph:

- ready fanout groups become one readable/full sync group
- done fanin groups become one writable/empty sync group

The optimizer must be conservative. If a pattern is not proven safe, leave it as
per-edge sync groups instead of merging it.

The qk/alpha/pacc example must stay unmerged where the two target-`@p1`
dependencies guard different operations.

### Stage 5: Port Local/SMEM to the Same Edge Model

Local/SMEM currently has the desired fanout shape, but the final design should
not leave it as a separate special scheduler.

Port local/SMEM to:

- emit raw per-edge dependency `SyncEdge`s
- use the same structured region ownership planner as TMEM
- run the same fanout/fanin DAG-level optimizer
- emit final semaphore IR from the optimized `SyncGroup`s
- preserve current output shape for existing fanout tests

The local path must still preserve:

- descriptor-load and descriptor-gather writes
- descriptor-store-derived shared encodings from `insert-allocas`
- scalar splat handling
- mutable producer views and immutable consumer views
- transitive consumer discovery
- release-after-last-transitive-use placement
- stage/cluster metadata

### Stage 6: Remove Obsolete Two-Owner Helpers

Delete or stop using helpers that assume two owners or "the other partition".

Obsolete logic that should not survive as scheduling authority:

- ping/pong kind toggling
- `pickOtherPartition`-style ownership recovery
- two-owner close/reconcile helpers
- target-partition-only semaphore selection
- original TMEM async-token DAG scheduling

Replacement logic must target the exact dependency edge or exact combined edge
group.

### Stage 7: Verification

Build first:

```bash
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/
ninja triton triton-opt
```

Then run lit from the same build directory:

```bash
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build//bin/llvm-lit -v test
```

No pytest is part of this plan unless explicitly requested.

## Acceptance Criteria

The plan is complete only when all of these are true:

- TMEM insertion no longer uses ping-pong scheduling.
- TMEM scheduling is driven by access-DAG dependency edges.
- Access DAG construction uses per-member touches, so mixed-effect ops can read
  one member and write another in the same terminal event.
- Structured region ownership is assigned per `(logicalGroupId, resourceKey)`
  before raw `SyncEdge` creation or final semaphore IR emission.
- Produced versions and overwrite/reuse dependencies are keyed by physical
  conflict class, so overlapping members synchronize and provably non-overlapping
  members in the same buffer-id group do not over-synchronize.
- Local/SMEM scheduling uses the same edge model or a shared wrapper around it.
- Raw cross-owner `SyncEdge`s derived from the ownership plan are optimized into
  final `SyncGroup`s before any `nvws.semaphore.*` IR is emitted.
- Existing fanout lit tests keep the compact one-`FULL`/one-`EMPTY` shape.
- Existing linear handoff reuse tests keep the compact one-`EMPTY`/one-`FULL`
  shape when the access-DAG edges form a safe single-resource linear chain.
- The qk/alpha/pacc shared-buffer case does not merge unrelated target-`@p1`
  dependencies.
- N-owner access sequences are supported when representable in structured
  program order.
- Unsupported alias/control-flow patterns or unreconciled region ownership states
  produce hard diagnostics.
- No existing lit test is modified to relax behavior without explicit approval.
- `NVWS_INSERT_SEMA_DUMP_DAG=1` dumps access/ownership/raw-sync/optimized-sync
  DAGs in old-style control-flow tree form, not as flat event lists.
