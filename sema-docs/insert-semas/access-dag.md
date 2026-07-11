# ACCESS-DAG: accesses, owners, and boundaries

## Purpose

ACCESS-DAG answers the questions that must be settled before synchronization
is derived:

1. Which allocations are one synchronization group?
2. Which disjoint memory pieces does each allocation cover?
3. Which operations read or write those pieces?
4. Which partition owns each access?
5. What owner and memory effect does each `for` or `if` present at its
   boundary?

This stage does not insert semaphore nodes. It builds the region tree and
ordered access chains that SYNC-DAG consumes.

The diagrams below are schematic. The implementation dumps only the completed
SYNC-DAG, not a standalone ACCESS-DAG.

## The running example

Start with the loop introduced in the overview:

The excerpt is IR-shaped text with abbreviated types. The loop attributes are
included because they establish the `{0}` and `{1}` owners:

```text
%m0 = ttg.local_alloc {buffer.id = 104} : !memdesc
scf.for ... {
  ttg.local_store %value, %m0 {ttg.partition = array<i32: 0>}
  %v = ttg.local_load %m0 {ttg.partition = array<i32: 1>}
  "use"(%v) {ttg.partition = array<i32: 1>}
} {tt.warp_specialize, ttg.partition = array<i32: 0, 1>,
   ttg.warp_specialize.tag = 0 : i32}
```

ACCESS-DAG first gives the allocation a member name and a piece footprint:

```text
group 104
  m0 = local_alloc [0, 1)

piece       interval    covered by
P0          [0, 1)      m0

member      footprint
m0          {P0}
```

It recognizes the store as `W m0 {0}` and the load as `R m0 {1}`. The loop's
boundary owner for `P0` is its first toucher, `{0}`:

```text
func root
`- for pieces{P0:W:{0}}
   |- ENTER pieces{P0:W:{0}}
   |- W m0 pieces{P0} {0}
   |- R m0 pieces{P0} {1}
   `- EXIT pieces{P0:W:{0}}
```

The `W` effect at the boundary means that some path through the loop writes
`P0`; it does not mean the boundary itself performs a write. SYNC-DAG will use
this tree to derive the two edges shown in the overview.

## What is analyzed

The pass recognizes mutable:

- `ttng.tmem_alloc`; and
- `ttg.local_alloc`.

It ignores immutable local allocations. Allocations with the same memory kind
and `buffer.id` normally form one `GroupDag`. An allocation without a
`buffer.id` receives a private synthetic group id, so unrelated anonymous
allocations are never merged accidentally.

The current implementation recursively analyzes the first block of each
structured `scf.for` or `scf.if` region, beginning at
`funcOp.getBody().front()`. It does not model additional top-level CFG blocks.

## Groups and members

A group is one synchronization unit. Each allocation in it is a `Member`:

```text
Member
  allocOp       original allocation
  type          authored memdesc type
  offset        buffer.offset, or zero
  extent        TMEM size or leading local dimension
  circularStart buffer.start, or zero
```

Grouping has three special cases:

- A circular local allocation must have `buffer.id`, `buffer.copy`, and
  `buffer.start`, and must not have `buffer.offset`. Each circular allocation
  gets its own logical group even when another circular group has the same
  physical `buffer.id`; physical sharing is validated later.
- TMEM allocations with one `buffer.id` stay together unless every member has
  `buffer.copy` and the values differ. In that mixed-depth case each member is
  a separate logical group marked for possible physical aliasing.
- Ordinary local allocations with one `buffer.id` remain one group.

Keeping logical groups separate where necessary is important: synchronization
is derived per group, while EMIT-IR may still prove that several groups share
one physical backing object.

## Pieces

Members in one group may overlap. Synchronizing each whole member would either
miss an overlap or serialize disjoint storage, so ACCESS-DAG partitions their
address span into disjoint conceptual pieces.

For example:

```text
m0 = [0, 256)
m1 = [64, 192)

address: 0          64                    192         256
         |----------|---------------------|-----------|
piece:   P0         P1                    P2
cover:   {m0}       {m0,m1}               {m0}

footprint(m0) = {P0, P1, P2}
footprint(m1) = {P1}
```

The implementation collects every member endpoint, orders the intervals, and
assigns a `PieceId` when the covering-member set changes. It persists only
each member's `PieceId` footprint; the numeric intervals are construction
facts.

An access to `m1` touches `P1`. An access to `m0` touches all three pieces:

```text
W m1  => W P1
R m0  => R P0, R P1, R P2
```

SYNC-DAG tracks memory versions independently for `P0`, `P1`, and `P2`.

<a id="pieces-must-connect"></a>

### Pieces must form one component

One `buffer.id` group must describe one connected storage component.
ACCESS-DAG rejects either:

- an uncovered gap between ordered member endpoints; or
- adjacent coverage sets that share no member.

Valid overlap:

```text
m0 [0, 128) -------+
                    +--- shared member across the boundary
m1      [64, 192) --+
```

Rejected disjoint members:

```text
m0 [0, 64)          m1 [128, 192)
   no member connects the two components
```

The rest of InsertSemas relies on “one group equals one connected
synchronization unit,” so rejecting a disconnected group is safer than
silently constructing two unrelated synchronization plans under one id.

## Recognizing accesses

For each group, ACCESS-DAG maintains a map from memdesc SSA values to:

```text
(member id, alias path from the original allocation)
```

The original allocation result starts with an empty alias path. A supported
alias operation extends the path for its result. The recognized alias names
are:

- `ttg.memdesc_index`
- `ttg.memdesc_subview`
- `ttg.memdesc_trans`
- `ttg.memdesc_reinterpret`
- `ttg.memdesc_reshape`

Each `AliasStep` stores the operation, the aliased operand number, and the
authored result type. EMIT-IR later replays that path on an acquired semaphore
buffer.

Known memory operations receive explicit effects:

| Operation | Managed operand/result | Effect |
| --- | --- | --- |
| `ttng.tmem_load` | source | `R` |
| `ttg.local_load` | source | `R` |
| `ttng.tmem_store` | destination | `W` |
| `ttg.local_store` | destination | `W` |
| `nvws.descriptor_load` / `nvws.descriptor_gather` | result | `W` |
| TMEM/local allocation with a source | allocation result | `W` |
| MMA accumulator | accumulator | `W` |
| Other managed MMA operands | operand | `R` |

For an otherwise unknown operation, a managed alias operand already present in
this group's value map is treated as a write. This conservative default
prevents an unrecognized mutation from being misclassified as a read.

Control-flow operations, function return, and `scf.yield` may not carry a
managed memdesc directly. Such flow would bypass the explicit alias and region
model, so it is rejected.

## Owners

An access owner is resolved from its partition metadata:

```text
Owner = (ttg.partition, warp-specialization tag)
```

An operation resolves to a non-root owner only when it names exactly one
partition and has a warp-specialization tag on itself or a reachable enclosing
warp-specialized loop. No partition, several partitions, or no such tag
produces the empty owner, written `root` in the documents. Owners are attached
to access nodes immediately; there is no later ownership stage.

An access may touch several members in the same group. Its `Touch` records,
for each member:

- member id;
- `R` or `W` effect;
- the exact memdesc value used by the operation;
- the authored access type; and
- the alias path.

## Ordered chains

Each function body, loop body, and `if` branch becomes a doubly linked chain.
Only memory-relevant nodes appear:

```text
Access <-> Access <-> For <-> Access <-> If <-> Access
```

Ordinary arithmetic remains in the IR but is absent from the ACCESS-DAG.
Structured regions are nested nodes, not flattened into their parent:

```text
parent chain
  A -> [For] -> B
        |
        `- child chain: ENTER -> C -> D -> EXIT
```

This separation is essential. Parent analysis sees a summarized region event;
child analysis sees only the version supplied through its own `ENTER` and the
version returned through its own `EXIT`.

## Effects

`Effect` has two values, `R` and `W`, with `W` dominating:

```text
join(R, R) = R
join(R, W) = W
join(W, R) = W
join(W, W) = W
```

An access node's `slotEffect` is the join of its touches. A region's effect for
one piece is the join across all of its branches. This is deliberately a may
summary: if either branch writes a piece, the region boundary says `W`.

The summary also records a boundary owner for each piece. Effect and owner are
separate facts:

```text
pieces{P0:W:{0}, P1:R:{2}}
       ^  ^  ^
       |  |  boundary owner
       |  aggregate effect
       piece
```

## When an access finishes

Most accesses finish at their own operation. One important case extends the
logical use of local memory:

```text
local_load -> [optional convert_layout] -> descriptor_store
```

The local buffer cannot be released when the load executes if the loaded value
is still feeding the descriptor store. `deriveCompletionAnchor` therefore
records the store as the access's `completionAnchor` when all of the following
hold:

- the path is direct or has one `convert_layout`;
- each intermediate value has exactly one user;
- load, optional conversion, and store are in the same block;
- the store follows the load; and
- the store has the same owner as the load.

Fan-out, cross-control-flow completion, reverse order, or an owner change is
rejected. SYNC-DAG uses the recorded anchor for release placement and
scheduling, so the generated release follows the real completion point.

## Regions and boundaries

ACCESS-DAG constructs owners, summaries, and boundary nodes in one recursive
walk.

For every path of each retained memory-relevant `for` or `if`, it creates:

```text
ENTER -> child accesses/regions -> EXIT
```

`ENTER` means “the version visible when this path starts.” `EXIT` means “the
version returned when this path ends.” They are structural records, not
existing MLIR operations.

### Choosing a boundary owner

The boundary owner is selected per piece:

- `for`: the first owner in the body that touches the piece;
- `if`: the last preceding owner in the parent chain if one exists; otherwise
  the first owner touching the piece in then/else order;
- a warp-specialized `for` presents `root` to its enclosing parent chain, even
  though its child boundary retains the selected partition owner.

The `if` rule preserves an already-established parent version. The loop rule
chooses the owner that must receive the iteration's input before any later
body owner can use it.

### A loop can have a different owner per piece

Consider two overlapping members:

```text
m0 footprint = {P0, P1}
m1 footprint = {P1, P2}

for {
  W m0 {0}        // first touches P0 and P1
  R m1 {1}        // first touches P2
}
```

The loop summary is:

```text
P0: W, owner {0}
P1: W, owner {0}
P2: R, owner {1}
```

The boundary is therefore per-piece, not one owner for the whole region:

```text
for pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
|- ENTER pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
|- W m0 {0}
|- R m1 {1}
`- EXIT pieces{P0:W:{0}, P1:W:{0}, P2:R:{1}}
```

### An `if` prefers the preceding owner

```text
W m0 {2}
if %cond {
  R m0 {0}
} else {
  R m0 {1}
}
```

The version entering the `if` belongs to `{2}`, so both branch boundaries use
`{2}` even though neither branch's first access does:

```text
W m0 {2}
if pieces{P0:R:{2}}
|- then
|  |- ENTER pieces{P0:R:{2}}
|  |- R m0 {0}
|  `- EXIT pieces{P0:R:{2}}
`- else
   |- ENTER pieces{P0:R:{2}}
   |- R m0 {1}
   `- EXIT pieces{P0:R:{2}}
```

SYNC-DAG can now derive each branch from the same parent source and later join
the exact token supplied by each path.

## Construction algorithm

`collectGroups` first runs once for the function, classifies allocations,
creates each group's members, and seeds that group's allocation-result alias
map. Then, for each group:

1. `buildPieces` partitions the covered span and validates connectivity.
2. `buildChainForBlock` walks the function's first block in lexical order.
3. `collectTouches` either extends an alias path or recognizes memory effects.
4. An access node receives its owner, touches, `slotEffect`, and optional
   completion anchor.
5. A nested `for` or `if` is built recursively.
6. Child effects, first owners, and last owners are summarized per piece.
7. The region selects its boundary owner per piece and receives `ENTER`/`EXIT`
   nodes for every path.
8. The summarized region is appended as one event in the parent chain.

The transient `Chain` summary contains effects and first/last owners only while
construction is in progress. The persistent model remains `Node`-based.

## Output contract

After ACCESS-DAG:

- every group is one connected piece component;
- every member has a `PieceId` footprint;
- every recognized access has exact touches, effects, owner, and alias paths;
- every access/region has a sealed aggregate `slotEffect`;
- every retained memory-relevant `for` and `if` has child chains with explicit
  `ENTER` and `EXIT`;
- every touched region piece has an effect and boundary owner; and
- descriptor-store completion, where present, is explicit.

No synchronization edge or semaphore has been chosen yet. The next document
starts from exactly these facts.

## Code map

- Group formation: `collectGroups`
- Piece partition and connectivity: `buildPieces`
- Alias and access recognition: `collectTouches`
- Descriptor-store completion: `deriveCompletionAnchor`
- Effect/owner accumulation: `appendNode`
- Recursive regions and boundaries: `buildChainForBlock`
- Per-group entry point: `buildAccessDag`
