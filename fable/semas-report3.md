# NVWS InsertSemas — Design Specification

Semaphore insertion for warp-specialized programs, rebuilt as four mechanical
stages over **one DAG object per buffer group** (one node type end to end;
groups are independent token games — §4):

```
input IR ──► ACCESS-DAG ──clone+extend──► OWNER-DAG ──clone+inject──► SYNC-DAG ──render──► output IR
```

Every stage is a pure, mechanical function of its input. No heuristics, no
guesses, no repair passes. The algorithm is the one specified in
`plans/algorithm_handoff.md` (memory pieces are tokens; a partition may touch
a piece only while holding its token; semaphores are how tokens get passed),
adapted to structured SCF IR. Acquire/release placement follows
`plans/insert-acquire-releaase-mechical-plan.md`: the DAG is authoritative,
and emission is a 1:1 rendering of DAG nodes.

---

## 1. Contract

### 1.1 Input

Post-AutoWS IR: warp-specialized `scf.for` loops (`tt.warp_specialize`,
`ttg.warp_specialize.tag = T`) whose body ops carry `ttg.partition`
annotations. The program is still sequential; partitions communicate values
through aliasing memory buffers. Allocs carry `buffer.id` (logical group =
one backing allocation) and `buffer.offset`; member extents come from the
memdesc type (TMEM: tensor-memory columns; local: the leading dim of the
memdesc shape — the memory planner's offset unit, corpus-proven).

The communicating accesses, **uniform over both memory spaces**. R means
provably load-only; **everything else is W**:

| | W | R |
|---|---|---|
| SMEM | `local_alloc` with src (acts as a store), `local_store`, `descriptor_load`, `descriptor_gather` (TMA writes into SMEM) | `local_load`; MMA A/B operand touches on SMEM memdescs |
| TMEM | `tmem_alloc` with src (acts as a store), `tmem_store`; `tc_gen5_mma` **accumulator** touch | `tmem_load`; MMA A/B operand touches on TMEM memdescs |

An op may touch several buffers with different effects. An MMA is one event
with multiple touches: its A/B **operand touches are R** — whether the
operand lives in SMEM or in TMEM — and its **accumulator touch is W**.
Nothing more is read into op semantics.

**Async tokens** threaded through TMEM accesses (`tmem_load %a[%tok]`) are
**ignored as an ordering source** — they restate the sequential program
order, which we already have — and the emitter erases them. Ordering
authority in the output is semaphores plus per-partition program order, only.

**Effective owner** of an access: `(partition, wsTag)` resolved from the
annotations; an op with no partition annotation, or with a partition but no
reachable WS tag, is **root** — a distinct owner, not partition 0.

**Conflict model**: per memory piece, N concurrent readers XOR one writer.
Cross-partition conflicts are ordered through semaphores; same-partition
order is program order and costs nothing. If several partitions only ever
read a piece between writes, they all hold it simultaneously (fan-out); a
write requires collecting the piece back from every holder (fan-in).

### 1.2 Output

Per rewritten access, the bracket:

```mlir
%tok  = nvws.semaphore.acquire %sem {ttg.partition = ...} -> !ttg.async.token
%view = nvws.semaphore.buffer  %sem, %tok ...        ; the only legal buffer view
...   = <access> %view ...
        nvws.semaphore.release %sem, %tok [#nvws.async_op<none|tc5mma|tma_load>]
```

The release payload names the source access's hardware completion mechanism
and is a table lookup on the op (`getAsyncPayload`), recorded as a DAG fact —
never recomputed. Tokens that cross a region boundary thread through
`scf.for` iter_args / `scf.if` results. Buffer groups touched by a single
owner produce no edges and are left completely untouched.

---

## 2. The DAG representation — one object, three snapshots

The central structural decision: **the same node-based DAG type is used by
every stage**. ACCESS-DAG is built from the IR; OWNER-DAG is a clone of it
extended with ENTER/EXIT nodes and owner assignments; SYNC-DAG is a clone of
that with ACQUIRE/RELEASE nodes injected into the chains. The emitter then
walks the SYNC-DAG and emits IR node by node. Each snapshot is independently
dumpable and diffable; the dump and the emitter share one traversal, so what
is printed is — by construction — exactly what is emitted. No side tables of
anchors, no maps from edges to insertion points, no re-derivation.

```cpp
struct Node {
  enum Kind {
    Func, For, If,            // structural (op = the scf op; Func/region root)
    Enter, Exit,              // virtual region brackets (op = null)
    Access,                   // load / store / alloc-with-source / descriptor_load
                              //   / descriptor_gather / mma — classified by touches
    Acquire, Release          // sync nodes, injected by the SYNC stage
  };
  Kind kind;
  Operation *op;              // IR anchor for Func/For/If/Access; null otherwise

  // Structure — identical in all three snapshots.
  Node *parent;               // enclosing Func/For/If node
  Node *prev, *next;          // program-order chain within the parent region
  SmallVector<Node*> children;// For: body chain head; If: then head [, else head]

  // ACCESS payload (Access nodes).
  Owner owner;                // (partition, wsTag) or root
  SmallVector<Touch> touches; // {member, R|W, accessValue, aliasChain};
                              // pieces are DERIVED: footprint(member)

  // OWNER payload (Enter/Exit/For/If nodes).
  DenseMap<PieceId, PieceInfo> pieceInfo; // {carried owner, effect R|W} per live piece

  // SYNC payload (Acquire/Release nodes).
  SemaId sema;                // which semaphore
  unsigned count;             // acquire pending count (releases are always 1)
  AsyncOp payload;            // release payload, from the source access
  Node *sat;                  // Release: the ONE Acquire it satisfies (forward
                              // link; an Acquire has count-many incoming)
};
```

Traversal is one recursive chain walk (`node = head; while (node) { visit;
recurse into children; node = node->next; }`). `Release → Acquire`
satisfaction links (`sat`) are the dependency cross-edges; they always point
**forward** in traversal order (§5.4). The whole structure is a DAG in the
strict sense — SCF has no back edges, and this design introduces none.

Alongside the DAG, two flat tables: the **member/piece table** (per group:
members, intervals, footprints) and the **semaphore table** (per semaphore:
id, group/pieces, initial permits, backing). Both are filled mechanically and
never edited after their stage completes.

---

## 3. Stage 1 — ACCESS-DAG

Built directly from the IR, uniformly for TMEM and local:

1. **Groups and members**: bucket allocs by `buffer.id` (fresh synthetic id
   when absent). Each member: interval `[offset, offset + extent)` in the
   space's native unit — TMEM: columns (`getTmemAllocSizes().numCols`;
   f16 packs two elements per 32-bit column, so 128x128xf16 spans 64);
   local: the leading dim of the memdesc shape, which is the planner's
   `buffer.offset` unit (corpus-proven by `insert_semas_local_buffer_reuse`:
   128x128xf16 members at offsets 0/64 must overlap while 0/256 must not —
   byte extents would merge the disjoint pair).
2. **Pieces** (the handoff doc's Phase A): collect every member's start/end
   as cut points; each interval between adjacent cuts gets a cover set
   (which members contain it); merge equal-cover neighbors; drop empty
   covers. `footprint(member) = set of pieces`. Guaranteed invariant: *two
   members overlap ⟺ their footprints intersect*. Pieces give minimal
   granularity automatically: e.g. for qk `[0,128)`, alpha `[64,65)`,
   p_acc `[0,64)` (the attention reuse group), the pieces are
   `A=[0,64){qk,p_acc}`, `B=[64,65){qk,alpha}`, `C=[65,128){qk}` — alpha and
   p_acc share no piece and never synchronize against each other, only
   through qk. A member whose offset is not statically known gets the
   conservative footprint = all pieces of its group; non-overlap is never
   silently assumed.
3. **Nodes**: one program-order walk creates `Func`/`For`/`If` structural
   nodes (only where the subtree contains accesses of the group) and one
   `Access` node per terminal access op, with its owner and per-member
   touches. `footprint(touch) = footprint(member)`;
   `footprint(node) = ∪ touches`. Alias chains
   (`memdesc_index/subview/trans/reinterpret/reshape`) are tracked on the
   touch; an unsupported alias use is a hard diagnostic.
4. **Region effect summaries**: every `For`/`If` node gets its per-piece
   map `PieceId → Effect` — presence in the map *is* the region's footprint;
   the value is the bottom-up OR of the subtree's touch effects (`W` iff any
   subtree access writes the piece). Purely touch-derived, hence computed
   here, in this stage; stage 2 adds only the *owner* half of `pieceInfo`
   and copies both halves onto the new `Enter`/`Exit` nodes. This summary
   is what makes the later stages flat: stage 2's first-toucher/next-toucher
   scans treat a region row as an ordinary toucher of exactly the pieces in
   its summary (no recursive subtree probing during owner assignment), and
   stage 3's super-node touch (rule 6) is a plain table lookup.

Dump: per group, the member/piece table plus the access tree (control-flow
shape, `R`/`W` rows with member, op name, owner; `FOR`/`IF` rows annotated
with their per-piece effects).

---

## 4. Stage 2 — OWNER-DAG

Clone the ACCESS-DAG; insert one `Enter` node at the head and one `Exit` node
at the tail of every **`For`/`If` region chain** (`Exit` sits where the
region terminator `scf.yield` is; the `Func` chain gets **neither** — no
carried owner exists at function level, so rules 4–5 never apply there); assign the **owner** half of `pieceInfo` to
`Enter`/`Exit`/`For`/`If` nodes (the **effect** half was already computed at
stage 1 on the `For`/`If` nodes — §3 item 4 — and is copied, with the
owners, onto the new `Enter`/`Exit` nodes).
One DAG **per buffer group** (groups never share a semaphore — a
`semaphore.create` lists one group's backings; an op touching two groups,
e.g. an MMA reading an SMEM group and writing a TMEM group, appears as an
Access row in *each* group's DAG and synchronizes in each game
independently — the handoff doc's emergent decomposition). Within a group's
DAG, the plan covers **all of that group's pieces at once** — a region row
carries one owner *per live piece*, not one global owner.

Owner assignment is a deterministic function of access order — no policy:

- **Loop body** (`scf.for`): per piece, carried owner := owner of the body's
  **first toucher** of that piece (a nested region counts via its own carried
  owner; the WS scope-barrier rule applies). `Enter`, `Exit`, and the `For`
  node all carry this owner. Invariant: `For == Enter == Exit` per piece.
- **`scf.if`**: per piece, branch owner := owner of the **first toucher of
  the piece in the if's subtree** (then chain first, then else chain, in
  chain order). No fallbacks: an if-row exists for a piece only when its
  summary contains it, so a first in-branch toucher always exists. This
  choice is load-bearing: it puts the handoff to/from the branch at the
  **body level**, sourced from real access rows (a cross-owner if as a
  loop body's last toucher then produces the body-level regain
  `IfRow → EXIT` that the entry acquire duplicates — token genesis is the
  standard rule, no special case), and it minimizes foreign-owner sync
  rows inside branches. The not-taken path costs one empty, unconditional
  acquire→release bracket on the owner's stream — balanced by
  construction, and the shape downstream's loop scheduler prefers anyway
  (sync outside the `scf.if`). (This also covers an if that is
  the piece's only toucher in its region — without this the owner, and with
  it the regain/entry acquire, would be undefined). Both branches get
  identical `Enter`/`Exit` owners.
  **Else rule:** every managed `scf.if` gets an else chain in the DAG —
  `Enter`+`Exit` rows — even when the IR has no else region (a **virtual
  else**; DAG nodes are ours, the IR is untouched until emission). Under
  local state seeding (§5.1 rule 5) the virtual else's game is seeded
  `Exclusive(branch owner)` with no events, so it **never carries sync
  rows** — parent-level holders (e.g. an outside reader `{3}` live across a
  then-only if) are closed at the *parent* level, path-independently, by
  the next parent-game touch; nothing about the not-taken path needs
  fixing inside the else. The else chain exists so both branch games are
  uniform objects and so token-result threading has a place to yield; at
  emission the else region is materialized iff the if carries token
  results (`replaceIfOpWithNewSignature` creates it automatically).
  Invariant: `If == then.Enter == then.Exit == else.Enter == else.Exit`.
- **Super-node rule**: in its parent's chain, a `For`/`If` node is **one
  row** whose per-piece owner is its carried owner. Parents never look
  inside; bodies never look outside. All cross-region interaction goes
  through this row (§5).
- **Per-piece effect on region rows**: every region row is qualified, per
  piece, as **R or W** — `W` iff *any* subtree access to that piece writes
  (computed at stage 1 on the `For`/`If` nodes, §3 item 4; this stage copies
  it onto `Enter`/`Exit`). The full region record is
  `pieceInfo : PieceId → {owner, effect}`. This is what lets the
  super-node participate in the walk as an ordinary R or W touch: an
  R-qualified region *joins the reader set* instead of retiring the
  version. Without it, two read-only loops on the same piece would
  serialize needlessly:
  `W m {1}; FOR{2}[R m]; FOR{3}[R m]; W m {4}` — with effects, both For
  rows are R-touches sharing {1}'s version (loops run concurrently), and
  `W{4}` fan-ins `acq(2)` from both loop rows. Mixed case:
  `op1{1}; R1{2}; FOR R2{2}; FOR R3{3}; FOR W1{3}; op2{4}` yields
  `op1 → {R1·R2 ∥ R3} → W1 → op2` — the only semaphore into W1 is
  `{2}`'s done-edge after the R2 loop (count 1; `{3}` self-orders R3
  before W1 by program order, and `op1 → W1` is transitive), so a long R2
  overlaps R3 with no race and no deadlock.
- **Scope barrier / root**: a WS-tagged `scf.for` is a scope boundary for
  owner propagation; root propagates freely; an op carrying both partition
  and an intrinsic WS tag is an ordinary owner everywhere.

### 4.1 ENTER/EXIT are the virtual first and last touchers

The load-bearing identity, per region and piece:

```
first toucher == last toucher == ENTER owner == EXIT owner   (the carried owner)
```

holds over the **augmented** chain even when the carried owner has no real
access at the head, at the tail, or at all: `Enter` and `Exit` *are* its
first and last touch. This is what makes every region a self-contained token
game (and it plays exactly the role of "rotate the loop body" in the handoff
doc — entry/exit ownership is pinned instead of ops being reordered). The
cases, enumerated:

**E1 — carried owner has both first and last real access.** No region-bracket
edges at all; the handback lands on its real access:

```
ENTER {1} · W{1} · r S0{1} · a S0{2} · R{2} · r S1{2} · a S1{1} · W'{1} · EXIT {1}
```

**E2 — carried owner has the first access but not the last** (the canonical
loop shape, §5.3): the last holder hands back to `Exit`; the acquire is
followed directly by `EXIT`:

```
ENTER {1} · W{1} · r S0{1} · a S0{2} · R{2} · r S1{2} · a S1{1} · EXIT {1}
```

**E3 — carried owner has no first access in *this* chain.** Under
first-toucher ownership this arises only for the *other* branch of an if
whose piece is touched in both branches by different partitions (the owner
is the then-chain's first toucher, so in the else chain it may have no
access); it cannot arise for loops or single-branch ifs. `Enter` then
sources the first release — `ENTER` followed immediately by the release:

```
ENTER {1} · r S0{1} · a S0{2} · R{2} · r S1{2} · a S1{1} · EXIT {1}
```

**E4 — carried owner has no access at all in the region** (it only brackets):
both brackets are virtual — E3's shape; the owner's only rows are the
region-start release and the pre-exit acquire. Fires only when the region
executes; releases and acquires stay balanced per execution.

**E5 — empty branch** (`ENTER {1} · EXIT {1}`): holders already equal the
carried owner on that path; no edges. Both branches of an `scf.if` therefore
exit with identical holder states — branch reconciliation is an invariant to
assert, never a choice to make.

**E6 — single-owner region**: no cross-owner touches, no edges; if the whole
group is single-owner, the group is left untouched entirely.

Dump: per piece, the v4-style ownership tree — region rows, `ENTER`/`EXIT`
rows, and `use {p}` rows, with `{@tag.p}` display outside the owning WS loop.

---

## 5. Stage 3 — SYNC-DAG

Clone the OWNER-DAG; run **one stateful walk** that derives handoff edges;
dedupe and group them; inject `Acquire`/`Release` nodes into the chains. One
stage, one DAG — fan-in is part of the construction, not an optimization
pass.

### 5.1 Walk state and rules

Per piece `r`: `holders(r)` — either `Exclusive(p)` or
`Shared(producer p, readers {q…})` — and per holder its most recent row.
Rows are visited in chain order; the complete rule set:

1. **W by `p` on `r`**: emit one edge `lastRow(h) → thisRow` for every
   holder `h ≠ p` (every reader if Shared, the single owner if Exclusive) —
   the fan-in. Then `holders(r) := Exclusive(p)`.
2. **R by `p` on `r`**: if `p ≠ versionProducer(r)` and `p` is not already
   a reader, emit one edge `lastRow(holder) → thisRow` (from the current
   token holder's most recent row) and add `p` to the readers — the
   fan-out. If `p == versionProducer(r)` — the partition that produced the
   current version re-reading its own data — **no edge**: `p`'s own program
   order covers it, even across region boundaries (partition-loops clones
   the region skeleton into every participating partition's stream, so a
   pre-region write and an in-region read by the same partition sit on one
   sequential stream). An already-reader or producer touch updates that
   holder's `lastRow`. **Never an edge between two readers.**
   *Reader-set invariant (the fan-out legality rule, emergent from rules
   1–2 and asserted by the verifier):* a `Shared` holder set only ever
   contains partitions whose accesses to the piece in the **current
   version** are all R; the first W by anyone — including a current
   reader — retires the set (rule 1 collects a done-edge from every
   co-holder; the writer's own prior reads are covered by its program
   order). Fan-out is therefore never "permitted" by a check — it is
   structurally impossible for a writer to co-hold:
   `W{1}; R{2}; R{2}; R{3}; R{3}; W{4}` shares `{2}∥{3}` and fan-ins
   `acq(2)` at `{4}`; `W{1}; R{2}; R{2}; R{3}; W{3}; W{4}` still overlaps
   the *reads* of `{2}` and `{3}`, but `W{3}` waits on `{2}`'s last read
   (edge `R{2}→W{3}`) — the version model: reads overlap, writes gate.
3. **Same-owner touch**: no edge; update the row. (If one row carries both
   an R and a W touch on the *same* piece — e.g. an MMA whose operand piece
   overlaps its accumulator — the row's effect for that piece is the OR:
   W wins, rule 1 applies.)
4. **EXIT row** (carried owner `c`): closes the **region's own game**: for
   every piece whose *in-body* holders ≠ `{c}`, emit `lastRow(h) → EXIT`
   per in-body cross holder. The region's local state then ends; the
   parent's state was never touched by the body. The consumer is guaranteed
   (the carried owner's continuation), so no "release into void" case
   exists.
5. **ENTER row — seeds the region's local game.** Every region body walks
   a **fresh local state**, per piece in the region's summary:
   `holders(r) := Exclusive(carried owner)`, `lastRow := Enter`,
   `lastPayload := none`, plus one imported read-only fact:
   `versionProducer(r) :=` the parent game's current version producer for
   the piece (a single partition value, known to the parent walk at the
   super-node row — like the carried-owner annotation, an input to the
   seed, not parent state). The parent's holder state **never enters the
   body**, and the body never modifies the parent's state — region
   locality at the *state* level, which is what makes the §4.1
   `ENTER == EXIT == carried owner` identity hold by construction.
   **Why `lastPayload := none` is safe (theorem, not assumption):** under
   first-toucher ownership the seed owner is the region's first toucher,
   so every in-region cross edge sources from one of the owner's *real*
   access rows (payload correct by construction) — an `Enter`-sourced
   edge can only arise in the other-branch E3 case, where the owner
   provably acquired the piece through a correctly-payloaded parent edge
   before the region, and its branch-start release follows that acquire by
   its own program order: the async completion is already ordered, `none`
   adds nothing and misses nothing. All in-body edges stay in-body.
6. **Region super-node rows — uniform for `For` and `If`.** In the
   *parent's* game, a region row is one ordinary touch per piece in its
   summary, by its carried owner, with its per-piece **effect** (§4): an
   R-qualified row obeys rule 2 (joins the readers — two read-only regions
   of the same version run concurrently), a W-qualified row obeys rule 1
   (retires the version). As an edge *destination*, the acquire is placed
   before the region op; as a *source*, the release after it — both fire
   exactly once per execution of the parent region, unconditionally, so
   counts always balance (an external producer feeding a loop can never
   create a one-release-versus-N-acquires imbalance; the body inherits the
   carrier). Conditional handoffs need no special case: sync that must
   fire iff a branch fires belongs to that branch's **local** game (rule
   5) and is balanced per path by construction; everything at the parent
   level is path-independent because both `scf.if` branch games end at the
   branch owner (asserted, never picked). The source payload of a region
   row's outgoing edge is the per-piece `lastPayload` of its own game (the
   carried owner's last real in-body access; `none` if it had none — its
   pre-EXIT acquires already ordered the in-body handback payloads); for an
   `If` row this is the **union over its two branch games** — the release
   fires on both paths, and an async element whose op didn't run commits an
   empty set and arrives immediately, so the union is safe path-wise. Walk
   order at a region row, pinned for determinism: apply the super-node
   touch to the parent state (destination edges) **first**, then recurse
   into the child game(s), then the row is available as an edge source with
   its games' payloads known — no backpatching. One mechanical exception:
   at a **WS-tagged** `For` row, a piece whose parent holder is **root** is
   adopted (state updated per the effect, **no edge**) — WS-loop entry is
   implicitly ordered after the enclosing code.

Each edge records its facts at creation: source row, destination row, source
owner, destination owner, pieces, release payload, stage/cluster anchors.
The emitter reads only these. **Payload doctrine:** the walk tracks, per
holder per piece, `(lastRow, lastPayload)`; real accesses update both; a
region super-node row sets the parent-game `lastRow` to itself and
`lastPayload` to its games' per-piece payload (union for `If` — rule 6), so
a payload always survives a row becoming virtual: a post-loop release whose
loop's last real write was an MMA carries `tc5mma` (lowering to the tcgen05
commit, arriving on MMA *completion*; a `none` release arrives at issue
time). The only `none`-payloaded releases with an async producer upstream
are `Enter`-sourced (rule 5's theorem): there, the same-partition
pre-region acquire is the ordering witness.

### 5.2 From edges to semaphores

- **Dedupe** (handoff C1): edges with identical `(srcRow, dstRow,
  srcOwner)` — the same handoff realized through several pieces — collapse
  to one. (This is what makes fine pieces free: overwriting all of qk held
  by one reader is one edge, not three.) The source **owner** is part of the
  key because re-anchoring can park several *different* holders' last-rows
  on the same ENTER node; their edges to a common destination are distinct
  synchronization obligations (one release per holder, fan-in count = number
  of holders) and must never merge. Row-only dedupe would undercount the
  fan-in — a race. **Payload merge rule:** edges collapsing under one key
  may carry different payloads (one holder's last real accesses can differ
  per piece — e.g. an MMA on one piece, a store on another, both summarized
  by the same super-node source row); the deduped edge takes the **union**,
  and the emitted release names all of them — `async_ops` is already an
  array attribute. Picking one would be a race; splitting into two releases
  by the same partition would break the per-wave distinct-partition count.
- **Group by destination** (handoff C2): one counting semaphore per
  destination row; `count = |sources|`; each source releases 1. The fan-in
  `acquire(N)` **is** this grouping. Grouping by destination is always safe;
  grouping by source is forbidden (token stealing) — a producer with N
  readers performs N releases on N semaphores, exactly as in the handoff
  doc's output (`use q; rel s1; rel s2`). There is no fan-out combine and no
  separate "optimized" DAG. If a case is ever found that this single
  SYNC-DAG cannot express, that concrete example — not a theory — is what
  would reopen the question.
- **Inject nodes**: for every semaphore, one `Release` node immediately
  **after** each source row (owner = source owner, payload = edge fact) and
  one `Acquire` node immediately **before** the destination row (owner =
  destination owner, count = group size), spliced into the chains. `Enter`
  as source ⇒ release at region start; `Exit` as destination ⇒ acquire
  before the terminator; super-node ⇒ before/after the `scf.for`/`scf.if`
  op. Each `Release` carries its single `sat` link to the one Acquire it
  feeds (the Acquire's incoming count is the group size).

### 5.3 The canonical structure — entry acquire and the ownership chain

The required shape of the SYNC-DAG for a loop whose carried owner is `p1`
(brackets mark rows that may be absent — see E1–E4):

```
a  S1(k) {p1}                  ; ENTRY ACQUIRE — executes exactly once, before the loop
FOR (WS, tag=T) {p1}
   ENTER {p1}
   [ use … {p1} ]              ; optional — ENTER may be followed directly by the release
   r  S2 {p1}
   a  S2 {p2}
   use … {p2}
   …
   use … {pn}
   r  S1 {pn}                  ; the last holder hands ownership back
   a  S1(k) {p1}               ; the carried owner regains; this token is the carrier
   [ use … {p1} ]              ; optional — the acquire may be followed directly by EXIT
   EXIT {p1}
```

Rules that produce and govern this shape:

- **Every access holds a token from a real acquire.** The view op
  `nvws.semaphore.buffer` requires one; there are no poison tokens and no
  tokenless accesses. An access without its own acquire uses the **carrier**:
  the token of its owner's most recent acquire in chain order.
- **Entry acquire (token genesis).** Per connected component of pieces (the
  per-resource token game — components are discovered, not declared): one
  additional `Acquire` of the component's *regain* semaphore — the carried
  owner's **last** acquire in the loop body's own chain — child chains
  excluded, "last" in chain order, hence deterministic (`S1` above, whether it sits
  before a real access or before `EXIT`) — is placed so it executes **exactly
  once**: immediately before the component's first access, hoisted before the
  outermost enclosing loop. Its owner is the component's first holder (root,
  if the component starts with an unannotated producer). Its token seeds the
  carrier. If the component is not loop-carried (no regain acquire exists —
  a purely acyclic chain at function level), the entry acquire gets a
  dedicated semaphore, released once immediately after the component's
  terminal access.
- **Initial permits — a static, per-create fact.** A semaphore whose *first
  event in chain order is an acquire* (exactly the entry-acquired ones) is
  created with **initial permits equal to that acquire's count** (`k`
  above); every other semaphore is created with zero. This is read off the
  DAG, asserted by the verifier, and never repaired.
  **IR realization:** `k` is DAG-level accounting, not an IR attribute. In
  IR the create carries only `is_released : i1` (`true` iff entry), and
  that suffices because the lowering is *phase-based*, not counter-based:
  `is_released = true` pre-completes the mbarrier's initial phase, so the
  entry acquire passes immediately regardless of `k`; per-phase expected
  arrive counts are re-derived by `nvws-lower-semaphore` as **Σ over the
  wave's distinct releasing partitions of `|async_ops|`** — each array
  element produces exactly one arrive, so a union-payload release counts
  once per element (this is the formula §4c hand-balancing must use).
  Corollary (verifier
  obligation): a fan-in group's sources must be distinct partitions — two
  releases by the *same* partition in one wave count once at lowering, so a
  DAG count that included both would hang.

### 5.4 Acyclicity — there is no back edge

Every `Release → Acquire` satisfaction is **forward** in traversal order:

- in-body pairs are adjacent-forward by injection (rule placement);
- the regain acquire `a S1` consumes the **same iteration's** `r S1` —
  release row precedes acquire row in the chain;
- the entry acquire consumes the create's initial permits — satisfied at
  node zero.

Balance, per full execution of a loop with `N` iterations: `S1` provides
`k·N` (releases) `+ k` (initial permits) and consumes `k·(N+1)` (entry + N
regains) — exact; every other semaphore provides and consumes equal counts
within each region execution (conditional regions fire neither side or
both). The loop recurrence itself is carried by two things only: the carried
owner's **program order** (its regain in iteration *i* precedes everything
it does in iteration *i+1*) and the carrier token threaded through
`iter_args`. No semaphore edge crosses an iteration boundary backward; the
SYNC-DAG is acyclic, as the structure of SCF demands.

### 5.5 Worked examples

**Minimal two-partition loop** (`W m0 {0}; R m0 {1}`) — shape E2:

```
a  S1(1) {0}                   ; entry — S1 created with 1 permit
FOR (WS, tag=0) {0}
   ENTER {0}
   W m0 {0}
   r  S0 {0}                   ; S0 created with 0
   a  S0 {1}
   R m0 {1}
   r  S1 {1}
   a  S1(1) {0}
   EXIT {0}
```

**Fan-out / fan-in falls out of the walk** — pieces `A={q,p}`, `B={q,v}`,
carried owner @1 for both; sequential body
`use q @1; st p @2; use p @2; st v @3; use v @3; st q @1` — shape E1 (the
regain sits before `st q`, not before `EXIT`):

```
a  SJ(2) {1}                   ; entry — SJ created with 2 permits
FOR (WS, tag=0) {A:@1, B:@1}
   ENTER {A:@1, B:@1}
   R q  {1}                    ; pieces A,B — same owner, no edge
   r  S2 {1}                   ; A: useq@1 → stp@2
   r  S3 {1}                   ; B: useq@1 → stv@3
   a  S2 {2}
   W p  {2} · R p {2}
   r  SJ {2}                   ; A: usep@2 → stq@1
   a  S3 {3}
   W v  {3} · R v {3}
   r  SJ {3}                   ; B: usev@3 → stq@1
   a  SJ(2) {1}                ; fan-in: count 2, grouped by destination
   W q  {1}                    ; pieces A,B
   EXIT {A:@1, B:@1}
```

@2 and @3 run concurrently (disjoint pieces, no edge between them); `st q`
collects both holders with one `acquire(2)`; `st q` of iteration *i* precedes
`use q` of iteration *i+1* on @1's own stream. This is byte-for-byte the
handoff doc's target output — produced by the single walk plus
group-by-destination, with no second DAG and no combine pass.

**Conditional consumption** (`W{1}; if { R{2} }; W{1}'`) — branch owner =
first in-branch toucher `{2}`; all sync is **body-level and unconditional**,
sourced from real rows:

```
W {1}
r S0 {1}                        ; real source: the W row — payload correct by construction
a S0 {2}                        ; before the if; fires every path
IF(R) {2}
   then: ENTER {2} · R {2} · EXIT {2}     ; same-owner — no in-branch sync
   else: ENTER {2} · EXIT {2}             ; empty
r S1 {2}                        ; IfRow → W' close; after the if, fires every path
a S1 {1}
W' {1}
```

`S0`/`S1` fire exactly once per execution on **both** paths (a not-taken
iteration is an empty acquire→release bracket on `{2}`) — path-independent
balance, and the sync sits outside the `scf.if`, the shape the loop
scheduler wants.

Dump: v4-style tree, region rows annotated with their per-piece effect
(e.g. `FOR(R) {2}` / `FOR(W) {3}`; the examples above omit it for brevity),
`r`/`a` rows inline at chain depth, semaphore names
`S0, S1, …` per component, pending counts on grouped acquires, entry
acquires rendered before their loop. One sync view per group — there is one
SYNC-DAG per group, no RAW/OPT split.

---

## 6. Stage 4 — EMIT-IR

The emitter is a single in-order traversal of the SYNC-DAG with **exactly one
action per node kind** — placement is the node's chain position, so the
mechanical placement law (release immediately after its source row, acquire
immediately before its destination row, ENTER ↦ region start, EXIT ↦ before
the terminator) holds by construction:

| node | action |
|---|---|
| `Func`/`For`/`If` | recurse; thread the live carrier tokens through `iter_args` / if-results (both branches yield the reconciled carrier; one slot per live component) |
| `Enter`/`Exit` | insertion-point markers only |
| `Acquire` | emit `nvws.semaphore.acquire`; its token becomes the owner's carrier; emit/cache the `nvws.semaphore.buffer` view |
| `Access` | retarget the op's memdesc operands onto the view (via the recorded alias chain); erase its original async-token plumbing |
| `Release` | emit `nvws.semaphore.release` with the node's recorded payload, consuming the owner's carrier token |

Plus the mechanical stamping rules: every sync op carries its node's owner
partition; a sync op owned by `{P}` but emitted outside the WS loop also
carries the loop's `ttg.warp_specialize.tag` (required for
`--tritongpu-partition-loops` routing and isolated-region co-location);
stage/cluster is read off the anchor access op, with the per-partition
last-seen cache for all virtual-row anchors (`Enter`/`Exit`/super-node);
`nvws.semaphore.create` ops (one per semaphore, `true` iff entry — §5.3 IR
realization) and the backing allocations are placed before the outermost WS
loop and carry no stage/cluster. A sync row owned by `{P}` placed inside a
region whose op does not list `P` extends that region op's `ttg.partition`
array (the region skeleton must exist in `P`'s stream for
`--tritongpu-partition-loops` to route the op), and the region's condition/
bounds must be available to `P` — verified, not assumed.

Phase/stage assignment of conditional acquires/releases (sync rows inside
`scf.if` branches) is downstream's responsibility — the NVWS
`AssignStagePhase` operand assigner plus the loop-scheduler workaround —
not this pass's; this pass only places the ops and stamps the attrs per
the rules above.

Hard rules, from the mechanical plan: never move a release later or an
acquire earlier; a release placed after a terminator is an invalid DAG row —
stop and report the row; if the DAG is valid and a node cannot be rendered,
the emitter is wrong; the emitter never adds edges, never moves anchors,
never re-derives owners or payloads.

Emission is bracketed by mechanical normalizations (attr/type-driven
rewrites, not decisions): a **pre-process** nukes all original TMEM async
tokens of managed groups before rendering; a **post-process** coalesces
overlapping TMEM backing allocs into subslice/reinterpret views of the
covering backing; and, last, the **loop-scheduler workaround** — the
`scf.if` acquire/release split reimplemented from
`InsertTmemSemaphore.cpp::workaroundForLoopScheduler` — reshapes `scf.if`
token plumbing for the downstream loop scheduler (critical for
`automatic-warp-specialization.mlir`); it makes no placement or
synchronization decisions.

---

## 7. Invariants and verifiers (asserted, never repaired)

Per stage, checked mechanically before the next stage runs:

- **ACCESS**: footprint invariant (overlap ⟺ intersecting footprints); every
  access op of a discovered group has a node; per-piece region **effect
  summaries** equal the OR of the subtree's touch effects (W iff any
  subtree write); unsupported alias ⇒ hard error.
- **OWNER**: `For == Enter == Exit` owner per piece;
  `If == then.Enter == then.Exit == else.Enter == else.Exit`; WS scope
  barrier respected; `Enter`/`Exit` `pieceInfo` (both halves) equals its
  region op's record — effects are copies of the stage-1 summaries, never
  recomputed.
- **SYNC**: every edge connects two rows of the same region chain and points
  forward; the reader-set invariant holds (`Shared` sets are R-only for the
  current version — rule 2); both if-branch walks exit with identical
  holder states; fan-in
  sources are same-chain siblings of their destination (count is therefore
  execution-uniform) **and are pairwise-distinct partitions** (the lowering
  counts arrives per wave — §5.3 formula); every release either sources
  from a real access row with that op's payload (union across pieces /
  branch games where merged), or is `Enter`-sourced with a same-partition
  pre-region acquire as its ordering witness (rule 5's theorem) — any other
  `none` downstream of an async producer is a hard error; per-semaphore
  balance:
  within each region execution,
  releases == acquires×count, with entry acquires covered exactly by initial
  permits; every access is preceded in chain order by an acquire of its
  component (the carrier exists — guaranteed by the entry acquire).
- **EMIT**: bijection between emitted `nvws.semaphore.{acquire,release}` ops
  and `Acquire`/`Release` nodes; every rewritten access uses a
  `semaphore.buffer` view whose token traces to a DAG acquire; no token
  crosses a region boundary except through a threaded iter_arg/result;
  stage/cluster present wherever the anchor carries one.

The dump is produced by the same traversal the emitter uses; dump/emit drift
is structurally impossible.

---

## 8. Relation to the existing pass

The current `InsertSemas` (8 files, ~7.3k LoC) is the first attempt: it
proved the pipeline shape but derives synchronization through per-resource
plans rebuilt mid-emission, two sync DAGs (raw + optimized) with
pattern-matched combine kinds, TMEM-specific deferral sets, emit-time
re-derivation of owners/payloads/anchors, and post-hoc repair passes. All of
that is structurally replaced here: one DAG object through four stages, edges
derived by one walk, fan-in by grouping, placement by node position,
emission 1:1. **No code is borrowed from the existing pass** — it is
informative only. The concepts that carry over qualitatively (discovery and
event collection, the owner/tag/payload semantics, the backing-allocation
policy, the dump formatting conventions) are re-derived from this spec; the
lit corpus is the shared verification asset. Region token threading, the
backing-stage check, and the loop-scheduler workaround are modeled on
`InsertTmemSemaphore.cpp` (read, not copied). The TMEM stage-count
(double-buffering) computation survives only as a **backing-size** input to
allocation — it sizes buffers; it never moves synchronization.

---

## 9. Bring-up plan and gates

Commit ladder (dump-only until the last commit; each commit verified against
the full set `test/NVWS/insert_semas*.mlir` +
`test/NVWS/tmem-buffer-reuse-semas.mlir` with verbatim stderr logs under
`logs/new-insert-semas/commit<N>/`):

- **Commit 0** — empty pass (registration only).
- **Commit 1** — ACCESS-DAG + dump.
- **Commit 2** — OWNER-DAG (clone + ENTER/EXIT + owners) + dump; owner
  invariants asserted.
- **Commit 3** — SYNC-DAG (walk + dedupe + group + node injection + entry
  acquires + permit table) + dump; sync invariants asserted.
- **Commit 4** — EMIT-IR (token-nuke pre-process, plan application, render,
  post-emit verifier, backing-coalescing post-process, loop-scheduler
  workaround).

Gates (the implementation plan is authoritative on commands and cwd):
`test/TritonGPU/automatic-warp-specialization.mlir` **must pass,
unmodified**; the single pytest case
`test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]` (60s
timeout — never the whole suite); `run_nvws.sh` and `run_nvws_1.sh` (60s
each). The FA kernels are the canonical deadlock regression — the
historical failure class (a fan-in count whose releasers are not all live
on every path) is excluded by the same-chain-siblings + distinct-partitions
invariants plus per-execution balance.

**Golden regeneration of `test/NVWS/insert_semas*` is deferred** to a
follow-up stage after this bring-up completes; those tests may fail
meanwhile. Expected eventual churn (intended): per-destination fan-out
semaphores, entry acquires rendered before loops, finer piece granularity
decoupling partially-overlapping members, tail acquires before `scf.yield`,
appended-only token slots.
