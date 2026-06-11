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
never recomputed. **The lookup applies to every real access row regardless
of R/W effect**: an MMA reading its operands is an ASYNC READER — it issues
and returns while the tensor core still streams the operand buffers, so a
release after the operand-read row must carry `tc5mma` (lowering to a
completion-gated arrive), or the producer overwrites the operands mid-MMA.
Synchronous readers (`tmem_load`, `local_load`) map to `none` through the
same table — never special-case "R rows" to `none`. Tokens that cross a region boundary thread through
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
with their per-piece effects). **Faithful-rendering rule**: the dump is the
exact rendering of the stage's DAG — a missing row means missing from the
DAG, an empty branch renders as a bare label. An `IF` row always shows
`then` (even with no access rows under it); it shows `else` iff the IR op
has an else region (the *virtual* else enters the DAG only at stage 2 and
renders there as `else (virtual)`).

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
- **`scf.if` — RULE A: the if keeps the incoming owner.** Per piece, the
  if-owner := the contribution of the **most recent toucher of the piece
  before the if in its chain** (same contribution filter as every scan:
  access row → its owner; plain region row → its record; WS-tagged For row
  → root). Fallback, only when the if is the piece's first toucher in its
  region (no incoming exists): the contribution of the first toucher
  **inside** the if's subtree (then chain first, then else). The rationale
  is load-bearing: **conditional code is sometimes skipped, and a skipped
  iteration must cost zero sync.** Keeping the incoming owner makes the if
  row a same-owner touch at the parent level (no per-iteration edges) and
  places the handoff pair **inside the branch**, firing iff taken — the
  not-taken path performs nothing. (This is why the step-7 loop-scheduler
  workaround stays load-bearing for conditional shapes, as in the old
  pass.)
  **RULE B — a recorded, contained extension (NOT implemented):** override
  the owner with the piece's next toucher **after** the if in the same
  chain, when one exists — hoisting the handoff before the if. Profitable
  only in the read-inside-then-read-after shape, where it saves one
  round-trip per TAKEN iteration; **it reduces to rule A whenever no
  post-if toucher exists**, so adopting it later changes output only in
  the shapes it was added for. Adopt only if a profiled workload shows
  that shape hot. The extension touches exactly one choke point (the
  stage-2 if-owner function) and requires right-to-left resolution of ifs
  within a chain (a forward scan may hit a later, not-yet-assigned if
  row). Downstream stages MUST consume `pieceInfo.owner` opaquely — never
  assume if-owner == incoming owner — so the override slot stays open.
  **Bracket records are restrictions, never copies of the union.** A
  branch's `Enter`/`Exit` carry only the pieces **that branch actually
  accesses** (its own chain footprint), with the **branch-local** effects;
  the owner per piece is the if-level branch owner, so owners agree
  piece-wise wherever both branches touch a piece. A branch with no
  accesses gets bare `Enter`/`Exit` — it is just there; nothing is
  invented, and at stage 3 no game exists for a piece the branch does not
  touch. Only the **if row itself** carries the union over both branches —
  its parent-facing super-node face.
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
  results (`replaceIfOpWithNewSignature` creates it automatically). The
  virtual else's brackets are always bare (its footprint is empty).
  Invariant: `If == then.Enter == then.Exit == else.Enter == else.Exit`
  **piece-wise on each branch's own footprint** (owners equal where
  present; bracket footprints are the branches' own).
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
  `W{4}` fan-ins from both loop rows plus the producer's
  redundant-but-safe edge (`acq(3)` — rule 1's direct-only skip). Mixed case:
  `op1{1}; R1{2}; FOR R2{2}; FOR R3{3}; FOR W1{3}; op2{4}` yields
  `op1 → {R1·R2 ∥ R3} → W1 → op2` — the only semaphore into W1 is
  `{2}`'s done-edge after the R2 loop (count 1; `{3}` self-orders R3
  before W1 by program order, and `op1 → W1` is transitive), so a long R2
  overlaps R3 with no race and no deadlock.
- **Scope barrier / root — the toucher-contribution rule (operational).**
  Partition owners exist only **within the WS-tagged `scf.for` that defines
  them**; outside it (the function chain, and every region that *contains*
  the WS loop) only the default-warp root stream exists. Mechanically, in
  any first-toucher scan the toucher's **contribution** is:
  - a **WS-tagged `For` row contributes `root`** — its partition system is
    sealed; its own record keeps the carried owner (the boundary's two
    faces: `scf.for (WS, tag=0) {1}` on the row itself, so the parent game
    can attribute its edges — a post-loop release is owned `{@0.1}` and
    tag-stamped — while nothing partition-valued escapes upward);
  - a plain `For`/`If` row contributes its carried owner (same scope
    continues; if its subtree's owners came from a deeper WS loop, its own
    carried owner is already `root` transitively);
  - an access row contributes its resolved owner (root when unannotated or
    outside any WS loop — root propagates freely, inside or outside; an
    **intrinsic-tag** op carrying both `ttg.partition` and
    `ttg.warp_specialize.tag` on itself self-names its partition system and
    is an ordinary owner anywhere, displayed `{@T.P}`).
  Consequence (matches the v4-era dumps): every region row and bracket
  strictly outside a WS loop is `root` unless an intrinsic-tag toucher
  decides it.

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

**E3 — carried owner has no first access in *this* chain.** Under rule A
this is the **normal conditional-consumption shape**: the branch owner is
the *incoming* owner, who typically has no access inside the branch.
`Enter` then sources the first release — `ENTER` followed immediately by
the release (carrying the seed-imported payload when the owner is the
producer):

```
ENTER {1} · r S0{1} · a S0{2} · R{2} · r S1{2} · a S1{1} · EXIT {1}
```

**E4 — carried owner has no access at all in the region** (it only brackets):
both brackets are virtual — E3's shape; the owner's only rows are the
region-start release and the pre-exit acquire. Fires only when the region
executes; releases and acquires stay balanced per execution.

**E5 — empty branch** (bare `ENTER · EXIT`, no piece records — a branch
that accesses nothing is just there, nothing invented): no game is seeded,
no edges. Both branches of an `scf.if` therefore
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

0. **First touch of an unheld piece** (function-level games start empty;
   region games are seeded by rule 5): the toucher initializes
   `Exclusive(toucher)` — **no edge** (there is no prior holder to hand
   off from; the entry acquire covers the permit).

1. **W by `p` on `r`**: emit one edge `lastRow(h) → thisRow` for every
   co-holder `h ≠ p` — readers AND the producer — **minus holders whose
   `syncedBehind` already contains `p`** (the §5.2 transitive-sync skip;
   it is deliberately DIRECT-only — full transitive closure is not
   tracked, so with an all-reader set the producer's edge may survive as
   redundant-but-safe). Readers-only sourcing would be minimal there but
   UNSOUND for producer-re-read shapes (`W{1}; R{2}; R{1}; W{3}`: the
   producer's re-read needs WAR protection too). This is the fan-in.
   Then `holders(r) := Exclusive(p)`.
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
   `acq(3)` at `{4}` — readers `{2}`,`{3}` plus the producer `{1}`, whose
   edge is redundant (both readers dominate it) but kept by the
   direct-only skip; `W{1}; R{2}; R{2}; R{3}; W{3}; W{4}` still overlaps
   the *reads* of `{2}` and `{3}`, but `W{3}` waits on `{2}`'s last read
   (edge `R{2}→W{3}`) — the version model: reads overlap, writes gate.
3. **Same-owner touch**: no edge; update the row. (If one row carries both
   an R and a W touch on the *same* piece — e.g. an MMA whose operand piece
   overlaps its accumulator — the row's effect for that piece is the OR:
   W wins, rule 1 applies.)
4. **EXIT row** (carried owner `c`): closes the **region's own game**: for
   every piece whose *in-body* holders ≠ `{c}`, emit `lastRow(h) → EXIT`
   per in-body cross holder — with two qualifiers:
   - **transitive-sync skip** (§5.2 bullet): a holder whose `syncedBehind`
     already contains `c` is NOT closed — `c`'s own acquire since that
     holder's last row already ordered it behind the holder's work;
   - **no drains**: the close is emitted only when it is load-bearing —
     the chain sits inside some loop (the recurrence reaches back every
     iteration), or the piece is touched again later in an ancestor chain
     (the re-anchored owner becomes the future release's ordering
     witness). A close with neither is a pure drain and is never
     synthesized (the `no_loop_exit_drain` golden; e.g. an if at function
     level whose buffer dies with the branch closes nothing).
   The region's local state then ends; the
   parent's state was never touched by the body. The consumer is guaranteed
   (the carried owner's continuation) whenever a close IS emitted, so no
   "release into void" case exists.

   **WAVE LOCALITY (USER RULING 10jun26 — run_nvws root cause).** The
   carrier token is partition-local: an acquire, the buffer views drawn
   from its token, and the releases consuming that token are ALWAYS one
   partition; a chain decomposes into bracketed waves (acquire by Q,
   accesses by Q, releases by Q). ALL edge-elision rules above (the
   transitive-sync skip, same-owner no-edge, reader-rejoin no-edge) are
   valid only while the toucher's wave is OPEN — the toucher still holds
   the carrier. A touch by Q while the carrier belongs to P != Q MUST
   take an edge (a fresh acquire by Q) even when Q is a co-holder or
   transitively synchronized: ordering soundness is not sufficient — the
   emitted single-token protocol cannot express a touch without a held
   token (meta-FA p-write after the alpha handoff: ordering-sound,
   token-unsound, destroyed-view crash in partition-loops). Enforced by
   two hard-error verifiers: the stage-3 chain verifier (every
   Access/Release row's owner equals its component's current carrier
   owner, tracked through Acquire rows and region seeds) and the emitter
   post-emit subpass (every `nvws.semaphore.buffer`/`release` token
   operand traces — through loop iter_args and if-results — to acquires
   of the op's own partition; a buffer's views are consumed only by ops
   of the view's partition; root-stamped entry acquires are the one
   sanctioned seed exemption).

   **FORWARD REGAIN PLACEMENT (USER RULING 10jun26, second instance of
   the class):** only ONE token is carried into the following traversal
   through the iter_args slot — the chain's first wave owner's (the
   yielded carrier; its pre-loop entry + trailing regain are the section
   5.3 canonical structure). Every OTHER carried owner's EXIT-close
   regain anchors at the START of its own wave inside the body
   (immediately before that partition's first touch row). This is the
   spec's own initial-permit rule: the semaphore's first chain event is
   then an acquire, so it is created with an initial permit (released
   state) and each release forward-satisfies the NEXT acquire occurrence
   in chain order — no pre-loop entry instance, no iter_arg slot, no
   token threading. Single-owner bodies reduce to the canonical
   structure unchanged. Evidence: local_buffer_reuse's three-partition
   serialized chain yielded only the LAST trailing regain's token ({2})
   while the following traversal's first wave ({0}) consumed it — caught
   by the post-emit verifier; the stage-3 verifier additionally checks,
   for every loop-body chain, that the final carrier owner equals the
   first wave owner (the carried token's consumer).

   **TRANSITIVE REDUCTION (USER RULING 10jun26 — pay-for-play).** After
   the walk, before dedupe/grouping, a per-chain reduction drops edges
   whose ordering is already implied: sweep rows in order keeping a sync
   vector per partition (advanced by its own program order, and on every
   KEPT acquire inheriting the source partition's vector AS OF the release
   row — snapshots, not current state); an Access-row edge P@s -> Q@d is
   dropped only when Q's vector at d already covers s. Scope: both
   endpoints in the same chain (If bodies are separate chains, so the
   historical conditional-scope hang class stays direct-only); region-row
   endpoints are never dropped; single-traversal facts only (traversal
   closure is a separate, later phase). Semantics are unchanged by construction —
   every drop has a witness chain of kept hard waits plus program order.
   **Closure verifier (hard error):** independently re-derives the
   happens-before closure from the FINAL edge set and re-checks EVERY
   originally generated edge, dropped or kept; an uncovered dropped edge
   fails analysis naming both rows. Under-synchronization cannot ship.
   **Phase B — traversal closure:** for loop-body chains the sweep
   continues over a SECOND TRAVERSAL of the chain — strictly forward:
   each partition's program order is sequential through repeated
   traversals, so its sync vector carries from the end of one traversal
   into the next, and kept forward satisfactions re-apply at their rows.
   An EXIT-close edge P@s -> Q drops iff, at Q's first touch of each
   closed piece in the following traversal, (a) coverage of s holds and
   (b) Q's wave is already opened there by a kept in-body acquire. The
   carrier close (acquirer = first wave owner, the yielded final) is
   NEVER dropped. The closure verifier re-checks all Phase-B drops under
   the same forward two-traversal propagation. The serialized chain then
   lands at its minimal form (local_n_owner: four semaphores).
5. **ENTER row — seeds the region's local game.** Every region body walks
   a **fresh local state**, per piece in the **chain's own footprint** (a
   branch that does not touch a piece has no game for it — nothing seeded,
   nothing invented):
   `holders(r) := Exclusive(carried owner)`, `lastRow := Enter`,
   `lastPayload := none`, plus one imported read-only fact:
   `versionProducer(r) :=` the parent game's current version producer for
   the piece (a single partition value, known to the parent walk at the
   super-node row — like the carried-owner annotation, an input to the
   seed, not parent state). The parent's holder state **never enters the
   body**, and the body never modifies the parent's state — region
   locality at the *state* level, which is what makes the §4.1
   `ENTER == EXIT == carried owner` identity hold by construction.
   **Payload seed — an import, not `none` (load-bearing under rule A):**
   `lastPayload[piece] :=` the carried owner's **parent-game**
   `lastPayload` for the piece when the carried owner is the parent game's
   current producer — both producer identity and payload taken from the
   **pre-touch snapshot** (captured just before the region row's own
   super-node touch transitions the parent game); `none` otherwise. Rule A
   makes the producer-brackets-
   its-own-branch shape the *normal* conditional-consumption case (e.g.
   `W mma {1}; if{ R {0} }` — if-owner `{1}` = the producer), and there
   the branch's `Enter`-sourced release MUST carry the producer's async
   payload (`tc5mma`) — a `none` release would let the consumer read
   mid-MMA, with no intervening acquire to witness completion. The `none`
   case stays safe by transitivity: a non-producer owner acquired the
   piece through a correctly-payloaded parent edge before the region, and
   its branch-start release follows that acquire by its own program
   order. All in-body edges stay in-body.
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
are `Enter`-sourced by a **non-producer** owner (rule 5's import seeds the
producer's payload; for non-producers the same-partition pre-region
acquire is the ordering witness).

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
  **Same-owner collapse (second pass):** edges with the same destination
  and the same source OWNER but **different source rows** (one partition
  holding different pieces at different rows — multi-piece games make this
  routine, e.g. meta_fa's qk overwrite closing one piece held at a store
  row and another held at a load row) collapse to the **latest source row**
  in chain order, payloads/pieces unioned: a partition's later release
  subsumes its earlier one (same instruction stream — an arrive after the
  later row implies the earlier row's work is complete, and `async_ops`
  counting covers the earlier async op at the later site). After both
  passes a destination's sources are pairwise-distinct partitions by
  construction — the §5.3 count formula's precondition, kept as a verifier
  assert.
- **Transitive-sync skip (all edge-emitting rules — 1, 2, and 4).** Per
  holder, the walk records which partitions have taken an edge FROM it
  since its `lastRow` (`syncedBehind`; cleared whenever `lastRow` moves).
  An edge whose destination partition is already in the source holder's
  set is **omitted**: that partition's acquire already happened after the
  holder's work, so the obligation is discharged transitively (its own
  program order carries it forward). This recovers the old token-transfer
  model's minimality inside the N-readers model — e.g.
  `live_tag_source_after_prior_loop_threading`: the post-loop reader's
  acquire discharges the inner-loop producer, so the outer EXIT emits **no
  close** (the spurious extra semaphore the unrefined rule produced), and
  the regain search then lands on the post-loop acquire — exactly the old
  pass's structure.
- **Group by destination** (handoff C2): one counting semaphore per
  **(destination row, destination owner)** pair — the owner is part of the
  key (old M3 identity): an EXIT row of a multi-piece chain can close
  pieces with different carried owners, and each owner class is its own
  phase-tracked waiter; `count = |sources|`; each source releases 1. The
  fan-in `acquire(N)` **is** this grouping.
  **Uniform pending count + release multiplicity.** A semaphore's pending
  count is a per-semaphore CONSTANT: every acquire site carries the same
  count, and every acquire cycle must receive exactly that many arrives.
  When several destination groups legally share one semaphore (the For-row
  unification below, or any future reuse), a group with fewer sources
  scales its releases: each `Release` node carries an arrive multiplicity
  (default 1, rendered `r S<k>(n)` only when n > 1), and per group
  Σ(multiplicities) == the semaphore's count. A single-source group
  against count k gets one release with multiplicity k; a group whose
  source total cannot meet the count exactly is a hard diagnostic, never a
  silent repair. Verifier: all acquire sites of one semaphore have equal
  counts; per group the multiplicities sum to the count. **For-row destinations unify with
  the loop's regain**: an edge whose destination row is a `For` row is the
  *entry instance of that loop's regain* (the edge-into-For-row-as-permit rule
  applied at the nested loop) — iteration 0 is fed by the outside release,
  iterations 1..N by the in-loop release; same acquirer class, M3-clean.
  Grouping therefore merges a For-row destination group into the loop's
  in-body regain group (the LAST destination group in the body's own
  chain with the same component and the same acquiring owner); the
  before-loop acquire keeps its own placement and count but shares the
  semaphore — the old pass's unified `S0`. No in-body match (e.g. the
  regain lives in a conditional branch) → a separate semaphore: sound,
  one extra. Grouping by destination is always safe;
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
a  S1(k) root                  ; ENTRY ACQUIRE — executes exactly once, before the
                               ; loop, in the root region (carrier inherit; the
                               ; semaphore's inheritStamp records {p1} for emission)
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
- **Entry acquire (token genesis) — root-owned, carrier inherit.** Per
  connected component of pieces (the per-resource token game — components
  are discovered, not declared): one additional `Acquire` of the
  component's *regain* semaphore — the **last** acquire of the component in
  the loop body's chain, where the search descends the body's **entire
  subtree** — if-branch chains (a conditional handback is a valid regain:
  the permit model is phase-based and a skipped iteration leaves the
  permit untouched) AND nested For bodies (an inner-loop regain is a valid
  seed — the entry instance fires once on the initial permit while the
  regain instance fires per inner iteration; differing acquire frequencies
  on one semaphore are exactly the For-row-unification situation, and the
  old pass's SEMA-UNION seeded nested-loop-carried buffers this way);
  "last" in chain order, hence deterministic — placed so it
  executes **exactly once**: immediately before the component's first row
  of its **placement chain** (the innermost chain reachable from the top by
  descending through single-involving-row `scf.if` branches — an if branch
  executes at most once; never through a For row — so the entry stays with
  the create and fires only on the path that uses the buffer). If no
  regain exists (a purely acyclic chain), the entry acquires a dedicated
  semaphore, released once immediately after the component's terminal row
  at the placement-chain level, **owned by that terminal row's record
  owner** (an access row's owner; a region row's pieceInfo owner for the
  component's first piece), payload `none` (no waiter exists — the permit
  merely returns). The carrier crosses skipped iterations
  unchanged and re-enters taken branches via the if's token results (the
  crossing rule covers this: the if contains acquires, so it gets a slot).
  **The entry row is owned by ROOT** — it executes in the root region
  (function level / pre-loop), and root passes the permit to the first
  in-loop holder by **carrier inherit**: no release/acquire edge for
  `root -> first holder`; the entry token simply seeds the carrier (v4's
  rule, preserved). The first-holder fact is NOT lost: it is recorded on
  the semaphore as **`inheritStamp`** — the component's **first access
  owner** (chain order, recursive; root when the component starts with an
  unannotated access). This matches the old pass's recorded seed acquirers
  across the whole corpus: root for root-seeded accumulators, the producer
  partition for operand buffers, and the in-loop first toucher for
  branch-local buffers (whose placement chain's ENTER owner is root — the
  ENTER owner is NOT the inherit fact). **Emission (USER RULING 10jun26 — root-outside rule, PARKED: see
  fable/attr-less-acquire-release-handoff.md; current emission still
  stamps inheritStamp): entry acquires are emitted ATTR-LESS, always.** They
  execute in the root block (partition-loops leaves unannotated ops in
  place); the one-time wait passes off the initial permit, so no phase
  needs communicating — an entry acquire is a phase SOURCE (a static
  constant), never a consumer. `inheritStamp` remains a recorded analysis
  fact (the component's first access owner, printed in SEMAS), but it is
  no longer an emission stamp. This supersedes the earlier
  replicate-the-previous-pass stamping ({p2, tag 0} on operand entries).
  **M3 acquirer-class criterion (verifier, hard error):** for every
  semaphore, the set of acquiring owners may contain **at most one
  concrete partition**; `root` is additionally allowed (the inherit case).
  `{root, p}` is valid; two distinct partitions — with or without root —
  are not expressible as one phase-tracked semaphore.
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

**Conditional consumption** (`W{1}; if { R{2} }; W{1}'`) — if-owner =
**incoming owner `{1}`** (rule A); all sync is **in-branch and
conditional** — a skipped iteration performs no sync at all:

```
W {1}
IF(R) {1}                       ; same-owner touch at the parent level — no edges
   then: ENTER {1}
         r S0 {1}               ; Enter-sourced; payload = seed import (W's payload)
         a S0 {2} · R {2}
         r S1 {2} · a S1 {1}    ; handback before the branch exits
         EXIT {1}
   else: ENTER · EXIT           ; bare — not taken ⇒ ZERO sync
W' {1}                          ; same owner — no edge, carrier inherit
```

`S0`/`S1` fire **iff the branch executes** — balanced per path. This is
the rule-A payoff: conditional consumption costs sync only when the
condition is true (cf. the pmatmul epilogue: `tmem_load {0}` on the rare
last-K iteration costs the hot path nothing).

Dump: v4-style tree, region rows annotated with their per-piece effect
(e.g. `FOR(R) {2}` / `FOR(W) {3}`; the examples above omit it for brevity),
`r`/`a` rows inline at chain depth, semaphore names
`S0, S1, …` per group in creation order (`E<k>` for dedicated entry
semaphores), pending counts on grouped acquires, entry
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
| `Func`/`For`/`If` | recurse; thread the live carrier tokens through `iter_args` / if-results — one slot per live component; the slot's owner and its final-carrier row are **stage-3 ThreadingPlan facts** (then/body yields the recorded final carrier's token; else/skip yields the incoming carrier unchanged — an SSA pass-through, which is why bare else brackets need no record). **Liveness (If rows)**: an If crossing exists only if the carrier is consumed after the If — a later row for the component in the enclosing chain, or the enclosing chain is a For body (recurrence), or, recursively, the enclosing If branch is itself live. A component whose last activity is inside the branch yields nothing (gate-2 evidence 10jun26: threading a dead token through a non-WS guard if trips AssignStagePhase's hasPartition assert). For rows always cross (recurrence) |
| `Enter`/`Exit` | insertion-point markers only |
| `Acquire` | emit `nvws.semaphore.acquire`; its token becomes the owner's carrier. Buffer VIEWS are NOT emitted here: `nvws.semaphore.buffer` is materialized lazily at each consuming access, in that access's region, stamped with that access's owner and stage/cluster; the view cache clears at every acquire and at every region boundary (a carried token gets a fresh view per region); one buffer op yields all member views of a multi-member semaphore |
| `Access` | retarget the op's memdesc operands onto the view (via the recorded alias chain); erase its original async-token plumbing. **Row independence (gate-2 evidence 10jun26):** a sourceful alloc's replacement RAUW must EXCLUDE the group's other access-row ops — each row retargets itself with its own owner's view (else a reader is left on the writer-partition's view: a cross-partition SSA edge partition-loops rejects); the original alloc's erase is deferred to final cleanup, after every row has retargeted |
| *(post-nuke)* | the token nuke leaves dead token-typed **signature slots** (a `scf.for` iter_arg whose region arg and result are both unused; a `scf.if` result that is unused); these are erased to a fixpoint before any semaphore IR is emitted — dropping the matching init/yield operands and `ttg.partition.outputs` entries. Gate-1 evidence (automatic-warp-specialization.mlir): surviving poison-husk slots change region signatures and break the downstream loop scheduler |
| `Release` | emit `nvws.semaphore.release` with the node's recorded payload, consuming the owner's carrier token |

Plus the mechanical stamping rules: INSIDE the WS loop every sync op
carries exactly its node's owner partition (mandatory — partition-loops
routes child ops by it). OUTSIDE the WS loop (USER RULING 10jun26 — root-outside rule, PARKED
pending LowerAref tolerance, see
fable/attr-less-acquire-release-handoff.md; current emission stamps
{P}+tag for every concrete owner): the default is ROOT, attr-less. Annotation `{P}` +
`ttg.warp_specialize.tag` is emitted only when P is NON-ZERO — i.e. the
op consumes a token/phase from a non-zero partition's chain and must be
routed into that warp-group region so its stage/phase SSA chain stays
local (unannotated it would force the phase to be exported into
partition 0 — a cross-warpgroup cost). Partition 0 and root are the same
cost domain (the default warpgroup executes the root block), so a
partition-0 owner outside the loop emits attr-less; entry acquires are
always attr-less (phase sources, see the inherit rule);
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
  barrier respected; each bracket's `pieceInfo` equals its **own chain's
  footprint** with branch-local effects and owners drawn from the region
  row's record (Enter == Exit; a For body's footprint equals its region
  summary, so For brackets equal the For record; a non-accessing if-branch
  has bare brackets) — owners are never invented for pieces a branch does
  not touch.
- **SYNC**: every edge connects two rows of the same region chain and points
  forward; the reader-set invariant holds (`Shared` sets are R-only for the
  current version — rule 2); both if-branch walks exit with identical
  holder states; fan-in
  sources are same-chain siblings of their destination (count is therefore
  execution-uniform) **and are pairwise-distinct partitions** (the lowering
  counts arrives per wave — §5.3 formula); every release either sources
  from a real access row with that op's payload (union across pieces /
  branch games where merged), or is `Enter`-sourced carrying the seed
  import (the producer-owner case — rule 5), or is `Enter`-sourced `none`
  by a non-producer with its same-partition pre-region acquire as the
  ordering witness — any other
  `none` downstream of an async producer is a hard error; per-semaphore
  balance:
  within each region execution,
  releases == acquires×count, with entry acquires covered exactly by initial
  permits; every access is preceded in chain order by an acquire of its
  component (the carrier exists — guaranteed by the entry acquire); **M3
  acquirer classes**: per semaphore, the acquiring owners contain at most
  one concrete partition (root additionally allowed — the inherit case).
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

---

## Addendum A — TOKEN RETENTION (evaluated, NOT implemented: perf regression)

**The redundancy.** A semaphore whose release and acquire satisfy all of:
released by the same completion event as another semaphore, acquired by
the SAME partition, and acquired LATER in that partition's program order,
can never block — by the time the partition reaches the wait, the permit
was granted long ago (the partition already waited on the same event at
its earlier acquire). Example: the meta-FA stats group, where the qk mma
releases both S0 (acquired by the softmax partition before `read qk_0`)
and S2 (acquired by the same partition later, before `store p`); S2
transfers no information. Same shape one level up for the l/m stats
stores (S5/S6).

**The elision rule (where it would land).** Stage 3, the wave-locality
`force` site in `walkChain` (see the placeholder comment in
`InsertSemasSyncDag.h`): the force may stand down — the touch riding the
partition's RETAINED token from its earlier wave, with no new acquire
and no semaphore — iff (a) the toucher held the carrier earlier in the
SAME chain (its token is live partition-local SSA), and (b) zero edges
are needed (every conflicting holder is already transitively
synchronized behind the toucher; an intervening conflicting touch
becomes a holder demanding a real edge, blocking the merge
automatically). Companion pieces: cross-game `syncedBehind` marking
(one release event covers every piece pending at that row), the chain
verifier accepting retained owners, an emitter retained
(component, partition) -> token map, and an owner-keyed view cache.

**It was implemented and measured** — full implementation, dedicated lit
test (`insert_semas_token_retention.mlir`, merge-fires + conflict-blocks
cases), and regenerated corpus live in commit `844bf8fa63`
("[InsertSemas rewrite 4.10/4]", branch
`egx/meta/sema10a-meta-new-sema-fresh-v5-fable`). Meta-FA stats groups
dropped 9 -> 7 semaphores; all gates green. **But it costs ~5% FA
performance** (user-measured; bisect-confirmed against `8642940bd1`).

**Root-cause findings (IR-level, from the 4.4 vs 4.5 corpus diff +
AssignStagePhase comparison):** every surviving op keeps byte-identical
`loop.stage`/`loop.cluster` — the software-pipeline schedule is
unchanged, so "the removed acquire perturbed the pipeliner" is ruled
out. The only real deltas are at runtime: two fewer mbarrier objects,
one fewer `tc5mma` arrive per mma, two fewer (always-satisfied) waits
per iteration — and, the surviving hypothesis: the merged stores now
ISSUE EARLIER in their partition's instruction stream (no wait
sequence before them), moving their TMEM writes from the post-commit
window into the mma's operand-streaming window — a contention-window
shift, plausibly the ~5%. Unproven at the IR level; confirming needs
profiles (TMEM/SMEM port stalls, mbarrier stall cycles).

**Status: NOT implemented, by user ruling (11jun26).** The "redundant"
semaphore is kept deliberately — it doubles as a runtime pacing point.
If ever revisited: the refinement suggested by the evidence is to merge
only when the riding touch lands in the SAME `loop.stage`/`loop.cluster`
as the retained acquire (the epilogue stats merges keep their benefit;
the cross-stage in-loop merge — the suspect — keeps its semaphore), and
to gate the whole feature behind an env knob for perf A/B.
