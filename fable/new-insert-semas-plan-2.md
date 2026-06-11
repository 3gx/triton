# Plan: New `nvws-insert-semas` — From-Scratch Implementation

Implements the design in `fable/semas-report3.md` (THE spec — four mechanical
stages over one node-based DAG: ACCESS-DAG → OWNER-DAG → SYNC-DAG → EMIT-IR).
The existing implementation is nuked and rebuilt from zero. This plan was
cross-checked, contract item by contract item, against the CHECK lines of the
full lit corpus (`test/NVWS/insert_semas*.mlir`,
`test/NVWS/tmem-buffer-reuse-semas.mlir`), with
`test/NVWS/insert_semas_meta_fa_fwd.mlir` as the keystone case; §3 records
the evidence.

## 0. Ground rules

1. **Nuke first.** The existing implementation is deleted at commit 0.
   **No code is borrowed from any `InsertSemas*` file — ever.** Nothing in
   them is of use; borrowing pollutes the implementation. Small semantic
   helpers (owner resolution, async-payload table) are re-derived from the
   spec (`semas-report3.md` §1.1), not copied.
2. **Reference reading (read, don't copy):**
   `third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertTmemSemaphore.cpp`:
   - region token threading: `processForRegion` (:979–1025),
     `processIfRegion` (:1027–1088), `processLocalBlock` (:964–977);
   - the TMEM backing-stage (1x/2x) decision — USER RULING: the
     producer-consumer pattern gate (`hasProducerConsumerPartitioning`) is
     DROPPED; the per-MMA-user multibuffering chain of
     `insertTmemSemaphoreSingle` (:1408-1425) is used VERBATIM as the
     whole decision (start true; every MMA user inside a loop vetoes via
     !hasAccReadModifyWrite && isAccMultibufferingPossible &&
     !getDisallowAccMultiBuffer && canDoubleBufferAcc; meta => 1).
     Historical note: `isMultiStagedMember`
     (:1379–1398; :1400–1434 is duplicate logic in
     `insertTmemSemaphoreSingle`), `canDoubleBufferAcc` (:1297), and the
     cross-group `numTmemBlocks` capacity accumulation (:1431, :1596);
   - the loop-scheduler shim: `workaroundForLoopScheduler` (:1640–).
   Its ping/pong scheduling is NOT the model — only these mechanics are.
3. **Allowed reuse — external utilities only:**
   `PartitionBuilder.h` (`createInto`; note the `{partitionId,
   stageCluster}` overload used throughout is InsertTmemSemaphore's local
   ~10-LoC wrapper (:367) — re-derive it in `InsertSemas.h`, don't copy),
   TritonGPU `Utility.h` (`addIterArgsToLoop` :151,
   `replaceIfOpWithNewSignature` :167), partition/tag helpers
   (`hasPartition`, `getPartitionIds`, `getPartitionOutputs`,
   `hasWarpSpecializeTag`, `getWarpSpecializeTag`), `getStageCluster`
   (PipeliningUtility), `getMemDescSize`, `MMAv5PipelineUtility.h`
   (`hasAccReadModifyWrite`, `isAccMultibufferingPossible`,
   `getDisallowAccMultiBuffer` — backing-stage policy only), NVWS
   `Utilities.h`/`WSUtility.h` where generic.
4. **No dialect changes.** Entry semaphores are
   `nvws.semaphore.create %bufs true`; all others `false`. Fan-in pending
   counts are implicit in the IR (one acquire op, N release sites; counted
   by `nvws-lower-semaphore`); the count lives only in the SYNC-DAG.
5. **All decisions before any mutation.** Stages 1–3 are pure analysis and
   additionally produce the **BackingPlan** and the per-node **Crossing**
   facts (§2 — crossings live on For/If DAG nodes; "ThreadingPlan" is their
   derived emission-time aggregation). The
   emitter applies plans; it decides nothing. In particular the TMEM 1x/2x
   stage check runs on the *unmodified* input IR, in the analysis path.
   Emission itself is bracketed by mechanical normalizations: a
   **pre-process** that nukes all original TMEM async tokens of managed
   groups, and two **post-processes** — backing coalescing into
   subslice/reinterpret views, then the loop-scheduler workaround (critical
   for `automatic-warp-specialization.mlir`). All are attr/type-driven
   rewrites, not decisions — §4 commit 4. One documented carve-out: the
   per-partition stage/cluster **cache** of contract I is filled during the
   render walk (it cannot exist earlier — it tracks the walk itself); it is
   deterministic transcription of anchor facts, not a decision.
6. **Determinism is mandatory.** The pass must be bit-deterministic from
   run to run: the same input produces byte-identical output IR and
   byte-identical dumps, every time. We have chased nondeterminism bugs in
   this pass family before; they are not tolerated. Concretely:
   - **Never iterate over a pointer-keyed `DenseMap`/`DenseSet` (or any
     hash-ordered container) anywhere the iteration order can reach the
     output** — emitted-op order, dump rows, ID/name assignment, plan
     construction, slot ordering, group/component processing order.
   - Hash containers are allowed for **lookup only**. Wherever ordered
     iteration is needed, use insertion-ordered containers
     (`llvm::MapVector`, `llvm::SetVector`, `SmallVector`) or sort by a
     deterministic key first (program-order rank, member index, piece id,
     edge id).
   - Every assigned ID (pieces, semaphore names `S0…`, components,
     crossing slot numbering, backing order) is allocated in program/insertion
     order — never in hash order.
   - The mechanical check, part of §4c self-verification at **every**
     commit: run the pass twice over the whole verification set and `diff`
     both the dumps and the output IR — any byte difference is a blocking
     finding.
7. **Banned concepts** (grep-able review checklist): post-emission
   `moveBefore`/`moveAfter` placement repair, **edge/semaphore** coalescing
   passes, payload or owner override chains, op-pattern-matched special
   cases, drain/loop-close synthesis, re-planning against mutated IR,
   RAW-vs-OPT DAG split, "back edge" framing. Named exceptions (the only
   ones): the §4 step-6 backing-view post-process and the step-7
   loop-scheduler workaround — both sanctioned by ground rule 5; nothing
   else may move or rewrite emitted sync. Inexpressible cases get a hard
   diagnostic naming the DAG node — never a workaround.
8. **`AGENTS.md` applies throughout**: build first, then lit (commands in
   §4a are the canonical forms); no pytest beyond the single user-sanctioned
   §5 gate-2 case; no guessing, no theorizing, no overreach beyond what this
   plan explicitly permits; re-read the plan at every commit. Anything that
   executes a kernel (the §5 runtime gates, any GPU repro) **requires the
   GPU and must run outside the sandbox**; build, lit, and dump capture are
   CPU-only and sandbox-safe.

## 1. Files and build

Delete at commit 0 (all seven):

```
third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemas{AccessDag,Common,
  Emitter,Model,OptSyncDag,OwnerDag,RawSyncDag}.h
```

`InsertSemas.cpp` is gutted to pass boilerplate (`NVWSInsertSemas` class +
`useMetaPartitioner` option, `Passes.td:142`; empty `runOnOperation`).

New layout — a **single translation unit**: one `.cpp` that includes one
header per stage (these files are not reused anywhere else, so no
CMakeLists changes at all). One header per stage also maps 1:1 onto the
commit ladder:

```
InsertSemas.cpp          pass + dispatcher (~150 LoC): gate on a WS-tagged
                         loop, then per group: stage 1 → 2 → 3, then stage 4;
                         includes the headers below in order
InsertSemas.h            shared model: Node, PieceTable, SemaTable,
                         BackingPlan, Crossing, traversal, owner/payload
                         helpers, dump/verifier shared utilities
InsertSemasAccessDag.h   stage 1: discovery + pieces + events + dump (commit 1)
InsertSemasOwnerDag.h    stage 2: Enter/Exit + owners + invariants + dump (commit 2)
InsertSemasSyncDag.h     stage 3: walk + grouping + entry acquires +
                         BackingPlan/node crossings + verifiers + dump (commit 3)
InsertSemasEmitIR.h      stage 4: token-nuke pre-process, plan application,
                         render, post-emit verifier, coalescing + loop-scheduler
                         post-processes (commit 4)
```

`InsertSemas.cpp` is purely the dispatcher over the stage headers. Note two
filenames (`InsertSemasAccessDag.h`, `InsertSemasOwnerDag.h`) coincide with
deleted old headers — they are deleted at commit 0 and **recreated from
scratch** at their commit; no content carries over (ground rule 1). Dump
stays behind `NVWS_INSERT_SEMA_DUMP_DAG=1` (tests reference this name).

## 2. Data model (`InsertSemas.h`, from spec §2)

Every type is defined here — nothing in `Node` or the tables is left as an
undeclared name.

```cpp
// ---- identifiers (indices into per-group tables; allocation order =
// ---- program/discovery order, never hash order — ground rule 6) ----
using MemberId = unsigned;     // index into PieceTable::members
using PieceId  = unsigned;     // index into PieceTable::pieces
using SemaId   = unsigned;     // index into SemaTable::semas
using CompId   = unsigned;     // connected component of pieces = one token game

// ---- owner: who executes an op / who is planned to hold a piece ----
using PartitionId = std::pair<int /*ttg.partition*/, int /*ws tag*/>;
using Owner       = std::optional<PartitionId>;  // std::nullopt = root/external
                                                 // (distinct from partition 0)

// ---- access classification ----
enum class Effect { R, W };    // R = provably load-only; W = everything else
                               // (spec §1.1: MMA A/B operand touches R,
                               //  accumulator W; sourceful allocs W)

// ---- region-row boundary plan, per piece (spec §4) ----
struct PieceInfo {
  Owner  owner;                // carried owner (loop: first toucher;
                               //  if: RULE A — the INCOMING owner (most
                               //  recent prior toucher in the chain;
                               //  fallback: first in-subtree toucher).
                               //  RULE B (post-if-toucher override) is a
                               //  recorded extension, NOT implemented —
                               //  see spec section 4 and the marked choke
                               //  point in InsertSemasOwnerDag.h)
  Effect effect;               // W iff ANY subtree access to the piece writes
};

// ---- one step of a memdesc alias chain (stage-1 fact, replayed at emit
// ---- to rebuild the view from the semaphore buffer) ----
struct AliasStep {
  Operation *op;               // ttg.memdesc_{index,subview,trans,
                               //              reinterpret,reshape}
  unsigned operandIdx;         // which operand carries the source memdesc
};

// ---- one buffer-touch of one access op ----
struct Touch {
  MemberId member;             // member alloc this access value resolves to
  Effect   effect;             // R | W for THIS touch
  Value    accessValue;        // the memdesc SSA value the op actually uses
  SmallVector<AliasStep, 2> alias; // chain: member alloc -> accessValue
};
// footprint(touch) = PieceTable::footprint[touch.member] — derived, never
// stored on the touch (pieces are a per-group property of the member).

struct Node {
  enum Kind { Func, For, If, Enter, Exit, Access, Acquire, Release };
  Kind kind;
  Operation *op;                    // For/If/Access anchor; null otherwise
  Node *parent, *prev, *next;       // program-order chain in parent region
  SmallVector<Node*> children;      // For: body head; If: then head[, else head]
  Owner owner;                      // Access/Acquire/Release: executing partition
  SmallVector<Touch, 2> touches;    // Access: one per touched member
  DenseMap<PieceId, PieceInfo> pieceInfo; // Enter/Exit/For/If only; iterated
                                          // sorted by PieceId (determinism)
  SemaId sema = 0; unsigned count = 0;    // Acquire(count) / Release
  AsyncOp payload = AsyncOp::NONE;  // Release: source holder's last REAL
                                    // access payload (carried through
                                    // re-anchoring — spec §5.1)
  Node *sat = nullptr;              // Release -> the ONE Acquire it satisfies
                                    // (an Acquire has count-many incoming)
  SmallVector<Crossing, 1> crossings; // For/If only, filled at stage 3:
                                      // carrier slots crossing this region
                                      // (see Crossing below) — rendered as
                                      // thread{c0:{1}} + per-chain EXIT
                                      // yield{} routes
  SmallVector<int, 2> requiredParts;  // For/If only, filled at stage 3:
                                      // sorted union of subtree row
                                      // partitions = the partitions that
                                      // receive a clone of this region
                                      // skeleton after partition-loops.
                                      // Emission extends the op's
                                      // ttg.partition array to exactly this
                                      // (the C10 rule — a transcription,
                                      // not a render-time derivation).
                                      // Rendered as parts{0,1}. ENTER/EXIT
                                      // need no owner: they execute once
                                      // per clone, in every parts{} member.
};
// AsyncOp is the existing NVWS enum (NONE | TC5MMA | TMALoad) — not a new type.
```

ACCESS → clone+extend (Enter/Exit + owners) → OWNER → clone+inject
(Acquire/Release) → SYNC; the emitter consumes SYNC 1:1. One recursive chain
traversal shared by builders, verifiers, dumps, and the emitter.

**Representation notes.** `prev`/`next` are the program-order chain within a
region (NOT dependency edges — those are the `sat` links); `children` are
the region-chain heads of `For`/`If` rows (structural containment, no
ordering/dependency semantics); `parent` answers region-scope queries
(locality invariant, Enter/Exit lookup, hoist anchors). `sat` is the only
dependency relation: each Release points to the one Acquire it satisfies
through its semaphore; an Acquire has count-many incoming links (fan-in);
the entry acquire has zero (satisfied by the create's initial permits —
which is also how `isEntry` is checked). `sat` always connects rows of the
same region chain and points forward; it serves verification (count
balance, acyclicity, locality, entry coverage) and dump rendering only —
the emitter never traverses it, placement being purely chain position.

Owner semantics — two distinct concepts, never mixed: `owner` (single
`(partition, wsTag)`) is **who executes the op** — its `ttg.partition`
annotation; an op runs on exactly one partition no matter how many pieces it
touches, so Access/Acquire/Release rows carry a single owner.
`pieceInfo` (map `PieceId → {owner, effect}`) is **the planned boundary holder per
buffer slice** — meaningful only for region rows (`Enter`/`Exit`/`For`/`If`),
which execute on no partition and where different pieces legitimately carry
different owners through the same region (qk_alpha `{1}` vs acc `{5}` in one
loop). A multi-touch op (MMA: R on A/B pieces, W on the accumulator piece)
whose touched pieces are held by different partitions does NOT get multiple
owners — it gets one incoming edge per cross-holder piece (then C2 fan-in:
one `acq(N)` executed by the op's own partition). Per-piece holders between
boundaries are the walk's transient state, stored on no node; their durable
trace is the edges.

What each snapshot populates (one Node type; fields fill in by stage):

| | ACCESS-DAG | OWNER-DAG | SYNC-DAG |
|---|---|---|---|
| node kinds | Func/For/If/Access | + Enter/Exit | + Acquire/Release |
| prev/next/parent/children | ✓ | ✓ (Enter/Exit spliced in) | ✓ (sync nodes spliced in) |
| owner + touches (Access) | ✓ | ✓ | ✓ |
| pieceInfo.effect (For/If rows: subtree OR) | ✓ | ✓ | ✓ |
| pieceInfo.owner (+ Enter/Exit rows, both halves copied) | — | ✓ | ✓ |
| sema / count / payload | — | — | ✓ (sync nodes) |
| `sat` | — (empty) | — (empty) | ✓ |
| crossings (For/If rows) | — | — | ✓ (slotOwner + per-chain finals) |
| requiredParts (For/If rows) | — | — | ✓ (clone set; C10 array) |

The chain is
an intrusive doubly-linked list so that stage-3 injection of sync nodes is
an O(1) splice that never invalidates any held `Node*` (edges, `sat`,
SemaTable, node crossings, op→node maps). The splice ↔ MLIR insertion
correspondence is exact and exhausts all cases: a Release spliced between
`src` and `src->next` (DAG insertAfter) emits via
`setInsertionPointAfter(prev->op)`; an Acquire spliced between `dst->prev`
and `dst` (DAG insertBefore) emits via `setInsertionPoint(next->op)`;
`Enter` ⇒ `setInsertionPointToStart(block)`; `Exit` ⇒
`setInsertionPoint(terminator)` (before `scf.yield`); a `For`/`If` neighbor
behaves like any op row (acquire before the loop fires once — the
super-node behavior). Placement is decided once, by the stage-3 splice;
stage 4 only transcribes chain order, which is also why dump and emitted IR
cannot diverge.

Side tables — all *decided* during analysis (stages 1–3) and frozen before
emission, with exactly two marked backpatch slots that receive SSA values
during emission (`Sema::create`, `BackingPlan::backing` — slots, not
decisions); every field defined:

```cpp
// ---- stage 1: discovery + pieces (per group) ----
struct Member {
  Operation  *allocOp;         // original ttng.tmem_alloc / ttg.local_alloc
                               // (buffer.* attrs live on it, preserved)
  MemDescType type;
  int64_t     offset, extent;  // [offset, offset+extent) in native units:
                               // TMEM = columns (getMemDescSize ->
                               // getTmemAllocSizes().numCols); local = the
                               // LEADING DIM of the memdesc shape — the
                               // planner's offset unit, corpus-proven
                               // (local_buffer_reuse: 128x128xf16 members at
                               // offsets 0/64 overlap, 0/256 do NOT — byte
                               // extents of 32768 would break the latter)
};
struct Piece {                 // cut-point interval (spec §3 item 2)
  int64_t lo, hi;              // [lo, hi) in native units
  SmallVector<MemberId, 2> cover; // members containing this piece, ascending
};
struct PieceTable {            // one per group
  SmallVector<Member> members;                  // discovery order
  SmallVector<Piece>  pieces;                   // ascending by lo; index = PieceId
  SmallVector<SmallVector<PieceId, 2>> footprint; // per member, ascending
  SmallVector<CompId> pieceComp;                // per piece: its component
};

// ---- stage 3: semaphores (per group) ----
struct Sema {
  CompId   component;          // the token game this semaphore belongs to
  SmallVector<PieceId, 2> pieces; // pieces its edges protect (dump/verify)
  unsigned count;              // pending count of the PRIMARY acquire (the
                               // first destination group); For-row groups
                               // merged onto this semaphore add their own
                               // acquire instances with their own counts
  unsigned expectedReleases;   // total release sites across all groups
                               // sharing this semaphore (+1 terminal for
                               // dedicated entries) — the balance check
  bool     isEntry;            // first event in chain order is an acquire
                               //   => nvws.semaphore.create ... true
  Owner    inheritStamp;       // entry semas: the component's FIRST
                               // ACCESS owner (old seed-acquirer fact)
                               // (carrier inherit). The entry-acquire NODE
                               // is owned by ROOT (it executes in the root
                               // region); emission stamps the op with this
                               // fact instead (partition+tag, or no attrs
                               // for root) — matches the previous pass's
                               // emitted IR. M3 verifier: per semaphore,
                               // acquirers = at most ONE concrete
                               // partition, root additionally allowed.
  Value    create;             // filled at emit step 2 (create op result)
};
struct SemaTable {             // index = SemaId; allocation order = the order
  SmallVector<Sema> semas;     // edges are grouped in program order (names S0…)
};

// ---- stage 3: backing plan (per group; §3.B/§3.C) ----
struct BackingPlan {
  int numStages;               // local: always 1; TMEM: 1 or 2
                               // (computeBackingStages, analysis-only)
  Operation *hoistAnchor;      // function scope, before the outermost WS loop
  SmallVector<Value> backing;  // per member; filled at emit step 2
                               // (ONE plain alloc per member; covering/view
                               //  shape is the commit-4 post-process)
};

// ---- stage 3: carrier threading plan (per function, all groups) ----
// Crossings LIVE ON THE DAG (a field of For/If Nodes — the DAG is the
// authority; no Operation*-keyed side table, hence nothing extra to patch
// during the commit-4 op-replacement fixup). Computed at stage 3, rendered
// by the SYNC dump on the region row as thread{c0:{1} final=a S1}.
struct Crossing {
  CompId comp;         // which token game (group implicit: the node's DAG)
  Owner slotOwner;     // the partition stamped on this slot's
                       // ttg.partition.outputs entry and yield attrs.
                       // Always equals the region row's pieceInfo owner
                       // for the component's pieces: EVERY chain's final
                       // carrier is held by that owner (EXIT-close hands
                       // back; pass-through was already held) — a COMMIT-3
                       // FACT (ground rule 5: emission decides nothing).
  SmallVector<Node *, 2> finals; // PER CHAIN, parallel to children: the
                       // chain's last carrier-producing row for the
                       // component (an Acquire node, or a nested region
                       // row with its own c-crossing — resolved
                       // hierarchically, innermost first). Its token is
                       // what THAT chain's yield returns — the NEW token.
                       // nullptr = PASS-THROUGH: that chain yields the
                       // INCOMING carrier unchanged — the OLD token (the
                       // skipped-path case; needs no other fact, which is
                       // why bare else brackets carry nothing).
};
// CROSSING RULE (mechanical): region op R gets a Crossing for component c
// iff R's subtree contains >=1 Acquire node of c. Wiring at render (pure
// transcription): For: init := enclosing carrier, yield := finals[0]'s
// token; If: thenYield := finals[0]'s token or incoming if null,
// elseYield := finals[1]'s token or incoming if null. Outer loops follow
// automatically (their subtree contains the inner acquires) —
// meta_fa_fwd's two-level threading (contract F). Dump rendering,
// faithful to the fields and INDEXED WHERE THEY APPLY: the region row
// prints the slot (thread{c0:{1}}), and EACH CHAIN'S EXIT row prints ITS
// OWN finals[] entry, so the route is reviewable per path:
//   EXIT ... yield{c0: a S1}     finals[i] = that Acquire row (NEW token)
//   EXIT ... yield{c0: scf.if}   finals[i] = a nested region row's result
//   EXIT ... yield{c0: pass}     finals[i] = nullptr — the OLD/incoming
//                                token passes through (the skipped path)
// yield{} is a ROUTE annotation, not an access/owner record: it never
// claims the branch touches a piece, so bare brackets stay honest.
//
// "ThreadingPlan" is therefore NOT a stored artifact: it is the DERIVED,
// emission-time aggregation of node crossings — commit-4 step 3 walks all
// groups' region rows in group order to number the appended slots across
// groups. Pure transcription; zero facts of its own.

// ---- per group, the whole artifact handed from stage to stage ----
struct GroupDag {
  unsigned    groupIdx;        // discovery order
  PieceTable  pieceTable;
  Node       *root;            // Func node of the current snapshot
  SemaTable   semaTable;       // empty until stage 3
  BackingPlan backingPlan;     // empty until stage 3
};

// ---- function-level driver state (InsertSemas.cpp) ----
//  - SmallVector<GroupDag> groups                 (discovery order)
//  - int numTmemBlocks                            (capacity accumulator,
//                                                  fed group-by-group into
//                                                  computeBackingStages)
//  - (slot numbering: derived at emit step 3 by walking all groups'
//     region-row crossings in group order — not stored)
//  - DenseMap<Operation*, Operation*> opFixup     (emit step 3: old->new
//                                                  scf ops; lookup only)
//  - Value poisonToken                            (emit step 1: the single
//                                                  function-level ub.poison)
```

Fresh helpers re-derived from spec §1.1 (~40 LoC): `resolveOwner(op)` →
`(partition, wsTag)` | root; `asyncPayloadOf(op)` → `tc5mma` for
`MMAv5OpInterface`, `tma_load` for `nvws.descriptor_load/gather` (including
through a sourceful alloc's src chain), else `none`.

## 3. Output contract — audited against the lit corpus

Each item below is what the CHECK lines mandate, with evidence. The plan's
stages must reproduce these qualitatively (goldens regenerate at commit 4;
sync shapes may change per spec §9, these structural rules may not).

**A. Backing allocs are new allocs, hoisted out of enclosing `scf.for`
ops — above the outermost WS loop — but NEVER across an `scf.if`.** In the
corpus the outermost WS loop sits at function level, so backings land at
function scope; in guarded persistent kernels (gate-2 case 2, 10jun26) the
WS loop lives in a function-level guard if and the oracle keeps backings,
creates, and entry acquires inside that branch — hoisting past the if
forces dead token threading through a partition-less if and trips
AssignStagePhase. Hoist law: from the first member alloc, ascend while the
parent is an `scf.for`; stop at `scf.if` or the function body.
`insert_semas_meta_fa_fwd.mlir`: originals for buffer.id=4 include
`%qk_0` *inside* the WS loop (:168) and the sourceful `%acc_164` *inside the
inner loop* (:302), yet the backing
`ttng.tmem_alloc {buffer.copy=1, buffer.id=4, buffer.offset=0} →
1x128x128xf32` and all its `semaphore.create`s are CHECKed at function level
(:85–:99), before the loop (:145). Same in
`insert_semas_nested_carrier.mlir` (:15–:19 vs :26–:27) and
`insert_semas.mlir::hoisted_alloc` (:1043–:1050).

**B. Backing depth: local always `1x`; TMEM `numStages ∈ {1,2}`, decided by
the InsertTmemSemaphore check, in analysis, before any mutation.**
Local `128x128xf16 → 1x128x128xf16` (meta_fa_fwd :69/:73/:77/:81/:123/:127;
`insert_semas.mlir` :800/:860; no 2x local exists anywhere). TMEM 2x cases:
`insert_semas.mlir` :70, :119, :175, :223, :396, :435, :1166;
`tmem-buffer-reuse-semas.mlir` :232. The 2x criteria visible in the corpus:
producer-side store (same partition as the MMA) + multibuffering possible +
not `tt.disallow_acc_multi_buffer` (forces 1x: :587/:616, :1101/:1145,
:272–:292) + TMEM capacity (`canDoubleBufferAcc` with `numTmemBlocks`
**accumulated across groups in discovery order** —
InsertTmemSemaphore.cpp:1431/:1596) ; `useMetaPartitioner` ⇒ 1
(meta_fa_fwd is all 1x). Implemented as `computeBackingStages(group,
numTmemBlocks&)` in **stage 3** output (BackingPlan), modeled on
the :1408-1425 veto chain per the user ruling (pattern gate dropped) —
never at emit time. **Zero-semaphore groups are exempt**: a group with no
semaphores is untouched at emission (contract H), so it gets NO stage
assignment and NO capacity charge — its BACKING line reports `untouched`.
Without this, phantom 2x charges from never-materialized groups are
order-dependent and can push a later REAL accumulator below capacity.

**C. Backing dedup and reuse views — emission per member, coalescing as a
post-process.** The final-IR shape in the goldens: members with identical
(offset, extent, type) share **one** backing value, listed once per member
in the create (`create [[ABUF]], [[ABUF]]` —
`tmem-buffer-reuse-semas.mlir` :16–:18, :77–:79, :265–:268); TMEM members
overlapping a covering member become **views of the covering backing**:
`tmem_subslice {N=offset}` + `memdesc_reinterpret` to the member's 1x type
(`tmem-buffer-reuse-semas.mlir` :184–:188; meta_fa_fwd :85–:98 — one
1x128x128xf32 backing for qk plus subslice/reinterpret views at N=64/65/66/0
for alpha, the two offsetkv members, and p, all five listed in every create;
`insert_semas_tmem_reuse_views.mlir` :12–:22 with `CHECK-NOT` for duplicate
id-42 allocs). This shape is produced the way the current pass correctly
does it — **as a post-process, after all semaphore IR is emitted and
verified**: emission creates one plain backing alloc per member (typed
`numStages × member shape`, `buffer.*` attrs preserved verbatim), and the
post-process then walks backing allocs per `buffer.id`, picks the covering
member per overlap component, RAUWs exact duplicates with the
representative directly, and replaces each contained/overlapping member's
backing with `tmem_subslice {N = offset − covering.offset}` +
`memdesc_reinterpret` to its type — RAUW updates the `semaphore.create`
operand lists automatically. The rewrite is keyed purely on the preserved
attrs + extents (mechanical, no decisions); all backings sit at the same
hoist anchor, so dominance is trivial. Local members are exempt: they keep
separate backing allocs with `buffer.id`/`buffer.offset` preserved
(`insert_semas_local_buffer_reuse.mlir` :25–:28, :63–:67) — physical
placement is realized downstream from the attrs.

**D. Sourceful allocs become acquire + buffer-view + explicit store.**
Row independence (gate-2 attention evidence, 10jun26): the replacement RAUW
excludes the group's other access-row ops — each row retargets itself with
its OWN owner's view at its own render; blanket RAUW left the PV mma on the
softmax store's partition-0 view (cross-partition SSA edge, rejected by
partition-loops in case 3, destroyed-with-uses crash in case 4). The
original alloc's erase is deferred to the final cleanup (it must outlive
every row that still matches operands against its result).
`ttng.tmem_alloc %src` → `semaphore.buffer` view + `arith.constant true` +
`ttng.tmem_store %src, view, true` at the original site
(`tmem-buffer-reuse-semas.mlir` :22–:25; meta_fa_fwd :302 → :306–:309 —
in-inner-loop sourceful alloc stores into view member #4 and releases
`<none>`); `ttg.local_alloc %src` → `ttg.local_store` into the view
(`insert_semas_local_buffer_reuse.mlir` :32–:41). The synthesized store
(and its `arith.constant true` / splat) inherits the ORIGINAL alloc's
owner and stage/cluster (mining gap 8). A SCALAR `%src`
(float/int) first materializes via `triton::SplatOp`, owner+stage
stamped, then the local_store (mining gap 7). **RESOLVED (mining gap 6,
user ruling):** a managed descriptor-fed buffer NEVER reaches this pass in
`tt.` form — `nvws-insert-allocas` (the immediately preceding pass,
InsertAllocas.cpp:451/:562/:575) converts cross-partition
`tt.descriptor_load/gather`+alloc pairs to `nvws.descriptor_load/gather`
upstream; a pair still in tt-form here is same-partition -> zero
semaphores -> untouched (contract H). The old emitter's conversion path
was dead code in the current pipeline. Stage 1 hard-errors if a managed
tt-form descriptor-fed alloc ever appears — documenting the pipeline
invariant, not guessing a behavior.

**E. ALL original TMEM tokens are nuked — as a pre-process, before any
semaphore IR is emitted.** Design rule: nuke *all* TMEM async-token plumbing
of every semaphore-managed group, then thread *all* semaphore tokens fresh.
The nuke is a dedicated pre-pass at the top of commit-4 emission: create
**one function-level `ub.poison : !ttg.async.token`** (InsertTmemSemaphore's
`replToken` pattern — one value for the whole function; THIS PRE-PROCESS
never creates poison ops inside loops — the step-7 workaround's in-loop
poison is a sanctioned exception), then clear dep/token operands of the groups' access ops
(empty `[]` brackets — `insert_semas.mlir` :600–:608; meta_fa_fwd
:322/:329) and RAUW their token results and dead token iter_arg slots —
init and yield — with that single value
(`tmem-buffer-reuse-semas.mlir` :80/:84/:95, :118/:134).
Carriers then travel **only** in slots the node crossings own (appended,
§F). If-crossings are LIVENESS-PRUNED at stage 3: no slot when no
later row consumes the carrier (see spec node table)."
 Reusing the original token slots for semaphore carriers (as the old
pass does in meta_fa_fwd :176) is rejected by design — the carrier set does
not correspond to the original token set, and mixing the two couples the
emitter to input plumbing it is supposed to erase. Untouched single-owner
groups (contract H) keep their tokens. After the nuke, dead token-typed
signature slots are ERASED to a fixpoint (scf.for iter_arg with region arg
AND result unused; scf.if result unused) — inits/yield operands dropped,
ttg.partition.outputs entries filtered, DAG anchors fixed up. The original
expectation that poisoned dead slots survive was FALSIFIED by gate 1
(10jun26): husk slots change loop/if signatures, the AWS loop scheduler
mis-stages, and descriptor ops never lower. Expected golden churn: dead
slots gone, appended carrier slots.

**F. Carrier threading: slots appended to `scf.for` iter_args / `scf.if`
results, set at yields, partition-stamped.** meta_fa_fwd: outer for 1→11
results (:145) carrying entry-acquire tokens from function level; inner for
9→13 (:176) fed from outer iter_args — carriers thread through *both* loop
levels. Yields set tokens with partition attrs (`insert_semas.mlir` :614,
:818, :1063; nested_carrier :44); for-op `ttg.partition.outputs` extended
(:616, :646; meta_fa_fwd :338/:399); scf.if token results
(`insert_semas.mlir::if_split_workaround` :1125–:1142;
`tmem-buffer-reuse-semas.mlir` :123/:134).

**G. Descriptor loads are retargeted, never converted.** The pass does NOT
rewrite `tt.descriptor_load` → `nvws.descriptor_load` (register-producing
`tt.descriptor_load`s are untouched — `insert_semas.mlir` :39–:40); where
the input already has `nvws.descriptor_load` writing SMEM, only its
destination operand is retargeted to the semaphore view (:866–:867;
meta_fa_fwd :162–:163), and the matching release carries
`[#nvws.async_op<tma_load>]` (:870; meta_fa_fwd :180/:186). MMA releases
carry `<tc5mma>` (:50, :88, :605, :1190; meta_fa_fwd :323); consumer-side
releases `<none>` (:92, :148; meta_fa_fwd :309/:314).

**H. Root entry, same-owner, errors.** Root producer feeding a WS loop:
entry acquire + root store into the view feed the loop directly, **no**
release/acquire between store and loop (`insert_semas_root_entry_tmem.mlir`
:23–:32); post-loop exit handoff appears after the loop (:50–:51).
Single-owner groups emit nothing (`CHECK-NOT: nvws.semaphore` —
`tmem-buffer-reuse-semas.mlir` :50–:68, `insert_semas.mlir` :788). Allocs
without `buffer.id` are synthetic-id singletons, fully handled
(`insert_semas_local_no_buffer_id.mlir` :11–:25). Unsupported alias use is a
hard diagnostic (`insert_semas_local_errors.mlir` — `RUN: not`, message
at :15).

**I. Stage/cluster stamping on emitted semaphore ops — critical, fully
specified.** Downstream loop scheduling depends on every emitted sync op
inside a pipelined loop carrying correct `loop.stage`/`loop.cluster`. The
rules, per op kind (this is how both InsertTmemSemaphore.cpp and the old
pass behave):

- `nvws.semaphore.create` (and backing allocs): **no** stage/cluster — they
  live above the pipelined region.
- `nvws.semaphore.acquire` anchored before a real access op: stage/cluster
  read off that access op (`getStageCluster(dstOp)`).
- `nvws.semaphore.buffer`: stamped with the CONSUMING ACCESS's owner and
  stage/cluster, NOT the acquire's (mining gap 3) — views are materialized
  LAZILY at the requesting access, per region: the view cache is cleared
  at every acquire AND at every region entry/exit, so a carried token gets
  a fresh view in each region that uses it. One buffer op yields ALL
  member views of a multi-member semaphore at once (`TypeRange` of member
  view types), cached until the next acquire or region boundary.
- `nvws.semaphore.release` anchored after a real access op: stage/cluster of
  that source access op.
- Sync ops anchored on virtual rows (`Enter`/`Exit`/super-node — no real
  anchor op): the **per-partition last-seen cache** — during the render
  traversal, every visited access updates `cache[owner] =
  getStageCluster(op)`; a virtual-row sync op owned by `{P}` uses
  `cache[P]`. This is InsertTmemSemaphore's mechanism verbatim: the
  `stageClusters` map updated at each access (:917–:918, :928–:929,
  :936–:937) and consulted by `getAcquireStageCluster` (:771–:781).
- Fallback: anchor carries no stage/cluster and the cache has no entry for
  the owner ⇒ omit the attrs (`createInto` with nullopt does not stamp) —
  the unpipelined-anchor behavior.

Real-op anchors are recorded as node facts at stage 3 (SYNC-DAG); the cache
is part of the deterministic render walk. Input access ops already carry
the attrs (set by the TritonGPU **loop scheduler**, upstream of this pass —
e.g. meta_fa_fwd :302 `loop.cluster = 4, loop.stage = 0`; the NVWS
`AssignStagePhase` pass is a different thing: it runs *after* insert-semas
and assigns the stage/phase *operands* on acquire/buffer/release, which is
why emitting them bare is legal); this pass only ever copies, never
derives a schedule. The post-emit verifier checks: every emitted
acquire/buffer/release inside a pipelined loop whose anchor access carries
stage/cluster carries it too — a miss is a hard diagnostic.

**Keystone walkthrough (meta_fa_fwd, buffer.id=4).** Members: qk `[0,128)`
f32 (in WS loop), p `[0,64)` f16 (sourceful, in inner loop), alpha
`[64,65)`, offsetkv `[65,66)`, `[66,67)` (function level). Pieces:
`[0,64){qk,p}`, `[64,65){qk,alpha}`, `[65,66)`, `[66,67)`, `[67,128){qk}`.
The plan must: nuke the group's original tokens (E, pre-process); hoist five
per-member backings to function scope (A); classify the mma writing qk as W,
the loads as R, the sourceful p-alloc as W rewritten to a store (D); derive
the @1→@5→{@0,@1} chain with fan-in per the spec walk; thread carriers
through outer and inner loops (F); stamp `tc5mma`/`tma_load`/`none` payloads
(G); and let the post-process coalesce the five backings into the golden
shape — one covering 1x128x128xf32 alloc + four subslice/reinterpret views
(C). Every requirement maps to a stage below; nothing requires a heuristic.

## 4. Commit ladder

Dump-only until commit 4; lit `insert_semas*` fails during bring-up AND may
still fail at the end of this plan (golden regeneration is deferred — §5).

**The verification set** (dumped, archived, §4c-verified at EVERY commit):
all of `test/NVWS/insert_semas*.mlir`, `test/NVWS/tmem-buffer-reuse-semas.mlir`,
**plus** `test/NVWS/insert_semas_release_count.mlir` (the release-
multiplicity lit test: a fan-in-2 regain unified with a single-source
For-row ready edge => r S(2)), **plus the real-kernel captures** —
`logs/moe-9jun26-v1/before-insert-semas-pmatmul.mlir` (MoE persistent matmul,
fp8×mxfp4 dequantize: root-stored accumulator, conditional epilogue read,
descriptor_load/gather producers, mma_scaled spanning five groups). Captures
are run WITHOUT `-split-input-file`; new captures of real kernels may be
appended to this list as they appear.

**4a. User-runnable diagnostics at every commit.** Like the current
`InsertSemas.cpp`, `NVWS_INSERT_SEMA_DUMP_DAG=1` prints, per backing group,
every stage built so far — cumulative as commits land: commit 1 → ACCESS-DAG
diagnostic; commit 2 → + OWNER-DAG diagnostic; commit 3 → + SYNC-DAG
diagnostic (entry acquires, counts, semaphore table, BackingPlan/
thread{...} crossing annotations on region rows + semaphore/backing
tables); commit 4 → all three remain available alongside
the transformed IR. The dump format is normatively defined by the spec's
examples (§4.1 E1–E6, §5.3, §5.5 — ENTER/EXIT rows, `R`/`W` access rows,
`r`/`a` sync rows with owners and counts); the old pass's dumps are not the
reference (that code is deleted), and the archived commit logs become the
reference corpus. The user can independently dump any stage at any commit:

Build and lit commands follow `AGENTS.md` (canonical; its rules apply:
**first build, then run lit**; no pytest unless explicitly sanctioned —
the §5 gate 2 case is user-sanctioned):

```bash
# BUILD (AGENTS.md): always first
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/build/cmake.linux-x86_64-cpython-3.12/ \
  && ninja triton triton-opt

# LIT (AGENTS.md): from that same build folder
/home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit -v test
#   (single suite/file: append its path under test/, e.g.
#    .../llvm-lit -v test/TritonGPU/automatic-warp-specialization.mlir)

# DUMP capture: from the repo root — test/NVWS/*.mlir and logs/ are
# repo-root-relative (build + lit run CPU-only and work inside the sandbox)
cd /home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git
mkdir -p logs/new-insert-semas/commit<N>
NVWS_INSERT_SEMA_DUMP_DAG=1 build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt \
  test/NVWS/<f>.mlir -split-input-file -allow-unregistered-dialect \
  --nvws-insert-semas 2> logs/new-insert-semas/commit<N>/<f>.txt > /dev/null
```

**4b. Per-commit checkpoint — user green-light required.** After each
commit's implementation is done and self-verified (per the definition in
4c): STOP and come back to the user with the archived dumps
(`logs/new-insert-semas/commit<N>/`), the 4c self-verification report, and
(commit 4) the gate results. The user reviews and green-lights; only then is
the commit made, and only then does work on the next commit start. No
commit without explicit approval.

**4c. Definition of "self-verified".** Self-verified means the DAG dumps
have been checked to be **100% consistent with this plan and the spec — no
violation of any plan contract** — by exhaustive per-file review, not spot
checks. **Fix-ordering obligation:** every rule change or bug fix lands in
the order spec -> plan -> code -> verification prompts; a fix that exists
in code or in a verifier prompt but not in the spec is itself a finding
(that is how spec gaps are born — the fleet would keep "verifying" a rule
the spec contradicts). **Old-reference cross-check:** for files with a
previous-architecture reference dump (`logs/per-edge-plan/commit5/`,
`-prev-semas` fixtures), structural correspondence of loop chains, entry
placement, and semaphore unification is part of the review — spec-faithful
output that diverges structurally from the old architecture without a
documented design reason is a finding, because the fleets verify against
the spec and are blind to spec gaps by construction. At **every** commit this includes the determinism check (ground
rule 6): run the pass twice over the whole verification set and `diff`
dumps and output IR byte-for-byte — any difference is a blocking finding.
Concretely, for **every** file in the verification set and **every**
function/group in it:

- **Commit 0**: clean build; the pass is a verified no-op (output IR ==
  input IR); the seven headers are gone.
- **Commit 1 (ACCESS-DAG)**: every dump row is checked against the input
  IR, and the input IR is checked against the dump (nothing missing,
  nothing extra): groups/members with correct `buffer.id`/offset/extent in
  native units (f16 vs f32 column widths checked explicitly); pieces match
  a hand computation of the cut-point construction; every access of a
  managed group present with the correct owner and R/W touches exactly per
  the §1.1 table (MMA A/B operands R, accumulator W, sourceful allocs W,
  descriptor loads W); every `FOR`/`IF` row's per-piece effect summary is
  re-derived by hand (OR over the subtree's touches) and matches the dump;
  control-flow nesting in the dump matches the IR;
  async tokens demonstrably ignored; alias diagnostics fire where the
  corpus expects them.
- **Commit 2 (OWNER-DAG)**: for each region and piece, the carried owner is
  re-derived by hand from the commit-1 access order (loop = first toucher;
  if = RULE A, the incoming owner — most recent prior toucher, falling back
  to the first in-subtree toucher; with the WS scope barrier: a WS-tagged
  For row contributes root upward) and matches the dump; bracket records are
  verified as **restrictions** — each branch's ENTER/EXIT carry only that
  branch's own footprint with branch-local effects (owners from the region
  row; non-accessing branches bare — nothing invented); ENTER/EXIT rows present
  with the owner invariants holding (`For == Enter == Exit`,
  `If == then.* == else.*`); WS scope-barrier and root/`{@tag.p}` rendering
  correct; each of the spec §4.1 cases E1–E6 located in the corpus and
  confirmed to render as specified.
- **Commit 3 (SYNC-DAG)**: for each component, the expected edge set is
  re-derived by hand-executing the §5.1 walk rules over the commit-2
  OWNER-DAG and compared 1:1 against the dumped edges — same sources, same
  destinations, same owners; group-by-destination counts match the
  hand-derived fan-in; exactly one entry acquire per component, on the
  correct semaphore, with `isEntry` set, hoisted to the correct anchor; the
  canonical §5.3 structure is visible (entry acquire before the loop,
  release/acquire chain in the body, acquire-before-EXIT where the last
  toucher differs); per-execution balance re-checked by hand on at least
  the keystone (meta_fa_fwd buffer.id=4 end-to-end) plus every conditional
  test; BackingPlan numbers (1x/2x per group) checked against the corpus
  evidence list in contract B; node crossings checked against the
  crossing rule (slot iff subtree contains an Acquire of the component);
  hand-balancing uses the lowering's arrive formula — Σ over a wave's
  distinct releasing partitions of `|async_ops|`, one arrive per array
  element (matters once union payloads exist).
- **Commit 4 (EMIT-IR)**: the compiled-in post-emit verifier reports clean
  on the whole set; the emitted IR of representative tests (at minimum:
  meta_fa_fwd, local_smem_fanout, per_edge_tmem, nested_carrier,
  local_cfg, root_entry, tmem-buffer-reuse) is read and checked against
  every §3 contract item A–I — hoisted backings and depths, sourceful→store,
  per-member backings then coalesced views, poisoned original tokens,
  appended-only carrier slots set at yields with partition attrs,
  retargeted descriptors, payload attrs, stage/cluster stamping per
  contract I; the three §5 gates are green.

Any discrepancy found is a **finding that blocks the checkpoint**: it is
either fixed (and the affected verification re-run from scratch) or
reported to the user as a plan/spec gap — it is never explained away. The
self-verification report presented at the 4b checkpoint enumerates: files
checked, what was re-derived by hand vs. mechanically asserted, findings
and their resolutions, and any contract item whose evidence is weaker than
"checked exhaustively".

### Commit 0 — nuke
Delete the seven headers; gut `InsertSemas.cpp`; build green.

### Commit 1 — ACCESS-DAG (creates `InsertSemas.h`, `InsertSemasAccessDag.h`)
Discovery (groups by `buffer.id`, synthetic ids; TMEM = every
`ttng.tmem_alloc`, local = mutable-memdesc `ttg.local_alloc`; member
intervals in native units: TMEM = columns via `getMemDescSize`, local =
leading dim of the memdesc shape (NOT bytes — see the `Member` comment in
§2) — f16/f32 element width
matters: meta_fa_fwd's 128x128xf16 p-member spans 64 columns, half of qk's
128). Pieces (cut-point construction; unknown offset ⇒ whole-group
footprint). Events with owner + touches, R/W per spec §1.1 (MMA A/B operand
touches R, accumulator W; sourceful allocs W; async tokens ignored). Alias
chains tracked; unsupported alias ⇒ diagnostic (contract H). Structural
nodes only where the subtree has group accesses; **every `For`/`If` node
gets its per-piece effect summary** (`PieceId → Effect`, the bottom-up OR
over subtree touches — spec §3 item 4; purely touch-derived, so it belongs
to this commit). Dump: member/piece table +
access tree with per-piece effects on `FOR`/`IF` rows. Verify by eye on the set; meta_fa_fwd's ten groups (4 TMEM —
buffer.id 2/3/4/5 — plus 6 synthetic-id locals) are the acid test.

### Commit 2 — OWNER-DAG (creates `InsertSemasOwnerDag.h`)
Clone; `Enter`/`Exit` per region; the **owner half** of `pieceInfo` by the
deterministic rules (spec §4: loop carried owner = first toucher per piece;
if-branch owner = RULE A, the incoming owner — conditional code costs sync
only when taken; RULE B (post-if-toucher override, hoists the handoff when
consumption continues after the join) is a recorded extension that reduces
to A when its shape is absent — adopt only on profiled need, at the single
marked choke point; super-node; the WS scope barrier as the operational
toucher-contribution
rule — a WS-tagged For row contributes ROOT upward, its own record keeps
the carried owner, so every region row outside a WS loop is root except
intrinsic-tag deciders; root propagates freely) —
where the commit-1 effect summaries make the toucher scans flat: a region
row is an ordinary toucher of exactly the pieces in its summary, no
recursive subtree probing. Effects are **copied** from the For/If stage-1
summaries onto `Enter`/`Exit`, never recomputed. The dump prints
owner+effect on region rows per piece, consistent with access rows. Assert the owner
invariants, per piece: **`For-row owner == ENTER owner == EXIT owner`** and
**`If-row owner == then.ENTER == then.EXIT == else.ENTER == else.EXIT`** —
where `Exit` is the row that sits immediately before `scf.yield` and
renders as `EXIT` in dumps (the row v4-era dumps called `YIELD`; the new
dumps standardize on `EXIT`, matching the spec's examples — dump-test CHECK
lines regenerate later anyway). These equalities make every region a
self-contained token game (spec §4.1); they are asserted, never repaired.
No `sat` links and no sync nodes exist at this stage. Dump per piece. Check virtual-toucher cases E1–E6 (spec §4.1)
against `insert_semas_local_cfg.mlir`, `insert_semas_nested_carrier.mlir`,
`insert_semas_local_read_lifetime.mlir`.

### Commit 3 — SYNC-DAG + plans (creates `InsertSemasSyncDag.h`)
Clone; the walk (spec §5.1 rules 1–6: reader-sharing with the reader-set
invariant; **local state seeding** — every region body walks a fresh state
seeded `Exclusive(carried owner)` at ENTER plus the imported
`versionProducer` fact (a producer re-reading its own data gets no edge —
program order spans region boundaries since partition-loops clones the
skeleton into every participating partition's stream), EXIT closes in-body
holders only, the parent state is touched solely by the super-node row as
one R- or W-touch per its effect; payloads tracked as
`(lastRow, lastPayload)` per holder per piece — releases source from real
access rows with that op's payload, or are ENTER-sourced with the
seed-imported producer payload (rule 5's import — load-bearing for rule A's
producer-brackets-own-branch epilogues) or, for non-producer owners, none
with the same-partition pre-region acquire as witness; If-row
outgoing payload = union over its branch games; destination-then-recurse
walk order pinned; WS-For root
adoption — contract H; the virtual else carries no sync rows; the
transitive-sync skip: per holder, partitions that already took an edge
from it since its lastRow are recorded (syncedBehind) and never re-edged —
rules 1/2/4 — GATED BY WAVE LOCALITY (user ruling 10jun26): every elision
applies only while the toucher still holds the carrier; a cross-partition
touch on a moved carrier always materializes an edge, and two hard-error
verifiers (stage-3 chain carrier locality; emitter post-emit token/view
locality) pin the class — see the spec's WAVE LOCALITY section; EXIT closes additionally only when load-bearing — under a
loop or with a later touch in an ancestor chain — never as drains); dedupe
(incl. the same-owner second collapse: same dst + same source owner +
different rows -> latest row, payload union — multi-piece games require it);
group-by-(destination row, destination OWNER) with the For-row unification
(a For-row destination merges into the loop's in-body regain group — same
component+acquirer, M3-clean); UNIFORM PENDING COUNT: all acquire sites of
one semaphore carry the same count; merged groups with fewer sources scale
their releases' arrive multiplicity (Node count on Release, default 1,
rendered r S<k>(n); per group the multiplicities sum to the count — hard
diagnostic when impossible)
`(srcRow, dstRow, srcOwner)` with **payload union** on collapse (the
release's `async_ops` array carries the union); group by destination;
inject `Acquire`/`Release` nodes with recorded owner/payload/count; entry
acquires per component (spec §5.3 — the regain is the carried owner's last
acquire in the body's own chain, child chains excluded), `isEntry` in the
SemaTable.

Also computed here, read-only (ground rule 5):
- **BackingPlan**: `computeBackingStages` per group in discovery order with
  accumulated `numTmemBlocks` (contract B) and hoist anchors. (The
  covering/view shape is NOT planned here — it is the attr-driven commit-4
  step-6 post-process, contract C.)
- **Entry acquires**: root-owned rows (carrier inherit), placed at the
  component's placement chain (descend single-involving-row if branches,
  never For rows); regain search descends if-branch chains (conditional
  handbacks are valid regains), never nested For bodies; acyclic
  components get a dedicated entry semaphore + terminal release;
  `inheritStamp` recorded per entry semaphore; M3 acquirer-class check.
- **Node crossings**: for each `scf.for`/`scf.if` DAG row, the components whose
  carrier crosses it (slot order fixed here), **and per crossing the
  `slotOwner` + per-chain `finals[]` facts** — resolved hierarchically from the
  SYNC-DAG (the region's last Acquire of the component, or a nested region
  row's own slot), innermost regions first. Emission transcribes these;
  it never re-derives an owner or searches for a carrier at render time.

Verifiers (hard errors, spec §7): edges same-region + forward; if-branch
holder states identical; fan-in sources same-chain siblings **and
pairwise-distinct partitions** (lowering counts distinct releasing
partitions per wave — spec §5.3 IR realization); release payload equals the
source holder's last real access payload; per-execution
balance with `isEntry` covering entry acquires; every access preceded by an
acquire of its component. Dump: canonical structure (entry acquires before
their loop, counts on grouped acquires). Eyeball against
`insert_semas.mlir::local_smem_fanout`, `insert_semas_per_edge_tmem.mlir`,
`insert_semas_local_cfg.mlir`, and the meta_fa_fwd keystone chain.

### Commit 4 — EMIT-IR (creates `InsertSemasEmitIR.h`)
Strict order — pre-process, apply frozen plans, render, post-process:

1. **Pre-process: nuke original TMEM tokens** (contract E): create **one
   function-level `ub.poison : !ttg.async.token`** for the whole function —
   exactly InsertTmemSemaphore's `replToken` pattern (token RAUW at
   :922/:933/:940; dead iter_arg/yield slots set to it at :998/:1015), one
   value instead of one-per-buffer since this pass processes all groups.
   Then, for every semaphore-managed group: clear dep/token operands on its
   access ops and RAUW all original token results and dead token iter_arg
   slots (init + yield operands) with that single poison value. **No poison
   ops are ever created inside loops** — in-loop uses merely reference the
   top-level value; sitting outside the WS loop, it needs no partition
   annotation. After this step the render stage never sees input token
   plumbing.
2. **Backings + creates + entry acquires** (BackingPlan + SemaTable): per
   group at the hoist anchor — **one backing alloc per member**, typed
   `<numStages × member shape>`, `buffer.*` attrs preserved verbatim; one
   `nvws.semaphore.create` per SemaTable row (`true` iff `isEntry`) listing
   the member backings in member order (contracts A–B); then render the
   **entry-acquire nodes** (function-level, before the outermost loops —
   they need no loop signatures). This ordering exists because step 3
   needs the entry tokens as real SSA init values; the step-4 traversal
   marks these nodes already-emitted and skips them.
3. **Signature rewrites** (aggregating node crossings across all groups,
   in group order — the derived "ThreadingPlan"): each `scf.for` gets its token
   iter_args appended via `addIterArgsToLoop` (init values = the step-2
   entry tokens / enclosing carriers), each `scf.if` its token results via
   `replaceIfOpWithNewSignature` — **exactly once per op**, aggregated
   across all groups; structured ops are processed **outside-in** (an inner
   loop's init may be the enclosing loop's new iter_arg, which must exist
   first). Extend `ttg.partition.outputs` for new results using each
   Crossing's recorded `slotOwner` — a commit-3 fact, never derived here
   (contract F). Utility asymmetries an implementer must handle:
   `addIterArgsToLoop` **erases** the old loop, `replaceIfOpWithNewSignature`
   **does not** — the husk `scf.if` must be erased manually; appended
   for-slots and materialized-else yields are **rebuilt** (via
   `appendToForOpYield` / new `scf.yield`), not `setOperand`-patched, so
   `Exit` insertion points are always resolved dynamically (terminator
   lookup), never cached. Both
   utilities replace the op: patch `Node::op` anchors of all SYNC-DAGs
   through an old→new map immediately (mechanical pointer fixup — the one
   engineering hazard; this replaces the old driver's re-planning).
   A managed `scf.if` whose DAG else chain is **virtual** (spec §4 else
   rule) gets its else region materialized here iff the if carries token
   results (a virtual else never carries sync rows;
   `replaceIfOpWithNewSignature` creates the block automatically).
4. **Render**: one traversal per SYNC-DAG, one action per node kind (spec §6
   table), all ops via `createInto` with stage/cluster stamped per contract
   I (real-op anchors from node facts; virtual-row anchors from the
   per-partition last-seen cache); `{P}`-owned sync ops outside a WS loop
   also get the loop's tag. Threading recipe modeled on
   InsertTmemSemaphore: For (cf. :992–1024) — init operand := carrier,
   body carrier := iter_arg, yield operand := body-final carrier, after :=
   loop result; If (cf. :1043–1087) — branch walks on copies, **assert**
   equal exit carrier owner (no reconcile-toggling; inequality = verifier
   failure), set both yields per the Crossing facts (each branch := its
   `finals[]` entry's token, or the incoming carrier when null — SSA
   pass-through; yield partition attrs are UNION-EXTENDED with `slotOwner` — start from
   the yield's existing `ttg.partition` ids and add the token owners,
   never overwrite (mining gap 5)), carrier := if
   result; no-crossing regions
   balance locally (cf. :964–977). Access nodes: retarget memdesc operands
   through the recorded alias chain onto the member's view; sourceful
   allocs → explicit store (contract D); descriptor destinations retargeted
   only (contract G). Region ops' `ttg.partition` arrays are extended to
   their node's recorded `requiredParts` (the stage-3 fact rendered as
   `parts{…}` — the C10 rule as transcription: the region skeleton must
   exist in every listed partition's stream for partition-loops routing),
   with the condition/bounds availability to each added partition verified
   (spec §6) — note `--tritongpu-partition-loops` does **not** rematerialize
   condition chains into a partition's stream; if the condition's defining
   ops do not already carry `P`, that is a hard diagnostic, not a fixup.
   **View types (mining gap 4):** a local member's view type = the access
   site's own memdesc type after walking leading `memdesc_index` alias
   steps, mutability forced `true`; alias-chain replay SKIPS
   `memdesc_index` steps whose result type already equals the current view
   type, and forces each cloned view op's result mutability to its
   source's; TMEM views retain the allocShape suffix.
   **RAUW discipline (mining gap 9):** replace only uses DOMINATED by the
   view; exclude the synthesized store and `SemaphoreCreateOp` users;
   track rewritten values so later retargeting matches the original OR
   the already-rewritten value. Erase fully-retargeted originals (alias
   ops in reverse order before allocs — uses before defs).
5. **Post-emit verifier**: emitted-op ↔ node bijection; every view token
   traces to a DAG acquire; tokens cross boundaries only via planned slots.
6. **Post-process: coalesce TMEM backings into views** (contract C): per
   `buffer.id`, RAUW exact-duplicate backings with the representative;
   replace contained/overlapping member backings with
   `tmem_subslice {N = Δoffset}` + `memdesc_reinterpret` views of the
   covering backing. Attr/extent-driven, runs only after the verifier has
   accepted the emitted semaphore IR. Local backings exempt.
7. **Post-process: loop-scheduler workaround, last** — a second, separate
   post-processor: the `scf.if` acquire/release split reimplemented from
   the OLD PASS's `splitSemaphoreIfForLoopScheduler`
   (`git show 5cfe0ac6e7^:.../InsertSemasEmitter.h` :2793–:2999) — NOT from
   `InsertTmemSemaphore.cpp:1640`, whose simplified variant hard-codes
   partitions 1/0 and only matches release-first/acquire-last then-blocks
   (mining report gap 1). The full rule set to reimplement: the candidate
   release need not be the branch's first op (skip ConstantLike and alias
   ops); ELSE-branch splits; RELEASE-ONLY splits (TMEM semaphore with a
   later acquire in the same branch -> exit-if only); the
   acquire-first-then-block pattern with a release immediately preceding
   the if; full-split REFUSAL when the TMEM semaphore has multiple member
   base types; exit-if stage = the release's stage with
   `inferPrecedingMmaStage(ifOp)` fallback; enter/exit partitions = the
   UNION of the release's and acquire's owners (owner-derived, never
   hard-coded); middle-if partitions = original minus the enter/exit ids
   with per-result outputs; `setPartitionOutputs(exitIf, {})`. The
   workaround legitimately creates ONE `ub.poison` inside the loop body
   (feeding the middle if's dead slot) — exempt from the step-1 rule. It reshapes
   `scf.if` token plumbing so the downstream loop scheduler can pipeline
   release and acquire into different stages. **This step is critical for
   `test/TritonGPU/automatic-warp-specialization.mlir` to pass** — AutoWS
   runs insert-semas internally, and without this workaround the loop
   scheduler rejects/mis-schedules the conditional sync. Covered also by
   `insert_semas.mlir::if_split_workaround`.

**Golden regeneration of `test/NVWS/insert_semas*` is OUT OF SCOPE of this
plan** — it is deferred to the next stage, after this plan completes and is
verified through the §5 gates. During and at the end of this plan, those lit
tests may fail; the verification currency is the per-stage dumps (§4a) plus
the §5 gates. Expected eventual churn (for the follow-up stage): entry
acquires before loops, non-entry creates `false`, per-destination fan-out
semaphores, finer piece granularity, tail acquires before `scf.yield`,
appended-only token slots. Error tests (`insert_semas_local_errors.mlir`)
must still error throughout.

## 5. Gates (commit 4 exit criteria)

`test/NVWS/insert_semas*` lit tests **may fail** — golden regeneration is
deferred to the next stage, after this plan is completed. The gates for this
plan are exactly the following (`third_party/tlx/killgpu.sh` on any hang).
**Always build first** (AGENTS.md): `cd build/cmake.linux-x86_64-cpython-3.12/
&& ninja triton triton-opt`. Gate 1 (lit) is CPU-only and runs from the
build dir, sandbox fine. **Gates 2–3 execute kernels on the GPU and MUST be
run outside the sandbox** (sandboxed shells have no GPU access — a sandboxed
run fails or silently tests nothing; disable sandboxing for these commands
or have the user run them); they run from the **repo root** —
`run_nvws*.sh` and the pytest path are repo-root-relative and fail from the
build dir:

1. **`test/TritonGPU/automatic-warp-specialization.mlir` must pass,
   unmodified.** AutoWS runs insert-semas internally; the
   loop-scheduler workaround post-processor (commit-4 step 7) and correct
   stage/cluster stamping (contract I) are what make this achievable. If
   the mechanical output cannot satisfy this test, STOP and report the
   exact diff to the user — do not relax the test and do not add
   workarounds beyond the specified post-processors.

   ```bash
   /home/egaburov/work/oai-triton/triton-src/llvm-project.git/build/bin/llvm-lit \
       -v test/TritonGPU/automatic-warp-specialization.mlir
   ```

**Gate ordering is strict (mining gap 10): no pytest runs until gate 1
passes UNMODIFIED. A timeout at any gate is a hang CAUSED BY THE CURRENT
CHANGE — stop and root-cause it; never retry, never broaden the test
selection.**

2. **Pytest cases, 60s timeout per test — DO NOT run the entire pytest suite:**

   Use PYTHONPATH=/home/scratch.egaburov_sw/oai-triton/triton-src/triton-solid-01.git/python/

   ```bash
   PYTHONPATH=<above> timeout 60s pytest -q \
     "python/test/unit/language/test_warp_specialization.py::test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]"
   ```
   As well as:
   ```
   python/test/unit/language/test_warp_specialization.py::test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]
   ```
   and
   ```
   python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]
   ```
   and
   ```
    python/test/unit/language/test_warp_specialization.py::test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024
    ```

3. **The two FA runtime scripts, 60s timeout each** (canonical deadlock
   regressions — the historical failure class, fan-in counts with
   not-all-live releasers, is excluded by the commit-3 verifiers; these
   runs prove it):


   ```bash
   PYTHONPATH=python timeout 60s sh run_nvws.sh
   PYTHONPATH=python timeout 60s sh run_nvws_1.sh
   ```

   ADDENDUM: DO NOT RUN THESE YET, when gate 2 is passing come back to user for verification and further instructions.

## 6. Acceptance criteria

- Seven old headers gone; no line of the old implementation survives; new
  implementation lives in the six files of §1 — one TU (`InsertSemas.cpp`
  as dispatcher) including one header per stage, zero CMakeLists changes
  (~2.5–3.5k LoC expected).
- One `Node` type end to end; three DAG snapshots; emitter renders SYNC 1:1;
  dumps and emission share one traversal.
- Every §3 contract item reproduced (A–I) — including the stage/cluster
  stamping rules (I) — with the BackingPlan (incl. the TMEM 1x/2x check)
  computed strictly in analysis, before any IR mutation; emission bracketed
  exactly as: token-nuke pre-process → plan application → render → verifier
  → backing-coalescing post-process → loop-scheduler workaround
  post-process.
- All stage verifiers on; canonical spec §5.3 structure visible in dumps
  (entry acquire before the loop; ENTER/EXIT as virtual first/last
  touchers); `NVWS_INSERT_SEMA_DUMP_DAG=1` diagnostics available
  cumulatively at every commit (§4a).
- Per-commit user checkpoint honored (§4b) with self-verification per the
  §4c definition: exhaustive dump-vs-plan consistency established and
  reported (hand-re-derived owners/edges/counts, contract items A–I
  checked), findings either fixed or escalated — never explained away;
  green light received, then committed — for every commit.
- Bit-deterministic (ground rule 6): no ordered iteration over hash
  containers anywhere output-reaching; run-twice diff clean at every
  commit.
- Banned-concepts list (§0.7) absent; the three §5 gates green
  (automatic-warp-specialization.mlir unmodified, the single pytest case,
  both FA scripts); `insert_semas*` golden regeneration explicitly deferred
  to the next stage.
