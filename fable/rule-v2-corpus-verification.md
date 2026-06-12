# Rule v2 corpus verification — report (12jun26)

## 0. The rules — final formulation (read this first)

Everything below this section is derivation history and evidence; this is
the contract. Vocabulary: access `op {p}`; owner; adjacency (plain /
through-Enter / through-Exit / bracket-pair); cut; hold (maximal uncut run);
merge; anchor; carrier; entry acquire; adoption; absorber; once-flip exit
semaphore; pseudo-access (ENTER/EXIT as carried-owner sequence elements).

### CURRENT RULE (the shipped pass)

> **At every ownership change, insert `rel ; aq` at the change point —
> anchors bind to the new owner's next op, where the FOR row, ENTER and
> EXIT count as ops of the carried owner.**

One uniform anchor; sound everywhere. Consequences: every synced component
in a loop gets the entry acquire + carried token + bottom regain (the EXIT
anchor), unconditionally; handoffs descend into if branches. The one flaw:
EXIT is an op but not an access, so anchoring to it pads holds with
body-length waiting — the FA P stall (−47 TFLOPS), paid by every buffer
whether or not its protocol needs the loop boundary.

### PROPOSED RULE — THE HOLD RULE (validated §7)

> 1. **Cut** every dependence edge of the buffer's ACCESS-DAG whose two
>    accesses have different execution contexts (owner differs, or the
>    enclosing predicate set differs).
> 2. Each maximal un-cut-connected group of accesses is a **hold**.
> 3. **`aq` immediately before the hold's first access; `rel` immediately
>    after its last.**

That is the entire rule. It differs from the current rule by one move:
anchors bind to the **hold's edges** (real accesses) instead of the
**change point** (where FOR/ENTER/EXIT count as ops). The substrate is the
same for both rules: the dependence DAG — adjacency exists only between
CONFLICTING accesses (stage-1 ACCESS-DAG edges, on the validated-not-flawed
list from the start). The linear "access sequence" used in the examples
below is shorthand, exact whenever all accesses pairwise conflict (one
shared slot); non-conflicting siblings (E3) have no adjacency and fan
in/out derives directly.

### The realization layer (NOT the rule — shared with the current rule, already implemented)

A placement rule does not execute; SCF + SSA + the hardware realize its
anchors. The current rule's "simplicity" already rests on exactly this
machinery (entry acquires, inheritStamp adoption, carriers, conditional
pairs, pending counts, exit semaphores — all in the shipped pass, none of
it stated in the current rule). The same layer serves the new anchors:

- **Pairing**: each cut is one semaphore handoff. Ends with equal dynamic
  multiplicity pair directly — including sequential loops, where the cycle
  simply continues on the same semaphores (E4). Ends with unequal
  multiplicity (per-iteration release vs once-per-run reader) use the
  standard bridge: the **absorber** (the wait that consumes the final
  async arrive — joins never wait for async) plus a fresh **once-flip**
  semaphore for the once-side (E1, §6 A1). The bridge is the realization
  of that cut, not a rule condition.
- **SSA**: tokens crossing iterations or branches materialize as
  iter_args / if-yields / region capture; a hold whose end is
  predicate-dependent realizes as conditional pairs with token
  pass-through (B1/B2). The component's first hold pairs with the
  semaphore's initial credit — that acquire is the entry acquire, emitted
  in root and adopted across the ws fork (never a root→partition pair).
- **Fan-out / fan-in** (ruled, E3): one release per non-conflicting
  successor edge; a multi-predecessor cut gets one acquire with
  pending_count = k — never serialize independent siblings. Counts come
  from DAG facts that already exist.
- **Elision** (lower-semaphore peephole): fork-join makes root-consumer
  exit pairs degenerate → elide; `[none]` final release → elide the
  absorber too (re-emitting the async-proxy fence where needFence would
  have). Absorber placement is flag-gated for perf A/B
  (`ABSORBER_IN_ROOT_P0/_ALL`, with the mandatory final-iteration peel).

Proof the layer needs nothing new: §7's validation — in 5 of 7 case
families the **existing pass machinery realizes the new anchors
op-for-op**; the other two were the E3/E4 rulings, both now folded in.
Scoping (component = footprint-overlapping allocs + alias specs;
single-owner components emit nothing) is likewise shared infrastructure.

### Why every device follows — the three constraints

The realization layer is not a list of devices; it is three constraints,
and each device is the unique solution to them at some cut:

- **C1 (SSA)**: every access takes a token; defs dominate uses; region
  edges need iter_arg/yield/capture; **no live token → find an acquire or
  insert one**.
- **C2 (pairing)**: every cut = one semaphore handoff: release at the
  upstream hold's end, acquire at the downstream hold's start.
- **C3 (balance)**: acquire/release executions balance per semaphore on
  every dynamic path.

| device | which constraint forces it |
|---|---|
| entry acquire | C1's "insert": a merged hold's init slot with no live token (E0); first hold pairs with the initial credit |
| adoption | C1's "find": a live token at the seam (case-4 pre-loop access; acc epilogue via the loop result) |
| carrier iter_arg | C1: token used above its def in the body → yield/iter_arg |
| bottom acquire + once-flip exit | C3: a per-iteration release cannot balance a once acquire → the once-side needs a once-executing post-loop release → C1: it needs a token → only source is a loop-result → which needs a yielded acquire (E1, §6 A1) |
| conditional in-branch pairs + if-yield pass-through | C3 *per dynamic path* + C1 across not-taken iterations (B1, B2) |
| fan-out releases / pending_count=k acquire | C2 per edge + C3 per cycle (E3 — ruled: never serialize) |
| nothing at sequential-loop seams | C3 already balances (matching recurring multiplicity) → C2 pairs directly (E4) |
| root-side elisions | not a derivation change at all — a lowering peephole over redundant ordering (fork-join) |

Teaching script: *hold region → aq just before, rel just after → thread
tokens, find-or-insert inits → C1/C2/C3 solve the rest.* Nothing in the
corpus required a decision outside this chain (§7.3).

### Expected behavior — all cases

| shape | emission under the proposed rule |
|---|---|
| in-loop ping-pong, nothing outside (P, qk, m_i, m_ij, k, v) | pure point-of-use; no entry, no carrier, no bottom acquire — the −47 TF fix |
| same-owner wrap `op1·p1 … op3·p1` (E0) | carrier from the MERGE of real accesses; entry acquire as its init |
| leading-only `op1·p1; op2·p2` | aq at body top; no carrier |
| acc (pre-loop init, in-loop, post-loop epilogue) | entry adoption; carrier; bottom absorber; same-owner epilogue adopts the loop result |
| post-loop reader in ROOT (E1) | uniform device, exit pair degenerate → elided; absorber stays iff final release is async |
| post-loop reader in another PARTITION (§6 A1) | full device: absorber + once-flip handoff — irreducible |
| consumer inside scf.if (B1) | both handoff sides in-branch, conditional; hold survives not-taken via if-yield |
| same-owner hold spanning a branch join (B2) | conditionality cut → branch-resident + unconditional holds, mediated handoff |
| nested loops (A4) | same rule per loop level; inner boundary device only on multiplicity mismatch; cross-owner inner exit = release-after-inner-for + fresh acquire |
| two sequential ws loops, same buffer (E4) | continuation on the same semaphores; no device |
| independent readers (E3 diamond) | fan-out releases / fan-in pending_count=k — readers stay unordered |
| single-owner buffer | nothing emitted |

## What was run

Multi-agent verification of the proposed placement rule ("rule v2": acquire
anchored immediately before the hold's first access, release after its last
access, brackets/region ops transparent) against:

- all 24 `test/NVWS/insert_semas*.mlir` lit files (one analyst per file; each
  derived v2's placement by hand on the input IR and compared against the
  **actual current-pass output** from `triton-opt --nvws-insert-semas`, not
  CHECK lines);
- the captured real FA-meta kernel
  (`logs/fa-11jun26-v3-root/passes-03/047-before-nvws-insert-semas.mlir` input
  vs `048-*.mlir` current output) — all 7 protected buffers / 15 semaphores;
- 3 red-team agents (control-flow, depth/async, aliasing/lifetime lenses)
  tasked to refute the "no carve-outs" claim;
- every CONCERN independently re-derived by an adversarial verifier before
  acceptance.

42 agents total. Tally over 98 function/component analyses:

| verdict | count | meaning |
|---|---|---|
| IDENTICAL | 36 | v2 prescribes exactly what the current pass emits |
| DIFFERS_SOUND | 46 | v2 places differently; soundness justified per case |
| CONCERN → **confirmed** | **13** | real rule gaps (verified adversarially) |
| CONCERN → refuted | 1 | analyst error (insert_semas.mlir attention_forward O) |
| SKIPPED | 2 | error-diagnostics test; unexercised alias clause |

**Bottom line: the core anchor change is validated everywhere it matters for
perf, but the claim "rule v2 needs no carve-outs" is REFUTED.** The 13
confirmed concerns + red-team breaks collapse into two structural families
plus a scoping-definition list. All three independently-derived amendment sets
converge on the same shape — which is, notably, exactly the empirical boundary
the in-tree fixup guards landed on.

## 1. What is validated (the perf-relevant core)

Verified against the real FA kernel capture (047→048):

- **All six in-loop FA components — k, v, m_i, m_ij, qk, p — are
  DIFFERS_SOUND**: reader-side acquires already identical (current pass is
  already point-of-use there); writer-side acquires move from the rotated
  bottom to point-of-use exactly as the in-tree fixup does today. The −47 TF
  placement fix is rule-derivable, not just empirically guarded.
- The acc-pattern **mid-loop commit-wait acquire** is correctly produced by v2
  under the merged-hold row order (tail…, then head…) — the body-order
  misreading is precisely the bug e51ec62358 fixed (red team:
  OK_AFTER_ANALYSIS).
- **Depth≥2 multibuffering**: `pending_count` is per-acquire-cycle fan-in
  arity, invariant under the anchor move (verified on qk depth-2; red team:
  OK_AFTER_ANALYSIS).
- **branch_local_init dominance trap does not bite**: with both in-body
  adjacencies cut there is no merged hold, hence no entry acquire to hoist
  (red team: OK_AFTER_ANALYSIS).
- If-resident handoffs (rule 8), inner-loop lifetime hoisting
  (local_read_lifetime), drain-less exits (tmem_no_loop_exit_drain), post-ws
  reads, and the FA stats interleaved-alias component
  (mixed_overlap_spanning_member) all check out.

## 2. Family A — loop-boundary handoffs (11 of 13 confirmed concerns)

**The gap.** When a buffer's body contains cuts AND its protocol crosses a
loop boundary, "one acquire+release pair per hold, anchored at accesses"
cannot express the boundary handoff:

- **One release, two downstream cuts**: the body-tail hold's single release
  must satisfy the next-iteration head hold's acquire on iterations 1..N−1
  and the post-loop hold's acquire at iteration N specifically. One release
  op signals one semaphore; every static realization deadlocks or completes
  the post-loop acquire prematurely (all four enumerated in §6 A1).
  Evidence: `insert_semas.mlir` @specialize_mma_only and @load_scale_mma_user;
  FA capture acc_tmem (047:130 MMA → 047:136 root epilogue load vs 047:114
  next-iteration p1 load); `insert_semas_root_entry_tmem.mlir`.
- **Entry adoption / double acquire**: when through-Enter is uncut (same
  owner) but the iteration adjacency is cut, the head access belongs to the
  entering merged hold on iteration 1 but to per-iteration holds on 2..N; an
  access-anchored acquire double-fires on iteration 1 → N+1 acquires vs N
  releases → deadlock at the first iteration.
  Evidence: `insert_semas_meta_fa_fwd.mlir` accumulators (buffer.id=2 and 3);
  `insert_semas_live_tag_source.mlir`.
- **Root-entry adoption contract**: `insert_semas_root_entry_tmem.mlir:20-22`
  *explicitly forbids* the root→partition release/acquire pair before the
  loop that v2's owner-cut at through-Enter would emit; the corpus mandates
  adoption (pre-loop token becomes the iter_arg init with no handoff pair) —
  warp_specialize entry implicitly synchronizes pre-loop root code.
- **Nested-loop multiplicity**: a cut adjacency is only realizable between a
  release site and an acquire site with equal dynamic execution multiplicity;
  mixed cut/uncut edges on one access at different nesting depths deadlock
  under any single static access-anchored placement.
  Evidence: `insert_semas_nested_carrier.mlir` (all three functions, deepest:
  N_inner×N_middle acquires vs 1 release per outer iteration);
  `insert_semas_release_count.mlir` @release_multiplicity_unified_fanin_regain
  (inner-loop release fires T times, T runtime-dynamic, against a per-outer
  acquire).

**The amendment (converged independently by ~6 verifiers).** Boundary
adjacencies (through-Enter / through-Exit) may remain *uncut* only in the
zero-in-body-cut case (the rule-6 hoist). If the body contains any cut for the
buffer, the loop boundary is realized by the **rotated boundary machinery**:

- iter_arg init = the pre-loop hold's live token (**adoption**, no
  release/acquire pair) when through-Enter would be uncut, else a fresh entry
  acquire before the for;
- a **bottom-of-body re-acquire** immediately after the body-tail hold's
  release — the one sanctioned exception to access-only anchoring — whose
  token is yielded;
- the loop-result token **drains/hands off after the for** (post-loop
  consumer or balancing release).

The same lift applies recursively at nested-loop boundaries (multiplicity
equalization).

**Perf check on the amendment**: P, qk, m_i, m_ij, k, v have **no pre/post-loop
accesses** — no boundary handoff exists — so they remain pure point-of-use.
acc keeps the rotated shape. This is *exactly* the partition the in-tree
fixup's `hasOneUse` guard discovered empirically. The amendment re-derives the
fixup's boundary as a rule instead of a pattern-match.

## 3. Family B — conditionality (1 confirmed concern + red-team break)

**The gap.** Cuts are static but holds are predicate-dependent:

- **Conditional consumer**: `insert_semas_conditional_multi_result.mlir`
  @conditional_multi_result_if_token (mma p1 unconditional; tmem_load p0
  inside `scf.if`, true only at iv==0). v2's unconditional point-of-use
  acquire next to the unconditional mma deadlocks on every not-taken
  iteration. The current pass's conditional re-acquire-if (acquire inside an
  if yielding the token) is the sound realization. Same family:
  `insert_semas_raw_if_token.mlir`, `insert_semas_if_split_metadata.mlir`.
- **Same-owner hold spanning a branch join**:
  `insert_semas_local_cfg.mlir` @local_if_consumption_continues_after_join —
  uncut same-owner adjacency from a branch-resident load to a post-join load
  puts the acquire inside the then-region and the release after the join:
  SSA dominance violation and an unprotected access on the else path. The
  current pass emits a mediated handoff (4-semaphore chain) here.

**The amendment.** (i) Add a **conditionality cut**: also cut same-owner
adjacencies whose two accesses do not execute under the same predicate (one
inside an if branch, the other outside or in the other branch); the resulting
handoff is mediated. (ii) A hold's anchors must execute under the same dynamic
condition as the accesses they protect; the conditional re-acquire-if shape
(rule 8 generalized) realizes predicate-dependent holds.

## 4. Family C — scoping definitions (1 confirmed concern + red-team clarifications)

Definition-level fixes, no placement consequences beyond their cases:

1. **Component formation = physical footprint, not buffer.id**: bucket by
   buffer.id, then union-find on transitive `[offset, offset+extent)`
   intersection (plus alias specs). Disjoint footprints sharing an id are
   independent (`insert_semas_local_buffer_reuse.mlir`
   @local_non_overlapping_aliased_buffers, lines 109–112: "no semaphores";
   `insert_semas_mixed_overlap_members.mlir` @tmem_disjoint_slivers_cross_partition:
   one id=521 group dissolves into four components).
2. **Zero-cut (single-owner) components emit nothing** — no degenerate
   semaphore pair (current pass skips them; rule 6's hoist applies only to
   multi-owner buffers' cut-free loops).
3. **Owner of partition-attributed ops outside ws loops** must be defined
   tag-scoped (`insert_semas_local_cfg.mlir`
   @local_root_external_distinct_from_ws_tag_zero: pre-loop op carries
   `ttg.partition=0` but is NOT in-loop p0).
4. **Opaque users of allocs** (unregistered ops, e.g.
   `insert_semas_tmem_reuse_views.mlir` "use") must be declared accesses
   (conservative: any transitive user touches).
5. **Immutable allocs are out of scope** (FA q_smem is unprotected by the
   current pass; v2 text needs the mutable-only filter explicit).
6. **Boundary adjacencies exist for every scf.for**, not only ws loops
   (`insert_semas_local_read_lifetime.mlir` inner non-ws loop).
7. The FA capture carries **no buffer.id attrs** — scope key needs an
   alloc-op-identity fallback.

Rides: out of v2's output domain by construction (release after last access);
the verifier invariant (no token buffer-use after release-use) stands. The
parked-retention hazard note in `fable/semas-report3.md` Addendum A is
unaffected.

## 5. The amended rule (v3 candidate)

Three layers, replacing "no carve-outs":

1. **In-body holds (cycle closes inside the body): access-anchored
   point-of-use.** aq before the hold's first access, rel after its last;
   merged (bracket-pair-uncut) holds get the carrier via iter_arg. This layer
   carries the entire FA perf fix.
2. **Loop-boundary handoffs of cut-bearing buffers: rotated boundary
   machinery.** Entry adoption or entry acquire → iter_arg; bottom re-acquire
   immediately after the body-tail release (the sanctioned non-access
   anchor); loop-result drain/handoff after the for. Applies recursively at
   nested-loop boundaries (multiplicity equalization). Boundary adjacencies
   stay uncut only for cut-free (rule-6) loops.
3. **Conditionality: cuts include predicate boundaries; anchors execute under
   their accesses' predicates** (conditional re-acquire-if / mediated
   handoffs, generalizing rule 8).

Convergence note: layer 1+2's partition of the FA components (six
point-of-use, acc rotated) is byte-for-byte the partition the in-tree
post-emission fixup enforces via its `hasOneUse` guard. The corpus analysis
derives the guard's empirical boundary from first principles.

## 6. Pseudo-IR illustrations of the confirmed gaps

Notation: `opN {p}` is an access to THE buffer under analysis, owned by
partition p; `aq S -> %t {p}` / `rel S %t {p}` are semaphore ops; `[%t]` marks
the token an access uses. Each example abstracts the cited corpus case; every
failure annotation and every sound shape below was independently verified
against the actual pass output (`triton-opt --nvws-insert-semas` on the cited
files) by adversarial agents; their corrections are folded in.

Vocabulary: holds = maximal same-owner runs of the buffer's access sequence
after cutting at owner changes; the rule under test (v2) anchors `aq` before a
hold's first access and `rel` after its last, with brackets transparent. Each
example names the **live bracket** v2 dropped — the bracket the current pass
treats as a carried-owner pseudo-access and the buffer genuinely needs.

### A1 — one release, two downstream cuts (dropped live EXIT)

Abstracts `insert_semas.mlir` @specialize_mma_only / @load_scale_mma_user and
the FA capture's acc_tmem.

```
for {
  op1 {p1}          // consumer-side access
  op2 {p2}          // producer (MMA)
}
op3 {p3}            // post-loop epilogue read by ANOTHER PARTITION
                    // (a ROOT consumer is the fork-join-degenerate variant — §7)
```
Cuts (all three adjacencies differ in owner): (op1,op2) plain → S_b;
(op2→op1) bracket pair → S_a; (op2→op3) through-Exit → S_c.
Holds: {op1}·p1 and {op2}·p2, executing once per iteration (N total, N
runtime-dynamic); {op3}·p3, executing once.

A cut means: the downstream hold must not begin before the upstream hold has
ended. The two cuts leaving {op2} therefore require:

- **(i)** the acquire before op1, execution i+1, is satisfied by the release
  after op2, execution i;
- **(ii)** the acquire before op3 is satisfied by the release after op2,
  **execution N specifically** — op3 must not run while any op2 remains.

Under v2, {op2} has ONE release (anchor: after op2; counting: one per hold) —
a single static op executing N times. A release signals one semaphore; an
acquire with pending_count k completes when k anonymous credits are
available. Every static realization violates (i) or (ii):

1. release → S_a only: (ii) has no signaler → **deadlock at op3**;
2. release → S_c only: (i) has no signaler → **deadlock at op1, iteration 2**
   (iteration 1 consumed the initial credit);
3. release → both S_a and S_c every execution: S_c gets a credit at iteration
   1, op3's acquire (pending_count=1) completes right then — op3 runs while
   op2 has N−1 executions left → **(ii) violated**. pending_count=N is not
   writable (static attribute, dynamic N);
4. op3's acquire shares S_a: aggregate counts balance (init + N credits vs
   N+1 acquires), but credits are anonymous and nothing orders p3's wait
   after p1's waits — there is NO join between partitions mid-region, so
   p3 reaches its acquire immediately and can consume the initial credit →
   op3 runs before the loop has done anything → **(ii) violated**,
   schedule-dependently. (For a ROOT consumer the join DOES order the wait
   last and option 4 becomes realizable — that is the ABSORBER_IN_ROOT
   variant, §7.)

The failure is structural: a release executing N times into static semaphores
cannot distinguish its last execution, so requirement (ii) is not
expressible under v2's anchors.

Sound shape (= current pass output; EXIT live as p1's pseudo-access):

```
S_c = sema.create               // dedicated exit semaphore
aq S_a -> %t0                   // entry acquire (root context, un-attributed)
%tN = for iter_args(%t = %t0) {
  op1 [%t] {p1}
  rel S_b %t {p1}
  aq S_b -> %u {p2}
  op2 [%u] {p2}
  rel S_a %u {p2}               // ONE release, ONE consumer:
  aq S_a -> %t' {p1}            // <- bottom re-acquire (= aq before live EXIT)
  yield %t'
}
rel S_c %tN {p1}                // exit release CONSUMES the loop result
aq S_c -> %v {p3}               // op3 re-acquires via the exit semaphore
op3 [%v] {p3}
```

Why this satisfies both requirements, in the same vocabulary: with EXIT in
the sequence as the carried owner's pseudo-access, the bracket pair
(EXIT, op1) is same-owner → MERGE → hold (EXIT | op1). Hold {op2} now has
exactly ONE downstream cut, (op2, EXIT), satisfied N:N by the acquire before
EXIT — the bottom re-acquire is nothing but that merged hold's anchor ("aq
before the hold's first access"). Requirement (i) is carried by the merged
hold's token through the iter_arg. The cut (EXIT, op3): on the last instance
the merged hold is truncated at EXIT(N), so "rel after the hold's last
access" lands AFTER the for, executing once, transitively after op2's
execution N (its token came from the bottom acquire that release satisfied) —
requirement (ii). Note the post-loop reader does **not** adopt the loop
result here: the (EXIT·p1, op3·root) cut is cross-owner, so the exit handoff
is an explicit release + fresh acquire. Direct loop-result adoption occurs
only for a same-owner through-Exit (e.g. the meta-FA p0 epilogue load
consuming the inner-loop result token directly) — the same cut criterion
decides both.

### A2 — entry adoption / double acquire (dropped live ENTER)

Abstracts `insert_semas_meta_fa_fwd.mlir` accumulators (buffer.id=2/3) and
`insert_semas_live_tag_source.mlir`.

```
op0 {p1}            // pre-loop init store
for {
  op1 {p1}          // head access (load)
  op2 {p2}          // MMA
}
```
(op0→op1) through-Enter UNCUT (same owner); (op1,op2) cut; (op2→op1)
iteration CUT.

The conflict: iteration 1's op1 is already covered by the entering hold
{op0, op1@1} (op0's token is live; no release has occurred); iterations
2..N's op1 each need an acquire paired with the PREVIOUS iteration's release.
v2's only tool is a static `aq` before op1, which fires on EVERY iteration:

```
op0 [%t0] {p1}
for {
  aq S_a -> %t {p1}  // <- fires at iteration 1 too, where the buffer is
  op1 [%t] {p1}      //    already held via %t0 and NO release has happened
  ...                //    yet -> blocks forever -> DEADLOCK at iteration 1
  rel S_a ... {p2}   //    (N acquires, but only the N-1 releases of
}                    //    iterations 1..N-1 can precede them; iteration 1's
                     //    acquire has no possible releaser)
```
Omitting the acquire instead leaves iterations 2..N unprotected. No static
access-anchored choice is right for both iteration 1 and the rest.

Sound shape (adoption + rotation; ENTER live):

```
op0 [%t0] {p1}                    // NO release after op0
for iter_args(%t = %t0) {         // op0's token ADOPTED as the init
  op1 [%t] {p1}                   // iter 1: op0's token; 2..N: bottom token
  rel S_b %t {p1}
  aq S_b -> %u {p2}
  op2 [%u] {p2}
  rel S_a %u {p2}
  aq S_a -> %t' {p1}              // bottom re-acquire
  yield %t'
}
// loop-result token drained/balanced after the for
```
N bottom acquires pair with N releases; iteration 1 needs none. Verified
op-for-op against the meta_fa_fwd ground truth (init store uses the outer
token with no release after it; inner loop adopts it as the iter_arg init;
balanced at trip count 0).

### A3 — root-entry adoption contract

Abstracts `insert_semas_root_entry_tmem.mlir` (citation line-exact:
lines 20–22, "it must not emit a root->partition semaphore release/acquire
pair before the loop").

```
op0 {root}          // pre-loop init by root
for ws {
  op1 {p1}
  op2 {p2}
}
op3 {root}          // post-loop read by root
```
v2-literal: (op0→op1) through-Enter is different-owner → CUT → a
release/acquire pair:

```
op0 [%r] {root}
rel S_e %r {root}    // <- exactly the pair the corpus contract forbids:
aq S_e -> %t0 {p1}   //    the ws-loop ENTRY already synchronizes pre-loop
for ... }            //    root code against all partitions
```
(The acquire is shown hoisted to its loop-seed position; strict v2 anchors it
in-loop before op1, which additionally reproduces A2's per-iteration
deadlock. The entry pair stays a contract violation under every static
choice.)

Mandated shape: ADOPTION — root's entry token covers the init store and seeds
the iter_args directly, with no semaphore handoff. The exit side is NOT
symmetric: post-loop root code is not implicitly synchronized with the
partitions' loop completion, so the exit handoff stays explicit — the
loop-result token feeds a partition-attributed exit release and root
re-acquires through a dedicated exit semaphore (the A1 exit device). Verified:
actual output has zero protocol ops on the entry path and the explicit
release/acquire pair on the exit path; counts balance at trip count 0 (the
entry token exits as the loop result and feeds the exit release).

### A4 — nested-loop mixed edges (dropped live inner brackets)

Abstracts `insert_semas_nested_carrier.mlir`
@outer_sourceful_alloc_inner_loop_reentry; the multiplicity form is
`insert_semas_release_count.mlir`.

```
for outer {
  op0 {p1}          // outer-top store (load-bearing in the real emission)
  for inner {
    op1 {p2}        // MMA
    op2 {p1}        // load; the inner body's LAST access
  }
  op3 {p1}          // outer-body access after the inner loop, same owner as op2
}
```
Adjacencies at op2: inner-iteration (op2→op1) CUT — demands a release after
op2 on EVERY inner iteration (it feeds the next op1's acquire); inner
through-Exit (op2→op3) UNCUT (same owner) — demands NO release after op2 on
the LAST inner iteration (the hold continues to op3). The static position
"after op2" cannot do both:

- release PRESENT: per-round counts balance, but op3 reads after p1's release
  — a protection gap: the read races the next outer round's op1 (use-after-
  release, the verifier-invariant violation);
- release OMITTED: iteration 1 consumes the semaphore's only credit and op1
  blocks at inner iteration 2 — deadlock.

(`release_count` pins the multiplicity form of the same break: the release
site executes T times per outer iteration — T runtime-dynamic — against an
acquire site that executes once per outer iteration.)

Sound shape (= current pass output, verified by running the pass; the inner
brackets are live, and the cross-owner exit needs the A1 device on top):

```
S_c = sema.create FULL      // outer semaphore (created signaled)
S_b = sema.create empty     // op2/op0 -> op1
S_a = sema.create empty     // op1 -> op2
aq S_c -> %t0               // prologue root acquire
for outer iter_args(%tc = %t0) {
  op0 [%tc] {p1}            // rides the outer carried token
  rel S_b %tc {p1}          // once per outer iter; feeds the inner ENTRY acquire
  aq S_b -> %e {p2}         // entry acquire, hoisted above the inner loop
  %tl = for inner iter_args(%u = %e) {
    op1 [%u] {p2}           // MMA
    rel S_a %u {p2} [tc5mma]
    aq S_a -> %t {p1}
    op2 [%t] {p1}           // load
    rel S_b %t {p1}         // EVERY iteration, incl. iter T
    aq S_b -> %u2 {p2}      // BOTTOM re-acquire by p2 (the rotation)
    yield %u2               // carried; loop token output owned by p2
  }
  rel S_c %tl {p2} [tc5mma] // p2 ADOPTS the inner-loop result token
  aq S_c -> %t3 {p1}        // fresh acquire = op3's token
  op3 [%t3] {p1}
  yield %t3                 // op3's token rides to the next outer op0
}
```

The iteration-T paradox dissolves because op2's iteration-T release feeds
p2's bottom re-acquire — whose token exits the loop and is spent by p2's
S_c release — not op3; op3 is protected by the fresh S_c acquire. Note op3
CANNOT adopt the inner carried token (its owner p1 differs from the carrier
p2): this is A1's rotation at the inner boundary PLUS a cross-owner
release/acquire handoff. Counts balance on all paths, including inner trip
count 0 (the unused entry token passes through and is spent by the S_c
release).

### B1 — conditional consumer (dropped live branch brackets)

Abstracts `insert_semas_conditional_multi_result.mlir` and
`insert_semas_raw_if_token.mlir`.

```
for {
  op1 {p1}                  // producer (MMA), unconditional
  if %cond {                // true on SOME iterations only
    op2 {p2}                // consumer
  }
}
```
v2-literal: static unconditional anchors on the op1 side vs a conditional
release on the op2 side:

```
for {
  aq S_e -> %t {p1}    // unconditional wait...
  op1 [%t] {p1}
  rel S_f %t {p1}
  if %cond {
    aq S_f -> %u {p2}
    op2 [%u] {p2}
    rel S_e %u {p2}    // ...on a CONDITIONAL release
  }
}
```
The failure (verified by trace; partitions run concurrently, coupled only by
the semaphores): on the first not-taken iteration k, `rel S_e` never fires,
so p1's iteration-(k+1) `aq S_e` stalls. p2 is NOT blocked: its next taken
iteration spends the leftover S_f credit from iteration k and reads STALE
data (p1 has only written through iteration k) — the protocol silently slips
one buffer generation per not-taken iteration. The hard deadlock is p1's
`aq S_e` at iteration N+2 (N = total taken iterations): it needs N+1 releases
of S_e but only N can ever fire; this point exists whenever the loop has ≥ 2
not-taken iterations (with exactly one, the loop drains with slipped —
corrupt — reads and one dangling S_f credit). So: **wrong-iteration
consumption first, deadlock second.**

Sound shape (both sides under the same replicated predicate; the producer
side carries the else-yield — confirmed against the
@conditional_multi_result_if_token output):

```
aq S_e -> %t0 {p1}
for iter_args(%t = %t0) {
  op1 [%t] {p1}
  %t1 = if %cond {
    rel S_f %t {p1}
    aq S_f -> %u {p2}
    op2 [%u] {p2}
    rel S_e %u {p2}
    aq S_e -> %t' {p1}
    yield %t'
  } else {
    yield %t             // hold passes through untouched
  }
  yield %t1
}
```
On not-taken iterations NO protocol op fires for this buffer — p1 simply
keeps the hold. Per-taken-iteration counts are +1/−1 on both semaphores;
trip-count-0 safe.

### B2 — same-owner hold spanning a branch join

Abstracts `insert_semas_local_cfg.mlir`
@local_if_consumption_continues_after_join.

```
for {
  op1 {p1}            // producer store
  if %cond {
    op2 {p2}          // conditional consumer
  }
  op3 {p2}            // unconditional consumer AFTER the join, same owner
}
```
v2: the (op2→op3) adjacency is same-owner → UNCUT → one hold {op2, op3}:

```
  if %cond {
    aq S_f -> %u {p2}   // token born inside the then-region...
    op2 [%u] {p2}
  }
  op3 [%u??] {p2}       // ...does not dominate this use (SSA violation);
  rel S_e %u?? {p2}     //    and on the not-taken path op3 runs with NO
                        //    acquire at all (unprotected access)
```
Two distinct failures: dominance (the token would have to be yielded out of
the if, but the else branch has nothing to yield) and an unprotected access
on the else path.

Sound shape: a conditionality cut splits the hold — {op2} branch-resident
(acquire+release inside the branch) and {op3} unconditional (acquire before
op3, release after) — with a mediated handoff so the producer's release
reaches both consumer holds {op2} and {op3} in order; the corpus golden realizes this with a
chained multi-semaphore arrangement (4 semaphores).

### C — footprint vs buffer.id scoping

Abstracts `insert_semas_local_buffer_reuse.mlir`
@local_non_overlapping_aliased_buffers.

```
allocA {buffer.id=402, offset=0,   extent=128}    // accessed only by p0
allocB {buffer.id=402, offset=256, extent=128}    // accessed only by p1
```
buffer.id-keyed scoping fuses these into one two-owner "buffer": per-iteration
stream W_A{0}, R_A{0}, W_B{1}, R_B{1} gets a plain cut at (R_A, W_B) and an
iteration cut at (R_B, W_A) → a live (not deadlocked) 2-semaphore p0↔p1 ring
every iteration — spurious cross-serialization where the corpus mandates NONE
("the pass must leave this function untouched: no semaphores, no buffer
views"). Footprint-keyed scoping: [0,128) and [256,384) are disjoint → two
independent single-owner components → zero cuts → nothing emitted. (The
converse: `insert_semas_mixed_overlap_members.mlir`
@tmem_disjoint_slivers_cross_partition, where ONE buffer.id group dissolves
into FOUR components by footprint.)

## 7. THE HOLD RULE — final form (validated 12jun26, supersedes §5)

After the fork-join execution model was established (user-corrected, then
verified to file:line), the rule was restated in the agreed vocabulary and
re-validated end-to-end: 7 case families against live pass output, 2
adversaries, plus a supplement on the exit-handoff elisions. Result: **5/7
derive ground truth exactly, 1 sound divergence, 1 amendment forced** (the
conditionality cut). This section is the authoritative statement.

### 7.1 Execution-model facts (verified, file:line)

- **Fork-join for root.** warp_specialize entry and exit are CTA-wide
  `bar.sync` index 1 rendezvous (`llvm.nvvm.barrier.cta.sync.all`,
  `ConvertWarpSpecializeToLLVM.cpp:34-60`): root pre-region code
  happens-before every partition's body
  (`WarpSpecializeUtility.cpp:329-333` fork barriers vs `:191` partition
  entry), and root post-region code happens-after every partition's
  WarpReturn (`:196` join arrive vs `:337` default-side join). Between fork
  and join, partitions order against each other **only via semaphores**
  (their internal barriers are private per-partition indices ≥ 2,
  `ConvertWarpSpecializeToLLVM.cpp:103-114`).
- **Joins do not wait for async.** tc5mma/tmem_copy releases lower to
  `TCGen5CommitOp` — mbarrier arrive on async completion
  (`LowerAref.cpp:347-356`); TMA-load releases attach the mbarrier to the
  TMA op itself (`:204-264`). A `bar.sync` orders warps, never in-flight
  async; only a semaphore WAIT absorbs the arrive.
- **Synchronous TMEM stores complete at the op site** (`tcgen05.wait::st`
  unconditionally emitted, `TensorMemoryToLLVM.cpp:468`) — this, plus join
  ordering, is what makes elisions sound for TMEM; `bar.sync` memory
  semantics alone would not be (the repo never emits `tcgen05.fence`).

### 7.2 The rule

Per component (footprint-overlapping allocs + alias specs; single-owner
components emit nothing):

1. **SEQUENCE**: the buffer's real accesses in chain order with owners
   (root for un-stamped ops outside the ws loop). For each For loop, that
   loop's ENTER and EXIT join the sequence as pseudo-accesses of the
   carried owner **iff the loop boundary needs a device — i.e. iff some
   boundary cut pairs a once-per-run outside endpoint against a
   per-iteration inside endpoint** (unequal multiplicities; evaluated per
   loop, never per nest). A once-event outside (E1's post-loop reader,
   acc's init/epilogue) → pseudo-accesses join; an outside endpoint that
   belongs to the same recurring cycle (sequential loops, E4) → no
   pseudo-accesses, the protocol continues on the same semaphores.
   [Corrected from "crosses = any access outside": that over-triggers on
   sequential loops — E4.] The bracket-pair (iteration) adjacency exists
   for **every** loop regardless — it is never gated.
2. **CUT where the execution context changes**: the owner differs, **or**
   the set of enclosing predicates differs (the conditionality cut — the
   one amendment validation forced; without it the same-owner
   branch-to-post-join hold of `local_cfg`
   @local_if_consumption_continues_after_join gets an acquire inside the
   then-region that neither dominates nor protects the else path).
   Maximal uncut runs = **holds**; an uncut bracket-pair adjacency merges
   tail and head into one hold (the carrier).
3. **ANCHOR**: `aq` before the hold's first element; `rel` after its last;
   one pair per hold instance. A pair at a cut renders in the innermost
   region containing the cut — in-branch cuts put both sides under the
   replicated predicate, and a hold surviving a not-taken branch routes its
   token through the if-yield. Exception for ROOT-side anchors: they hoist
   **above** predicates rather than render under them (the entry acquire
   above a conditional init — that hoist is what makes the adopted token
   dominate; verified emission behavior).
4. **BOUNDARY = SSA + fork/join, not extra rules**:
   - the component's first hold begins with the **entry acquire**, emitted
     in root against the initial credit; a cross-owner through-Enter from
     root is realized as **adoption** (the entry token seeds pre-loop
     accesses and the iter_arg init; never a release/acquire pair — the
     fork orders root before all partitions);
   - tokens crossing a bracket pair materialize as yield + iter_arg
     (init = adopted live token, else the entry acquire);
   - a **cut through-Exit**: the upstream hold's last element is EXIT, so
     its release lands after the for, once, in the carried owner's stream,
     consuming the loop-result token (whose in-loop bottom acquire is the
     **absorber** of the final async arrive); the downstream hold acquires
     a fresh **once-flip** semaphore (statically known phase). An **uncut**
     through-Exit is hold continuation: direct loop-result adoption, no
     pair.

### 7.3 Validation results

| case family | corpus anchor | verdict |
|---|---|---|
| P ping-pong (non-crossing) | FA capture m_i / p_tmem writer sides | DERIVES ground truth (the fixup's converted shape; no entry, no carrier, no bottom acquire) |
| same-owner wrap (non-crossing, merge of real accesses) | local_cfg @local_if_conditional_only | DERIVES (carrier + entry-as-init from the merge — crossing not required for carriers) |
| exit-to-root | @specialize_mma_only, root_entry_tmem | DERIVES (absorber = bottom acquire; root pair degenerate; entry adoption) |
| exit-to-partition | live_tag_source, post_ws_read_tag | DERIVES (release-after-for consuming loop result + fresh once-flip acquire; `post_ws_read_tag` even ships the FUSED one-wait form) |
| entry adoption | meta_fa_fwd accumulators | DERIVES (op-for-op, incl. same-owner epilogue adoption) |
| nested / multiplicity | nested_carrier ×3, release_count | DERIVES nested ×3 exactly; release_count diverges soundly (see 7.5.2) |
| conditional | conditional_multi_result, local_cfg join | FnA derives; FnB forced the conditionality cut (now rule 2) |

### 7.4 Exit-handoff refinement (supplement, verified)

Elision table — a **lower-semaphore peephole**, not a rule change (insert-semas
keeps the uniform emission; lower-semaphore, which owns the `needFence`
logic at `LowerAref.cpp:307-334`, lowers join-redundant pairs to their
residue):

| consumer | final rel kind | lowering of the exit device |
|---|---|---|
| root | `[none]` | nothing (join + at-op-site completion); EXCEPT re-emit the proxy fence if the consumer reads SMEM through the async proxy |
| root | async | the absorber wait only; pair → nothing |
| partition p3 | any | full device — **mechanically irreducible** (one WaitBarrierOp per acquire; per-iteration phase flips; no trip-count phase arithmetic anywhere; cross-partition SSA rejected at `PartitionLoops.cpp:364-403`) |

Absorber-placement experiment (perf A/B, flags in `GetEnv.hpp`, legs run
with `TRITON_ALWAYS_COMPILE=1` + per-leg IR fingerprint):

- default — absorber in its owning partition (today's shape);
- `ABSORBER_IN_ROOT_P0=1` — p0 absorbers move to root (zero plumbing: p0's
  phase chain already exits as canonical loop results);
- `ABSORBER_IN_ROOT_ALL=1` — all absorbers move to root (requires yielding
  the other partitions' phase chains).

Feasibility verified in source (AssignStagePhase threads phase state out as
i32 for-results, `AssignStagePhase.cpp:964-968`; `rewriteAcquire` bakes no
static parity, `LowerAref.cpp:276-289`), with three prerequisites: the
last-pid-wins phase fix (`AssignStagePhase.cpp:1108-1137`), an
Exit-group semaphore-adoption path (`InsertSemasSyncDag.cpp:900-953`), and
**peeling/predicating the loop partition's final-iteration bottom acquire**
(count balance: N−1 in-loop + 1 root = N releases) — i.e. a loop-body
protocol change, not a pure peephole. `insert_semas_post_ws_read_tag.mlir`
already ships the fused one-wait-in-root exit and is the implementation
template.

### 7.5 Known divergences and open items

1. **Conditionality cut** — folded into rule 2 (was the single RULE_FAILS).
2. **Multiplicity contracts — RESOLVED as a derivation-error fix (user
   ruling 12jun26: keep the diamond; then shown to DERIVE).** The serial
   chain came from linearizing non-conflicting accesses that have no
   dependence edge; on the ACCESS-DAG substrate (normative for both rules)
   fan-out (one release per successor edge) and fan-in (pending_count=k)
   derive directly. `release_count` / `per_edge_tmem` behavior is
   preserved by construction. See E3.
3. **Two sequential ws loops sharing a buffer — RESOLVED (user, 12jun26):
   continuation on the same semaphores, no boundary device** (see E4, incl.
   the crossing-criterion correction it forced). Residual: add a golden to
   pin pass behavior; zero corpus coverage today.
4. **Mechanical constraints found by the red team** (pre-existing, now
   recorded): AssignStagePhase aborts on protocol ops inside partition-less
   `scf.if` (`assignStateInIfOp` asserts partition metadata — root
   conditional pre/post-loop code must hoist its protocol ops); latent
   crash if one semaphore group ever contains two carriers
   (`propagateStage` DenseMap::at); partition-metadata preconditions are
   asserted by AssignStagePhase but never verified between passes (another
   instance of the recorded oracle gap).
5. **§6 A1 correction**: the irreducible exit device is the
   partition-consumer case (op3 {p3}); for a root consumer the fork-join
   makes the sharing option realizable (= the ABSORBER_IN_ROOT variant) and
   the device degenerates per the elision table. A1's text has been updated
   accordingly.

### 7.6 Pseudo-IR illustrations (E-series; notation as in §6)

The conditionality cut (rule 2's amendment) is already illustrated by §6 B2.
These cover the §7 findings that had no pseudo-IR yet.

#### E0 — carriers do NOT come from crossing (the red-team misreading)

```
for {
  op1 {p1} ; op2 {p2} ; op3 {p1}
}                                  // NOTHING outside: crossing = NO
```
No pseudo-accesses join the sequence — but the bracket-pair adjacency of the
REAL accesses still exists, (op3 → op1) is same-owner → **merge** → hold
`(op3 | op1)` → carrier iter_arg + entry acquire as its init:

```
aq S → %t0                  // entry: the merged hold's init, not a "boundary device"
for iter_args(%t = %t0) {
  op1 [%t] {p1}
  rel ; aq {p1→p2} ; op2 {p2} ; rel ; 
  aq S → %t' {p1}           // before op3 — the merged hold's anchor
  op3 [%t'] {p1}
  yield %t'
}
```
Crossing gates only ENTER/EXIT pseudo-accesses; merges of real accesses
produce carriers regardless. (Validated: `local_cfg`
@local_if_conditional_only derives ground truth exactly.)

#### E1 — exit-to-root: the uniform device, with the degenerate part marked

```
for {
  op1 {p1}          // consumer-side access (load/store acc)
  op2 {p2}          // producer (MMA, async completion)
}
op3 {root}          // post-loop read by ROOT: crossing = YES → brackets live
```
Cuts: (op1,op2); (op2,EXIT) — the absorber handoff; bracket pair
(EXIT,op1) same-owner → merge → carrier; (EXIT,op3) through-Exit cross-owner
→ exit pair. Emission (= the actual `@specialize_mma_only` output):

```
S_a = create true ; S_x = create false       // S_x: exit semaphore
%t0 = aq S_a                                 // entry (root)
%tN = for iter_args(%t = %t0) {
  op1 [%t] {p0} ; rel S_b {p0}
  aq S_b {p1} ; op2 {p1} ; rel S_a [tc5mma] {p1}
  aq S_a → %t' {p0}        // bottom acquire — the ABSORBER: its iteration-N
  yield %t'                //  firing waits out the LAST MMA's async arrive,
}                          //  so p0 cannot reach the JOIN before MMA(N) lands
rel S_x %tN                // ← both in root, after the join:
%v = aq S_x                // ← zero synchronization, pure token bookkeeping
op3 [%v] {root}
```
Load-bearing: the absorber (async is invisible to the join's bar.sync).
Degenerate: the S_x pair (the join already orders root after p0). Hence the
elision rows:

```
final rel [none] :  elide absorber AND pair    (join + at-op-site completion)
final rel async  :  keep absorber, elide pair
consumer = p3    :  keep everything            (§6 A1 — no join mid-region)
```

#### E2 — ABSORBER_IN_ROOT: the variant and why the peel is mandatory

```
for {
  op1 {p1}          // same pre-sync IR as E1
  op2 {p2}          // producer (MMA, async completion)
}
op3 {root}          // post-loop read by ROOT
```
Same cuts/holds as E1 — only the ABSORBER's placement changes: instead of
the carried owner's bottom acquire firing on the last iteration, root
absorbs the final async arrive after the join (root's wait parity comes
from the loop-result phase chain per `AssignStagePhase.cpp:964-968`):

```
%t0 = aq S_a                                 // entry (root)
for iter_args(%t = %t0) {                    // bottom acquire fires N−1 times
  ...
  rel S_a [tc5mma] {p1}
  aq S_a → %t' {p0}  ONLY for i < N−1        // ← THE PEEL (predicate or peel
  yield %t'                                  //    the last iteration)
}
%v = aq S_a {root}                           // root absorbs rel S_a(N), post-join
op3 [%v]
```
Count ledger on S_a — why the peel is not optional:

```
credits:  init(1) + N releases                      = N+1
waits:    entry(1) + (N−1) bottom + 1 root          = N+1   ✓ balanced
without peel: entry(1) + N bottom + 1 root          = N+2   → one wait starves
                                                              → DEADLOCK
```
So the flag is a loop-body protocol change (peel/predicate), not a post-loop
peephole. Perf rationale: p0 reaches the join (and its next persistent tile)
without blocking on MMA(N); the MMA latency hides behind the join.

#### E3 — the release_count divergence: fan-in vs serial chain

```
for {
  op1 {p3}          // producer store
  op2 {p0}          // reader A ┐ non-conflicting accesses (disjoint
  op3 {p1}          // reader B ┘ slices) — no order needed between them
}                   // nothing outside
```
Ground truth (`insert_semas_release_count.mlir` — see the file for the full
multiplicity wiring): the producer's regain is a **fan-in** — ONE acquire
gated on BOTH readers via pending_count, leaving the readers unordered:

```
S = create {pending_count = 2}
for {
  aq S → %t {p3}             // single wait, completes after BOTH arrives
  op1 [%t] {p3}
  ... ; op2 {p0} ; rel S {p0}     // arrive 1 ┐ op2, op3 free to finish
  ... ; op3 {p1} ; rel S {p1}     // arrive 2 ┘ in EITHER order
}
```
THE RULE emits a serial chain instead (cuts are pairwise — (op2,op3) is
cross-owner, so it becomes a handoff):

```
for {
  aq S2 → %t {p3} ; op1 [%t] {p3} ; rel S1 {p3}
  aq S1 {p0} ; op2 {p0} ; rel S0 {p0}        // ← imposes op2 → op3,
  aq S0 {p1} ; op3 {p1} ; rel S2 {p1}        //    an order GT doesn't have
}
```
Both sound; the difference is concurrency: fan-in leaves the readers free,
the chain orders them — i.e. the serial chain is a REGRESSION.
**RULING (user, 12jun26): keep the diamond** — and on inspection the
ruling is a DERIVATION ERROR FIX, not added rule content: the serial chain
came from linearizing op2/op3, which are non-conflicting and therefore
have NO dependence edge — no adjacency, no cut between them. On the true
substrate (the ACCESS-DAG) the rule derives the diamond directly:

```
        op1 {p3}
       /        \           edges: op1→op2, op1→op3,
  op2 {p0}    op3 {p1}             op2→op1@next, op3→op1@next;
       \        /           op2↔op3: NO edge (non-conflicting)
        op1 {p3}
```

Two cuts out of op1's hold → one release per edge (fan-out); two cuts into
op1@next's hold → one acquire with pending_count = 2 (fan-in). The current
rule has exactly the same status: its one-sentence statement read linearly
also serializes; the shipped pass emits the diamond because it runs on the
DAG. Both rules inherit fan-in/out from the substrate — no ruling needed
beyond "the DAG is normative; the linear chain in examples is shorthand,
exact only when all accesses pairwise conflict."

#### E4 — two sequential ws loops sharing a buffer (RESOLVED: continuation; golden still wanted)

```
for A {
  op1 {p1} ; op2 {p2}     // ws loop A
}                          // nothing between the loops
for B {
  op3 {p1} ; op4 {p2}     // ws loop B, same buffer
}
```
RESOLVED (user, 12jun26): no boundary device at all — the cycle simply
**continues on the same semaphores** across the loop boundary:

```
for A {
  aq S1 {p1} ; op1 ; rel S2 {p1}
  aq S2 {p2} ; op2 ; rel S1 {p2}
}
for B {
  aq S1 {p1} ; op3 ; rel S2 {p1}    // ← B's first aq S1 pairs with A's
  aq S2 {p2} ; op4 ; rel S1 {p2}    //   last rel S1: ordinary cut, and the
}                                   //   async absorber if A's rel is async
```
Ledger: S1 credits = init(1) + N_A + N_B vs waits N_A + N_B ✓ (standard
one-surplus). Phases: p1's chain continues (N_A waits then N_B more) —
AssignStagePhase's linear state threading across sequential ops, rebinding
through each for's results, is built for exactly this.

This case exposed an over-trigger in the §7.2 crossing criterion as first
stated ("any access outside the loop" fires here and would bolt on
unnecessary machinery). Corrected criterion: **the boundary device bridges
unequal multiplicities** — needed iff a boundary cut's outside endpoint is
a once-per-run event pairing against a per-iteration event (E1's op3); when
the outside endpoint belongs to the same recurring cycle (sequential
loops), multiplicities match and the protocol continues with no device.
acc remains covered (init/epilogue are once-events → adoption/exit device).
A golden should still be added to pin pass behavior (§7.5.3), but the
design question is closed.

## 8. Provenance

Workflow runs: `wf_981a472a-51e` (42 agents, corpus sweep + red team +
adversarial verification of v2); `wf_d5acc22f-a5b` (4 agents, pseudo-IR
example verification — corrections folded into §6); `wf_6997b31c-322`
(11 agents, final-form validation: fork-join/async facts + 7 case families
+ 2 adversaries — §7); `wf_f587b052-ea5` (2 agents, exit-handoff elision +
absorber placement — §7.4). Full structured results under
`/tmp/claude-25502/.../tasks/` (machine-local). Ground truth for every
comparison: `triton-opt --nvws-insert-semas` at HEAD (`c31a6b7a37`) and the
047/048 FA capture. NOTE: the archived 048 dump predates the in-tree
point-of-use fixup commits — the P-pingpong validation re-ran the current
pass on the 047 input instead of trusting 048.
