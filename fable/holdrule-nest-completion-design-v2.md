# Holding regions for arbitrary loop nests (design v2)

This is the generic extension of the holding-region construction (THE HOLD
RULE, `rule-v2-corpus-verification.md` §7.2) from a single loop to an
arbitrary loop nest — perfect or imperfect, any depth. There are no
per-kernel cases, no per-depth cases, and no fall-back-to-gated paths: one
uniform recursive construction places every component.

Scope of this document: the PLACEMENT construction (where each acquire and
release goes). Downstream realizability of a placement (stage/phase
threading, pending-count, TMA arrive locality) is a separate concern and is
NOT a placement exception — see §6. Performance is taken as given: a
smaller critical region is the goal; this document does not argue it.

Verified against `logs/nested-12jun26-v1/`: `grouped_gemm` (3-level nest),
`attn_persistent` (2-level), `matmul_persistent_flatten0` (2-level,
flatten=false), plus the persistent-FA captures `insert_semas_meta_fa_fwd`
and `pfa-before-insert-allocas`.

## 1. The holding region (recap)

Per component (a buffer plus its footprint-overlapping aliases;
single-owner components emit nothing):

- A **hold** is a maximal run of the component's accesses under one
  **execution context** — same owner AND same set of enclosing predicates.
- A **cut** falls where the execution context changes: the owner differs
  (a producer→consumer handoff), or the enclosing-predicate set differs (a
  conditional).
- Each hold is one **critical region**: bracket it with exactly one
  `acquire` before its first access and one `release` after its last.

That is the whole idea, and it does not mention loops. A loop is just a
region the accesses sit in; the construction below makes loops participate
without changing the idea.

## 2. The generic recursive construction

For one component, over the whole nest:

1. **SEQUENCE.** Walk the component's real accesses in program order across
   every nest level, each tagged with its owner and its enclosing region
   chain (which loops/ifs contain it). A loop participates in an enclosing
   level's sequence as a single **pseudo-access pair** (its ENTER and EXIT)
   of the owner that carries the component across that loop — present
   exactly when a hold spans the loop (the component is held across the
   whole loop by one owner) or when the loop boundary pairs endpoints of
   unequal multiplicity (a once-per-ENTER/EXIT endpoint outside against a
   per-iteration endpoint inside). A loop whose every adjacency is the same
   recurring cycle contributes no pseudo-access — the protocol simply
   continues across its iterations.

2. **CUT** where the execution context changes (owner or predicate set).
   Maximal uncut run = one hold. At a loop, the iteration adjacency
   (last-access-of-iteration → first-access-of-next-iteration) is itself a
   cut when those two accesses differ in context, and a continuation when
   they do not.

3. **ANCHOR.** `acquire` immediately before the hold's first element, in
   that element's region; `release` immediately after the hold's last
   element, in that element's region; one pair per hold. The acquire and
   release **need not sit at the same nest level**: a hold whose first
   access is at an outer level and whose last access is inside an inner
   loop opens its region at the outer level and closes it inside the inner
   loop. The phrase "innermost region containing the cut" applies to the
   **cut point** between two consecutive holds — where an upstream hold's
   release meets a downstream hold's acquire, that release/acquire sits at
   the cut, in the lowest region enclosing both sides of it (a plain walk
   up the region chain to their common ancestor — per cut, never per
   nest). It is not a claim that one hold's two ends share a region.

Steps 1–3 are applied identically at every level. Nothing in them is
depth-specific. The placement of a hold is therefore determined by where
its first and last elements live, not by how deep the kernel nests.

## 3. Where a hold's anchor lands (consequences, not cases)

The placement law in §2.3 yields three shapes. These are **not special
cases** — each is the same law evaluated for a hold whose elements happen
to sit at a particular level:

- **Hold entirely inside the innermost loop** → its first and last
  elements are inside that loop → anchor inside it. The hold recurs across
  the loop's iterations; the iteration adjacency carries it. No token
  leaves the loop.

- **Hold spans an inner loop** (one owner holds the buffer across the whole
  inner loop — e.g. a value produced once in the outer body and read every
  inner iteration; or an accumulation written every inner iteration and
  read once after). Then the inner loop's ENTER/EXIT are the hold's
  first/last pseudo-access at the enclosing level → the pair anchors at the
  **enclosing level**, before and after the inner loop. The acquire is not
  placed inside the inner loop. This is the case a single-loop view cannot
  express: a per-inner-iteration acquire against a once-per-outer-iteration
  release does not balance (verified on `attn_persistent`'s `q`: produced
  in the outer body {p2}, read every inner iteration {p1} → one acquire
  above the inner loop, release after it).

- **Hold spans a loop's iteration bracket with the same owner** → the value
  threads that loop as a carried iter_arg.

A loop need not be carried by a single owner. When the buffer is handed
between owners **across** an inner loop — held by one owner at the loop's
entry, a different owner at its exit (a ping-pong loop) — the carried owner
**differs at the loop's entry vs its exit**. The construction connects the
outer prefix to the inner loop's first access and the inner loop's last
access to the outer suffix: the outer-level holds bracket the inner loop on
each side, while the in-loop holds ping-pong within it. (Verified on the
n-level form: an outer same-owner prefix merges with the first
inner-iteration access of that owner, up to the first owner cut; the inner
loop then ping-pongs; the outer suffix resumes after the loop's exit-owner
access.)

One component composes these freely up its nest, each boundary
independently. Worked example — the accumulator of `grouped_gemm` (3-level:
ws group-loop ⊃ tile-loop ⊃ k-loop), one op per line, owners in braces:

```
acc = tmem_alloc init {root}                 // L0 (root), before the ws loop
for g (ws) {                                  // L0 bracket
  for tile {                                  // L1
    for k {                                   // L2
      mma acc {p1}                            //   written every k-iteration
    }                                         // L2 EXIT
    read acc {p0}                             // L1: read once per tile, after k
  }
}
```
- L0 (ws bracket): root init → first use is the mma {p1}, cross-owner from
  root → **adoption** (the init value seeds the carried iter_arg; the fork
  orders root before the partitions, so no pair).
- L1 (tile bracket): acc cycles per tile with the same owner → **carried**
  iter_arg across the tile bracket.
- L2 (k boundary): the mma-write hold ({p1}, spanning the whole k-loop)
  ends at k-EXIT; the reader {p0} is in the tile body, once per tile →
  cross-owner handoff at unequal multiplicity → the pair anchors in the
  innermost region containing the cut = the **tile body**: release after
  the k-loop, acquire before the read. The in-k re-acquire that seeds the
  next k-iteration is where the final async write is awaited.

`matmul_persistent_flatten0` is the same composition with the tile level
removed (2-level); `attn_persistent` is the same with the accumulator's
reader being the per-tile epilogue. The construction is identical; only the
depth differs.

## 3.5 Per buffer; single-loop and perfect nests are the degenerate cases

The construction is **per component** (per buffer): each buffer places on
its own access set, independently of the others in the kernel.

- A buffer whose accesses all sit in one loop places entirely at that loop
  — the enclosing loops have no access of it, so they contribute no
  pseudo-access (no outside endpoint) and carry no acquire/release for it.
  Its emitted sync is identical to a single-loop kernel's. This holds **per
  buffer**, even inside an otherwise-imperfect kernel (verified:
  `grouped_gemm`'s A/B tiles are inner-confined → pure in-loop ping-pong,
  the enclosing tile- and group-loops transparent to them).
- A **perfect nest** (every buffer confined to the innermost loop) is
  therefore the case where every buffer emits single-loop-like and the
  outer loops carry no sync at all.
- **Single-loop is the depth-1 instance of this same construction — not a
  special case.** Run §2 on a one-level nest and the recursion has nothing
  deeper to descend into: one level to sequence, cut, and anchor over. The
  multi-level work engages **only** for buffers genuinely accessed at more
  than one level (the imperfect-nest buffers — e.g. an accumulator with an
  outer-level epilogue). This **inverts** the prior realization, where
  single-loop was the only handled shape and nested crossings were bailed
  to the rotated device; here there is one construction and single-loop is
  its smallest instance.
- **Single-loop preservation is by construction, not a caveat.** Because §2
  at depth 1 is the existing single-loop rule and the removed nested
  short-circuits never fire on a single-loop kernel, flat kernels emit
  exactly as before. The flat lit goldens staying byte-identical is the
  *prediction* that confirms the **implementation** faithfully realizes
  this construction — a churn means a coding bug, not a design change.
- Plumbing note: an inner-confined buffer that is *allocated* outside the
  loops still threads its value inward as a carried iter_arg / adoption
  through the outer brackets — SSA wiring, not a critical-region device; no
  acquire/release is added at the outer level.

## 4. Boundary realization is SSA, not extra rules

The brackets above are realized in SSA, the same way at every level:

- A component's first hold opens with the **entry acquire** against the
  semaphore's initial credit; a cross-owner entry from root is realized as
  **adoption** (the live value seeds the first hold and the carried
  iter_arg — never a pair; the fork already orders root first).
- A hold spanning a loop's iteration bracket materializes as a carried
  iter_arg (init = the adopted value or the entry acquire), with the
  in-loop re-acquire seeding the next iteration.
- A cross-owner hold whose last element is a loop EXIT releases after that
  loop, once, in the carried owner's stream (the in-loop re-acquire awaits
  the last async write); the downstream hold acquires a fresh
  statically-phased semaphore. A same-owner continuation through a loop
  EXIT carries the value directly, no pair.

## 5. The classification is per (component, boundary)

`gateCrossing`'s current binary verdict (point-of-use vs the rotated
single-loop device) becomes, per (component, loop boundary), the §2
construction's outcome for that boundary: in-loop hold, spanning-inner-loop
pair anchored one level out, or carried bracket. The two short-circuits
that today force the rotated device for any non-ws or nested-region
crossing (`non-ws-loop`, `nested-final`) are removed: a crossing over an
inner loop is classified by §2, not bailed out of. The existing
outside-endpoint scans already compute the facts §2 needs (the multiplicity
test and the owner/predicate of each endpoint); what is added is applying
them at every level — anchoring each hold's acquire/release at its
elements' regions (which may differ in level) and each cut's
release/acquire at the innermost region containing that cut.

## 6. No placement exceptions

There is no fall-back-to-gated. Every component places by §2. Two items
that earlier drafts treated as gated fallbacks are not placement
exceptions:

- A TMA-load producer's accesses (the loads) and the release that publishes
  them are one hold under one owner; §2 anchors that hold where the loads
  are, so the release is never separated from its loads. There is nothing
  to forbid — the construction co-locates them by definition.
- The one-carried-value-per-semaphore-group requirement of the downstream
  stage/phase assignment is a property of the REALIZATION, not of the
  placement. If a placement the construction produces is not directly
  representable downstream, that is a realization gap to close in the
  realization (or a hard diagnostic naming the component) — never a reason
  to widen the critical region back to the rotated form. Per the standing
  scope, downstream passes (stage/phase, pending-count, pipelining) are out
  of scope for this document; they do not constrain the placement here.

## 7. Verification against the corpus

Traced in `logs/nested-12jun26-v1/`:

- `grouped_gemm` (3-level): A/B smem = in-k holds; acc = adoption(L0) +
  carried(L1) + spanning-inner-loop handoff(L2 anchored at L1). Places
  unambiguously.
- `matmul_persistent_flatten0` (2-level): same acc composition without L1;
  A/B in-k. Places unambiguously.
- `attn_persistent` (2-level): K/V/p/qk in-loop holds; `q` =
  spanning-inner-loop (the single-loop view would not balance); acc =
  adoption + in-loop + spanning-inner-loop handoff to the epilogue. Places
  unambiguously.
- `insert_semas_meta_fa_fwd`, `pfa-before-insert-allocas`: additional
  persistent-FA bodies for cross-checking the same construction.

Every component in the genuine nests places by the single §2 construction,
at depths 2 and 3, perfect (in-loop ping-pong) and imperfect
(prologue/epilogue spanning an inner loop). No component required a special
case or a gated fallback.

## 8. References

- Rule: `rule-v2-corpus-verification.md` §7.2 (cut/hold/anchor) and §7.6
  (the E-series illustrations).
- Bodies: `logs/nested-12jun26-v1/`.
- Implementation: `extend-design-to-nested-loops-plan-v2.md` (the plan
  carries the code touch-points and gates; this document is the
  construction only).
