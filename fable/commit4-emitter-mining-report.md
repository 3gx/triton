# Commit-4 emitter mining report

Proactive mining of the OLD emission rules (per the §4c old-reference
obligation): all six `plans/` emitter documents, the old emitter source
(`git show 5cfe0ac6e7^:.../InsertSemasEmitter.h`), `InsertTmemSemaphore.cpp`,
and the emitted-IR ground truth
(`logs/new-insert-semas/commit3/before-insert-semas-pmatmul-prev-semas.out`),
diffed against spec §6 and plan commit-4. Produced by a background explorer,
2026-06-10; verdicts to be ratified by the user before commit-4 code.

## Ground-truth op catalog (from the old pass's emitted IR)

- **create**: `nvws.semaphore.create %backing true|false : <[memdesc...]>` at
  function level right after each backing alloc; NO partition/stage/tag attrs
  ever. Backing allocs are token-less re-creations; TMEM backing keeps
  allocShape (`1x128x256`), local backing gets leading `1x`.
- **acquire**: bare when root-owned; `{ttg.partition, ttg.warp_specialize.tag}`
  when partition-owned OUTSIDE the WS loop; `{loop.cluster, loop.stage,
  ttg.partition}` and NO tag inside the loop.
- **buffer**: `nvws.semaphore.buffer %sem, %tok {loop.cluster, loop.stage,
  ttg.partition}` — stamped with the CONSUMING ACCESS's partition/stage, not
  the acquire's. TMEM view keeps allocShape suffix; SMEM view has none.
- **release**: `nvws.semaphore.release %sem, %tok [#nvws.async_op<...>]
  {loop.cluster, loop.stage, ttg.partition}`.
- **token threading**: appended `!ttg.async.token` iter_args; yields stamped
  with UNION-extended `ttg.partition`; for-op `ttg.partition.outputs` extended
  one owner-stamped array per slot; if-results with `ttg.partition.outputs`.
- **workaround shape**: result-less exit-if holding the release; `ub.poison`
  INSIDE the loop body feeding the middle if's dead slot; acquire-if yielding
  new token / old carrier.
- **poison**: old pass minted one `ub.poison` per replaced site/group at
  function level (new design: ONE per function — deliberate change).
- Access ops keep empty dep brackets `op %buf[]` with dead tokens.

## GENUINE GAPS (need spec/plan amendments before commit-4 code)

1. **If-split workaround reference is WRONG in the plan.** The behavior gate 1
   depends on is the old `splitSemaphoreIfForLoopScheduler`
   (`InsertSemasEmitter.h:2793-2999` at `5cfe0ac6e7^`), NOT
   `InsertTmemSemaphore.cpp:1640` (which hard-codes partitions 1/0 and only
   matches release-first/acquire-last then-blocks). The old rule set includes:
   release not necessarily first (skips ConstantLike + alias ops); ELSE-branch
   splits; RELEASE-ONLY splits (TMEM semaphore + later acquire in the same
   branch → exit-if only); acquire-first-then-block with
   release-immediately-preceding-the-if; full-split REFUSAL when the TMEM
   semaphore has multiple member base types; exit-if stage = release's stage
   with `inferPrecedingMmaStage(ifOp)` fallback; enter/exit partitions =
   union of release+acquire owners (owner-derived, never hard-coded);
   middle-if partitions = original minus enter/exit ids with per-result
   outputs; `setPartitionOutputs(exitIf, {})`.
2. **In-loop poison exemption.** The workaround legitimately creates
   `ub.poison` inside the loop body; plan step 1's "no poison ops are ever
   created inside loops" must be scoped to the token-nuke pre-process only.
3. **Buffer-view materialization is per-region/per-access-site.** Views are
   (re)materialized lazily at the requesting op, stamped with THAT access's
   partition + stage/cluster; the view cache is cleared at every acquire AND
   at every region entry/exit (a carried token gets a fresh view per region).
   One `SemaphoreBufferOp` yields ALL member views at once for multi-member
   semaphores. Spec §6's "emit/cache at the Acquire" and contract I's
   "identical to the acquire" are wrong for carried tokens.
4. **View-type rules.** Local member view type = the access site's own memdesc
   type after walking leading `memdesc_index` alias steps, mutability forced
   true; alias-chain replay SKIPS `memdesc_index` steps whose result type
   equals the current view type and forces each cloned view op's result
   mutability to the source's. TMEM views retain allocShape.
5. **Yield partition stamping is UNION-EXTEND**, never assignment: start from
   the yield's existing `ttg.partition` ids and add the token owners. (Plan
   step 4's "yield partition attrs := slotOwner" would drop existing ids.)
6. **Descriptor-sourced local alloc** (managed `ttg.local_alloc %src` where
   src is a `tt.descriptor_load/gather`): old pass converted to
   `nvws.descriptor_load/gather` writing the view directly; new contract D
   would round-trip through registers via local_store. No corpus input
   exercises it today; needs an explicit decision (keep contract D and
   document, or adopt the conversion).
7. **Scalar-sourced local alloc** → `triton::SplatOp` (owner+stage stamped)
   then local_store. Contract D assumes tensor sources; one sentence fixes it.
8. **Synthesized sourceful-store triple stamping**: the emitted store (and
   splat/constant true) inherit the ORIGINAL alloc's owner + stage/cluster.
9. **RAUW discipline for replaced alloc values**: replace only uses DOMINATED
   by the view; exclude the new store and `SemaphoreCreateOp` users; track
   rewritten values so later retargeting matches the original OR the
   already-rewritten value. (Without the dominance filter: use-before-def.)
10. **Process (plan §5)**: no pytest until gate 1 passes unmodified; a 60s
    timeout means a hang CAUSED BY THE CHANGE — root-cause it, never retry or
    broaden.

## ALREADY COVERED (verified, no action)

- Outside-WS-loop tag stamping + reasons (spec §6 has it; keep
  insert_semas_post_ws_read_tag on the verification list).
- Stage/cluster per-op rules, per-partition cache, omit-fallback, verifier
  hook (contract I), modulo the buffer-op nuance in gap 3.
- Mechanical placement law incl. release-after-terminator ban and
  stop-and-report (spec §6 hard rules).
- M1/M2/M3, deterministic create order (spec §5.3 + ground rule 6; M1 relaxed
  to per-component entries by design).
- Create anchors (TMEM hoist vs local last-alloc): both land before the WS
  loop on the whole corpus; relative interleaving = acknowledged golden churn.

## SUPERSEDED BY NEW DESIGN (deliberate, documented)

- Linear-chain → compact FULL/EMPTY two-semaphore lowering (replaced by
  group-by-destination, spec §5.2; golden churn deferred).
- replToken slot-filling / carrier-slot picking / slot normalization /
  original-slot reuse (contract E rejects reuse by design; one slot per live
  component from stage-3 Crossing facts).
- Per-site poison multiplicity (new: ONE function-level poison).

## Open items from the commit-3 v4 verification round (user rulings needed)

A. **`@nested_loop_no_double_buffer` numStages=2 vs golden 1x.** The verbatim
   veto-chain ruling (pattern gate dropped) double-buffers a shape the old
   pass single-buffered ONLY via the dropped gate (consumer-side in-loop
   store). The in-tree golden pins `1x128x128`. **RULED (user): 2x is
   CORRECT** — the dropped gate was a condition InsertTmemSemaphore needed
   for its own pipeline, not something this pass requires; the golden
   regenerates to 2x at commit 4.
B. **Sync-free TMEM groups inflate the capacity budget.** The veto chain
   evaluates a scales group against its MMA's ACCUMULATOR shape with
   cumulative numTmemBlocks, so identical immutable, semaphore-less scales
   groups get order-dependent stages (2,2,1 observed) AND their 2x
   accounting pushed a later real accumulator group below capacity.
   Candidate rule: groups with zero semaphores are untouched at emission —
   exclude them from numStages/capacity accounting entirely.
C. **Carrier crossing by SSA capture vs threading.** An inner consumer loop
   with no acquires gets no token slot (the crossing rule), yet its in-loop
   MMA consumes the carrier acquired before the loop; the old pass threaded
   the token through the inner loop regardless. At emission the buffer view
   token would cross the region boundary by plain SSA capture. Decide:
   bless capture (MLIR-legal; check partition-loops tolerates it) or widen
   the crossing rule to "region USES the carrier", restoring old threading.
