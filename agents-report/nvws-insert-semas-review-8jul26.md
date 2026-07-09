# NVWS InsertSemas final refactor review — 8 Jul 2026

## Executive summary

**Verdict: approved; no blocking findings remain.**

The final working-tree refactor preserves the placement-invariant performance
fix from `a207d5ba2b`, resolves all four correctness concerns from the initial
review, and substantially simplifies the pass without changing any lit test.

- `InsertSemasSyncDag.cpp` is **1,989 lines**, down from 2,666 (`-677`, 25%).
- The InsertSemas implementation is **1,010 lines smaller**; including the
  deleted CMake source entry, the tracked pass-wide diff is **-1,011 net lines**.
- `InsertSemasOwnerDag.cpp` is gone; owner analysis is folded into access-DAG
  construction.
- Build passes, NVWS lit is **118/118**, and the full suite is clean: **502
  passed, 13 expected failures, 9 unsupported** out of 524.
- `git diff --name-only -- test/NVWS` is empty. No test was added, removed, or
  edited for this refactor.

The two remaining edge reducers were deliberately retained. They are not
duplicate cleanup passes: one proves token-sensitive straight-chain
transitivity and the other handles structured loop-closing recurrence edges.
Removing either changed existing outputs and broke established lit contracts.
The refactor instead removes their duplicated plumbing while preserving the
distinct proofs.

## Scope

I reviewed the working-tree refactor against `1aebbadacd`, with particular
attention to the performance regression introduced by `715e024eab`, the
placement-invariant correction in `a207d5ba2b`, and the original review
findings concerning recurrence scheduling, covered-sender counts, release
placement, and async completion payloads.

Changed implementation files:

- `InsertSemas.cpp`
- `InsertSemas.h`
- `InsertSemasAccessDag.cpp`
- `InsertSemasEmitIR.cpp`
- `InsertSemasSyncDag.cpp`
- `InsertSemasOwnerDag.cpp` (deleted)
- `CMakeLists.txt`

No design document or test expectation was changed by this refactor.

## Correctness and performance review

### Placement invariance is preserved across every reduction stage

The walk remains fact-complete: covered edges are annotated, not suppressed.
Before either legacy reducer runs, `buildEdgesAndSemas` records the latest raw
source for each `(destination, destination owner, source owner)` handoff. The
same-sender merge then uses that independent release floor even if an edge was
removed as ordering-redundant.

Consequently, deleting a redundant edge can reduce an arrival count or remove
an entire handoff, but it cannot move a surviving sender release earlier. This
closes the broader pre-merge counterexample from the initial review, not only
the motivating FA EXIT-edge case.

### Covered-sender deletion remains conservative

A sender is removed only when all of its merged contributions are covered and
both covering legs remain valid. Candidate covers are refined to a survival
fixpoint. Loop-entry and recurrence handoffs that share one fixed-count
semaphore are additionally refined together, reviving a covered sender if
independent pruning would make their pending counts incompatible.

This resolves the previously reproduced legal 2-arrival entry / 1-arrival
recurrence failure.

### Async completion cannot be erased as mere ordering redundancy

Exact-source async RAW/WAW/WAR/EXIT edges retain the `preserve` guard, and both
the straight-edge and loop-close reducers honor it. An alternate path that
proves program order therefore cannot erase the only TMA/MMA completion wait.

`preserve` is intentionally retained as a payload-safety property; it is not
part of covered-sender placement logic.

### Recurrence placement and live waits are represented separately

The old per-node nullable `release->sat` relation is replaced by `ProtocolArc`.
Each arc keeps immutable acquire/consumer placement facts for physical-slot
replay and a separate mutable `wait` relation for alias and owner scheduling.
When point-of-use lowering moves or detaches an acquire, only the live wait is
retained, retargeted, or cleared under the original same-chain rules; slot
provenance is not lost or mistaken for a live schedule dependency. A nested
closing release creates a live recurrence wait, while a synthetic entry
closure has no schedule arc and bridges remain schedule-inactive. Canonical
root-owned acquires still participate in physical-slot replay.

This removes stale per-node cross-links while preserving the
read-to-next-iteration recurrence dependency identified in the initial review.

The compact point-of-use planner also retains two small but semantic rules:

- a carried `scf.for` result keeps the incoming semaphore channel, including
  the zero-trip path;
- a planner-created post-loop acquire is explicitly marked, inherits the
  cached same-owner boundary schedule, and cannot be reconsidered as an
  enclosing loop's recurrence exit.

Region completion is also summarized path-sensitively. Branches with different
terminal completion stages—including scheduled versus unscheduled fallbacks—
retain their carried flow instead of being accepted according to traversal
order. This summary is stored independently of optional token flow, so lowering
a child cannot erase the completion facts needed by its parent; pruning a dead
conditional clears the corresponding summary with its flow.

Those rules preserve the established sequential-loop and meta-FA schedules
without broad fallback heuristics.

## Simplification assessment

The pass now follows a clearer four-stage structure:

1. collect groups and build access/owner facts;
2. build the complete sync DAG, reduce handoffs, and plan region flows;
3. finalize physical-slot and pipeline schedule constraints;
4. render semaphore IR and token/capability flow.

The main reductions are substantive rather than cosmetic:

- owner-DAG construction was folded into the access-DAG traversal;
- `ProtocolArc` replaces parallel mutable scheduling links and their repair
  logic;
- physical backing domains are planned and materialized once;
- region token flow is represented by one compact `RegionFlow` summary and a
  direct point-of-use lowering path;
- render-time capability state replaces duplicated late repair/coalescing;
- verifier walks and helper layers that restated construction invariants were
  removed;
- custom DAG-dump infrastructure was replaced by the existing IR dump switch.

The remaining straight-edge reduction, loop-close reduction, same-sender
merge, and covered-sender refinement operate at different proof granularities.
Combining them today would be an output-changing algorithm rewrite, not a safe
dead-pass deletion.

## Line-count ledger

| File | `1aebbadacd` | Final | Net |
| --- | ---: | ---: | ---: |
| `InsertSemas.h` | 399 | 403 | +4 |
| `InsertSemas.cpp` | 65 | 59 | -6 |
| `InsertSemasAccessDag.cpp` | 458 | 417 | -41 |
| `InsertSemasOwnerDag.cpp` | 163 | 0 | -163 |
| `InsertSemasSyncDag.cpp` | 2,666 | **1,989** | **-677** |
| `InsertSemasEmitIR.cpp` | 1,543 | 1,416 | -127 |
| **Implementation total** | **5,294** | **4,284** | **-1,010** |

The CMake source-list deletion contributes one additional removed line, making
the full tracked pass-wide result **1,666 insertions / 2,677 deletions = -1,011
net lines**.

## Verification

Performed on the final source, in the repository-required order:

1. `ninja triton triton-opt` — passed.
2. NVWS lit — 118/118 passed.
3. Full lit — 524 discovered; 502 passed, 13 expected failures, 9 unsupported.
4. Clang-format 19.1.6 dry run with `--Werror` — clean.
5. `git diff --check` — clean.
6. `git diff --name-only -- test/NVWS` — empty.

No pytest was run, per repository instructions.

I did not rerun the B300 performance benchmark or independently reproduce the
reported cubin hashes. The supplied performance data remains the empirical
evidence; this review verifies that the source-level placement invariant that
prevents the regression is retained through all edge-reduction stages.

## Suggestions

No further change is required for this refactor. In particular, do not remove
the release-floor table, async `preserve` guard, shared-count refinement, or
post-loop-acquire marker merely to save a few lines; each carries a distinct
correctness obligation.

If further reduction is pursued separately:

1. Replace the two topology-specific reducers and handoff refinement with one
   exact global reduction only as an explicitly output-changing project. It
   could retire more code, but it first needs a single model for token payload,
   structured control flow, fixed semaphore counts, and placement floors.
2. Audit `InsertSemasEmitIR.cpp` next. Its 1,416 lines now exceed every other
   component, and region result/capability rendering offers the largest
   remaining opportunity for helper consolidation without reopening edge
   semantics.
3. Keep future cleanup source-only and require the same unchanged-test gate:
   build first, then 118/118 NVWS, then the full lit suite.
