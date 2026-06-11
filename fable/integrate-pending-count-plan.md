# Plan: first-class pending/arrive counts on NVWS semaphore ops

Goal: the counts the lowering needs become typed IR authored by
`nvws-insert-semas`; `nvws-lower-semaphore` (LowerAref.cpp) transcribes
and verifies, never derives. This retires the discardable
`nvws.dag_pending_count` side-channel, closes the silent under-sync
hazard on the `r S(n>1)` path (emitter computes `rel->count` at
SyncDag:1014 and then drops it at EmitIR:869), and makes the
pending-count analysis an exact-equality verifier forever — with
`arrive_count` first-class, the analysis is no longer blind to
multiplicity, so no `>=` relaxation is ever needed.

## Design (user-ruled, 11jun26)

- `nvws.semaphore.create` gains `OptionalAttr<I32Attr>:$pending_count`;
  insert-semas sets it to the DAG fan-in (`s.count`). The
  `nvws.dag_pending_count` discardable attr is retired.
- `nvws.semaphore.release` gains `OptionalAttr<I32Attr>:$arrive_count`;
  insert-semas sets it to the DAG arrive multiplicity (`rel->count`,
  1 or N).
- LowerAref `getPendingCount`: REQUIRES `pending_count` — hard
  `emitError` if absent; then runs `analyzeSemaphorePendingCount` and
  hard-errors on disagreement (exact equality; asserts become
  diagnostics).
- LowerAref `rewriteRelease`: REQUIRES `arrive_count` — hard error if
  absent; passes it to `ttng.arrive_barrier` (which already takes
  count >= 1; today hardcoded 1 at LowerAref:318).
- `SemaphorePendingCount.cpp`: per-release contribution becomes
  `|async_ops| * arrive_count` (absent = 1 for analysis of old-pass
  IR), restoring exact agreement even for multiplicity > 1.
- Enforcement lives in the LOWERING, not the op verifier — the old
  `InsertSemaphore`/`InsertTmemSemaphore` passes stay valid-IR
  producers (they are out of the live pipeline,
  AutomaticWarpSpecialization.cpp:122-123, and never lowered in-tree);
  they are NOT stamped (user stance: no scope creep).

## RULING (user-approved, 11jun26)

`arrive_count > 1` is directly lowerable only for sync kinds
(`none`/`wgmma` -> ArriveBarrierOp count). `tc5mma`/`tmem_copy` lower
to TCGen5CommitOp (no count); `tma_load` arrives via hardware (no op).
RULING: no use cases exist for multiplicity on async kinds — the
lowering switch uses `llvm::report_fatal_error` (llvm_unreachable
style) when it sees `arrive_count > 1` with a `tc5mma`/`tmem_copy`/
`tma_load` kind. No verifier machinery, no lit-testable diagnostic; if
a real asymmetric kernel ever hits it, the crash message names this
plan and the semantics question gets answered then.

## Code changes

| file | change |
|---|---|
| `NVWSOps.td:44` (create), `:77` (release) | optional attrs + asm format |
| `Ops.cpp` | create verify: field-vs-analysis exact match when present; release verify: `arrive_count >= 1` |
| `SemaphorePendingCount.cpp` | contribution *= arrive_count (default 1) |
| `InsertSemasEmitIR.cpp:293` | create: set `pending_count = s.count`; delete dag_pending_count setAttr |
| `InsertSemasEmitIR.cpp:869` | release: set `arrive_count = n->count` (currently ignored); retire the EmitIR:105 deferral comment |
| `LowerAref.cpp:131` | getPendingCount: require field, verify vs analysis, diagnostics not asserts |
| `LowerAref.cpp:271/318` | rewriteRelease: require field, pass to ArriveBarrierOp; fatal-error on count>1 with async kind |

## Gate sequencing (USER-MANDATED — do not deviate)

1. **Phase A — code only.** Implement everything above. DO NOT touch
   lit tests. Lit WILL fail (printed form of every create/release
   changes); that is expected and ignored in this phase.
2. **Phase B — runtime gates.** All five must pass:
   - the mxfp4 regression we debugged:
     `pytest python/triton_kernels/tests/test_matmul.py::test_op[True-False-False-False-None-16-768-512-1024-ragged-bfloat16-mxfloat4_e2m1-10-1-False-True-None-False-False-False-True-None]`
   - the four pytest gates from `fable/new-insert-semas-plan-2.md`:
     `test_warp_specialize_tma_matmul[False-4-2-64-128-128-8192-8192-512]`,
     `test_warp_specialize_tma_matmul_persistent[True-False-8-2-128-128-128-32-32-32]`,
     `test_warp_specialize_attention_forward[False-4-True-3-128-128-1024-1024]`,
     `test_warp_specialize_attention_persistent_forward[True-8-True-3-128-128-1024-1024]`
     (in `python/test/unit/language/test_warp_specialization.py`)
3. **Phase C — STOP and report to user.** User verifies on the full
   runtime suite. No commits before user approval.
4. **Phase D — after approval:** commit the code change; THEN update
   lit tests (below) and commit those.

## Lit test work (Phase D only)

- **Regen all `insert_semas*` goldens** (~24 files,
  `fable/gen_insert_semas_checks.py --apply` + restore known
  CHECK-NOT/manual blocks): every create (~186) gains
  `pending_count(...)`, every release (~197) gains `arrive_count(...)`,
  `dag_pending_count` disappears.
- **`lower_semaphore.mlir`** (26 funcs, 51 creates, 59 releases):
  hand-add `pending_count` to every input create with the value read
  off its OWN pinned `init_barrier` count — distribution today:
  39x count 1, 7x count 2, 1x count 3 (fan-in cases EXIST; do not
  blanket-1). Add `arrive_count = 1` to every input release (all
  current cases are multiplicity-1: pinned arrive_barrier counts are
  18x1, rest lower to commit/TMA).
- **NEW count>1 coverage (user-required, must exist before close):**
  1. lower-semaphore-level: hand-written case(s) with
     `pending_count = 2` on the create satisfied by ONE release with
     `arrive_count = 2`, payload `[none]` — CHECK `init_barrier ..., 2`
     and `arrive_barrier ..., 2`. Plus a 3-arrival mixed case
     (one release x2 + one release x1) if cheap.
  2. insert-semas-level: an input shape whose SYNC-DAG takes the
     UNIFORM PENDING COUNT scaling path (SyncDag:966-976,
     m == 1 with s.count > 1, i.e. merged dst-groups sharing a
     semaphore with asymmetric source counts) so the PASS ITSELF emits
     `release ... arrive_count(2)` — pin it, and feed the same case
     through `--nvws-insert-semas --nvws-lower-semaphore` to pin the
     lowered `arrive_barrier ..., 2`. EXPLORATION TASK: construct the
     triggering input (candidate source: the asymmetric-wave FA shapes
     referenced by the retired EmitIR:105 ruling); if no current input
     reaches the path, say so explicitly and ship the hand-written
     lower-level case plus a TODO — do not fake the shape with
     hand-mutated metadata (feedback-partition-metadata-semantics).
- **Negative tests** (3): create missing pending_count at lowering;
  release missing arrive_count at lowering; pending_count != analysis.
  (count>1-with-async-kind is a fatal-error path per the ruling — not
  lit-testable, intentionally.)
- Old-pass lit tests (6 files using nvws-insert-semaphore /
  nvws-insert-tmem-semaphore): untouched, they never lower.

## Battery (Phase D close)

NVWS suite + gate-1 lit; the 5 runtime gates again on the final tree;
full lit suite (expect only the 2 known pre-existing failures:
TLX/tlx-verifier.mlir, Conversion/tritongpu_to_llvm_blackwell.mlir).
run_nvws.sh and perf are the user's to run.
