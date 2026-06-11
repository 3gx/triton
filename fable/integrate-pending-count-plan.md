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

## RULING (user-approved, 11jun26; revised same day)

`tc5mma`/`tmem_copy`/`tma_load` remain fully valid async kinds with
`arrive_count = 1` — nothing changes for them. ONLY the combination
`arrive_count > 1` AND an async kind is rejected: those kinds lower to
TCGen5CommitOp / hardware-completed TMA, which cannot arrive N times.
Multiplicity > 1 is supported exactly for `[none]`/`[wgmma]`
(ArriveBarrierOp takes a count). REJECTION MECHANISM (user-revised):
a proper DIAGNOSTIC — `emitError` on the release op and pass failure
in the lowering — NOT report_fatal_error; it is lit-testable and
matches the pass ethos (hard diagnostic, never a silent repair).

## Code changes

| file | change |
|---|---|
| `NVWSOps.td:44` (create), `:77` (release) | optional attrs + asm format |
| `Ops.cpp` | create verify: field-vs-analysis exact match when present; release verify: `arrive_count >= 1` |
| `SemaphorePendingCount.cpp` | contribution *= arrive_count (default 1) |
| `InsertSemasEmitIR.cpp:293` | create: set `pending_count = s.count`; delete dag_pending_count setAttr |
| `InsertSemasEmitIR.cpp:869` | release: set `arrive_count = n->count` (currently ignored); retire the EmitIR:105 deferral comment |
| `LowerAref.cpp:131` | getPendingCount: require field, verify vs analysis, diagnostics not asserts |
| `LowerAref.cpp:271/318` | rewriteRelease: require field, pass to ArriveBarrierOp; emitError + fail on count>1 with async kind |

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
     lowered `arrive_barrier ..., 2`. RESOLVED (11jun26, second pass):
     the triggering shape already existed in-tree —
     insert_semas_release_count.mlir (rewrite commit 3.1/4) drives the
     SyncDag For-row adoption with a 1-source outer group adopting a
     2-source regain (its DAG dump pins `r S0(2)`); it was missed
     because its RUN piped to /dev/null, so no emitted-IR golden carried
     arrive_count. The test now has EMIT and LOWER check prefixes
     pinning `pending_count = 2` / `arrive_count = 2` in emitted IR and
     `init_barrier/arrive_barrier ..., 2` after lowering — end-to-end
     coverage from a real pass-produced shape. (Hand-built probes that
     do NOT trigger the path, for the record: an in-body asymmetric
     second reader resolves to separate symmetric semaphores via an
     EXIT-close group, and a pre-loop toucher resolves via entry-permit
     semantics; the adoption specifically needs an outer For-row edge
     whose acquirer's LAST in-body group has more sources.)
- **Negative tests** (4): create missing pending_count at lowering;
  release missing arrive_count at lowering; pending_count != analysis;
  `arrive_count > 1` with an async kind (expects the lowering
  diagnostic).
- Old-pass lit tests (6 files using nvws-insert-semaphore /
  nvws-insert-tmem-semaphore): untouched, they never lower.

## Battery (Phase D close)

NVWS suite + gate-1 lit; the 5 runtime gates again on the final tree;
full lit suite (expect only the 2 known pre-existing failures:
TLX/tlx-verifier.mlir, Conversion/tritongpu_to_llvm_blackwell.mlir).
run_nvws.sh and perf are the user's to run.
