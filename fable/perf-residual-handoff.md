# Hand-off: 06-fa.py perf residual (old pass vs insert-semas family) + machine-wide drop

Status: OPEN OBSERVATION, recorded 12jun26 (evening, after hold-rule M3
gates). Nothing here blocks anything: the hold rule is at parity with its
whole family and all gates passed. This doc exists so the residual can be
investigated later without re-deriving the session.

## What was measured (12jun26, single session, same GPU, same script)

Script: `06-fa.py` (untracked, repo root) — FA fwd WS, BATCH=4 H=32
N_CTX=65536 D=128, causal=False, warp_specialize=True. Every run with
`TRITON_ALWAYS_COMPILE=1`. Runs listed in EXECUTION ORDER within each row;
the rows themselves were measured in the order: solid-tip block, fixup-era
block, band-commit block, solid-tip once more, 01.git block, 03.git block.

| build | sync pass | runs (TFLOPS) | campaign record |
|---|---|---|---|
| solid tip (`bcf4b40e20`-era binary, hold rule native) | insert-semas, native hold rule | 623.4, 626.7, 610.8, 610.8 | — (new) |
| solid w/ fixup-era `Transforms/` (`f2c9a22840` dir-swap) | insert-semas + point-of-use fixup | 624.6, 610.4 | 657–668 |
| solid @ `c31a6b7a37` FULL checkout (the band commit) | insert-semas + fixup | 625.9, 617.0, 609.6 | 657–668 (its own record) |
| `../triton-01.git` (`egx/nvws-semaphore`) | OLD InsertSemaphore pass | 638.3, 634.1, 630.0 | 664–670 |
| `../triton-03.git` (`egx/nvws-semaphore-insert-semas`, tip ab6c27cc6c) | insert-semas ported + fixes (no ride guard) | 625.9, 622.4, 619.2 | ≈ baseline ("gap closed", study §14) |

## Established facts (measured, not interpreted)

1. **Machine-wide drop**: every build, including the untouched 01.git
   baseline, measures ~30–45 TF below its campaign record. The band commit
   `c31a6b7a37` cannot reproduce its own 657–668.
2. **Insert-semas family parity**: port (03.git), fixup-era, and native
   hold rule are mutually indistinguishable (~610–627, overlapping).
   The hold rule is exonerated twice over — the residual exists in builds
   containing none of its code.
3. **Old-pass residual TODAY**: 01.git (630–638) sits ~12–16 TF above the
   whole insert-semas family (610–627), NON-overlapping ranges. Run
   ordering does not explain it: 01.git's runs were sandwiched between
   insert-semas runs that stayed at ~620 on both sides.
4. **The same comparison was ≈closed at campaign time** (657–668 vs
   664–670; ~1%). Today it reads ~2.3%.

## Explicitly NOT claimed

- The cause of the machine-wide drop (clocks/driver/thermals/tenant —
  unknown, not investigated).
- The cause of the re-opened residual. One sentence of hypothesis, clearly
  labeled as such: the FA study's regime framing (stall-bound vs
  issue-bound, fable/fa-perf-study-regressions-analysis.md §10/§13) makes
  a clock-dependent sensitivity plausible — lower clocks could re-expose a
  latency the campaign conditions hid. UNVERIFIED.

## State caveats for whoever picks this up

- `06-fa.py` is UNTRACKED. All legs this session ran the identical script,
  but identity vs the campaign-time script is unverifiable from git.
- 01.git still carries the campaign's uncommitted TEMPORARY combine-disable
  in LowerAref (study §14.1 measured combine inert for FA, but for a
  pristine-pristine baseline leg, revert + rebuild first).
- 03.git tip ab6c27cc6c lacks the ride guard and everything after it (it is
  frozen at the port-experiment state).

## Investigation protocol (when picked up)

1. **First re-measure when the machine is back in its usual state.** If
   01.git returns to ~665 and solid to ~660, the residual closed itself and
   only fact 1 needs explaining (or ignoring).
2. **If the residual persists: strict interleaving.** Alternate
   `01.git, solid-tip, 01.git, solid-tip, …` 5–6 pairs back-to-back so
   drift cancels; paired statistics decide. A/B discipline throughout
   (`TRITON_ALWAYS_COMPILE=1`, one-time per-leg IR fingerprint; env knobs
   are not in the cache key).
3. **If real: it is the ROOT-OUTSIDE / placement-second-order class, not
   the hold rule.** Diff per-pass IR 01 vs solid for this kernel exactly as
   the campaign study did (methodology + capture layout:
   fable/fa-perf-study-regressions-analysis.md §11–§13; old captures under
   logs/fa-11jun26-*). The study's decomposition (§14: ~47 TF point-of-use
   + ~15 TF ROOT-OUTSIDE) closed the campaign gap; the residual would be a
   third, smaller term that today's conditions amplify.
4. **Useful technique from this session**: for binary A/Bs, in-place
   `git checkout <commit>` (tree must be CLEAN) + incremental ninja beats
   worktree builds by an order of magnitude; restore with
   `git checkout <branch>` + ninja, then re-run the lit gates as a sanity
   check. Dir-level swaps (`git checkout <commit> -- path/`) work too but
   carry an assumption about what matters — full checkout when it counts.

## References

- Hold-rule implementation + gates: fable/hold-rule-implementation-plan.md
  (COMPLETE; M3 evidence in Status), commits 47a3e1653f, 918329d015,
  bcf4b40e20.
- Design + validation: fable/rule-v2-corpus-verification.md §0/§7.
- Campaign perf study (methodology to reuse):
  fable/fa-perf-study-regressions-analysis.md §10–§14.
- Real-kernel placement fingerprint (12jun26): holdrule on 06-fa compile =
  acc gated(entry-consumed); qk/p/m_i/m_ij/k/v pointofuse.
