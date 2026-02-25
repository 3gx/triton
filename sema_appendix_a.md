# Appendix A: Proof — When is Multiphase Required?

Referenced by: `sema_phase3_spec.md` section 2.4 and `sema_master_plan.md` decision 3.

## A.1 Definitions

**mbarrier.** Hardware object with internal phase parity `phi(b) in {0,1}`. After
`init`, `phi = 0`. Each completed acquire-release cycle flips `phi`.

**Semaphore.** Guards a buffer with `D` stages. Stage `s` is backed by `mbar[s]`.
The k-th acquire of `mbar[s]` requires phase = `phi_0 XOR (k-1 mod 2)`.

**Single-phase scheme.** One scalar phase, flips on stage wrap:
```
p(n) = phi_0 XOR (W(n) mod 2)
```
where `W(n)` = number of wraps (stage D-1 -> 0) before acquire n.

**Multiphase scheme.** Per-stage bit vector, flips at every acquire:
```
p(n) = phi_0 XOR (K(s_n, n) mod 2)
```
where `K(s, n)` = number of prior acquires of stage s before acquire n.
This is **always correct** by definition.

**A(s).** Number of acquires of stage s within one cycle (one pass through all D
stages). Assumed stationary across cycles.

## A.2 Theorem

**Single-phase is correct if and only if A(s) = 1 for all s in {0,...,D-1}.**

## A.3 Proof of sufficiency (A(s)=1 implies single-phase correct)

When A(s)=1, each stage is acquired exactly once per cycle. The stage sequence is
a sequential pass: `0, 1, ..., D-1, 0, 1, ...`. The n-th acquire targets stage
`s_n = n mod D`, in cycle `j = floor(n/D)`.

The number of prior acquires of stage s_n is:
```
K(s_n, n) = j = floor(n/D) = W(n)
```

Because stage s_n appeared exactly once in each of the j prior cycles, and hasn't
appeared yet in the current cycle.

Therefore: `p_required(n) = phi_0 XOR (K(s_n,n) mod 2) = phi_0 XOR (W(n) mod 2) = p_SP(n)`. QED

## A.4 Proof of necessity (A(s)>1 implies single-phase incorrect)

Suppose A(s*) > 1 for some stage s*. Within cycle j, let n1 < n2 be the indices of
the first and second acquires of s* (both in `[w_j, w_{j+1})`).

**Required phases:**
```
p_required(n1) = phi_0 XOR (K1 mod 2)
p_required(n2) = phi_0 XOR ((K1+1) mod 2) = p_required(n1) XOR 1
```

They MUST differ (the mbarrier flipped between them).

**Single-phase produces:**
```
p_SP(n1) = phi_0 XOR (W(n1) mod 2) = phi_0 XOR (j mod 2)
p_SP(n2) = phi_0 XOR (W(n2) mod 2) = phi_0 XOR (j mod 2)
```

No wrap occurs between n1 and n2 (same cycle). So p_SP(n1) = p_SP(n2).

But the required phases differ. Single-phase is INCORRECT for n2. QED

## A.5 IR condition

A(s) equals the number of `tmem_load` (observation) operations that execute at
`bufId = s` between consecutive stage advances through s. A stage advance occurs
at `mma useD=false` or `tmem_store` (fresh write) when `was_observed = true`.

**The condition in plain language:** multiphase is required if and only if two
observations occur on the same buffer stage without an intervening stage advance.

**Pigeonhole shortcut:** If O > V (observations per iteration > stage advances per
iteration), multiphase is required.

| Pattern | O/iter | V/iter | A(s) | Single-phase? |
|---------|--------|--------|------|---------------|
| SMEM producer-consumer | 1 | 1 | 1 | Yes |
| TMEM nested matmul | 1 | 1 | 1 | Yes |
| 2 obs + 2 advances (flash attn) | 2 | 2 | 1 | Yes |
| 2 obs + 1 advance (pathological) | 2 | 1 | 2 | No -> multiphase |

## A.6 Worked examples

**Example 1: A(s) = 1 (single-phase sufficient)**
```
for { mma F; mma T; mma T; tmem_load; mma F; mma T; tmem_load }
```
- Obs 1 at stage 0 -> advance -> obs 2 at stage 1. Different stages. A(0)=1, A(1)=1.

**Example 2: A(s) = 2 (multiphase required)**
```
for { mma F; tmem_load; mma T; tmem_load }
```
- Obs 1 at stage 0 -> mma T (no advance, useD=T) -> obs 2 at stage 0. Same stage.
  A(0)=2. Single-phase gives same phase for both. Mbarrier flipped. WRONG.

**Example 3: Nested matmul A(s) = 1**
```
for outer { for inner { mma T }; tmem_load }
```
- One obs per outer iteration, one advance per outer iteration. A(s)=1.

**Example 4: SMEM producer-consumer A(s) = 1**
```
for { TMA_load buf @producer; MMA reads buf @consumer }
```
- One obs (MMA read), one advance (TMA write). A(s)=1.
