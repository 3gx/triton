# Formal Proofs

This document contains the formal proofs that underpin the semaphore stage/phase
assignment algorithm. These proofs are verified against the implementation in
`AssignSemaphoreStagePhase.cpp`.

## Proof A: When is Multiphase Required?

### Definitions

**mbarrier**: Hardware object with internal phase parity `phi(b)` in `{0, 1}`.
After `init`, `phi = 0`. Each completed acquire-release cycle flips `phi`.

**Semaphore**: Guards a buffer with `D` stages. Stage `s` is backed by `mbar[s]`.
The k-th acquire of `mbar[s]` requires phase = `phi_0 XOR ((k-1) mod 2)`.

**Single-phase scheme**: One scalar phase, flips on stage wrap:
```
p_SP(n) = phi_0 XOR (W(n) mod 2)
```
where `W(n)` = number of wraps (stage D-1 -> 0) before acquire n.

**Multiphase scheme**: Per-stage bit vector, flips at every acquire:
```
p_MP(n) = phi_0 XOR (K(s_n, n) mod 2)
```
where `K(s, n)` = number of prior acquires of stage s before acquire n.
This is **always correct** by definition (it directly tracks the mbarrier's phase).

**A(s)**: Number of acquires of stage s within one cycle (one pass through all D
stages). Assumed stationary across cycles.

### Theorem

**Single-phase is correct if and only if A(s) = 1 for all s in {0,...,D-1}.**

### Proof of Sufficiency (A(s)=1 => single-phase correct)

When A(s)=1, each stage is acquired exactly once per cycle. The stage sequence is
a sequential pass: `0, 1, ..., D-1, 0, 1, ...`. The n-th acquire targets stage
`s_n = n mod D`, in cycle `j = floor(n/D)`.

The number of prior acquires of stage `s_n` is:
```
K(s_n, n) = j = floor(n/D) = W(n)
```

Because stage `s_n` appeared exactly once in each of the j prior cycles, and hasn't
appeared yet in the current cycle.

Therefore:
```
p_required(n) = phi_0 XOR (K(s_n,n) mod 2) = phi_0 XOR (W(n) mod 2) = p_SP(n)
```

Single-phase matches multiphase. QED.

### Proof of Necessity (A(s)>1 => single-phase incorrect)

Suppose `A(s*) > 1` for some stage `s*`. Within cycle j, let `n1 < n2` be the
indices of the first and second acquires of `s*` (both in cycle j).

**Required phases:**
```
p_required(n1) = phi_0 XOR (K1 mod 2)
p_required(n2) = phi_0 XOR ((K1+1) mod 2) = p_required(n1) XOR 1
```

They **must differ** (the mbarrier flipped between them).

**Single-phase produces:**
```
p_SP(n1) = phi_0 XOR (W(n1) mod 2) = phi_0 XOR (j mod 2)
p_SP(n2) = phi_0 XOR (W(n2) mod 2) = phi_0 XOR (j mod 2)
```

No wrap occurs between `n1` and `n2` (same cycle). So `p_SP(n1) = p_SP(n2)`.

But the required phases differ. Single-phase is **incorrect** for `n2`. QED.

### IR Condition

A(s) equals the number of acquires (mbarrier waits) of mbar[s] per cycle. In the
IR, a stage advance occurs at an acquire whose first buffer use is a FreshWrite
(the fresh-write rule). The condition translates to:

**Multiphase is required if and only if two acquires hit the same stage in one
iteration without an intervening stage advance.**

The eligibility analysis (`walkBlockForEligibility`) directly checks this by
tracking `(semaphore, partition, virtual_stage)` tuples and detecting duplicates.

### Worked Examples

**Example 1: A(s)=1 (single-phase correct)**
```
for { acquire P (advance) -> acquire C (observe) }
```
P advances to vs=1, C stays at vs=1. No duplicates. A(0)=1, A(1)=1.

**Example 2: A(s)=2 (multiphase required)**
```
for { acquire P (advance) -> acquire C (observe) -> acquire P (no advance, useD=T) -> acquire C (observe) }
```
P@vs1, C@vs1, P@vs1 (duplicate!), C@vs1 (duplicate!). A(0)=2.

**Example 3: Three semaphores, A(s)=1 (single-phase correct)**
```
for { acquire P (advance) -> release C -> release D -> acquire C -> acquire D }
```
P@vs1, C@vs1, D@vs1. No duplicates (different semaphores). A(s)=1.

This example proves the pigeonhole shortcut `O > V` is too conservative:
O=2, V=1, but A(s)=1 because the observations are on different semaphores.

## Proof B: Shared Stage Counter for N Semaphores

### Claim

For any N >= 2 semaphores `sem_0, ..., sem_{N-1}` sharing the same
multi-buffered allocation `buf` with depth D, all semaphores **must** use the
same stage counter `%bufId`. Furthermore, a shared counter is **sufficient**
for correctness.

### Definitions

**Stage counter**: `%bufId` in `{0,...,D-1}` selects which physical slot
`buf[%bufId]` is active. `semaphore.buffer` returns `buf[%bufId]`.

**Ownership ring**: N partitions form a directed chain:
```
P_0 -> P_1 -> P_2 -> ... -> P_{N-1} -> P_0
```
`P_i` acquires `sem_i`, accesses `buf[%bufId]`, releases `sem_{(i+1) mod N}`.

**Correctness properties**:
- (a) Mutual exclusion: at most one partition accesses `buf[s]` at any time
- (b) Consistent reads: consumer reads what producer wrote (same physical slot)
- (c) No lost writes: every write has a corresponding read

### Base Case: N = 2

**Necessity**: Suppose `P_0` uses `s_0` and `P_1` uses `s_1 != s_0`. `P_0`
writes `buf[s_0]`. `P_1` reads `buf[s_1]`. Different physical slots -> stale
read (property b violated). `P_0`'s write to `buf[s_0]` is never consumed ->
lost write (property c violated). QED.

**Sufficiency**: When both use `s_0 = s_1 = s`: `P_0` writes `buf[s]`, releases
`sem_1`. `P_1` acquires `sem_1`, reads `buf[s]`. Release-acquire pair on same
mbarrier provides memory ordering. All three properties hold. QED.

### Inductive Hypothesis

Assume the claim holds for any K semaphores sharing the same buffer, for all
`2 <= K <= N-1`.

### Inductive Step: N Semaphores

**Necessity (all must use same stage)**:

Consider any two consecutive partitions `P_i`, `P_{i+1}` in the ownership chain.
By the base case (N=2), they must use the same stage.

Define equivalence: `sem_i ~ sem_j` iff they must use the same stage.

```
sem_0 ~ sem_1    (pair {P_0, P_1})
sem_1 ~ sem_2    (pair {P_1, P_2})
...
sem_{N-2} ~ sem_{N-1}  (pair {P_{N-2}, P_{N-1}})
sem_{N-1} ~ sem_0      (pair {P_{N-1}, P_0}, ring closure)
```

By transitivity: all semaphores are equivalent -> all must use same stage. QED.

**Sufficiency (shared counter is correct)**:

When all N semaphores use `%bufId = s`:

**(a) Mutual exclusion**: The ownership ring ensures sequential access. `P_i`
must release before `P_{i+1}` can acquire. No concurrent access to `buf[s]`.

**(b) Consistent reads**: The release-acquire chain provides transitive memory
ordering: `P_i` release -> `P_{i+1}` acquire -> ... -> `P_j` acquire. Since
all use `buf[s]`, `P_j` reads `P_i`'s writes.

**(c) No lost writes**: The ring is closed. `P_0` does not advance `%bufId`
until the full chain completes. Every write to `buf[s]` is observed before
`s` advances. QED.

### Fundamental Reason

The stage counter is an addressing function `addr(t) = t mod D` that maps a
logical iteration to a physical slot. For communication through a shared buffer,
all communicating parties must agree on which slot holds the data. A
partition-dependent addressing function `addr(t, P_i) != addr(t, P_j)` violates
this for at least one iteration, breaking consistent reads. The shared counter
is the unique partition-independent addressing function.

### Implementation

In `AssignSemaphoreStagePhase.cpp`, semaphores are grouped by their first buffer
operand. All semaphores in a group share a single `state.stage` value. The
stage is computed once (via the fresh-write rule) and applied to all semaphores
in the group:

```cpp
llvm::MapVector<Value, SmallVector<SemaphoreCreateOp>> semaGroups;
for (auto semaOp : semaOps) {
    semaGroups[semaOp.getBuffers().front()].push_back(semaOp);
}
```

## Pending Count Rule

### Definition

```
pendingCount(sem) = sum over distinct releasing partitions P of arrivalCount(sem, P)
```

Where `arrivalCount(sem, P) = |asyncOps|` on partition P's release of `sem`.

### How It Maps to mbarrier

`InitBarrierOp(mbar, N)` sets expected arrivals to N. The mbarrier completes
when N arrivals occur.

Each `AsyncOp` kind generates exactly one arrival:

| Kind | Arrive Op | Arrivals |
|------|-----------|----------|
| `none` | `ArriveBarrierOp(mbar, 1)` | 1 |
| `wgmma` | `ArriveBarrierOp(mbar, 1)` | 1 |
| `tc5mma` | `TCGen5CommitOp(mbar)` | 1 |
| `tmem_copy` | `TCGen5CommitOp(mbar)` | 1 |
| `tma_load` | hardware | 1 |

**TMA byte tracking is orthogonal**: `BarrierExpectOp(mbar, txCount)` sets
expected bytes independently of the arrival counter. The mbarrier completes
when **both** counters reach zero.

### Guarantee Levels

**Local op verifier** (always runs):
- `SemaphoreReleaseOp`: no duplicate async kinds
- `SemaphoreCreateOp`: type consistency, pending-count analysis

**Pass-level guarantees** (by construction):
- Conditional acquire/release matching (requires control-flow analysis)
- pendingCount correctness (dedup by partition, sum |asyncOps|)

### Conditional Release Safety

**Valid**: acquire and release in same conditional scope:
```
@1 { for { if (c) { work; release sem } } }
@2 { for { if (c) { acquire sem; use } } }
```
When `c=true`: matched. When `c=false`: neither fires.

**Invalid**: unconditional acquire, conditional release:
```
@1 { for { if (c) { work; release sem } } }
@2 { for { acquire sem; use } }
```
When `c=false`: acquire fires, 0 arrivals -> **deadlock**.

**Rule**: On every execution path where an acquire fires, exactly `pendingCount`
releases must also fire.
