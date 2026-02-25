# Appendix B: Proof — Shared Stage Counter for N Semaphores (by Induction)

Referenced by: `sema_phase3_spec.md` section 2.3 and `sema_master_plan.md` decision 4.

## B.1 Claim

For any N >= 2 semaphores `sem_0, sem_1, ..., sem_{N-1}` sharing the same
multi-buffered allocation `buf` with depth D, all semaphores must use the same
stage counter `%bufId`. This is both **necessary** (separate counters break
correctness) and **sufficient** (shared counter + semaphore protocol is correct).

## B.2 Definitions

**Stage counter.** `%bufId in {0,...,D-1}` selects which physical slot `buf[%bufId]`
is active. The `semaphore.buffer` op returns `buf[%bufId]`.

**Ownership ring.** N partitions form a directed chain of ownership transfers:
```
P_0 -> P_1 -> P_2 -> ... -> P_{N-1} -> P_0
```
P_i acquires `sem_i`, accesses `buf[%bufId]`, releases `sem_{i+1 mod N}`.

**Correctness properties:**
- (a) Mutual exclusion: at most one partition accesses `buf[s]` at any time
- (b) Consistent reads: consumer reads what producer wrote (same physical slot)
- (c) No lost writes: every write has a corresponding read

## B.3 Base case: N = 2

Two semaphores `sem_0, sem_1` share `buf`.

**Necessity.** Suppose P_0 uses `s_0` and P_1 uses `s_1 != s_0`. P_0 writes
`buf[s_0]`. P_1 reads `buf[s_1]`. Different physical slots -> stale read
(property b violated). P_0's write to `buf[s_0]` is never consumed -> lost write
(property c violated). QED

**Sufficiency.** When both use `s_0 = s_1 = s`: P_0 writes `buf[s]`, releases
`sem_1` (release barrier). P_1 acquires `sem_1` (acquire barrier), reads `buf[s]`.
Release-acquire pair on same mbarrier provides memory ordering. All three
properties hold. QED

## B.4 Inductive hypothesis

Assume the claim holds for any K semaphores sharing the same buffer, for all
2 <= K <= N-1.

## B.5 Inductive step: N semaphores

**Necessity (all must use same stage).**

Consider any two consecutive partitions P_i, P_{i+1} in the ownership chain.
P_i accesses `buf` and releases to P_{i+1}. These two semaphores `sem_i` and
`sem_{i+1}` share `buf` — by the base case (N=2), they must use the same stage.

Define equivalence relation: `sem_i ~ sem_j` iff they must use the same stage.

By the base case applied to each consecutive pair:
```
sem_0 ~ sem_1    (from pair {P_0, P_1})
sem_1 ~ sem_2    (from pair {P_1, P_2})
sem_2 ~ sem_3    (from pair {P_2, P_3})
...
sem_{N-2} ~ sem_{N-1}  (from pair {P_{N-2}, P_{N-1}})
sem_{N-1} ~ sem_0      (from pair {P_{N-1}, P_0}, ring closure)
```

By transitivity: `sem_0 ~ sem_1 ~ sem_2 ~ ... ~ sem_{N-1}`.

All semaphores are in the same equivalence class -> all must use the same stage. QED

**Sufficiency (shared counter is correct).**

When all N semaphores use `%bufId = s`:

**(a) Mutual exclusion.** The ownership ring ensures sequential access: P_i must
release before P_{i+1} can acquire. By inductive hypothesis, any K < N consecutive
partitions have correct mutual exclusion. Adding P_N extends the chain by one link.
P_{N-1} releases `sem_0` only after finishing, so P_0 in the next cycle cannot
start early. No concurrent access to `buf[s]`. QED

**(b) Consistent reads.** For any producer P_i and consumer P_j (j after i in chain):
the release-acquire chain provides transitive memory ordering. P_i release ->
P_{i+1} acquire -> ... -> P_j acquire. Since all use `buf[s]`, P_j reads P_i's
writes. QED

**(c) No lost writes.** The ring is closed. P_0 does not advance `%bufId` until the
full chain completes (P_{N-1} releases back to P_0). Every write to `buf[s]` is
observed before `s` advances. QED

## B.6 Conclusion

By induction on N, for any N >= 2 semaphores sharing the same buffer:

1. **Necessary:** All must use the same stage counter. (By base case on consecutive
   pairs + transitivity along the ownership chain.)
2. **Sufficient:** Shared counter + semaphore protocol provides mutual exclusion,
   consistent reads, and no lost writes. (By extending the ownership chain one link
   at each inductive step.)

**Fundamental reason:** The stage counter is an addressing function
`addr(t) = t mod D` that maps a logical iteration to a physical slot. For
communication through a shared buffer, all communicating parties must agree on
which slot holds the data. A partition-dependent addressing function
`addr(t, P_i) != addr(t, P_j)` violates this for at least one iteration, breaking
property (b). The shared counter is the unique partition-independent addressing
function.

## B.7 Counterexample search (none found)

Six potential counterexamples were examined:
1. Asymmetric access (one partition reads every other iter) -> stale read
2. Multiple producers writing different parts of buffer -> incoherent read
3. Nested loops with different advancement rates -> wrong slot
4. Independent consumers -> no benefit to separate counters
5. Document's flattened loop with conditional access -> shared counter works
6. Non-deterministic access patterns -> divergent counters break communication

No counterexample exists because the stage counter addresses shared physical memory.
Agreement on the address is a precondition for communication, not a design choice.
