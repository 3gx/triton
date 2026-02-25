# Why Semaphores Instead of Arefs

This document describes limitations of the aref (asynchronous
reference) abstraction that motivated the switch to explicit semaphores.
Examples are drawn from the design rationale document [`aref_tmem`](https://docs.google.com/document/d/1rRGM5ibHopa8Us_NeSggs1ZMoH7F0pjOVjUTIqc2neM/edit?tab=t.0#heading=h.d0w2qew0lu25) google-doc.

## Background: The Aref Model

An aref provides a communication channel between a producer and a consumer
through exclusive access to a backing buffer:

```
%aref_buf = alloc : memdesc<1x..>  // aref backing buffer
%aref = aref_create %aref_buf

@1 {                               @2 {
  for .. {                            for .. {
     %buf = put.enter %aref              %buf = get.enter %aref
     produce %buf                        consume %buf
     put.exit %aref                      get.exit %aref
  }                                   }
}                                   }
```

The `put.enter`/`put.exit` bracket the producer's critical section;
`get.enter`/`get.exit` bracket the consumer's. An aref also embeds
buffer-staging -- `put.enter` and `get.enter` semantics automatically return successive
buffer slots, enabling software pipelining without explicit stage tracking.
This is done automatically by `AssignStagePhase.cpp`

## Limitation 1: Two-Party Restriction

### The Problem

An aref inherently models a two-party (producer/consumer) interaction. When
TMEM ownership must rotate among 3+ partitions, arefs require awkward workarounds.

### Example: Three-Partition Ownership Ring

With semaphores, a ring-based ownership transfer is natural:

```
%sem1 = semaphore_create    // @1 acquires, @3 releases
%sem2 = semaphore_create    // @2 acquires, @1 releases
%sem3 = semaphore_create    // @3 acquires, @2 releases

@1 {                    @2 {                    @3 {
  for .. {                for .. {                for .. {
    acquire %sem1           acquire %sem2           acquire %sem3
    update1 %buf            update2 %buf            update3 %buf
    release %sem2           release %sem3           release %sem1
  }                       }                       }
}                       }                       }
```

Each partition acquires its semaphore (released by the predecessor), performs work,
then releases the successor's semaphore. One semaphore per link in the chain.

### Aref Workaround

With arefs, this requires three arefs** to prevent races:

```
@1 {                       @2 {                       @3 {
  for .. {                   for .. {                   for .. {
    %buf = put.enter %a1       %buf = get.enter %a1       %buf = get.enter %a2
    update1 %buf               update2 %buf               update3 %buf
    put.exit %a1               get.exit %a1               get.exit %a2

    _ = get.enter %a3          _ = put.enter %a2          _ = put.enter %a3
    get.exit %a3               put.exit %a2               put.exit %a3
  }                          }                          }
}                          }                          }
```

This introduces an extra `enter`/`exit` pair per iteration per partition. 
These dummy operations have no data-transfer purpose -- they
exist solely to prevent @1 from racing ahead of @3.

When lowered to semaphores, each aref becomes two semaphores, so the 3-aref
variant creates 6 semaphores instead of the natural 3, with redundant
acquire/release pairs that downstream optimizations may or may not eliminate.

### Flash Attention Example

A real-world pattern from Flash Attention illustrates this further. Two MMAs
followed by loads on the same buffer:

**With semaphores** (3 semaphores, clean):
```
@1 {                       @2 {                   @3 {
  for .. {                   for .. {               for .. {
    acquire %sem1              acquire %sem2          acquire %sem3
    mma .., %buf, ..           load %buf              load %buf1
    release %sem2              release %sem1          release %sem1
    acquire %sem1
    mma .., %buf1, ..
    release %sem3
  }                          }                      }
}
```

**With arefs** (4 arefs = 8 semaphores when lowered):
```
@1 {                          @2 {                       @3 {
  for .. {                      for .. {                   for .. {
    %buf = put.enter %a1          %buf = get.enter %a1       %buf = get.enter %a3
    mma .., %buf, ..              load %buf                  load %buf
    put.exit %a1                  get.exit %a1               get.exit %a3

    _ = get.enter %a2             _ = put.enter %a2          _ = put.enter %a4
    get.exit %a2                  put.exit %a2               put.exit %a4

    %buf = put.enter %a3
    mma .., %buf, ..
    put.exit %a3

    _ = get.enter %a4
    get.exit %a4
  }                             }                          }
}
```

The aref version requires 4 arefs and introduces redundant synchronization.
An alternative formulation can reduced it to 3 arefs but entangles @2 and @3 -- @2 must
wait for @3 to complete before it can proceed, which is unnecessary and harms performance.

### Why This Matters

These extra synchronization points are not just code bloat -- they translate to
real mbarrier waits and arrives at runtime. Downstream passes would need to
detect and eliminate the redundant ones, which unclear if  such
optimization is feasible in the general case. Semaphores
avoid the problem entirely by allowing direct N-party communication.

### Per-Stage Semaphores

The semaphore version also naturally supports per-stage semaphores. With N
stages, there are N independent semaphores per direction. Stage `k`'s producer and
consumer can operate independently from stage `k+1`'s, enabling 
pipelining without unnecessary sequencing between stages.



## Summary: Why Semaphores

| Issue | Aref Limitation | Semaphore Solution |
|-------|----------------|-------------------|
| 3+ partition ownership | Requires extra arefs, redundant sync | Natural N-party ownership |
| Buffer staging correctness | Stage advance tied to put/get enter | Stage advance decoupled via fresh-write rule |
| Per-stage independence | All stages share one aref | Per-stage semaphore arrays |
| Code complexity | Extra enter/exit, buffer threading through control flow | Clean acquire/release pairs |
| Optimization burden | Downstream must eliminate redundant sync | No redundant sync generated |

Aref couples signaling, buffer access, and stage advancement into a single abstraction,
 which is an excellent fit for the two-party SMEM or TMEM producer/consumer model. 
However, for the TMEM ownership-transfer model, the aref abstraction introduces 
challenges. Semaphores decouple these concerns—signaling, buffer access, and stage 
advancement can be handled independently—while retaining the essential ideas of arefs, 
such as embedding N stages and the backing buffer to enable correct and efficient code 
generation for all patterns.
