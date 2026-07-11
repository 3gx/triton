# InsertAllocas

## Purpose

`NVWSInsertAllocas` converts cross-partition SSA communication into explicit
mutable SMEM or TMEM accesses on the default NVWS path. It inserts no
synchronization. Meta-NVWS instead consumes canonical Meta buffer-allocation
output through `MetaToNVWSConvert`. Terms are defined in the
[NVWS-AWS terminology](nvws-aws-overview.md#terminology).

## Input contract

The pass expects warp-specialized loops with finalized partition ownership:

- producers and consumers carry `ttg.partition`;
- region results carry `ttg.partition.outputs`;
- scheduled operations retain `loop.stage` and `loop.cluster`;
- WS tags distinguish nested or post-loop warp-specialized scopes.

It handles loop iter-args, ranked-tensor WS-loop results, sourceful
`ttg.local_alloc` operations, descriptor loads/gathers, ordinary tensor/scalar
results, values produced by regular `tt.load`, and sourceful
`ttng.tmem_alloc` operations.

## Algorithm

Before materializing SSA communication, a function containing a partitioned WS
loop normalizes each sourceful `ttng.tmem_alloc` whose producer and consumers
span more than one partition. Same-partition allocations remain sourceful:

- a sourceless mutable backing is placed before the outer WS loop while
  remaining in the allocation's top-level CFG block;
- an explicit `ttng.tmem_store` remains at the original scheduled point;
- physical planning attributes stay on the backing, while partition and loop
  schedule attributes stay on the store; and
- an initializer token seed is replaced with poison without breaking a
  loop-carried MMA token recurrence.

For each produced value:

1. Group uses by consumer partition after removing producer partitions.
2. Choose the communication memory. Existing memdescs keep their memory
   space. With `NVWS_USE_SSA_TMEM` set, a CUDA capability of 100+, and a
   rank-1 floating-point tensor of extent 64/128 and element width 16/32
   whose layout can be expanded to the rank-1 TMEM form
   (`getExpandedRank1TensorType`), the value may use TMEM; remaining tensors
   and scalars use SMEM.
3. Allocate one mutable communication buffer before the outer WS loop.
4. Materialize the producer write:
   - descriptor operations write directly into the buffer;
   - regular loads and sourceful `ttg.local_alloc` operations store their value
     into it;
   - other tensors use an SMEM or TMEM store;
   - floating-point and integer scalars are splatted and stored in SMEM.
5. Materialize one buffer view/load per consumer partition and rewrite the
   uses. Tensor and scalar values are rewritten strictly per consumer
   partition. A memdesc value (sourceful `ttg.local_alloc`) is instead rewired
   in one shot for all its uses (`replaceUsesAndPropagateType`), which assumes
   a single consumer partition; the assumption is checked only by a debug
   assertion.
6. Remove stale sourceful allocation/descriptor operations after rewiring.

Generated producer and consumer accesses carry the selected owner partition
and available `loop.stage`/`loop.cluster` annotation, plus a WS tag when the
rewritten path supplies one. The allocation operation itself deliberately
carries none of these annotations.

## Output contract

The output contains explicit producer writes and consumer reads over mutable
allocations. Cross-partition TMEM communication has no sourceful allocations,
while same-partition TMEM remains unchanged. There are no
`nvws.semaphore.*` operations. The buffers keep their direct shape — no extra
leading dimension for buffered copies is added (depth is decided later).
`InsertSemas` then observes the resulting memory accesses.

## Separation from synchronization

Allocation answers *where data is communicated*. `InsertSemas` independently
derives *who owns it and when ownership moves*.

## Code map

[`InsertAllocas.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAllocas.cpp):

- `createCommunicationBuffer`: memory-space and allocation choice.
- `normalizeSourcefulTmemAlloc`: hoisted TMEM backing and explicit initializer
  store.
- `createSemaphoreProducer` / `createSemaphoreConsumer`: retained historical
  names for producer/consumer access materialization.
- `insertSemaphoresForUses`: per-produced-value transformation.
- `NVWSInsertAllocas::runOnOperation`: selects allocation-only mode.
