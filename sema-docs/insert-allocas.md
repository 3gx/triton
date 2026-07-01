# InsertAllocas

## Purpose

`NVWSInsertAllocas` converts cross-partition SSA communication into explicit
mutable SMEM or TMEM accesses. It inserts no synchronization. On Meta-NVWS,
`MemoryPlanner` uses these accesses to assign storage and reuse before
`InsertSemas` derives synchronization from the resulting access topology.
Terms are defined in the
[NVWS-AWS terminology](nvws-aws-overview.md#terminology).

## Input contract

The pass expects warp-specialized loops with finalized partition ownership:

- producers and consumers carry `ttg.partition`;
- region results carry `ttg.partition.outputs`;
- scheduled operations retain `loop.stage` and `loop.cluster`;
- WS tags distinguish nested or post-loop warp-specialized scopes.

It handles loop iter-args, ranked-tensor WS-loop results, sourceful
`ttg.local_alloc` operations, descriptor loads/gathers, ordinary tensor/scalar
results, and values produced by regular `tt.load`.

## Algorithm

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
allocations. There are no `nvws.semaphore.*` operations, and the buffers keep
their direct shape — no extra leading dimension for buffered copies is added
(depth is decided later). On the Meta-NVWS path, `MemoryPlanner` subsequently
adds physical reuse and depth metadata; `InsertSemas` then observes the
resulting memory accesses.

## Differences from InsertSemaphore

The pass was adapted from
[`InsertSemaphore.cpp`](https://github.com/3gx/triton/blob/egx/nvws-semaphore/third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertSemaphore.cpp)
in [`PR10121`](https://github.com/triton-lang/triton/pull/10121).

| InsertSemaphore | Current InsertAllocas |
|---|---|
| Creates empty/full semaphore pairs while creating the buffer | Creates only the buffer and accesses |
| Places producer/consumer acquire, buffer, and release operations eagerly | Places data accesses but leaves semaphore ownership and synchronization placement to `InsertSemas` |
| Adds a leading buffered-copy dimension immediately | Keeps the direct mutable memdesc shape |
| Rejects regular-load/cp.async-shaped values | Represents a regular load with an explicit store into SMEM |
| Does not process WS loop results separately | Materializes ranked-tensor cross-partition WS-loop-result buffers |
| SMEM-oriented tensor/scalar communication | Adds the guarded rank-1 SSA-TMEM path described above |

The split is structural: allocation answers *where data is communicated*;
`InsertSemas` answers *who owns it and when ownership moves*.

## Code map

[`InsertAllocas.cpp`](../third_party/nvidia/lib/Dialect/NVWS/Transforms/InsertAllocas.cpp):

- `createCommunicationBuffer`: memory-space and allocation choice.
- `createSemaphoreProducer` / `createSemaphoreConsumer`: retained historical
  names for producer/consumer access materialization.
- `insertSemaphoresForUses`: per-produced-value transformation.
- `NVWSInsertAllocas::runOnOperation`: selects allocation-only mode.
