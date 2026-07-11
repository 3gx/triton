# NVWS semaphore documentation

This directory describes the NVWS automatic warp-specialization path from
partition selection through hardware barrier lowering. It is written for a
reader familiar with Meta AutoWS.

The documents describe algorithms and contracts. Source is authoritative.

## Reading order

1. [NVWS-AWS pipeline and terminology](nvws-aws-overview.md)
2. [InsertAllocas](insert-allocas.md)
3. [InsertSemas overview](insert-semas/overview.md)
4. [ACCESS-DAG](insert-semas/access-dag.md)
5. [SYNC-DAG](insert-semas/sync-dag.md)
6. [EMIT-IR](insert-semas/emit-ir.md)
7. [AssignStagePhase and LowerSemaphore](assign-stage-phase-and-lower-semaphores.md)

## Flags

### Meta-AutoWS

- `TRITON_USE_META_WS=1` uses the complete Meta AutoWS backend. This takes
  precedence over `TRITON_NVWS_USE_META`.

### NVWS

- `TRITON_NVWS_USE_META=1` uses canonical Meta partitioning/planning with the
  NVWS synchronization and lowering backend.
- `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1`, together with
  `TRITON_NVWS_USE_META=1`, uses `NVWSInsertAllocas` instead of Meta
  `WSBufferAllocation`. All subsequent Meta planning passes remain common.
- `NVWS_USE_SSA_TMEM=1` allows eligible cross-partition SSA communication to
  use TMEM. It applies on the default NVWS path and, on the Meta-NVWS path,
  only when `TRITON_NVWS_USE_META_NVWS_ALLOCAS=1` is enabled.
- `NVWS_INSERT_SEMA_DUMP_DAG=1` dumps the InsertSemas access, ownership, and
  synchronization DAGs.
- `TRITON_DUMP_WS_GRAPHS=/path` dumps memory-planner channel/allocation graphs.

### Other

- `TRITON_USE_LLM_SCHEDULE=1` uses the LLM scheduler.
- `TRITON_USE_MODULO_SCHEDULE=<1|sms|exhaustive|random>` uses modulo scheduling
  instead of ordinary latency scheduling.
- `TRITON_FP8_PROMOTE_TO_TMEM=0|1` controls FP8 LHS promotion into TMEM.
- `TRITON_ALWAYS_COMPILE=1` forces recompilation instead of using cached
  compilation.
- `MLIR_ENABLE_DIAGNOSTICS=warnings` enables compiler warnings.
- `MLIR_ENABLE_DUMP=1` dumps IR around compiler passes.
