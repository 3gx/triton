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
