# NVWS semaphore documentation

This directory describes the NVWS automatic warp-specialization path from
partition selection through hardware barrier lowering. It is written for a
reader familiar with Meta AutoWS.

The documents describe algorithms and contracts. Source is authoritative.

## Reading order

1. [NVWS-AWS pipeline and terminology](nvws-aws-overview.md)
2. [Meta ports](meta-ports.md)
3. [InsertAllocas](insert-allocas.md)
4. [InsertSemas overview](insert-semas/overview.md)
5. [ACCESS-DAG](insert-semas/access-dag.md)
6. [OWNER-DAG](insert-semas/owner-dag.md)
7. [SYNC-DAG](insert-semas/sync-dag.md)
8. [EMIT-IR](insert-semas/emit-ir.md)
9. [AssignStagePhase and LowerSemaphore](assign-stage-phase-and-lower-semaphores.md)
