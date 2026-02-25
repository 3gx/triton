// RUN: triton-opt %S/assign_stage_phase.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore | FileCheck %s --check-prefix=LOWER-ASP --implicit-check-not=nvws.aref.
// RUN: triton-opt %S/lower_aref_to_semaphore.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref-to-semaphore --nvws-assign-semaphore-stage-phase | FileCheck %s --check-prefix=ASSIGN-L2S --implicit-check-not=nvws.aref.

// LOWER-ASP: @warp_specialize_tma_matmul
// LOWER-ASP-DAG: nvws.semaphore.create
// LOWER-ASP-DAG: nvws.semaphore.acquire
// LOWER-ASP-DAG: nvws.semaphore.buffer
// LOWER-ASP-DAG: nvws.semaphore.release
// LOWER-ASP-DAG: #nvws.async_op<tma_load>
// LOWER-ASP-DAG: #nvws.async_op<tc5mma>
// LOWER-ASP-DAG: ttng.tc_gen5_mma
// LOWER-ASP-DAG: ttng.tmem_store
// LOWER-ASP-DAG: ttng.tmem_load

// ASSIGN-L2S: @basic
// ASSIGN-L2S-DAG: nvws.semaphore.acquire
// ASSIGN-L2S-DAG: nvws.semaphore.buffer
// ASSIGN-L2S-DAG: nvws.semaphore.release
// ASSIGN-L2S-DAG: arith.shli
// ASSIGN-L2S-DAG: arith.andi
// ASSIGN-L2S-DAG: #nvws.async_op<none>
// ASSIGN-L2S: @tma_preserved
// ASSIGN-L2S-DAG: nvws.descriptor_load
// ASSIGN-L2S-DAG: #nvws.async_op<tma_load>
