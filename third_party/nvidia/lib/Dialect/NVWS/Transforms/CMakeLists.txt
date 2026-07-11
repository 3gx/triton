add_triton_library(NVWSTransforms
  AssignStagePhase.cpp
  LowerAref.cpp
  LowerWarpGroup.cpp
  InsertAllocas.cpp
  MetaToNVWSConvert.cpp
  PackEpilogueSlices.cpp
  InsertSemaphore.cpp
  Utilities.cpp
  InsertTmemSemaphore.cpp
  InsertSemas.cpp
  InsertSemasAccessDag.cpp
  InsertSemasEmitIR.cpp
  InsertSemasSyncDag.cpp
  HoistTmemStore.cpp
  WSUtility.cpp
  WSDataPartition.cpp
  PartitionSchedulingMeta.cpp
  StripPartitionAttrsOutsideWS.cpp
  MemoryPlanner.cpp
  MemoryPlannerNVWSAdapter.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVHopperTransforms
  NVWSIR
  MLIRTransformUtils
)
