add_triton_library(NVWSTransforms
  AssignStagePhase.cpp
  LowerAref.cpp
  LowerWarpGroup.cpp
  InsertAllocas.cpp
  InsertSemaphore.cpp
  Utilities.cpp
  InsertTmemSemaphore.cpp
  InsertSemas.cpp
  HoistTmemStore.cpp
  WSUtility.cpp
  WSDataPartition.cpp
  PartitionSchedulingMeta.cpp
  StripPartitionAttrsOutsideWS.cpp
  MemoryPlanner.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
