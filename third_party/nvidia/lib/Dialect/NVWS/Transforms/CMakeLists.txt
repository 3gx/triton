add_triton_library(NVWSTransforms
  AssignSemaphoreStagePhase.cpp
  LowerSemaphore.cpp
  LowerWarpGroup.cpp
  InsertSemaphore.cpp
  Utilities.cpp
  InsertTmemSemaphore.cpp
  HoistTmemStore.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
