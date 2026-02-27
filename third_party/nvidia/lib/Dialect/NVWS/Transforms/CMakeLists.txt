add_triton_library(NVWSTransforms
  LowerAref.cpp
  LowerArefToSemaphore.cpp
  AssignSemaphoreStagePhase.cpp
  LowerSemaphore.cpp
  LowerWarpGroup.cpp
  InsertAref.cpp
  InsertSemaphore.cpp
  Utilities.cpp
  AssignStagePhase.cpp
  InsertTmemAref.cpp
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
