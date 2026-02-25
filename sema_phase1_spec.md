# Phase 1 Spec: Semaphore Dialect Infrastructure

**Depends on:** Nothing
**Checkpoint:** `ninja triton-opt` builds + `test/NVWS/semaphore_ops.mlir` passes
**Non-breakage:** All existing lit tests pass (purely additive changes)

## What this phase does

Add `SemaphoreType` and 4 semaphore ops to the NVWS dialect. Register 3 new passes
in `Passes.td` with STUB `.cpp` files (empty `runOnOperation()` bodies) to satisfy
the linker. NO Python bindings in Phase 1 — those are added in Phases 2/3/4.

## 1. SemaphoreType

Add to `third_party/nvidia/include/Dialect/NVWS/IR/NVWSTypes.td` after `NVWS_ArefType`:

```tablegen
def NVWS_SemaphoreType : NVWS_TypeDef<"Semaphore", "semaphore"> {
  let summary = "Semaphore with embedded buffer type";
  let description = [{
    A semaphore guards access to a multi-buffered allocation. It embeds the buffer
    types and the number of pipeline stages (buffer depth). Multiple semaphores can
    share the same buffer — they will use a shared stage counter.
  }];
  let parameters = (ins "TypeArrayAttr":$baseType,
                        DefaultValuedParameter<"int", "1">:$numStages);
  let assemblyFormat = "`<` $baseType `,` $numStages `>`";
}
```

`TypeArrayAttr` is the NVWS-custom attribute defined at `NVWSAttrDefs.td:34` as
`ArrayOfAttr<NVWS_Dialect, "TypeArray", "type_array", "Type">`. Same as `ArefType`.

## 2. Semaphore ops

Add to `third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td` before `NVWS_ArefCreateOp`
(line 44). Model after existing aref ops at lines 44-169 of the same file.

### SemaphoreCreateOp

```tablegen
def NVWS_SemaphoreCreateOp : NVWS_Op<"semaphore.create", [
    RangedTypesMatchWith<"input types match Semaphore output type",
                        "result", "buffers",
                        "::llvm::cast<SemaphoreType>($_self).getBaseType()">]> {
  let summary = "Create a semaphore guarding a multi-buffered allocation.";
  let arguments = (ins Variadic<TTG_MemDescType>:$buffers,
                       I1Attr:$is_released);
  let results = (outs NVWS_SemaphoreType:$result);
  let assemblyFormat = [{
    $buffers $is_released attr-dict `:` type($result)
  }];
}
```

### SemaphoreAcquireOp

```tablegen
def NVWS_SemaphoreAcquireOp : NVWS_Op<"semaphore.acquire", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<NVWS_ArefStageInterface>]> {
  let summary = "Acquire a semaphore. Returns a token proving ownership.";
  let arguments = (ins NVWS_SemaphoreType:$semaphore,
                       Optional<I32>:$stage,
                       Optional<I32>:$phase);
  let results = (outs TTG_AsyncToken:$token);
  let assemblyFormat = [{
    $semaphore (`[` $stage^ `,` $phase `]`)? attr-dict
    `:` type($semaphore) `->` type($token)
  }];
  let builders = [
    OpBuilder<(ins "Value":$semaphore, "Type":$tokenType), [{
      build($_builder, $_state, tokenType, semaphore, Value(), Value());
    }]>
  ];
}
```

### SemaphoreReleaseOp

```tablegen
def NVWS_SemaphoreReleaseOp : NVWS_Op<"semaphore.release", [
    DeclareOpInterfaceMethods<NVWS_ArefStageInterface>]> {
  let summary = "Release a semaphore, signaling the next partition.";
  let arguments = (ins NVWS_SemaphoreType:$semaphore,
                       TTG_AsyncToken:$token,
                       Optional<I32>:$stage,
                       NVWS_AsyncOpArrayAttr:$async_ops);
  let assemblyFormat = [{
    $semaphore (`[` $stage^ `]`)? `,` $token $async_ops attr-dict
    `:` type($semaphore) `,` type($token)
  }];
  let builders = [
    OpBuilder<(ins "Value":$semaphore, "Value":$token, "ArrayAttr":$async_ops), [{
      build($_builder, $_state, semaphore, token, Value(), async_ops);
    }]>
  ];
}
```

### SemaphoreBufferOp

```tablegen
def NVWS_SemaphoreBufferOp : NVWS_Op<"semaphore.buffer", [
    DeclareOpInterfaceMethods<NVWS_ArefStageInterface>]> {
  let summary = "Get buffer view from semaphore at a given stage.";
  let arguments = (ins NVWS_SemaphoreType:$semaphore,
                       TTG_AsyncToken:$token,
                       Optional<I32>:$stage);
  let results = (outs Variadic<TTG_MemDescType>:$buffers);
  let assemblyFormat = [{
    $semaphore (`[` $stage^ `]`)? `,` $token attr-dict
    `:` type($semaphore) `,` type($token) `->` type(results)
  }];
  let builders = [
    OpBuilder<(ins "Value":$semaphore, "TypeRange":$bufferTypes, "Value":$token), [{
      build($_builder, $_state, bufferTypes, semaphore, token, Value());
    }]>
  ];
}
```

**Key traits:**
- `AttrSizedOperandSegments` on `SemaphoreAcquireOp` (has 2 optional operands, same
  as `ArefGetEnterOp` at NVWSOps.td:83 and `ArefPutEnterOp` at NVWSOps.td:127)
- `DeclareOpInterfaceMethods<NVWS_ArefStageInterface>` on all ops with Optional stage
  (same as all aref ops). This provides `getStage()`/`setStage()` methods.
- `assemblyFormat` patterns match existing aref ops (stage in brackets, optional).
- `builders` provide no-stage/no-phase convenience constructors (same pattern as
  existing aref builders at NVWSOps.td:76-80, 99-103, 120-124, 143-147, 164-167).

## 3. setStage implementations in Ops.cpp

Add to `third_party/nvidia/lib/Dialect/NVWS/IR/Ops.cpp` (after existing aref
implementations at line 183). Same pattern as existing aref ops — ONLY `setStage()`.
`getStage()` is auto-generated by ODS from the `ArefStageInterface` declaration
and does NOT need a manual implementation.

```cpp
void SemaphoreAcquireOp::setStage(Value stage) { getStageMutable().assign(stage); }
void SemaphoreReleaseOp::setStage(Value stage) { getStageMutable().assign(stage); }
void SemaphoreBufferOp::setStage(Value stage) { getStageMutable().assign(stage); }
```

## 4. Pass registrations in Passes.td

Add to `third_party/nvidia/include/Dialect/NVWS/Transforms/Passes.td` BEFORE
`#endif // NVWS_PASSES` (line 190). Keep ALL existing pass defs unchanged.

```tablegen
def NVWSLowerArefToSemaphore : Pass<"nvws-lower-aref-to-semaphore", "mlir::ModuleOp"> {
  let summary = "Convert nvws.aref.* to nvws.semaphore.* ops.";
  let description = [{
    Lowers aref operations to semaphore operations. Each aref becomes two semaphores
    (one initially released, one not). The aref put/get enter/exit ops become
    semaphore acquire/release pairs. Buffer views are produced by semaphore.buffer.
    Stage and phase are NOT assigned by this pass — they are left as Optional absent.
  }];
  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">
  ];
}

def NVWSAssignSemaphoreStagePhase : Pass<"nvws-assign-semaphore-stage-phase", "mlir::ModuleOp"> {
  let summary = "Assign stage/phase to nvws.semaphore.* ops using observation rule.";
  let description = [{
    Assigns buffer stage and phase to semaphore ops based on the observation rule:
    advance %bufId when (was_observed AND is_fresh_write). Groups semaphores by
    their shared buffer. Threads %bufId, %was_observed, and per-semaphore %phase
    through scf.for/scf.if control flow. Uses MULTIPHASE (per-stage bit vector)
    for phase tracking.
  }];
  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSLowerSemaphore : Pass<"nvws-lower-semaphore", "mlir::ModuleOp"> {
  let summary = "Convert nvws.semaphore.* to ttng.*barrier* ops.";
  let description = [{
    Lowers semaphore operations to mbarrier wait/arrive primitives.
    Each semaphore becomes an mbarrier array (one per stage). Phase bit is extracted
    from the MULTIPHASE bit-vector. Handles TMA loads (BarrierExpectOp),
    tc5mma (TCGen5CommitOp), fences, and mbarrier cleanup.
  }];
  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}
```

## 5. Python bindings

**DO NOT add Python bindings in Phase 1.** The factory functions (`createNVWS*`)
won't link until the implementation `.cpp` files exist. Each later phase adds its
own binding when its `.cpp` file is created:

- Phase 2 adds: `ADD_PASS_WRAPPER_0("add_lower_aref_to_semaphore", mlir::triton::createNVWSLowerArefToSemaphore);`
- Phase 3 adds: `ADD_PASS_WRAPPER_0("add_assign_semaphore_stage_phase", mlir::triton::createNVWSAssignSemaphoreStagePhase);`
- Phase 4 adds: `ADD_PASS_WRAPPER_0("add_lower_semaphore", mlir::triton::createNVWSLowerSemaphore);`

All use `ADD_PASS_WRAPPER_0` (no options exposed to Python, matching existing
`add_lower_aref` at `triton_nvidia.cc:84`). The `numStages` option on
`LowerArefToSemaphore` uses the tablegen default (3).

**CRITICAL: Pass registration requires stub .cpp files.**

`Passes.h` (line 37) uses `#define GEN_PASS_REGISTRATION` which generates
`registerNVWSTransformsPasses()`. This is called from `RegisterTritonDialects.h:136`.
Adding pass defs to `Passes.td` causes the registration function to call the factory
functions (`createNVWSLowerArefToSemaphore` etc.). Without definitions, the linker fails.

**Solution:** Phase 1 MUST create 3 stub `.cpp` files:

```cpp
// LowerArefToSemaphoreStub.cpp (or add to LowerArefToSemaphore.cpp)
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
namespace mlir { namespace triton {
#define GEN_PASS_DEF_NVWSLOWERAREFTOSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
namespace {
class NVWSLowerArefToSemaphore
    : public impl::NVWSLowerArefToSemaphoreBase<NVWSLowerArefToSemaphore> {
  using impl::NVWSLowerArefToSemaphoreBase<NVWSLowerArefToSemaphore>::NVWSLowerArefToSemaphoreBase;
  void runOnOperation() override {} // stub — replaced in Phase 2
};
} // namespace
} // namespace triton
} // namespace mlir
```

Create similar stubs for `AssignSemaphoreStagePhase` and `LowerSemaphore`.
Add all 3 to `CMakeLists.txt`. Later phases replace stub bodies with real logic.

## 6. CMakeLists.txt

ADD the 3 stub `.cpp` files to `third_party/nvidia/lib/Dialect/NVWS/Transforms/CMakeLists.txt`:
- `LowerArefToSemaphore.cpp`
- `AssignSemaphoreStagePhase.cpp`
- `LowerSemaphore.cpp`

The `.td` changes are picked up by existing tablegen rules. `Ops.cpp` is already compiled.

## 7. Lit test

Create `test/NVWS/semaphore_ops.mlir`. The exact type syntax for
`!nvws.semaphore<..., N>` depends on how `TypeArrayAttr` prints — adjust after
first build. Use the same encoding/memory-space attributes as existing aref tests.

The test should verify parse→print round-trip for:
1. `semaphore.create` with `true`/`false` `is_released`
2. `semaphore.acquire` without stage/phase (Optional absent)
3. `semaphore.acquire` with stage and phase
4. `semaphore.buffer` without stage and with stage
5. `semaphore.release` with different `async_ops`

**Important:** Use the `assemblyFormat` as defined in section 2 above. The format
for `SemaphoreCreateOp` is `$buffers $is_released attr-dict : type($result)` —
NO parentheses around `$buffers`. Match the format in the test IR.

**Build command:** `ninja -C build/cmake.linux-x86_64-cpython-3.12/ triton-opt`

## 8. Verification

```bash
BUILD=build/cmake.linux-x86_64-cpython-3.12
TOPT=$BUILD/bin/triton-opt

# Build:
ninja -C $BUILD triton-opt

# Phase 1 lit test:
$TOPT test/NVWS/semaphore_ops.mlir -split-input-file | FileCheck test/NVWS/semaphore_ops.mlir

# Existing tests still pass:
$TOPT test/NVWS/insert_aref.mlir -split-input-file --allow-unregistered-dialect --nvws-insert-aref | FileCheck test/NVWS/insert_aref.mlir
$TOPT test/NVWS/aref-tmem-insertion.mlir -split-input-file --allow-unregistered-dialect -nvws-insert-tmem-aref -cse | FileCheck test/NVWS/aref-tmem-insertion.mlir
$TOPT test/NVWS/lower_aref.mlir -split-input-file --allow-unregistered-dialect --nvws-lower-aref | FileCheck test/NVWS/lower_aref.mlir
$TOPT test/NVWS/assign_stage_phase.mlir -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase -cse | FileCheck test/NVWS/assign_stage_phase.mlir
```
