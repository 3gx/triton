// v4 InsertSemas — empty pass scaffolding (commit 0 baseline).
//
// Per meta2nvws-plan/per-edge-sema-plan.v4.md, the entire prior
// implementation is removed in commit 0. Subsequent commits add:
//   1. Discovery + ACCESS DAG (dump-only)
//   2. + OWNERSHIP DAG (dump-only)
//   3. + RAW-SYNC DAG (dump-only)
//   4. + OPT-SYNC DAG (dump-only)
//   5. + EMIT (renders nvws.semaphore.* IR from the OPT-SYNC DAG)
//
// At this commit the pass is a no-op. lit tests under test/NVWS that
// depend on nvws.semaphore.* emission are expected to fail until
// commit 5 lands.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  void runOnOperation() override {
    // Empty: implementation lands in subsequent commits per the staged
    // plan in meta2nvws-plan/per-edge-sema-plan.v4.md.
  }
};

} // namespace triton
} // namespace mlir
