#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

// Commit 0 of fable/new-insert-semas-plan-2.md: the previous implementation
// is deleted and the pass is an intentional no-op while the new
// implementation is brought up stage by stage per fable/semas-report3.md
// (ACCESS-DAG -> OWNER-DAG -> SYNC-DAG -> EMIT-IR). lit failures of
// test/NVWS/insert_semas* are expected until the EMIT-IR commit lands.
class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<NVWSInsertSemas> {
public:
  using NVWSInsertSemasBase::NVWSInsertSemasBase;

  void runOnOperation() override {}
};

} // namespace triton
} // namespace mlir
