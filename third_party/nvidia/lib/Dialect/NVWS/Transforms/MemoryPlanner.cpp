/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nvws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

#define GEN_PASS_DEF_NVWSMEMORYPLANNER
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

namespace ttng = mlir::triton::nvidia_gpu;

static void setI32Attr(Operation *op, StringRef name, int32_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     value));
}

static LogicalResult doTmemMemoryPlanning(FuncOp funcOp,
                                          unsigned firstBufferId) {
  unsigned nextBufferId = firstBufferId;
  funcOp.walk([&](ttng::TMEMAllocOp alloc) {
    setI32Attr(alloc, "buffer.id", nextBufferId++);
    setI32Attr(alloc, "buffer.copy", 1);
    setI32Attr(alloc, "buffer.offset", 0);
    LLVM_DEBUG({
      LDBG("assigned TMEM allocation buffer.id=" << nextBufferId - 1);
      alloc->dump();
    });
  });
  return success();
}

class NVWSMemoryPlanner
    : public impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner> {
public:
  using impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner>::NVWSMemoryPlannerBase;

  void runOnOperation() override {
    getOperation()->walk([&](FuncOp funcOp) {
      if (failed(doTmemMemoryPlanning(funcOp, numBuffers)))
        signalPassFailure();
    });
  }
};
} // namespace

} // namespace mlir::triton
