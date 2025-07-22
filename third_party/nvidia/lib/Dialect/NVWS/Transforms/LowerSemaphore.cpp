/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-semaphore"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

std::pair<WarpGroupOp, int> getWarpGroupIdx(Operation *op) {
  if (auto wgOp = dyn_cast<WarpGroupOp>(op->getParentOp())) {
    auto region = op->getParentRegion();
    return {wgOp, region->getRegionNumber()};
  }
  if (isa<triton::FuncOp>(op))
    return {nullptr, -1};
  return getWarpGroupIdx(op->getParentOp());
}

int getPendingCount(SemaphoreCreateOp op) {
  std::optional<int> arrivalCount;

  for (auto user : op->getUsers()) {
    auto [wgOp, idx] = getWarpGroupIdx(user);
    auto numWarps = wgOp.getNumWarps()[idx];

    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user)) {
      int count = 0;
      for (auto prod : releaseOp.getAsyncOps()) {
        auto kind = dyn_cast<AsyncOpAttr>(prod).getValue();
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::TMALoad:
          count += 1;
          break;
        case AsyncOp::CpAsync:
          count += numWarps * 32;
          break;
        case AsyncOp::NONE:
          // TODO: this should be 'numWarps * 32' when we transition to
          //       multi-threaded arrive
          count += 1;
          break;
        default:
          llvm_unreachable("unknown producer kind");
        }
      }

      if (arrivalCount) {
        assert(*arrivalCount == count && "inconsistent producer arrival count");
      } else {
        arrivalCount = count;
      }
    }
  }

  assert(arrivalCount);

  return *arrivalCount;
}

Value createAndInitMbar(SemaphoreCreateOp op, PatternRewriter &rewriter) {
  auto pendingCount = getPendingCount(op);

  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto semaType = op.getType();
  auto depth = *semaType.getNumStages();

  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(builder, rewriter.getI64Type(), depth);
  auto lb = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto ub = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
  auto step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  auto dLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(dLoop.getBody());

  for (int i = 0; i < 2; ++i) {
    auto singleBarrier =
        createSingleBufferView(rewriter, mbars, dLoop.getInductionVar());
    rewriter.create<InitBarrierOp>(loc, singleBarrier, pendingCount);
  }
  return mbars;
}

void rewriteAcquireOp(SemaphoreCreateOp semaphoreOp, SemaphoreAcquireOp op,
                      PatternRewriter &rewriter, Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  rewriter.create<WaitBarrierOp>(loc, mbar, op.getPhase());
}

void rewriteReleaseOp(SemaphoreCreateOp semaphoreOp, SemaphoreReleaseOp op,
                      PatternRewriter &rewriter, Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  for (auto asyncOp : op.getAsyncOps()) {
    auto asyncOpEnum = cast<AsyncOpAttr>(asyncOp).getValue();
    switch (asyncOpEnum) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      rewriter.create<nvidia_gpu::ArriveBarrierOp>(loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      rewriter.create<nvidia_gpu::TCGen5CommitOp>(loc, mbar);
      break;

    case AsyncOp::TMALoad:
      // nothing to do, TMA load is handled by lowering putEnterOp
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unknown async op");
    }
  }
}

class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto mbars = createAndInitMbar(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      if (auto user = dyn_cast<SemaphoreAcquireOp>(userOp)) {
        opToDelete.insert(user);
        rewriteAcquireOp(op, user, rewriter, mbars);
      } else if (auto user = dyn_cast<SemaphoreReleaseOp>(userOp)) {
        opToDelete.insert(user);
        rewriteReleaseOp(op, user, rewriter, mbars);
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }
};

template <class... Ts> struct AssignIndex;
template <class T> struct AssignIndex<T> {
  struct Index {
    // Having stage and phase as separate values, rather than encoding them
    // into a single index, results in better performance. Same approach is used
    // in CUTLASS and CUTEDSL, and this may allow PTXAS to better optimize code.
    Value stage;
    Value phase;
  };
  using IndexMap = llvm::MapVector<Value, Index>;
  using UseSet = llvm::SetVector<Value>;

  static UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(opT.getOperand(0));
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        useSet = analyzeUseInBlock(forOp.getBody(), useSet);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        useSet = analyzeUseInBlock(ifOp.thenBlock(), useSet);
        if (ifOp.elseBlock())
          useSet = analyzeUseInBlock(ifOp.elseBlock(), useSet);
      }
    }
    return useSet;
  }

  static void assignInForOp(scf::ForOp forOp, IndexMap &indexMap) {

    // find uses of xops in forOp body
    auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
    if (useInBlock.empty())
      return;

    // add extra iterArgs to the forOp
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> indexRefs;
    for (auto sema : useInBlock) {
      auto index = indexMap.lookup(sema);
      extraIterArgs.push_back(index.stage);
      indexRefs.push_back(&indexMap[sema].stage);
      if (index.phase) {
        extraIterArgs.push_back(index.phase);
        indexRefs.push_back(&indexMap[sema].phase);
      }
    }

    // create new forOp with extra iterArgs
    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // update index with iterArgs in the forOp body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *indexRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // assign index in the forOp body
    auto indexMapInBlock = assignInBlock(forOp.getBody(), indexMap);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    for (auto sema : useInBlock) {
      auto &index = indexMapInBlock[sema];
      extraYieldArgs.push_back(index.stage);
      if (index.phase)
        extraYieldArgs.push_back(index.phase);
    }
    appendToForOpYield(forOp, extraYieldArgs);

    // update index with results from newForOp
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *indexRefs[idx - nArgs] = forOp.getResult(idx);
  }

  static void assignInIfOp(scf::IfOp ifOp, IndexMap &indexMap) {

    // find uses of xops in then-block
    auto useInIfOp = analyzeUseInBlock(ifOp.thenBlock(), {});
    if (useInIfOp.empty())
      return;

    // find uses of xops in else-block
    useInIfOp = ifOp.elseBlock()
                    ? analyzeUseInBlock(ifOp.elseBlock(), useInIfOp)
                    : useInIfOp;

    // add extra results to the ifOp
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> indexRefs;
    for (auto sema : useInIfOp) {
      auto index = indexMap.lookup(sema);
      extraIfResults.push_back(index.stage.getType());
      indexRefs.push_back(&indexMap[sema].stage);
      if (index.phase) {
        extraIfResults.push_back(index.phase.getType());
        indexRefs.push_back(&indexMap[sema].phase);
      }
    }

    // create new ifOp with extra results
    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // assign index in then-body
    auto indexInThenBlock = assignInBlock(newIfOp.thenBlock(), indexMap);

    // assign index in else-body
    auto indexInElseBlock = ifOp.elseBlock()
                                ? assignInBlock(newIfOp.elseBlock(), indexMap)
                                : indexMap;

    // update yieldOp to return new indexes
    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();
    // insert new indexes to the yieldOp
    for (auto sema : useInIfOp) {
      auto &thenIndex = indexInThenBlock[sema];
      auto &elseIndex = indexInElseBlock[sema];
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  thenIndex.stage);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  elseIndex.stage);
      if (thenIndex.phase) {
        thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                    thenIndex.phase);
        elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                    elseIndex.phase);
      }
    }
    ifOp.erase();

    // update index with results from newIfOp
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *indexRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  static IndexMap assignInBlock(Block *block, IndexMap indexMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        auto index = indexMap.lookup(opT.getOperand(0));

        OpBuilder builder(opT);
        builder.setInsertionPointAfter(opT);

        // compute next stage
        opT.getStageMutable().assign(index.stage);
        auto nextStage = builder.create<arith::AddIOp>(
            opT.getLoc(), index.stage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
        // auto arefBuf = opT.getAref()
        //                    .template getDefiningOp<nvws::ArefCreateOp>()
        //                    .getOperand(0);
        auto depth = *opT.getSemaphore().getType().getNumStages();
        //            depth 2; //
        //            cast<MemDescType>(arefBuf.getType()).getShape().front();

        auto cnd = builder.create<arith::CmpIOp>(
            opT.getLoc(), arith::CmpIPredicate::eq, nextStage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), depth, 32));
        auto zero = builder.create<arith::ConstantIntOp>(opT.getLoc(), 0, 32);
        indexMap[opT.getOperand(0)].stage =
            builder.create<arith::SelectOp>(opT.getLoc(), cnd, zero, nextStage);

        if constexpr (std::is_same_v<T, SemaphoreAcquireOp>) {
          // if this is an enterOp, compute next phase
          opT.getPhaseMutable().assign(index.phase);
          auto nextPhase = builder.create<arith::XOrIOp>(
              opT.getLoc(), index.phase,
              builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
          indexMap[opT.getOperand(0)].phase = builder.create<arith::SelectOp>(
              opT.getLoc(), cnd, nextPhase, index.phase);
        }

      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignInForOp(forOp, indexMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignInIfOp(ifOp, indexMap);
      }
    }

    return indexMap;
  }

  static LogicalResult run(WarpGroupOp wgOp, std::string opName) {
    llvm::errs() << " --- Xrun " << opName << "\n";
    // Verify that all puts and gets are in the same group; otherwise, the stage
    // would need to be communicated across groups, not currently supported.
    Region *opRegion = {};

    // SetVector<Value> anchors;
    // wgOp.walk([&](T op) { anchors.insert(op.getOperand(0)); });
    // for (auto anchor : anchors) {
    //   for (auto user : anchor.getUsers()) {
    //     if (isa<T>(user)) {
    //       auto [wg, idx] = getWarpGroupIdx(user);
    //       auto region = &wg.getPartitionRegions()[idx];
    //       if (opRegion && opRegion != region) {
    //         return mlir::emitWarning(user->getLoc(),
    //                                  "All " + opName +
    //                                      " must be in the same warp-group");
    //       }
    //       opRegion = region;
    //     }
    //   }
    // }

    UseSet use;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      use = analyzeUseInBlock(block, use);
    }

    // initialize indexes
    IndexMap indexMap;
    for (auto anchor : use) {
      OpBuilder builder(anchor.getDefiningOp());
      builder.setInsertionPointAfter(anchor.getDefiningOp());
      indexMap[anchor].stage =
          builder.create<arith::ConstantIntOp>(anchor.getLoc(), 0, 32);
      if (std::is_same_v<T, SemaphoreAcquireOp>) {
        auto semaOp = anchor.getDefiningOp<SemaphoreCreateOp>();
        assert(semaOp);
        bool isReleased = semaOp.getIsReleased();
        indexMap[anchor].phase = builder.create<arith::ConstantIntOp>(
            anchor.getLoc(), isReleased, 32);
      } else {
        indexMap[anchor].phase = {};
      }
    }

    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      assignInBlock(block, indexMap);
    }
    return success();
  }
};

template <> struct AssignIndex<> {
  static LogicalResult run(WarpGroupOp wgOp) {
    if (failed(
            AssignIndex<SemaphoreAcquireOp>::run(wgOp, "SemaphoreAcquireOp"))) {
      assert(0);
      return failure();
    }
    if (failed(
            AssignIndex<SemaphoreReleaseOp>::run(wgOp, "SemaphoreReleaseOp"))) {
      assert(0);
      return failure();
    }
    return success();
  }
};

} // anonymous namespace

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    llvm::errs() << " --- XwgOps.size() " << wgOps.size() << "\n";
    for (auto wgOp : wgOps) {
      llvm::errs() << " --- XwgOp " << wgOp << "\n";
      if (failed(AssignIndex<>::run(wgOp)))
        signalPassFailure();
    }
    // LLVM_DEBUG(llvm::dbgs() << "After semaphoreIndexAssignment\n" << m <<
    // "\n");
    llvm::errs() << "After semaphoreIndexAssignment\n" << m << "\n";
    //    assert(0);

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerSemaphoreCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
