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

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERAREF
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

// ----------------------------------------------------------------------------

struct ArefValue {
  Value emptySemaphore;
  Value fullSemaphore;
  SmallVector<Value> buffers;
};

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter) {
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto arefTy = op.getType();
  auto baseType = arefTy.getBaseType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto semaTy = triton::nvws::SemaphoreType::get(b.getContext(), depth);
  auto emptySempahore = b.create<SemaphoreCreateOp>(loc, semaTy, true);
  auto fullSempahore = b.create<SemaphoreCreateOp>(loc, semaTy, false);

  return ArefValue{emptySempahore, fullSempahore, op.getOperands()};
}

SmallVector<Value> getSubViews(ArefValue arefVal, Value stage, Location loc,
                               OpBuilder &rewriter) {
  SmallVector<Value> views;
  for (auto buffer : arefVal.buffers) {
    SmallVector<Value> offsetsVal{stage};
    auto memDescType = cast<MemDescType>(buffer.getType());
    auto shape = memDescType.getShape();
    auto rank = shape.size() - 1;

    for (int i = 0; i < rank; ++i) {
      offsetsVal.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, 0, rewriter.getIntegerType(32)));
    }
    SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
    auto memDescTypeNew = MemDescType::get(
        tensorShape, memDescType.getElementType(), memDescType.getEncoding(),
        memDescType.getMemorySpace(), true);
    Value singleBuffer = rewriter.create<MemDescSubviewOp>(loc, memDescTypeNew,
                                                           buffer, offsetsVal);
    views.push_back(singleBuffer);
  }

  return views;
}

void lowerAsyncLoads(ArefPutEnterOp op, PatternRewriter &rewriter,
                     ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> loadOps;
  for (auto result : op.getResults())
    for (auto user : result.getUsers()) {
      // Temporary workaround for lit testing: handle TMA loads here until a
      // dedicated tma_load op is added to the NVWS dialect
      if (user->getName().getStringRef() == "tma_load")
        loadOps.push_back(user);
    }
  assert(loadOps.size() <= op.getResults().size());
  if (loadOps.empty())
    return;

  // matching ArefPutExitOp is with ArefPutEnterOp
  // we use aref_tag to match the two
  //   %bufs:n = aref_put.enter %aref[%enter_idx] {aref_tag = tag}
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx] {aref_tag = tag}

  // locate the matching aref_put.exit with the same tag, to get full barrier
  ArefPutExitOp arefPutExitOp;
  auto arefTag = op->getAttrOfType<StringAttr>("aref_tag").str();
  for (auto user : op.getAref().getUsers()) {
    if (auto exitOp = dyn_cast<ArefPutExitOp>(user)) {
      if (exitOp->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
        arefPutExitOp = exitOp;
        break;
      }
    }
  }
  assert(arefPutExitOp);
  assert(arefPutExitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

#if 0
Value fullBarrier =
      getFullBarrier(rewriter, loc, arefVal, arefPutExitOp.getStage());
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier, 0,
                                                       pred);
#else
  llvm_unreachable("not implemented");
#endif
  return;
}

void rewritePutEnterOp(ArefCreateOp arefOp, ArefPutEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  rewriter.create<SemaphoreAcquireOp>(loc, arefVal.emptySemaphore,
                                      op.getStage(), Value());
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getResults().size());

  // TMA load need special handling as it requires fullMbarrier that
  // we need to get from matching ArefPutExitOp
  lowerAsyncLoads(op, rewriter, arefVal);

  // replaces uses with views
  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);
}

void rewriteGetEnterOp(ArefCreateOp arefOp, ArefGetEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  rewriter.create<SemaphoreAcquireOp>(loc, arefVal.fullSemaphore, op.getStage(),
                                      Value());
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getResults().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);
}

void rewritePutExitOp(ArefPutExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<SemaphoreReleaseOp>(loc, arefVal.fullSemaphore, op.getStage(),
                                      op.getAsyncOps());
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<SemaphoreReleaseOp>(loc, arefVal.emptySemaphore,
                                      op.getStage(), op.getAsyncOps());
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createAndInitMbar(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        opToDelete.insert(user);
        rewritePutEnterOp(op, user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        opToDelete.insert(user);
        rewriteGetEnterOp(op, user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
        opToDelete.insert(user);
        rewritePutExitOp(user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
        opToDelete.insert(user);
        rewriteGetExitOp(user, rewriter, aref);
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }
};

// ----------------------------------------------------------------------------
template <class T> struct ArefStage {
  using StageMap = llvm::MapVector<Value /*aref*/, Value /*stage*/>;
  using UseSet = llvm::SetVector<Value /*aref*/>;

  static UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(opT.getAref());
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

  static void assignStageInForOp(scf::ForOp forOp, StageMap &stageMap) {

    // find uses of arefs in forOp body
    auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
    if (useInBlock.empty())
      return;

    // add extra iterArgs to the forOp
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> stageRefs;
    for (auto aref : useInBlock) {
      extraIterArgs.push_back(stageMap.lookup(aref));
      stageRefs.push_back(&stageMap[aref]);
    }

    // create new forOp with extra iterArgs
    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // update arefIndex with iterArgs in the forOp body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *stageRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // assign arefIndex in the forOp body
    auto stageMapInBlock = assignStageInBlock(forOp.getBody(), stageMap);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    for (auto aref : useInBlock)
      extraYieldArgs.push_back(stageMapInBlock[aref]);
    appendToForOpYield(forOp, extraYieldArgs);

    // update stage with results from newForOp
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *stageRefs[idx - nArgs] = forOp.getResult(idx);
  }

  static void assignStageInIfOp(scf::IfOp ifOp, StageMap &stageMap) {

    // find uses of aref in then-block
    auto useInBlock = analyzeUseInBlock(ifOp.thenBlock(), {});
    if (useInBlock.empty())
      return;

    // find uses of aref in else-block
    useInBlock = ifOp.elseBlock()
                     ? analyzeUseInBlock(ifOp.elseBlock(), useInBlock)
                     : useInBlock;

    // add extra results to the ifOp
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> stageRefs;
    for (auto aref : useInBlock) {
      extraIfResults.push_back(stageMap.lookup(aref).getType());
      stageRefs.push_back(&stageMap[aref]);
    }

    // create new ifOp with extra results
    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // assign arefIndex in then-body
    auto stageMapInThenBlock =
        assignStageInBlock(newIfOp.thenBlock(), stageMap);

    // assign arefIndex in else-body
    auto stageMapInElseBlock =
        ifOp.elseBlock() ? assignStageInBlock(newIfOp.elseBlock(), stageMap)
                         : stageMap;

    // update yieldOp to return new indexes
    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();
    // insert new indexes to the yieldOp
    for (auto aref : useInBlock) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  stageMapInThenBlock[aref]);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  stageMapInElseBlock[aref]);
    }
    ifOp.erase();

    // update arefIndex with results from newIfOp
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *stageRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  static StageMap assignStageInBlock(Block *block, StageMap stageMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        auto stage = stageMap.lookup(opT.getAref());

        OpBuilder builder(opT);
        builder.setInsertionPointAfter(opT);

        // compute next stage
        opT.getStageMutable().assign(stage);
        auto nextStage = builder.create<arith::AddIOp>(
            opT.getLoc(), stage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
        auto arefBuf = opT.getAref()
                           .template getDefiningOp<nvws::ArefCreateOp>()
                           .getOperand(0);
        auto depth = cast<MemDescType>(arefBuf.getType()).getShape().front();

        auto cnd = builder.create<arith::CmpIOp>(
            opT.getLoc(), arith::CmpIPredicate::eq, nextStage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), depth, 32));
        auto zero = builder.create<arith::ConstantIntOp>(opT.getLoc(), 0, 32);
        stageMap[opT.getAref()] =
            builder.create<arith::SelectOp>(opT.getLoc(), cnd, zero, nextStage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignStageInForOp(forOp, stageMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignStageInIfOp(ifOp, stageMap);
      }
    }

    return stageMap;
  }

  static void run(WarpGroupOp wgOp) {
    UseSet useSet;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      useSet = analyzeUseInBlock(block, useSet);
    }

    // initialize indexes
    StageMap stageMap;
    for (auto aref : useSet) {
      OpBuilder builder(aref.getDefiningOp());
      builder.setInsertionPointAfter(aref.getDefiningOp());
      stageMap[aref] =
          builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
    }

    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      assignStageInBlock(block, stageMap);
    }
  }
};

// ----------------------------------------------------------------------------

} // anonymous namespace

class NVWSLowerAref : public impl::NVWSLowerArefBase<NVWSLowerAref> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    for (auto wgOp : wgOps) {
      ArefStage<ArefPutEnterOp>::run(wgOp);
      ArefStage<ArefPutExitOp>::run(wgOp);
      ArefStage<ArefGetEnterOp>::run(wgOp);
      ArefStage<ArefGetExitOp>::run(wgOp);
    }
    LLVM_DEBUG(llvm::dbgs() << "After ArefStageAssignment\n" << m << "\n");

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
