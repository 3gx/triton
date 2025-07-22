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

#if 1
#define MULTIPHASE
#endif

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
  auto numStages = *op.getType().getNumStages();

  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(builder, rewriter.getI64Type(), numStages);
  for (int i = 0; i < numStages; i++) {
    auto singleBarrier = createSingleBufferView(rewriter, mbars, i);
    rewriter.create<InitBarrierOp>(loc, singleBarrier, pendingCount);
  }
  return mbars;
}

void rewriteAcquireOp(SemaphoreCreateOp semaphoreOp, SemaphoreAcquireOp op,
                      PatternRewriter &rewriter, Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
#ifdef MULTIPHASE
  // extract phase for a given stage from the phase bit-vector
  // phase = (phase >> stage) & 1
  Value phaseBit =
      rewriter.create<arith::ShRSIOp>(loc, op.getPhase(), op.getStage());
  phaseBit = rewriter.create<arith::AndIOp>(
      loc, phaseBit, rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
#else
  Value phaseBit = op.getPhase();
#endif
  rewriter.create<WaitBarrierOp>(loc, mbar, phaseBit);
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

namespace AssignPhase {
using PhaseMap = llvm::MapVector<Value /*semaphore*/, Value /*phase*/>;
using UseSet = llvm::SetVector<Value /*semaphore*/>;
static PhaseMap assignInBlock(Block *block, PhaseMap phaseMap);

UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
  for (auto &op : *block) {
    if (auto opT = dyn_cast<SemaphoreAcquireOp>(op)) {
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

void assignInForOp(scf::ForOp forOp, PhaseMap &phaseMap) {

  // find uses of xops in forOp body
  auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
  if (useInBlock.empty())
    return;

  // add extra iterArgs to the forOp
  SmallVector<Value> extraIterArgs;
  SmallVector<Value *> indexRefs;
  for (auto sema : useInBlock) {
    extraIterArgs.push_back(phaseMap.lookup(sema));
    indexRefs.push_back(&phaseMap[sema]);
  }

  // create new forOp with extra iterArgs
  OpBuilder builder(forOp);
  size_t nArgs = forOp.getRegionIterArgs().size();
  forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

  // update index with iterArgs in the forOp body
  for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
    *indexRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

  // assign index in the forOp body
  auto phaseMapInBlock = assignInBlock(forOp.getBody(), phaseMap);

  // update yieldOp to return new indexes
  SmallVector<Value> extraYieldArgs;
  for (auto sema : useInBlock)
    extraYieldArgs.push_back(phaseMapInBlock[sema]);
  appendToForOpYield(forOp, extraYieldArgs);

  // update index with results from newForOp
  for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
    *indexRefs[idx - nArgs] = forOp.getResult(idx);
}

void assignInIfOp(scf::IfOp ifOp, PhaseMap &phaseMap) {

  // find uses of xops in then-block
  auto useInIfOp = analyzeUseInBlock(ifOp.thenBlock(), {});
  if (useInIfOp.empty())
    return;

  // find uses of xops in else-block
  useInIfOp = ifOp.elseBlock() ? analyzeUseInBlock(ifOp.elseBlock(), useInIfOp)
                               : useInIfOp;

  // add extra results to the ifOp
  SmallVector<Type> extraIfResults;
  SmallVector<Value *> phaseRefs;
  for (auto sema : useInIfOp) {
    extraIfResults.push_back(phaseMap.lookup(sema).getType());
    phaseRefs.push_back(&phaseMap[sema]);
  }

  // create new ifOp with extra results
  OpBuilder builder(ifOp);
  size_t nArgs = ifOp.getResults().size();
  auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

  // assign index in then-body
  auto phaseInThenBlock = assignInBlock(newIfOp.thenBlock(), phaseMap);

  // assign index in else-body
  auto phaseInElseBlock = ifOp.elseBlock()
                              ? assignInBlock(newIfOp.elseBlock(), phaseMap)
                              : phaseMap;

  // update yieldOp to return new indexes
  auto thenYieldOp = newIfOp.thenYield();
  auto elseYieldOp = newIfOp.elseYield();
  // insert new indexes to the yieldOp
  for (auto sema : useInIfOp) {
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                phaseInThenBlock[sema]);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                phaseInElseBlock[sema]);
  }
  ifOp.erase();

  // update index with results from newIfOp
  for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
    *phaseRefs[idx - nArgs] = newIfOp.getResult(idx);
}

PhaseMap assignInBlock(Block *block, PhaseMap phaseMap) {
  for (auto &op : llvm::make_early_inc_range(*block)) {
    if (auto opT = dyn_cast<SemaphoreAcquireOp>(op)) {
      auto phase = phaseMap.lookup(opT.getOperand(0));

      OpBuilder builder(opT);
      builder.setInsertionPointAfter(opT);

#ifdef MULTIPHASE
      opT.getPhaseMutable().assign(phase);
      SmallVector<Operation *> users;
      for (auto user : opT.getStage().getUsers())
        users.push_back(user);
      assert(!users.empty());
      auto user = users.back();
      // the phase is a bit-vector, each bit for each stage
      // next_phase = phase ^ (1 << stage)
      auto phaseBit = builder.create<arith::ShLIOp>(
          opT.getLoc(),
          builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32),
          opT.getStage());
      phaseMap[opT.getOperand(0)] =
          builder.create<arith::XOrIOp>(opT.getLoc(), phase, phaseBit);
#else
      opT.getPhaseMutable().assign(phase);

      Operation *addi = {};
      for (auto user : opT.getStage().getUsers()) {
        if (isa<arith::AddIOp>(user)) {
          assert(!addi);
          addi = user;
        }
      }
      assert(addi);
      Operation *cnd = {};
      for (auto user : addi->getUsers()) {
        if (isa<arith::CmpIOp>(user)) {
          assert(!cnd);
          cnd = user;
        }
      }
      assert(cnd);
      Operation *select = {};
      for (auto user : cnd->getUsers()) {
        if (isa<arith::SelectOp>(user)) {
          assert(!select);
          select = user;
        }
      }
      assert(select);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(select);
        auto nextPhase = builder.create<arith::XOrIOp>(
            opT.getLoc(), phase,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
        phaseMap[opT.getOperand(0)] = builder.create<arith::SelectOp>(
            opT.getLoc(), cnd->getResult(0), nextPhase, phase);
      }

#endif

    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      assignInForOp(forOp, phaseMap);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      assignInIfOp(ifOp, phaseMap);
    }
  }

  return phaseMap;
}

void run(WarpGroupOp wgOp) {
  UseSet useSet;
  for (auto region : wgOp.getRegions()) {
    auto block = &region->getBlocks().front();
    useSet = analyzeUseInBlock(block, useSet);
  }

  // initialize indexes
  PhaseMap phaseMap;
  for (auto sema : useSet) {
    OpBuilder builder(sema.getDefiningOp());
    builder.setInsertionPointAfter(sema.getDefiningOp());
    auto semaOp = sema.getDefiningOp<SemaphoreCreateOp>();
    assert(semaOp);
    bool isReleased = semaOp.getIsReleased();
#ifdef MULTIPHASE
    phaseMap[sema] = builder.create<arith::ConstantIntOp>(
        sema.getLoc(), isReleased ? 0xFFFFFFFF : 0x00000000, 32);
#else
    phaseMap[sema] = builder.create<arith::ConstantIntOp>(
        sema.getLoc(), isReleased ? 1 : 0, 32);
#endif
  }

  for (auto region : wgOp.getRegions()) {
    auto block = &region->getBlocks().front();
    assignInBlock(block, phaseMap);
  }
}
} // namespace AssignPhase

} // anonymous namespace

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    for (auto wgOp : wgOps)
      AssignPhase::run(wgOp);
    LLVM_DEBUG(llvm::dbgs() << "After semaphoreIndexAssignment\n" << m << "\n");

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerSemaphoreCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
