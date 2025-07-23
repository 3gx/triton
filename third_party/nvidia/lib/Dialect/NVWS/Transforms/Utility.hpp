#ifndef NVWS_TRANSFORMS_UTILITY_HPP
#define NVWS_TRANSFORMS_UTILITY_HPP

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

template <class T> struct ArefStage {
  std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)> updateValue;
  using StageMap = llvm::MapVector<Value /*aref*/, Value /*stage*/>;
  using UseSet = llvm::SetVector<Value /*aref*/>;

  UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(op.getOperand(0));
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

  void assignStageInForOp(scf::ForOp forOp, StageMap &stageMap) {

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

  void assignStageInIfOp(scf::IfOp ifOp, StageMap &stageMap) {

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

  StageMap assignStageInBlock(Block *block, StageMap stageMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        ImplicitLocOpBuilder b(op.getLoc(), &op);
        b.setInsertionPointAfter(&op);
        auto value = stageMap.lookup(op.getOperand(0));
        op.setOperand(1, value);
        stageMap[op.getOperand(0)] = updateValue(b, value, &op);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignStageInForOp(forOp, stageMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignStageInIfOp(ifOp, stageMap);
      }
    }

    return stageMap;
  }

  static void
  run(triton::nvws::WarpGroupOp wgOp,
      std::function<Value(ImplicitLocOpBuilder &, Operation *)> initValue,
      std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)>
          updateValue) {
    ArefStage<T> stage{updateValue};
    UseSet useSet;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      useSet = stage.analyzeUseInBlock(block, useSet);
    }

    // initialize indexes
    StageMap stageMap;
    for (auto aref : useSet) {
      ImplicitLocOpBuilder b(aref.getLoc(), aref.getDefiningOp());
      b.setInsertionPointAfter(aref.getDefiningOp());
      stageMap[aref] = initValue(b, aref.getDefiningOp());
    }

    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      stage.assignStageInBlock(block, stageMap);
    }
  }
};
} // namespace mlir

#endif // NVWS_TRANSFORMS_UTILITY_HPP