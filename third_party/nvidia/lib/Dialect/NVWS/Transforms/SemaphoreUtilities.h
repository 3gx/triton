#ifndef NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H
#define NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::nvws {

// ThreadValue<T> threads a value through control flow (scf.for, scf.if),
// updating it at every occurrence of operation type T.
//
// Usage:
//   ThreadValue<SemaphoreAcquireOp>::run(wgOp, initValue, updateValue);
//
// Where:
//   initValue(builder, defOp) -> Value: creates the initial value from the
//     defining op (e.g., SemaphoreCreateOp)
//   updateValue(builder, currentValue, triggerOp) -> Value: computes the next
//     value at each trigger op occurrence
//
// The template:
// 1. Walks all regions of wgOp to find uses of T's first operand (the key)
// 2. Initializes a ValueMap mapping each key to its initial value
// 3. Walks blocks: at each T op, calls updateValue; at scf.for/scf.if,
//    threads the value through iter_args/results

template <class T> struct ThreadValue {
  std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)> updateValue;
  using ValueMap = llvm::MapVector<Value /*key*/, Value /*value*/>;
  using UseSet = llvm::SetVector<Value /*key*/>;

  // Find all keys (first operand of T) used in a block and nested regions.
  UseSet analyzeUseInBlock(Block *block, UseSet useSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        useSet.insert(op.getOperand(0)); // key = semaphore SSA value
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

  // Thread value through a scf.for loop.
  void assignValueInForOp(scf::ForOp forOp, ValueMap &valueMap) {
    auto useInBlock = analyzeUseInBlock(forOp.getBody(), {});
    if (useInBlock.empty())
      return;

    // Add extra iter_args for each key used in the loop body.
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIterArgs.push_back(valueMap.lookup(key));
      valueRefs.push_back(&valueMap[key]);
    }

    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // Update valueMap with the new iter_args inside the loop body.
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // Recursively assign values in the loop body.
    auto valueMapInBlock = assignValueInBlock(forOp.getBody(), valueMap);

    // Append updated values to the yield op.
    SmallVector<Value> extraYieldArgs;
    for (auto key : useInBlock)
      extraYieldArgs.push_back(valueMapInBlock[key]);
    appendToForOpYield(forOp, extraYieldArgs);

    // Update valueMap with loop results.
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *valueRefs[idx - nArgs] = forOp.getResult(idx);
  }

  // Thread value through a scf.if op.
  void assignValueInIfOp(scf::IfOp ifOp, ValueMap &valueMap) {
    auto useInBlock = analyzeUseInBlock(ifOp.thenBlock(), {});
    if (useInBlock.empty())
      return;

    useInBlock = ifOp.elseBlock() ? analyzeUseInBlock(ifOp.elseBlock(), useInBlock)
                                  : useInBlock;

    // Add extra results to the if op.
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> valueRefs;
    for (auto key : useInBlock) {
      extraIfResults.push_back(valueMap.lookup(key).getType());
      valueRefs.push_back(&valueMap[key]);
    }

    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // Assign in then-block and else-block.
    auto thenMap = assignValueInBlock(newIfOp.thenBlock(), valueMap);
    auto elseMap =
        newIfOp.elseBlock() ? assignValueInBlock(newIfOp.elseBlock(), valueMap)
                            : valueMap;

    // Append values to yields.
    auto thenYield = newIfOp.thenYield();
    auto elseYield = newIfOp.elseYield();
    for (auto key : useInBlock) {
      thenYield->insertOperands(thenYield.getNumOperands(), thenMap[key]);
      elseYield->insertOperands(elseYield.getNumOperands(), elseMap[key]);
    }
    ifOp.erase();

    // Update valueMap with if results.
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *valueRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  // Walk a block, updating values at each T op and threading through control
  // flow.
  ValueMap assignValueInBlock(Block *block, ValueMap valueMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        ImplicitLocOpBuilder b(op.getLoc(), &op);
        b.setInsertionPointAfter(&op);
        auto value = valueMap.lookup(op.getOperand(0));
        valueMap[op.getOperand(0)] = updateValue(b, value, &op);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignValueInForOp(forOp, valueMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignValueInIfOp(ifOp, valueMap);
      }
    }
    return valueMap;
  }

  // Entry point: run on a WarpGroupOp.
  static void run(
      WarpGroupOp wgOp,
      std::function<Value(ImplicitLocOpBuilder &, Operation *)> initValue,
      std::function<Value(ImplicitLocOpBuilder &, Value, Operation *)>
          updateValue) {
    ThreadValue<T> value{updateValue};
    UseSet useSet;
    for (auto &region : wgOp->getRegions()) {
      auto block = &region.getBlocks().front();
      useSet = value.analyzeUseInBlock(block, useSet);
    }

    // Initialize values.
    ValueMap valueMap;
    for (auto key : useSet) {
      auto *def = key.getDefiningOp();
      if (!def)
        continue;
      ImplicitLocOpBuilder b(key.getLoc(), def);
      b.setInsertionPointAfter(def);
      valueMap[key] = initValue(b, def);
    }

    // Assign in all regions.
    for (auto &region : wgOp->getRegions()) {
      auto block = &region.getBlocks().front();
      value.assignValueInBlock(block, valueMap);
    }
  }
};

} // namespace mlir::triton::nvws

#endif // NVWS_TRANSFORMS_SEMAPHORE_UTILITIES_H
