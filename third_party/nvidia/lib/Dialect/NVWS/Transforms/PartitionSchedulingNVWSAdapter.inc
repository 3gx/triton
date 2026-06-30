// NVWS-only completion of partition metadata required by PartitionLoops.
// Included inside PartitionSchedulingMeta.cpp's anonymous namespace so the
// copied Meta scheduling algorithm remains mechanically recognizable.

static SetVector<int> getChildPartitionUnion(Operation *op) {
  SetVector<int> result;
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &child : block.getOperations()) {
        if (isa<ub::PoisonOp>(child))
          continue;
        if (hasPartition(&child))
          insertAll(result, getPartitionIds(&child));
      }
    }
  }
  return result;
}
static SetVector<int> getConsumerPartitionUnion(Value value) {
  SetVector<int> result;
  DenseSet<Value> visited;

  std::function<void(Value)> collect = [&](Value value) {
    if (!visited.insert(value).second)
      return;

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yieldOp->getParentOp();
        unsigned idx = use.getOperandNumber();
        if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          if (idx < forOp.getNumRegionIterArgs())
            collect(forOp.getRegionIterArg(idx));
          if (idx < forOp.getNumResults())
            collect(forOp.getResult(idx));
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          if (idx < ifOp.getNumResults())
            collect(ifOp.getResult(idx));
          continue;
        }
      }

      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        int idx = use.getOperandNumber() - forOp.getNumControlOperands();
        if (idx >= 0) {
          // PartitionLoops sees the region op itself as the direct SSA
          // consumer for iter_args. Scalar/address producers therefore need
          // to cover the whole loop partition set, not only the partitions of
          // the block-argument uses inside the loop body.
          if (hasPartition(forOp))
            insertAll(result, getPartitionIds(forOp));
          if (static_cast<unsigned>(idx) < forOp.getNumRegionIterArgs())
            collect(forOp.getRegionIterArg(idx));
          if (static_cast<unsigned>(idx) < forOp.getNumResults())
            collect(forOp.getResult(idx));
          continue;
        }
      }

      if (auto reduceReturn = dyn_cast<ReduceReturnOp>(user)) {
        if (auto reduceOp = dyn_cast<ReduceOp>(reduceReturn->getParentOp())) {
          unsigned idx = use.getOperandNumber();
          if (idx < reduceOp->getNumResults())
            collect(reduceOp->getResult(idx));
        }
        continue;
      }

      if (hasPartition(user))
        insertAll(result, getPartitionIds(&use));
    }
  };

  collect(value);
  return result;
}

static bool isScalarOrAddressGlueType(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<PointerType>(type))
    return true;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorType.getElementType();
    return elementType.isIntOrIndex() || isa<PointerType>(elementType);
  }
  return false;
}

static bool isScalarOrAddressPartitionGlueOp(Operation *op) {
  if (op->getNumResults() == 0)
    return false;
  return llvm::all_of(op->getResults(), [](Value value) {
    return isScalarOrAddressGlueType(value.getType());
  });
}

static SetVector<int> getOperandDefPartitionUnion(Operation *op) {
  SetVector<int> result;
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (hasPartition(def))
        insertAll(result, getPartitionIds(def));
    }
  }
  return result;
}

static void collectYieldedValues(Operation *op, unsigned resultIdx,
                                 SmallVectorImpl<Value> &values) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Operation *terminator = forOp.getBody()->getTerminator();
    if (resultIdx < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIdx));
    return;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      Operation *terminator = region->front().getTerminator();
      if (resultIdx < terminator->getNumOperands())
        values.push_back(terminator->getOperand(resultIdx));
    }
    return;
  }

  if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
    Region &region = reduceOp.getCombineOp();
    if (region.empty())
      return;
    Operation *terminator = region.front().getTerminator();
    if (resultIdx < terminator->getNumOperands())
      values.push_back(terminator->getOperand(resultIdx));
  }
}

static void insertPartitionForUser(SetVector<int> &result, Operation *user,
                                   Block *scope = nullptr) {
  if (hasPartition(user)) {
    insertAll(result, getPartitionIds(user));
    return;
  }
  if (!scope)
    return;
  if (Operation *ancestor = scope->findAncestorOpInBlock(*user)) {
    if (hasPartition(ancestor))
      insertAll(result, getPartitionIds(ancestor));
  }
}

static SetVector<int> inferAsyncTokenResultPartition(Operation *op,
                                                     unsigned resultIdx) {
  SetVector<int> result;
  if (resultIdx >= op->getNumResults() ||
      !isa<AsyncTokenType>(op->getResult(resultIdx).getType()))
    return result;

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (resultIdx >= forOp.getNumRegionIterArgs())
      return result;
    BlockArgument arg = forOp.getRegionIterArg(resultIdx);
    for (OpOperand &use : arg.getUses())
      insertPartitionForUser(result, use.getOwner(), forOp.getBody());
    return result;
  }

  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIdx, yieldedValues);
  for (Value value : yieldedValues) {
    if (Operation *def = value.getDefiningOp())
      insertPartitionForUser(result, def);
  }
  return result;
}

static void insertYieldedProducerPartitions(SetVector<int> &result,
                                            Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return;

  if (def->getNumRegions() != 0 && def->hasAttr(kPartitionOutputsAttrName)) {
    auto it = llvm::find(def->getResults(), value);
    if (it != def->result_end()) {
      unsigned pos = std::distance(def->result_begin(), it);
      auto outputs = getPartitionOutputs(def);
      if (pos < outputs.size()) {
        insertAll(result, outputs[pos]);
        return;
      }
    }
  }

  if (hasPartition(def))
    insertAll(result, getPartitionIds(def));
}

static SetVector<int> inferResultPartition(Operation *op, unsigned resultIdx) {
  SetVector<int> tokenResult = inferAsyncTokenResultPartition(op, resultIdx);
  if (!tokenResult.empty())
    return tokenResult;

  SetVector<int> result;
  SmallVector<Value> yieldedValues;
  collectYieldedValues(op, resultIdx, yieldedValues);
  for (Value value : yieldedValues)
    insertYieldedProducerPartitions(result, value);

  if (result.empty() && resultIdx < op->getNumResults())
    insertAll(result, getConsumerPartitionUnion(op->getResult(resultIdx)));

  if (result.empty() && hasPartition(op))
    insertAll(result, getPartitionIds(op));
  return result;
}

static SetVector<int> inferOpPartition(Operation *op) {
  SetVector<int> result;
  for (Value value : op->getResults())
    insertAll(result, getConsumerPartitionUnion(value));
  insertAll(result, getChildPartitionUnion(op));
  if (result.empty())
    insertAll(result, getOperandDefPartitionUnion(op));
  if (result.empty()) {
    if (Operation *parent = op->getParentOp()) {
      if (hasPartition(parent))
        insertAll(result, getPartitionIds(parent));
    }
  }
  return result;
}

static SetVector<int> inferScalarPartition(Operation *op) {
  SetVector<int> result;
  if (hasPartition(op))
    insertAll(result, getPartitionIds(op));
  insertAll(result, inferOpPartition(op));
  return result;
}

static bool setPartitionIfChanged(Operation *op, const SetVector<int> &ids) {
  if (ids.empty())
    return false;
  if (hasPartition(op) && getPartitionIds(op) == ids)
    return false;
  setPartition(op, ids);
  return true;
}

static void setRegionResultOutputs(Operation *op) {
  if (!isa<scf::ForOp, scf::IfOp, ReduceOp>(op) || op->getNumResults() == 0)
    return;

  SmallVector<SetVector<int>> outputs;
  outputs.reserve(op->getNumResults());
  SetVector<int> unionIds;
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    SetVector<int> ids = inferResultPartition(op, i);
    if (ids.empty() && hasPartition(op))
      insertAll(ids, getPartitionIds(op));
    outputs.push_back(ids);
    insertAll(unionIds, ids);
  }

  if (hasPartition(op))
    insertAll(unionIds, getPartitionIds(op));
  insertAll(unionIds, getChildPartitionUnion(op));
  if (!unionIds.empty())
    setPartition(op, unionIds);
  setPartitionOutputs(op, outputs);
}

static void finalizePartitionAnnotations(scf::ForOp loop) {
  bool changed = false;
  do {
    changed = false;
    loop.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isa<ub::PoisonOp>(op))
        return;

      if (isScalarOrAddressPartitionGlueOp(op))
        changed |= setPartitionIfChanged(op, inferScalarPartition(op));
      else if (!hasPartition(op))
        changed |= setPartitionIfChanged(op, inferOpPartition(op));

      if (op->getNumRegions() != 0) {
        SetVector<int> ids;
        if (hasPartition(op))
          insertAll(ids, getPartitionIds(op));
        insertAll(ids, getChildPartitionUnion(op));
        for (unsigned i = 0; i < op->getNumResults(); ++i)
          insertAll(ids, inferResultPartition(op, i));
        changed |= setPartitionIfChanged(op, ids);
      }
    });
  } while (changed);

  loop.walk<WalkOrder::PostOrder>(
      [&](Operation *op) { setRegionResultOutputs(op); });
}

static bool isNestedInWarpSpecializeLoop(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return true;
    }
  }
  return false;
}

static bool isWarpSpecializeLoopOp(Operation *op) {
  auto forOp = dyn_cast<scf::ForOp>(op);
  return forOp && forOp->hasAttr(kWarpSpecializeAttrName);
}

static std::optional<int> getEnclosingWarpSpecializeTag(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return getWarpSpecializeTag(forOp);
    }
  }
  return std::nullopt;
}

static std::optional<int> getAssociatedWarpSpecializeTag(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    if (forOp->hasAttr(kWarpSpecializeAttrName))
      return getWarpSpecializeTag(forOp);
  }
  if (auto tag = getWarpSpecializeTag(def))
    return tag;
  return getEnclosingWarpSpecializeTag(def);
}

static SetVector<int> inferAssociatedWarpSpecializeTags(Operation *op) {
  SetVector<int> tags;

  for (Value operand : op->getOperands()) {
    if (auto tag = getAssociatedWarpSpecializeTag(operand))
      tags.insert(*tag);
  }

  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (auto tag = getWarpSpecializeTag(user))
        tags.insert(*tag);
      if (auto tag = getEnclosingWarpSpecializeTag(user))
        tags.insert(*tag);
    }
  }

  return tags;
}

static LogicalResult tagPartitionedOpsForFunc(FuncOp funcOp,
                                              ArrayRef<scf::ForOp> loops) {
  SetVector<int> loopTags;
  for (scf::ForOp loop : loops) {
    auto tag = getWarpSpecializeTag(loop);
    assert(tag && "WS loop must be tagged before final tag propagation");
    loopTags.insert(*tag);
  }

  bool changed = false;
  do {
    changed = false;
    Operation *ambiguous = nullptr;
    SetVector<int> ambiguousTags;

    funcOp.walk([&](Operation *op) {
      if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
          isWarpSpecializeLoopOp(op))
        return WalkResult::advance();

      SetVector<int> tags = inferAssociatedWarpSpecializeTags(op);
      if (tags.size() > 1) {
        ambiguous = op;
        ambiguousTags = tags;
        return WalkResult::interrupt();
      }

      if (tags.empty() && loopTags.size() == 1)
        tags.insert(loopTags.front());

      if (tags.size() == 1) {
        int inferredTag = tags.front();
        if (auto existingTag = getWarpSpecializeTag(op)) {
          if (*existingTag != inferredTag) {
            ambiguous = op;
            ambiguousTags.insert(*existingTag);
            ambiguousTags.insert(inferredTag);
            return WalkResult::interrupt();
          }
        } else {
          setWarpSpecializeTag(op, inferredTag);
          changed = true;
        }
      }

      return WalkResult::advance();
    });

    if (ambiguous) {
      return ambiguous->emitError(
          "partitioned op outside tt.warp_specialize loop has ambiguous "
          "ttg.warp_specialize.tag association");
    }
  } while (changed);

  Operation *untagged = nullptr;
  funcOp.walk([&](Operation *op) {
    if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
        isWarpSpecializeLoopOp(op))
      return WalkResult::advance();
    if (!hasWarpSpecializeTag(op)) {
      untagged = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (untagged) {
    return untagged->emitError(
        "partitioned op outside tt.warp_specialize loop is missing an "
        "unambiguous ttg.warp_specialize.tag association");
  }

  Operation *badUseClosure = nullptr;
  funcOp.walk([&](Operation *op) {
    if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op) ||
        isWarpSpecializeLoopOp(op))
      return WalkResult::advance();
    std::optional<int> tag = getWarpSpecializeTag(op);
    if (!tag)
      return WalkResult::advance();
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (isNestedInWarpSpecializeLoop(user)) {
          auto userTag = getEnclosingWarpSpecializeTag(user);
          if (!userTag || *userTag != *tag) {
            badUseClosure = op;
            return WalkResult::interrupt();
          }
          continue;
        }
        if (!hasPartition(user) || !hasWarpSpecializeTag(user) ||
            *getWarpSpecializeTag(user) != *tag) {
          badUseClosure = op;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (badUseClosure) {
    return badUseClosure->emitError(
        "partitioned op outside tt.warp_specialize loop has a result use "
        "outside its WS tag closure");
  }

  return success();
}

static LogicalResult verifyFinalizedPartitionAnnotations(FuncOp funcOp,
                                                         scf::ForOp loop) {
  Operation *missingPartition = nullptr;
  loop.walk([&](Operation *op) {
    if (isa<ub::PoisonOp>(op))
      return WalkResult::advance();
    if (!hasPartition(op)) {
      missingPartition = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (missingPartition)
    return missingPartition->emitError("missing ttg.partition after NVWS "
                                       "partition-scheduling-meta finalizer");

  Operation *badOutputs = nullptr;
  loop.walk([&](Operation *op) {
    if (isa<scf::ForOp, scf::IfOp, ReduceOp>(op) && op->getNumResults() != 0 &&
        !op->hasAttr(kPartitionOutputsAttrName)) {
      badOutputs = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (badOutputs)
    return badOutputs->emitError("missing ttg.partition.outputs after NVWS "
                                 "partition-scheduling-meta finalizer");

  Operation *badGlueUse = nullptr;
  loop.walk([&](Operation *op) {
    if (!isScalarOrAddressPartitionGlueOp(op) || !hasPartition(op))
      return WalkResult::advance();
    SetVector<int> producerIds = getPartitionIds(op);
    for (Value result : op->getResults()) {
      SetVector<int> consumerIds = getConsumerPartitionUnion(result);
      for (int id : consumerIds) {
        if (!producerIds.contains(id)) {
          badGlueUse = op;
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (badGlueUse)
    return badGlueUse->emitError(
        "scalar/address glue op partition does not cover all partitioned "
        "consumers after NVWS partition-scheduling-meta finalizer");

  Operation *badTag = nullptr;
  funcOp.walk([&](Operation *op) {
    if (!hasPartition(op) || isNestedInWarpSpecializeLoop(op))
      return WalkResult::advance();
    if (!op->hasAttr(kWarpSpecializeTagAttrName)) {
      badTag = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (badTag)
    return badTag->emitError("partitioned op outside tt.warp_specialize loop "
                             "is missing ttg.warp_specialize.tag");

  return verifyPartitionedLoop(loop);
}
