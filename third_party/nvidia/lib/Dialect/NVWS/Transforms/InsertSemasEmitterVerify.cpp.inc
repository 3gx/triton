static unsigned countPlannedAnchors(const DenseMap<Operation *, SmallVector<unsigned, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(const DenseMap<Region *, SmallVector<unsigned, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static unsigned countPlannedAnchors(
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &map) {
  unsigned count = 0;
  for (auto &kv : map)
    count += kv.second.size();
  return count;
}

static const char *syncGroupKindName(SyncGroupKind kind) {
  switch (kind) {
  case SyncGroupKind::InitialEmpty: return "initial-empty";
  case SyncGroupKind::Singleton: return "singleton";
  case SyncGroupKind::ReadyFanout: return "ready-fanout";
  case SyncGroupKind::DoneFanin: return "done-fanin";
  case SyncGroupKind::LinearChain: return "linear-chain";
  }
  llvm_unreachable("unknown sync group kind");
}

static bool hasSameSrcKey(const SyncEdge &a, const SyncEdge &b) {
  return a.srcOwner == b.srcOwner && a.srcEpoch == b.srcEpoch &&
         a.srcOp == b.srcOp && a.srcYieldRegion == b.srcYieldRegion;
}

static bool hasSameDstKey(const SyncEdge &a, const SyncEdge &b) {
  return a.dstOwner == b.dstOwner && a.dstOp == b.dstOp &&
         a.dstYieldRegion == b.dstYieldRegion;
}

static LogicalResult verifyLinearChainGroup(triton::FuncOp funcOp,
                                            const SyncPlan &sp,
                                            const SyncGroup &group) {
  if (group.edgeIdxs.size() < 2)
    return funcOp.emitError("nvws-insert-semas: verifier: linear-chain group "
                            "has fewer than two edges");
  SmallVector<ReadyFanoutKey, 4> srcKeys;
  SmallVector<DoneFaninKey, 4> dstKeys;
  for (auto [pos, idx] : llvm::enumerate(group.edgeIdxs)) {
    const SyncEdge &edge = sp.edges[idx];
    if ((!edge.srcOp && !edge.srcYieldRegion) ||
        (!edge.dstOp && !edge.dstYieldRegion))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has an unanchored endpoint");
    if (edge.srcYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has a yield source");
    if (edge.dstYieldRegion && !isIfYieldRegion(edge.dstYieldRegion) &&
        (pos + 1 != group.edgeIdxs.size() || edge.kind != SyncEdgeKind::Done))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "has a non-terminal yield target");
    if (edge.srcOp && isa<scf::ForOp, scf::IfOp>(edge.srcOp))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain edge "
                              "crosses a structured CFG anchor");
    ReadyFanoutKey srcKey{edge.srcOwner, edge.srcEpoch, edge.srcOp,
                          edge.srcYieldRegion};
    if (llvm::is_contained(srcKeys, srcKey))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain has "
                              "fanout from one produced version");
    srcKeys.push_back(srcKey);
    DoneFaninKey dstKey{edge.dstOwner, edge.dstOp, edge.dstYieldRegion};
    if (llvm::is_contained(dstKeys, dstKey))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain has "
                              "fanin to one target event");
    dstKeys.push_back(dstKey);
  }
  for (unsigned i = 1; i < group.edgeIdxs.size(); ++i) {
    const SyncEdge &prev = sp.edges[group.edgeIdxs[i - 1]];
    const SyncEdge &cur = sp.edges[group.edgeIdxs[i]];
    if (!sameOwner(prev.dstOwner, cur.srcOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: linear-chain owner "
                              "sequence is not contiguous");
  }
  return success();
}

static LogicalResult verifyPlanBeforeEmission(triton::FuncOp funcOp,
                                              BufferGroup &group,
                                              const ResourcePlan &plan,
                                              const SyncPlan &sp,
                                              const OptSyncDag &dag) {
  if (sp.resource != dag.resource || sp.groupIdx != dag.groupIdx)
    return funcOp.emitError("nvws-insert-semas: verifier: OPT-SYNC resource "
                            "does not match RAW-SYNC resource");
  if (dag.edgeToGroup.size() != sp.edges.size())
    return funcOp.emitError("nvws-insert-semas: verifier: edge-to-group map "
                            "does not cover every raw edge");

  for (Operation *op : dag.accessOps) {
    if (!plan.useOwner.contains(op))
      return op->emitError("nvws-insert-semas: verifier: planned access has no "
                           "ownership record");
    const AccessEvent *event = findEvent(group, op);
    if (!event || !eventTouchesResource(*event, dag.resource.second))
      return op->emitError("nvws-insert-semas: verifier: planned access has no "
                           "touch for this resource");
  }

  for (const SyncEdge &edge : sp.edges) {
    if (edge.producedVersion.resourceKey != sp.resource.second)
      return funcOp.emitError("nvws-insert-semas: verifier: edge produced "
                              "version does not include this resource key");
    if (edge.producedVersion.epoch == 0)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "produced version epoch");
    if (!sameOwner(edge.postOwner, edge.dstOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: edge post-owner "
                              "does not match destination owner");
    if (edge.carrierChoice == CarrierTokenChoice::None)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "carrier-token choice");
    if (!edge.dstOp && !edge.dstYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "planned acquire anchor");
    if (!edge.srcOp && !edge.srcYieldRegion)
      return funcOp.emitError("nvws-insert-semas: verifier: edge has no "
                              "planned release anchor");
    if (edge.srcOp && edge.dstOp && edge.dstOp->isProperAncestor(edge.srcOp))
      return edge.dstOp->emitError(
          "nvws-insert-semas: verifier: acquire would wait on a release "
          "control-dependent on the same acquire");
    if ((edge.kind == SyncEdgeKind::Ready ||
         edge.kind == SyncEdgeKind::Handoff) &&
        !sameOwner(edge.preOwner, edge.srcOwner))
      return funcOp.emitError("nvws-insert-semas: verifier: edge pre-owner "
                              "does not match source owner");
  }

  std::optional<PartitionId> firstWriterOwner;
  Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner);
  unsigned initialGroups = 0;
  for (auto [groupIdx, syncGroup] : llvm::enumerate(dag.groups)) {
    switch (syncGroup.kind) {
    case SyncGroupKind::InitialEmpty:
      ++initialGroups;
      if (!firstWriter)
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group without a writer");
      if (syncGroup.initialOp != firstWriter ||
          syncGroup.initialOwner != firstWriterOwner)
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group is not anchored at the first writer");
      if (!syncGroup.edgeIdxs.empty())
        return funcOp.emitError("nvws-insert-semas: verifier: initial-empty "
                                "group must not own raw edges");
      break;
    case SyncGroupKind::Singleton:
      if (syncGroup.edgeIdxs.size() != 1)
        return funcOp.emitError("nvws-insert-semas: verifier: singleton group "
                                "does not contain exactly one edge");
      break;
    case SyncGroupKind::ReadyFanout: {
      if (syncGroup.edgeIdxs.size() < 2)
        return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                "group has fewer than two edges");
      const SyncEdge &probe = sp.edges[syncGroup.edgeIdxs.front()];
      if (!probe.srcOp && !probe.srcYieldRegion)
        return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                "group has no release anchor");
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (edge.kind != SyncEdgeKind::Ready)
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "contains a non-ready edge");
        if (!hasSameSrcKey(probe, edge))
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "group does not share one produced version");
        if (!(probe.producedVersion == edge.producedVersion))
          return funcOp.emitError("nvws-insert-semas: verifier: ready-fanout "
                                  "group does not share one produced version "
                                  "key");
        if (edge.dstOp) {
          const AccessEvent *dst = findEvent(group, edge.dstOp);
          if (dst && (!eventConsumes(*dst, dag.resource.second) ||
                      eventProduces(*dst, dag.resource.second)))
            return edge.dstOp->emitError(
                "nvws-insert-semas: verifier: ready-fanout target is not a "
                "read-only consumer");
        }
      }
      break;
    }
    case SyncGroupKind::DoneFanin: {
      if (syncGroup.edgeIdxs.size() < 2)
        return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                "group has fewer than two edges");
      const SyncEdge &probe = sp.edges[syncGroup.edgeIdxs.front()];
      SmallVector<std::optional<PartitionId>, 4> srcOwners;
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (edge.kind != SyncEdgeKind::Done)
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "contains a non-done edge");
        if (!hasSameDstKey(probe, edge))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "group does not share one target event");
        if (!(probe.producedVersion == edge.producedVersion))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin "
                                  "group does not retire one produced version "
                                  "key");
        if (llvm::is_contained(srcOwners, edge.srcOwner))
          return funcOp.emitError("nvws-insert-semas: verifier: done-fanin has "
                                  "multiple releases from one source owner");
        srcOwners.push_back(edge.srcOwner);
      }
      break;
    }
    case SyncGroupKind::LinearChain:
      if (failed(verifyLinearChainGroup(funcOp, sp, syncGroup)))
        return failure();
      break;
    }

    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      if (edgeIdx >= sp.edges.size())
        return funcOp.emitError("nvws-insert-semas: verifier: ")
               << syncGroupKindName(syncGroup.kind)
               << " group references an out-of-range raw edge";
      if (dag.edgeToGroup[edgeIdx] != groupIdx)
        return funcOp.emitError("nvws-insert-semas: verifier: edge-to-group "
                                "map disagrees with OPT-SYNC group");
    }
  }
  if (firstWriter && initialGroups != 1)
    return firstWriter->emitError("nvws-insert-semas: verifier: expected one "
                                  "initial-empty group for first writer");
  if (!firstWriter && initialGroups != 0)
    return funcOp.emitError("nvws-insert-semas: verifier: unexpected "
                            "initial-empty group");

  auto emitReleasePlanError = [&](Operation *anchor, const Twine &message) {
    if (anchor)
      return anchor->emitError(message);
    return funcOp.emitError(message);
  };
  auto verifyReleaseAction = [&](const PlannedRelease &action,
                                 Operation *anchor) -> LogicalResult {
    if (action.groupIdx >= dag.groups.size())
      return emitReleasePlanError(
          anchor, "nvws-insert-semas: verifier: planned release references an "
                  "invalid group");
    if (action.edgeIdxs.empty())
      return emitReleasePlanError(
          anchor, "nvws-insert-semas: verifier: planned release has no "
                  "transition edge");
    for (unsigned edgeIdx : action.edgeIdxs) {
      if (edgeIdx >= sp.edges.size())
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release references "
                    "an invalid edge");
      if (edgeIdx >= dag.edgeToGroup.size() ||
          dag.edgeToGroup[edgeIdx] != action.groupIdx)
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release edge does "
                    "not belong to its group");
      if (!edgeRequiresRelease(sp.edges[edgeIdx]))
        return emitReleasePlanError(
            anchor, "nvws-insert-semas: verifier: planned release is not "
                    "backed by a partition transition edge");
    }
    return success();
  };
  auto verifyReleaseOpMap =
      [&](const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &map)
      -> LogicalResult {
    for (auto &kv : map)
      for (const PlannedRelease &action : kv.second)
        if (failed(verifyReleaseAction(action, kv.first)))
          return failure();
    return success();
  };
  auto verifyReleaseYieldMap =
      [&](const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &map)
      -> LogicalResult {
    for (auto &kv : map)
      for (const PlannedRelease &action : kv.second)
        if (failed(verifyReleaseAction(
                action, kv.first ? kv.first->getParentOp() : nullptr)))
          return failure();
    return success();
  };
  if (failed(verifyReleaseOpMap(dag.releaseBeforeOp)) ||
      failed(verifyReleaseOpMap(dag.releaseAfterOp)) ||
      failed(verifyReleaseYieldMap(dag.releaseBeforeYield)) ||
      failed(verifyReleaseYieldMap(dag.releaseAfterYield)))
    return failure();

  WalkResult walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionContainsAccess(forOp.getRegion(), dag))
        return WalkResult::advance();
      if (!dag.threadForOps.contains(forOp.getOperation()))
        return WalkResult::advance();
      auto it = plan.regionOwners.find(&forOp.getRegion());
      if (it == plan.regionOwners.end() || !it->second.carried ||
          !sameOwner(it->second.entry, it->second.exit)) {
        op->emitError("nvws-insert-semas: verifier: scf.for has no known "
                      "loop-carried owner/state for this resource");
        return WalkResult::interrupt();
      }
      if (!plan.yieldOwner.contains(getForYieldOp(forOp))) {
        op->emitError("nvws-insert-semas: verifier: scf.for has no planned "
                      "yield owner for this resource");
        return WalkResult::interrupt();
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionContainsAccess(ifOp.getThenRegion(), dag);
      bool elseHas = regionContainsAccess(ifOp.getElseRegion(), dag);
      if (!thenHas && !elseHas)
        return WalkResult::advance();
      if (!dag.threadIfOps.contains(ifOp.getOperation()))
        return WalkResult::advance();
      auto thenIt = plan.regionOwners.find(&ifOp.getThenRegion());
      auto elseIt = plan.regionOwners.find(&ifOp.getElseRegion());
      if (thenIt == plan.regionOwners.end() ||
          elseIt == plan.regionOwners.end() ||
          !sameOwner(thenIt->second.entry, thenIt->second.exit) ||
          !sameOwner(elseIt->second.entry, elseIt->second.exit) ||
          !sameOwner(thenIt->second.exit, elseIt->second.exit)) {
        op->emitError("nvws-insert-semas: verifier: scf.if join has no known "
                      "owner/state on every path for this resource");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

static bool hasSyncRecord(ArrayRef<EmittedSyncRecord> records, unsigned groupIdx,
                          SyncAnchorKind kind, Operation *anchor,
                          Region *yieldRegion) {
  return llvm::any_of(records, [&](const EmittedSyncRecord &record) {
    return record.groupIdx == groupIdx && record.kind == kind &&
           record.anchor == anchor && record.yieldRegion == yieldRegion &&
           record.op;
  });
}

static bool hasSyncRecord(ArrayRef<EmittedSyncRecord> records,
                          const PlannedRelease &action, SyncAnchorKind kind,
                          Operation *anchor, Region *yieldRegion) {
  return llvm::any_of(records, [&](const EmittedSyncRecord &record) {
    return record.groupIdx == action.groupIdx && record.edgeIdxs == action.edgeIdxs &&
           record.kind == kind && record.anchor == anchor &&
           record.yieldRegion == yieldRegion && record.op;
  });
}

static bool isRecordPlanned(const EmittedSyncRecord &record,
                            const DenseMap<Operation *, SmallVector<unsigned, 2>>
                                &opMap,
                            const DenseMap<Region *, SmallVector<unsigned, 2>>
                                &yieldMap) {
  if (record.anchor) {
    auto it = opMap.find(record.anchor);
    return it != opMap.end() && llvm::is_contained(it->second, record.groupIdx);
  }
  if (record.yieldRegion) {
    auto it = yieldMap.find(record.yieldRegion);
    return it != yieldMap.end() &&
           llvm::is_contained(it->second, record.groupIdx);
  }
  return false;
}

static bool isRecordPlanned(
    const EmittedSyncRecord &record,
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &opMap,
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &yieldMap) {
  auto matches = [&](ArrayRef<PlannedRelease> planned) {
    return llvm::any_of(planned, [&](const PlannedRelease &action) {
      return action.groupIdx == record.groupIdx &&
             action.edgeIdxs == record.edgeIdxs;
    });
  };
  if (record.anchor) {
    auto it = opMap.find(record.anchor);
    return it != opMap.end() && matches(it->second);
  }
  if (record.yieldRegion) {
    auto it = yieldMap.find(record.yieldRegion);
    return it != yieldMap.end() && matches(it->second);
  }
  return false;
}

static LogicalResult verifyPlannedSyncRecords(
    triton::FuncOp funcOp, SyncAnchorKind opKind, SyncAnchorKind yieldKind,
    const DenseMap<Operation *, SmallVector<unsigned, 2>> &opMap,
    const DenseMap<Region *, SmallVector<unsigned, 2>> &yieldMap,
    ArrayRef<EmittedSyncRecord> records) {
  for (auto &kv : opMap)
    for (unsigned groupIdx : kv.second)
      if (!hasSyncRecord(records, groupIdx, opKind, kv.first, nullptr))
        return kv.first->emitError("nvws-insert-semas: post-emission verifier: "
                                   "planned sync op is missing at op anchor");
  for (auto &kv : yieldMap)
    for (unsigned groupIdx : kv.second)
      if (!hasSyncRecord(records, groupIdx, yieldKind, nullptr, kv.first))
        return funcOp.emitError("nvws-insert-semas: post-emission verifier: "
                                "planned sync op is missing at yield anchor");
  return success();
}

static LogicalResult verifyPlannedSyncRecords(
    triton::FuncOp funcOp, SyncAnchorKind opKind, SyncAnchorKind yieldKind,
    const DenseMap<Operation *, SmallVector<PlannedRelease, 2>> &opMap,
    const DenseMap<Region *, SmallVector<PlannedRelease, 2>> &yieldMap,
    ArrayRef<EmittedSyncRecord> records) {
  for (auto &kv : opMap)
    for (const PlannedRelease &action : kv.second)
      if (!hasSyncRecord(records, action, opKind, kv.first, nullptr))
        return kv.first->emitError("nvws-insert-semas: post-emission verifier: "
                                   "planned release is missing at op anchor");
  for (auto &kv : yieldMap)
    for (const PlannedRelease &action : kv.second)
      if (!hasSyncRecord(records, action, yieldKind, nullptr, kv.first))
        return funcOp.emitError("nvws-insert-semas: post-emission verifier: "
                                "planned release is missing at yield anchor");
  return success();
}

static LogicalResult verifyStageCluster(Operation *emitted,
                                        StageCluster expected) {
  if (!expected)
    return success();
  StageCluster actual = getStageCluster(emitted);
  if (!actual || *actual != *expected)
    return emitted->emitError("nvws-insert-semas: post-emission verifier: "
                              "emitted semaphore op is missing planned "
                              "stage/cluster attrs");
  return success();
}

static bool operationUsesValue(Operation *op, Value value) {
  for (OpOperand &operand : op->getOpOperands())
    if (operand.get() == value)
      return true;
  return false;
}

static Operation *nearestStructuredParent(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<scf::ForOp, scf::IfOp>(parent))
      return parent;
  return nullptr;
}

static bool tokenEscapesWithoutYield(Value token) {
  Operation *def = token.getDefiningOp();
  if (!def)
    return false;
  Operation *boundary = nearestStructuredParent(def);
  if (!boundary)
    return false;
  for (Operation *user : token.getUsers()) {
    if (boundary->isAncestor(user))
      continue;
    if (isa<scf::YieldOp>(user))
      continue;
    return true;
  }
  return false;
}

static bool partitionSetMatchesOwner(SetVector<int> partitions,
                                     std::optional<PartitionId> owner) {
  if (!owner)
    return true;
  return partitions.size() == 1 && *partitions.begin() == owner->first;
}

static LogicalResult verifyThreadRecord(const ThreadRecord &record,
                                        const ResourcePlan &plan,
                                        const SyncPlan &sp,
                                        const OptSyncDag &dag) {
  if (!record.token || !isa<AsyncTokenType>(record.token.getType()))
    return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                "planned CFG-threaded carrier is not an "
                                "async token");
  switch (record.kind) {
  case ThreadRecordKind::ForIterArg: {
    auto forOp = dyn_cast<scf::ForOp>(record.op);
    if (!forOp || !isa<BlockArgument>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.for iter_arg carrier is absent");
    Region *region = record.plannedRegion ? record.plannedRegion
                                           : &forOp.getRegion();
    auto it = plan.regionOwners.find(region);
    if (it == plan.regionOwners.end() ||
        !sameOwner(it->second.entry, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.for iter_arg carrier owner does not "
                                  "match the ownership plan");
    return success();
  }
  case ThreadRecordKind::ForResult: {
    auto forOp = dyn_cast<scf::ForOp>(record.op);
    if (!forOp || !isa<OpResult>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.for result carrier is absent");
    Region *region = record.plannedRegion ? record.plannedRegion
                                           : &forOp.getRegion();
    auto it = plan.regionOwners.find(region);
    std::optional<PartitionId> expectedOwner;
    if (it != plan.regionOwners.end())
      expectedOwner = it->second.exit;
    if (auto linearOwner =
            linearChainLoopYieldSourceOwner(forOp.getOperation(), dag, sp))
      expectedOwner = linearOwner;
    if (it == plan.regionOwners.end() || !sameOwner(expectedOwner, record.owner))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "scf.for result carrier owner does not "
                                  "match the ownership plan");
    if (hasPartition(forOp)) {
      auto outputs = getPartitionOutputs(forOp);
      auto result = cast<OpResult>(record.token);
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber >= outputs.size() ||
          !partitionSetMatchesOwner(outputs[resultNumber], record.owner))
        return record.op->emitError(
            "nvws-insert-semas: post-emission verifier: scf.for result "
            "carrier partition output does not match the ownership plan");
    }
    return success();
  }
  case ThreadRecordKind::IfResult: {
    auto ifOp = dyn_cast<scf::IfOp>(record.op);
    if (!ifOp || !isa<OpResult>(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "planned scf.if result carrier is absent");
    return success();
  }
  }
  llvm_unreachable("unknown thread record kind");
}

static LogicalResult verifyPostEmission(const OptSyncDag &dag, BufferGroup &group,
                                        const ResourcePlan &plan,
                                        const SyncPlan &sp,
                                        const EmitState &state) {
  unsigned plannedAcquires =
      countPlannedAnchors(dag.acquireBeforeOp) +
      countPlannedAnchors(dag.acquireBeforeYield);
  unsigned plannedReleases =
      countPlannedAnchors(dag.releaseBeforeOp) +
      countPlannedAnchors(dag.releaseBeforeYield) +
      countPlannedAnchors(dag.releaseAfterOp) +
      countPlannedAnchors(dag.releaseAfterYield);
  if (state.emittedAcquires.size() < plannedAcquires)
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: post-emission verifier: fewer acquires emitted "
        "than planned");
  if (state.emittedReleases.size() < plannedReleases)
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: post-emission verifier: fewer releases emitted "
        "than planned");

  auto funcOp = group.members.front().allocOp->getParentOfType<triton::FuncOp>();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::AcquireBeforeOp,
                                      SyncAnchorKind::AcquireBeforeYield,
                                      dag.acquireBeforeOp,
                                      dag.acquireBeforeYield,
                                      state.emittedAcquires)))
    return failure();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::ReleaseBeforeOp,
                                      SyncAnchorKind::ReleaseBeforeYield,
                                      dag.releaseBeforeOp,
                                      dag.releaseBeforeYield,
                                      state.emittedReleases)))
    return failure();
  if (failed(verifyPlannedSyncRecords(funcOp,
                                      SyncAnchorKind::ReleaseAfterOp,
                                      SyncAnchorKind::ReleaseAfterYield,
                                      dag.releaseAfterOp,
                                      dag.releaseAfterYield,
                                      state.emittedReleases)))
    return failure();

  for (const EmittedSyncRecord &record : state.emittedAcquires) {
    if (!isRecordPlanned(record, dag.acquireBeforeOp, dag.acquireBeforeYield))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "unplanned acquire was emitted");
    if (failed(verifyStageCluster(record.op, record.expectedStageCluster)))
      return failure();
    if (tokenEscapesWithoutYield(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "branch-local acquire token escaped without "
                                  "CFG threading");
  }
  auto releaseIsPlanned = [&](const EmittedSyncRecord &record) {
    switch (record.kind) {
    case SyncAnchorKind::ReleaseBeforeOp:
    case SyncAnchorKind::ReleaseBeforeYield:
      return isRecordPlanned(record, dag.releaseBeforeOp, dag.releaseBeforeYield);
    case SyncAnchorKind::ReleaseAfterOp:
    case SyncAnchorKind::ReleaseAfterYield:
      return isRecordPlanned(record, dag.releaseAfterOp, dag.releaseAfterYield);
    case SyncAnchorKind::AcquireBeforeOp:
    case SyncAnchorKind::AcquireBeforeYield:
      return false;
    }
    llvm_unreachable("unknown sync anchor kind");
  };
  for (const EmittedSyncRecord &record : state.emittedReleases) {
    if (!releaseIsPlanned(record))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "unplanned release was emitted");
    if (!state.knownCarrierTokens.contains(record.token))
      return record.op->emitError("nvws-insert-semas: post-emission verifier: "
                                  "release token does not trace to a planned "
                                  "carrier acquire");
    if (failed(verifyStageCluster(record.op, record.expectedStageCluster)))
      return failure();
  }

  for (Operation *op : dag.accessOps) {
    const AccessEvent *event = findEvent(group, op);
    if (!event) continue;
    SmallVector<const AccessTouch *, 4> touches;
    collectTouchesForResource(*event, dag.resource.second, touches);
    if (touches.empty()) continue;
    if (!state.protectedAccesses.contains(op))
      return op->emitError("nvws-insert-semas: post-emission verifier: "
                           "planned access was not rewritten through "
                           "nvws.semaphore.buffer");
    SmallVector<unsigned, 4> consumedRecords;
    for (const AccessTouch *touch : touches) {
      auto bufferIt =
          llvm::find_if(state.emittedBuffers, [&](const EmittedBufferRecord &rec) {
            if (rec.accessOp != op || rec.memberIdx != touch->memberIdx)
              return false;
            unsigned recordIdx = static_cast<unsigned>(&rec - state.emittedBuffers.data());
            return !llvm::is_contained(consumedRecords, recordIdx);
          });
      if (bufferIt == state.emittedBuffers.end())
        return op->emitError("nvws-insert-semas: post-emission verifier: "
                             "planned access has no semaphore.buffer op for a "
                             "planned member touch");
      unsigned recordIdx =
          static_cast<unsigned>(&*bufferIt - state.emittedBuffers.data());
      consumedRecords.push_back(recordIdx);
      if (bufferIt->backingIdx >= bufferIt->buffers.size())
        return bufferIt->bufferOp->emitError(
            "nvws-insert-semas: post-emission verifier: semaphore.buffer member "
            "does not match planned touch");
      if (!operationUsesValue(bufferIt->retargetOp, bufferIt->accessBuffer))
        return bufferIt->retargetOp->emitError(
            "nvws-insert-semas: post-emission verifier: planned memory access "
            "does not use the planned nvws.semaphore.buffer result");
      if (!state.knownCarrierTokens.contains(bufferIt->token))
        return bufferIt->bufferOp->emitError(
            "nvws-insert-semas: post-emission verifier: semaphore.buffer token "
            "does not trace to a planned acquire");
      if (failed(verifyStageCluster(bufferIt->bufferOp,
                                    bufferIt->expectedStageCluster)))
        return failure();
    }
  }

  for (const ThreadRecord &record : state.threadedTokens)
    if (failed(verifyThreadRecord(record, plan, sp, dag)))
      return failure();

  // M1 / M3 invariants (§2, plan step 5). These are HARD-FAIL per the §1 Hard
  // Rule — a violation means the DAG is invalid or the model is wrong; it is
  // never repaired here. M2 (first-event consistency) is audited via the
  // schedule dump (§9-V6a), not asserted here.
  SmallVector<Value, 4> resourceSemas;
  auto addSema = [&](Value v) {
    if (v && !llvm::is_contained(resourceSemas, v))
      resourceSemas.push_back(v);
  };
  for (auto &entry : state.semas.byClass)
    addSema(entry.second);
  // M1 — exactly one released seed for a resource that emits semaphores.
  unsigned releasedCount = 0;
  for (Value v : resourceSemas)
    if (auto cr = v.getDefiningOp<SemaphoreCreateOp>())
      if (cr.getIsReleased())
        ++releasedCount;
  if (!resourceSemas.empty() && releasedCount != 1)
    return group.members.front().allocOp->emitError(
               "nvws-insert-semas: post-emission verifier: M1 violation — a "
               "resource that emits semaphores must have exactly one released "
               "seed, found ")
           << releasedCount;
  // M3 — each semaphore acquired by root and at most one concrete owner.
  for (Value v : resourceSemas) {
    SetVector<int> concreteOwners;
    Operation *witness = nullptr;
    for (const EmittedSyncRecord &rec : state.emittedAcquires) {
      if (rec.semaphore != v || !rec.op)
        continue;
      if (std::optional<SetVector<int>> ids = nearestPartitionIds(rec.op))
        for (int id : *ids) {
          if (concreteOwners.insert(id) && !witness)
            witness = rec.op;
        }
    }
    if (concreteOwners.size() > 1)
      return (witness ? witness : group.members.front().allocOp)
                 ->emitError("nvws-insert-semas: post-emission verifier: M3 "
                             "violation — semaphore acquired by ")
             << concreteOwners.size() << " distinct concrete owners";
  }
  return success();
}

static LogicalResult emitResourceBlock(Block &block, const OptSyncDag &dag,
                                       const SyncPlan &sp,
                                       const ResourcePlan &plan,
                                       BufferGroup &group,
                                       const GroupBacking &backing,
                                       EmitState &state,
                                       Region *plannedRegion);

static LogicalResult emitResourceRegion(Region &region, const OptSyncDag &dag,
                                        const SyncPlan &sp,
                                        const ResourcePlan &plan,
                                        BufferGroup &group,
                                        const GroupBacking &backing,
                                        EmitState &state,
                                        Region *plannedRegion = nullptr) {
  Region *regionKey = plannedRegion ? plannedRegion : &region;
  for (Block &block : region)
    if (failed(emitResourceBlock(block, dag, sp, plan, group, backing, state,
                                 regionKey)))
      return failure();
  return success();
}

static LogicalResult emitResourceBlock(Block &block, const OptSyncDag &dag,
                                       const SyncPlan &sp,
                                       const ResourcePlan &plan,
                                       BufferGroup &group,
                                       const GroupBacking &backing,
                                       EmitState &state,
                                       Region *plannedRegion) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<scf::YieldOp>(op)) {
      SmallVector<AcquireRecord, 2> yieldAcquires;
      if (failed(emitBeforeYieldSync(&op, plannedRegion, dag, sp, group,
                                     state, yieldAcquires)))
        return failure();
      continue;
    }

    AccessEvent *event = nullptr;
    SmallVector<const AccessTouch *, 4> touches;
    if (dag.accessOps.contains(&op)) {
      for (AccessEvent &candidate : group.events)
        if (candidate.op == &op) {
          event = &candidate;
          collectTouchesForResource(candidate, dag.resource.second, touches);
          break;
        }
      if (event && event->owner)
        state.stageCache[*event->owner] = getStageCluster(&op);
    }

    SmallVector<AcquireRecord, 2> acquires;
    if (failed(emitBeforeOpSync(&op, dag, sp, group, state, acquires)))
      return failure();

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Operation *oldForOp = forOp.getOperation();
      Region *plannedForRegion = &forOp.getRegion();
      scf::ForOp activeForOp = forOp;
      EmitState bodyState = state;
      bool threaded = shouldThreadForRegion(forOp, dag);
      SmallVector<const AccessTouch *, 4> prebufferTouches;
      if (threaded && canPrebufferLocalRegionEntry(
                          forOp.getOperation(), forOp.getRegion(), acquires,
                          dag, group, prebufferTouches)) {
        OpBuilder prebufferBuilder(forOp);
        prebufferLocalRegionEntry(prebufferBuilder, forOp.getOperation(),
                                  prebufferTouches, acquires.front(), group,
                                  backing, state);
        bodyState = state;
        threaded = false;
      }
      OpBuilder loopBuilder(forOp);
      if (threaded) {
        std::optional<PartitionId> recordOwner = state.currentOwner;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end())
          recordOwner = regionOwnerIt->second.entry;
        FailureOr<scf::ForOp> threadedFor =
            threadCarrierThroughFor(loopBuilder, forOp, bodyState,
                                    plannedForRegion, recordOwner, group,
                                    dag.memberIndices, dag.resource.second);
        if (failed(threadedFor))
          return failure();
        activeForOp = *threadedFor;
      }
      if (failed(emitResourceRegion(activeForOp.getRegion(), dag, sp, plan,
                                    group, backing, bodyState,
                                    plannedForRegion)))
        return failure();
      if (threaded) {
        std::optional<PartitionId> ownerAtYield = bodyState.currentOwner;
        bool overrideOwnerAtYield = false;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end() &&
            (!linearChainEntersFor(oldForOp, dag, sp) ||
             dag.acquireBeforeYield.contains(plannedForRegion))) {
          ownerAtYield = regionOwnerIt->second.exit;
          overrideOwnerAtYield = true;
        }
        closeCarrierForLoop(activeForOp, bodyState, state, ownerAtYield,
                            overrideOwnerAtYield, plannedForRegion);
        if (failed(emitTmemLinearLoopExitDrain(activeForOp, plannedForRegion,
                                               dag, sp, group, state)))
          return failure();
      } else {
        closeExistingCarrierForLoop(activeForOp, bodyState, state);
      }
      if (failed(emitAfterOpSync(oldForOp, activeForOp.getOperation(), dag, sp,
                                 group, state)))
        return failure();
      if (failed(emitDeferredNestedAfterOpSync(activeForOp.getOperation(), dag,
                                               sp, group, state)))
        return failure();
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      Operation *oldIfOp = ifOp.getOperation();
      Region *plannedThenRegion = &ifOp.getThenRegion();
      Region *plannedElseRegion = &ifOp.getElseRegion();
      SmallVector<const AccessTouch *, 4> prebufferTouches;
      if (canPrebufferLocalIfEntry(ifOp, acquires, dag, group,
                                   prebufferTouches)) {
        OpBuilder prebufferBuilder(ifOp);
        prebufferLocalRegionEntry(prebufferBuilder, ifOp.getOperation(),
                                  prebufferTouches, acquires.front(), group,
                                  backing, state);
      }
      bool threaded = shouldThreadIfRegion(ifOp, dag);
      scf::IfOp activeIfOp = ifOp;
      unsigned oldNumResults = ifOp.getNumResults();
      auto oldPartitionIds =
          hasPartition(ifOp) ? getPartitionIds(ifOp) : SetVector<int>();
      auto oldPartitionOutputs =
          hasPartition(ifOp) ? getPartitionOutputs(ifOp)
                             : SmallVector<SetVector<int>, 4>();
      SmallVector<unsigned, 4> reusableIfTokenResults;
      if (threaded)
        for (auto [idx, result] : llvm::enumerate(ifOp.getResults()))
          if (isa<AsyncTokenType>(result.getType()))
            reusableIfTokenResults.push_back(static_cast<unsigned>(idx));
      std::optional<unsigned> reusableIfTokenResult;
      if (!reusableIfTokenResults.empty())
        reusableIfTokenResult = reusableIfTokenResults.front();
      bool reuseExistingTokenResult = reusableIfTokenResult.has_value();
      if (threaded) {
        if (ifOp.getElseRegion().empty())
          return ifOp.emitError(
              "nvws-insert-semas: planned scf.if carrier threading requires "
              "an else path producer");
        if (!reuseExistingTokenResult) {
          OpBuilder ifBuilder(ifOp);
          activeIfOp = threadCarrierThroughIf(ifBuilder, ifOp);
        }
      }
      Value incomingToken = state.currentToken;
      Value incomingSemaphore = state.currentSemaphore;
      EmitState thenState = state;
      if (failed(emitResourceRegion(activeIfOp.getThenRegion(), dag, sp, plan,
                                    group, backing, thenState,
                                    plannedThenRegion)))
        return failure();
      EmitState elseState = state;
      if (!activeIfOp.getElseRegion().empty() &&
          failed(emitResourceRegion(activeIfOp.getElseRegion(), dag, sp, plan,
                                    group, backing, elseState,
                                    plannedElseRegion)))
        return failure();
      if (threaded) {
        Value thenToken =
            thenState.currentToken ? thenState.currentToken : incomingToken;
        Value elseToken =
            elseState.currentToken ? elseState.currentToken : incomingToken;
        if (!thenToken || !elseToken)
          return activeIfOp.emitError(
              "nvws-insert-semas: planned scf.if carrier threading has no "
              "token producer on every path");
        if (reuseExistingTokenResult) {
          unsigned tokenResultIdx = *reusableIfTokenResult;
          Value poison;
          if (reusableIfTokenResults.size() > 1) {
            Operation *poisonAnchor =
                getDominatingPoisonAnchor(activeIfOp.getOperation());
            OpBuilder poisonBuilder(poisonAnchor);
            poisonBuilder.setInsertionPoint(poisonAnchor);
            poison = ub::PoisonOp::create(poisonBuilder, activeIfOp.getLoc(),
                                          poisonBuilder.getType<AsyncTokenType>());
          }
          for (unsigned resultIdx : reusableIfTokenResults) {
            activeIfOp.thenYield().setOperand(
                resultIdx, resultIdx == tokenResultIdx ? thenToken : poison);
            activeIfOp.elseYield().setOperand(
                resultIdx, resultIdx == tokenResultIdx ? elseToken : poison);
          }
          stampTokenYieldPartition(activeIfOp.thenYield(), thenToken,
                                   thenState.currentOwner);
          stampTokenYieldPartition(activeIfOp.elseYield(), elseToken,
                                   elseState.currentOwner);
        } else {
          appendTokenToYield(activeIfOp.thenYield(), thenToken,
                             thenState.currentOwner);
          appendTokenToYield(activeIfOp.elseYield(), elseToken,
                             elseState.currentOwner);
        }
        thenState.knownCarrierTokens.insert(thenToken);
        elseState.knownCarrierTokens.insert(elseToken);
        std::optional<PartitionId> outOwner =
            thenState.currentOwner == elseState.currentOwner
                ? thenState.currentOwner
                : std::nullopt;
        SetVector<int> outPartition = partitionSetForOwner(outOwner);
        if (outPartition.empty()) {
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            thenToken, thenState.currentOwner,
                                            activeIfOp.getOperation()));
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            elseToken, elseState.currentOwner,
                                            activeIfOp.getOperation()));
        }
        if (hasPartition(activeIfOp)) {
          addPartitionIds(oldPartitionIds, outPartition);
          if (!reuseExistingTokenResult)
            oldPartitionOutputs.push_back(outPartition);
          setPartition(activeIfOp, oldPartitionIds);
          setPartitionOutputs(activeIfOp, oldPartitionOutputs);
        }
        Value thenSemaphore =
            thenState.currentSemaphore ? thenState.currentSemaphore
                                       : incomingSemaphore;
        Value elseSemaphore =
            elseState.currentSemaphore ? elseState.currentSemaphore
                                       : incomingSemaphore;
        Value joinedSemaphore =
            thenSemaphore == elseSemaphore
                ? thenSemaphore
                : (outOwner ? (thenSemaphore ? thenSemaphore : elseSemaphore)
                            : Value());
        EmitState joinedState = state;
        mergeProtectedAccesses(joinedState, thenState);
        mergeProtectedAccesses(joinedState, elseState);
        joinedState.currentToken =
            activeIfOp.getResult(reuseExistingTokenResult
                                     ? *reusableIfTokenResult
                                     : oldNumResults);
        joinedState.currentSemaphore = joinedSemaphore;
        joinedState.currentOwner = outOwner;
        joinedState.currentBuffers.clear();
        joinedState.knownCarrierTokens.insert(joinedState.currentToken);
        joinedState.threadedTokens.push_back(
            {activeIfOp.getOperation(), joinedState.currentToken, outOwner,
             ThreadRecordKind::IfResult, plannedThenRegion, plannedElseRegion});
        state = joinedState;
      } else {
        mergeProtectedAccesses(state, thenState);
        mergeProtectedAccesses(state, elseState);
      }
      if (failed(emitAfterOpSync(oldIfOp, activeIfOp.getOperation(), dag, sp,
                                 group, state)))
        return failure();
      if (failed(emitDeferredNestedAfterOpSync(activeIfOp.getOperation(), dag,
                                               sp, group, state)))
        return failure();
      if (threaded && !reuseExistingTokenResult)
        state.eraseAfterEmission.insert(oldIfOp);
      continue;
    }

    if (event && !touches.empty()) {
      OpBuilder b(&op);
      b.setInsertionPoint(&op);
      if (failed(emitAccessEvent(b, *event, touches, acquires, group, dag,
                                 backing, state)))
        return failure();
    }
    if (failed(emitAfterOpSync(&op, dag, sp, group, state)))
      return failure();
  }
  return success();
}

static LogicalResult emitResource(triton::FuncOp funcOp, BufferGroup &group,
                                  const ResourcePlan &plan, const SyncPlan &sp,
                                  const OptSyncDag &dag,
                                  DenseMap<BackingKey, GroupBacking> &backings,
                                  const DenseMap<unsigned, int> &numStagesByGroup,
                                  SetVector<Operation *> &eraseAfterEmission) {
  if (dag.groups.empty()) return success();
  if (failed(verifyPlanBeforeEmission(funcOp, group, plan, sp, dag)))
    return failure();
  GroupBacking &backing =
      ensureGroupBacking(group, dag.groupIdx, dag.resource.second,
                         plan.memberIndices, backings, numStagesByGroup);
  EmitState state;
  state.semas = createResourceSemaphores(dag, sp, group, backing);
  if (failed(emitResourceRegion(funcOp.getBody(), dag, sp, plan, group, backing,
                                state)))
    return failure();
  DenseSet<Operation *> poisonedTokenOps;
  for (const PoisonTokenRecord &record :
       state.poisonTokenResultsAfterEmission) {
    if (!record.op || !poisonedTokenOps.insert(record.op).second)
      continue;
    OpBuilder poisonBuilder(record.insertBefore ? record.insertBefore
                                                : record.op);
    poisonTokenResults(poisonBuilder, record.op, record.insertBefore);
  }
  if (failed(verifyPostEmission(dag, group, plan, sp, state)))
    return failure();
  for (Operation *op : llvm::reverse(state.eraseAfterEmission))
    eraseAfterEmission.insert(op);
  return success();
}
