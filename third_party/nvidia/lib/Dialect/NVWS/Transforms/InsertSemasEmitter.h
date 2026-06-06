#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMITTER_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMITTER_H_

// ---------------------------------------------------------------------------
// v4 commit 5 — semaphore IR emission.
// ---------------------------------------------------------------------------

static bool shouldDumpDag() {
  const char *value = std::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  if (!value) return false;
  std::string s(value);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s == "1" || s == "true" || s == "on";
}

template <typename OpT, typename... Args>
static OpT createIntoPartition(
    OpBuilder &b, Location loc,
    std::pair<std::optional<PartitionId>, StageCluster> partitionIdStageCluster,
    Args &&...args) {
  std::optional<SetVector<int>> partitionIds = SetVector<int>();
  std::optional<int> wsTag;
  if (partitionIdStageCluster.first) {
    auto [id, tag] = *partitionIdStageCluster.first;
    wsTag = tag;
    partitionIds->insert(id);
  } else {
    partitionIds = std::nullopt;
  }
  auto op = triton::gpu::createInto<OpT>(b, loc, partitionIds,
                                         partitionIdStageCluster.second,
                                         std::forward<Args>(args)...);
  if (wsTag) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !hasWarpSpecializeTag(forOp))
      forOp = forOp->template getParentOfType<scf::ForOp>();
    if (!forOp)
      setWarpSpecializeTag(op, *wsTag);
  }
  return op;
}

static void copyBufferAttrs(Operation *src, Operation *dst) {
  for (StringRef attrName :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName}) {
    if (Attribute attr = src->getAttr(attrName))
      dst->setAttr(attrName, attr);
  }
}

static Operation *getSemaphoreInsertionAnchor(BufferGroup &group) {
  Operation *anchor = group.members.front().allocOp;
  auto outerWsLoop = anchor->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  return outerWsLoop ? outerWsLoop.getOperation() : anchor;
}

static Operation *getLocalSemaphoreCreateAnchor(BufferGroup &group) {
  Operation *anchor = group.members.front().allocOp;
  Block *block = anchor->getBlock();
  for (BufferMember &member : group.members) {
    if (member.allocOp->getBlock() != block)
      continue;
    if (anchor->isBeforeInBlock(member.allocOp))
      anchor = member.allocOp;
  }
  return anchor;
}

static bool canDoubleBufferAcc(MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  auto blockM = tmemDesc.getShape()[0];
  auto blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + (blockM * blockN * 2) > numTMEMRows * numTMEMColumns)
    return false;
  if (isa<TCGen5MMAScaledOp>(mmaOp) && blockN == 256)
    return false;
  return true;
}

static int computeTmemSemaphoreNumStages(BufferGroup &group, int numTmemBlocks,
                                         bool useMetaPartitioner) {
  bool isMultiStaged = true;
  for (BufferMember &member : group.members) {
    auto allocOp = cast<TMEMAllocOp>(member.allocOp);
    for (auto user : allocOp.getResult().getUsers()) {
      if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          auto wsLoop = getOuterWSLoop(loop);
          // Determine if the MMA accumulator can be multibuffered.
          bool accIsMultiBuffered =
              // MMAs in subsequent iterations can be overlapped.
              !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              // The accumulator is reset at some point, thus allowing
              // multibuffering.
              isAccMultibufferingPossible(mmaOp, loop) &&
              // The user didn't disable it with a flag.
              !getDisallowAccMultiBuffer(wsLoop) &&
              canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiStaged = isMultiStaged && accIsMultiBuffered;
        }
      }
    }
  }
  auto numStages =
      useMetaPartitioner ? 1 + 0 * isMultiStaged : 1 + 1 * isMultiStaged;
  return numStages;
}

static void updateNumTmemBlocks(BufferGroup &group, int numStages,
                                int &numTmemBlocks) {
  for (BufferMember &member : group.members) {
    auto shape = member.type.getShape();
    if (shape.size() >= 2)
      numTmemBlocks += shape[0] * shape[1] * numStages;
  }
}

static GroupBacking &
ensureGroupBacking(BufferGroup &group, unsigned groupIdx,
                   int64_t resourceKey, ArrayRef<unsigned> memberIndices,
                   DenseMap<BackingKey, GroupBacking> &backings,
                   const DenseMap<unsigned, int> &numStagesByGroup) {
  BackingKey key{groupIdx, resourceKey};
  auto it = backings.find(key);
  if (it != backings.end())
    return it->second;

  GroupBacking backing;
  backing.memberIndices.append(memberIndices.begin(), memberIndices.end());
  OpBuilder b(getSemaphoreInsertionAnchor(group));
  b.setInsertionPoint(getSemaphoreInsertionAnchor(group));
  int depth = 1;
  if (group.isTmem()) {
    auto it = numStagesByGroup.find(groupIdx);
    assert(it != numStagesByGroup.end() &&
           "TMEM semaphore numStages must be precomputed before emission");
    depth = it->second;
  }
  for (auto [backingIdx, memberIdx] : llvm::enumerate(memberIndices)) {
    BufferMember &member = group.members[memberIdx];
    backing.memberToBackingIndex[memberIdx] = static_cast<unsigned>(backingIdx);
    MemDescType semBufType = member.type;
    if (group.isTmem()) {
      semBufType = getSemaphoreMultiBufferedType(member.type, depth);
      Operation *semAlloc = createAlloc(b, member.allocOp->getLoc(), semBufType,
                                        Value());
      semAlloc->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      copyBufferAttrs(member.allocOp, semAlloc);
      backing.buffers.append(semAlloc->getResults().begin(),
                             semAlloc->getResults().end());
    } else if (auto localAlloc = dyn_cast<LocalAllocOp>(member.allocOp);
               localAlloc && !localAlloc.getSrc()) {
      member.allocOp->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      backing.buffers.push_back(member.value);
    } else {
      semBufType = getMultiBufferedType(member.type, depth);
      Operation *semAlloc = createAlloc(b, member.allocOp->getLoc(), semBufType,
                                        Value());
      semAlloc->setAttr("nvws.semaphore.backing", b.getUnitAttr());
      copyBufferAttrs(member.allocOp, semAlloc);
      backing.buffers.append(semAlloc->getResults().begin(),
                             semAlloc->getResults().end());
    }
    backing.bufferTypes.push_back(semBufType);
  }
  auto inserted = backings.insert({key, std::move(backing)});
  return inserted.first->second;
}

static Value createSemaphore(OpBuilder &b, Location loc,
                             const GroupBacking &backing, bool released) {
  auto baseTypes = TypeArrayAttr::get(b.getContext(), backing.bufferTypes);
  auto semaTy = SemaphoreType::get(b.getContext(), baseTypes);
  auto op = SemaphoreCreateOp::create(b, loc, semaTy, backing.buffers, released);
  return op.getResult();
}

static void setPartitionFromAnchor(Operation *op, Operation *anchor);

static std::optional<unsigned> findEdgeIndex(const SyncPlan &sp,
                                             const SyncEdge *edge) {
  if (!edge)
    return std::nullopt;
  for (auto [idx, candidate] : llvm::enumerate(sp.edges))
    if (&candidate == edge)
      return static_cast<unsigned>(idx);
  return std::nullopt;
}

static const AccessEvent *findEvent(BufferGroup &group, Operation *op) {
  for (AccessEvent &event : group.events)
    if (event.op == op)
      return &event;
  return nullptr;
}

static bool edgeDstWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (!edge.dstOp) return false;
  const AccessEvent *event = findEvent(group, edge.dstOp);
  return event && eventProduces(*event, resourceKey);
}

static bool edgeDstReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey) {
  if (!edge.dstOp) return false;
  const AccessEvent *event = findEvent(group, edge.dstOp);
  return event && eventConsumes(*event, resourceKey);
}

static bool edgeSrcReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey) {
  if (!edge.srcOp) return false;
  const AccessEvent *event = findEvent(group, edge.srcOp);
  return event && eventConsumes(*event, resourceKey);
}

static bool edgeSrcWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (!edge.srcOp) return false;
  const AccessEvent *event = findEvent(group, edge.srcOp);
  return event && eventProduces(*event, resourceKey);
}

static bool isRootToWsLoopEntryEdge(const SyncEdge &edge, BufferGroup &group) {
  if (!group.isTmem() || edge.srcOwner || !edge.dstOwner)
    return false;
  auto forOp = dyn_cast_or_null<scf::ForOp>(edge.dstOp);
  return forOp && hasWarpSpecializeTag(forOp.getOperation());
}

static bool edgeStartsAtRootTmemStoreInitializer(const SyncEdge &edge,
                                                 BufferGroup &group,
                                                 int64_t resourceKey) {
  if (!group.isTmem() || edge.srcOwner || !isa_and_nonnull<TMEMStoreOp>(edge.srcOp))
    return false;
  for (const AccessEvent &event : group.events) {
    if (!eventProduces(event, resourceKey))
      continue;
    return event.op == edge.srcOp && !event.owner;
  }
  return false;
}

static bool edgeUsesEmpty(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey) {
  if (edge.forceFullSemaphore)
    return false;
  if (edgeStartsAtRootTmemStoreInitializer(edge, group, resourceKey))
    return true;
  if (edge.kind == SyncEdgeKind::Done)
    return true;
  if (edge.kind == SyncEdgeKind::Ready)
    return false;
  if (group.isTmem() && isa_and_nonnull<MMAv5OpInterface>(edge.srcOp) &&
      !edge.dstOp && edge.dstYieldRegion && edge.dstOwner)
    return true;
  if (edgeSrcWrites(edge, group, resourceKey) &&
      edgeDstWrites(edge, group, resourceKey))
    return false;
  if (edgeDstWrites(edge, group, resourceKey))
    return true;
  if (edgeDstReads(edge, group, resourceKey))
    return false;
  if (edgeSrcReads(edge, group, resourceKey) &&
      !edgeSrcWrites(edge, group, resourceKey))
    return true;
  return false;
}

static bool edgeNeedsTerminalReadRelease(const SyncEdge &edge,
                                         BufferGroup &group,
                                         int64_t resourceKey) {
  return edge.dstOp && edgeDstReads(edge, group, resourceKey) &&
         !edgeDstWrites(edge, group, resourceKey) &&
         isa<TMEMLoadOp, LocalLoadOp>(edge.dstOp);
}

static bool opReadsOnlyResource(Operation *op, BufferGroup &group,
                                int64_t resourceKey) {
  const AccessEvent *event = findEvent(group, op);
  return event && eventConsumes(*event, resourceKey) &&
         !eventProduces(*event, resourceKey);
}

static bool operationOccursAtOrAfterInSequentialScope(Operation *anchor,
                                                      Operation *candidate);

static bool hasLaterSameOwnerResourceAccessBeforeOwnerChange(
    Operation *op, BufferGroup &group, int64_t resourceKey) {
  const AccessEvent *anchor = findEvent(group, op);
  if (!anchor || !eventTouchesResource(*anchor, resourceKey))
    return false;

  for (const AccessEvent &event : group.events) {
    if (!eventTouchesResource(event, resourceKey))
      continue;
    if (event.op == op) {
      continue;
    }
    if (!operationOccursAtOrAfterInSequentialScope(op, event.op))
      continue;
    if (!sameOwner(event.owner, anchor->owner))
      return false;
    return true;
  }
  return false;
}

static bool operationOccursAtOrAfterInSequentialScope(Operation *anchor,
                                                      Operation *candidate) {
  if (!anchor || !candidate)
    return false;
  if (anchor == candidate)
    return true;
  for (Operation *anchorScope = anchor; anchorScope;
       anchorScope = anchorScope->getParentOp()) {
    Block *block = anchorScope->getBlock();
    if (!block)
      continue;
    for (Operation *candidateScope = candidate; candidateScope;
         candidateScope = candidateScope->getParentOp()) {
      if (candidateScope->getBlock() != block)
        continue;
      if (anchorScope == candidateScope)
        return false;
      return anchorScope->isBeforeInBlock(candidateScope);
    }
  }
  return false;
}

static bool regionContainsOp(Region *region, Operation *op) {
  if (!region || !op)
    return false;
  for (Region *parentRegion = op->getParentRegion(); parentRegion;) {
    if (parentRegion == region)
      return true;
    Operation *parentOp = parentRegion->getParentOp();
    parentRegion = parentOp ? parentOp->getParentRegion() : nullptr;
  }
  return false;
}

static bool yieldOccursAtOrAfterInSequentialScope(Region *yieldRegion,
                                                  Operation *anchor) {
  if (!yieldRegion || !anchor)
    return false;
  Operation *parent = yieldRegion->getParentOp();
  if (!parent)
    return false;
  if (regionContainsOp(yieldRegion, anchor))
    return true;
  return operationOccursAtOrAfterInSequentialScope(anchor, parent);
}

static bool ownerHasPlannedOutgoingEdgeAtOrAfterRead(
    Operation *op, const SyncPlan &sp, BufferGroup &group,
    int64_t resourceKey) {
  const AccessEvent *anchor = findEvent(group, op);
  if (!anchor || !eventTouchesResource(*anchor, resourceKey))
    return false;
  for (const SyncEdge &edge : sp.edges) {
    if (!sameOwner(edge.srcOwner, anchor->owner))
      continue;
    if (edge.srcOp &&
        operationOccursAtOrAfterInSequentialScope(op, edge.srcOp))
      return true;
    if (edge.srcYieldRegion &&
        yieldOccursAtOrAfterInSequentialScope(edge.srcYieldRegion, op))
      return true;
  }
  return false;
}

static Operation *findTerminalReadReleaseAnchor(Operation *op,
                                                const SyncPlan &sp,
                                                BufferGroup &group,
                                                int64_t resourceKey) {
  if (!opReadsOnlyResource(op, group, resourceKey))
    return nullptr;
  if (hasLaterSameOwnerResourceAccessBeforeOwnerChange(op, group, resourceKey))
    return nullptr;
  if (ownerHasPlannedOutgoingEdgeAtOrAfterRead(op, sp, group, resourceKey))
    return nullptr;
  return op;
}

static bool linearChainNeedsPerEdgeFulls(const SyncGroup &syncGroup,
                                         const SyncPlan &sp,
                                         BufferGroup &group,
                                         int64_t resourceKey) {
  if (!group.isTmem())
    return false;
  std::optional<int64_t> firstOffset;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    for (const AccessTouch &touch : sp.edges[edgeIdx].touches) {
      if (touch.resourceKey != resourceKey ||
          touch.memberIdx >= group.members.size())
        continue;
      int64_t offset = group.members[touch.memberIdx].offset;
      if (!firstOffset) {
        firstOffset = offset;
        continue;
      }
      if (*firstOffset != offset)
        return true;
    }
  }
  return false;
}

static ResourceSemaphores createResourceSemaphores(const OptSyncDag &dag,
                                                   const SyncPlan &sp,
                                                   BufferGroup &group,
                                                   GroupBacking &backing) {
  Operation *anchor = group.isTmem() ? getSemaphoreInsertionAnchor(group)
                                     : getLocalSemaphoreCreateAnchor(group);
  if (!group.isTmem()) {
    for (Value buffer : backing.buffers) {
      Operation *def = buffer.getDefiningOp();
      if (!def)
        continue;
      if (!anchor || anchor->getBlock() != def->getBlock()) {
        anchor = def;
        continue;
      }
      if (anchor->isBeforeInBlock(def))
        anchor = def;
    }
  }
  OpBuilder b(anchor);
  if (group.isTmem())
    b.setInsertionPoint(anchor);
  else
    b.setInsertionPointAfter(anchor);
  ResourceSemaphores semas;
  Location loc = group.members.front().allocOp->getLoc();
  semas.seedId = static_cast<unsigned>(sp.edges.size());

  // Mechanical identity: exactly one semaphore per canonical `semaRep` class.
  // A class is created released=true iff it is the seed's class (the single
  // initial permit — M1); every other class is released=false (M2). Creation
  // order is the deterministic §6.5 order: walk dag.groups in order (the
  // InitialEmpty seed group is group 0, so the released seed is created first),
  // then each group's edgeIdxs in order. No empty/full, no acquirer keying, no
  // op-kind.
  unsigned seedRep = sp.semaFind(semas.seedId);
  auto ensureClass = [&](unsigned id) {
    unsigned rep = sp.semaFind(id);
    if (semas.byClass.count(rep))
      return;
    Value sem = createSemaphore(b, loc, backing, /*released=*/rep == seedRep);
    if (!group.isTmem())
      setPartitionFromAnchor(sem.getDefiningOp(), anchor);
    semas.byClass[rep] = sem;
  };
  for (const SyncGroup &syncGroup : dag.groups) {
    if (syncGroup.kind == SyncGroupKind::InitialEmpty) {
      ensureClass(semas.seedId);
      continue;
    }
    for (unsigned edgeIdx : syncGroup.edgeIdxs)
      ensureClass(edgeIdx);
  }
  // Safety: ensure the seed class exists even if no InitialEmpty group is present
  // (an edge-bearing resource with no separate seed marker still needs its seed
  // class materialized for forClass()).
  ensureClass(semas.seedId);
  return semas;
}

static const SyncEdge *findEdgeForAnchor(const SyncGroup &group,
                                         const SyncPlan &sp,
                                         const OptSyncDag &dag,
                                         SyncAnchorKind kind,
                                         Operation *anchor,
                                         Region *yieldRegion,
                                         Operation *liveAnchor = nullptr) {
  if (group.kind == SyncGroupKind::InitialEmpty)
    return nullptr;
  if (group.kind == SyncGroupKind::ReadyFanout &&
      (kind == SyncAnchorKind::ReleaseAfterOp ||
       kind == SyncAnchorKind::ReleaseAfterYield))
    return &sp.edges[group.edgeIdxs.front()];
  if (group.kind == SyncGroupKind::DoneFanin &&
      (kind == SyncAnchorKind::AcquireBeforeOp ||
       kind == SyncAnchorKind::AcquireBeforeYield))
    return &sp.edges[group.edgeIdxs.front()];

  for (unsigned ei : group.edgeIdxs) {
    const SyncEdge &edge = sp.edges[ei];
    auto dstBranchIt = dag.dstBranchEntryAnchor.find(ei);
    Operation *dstBranchEntry =
        dstBranchIt == dag.dstBranchEntryAnchor.end() ? nullptr
                                                      : dstBranchIt->second;
    switch (kind) {
    case SyncAnchorKind::AcquireBeforeOp:
      if (dag.loopEntryHandoffAccess.lookup(ei) == anchor)
        return &edge;
      if (edge.dstOp == anchor || dstBranchEntry == anchor)
        return &edge;
      break;
    case SyncAnchorKind::AcquireBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeOp:
      if (dag.loopEntryHandoffAccess.lookup(ei) == anchor)
        return &edge;
      if (edge.dstOp == anchor || dstBranchEntry == anchor)
        return &edge;
      break;
    case SyncAnchorKind::ReleaseBeforeYield:
      if (edge.dstYieldRegion == yieldRegion) return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterOp:
      if (edge.srcOp == anchor)
        return &edge;
      if (Operation *ancestorAnchor = liveAnchor ? liveAnchor : anchor)
        if (isa<scf::ForOp>(ancestorAnchor) && edge.srcOp &&
            ancestorAnchor->isProperAncestor(edge.srcOp) &&
            (!edge.dstOp || !ancestorAnchor->isProperAncestor(edge.dstOp)))
          return &edge;
      break;
    case SyncAnchorKind::ReleaseAfterYield:
      if (edge.srcYieldRegion == yieldRegion) return &edge;
      break;
    }
  }
  if (group.kind == SyncGroupKind::LinearChain &&
      (kind == SyncAnchorKind::ReleaseBeforeOp ||
       kind == SyncAnchorKind::AcquireBeforeOp) &&
      anchor) {
    for (unsigned ei : group.edgeIdxs) {
      const SyncEdge &edge = sp.edges[ei];
      auto joinIt = dag.ifYieldJoinAccess.find(ei);
      if (joinIt != dag.ifYieldJoinAccess.end() && joinIt->second == anchor)
        return &edge;
    }
  }
  if (group.kind == SyncGroupKind::LinearChain &&
      kind == SyncAnchorKind::ReleaseAfterOp && anchor) {
    for (unsigned ei : group.edgeIdxs) {
      const SyncEdge &edge = sp.edges[ei];
      if (edge.dstOp == anchor)
        return &edge;
    }
  }
  return group.edgeIdxs.empty() ? nullptr : &sp.edges[group.edgeIdxs.front()];
}

static Value getSemaphoreForGroup(unsigned groupIdx, const SyncEdge *edge,
                                  const OptSyncDag &dag, const SyncPlan &sp,
                                  BufferGroup &group,
                                  ResourceSemaphores &semas) {
  // Mechanical: the semaphore is the one created for this edge's canonical
  // `semaRep` class; InitialEmpty / a null edge resolve to the seed class.
  // No empty/full split and no op-kind heuristic: a single fact lookup.
  if (dag.groups[groupIdx].kind == SyncGroupKind::InitialEmpty || !edge)
    return semas.forClass(sp, std::nullopt);
  return semas.forClass(sp, findEdgeIndex(sp, edge));
}

static const SyncEdge *getRepresentativeReleaseEdge(const PlannedRelease &action,
                                                    const SyncPlan &sp) {
  if (action.edgeIdxs.empty())
    return nullptr;
  unsigned edgeIdx = action.edgeIdxs.front();
  if (edgeIdx >= sp.edges.size())
    return nullptr;
  return &sp.edges[edgeIdx];
}

static StageCluster stageForYieldOwner(std::optional<PartitionId> owner,
                                       EmitState &state) {
  if (!owner) return std::nullopt;
  auto it = state.stageCache.find(*owner);
  return it == state.stageCache.end() ? StageCluster{} : it->second;
}

static SetVector<int> partitionSetForOwner(std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (owner)
    ids.insert(owner->first);
  return ids;
}

static std::optional<SetVector<int>> nearestPartitionIds(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp())
    if (hasPartition(parent))
      return getPartitionIds(parent);
  return std::nullopt;
}

static void addPartitionIds(SetVector<int> &dst, const SetVector<int> &src) {
  dst.insert(src.begin(), src.end());
}

static SetVector<int> partitionSetForValue(Value value) {
  SetVector<int> ids;
  if (!value)
    return ids;
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *def = result.getOwner();
    if (!hasPartition(def))
      return ids;
    if (def->getNumRegions() > 0) {
      auto outputs = getPartitionOutputs(def);
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber < outputs.size())
        addPartitionIds(ids, outputs[resultNumber]);
    }
    if (ids.empty())
      addPartitionIds(ids, getPartitionIds(def));
    return ids;
  }
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return ids;
  Operation *parent = arg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
    if (arg.getOwner() == forOp.getBody() && arg != forOp.getInductionVar()) {
      unsigned iterArgIdx = arg.getArgNumber() - 1;
      if (hasPartition(forOp)) {
        auto outputs = getPartitionOutputs(forOp);
        if (iterArgIdx < outputs.size())
          addPartitionIds(ids, outputs[iterArgIdx]);
        if (ids.empty())
          addPartitionIds(ids, getPartitionIds(forOp));
      }
    }
  } else if (parent && hasPartition(parent)) {
    addPartitionIds(ids, getPartitionIds(parent));
  }
  return ids;
}

static std::optional<int> wsTagForValue(Value value) {
  if (!value)
    return std::nullopt;
  if (auto result = dyn_cast<OpResult>(value))
    return tryGetWsTag(result.getOwner());
  if (auto arg = dyn_cast<BlockArgument>(value))
    return tryGetWsTag(arg.getOwner()->getParentOp());
  return std::nullopt;
}

static SetVector<int>
partitionSetForTokenOrOwner(Value token, std::optional<PartitionId> owner,
                            Operation *fallbackAnchor = nullptr) {
  SetVector<int> ids = partitionSetForOwner(owner);
  if (ids.empty())
    addPartitionIds(ids, partitionSetForValue(token));
  if (ids.empty())
    if (auto parentIds = nearestPartitionIds(fallbackAnchor))
      addPartitionIds(ids, *parentIds);
  return ids;
}

static void setWarpTagOutsideWsLoop(Operation *op, int tag) {
  auto forOp = op->getParentOfType<scf::ForOp>();
  while (forOp && !hasWarpSpecializeTag(forOp))
    forOp = forOp->getParentOfType<scf::ForOp>();
  if (!forOp && !hasWarpSpecializeTag(op))
    setWarpSpecializeTag(op, tag);
}

static void addOwnerPartition(Operation *op, std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (hasPartition(op))
    ids = getPartitionIds(op);
  else if (auto parentIds = nearestPartitionIds(op->getParentOp()))
    ids = *parentIds;
  if (owner) {
    ids.insert(owner->first);
    setWarpTagOutsideWsLoop(op, owner->second);
  }
  if (!ids.empty())
    setPartition(op, ids);
}

static void setSingleOwnerPartition(Operation *op,
                                    std::optional<PartitionId> owner) {
  if (!op || !owner)
    return;
  SetVector<int> ids;
  ids.insert(owner->first);
  setPartition(op, ids);
  setWarpTagOutsideWsLoop(op, owner->second);
}

static void setPartitionFromAnchor(Operation *op, Operation *anchor) {
  if (!op || !anchor || hasPartition(op) || !hasPartition(anchor))
    return;
  auto ids = getPartitionIds(anchor);
  if (ids.size() == 1) {
    setPartition(op, ids);
    if (auto tag = tryGetWsTag(anchor))
      setWarpTagOutsideWsLoop(op, *tag);
  }
}

static bool parentRequiresPartition(Operation *op) {
  return op && nearestPartitionIds(op->getParentOp()).has_value();
}

static void setPartitionFromTokenIfParentPartitioned(Operation *op,
                                                     Value token) {
  if (!op || hasPartition(op) || !parentRequiresPartition(op))
    return;
  auto ids = partitionSetForValue(token);
  if (ids.size() != 1)
    return;
  setPartition(op, ids);
  if (auto tag = wsTagForValue(token))
    setWarpTagOutsideWsLoop(op, *tag);
}

static bool regionContainsAccess(Region &region, const OptSyncDag &dag) {
  bool found = false;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (dag.accessOps.contains(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static ArrayAttr asyncPayloadArray(OpBuilder &b, AsyncOp payload) {
  return b.getArrayAttr(
      SmallVector<Attribute>{AsyncOpAttr::get(b.getContext(), payload)});
}

static MemDescType withMutableMemory(MemDescType type, bool mutableMemory) {
  if (type.getMutableMemory() == mutableMemory)
    return type;
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          mutableMemory, type.getAllocShape());
}

static MemDescType getLocalSemaphoreBufferType(
    unsigned memberIdx, ArrayRef<const AccessTouch *> touches, Type backingType,
    bool mutableMemory) {
  for (const AccessTouch *touch : touches) {
    if (touch->memberIdx != memberIdx) continue;
    Value viewValue = touch->accessValue;
    if (!touch->alias.steps.empty()) {
      AliasStep first = touch->alias.steps.front();
      viewValue = first.op->getOperand(first.sourceOperand);
    }
    for (AliasStep step : touch->alias.steps) {
      if (step.op->getName().getStringRef() != "ttg.memdesc_index")
        break;
      viewValue = step.op->getResult(0);
    }
    if (auto accessTy = dyn_cast<MemDescType>(viewValue.getType()))
      return withMutableMemory(accessTy, mutableMemory);
  }
  return withMutableMemory(
      getSemaphoreViewBufferType(cast<MemDescType>(backingType)),
      mutableMemory);
}

static FailureOr<unsigned> getBackingIndex(Operation *op,
                                           const GroupBacking &backing,
                                           unsigned memberIdx) {
  auto it = backing.memberToBackingIndex.find(memberIdx);
  if (it == backing.memberToBackingIndex.end())
    return op->emitError("nvws-insert-semas: semaphore backing has no member "
                         "for planned resource touch");
  return it->second;
}

static SmallVector<Type, 4> getSemaphoreBufferViewTypes(BufferGroup &group,
                                                        const GroupBacking &backing,
                                                        ArrayRef<const AccessTouch *> touches,
                                                        bool mutableMemory) {
  SmallVector<Type, 4> viewTypes;
  for (auto [idx, type] : llvm::enumerate(backing.bufferTypes)) {
    auto memDescType = cast<MemDescType>(type);
    if (group.isTmem())
      viewTypes.push_back(getSemaphoreViewBufferType(memDescType));
    else
      viewTypes.push_back(getLocalSemaphoreBufferType(
          backing.memberIndices[static_cast<unsigned>(idx)], touches, type,
          mutableMemory));
  }
  return viewTypes;
}

static SemaphoreBufferOp
emitSemaphoreBuffer(OpBuilder &b, Location loc, Value sem, Value token,
                    std::optional<PartitionId> owner, StageCluster stageCluster,
                    BufferGroup &group, const GroupBacking &backing,
                    ArrayRef<const AccessTouch *> touches,
                    bool mutableMemory) {
  SmallVector<Type, 4> viewTypes =
      getSemaphoreBufferViewTypes(group, backing, touches, mutableMemory);
  return createIntoPartition<SemaphoreBufferOp>(
      b, loc, {owner, stageCluster}, sem, TypeRange(viewTypes), token);
}

static SemaphoreAcquireOp emitAcquire(OpBuilder &b, Location loc, Value sem,
                                      std::optional<PartitionId> owner,
                                      StageCluster stageCluster) {
  return createIntoPartition<SemaphoreAcquireOp>(
      b, loc, {owner, stageCluster}, sem, b.getType<AsyncTokenType>());
}

static SemaphoreReleaseOp emitRelease(OpBuilder &b, Location loc, Value sem,
                                      Value token,
                                      std::optional<PartitionId> owner,
                                      StageCluster stageCluster,
                                      AsyncOp payload) {
  return createIntoPartition<SemaphoreReleaseOp>(
      b, loc, {owner, stageCluster}, sem, token, asyncPayloadArray(b, payload));
}

static StageCluster expectedStampedStage(std::optional<PartitionId> owner,
                                         StageCluster stageCluster) {
  return owner ? stageCluster : StageCluster{};
}

static Operation *createNVWSDescriptorLoadOp(
    OpBuilder &b, Operation *ttDescLoadOp, Value dataBuf,
    std::optional<PartitionId> owner, StageCluster stageCluster, Location loc) {
  int txCount = getTxCount(ttDescLoadOp);
  if (auto descLoad = dyn_cast<triton::DescriptorLoadOp>(ttDescLoadOp)) {
    auto newDescLoad = createIntoPartition<triton::nvws::DescriptorLoadOp>(
        b, loc, {owner, stageCluster}, descLoad.getDesc(),
        descLoad.getIndices(), txCount, dataBuf, descLoad.getCache(),
        descLoad.getEvict());
    newDescLoad->setAttrs(descLoad->getAttrs());
    setStageCluster(b, newDescLoad, stageCluster);
    if (owner)
      setPartition(newDescLoad, partitionSetForOwner(owner));
    return newDescLoad.getOperation();
  }
  if (auto descGather = dyn_cast<triton::DescriptorGatherOp>(ttDescLoadOp)) {
    auto newDescGather = createIntoPartition<triton::nvws::DescriptorGatherOp>(
        b, loc, {owner, stageCluster}, descGather.getDesc(),
        descGather.getXOffsets(), descGather.getYOffset(), txCount, dataBuf);
    newDescGather->setAttrs(descGather->getAttrs());
    setStageCluster(b, newDescGather, stageCluster);
    if (owner)
      setPartition(newDescGather, partitionSetForOwner(owner));
    return newDescGather.getOperation();
  }
  llvm_unreachable("unknown descriptor op");
}

static Operation *latestSameBlockConsumer(Operation *anchor) {
  Operation *latest = anchor;
  Block *block = anchor->getBlock();
  SmallVector<Operation *, 8> worklist;
  DenseSet<Operation *> seen;
  for (Value result : anchor->getResults())
    for (Operation *user : result.getUsers())
      worklist.push_back(user);

  while (!worklist.empty()) {
    Operation *user = worklist.pop_back_val();
    if (!seen.insert(user).second)
      continue;
    Operation *ancestor = block->findAncestorOpInBlock(*user);
    if (!ancestor)
      continue;
    if (latest->isBeforeInBlock(ancestor))
      latest = ancestor;
    for (Value result : user->getResults())
      for (Operation *next : result.getUsers())
        worklist.push_back(next);
  }
  return latest;
}

static bool hasMemDescResult(Operation *op) {
  return llvm::any_of(op->getResults(),
                      [](Value result) { return isa<MemDescType>(result.getType()); });
}

static void collectTransitiveConsumers(Operation *producer, Block *anchorBlock,
                                       DenseSet<Operation *> &seen,
                                       SetVector<Operation *> &consumers) {
  if (!seen.insert(producer).second)
    return;
  for (Value result : producer->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (hasMemDescResult(user)) {
        collectTransitiveConsumers(user, anchorBlock, seen, consumers);
        continue;
      }
      Operation *ancestor = anchorBlock->findAncestorOpInBlock(*user);
      consumers.insert(ancestor ? ancestor : user);
    }
  }
}

static Operation *latestTransitiveConsumer(Operation *anchor) {
  SetVector<Operation *> consumers;
  DenseSet<Operation *> seen;
  collectTransitiveConsumers(anchor, anchor->getBlock(), seen, consumers);
  if (consumers.empty())
    return latestSameBlockConsumer(anchor);
  SmallVector<Operation *, 8> consumerOps(consumers.begin(), consumers.end());
  Operation *scope = nullptr;
  if (auto funcOp = anchor->getParentOfType<triton::FuncOp>())
    scope = funcOp.getOperation();
  PostDominanceInfo dom(scope ? scope : anchor->getParentOp());
  Operation *postDom = findNearestCommonPostDominator(consumerOps, dom);
  if (!postDom)
    return latestSameBlockConsumer(anchor);
  Operation *ancestor = anchor->getBlock()->findAncestorOpInBlock(*postDom);
  return ancestor ? ancestor : postDom;
}

static bool sameMemDescViewType(Type a, Type b) {
  if (a == b)
    return true;
  auto aTy = dyn_cast<MemDescType>(a);
  auto bTy = dyn_cast<MemDescType>(b);
  if (!aTy || !bTy)
    return false;
  return aTy.getShape() == bTy.getShape() &&
         aTy.getElementType() == bTy.getElementType() &&
         aTy.getEncoding() == bTy.getEncoding() &&
         aTy.getMemorySpace() == bTy.getMemorySpace() &&
         aTy.getMutableMemory() == bTy.getMutableMemory();
}

static Value materializeAliasForBuffer(OpBuilder &b, const AccessTouch &touch,
                                       Value memberBuffer) {
  Value cur = memberBuffer;
  for (AliasStep step : touch.alias.steps) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" &&
        old->getNumResults() == 1 &&
        sameMemDescViewType(old->getResult(0).getType(), cur.getType()))
      continue;
    IRMapping mapping;
    for (auto [idx, operand] : llvm::enumerate(old->getOperands()))
      mapping.map(operand, idx == step.sourceOperand ? cur : operand);
    Operation *cloned = b.clone(*old, mapping);
    cur = cloned->getResult(0);
  }
  return cur;
}

static void replaceUsesExcept(Value oldValue, Value newValue,
                              Operation *except) {
  SmallVector<OpOperand *> uses;
  DominanceInfo domInfo(
      except->getParentOfType<triton::FuncOp>().getOperation());
  for (OpOperand &use : oldValue.getUses())
    if (use.getOwner() != except && !isa<SemaphoreCreateOp>(use.getOwner()) &&
        domInfo.dominates(newValue, use.getOwner()))
      uses.push_back(&use);
  for (OpOperand *use : uses)
    use->set(newValue);
}

static void replaceTokenResults(Operation *op, Value token) {
  if (!token) return;
  for (Value result : op->getResults())
    if (isa<AsyncTokenType>(result.getType()))
      result.replaceAllUsesWith(token);
}

static void poisonTokenResults(OpBuilder &b, Operation *op,
                               Operation *insertBefore = nullptr) {
  bool hasTokenResult = llvm::any_of(op->getResults(), [](Value result) {
    return isa<AsyncTokenType>(result.getType());
  });
  if (!hasTokenResult)
    return;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(insertBefore ? insertBefore : op);
  Value poison =
      ub::PoisonOp::create(b, op->getLoc(), b.getType<AsyncTokenType>());
  replaceTokenResults(op, poison);
}

static void poisonOriginalTmemAllocTokens(BufferGroup &group) {
  if (!group.isTmem())
    return;
  Operation *anchor = nullptr;
  for (BufferMember &member : group.members) {
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getToken() || allocOp.getToken().use_empty())
      continue;
    anchor = allocOp.getOperation();
    break;
  }
  if (!anchor)
    return;

  OpBuilder b(anchor);
  b.setInsertionPoint(anchor);
  Value poison =
      ub::PoisonOp::create(b, anchor->getLoc(), b.getType<AsyncTokenType>());
  for (BufferMember &member : group.members) {
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getToken())
      continue;
    allocOp.getToken().replaceAllUsesWith(poison);
  }
}

static bool touchesWrittenAccumulator(ArrayRef<const AccessTouch *> touches,
                                      Value accumulator) {
  return llvm::any_of(touches, [&](const AccessTouch *touch) {
    return touch->accessValue == accumulator && touchWrites(*touch);
  });
}

static void clearOwnedTmemTokenOperands(Operation *op) {
  if (auto tmemLoad = dyn_cast<TMEMLoadOp>(op)) {
    tmemLoad.getDepMutable().clear();
    return;
  }
  if (auto tmemStore = dyn_cast<TMEMStoreOp>(op)) {
    tmemStore.getDepMutable().clear();
    return;
  }
  if (auto mma = dyn_cast<MMAv5OpInterface>(op))
    mma.getAccDepMutable().clear();
}

static bool accessOwnsAsyncToken(Operation *op,
                                 ArrayRef<const AccessTouch *> touches,
                                 BufferGroup &group) {
  if (!group.isTmem())
    return false;
  if (isa<TMEMAllocOp, TMEMLoadOp, TMEMStoreOp>(op))
    return true;
  if (auto mma = dyn_cast<MMAv5OpInterface>(op))
    return llvm::any_of(touches, [](const AccessTouch *touch) {
      return touchWrites(*touch);
    });
  return false;
}

static Operation *getBufferDefiningOp(ArrayRef<Value> buffers) {
  for (Value buffer : buffers)
    if (Operation *def = buffer.getDefiningOp())
      return def;
  return nullptr;
}

static bool allResultsUnused(Operation *op) {
  for (Value result : op->getResults())
    if (!result.use_empty())
      return false;
  return true;
}

static void eraseUnusedOriginals(BufferGroup &group) {
  for (Operation *op : llvm::reverse(group.aliasOps))
    if (allResultsUnused(op))
      op->erase();
  for (BufferMember &member : group.members)
    if (!isSemaphoreBackingAlloc(member.allocOp) && allResultsUnused(member.allocOp))
      member.allocOp->erase();
}

static bool isEligibleTmemReuseAlloc(TMEMAllocOp allocOp) {
  if (!getBufferId(allocOp))
    return false;
  if (allocOp.getSrc())
    return false;
  if (auto token = allocOp.getToken(); token && !token.use_empty())
    return false;

  auto type = allocOp.getResult().getType();
  if (type.getRank() < 2)
    return false;
  if (!isa<TensorMemorySpaceAttr>(type.getMemorySpace()))
    return false;
  if (!isa<TensorMemoryEncodingAttr>(type.getEncoding()))
    return false;

  return true;
}

static int64_t getI64AttrOr(Operation *op, StringRef name,
                            int64_t defaultValue) {
  return getI64Attr(op, name).value_or(defaultValue);
}

using TmemReuseKey = std::pair<int64_t, int64_t>;

static TmemReuseKey getTmemReuseKey(TMEMAllocOp allocOp) {
  return {*getBufferId(allocOp),
          getI64AttrOr(allocOp, kBufferCopyAttrName, -1)};
}

struct TmemReuseView {
  int64_t offset = 0;
  int64_t sliceSize = 0;
};

static std::optional<TmemReuseView> getTmemReuseView(
    TMEMAllocOp representative, TMEMAllocOp duplicate) {
  if (getBufferOffset(representative) != 0)
    return std::nullopt;

  auto baseType = representative.getResult().getType();
  auto duplicateType = duplicate.getResult().getType();
  if (baseType.getRank() != duplicateType.getRank())
    return std::nullopt;

  ArrayRef<int64_t> baseShape = baseType.getShape();
  ArrayRef<int64_t> duplicateShape = duplicateType.getShape();
  for (int i = 0, e = baseType.getRank() - 1; i < e; ++i)
    if (baseShape[i] != duplicateShape[i])
      return std::nullopt;

  int64_t duplicateOffset = getBufferOffset(duplicate);
  if (duplicateOffset < 0)
    return std::nullopt;

  int64_t baseBlockN = baseShape.back();
  int64_t duplicateBlockN = duplicateShape.back();
  int64_t baseElemWidth = baseType.getElementTypeBitWidth();
  int64_t duplicateElemWidth = duplicateType.getElementTypeBitWidth();

  int64_t sliceSize = 0;
  if (baseElemWidth == duplicateElemWidth) {
    sliceSize = duplicateBlockN;
  } else if (baseElemWidth == duplicateElemWidth * 2) {
    if (duplicateBlockN % 2 != 0)
      return std::nullopt;
    sliceSize = duplicateBlockN / 2;
  } else {
    return std::nullopt;
  }

  if (sliceSize <= 0 || duplicateOffset + sliceSize > baseBlockN)
    return std::nullopt;

  return TmemReuseView{duplicateOffset, sliceSize};
}

static bool canRepresentTmemReuseGroup(TMEMAllocOp representative,
                                       ArrayRef<TMEMAllocOp> group) {
  return llvm::all_of(group, [&](TMEMAllocOp duplicate) {
    return duplicate == representative ||
           getTmemReuseView(representative, duplicate).has_value();
  });
}

static TMEMAllocOp chooseTmemReuseRepresentative(ArrayRef<TMEMAllocOp> group) {
  for (TMEMAllocOp candidate : group)
    if (canRepresentTmemReuseGroup(candidate, group))
      return candidate;
  return {};
}

static bool moveRepresentativeBeforeGroup(TMEMAllocOp representative,
                                          ArrayRef<TMEMAllocOp> group) {
  Block *block = representative->getBlock();
  Operation *earliest = representative.getOperation();
  for (TMEMAllocOp allocOp : group) {
    if (allocOp->getBlock() != block)
      return false;
    if (allocOp->isBeforeInBlock(earliest))
      earliest = allocOp.getOperation();
  }
  if (earliest != representative.getOperation())
    representative->moveBefore(earliest);
  return true;
}

static Value createTmemReuseView(OpBuilder &builder,
                                 TMEMAllocOp representative,
                                 TMEMAllocOp duplicate,
                                 TmemReuseView view) {
  auto duplicateType = duplicate.getResult().getType();
  if (representative.getResult().getType() == duplicateType &&
      view.offset == 0)
    return representative.getResult();

  builder.setInsertionPoint(duplicate);
  auto subSlice = TMEMSubSliceOp::create(builder, duplicate.getLoc(),
                                         representative.getResult(),
                                         view.offset, view.sliceSize);
  auto reinterpret = MemDescReinterpretOp::create(
      builder, duplicate.getLoc(), duplicateType, subSlice);
  setPartitionFromAnchor(subSlice, duplicate);
  setPartitionFromAnchor(reinterpret, duplicate);
  if (StageCluster stageCluster = getStageCluster(duplicate)) {
    setStageCluster(builder, subSlice, stageCluster);
    setStageCluster(builder, reinterpret, stageCluster);
  }
  return reinterpret.getResult();
}

static void coalesceTmemAllocsByBufferIdIntoViews(triton::FuncOp funcOp) {
  SmallVector<TMEMAllocOp> allocs;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isEligibleTmemReuseAlloc(allocOp))
      allocs.push_back(allocOp);
  });

  llvm::MapVector<TmemReuseKey, SmallVector<TMEMAllocOp>> groups;
  for (TMEMAllocOp allocOp : allocs)
    groups[getTmemReuseKey(allocOp)].push_back(allocOp);

  OpBuilder builder(funcOp.getContext());
  for (auto &entry : groups) {
    SmallVector<TMEMAllocOp> &group = entry.second;
    if (group.size() < 2)
      continue;

    TMEMAllocOp representative = chooseTmemReuseRepresentative(group);
    if (!representative)
      continue;
    if (!moveRepresentativeBeforeGroup(representative, group))
      continue;

    DominanceInfo domInfo(funcOp);
    if (!llvm::all_of(group, [&](TMEMAllocOp duplicate) {
          return duplicate == representative ||
                 domInfo.dominates(representative.getOperation(),
                                   duplicate.getOperation());
        }))
      continue;

    for (TMEMAllocOp duplicate : group) {
      if (duplicate == representative)
        continue;
      std::optional<TmemReuseView> view =
          getTmemReuseView(representative, duplicate);
      if (!view)
        continue;
      Value replacement =
          createTmemReuseView(builder, representative, duplicate, *view);
      duplicate.getResult().replaceAllUsesWith(replacement);
    }
  }
}

static void eraseDeadTmemAllocs(triton::FuncOp funcOp) {
  SmallVector<TMEMAllocOp> allocs;
  funcOp.walk([&](TMEMAllocOp allocOp) { allocs.push_back(allocOp); });
  for (TMEMAllocOp allocOp : llvm::reverse(allocs))
    if (allResultsUnused(allocOp))
      allocOp.erase();
}

static LogicalResult emitAccessEvent(OpBuilder &b, AccessEvent &event,
                                     ArrayRef<const AccessTouch *> touches,
                                     ArrayRef<AcquireRecord> acquires,
                                     BufferGroup &group,
                                     const OptSyncDag &dag,
                                     const GroupBacking &backing,
                                     EmitState &state) {
  Operation *op = event.op;
  bool writes = llvm::any_of(touches, [](const AccessTouch *touch) {
    return touchWrites(*touch);
  });

  Value sem;
  Value token;
  if (!acquires.empty()) {
    sem = acquires.front().semaphore;
    token = acquires.front().token;
  } else if (state.currentToken && state.currentSemaphore) {
    sem = state.currentSemaphore;
    token = state.currentToken;
  } else if (writes) {
    return op->emitError("nvws-insert-semas: missing planned EMPTY/FULL carrier "
                         "token for writer");
  } else {
    return success();
  }

  StageCluster stageCluster = getStageCluster(op);
  Operation *bufferOperation = nullptr;
  SmallVector<Value, 4> buffers;
  StageCluster bufferExpectedStageCluster =
      expectedStampedStage(event.owner, stageCluster);
  bool canReuseCurrentBuffers =
      acquires.empty() && state.currentToken == token &&
      state.currentSemaphore == sem &&
      state.currentBuffers.size() == backing.bufferTypes.size();
  if (canReuseCurrentBuffers) {
    buffers.assign(state.currentBuffers.begin(), state.currentBuffers.end());
    bufferOperation = getBufferDefiningOp(buffers);
    bufferExpectedStageCluster =
        bufferOperation ? getStageCluster(bufferOperation) : StageCluster{};
  } else {
    SemaphoreBufferOp bufferOp =
        emitSemaphoreBuffer(b, op->getLoc(), sem, token, event.owner,
                            stageCluster, group, backing, touches, writes);
    if (!event.owner)
      setPartitionFromAnchor(bufferOp.getOperation(), op);
    if (!event.owner)
      setPartitionFromTokenIfParentPartitioned(bufferOp.getOperation(), token);
    bufferOperation = bufferOp.getOperation();
    buffers.assign(bufferOp.getBuffers().begin(), bufferOp.getBuffers().end());
    state.currentBuffers = buffers;
  }
  state.eventBuffers[op] = buffers;
  Operation *retargetOp = op;
  bool ownsAsyncToken = accessOwnsAsyncToken(op, touches, group);

  if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful TMEM alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    FailureOr<unsigned> backingIdx = getBackingIndex(op, backing, touch.memberIdx);
    if (failed(backingIdx))
      return failure();
    if (*backingIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[*backingIdx]);
    if (Value src = tmemAlloc.getSrc()) {
      auto vTrue = createIntoPartition<arith::ConstantIntOp>(
          b, op->getLoc(), {event.owner, getStageCluster(op)}, true, 1);
      auto store = createIntoPartition<TMEMStoreOp>(
          b, op->getLoc(), {event.owner, getStageCluster(op)}, Type(),
          accessBuffer, Value(), src, vTrue);
      retargetOp = store.getOperation();
      replaceUsesExcept(tmemAlloc.getResult(), accessBuffer, store);
      state.rewrittenAccessValue[tmemAlloc.getResult()] = accessBuffer;
    }
    state.emittedBuffers.push_back(EmittedBufferRecord{
        op, retargetOp, bufferOperation, sem, token, accessBuffer,
        state.eventBuffers[op], touch.memberIdx, *backingIdx,
        bufferExpectedStageCluster});
  } else if (auto localAlloc = dyn_cast<LocalAllocOp>(op)) {
    if (touches.size() != 1)
      return op->emitError("nvws-insert-semas: sourceful local alloc has "
                           "multiple touches for one resource");
    const AccessTouch &touch = *touches.front();
    FailureOr<unsigned> backingIdx = getBackingIndex(op, backing, touch.memberIdx);
    if (failed(backingIdx))
      return failure();
    if (*backingIdx >= buffers.size())
      return op->emitError("nvws-insert-semas: semaphore buffer member index out "
                           "of range");
    Value accessBuffer =
        materializeAliasForBuffer(b, touch, buffers[*backingIdx]);
    if (Value src = localAlloc.getSrc()) {
      if (Operation *def = src.getDefiningOp();
          def && isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(def)) {
        retargetOp = createNVWSDescriptorLoadOp(
            b, def, accessBuffer, event.owner, getStageCluster(op),
            op->getLoc());
      } else {
        Value storeValue = src;
        if (isa<FloatType, IntegerType>(src.getType())) {
          auto splat = createIntoPartition<triton::SplatOp>(
              b, op->getLoc(), {event.owner, getStageCluster(op)},
              getTensorTypeFromScalar(b, src), src);
          storeValue = splat;
        }
        auto store = createIntoPartition<LocalStoreOp>(
            b, op->getLoc(), {event.owner, getStageCluster(op)}, storeValue,
            accessBuffer);
        retargetOp = store.getOperation();
      }
      replaceUsesExcept(localAlloc.getResult(), accessBuffer, retargetOp);
      state.rewrittenAccessValue[localAlloc.getResult()] = accessBuffer;
    }
    state.emittedBuffers.push_back(EmittedBufferRecord{
        op, retargetOp, bufferOperation, sem, token, accessBuffer,
        state.eventBuffers[op], touch.memberIdx, *backingIdx,
        bufferExpectedStageCluster});
  } else {
    SmallVector<std::pair<OpOperand *, Value>, 4> replacements;
    for (const AccessTouch *touch : touches) {
      FailureOr<unsigned> backingIdx =
          getBackingIndex(op, backing, touch->memberIdx);
      if (failed(backingIdx))
        return failure();
      if (*backingIdx >= buffers.size())
        return op->emitError("nvws-insert-semas: semaphore buffer member index "
                             "out of range");
      Value accessBuffer =
          materializeAliasForBuffer(b, *touch, buffers[*backingIdx]);
      Value currentAccessValue = state.rewrittenAccessValue.lookup(
          touch->accessValue);
      for (OpOperand &operand : op->getOpOperands())
        if (operand.get() == touch->accessValue ||
            (currentAccessValue && operand.get() == currentAccessValue))
          replacements.push_back({&operand, accessBuffer});
      state.emittedBuffers.push_back(EmittedBufferRecord{
          op, retargetOp, bufferOperation, sem, token, accessBuffer,
          state.eventBuffers[op], touch->memberIdx, *backingIdx,
          bufferExpectedStageCluster});
    }
    for (auto [operand, accessBuffer] : replacements)
      operand->set(accessBuffer);
    if (ownsAsyncToken) {
      clearOwnedTmemTokenOperands(op);
      Operation *poisonAnchor = nullptr;
      if (auto createOp = sem.getDefiningOp<SemaphoreCreateOp>())
        if (!createOp.getBuffers().empty())
          poisonAnchor = createOp.getBuffers().front().getDefiningOp();
      if (!poisonAnchor)
        poisonAnchor =
            group.members.empty() ? bufferOperation : group.members.front().allocOp;
      state.poisonTokenResultsAfterEmission.push_back({op, poisonAnchor});
    }
  }

  if (!event.owner && retargetOp)
    setPartitionFromTokenIfParentPartitioned(retargetOp, token);

  state.eventToken[op] = token;
  state.eventSemaphore[op] = sem;
  state.protectedAccesses.insert(op);
  state.knownCarrierTokens.insert(token);
  state.currentToken = token;
  state.currentSemaphore = sem;
  state.currentOwner = event.owner;
  return success();
}

static bool valueScopeCanReachBlock(Value value, Block *block) {
  if (!value || !block)
    return false;
  Region *valueRegion = value.getParentRegion();
  Region *insertRegion = block->getParent();
  return valueRegion == insertRegion || valueRegion->isAncestor(insertRegion);
}

static FailureOr<Value> lookupReleaseToken(Location loc, const SyncEdge *edge,
                                           EmitState &state,
                                           Block *insertBlock) {
  if (edge && edge->srcOp) {
    auto it = state.eventToken.find(edge->srcOp);
    if (it != state.eventToken.end()) {
      if (!insertBlock || valueScopeCanReachBlock(it->second, insertBlock))
        return it->second;
      if (state.currentToken &&
          valueScopeCanReachBlock(state.currentToken, insertBlock))
        return state.currentToken;
    }
  }
  if (state.currentToken &&
      (!insertBlock || valueScopeCanReachBlock(state.currentToken, insertBlock)))
    return state.currentToken;
  emitError(loc, "nvws-insert-semas: planned release has no carrier token "
                 "producer");
  return failure();
}

static bool isConstantTrue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIntOp>();
  return constant && constant.value() != 0;
}

static bool isConditionalTmemStore(Operation *op) {
  auto store = dyn_cast_or_null<TMEMStoreOp>(op);
  return store && !isConstantTrue(store.getPred());
}

static bool nextLinearEdgeDstIsConditionalStore(const SyncGroup &syncGroup,
                                                const SyncPlan &sp,
                                                const SyncEdge *edge) {
  if (!edge)
    return false;
  for (auto [pos, edgeIdx] : llvm::enumerate(syncGroup.edgeIdxs)) {
    if (&sp.edges[edgeIdx] != edge)
      continue;
    if (pos + 1 >= syncGroup.edgeIdxs.size())
      return false;
    const SyncEdge &nextEdge = sp.edges[syncGroup.edgeIdxs[pos + 1]];
    return isConditionalTmemStore(nextEdge.dstOp);
  }
  return false;
}

static bool shouldForceNonePayload(const SyncGroup &syncGroup,
                                   const SyncPlan &sp, const SyncEdge *edge,
                                   SyncAnchorKind kind) {
  return syncGroup.kind == SyncGroupKind::LinearChain &&
         kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
         isa<MMAv5OpInterface>(edge->srcOp) && edge->dstOp &&
         isa<TMEMLoadOp>(edge->dstOp) &&
         nextLinearEdgeDstIsConditionalStore(syncGroup, sp, edge);
}

static bool releaseShouldPrecedeFollowingSemaphores(const SyncGroup &syncGroup,
                                                    const SyncEdge *edge,
                                                    BufferGroup &group,
                                                    int64_t resourceKey,
                                                    Operation *anchor) {
  if (!group.isTmem() || !edge || !isa<MMAv5OpInterface>(anchor))
    return false;
  bool linearReadRelease =
      syncGroup.kind == SyncGroupKind::LinearChain && edge->srcOp == anchor &&
      edgeSrcReads(*edge, group, resourceKey) &&
      !edgeSrcWrites(*edge, group, resourceKey);
  bool terminalReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      edge->dstOp == anchor && edgeDstReads(*edge, group, resourceKey) &&
      !edgeDstWrites(*edge, group, resourceKey);
  bool mmaSourceRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      edge->srcOp == anchor;
  return linearReadRelease || terminalReadRelease || mmaSourceRelease;
}

static const AccessEvent *findLastProducerInRegion(Region *region,
                                                   BufferGroup &group,
                                                   int64_t resourceKey) {
  const AccessEvent *lastProducer = nullptr;
  Operation *parent = region ? region->getParentOp() : nullptr;
  if (!parent)
    return nullptr;
  for (const AccessEvent &event : group.events)
    if (event.op && parent->isProperAncestor(event.op) &&
        eventProduces(event, resourceKey))
      lastProducer = &event;
  return lastProducer;
}

static bool hasLaterBackingGroupAccessInSameBlock(Operation *op,
                                                  BufferGroup &group) {
  if (!op || !op->getBlock())
    return false;
  bool seen = false;
  for (const AccessEvent &event : group.events) {
    if (event.op == op) {
      seen = true;
      continue;
    }
    if (!seen || !event.op || event.op->getBlock() != op->getBlock())
      continue;
    if (op->isBeforeInBlock(event.op))
      return true;
  }
  return false;
}

static void moveAfterExistingReleasesBeforeAcquire(Operation *op,
                                                   Operation *source) {
  Operation *insertBefore = source->getNextNode();
  while (insertBefore && isa<SemaphoreReleaseOp>(insertBefore))
    insertBefore = insertBefore->getNextNode();
  if (insertBefore && isa<SemaphoreAcquireOp>(insertBefore))
    op->moveBefore(insertBefore);
  else
    op->moveAfter(source);
}

static void moveAfterLoopBeforeFollowingSemaphores(Operation *op,
                                                   scf::ForOp forOp) {
  Operation *insertBefore = forOp->getNextNode();
  if (insertBefore &&
      isa<SemaphoreReleaseOp, SemaphoreAcquireOp>(insertBefore))
    op->moveBefore(insertBefore);
  else
    op->moveAfter(forOp);
}

static Operation *findLastSameBlockNonTokenResultUser(Operation *op) {
  if (!op || !op->getBlock())
    return nullptr;
  Operation *lastUser = nullptr;
  for (Value result : op->getResults()) {
    if (isa<AsyncTokenType>(result.getType()))
      continue;
    for (Operation *user : result.getUsers()) {
      if (user->getBlock() != op->getBlock() || user == op ||
          !op->isBeforeInBlock(user))
        continue;
      if (!lastUser || lastUser->isBeforeInBlock(user))
        lastUser = user;
    }
  }
  return lastUser;
}

static LogicalResult emitReleaseAction(OpBuilder &b, Location loc,
                                       SyncAnchorKind kind, Operation *anchor,
                                       Region *yieldRegion,
                                       const PlannedRelease &action,
                                       const OptSyncDag &dag, const SyncPlan &sp,
                                       BufferGroup &group, EmitState &state,
                                       StageCluster stageCluster,
                                       Operation *liveAnchor = nullptr) {
  unsigned groupIdx = action.groupIdx;
  if (groupIdx >= dag.groups.size())
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: planned release references an invalid group");
  if (action.edgeIdxs.empty())
    return group.members.front().allocOp->emitError(
        "nvws-insert-semas: planned release has no transition edge");
  for (unsigned edgeIdx : action.edgeIdxs) {
    if (edgeIdx >= sp.edges.size())
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release references an invalid edge");
    if (!edgeRequiresRelease(sp.edges[edgeIdx]))
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release is not backed by a partition "
          "transition edge");
    if (edgeIdx >= dag.edgeToGroup.size() || dag.edgeToGroup[edgeIdx] != groupIdx)
      return group.members.front().allocOp->emitError(
          "nvws-insert-semas: planned release edge does not belong to its group");
  }
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge = getRepresentativeReleaseEdge(action, sp);
  std::optional<unsigned> edgeIdx = action.edgeIdxs.front();
  for (const EmittedSyncRecord &record : state.emittedReleases)
    if (record.groupIdx == groupIdx && record.kind == kind &&
        record.anchor == anchor && record.yieldRegion == yieldRegion &&
        record.edgeIdxs == action.edgeIdxs)
      return success();
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  bool terminalDstReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->dstOp == anchor &&
      edge->srcOp != anchor;
  bool terminalLoopExitReadRelease =
      syncGroup.kind == SyncGroupKind::Singleton &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && anchor &&
      edgeIdx && dag.tmemLoopExitRead.lookup(*edgeIdx) == anchor;
  bool sourceReadRelease =
      group.isTmem() && kind == SyncAnchorKind::ReleaseAfterOp && edge &&
      anchor && edge->srcOp == anchor &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second);
  bool delayReadReleaseForUsers = group.isTmem() && group.members.size() > 1;
  bool readCompletionRelease =
      delayReadReleaseForUsers &&
      (terminalDstReadRelease || terminalLoopExitReadRelease ||
       sourceReadRelease);
  // (No semaphore override: `sem` is the mechanical class lookup above. The
  // terminal/loop-exit read-release flags below still drive owner/payload
  // materialization only, not semaphore identity.)
  bool useStructuredCarrier =
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
      edge->srcOp != anchor && state.currentToken;
  useStructuredCarrier |= kind == SyncAnchorKind::ReleaseBeforeOp && edge &&
                          edge->srcOp && edge->srcOp != anchor &&
                          state.currentToken;
  SetVector<int> structuredCarrierPartition;
  if (useStructuredCarrier)
    structuredCarrierPartition = partitionSetForValue(state.currentToken);
  FailureOr<Value> token =
      useStructuredCarrier ? FailureOr<Value>(state.currentToken)
                           : lookupReleaseToken(loc, edge, state,
                                                b.getInsertionBlock());
  if (failed(token))
    return failure();
  std::optional<PartitionId> owner = edge ? edge->srcOwner : std::nullopt;
  if (terminalDstReadRelease || terminalLoopExitReadRelease)
    owner = edge->dstOwner;
  if (terminalLoopExitReadRelease)
    owner = getPartitionId(anchor);
  if (!owner && kind == SyncAnchorKind::ReleaseBeforeOp && state.currentOwner)
    owner = state.currentOwner;
  if (edge && !edge->srcOwner && kind == SyncAnchorKind::ReleaseBeforeOp &&
      syncGroup.kind == SyncGroupKind::LinearChain &&
      syncGroup.edgeIdxs.size() > 1 && &sp.edges[syncGroup.edgeIdxs.front()] == edge)
    owner = sp.edges[syncGroup.edgeIdxs[1]].dstOwner;
  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  AsyncOp payload =
      (terminalDstReadRelease || terminalLoopExitReadRelease)
          ? getAsyncPayload(anchor)
          : (edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
  if (group.isTmem() && edge && edge->srcYieldRegion &&
      !terminalDstReadRelease && !terminalLoopExitReadRelease &&
      payload == AsyncOp::NONE)
    if (const AccessEvent *producer = findLastProducerInRegion(
            edge->srcYieldRegion, group, dag.resource.second))
      if (sameOwner(producer->owner, edge->srcOwner))
        payload = getAsyncPayload(producer->op);
  if (shouldForceNonePayload(syncGroup, sp, edge, kind))
    payload = AsyncOp::NONE;
  SemaphoreReleaseOp release =
      emitRelease(b, loc, sem, *token, owner, stageCluster, payload);
  if (readCompletionRelease && anchor)
    if (Operation *lastUser = findLastSameBlockNonTokenResultUser(anchor))
      if (release->getBlock() == lastUser->getBlock() &&
          release->isBeforeInBlock(lastUser))
        release->moveAfter(lastUser);
  bool readOnlyMmaSource =
      edge && isa_and_nonnull<MMAv5OpInterface>(edge->srcOp) &&
      edge->dstYieldRegion && edge->dstOwner &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second) &&
      !hasLaterBackingGroupAccessInSameBlock(edge->srcOp, group);
  bool writeMmaSource =
      edge && isa_and_nonnull<MMAv5OpInterface>(edge->srcOp) &&
      edge->dstYieldRegion && edge->dstOwner &&
      edgeSrcWrites(*edge, group, dag.resource.second) &&
      !hasLaterBackingGroupAccessInSameBlock(edge->srcOp, group);
  bool readOnlyTmemLoadSource =
      edge && isa_and_nonnull<TMEMLoadOp>(edge->srcOp) &&
      edgeSrcReads(*edge, group, dag.resource.second) &&
      !edgeSrcWrites(*edge, group, dag.resource.second);
  bool moveReadOnlyTmemLoadSource =
      readOnlyTmemLoadSource && group.members.size() == 1;
  if (group.isTmem() && kind == SyncAnchorKind::ReleaseBeforeYield && edge &&
      (moveReadOnlyTmemLoadSource || readOnlyMmaSource || writeMmaSource) &&
      operationIsAttached(edge->srcOp) &&
      release->getBlock() == edge->srcOp->getBlock()) {
    if (writeMmaSource)
      moveAfterExistingReleasesBeforeAcquire(release.getOperation(), edge->srcOp);
    else if (moveReadOnlyTmemLoadSource)
      release->moveAfter(edge->srcOp);
    else {
      Operation *insertAfter = edge->srcOp;
      if (Operation *lastUser =
              findLastSameBlockNonTokenResultUser(edge->srcOp))
        if (insertAfter->isBeforeInBlock(lastUser))
          insertAfter = lastUser;
      release->moveAfter(insertAfter);
    }
  }
  if (group.isTmem() && kind == SyncAnchorKind::ReleaseBeforeOp && edge &&
      edge->srcYieldRegion && anchor && opReadsOnlyResource(anchor, group,
                                                            dag.resource.second)) {
    if (edgeIdx && dag.srcYieldParentWarpFor.contains(*edgeIdx)) {
      for (Operation *candidate = release->getPrevNode(); candidate;
           candidate = candidate->getPrevNode()) {
        if (isa<SemaphoreAcquireOp, SemaphoreReleaseOp>(candidate))
          continue;
        auto forOp = dyn_cast<scf::ForOp>(candidate);
        if (!forOp)
          break;
        if (hasWarpSpecializeTag(forOp) &&
            release->getBlock() == forOp->getBlock())
          moveAfterLoopBeforeFollowingSemaphores(release.getOperation(), forOp);
        break;
      }
    }
  }
  if (useStructuredCarrier && structuredCarrierPartition.size() == 1 && !owner) {
    setPartition(release.getOperation(), structuredCarrierPartition);
    if (auto tag = wsTagForValue(state.currentToken))
      setWarpTagOutsideWsLoop(release.getOperation(), *tag);
  }
  if (!owner) {
    std::optional<PartitionId> fallbackOwner =
        edge ? edge->dstOwner : std::nullopt;
    setSingleOwnerPartition(release.getOperation(), fallbackOwner);
    if (!fallbackOwner)
      setPartitionFromAnchor(release.getOperation(),
                             anchor ? anchor
                                    : (yieldRegion ? yieldRegion->getParentOp()
                                                   : nullptr));
  }
  state.emittedReleases.push_back(EmittedSyncRecord{
      groupIdx, kind, anchor, yieldRegion, release.getOperation(), sem, *token,
      expectedStampedStage(owner, stageCluster)});
  state.emittedReleases.back().edgeIdxs.append(action.edgeIdxs.begin(),
                                               action.edgeIdxs.end());
  return success();
}

static AcquireRecord emitAcquireForGroup(OpBuilder &b, Location loc,
                                         SyncAnchorKind kind, Operation *anchor,
                                         Region *yieldRegion,
                                         unsigned groupIdx,
                                         const OptSyncDag &dag,
                                         const SyncPlan &sp, BufferGroup &group,
                                         EmitState &state,
                                         StageCluster stageCluster) {
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge =
      findEdgeForAnchor(syncGroup, sp, dag, kind, anchor, yieldRegion);
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, state.semas);
  std::optional<PartitionId> owner =
      edge ? edge->dstOwner : syncGroup.initialOwner;
  SemaphoreAcquireOp acquire = emitAcquire(b, loc, sem, owner, stageCluster);
  if (!owner) {
    std::optional<PartitionId> fallbackOwner =
        parentRequiresPartition(acquire.getOperation()) && edge
            ? edge->srcOwner
            : std::nullopt;
    setSingleOwnerPartition(acquire.getOperation(), fallbackOwner);
    setPartitionFromAnchor(acquire.getOperation(),
                           anchor ? anchor
                                  : (yieldRegion ? yieldRegion->getParentOp()
                                                 : nullptr));
  }
  Value token = acquire.getToken();
  state.emittedAcquires.push_back(EmittedSyncRecord{
      groupIdx, kind, anchor, yieldRegion, acquire.getOperation(), sem, token,
      expectedStampedStage(owner, stageCluster)});
  state.knownCarrierTokens.insert(token);
  state.currentToken = token;
  state.currentSemaphore = sem;
  state.currentOwner = owner;
  state.currentBuffers.clear();
  return AcquireRecord{groupIdx, sem, token, owner};
}

static LogicalResult
emitBeforeOpSync(Operation *anchor, const OptSyncDag &dag, const SyncPlan &sp,
                 BufferGroup &group, EmitState &state,
                 SmallVectorImpl<AcquireRecord> &acquires) {
  OpBuilder b(anchor);
  b.setInsertionPoint(anchor);
  auto rIt = dag.releaseBeforeOp.find(anchor);
  if (rIt != dag.releaseBeforeOp.end())
    for (const PlannedRelease &release : rIt->second)
      if (failed(emitReleaseAction(
              b, anchor->getLoc(), SyncAnchorKind::ReleaseBeforeOp, anchor,
              nullptr, release, dag, sp, group, state, getStageCluster(anchor))))
        return failure();
  auto aIt = dag.acquireBeforeOp.find(anchor);
  if (aIt != dag.acquireBeforeOp.end())
    for (unsigned gi : aIt->second)
      acquires.push_back(emitAcquireForGroup(
          b, anchor->getLoc(), SyncAnchorKind::AcquireBeforeOp, anchor, nullptr,
          gi, dag, sp, group, state, getStageCluster(anchor)));
  return success();
}

static LogicalResult emitAfterOpSync(Operation *anchor, Operation *insertAfter,
                                     const OptSyncDag &dag,
                                     const SyncPlan &sp, BufferGroup &group,
                                     EmitState &state) {
  auto rIt = dag.releaseAfterOp.find(anchor);
  if (rIt == dag.releaseAfterOp.end()) return success();
  Operation *releaseAfter = insertAfter;
  if (!group.isTmem() && isa<LocalLoadOp>(insertAfter))
    releaseAfter = latestTransitiveConsumer(insertAfter);
  if (anchor == insertAfter && operationIsAttached(releaseAfter) &&
      operationIsAttached(anchor) &&
      releaseAfter->isProperAncestor(anchor))
    return success();
  OpBuilder b(releaseAfter);
  bool beforeFollowingSemaphores = false;
  bool afterLoopRelease = group.isTmem() && isa<scf::ForOp>(releaseAfter);
  if (group.isTmem()) {
    for (const PlannedRelease &release : rIt->second) {
      const SyncGroup &syncGroup = dag.groups[release.groupIdx];
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (releaseShouldPrecedeFollowingSemaphores(
              syncGroup, edge, group, dag.resource.second, insertAfter)) {
        beforeFollowingSemaphores = true;
        break;
      }
    }
    beforeFollowingSemaphores |= afterLoopRelease;
  }
  if (isa<scf::YieldOp>(releaseAfter)) {
    b.setInsertionPoint(releaseAfter);
  } else if (beforeFollowingSemaphores) {
    Operation *insertBefore = releaseAfter->getNextNode();
    while (insertBefore && isa<SemaphoreReleaseOp>(insertBefore))
      insertBefore = insertBefore->getNextNode();
    if (insertBefore && isa<SemaphoreAcquireOp>(insertBefore))
      b.setInsertionPoint(insertBefore);
    else
      b.setInsertionPointAfter(releaseAfter);
  } else {
    b.setInsertionPointAfter(releaseAfter);
  }
  std::optional<Value> savedEventToken;
  bool overrideEventToken = releaseAfter != anchor && state.currentToken;
  if (overrideEventToken) {
    auto it = state.eventToken.find(anchor);
    if (it != state.eventToken.end())
      savedEventToken = it->second;
    state.eventToken[anchor] = state.currentToken;
  }
  SmallVector<PlannedRelease, 4> releaseActions(rIt->second.begin(),
                                                rIt->second.end());
  if (group.isTmem())
    llvm::stable_sort(releaseActions, [&](const PlannedRelease &lhs,
                                          const PlannedRelease &rhs) {
      const SyncGroup &lhsGroup = dag.groups[lhs.groupIdx];
      const SyncEdge *lhsEdge = getRepresentativeReleaseEdge(lhs, sp);
      const SyncGroup &rhsGroup = dag.groups[rhs.groupIdx];
      const SyncEdge *rhsEdge = getRepresentativeReleaseEdge(rhs, sp);
      bool lhsPrecedes = releaseShouldPrecedeFollowingSemaphores(
          lhsGroup, lhsEdge, group, dag.resource.second, insertAfter);
      bool rhsPrecedes = releaseShouldPrecedeFollowingSemaphores(
          rhsGroup, rhsEdge, group, dag.resource.second, insertAfter);
      return lhsPrecedes && !rhsPrecedes;
    });
  LogicalResult result = success();
  for (const PlannedRelease &release : releaseActions)
    if (failed(emitReleaseAction(
            b, insertAfter->getLoc(), SyncAnchorKind::ReleaseAfterOp, anchor,
            nullptr, release, dag, sp, group, state, getStageCluster(insertAfter),
            insertAfter))) {
      result = failure();
      break;
    }
  if (overrideEventToken) {
    if (savedEventToken)
      state.eventToken[anchor] = *savedEventToken;
    else
      state.eventToken.erase(anchor);
  }
  return result;
}

static LogicalResult emitAfterOpSync(Operation *anchor, const OptSyncDag &dag,
                                     const SyncPlan &sp, BufferGroup &group,
                                     EmitState &state) {
  return emitAfterOpSync(anchor, anchor, dag, sp, group, state);
}

static LogicalResult emitDeferredNestedAfterOpSync(Operation *releaseAfter,
                                                   const OptSyncDag &dag,
                                                   const SyncPlan &sp,
                                                   BufferGroup &group,
                                                   EmitState &state) {
  SmallVector<Operation *, 4> anchors;
  releaseAfter->walk([&](Operation *op) {
    if (op == releaseAfter)
      return;
    if (!isa<LocalLoadOp>(op))
      return;
    if (!dag.releaseAfterOp.contains(op))
      return;
    if (latestTransitiveConsumer(op) == releaseAfter)
      anchors.push_back(op);
  });
  for (Operation *anchor : anchors)
    if (failed(emitAfterOpSync(anchor, releaseAfter, dag, sp, group, state)))
      return failure();
  return success();
}

static bool linearChainAnchorsLoopExit(const SyncGroup &syncGroup,
                                       const SyncPlan &sp, Operation *forOp,
                                       Region *region) {
  if (!forOp || !region)
    return false;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    const SyncEdge &edge = sp.edges[edgeIdx];
    if (edge.srcOp == forOp || edge.dstOp == forOp ||
        edge.srcYieldRegion == region || edge.dstYieldRegion == region)
      return true;
  }
  return false;
}

static const PlannedRelease *
findPlannedAfterOpReleaseForGroup(Operation *anchor, unsigned groupIdx,
                                  const OptSyncDag &dag) {
  if (!anchor)
    return nullptr;
  auto releaseIt = dag.releaseAfterOp.find(anchor);
  if (releaseIt == dag.releaseAfterOp.end())
    return nullptr;
  for (const PlannedRelease &action : releaseIt->second)
    if (action.groupIdx == groupIdx)
      return &action;
  return nullptr;
}

static bool linearChainNeedsLoopExitDrain(unsigned groupIdx,
                                          const SyncGroup &syncGroup,
                                          Operation *forOp, Region *region,
                                          const OptSyncDag &dag,
                                          const SyncPlan &sp) {
  if (!forOp || !region)
    return false;
  if (findPlannedAfterOpReleaseForGroup(forOp, groupIdx, dag))
    return true;
  if (dag.skippedInitialLoopCarrierRegion.lookup(groupIdx) == region)
    for (unsigned edgeIdx : syncGroup.edgeIdxs)
      if (dag.edgesDeferringToSkippedLoopExit.contains(edgeIdx))
        return true;
  for (unsigned edgeIdx : syncGroup.edgeIdxs) {
    const SyncEdge &edge = sp.edges[edgeIdx];
    if (dag.loopEntryHandoffAccess.contains(edgeIdx))
      return true;
    if (edge.dstYieldRegion == region &&
        dag.terminalLoopReadEdgesDeferringToExit.contains(edgeIdx))
      return true;
  }
  return false;
}

static LogicalResult emitTmemLinearLoopExitDrain(scf::ForOp forOp,
                                                 Region *region,
                                                 const OptSyncDag &dag,
                                                 const SyncPlan &sp,
                                                 BufferGroup &group,
                                                 EmitState &state) {
  if (!group.isTmem() || !state.currentToken)
    return success();
  if (!hasWarpSpecializeTag(forOp))
    return success();
  SmallVector<unsigned, 2> groupIds;
  auto it = dag.acquireBeforeYield.find(region);
  if (it != dag.acquireBeforeYield.end())
    groupIds.append(it->second.begin(), it->second.end());
  else
    for (auto [idx, syncGroup] : llvm::enumerate(dag.groups))
      if (syncGroup.kind == SyncGroupKind::LinearChain &&
          linearChainAnchorsLoopExit(syncGroup, sp, forOp.getOperation(),
                                     region))
        groupIds.push_back(static_cast<unsigned>(idx));

  for (unsigned gi : groupIds) {
    const SyncGroup &syncGroup = dag.groups[gi];
    if (syncGroup.kind != SyncGroupKind::LinearChain ||
        syncGroup.edgeIdxs.size() < 2)
      continue;
    if (linearChainNeedsPerEdgeFulls(syncGroup, sp, group, dag.resource.second))
      continue;
    if (!linearChainNeedsLoopExitDrain(gi, syncGroup, forOp.getOperation(),
                                       region, dag, sp))
      continue;

    const SyncEdge &firstEdge = sp.edges[syncGroup.edgeIdxs.front()];
    const SyncEdge &secondEdge = sp.edges[syncGroup.edgeIdxs[1]];
    auto loopExitPayload = [&]() {
      Operation *lastWriter = nullptr;
      for (const AccessEvent &event : group.events)
        if (event.op && forOp->isProperAncestor(event.op) &&
            eventProduces(event, dag.resource.second))
          lastWriter = event.op;
      return getAsyncPayload(lastWriter);
    };
    OpBuilder b(forOp);
    b.setInsertionPointAfter(forOp);
    Location loc = forOp.getLoc();
    auto emitDrainRelease = [&](Value sem, Value token,
                                std::optional<PartitionId> owner,
                                AsyncOp payload,
                                const SyncEdge &edge) -> LogicalResult {
      if (!edgeRequiresRelease(edge))
        return forOp->emitError(
            "nvws-insert-semas: loop-exit drain release is not backed by a "
            "partition transition edge");
      emitRelease(b, loc, sem, token, owner, StageCluster{}, payload);
      return success();
    };
    auto findPlannedAfterLoopRelease =
        [&](const SyncEdge &edge) -> const PlannedRelease * {
      std::optional<unsigned> edgeIdx = findEdgeIndex(sp, &edge);
      if (!edgeIdx)
        return nullptr;
      auto releaseIt = dag.releaseAfterOp.find(forOp.getOperation());
      if (releaseIt == dag.releaseAfterOp.end())
        return nullptr;
      for (const PlannedRelease &action : releaseIt->second)
        if (llvm::is_contained(action.edgeIdxs, *edgeIdx))
          return &action;
      return nullptr;
    };
    bool skipsInitialCarrier =
        dag.skippedInitialLoopCarrierRegion.lookup(gi) == region;
    const SyncEdge *loopEntryHandoffEdge = nullptr;
    for (unsigned edgeIdx : syncGroup.edgeIdxs)
      if (dag.loopEntryHandoffAccess.contains(edgeIdx)) {
        loopEntryHandoffEdge = &sp.edges[edgeIdx];
        break;
      }
    bool deferredTerminalLoopRead = llvm::any_of(
        syncGroup.edgeIdxs, [&](unsigned edgeIdx) {
          const SyncEdge &edge = sp.edges[edgeIdx];
          return edge.dstYieldRegion == region &&
                 dag.terminalLoopReadEdgesDeferringToExit.contains(edgeIdx);
        });
    if (skipsInitialCarrier) {
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        if (!dag.edgesDeferringToSkippedLoopExit.contains(edgeIdx))
          continue;
        Value sem = getSemaphoreForGroup(gi, &edge, dag, sp, group,
                                         state.semas);
        if (failed(emitDrainRelease(sem, state.currentToken, state.currentOwner,
                                    edge.asyncPayload, edge)))
          return failure();
        state.currentSemaphore = sem;
        state.currentBuffers.clear();
        return success();
      }
    }
    std::optional<PartitionId> drainOwner = firstEdge.dstOwner;
    std::optional<PartitionId> releaseOwner = state.currentOwner;
    AsyncOp drainPayload = AsyncOp::NONE;
    AsyncOp emptyPayload = AsyncOp::NONE;
    if (loopEntryHandoffEdge) {
      releaseOwner = loopEntryHandoffEdge->srcOwner;
      drainOwner = loopEntryHandoffEdge->dstOwner;
      emptyPayload = loopExitPayload();
    } else if (skipsInitialCarrier) {
      drainOwner = secondEdge.dstOwner;
      drainPayload = loopExitPayload();
    } else if (deferredTerminalLoopRead) {
      emptyPayload = loopExitPayload();
    } else if (firstEdge.dstOp == forOp.getOperation()) {
      drainPayload = secondEdge.asyncPayload;
    } else if (isa_and_nonnull<MMAv5OpInterface>(firstEdge.srcOp) &&
               edgeDstReads(firstEdge, group, dag.resource.second) &&
               !edgeDstWrites(firstEdge, group, dag.resource.second)) {
      drainPayload = loopExitPayload();
    } else if (isa_and_nonnull<MMAv5OpInterface>(secondEdge.srcOp) &&
               edgeDstReads(secondEdge, group, dag.resource.second) &&
               !edgeDstWrites(secondEdge, group, dag.resource.second)) {
      emptyPayload = secondEdge.asyncPayload;
    }
    const SyncEdge *fullReleaseEdge =
        loopEntryHandoffEdge ? loopEntryHandoffEdge : &firstEdge;
    // Mechanical identity: the drain's "full" side is the class of the
    // full-release edge; the "empty" side is the class of the second (carry)
    // edge. Both are created in createResourceSemaphores.
    Value fullSem = state.semas.forClass(sp, findEdgeIndex(sp, fullReleaseEdge));
    Value emptySem = state.semas.forClass(sp, findEdgeIndex(sp, &secondEdge));
    if (const PlannedRelease *releaseAction =
            findPlannedAfterLoopRelease(*fullReleaseEdge)) {
      if (failed(emitReleaseAction(b, loc, SyncAnchorKind::ReleaseAfterOp,
                                   forOp.getOperation(), nullptr,
                                   *releaseAction, dag, sp, group, state,
                                   StageCluster{})))
        return failure();
    } else if (failed(emitDrainRelease(fullSem, state.currentToken,
                                       releaseOwner, drainPayload,
                                       *fullReleaseEdge))) {
      return failure();
    }
    SemaphoreAcquireOp acquire =
        emitAcquire(b, loc, fullSem, drainOwner, StageCluster{});
    state.knownCarrierTokens.insert(acquire.getToken());
    if (failed(emitDrainRelease(emptySem, acquire.getToken(), drainOwner,
                                emptyPayload, secondEdge)))
      return failure();
    state.currentToken = acquire.getToken();
    state.currentSemaphore = emptySem;
    state.currentOwner = drainOwner;
    return success();
  }
  return success();
}

static bool collectFirstReadOnlyRegionAccess(Region &region, const OptSyncDag &dag,
                                             BufferGroup &group,
                                             SmallVectorImpl<const AccessTouch *> &touches) {
  Operation *firstAccess = nullptr;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (!dag.accessOps.contains(op))
      return WalkResult::advance();
    for (AccessEvent &event : group.events) {
      if (event.op != op)
        continue;
      collectTouchesForResource(event, dag.resource.second, touches);
      firstAccess = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!firstAccess || touches.empty())
    return false;
  return llvm::all_of(touches, [](const AccessTouch *touch) {
    return !touchWrites(*touch);
  });
}

static bool canPrebufferLocalRegionEntry(Operation *anchor, Region &region,
                                         ArrayRef<AcquireRecord> acquires,
                                         const OptSyncDag &dag,
                                         BufferGroup &group,
                                         SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!isa<scf::ForOp>(anchor))
    return false;
  if (!dag.acquireBeforeOp.contains(anchor) || dag.releaseAfterOp.contains(anchor) ||
      dag.acquireBeforeYield.contains(&region))
    return false;
  return collectFirstReadOnlyRegionAccess(region, dag, group, touches);
}

static bool canPrebufferLocalIfEntry(scf::IfOp ifOp,
                                     ArrayRef<AcquireRecord> acquires,
                                     const OptSyncDag &dag, BufferGroup &group,
                                     SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!dag.acquireBeforeOp.contains(ifOp.getOperation()) ||
      dag.releaseAfterOp.contains(ifOp.getOperation()) ||
      dag.acquireBeforeYield.contains(&ifOp.getThenRegion()) ||
      dag.acquireBeforeYield.contains(&ifOp.getElseRegion()))
    return false;
  if (collectFirstReadOnlyRegionAccess(ifOp.getThenRegion(), dag, group,
                                       touches))
    return true;
  return collectFirstReadOnlyRegionAccess(ifOp.getElseRegion(), dag, group,
                                          touches);
}

static void prebufferLocalRegionEntry(OpBuilder &b, Operation *anchor,
                                      ArrayRef<const AccessTouch *> touches,
                                      const AcquireRecord &acquire,
                                      BufferGroup &group,
                                      const GroupBacking &backing,
                                      EmitState &state) {
  b.setInsertionPoint(anchor);
  SemaphoreBufferOp bufferOp =
      emitSemaphoreBuffer(b, anchor->getLoc(), acquire.semaphore, acquire.token,
                          acquire.owner, getStageCluster(anchor), group, backing,
                          touches, /*mutableMemory=*/false);
  if (!acquire.owner)
    setPartitionFromAnchor(bufferOp.getOperation(), anchor);
  if (!acquire.owner)
    setPartitionFromTokenIfParentPartitioned(bufferOp.getOperation(),
                                             acquire.token);
  state.currentBuffers.assign(bufferOp.getBuffers().begin(),
                              bufferOp.getBuffers().end());
}

static LogicalResult
emitBeforeYieldSync(Operation *yieldOp, Region *region, const OptSyncDag &dag,
                    const SyncPlan &sp, BufferGroup &group, EmitState &state,
                    SmallVectorImpl<AcquireRecord> &acquires) {
  OpBuilder b(yieldOp);
  b.setInsertionPoint(yieldOp);
  auto rIt = dag.releaseBeforeYield.find(region);
  if (rIt != dag.releaseBeforeYield.end())
    for (const PlannedRelease &release : rIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (failed(emitReleaseAction(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseBeforeYield, nullptr,
              region, release, dag, sp, group, state,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  auto aIt = dag.acquireBeforeYield.find(region);
  if (aIt != dag.acquireBeforeYield.end())
    for (unsigned gi : aIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            dag,
                            SyncAnchorKind::AcquireBeforeYield, nullptr, region);
      acquires.push_back(emitAcquireForGroup(
          b, yieldOp->getLoc(), SyncAnchorKind::AcquireBeforeYield, nullptr,
          region, gi, dag, sp, group, state,
          stageForYieldOwner(edge ? edge->dstOwner : std::nullopt, state)));
    }
  auto arIt = dag.releaseAfterYield.find(region);
  if (arIt != dag.releaseAfterYield.end())
    for (const PlannedRelease &release : arIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (failed(emitReleaseAction(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseAfterYield, nullptr,
              region, release, dag, sp, group, state,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  return success();
}

static bool shouldThreadForRegion(scf::ForOp forOp, const OptSyncDag &dag) {
  return dag.threadForOps.contains(forOp.getOperation());
}

static bool shouldThreadIfRegion(scf::IfOp ifOp, const OptSyncDag &dag) {
  return dag.threadIfOps.contains(ifOp.getOperation());
}

static Operation *getDominatingPoisonAnchor(Operation *op) {
  Operation *anchor = op;
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp())
    if (isa<scf::ForOp>(parent))
      anchor = parent;
  return anchor;
}

static bool linearChainEntersFor(Operation *forOp, const OptSyncDag &dag,
                                 const SyncPlan &sp) {
  if (!forOp)
    return false;
  for (const SyncGroup &syncGroup : dag.groups)
    if (syncGroup.kind == SyncGroupKind::LinearChain &&
        !syncGroup.edgeIdxs.empty() &&
        sp.edges[syncGroup.edgeIdxs.front()].dstOp == forOp)
      return true;
  return false;
}

static std::optional<PartitionId>
linearChainLoopYieldSourceOwner(Operation *forOp, const OptSyncDag &dag,
                                const SyncPlan &sp) {
  auto loop = dyn_cast_or_null<scf::ForOp>(forOp);
  if (!loop)
    return std::nullopt;
  Region *region = &loop.getRegion();
  for (auto [groupIdx, syncGroup] : llvm::enumerate(dag.groups)) {
    if (syncGroup.kind != SyncGroupKind::LinearChain ||
        syncGroup.edgeIdxs.empty() ||
        sp.edges[syncGroup.edgeIdxs.front()].dstOp != forOp)
      continue;
    for (unsigned edgeIdx : llvm::reverse(syncGroup.edgeIdxs)) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (edge.dstYieldRegion == region) {
        auto acquireIt = dag.acquireBeforeYield.find(region);
        if (acquireIt != dag.acquireBeforeYield.end() &&
            llvm::is_contained(acquireIt->second,
                               static_cast<unsigned>(groupIdx)))
          return edge.dstOwner;
        return edge.srcOwner;
      }
    }
  }
  return std::nullopt;
}

static std::optional<PartitionId>
linearChainIfYieldSourceOwner(Operation *ifOp, const OptSyncDag &dag,
                              const SyncPlan &sp) {
  if (!isa_and_nonnull<scf::IfOp>(ifOp))
    return std::nullopt;
  for (const SyncGroup &syncGroup : dag.groups) {
    if (syncGroup.kind != SyncGroupKind::LinearChain)
      continue;
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (isIfYieldRegion(edge.dstYieldRegion) &&
          edge.dstYieldRegion->getParentOp() == ifOp)
        return edge.srcOwner;
    }
  }
  return std::nullopt;
}

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src);

static FailureOr<scf::ForOp> threadCarrierThroughFor(OpBuilder &b,
                                                     scf::ForOp forOp,
                                                     EmitState &state,
                                                     Region *plannedRegion,
                                                     std::optional<PartitionId>
                                                         recordOwner,
                                                     BufferGroup &group,
                                                     ArrayRef<unsigned>
                                                         memberIndices,
                                                     int64_t resourceKey) {
  unsigned oldNumResults = forOp.getNumResults();
  auto oldPartitionIds =
      hasPartition(forOp) ? getPartitionIds(forOp) : SetVector<int>();
  auto oldPartitionOutputs =
      hasPartition(forOp) ? getPartitionOutputs(forOp)
                          : SmallVector<SetVector<int>, 4>();

  SmallVector<unsigned, 4> reusableSlots =
      findReusableTmemTokenSlots(forOp, group, memberIndices, resourceKey);
  Value init = state.currentToken;
  if (!init && !reusableSlots.empty())
    init = forOp->getOperand(3 + reusableSlots.front());
  if (!init) {
    forOp.emitError("nvws-insert-semas: planned scf.for carrier threading has "
                    "no token producer at loop entry");
    return failure();
  }
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(init, state.currentOwner, forOp.getOperation());
  if (!reusableSlots.empty()) {
    unsigned carrierSlot = reusableSlots.front();
    Value poison;
    if (reusableSlots.size() > 1) {
      OpBuilder::InsertionGuard guard(b);
      if (Operation *def = init.getDefiningOp())
        b.setInsertionPoint(def);
      else
        b.setInsertionPoint(forOp);
      poison =
          ub::PoisonOp::create(b, forOp.getLoc(), b.getType<AsyncTokenType>());
    }
    for (unsigned slot : reusableSlots)
      forOp->setOperand(3 + slot, slot == carrierSlot ? init : poison);

    state.currentToken = forOp.getRegionIterArg(carrierSlot);
    state.knownCarrierTokens.insert(state.currentToken);
    state.currentBuffers.clear();
    state.reusedForCarrierSlots[forOp.getOperation()] = carrierSlot;
    state.reusedForTokenSlots[forOp.getOperation()] = reusableSlots;
    if (poison)
      state.reusedForPoisonTokens[forOp.getOperation()] = poison;
    state.threadedTokens.push_back({forOp.getOperation(), state.currentToken,
                                    recordOwner,
                                    ThreadRecordKind::ForIterArg,
                                    plannedRegion, nullptr});
    if (hasPartition(forOp)) {
      addPartitionIds(oldPartitionIds, carrierPartition);
      if (carrierSlot < oldPartitionOutputs.size())
        oldPartitionOutputs[carrierSlot] = carrierPartition;
      setPartition(forOp, oldPartitionIds);
      setPartitionOutputs(forOp, oldPartitionOutputs);
    }
    return forOp;
  }

  b.setInsertionPoint(forOp);
  scf::ForOp newFor = addIterArgsToLoop(b, forOp, {init});
  state.currentToken = newFor.getRegionIterArg(oldNumResults);
  state.knownCarrierTokens.insert(state.currentToken);
  state.currentBuffers.clear();
  state.threadedTokens.push_back({newFor.getOperation(), state.currentToken,
                                  recordOwner,
                                  ThreadRecordKind::ForIterArg,
                                  plannedRegion, nullptr});
  if (hasPartition(newFor)) {
    addPartitionIds(oldPartitionIds, carrierPartition);
    oldPartitionOutputs.push_back(carrierPartition);
    setPartition(newFor, oldPartitionIds);
    setPartitionOutputs(newFor, oldPartitionOutputs);
  }
  return newFor;
}

static void closeCarrierForLoop(scf::ForOp forOp, EmitState &bodyState,
                                EmitState &parentState,
                                std::optional<PartitionId> ownerAtYield,
                                bool overrideOwnerAtYield,
                                Region *plannedRegion) {
  auto reusedIt = bodyState.reusedForCarrierSlots.find(forOp.getOperation());
  if (reusedIt != bodyState.reusedForCarrierSlots.end()) {
    unsigned slot = reusedIt->second;
    Value yieldedToken = bodyState.currentToken
                             ? bodyState.currentToken
                             : forOp.getRegionIterArg(slot);
    std::optional<PartitionId> resultOwner =
        overrideOwnerAtYield ? ownerAtYield
                             : (ownerAtYield ? ownerAtYield
                                             : bodyState.currentOwner);
    auto yieldOp = getForYieldOp(forOp);
    auto slotsIt = bodyState.reusedForTokenSlots.find(forOp.getOperation());
    ArrayRef<unsigned> tokenSlots =
        slotsIt == bodyState.reusedForTokenSlots.end()
            ? ArrayRef<unsigned>(slot)
            : ArrayRef<unsigned>(slotsIt->second);
    Value poison = bodyState.reusedForPoisonTokens.lookup(forOp.getOperation());
    for (unsigned tokenSlot : tokenSlots)
      yieldOp.setOperand(tokenSlot,
                         tokenSlot == slot ? yieldedToken : poison);
    SetVector<int> carrierPartition = partitionSetForTokenOrOwner(
        yieldedToken, resultOwner, forOp.getOperation());
    SetVector<int> yieldPartition;
    if (hasPartition(yieldOp))
      yieldPartition = getPartitionIds(yieldOp);
    addPartitionIds(yieldPartition, carrierPartition);
    if (!yieldPartition.empty())
      setPartition(yieldOp.getOperation(), yieldPartition);
    if (!carrierPartition.empty() && hasPartition(forOp)) {
      auto partitionIds = getPartitionIds(forOp);
      addPartitionIds(partitionIds, carrierPartition);
      auto partitionOutputs = getPartitionOutputs(forOp);
      if (slot < partitionOutputs.size())
        partitionOutputs[slot] = carrierPartition;
      setPartitionOutputs(forOp, partitionOutputs);
    }
    parentState = bodyState;
    parentState.currentToken = forOp.getResult(slot);
    parentState.knownCarrierTokens.insert(parentState.currentToken);
    parentState.currentOwner = resultOwner;
    parentState.currentBuffers.clear();
    parentState.threadedTokens.push_back({forOp.getOperation(),
                                          parentState.currentToken,
                                          parentState.currentOwner,
                                          ThreadRecordKind::ForResult,
                                          plannedRegion, nullptr});
    return;
  }

  Value yieldedToken = bodyState.currentToken
                           ? bodyState.currentToken
                           : forOp.getRegionIterArg(forOp.getNumResults() - 1);
  std::optional<PartitionId> resultOwner =
      overrideOwnerAtYield ? ownerAtYield
                           : (ownerAtYield ? ownerAtYield
                                           : bodyState.currentOwner);
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(yieldedToken, resultOwner, forOp.getOperation());
  appendToForYield(forOp, yieldedToken);
  if (!carrierPartition.empty()) {
    if (hasPartition(forOp)) {
      auto partitionIds = getPartitionIds(forOp);
      addPartitionIds(partitionIds, carrierPartition);
      auto partitionOutputs = getPartitionOutputs(forOp);
      if (partitionOutputs.size() == forOp.getNumResults())
        partitionOutputs.back() = carrierPartition;
      setPartitionOutputs(forOp, partitionOutputs);
    }
    scf::YieldOp yieldOp = getForYieldOp(forOp);
    SetVector<int> yieldPartition;
    if (hasPartition(yieldOp))
      yieldPartition = getPartitionIds(yieldOp);
    addPartitionIds(yieldPartition, carrierPartition);
    setPartition(yieldOp, yieldPartition);
  }
  parentState = bodyState;
  parentState.currentToken = forOp.getResult(forOp.getNumResults() - 1);
  parentState.knownCarrierTokens.insert(parentState.currentToken);
  parentState.currentOwner = resultOwner;
  parentState.currentBuffers.clear();
  parentState.threadedTokens.push_back({forOp.getOperation(),
                                        parentState.currentToken,
                                        parentState.currentOwner,
                                        ThreadRecordKind::ForResult,
                                        plannedRegion, nullptr});
}

static void closeExistingCarrierForLoop(scf::ForOp forOp, EmitState &bodyState,
                                        EmitState &parentState) {
  mergeProtectedAccesses(parentState, bodyState);
  if (!bodyState.currentToken)
    return;
  auto yieldOp = getForYieldOp(forOp);
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    if (operand != bodyState.currentToken || idx >= forOp.getNumResults())
      continue;
    Value result = forOp.getResult(idx);
    if (!isa<AsyncTokenType>(result.getType()))
      continue;
    parentState.currentToken = result;
    parentState.currentSemaphore = bodyState.currentSemaphore;
    parentState.currentOwner = bodyState.currentOwner;
    parentState.currentBuffers.clear();
    parentState.knownCarrierTokens.insert(result);
    return;
  }
}

static scf::IfOp threadCarrierThroughIf(OpBuilder &b, scf::IfOp ifOp) {
  b.setInsertionPoint(ifOp);
  return replaceIfOpWithNewSignature(b, ifOp,
                                     TypeRange{b.getType<AsyncTokenType>()});
}

static void stampTokenYieldPartition(scf::YieldOp yieldOp, Value token,
                                     std::optional<PartitionId> owner) {
  SetVector<int> ids;
  if (hasPartition(yieldOp))
    ids = getPartitionIds(yieldOp);
  addPartitionIds(
      ids, partitionSetForTokenOrOwner(token, owner, yieldOp.getOperation()));
  if (!ids.empty())
    setPartition(yieldOp.getOperation(), ids);
}

static void appendTokenToYield(scf::YieldOp yieldOp, Value token,
                               std::optional<PartitionId> owner) {
  yieldOp->insertOperands(yieldOp.getNumOperands(), token);
  stampTokenYieldPartition(yieldOp, token, owner);
}

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src) {
  for (Operation *op : src.protectedAccesses)
    dst.protectedAccesses.insert(op);
  for (auto &kv : src.eventToken)
    if (!dst.eventToken.contains(kv.first))
      dst.eventToken[kv.first] = kv.second;
  for (auto &kv : src.eventSemaphore)
    if (!dst.eventSemaphore.contains(kv.first))
      dst.eventSemaphore[kv.first] = kv.second;
  for (auto &kv : src.eventBuffers)
    if (!dst.eventBuffers.contains(kv.first))
      dst.eventBuffers[kv.first] = kv.second;
  for (Value token : src.knownCarrierTokens)
    dst.knownCarrierTokens.insert(token);
  auto appendSync = [](auto &dstVec, const auto &srcVec) {
    for (const auto &record : srcVec) {
      bool seen = llvm::any_of(dstVec, [&](const auto &existing) {
        return existing.op == record.op;
      });
      if (!seen)
        dstVec.push_back(record);
    }
  };
  appendSync(dst.emittedAcquires, src.emittedAcquires);
  appendSync(dst.emittedReleases, src.emittedReleases);
  for (const EmittedBufferRecord &record : src.emittedBuffers) {
    bool seen = llvm::any_of(dst.emittedBuffers,
                             [&](const EmittedBufferRecord &existing) {
                               return existing.accessOp == record.accessOp &&
                                      existing.memberIdx == record.memberIdx &&
                                      existing.accessBuffer ==
                                          record.accessBuffer;
                             });
    if (!seen)
      dst.emittedBuffers.push_back(record);
  }
  for (const ThreadRecord &record : src.threadedTokens) {
    bool seen = llvm::any_of(dst.threadedTokens, [&](const ThreadRecord &existing) {
      return existing.op == record.op && existing.token == record.token;
    });
    if (!seen)
      dst.threadedTokens.push_back(record);
  }
  for (const PoisonTokenRecord &record :
       src.poisonTokenResultsAfterEmission) {
    bool seen = llvm::any_of(
        dst.poisonTokenResultsAfterEmission,
        [&](const PoisonTokenRecord &existing) {
          return existing.op == record.op;
        });
    if (!seen)
      dst.poisonTokenResultsAfterEmission.push_back(record);
  }
  for (Operation *op : src.eraseAfterEmission)
    dst.eraseAfterEmission.insert(op);
  for (auto &kv : src.reusedForCarrierSlots)
    dst.reusedForCarrierSlots[kv.first] = kv.second;
  for (auto &kv : src.reusedForTokenSlots)
    dst.reusedForTokenSlots[kv.first] = kv.second;
  for (auto &kv : src.reusedForPoisonTokens)
    dst.reusedForPoisonTokens[kv.first] = kv.second;
  for (auto &kv : src.stageCache)
    dst.stageCache[kv.first] = kv.second;
}

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

static SetVector<int> unionPartitionIds(Operation *lhs, Operation *rhs) {
  SetVector<int> ids;
  if (lhs && hasPartition(lhs))
    addPartitionIds(ids, getPartitionIds(lhs));
  if (rhs && hasPartition(rhs))
    addPartitionIds(ids, getPartitionIds(rhs));
  return ids;
}

static SetVector<int> subtractPartitionIds(const SetVector<int> &ids,
                                           const SetVector<int> &excluded) {
  SetVector<int> result;
  for (int id : ids)
    if (!llvm::is_contained(excluded, id))
      result.insert(id);
  return result;
}

static void assignStageIfKnown(OpBuilder &b, Operation *op,
                               StageCluster stageCluster) {
  if (stageCluster)
    setStageCluster(b, op, stageCluster);
}

static bool semaphoreUsesTmem(Value semaphore) {
  auto semaType = dyn_cast<SemaphoreType>(semaphore.getType());
  if (!semaType || semaType.getBaseType().empty())
    return false;
  auto memDescType = dyn_cast<MemDescType>(semaType.getBaseType().front());
  return memDescType &&
         memDescType.getMemorySpace() ==
             TensorMemorySpaceAttr::get(semaphore.getContext());
}

static unsigned semaphoreBaseTypeCount(Value semaphore) {
  auto semaType = dyn_cast<SemaphoreType>(semaphore.getType());
  return semaType ? semaType.getBaseType().size() : 0;
}

struct SemaphoreIfSplitCandidate {
  scf::IfOp ifOp;
  bool branchIsThen = true;
  SemaphoreReleaseOp releaseOp;
  SemaphoreAcquireOp acquireOp;
  unsigned tokenResultIdx = 0;
  bool releaseOnly = false;
};

static SemaphoreReleaseOp findBranchReleaseForSplit(Block *block) {
  if (!block)
    return nullptr;
  for (Operation &op : *block) {
    if (isa<scf::YieldOp>(op))
      return nullptr;
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(&op))
      return releaseOp;
    if (isa<SemaphoreAcquireOp>(op))
      return nullptr;
    if (op.hasTrait<OpTrait::ConstantLike>() || isSupportedAliasOp(&op))
      continue;
    return nullptr;
  }
  return nullptr;
}

static SemaphoreAcquireOp findBranchTrailingAcquire(Block *block) {
  if (!block || !block->getTerminator())
    return nullptr;
  Operation *lastOp = block->getTerminator()->getPrevNode();
  return dyn_cast_or_null<SemaphoreAcquireOp>(lastOp);
}

static bool branchHasAcquireAfter(SemaphoreReleaseOp releaseOp) {
  if (!releaseOp)
    return false;
  for (Operation *op = releaseOp->getNextNode(); op; op = op->getNextNode()) {
    if (isa<scf::YieldOp>(op))
      return false;
    if (isa<SemaphoreAcquireOp>(op))
      return true;
  }
  return false;
}

static StageCluster inferPrecedingMmaStage(scf::IfOp ifOp) {
  for (Operation *op = ifOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<MMAv5OpInterface>(op))
      return getStageCluster(op);
  return StageCluster{};
}

static void splitSemaphoreIfForLoopScheduler(triton::FuncOp funcOp) {
  SmallVector<SemaphoreIfSplitCandidate, 4> ifOps;
  funcOp.walk([&](scf::IfOp ifOp) {
    if (ifOp.thenBlock()->empty())
      return;
    auto makeReleaseOnlyCandidate = [&](bool branchIsThen)
        -> std::optional<SemaphoreIfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto releaseOp = findBranchReleaseForSplit(block);
      if (!releaseOp || !semaphoreUsesTmem(releaseOp.getSemaphore()) ||
          !branchHasAcquireAfter(releaseOp))
        return std::nullopt;
      return SemaphoreIfSplitCandidate{
          ifOp, branchIsThen, releaseOp, SemaphoreAcquireOp(), 0,
          /*releaseOnly=*/true};
    };
    auto makeCandidate = [&](bool branchIsThen)
        -> std::optional<SemaphoreIfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto releaseOp = findBranchReleaseForSplit(block);
      auto acquireOp = findBranchTrailingAcquire(block);
      if (!releaseOp || !acquireOp)
        return std::nullopt;
      if (semaphoreUsesTmem(releaseOp.getSemaphore()) &&
          semaphoreBaseTypeCount(releaseOp.getSemaphore()) > 1)
        return std::nullopt;
      scf::YieldOp yieldOp =
          branchIsThen ? ifOp.thenYield() : ifOp.elseYield();
      auto pos =
          findValuePosInRange(yieldOp->getOperands(), acquireOp.getToken());
      if (!pos)
        return std::nullopt;
      return SemaphoreIfSplitCandidate{
          ifOp, branchIsThen, releaseOp, acquireOp,
          static_cast<unsigned>(*pos), /*releaseOnly=*/false};
    };

    if (auto candidate = makeCandidate(/*branchIsThen=*/true)) {
      ifOps.push_back(*candidate);
      return;
    }
    if (auto candidate = makeCandidate(/*branchIsThen=*/false)) {
      ifOps.push_back(*candidate);
      return;
    }

    if (auto candidate = makeReleaseOnlyCandidate(/*branchIsThen=*/true)) {
      ifOps.push_back(*candidate);
      return;
    }
    if (auto candidate = makeReleaseOnlyCandidate(/*branchIsThen=*/false)) {
      ifOps.push_back(*candidate);
      return;
    }

    Operation *firstOp = &ifOp.thenBlock()->front();
    auto acquireOp = dyn_cast_or_null<SemaphoreAcquireOp>(firstOp);
    if (acquireOp) {
      Operation *prev = ifOp->getPrevNode();
      if (prev && ifOp.getCondition().getDefiningOp() == prev)
        prev = prev->getPrevNode();
      auto releaseOp = dyn_cast_or_null<SemaphoreReleaseOp>(prev);
      if (!releaseOp)
        return;
      scf::YieldOp yieldOp = ifOp.thenYield();
      auto pos =
          findValuePosInRange(yieldOp->getOperands(), acquireOp.getToken());
      if (!pos)
        return;
      ifOps.push_back(SemaphoreIfSplitCandidate{
          ifOp, /*branchIsThen=*/true, releaseOp, acquireOp,
          static_cast<unsigned>(*pos), /*releaseOnly=*/false});
    }
  });

  for (SemaphoreIfSplitCandidate candidate : ifOps) {
    scf::IfOp ifOp = candidate.ifOp;
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();

    b.setInsertionPoint(ifOp);
    auto exitIf = scf::IfOp::create(
        b, loc, TypeRange{}, ifOp.getCondition(),
        /*withElseRegion=*/!candidate.branchIsThen);
    Block *exitBlock =
        candidate.branchIsThen ? exitIf.thenBlock() : exitIf.elseBlock();
    candidate.releaseOp->moveBefore(exitBlock, exitBlock->begin());
    exitIf->setAttrs(ifOp->getAttrs());
    StageCluster releaseStage = getStageCluster(candidate.releaseOp);
    if (!releaseStage)
      releaseStage = inferPrecedingMmaStage(ifOp);
    assignStageIfKnown(b, candidate.releaseOp, releaseStage);
    assignStageIfKnown(b, exitIf, releaseStage);
    SetVector<int> exitIds;
    if (hasPartition(candidate.releaseOp.getOperation()))
      exitIds = getPartitionIds(candidate.releaseOp.getOperation());
    else if (hasPartition(ifOp))
      exitIds = getPartitionIds(ifOp);
    if (!exitIds.empty())
      setPartition(exitIf, exitIds);
    setPartitionOutputs(exitIf, {});
    if (candidate.releaseOnly)
      continue;

    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(b, loc, TypeRange{b.getType<AsyncTokenType>()},
                                     ifOp.getCondition(),
                                     /*withElseRegion=*/true);
    Block *enterAcquireBlock =
        candidate.branchIsThen ? enterIf.thenBlock() : enterIf.elseBlock();
    candidate.acquireOp->moveBefore(enterAcquireBlock,
                                    enterAcquireBlock->begin());

    ifOp.getResult(candidate.tokenResultIdx)
        .replaceAllUsesWith(enterIf.getResult(0));

    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(
        b, loc,
        candidate.branchIsThen
            ? candidate.acquireOp.getToken()
            : ifOp.thenYield().getOperand(candidate.tokenResultIdx));
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(
        b, loc,
        candidate.branchIsThen
            ? ifOp.elseYield().getOperand(candidate.tokenResultIdx)
            : candidate.acquireOp.getToken());

    b.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(b, loc, b.getType<AsyncTokenType>());
    ifOp.thenYield().setOperand(candidate.tokenResultIdx, poison);
    ifOp.elseYield().setOperand(candidate.tokenResultIdx, poison);

    enterIf->setAttrs(ifOp->getAttrs());
    StageCluster acquireStage = getStageCluster(candidate.acquireOp);
    if (!releaseStage)
      releaseStage = acquireStage;
    assignStageIfKnown(b, enterIf, acquireStage);

    SetVector<int> enterExitIds =
        unionPartitionIds(candidate.releaseOp.getOperation(),
                          candidate.acquireOp.getOperation());
    if (!enterExitIds.empty()) {
      setPartition(exitIf, enterExitIds);
      setPartition(enterIf, enterExitIds);
      setPartitionOutputs(exitIf, {});
      SmallVector<SetVector<int>, 1> enterOutputs{enterExitIds};
      setPartitionOutputs(enterIf, enterOutputs);
    }

    SetVector<int> middleIds;
    if (hasPartition(ifOp))
      middleIds = subtractPartitionIds(getPartitionIds(ifOp), enterExitIds);
    if (middleIds.empty() && ifOp.getNumResults() > 0)
      middleIds = partitionSetForValue(ifOp.getResult(0));
    if (!middleIds.empty()) {
      SetVector<int> ifIds = middleIds;
      SmallVector<SetVector<int>, 4> outputs;
      outputs.reserve(ifOp.getNumResults());
      for (Value result : ifOp.getResults()) {
        SetVector<int> resultIds = partitionSetForValue(result);
        if (resultIds.empty())
          resultIds = middleIds;
        addPartitionIds(ifIds, resultIds);
        outputs.push_back(resultIds);
      }
      setPartition(ifOp, ifIds);
      setPartitionOutputs(ifOp, outputs);
    }
  }
}

static bool isReleasedCreate(SemaphoreCreateOp createOp) {
  auto attr = dyn_cast_or_null<BoolAttr>(createOp->getAttr("is_released"));
  return attr && attr.getValue();
}

static bool isFirstAcquireAfterCreate(SemaphoreCreateOp createOp,
                                      SemaphoreAcquireOp acquireOp) {
  Operation *create = createOp.getOperation();
  Operation *acquire = acquireOp.getOperation();
  if (create->getBlock() != acquire->getBlock() ||
      !create->isBeforeInBlock(acquire))
    return false;
  Value semaphore = createOp.getResult();
  for (Operation *user : semaphore.getUsers()) {
    if (user == acquire || user->getBlock() != create->getBlock())
      continue;
    if (!isa<SemaphoreAcquireOp, SemaphoreReleaseOp>(user))
      continue;
    if (create->isBeforeInBlock(user) && user->isBeforeInBlock(acquire))
      return false;
  }
  return true;
}

static Operation *findEarliestTokenUserInBlock(Value token, Block *block) {
  Operation *earliest = nullptr;
  Operation *defOp = token.getDefiningOp();
  for (OpOperand &use : token.getUses()) {
    Operation *user = use.getOwner();
    Operation *ancestor = block ? block->findAncestorOpInBlock(*user) : user;
    if (!ancestor || ancestor == defOp)
      continue;
    if (!earliest || ancestor->isBeforeInBlock(earliest))
      earliest = ancestor;
  }
  return earliest;
}

static bool opBetweenFeedsTarget(Operation *first, Operation *last,
                                 Operation *target) {
  if (!first || !last || !target || first == target)
    return false;
  for (Operation *op = first; op && op != target; op = op->getNextNode()) {
    if (isa<SemaphoreCreateOp, SemaphoreAcquireOp, SemaphoreReleaseOp,
            SemaphoreBufferOp>(op))
      continue;
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (user == target)
          return true;
      }
    }
  }
  return false;
}

static void hoistInitialEmptyAcquires(triton::FuncOp funcOp) {
  SmallVector<SemaphoreAcquireOp, 8> acquires;
  funcOp.walk([&](SemaphoreAcquireOp acquireOp) {
    auto createOp =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (!createOp || !semaphoreUsesTmem(createOp.getResult()) ||
        !isReleasedCreate(createOp) ||
        !isFirstAcquireAfterCreate(createOp, acquireOp))
      return;
    acquires.push_back(acquireOp);
  });

  for (SemaphoreAcquireOp acquireOp : acquires) {
    auto createOp =
        acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
    if (!createOp || !isFirstAcquireAfterCreate(createOp, acquireOp))
      continue;
    Operation *insertAfter = createOp.getOperation();
    for (Operation *next = insertAfter->getNextNode();
         next && isa<SemaphoreCreateOp>(next); next = next->getNextNode())
      insertAfter = next;
    if (Operation *firstUser =
            findEarliestTokenUserInBlock(acquireOp.getToken(),
                                         acquireOp->getBlock())) {
      if (createOp->isBeforeInBlock(firstUser) &&
          firstUser != acquireOp.getOperation()) {
        Operation *afterCreates = insertAfter->getNextNode();
        if (opBetweenFeedsTarget(afterCreates, firstUser, firstUser)) {
          if (firstUser->getPrevNode() != acquireOp.getOperation())
            acquireOp->moveBefore(firstUser);
        } else if (insertAfter->getNextNode() != acquireOp.getOperation()) {
          acquireOp->moveAfter(insertAfter);
        }
      }
    } else if (insertAfter->getNextNode() != acquireOp.getOperation()) {
      acquireOp->moveAfter(insertAfter);
    }
  }
}

static Value findLatestSemaphoreCarrierInit(scf::ForOp forOp,
                                            ArrayRef<unsigned> slots) {
  if (slots.empty())
    return {};

  Value latestInit = forOp.getInitArgs()[slots.front()];
  Operation *latestAcquire = nullptr;
  for (unsigned slot : slots) {
    auto acquireOp = forOp.getInitArgs()[slot].getDefiningOp<SemaphoreAcquireOp>();
    if (!acquireOp || acquireOp->getBlock() != forOp->getBlock() ||
        !acquireOp->isBeforeInBlock(forOp))
      continue;
    if (!latestAcquire || latestAcquire->isBeforeInBlock(acquireOp)) {
      latestAcquire = acquireOp;
      latestInit = forOp.getInitArgs()[slot];
    }
  }
  return latestInit;
}

static void collectSemaphoreBackingsForToken(Value token,
                                             SetVector<Value> &backings,
                                             DenseSet<Value> &visited) {
  if (!token || !visited.insert(token).second)
    return;

  auto addSemaphoreBacking = [&](Value semaphore) {
    auto createOp = semaphore.getDefiningOp<SemaphoreCreateOp>();
    if (createOp && !createOp.getBuffers().empty())
      backings.insert(createOp.getBuffers().front());
  };

  for (Operation *user : token.getUsers()) {
    if (auto bufferOp = dyn_cast<SemaphoreBufferOp>(user)) {
      addSemaphoreBacking(bufferOp.getSemaphore());
      continue;
    }
    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user)) {
      addSemaphoreBacking(releaseOp.getSemaphore());
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned controlOperands = forOp.getNumControlOperands();
      for (OpOperand &operand : forOp->getOpOperands()) {
        if (operand.get() != token ||
            operand.getOperandNumber() < controlOperands)
          continue;
        unsigned slot = operand.getOperandNumber() - controlOperands;
        collectSemaphoreBackingsForToken(forOp.getRegionIterArg(slot),
                                         backings, visited);
      }
    }
  }
}

static std::optional<Value> inferSemaphoreBackingForCarrierSlot(scf::ForOp forOp,
                                                                unsigned slot) {
  SetVector<Value> backings;
  DenseSet<Value> visited;
  collectSemaphoreBackingsForToken(forOp.getRegionIterArg(slot), backings,
                                   visited);
  if (backings.size() != 1)
    return std::nullopt;
  return backings.front();
}

static bool isPoisonAsyncToken(Value value) {
  return value && isa<AsyncTokenType>(value.getType()) &&
         value.getDefiningOp<ub::PoisonOp>();
}

static void poisonDuplicateUnbackedTokenSlots(
    scf::ForOp forOp, ArrayRef<unsigned> asyncTokenSlots,
    const DenseMap<unsigned, Value> &backingBySlot) {
  llvm::MapVector<Value, SmallVector<unsigned, 4>> slotsByInit;
  for (unsigned slot : asyncTokenSlots)
    slotsByInit[forOp.getInitArgs()[slot]].push_back(slot);

  scf::YieldOp yieldOp = getForYieldOp(forOp);
  unsigned controlOperands = forOp.getNumControlOperands();
  Value poison;
  for (auto &it : slotsByInit) {
    ArrayRef<unsigned> slots = it.second;
    if (slots.size() < 2)
      continue;
    bool hasSemaphoreBackedSlot = false;
    for (unsigned slot : slots)
      hasSemaphoreBackedSlot |= backingBySlot.contains(slot);
    if (!hasSemaphoreBackedSlot)
      continue;

    for (unsigned slot : slots) {
      if (backingBySlot.contains(slot) ||
          !isPoisonAsyncToken(yieldOp.getOperand(slot)))
        continue;
      if (!poison) {
        OpBuilder b(forOp);
        b.setInsertionPoint(forOp);
        poison =
            ub::PoisonOp::create(b, forOp.getLoc(), b.getType<AsyncTokenType>());
      }
      forOp->setOperand(controlOperands + slot, poison);
      yieldOp.setOperand(slot, poison);
    }
  }
}

static void coalesceSemaphoreForCarriers(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp, 8> loops;
  funcOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
    llvm::MapVector<Value, SmallVector<unsigned, 4>> slotsByBacking;
    DenseMap<unsigned, Value> backingBySlot;
    SmallVector<unsigned, 4> asyncTokenSlots;
    for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
      if (!isa<AsyncTokenType>(init.getType()))
        continue;
      unsigned slot = static_cast<unsigned>(idx);
      asyncTokenSlots.push_back(slot);
      auto acquireOp = init.getDefiningOp<SemaphoreAcquireOp>();
      if (acquireOp) {
        auto createOp =
            acquireOp.getSemaphore().getDefiningOp<SemaphoreCreateOp>();
        if (createOp && !createOp.getBuffers().empty()) {
          Value backing = createOp.getBuffers().front();
          backingBySlot[slot] = backing;
          slotsByBacking[backing].push_back(slot);
          continue;
        }
      }
      if (std::optional<Value> backing = inferSemaphoreBackingForCarrierSlot(
              forOp, slot)) {
        backingBySlot[slot] = *backing;
        slotsByBacking[*backing].push_back(slot);
      }
    }

    poisonDuplicateUnbackedTokenSlots(forOp, asyncTokenSlots, backingBySlot);

    for (auto &it : slotsByBacking) {
      ArrayRef<unsigned> semaphoreTokenSlots = it.second;
      if (semaphoreTokenSlots.size() < 2)
        continue;

      Value carrierInit =
          findLatestSemaphoreCarrierInit(forOp, semaphoreTokenSlots);
      if (!carrierInit)
        continue;

      unsigned canonicalSlot = semaphoreTokenSlots.front();
      Value canonicalIterArg = forOp.getRegionIterArg(canonicalSlot);
      Value canonicalResult = forOp.getResult(canonicalSlot);
      scf::YieldOp yieldOp = getForYieldOp(forOp);
      Value canonicalYield = yieldOp.getOperand(canonicalSlot);
      unsigned controlOperands = forOp.getNumControlOperands();
      forOp->setOperand(controlOperands + canonicalSlot, carrierInit);
      for (unsigned slot : semaphoreTokenSlots) {
        if (slot == canonicalSlot)
          continue;
        forOp.getRegionIterArg(slot).replaceAllUsesWith(canonicalIterArg);
        forOp.getResult(slot).replaceAllUsesWith(canonicalResult);
        yieldOp.setOperand(slot, canonicalYield);
        forOp->setOperand(controlOperands + slot, carrierInit);
      }
    }
  }
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMITTER_H_
