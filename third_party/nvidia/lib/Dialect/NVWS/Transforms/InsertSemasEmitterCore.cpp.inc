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
