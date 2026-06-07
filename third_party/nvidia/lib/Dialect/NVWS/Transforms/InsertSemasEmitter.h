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

struct ActiveSemaphoreState {
  Value token, semaphore;
  std::optional<PartitionId> owner;
  std::optional<AsyncOp> payload;
  SmallVector<Value, 4> buffers;

  bool canReuseBuffer(Value sem, Value tok, unsigned bufferCount) const {
    return token == tok && semaphore == sem && buffers.size() == bufferCount;
  }
  void clearBuffers() { buffers.clear(); }
  template <typename RangeT> void cacheBuffers(RangeT newBuffers) {
    buffers.assign(newBuffers.begin(), newBuffers.end());
  }
  void setCarrier(Value sem, Value tok, std::optional<PartitionId> newOwner) {
    semaphore = sem;
    token = tok;
    owner = newOwner;
    payload = std::nullopt;
    clearBuffers();
  }
  void setAfterAccess(Value sem, Value tok, std::optional<PartitionId> newOwner,
                      AsyncOp newPayload) {
    semaphore = sem;
    token = tok;
    owner = newOwner;
    payload = newPayload;
  }
  void setToken(Value tok) { token = tok; clearBuffers(); }
  void setTokenOwner(Value tok, std::optional<PartitionId> newOwner) {
    token = tok; owner = newOwner; clearBuffers();
  }
  void setSemaphore(Value sem) { semaphore = sem; clearBuffers(); }
};

struct EmitState {
  ResourceSemaphores semas;
  DenseMap<Operation *, Value> eventToken;
  DenseMap<Value, Value> rewrittenAccessValue;
  DenseMap<Operation *, unsigned> reusedForCarrierSlots;
  DenseMap<Operation *, SmallVector<unsigned, 4>> reusedForTokenSlots;
  DenseMap<Operation *, Value> reusedForPoisonTokens;
  const EmitterTransitionPlan *transitions = nullptr;
  ActiveSemaphoreState current;
  SmallVector<EmittedSyncRecord, 8> emittedReleases;
  SmallVector<PoisonTokenRecord, 8> poisonTokenResultsAfterEmission;
  SetVector<Operation *> eraseAfterEmission;
  DenseMap<PartitionId, StageCluster> stageCache;

  AcquireRecord acquireForGroup(OpBuilder &b, Location loc, SyncAnchorKind kind,
                                Operation *anchor, Region *yieldRegion,
                                unsigned groupIdx, const OptSyncDag &dag,
                                const SyncPlan &sp, BufferGroup &group,
                                StageCluster stageCluster);
  LogicalResult release(OpBuilder &b, Location loc, SyncAnchorKind kind,
                        Operation *anchor, Region *yieldRegion,
                        const PlannedRelease &action, const OptSyncDag &dag,
                        const SyncPlan &sp, BufferGroup &group,
                        StageCluster stageCluster, Operation *liveAnchor = nullptr);
  SmallVector<Value, 4>
  getBuffers(OpBuilder &b, Operation *op, Value sem, Value token,
             std::optional<PartitionId> owner, bool freshAcquire,
             BufferGroup &group, const GroupBacking &backing,
             ArrayRef<const AccessTouch *> touches, bool writes);
  const EmitterTransitionPlan &transitionPlan() const { return *transitions; }
};

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
  ensureClass(semas.seedId);
  return semas;
}

static Value getSemaphoreForGroup(unsigned groupIdx, const SyncEdge *edge,
                                  const OptSyncDag &dag, const SyncPlan &sp,
                                  BufferGroup &group,
                                  ResourceSemaphores &semas) {
  if (dag.groups[groupIdx].kind == SyncGroupKind::InitialEmpty || !edge)
    return semas.forClass(sp, std::nullopt);
  return semas.forClass(sp, findEdgeIndex(sp, edge));
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

static bool semaphoreUsesTmem(Value semaphore) {
  auto semaType = dyn_cast<SemaphoreType>(semaphore.getType());
  if (!semaType || semaType.getBaseType().empty())
    return false;
  auto memDescType = dyn_cast<MemDescType>(semaType.getBaseType().front());
  return memDescType &&
         memDescType.getMemorySpace() ==
             TensorMemorySpaceAttr::get(semaphore.getContext());
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

SmallVector<Value, 4>
EmitState::getBuffers(OpBuilder &b, Operation *op, Value sem, Value token,
                      std::optional<PartitionId> owner, bool freshAcquire,
                      BufferGroup &group, const GroupBacking &backing,
                      ArrayRef<const AccessTouch *> touches, bool writes) {
  StageCluster stageCluster = getStageCluster(op);
  bool canReuse =
      !freshAcquire && current.canReuseBuffer(sem, token, backing.bufferTypes.size());
  if (canReuse)
    return SmallVector<Value, 4>(current.buffers.begin(), current.buffers.end());
  SemaphoreBufferOp bufferOp =
      emitSemaphoreBuffer(b, op->getLoc(), sem, token, owner, stageCluster, group,
                          backing, touches, writes);
  if (!owner)
    setPartitionFromAnchor(bufferOp.getOperation(), op);
  if (!owner)
    setPartitionFromTokenIfParentPartitioned(bufferOp.getOperation(), token);
  SmallVector<Value, 4> buffers(bufferOp.getBuffers().begin(),
                                bufferOp.getBuffers().end());
  current.cacheBuffers(ArrayRef<Value>(buffers));
  return buffers;
}

static LogicalResult emitAccessTransition(OpBuilder &b, AccessEvent &event,
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
  } else if (state.current.token && state.current.semaphore) {
    sem = state.current.semaphore;
    token = state.current.token;
  } else if (writes) {
    return op->emitError("nvws-insert-semas: missing planned EMPTY/FULL carrier "
                         "token for writer");
  } else {
    return success();
  }

  SmallVector<Value, 4> buffers =
      state.getBuffers(b, op, sem, token, event.owner, !acquires.empty(), group,
                       backing, touches, writes);
  Operation *bufferOperation = getBufferDefiningOp(buffers);
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
  state.current.setAfterAccess(sem, token, event.owner, getAsyncPayload(retargetOp));
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
      if (state.current.token &&
          valueScopeCanReachBlock(state.current.token, insertBlock))
        return state.current.token;
    }
  }
  if (state.current.token &&
      (!insertBlock || valueScopeCanReachBlock(state.current.token, insertBlock)))
    return state.current.token;
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

static std::optional<PartitionId>
computeReleaseOwner(const SyncGroup &syncGroup, const SyncPlan &sp,
                    const SyncEdge *edge, Operation *anchor, SyncAnchorKind kind,
                    bool terminalDstReadRelease,
                    bool terminalLoopExitReadRelease,
                    std::optional<PartitionId> carriedOwner) {
  std::optional<PartitionId> owner = edge ? edge->srcOwner : std::nullopt;
  if (terminalDstReadRelease || terminalLoopExitReadRelease)
    owner = edge->dstOwner;
  if (terminalLoopExitReadRelease)
    owner = getPartitionId(anchor);
  if (!owner && kind == SyncAnchorKind::ReleaseBeforeOp && carriedOwner)
    owner = carriedOwner;
  if (edge && !edge->srcOwner && kind == SyncAnchorKind::ReleaseBeforeOp &&
      syncGroup.kind == SyncGroupKind::LinearChain &&
      syncGroup.edgeIdxs.size() > 1 &&
      &sp.edges[syncGroup.edgeIdxs.front()] == edge)
    owner = sp.edges[syncGroup.edgeIdxs[1]].dstOwner;
  return owner;
}

static AsyncOp computeReleasePayload(
    const SyncGroup &syncGroup, const SyncPlan &sp, const SyncEdge *edge,
    Operation *anchor, SyncAnchorKind kind, bool terminalDstReadRelease,
    bool terminalLoopExitReadRelease, BufferGroup &group, int64_t resourceKey,
    std::optional<AsyncOp> carriedPayload) {
  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  AsyncOp payload =
      (terminalDstReadRelease || terminalLoopExitReadRelease)
          ? getAsyncPayload(anchor)
          : (edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
  if ((terminalDstReadRelease || terminalLoopExitReadRelease) && carriedPayload)
    payload = *carriedPayload;
  if (group.isTmem() && edge && edge->srcYieldRegion &&
      !terminalDstReadRelease && !terminalLoopExitReadRelease &&
      payload == AsyncOp::NONE)
    if (const AccessEvent *producer =
            findLastProducerInRegion(edge->srcYieldRegion, group, resourceKey))
      if (sameOwner(producer->owner, edge->srcOwner))
        payload = getAsyncPayload(producer->op);
  if (shouldForceNonePayload(syncGroup, sp, edge, kind))
    payload = AsyncOp::NONE;
  return payload;
}

LogicalResult EmitState::release(OpBuilder &b, Location loc, SyncAnchorKind kind,
                                 Operation *anchor, Region *yieldRegion,
                                 const PlannedRelease &action,
                                 const OptSyncDag &dag, const SyncPlan &sp,
                                 BufferGroup &group, StageCluster stageCluster,
                                 Operation *liveAnchor) {
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
  for (const EmittedSyncRecord &record : emittedReleases)
    if (record.groupIdx == groupIdx && record.kind == kind &&
        record.anchor == anchor && record.yieldRegion == yieldRegion &&
        record.edgeIdxs == action.edgeIdxs)
      return success();
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, semas);
  bool terminalDstReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->dstOp == anchor &&
      edge->srcOp != anchor;
  bool terminalLoopExitReadRelease =
      syncGroup.kind == SyncGroupKind::Singleton &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && anchor &&
      edgeIdx && dag.tmemLoopExitRead.lookup(*edgeIdx) == anchor;
  bool useStructuredCarrier =
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
      edge->srcOp != anchor && current.token;
  useStructuredCarrier |= kind == SyncAnchorKind::ReleaseBeforeOp && edge &&
                          edge->srcOp && edge->srcOp != anchor && current.token;
  SetVector<int> structuredCarrierPartition;
  if (useStructuredCarrier)
    structuredCarrierPartition = partitionSetForValue(current.token);
  FailureOr<Value> token =
      useStructuredCarrier ? FailureOr<Value>(current.token)
                           : lookupReleaseToken(loc, edge, *this,
                                                b.getInsertionBlock());
  if (failed(token))
    return failure();
  std::optional<PartitionId> owner = computeReleaseOwner(
      syncGroup, sp, edge, anchor, kind, terminalDstReadRelease,
      terminalLoopExitReadRelease, current.owner);
  AsyncOp payload = computeReleasePayload(
      syncGroup, sp, edge, anchor, kind, terminalDstReadRelease,
      terminalLoopExitReadRelease, group, dag.resource.second, current.payload);
  SemaphoreReleaseOp release =
      emitRelease(b, loc, sem, *token, owner, stageCluster, payload);
  if (useStructuredCarrier && structuredCarrierPartition.size() == 1 && !owner) {
    setPartition(release.getOperation(), structuredCarrierPartition);
    if (auto tag = wsTagForValue(current.token))
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
  emittedReleases.push_back(
      EmittedSyncRecord{groupIdx, kind, anchor, yieldRegion});
  emittedReleases.back().edgeIdxs.append(action.edgeIdxs.begin(),
                                         action.edgeIdxs.end());
  return success();
}

static bool hasInterveningLiveTmemAllocCarrier(Operation *first,
                                               Operation *target,
                                               BufferGroup &group) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(target);
  if (!first || !forOp)
    return false;
  auto initArgs = forOp.getInitArgs();
  for (Operation *op = first; op && op != target; op = op->getNextNode()) {
    auto allocOp = dyn_cast<TMEMAllocOp>(op);
    if (!allocOp || allocOp.getSrc() || !allocOp.getToken())
      continue;
    bool isCurrentBackingMember = llvm::any_of(group.members,
                                               [&](const BufferMember &member) {
                                                 return member.allocOp == op;
                                               });
    if (!isCurrentBackingMember && llvm::is_contained(initArgs, allocOp.getToken()))
      return true;
  }
  return false;
}

AcquireRecord EmitState::acquireForGroup(OpBuilder &b, Location loc,
                                         SyncAnchorKind kind, Operation *anchor,
                                         Region *yieldRegion, unsigned groupIdx,
                                         const OptSyncDag &dag,
                                         const SyncPlan &sp, BufferGroup &group,
                                         StageCluster stageCluster) {
  const SyncGroup &syncGroup = dag.groups[groupIdx];
  const SyncEdge *edge =
      findEdgeForAnchor(syncGroup, sp, dag, kind, anchor, yieldRegion);
  Value sem = getSemaphoreForGroup(groupIdx, edge, dag, sp, group, semas);
  std::optional<PartitionId> owner =
      edge ? edge->dstOwner : syncGroup.initialOwner;
  if (syncGroup.kind == SyncGroupKind::InitialEmpty && anchor) {
    auto createOp = sem.getDefiningOp<SemaphoreCreateOp>();
    if (createOp && semaphoreUsesTmem(createOp.getResult()) &&
        createOp->getBlock() == anchor->getBlock() &&
        createOp->isBeforeInBlock(anchor)) {
      Operation *insertAfter = createOp.getOperation();
      for (Operation *next = insertAfter->getNextNode();
           next && isa<SemaphoreCreateOp>(next); next = next->getNextNode())
        insertAfter = next;
      bool tokenUserIsAnchor =
          (isa<scf::ForOp>(anchor) && dag.threadForOps.contains(anchor)) ||
          (isa<scf::IfOp>(anchor) && dag.threadIfOps.contains(anchor));
      if (!tokenUserIsAnchor ||
          !hasInterveningLiveTmemAllocCarrier(insertAfter->getNextNode(),
                                              anchor, group))
        b.setInsertionPointAfter(insertAfter);
    }
  }
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
  current.setCarrier(sem, token, owner);
  return AcquireRecord{sem, token, owner};
}

static LogicalResult emitOperationEntryTransitions(
    Operation *anchor, const OptSyncDag &dag, const SyncPlan &sp,
    BufferGroup &group, EmitState &state, SmallVectorImpl<AcquireRecord> &acquires) {
  const EmitterTransitionPlan &transitions = state.transitionPlan();
  auto rIt = transitions.opEntryReleases.find(anchor);
  if (rIt != transitions.opEntryReleases.end())
    for (const PlannedRelease &release : rIt->second) {
      OpBuilder releaseBuilder(anchor);
      if (Operation *insertBefore =
              transitions.opEntryReleaseInsertBefore.lookup(anchor))
        releaseBuilder.setInsertionPoint(insertBefore);
      else if (Operation *insertAfter =
                   transitions.opEntryReleaseInsertAfter.lookup(anchor))
        releaseBuilder.setInsertionPointAfter(insertAfter);
      else
        releaseBuilder.setInsertionPoint(anchor);
      if (failed(state.release(
              releaseBuilder, anchor->getLoc(), SyncAnchorKind::ReleaseBeforeOp,
              anchor, nullptr, release, dag, sp, group, getStageCluster(anchor))))
        return failure();
    }
  OpBuilder b(anchor);
  b.setInsertionPoint(anchor);
  auto aIt = transitions.opEntryAcquires.find(anchor);
  if (aIt != transitions.opEntryAcquires.end())
    for (unsigned gi : aIt->second)
      acquires.push_back(state.acquireForGroup(
          b, anchor->getLoc(), SyncAnchorKind::AcquireBeforeOp, anchor, nullptr,
          gi, dag, sp, group, getStageCluster(anchor)));
  return success();
}

static LogicalResult emitOperationExitTransitions(
    Operation *anchor, Operation *insertAfter, const OptSyncDag &dag,
    const SyncPlan &sp, BufferGroup &group, EmitState &state) {
  const EmitterTransitionPlan &transitions = state.transitionPlan();
  auto rIt = transitions.opExitReleases.find(anchor);
  if (rIt == transitions.opExitReleases.end()) return success();
  Operation *releaseAfter = insertAfter;
  if (Operation *plannedAfter =
          transitions.opExitReleaseInsertAfter.lookup(anchor))
    releaseAfter = plannedAfter == anchor ? insertAfter : plannedAfter;
  if (anchor == insertAfter && operationIsAttached(releaseAfter) &&
      operationIsAttached(anchor) &&
      releaseAfter->isProperAncestor(anchor))
    return success();
  OpBuilder b(releaseAfter);
  if (Operation *insertBefore =
          transitions.opExitReleaseInsertBefore.lookup(anchor)) {
    b.setInsertionPoint(insertBefore);
  } else if (isa<scf::YieldOp>(releaseAfter)) {
    b.setInsertionPoint(releaseAfter);
  } else {
    b.setInsertionPointAfter(releaseAfter);
  }
  std::optional<Value> savedEventToken;
  bool overrideEventToken = releaseAfter != anchor && state.current.token;
  if (overrideEventToken) {
    auto it = state.eventToken.find(anchor);
    if (it != state.eventToken.end())
      savedEventToken = it->second;
    state.eventToken[anchor] = state.current.token;
  }
  LogicalResult result = success();
  for (const PlannedRelease &release : rIt->second)
    if (failed(state.release(
            b, insertAfter->getLoc(), SyncAnchorKind::ReleaseAfterOp, anchor,
            nullptr, release, dag, sp, group, getStageCluster(insertAfter),
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

static LogicalResult emitOperationExitTransitions(
    Operation *anchor, const OptSyncDag &dag, const SyncPlan &sp,
    BufferGroup &group, EmitState &state) {
  return emitOperationExitTransitions(anchor, anchor, dag, sp, group, state);
}

static LogicalResult emitDeferredNestedExitTransitions(
    Operation *releaseAfter, const OptSyncDag &dag, const SyncPlan &sp,
    BufferGroup &group, EmitState &state) {
  const EmitterTransitionPlan &transitions = state.transitionPlan();
  auto deferredIt = transitions.deferredOpExitAnchors.find(releaseAfter);
  if (deferredIt == transitions.deferredOpExitAnchors.end())
    return success();
  for (Operation *anchor : deferredIt->second)
    if (failed(emitOperationExitTransitions(anchor, releaseAfter, dag, sp,
                                            group, state)))
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
                                  const EmitterTransitionPlan &transitions) {
  if (!anchor)
    return nullptr;
  auto releaseIt = transitions.opExitReleases.find(anchor);
  if (releaseIt == transitions.opExitReleases.end())
    return nullptr;
  for (const PlannedRelease &action : releaseIt->second)
    if (action.groupIdx == groupIdx)
      return &action;
  return nullptr;
}

static bool linearChainNeedsLoopExitDrain(unsigned groupIdx,
                                          const SyncGroup &syncGroup, Operation *forOp, Region *region,
                                          const EmitterTransitionPlan &transitions,
                                          const OptSyncDag &dag,
                                          const SyncPlan &sp) {
  if (!forOp || !region)
    return false;
  if (findPlannedAfterOpReleaseForGroup(forOp, groupIdx, transitions))
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
  if (!group.isTmem() || !state.current.token)
    return success();
  if (!hasWarpSpecializeTag(forOp))
    return success();
  const EmitterTransitionPlan &transitions = state.transitionPlan();
  SmallVector<unsigned, 2> groupIds;
  auto it = transitions.regionEntryAcquires.find(region);
  if (it != transitions.regionEntryAcquires.end())
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
                                       region, transitions, dag, sp))
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
      auto releaseIt = transitions.opExitReleases.find(forOp.getOperation());
      if (releaseIt == transitions.opExitReleases.end())
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
        if (failed(emitDrainRelease(sem, state.current.token, state.current.owner,
                                    edge.asyncPayload, edge)))
          return failure();
        state.current.setSemaphore(sem);
        return success();
      }
    }
    std::optional<PartitionId> drainOwner = firstEdge.dstOwner;
    std::optional<PartitionId> releaseOwner = state.current.owner;
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
    Value fullSem = state.semas.forClass(sp, findEdgeIndex(sp, fullReleaseEdge));
    Value emptySem = state.semas.forClass(sp, findEdgeIndex(sp, &secondEdge));
    if (const PlannedRelease *releaseAction =
            findPlannedAfterLoopRelease(*fullReleaseEdge)) {
      if (failed(state.release(b, loc, SyncAnchorKind::ReleaseAfterOp,
                               forOp.getOperation(), nullptr, *releaseAction, dag,
                               sp, group, StageCluster{})))
        return failure();
    } else if (failed(emitDrainRelease(fullSem, state.current.token,
                                       releaseOwner, drainPayload,
                                       *fullReleaseEdge))) {
      return failure();
    }
    SemaphoreAcquireOp acquire =
        emitAcquire(b, loc, fullSem, drainOwner, StageCluster{});
    if (failed(emitDrainRelease(emptySem, acquire.getToken(), drainOwner,
                                emptyPayload, secondEdge)))
      return failure();
    state.current.setCarrier(emptySem, acquire.getToken(), drainOwner);
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
                                         const EmitterTransitionPlan &transitions,
                                         const OptSyncDag &dag,
                                         BufferGroup &group,
                                         SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!isa<scf::ForOp>(anchor))
    return false;
  if (!transitions.opEntryAcquires.contains(anchor) ||
      transitions.opExitReleases.contains(anchor) ||
      transitions.regionEntryAcquires.contains(&region))
    return false;
  return collectFirstReadOnlyRegionAccess(region, dag, group, touches);
}

static bool canPrebufferLocalIfEntry(scf::IfOp ifOp,
                                     ArrayRef<AcquireRecord> acquires,
                                     const EmitterTransitionPlan &transitions,
                                     const OptSyncDag &dag, BufferGroup &group,
                                     SmallVectorImpl<const AccessTouch *> &touches) {
  if (group.isTmem() || acquires.size() != 1)
    return false;
  if (!transitions.opEntryAcquires.contains(ifOp.getOperation()) ||
      transitions.opExitReleases.contains(ifOp.getOperation()) ||
      transitions.regionEntryAcquires.contains(&ifOp.getThenRegion()) ||
      transitions.regionEntryAcquires.contains(&ifOp.getElseRegion()))
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
  state.current.cacheBuffers(bufferOp.getBuffers());
}

static LogicalResult emitRegionCloseTransitions(
    Operation *yieldOp, Region *region, const OptSyncDag &dag,
    const SyncPlan &sp, BufferGroup &group, EmitState &state,
    SmallVectorImpl<AcquireRecord> &acquires) {
  const EmitterTransitionPlan &transitions = state.transitionPlan();
  OpBuilder b(yieldOp);
  b.setInsertionPoint(yieldOp);
  auto rIt = transitions.regionEntryReleases.find(region);
  if (rIt != transitions.regionEntryReleases.end())
    for (const PlannedRelease &release : rIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      OpBuilder releaseBuilder(yieldOp);
      if (Operation *insertBefore =
              transitions.regionEntryReleaseInsertBefore.lookup(region))
        releaseBuilder.setInsertionPoint(insertBefore);
      else if (Operation *insertAfter =
                   transitions.regionEntryReleaseInsertAfter.lookup(region))
        releaseBuilder.setInsertionPointAfter(insertAfter);
      else
        releaseBuilder.setInsertionPoint(yieldOp);
      if (failed(state.release(
              releaseBuilder, yieldOp->getLoc(),
              SyncAnchorKind::ReleaseBeforeYield, nullptr, region, release, dag, sp,
              group,
              stageForYieldOwner(edge ? edge->srcOwner : std::nullopt, state))))
        return failure();
    }
  auto aIt = transitions.regionEntryAcquires.find(region);
  if (aIt != transitions.regionEntryAcquires.end())
    for (unsigned gi : aIt->second) {
      const SyncEdge *edge =
          findEdgeForAnchor(dag.groups[gi], sp,
                            dag,
                            SyncAnchorKind::AcquireBeforeYield, nullptr, region);
      acquires.push_back(state.acquireForGroup(
          b, yieldOp->getLoc(), SyncAnchorKind::AcquireBeforeYield, nullptr,
          region, gi, dag, sp, group,
          stageForYieldOwner(edge ? edge->dstOwner : std::nullopt, state)));
    }
  auto arIt = transitions.regionExitReleases.find(region);
  if (arIt != transitions.regionExitReleases.end())
    for (const PlannedRelease &release : arIt->second) {
      const SyncEdge *edge = getRepresentativeReleaseEdge(release, sp);
      if (failed(state.release(
              b, yieldOp->getLoc(), SyncAnchorKind::ReleaseAfterYield, nullptr,
              region, release, dag, sp, group,
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

static void mergeProtectedAccesses(EmitState &dst, const EmitState &src);

static FailureOr<scf::ForOp> threadCarrierThroughFor(OpBuilder &b,
                                                     scf::ForOp forOp,
                                                     EmitState &state,
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
  Value init = state.current.token;
  if (!init && !reusableSlots.empty())
    init = forOp->getOperand(3 + reusableSlots.front());
  if (!init) {
    forOp.emitError("nvws-insert-semas: planned scf.for carrier threading has "
                    "no token producer at loop entry");
    return failure();
  }
  SetVector<int> carrierPartition =
      partitionSetForTokenOrOwner(init, state.current.owner, forOp.getOperation());
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

    state.current.setToken(forOp.getRegionIterArg(carrierSlot));
    state.reusedForCarrierSlots[forOp.getOperation()] = carrierSlot;
    state.reusedForTokenSlots[forOp.getOperation()] = reusableSlots;
    if (poison)
      state.reusedForPoisonTokens[forOp.getOperation()] = poison;
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
  state.current.setToken(newFor.getRegionIterArg(oldNumResults));
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
                                bool overrideOwnerAtYield) {
  auto reusedIt = bodyState.reusedForCarrierSlots.find(forOp.getOperation());
  if (reusedIt != bodyState.reusedForCarrierSlots.end()) {
    unsigned slot = reusedIt->second;
    Value yieldedToken = bodyState.current.token
                             ? bodyState.current.token
                             : forOp.getRegionIterArg(slot);
    std::optional<PartitionId> resultOwner =
        overrideOwnerAtYield ? ownerAtYield
                             : (ownerAtYield ? ownerAtYield
                                             : bodyState.current.owner);
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
    parentState.current.setTokenOwner(forOp.getResult(slot), resultOwner);
    return;
  }

  Value yieldedToken = bodyState.current.token
                           ? bodyState.current.token
                           : forOp.getRegionIterArg(forOp.getNumResults() - 1);
  std::optional<PartitionId> resultOwner =
      overrideOwnerAtYield ? ownerAtYield
                           : (ownerAtYield ? ownerAtYield
                                           : bodyState.current.owner);
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
  parentState.current.setTokenOwner(forOp.getResult(forOp.getNumResults() - 1),
                                    resultOwner);
}

static void closeExistingCarrierForLoop(scf::ForOp forOp, EmitState &bodyState,
                                        EmitState &parentState) {
  mergeProtectedAccesses(parentState, bodyState);
  if (!bodyState.current.token)
    return;
  auto yieldOp = getForYieldOp(forOp);
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    if (operand != bodyState.current.token || idx >= forOp.getNumResults())
      continue;
    Value result = forOp.getResult(idx);
    if (!isa<AsyncTokenType>(result.getType()))
      continue;
    parentState.current.setCarrier(bodyState.current.semaphore, result,
                                   bodyState.current.owner);
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
  for (auto &kv : src.eventToken)
    if (!dst.eventToken.contains(kv.first))
      dst.eventToken[kv.first] = kv.second;
  for (const EmittedSyncRecord &record : src.emittedReleases) {
    bool seen = llvm::any_of(dst.emittedReleases,
                             [&](const EmittedSyncRecord &existing) {
                               return existing.groupIdx == record.groupIdx &&
                                      existing.kind == record.kind &&
                                      existing.anchor == record.anchor &&
                                      existing.yieldRegion ==
                                          record.yieldRegion &&
                                      existing.edgeIdxs == record.edgeIdxs;
                             });
    if (!seen)
      dst.emittedReleases.push_back(record);
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

static LogicalResult emitResourceBlock(Block &block, const OptSyncDag &dag,
                                       const SyncPlan &sp, const ResourcePlan &plan,
                                       BufferGroup &group, const GroupBacking &backing,
                                       EmitState &state, Region *plannedRegion);

static LogicalResult emitResourceRegion(Region &region, const OptSyncDag &dag,
                                        const SyncPlan &sp, const ResourcePlan &plan,
                                        BufferGroup &group, const GroupBacking &backing,
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
                                       const SyncPlan &sp, const ResourcePlan &plan,
                                       BufferGroup &group, const GroupBacking &backing,
                                       EmitState &state, Region *plannedRegion) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<scf::YieldOp>(op)) {
      SmallVector<AcquireRecord, 2> yieldAcquires;
      if (failed(emitRegionCloseTransitions(&op, plannedRegion, dag, sp,
                                            group, state,
                                            yieldAcquires)))
        return failure();
      continue;
    }

    AccessEvent *event = nullptr;
    SmallVector<const AccessTouch *, 4> touches;
    if (dag.accessOps.contains(&op)) {
      event = findEvent(group, &op);
      if (event)
        collectTouchesForResource(*event, dag.resource.second, touches);
      if (event && event->owner)
        state.stageCache[*event->owner] = getStageCluster(&op);
    }

    SmallVector<AcquireRecord, 2> acquires;
    if (failed(emitOperationEntryTransitions(&op, dag, sp, group, state,
                                             acquires)))
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
                          state.transitionPlan(), dag, group, prebufferTouches)) {
        OpBuilder prebufferBuilder(forOp);
        prebufferLocalRegionEntry(prebufferBuilder, forOp.getOperation(),
                                  prebufferTouches, acquires.front(), group,
                                  backing, state);
        bodyState = state;
        threaded = false;
      }
      OpBuilder loopBuilder(forOp);
      if (threaded) {
        FailureOr<scf::ForOp> threadedFor =
            threadCarrierThroughFor(loopBuilder, forOp, bodyState, group,
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
        std::optional<PartitionId> ownerAtYield = bodyState.current.owner;
        bool overrideOwnerAtYield = false;
        auto regionOwnerIt = plan.regionOwners.find(plannedForRegion);
        if (regionOwnerIt != plan.regionOwners.end() &&
            (!linearChainEntersFor(oldForOp, dag, sp) ||
             state.transitionPlan().regionEntryAcquires.contains(plannedForRegion))) {
          ownerAtYield = regionOwnerIt->second.exit;
          overrideOwnerAtYield = true;
        }
        closeCarrierForLoop(activeForOp, bodyState, state, ownerAtYield,
                            overrideOwnerAtYield);
        if (failed(emitTmemLinearLoopExitDrain(activeForOp, plannedForRegion,
                                               dag, sp, group, state)))
          return failure();
      } else {
        closeExistingCarrierForLoop(activeForOp, bodyState, state);
      }
      if (failed(emitOperationExitTransitions(oldForOp, activeForOp.getOperation(),
                                              dag, sp, group, state)))
        return failure();
      if (failed(emitDeferredNestedExitTransitions(activeForOp.getOperation(),
                                                   dag, sp, group, state)))
        return failure();
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      Operation *oldIfOp = ifOp.getOperation();
      Region *plannedThenRegion = &ifOp.getThenRegion();
      Region *plannedElseRegion = &ifOp.getElseRegion();
      SmallVector<const AccessTouch *, 4> prebufferTouches;
      if (canPrebufferLocalIfEntry(ifOp, acquires, state.transitionPlan(), dag,
                                   group, prebufferTouches)) {
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
      Value incomingToken = state.current.token;
      Value incomingSemaphore = state.current.semaphore;
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
            thenState.current.token ? thenState.current.token : incomingToken;
        Value elseToken =
            elseState.current.token ? elseState.current.token : incomingToken;
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
                                   thenState.current.owner);
          stampTokenYieldPartition(activeIfOp.elseYield(), elseToken,
                                   elseState.current.owner);
        } else {
          appendTokenToYield(activeIfOp.thenYield(), thenToken,
                             thenState.current.owner);
          appendTokenToYield(activeIfOp.elseYield(), elseToken,
                             elseState.current.owner);
        }
        std::optional<PartitionId> outOwner =
            thenState.current.owner == elseState.current.owner
                ? thenState.current.owner
                : std::nullopt;
        SetVector<int> outPartition = partitionSetForOwner(outOwner);
        if (outPartition.empty()) {
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            thenToken, thenState.current.owner,
                                            activeIfOp.getOperation()));
          addPartitionIds(outPartition, partitionSetForTokenOrOwner(
                                            elseToken, elseState.current.owner,
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
            thenState.current.semaphore ? thenState.current.semaphore
                                       : incomingSemaphore;
        Value elseSemaphore =
            elseState.current.semaphore ? elseState.current.semaphore
                                       : incomingSemaphore;
        Value joinedSemaphore =
            thenSemaphore == elseSemaphore
                ? thenSemaphore
                : (outOwner ? (thenSemaphore ? thenSemaphore : elseSemaphore)
                            : Value());
        EmitState joinedState = state;
        mergeProtectedAccesses(joinedState, thenState);
        mergeProtectedAccesses(joinedState, elseState);
        joinedState.current.setCarrier(
            joinedSemaphore,
            activeIfOp.getResult(reuseExistingTokenResult
                                     ? *reusableIfTokenResult
                                     : oldNumResults),
            outOwner);
        state = joinedState;
      } else {
        mergeProtectedAccesses(state, thenState);
        mergeProtectedAccesses(state, elseState);
      }
      if (failed(emitOperationExitTransitions(oldIfOp, activeIfOp.getOperation(),
                                              dag, sp, group, state)))
        return failure();
      if (failed(emitDeferredNestedExitTransitions(activeIfOp.getOperation(),
                                                   dag, sp, group, state)))
        return failure();
      if (threaded && !reuseExistingTokenResult)
        state.eraseAfterEmission.insert(oldIfOp);
      continue;
    }

    if (event && !touches.empty()) {
      OpBuilder b(&op);
      b.setInsertionPoint(&op);
      if (failed(emitAccessTransition(b, *event, touches, acquires, group, dag,
                                      backing, state)))
        return failure();
    }
    if (failed(emitOperationExitTransitions(&op, dag, sp, group, state)))
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
  GroupBacking &backing =
      ensureGroupBacking(group, dag.groupIdx, dag.resource.second,
                         plan.memberIndices, backings, numStagesByGroup);
  EmitState state;
  state.semas = createResourceSemaphores(dag, sp, group, backing);
  EmitterTransitionPlan transitions = buildEmitterTransitionPlan(dag, sp, group);
  state.transitions = &transitions;
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
    auto makeCandidate = [&](bool branchIsThen, bool releaseOnly)
        -> std::optional<SemaphoreIfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto releaseOp = findBranchReleaseForSplit(block);
      if (!releaseOp)
        return std::nullopt;
      if (releaseOnly) {
        if (!semaphoreUsesTmem(releaseOp.getSemaphore()) ||
            !branchHasAcquireAfter(releaseOp))
          return std::nullopt;
        return SemaphoreIfSplitCandidate{
            ifOp, branchIsThen, releaseOp, SemaphoreAcquireOp(), 0,
            /*releaseOnly=*/true};
      }
      auto acquireOp = findBranchTrailingAcquire(block);
      if (!acquireOp)
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

    for (bool releaseOnly : {false, true})
      for (bool branchIsThen : {true, false})
        if (auto candidate = makeCandidate(branchIsThen, releaseOnly)) {
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

static void poisonUnbackedCarrierTokenSlots(triton::FuncOp funcOp) {
  SmallVector<scf::ForOp, 8> loops;
  funcOp.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  for (scf::ForOp forOp : loops) {
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
          backingBySlot[slot] = createOp.getBuffers().front();
          continue;
        }
      }
      if (std::optional<Value> backing =
              inferSemaphoreBackingForCarrierSlot(forOp, slot))
        backingBySlot[slot] = *backing;
    }
    poisonDuplicateUnbackedTokenSlots(forOp, asyncTokenSlots, backingBySlot);
  }
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMITTER_H_
