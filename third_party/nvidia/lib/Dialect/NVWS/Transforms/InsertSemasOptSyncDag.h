#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OPT_SYNC_DAG_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OPT_SYNC_DAG_H_

// ---------------------------------------------------------------------------
// v4 §Final Combine Subpass / §End-to-End examples — OPT-SYNC DAG.
// Classify each raw SyncEdge into one of:
//   - Singleton:   one src -> one dst (default, identical render to RAW)
//   - ReadyFanout: one producer event -> N consumer events (Combine A)
//   - DoneFanin:   N reader events -> one next-writer event   (Combine B)
//   - LinearChain: not detected here; LinearChain is a render-time
//                  optimization at emit-stage and need not change the
//                  dump structure. We emit it explicitly only when the
//                  pattern is unambiguous: a singleton handoff whose
//                  src/dst chain forms a linear sequence with no other
//                  fanout/fanin participants. For commit 4 we leave
//                  linear handoffs as Singletons; combine C is an emit
//                  optimization deferred to commit 5.
// ---------------------------------------------------------------------------

static bool edgeRequiresRelease(const SyncEdge &edge) {
  if (!edge.srcOwner)
    return false;
  if (edge.dstOwner && sameOwner(edge.srcOwner, edge.dstOwner))
    return false;
  return true;
}

template <typename AnchorT>
static void addPlannedRelease(
    llvm::MapVector<AnchorT *, SmallVector<PlannedRelease, 2>> &anchors,
    AnchorT *anchor, const SyncPlan &sp, unsigned groupIdx,
    ArrayRef<unsigned> edgeIdxs) {
  if (!anchor)
    return;
  PlannedRelease action;
  action.groupIdx = groupIdx;
  for (unsigned edgeIdx : edgeIdxs) {
    if (edgeIdx >= sp.edges.size())
      continue;
    if (!edgeRequiresRelease(sp.edges[edgeIdx]))
      continue;
    action.edgeIdxs.push_back(edgeIdx);
  }
  if (action.edgeIdxs.empty())
    return;
  SmallVector<PlannedRelease, 2> &planned = anchors[anchor];
  if (!llvm::is_contained(planned, action))
    planned.push_back(std::move(action));
}

template <typename AnchorT>
static void addPlannedRelease(
    llvm::MapVector<AnchorT *, SmallVector<PlannedRelease, 2>> &anchors,
    AnchorT *anchor, const SyncPlan &sp, unsigned groupIdx, unsigned edgeIdx) {
  addPlannedRelease(anchors, anchor, sp, groupIdx,
                    ArrayRef<unsigned>(&edgeIdx, 1));
}

static std::string makeGroupName(unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "S_g" << serial;
  return s;
}

static std::optional<unsigned> findEdgeIndex(const SyncPlan &sp,
                                             const SyncEdge *edge) {
  if (!edge)
    return std::nullopt;
  for (auto [idx, candidate] : llvm::enumerate(sp.edges))
    if (&candidate == edge)
      return static_cast<unsigned>(idx);
  return std::nullopt;
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

static bool isIfYieldRegion(Region *region);
static bool opReadsOnlyResource(Operation *op, BufferGroup &group,
                                int64_t resourceKey);
static Operation *findTerminalReadReleaseAnchor(Operation *op,
                                                const SyncPlan &sp,
                                                BufferGroup &group,
                                                int64_t resourceKey);

static bool collectLinearChainEdges(const SyncPlan &sp, ArrayRef<bool> claimed,
                                    SmallVectorImpl<unsigned> &chain) {
  chain.clear();
  for (unsigned i = 0; i < sp.edges.size(); ++i)
    if (!claimed[i])
      chain.push_back(i);
  if (chain.size() < 2) return false;

  SmallVector<ReadyFanoutKey, 4> srcKeys;
  SmallVector<DoneFaninKey, 4> dstKeys;
  for (auto [pos, idx] : llvm::enumerate(chain)) {
    const SyncEdge &e = sp.edges[idx];
    if ((!e.srcOp && !e.srcYieldRegion) || (!e.dstOp && !e.dstYieldRegion))
      return false;
    if (e.srcYieldRegion)
      return false;
    if (e.dstYieldRegion && !isIfYieldRegion(e.dstYieldRegion) &&
        (pos + 1 != chain.size() || e.kind != SyncEdgeKind::Done))
      return false;
    if (e.srcOp && isa<scf::ForOp, scf::IfOp>(e.srcOp))
      return false;
    ReadyFanoutKey srcKey{e.srcOwner, e.srcEpoch, e.srcOp, e.srcYieldRegion};
    if (llvm::is_contained(srcKeys, srcKey))
      return false;
    srcKeys.push_back(srcKey);
    DoneFaninKey dstKey{e.dstOwner, e.dstOp, e.dstYieldRegion};
    if (llvm::is_contained(dstKeys, dstKey))
      return false;
    dstKeys.push_back(dstKey);
  }

  for (unsigned i = 1; i < chain.size(); ++i) {
    const SyncEdge &prev = sp.edges[chain[i - 1]];
    const SyncEdge &cur = sp.edges[chain[i]];
    if (!sameOwner(prev.dstOwner, cur.srcOwner))
      return false;
  }

  return true;
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

static bool yieldedAccessTokenRequiresCarrier(Region *region,
                                              const SyncPlan &sp) {
  if (!region) return false;
  for (Block &block : *region) {
    auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
    if (!yieldOp) continue;
    for (Value operand : yieldOp.getOperands()) {
      if (!isa<AsyncTokenType>(operand.getType())) continue;
      auto result = dyn_cast<OpResult>(operand);
      if (result && sp.accessOps.contains(result.getOwner()))
        return true;
    }
  }
  return false;
}

static bool syncYieldRequiresCarrier(Region *region, const SyncPlan &sp) {
  if (!region) return false;
  Operation *parent = region->getParentOp();
  // A sync edge targeting a structured yield makes the semaphore token the
  // state that crosses the CFG boundary. Loops need it for re-entry even when
  // the source access did not already produce an async token in the input IR.
  if (isa_and_nonnull<scf::ForOp, scf::IfOp>(parent))
    return true;
  return yieldedAccessTokenRequiresCarrier(region, sp);
}

static Operation *findFirstAccessInRegion(Region &region,
                                          const OptSyncDag &dag) {
  Operation *firstAccess = nullptr;
  region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (!dag.accessOps.contains(op))
      return WalkResult::advance();
    firstAccess = op;
    return WalkResult::interrupt();
  });
  return firstAccess;
}

static bool isIfYieldRegion(Region *region) {
  return region && isa<scf::IfOp>(region->getParentOp());
}

static Operation *findFirstAccessAfter(Operation *op, const OptSyncDag &dag) {
  if (!op || !op->getBlock())
    return nullptr;
  auto it = op->getIterator();
  for (++it; it != op->getBlock()->end(); ++it) {
    Operation *candidate = &*it;
    if (dag.accessOps.contains(candidate))
      return candidate;
    Operation *nestedAccess = nullptr;
    candidate->walk<WalkOrder::PreOrder>([&](Operation *nested) -> WalkResult {
      if (!dag.accessOps.contains(nested))
        return WalkResult::advance();
      nestedAccess = nested;
      return WalkResult::interrupt();
    });
    if (nestedAccess)
      return nestedAccess;
  }
  return nullptr;
}

static scf::ForOp getContainingWsFor(Operation *op) {
  if (!op)
    return {};
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    if (hasWarpSpecializeTag(forOp))
      return forOp;
  for (auto forOp = op->getParentOfType<scf::ForOp>(); forOp;
       forOp = forOp->getParentOfType<scf::ForOp>())
    if (hasWarpSpecializeTag(forOp))
      return forOp;
  return {};
}

static bool operationIsAttached(Operation *op) {
  return op && op->getBlock() && op->getBlock()->getParent();
}

static Operation *getIfBranchEntryAnchor(Operation *op) {
  if (!operationIsAttached(op))
    return op;
  Region *parentRegion = op->getBlock()->getParent();
  if (!parentRegion ||
      !isa_and_nonnull<scf::IfOp>(parentRegion->getParentOp()))
    return op;
  Operation *first = &op->getBlock()->front();
  if (isa<scf::YieldOp>(first))
    return op;
  return first;
}

static Operation *findTmemLoopExitReadForEdge(const SyncEdge &edge,
                                              const SyncPlan &sp,
                                              const OptSyncDag &dag,
                                              BufferGroup &group,
                                              int64_t resourceKey) {
  if (!group.isTmem() || !edge.dstYieldRegion || edge.dstOwner || !edge.srcOp)
    return nullptr;
  auto forOp =
      dyn_cast_or_null<scf::ForOp>(edge.dstYieldRegion->getParentOp());
  if (!forOp || !hasWarpSpecializeTag(forOp) ||
      !forOp->isProperAncestor(edge.srcOp))
    return nullptr;
  Operation *afterLoop = findFirstAccessAfter(forOp.getOperation(), dag);
  return findTerminalReadReleaseAnchor(afterLoop, sp, group, resourceKey);
}

static Operation *findFirstWriter(const SyncPlan &sp, BufferGroup &group,
                                  std::optional<PartitionId> &owner) {
  for (AccessEvent &event : group.events) {
    if (!sp.accessOps.contains(event.op))
      continue;
    if (!eventProduces(event, sp.resource.second))
      continue;
    owner = event.owner;
    return event.op;
  }
  return nullptr;
}

static scf::ForOp getSkippedInitialLoopCarrierFor(const SyncGroup &syncGroup,
                                                  const SyncPlan &sp,
                                                  BufferGroup &group) {
  if (!group.isTmem() || syncGroup.edgeIdxs.empty())
    return {};
  std::optional<PartitionId> firstWriterOwner;
  Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner);
  if (!firstWriter)
    return {};
  const SyncEdge &firstEdge = sp.edges[syncGroup.edgeIdxs.front()];
  if (firstEdge.srcOp != firstWriter)
    return {};
  auto forOp = dyn_cast_or_null<scf::ForOp>(firstEdge.dstOp);
  if (!forOp)
    return {};
  for (unsigned memberIdx : sp.memberIndices) {
    if (memberIdx >= group.members.size())
      continue;
    if (group.members[memberIdx].allocOp->hasAttr("buffer.copy"))
      return {};
  }
  if (forOp.getOperation()->isProperAncestor(firstWriter))
    return {};
  return forOp;
}

static bool edgeDefersToSkippedLoopExit(const SyncEdge &edge,
                                        scf::ForOp forOp) {
  if (!forOp || !edge.srcOp || !edge.dstOp)
    return false;
  Operation *forOperation = forOp.getOperation();
  return forOperation->isProperAncestor(edge.srcOp) &&
         !forOperation->isProperAncestor(edge.dstOp);
}

static bool edgeDstReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey);
static bool edgeDstWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);
static bool edgeSrcReads(const SyncEdge &edge, BufferGroup &group,
                         int64_t resourceKey);
static bool edgeSrcWrites(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);
static bool edgeNeedsTerminalReadRelease(const SyncEdge &edge,
                                         BufferGroup &group,
                                         int64_t resourceKey);
static bool isRootToWsLoopEntryEdge(const SyncEdge &edge, BufferGroup &group);
static bool edgeUsesEmpty(const SyncEdge &edge, BufferGroup &group,
                          int64_t resourceKey);

static bool hasOutgoingEdgeFromOp(const SyncPlan &sp, Operation *op) {
  if (!op)
    return false;
  for (const SyncEdge &edge : sp.edges)
    if (edge.srcOp == op)
      return true;
  return false;
}

static Operation *getNonWsForAncestorExitingTo(Operation *srcOp,
                                               Operation *dstOp) {
  if (!srcOp || !dstOp)
    return nullptr;
  for (Operation *parent = srcOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto forOp = dyn_cast<scf::ForOp>(parent);
    if (!forOp || hasWarpSpecializeTag(parent))
      continue;
    if (!parent->isProperAncestor(dstOp))
      return parent;
  }
  return nullptr;
}

static bool isOriginalTmemTokenForMember(Value token, BufferGroup &group,
                                         ArrayRef<unsigned> memberIndices) {
  if (!group.isTmem() || !token || !isa<AsyncTokenType>(token.getType()))
    return false;
  for (unsigned memberIdx : memberIndices) {
    if (memberIdx >= group.members.size())
      continue;
    auto allocOp = dyn_cast<TMEMAllocOp>(group.members[memberIdx].allocOp);
    if (!allocOp || !allocOp.getToken())
      continue;
    if (token == allocOp.getToken())
      return true;
  }
  return false;
}

static bool isTmemAccessTokenForResource(Value token, BufferGroup &group,
                                         int64_t resourceKey) {
  if (!group.isTmem() || !token || !isa<AsyncTokenType>(token.getType()))
    return false;
  Operation *def = token.getDefiningOp();
  if (!def)
    return false;
  for (const AccessEvent &event : group.events)
    if (event.op == def && eventTouchesResource(event, resourceKey))
      return true;
  return false;
}

static bool isReusableTmemTokenForResource(Value token, BufferGroup &group,
                                           ArrayRef<unsigned> memberIndices,
                                           int64_t resourceKey) {
  return isOriginalTmemTokenForMember(token, group, memberIndices) ||
         isTmemAccessTokenForResource(token, group, resourceKey);
}

static bool tokenResultIsYielded(Operation *op) {
  if (!op)
    return false;
  for (Value result : op->getResults()) {
    if (!isa<AsyncTokenType>(result.getType()))
      continue;
    for (Operation *user : result.getUsers())
      if (isa<scf::YieldOp>(user))
        return true;
  }
  return false;
}

static bool shouldDeferTerminalLoopReadToExit(const SyncEdge &edge,
                                              BufferGroup &group,
                                              int64_t resourceKey) {
  if (!group.isTmem() || group.members.size() != 1 || !edge.dstYieldRegion ||
      !edge.srcOp || !tokenResultIsYielded(edge.srcOp))
    return false;
  if (group.members.empty() ||
      !group.members.front().allocOp->hasAttr("buffer.copy"))
    return false;
  auto forOp =
      dyn_cast_or_null<scf::ForOp>(edge.dstYieldRegion->getParentOp());
  return forOp && hasWarpSpecializeTag(forOp) &&
         forOp->isProperAncestor(edge.srcOp) &&
         edgeSrcReads(edge, group, resourceKey) &&
         !edgeSrcWrites(edge, group, resourceKey);
}

static std::optional<unsigned>
findReusableTmemTokenSlot(scf::ForOp forOp, BufferGroup &group,
                          ArrayRef<unsigned> memberIndices,
                          int64_t resourceKey) {
  if (!group.isTmem())
    return std::nullopt;
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs())) {
    if (isReusableTmemTokenForResource(init, group, memberIndices, resourceKey))
      return static_cast<unsigned>(idx);
  }
  return std::nullopt;
}

static SmallVector<unsigned, 4>
findReusableTmemTokenSlots(scf::ForOp forOp, BufferGroup &group,
                           ArrayRef<unsigned> memberIndices,
                           int64_t resourceKey) {
  SmallVector<unsigned, 4> slots;
  if (!group.isTmem())
    return slots;
  for (auto [idx, init] : llvm::enumerate(forOp.getInitArgs()))
    if (isReusableTmemTokenForResource(init, group, memberIndices, resourceKey))
      slots.push_back(static_cast<unsigned>(idx));
  return slots;
}

static scf::YieldOp getForYieldOp(scf::ForOp forOp) {
  for (Operation &op :
       llvm::reverse(forOp.getRegion().front().getOperations()))
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op))
      return yieldOp;
  llvm_unreachable("scf.for body has no scf.yield");
}

static scf::YieldOp appendToForYield(scf::ForOp forOp,
                                     ArrayRef<Value> newOperands) {
  scf::YieldOp yieldOp = getForYieldOp(forOp);
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());
  OpBuilder builder(yieldOp);
  auto newYield = scf::YieldOp::create(builder, yieldOp.getLoc(), operands);
  newYield->setAttrs(yieldOp->getAttrs());
  yieldOp->erase();
  return newYield;
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

static bool plannedReleaseForcesNonePayload(const SyncGroup &syncGroup,
                                            const SyncPlan &sp,
                                            const SyncEdge *edge,
                                            SyncAnchorKind kind) {
  return syncGroup.kind == SyncGroupKind::LinearChain &&
         kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->srcOp &&
         isa<MMAv5OpInterface>(edge->srcOp) && edge->dstOp &&
         isa<TMEMLoadOp>(edge->dstOp) &&
         nextLinearEdgeDstIsConditionalStore(syncGroup, sp, edge);
}

static void resolvePlannedReleaseState(PlannedRelease &action,
                                       SyncAnchorKind kind, Operation *anchor,
                                       const OptSyncDag &dag,
                                       const SyncPlan &sp,
                                       BufferGroup &group) {
  if (action.groupIdx >= dag.groups.size())
    return;
  const SyncGroup &syncGroup = dag.groups[action.groupIdx];
  const SyncEdge *edge = getRepresentativeReleaseEdge(action, sp);
  std::optional<unsigned> edgeIdx =
      action.edgeIdxs.empty() ? std::nullopt
                              : std::optional<unsigned>(action.edgeIdxs.front());
  bool terminalDstReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->dstOp == anchor &&
      edge->srcOp != anchor;
  bool terminalLoopExitReadRelease =
      syncGroup.kind == SyncGroupKind::Singleton &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && anchor && edgeIdx &&
      dag.tmemLoopExitRead.lookup(*edgeIdx) == anchor;

  action.owner = edge ? edge->srcOwner : std::nullopt;
  if (terminalDstReadRelease || terminalLoopExitReadRelease)
    action.owner = edge->dstOwner;
  if (terminalLoopExitReadRelease)
    action.owner = getPartitionId(anchor);
  action.useCarriedOwner =
      !action.owner && kind == SyncAnchorKind::ReleaseBeforeOp;
  if (edge && !edge->srcOwner && kind == SyncAnchorKind::ReleaseBeforeOp &&
      syncGroup.kind == SyncGroupKind::LinearChain &&
      syncGroup.edgeIdxs.size() > 1 &&
      &sp.edges[syncGroup.edgeIdxs.front()] == edge) {
    action.owner = sp.edges[syncGroup.edgeIdxs[1]].dstOwner;
    action.useCarriedOwner = false;
  }

  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  action.payload =
      (terminalDstReadRelease || terminalLoopExitReadRelease)
          ? getAsyncPayload(anchor)
          : (edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
  action.useCarriedPayload =
      terminalDstReadRelease || terminalLoopExitReadRelease;
  if (plannedReleaseForcesNonePayload(syncGroup, sp, edge, kind)) {
    action.payload = AsyncOp::NONE;
    action.useCarriedPayload = false;
  }
}

template <typename AnchorT>
static void addTransitionAcquire(
    llvm::MapVector<AnchorT *, SmallVector<unsigned, 2>> &anchors, AnchorT *anchor,
    unsigned groupIdx, bool unique = false) {
  if (!anchor)
    return;
  SmallVector<unsigned, 2> &groups = anchors[anchor];
  if (!unique || !llvm::is_contained(groups, groupIdx))
    groups.push_back(groupIdx);
}

static Operation *findInitialAcquireAnchor(const SyncGroup &syncGroup,
                                           const OptSyncDag &dag,
                                           BufferGroup &group,
                                           bool reanchorInitialAcquire) {
  if (!reanchorInitialAcquire || !syncGroup.initialOp)
    return syncGroup.initialOp;
  Operation *threadedRegionOp = nullptr;
  for (Operation *parent = syncGroup.initialOp->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (dag.threadForOps.contains(parent) ||
          findReusableTmemTokenSlot(forOp, group, dag.memberIndices,
                                    dag.resource.second))
        threadedRegionOp = parent;
      continue;
    }
    // Initial-empty acquire seeds the first writer or a loop carrier. Do not
    // re-anchor it to an enclosing scf.if: branch-local semaphore creates do not
    // dominate the parent if.
    if (isa<scf::IfOp>(parent))
      continue;
  }
  return threadedRegionOp ? threadedRegionOp : syncGroup.initialOp;
}

static EmitterTransitionPlan
buildEmitterTransitionPlan(const OptSyncDag &dag, const SyncPlan &sp,
                           BufferGroup &group, triton::FuncOp funcOp,
                           bool reanchorInitialAcquire = true) {
  EmitterTransitionPlan transitions;
  SmallVector<char, 8> readyFanoutReleaseSeen(dag.groups.size(), 0);
  SmallVector<char, 8> doneFaninAcquireSeen(dag.groups.size(), 0);

  auto siteIsYield = [](SyncRenderSite site) {
    return site.anchor && isa<scf::YieldOp>(site.anchor);
  };
  auto addSiteReleases = [&](SyncRenderSite site, unsigned groupIdx,
                             ArrayRef<unsigned> edgeIdxs) {
    if (siteIsYield(site)) {
      addPlannedRelease(transitions.regionEntryReleases, site.region, sp,
                        groupIdx, edgeIdxs);
      return;
    }
    addPlannedRelease(transitions.opEntryReleases, site.anchor, sp, groupIdx,
                      edgeIdxs);
  };
  auto addSiteRelease = [&](SyncRenderSite site, unsigned groupIdx,
                            unsigned edgeIdx) {
    addSiteReleases(site, groupIdx, ArrayRef<unsigned>(&edgeIdx, 1));
  };
  auto addSiteAcquire = [&](SyncRenderSite site, unsigned groupIdx,
                            bool unique = false) {
    if (siteIsYield(site)) {
      addTransitionAcquire(transitions.regionEntryAcquires, site.region,
                           groupIdx, unique);
      return;
    }
    addTransitionAcquire(transitions.opEntryAcquires, site.anchor, groupIdx,
                         unique);
  };

  for (unsigned groupIdx = 0; groupIdx < dag.groups.size(); ++groupIdx) {
    const SyncGroup &syncGroup = dag.groups[groupIdx];
    if (syncGroup.kind != SyncGroupKind::InitialEmpty)
      continue;
    addTransitionAcquire(
        transitions.opEntryAcquires,
        findInitialAcquireAnchor(syncGroup, dag, group, reanchorInitialAcquire),
        groupIdx);
  }

  (void)walkRenderedSyncSites(
      sp, funcOp, [&](ArrayRef<unsigned> edges, SyncRenderSite site)
                      -> LogicalResult {
        SmallVector<unsigned, 2> faninHere;
        for (unsigned edgeIdx : edges) {
          if (edgeIdx >= dag.edgeToGroup.size())
            continue;
          unsigned groupIdx = dag.edgeToGroup[edgeIdx];
          if (groupIdx >= dag.groups.size())
            continue;
          const SyncGroup &syncGroup = dag.groups[groupIdx];
          switch (syncGroup.kind) {
    case SyncGroupKind::InitialEmpty:
      break;
    case SyncGroupKind::ReadyFanout:
      if (!readyFanoutReleaseSeen[groupIdx]) {
        addSiteReleases(site, groupIdx, syncGroup.edgeIdxs);
        readyFanoutReleaseSeen[groupIdx] = 1;
      }
      addSiteAcquire(site, groupIdx);
      break;
    case SyncGroupKind::DoneFanin:
      addSiteRelease(site, groupIdx, edgeIdx);
      if (!llvm::is_contained(faninHere, groupIdx))
        faninHere.push_back(groupIdx);
      break;
    case SyncGroupKind::Singleton:
    case SyncGroupKind::LinearChain:
      addSiteRelease(site, groupIdx, edgeIdx);
      addSiteAcquire(site, groupIdx);
      break;
      }
    }
    for (unsigned groupIdx : faninHere) {
      if (doneFaninAcquireSeen[groupIdx])
        continue;
      addSiteAcquire(site, groupIdx);
      doneFaninAcquireSeen[groupIdx] = 1;
    }
    return success();
  });

  if (sp.initialPermitReleaseAfterOp) {
    if (sp.initialPermitEdgeIdx >= 0) {
      unsigned edgeIdx = static_cast<unsigned>(sp.initialPermitEdgeIdx);
      if (edgeIdx < sp.edges.size() && edgeIdx < dag.edgeToGroup.size())
        addPlannedRelease(transitions.opExitReleases,
                          sp.initialPermitReleaseAfterOp, sp,
                          dag.edgeToGroup[edgeIdx], edgeIdx);
    } else {
      for (auto [groupIdxIt, syncGroup] : llvm::enumerate(dag.groups)) {
        if (syncGroup.kind != SyncGroupKind::InitialEmpty)
          continue;
        PlannedRelease release;
        release.groupIdx = static_cast<unsigned>(groupIdxIt);
        release.initialPermitTerminalRelease = true;
        SmallVector<PlannedRelease, 2> &planned =
            transitions.opExitReleases[sp.initialPermitReleaseAfterOp];
        if (!llvm::is_contained(planned, release))
          planned.push_back(std::move(release));
        break;
      }
    }
  }

  auto resolveOpReleases = [&](auto &releasesByAnchor, SyncAnchorKind kind) {
    for (auto &[anchor, releases] : releasesByAnchor)
      for (PlannedRelease &release : releases)
        resolvePlannedReleaseState(release, kind, anchor, dag, sp, group);
  };
  auto resolveRegionReleases = [&](auto &releasesByRegion,
                                   SyncAnchorKind kind) {
    for (auto &[region, releases] : releasesByRegion)
      for (PlannedRelease &release : releases)
        resolvePlannedReleaseState(release, kind, nullptr, dag, sp, group);
  };
  resolveOpReleases(transitions.opEntryReleases,
                    SyncAnchorKind::ReleaseBeforeOp);
  resolveOpReleases(transitions.opExitReleases,
                    SyncAnchorKind::ReleaseAfterOp);
  resolveRegionReleases(transitions.regionEntryReleases,
                        SyncAnchorKind::ReleaseBeforeYield);
  return transitions;
}

static OptSyncDag buildOptSyncDag(const SyncPlan &sp, const ResourcePlan &plan,
                                  BufferGroup &group, triton::FuncOp funcOp) {
  OptSyncDag dag;
  dag.resource = sp.resource;
  dag.groupIdx = sp.groupIdx;
  dag.memberIndices = sp.memberIndices;
  dag.accessOps = sp.accessOps;
  dag.edgeToGroup.assign(sp.edges.size(), 0u);
  for (auto [edgeIdx, edge] : llvm::enumerate(sp.edges)) {
    unsigned idx = static_cast<unsigned>(edgeIdx);
    if (edge.dstOp)
      dag.dstBranchEntryAnchor[idx] = getIfBranchEntryAnchor(edge.dstOp);
    if (!isIfYieldRegion(edge.dstYieldRegion))
      continue;
    Operation *ifOp = edge.dstYieldRegion->getParentOp();
    if (Operation *joinAccess = findFirstAccessAfter(ifOp, dag))
      dag.ifYieldJoinAccess[idx] = joinAccess;
  }
  unsigned groupSerial = 0;

  // Initial writable permit: one planned initially-released EMPTY state per
  // managed (logicalGroupId, resourceKey), consumed by the first writer.
  std::optional<PartitionId> firstWriterOwner;
  if (!sp.edges.empty())
    if (Operation *firstWriter = findFirstWriter(sp, group, firstWriterOwner)) {
      SyncGroup g;
      g.name = makeGroupName(groupSerial++);
      g.kind = SyncGroupKind::InitialEmpty;
      g.initialOp = firstWriter;
      g.initialOwner = firstWriterOwner;
      dag.groups.push_back(std::move(g));
  }

  // First, partition edges by ReadyFanout key. Any bucket with >=2
  // entries collapses to one ReadyFanout group; size-1 buckets defer to
  // DoneFanin classification below.
  SmallVector<ReadyFanoutKey, 0> rfKeys;
  SmallVector<SmallVector<unsigned, 2>, 0> rfBuckets;
  auto rfFind = [&](const ReadyFanoutKey &k) -> int {
    for (unsigned i = 0; i < rfKeys.size(); ++i)
      if (rfKeys[i] == k) return static_cast<int>(i);
    return -1;
  };
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    const SyncEdge &e = sp.edges[i];
    ReadyFanoutKey k{e.srcOwner, e.srcEpoch, e.srcOp, e.srcYieldRegion};
    int idx = rfFind(k);
    if (idx < 0) {
      rfKeys.push_back(k);
      rfBuckets.emplace_back();
      idx = static_cast<int>(rfKeys.size() - 1);
    }
    rfBuckets[idx].push_back(i);
  }

  // Track which edges have been claimed by ReadyFanout groups so they
  // do not later participate in DoneFanin classification.
  SmallVector<bool> claimed(sp.edges.size(), false);

  for (unsigned i = 0; i < rfBuckets.size(); ++i) {
    if (rfBuckets[i].size() < 2) continue;
    // Sanity: a ReadyFanout group must have a usable src anchor (either
    // a real op or a yield-region). Otherwise it's a degenerate group
    // with no place to render the shared release, so leave as singletons.
    const SyncEdge &probe = sp.edges[rfBuckets[i].front()];
    if (!probe.srcOp && !probe.srcYieldRegion) continue;
    // A ReadyFanout shares ONE semaphore that every consumer edge acquires, at
    // its own consumer partition. A semaphore may be released from any
    // partition but acquired in only one, so only collapse when all consumer
    // edges have the same acquirer (dstOwner). Otherwise leave them as per-edge
    // Singletons so each consumer gets its own semaphore.
    if (llvm::any_of(rfBuckets[i], [&](unsigned e) {
          return !sameOwner(sp.edges[e].dstOwner, probe.dstOwner);
        }))
      continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::ReadyFanout;
    g.edgeIdxs.append(rfBuckets[i].begin(), rfBuckets[i].end());
    for (unsigned e : g.edgeIdxs) claimed[e] = true;
    dag.groups.push_back(std::move(g));
  }

  // DoneFanin among the not-yet-claimed edges.
  SmallVector<DoneFaninKey, 0> dfKeys;
  SmallVector<SmallVector<unsigned, 2>, 0> dfBuckets;
  auto dfFind = [&](const DoneFaninKey &k) -> int {
    for (unsigned i = 0; i < dfKeys.size(); ++i)
      if (dfKeys[i] == k) return static_cast<int>(i);
    return -1;
  };
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    if (claimed[i]) continue;
    const SyncEdge &e = sp.edges[i];
    DoneFaninKey k{e.dstOwner, e.dstOp, e.dstYieldRegion};
    int idx = dfFind(k);
    if (idx < 0) {
      dfKeys.push_back(k);
      dfBuckets.emplace_back();
      idx = static_cast<int>(dfKeys.size() - 1);
    }
    dfBuckets[idx].push_back(i);
  }
  for (unsigned i = 0; i < dfBuckets.size(); ++i) {
    if (dfBuckets[i].size() < 2) continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::DoneFanin;
    g.edgeIdxs.append(dfBuckets[i].begin(), dfBuckets[i].end());
    for (unsigned e : g.edgeIdxs) claimed[e] = true;
    dag.groups.push_back(std::move(g));
  }

  // Remaining unclaimed singleton edges that form a program-order handoff chain
  // render as one compact LinearChain group (Combine C).
  SmallVector<unsigned, 4> linearChain;
  if (collectLinearChainEdges(sp, claimed, linearChain)) {
    // A LinearChain collapses all of its writable-permit (EMPTY) handoffs onto
    // one shared EMPTY semaphore. That is sound only if every EMPTY edge is
    // acquired by the SAME partition: a semaphore may be released from any
    // partition, but it must always be acquired in exactly one. If the chain's
    // EMPTY edges span more than one acquirer (dstOwner) — e.g. a merged
    // resource whose qk write (p1) and P write (p5) are both writer phases —
    // collapsing would make one semaphore acquired by several partitions, which
    // is illegal and deadlocks. Leave such edges unclaimed so they fall back to
    // per-edge Singletons, each with a single acquirer (the RAW model).
    //
    // The same rule applies to the chain's data-ready edges, which may share a
    // single semaphore at emit: only collapse when both the writable (EMPTY)
    // edges and the data-ready (non-EMPTY) edges each have a single acquirer.
    SmallVector<std::optional<PartitionId>, 2> emptyAcquirers, fullAcquirers;
    for (unsigned idx : linearChain) {
      const SyncEdge &e = sp.edges[idx];
      auto &bucket = edgeUsesEmpty(e, group, sp.resource.second) ? emptyAcquirers
                                                                 : fullAcquirers;
      if (llvm::none_of(bucket,
                        [&](const std::optional<PartitionId> &o) {
                          return sameOwner(o, e.dstOwner);
                        }))
        bucket.push_back(e.dstOwner);
    }
    if (emptyAcquirers.size() <= 1 && fullAcquirers.size() <= 1) {
      SyncGroup g;
      g.name = makeGroupName(groupSerial++);
      g.kind = SyncGroupKind::LinearChain;
      g.edgeIdxs.append(linearChain.begin(), linearChain.end());
      for (unsigned e : g.edgeIdxs) claimed[e] = true;
      dag.groups.push_back(std::move(g));
    }
  }

  // Remaining: Singleton groups for each unclaimed edge.
  for (unsigned i = 0; i < sp.edges.size(); ++i) {
    if (claimed[i]) continue;
    SyncGroup g;
    g.name = makeGroupName(groupSerial++);
    g.kind = SyncGroupKind::Singleton;
    g.edgeIdxs.push_back(i);
    dag.groups.push_back(std::move(g));
  }

  auto markThreadedOp = [&](Operation *op) {
    if (!op) return;
    if (isa<scf::ForOp>(op))
      dag.threadForOps.insert(op);
    else if (isa<scf::IfOp>(op))
      dag.threadIfOps.insert(op);
  };

  // Populate edgeToGroup and transition metadata. Release/acquire insertion
  // maps are derived separately by buildEmitterTransitionPlan().
  for (unsigned gi = 0; gi < dag.groups.size(); ++gi) {
    const SyncGroup &g = dag.groups[gi];
    for (unsigned ei : g.edgeIdxs) dag.edgeToGroup[ei] = gi;

    switch (g.kind) {
    case SyncGroupKind::InitialEmpty:
      break;
    case SyncGroupKind::Singleton: {
      const SyncEdge &e = sp.edges[g.edgeIdxs.front()];
      if (e.dstOp) {
        if (isRootToWsLoopEntryEdge(e, group)) {
          markThreadedOp(e.dstOp);
          break;
        }
      } else if (e.dstYieldRegion) {
        if (group.isTmem()) {
          auto forOp = dyn_cast_or_null<scf::ForOp>(
              e.dstYieldRegion->getParentOp());
          if (forOp && hasWarpSpecializeTag(forOp) && !e.dstOwner &&
              e.srcOp && forOp->isProperAncestor(e.srcOp)) {
            markThreadedOp(forOp.getOperation());
            break;
          }
        }
        if (syncYieldRequiresCarrier(e.dstYieldRegion, sp))
          markThreadedOp(e.dstYieldRegion->getParentOp());
      }
      break;
    }
    case SyncGroupKind::ReadyFanout:
      for (unsigned ei : g.edgeIdxs)
        if (Region *region = sp.edges[ei].dstYieldRegion)
          markThreadedOp(region->getParentOp());
      break;
    case SyncGroupKind::DoneFanin: {
      const SyncEdge &probe = sp.edges[g.edgeIdxs.front()];
      if (probe.dstYieldRegion && syncYieldRequiresCarrier(probe.dstYieldRegion, sp))
        markThreadedOp(probe.dstYieldRegion->getParentOp());
      break;
    }
    case SyncGroupKind::LinearChain: {
      Operation *initialOp = nullptr;
      for (const SyncGroup &candidate : dag.groups)
        if (candidate.kind == SyncGroupKind::InitialEmpty) {
          initialOp = candidate.initialOp;
          break;
        }
      scf::ForOp skippedCarrierFor =
          getSkippedInitialLoopCarrierFor(g, sp, group);
      if (skippedCarrierFor) {
        dag.skippedInitialLoopCarrierRegion[gi] =
            &skippedCarrierFor.getRegion();
        for (unsigned edgeIdx : g.edgeIdxs)
          if (edgeDefersToSkippedLoopExit(sp.edges[edgeIdx],
                                          skippedCarrierFor))
            dag.edgesDeferringToSkippedLoopExit.insert(edgeIdx);
      }
      bool chainEntersLoop =
          group.isTmem() && !g.edgeIdxs.empty() &&
          isa_and_nonnull<scf::ForOp>(sp.edges[g.edgeIdxs.front()].dstOp);
      bool chainHasIfYield =
          llvm::any_of(g.edgeIdxs, [&](unsigned edgeIdx) {
            return isIfYieldRegion(sp.edges[edgeIdx].dstYieldRegion);
          });
      DenseMap<Operation *, unsigned> ifYieldCounts;
      for (unsigned edgeIdx : g.edgeIdxs) {
        Region *yieldRegion = sp.edges[edgeIdx].dstYieldRegion;
        if (!isIfYieldRegion(yieldRegion))
          continue;
        ++ifYieldCounts[yieldRegion->getParentOp()];
      }
      for (unsigned ei : g.edgeIdxs) {
        const SyncEdge &e = sp.edges[ei];
        if (group.isTmem()) {
          if (auto forOp = dyn_cast_or_null<scf::ForOp>(e.dstOp)) {
            if (Operation *firstAccess =
                    findFirstAccessInRegion(forOp.getRegion(), dag)) {
              markThreadedOp(forOp.getOperation());
              if (skippedCarrierFor && forOp == skippedCarrierFor &&
                  ei == g.edgeIdxs.front() && initialOp && e.srcOp == initialOp)
                continue;
              continue;
            }
          }
        }
        if (isIfYieldRegion(e.dstYieldRegion)) {
          auto ifOp = cast<scf::IfOp>(e.dstYieldRegion->getParentOp());
          if (ifYieldCounts.lookup(ifOp.getOperation()) > 1) {
            if (Operation *joinAccess =
                    findFirstAccessAfter(ifOp.getOperation(), dag)) {
              (void)joinAccess;
              markThreadedOp(ifOp.getOperation());
              continue;
            }
          }
        }
        if (chainEntersLoop && e.dstYieldRegion &&
            !isIfYieldRegion(e.dstYieldRegion)) {
          if (shouldDeferTerminalLoopReadToExit(e, group, sp.resource.second)) {
            auto forOp = dyn_cast_or_null<scf::ForOp>(
                e.dstYieldRegion->getParentOp());
            if (forOp)
              if (Operation *firstAccess =
                      findFirstAccessInRegion(forOp.getRegion(), dag)) {
                dag.loopEntryHandoffAccess[ei] = firstAccess;
                markThreadedOp(forOp.getOperation());
                continue;
              }
            dag.terminalLoopReadEdgesDeferringToExit.insert(ei);
            continue;
          }
          if (e.srcOp && edgeSrcReads(e, group, sp.resource.second) &&
              !edgeSrcWrites(e, group, sp.resource.second) &&
              isa<TMEMLoadOp>(e.srcOp)) {
            if (syncYieldRequiresCarrier(e.dstYieldRegion, sp))
              markThreadedOp(e.dstYieldRegion->getParentOp());
          }
          continue;
        }
        if (e.dstYieldRegion) {
          if (shouldDeferTerminalLoopReadToExit(e, group, sp.resource.second)) {
            auto forOp = dyn_cast_or_null<scf::ForOp>(
                e.dstYieldRegion->getParentOp());
            if (forOp)
              if (Operation *firstAccess =
                      findFirstAccessInRegion(forOp.getRegion(), dag)) {
                dag.loopEntryHandoffAccess[ei] = firstAccess;
                markThreadedOp(forOp.getOperation());
                continue;
              }
            dag.terminalLoopReadEdgesDeferringToExit.insert(ei);
            continue;
          }
        }
        if (chainHasIfYield && e.dstOp)
          (void)getIfBranchEntryAnchor(e.dstOp);
        if (e.dstYieldRegion && syncYieldRequiresCarrier(e.dstYieldRegion, sp))
          markThreadedOp(e.dstYieldRegion->getParentOp());
      }
      break;
    }
    }
  }

  for (auto [edgeIdx, edge] : llvm::enumerate(sp.edges))
    if (Operation *loopExitRead = findTmemLoopExitReadForEdge(
            edge, sp, dag, group, sp.resource.second))
      dag.tmemLoopExitRead[static_cast<unsigned>(edgeIdx)] = loopExitRead;

  EmitterTransitionPlan preliminaryTransitions =
      buildEmitterTransitionPlan(dag, sp, group, funcOp,
                                 /*reanchorInitialAcquire=*/false);
  auto markOpAnchors = [&](const auto &anchors) {
    for (auto &kv : anchors)
      if (isa<scf::ForOp>(kv.first))
        markThreadedOp(kv.first);
  };
  markOpAnchors(preliminaryTransitions.opEntryReleases);
  markOpAnchors(preliminaryTransitions.opEntryAcquires);
  markOpAnchors(preliminaryTransitions.opExitReleases);
  for (auto &kv : preliminaryTransitions.regionEntryAcquires)
    markThreadedOp(kv.first->getParentOp());

  auto markEscapingSourceToken = [&](Operation *srcOp, Operation *dstAnchor) {
    if (!srcOp || !dstAnchor)
      return;
    for (Operation *parent = srcOp->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(parent))
        continue;
      if (parent == dstAnchor || parent->isProperAncestor(dstAnchor))
        break;
      if (auto ifOp = dyn_cast<scf::IfOp>(parent))
        if (ifOp.getElseRegion().empty())
          continue;
      markThreadedOp(parent);
    }
  };
  for (const SyncEdge &edge : sp.edges) {
    Operation *dstAnchor =
        edge.dstOp ? edge.dstOp
                   : (edge.dstYieldRegion ? edge.dstYieldRegion->getParentOp()
                                          : nullptr);
    markEscapingSourceToken(edge.srcOp, dstAnchor);
  }

  auto carriedForRegion = [&](scf::ForOp forOp) {
    auto it = plan.regionOwners.find(&forOp.getRegion());
    return it != plan.regionOwners.end() && it->second.carried &&
           it->second.hasEventsInSubtree &&
           sameOwner(it->second.entry, it->second.exit);
  };
  auto canThreadIfRegion = [&](scf::IfOp ifOp) {
    if (!ifOp || ifOp.getElseRegion().empty())
      return false;
    auto thenIt = plan.regionOwners.find(&ifOp.getThenRegion());
    auto elseIt = plan.regionOwners.find(&ifOp.getElseRegion());
    bool thenHas = thenIt != plan.regionOwners.end() &&
                   thenIt->second.hasEventsInSubtree;
    bool elseHas = elseIt != plan.regionOwners.end() &&
                   elseIt->second.hasEventsInSubtree;
    return thenHas || elseHas;
  };
  auto hasEnclosingCarriedFor = [&](Operation *op) {
    for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
         parent = parent->getParentOp())
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        if (carriedForRegion(forOp))
          return true;
    return false;
  };
  auto carrierCrossesIfBoundary = [&](scf::IfOp ifOp) {
    return hasEnclosingCarriedFor(ifOp.getOperation()) ||
           findFirstAccessAfter(ifOp.getOperation(), dag);
  };
  auto propagateCarriedThreading = [&]() {
    // Edge groups and sync anchors are fixed at this point. This pass derives
    // only SSA carrier plumbing from that completed OPT-SYNC-DAG: when a
    // child structured op is already known to carry the semaphore state, every
    // enclosing loop/if that the state crosses must also return it. Re-entry
    // into a carried loop is itself the next use, so loop propagation must not
    // depend on finding a later access after the child in the same iteration.
    // An if has no re-entry by itself; only thread it when the carrier exits
    // the if to a later access or to an enclosing carried loop.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *, 8> seeds;
      seeds.append(dag.threadForOps.begin(), dag.threadForOps.end());
      seeds.append(dag.threadIfOps.begin(), dag.threadIfOps.end());
      for (Operation *seed : seeds) {
        for (Operation *parent = seed ? seed->getParentOp() : nullptr; parent;
             parent = parent->getParentOp()) {
          if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
            if (carriedForRegion(forOp))
              changed |= dag.threadForOps.insert(parent);
            continue;
          }
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            if (canThreadIfRegion(ifOp) && carrierCrossesIfBoundary(ifOp))
              changed |= dag.threadIfOps.insert(parent);
          }
        }
      }
    }
  };

  auto findLinearChainCarrierLoop = [&](const SyncGroup &syncGroup) -> scf::ForOp {
    if (!group.isTmem() || syncGroup.kind != SyncGroupKind::LinearChain)
      return {};
    SmallVector<scf::ForOp, 4> candidates;
    auto addCandidate = [&](scf::ForOp forOp) {
      if (forOp && hasWarpSpecializeTag(forOp) &&
          !llvm::is_contained(candidates, forOp))
        candidates.push_back(forOp);
    };
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      const SyncEdge &edge = sp.edges[edgeIdx];
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(
              edge.dstYieldRegion ? edge.dstYieldRegion->getParentOp()
                                  : nullptr))
        addCandidate(forOp);
      addCandidate(getContainingWsFor(edge.srcOp));
      addCandidate(getContainingWsFor(edge.dstOp));
    }
    for (scf::ForOp forOp : candidates)
      if (findReusableTmemTokenSlot(forOp, group, dag.memberIndices,
                                    dag.resource.second))
        return forOp;
    return {};
  };

  for (const SyncGroup &syncGroup : dag.groups)
    if (scf::ForOp forOp = findLinearChainCarrierLoop(syncGroup))
      markThreadedOp(forOp.getOperation());

  // A token may be produced by an inner carried loop and be the ownership state
  // that an enclosing carried loop must reuse on its next iteration. Those
  // same-owner reentry requirements are visible in the ownership DAG as
  // `YIELD {P}` rows, not necessarily as extra sync edges, so promote threading
  // through all enclosing carried regions before placing the initial acquire.
  propagateCarriedThreading();

  // When the first writer is inside a loop whose resource state is loop-carried,
  // the initial EMPTY acquire must produce the loop-carried carrier before the
  // scf.for. The semaphore.buffer remains at the first writer and consumes the
  // iter_arg token inside the body.
  for (const SyncGroup &syncGroup : dag.groups) {
    if (syncGroup.kind != SyncGroupKind::InitialEmpty || !syncGroup.initialOp)
      continue;
    Operation *anchor =
        findInitialAcquireAnchor(syncGroup, dag, group,
                                 /*reanchorInitialAcquire=*/true);
    if (anchor != syncGroup.initialOp)
      markThreadedOp(anchor);
  }

  // §6.3 released-bit fact (stage-2 diagnostic). Forward program-order scan over
  // semaphore classes keyed (groupIdx, acquirer). The scan is uniform: for every
  // edge the destination is an *acquire* and the source is a *release* (an empty
  // edge has the reader release and the next writer acquire; a full edge has the
  // producer release and the consumer acquire — both place the acquire at dst and
  // the release at src). The InitialEmpty seed group contributes a single acquire
  // at its first writer with no preceding release. A class is created
  // `is_released = true` iff its earliest event is an acquire (§6.3): nothing
  // releases it before its first acquire. M1 expects exactly one such class for a
  // seeded resource; this is asserted/inspected, never repaired (§1).
  {
    DenseMap<Operation *, unsigned> rank;
    if (!dag.groups.empty()) {
      Operation *anchor = group.members.front().allocOp;
      if (auto funcOp = anchor->getParentOfType<triton::FuncOp>())
        buildProgramOrderRank(funcOp, rank);
    }
    auto rankOf = [&](Operation *op, Region *region) -> std::optional<unsigned> {
      Operation *keyOp = op ? op : (region ? region->getParentOp() : nullptr);
      if (!keyOp)
        return std::nullopt;
      auto it = rank.find(keyOp);
      if (it == rank.end())
        return std::nullopt;
      return it->second;
    };
    struct ClassEvents {
      unsigned groupIdx = 0;
      std::optional<PartitionId> acquirer;
      std::optional<unsigned> firstAcquire;
      std::optional<unsigned> firstRelease;
    };
    SmallVector<ClassEvents, 4> classes;
    auto classFor = [&](unsigned gIdx,
                        std::optional<PartitionId> acquirer) -> ClassEvents & {
      for (ClassEvents &c : classes)
        if (c.groupIdx == gIdx && sameOwner(c.acquirer, acquirer))
          return c;
      classes.push_back(ClassEvents{gIdx, acquirer, std::nullopt, std::nullopt});
      return classes.back();
    };
    auto noteEvent = [](std::optional<unsigned> &slot, std::optional<unsigned> r) {
      if (r && (!slot || *r < *slot))
        slot = r;
    };
    for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
      unsigned gIdx = static_cast<unsigned>(idx);
      if (syncGroup.kind == SyncGroupKind::InitialEmpty) {
        ClassEvents &c = classFor(gIdx, syncGroup.initialOwner);
        noteEvent(c.firstAcquire, rankOf(syncGroup.initialOp, nullptr));
        continue;
      }
      for (unsigned edgeIdx : syncGroup.edgeIdxs) {
        const SyncEdge &edge = sp.edges[edgeIdx];
        ClassEvents &c = classFor(gIdx, edge.dstOwner);
        noteEvent(c.firstAcquire, rankOf(edge.dstOp, edge.dstYieldRegion));
        noteEvent(c.firstRelease, rankOf(edge.srcOp, edge.srcYieldRegion));
      }
    }
    for (const ClassEvents &c : classes) {
      bool released = c.firstAcquire &&
                      (!c.firstRelease || *c.firstAcquire < *c.firstRelease);
      if (released)
        dag.releasedSemaphores.push_back({c.groupIdx, c.acquirer});
    }
  }
  return dag;
}


std::string ownerSetStr(Operation *anchor,
                        ArrayRef<std::optional<PartitionId>> owners) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{";
  bool first = true;
  for (const auto &o : owners) {
    if (!first)
      os << ",";
    os << ownerStr(anchor, o);
    first = false;
  }
  os << "}";
  return s;
}

static LogicalResult
verifyOptSyncDagStructuralInvariant(const OptSyncDag &dag, const SyncPlan &sp,
                                    triton::FuncOp funcOp) {
  if (dag.edgeToGroup.size() != sp.edges.size())
    return emitStructuralSyncInvariantError(
        funcOp, "OPT-SYNC-DAG", "edge-to-group map has the wrong size");

  SmallVector<SyncInvariantSubject, 8> expectedSubjects;
  SmallVector<int, 8> groupSubject(dag.groups.size(), -1);
  SmallVector<int, 8> edgeSubject(sp.edges.size(), -1);

  auto addSubject = [&](unsigned releases, unsigned acquires) {
    expectedSubjects.emplace_back();
    SyncInvariantSubject &subject = expectedSubjects.back();
    subject.expectedReleases = releases;
    subject.expectedAcquires = acquires;
    return static_cast<unsigned>(expectedSubjects.size() - 1);
  };

  for (auto [groupIdxIt, group] : llvm::enumerate(dag.groups)) {
    unsigned groupIdx = static_cast<unsigned>(groupIdxIt);
    if (group.kind == SyncGroupKind::InitialEmpty) {
      if (!group.edgeIdxs.empty())
        return emitStructuralSyncInvariantError(
            funcOp, "OPT-SYNC-DAG", "initial-empty group owns sync edges");
      if (!group.initialOp)
        return emitStructuralSyncInvariantError(
            funcOp, "OPT-SYNC-DAG", "initial-empty group has no anchor");
      continue;
    }
    if (group.edgeIdxs.empty())
      return emitStructuralSyncInvariantError(
          funcOp, "OPT-SYNC-DAG", "non-initial group has no sync edges");
    for (unsigned edgeIdx : group.edgeIdxs) {
      if (edgeIdx >= sp.edges.size())
        return emitStructuralSyncInvariantError(
            funcOp, "OPT-SYNC-DAG", "group edge index is out of range");
      if (dag.edgeToGroup[edgeIdx] != groupIdx)
        return emitStructuralSyncInvariantError(
            funcOp, "OPT-SYNC-DAG", "edge-to-group map is inconsistent");
    }

    switch (group.kind) {
    case SyncGroupKind::InitialEmpty:
      llvm_unreachable("handled above");
    case SyncGroupKind::ReadyFanout:
      groupSubject[groupIdx] =
          static_cast<int>(addSubject(/*releases=*/1,
                                      /*acquires=*/group.edgeIdxs.size()));
      break;
    case SyncGroupKind::DoneFanin:
      groupSubject[groupIdx] =
          static_cast<int>(addSubject(/*releases=*/group.edgeIdxs.size(),
                                      /*acquires=*/1));
      break;
    case SyncGroupKind::Singleton:
    case SyncGroupKind::LinearChain:
      for (unsigned edgeIdx : group.edgeIdxs) {
        if (edgeSubject[edgeIdx] >= 0)
          return emitStructuralSyncInvariantError(
              funcOp, "OPT-SYNC-DAG", "sync edge is assigned twice");
        edgeSubject[edgeIdx] =
            static_cast<int>(addSubject(/*releases=*/1, /*acquires=*/1));
      }
      break;
    }
  }

  SyncDagStructuralVerifier verifier(funcOp, "OPT-SYNC-DAG", sp.edges.size(),
                                     expectedSubjects.size());
  for (auto [idx, subject] : llvm::enumerate(expectedSubjects))
    verifier.subjects[static_cast<unsigned>(idx)] = subject;

  SmallVector<char, 8> groupRendered(dag.groups.size(), 0);
  if (failed(walkRenderedSyncSites(
          sp, funcOp, [&](ArrayRef<unsigned> edges, SyncRenderSite site)
                          -> LogicalResult {
            if (failed(verifier.verifySite(edges, site, sp.edges.size())))
              return failure();
            SmallVector<unsigned, 2> faninHere;
            for (unsigned edgeIdx : edges) {
              unsigned groupIdx = dag.edgeToGroup[edgeIdx];
              if (groupIdx >= dag.groups.size())
                return emitStructuralSyncInvariantError(
                    funcOp, "OPT-SYNC-DAG", "group index is out of range");
              const SyncGroup &group = dag.groups[groupIdx];
              switch (group.kind) {
              case SyncGroupKind::InitialEmpty:
                return emitStructuralSyncInvariantError(
                    funcOp, "OPT-SYNC-DAG",
                    "sync edge maps to initial-empty group");
              case SyncGroupKind::ReadyFanout: {
                int subjectIdx = groupSubject[groupIdx];
                if (subjectIdx < 0)
                  return emitStructuralSyncInvariantError(
                      funcOp, "OPT-SYNC-DAG",
                      "ready-fanout group has no sync subject");
                if (!groupRendered[groupIdx]) {
                  if (failed(verifier.noteRelease(
                          static_cast<unsigned>(subjectIdx), site)))
                    return failure();
                  groupRendered[groupIdx] = 1;
                }
                if (failed(verifier.noteAcquire(
                        static_cast<unsigned>(subjectIdx), site)))
                  return failure();
                break;
              }
              case SyncGroupKind::DoneFanin: {
                int subjectIdx = groupSubject[groupIdx];
                if (subjectIdx < 0)
                  return emitStructuralSyncInvariantError(
                      funcOp, "OPT-SYNC-DAG",
                      "done-fanin group has no sync subject");
                if (failed(verifier.noteRelease(
                        static_cast<unsigned>(subjectIdx), site)))
                  return failure();
                if (!llvm::is_contained(faninHere, groupIdx))
                  faninHere.push_back(groupIdx);
                break;
              }
              case SyncGroupKind::Singleton:
              case SyncGroupKind::LinearChain: {
                int subjectIdx = edgeSubject[edgeIdx];
                if (subjectIdx < 0)
                  return emitStructuralSyncInvariantError(
                      funcOp, "OPT-SYNC-DAG",
                      "sync edge has no sync subject");
                if (failed(verifier.noteRelease(
                        static_cast<unsigned>(subjectIdx), site)))
                  return failure();
                if (failed(verifier.noteAcquire(
                        static_cast<unsigned>(subjectIdx), site)))
                  return failure();
                break;
              }
              }
            }
            for (unsigned groupIdx : faninHere) {
              if (groupRendered[groupIdx])
                continue;
              int subjectIdx = groupSubject[groupIdx];
              if (subjectIdx < 0)
                return emitStructuralSyncInvariantError(
                    funcOp, "OPT-SYNC-DAG",
                    "done-fanin group has no acquire subject");
              if (failed(verifier.noteAcquire(static_cast<unsigned>(subjectIdx),
                                              site)))
                return failure();
              groupRendered[groupIdx] = 1;
            }
            return success();
          })))
    return failure();

  return verifier.finish();
}


void dumpOptSyncDag(const OptSyncDag &dag, SyncPlan &sp,
                    const ResourcePlan &plan, BufferGroup &group,
                    mlir::triton::FuncOp funcOp) {
  unsigned nInitial = 0, nFanout = 0, nFanin = 0, nSingleton = 0, nLinear = 0;
  for (const SyncGroup &g : dag.groups) {
    switch (g.kind) {
    case SyncGroupKind::InitialEmpty:
      ++nInitial;
      break;
    case SyncGroupKind::ReadyFanout:
      ++nFanout;
      break;
    case SyncGroupKind::DoneFanin:
      ++nFanin;
      break;
    case SyncGroupKind::Singleton:
      ++nSingleton;
      break;
    case SyncGroupKind::LinearChain:
      ++nLinear;
      break;
    }
  }
  llvm::errs() << "OPT-SYNC-DAG buffer.id=" << dag.resource.first
               << " resourceKey=" << dag.resource.second
               << " groups=" << dag.groups.size()
               << " (initial-empty=" << nInitial
               << " singleton=" << nSingleton
               << " ready-fanout=" << nFanout
               << " done-fanin=" << nFanin
               << " linear-chain=" << nLinear << ")\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  OptDumpCtx octx{&dag, {}};
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1, &octx);
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OPT_SYNC_DAG_H_
