#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_SCHEDULE_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_SCHEDULE_H_

// ---------------------------------------------------------------------------
// EMIT-SCHEDULE (commit 4.5, section 5 stage 2 / section 6.7).
//
// buildEmitSchedule() turns a completed OPT-SYNC-DAG into an explicit, ordered
// list of typed emit actions (CreateSemaphore / Acquire / Release / Buffer /
// ThreadToken). Every action carries a stable semaphore identity resolved by
// section 6.4 -- the released seed resolves to the single released semaphore;
// every other action is keyed by (groupIdx, edge.dstOwner) -- plus a DAG-derived
// endpoint / placement basis. Each action carries only model-legal data: a
// semaphore identity, the single is_released bit (section 6.3), the release
// owner / async payload already stamped on the edge, and the endpoint. There is
// no "empty"/"full" notion: a semaphore is created released or unreleased, and
// that is all (section 2).
//
// The schedule is a pure value; it mutates no IR. dumpEmitSchedule() prints it
// (under NVWS_INSERT_SEMA_DUMP_DAG) and runs the in-plan M3 acquirer-set check;
// materializeSchedule() (commit 5) consumes it.
// EmitActionKind / EmitAction / EmitSchedule are defined in InsertSemasModel.h
// so EmitState can carry the schedule. Everything here lives in the same
// (outer) anonymous namespace as InsertSemasEmitter.h so the forward-declared
// helpers below resolve to their definitions there.

// Defined in InsertSemasEmitter.h (later in the same TU); the schedule builder
// uses them to settle the release payload as a DAG fact.
static const AccessEvent *findLastProducerInRegion(Region *region,
                                                   BufferGroup &group,
                                                   int64_t resourceKey);
static bool shouldForceNonePayload(const SyncGroup &syncGroup,
                                   const SyncPlan &sp, const SyncEdge *edge,
                                   SyncAnchorKind kind);

// The release's async payload, settled once from DAG facts (section 6.6). This
// is the single source of truth the emitter reads -- no emit-time re-derivation.
// terminal* flags mirror the emitter's anchor classification; both are pure
// functions of (edge, anchor, kind, group), no EmitState dependence.
static AsyncOp computeReleasePayload(const SyncGroup &syncGroup,
                                     const SyncPlan &sp, const SyncEdge *edge,
                                     Operation *anchor, SyncAnchorKind kind,
                                     std::optional<unsigned> edgeIdx,
                                     const OptSyncDag &dag, BufferGroup &group) {
  bool terminalDstReadRelease =
      (syncGroup.kind == SyncGroupKind::LinearChain ||
       syncGroup.kind == SyncGroupKind::Singleton) &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && edge->dstOp == anchor &&
      edge->srcOp != anchor;
  bool terminalLoopExitReadRelease =
      syncGroup.kind == SyncGroupKind::Singleton &&
      kind == SyncAnchorKind::ReleaseAfterOp && edge && anchor && edgeIdx &&
      dag.tmemLoopExitRead.lookup(*edgeIdx) == anchor;
  Operation *payloadOp = edge ? edge->srcOp : nullptr;
  AsyncOp payload =
      (terminalDstReadRelease || terminalLoopExitReadRelease)
          ? getAsyncPayload(anchor)
          : (edge ? edge->asyncPayload : getAsyncPayload(payloadOp));
  if (group.isTmem() && edge && edge->srcYieldRegion &&
      !terminalDstReadRelease && !terminalLoopExitReadRelease &&
      payload == AsyncOp::NONE)
    if (const AccessEvent *producer =
            findLastProducerInRegion(edge->srcYieldRegion, group, dag.resource.second))
      if (sameOwner(producer->owner, edge->srcOwner))
        payload = getAsyncPayload(producer->op);
  if (shouldForceNonePayload(syncGroup, sp, edge, kind))
    payload = AsyncOp::NONE;
  return payload;
}

// Format an optional owner the same way the RELEASED-SEMAPHORES dump does.
static std::string ownerLabel(const std::optional<PartitionId> &o) {
  if (!o)
    return "{root}";
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{p" << o->first << ",ws" << o->second << "}";
  return os.str();
}

// section 6.4 semaphore-id resolution. The seed is the single releasedSemaphores
// entry; every other action is keyed (groupIdx, edge.dstOwner).
static void resolveSemaId(const OptSyncDag &dag, unsigned groupIdx,
                          const SyncEdge *edge, EmitAction &a) {
  if (dag.groups[groupIdx].kind == SyncGroupKind::InitialEmpty) {
    a.isSeed = true;
    a.acquirer = dag.releasedSemaphores.empty()
                     ? dag.groups[groupIdx].initialOwner
                     : dag.releasedSemaphores.front().second;
    a.semaKey = "SEED";
    return;
  }
  a.isSeed = false;
  a.acquirer = edge ? edge->dstOwner : std::nullopt;
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "g" << groupIdx << ownerLabel(a.acquirer);
  a.semaKey = os.str();
}

// Build the ordered emit schedule from the completed OPT-SYNC-DAG. Pure: no IR
// is mutated. Mirrors the legacy emit anchors (acquire/release maps, thread
// sets, releasedSemaphores seed) so that materializeSchedule reproduces the
// same IR mechanically.
static EmitSchedule buildEmitSchedule(const OptSyncDag &dag, const SyncPlan &sp,
                                      BufferGroup &group,
                                      triton::FuncOp funcOp) {
  DenseMap<Operation *, unsigned> rank;
  buildProgramOrderRank(funcOp, rank);
  auto rankOf = [&](Operation *op, Region *region) -> unsigned {
    Operation *keyOp = op ? op : (region ? region->getParentOp() : nullptr);
    if (!keyOp)
      return 0;
    auto it = rank.find(keyOp);
    return it == rank.end() ? 0 : it->second;
  };

  EmitSchedule schedule;

  // 1) CreateSemaphore: one per (edge-bearing group, first-seen dstOwner) in
  // deterministic order (section 6.5), plus the released seed group. The seed
  // is created released (section 6.3); every other class is unreleased (M2).
  for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
    unsigned groupIdx = static_cast<unsigned>(idx);
    if (syncGroup.kind == SyncGroupKind::InitialEmpty) {
      EmitAction a;
      a.kind = EmitActionKind::CreateSemaphore;
      a.groupIdx = groupIdx;
      a.released = true;
      resolveSemaId(dag, groupIdx, nullptr, a);
      schedule.push_back(std::move(a));
      continue;
    }
    SmallVector<std::optional<PartitionId>, 2> seen;
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      std::optional<PartitionId> owner = sp.edges[edgeIdx].dstOwner;
      if (llvm::any_of(seen, [&](auto &o) { return sameOwner(o, owner); }))
        continue;
      seen.push_back(owner);
      EmitAction a;
      a.kind = EmitActionKind::CreateSemaphore;
      a.groupIdx = groupIdx;
      a.edgeIdx = edgeIdx;
      a.released = false; // non-seed classes are unreleased (M2)
      resolveSemaId(dag, groupIdx, &sp.edges[edgeIdx], a);
      schedule.push_back(std::move(a));
    }
  }

  // 2) Acquire actions from the acquire anchors.
  auto addAcquire = [&](unsigned groupIdx, const SyncEdge *edge,
                        SyncAnchorKind anchorKind, Operation *op,
                        Region *region) {
    EmitAction a;
    a.kind = EmitActionKind::Acquire;
    a.groupIdx = groupIdx;
    a.edgeIdx = findEdgeIndex(sp, edge);
    a.anchorKind = anchorKind;
    a.anchorOp = op;
    a.anchorRegion = region;
    a.rank = rankOf(op, region);
    a.priority = 1;
    resolveSemaId(dag, groupIdx, edge, a);
    schedule.push_back(std::move(a));
  };
  for (auto &[op, groups] : dag.acquireBeforeOp)
    for (unsigned g : groups) {
      const SyncEdge *e = findEdgeForAnchor(dag.groups[g], sp, dag,
                                            SyncAnchorKind::AcquireBeforeOp, op,
                                            nullptr);
      addAcquire(g, e, SyncAnchorKind::AcquireBeforeOp, op, nullptr);
    }
  for (auto &[region, groups] : dag.acquireBeforeYield)
    for (unsigned g : groups) {
      const SyncEdge *e = findEdgeForAnchor(dag.groups[g], sp, dag,
                                            SyncAnchorKind::AcquireBeforeYield,
                                            nullptr, region);
      addAcquire(g, e, SyncAnchorKind::AcquireBeforeYield, nullptr, region);
    }

  // 3) Release actions from the release anchors. owner / payload come straight
  // off the representative edge (section 6.6 facts), not from emit-time sniffing.
  auto addRelease = [&](const PlannedRelease &pr, SyncAnchorKind anchorKind,
                        Operation *op, Region *region) {
    EmitAction a;
    a.kind = EmitActionKind::Release;
    a.groupIdx = pr.groupIdx;
    a.anchorKind = anchorKind;
    a.anchorOp = op;
    a.anchorRegion = region;
    a.release = pr;
    a.rank = rankOf(op, region);
    a.priority = 0;
    const SyncEdge *e = getRepresentativeReleaseEdge(pr, sp);
    a.edgeIdx = findEdgeIndex(sp, e);
    a.owner = e ? e->srcOwner : std::nullopt;
    a.payload = computeReleasePayload(dag.groups[pr.groupIdx], sp, e, op,
                                      anchorKind, a.edgeIdx, dag, group);
    resolveSemaId(dag, pr.groupIdx, e, a);
    schedule.push_back(std::move(a));
  };
  for (auto &[op, rels] : dag.releaseBeforeOp)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, SyncAnchorKind::ReleaseBeforeOp, op, nullptr);
  for (auto &[op, rels] : dag.releaseAfterOp)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, SyncAnchorKind::ReleaseAfterOp, op, nullptr);
  for (auto &[region, rels] : dag.releaseBeforeYield)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, SyncAnchorKind::ReleaseBeforeYield, nullptr, region);
  for (auto &[region, rels] : dag.releaseAfterYield)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, SyncAnchorKind::ReleaseAfterYield, nullptr, region);

  // 4) ThreadToken actions (forward token threading, section 6.6).
  for (Operation *op : dag.threadForOps) {
    EmitAction a;
    a.kind = EmitActionKind::ThreadToken;
    a.threadOp = op;
    a.rank = rankOf(op, nullptr);
    a.priority = 3;
    a.semaKey = "thread-for";
    schedule.push_back(std::move(a));
  }
  for (Operation *op : dag.threadIfOps) {
    EmitAction a;
    a.kind = EmitActionKind::ThreadToken;
    a.threadOp = op;
    a.rank = rankOf(op, nullptr);
    a.priority = 3;
    a.semaKey = "thread-if";
    schedule.push_back(std::move(a));
  }

  // CreateSemaphore actions first (section 6.5 order, kept stable), then by
  // program-order rank, priority, and stable key (section 6.7).
  std::stable_sort(schedule.begin(), schedule.end(),
                   [](const EmitAction &a, const EmitAction &b) {
                     bool ac = a.kind == EmitActionKind::CreateSemaphore;
                     bool bc = b.kind == EmitActionKind::CreateSemaphore;
                     if (ac != bc)
                       return ac;
                     if (ac && bc)
                       return false;
                     if (a.rank != b.rank)
                       return a.rank < b.rank;
                     if (a.priority != b.priority)
                       return a.priority < b.priority;
                     return a.semaKey < b.semaKey;
                   });
  return schedule;
}

static void dumpEmitSchedule(const OptSyncDag &dag, const SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
                             triton::FuncOp funcOp) {
  EmitSchedule schedule = buildEmitSchedule(dag, sp, group, funcOp);

  llvm::errs() << "EMIT-SCHEDULE buffer.id=" << dag.resource.first
               << " resourceKey=" << dag.resource.second
               << " actions=" << schedule.size()
               << " initialPermitEdgeIdx=" << sp.initialPermitEdgeIdx
               << " initialPermitName=" << sp.initialPermitName << "\n";
  if (!sp.semaRep.empty()) {
    unsigned seedId = static_cast<unsigned>(sp.edges.size());
    llvm::errs() << "  SEMA-UNION:";
    SmallVector<SmallVector<unsigned, 2>, 4> classes(sp.semaRep.size());
    for (unsigned i = 0; i < sp.semaRep.size(); ++i)
      classes[sp.semaFind(i)].push_back(i);
    for (auto &cls : classes) {
      if (cls.size() < 2)
        continue;
      llvm::errs() << " {";
      for (unsigned i : cls)
        llvm::errs() << (i == seedId ? "SEED" : ("S" + std::to_string(i)))
                     << " ";
      llvm::errs() << "}";
      // Soundness: a class must not unite two distinct concrete owners.
      SmallVector<PartitionId, 2> concrete;
      for (unsigned i : cls) {
        if (i >= sp.edges.size())
          continue; // seed acquires as root in the RAW model
        if (std::optional<PartitionId> o = sp.edges[i].dstOwner)
          if (!llvm::is_contained(concrete, *o))
            concrete.push_back(*o);
      }
      if (concrete.size() > 1)
        llvm::errs() << "<<UNSOUND-UNION: " << concrete.size()
                     << " concrete owners>>";
    }
    llvm::errs() << "\n";
  }

  auto endpointDesc = [&](const EmitAction &a) -> std::string {
    Operation *keyOp = a.anchorOp ? a.anchorOp
                       : a.threadOp ? a.threadOp
                       : a.anchorRegion ? a.anchorRegion->getParentOp()
                                        : nullptr;
    if (!keyOp)
      return a.kind == EmitActionKind::CreateSemaphore ? "create-scope"
                                                       : "<none>";
    std::string s;
    llvm::raw_string_ostream os(s);
    const char *pfx = a.anchorKind == SyncAnchorKind::ReleaseAfterOp ||
                              a.anchorKind == SyncAnchorKind::ReleaseAfterYield
                          ? "after:"
                      : a.kind == EmitActionKind::Release ? "before:"
                                                          : "";
    os << pfx << keyOp->getName().getStringRef() << "@" << a.rank;
    if (a.anchorRegion)
      os << "/yield";
    return os.str();
  };

  // Aggregate per-semaphore acquirer sets for an in-plan M3 check.
  SmallVector<std::pair<std::string, SmallVector<std::optional<PartitionId>, 2>>,
              4>
      acqSet;
  auto acqSetFor =
      [&](const std::string &key) -> SmallVector<std::optional<PartitionId>, 2> & {
    for (auto &kv : acqSet)
      if (kv.first == key)
        return kv.second;
    acqSet.push_back({key, {}});
    return acqSet.back().second;
  };
  for (const EmitAction &a : schedule) {
    const char *k = a.kind == EmitActionKind::CreateSemaphore ? "create "
                    : a.kind == EmitActionKind::Acquire       ? "acquire"
                    : a.kind == EmitActionKind::Release       ? "release"
                    : a.kind == EmitActionKind::Buffer        ? "buffer "
                                                              : "thread ";
    llvm::errs() << "  " << k << "  sema="
                 << (a.semaKey == "SEED" ? ("SEED" + ownerLabel(a.acquirer))
                     : a.semaKey == "thread-for" || a.semaKey == "thread-if"
                         ? a.semaKey
                         : a.semaKey);
    if (a.edgeIdx)
      llvm::errs() << " edge=" << *a.edgeIdx;
    if (a.kind == EmitActionKind::CreateSemaphore)
      llvm::errs() << " released=" << (a.released ? "true" : "false");
    llvm::errs() << "  @ " << endpointDesc(a) << "\n";
    if (a.kind == EmitActionKind::Acquire) {
      auto &set = acqSetFor(a.semaKey);
      if (!llvm::any_of(set, [&](auto &o) { return sameOwner(o, a.acquirer); }))
        set.push_back(a.acquirer);
    }
  }
  for (auto &[key, owners] : acqSet) {
    unsigned concrete = 0;
    for (auto &o : owners)
      if (o)
        ++concrete;
    if (concrete > 1) {
      llvm::errs() << "  <<M3-VIOLATION: sema " << key << " acquired by "
                   << concrete << " concrete owners";
      for (auto &o : owners)
        llvm::errs() << " " << ownerLabel(o);
      llvm::errs() << ">>\n";
    }
  }
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_SCHEDULE_H_
