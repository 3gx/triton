#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_SCHEDULE_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_SCHEDULE_H_

// ---------------------------------------------------------------------------
// EMIT-SCHEDULE dump (commit 4.5, section 6.7). Stage-2 diagnostic only: it
// enumerates the mechanical emit plan (CreateSemaphore / Acquire / Release /
// ThreadToken) directly from the completed OPT-SYNC-DAG anchors, resolving each
// action's semaId by section 6.4. The released seed resolves to the single
// released semaphore; every other action is keyed by (groupIdx, edge.dstOwner).
// No IR is mutated. The dump exposes the per-semaphore acquirer set so M3
// (single concrete acquirer, with optional root) is inspectable in the plan,
// before emission is wired.
namespace {
// Format an optional owner the same way the RELEASED-SEMAPHORES dump does.
static std::string ownerLabel(const std::optional<PartitionId> &o) {
  if (!o)
    return "{root}";
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{p" << o->first << ",ws" << o->second << "}";
  return os.str();
}

struct SchedAction {
  enum Kind { Create, Acquire, Release, Thread } kind;
  unsigned rank = 0;     // program-order rank of the endpoint
  unsigned priority = 0; // section 6.7 tie-break: Release<Acquire<Buffer<Thread
  std::string semaKey;   // identity key for aggregation
  std::string semaLabel; // human label
  std::optional<PartitionId> acquirer;
  bool isSeed = false;
  bool released = false;
  std::optional<unsigned> edgeIdx;
  std::string endpoint;
};
} // namespace

static void dumpEmitSchedule(const OptSyncDag &dag, SyncPlan &sp,
                             const ResourcePlan &plan, BufferGroup &group,
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
  auto endpointDesc = [&](Operation *op, Region *region) -> std::string {
    Operation *keyOp = op ? op : (region ? region->getParentOp() : nullptr);
    if (!keyOp)
      return "<none>";
    std::string s;
    llvm::raw_string_ostream os(s);
    os << keyOp->getName().getStringRef() << "@" << rankOf(op, region);
    if (region)
      os << "/yield";
    return os.str();
  };
  // semaId resolution. The seed is the single releasedSemaphores entry.
  std::optional<std::pair<unsigned, std::optional<PartitionId>>> seed;
  if (!dag.releasedSemaphores.empty())
    seed = dag.releasedSemaphores.front();
  auto semaFor = [&](unsigned groupIdx, const SyncEdge *edge,
                     std::string &key, std::string &label,
                     std::optional<PartitionId> &acquirer, bool &isSeed) {
    if (dag.groups[groupIdx].kind == SyncGroupKind::InitialEmpty) {
      isSeed = true;
      acquirer = seed ? seed->second : dag.groups[groupIdx].initialOwner;
      key = "SEED";
      label = "SEED" + ownerLabel(acquirer);
      return;
    }
    isSeed = false;
    acquirer = edge ? edge->dstOwner : std::nullopt;
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "g" << groupIdx << ownerLabel(acquirer);
    key = os.str();
    label = s;
  };

  SmallVector<SchedAction, 16> actions;

  // 1) CreateSemaphore: one per (edge-bearing group, first-seen dstOwner) in
  // deterministic order, plus the released seed group.
  for (auto [idx, syncGroup] : llvm::enumerate(dag.groups)) {
    unsigned groupIdx = static_cast<unsigned>(idx);
    if (syncGroup.kind == SyncGroupKind::InitialEmpty) {
      SchedAction a;
      a.kind = SchedAction::Create;
      a.isSeed = true;
      a.released = true;
      a.acquirer = seed ? seed->second : syncGroup.initialOwner;
      a.semaKey = "SEED";
      a.semaLabel = "SEED" + ownerLabel(a.acquirer);
      a.endpoint = "create-scope";
      actions.push_back(std::move(a));
      continue;
    }
    SmallVector<std::optional<PartitionId>, 2> seen;
    for (unsigned edgeIdx : syncGroup.edgeIdxs) {
      std::optional<PartitionId> owner = sp.edges[edgeIdx].dstOwner;
      if (llvm::any_of(seen, [&](auto &o) { return sameOwner(o, owner); }))
        continue;
      seen.push_back(owner);
      SchedAction a;
      a.kind = SchedAction::Create;
      semaFor(groupIdx, &sp.edges[edgeIdx], a.semaKey, a.semaLabel, a.acquirer,
              a.isSeed);
      a.released = false; // non-seed classes are unreleased (M2)
      a.endpoint = "create-scope";
      actions.push_back(std::move(a));
    }
  }

  // 2) Acquire actions from the acquire anchors.
  auto addAcquire = [&](unsigned groupIdx, const SyncEdge *edge, unsigned rk,
                        const std::string &ep) {
    SchedAction a;
    a.kind = SchedAction::Acquire;
    a.rank = rk;
    a.priority = 1;
    a.edgeIdx = findEdgeIndex(sp, edge);
    semaFor(groupIdx, edge, a.semaKey, a.semaLabel, a.acquirer, a.isSeed);
    a.endpoint = ep;
    actions.push_back(std::move(a));
  };
  for (auto &[op, groups] : dag.acquireBeforeOp)
    for (unsigned g : groups) {
      const SyncEdge *e = findEdgeForAnchor(dag.groups[g], sp, dag,
                                            SyncAnchorKind::AcquireBeforeOp, op,
                                            nullptr);
      addAcquire(g, e, rankOf(op, nullptr), endpointDesc(op, nullptr));
    }
  for (auto &[region, groups] : dag.acquireBeforeYield)
    for (unsigned g : groups) {
      const SyncEdge *e = findEdgeForAnchor(dag.groups[g], sp, dag,
                                            SyncAnchorKind::AcquireBeforeYield,
                                            nullptr, region);
      addAcquire(g, e, rankOf(nullptr, region), endpointDesc(nullptr, region));
    }

  // 3) Release actions from the release anchors.
  auto addRelease = [&](const PlannedRelease &pr, unsigned rk,
                        const std::string &ep) {
    SchedAction a;
    a.kind = SchedAction::Release;
    a.rank = rk;
    a.priority = 0;
    const SyncEdge *e = getRepresentativeReleaseEdge(pr, sp);
    a.edgeIdx = findEdgeIndex(sp, e);
    semaFor(pr.groupIdx, e, a.semaKey, a.semaLabel, a.acquirer, a.isSeed);
    a.endpoint = ep;
    actions.push_back(std::move(a));
  };
  for (auto &[op, rels] : dag.releaseBeforeOp)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, rankOf(op, nullptr), "before:" + endpointDesc(op, nullptr));
  for (auto &[op, rels] : dag.releaseAfterOp)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, rankOf(op, nullptr), "after:" + endpointDesc(op, nullptr));
  for (auto &[region, rels] : dag.releaseBeforeYield)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, rankOf(nullptr, region),
                 "before:" + endpointDesc(nullptr, region));
  for (auto &[region, rels] : dag.releaseAfterYield)
    for (const PlannedRelease &pr : rels)
      addRelease(pr, rankOf(nullptr, region),
                 "after:" + endpointDesc(nullptr, region));

  // 4) ThreadToken actions (forward token threading).
  for (Operation *op : dag.threadForOps) {
    SchedAction a;
    a.kind = SchedAction::Thread;
    a.rank = rankOf(op, nullptr);
    a.priority = 3;
    a.semaLabel = "thread-for";
    a.endpoint = endpointDesc(op, nullptr);
    actions.push_back(std::move(a));
  }
  for (Operation *op : dag.threadIfOps) {
    SchedAction a;
    a.kind = SchedAction::Thread;
    a.rank = rankOf(op, nullptr);
    a.priority = 3;
    a.semaLabel = "thread-if";
    a.endpoint = endpointDesc(op, nullptr);
    actions.push_back(std::move(a));
  }

  // CreateSemaphore actions first, then program order, priority, and stable key.
  std::stable_sort(actions.begin(), actions.end(),
                   [](const SchedAction &a, const SchedAction &b) {
                     bool ac = a.kind == SchedAction::Create;
                     bool bc = b.kind == SchedAction::Create;
                     if (ac != bc)
                       return ac; // creates first; keep their insertion order
                     if (ac && bc)
                       return false;
                     if (a.rank != b.rank)
                       return a.rank < b.rank;
                     if (a.priority != b.priority)
                       return a.priority < b.priority;
                     return a.semaKey < b.semaKey;
                   });

  llvm::errs() << "EMIT-SCHEDULE buffer.id=" << dag.resource.first
               << " resourceKey=" << dag.resource.second
               << " actions=" << actions.size()
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
  for (const SchedAction &a : actions) {
    const char *k = a.kind == SchedAction::Create   ? "create "
                    : a.kind == SchedAction::Acquire ? "acquire"
                    : a.kind == SchedAction::Release ? "release"
                                                     : "thread ";
    llvm::errs() << "  " << k << "  sema=" << a.semaLabel;
    if (a.edgeIdx)
      llvm::errs() << " edge=" << *a.edgeIdx;
    if (a.kind == SchedAction::Create)
      llvm::errs() << " released=" << (a.released ? "true" : "false");
    llvm::errs() << "  @ " << a.endpoint << "\n";
    if (a.kind == SchedAction::Acquire) {
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
