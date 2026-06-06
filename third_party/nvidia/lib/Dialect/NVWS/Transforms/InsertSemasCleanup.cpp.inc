// ---------------------------------------------------------------------------
// EMIT-SCHEDULE dump (commit 4.5, §6.7). Stage-2 diagnostic only: it enumerates
// the mechanical emit plan (CreateSemaphore / Acquire / Release / ThreadToken)
// directly from the completed OPT-SYNC-DAG anchors, resolving each action's
// semaId by §6.4 — InitialEmpty resolves to the single released seed; every
// other action is keyed (groupIdx, edge.dstOwner). No IR is mutated. The dump
// exposes the per-semaphore acquirer set so M3 (single concrete acquirer, with
// optional root) is inspectable in the plan, before emission is wired.
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
  unsigned rank = 0;          // program-order rank of the endpoint
  unsigned priority = 0;      // §6.7 tie-break: Release<Acquire<Buffer<Thread
  std::string semaKey;        // identity key for aggregation
  std::string semaLabel;      // human label
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
  // §6.4 semaId resolution. The seed is the single releasedSemaphores entry.
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

  // 1) CreateSemaphore — one per (edge-bearing group, first-seen dstOwner) in
  //    §6.5 deterministic order, plus the seed for the InitialEmpty group.
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

  // 4) ThreadToken actions (forward token threading, §6.6).
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

  // §6.7 ordering: CreateSemaphore first (creation order), then by program-order
  // rank, tie-broken by priority then stable key.
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
        llvm::errs() << (i == seedId ? "SEED" : ("S" + std::to_string(i))) << " ";
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
