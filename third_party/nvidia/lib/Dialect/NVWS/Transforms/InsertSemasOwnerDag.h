#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_

// ---------------------------------------------------------------------------
// Helpers used by the planner.
// ---------------------------------------------------------------------------

namespace {

// Recursive region-ownership planner (v4 §Structured Region Ownership).
// One Planner per ResourcePlan.
struct Planner {
  triton::FuncOp funcOp;
  ResourcePlan &plan;
  const DenseMap<Operation *, unsigned> &rank;
  // Events touching this resource, sorted by program order.
  SmallVector<Operation *> orderedEventOps;

  Planner(triton::FuncOp f, ResourcePlan &p,
          const DenseMap<Operation *, unsigned> &r)
      : funcOp(f), plan(p), rank(r) {}

  // First in-scope event (by program order) anywhere inside `region`.
  // v4 §Debug DAG Dumps scope-barrier rule: an event is in-scope for the
  // body region of a WS-tagged scf.for, but not for ancestor regions
  // outside that for-loop.
  std::optional<PartitionId> firstEventOwnerIn(Region &region) {
    Operation *foundOp = nullptr;
    region.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (plan.useOwner.find(op) == plan.useOwner.end())
        return WalkResult::advance();
      Operation *tagSource = plan.useTagSource.lookup(op);
      if (!isEventInScopeForRegion(tagSource, op, &region))
        return WalkResult::advance();
      foundOp = op;
      return WalkResult::interrupt();
    });
    if (!foundOp) return std::nullopt;
    return plan.useOwner.find(foundOp)->second;
  }

  // Owner of next in-scope event for `contextRegion`, in program order
  // after `op`'s subtree. Used to choose the branchOwner for an scf.if
  // sitting inside `contextRegion`.
  std::optional<PartitionId> nextEventOwnerAfter(Operation *op,
                                                 Region &contextRegion) {
    unsigned cutoff = maxRankInSubtree(op, rank);
    for (Operation *evOp : orderedEventOps) {
      if (rank.lookup(evOp) <= cutoff) continue;
      Operation *tagSource = plan.useTagSource.lookup(evOp);
      if (!isEventInScopeForRegion(tagSource, evOp, &contextRegion))
        continue;
      return plan.useOwner.find(evOp)->second;
    }
    return std::nullopt;
  }

  // Plan `region`. Returns the chosen exit owner. For loop bodies
  // (isLoopBody=true) entry == exit == carried owner.
  std::optional<PartitionId> planRegion(Region &region,
                                        std::optional<PartitionId> entry,
                                        bool isLoopBody = false);
};

std::optional<PartitionId>
Planner::planRegion(Region &region, std::optional<PartitionId> entry,
                    bool isLoopBody) {
  RegionOwnership rec;
  rec.entry = entry;
  rec.carried = isLoopBody;
  bool subtreeHasEvents = false;

  std::optional<PartitionId> current = entry;
  if (isLoopBody) {
    // Loop-body carried owner: pick the first use inside the body. The
    // body's natural last use may differ from this; a handoff back to
    // `current` before scf.yield is required (raw-sync edge, commit 3).
    auto firstInside = firstEventOwnerIn(region);
    if (firstInside) current = firstInside;
    rec.entry = current;
  }

  for (Block &block : region) {
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // v4: scf.if branch owner = post-if next-use in this region;
        // filtered to events in-scope for *this* region (the one
        // containing the scf.if).
        auto postIf = nextEventOwnerAfter(&op, region);
        std::optional<PartitionId> branchOwner = postIf ? postIf : current;

        planRegion(ifOp.getThenRegion(), branchOwner,
                   /*isLoopBody=*/false);
        plan.regionOwners[&ifOp.getThenRegion()].entry = branchOwner;
        plan.regionOwners[&ifOp.getThenRegion()].exit = branchOwner;

        // Always emit a record for the else region (even if empty).
        RegionOwnership elseRec{branchOwner, branchOwner, /*carried=*/false,
                                /*hasEventsInSubtree=*/false};
        plan.regionOwners[&ifOp.getElseRegion()] = elseRec;
        if (!ifOp.getElseRegion().empty()) {
          planRegion(ifOp.getElseRegion(), branchOwner,
                     /*isLoopBody=*/false);
          plan.regionOwners[&ifOp.getElseRegion()].entry = branchOwner;
          plan.regionOwners[&ifOp.getElseRegion()].exit = branchOwner;
        }
        bool thenHas =
            plan.regionOwners[&ifOp.getThenRegion()].hasEventsInSubtree;
        bool elseHas =
            plan.regionOwners[&ifOp.getElseRegion()].hasEventsInSubtree;
        if (thenHas || elseHas) {
          subtreeHasEvents = true;
          current = branchOwner;
        }
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        planRegion(forOp.getRegion(), current, /*isLoopBody=*/true);
        RegionOwnership bodyRec =
            plan.regionOwners.lookup(&forOp.getRegion());
        if (bodyRec.hasEventsInSubtree) {
          subtreeHasEvents = true;
          // v4 scope-barrier rule: a WS-tagged scf.for is a scope
          // boundary. Its body owners are anchored to the for-loop and
          // do not propagate to the parent region. A plain scf.for
          // shares its parent's scope so its body exit owner is
          // in-scope for the parent and may propagate.
          if (!hasWarpSpecializeTag(&op))
            current = bodyRec.exit;
        }
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        if (isLoopBody)
          plan.yieldOwner[yieldOp.getOperation()] = rec.entry;
        continue;
      }
      auto useIt = plan.useOwner.find(&op);
      if (useIt != plan.useOwner.end()) {
        subtreeHasEvents = true;
        // Only update `current` when this event is in scope for the
        // region we're planning. Out-of-scope events still count for
        // transitive presence (subtreeHasEvents) but do not propagate
        // their owner across the WS scope barrier.
        Operation *tagSource = plan.useTagSource.lookup(&op);
        if (isEventInScopeForRegion(tagSource, &op, &region))
          current = useIt->second;
      }
    }
  }

  rec.exit = isLoopBody ? rec.entry : current;
  rec.hasEventsInSubtree = subtreeHasEvents;
  plan.regionOwners[&region] = rec;
  return rec.exit;
}

} // namespace

static ResourcePlan
planResource(triton::FuncOp funcOp, unsigned groupIdx, BufferGroup &group,
             int64_t resourceKey,
             const DenseMap<Operation *, unsigned> &rank) {
  ResourcePlan plan;
  plan.resource = {group.logicalId, resourceKey};
  plan.groupIdx = groupIdx;
  for (auto [idx, member] : llvm::enumerate(group.members))
    if (member.resourceKey == resourceKey)
      plan.memberIndices.push_back(static_cast<unsigned>(idx));

  for (AccessEvent &event : group.events) {
    bool found = false;
    for (auto [tIdx, touch] : llvm::enumerate(event.touches)) {
      if (touch.resourceKey == resourceKey) {
        if (!found) {
          std::optional<PartitionId> owner = getPartitionId(event.op);
          plan.useOwner[event.op] = owner;
          plan.useTagSource[event.op] =
              owner ? getTagSourceOp(event.op) : nullptr;
          plan.useTouchIdx[event.op] = static_cast<unsigned>(tIdx);
          found = true;
        }
        plan.useTouchIdxs[event.op].push_back(static_cast<unsigned>(tIdx));
      }
    }
  }

  Planner planner(funcOp, plan, rank);
  for (AccessEvent &event : group.events)
    if (eventTouchesResource(event, resourceKey))
      planner.orderedEventOps.push_back(event.op);
  llvm::sort(planner.orderedEventOps, [&](Operation *a, Operation *b) {
    return rank.lookup(a) < rank.lookup(b);
  });

  // v4: the function region is never annotated, so we don't seed it with
  // a precomputed entry owner. The planner threads in-scope events
  // (root + intrinsic-tagged) through `current` if any exist; otherwise
  // function-region entry/exit stay nullopt.
  planner.planRegion(funcOp.getBody(), /*entry=*/std::nullopt,
                     /*isLoopBody=*/false);
  return plan;
}


bool regionHasEvents(Region &region, ResourcePlan &plan) {
  auto it = plan.regionOwners.find(&region);
  return it != plan.regionOwners.end() && it->second.hasEventsInSubtree;
}

std::string regionOpLabel(Operation *op, const ResourcePlan &plan) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (hasWarpSpecializeTag(op))
      os << "scf.for (WS, tag=" << *getWarpSpecializeTag(op) << ")";
    else
      os << "scf.for";
  } else if (isa<scf::IfOp>(op)) {
    os << "scf.if";
  }
  auto part = regionOpPartition(op, plan);
  os << " " << ownerStr(op, part);
  return s;
}

void renderEnterRow(unsigned depth, Operation *anchor,
                    std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- ENTER " << ownerStr(anchor, partition)
               << "\n";
}

void renderYieldRow(unsigned depth, Operation *anchor,
                    std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- YIELD " << ownerStr(anchor, partition)
               << "\n";
}

void dumpOwnershipBlock(Block &block, ResourcePlan &plan, BufferGroup &group,
                        unsigned depth);

void dumpOwnershipRegion(Region &region, ResourcePlan &plan,
                         BufferGroup &group, Operation *anchorOp,
                         unsigned depth) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end())
    part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region)
    dumpOwnershipBlock(b, plan, group, depth);
  renderYieldRow(depth, anchorOp, part);
}

void dumpOwnershipBlock(Block &block, ResourcePlan &plan, BufferGroup &group,
                        unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionHasEvents(forOp.getRegion(), plan))
        continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpOwnershipRegion(forOp.getRegion(), plan, group, &op, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionHasEvents(ifOp.getThenRegion(), plan);
      bool elseHas = regionHasEvents(ifOp.getElseRegion(), plan);
      if (!thenHas && !elseHas)
        continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpOwnershipRegion(ifOp.getThenRegion(), plan, group, &op, depth + 2);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpOwnershipRegion(ifOp.getElseRegion(), plan, group, &op, depth + 2);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op))
      continue;
    auto useIt = plan.useOwner.find(&op);
    if (useIt == plan.useOwner.end())
      continue;
    unsigned tIdx = plan.useTouchIdx.lookup(&op);
    AccessEvent *event = findEvent(group, &op);
    if (!event || tIdx >= event->touches.size())
      continue;
    SmallVector<unsigned, 1> fallbackTouchIdx{tIdx};
    auto allTouchIdxs = plan.useTouchIdxs.find(&op);
    ArrayRef<unsigned> touchIdxs =
        allTouchIdxs == plan.useTouchIdxs.end()
            ? ArrayRef<unsigned>(fallbackTouchIdx)
            : ArrayRef<unsigned>(allTouchIdxs->second);
    for (unsigned touchIdx : touchIdxs) {
      if (touchIdx >= event->touches.size())
        continue;
      AccessTouch &touch = event->touches[touchIdx];
      bool reads = hasRead(touch.effect);
      bool writes = hasWrite(touch.effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m" << touch.memberIdx
                   << "  " << op.getName().getStringRef() << "  use "
                   << ownerStr(&op, useIt->second) << "\n";
    }
  }
}


void dumpOwnershipDag(ResourcePlan &plan, BufferGroup &group,
                      mlir::triton::FuncOp funcOp) {
  llvm::errs() << "OWNERSHIP-DAG buffer.id=" << plan.resource.first
               << " resourceKey=" << plan.resource.second << " members:";
  for (unsigned idx : plan.memberIndices)
    llvm::errs() << " m" << idx;
  llvm::errs() << "\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpOwnershipBlock(b, plan, group, /*depth=*/1);
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_
