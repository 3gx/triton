#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_RAW_SYNC_DAG_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_RAW_SYNC_DAG_H_

// ---------------------------------------------------------------------------
// v4 §Dependency Edges / §Raw DAG Edge Model (commit 3, post-revision).
// Per-edge counter model:
//   - one counter per cross-owner edge (initial 0)
//   - source partition releases (counter += 1) after its op
//   - destination partition acquires (counter -= 1, blocks if 0) before
//     its op
//   - no FULL/EMPTY, no ready/done/handoff kind, no S_init
//   - sequential name "S_e<N>" per resource
//
// Edges are derived from version tracking applied to the augmented
// program-order stream (real accesses + virtual ENTER/YIELD rows + region
// op partitions). Concurrent readers of the same version produce no edge
// between themselves. Loop-carry close edges emerge naturally as
// cross-owner edges whose destination is a YIELD virtual row.
// ---------------------------------------------------------------------------

static bool isExplicitOffsetSourcefulTmemSelfContained(BufferGroup &group,
                                                       int64_t resourceKey) {
  if (!group.isTmem())
    return false;
  bool sawMember = false;
  for (BufferMember &member : group.members) {
    if (member.resourceKey != resourceKey)
      continue;
    sawMember = true;
    auto allocOp = dyn_cast<TMEMAllocOp>(member.allocOp);
    if (!allocOp || !allocOp.getSrc() ||
        !member.allocOp->hasAttr(kBufferOffsetAttrName))
      return false;
  }
  if (!sawMember)
    return false;

  for (const AccessEvent &event : group.events) {
    SmallVector<const AccessTouch *, 2> touches;
    collectTouchesForResource(event, resourceKey, touches);
    if (touches.empty())
      continue;
    if (!isa<TMEMAllocOp, TMEMLoadOp>(event.op))
      return false;
    for (const AccessTouch *touch : touches) {
      if (touch->memberIdx >= group.members.size())
        return false;
      auto allocOp =
          dyn_cast<TMEMAllocOp>(group.members[touch->memberIdx].allocOp);
      if (!allocOp || !sameOwner(event.owner, getPartitionId(allocOp)))
        return false;
      if (isa<TMEMAllocOp>(event.op) && !touchWrites(*touch))
        return false;
      if (isa<TMEMLoadOp>(event.op) && !touchReads(*touch))
        return false;
    }
  }
  bool sawOwner = false;
  int previousPartition = -1;
  for (const AccessEvent &event : group.events) {
    if (!eventTouchesResource(event, resourceKey) || !isa<TMEMAllocOp>(event.op))
      continue;
    if (!event.owner || (!sawOwner && event.owner->first != 0) ||
        (sawOwner && event.owner->first < previousPartition))
      return false;
    sawOwner = true;
    previousPartition = event.owner->first;
  }
  return true;
}

static std::string makeEdgeName(unsigned serial) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "S" << serial;
  return s;
}

static RegionOwnership ownershipForAnchor(Operation *op, Region *yieldRegion,
                                          const ResourcePlan &rp) {
  if (yieldRegion) {
    auto it = rp.regionOwners.find(yieldRegion);
    return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
  }
  if (!op)
    return {};
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto it = rp.regionOwners.find(&forOp.getRegion());
    return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto thenIt = rp.regionOwners.find(&ifOp.getThenRegion());
    if (thenIt != rp.regionOwners.end() && thenIt->second.hasEventsInSubtree)
      return thenIt->second;
    auto elseIt = rp.regionOwners.find(&ifOp.getElseRegion());
    return elseIt == rp.regionOwners.end() ? RegionOwnership{} : elseIt->second;
  }
  Region *parentRegion = op->getParentRegion();
  auto it = rp.regionOwners.find(parentRegion);
  return it == rp.regionOwners.end() ? RegionOwnership{} : it->second;
}

// Record an edge anchored at its destination. Both the release and the
// acquire render right before the destination's row at the destination's
// tree depth.
static void recordEdge(SyncPlan &sp, SyncEdge edge) {
  unsigned idx = sp.edges.size();
  if (edge.dstOp)
    sp.beforeOp[edge.dstOp].push_back(idx);
  else if (edge.dstYieldRegion)
    sp.beforeYield[edge.dstYieldRegion].push_back(idx);
  sp.edges.push_back(std::move(edge));
}

// Build the raw-sync DAG via a single unified walk of the OWNERSHIP-DAG.
// State is (writer, readers). Walk every row in program order:
//   - R{P}: if writer cross-partition and P not in readers, emit (writer→P).
//           Add P to readers.
//   - W{P}: close cross-partition readers; if no readers and cross writer,
//           emit handoff; set writer=P, clear readers.
//   - Structural row (scf.for-op / scf.if-op / YIELD body): same as W's
//     close+handoff logic, EXCEPT (a) writer is updated only when an edge
//     actually fires, (b) handoff is skipped when no real access exists
//     downstream (release-into-void suppression).
//
// scf.if: each branch walks from a snapshot; planner guarantees both
// branches converge on the same partition at exit.
//
// This unified rule replaces the prior bifurcated walker+loop-carry pair,
// and is sound by direct trace verification across all 86 commit3 OWNERSHIP
// DAGs.
static SyncPlan buildSyncPlan(BufferGroup &group, const ResourcePlan &rp,
                              triton::FuncOp funcOp) {
  SyncPlan sp;
  sp.resource = rp.resource;
  sp.groupIdx = rp.groupIdx;
  sp.memberIndices = rp.memberIndices;

  // Collect access ops for this resource.
  DenseMap<Operation *, const AccessEvent *> opToEvent;
  for (const AccessEvent &e : group.events)
    if (eventTouchesResource(e, rp.resource.second)) {
      sp.accessOps.insert(e.op);
      opToEvent[e.op] = &e;
    }

  // Program-order rank per op (depth-first pre-order over the function).
  DenseMap<Operation *, unsigned> opRank;
  {
    unsigned ctr = 0;
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) { opRank[op] = ctr++; });
  }
  // Max rank of any access op — used to answer "any access after this row?".
  unsigned maxAccessRank = 0;
  bool anyAccess = false;
  for (const AccessEvent &event : group.events) {
    if (!eventTouchesResource(event, rp.resource.second))
      continue;
    Operation *op = event.op;
    auto it = opRank.find(op);
    if (it == opRank.end()) continue;
    if (!anyAccess || it->second > maxAccessRank) {
      maxAccessRank = it->second;
      anyAccess = true;
    }
  }
  auto hasDownstreamAccess = [&](unsigned cutoff) {
    return anyAccess && maxAccessRank > cutoff;
  };

  // Owner = std::optional<PartitionId>: nullopt represents "root" (an
  // event outside any WS scope, still a valid producer/consumer).
  // State.writerSet distinguishes "no writer has happened yet" (false)
  // from "writer is root or some partition" (true).
  //
  // writerOp / writerYieldRegion identify where the writer became
  // `writer` (a real W access op, a structural row's op for region
  // transitions, or a yield region for body-yield closes). writerEpoch
  // is bumped every time the writer is updated, so edges that share a
  // (srcOwner, srcEpoch) provably came from the same producer event.
  using Owner = std::optional<PartitionId>;
  struct State {
    bool writerSet = false;
    Owner writer;
    Operation *writerOp = nullptr;
    Region *writerYieldRegion = nullptr;
    unsigned writerEpoch = 0;
    // Parallel arrays indexed together: `readers[i]` is the partition,
    // `readerOps[i]` / `readerYields[i]` is where its last access for
    // this resource happened (used as the SRC anchor when a close edge
    // fires from this reader).
    SmallVector<Owner, 4> readers;
    SmallVector<Operation *, 4> readerOps;
    SmallVector<Region *, 4> readerYields;
    SmallVector<unsigned, 4> readerEpochs;
  };

  auto containsReader = [](const State &s, const Owner &p) {
    return llvm::any_of(s.readers, [&](const Owner &q) { return sameOwner(p, q); });
  };

  unsigned serial = 0;
  unsigned writerEpochCtr = 0;

  auto readerIdx = [&](const State &s, const Owner &p) -> int {
    for (unsigned i = 0; i < s.readers.size(); ++i)
      if (sameOwner(s.readers[i], p)) return static_cast<int>(i);
    return -1;
  };

  auto populateEdgeInfo = [&](SyncEdge &edge, SyncEdgeKind kind,
                              unsigned producedEpoch, Owner preOwner,
                              Owner postOwner) {
    auto regionContainsOp = [](Region *region, Operation *op) {
      if (!region || !op)
        return false;
      Region *opRegion = op->getParentRegion();
      return opRegion == region || region->isAncestor(opRegion);
    };
    auto regionContainsRegion = [](Region *region, Region *other) {
      if (!region || !other)
        return false;
      return other == region || region->isAncestor(other);
    };
    edge.kind = kind;
    edge.producedVersion = {rp.resource.second, producedEpoch};
    edge.preOwner = preOwner;
    edge.postOwner = postOwner;
    edge.srcRegionOwner =
        ownershipForAnchor(edge.srcOp, edge.srcYieldRegion, rp);
    edge.dstRegionOwner =
        ownershipForAnchor(edge.dstOp, edge.dstYieldRegion, rp);
    edge.releaseOutsideControlFlow =
        edge.srcOp && isa<scf::ForOp, scf::IfOp>(edge.srcOp);
    edge.acquireOutsideControlFlow =
        edge.dstOp && isa<scf::ForOp, scf::IfOp>(edge.dstOp);
    edge.carrierChoice = CarrierTokenChoice::Destination;
    if (edge.srcOp)
      edge.releaseStageCluster = getStageCluster(edge.srcOp);
    if (edge.dstOp)
      edge.acquireStageCluster = getStageCluster(edge.dstOp);
    edge.asyncPayload = getAsyncPayload(edge.srcOp);
    if (edge.srcOp) {
      auto it = opToEvent.find(edge.srcOp);
      if (it != opToEvent.end()) {
        SmallVector<const AccessTouch *, 4> touches;
        collectTouchesForResource(*it->second, rp.resource.second, touches);
        for (const AccessTouch *touch : touches)
          edge.touches.push_back(*touch);
      }
    }
    if (edge.dstOp) {
      auto it = opToEvent.find(edge.dstOp);
      if (it != opToEvent.end()) {
        SmallVector<const AccessTouch *, 4> touches;
        collectTouchesForResource(*it->second, rp.resource.second, touches);
        for (const AccessTouch *touch : touches)
          edge.touches.push_back(*touch);
      }
    }

    Region *dstRegion =
        edge.dstOp ? edge.dstOp->getParentRegion() : edge.dstYieldRegion;
    if (edge.srcYieldRegion) {
      edge.releaseSiteKind = SyncReleaseSiteKind::AfterOp;
      edge.releaseAfterOp = edge.srcYieldRegion->getParentOp();
      edge.releaseAfterEnterRegion = nullptr;
      return;
    }
    if (dstRegion && sameOwner(edge.srcOwner, edge.dstRegionOwner.entry) &&
        !regionContainsOp(dstRegion, edge.srcOp) &&
        !regionContainsRegion(dstRegion, edge.srcYieldRegion)) {
      edge.releaseSiteKind = SyncReleaseSiteKind::AfterEnter;
      edge.releaseAfterOp = nullptr;
      edge.releaseAfterEnterRegion = dstRegion;
      return;
    }
    Operation *srcAnchor = edge.srcOp;
    Operation *dstAnchor =
        edge.dstOp
            ? edge.dstOp
            : (edge.dstYieldRegion ? edge.dstYieldRegion->getParentOp()
                                   : nullptr);
    Operation *best = srcAnchor;
    for (Operation *parent = srcAnchor ? srcAnchor->getParentOp() : nullptr;
         parent; parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scf::IfOp>(parent))
        continue;
      if (dstAnchor &&
          (parent == dstAnchor || parent->isProperAncestor(dstAnchor)))
        break;
      best = parent;
    }
    edge.releaseSiteKind = SyncReleaseSiteKind::AfterOp;
    edge.releaseAfterOp = best;
    edge.releaseAfterEnterRegion = nullptr;
  };

  // Apply the "close cross-readers + optional handoff" rule. Used by W
  // (isW=true) and structural rows (isW=false). Returns true if any edge
  // was emitted at this row.
  // `consumerGuaranteed`: the destination owner P is reached by a use that is
  // guaranteed to exist even when no access follows this row in static program
  // order — specifically a loop-carry back-edge (this row's YIELD flows to the
  // loop's next-iteration ENTER, whose body has events). An owner change into
  // such a carry is a real handoff and must emit, so it bypasses the
  // static-downstream "release-into-void" suppression below.
  auto emitClose = [&](State &state, const Owner &P, Operation *anchorOp,
                       Region *anchorYieldRegion, bool isW,
                       unsigned anchorRank,
                       bool consumerGuaranteed = false) -> bool {
    SmallVector<unsigned, 4> crossIdxs;
    for (unsigned i = 0; i < state.readers.size(); ++i)
      if (!sameOwner(state.readers[i], P)) crossIdxs.push_back(i);
    bool fired = false;
    if (!crossIdxs.empty()) {
      for (unsigned i : crossIdxs) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        bool sameOwnerReadAfterWrite =
            isW && state.writerSet && sameOwner(state.readers[i], state.writer);
        edge.srcOwner =
            sameOwnerReadAfterWrite ? state.writer : state.readers[i];
        edge.dstOwner = P;
        edge.srcOp = state.readerOps[i];
        edge.srcYieldRegion = state.readerYields[i];
        if (sameOwnerReadAfterWrite) {
          edge.srcEpoch = state.writerEpoch;
        } else {
          // Each reader's last-access is itself a fresh source event for
          // Combine A grouping purposes; assign a unique epoch per
          // contributing reader so cross-reader closes are not falsely
          // collapsed into a ReadyFanout.
          edge.srcEpoch = ++writerEpochCtr;
        }
        if (anchorOp)
          edge.dstOp = anchorOp;
        else
          edge.dstYieldRegion = anchorYieldRegion;
        populateEdgeInfo(edge,
                         sameOwnerReadAfterWrite ? SyncEdgeKind::Handoff
                                                 : SyncEdgeKind::Done,
                         sameOwnerReadAfterWrite ? state.writerEpoch
                                                 : state.readerEpochs[i],
                         edge.srcOwner, P);
        edge.forceFullSemaphore = sameOwnerReadAfterWrite;
        recordEdge(sp, edge);
      }
      state.readers.clear();
      state.readerOps.clear();
      state.readerYields.clear();
      state.readerEpochs.clear();
      fired = true;
    } else if (state.readers.empty() && state.writerSet &&
               !sameOwner(state.writer, P)) {
      // Handoff: W always fires; a structural row fires if a real access
      // exists downstream in program order (release-into-void suppression) OR
      // the destination is reached via a guaranteed loop-carry consumer (op2
      // is the next iteration, at a lower static rank).
      if (isW || consumerGuaranteed || hasDownstreamAccess(anchorRank)) {
        SyncEdge edge;
        edge.name = makeEdgeName(serial++);
        edge.srcOwner = state.writer;
        edge.dstOwner = P;
        edge.srcOp = state.writerOp;
        edge.srcYieldRegion = state.writerYieldRegion;
        edge.srcEpoch = state.writerEpoch;
        if (anchorOp)
          edge.dstOp = anchorOp;
        else
          edge.dstYieldRegion = anchorYieldRegion;
        populateEdgeInfo(edge, SyncEdgeKind::Handoff, state.writerEpoch,
                         state.writer, P);
        recordEdge(sp, edge);
        fired = true;
      }
    }
    if (isW) {
      state.writerSet = true;
      state.writer = P;
      state.writerOp = anchorOp;
      state.writerYieldRegion = anchorYieldRegion;
      state.writerEpoch = ++writerEpochCtr;
      state.readers.clear();
      state.readerOps.clear();
      state.readerYields.clear();
      state.readerEpochs.clear();
    } else if (fired) {
      // Structural row only promotes itself to writer when it actually
      // settled a transition.
      state.writerSet = true;
      state.writer = P;
      state.writerOp = anchorOp;
      state.writerYieldRegion = anchorYieldRegion;
      state.writerEpoch = ++writerEpochCtr;
    }
    return fired;
  };

  auto emitRead = [&](State &state, const Owner &P, Operation *anchorOp) {
    if (state.writerSet && !sameOwner(state.writer, P) &&
        !containsReader(state, P)) {
      SyncEdge edge;
      edge.name = makeEdgeName(serial++);
      edge.srcOwner = state.writer;
      edge.dstOwner = P;
      edge.srcOp = state.writerOp;
      edge.srcYieldRegion = state.writerYieldRegion;
      edge.srcEpoch = state.writerEpoch;
      edge.dstOp = anchorOp;
      populateEdgeInfo(edge, SyncEdgeKind::Ready, state.writerEpoch,
                       state.writer, P);
      recordEdge(sp, edge);
    }
    int idx = readerIdx(state, P);
    unsigned readEpoch = state.writerSet ? state.writerEpoch : ++writerEpochCtr;
    if (idx < 0) {
      state.readers.push_back(P);
      state.readerOps.push_back(anchorOp);
      state.readerYields.push_back(nullptr);
      state.readerEpochs.push_back(readEpoch);
    } else {
      state.readerOps[idx] = anchorOp;
      state.readerYields[idx] = nullptr;
      state.readerEpochs[idx] = readEpoch;
    }
  };

  std::function<void(Region &, State &)> walkRegion;
  walkRegion = [&](Region &region, State &state) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
          auto regIt = rp.regionOwners.find(&forOp.getRegion());
          if (regIt == rp.regionOwners.end() ||
              !regIt->second.hasEventsInSubtree)
            continue; // No events in subtree — skip entirely.
          Owner pp = regIt->second.entry; // may be nullopt (== root partition)
          unsigned rank = opRank.lookup(&op);
          // v4 §Region-Boundary Edges: a root/external producer (nullopt
          // effective owner) entering a warp-specialized loop is
          // *carrier-inherit*, not a handoff. Region entry already orders
          // the producer's writes before every partition, so the loop's
          // entry partition adopts the producer's state with no semaphore
          // edge. Applies uniformly to true-root and annotated-external
          // (both have nullopt effective owner). Without this, the
          // root->{P} edge would emit a bogus handoff (and, for a loop
          // that re-acquires per iteration against a single pre-loop
          // release, deadlock).
          if (hasWarpSpecializeTag(&op) && state.writerSet && !state.writer &&
              pp.has_value()) {
            state.writer = pp;
            state.writerOp = &op;
            state.writerYieldRegion = nullptr;
            state.writerEpoch = ++writerEpochCtr;
            state.readers.clear();
            state.readerOps.clear();
            state.readerYields.clear();
            state.readerEpochs.clear();
          } else {
            // Transition at the scf.for header row.
            emitClose(state, pp, &op, /*anchorYieldRegion=*/nullptr,
                      /*isW=*/false, rank);
          }
          walkRegion(forOp.getRegion(), state);
          // Transition at the body's YIELD row (loop-carry close). op2 is the
          // loop's next-iteration ENTER (body has events by the precondition
          // above), so an owner change here is a real carry handoff and must
          // emit even with nothing after the loop in static order.
          Operation *yieldOp = forOp.getRegion().front().getTerminator();
          unsigned yieldRank = opRank.lookup(yieldOp);
          emitClose(state, pp, /*anchorOp=*/nullptr, &forOp.getRegion(),
                    /*isW=*/false, yieldRank, /*consumerGuaranteed=*/true);
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
          // The scf.if's partition is the entry partition of whichever
          // branch has events. May be nullopt (== root partition).
          auto thenIt = rp.regionOwners.find(&ifOp.getThenRegion());
          auto elseIt = rp.regionOwners.find(&ifOp.getElseRegion());
          bool thenHas = thenIt != rp.regionOwners.end() &&
                         thenIt->second.hasEventsInSubtree;
          bool elseHas = elseIt != rp.regionOwners.end() &&
                         elseIt->second.hasEventsInSubtree;
          if (!thenHas && !elseHas) continue;
          Owner pp = thenHas ? thenIt->second.entry : elseIt->second.entry;
          unsigned rank = opRank.lookup(&op);
          // Transition at the scf.if header row.
          emitClose(state, pp, &op, /*anchorYieldRegion=*/nullptr,
                    /*isW=*/false, rank);
          // Walk each branch from a snapshot of the post-header state.
          State snap = state;
          State thenExit = snap;
          walkRegion(ifOp.getThenRegion(), thenExit);
          {
            Operation *thenYield =
                ifOp.getThenRegion().front().getTerminator();
            unsigned ry = opRank.lookup(thenYield);
            emitClose(thenExit, pp, /*anchorOp=*/nullptr,
                      &ifOp.getThenRegion(), /*isW=*/false, ry);
          }
          State elseExit = snap;
          bool hasElse = !ifOp.getElseRegion().empty();
          if (hasElse) {
            walkRegion(ifOp.getElseRegion(), elseExit);
            Operation *elseYield =
                ifOp.getElseRegion().front().getTerminator();
            unsigned ry = opRank.lookup(elseYield);
            emitClose(elseExit, pp, /*anchorOp=*/nullptr,
                      &ifOp.getElseRegion(), /*isW=*/false, ry);
          }
          // Planner enforces branches converge. Use last walked branch.
          state = hasElse ? elseExit : thenExit;
          continue;
        }
        if (isa<scf::YieldOp>(op)) continue;
        if (!sp.accessOps.contains(&op)) continue;
        auto evIt = opToEvent.find(&op);
        if (evIt == opToEvent.end()) continue;
        const AccessEvent *ev = evIt->second;
        // ev->owner is nullopt for "root" (outside any WS scope); still
        // a valid participating partition.
        const Owner &P = ev->owner;
        bool reads = eventConsumes(*ev, rp.resource.second);
        bool writes = eventProduces(*ev, rp.resource.second);
        unsigned rank = opRank.lookup(&op);
        // For RMW (both reads and writes): apply R first (emits RAW edge
        // from prior writer), then W (closes cross-readers, becomes new
        // writer). For W-only: skip R. For R-only: skip W.
        if (reads) emitRead(state, P, &op);
        if (writes)
          emitClose(state, P, &op, /*anchorYieldRegion=*/nullptr,
                    /*isW=*/true, rank);
      }
    }
  };

  State state;
  walkRegion(funcOp.getBody(), state);

  // -------- Initial writable permit (RAW-SYNC-DAG dump only). ----------
  // v4 plan "Every access begins with an acquire": the first access of a
  // semaphore-managed (>=2 distinct owners) resource acquires the initial
  // writable permit. Cyclic chains reuse the loop-carry back edge into the
  // loop's entry owner; acyclic chains mint a distinct permit released at the
  // terminal (top-level) consumer. This only annotates the dump; the emit is
  // unchanged. Single-owner resources stay bare (no semaphore).
  {
    SmallVector<Owner, 4> distinctOwners;
    Operation *firstAccess = nullptr, *lastAccess = nullptr;
    unsigned firstRank = 0, lastRank = 0;
    for (const AccessEvent &event : group.events) {
      if (!eventTouchesResource(event, rp.resource.second))
        continue;
      Operation *op = event.op;
      const Owner &o = event.owner;
      if (!llvm::any_of(distinctOwners,
                        [&](const Owner &q) { return sameOwner(o, q); }))
        distinctOwners.push_back(o);
      unsigned r = opRank.lookup(op);
      if (!firstAccess || r < firstRank) { firstAccess = op; firstRank = r; }
      if (!lastAccess || r > lastRank) { lastAccess = op; lastRank = r; }
    }
    if (distinctOwners.size() >= 2 && firstAccess) {
      scf::ForOp wsLoop;
      for (Operation *p = firstAccess->getParentOp(); p; p = p->getParentOp())
        if (auto f = dyn_cast<scf::ForOp>(p))
          if (hasWarpSpecializeTag(f)) wsLoop = f;
      Operation *anchorOp = wsLoop ? wsLoop.getOperation() : firstAccess;
      scf::ForOp cycleLoop = wsLoop;
      if (!cycleLoop)
        for (const AccessEvent &event : group.events) {
          if (!eventTouchesResource(event, rp.resource.second))
            continue;
          Operation *op = event.op;
          scf::ForOp outer;
          for (Operation *p = op->getParentOp(); p; p = p->getParentOp())
            if (auto f = dyn_cast<scf::ForOp>(p))
              if (hasWarpSpecializeTag(f)) outer = f;
          if (outer) { cycleLoop = outer; break; }
        }
      std::optional<PartitionId> loopOwner =
          cycleLoop ? regionOpPartition(cycleLoop.getOperation(), rp)
                    : std::nullopt;
      std::string reuseName;
      int reuseIdx = -1;
      if (cycleLoop && loopOwner)
        for (auto [ei, e] : llvm::enumerate(sp.edges))
          if (e.dstYieldRegion == &cycleLoop.getRegion() &&
              sameOwner(e.dstOwner, loopOwner)) {
            reuseName = e.name;
            reuseIdx = (int)ei;
            break;
          }
      sp.initialPermitBeforeOp = anchorOp;
      if (!reuseName.empty()) {
        sp.initialPermitName = reuseName; // cyclic: reuse loop-carry back edge
        sp.initialPermitEdgeIdx = reuseIdx;
      } else {
        sp.initialPermitName = makeEdgeName((unsigned)sp.edges.size()); // mint
        if (lastAccess && lastAccess->getParentOp() == funcOp.getOperation())
          sp.initialPermitReleaseAfterOp = lastAccess; // acyclic terminal
      }
    }
  }

  // ---- Semaphore-identity unification (commit 3) ----
  // Reaching-acquire rule: an access's buffer permit is provided by the nearest
  // acquire on each control-flow path. The loop back-edge means a loop's first
  // carried access is reached by BOTH the acquire entering the loop and the
  // carried (last) acquire of the body — so those semaphores MUST be the same.
  // Walk forward (modelling each loop body's back-edge) and union the semaphores
  // feeding each access. Element ids: 0..edges-1 = edges, edges.size() = seed.
  // No op-kind / read-vs-write / parity assumption.
  {
    unsigned seedId = static_cast<unsigned>(sp.edges.size());
    sp.semaRep.resize(seedId + 1);
    for (unsigned i = 0; i <= seedId; ++i)
      sp.semaRep[i] = i;
    auto unite = [&](unsigned a, unsigned b) {
      a = sp.semaFind(a);
      b = sp.semaFind(b);
      if (a != b)
        sp.semaRep[std::max(a, b)] = std::min(a, b);
    };
    DenseMap<Operation *, SmallVector<unsigned, 2>> acqByOp;
    DenseMap<Region *, SmallVector<unsigned, 2>> acqByYield;
    for (auto [i, e] : llvm::enumerate(sp.edges)) {
      if (e.dstOp)
        acqByOp[e.dstOp].push_back(static_cast<unsigned>(i));
      else if (e.dstYieldRegion)
        acqByYield[e.dstYieldRegion].push_back(static_cast<unsigned>(i));
    }
    // Edges acquired at the same anchor (op or region-yield) are one acquire
    // for that access — unite them and return the canonical id.
    auto uniteAnchor = [&](ArrayRef<unsigned> es) -> unsigned {
      for (unsigned e : es)
        unite(es.front(), e);
      return es.front();
    };
    // Forward walk threading the "current permit" (the most recent acquire). At
    // each loop, the iter_arg carrying this resource has init = the permit
    // entering the loop and yield = the permit live at the body's end; the
    // loop's first carried access is therefore fed by BOTH, so they are the same
    // semaphore — unite(enterPermit, bodyExit). Returns the permit live at the
    // region's end.
    std::function<unsigned(Region &, unsigned)> walk =
        [&](Region &region, unsigned incoming) -> unsigned {
      unsigned current = incoming;
      for (Block &blk : region) {
        for (Operation &op : blk) {
          if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
            bool carriesResource = false;
            forOp.getRegion().walk([&](Operation *o) {
              if (sp.accessOps.contains(o))
                carriesResource = true;
            });
            if (carriesResource) {
              unsigned enter = current;
              auto it = acqByOp.find(&op);
              if (it != acqByOp.end())
                enter = uniteAnchor(it->second);
              unsigned bodyExit = walk(forOp.getRegion(), enter);
              unite(enter, bodyExit); // carry: iter_arg init == yield
              current = bodyExit;
              continue;
            }
          }
          // scf.if: a guarded acquire still produces the permit live after the
          // branch. Descend into both regions with the same incoming permit and
          // unite their exits (they feed the one scf.if result / carried slot),
          // so a re-acquire nested in a branch is unified with the loop carry.
          if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
            unsigned thenExit = walk(ifOp.getThenRegion(), current);
            unsigned elseExit = current;
            if (!ifOp.getElseRegion().empty())
              elseExit = walk(ifOp.getElseRegion(), current);
            unite(thenExit, elseExit);
            current = sp.semaFind(thenExit);
            continue;
          }
          auto it = acqByOp.find(&op);
          if (it != acqByOp.end())
            current = uniteAnchor(it->second);
        }
      }
      auto yit = acqByYield.find(&region);
      if (yit != acqByYield.end())
        current = uniteAnchor(yit->second);
      return current;
    };
    walk(funcOp.getBody(), seedId);
  }
  return sp;
}


void renderAcquireRow(unsigned depth, Operation *anchor, const SyncEdge &edge,
                      StringRef name) {
  llvm::errs() << treePrefix(depth) << "|- a  " << name << "  acquire  "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

void renderReleaseRow(unsigned depth, Operation *anchor, const SyncEdge &edge,
                      StringRef name) {
  llvm::errs() << treePrefix(depth) << "|- r  " << name << "  release  "
               << ownerStr(anchor, edge.srcOwner) << " -> "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

StringRef canonicalSemaName(const SyncPlan &sp, unsigned idx) {
  if (sp.semaRep.empty() || idx >= sp.edges.size())
    return sp.edges[idx].name;
  unsigned rep = sp.semaFind(idx);
  return rep < sp.edges.size() ? StringRef(sp.edges[rep].name)
                               : StringRef(sp.edges[idx].name);
}

std::string ownerSetStr(Operation *anchor,
                        ArrayRef<std::optional<PartitionId>> owners);

struct OptDumpCtx {
  const OptSyncDag *dag;
  DenseSet<unsigned> rendered;
};

struct SyncRenderSite {
  Region *region = nullptr;
  Block *block = nullptr;
  Operation *anchor = nullptr;
  unsigned ordinal = 0;
};

static SyncRenderSite makeOpRenderSite(Operation *op, unsigned ordinal = 0) {
  return {op ? op->getParentRegion() : nullptr, op ? op->getBlock() : nullptr,
          op, ordinal};
}

static SyncRenderSite makeYieldRenderSite(Region &region,
                                          unsigned ordinal = 0) {
  Operation *yieldOp = region.empty() ? nullptr : region.front().getTerminator();
  return {&region, region.empty() ? nullptr : &region.front(), yieldOp,
          ordinal};
}

static bool sameSyncRenderRegion(const SyncRenderSite &lhs,
                                 const SyncRenderSite &rhs) {
  return lhs.region == rhs.region && lhs.block == rhs.block;
}

static bool hasRenderedAccessInSubtree(Operation *op, const SyncPlan &sp) {
  bool hasAccess = false;
  op->walk([&](Operation *nested) -> WalkResult {
    if (sp.accessOps.contains(nested)) {
      hasAccess = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasAccess;
}

static LogicalResult emitStructuralSyncInvariantError(triton::FuncOp funcOp,
                                                      StringRef dagName,
                                                      StringRef reason) {
  InFlightDiagnostic diag = funcOp.emitError("nvws-insert-semas: ");
  diag << dagName << " violates structural release/acquire invariant: "
       << reason;
  return failure();
}

struct SyncInvariantSubject {
  unsigned expectedReleases = 0;
  unsigned expectedAcquires = 0;
  unsigned releases = 0;
  unsigned acquires = 0;
  std::optional<SyncRenderSite> firstSite;
  std::optional<unsigned> maxReleaseOrdinal;
  std::optional<unsigned> minAcquireOrdinal;
};

struct SyncDagStructuralVerifier {
  triton::FuncOp funcOp;
  StringRef dagName;
  SmallVector<unsigned, 8> edgeRenderCount;
  SmallVector<SyncInvariantSubject, 8> subjects;

  SyncDagStructuralVerifier(triton::FuncOp funcOp, StringRef dagName,
                            unsigned edgeCount, unsigned subjectCount)
      : funcOp(funcOp), dagName(dagName), edgeRenderCount(edgeCount, 0),
        subjects(subjectCount) {}

  LogicalResult verifySite(ArrayRef<unsigned> edges, SyncRenderSite site,
                           unsigned edgeCount) {
    if (!site.region || !site.block || !site.anchor)
      return emitStructuralSyncInvariantError(funcOp, dagName,
                                              "invalid rendered sync site");
    for (unsigned edgeIdx : edges) {
      if (edgeIdx >= edgeCount)
        return emitStructuralSyncInvariantError(
            funcOp, dagName, "rendered edge index is out of range");
      ++edgeRenderCount[edgeIdx];
    }
    return success();
  }

  LogicalResult noteSubjectSite(unsigned subjectIdx, SyncRenderSite site) {
    if (subjectIdx >= subjects.size())
      return emitStructuralSyncInvariantError(funcOp, dagName,
                                              "sync subject is out of range");
    SyncInvariantSubject &subject = subjects[subjectIdx];
    if (!subject.firstSite) {
      subject.firstSite = site;
      return success();
    }
    if (!sameSyncRenderRegion(*subject.firstSite, site))
      return emitStructuralSyncInvariantError(
          funcOp, dagName, "release/acquire rows are not in the same region");
    return success();
  }

  LogicalResult noteRelease(unsigned subjectIdx, SyncRenderSite site) {
    if (failed(noteSubjectSite(subjectIdx, site)))
      return failure();
    SyncInvariantSubject &subject = subjects[subjectIdx];
    ++subject.releases;
    if (!subject.maxReleaseOrdinal ||
        site.ordinal > *subject.maxReleaseOrdinal)
      subject.maxReleaseOrdinal = site.ordinal;
    return success();
  }

  LogicalResult noteAcquire(unsigned subjectIdx, SyncRenderSite site) {
    if (failed(noteSubjectSite(subjectIdx, site)))
      return failure();
    SyncInvariantSubject &subject = subjects[subjectIdx];
    ++subject.acquires;
    if (!subject.minAcquireOrdinal ||
        site.ordinal < *subject.minAcquireOrdinal)
      subject.minAcquireOrdinal = site.ordinal;
    return success();
  }

  LogicalResult finish() {
    for (unsigned count : edgeRenderCount)
      if (count != 1)
        return emitStructuralSyncInvariantError(
            funcOp, dagName,
            count == 0 ? "edge is not rendered"
                       : "edge is rendered more than once");
    for (const SyncInvariantSubject &subject : subjects) {
      if (subject.releases != subject.expectedReleases)
        return emitStructuralSyncInvariantError(
            funcOp, dagName, "release row count does not match the DAG");
      if (subject.acquires != subject.expectedAcquires)
        return emitStructuralSyncInvariantError(
            funcOp, dagName, "acquire row count does not match the DAG");
      if (!subject.expectedReleases || !subject.expectedAcquires)
        continue;
      if (!subject.maxReleaseOrdinal || !subject.minAcquireOrdinal)
        return emitStructuralSyncInvariantError(
            funcOp, dagName, "release/acquire rows are incomplete");
      if (*subject.maxReleaseOrdinal > *subject.minAcquireOrdinal)
        return emitStructuralSyncInvariantError(
            funcOp, dagName, "release row is rendered after acquire row");
    }
    return success();
  }
};

static LogicalResult walkRenderedSyncSites(
    const SyncPlan &sp, triton::FuncOp funcOp,
    const std::function<LogicalResult(ArrayRef<unsigned>, SyncRenderSite)> &fn) {
  unsigned ordinal = 0;
  std::function<LogicalResult(Region &)> walkRegion;
  std::function<LogicalResult(Block &)> walkBlock;

  auto visitOpSite = [&](Operation *op, ArrayRef<unsigned> edges) {
    SyncRenderSite site = makeOpRenderSite(op, ordinal++);
    return fn(edges, site);
  };
  auto visitYieldSite = [&](Region &region, ArrayRef<unsigned> edges) {
    SyncRenderSite site = makeYieldRenderSite(region, ordinal++);
    return fn(edges, site);
  };

  walkRegion = [&](Region &region) -> LogicalResult {
    for (Block &block : region)
      if (failed(walkBlock(block)))
        return failure();
    auto yIt = sp.beforeYield.find(&region);
    if (yIt != sp.beforeYield.end())
      if (failed(visitYieldSite(region, yIt->second)))
        return failure();
    return success();
  };

  walkBlock = [&](Block &block) -> LogicalResult {
    for (Operation &op : block) {
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        bool show = hasRenderedAccessInSubtree(forOp.getOperation(), sp);
        if (!show && !sp.beforeYield.count(&forOp.getRegion()))
          continue;
        auto bIt = sp.beforeOp.find(&op);
        if (bIt != sp.beforeOp.end())
          if (failed(visitOpSite(&op, bIt->second)))
            return failure();
        if (failed(walkRegion(forOp.getRegion())))
          return failure();
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (!hasRenderedAccessInSubtree(ifOp.getOperation(), sp))
          continue;
        auto bIt = sp.beforeOp.find(&op);
        if (bIt != sp.beforeOp.end())
          if (failed(visitOpSite(&op, bIt->second)))
            return failure();
        if (failed(walkRegion(ifOp.getThenRegion())))
          return failure();
        if (!ifOp.getElseRegion().empty())
          if (failed(walkRegion(ifOp.getElseRegion())))
            return failure();
        continue;
      }
      if (isa<scf::YieldOp>(op))
        continue;
      if (!sp.accessOps.contains(&op))
        continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        if (failed(visitOpSite(&op, bIt->second)))
          return failure();
    }
    return success();
  };

  return walkRegion(funcOp.getBody());
}

static LogicalResult verifyRawSyncDagStructuralInvariant(const SyncPlan &sp,
                                                         triton::FuncOp funcOp) {
  SyncDagStructuralVerifier verifier(funcOp, "RAW-SYNC-DAG", sp.edges.size(),
                                     sp.edges.size());
  for (SyncInvariantSubject &subject : verifier.subjects) {
    subject.expectedReleases = 1;
    subject.expectedAcquires = 1;
  }
  if (failed(walkRenderedSyncSites(
          sp, funcOp, [&](ArrayRef<unsigned> edges, SyncRenderSite site)
                          -> LogicalResult {
            if (failed(verifier.verifySite(edges, site, sp.edges.size())))
              return failure();
            for (unsigned edgeIdx : edges) {
              if (failed(verifier.noteRelease(edgeIdx, site)))
                return failure();
              if (failed(verifier.noteAcquire(edgeIdx, site)))
                return failure();
            }
            return success();
          })))
    return failure();
  return verifier.finish();
}

void printEdgesAt(SmallVector<unsigned, 2> *edgeIdxs, SyncPlan &sp,
                  Operation *anchor, unsigned depth,
                  OptDumpCtx *octx = nullptr) {
  if (!edgeIdxs)
    return;
  if (!octx) {
    for (unsigned idx : *edgeIdxs) {
      StringRef nm = canonicalSemaName(sp, idx);
      renderReleaseRow(depth, anchor, sp.edges[idx], nm);
      renderAcquireRow(depth, anchor, sp.edges[idx], nm);
    }
    return;
  }
  const OptSyncDag &dag = *octx->dag;
  SmallVector<unsigned, 2> faninHere;
  for (unsigned idx : *edgeIdxs) {
    const SyncEdge &edge = sp.edges[idx];
    unsigned gi = dag.edgeToGroup[idx];
    const SyncGroup &g = dag.groups[gi];
    if (g.kind == SyncGroupKind::ReadyFanout) {
      StringRef nm = canonicalSemaName(sp, g.edgeIdxs.front());
      if (octx->rendered.insert(gi).second) {
        SmallVector<std::optional<PartitionId>, 4> dsts;
        for (unsigned ei : g.edgeIdxs)
          dsts.push_back(sp.edges[ei].dstOwner);
        llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                     << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].srcOwner)
                     << " -> " << ownerSetStr(anchor, dsts) << "\n";
      }
      llvm::errs() << treePrefix(depth) << "|- a  " << nm << "  acquire  "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
    } else if (g.kind == SyncGroupKind::DoneFanin) {
      StringRef nm = canonicalSemaName(sp, idx);
      llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                   << ownerStr(anchor, edge.srcOwner) << " -> "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
      if (!llvm::is_contained(faninHere, gi))
        faninHere.push_back(gi);
    } else {
      StringRef nm = canonicalSemaName(sp, idx);
      llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                   << ownerStr(anchor, edge.srcOwner) << " -> "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
      llvm::errs() << treePrefix(depth) << "|- a  " << nm << "  acquire  "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
    }
  }
  for (unsigned gi : faninHere) {
    if (!octx->rendered.insert(gi).second)
      continue;
    const SyncGroup &g = dag.groups[gi];
    SmallVector<std::optional<PartitionId>, 4> srcs;
    for (unsigned ei : g.edgeIdxs)
      srcs.push_back(sp.edges[ei].srcOwner);
    llvm::errs() << treePrefix(depth) << "|- a  "
                 << canonicalSemaName(sp, g.edgeIdxs.front())
                 << "  acquire  pending=" << ownerSetStr(anchor, srcs) << "  "
                 << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].dstOwner)
                 << "\n";
  }
}

void dumpRawSyncBlock(Block &block, SyncPlan &sp, const ResourcePlan &plan,
                      BufferGroup &group, unsigned depth,
                      OptDumpCtx *octx = nullptr);

void dumpRawSyncRegion(Region &region, SyncPlan &sp, const ResourcePlan &plan,
                       BufferGroup &group, Operation *anchorOp, unsigned depth,
                       OptDumpCtx *octx = nullptr) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end())
    part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region)
    dumpRawSyncBlock(b, sp, plan, group, depth, octx);
  auto yIt = sp.beforeYield.find(&region);
  if (yIt != sp.beforeYield.end())
    printEdgesAt(&yIt->second, sp, anchorOp, depth, octx);
  renderYieldRow(depth, anchorOp, part);
}

void dumpRawSyncBlock(Block &block, SyncPlan &sp, const ResourcePlan &plan,
                      BufferGroup &group, unsigned depth, OptDumpCtx *octx) {
  for (Operation &op : block) {
    if (&op == sp.initialPermitBeforeOp && !sp.initialPermitName.empty()) {
      StringRef entryName = sp.initialPermitName;
      if (!sp.semaRep.empty()) {
        unsigned seedId = static_cast<unsigned>(sp.edges.size());
        unsigned rep = sp.semaFind(seedId);
        if (rep < sp.edges.size())
          entryName = sp.edges[rep].name;
      }
      llvm::errs() << treePrefix(depth) << "|- a  " << entryName
                   << "  acquire  root\n";
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      bool show = false;
      forOp.walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show && !sp.beforeYield.count(&forOp.getRegion()))
        continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpRawSyncRegion(forOp.getRegion(), sp, plan, group, &op, depth + 1,
                        octx);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool show = false;
      ifOp->walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show)
        continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpRawSyncRegion(ifOp.getThenRegion(), sp, plan, group, &op, depth + 2,
                        octx);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpRawSyncRegion(ifOp.getElseRegion(), sp, plan, group, &op,
                          depth + 2, octx);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op))
      continue;
    if (!sp.accessOps.contains(&op))
      continue;
    auto bIt = sp.beforeOp.find(&op);
    if (bIt != sp.beforeOp.end())
      printEdgesAt(&bIt->second, sp, &op, depth, octx);
    AccessEvent *event = findEvent(group, &op);
    SmallVector<const AccessTouch *, 4> touches;
    if (event)
      collectTouchesForResource(*event, sp.resource.second, touches);
    for (const AccessTouch *touch : touches) {
      bool reads = hasRead(touch->effect);
      bool writes = hasWrite(touch->effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
    if (&op == sp.initialPermitReleaseAfterOp && !sp.initialPermitName.empty())
      llvm::errs() << treePrefix(depth) << "|- r  " << sp.initialPermitName
                   << "  release  root\n";
  }
}


void dumpRawSyncDag(SyncPlan &sp, const ResourcePlan &plan, BufferGroup &group,
                    mlir::triton::FuncOp funcOp) {
  llvm::errs() << "RAW-SYNC-DAG buffer.id=" << sp.resource.first
               << " resourceKey=" << sp.resource.second << " edges="
               << sp.edges.size() << "\n";
  if (!sp.semaRep.empty()) {
    unsigned seedId = static_cast<unsigned>(sp.edges.size());
    SmallVector<SmallVector<unsigned, 2>, 4> classes(sp.semaRep.size());
    for (unsigned i = 0; i < sp.semaRep.size(); ++i)
      classes[sp.semaFind(i)].push_back(i);
    bool any = false;
    for (auto &cls : classes)
      if (cls.size() >= 2)
        any = true;
    if (any) {
      llvm::errs() << "|  semaphore-classes:";
      for (auto &cls : classes) {
        if (cls.size() < 2)
          continue;
        llvm::errs() << " {";
        bool first = true;
        for (unsigned i : cls) {
          llvm::errs() << (first ? "" : "=")
                       << (i == seedId ? std::string("SEED")
                                       : sp.edges[i].name);
          first = false;
        }
        llvm::errs() << "}";
      }
      llvm::errs() << "\n";
    }
  }
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1);
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_RAW_SYNC_DAG_H_
