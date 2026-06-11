#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_

// Stage 3 of nvws-insert-semas: SYNC-DAG (spec section 5; plan commit 3).
//
// One ownership walk over the OWNER-DAG plays the token game per piece and
// records every handoff as an edge; edges are deduped (payload union),
// grouped by destination (fan-in), materialized as Acquire/Release nodes
// spliced into the chains, then completed with per-component entry
// acquires, carrier crossings (threading facts ON the For/If nodes),
// requiredParts clone sets, and the BackingPlan. Pure analysis: the IR is
// not touched.
//
// Walk rules (spec section 5.1):
//   1. W by p: edge from every holder != p; piece becomes Exclusive(p).
//   2. R by p: producer/holder reread updates in place; a new reader takes
//      an edge from the current primary holder and joins the readers.
//   3. Same-owner touches move lastRow/lastPayload, no edge.
//   4. EXIT closes in-body holders != carried owner — except holders the
//      carried owner already synchronized behind (transitive-sync skip),
//      and only when load-bearing (under a loop, or the piece is touched
//      later) — never as drains.
//   5. Region bodies walk FRESH local games: Exclusive(carried owner)
//      seeded at ENTER, versionProducer imported from the parent game, and
//      the payload seed IMPORTED when the carried owner is the parent
//      game's producer (rule A makes producer-bracketed branches the
//      normal conditional shape — their Enter-sourced release must carry
//      the producer's async payload).
//   6. A region row is one super-node touch per piece in the parent game
//      (touch first, then recurse); WS-For rows ADOPT root-held pieces
//      without an edge; after recursion the region row's holder carries
//      its games' final payloads (union for If).

// ---------------------------------------------------------------------------
// Walk state.
// ---------------------------------------------------------------------------
struct HolderRec {
  Owner owner;
  Node *lastRow = nullptr;
  SmallVector<AsyncOp, 1> lastPayloads;
  // Partitions that have taken an edge FROM this holder since lastRow:
  // a destination already in this set is TRANSITIVELY synchronized with
  // the holder (its acquire happened after the holder's work), so further
  // edges to it are redundant — the old token-transfer model's behavior,
  // recovered inside the N-readers model. Cleared whenever lastRow moves.
  SmallVector<int64_t, 2> syncedBehind;
};

struct PieceGame {
  bool live = false;
  Owner versionProducer; // current version's producer (root = nullopt)
  SmallVector<HolderRec, 2> holders; // producer and/or readers, join order
};

// std::map: deterministic piece order without sorting at every step.
struct ChainState {
  std::map<PieceId, PieceGame> games;
};

struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  SmallVector<AsyncOp, 1> payloads;
  SmallVector<PieceId, 2> pieces;
};

struct SyncCtx {
  SmallVector<EdgeRec> edges; // walk order
};

static int64_t ownerKey(const Owner &o) {
  if (!o)
    return -1;
  return (static_cast<int64_t>(o->second) << 32) |
         static_cast<uint32_t>(o->first);
}

static HolderRec *findHolder(PieceGame &gm, const Owner &who) {
  for (HolderRec &h : gm.holders)
    if (sameOwner(h.owner, who))
      return &h;
  return nullptr;
}

static void unionPayloads(SmallVector<AsyncOp, 1> &into,
                          const SmallVector<AsyncOp, 1> &from) {
  for (AsyncOp p : from)
    if (!llvm::is_contained(into, p))
      into.push_back(p);
  llvm::sort(into, [](AsyncOp a, AsyncOp b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
}

// ---------------------------------------------------------------------------
// One touch of one piece on one game (rules 1-3 + bottom/adoption).
// `wsAdopt`: the toucher is a WS-tagged For row — root-held pieces are
// adopted (state transition, no edge): outside the loop only the root
// stream exists, and the loop's first toucher inherits root's data without
// sync (program order: the launch).
// ---------------------------------------------------------------------------
// `force`: WAVE LOCALITY (spec rule, user ruling 10jun26) — the toucher's
// wave is closed (the carrier moved to another partition), so every edge
// elision is suspended: the touch MUST take a fresh edge. `waveSrc` is the
// wave owner's last chain row (the baton handoff site) used as fallback
// source when the piece's own holders provide none.
static void applyTouch(ChainState &st, PieceId p, const Owner &who,
                       Effect effect, Node *row,
                       const SmallVector<AsyncOp, 1> &rowPayloads,
                       SyncCtx &ctx, bool wsAdopt, bool force = false,
                       Node *waveSrc = nullptr, Owner waveOwner = Owner(),
                       const SmallVector<AsyncOp, 1> &wavePay = {}) {
  PieceGame &gm = st.games[p];
  if (!gm.live) { // first toucher in an unseeded (function-level) game
    gm.live = true;
    gm.versionProducer = who;
    gm.holders.assign(1, HolderRec{who, row, rowPayloads, {}});
    return;
  }
  if (effect == Effect::W) {
    bool edged = false;
    for (HolderRec &h : gm.holders) {
      if (sameOwner(h.owner, who))
        continue;
      if (wsAdopt && !h.owner.has_value())
        continue; // adoption: no edge from a root holder
      if (!force && llvm::is_contained(h.syncedBehind, ownerKey(who)))
        continue; // transitively synchronized — edge redundant
      ctx.edges.push_back(
          EdgeRec{h.lastRow, row, h.owner, who, h.lastPayloads, {p}});
      edged = true;
    }
    if (force && !edged && waveSrc && !sameOwner(waveOwner, who))
      ctx.edges.push_back(
          EdgeRec{waveSrc, row, waveOwner, who, wavePay, {p}});
    gm.holders.assign(1, HolderRec{who, row, rowPayloads, {}});
    gm.versionProducer = who;
    return;
  }
  // R
  if (HolderRec *h = findHolder(gm, who)) { // reread (producer or reader)
    if (force) { // wave closed: the reread must re-acquire
      HolderRec *primary = findHolder(gm, gm.versionProducer);
      if (primary && !sameOwner(primary->owner, who)) {
        ctx.edges.push_back(EdgeRec{primary->lastRow, row, primary->owner,
                                    who, primary->lastPayloads,
                                    {p}});
        primary->syncedBehind.push_back(ownerKey(who));
      } else if (waveSrc && !sameOwner(waveOwner, who)) {
        ctx.edges.push_back(
            EdgeRec{waveSrc, row, waveOwner, who, wavePay, {p}});
      }
    }
    h->lastRow = row;
    h->lastPayloads = rowPayloads;
    h->syncedBehind.clear(); // lastRow moved
    return;
  }
  HolderRec *primary = findHolder(gm, gm.versionProducer);
  if (!primary)
    primary = &gm.holders.front();
  if (!(wsAdopt && !primary->owner.has_value())) {
    ctx.edges.push_back(EdgeRec{primary->lastRow, row, primary->owner, who,
                                primary->lastPayloads,
                                {p}});
    primary->syncedBehind.push_back(ownerKey(who));
  }
  gm.holders.push_back(HolderRec{who, row, rowPayloads, {}});
}

// ---------------------------------------------------------------------------
// The walk. Returns, per piece of THIS chain's footprint, the carried
// owner's final lastPayloads — the payload a region row's outgoing release
// carries (rule 6).
// ---------------------------------------------------------------------------
// Does any later row (after `fromRow` in its own chain, then ancestor
// chains upward) touch the piece? Used to suppress EXIT-close drains: a
// close at an if-branch EXIT is load-bearing only when the piece is used
// again (the re-anchored owner is the future release's ordering witness);
// with no future use it is a pure drain — the old corpus bans those
// (no_loop_exit_drain).
static bool rowTouchesPiece(GroupDag &g, Node *n, PieceId piece) {
  if (n->kind == Node::Access) {
    for (const Touch &t : n->touches)
      for (PieceId p : g.pieceTable.footprint[t.member])
        if (p == piece)
          return true;
    return false;
  }
  if (n->kind == Node::For || n->kind == Node::If)
    return n->pieceInfo.count(piece) > 0;
  return false;
}

static bool pieceTouchedAfter(GroupDag &g, Node *regionRow, PieceId piece) {
  for (Node *r = regionRow; r && r->kind != Node::Func;) {
    for (Node *m = r->next; m; m = m->next)
      if (rowTouchesPiece(g, m, piece))
        return true;
    r = r->parent;
  }
  return false;
}

static std::map<PieceId, SmallVector<AsyncOp, 1>>
walkChain(GroupDag &g, Node *head, ChainState &st, SyncCtx &ctx,
          bool underFor) {
  // Carried owner per piece (from the ENTER seed; empty for the top chain).
  std::map<PieceId, Owner> carried;
  if (head->kind == Node::Enter)
    for (auto &[p, pi] : sortedPieceInfo(head))
      carried[p] = pi.owner;

  // WAVE LOCALITY state (spec rule): the carrier owner PER COMPONENT —
  // the carrier/token is per component (stage-1 pieceComp), so two
  // disjoint buffer streams in one group track independent waves.
  // valid=false means "no constraint yet" (chain start / mixed owners).
  struct WaveSt {
    Owner owner;
    bool valid = false;
    Node *lastRow = nullptr;
    SmallVector<AsyncOp, 1> pay;
  };
  std::map<CompId, WaveSt> wave;
  auto compOf = [&](PieceId p) { return g.pieceTable.pieceComp[p]; };
  if (head->kind == Node::Enter)
    for (auto &[p, pi] : sortedPieceInfo(head)) {
      WaveSt &w = wave[compOf(p)];
      if (!w.lastRow) {
        w.owner = pi.owner;
        w.valid = true;
        w.lastRow = head;
        w.pay.assign(1, AsyncOp::NONE);
      } else if (!sameOwner(w.owner, pi.owner))
        w.valid = false;
    }

  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
      break; // seeding happened at chain entry
    case Node::Access: {
      // Per-piece effect of this row (W wins across touches).
      std::map<PieceId, Effect> eff;
      for (const Touch &t : n->touches)
        for (PieceId p : g.pieceTable.footprint[t.member]) {
          auto it = eff.find(p);
          if (it == eff.end())
            eff.emplace(p, t.effect);
          else
            it->second = joinEffect(it->second, t.effect);
        }
      for (auto &[p, e] : eff) {
        // Payload = the op's completion mechanism, REGARDLESS of R/W: an
        // MMA reading its operands is an async reader — its release must
        // gate on tc5mma completion. Synchronous loads map to NONE
        // through the same table (spec section 1.2).
        SmallVector<AsyncOp, 1> pay;
        pay.push_back(asyncPayloadOf(n->op));
        WaveSt &w = wave[compOf(p)];
        bool force = w.valid && n->owner.has_value() &&
                     w.owner.has_value() && !sameOwner(w.owner, n->owner);
        applyTouch(st, p, n->owner, e, n, pay, ctx, /*wsAdopt=*/false,
                   force, w.lastRow, w.owner, w.pay);
        if (n->owner.has_value()) {
          w.owner = n->owner;
          w.valid = true;
          w.lastRow = n;
          w.pay.assign(1, asyncPayloadOf(n->op));
        }
      }
      break;
    }
    case Node::For:
    case Node::If: {
      bool wsAdopt =
          n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op);
      auto infos = sortedPieceInfo(n);
      // Pre-touch producer snapshot: the seed-import facts (rule 5).
      std::map<PieceId, std::pair<Owner, SmallVector<AsyncOp, 1>>> preProd;
      for (auto &[p, pi] : infos) {
        auto it = st.games.find(p);
        if (it == st.games.end() || !it->second.live)
          continue;
        PieceGame &gm = it->second;
        HolderRec *ph = findHolder(gm, gm.versionProducer);
        SmallVector<AsyncOp, 1> pay;
        if (ph)
          pay = ph->lastPayloads;
        else
          pay.push_back(AsyncOp::NONE);
        preProd.emplace(p, std::make_pair(gm.versionProducer, pay));
      }
      // 1. Super-node touch on the parent game (destination edges first).
      for (auto &[p, pi] : infos) {
        SmallVector<AsyncOp, 1> none;
        none.push_back(AsyncOp::NONE); // placeholder; replaced below
        applyTouch(st, p, pi.owner, pi.effect, n, none, ctx, wsAdopt);
      }
      // 2. Recurse into the children with locally seeded games.
      std::map<PieceId, SmallVector<AsyncOp, 1>> unionRet;
      for (Node *childHead : n->children) {
        ChainState child;
        for (auto &[p, pi] : sortedPieceInfo(childHead)) {
          PieceGame gm;
          gm.live = true;
          Owner childCarried = pi.owner;
          auto pre = preProd.find(p);
          gm.versionProducer =
              pre != preProd.end() ? pre->second.first : childCarried;
          SmallVector<AsyncOp, 1> seedPay;
          if (pre != preProd.end() &&
              sameOwner(childCarried, pre->second.first))
            seedPay = pre->second.second; // payload-seed IMPORT
          else
            seedPay.push_back(AsyncOp::NONE); // transitivity witness
          gm.holders.assign(1, HolderRec{childCarried, childHead, seedPay, {}});
          child.games.emplace(p, std::move(gm));
        }
        auto ret = walkChain(g, childHead, child, ctx,
                             underFor || n->kind == Node::For);
        for (auto &[p, pay] : ret)
          unionPayloads(unionRet[p], pay);
      }
      // 3. The region row's holder now carries its games' final payloads.
      for (auto &[p, pi] : infos) {
        PieceGame &gm = st.games[p];
        if (HolderRec *h = findHolder(gm, pi.owner)) {
          auto it = unionRet.find(p);
          if (it != unionRet.end())
            h->lastPayloads = it->second;
        }
      }
      // WAVE LOCALITY: after the region each component's carrier is back
      // with its carried owner (the EXIT regain). Mixed owners per
      // component reset that component's wave.
      {
        std::map<CompId, std::pair<Owner, bool>> ro; // owner, uniform
        for (auto &[p, pi] : infos) {
          auto [it, fresh] =
              ro.try_emplace(compOf(p), std::make_pair(pi.owner, true));
          if (!fresh && !sameOwner(it->second.first, pi.owner))
            it->second.second = false;
        }
        for (auto &[c, ou] : ro) {
          WaveSt &w = wave[c];
          if (ou.second && ou.first.has_value()) {
            w.owner = ou.first;
            w.valid = true;
            w.lastRow = n;
            w.pay.assign(1, AsyncOp::NONE);
          } else {
            w.valid = false;
          }
        }
      }
      break;
    }
    case Node::Exit: {
      // Rule 4: close every in-body holder != carried owner — but ONLY
      // when the close is load-bearing: inside any loop (the recurrence
      // reaches back every iteration), or when the piece has a later
      // touch in an ancestor chain (the re-anchored owner becomes the
      // future release's ordering witness). Otherwise it is a drain —
      // never synthesized (no_loop_exit_drain).
      for (auto &[p, pi] : sortedPieceInfo(n)) {
        auto it = st.games.find(p);
        if (it == st.games.end())
          continue;
        PieceGame &gm = it->second;
        bool needed = underFor || pieceTouchedAfter(g, n->parent, p);
        if (needed)
          for (HolderRec &h : gm.holders) {
            if (sameOwner(h.owner, pi.owner))
              continue;
            if (llvm::is_contained(h.syncedBehind, ownerKey(pi.owner)))
              continue; // carried owner already synchronized — no close
            ctx.edges.push_back(
                EdgeRec{h.lastRow, n, h.owner, pi.owner, h.lastPayloads, {p}});
          }
        // The carried owner re-holds exclusively past the EXIT.
        HolderRec keep;
        if (HolderRec *ch = findHolder(gm, pi.owner))
          keep = *ch;
        else
          keep = HolderRec{pi.owner, n, {AsyncOp::NONE}, {}};
        gm.holders.assign(1, keep);
        gm.versionProducer = pi.owner;
      }
      break;
    }
    case Node::Acquire:
    case Node::Release:
    case Node::Func:
      break; // not present during the walk
    }
  }

  // Final payloads per piece: the carried owner's holder at chain end
  // ({NONE} if it was displaced — its pre-EXIT acquire is the witness).
  std::map<PieceId, SmallVector<AsyncOp, 1>> result;
  for (auto &[p, who] : carried) {
    auto it = st.games.find(p);
    SmallVector<AsyncOp, 1> pay;
    if (it != st.games.end())
      if (HolderRec *h = findHolder(it->second, who))
        pay = h->lastPayloads;
    if (pay.empty())
      pay.push_back(AsyncOp::NONE);
    result.emplace(p, pay);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Chain splicing for injected sync nodes.
// ---------------------------------------------------------------------------
static void spliceBefore(Node *node, Node *before) {
  node->parent = before->parent;
  node->prev = before->prev;
  node->next = before;
  if (before->prev)
    before->prev->next = node;
  else if (node->parent) // chain head: repoint the parent's children slot
    for (Node *&slot : node->parent->children)
      if (slot == before)
        slot = node;
  before->prev = node;
}

static void spliceAfter(Node *node, Node *after) {
  node->parent = after->parent;
  node->next = after->next;
  node->prev = after;
  if (after->next)
    after->next->prev = node;
  after->next = node;
}


// ---------------------------------------------------------------------------
// TRANSITIVE REDUCTION (spec section; user ruling 10jun26) + closure
// verifier. Pay-for-play: drop same-chain Access-row edges whose ordering
// is already implied by kept hard waits plus per-partition program order.
// ---------------------------------------------------------------------------
namespace {
using SyncVec = std::map<int64_t, unsigned>; // partitionKey -> row index

struct ChainIndex {
  DenseMap<Node *, unsigned> idx; // row -> position within its chain
  DenseMap<Node *, Node *> chainOf; // row -> chain head
};

static void indexChains(Node *head, ChainIndex &ci) {
  unsigned i = 0;
  for (Node *n = head; n; n = n->next) {
    ci.idx[n] = i++;
    ci.chainOf[n] = head;
    if (n->kind == Node::For || n->kind == Node::If)
      for (Node *child : n->children)
        indexChains(child, ci);
  }
}

static bool covers(const SyncVec &v, int64_t key, unsigned srcIdx) {
  auto it = v.find(key);
  return it != v.end() && it->second >= srcIdx;
}

// One sweep over a chain with the given edge set (drop[i] marks edges to
// ignore). When `reduce` is set, fills drop[] for newly-implied edges;
// otherwise verifies that every edge in `check` is covered.
static LogicalResult sweepChain(GroupDag &g, Node *head, ChainIndex &ci,
                                SmallVector<EdgeRec> &edges,
                                std::vector<bool> &drop, bool reduce,
                                ArrayRef<unsigned> checkIdxs) {
  // Edges grouped by destination row, original order preserved.
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  for (auto [i, e] : llvm::enumerate(edges))
    if (ci.chainOf.lookup(e.src) == head && ci.chainOf.lookup(e.dst) == head)
      atDst[e.dst].push_back(i);
  // Latest source first: its inherited snapshot subsumes earlier ones, so
  // earlier-source edges at the same destination become provably implied.
  for (auto &[d, v] : atDst)
    llvm::stable_sort(v, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snapAtRow; // source snapshots
  // WAVE GUARD: ordering-implied is not token-allowed — an edge may be
  // dropped only when the destination's wave is already open (carrier
  // with Q via a kept acquire). Track the wave owner per component.
  std::map<CompId, int64_t> waveOf;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner)
        waveOf[g.pieceTable.pieceComp[pc]] = ownerKey(pi.owner);
  auto compOfEdge = [&](const EdgeRec &e) {
    return g.pieceTable.pieceComp[e.pieces.front()];
  };
  auto ownerOfRow = [&](Node *n) -> Owner {
    return n->kind == Node::Exit ? Owner() : n->owner;
  };
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        EdgeRec &e = edges[ei];
        if (!e.srcOwner || !e.dstOwner || e.src->kind != Node::Access)
          continue; // region/root endpoints: never reduced
        int64_t sk = ownerKey(e.srcOwner), dk = ownerKey(e.dstOwner);
        unsigned srcIdx = ci.idx.lookup(e.src);
        bool implied = covers(behind[dk], sk, srcIdx);
        bool waveOpen = waveOf.count(compOfEdge(e)) &&
                        waveOf[compOfEdge(e)] == dk;
        if (reduce && !drop[ei] && implied && waveOpen &&
            e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        if (drop[ei])
          continue;
        waveOf[compOfEdge(e)] = dk; // kept acquire opens Q's wave
        // Kept acquire: dst inherits the source's snapshot + the edge.
        SyncVec &dv = behind[dk];
        auto snap = snapAtRow.find(e.src);
        if (snap != snapAtRow.end())
          for (auto &[k, v] : snap->second)
            if (!covers(dv, k, v))
              dv[k] = std::max(dv[k], v);
        dv[sk] = std::max(dv[sk], srcIdx);
      }
    if (Owner o = ownerOfRow(n)) {
      behind[ownerKey(o)][ownerKey(o)] = ci.idx.lookup(n);
      snapAtRow[n] = behind[ownerKey(o)];
    }
  }
  // Verification mode: every checked (dropped) edge must now be implied.
  if (!reduce)
    for (unsigned ei : checkIdxs) {
      EdgeRec &e = edges[ei];
      if (ci.chainOf.lookup(e.src) != head)
        continue;
      if (e.dst->kind == Node::Exit)
        continue; // back-edge closes are re-verified by sweepTraversalClosure
      if (!covers(behind[ownerKey(e.dstOwner)], ownerKey(e.srcOwner),
                  ci.idx.lookup(e.src)))
        return (e.dst->op ? e.dst->op : g.root->op)
            ->emitError("nvws-insert-semas: transitive-reduction closure "
                        "violation: dropped edge is not implied by the "
                        "final edge set");
    }
  return success();
}


// Phase B: traversal-closure for one loop-body chain (spec). Sweeps the
// forward — the sweep continues over a SECOND TRAVERSAL; per-partition
// program order is sequential through repeated traversals so vectors
// carry forward. An EXIT-close edge drops iff at the closing owner's
// first touch of each closed piece in the following traversal the source
// is covered AND the wave is already open via a kept in-body acquire.
// The carrier close (acquirer == first wave owner, the yielded final)
// never drops. `reduce` toggles reduction vs re-verification.
static LogicalResult sweepTraversalClosure(GroupDag &g, Node *head, ChainIndex &ci,
                                   SmallVector<EdgeRec> &edges,
                                   std::vector<bool> &drop, bool reduce,
                                   ArrayRef<unsigned> checkIdxs) {
  // First wave owner (the protected carrier close's acquirer).
  Owner firstWaveOwner;
  for (Node *n = head; n && !firstWaveOwner; n = n->next)
    if (n->kind == Node::Access && n->owner)
      firstWaveOwner = n->owner;
  if (!firstWaveOwner)
    return success();
  constexpr unsigned kPass2 = 1u << 20;
  // In-body kept edges by destination row; EXIT closes listed separately.
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  SmallVector<unsigned, 4> closes;
  for (auto [i, e] : llvm::enumerate(edges)) {
    if (drop[i] || ci.chainOf.lookup(e.src) != head ||
        ci.chainOf.lookup(e.dst) != head)
      continue;
    if (e.dst->kind == Node::Exit && e.src->kind == Node::Access &&
        e.srcOwner && e.dstOwner)
      closes.push_back(i);
    else
      atDst[e.dst].push_back(i);
  }
  if (closes.empty())
    return success();
  for (auto &[d, v] : atDst)
    llvm::stable_sort(v, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
  // First pass-2 touch row per (owner, piece) for the close checks.
  auto firstTouch = [&](const Owner &q, PieceId pc) -> Node * {
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Access && n->owner && sameOwner(n->owner, q))
        for (const Touch &t : n->touches)
          for (PieceId fp : g.pieceTable.footprint[t.member])
            if (fp == pc)
              return n;
    return nullptr;
  };
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snap1, snap2;
  std::map<int64_t, unsigned> waveOpenAt; // ownerKey -> pass-2 open row
  auto applyKept = [&](EdgeRec &e, unsigned srcIdx,
                       DenseMap<Node *, SyncVec> &snaps) {
    SyncVec &dv = behind[ownerKey(e.dstOwner)];
    auto sn = snaps.find(e.src);
    if (sn != snaps.end())
      for (auto &[k, v] : sn->second)
        dv[k] = std::max(dv[k], v);
    int64_t sk = ownerKey(e.srcOwner);
    dv[sk] = std::max(dv[sk], srcIdx);
  };
  // Pass 1.
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        applyKept(edges[ei], ci.idx.lookup(edges[ei].src), snap1);
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = ci.idx.lookup(n);
      snap1[n] = behind[ownerKey(n->owner)];
    }
  }
  // Second traversal (indices offset by kPass2); kept closes apply at
  // their true destinations (closing owner's first touches), in order.
  DenseMap<Node *, SmallVector<unsigned, 2>> closeAt;
  for (unsigned ei : closes) {
    EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    for (PieceId pc : e.pieces)
      if (Node *ft = firstTouch(e.dstOwner, pc))
        if (!latest || ci.idx.lookup(ft) > ci.idx.lookup(latest))
          latest = ft;
    if (latest)
      closeAt[latest].push_back(ei);
  }
  LogicalResult result = success();
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        if (drop[ei])
          continue;
        applyKept(edges[ei], kPass2 + ci.idx.lookup(edges[ei].src), snap2);
        waveOpenAt.try_emplace(ownerKey(edges[ei].dstOwner),
                               kPass2 + ci.idx.lookup(n));
      }
    auto ct = closeAt.find(n);
    if (ct != closeAt.end())
      for (unsigned ei : ct->second) {
        EdgeRec &e = edges[ei];
        int64_t dk = ownerKey(e.dstOwner);
        bool covered = covers(behind[dk], ownerKey(e.srcOwner),
                              ci.idx.lookup(e.src));
        bool open = waveOpenAt.count(dk);
        bool isCarrierClose = sameOwner(e.dstOwner, firstWaveOwner);
        if (reduce) {
          if (!drop[ei] && covered && open && !isCarrierClose)
            drop[ei] = true;
          if (!drop[ei]) // kept close: provides its ordering at dst
            applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (!drop[ei]) {
          applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (llvm::is_contained(checkIdxs, ei) &&
                   !(covered && open)) {
          result = (e.src->op ? e.src->op : g.root->op)
                       ->emitError("nvws-insert-semas: traversal-closure "
                                   "violation: dropped close not implied");
        }
      }
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] =
          kPass2 + ci.idx.lookup(n);
      snap2[n] = behind[ownerKey(n->owner)];
    }
  }
  return result;
}

static LogicalResult reduceEdges(GroupDag &g, SyncCtx &ctx) {
  if (g.root->children.empty() || ctx.edges.empty())
    return success();
  ChainIndex ci;
  indexChains(g.root->children[0], ci);
  std::vector<bool> drop(ctx.edges.size(), false);
  SmallVector<Node *, 8> heads;
  DenseSet<Node *> seen;
  for (auto &[row, h] : ci.chainOf)
    if (seen.insert(h).second)
      heads.push_back(h);
  llvm::sort(heads, [&](Node *a, Node *b) {
    return ci.idx.lookup(a) < ci.idx.lookup(b);
  });
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, ctx.edges, drop, /*reduce=*/true, {})))
      return failure();
  // Phase B: traversal-closure on loop-body chains.
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, ctx.edges, drop, /*reduce=*/true,
                               {})))
        return failure();
  SmallVector<unsigned, 8> dropped;
  for (auto [i, d] : llvm::enumerate(drop))
    if (d)
      dropped.push_back(i);
  if (dropped.empty())
    return success();
  // CLOSURE VERIFIER: re-derive coverage from the kept set only and
  // re-check every dropped edge (hard error on under-synchronization).
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, ctx.edges, drop, /*reduce=*/false,
                          dropped)))
      return failure();
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, ctx.edges, drop, /*reduce=*/false,
                               dropped)))
        return failure();
  if (std::getenv("NVWS_EDGE_DEBUG"))
    llvm::errs() << "[reduce] dropped " << dropped.size() << " of "
                 << ctx.edges.size() << " edges\n";
  SmallVector<EdgeRec> kept;
  for (auto [i, e] : llvm::enumerate(ctx.edges))
    if (!drop[i])
      kept.push_back(e);
  ctx.edges = std::move(kept);
  return success();
}
} // namespace

// ---------------------------------------------------------------------------
// Edge dedupe + fan-in grouping + node injection (spec section 5.2).
// ---------------------------------------------------------------------------
static LogicalResult buildEdgesAndSemas(GroupDag &g, SyncCtx &ctx) {
  auto ownerToStr = [](const Owner &o) {
    return o ? ("{" + std::to_string(o->first) + "}") : std::string("root");
  };
  (void)ownerToStr;
  if (std::getenv("NVWS_EDGE_DEBUG")) {
    llvm::errs() << "[edges] group members=" << g.pieceTable.members.size()
                 << " raw=" << ctx.edges.size() << "\n";
    auto rowStr = [&](Node *n) {
      std::string k = n->kind == Node::Enter   ? "ENTER"
                      : n->kind == Node::Exit  ? "EXIT"
                      : n->kind == Node::For   ? "FOR"
                      : n->kind == Node::If    ? "IF"
                                               : (n->op ? n->op->getName()
                                                              .getStringRef()
                                                              .str()
                                                        : "?");
      return k;
    };
    for (EdgeRec &e : ctx.edges) {
      llvm::errs() << "  " << rowStr(e.src) << " " << ownerToStr(e.srcOwner)
                   << " -> " << rowStr(e.dst) << " " << ownerToStr(e.dstOwner)
                   << " pieces[";
      for (PieceId p : e.pieces)
        llvm::errs() << p << " ";
      llvm::errs() << "]\n";
    }
  }
  // Transitive reduction (spec section) — drops implied edges, then the
  // closure verifier re-checks every drop against the kept set.
  if (failed(reduceEdges(g, ctx)))
    return failure();
  // Components are independent (spec): no merge layer below may join
  // pieces of different components — one semaphore would otherwise span
  // components (the createSema guard refuses that). Every merge key
  // therefore carries the edge's component. Edges are born per chain
  // walk, i.e. per component, so the first piece determines it.
  auto edgeComp = [&](const EdgeRec &e) -> CompId {
    return g.pieceTable.pieceComp[e.pieces.front()];
  };
  // Dedupe by (src, dst, srcOwner, component) with payload + piece union.
  SmallVector<EdgeRec> deduped;
  DenseMap<std::tuple<Node *, Node *, int64_t, CompId>, unsigned>
      index; // lookup
  for (EdgeRec &e : ctx.edges) {
    auto key = std::make_tuple(e.src, e.dst, ownerKey(e.srcOwner),
                               edgeComp(e));
    auto it = index.find(key);
    if (it == index.end()) {
      index.try_emplace(key, deduped.size());
      llvm::sort(e.payloads, [](AsyncOp a, AsyncOp b) {
        return static_cast<int>(a) < static_cast<int>(b);
      });
      llvm::sort(e.pieces);
      deduped.push_back(e);
      continue;
    }
    EdgeRec &d = deduped[it->second];
    unionPayloads(d.payloads, e.payloads);
    for (PieceId p : e.pieces)
      if (!llvm::is_contained(d.pieces, p))
        d.pieces.push_back(p);
    llvm::sort(d.pieces);
  }

  // Second collapse — same destination, same source OWNER, different
  // source rows (multi-piece games: one partition holds different pieces
  // at different rows): a partition's later release subsumes its earlier
  // one (same instruction stream — an arrive after the later row implies
  // the earlier row is done), so keep the LATEST source row and union
  // payloads/pieces. After this, a destination's sources are pairwise
  // distinct partitions by construction (the section 5.3 count formula).
  auto isLaterInChain = [](Node *a, Node *b) { // true iff a is after b
    for (Node *m = b->next; m; m = m->next)
      if (m == a)
        return true;
    return false;
  };
  SmallVector<EdgeRec> collapsed;
  DenseMap<std::tuple<Node *, int64_t, CompId>, unsigned> cidx; // lookup
  for (EdgeRec &e : deduped) {
    auto key = std::make_tuple(e.dst, ownerKey(e.srcOwner), edgeComp(e));
    auto it = cidx.find(key);
    if (it == cidx.end()) {
      cidx.try_emplace(key, collapsed.size());
      collapsed.push_back(e);
      continue;
    }
    EdgeRec &d = collapsed[it->second];
    if (isLaterInChain(e.src, d.src))
      d.src = e.src;
    unionPayloads(d.payloads, e.payloads);
    for (PieceId p : e.pieces)
      if (!llvm::is_contained(d.pieces, p))
        d.pieces.push_back(p);
    llvm::sort(d.pieces);
  }

  // Group by destination, first-seen order: one semaphore per destination —
  // EXCEPT For-row destinations, which unify with the loop's in-body regain
  // group (spec section 5.2): an edge into a For row is the entry instance
  // of that loop's regain (iteration 0 fed from outside, iterations 1..N
  // by the in-loop release; same acquirer class, M3-clean).
  struct DstGroup {
    Node *dst;
    SmallVector<unsigned, 2> idxs;
    int sema = -1;
  };
  // Key = (destination row, destination OWNER, component): an EXIT row of
  // a multi-piece chain can close pieces with different carried owners, and
  // each owner class is its own phase-tracked waiter (old M3 identity).
  // The component keeps independent token games apart even when their
  // regains share a destination row and acquirer (e.g. disjoint slivers
  // written by one partition: same For row, same owner, separate games).
  llvm::MapVector<std::tuple<Node *, int64_t, CompId>, unsigned> dstIndex;
  SmallVector<DstGroup> groups;
  for (auto [i, e] : llvm::enumerate(collapsed)) {
    auto key = std::make_tuple(e.dst, ownerKey(e.dstOwner), edgeComp(e));
    auto it = dstIndex.find(key);
    if (it == dstIndex.end()) {
      dstIndex.try_emplace(key, groups.size());
      groups.push_back(DstGroup{e.dst, {static_cast<unsigned>(i)}, -1});
    } else {
      groups[it->second].idxs.push_back(i);
    }
  }

  auto groupComp = [&](const DstGroup &grp) {
    return g.pieceTable.pieceComp[collapsed[grp.idxs.front()].pieces.front()];
  };
  auto groupAcquirer = [&](const DstGroup &grp) {
    return collapsed[grp.idxs.front()].dstOwner;
  };
  // The regain group of a For-row destination: the LAST destination group
  // in the For's body OWN chain with the same component and acquirer.
  auto findRegainGroup = [&](Node *forRow, const Owner &acq,
                             CompId comp) -> int {
    int best = -1;
    for (Node *m = forRow->children[0]; m; m = m->next) {
      auto it = dstIndex.find(std::make_tuple(m, ownerKey(acq), comp));
      if (it == dstIndex.end())
        continue;
      DstGroup &cand = groups[it->second];
      if (groupComp(cand) == comp)
        best = static_cast<int>(it->second);
    }
    return best;
  };

  auto createSema = [&](DstGroup &grp) -> LogicalResult {
    SemaId sid = g.semaTable.semas.size();
    Sema s;
    s.name = "S" + std::to_string(sid);
    s.count = grp.idxs.size();
    for (unsigned idx : grp.idxs)
      for (PieceId p : collapsed[idx].pieces)
        if (!llvm::is_contained(s.pieces, p))
          s.pieces.push_back(p);
    llvm::sort(s.pieces);
    s.component = g.pieceTable.pieceComp[s.pieces.front()];
    for (PieceId p : s.pieces)
      if (g.pieceTable.pieceComp[p] != s.component)
        return (grp.dst->op ? grp.dst->op : g.root->op)
            ->emitError("nvws-insert-semas: one destination joins pieces of "
                        "different components");
    grp.sema = static_cast<int>(sid);
    g.semaTable.semas.push_back(std::move(s));
    return success();
  };

  // Sema assignment in first-seen order; a For-row group adopts its regain
  // group's semaphore (creating it eagerly so the unified semaphore gets
  // the earlier number, like the old pass's S0).
  for (DstGroup &grp : groups) {
    if (grp.sema != -1)
      continue;
    // Pairwise-distinct source partitions per destination group.
    for (unsigned i = 0; i < grp.idxs.size(); ++i)
      for (unsigned j = i + 1; j < grp.idxs.size(); ++j)
        if (sameOwner(collapsed[grp.idxs[i]].srcOwner,
                      collapsed[grp.idxs[j]].srcOwner))
          return (grp.dst->op ? grp.dst->op : g.root->op)
              ->emitError("nvws-insert-semas: fan-in sources share a "
                          "partition — not expressible as one semaphore");
    if (grp.dst->kind == Node::For) {
      int t = findRegainGroup(grp.dst, groupAcquirer(grp), groupComp(grp));
      if (t >= 0) {
        if (groups[t].sema == -1)
          if (failed(createSema(groups[t])))
            return failure();
        grp.sema = groups[t].sema;
        Sema &s = g.semaTable.semas[grp.sema];
        for (unsigned idx : grp.idxs)
          for (PieceId p : collapsed[idx].pieces)
            if (!llvm::is_contained(s.pieces, p))
              s.pieces.push_back(p);
        llvm::sort(s.pieces);
        continue;
      }
    }
    if (failed(createSema(grp)))
      return failure();
  }

  // Injection: one Acquire per destination group (its own count), one
  // Release per source; merged groups share the semaphore.
  DenseMap<Node *, Node *> lastAfter; // release insertion cursor per source
  for (DstGroup &grp : groups) {
    Sema &s = g.semaTable.semas[grp.sema];
    // UNIFORM PENDING COUNT: every acquire site of one semaphore carries
    // the semaphore's count; a merged group with fewer sources scales its
    // releases' arrive multiplicity so the per-cycle arrives still sum to
    // the count (spec section 5.2). m == count -> multiplicity 1;
    // m == 1 -> one release with multiplicity count; anything else is
    // inexpressible — hard diagnostic, never a silent repair.
    unsigned m = grp.idxs.size();
    unsigned relCount = 1;
    if (m != s.count) {
      if (m == 1)
        relCount = s.count;
      else
        return (grp.dst->op ? grp.dst->op : g.root->op)
            ->emitError("nvws-insert-semas: destination group with ")
               << m << " sources cannot meet semaphore " << s.name
               << " pending count " << s.count;
    }
    Node *acq = g.newNode(Node::Acquire, /*op=*/nullptr, grp.dst->parent);
    acq->owner = groupAcquirer(grp);
    acq->sema = static_cast<SemaId>(grp.sema);
    acq->count = s.count;
    // BACK-EDGE PLACEMENT (spec wave-locality section): an EXIT-close
    // regain whose acquirer is NOT the chain's first wave owner anchors
    // at the start of its own wave (before its partition's first touch);
    // iteration 0 is satisfied by the initial permit (isEntry), and no
    // token threads through iter_args for it.
    Node *dstAnchor = grp.dst;
    if (grp.dst->kind == Node::Exit && acq->owner) {
      Node *head = grp.dst;
      while (head->prev)
        head = head->prev;
      Owner firstWaveOwner;
      Node *firstTouch = nullptr;
      for (Node *r = head; r; r = r->next) {
        if (r->kind != Node::Access || !r->owner)
          continue;
        if (!firstWaveOwner)
          firstWaveOwner = r->owner;
        if (!firstTouch && sameOwner(r->owner, acq->owner))
          firstTouch = r;
      }
      if (firstWaveOwner && !sameOwner(firstWaveOwner, acq->owner) && firstTouch) {
        dstAnchor = firstTouch;
        s.isEntry = true; // initial permit; no pre-loop entry instance
        s.inheritStamp = acq->owner;
      }
    }
    spliceBefore(acq, dstAnchor);
    s.expectedReleases += m * relCount;
    for (unsigned idx : grp.idxs) {
      EdgeRec &e = collapsed[idx];
      Node *rel = g.newNode(Node::Release, /*op=*/nullptr, e.src->parent);
      rel->owner = e.srcOwner;
      rel->sema = static_cast<SemaId>(grp.sema);
      rel->count = relCount; // arrive multiplicity (default 1)
      rel->payloads = e.payloads;
      rel->sat = acq;
      Node *anchor = lastAfter.lookup(e.src);
      spliceAfter(rel, anchor ? anchor : e.src);
      lastAfter[e.src] = rel;
    }
  }
  return success();
}

// ---------------------------------------------------------------------------
// Entry acquires (spec section 5.3): per component with sync, one acquire
// that executes exactly once before the component's first row. Loop-carried
// components duplicate the carried owner's REGAIN (the last acquire in the
// loop body's own chain); components without one (acyclic chains, or
// conditional-only acquires) get a dedicated entry semaphore, released once
// after the component's terminal top-level row.
// ---------------------------------------------------------------------------
static bool nodeInvolvesComp(GroupDag &g, Node *n, CompId comp) {
  if (n->kind == Node::Access) {
    for (const Touch &t : n->touches)
      for (PieceId p : g.pieceTable.footprint[t.member])
        if (g.pieceTable.pieceComp[p] == comp)
          return true;
    return false;
  }
  if (n->kind == Node::For || n->kind == Node::If)
    for (auto &[p, pi] : n->pieceInfo)
      if (g.pieceTable.pieceComp[p] == comp)
        return true;
  return false;
}

// Regain search: the LAST acquire of the component in the body's ENTIRE
// subtree — if branches (a conditional handback is a valid regain) and
// nested For bodies (an inner-loop regain is a valid seed: the entry
// instance fires once on the initial permit; differing acquire
// frequencies on one semaphore are the For-row-unification situation, and
// the old pass's SEMA-UNION seeded nested-loop buffers this way).
static Node *lastAcquireOfCompInChain(GroupDag &g, Node *head, CompId comp) {
  Node *found = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire &&
        g.semaTable.semas[n->sema].component == comp)
      found = n;
    if (n->kind == Node::If || n->kind == Node::For)
      for (Node *child : n->children)
        if (Node *f = lastAcquireOfCompInChain(g, child, comp))
          found = f;
  }
  return found;
}

// The component's first access row's owner, chain order, recursive.
static Owner firstAccessOwnerOfComp(GroupDag &g, Node *head, CompId comp,
                                    bool &found) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access) {
      for (const Touch &t : n->touches)
        for (PieceId p : g.pieceTable.footprint[t.member])
          if (g.pieceTable.pieceComp[p] == comp) {
            found = true;
            return n->owner;
          }
    }
    if (n->kind == Node::For || n->kind == Node::If)
      for (Node *child : n->children) {
        Owner o = firstAccessOwnerOfComp(g, child, comp, found);
        if (found)
          return o;
      }
  }
  return std::nullopt;
}

static LogicalResult insertEntryAcquires(GroupDag &g) {
  unsigned numComps = 0;
  for (CompId c : g.pieceTable.pieceComp)
    numComps = std::max(numComps, c + 1);
  Node *top = g.root->children.empty() ? nullptr : g.root->children[0];
  if (!top)
    return success();

  for (CompId comp = 0; comp < numComps; ++comp) {
    // Skip components with no synchronization at all (same-owner only).
    bool hasSync = false;
    for (const Sema &s : g.semaTable.semas)
      if (s.component == comp)
        hasSync = true;
    if (!hasSync)
      continue;

    // Placement chain: start at the top chain and DESCEND through scf.if
    // branches while the component lives entirely inside one branch — an
    // if branch executes at most once, so the entry stays once-executed
    // and fires only on the path that uses the buffer (the old
    // architecture's "initial acquire stays with the create"). NEVER
    // descend into a For row: its body executes per iteration.
    Node *chainHead = top;
    SmallVector<Node *, 4> rows;
    auto collectRows = [&](Node *head) {
      rows.clear();
      for (Node *n = head; n; n = n->next)
        if (nodeInvolvesComp(g, n, comp))
          rows.push_back(n);
    };
    collectRows(chainHead);
    while (rows.size() == 1 && rows[0]->kind == Node::If) {
      Node *onlyChild = nullptr;
      int cnt = 0;
      for (Node *child : rows[0]->children) {
        bool involves = false;
        for (Node *n = child; n; n = n->next)
          if (nodeInvolvesComp(g, n, comp))
            involves = true;
        if (involves) {
          onlyChild = child;
          ++cnt;
        }
      }
      if (cnt != 1)
        break;
      chainHead = onlyChild;
      collectRows(chainHead);
    }
    if (rows.empty())
      return g.root->op->emitError(
          "nvws-insert-semas: component with sync but no placement rows");

    // Inherit fact = the component's FIRST ACCESS owner, chain order,
    // recursive (ground truth: matches the old pass's recorded seed
    // acquirers across the corpus — root for root-seeded accumulators,
    // the producer partition for operand buffers, the in-loop first
    // toucher for branch-local buffers).
    bool fhFound = false;
    Owner firstHolder = firstAccessOwnerOfComp(g, top, comp, fhFound);
    if (!fhFound)
      return g.root->op->emitError(
          "nvws-insert-semas: component has no access rows");

    // Regain: the carried owner's last acquire in the body's OWN chain of
    // the last top-level loop carrying the component.
    Node *regain = nullptr;
    for (Node *row : llvm::reverse(rows))
      if (row->kind == Node::For) {
        regain = lastAcquireOfCompInChain(g, row->children[0], comp);
        if (regain)
          break;
      }

    if (regain) {
      Sema &s = g.semaTable.semas[regain->sema];
      s.isEntry = true; // first event in chain order is an acquire
      s.inheritStamp = firstHolder; // carrier inherit: emission stamps this
      Node *acq = g.newNode(Node::Acquire, nullptr, rows.front()->parent);
      acq->owner = std::nullopt; // ROOT — executes in the root region
      acq->sema = regain->sema;
      acq->count = regain->count; // duplicate the regain instance's count
      spliceBefore(acq, rows.front());
    } else {
      SemaId sid = g.semaTable.semas.size();
      Sema s;
      s.name = "E" + std::to_string(sid);
      s.component = comp;
      for (auto [p, c] : llvm::enumerate(g.pieceTable.pieceComp))
        if (c == comp)
          s.pieces.push_back(p);
      s.count = 1;
      s.isEntry = true;
      s.expectedReleases = 1; // the terminal release
      s.inheritStamp = firstHolder; // carrier inherit: emission stamps this
      Node *acq = g.newNode(Node::Acquire, nullptr, rows.front()->parent);
      acq->owner = std::nullopt; // ROOT — executes in the root region
      acq->sema = sid;
      acq->count = 1;
      spliceBefore(acq, rows.front());
      // Terminal release at the same chain level: returns the permit; no
      // future acquire waits on it (sat stays null).
      Node *terminal = rows.back();
      Node *rel = g.newNode(Node::Release, nullptr, terminal->parent);
      rel->owner = terminal->kind == Node::Access
                       ? terminal->owner
                       : sortedPieceInfo(terminal).front().second.owner;
      rel->sema = sid;
      rel->payloads.push_back(AsyncOp::NONE);
      // Place after any releases already trailing the terminal row.
      Node *anchor = terminal;
      while (anchor->next && anchor->next->kind == Node::Release)
        anchor = anchor->next;
      spliceAfter(rel, anchor);
      g.semaTable.semas.push_back(std::move(s));
    }
  }
  return success();
}

// ---------------------------------------------------------------------------
// Crossings (threading facts ON the nodes) + requiredParts (clone sets).
// Post-order: nested region rows resolve before their parents.
// ---------------------------------------------------------------------------
static CompId compOfMember(GroupDag &g, MemberId m) {
  return g.pieceTable.pieceComp[g.pieceTable.footprint[m].front()];
}

static Node *chainFinalForComp(GroupDag &g, Node *head, CompId comp) {
  Node *final = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire &&
        g.semaTable.semas[n->sema].component == comp)
      final = n;
    if ((n->kind == Node::For || n->kind == Node::If))
      for (const Crossing &c : n->crossings)
        if (c.comp == comp)
          final = n;
  }
  return final;
}

static Owner finalOwner(GroupDag &g, Node *final, CompId comp) {
  if (final->kind == Node::Acquire)
    return final->owner;
  for (const Crossing &c : final->crossings)
    if (c.comp == comp)
      return c.slotOwner;
  return std::nullopt;
}

static void computeCrossings(GroupDag &g, Node *head, unsigned numComps) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind != Node::For && n->kind != Node::If)
      continue;
    for (Node *child : n->children)
      computeCrossings(g, child, numComps);
    for (CompId comp = 0; comp < numComps; ++comp) {
      Crossing cr;
      cr.comp = comp;
      bool any = false;
      for (Node *child : n->children) {
        Node *f = chainFinalForComp(g, child, comp);
        cr.finals.push_back(f);
        if (f) {
          any = true;
          cr.slotOwner = finalOwner(g, f, comp);
        }
      }
      if (any)
        n->crossings.push_back(std::move(cr));
    }
  }
}


// ---------------------------------------------------------------------------
// Liveness prune for If crossings (spec node table): a crossing survives
// only if the escaped carrier is consumed after the If — a later component
// row in the chain, or the enclosing region's own surviving crossing
// (For-body recurrence / live parent branch). Function-chain end consumes
// nothing. Reverse chain order so later prunes are visible to earlier
// rows; parents prune before children descend.
// ---------------------------------------------------------------------------
static bool carrierConsumedAfter(GroupDag &g, Node *start, CompId comp) {
  for (Node *n = start; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      if (g.semaTable.semas[n->sema].component == comp)
        return false; // fresh carrier supersedes the escaped one
      break;
    case Node::Release:
      if (g.semaTable.semas[n->sema].component == comp)
        return true;
      break;
    case Node::Access:
      for (const Touch &t : n->touches)
        if (compOfMember(g, t.member) == comp)
          return true;
      break;
    case Node::For:
    case Node::If:
      for (const Crossing &c : n->crossings)
        if (c.comp == comp)
          return true;
      break;
    default:
      break;
    }
  }
  return false;
}

static bool regionLiveFor(const Node *region, CompId comp) {
  if (!region)
    return false; // function chain
  for (const Crossing &c : region->crossings)
    if (c.comp == comp)
      return true;
  return false;
}

static void pruneDeadIfCrossings(GroupDag &g, Node *head, Node *region) {
  SmallVector<Node *, 8> rows;
  for (Node *n = head; n; n = n->next)
    rows.push_back(n);
  for (Node *n : llvm::reverse(rows))
    if (n->kind == Node::If)
      llvm::erase_if(n->crossings, [&](const Crossing &c) {
        return !carrierConsumedAfter(g, n->next, c.comp) &&
               !regionLiveFor(region, c.comp);
      });
  for (Node *n : rows)
    if (n->kind == Node::For || n->kind == Node::If)
      for (Node *child : n->children)
        pruneDeadIfCrossings(g, child, n);
}

static void collectParts(Node *head, SmallVector<int, 4> &parts) {
  for (Node *n = head; n; n = n->next) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire ||
         n->kind == Node::Release) &&
        n->owner.has_value())
      if (!llvm::is_contained(parts, n->owner->first))
        parts.push_back(n->owner->first);
    if (n->kind == Node::For || n->kind == Node::If)
      for (Node *child : n->children)
        collectParts(child, parts);
  }
}

static void computeRequiredParts(Node *head) {
  for (Node *n = head; n; n = n->next)
    if (n->kind == Node::For || n->kind == Node::If) {
      for (Node *child : n->children)
        computeRequiredParts(child);
      SmallVector<int, 4> parts;
      for (Node *child : n->children)
        collectParts(child, parts);
      llvm::sort(parts);
      n->requiredParts.assign(parts.begin(), parts.end());
    }
}

// ---------------------------------------------------------------------------
// BackingPlan (plan contract B; reference semantics from
// InsertTmemSemaphore.cpp:1297/1340/1379 — re-derived, not copied).
// ---------------------------------------------------------------------------
static bool canDoubleBufferAcc(nvidia_gpu::MMAv5OpInterface mmaOp,
                               int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  int64_t blockM = tmemDesc.getShape()[0];
  int64_t blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + blockM * blockN * 2 > numTMEMRows * numTMEMColumns)
    return false;
  if (isa<nvidia_gpu::TCGen5MMAScaledOp>(mmaOp.getOperation()) &&
      blockN == 256)
    return false;
  return true;
}

static scf::ForOp outerWSLoop(scf::ForOp loop) {
  scf::ForOp ws = loop;
  for (Operation *p = loop; p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      if (gpu::hasWarpSpecializeTag(f))
        ws = f;
  return ws;
}

// USER RULING (plan ground rule 2): the producer-consumer pattern gate is
// dropped; the per-MMA-user multibuffering veto chain of
// InsertTmemSemaphore.cpp:1408-1425 is used VERBATIM as the whole decision.
static bool isMultiStagedGroup(GroupDag &g, int numTmemBlocks) {
  bool isMultiStaged = true;
  for (const Member &m : g.pieceTable.members) {
    for (Operation *user : m.allocOp->getResult(0).getUsers()) {
      if (auto mmaOp = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          scf::ForOp wsLoop = outerWSLoop(loop);
          // Determine if the MMA accumulator can be multibuffered.
          bool accIsMultiBuffered =
              // MMAs in subsequent iterations can be overlapped.
              !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              // The accumulator is reset at some point, thus allowing
              // multibuffering.
              nvidia_gpu::isAccMultibufferingPossible(mmaOp, loop) &&
              // The user didn't disable it with a flag.
              !getDisallowAccMultiBuffer(wsLoop) &&
              canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiStaged = isMultiStaged && accIsMultiBuffered;
        }
      }
    }
  }
  return isMultiStaged;
}

static void computeBackingPlan(GroupDag &g, triton::FuncOp funcOp,
                               bool useMetaPartitioner, int &numTmemBlocks) {
  // POSTERITY: this is the ONLY decision in the whole pass that consumes
  // useMetaPartitioner (audited against the pre-rewrite pass, which used it
  // identically and nowhere else): the meta partitioner makes its own
  // pipelining/multibuffering arrangements, so insert-semas must not
  // double-buffer the TMEM accumulator on top of it — meta => numStages=1
  // always (plan contract B). The flag never influences discovery, pieces,
  // owners, the walk, edges/semaphores, entry acquires, crossings, or
  // placement.
  g.backingPlan.numStages = 1;
  // Zero-semaphore groups are untouched at emission (contract H): no stage
  // assignment, no capacity charge — phantom charges from groups that will
  // never be materialized are order-dependent and can push a later REAL
  // accumulator below capacity (plan contract B).
  bool untouched = g.semaTable.semas.empty();
  if (g.isTmem() && !untouched && !useMetaPartitioner &&
      isMultiStagedGroup(g, numTmemBlocks))
    g.backingPlan.numStages = 2;
  // Hoist anchor: before the first WS-tagged loop (function scope).
  Operation *anchor = nullptr;
  funcOp.walk([&](scf::ForOp forOp) {
    if (!anchor && forOp->hasAttr(triton::kWarpSpecializeAttrName))
      anchor = forOp;
    return anchor ? WalkResult::interrupt() : WalkResult::advance();
  });
  g.backingPlan.hoistAnchor = anchor ? anchor : &funcOp.getBody().front().front();
  if (g.isTmem() && !untouched)
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.backingPlan.numStages;
    }
}

// ---------------------------------------------------------------------------
// Verifiers (spec section 7, SYNC).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// WAVE-LOCALITY VERIFIER (hard error; spec rule, user ruling 10jun26):
// walking each chain, every Access/Release row owned by a concrete
// partition must find its component's carrier held by that partition
// (set by Acquire rows; entry rows seed their inheritStamp; region heads
// seed the carried owner). Catches any unbracketed wave at analysis time.
// ---------------------------------------------------------------------------
static LogicalResult verifyCarrierLocality(GroupDag &g, Node *head) {
  std::map<CompId, Owner> carrier; // tracked comps only
  auto seedFromPieces = [&](Node *n) {
    for (auto &[p, pi] : sortedPieceInfo(n)) {
      CompId c = g.pieceTable.pieceComp[p];
      auto it = carrier.find(c);
      if (it == carrier.end())
        carrier.emplace(c, pi.owner);
      else if (!sameOwner(it->second, pi.owner))
        carrier.erase(it); // mixed: untracked
    }
  };
  if (head->kind == Node::Enter)
    seedFromPieces(head);
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      const Sema &sm = g.semaTable.semas[n->sema];
      carrier[sm.component] = sm.isEntry && !n->owner ? sm.inheritStamp
                                                      : n->owner;
      break;
    }
    case Node::Release: {
      const Sema &sm = g.semaTable.semas[n->sema];
      auto it = carrier.find(sm.component);
      if (n->owner && it != carrier.end() && it->second &&
          !sameOwner(it->second, n->owner))
        return (n->sat && n->sat->op ? n->sat->op : g.root->op)
                   ->emitError("nvws-insert-semas: wave-locality violation: "
                               "release owned by partition ")
               << n->owner->first << " consumes a carrier held by partition "
               << it->second->first;
      break;
    }
    case Node::Access: {
      if (!n->owner)
        break; // root rows: outside partition discipline
      for (const Touch &t : n->touches) {
        CompId c = compOfMember(g, t.member);
        auto it = carrier.find(c);
        if (it != carrier.end() && it->second &&
            !sameOwner(it->second, n->owner))
          return n->op->emitError(
                     "nvws-insert-semas: wave-locality violation: access "
                     "owned by partition ")
                 << n->owner->first
                 << " touches a carrier held by partition "
                 << it->second->first;
      }
      break;
    }
    case Node::For:
    case Node::If: {
      for (Node *child : n->children)
        if (failed(verifyCarrierLocality(g, child)))
          return failure();
      seedFromPieces(n); // EXIT regain: carrier back with carried owner
      break;
    }
    default:
      break;
    }
  }
  // Traversal-boundary locality (spec wave-locality section): under a
  // loop the chain's final carrier owner must equal its first wave
  // owner — only that token is carried into the following traversal.
  if (head->parent && head->parent->kind == Node::For) {
    // First wave = the first carrier consumer next iteration; a leading
    // region row consumes it in ITS body top — descend.
    std::function<Owner(Node *)> firstWaveOf = [&](Node *h) -> Owner {
      for (Node *n = h; n; n = n->next) {
        if (n->kind == Node::Access && n->owner)
          return n->owner;
        if (n->kind == Node::Acquire && n->owner)
          return n->owner; // a leading in-body acquire opens the wave
        if ((n->kind == Node::For || n->kind == Node::If) &&
            !n->children.empty())
          if (Owner o = firstWaveOf(n->children[0]))
            return o;
      }
      return std::nullopt;
    };
    Owner firstWave = firstWaveOf(head);
    Owner finalCarrier;
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Acquire && n->owner)
        finalCarrier = n->owner;
    if (firstWave && finalCarrier && !sameOwner(firstWave, finalCarrier))
      return (g.root->op ? g.root->op : head->parent->op)
                 ->emitError("nvws-insert-semas: traversal-boundary wave-locality "
                             "violation: loop body's final carrier owner ")
             << finalCarrier->first << " differs from its first wave owner "
             << firstWave->first;
  }
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (!g.semaTable.semas.empty() && !g.root->children.empty())
    if (failed(verifyCarrierLocality(g, g.root->children[0])))
      return failure();
  // Per sema: #releases == count; release precedes its sat acquire in the
  // same chain; payloads non-empty.
  SmallVector<unsigned> releaseCount(g.semaTable.semas.size(), 0);
  std::function<LogicalResult(Node *)> walk = [&](Node *head) -> LogicalResult {
    for (Node *n = head; n; n = n->next) {
      if (n->kind == Node::Release) {
        releaseCount[n->sema] += std::max(1u, n->count);
        if (n->payloads.empty())
          return g.root->op->emitError(
              "nvws-insert-semas: release without payload record");
        if (n->sat) {
          if (n->sat->parent != n->parent)
            return g.root->op->emitError(
                "nvws-insert-semas: release and its acquire are in "
                "different chains");
          bool forward = false;
          for (Node *m = n->next; m; m = m->next)
            if (m == n->sat)
              forward = true;
          if (!forward && !g.semaTable.semas[n->sema].isEntry)
            return g.root->op->emitError(
                "nvws-insert-semas: release does not precede its acquire");
        }
      }
      if (n->kind == Node::For || n->kind == Node::If)
        for (Node *child : n->children)
          if (failed(walk(child)))
            return failure();
    }
    return success();
  };
  if (!g.root->children.empty())
    if (failed(walk(g.root->children[0])))
      return failure();
  for (auto [sid, s] : llvm::enumerate(g.semaTable.semas)) {
    if (releaseCount[sid] != s.expectedReleases)
      return g.root->op->emitError("nvws-insert-semas: semaphore ")
             << s.name << " has " << releaseCount[sid] << " releases, expected "
             << s.expectedReleases;
  }
  // M3 acquirer-class criterion (spec section 5.3): per semaphore, the
  // acquiring owners contain at most ONE concrete partition; root is
  // additionally allowed (the carrier-inherit case). Two distinct
  // partitions are not expressible as one phase-tracked semaphore.
  SmallVector<std::optional<int64_t>> acqClass(g.semaTable.semas.size(),
                                               std::nullopt);
  std::function<LogicalResult(Node *)> m3 = [&](Node *head) -> LogicalResult {
    for (Node *n = head; n; n = n->next) {
      if (n->kind == Node::Acquire &&
          n->count != g.semaTable.semas[n->sema].count)
        return g.root->op->emitError("nvws-insert-semas: semaphore ")
               << g.semaTable.semas[n->sema].name
               << " acquired with non-uniform pending count";
      if (n->kind == Node::Acquire && n->owner.has_value()) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema] && *acqClass[n->sema] != k)
          return g.root->op->emitError("nvws-insert-semas: semaphore ")
                 << g.semaTable.semas[n->sema].name
                 << " acquired by two distinct partitions (M3 violation)";
        acqClass[n->sema] = k;
      }
      if (n->kind == Node::For || n->kind == Node::If)
        for (Node *child : n->children)
          if (failed(m3(child)))
            return failure();
    }
    return success();
  };
  if (!g.root->children.empty())
    if (failed(m3(g.root->children[0])))
      return failure();
  return success();
}

// ---------------------------------------------------------------------------
// Driver for stage 3.
// ---------------------------------------------------------------------------
static LogicalResult buildSyncDag(GroupDag &g, triton::FuncOp funcOp,
                                  bool useMetaPartitioner,
                                  int &numTmemBlocks) {
  SyncCtx ctx;
  if (!g.root->children.empty()) {
    ChainState top; // function chain: games start at bottom (first-touch)
    walkChain(g, g.root->children[0], top, ctx, /*underFor=*/false);
  }
  if (failed(buildEdgesAndSemas(g, ctx)))
    return failure();
  if (failed(insertEntryAcquires(g)))
    return failure();
  unsigned numComps = 0;
  for (CompId c : g.pieceTable.pieceComp)
    numComps = std::max(numComps, c + 1);
  if (!g.root->children.empty()) {
    computeCrossings(g, g.root->children[0], numComps);
    pruneDeadIfCrossings(g, g.root->children[0], /*region=*/nullptr);
    computeRequiredParts(g.root->children[0]);
  }
  computeBackingPlan(g, funcOp, useMetaPartitioner, numTmemBlocks);
  // Pipeline-invariant guard (contract D / mining gap 6): a managed (=
  // synchronized) group must not contain a tt-form descriptor-fed
  // sourceful alloc — nvws-insert-allocas normalizes those upstream.
  if (!g.semaTable.semas.empty())
    for (Operation *alloc : g.ttDescriptorFedMembers)
      return alloc->emitError(
          "nvws-insert-semas: managed local_alloc sourced from a tt-form "
          "descriptor load — nvws-insert-allocas must convert this "
          "upstream (pipeline invariant violated)");
  return verifySyncDag(g);
}

// ---------------------------------------------------------------------------
// Dump: the OWNER tree extended with sync rows, thread/parts/yield
// annotations, and the semaphore/backing tables (faithful: every printed
// token is a struct field).
// ---------------------------------------------------------------------------
static StringRef asyncOpStr(AsyncOp a) {
  switch (a) {
  case AsyncOp::TC5MMA:
    return "tc5mma";
  case AsyncOp::TMALoad:
    return "tma_load";
  case AsyncOp::CpAsync:
    return "cp_async";
  case AsyncOp::WGMMA:
    return "wgmma";
  case AsyncOp::TMEMCopy:
    return "tmem_copy";
  default:
    return "none";
  }
}

static std::string syncRowLabel(GroupDag &g, const Node *n) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (n->kind == Node::Acquire)
    os << "a " << g.semaTable.semas[n->sema].name;
  else if (n->kind == Node::Release)
    os << "r " << g.semaTable.semas[n->sema].name;
  else if (n->kind == Node::For)
    os << "scf.for";
  else if (n->kind == Node::If)
    os << "scf.if";
  return s;
}

static void printThreadInfo(llvm::raw_ostream &os, GroupDag &g,
                            const Node *n) {
  if (!n->requiredParts.empty()) {
    os << " parts{";
    bool first = true;
    for (int p : n->requiredParts) {
      if (!first)
        os << ",";
      first = false;
      os << p;
    }
    os << "}";
  }
  if (!n->crossings.empty()) {
    os << " thread{";
    bool first = true;
    for (const Crossing &c : n->crossings) {
      if (!first)
        os << ",";
      first = false;
      os << "c" << c.comp << ":" << ownerStr(n->op, c.slotOwner);
    }
    os << "}";
  }
}

static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g,
                           const Node *exit, const Node *region,
                           unsigned chainIdx) {
  if (!region || region->crossings.empty())
    return;
  os << " yield{";
  bool first = true;
  for (const Crossing &c : region->crossings) {
    if (!first)
      os << ",";
    first = false;
    os << "c" << c.comp << ": ";
    Node *f = chainIdx < c.finals.size() ? c.finals[chainIdx] : nullptr;
    os << (f ? syncRowLabel(g, f) : std::string("pass"));
  }
  os << "}";
}

static void dumpSyncChain(GroupDag &g, const Node *head, unsigned depth,
                          const Node *region, unsigned chainIdx) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    Operation *anchor = n->parent ? n->parent->op : nullptr;
    switch (n->kind) {
    case Node::Access: {
      // Per-member effects (one op = one row; a member the op only reads
      // is shown R even when another member is written).
      os << treePrefix(depth) << "|- ";
      bool first = true;
      for (const Touch &t : n->touches) {
        if (!first)
          os << ",";
        first = false;
        os << (t.effect == Effect::W ? "W" : "R") << " m" << t.member;
      }
      os << "  " << n->op->getName().getStringRef() << " "
         << ownerStr(n->op, n->owner) << "\n";
      break;
    }
    case Node::Acquire: {
      const Sema &s = g.semaTable.semas[n->sema];
      os << treePrefix(depth) << "|- a  " << s.name;
      if (n->count > 1)
        os << "(" << n->count << ")";
      os << "  " << ownerStr(anchor, n->owner);
      // The entry instance: the root-owned acquire of an entry semaphore
      // (the in-loop regain acquirer is always a concrete partition).
      if (s.isEntry && !n->owner.has_value())
        os << "  ; entry";
      os << "\n";
      break;
    }
    case Node::Release: {
      const Sema &s = g.semaTable.semas[n->sema];
      os << treePrefix(depth) << "|- r  " << s.name;
      if (n->count > 1)
        os << "(" << n->count << ")";
      os << "" << "  "
         << ownerStr(anchor, n->owner) << " [";
      bool first = true;
      for (AsyncOp p : n->payloads) {
        if (!first)
          os << ",";
        first = false;
        os << asyncOpStr(p);
      }
      os << "]\n";
      break;
    }
    case Node::For: {
      os << treePrefix(depth) << "|- scf.for";
      if (gpu::hasWarpSpecializeTag(n->op))
        os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      printPieceRecord(os, n, n->op);
      printThreadInfo(os, g, n);
      os << "\n";
      dumpSyncChain(g, n->children[0], depth + 1, n, 0);
      break;
    }
    case Node::If: {
      os << treePrefix(depth) << "|- scf.if";
      printPieceRecord(os, n, anchor);
      printThreadInfo(os, g, n);
      os << "\n";
      os << treePrefix(depth + 1) << "|- then\n";
      dumpSyncChain(g, n->children[0], depth + 2, n, 0);
      bool virtualElse = !cast<scf::IfOp>(n->op).elseBlock();
      os << treePrefix(depth + 1) << "|- else"
         << (virtualElse ? " (virtual)" : "") << "\n";
      if (n->children.size() > 1)
        dumpSyncChain(g, n->children[1], depth + 2, n, 1);
      break;
    }
    case Node::Enter:
      os << treePrefix(depth) << "|- ENTER";
      printPieceRecord(os, n, anchor);
      os << "\n";
      break;
    case Node::Exit:
      os << treePrefix(depth) << "|- EXIT";
      printPieceRecord(os, n, anchor);
      printYieldInfo(os, g, n, region, chainIdx);
      os << "\n";
      break;
    case Node::Func:
      break;
    }
  }
}

static void dumpGroupSyncDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "SYNC-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  if (!g.root->children.empty())
    dumpSyncChain(g, g.root->children[0], 1, nullptr, 0);
  // Semaphore table, grouped by component ascending, then sema id.
  unsigned numComps = 0;
  for (CompId c : g.pieceTable.pieceComp)
    numComps = std::max(numComps, c + 1);
  for (CompId comp = 0; comp < numComps; ++comp) {
    std::string line;
    llvm::raw_string_ostream ls(line);
    bool any = false;
    for (const Sema &s : g.semaTable.semas) {
      if (s.component != comp)
        continue;
      if (any)
        ls << " ";
      any = true;
      ls << s.name << "{count=" << s.count;
      if (s.isEntry)
        ls << " entry inherit=" << ownerStr(nullptr, s.inheritStamp);
      ls << "}";
    }
    if (any)
      os << "  SEMAS c" << comp << ": " << ls.str() << "\n";
  }
  if (g.semaTable.semas.empty()) {
    os << "  BACKING: untouched (no semaphores)\n";
    return;
  }
  os << "  BACKING: numStages=" << g.backingPlan.numStages << " anchor=";
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(g.backingPlan.hoistAnchor))
    os << "before scf.for(tag="
       << (gpu::hasWarpSpecializeTag(forOp)
               ? std::to_string(*gpu::getWarpSpecializeTag(forOp))
               : std::string("-"))
       << ")";
  else
    os << "function-entry";
  os << "\n";
}

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_
