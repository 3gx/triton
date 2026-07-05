// SYNC analysis and scheduling; see sema-docs/insert-semas/sync-dag-1.md.
#include "InsertSemas.h"
#include <limits>
#include <numeric>

namespace mlir::triton::nvws_semas {

struct ActiveUse {
  Owner owner;
  Node *node = nullptr;
  SmallVector<AsyncOp, 1> payloads;
  // Owners whose existing dependency already orders this node before them.
  SmallVector<int64_t, 2> orderedBefore;
};

struct VersionSource {
  Owner producer;    // logical producer of the current version
  Owner sourceOwner; // owner of the chain-local source node
  Node *node = nullptr;
  SmallVector<AsyncOp, 1> payloads;
};

struct PieceState {
  // Reads move their active use but not this source. New readers therefore
  // fan out from the write, or from the ENTER node that represents an outer
  // write in a child chain.
  VersionSource source;
  SmallVector<ActiveUse, 2> uses; // producer and/or readers, join order
  bool initialized() const { return source.node != nullptr; }
};
using ChainState = std::map<PieceId, PieceState>;

// Live tokens known while walking one chain. Their order records the existing
// deterministic handoff policy: the last token is used when an access cannot
// reuse its owner's token and no memory edge already supplies one.
struct Tokens {
  struct Token {
    Owner owner;
    Node *node = nullptr;
    SmallVector<AsyncOp, 1> payloads;
  };
  SmallVector<Token, 2> live;

  const Token *find(const Owner &owner) const {
    if (!owner)
      return nullptr;
    for (const Token &token : live)
      if (sameOwner(token.owner, owner))
        return &token;
    return nullptr;
  }
  const Token *last() const {
    return !live.empty() && live.back().node ? &live.back() : nullptr;
  }
  void remember(const Owner &owner) {
    if (owner && !find(owner))
      live.push_back(Token{owner, nullptr, {}});
  }
  void record(const Owner &owner, Node *node,
              const SmallVector<AsyncOp, 1> &payloads) {
    auto it = llvm::find_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
    if (it != live.end())
      live.erase(it);
    live.push_back(Token{owner, node, payloads});
  }
  void clear() { live.clear(); }
};

struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  SmallVector<AsyncOp, 1> payloads;
  SmallVector<PieceId, 2> pieces;
};
static ActiveUse *findUse(PieceState &piece, const Owner &who) {
  for (ActiveUse &use : piece.uses)
    if (sameOwner(use.owner, who))
      return &use;
  return nullptr;
}
static void setVersionSource(PieceState &piece, const Owner &producer,
                             const Owner &sourceOwner, Node *node,
                             const SmallVector<AsyncOp, 1> &payloads) {
  piece.source = VersionSource{producer, sourceOwner, node, payloads};
}
static void unionPayloads(SmallVector<AsyncOp, 1> &into, const SmallVector<AsyncOp, 1> &from) {
  for (AsyncOp p : from)
    if (!llvm::is_contained(into, p))
      into.push_back(p);
  llvm::sort(into, [](AsyncOp a, AsyncOp b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
}
// Advance one piece's data game. This emits only RAW/WAR dependencies; the
// chain walk separately adds a token handoff when no data edge can provide
// a token for the toucher.
static void applyTouch(ChainState &st, PieceId p, const Owner &who, Effect effect, Node *node,
                       const SmallVector<AsyncOp, 1> &payloads,
                       SmallVector<EdgeRec> &edges, bool wsAdopt) {
  PieceState &piece = st[p];
  if (!piece.initialized()) { // first touch in an unseeded function chain
    setVersionSource(piece, who, who, node, payloads);
    piece.uses.assign(1, ActiveUse{who, node, payloads, {}});
    return;
  }
  if (effect == Effect::W) {
    for (ActiveUse &use : piece.uses) {
      if (sameOwner(use.owner, who))
        continue;
      if (wsAdopt && !use.owner.has_value())
        continue; // adoption: no edge from a root use
      if (llvm::is_contained(use.orderedBefore, ownerKey(who)))
        continue; // transitively synchronized — edge redundant
      edges.push_back(
          EdgeRec{use.node, node, use.owner, who, use.payloads, {p}});
    }
    piece.uses.assign(1, ActiveUse{who, node, payloads, {}});
    setVersionSource(piece, who, who, node, payloads);
    return;
  }
  if (ActiveUse *use = findUse(piece, who)) { // reread (producer or reader)
    use->node = node;
    use->payloads = payloads;
    use->orderedBefore.clear(); // the active node moved
    return;
  }
  assert(piece.source.node && "initialized piece without a version source");
  if (!(wsAdopt && !piece.source.sourceOwner.has_value()) &&
      !sameOwner(piece.source.sourceOwner, who)) {
    edges.push_back(EdgeRec{piece.source.node, node, piece.source.sourceOwner,
                            who, piece.source.payloads, {p}});
    // The source edge orders an active use only while it still names the
    // source node. After a producer reread, the old write cannot discharge the
    // reread's WAR obligation to a later foreign writer.
    if (ActiveUse *source = findUse(piece, piece.source.sourceOwner))
      if (source->node == piece.source.node &&
          !llvm::is_contained(source->orderedBefore, ownerKey(who)))
        source->orderedBefore.push_back(ownerKey(who));
  }
  piece.uses.push_back(ActiveUse{who, node, payloads, {}});
}

static bool canReuseTokenForPiece(ChainState &st, PieceId p, const Owner &who,
                                  Effect effect) {
  auto it = st.find(p);
  if (it == st.end() || !it->second.initialized())
    return false;
  PieceState &piece = it->second;
  if (effect == Effect::W) {
    for (ActiveUse &use : piece.uses)
      if (!sameOwner(use.owner, who) &&
          !llvm::is_contained(use.orderedBefore, ownerKey(who)))
        return false;
    return true;
  }
  return findUse(piece, who) != nullptr;
}

static bool nodeTouchesPiece(GroupDag &g, Node *n, PieceId piece) {
  if (n->kind == Node::Access)
    return touchesPiece(g, n, piece);
  if (n->isRegion())
    return n->pieceInfo.count(piece) > 0;
  return false;
}
static bool pieceTouchedAfter(GroupDag &g, Node *regionNode, PieceId piece) {
  for (Node *r = regionNode; r && r->kind != Node::Func;) {
    for (Node *m = r->next; m; m = m->next)
      if (nodeTouchesPiece(g, m, piece))
        return true;
    r = r->parent;
  }
  return false;
}
static std::map<PieceId, SmallVector<AsyncOp, 1>>
walkChain(GroupDag &g, Node *head, ChainState &st, SmallVector<EdgeRec> &edges, bool underFor) {
  std::map<PieceId, Owner> carried;
  if (head->kind == Node::Enter)
    for (auto &[p, pi] : sortedPieceInfo(head))
      carried[p] = pi.owner;
  // A uniform ENTER owner has a reusable incoming token. It has no source
  // node until an access uses it, so it cannot yet supply a handoff edge.
  Tokens tokens;
  if (head->kind == Node::Enter)
    if (auto owner = uniformPieceOwner(head); owner && owner->has_value())
      tokens.remember(*owner);
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
      break; // seeding happened at chain entry
    case Node::Access: {
      std::map<PieceId, Effect> eff;
      forEachTouchedPiece(g, n, [&](PieceId p, Effect e) { mergeEffect(eff, p, e); });
      const Tokens::Token *lastToken = tokens.last();
      bool ownerDiffers = lastToken && n->owner &&
                          !sameOwner(lastToken->owner, n->owner);
      bool canReuse = tokens.find(n->owner) &&
                      llvm::all_of(eff, [&](const auto &item) {
                        return canReuseTokenForPiece(
                            st, item.first, n->owner, item.second);
                      });
      SmallVector<PieceId, 2> pieces;
      for (auto &[p, e] : eff)
        pieces.push_back(p);
      size_t edgeStart = edges.size();
      SmallVector<AsyncOp, 1> pay;
      pay.push_back(asyncPayloadOf(n->op));
      for (auto &[p, e] : eff) {
        applyTouch(st, p, n->owner, e, n, pay, edges,
                   /*wsAdopt=*/false);
      }
      bool reusesToken = edges.size() == edgeStart && canReuse;
      bool keepsLastToken = ownerDiffers && reusesToken;
      if (reusesToken) {
        markTokenReuse(n, n->owner);
      } else if (ownerDiffers && edges.size() == edgeStart) {
        assert(lastToken && lastToken->node &&
               "last token without a source node");
        edges.push_back(EdgeRec{lastToken->node, n, lastToken->owner,
                                n->owner, lastToken->payloads, pieces});
      }
      if (n->owner.has_value() && !keepsLastToken)
        tokens.record(n->owner, n, pay);
      break;
    }
    case Node::For:
    case Node::If: {
      bool wsAdopt = n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op);
      auto infos = sortedPieceInfo(n);
      struct PreVersion {
        Owner producer;
        SmallVector<AsyncOp, 1> payloads;
      };
      std::map<PieceId, PreVersion> preVersion;
      for (auto &[p, pi] : infos) {
        auto it = st.find(p);
        if (it == st.end() || !it->second.initialized())
          continue;
        PieceState &piece = it->second;
        preVersion.emplace(p,
                           PreVersion{piece.source.producer,
                                      piece.source.payloads});
      }
      for (auto &[p, pi] : infos) {
        SmallVector<AsyncOp, 1> none;
        none.push_back(AsyncOp::NONE); // placeholder; replaced below
        applyTouch(st, p, pi.owner, pi.effect, n, none, edges, wsAdopt);
      }
      std::map<PieceId, SmallVector<AsyncOp, 1>> unionRet;
      for (Node *childHead : n->children) {
        ChainState child;
        for (auto &[p, pi] : sortedPieceInfo(childHead)) {
          PieceState piece;
          Owner childCarried = pi.owner;
          auto pre = preVersion.find(p);
          Owner producer =
              pre != preVersion.end() ? pre->second.producer : childCarried;
          SmallVector<AsyncOp, 1> seedPay;
          if (pre != preVersion.end() &&
              sameOwner(childCarried, pre->second.producer))
            seedPay = pre->second.payloads; // payload-seed IMPORT
          else
            seedPay.push_back(AsyncOp::NONE); // transitivity witness
          setVersionSource(piece, producer, childCarried, childHead, seedPay);
          piece.uses.assign(
              1, ActiveUse{childCarried, childHead, seedPay, {}});
          child.emplace(p, std::move(piece));
        }
        auto ret = walkChain(g, childHead, child, edges, underFor || n->kind == Node::For);
        for (auto &[p, pay] : ret)
          unionPayloads(unionRet[p], pay);
      }
      for (auto &[p, pi] : infos) {
        PieceState &piece = st[p];
        if (ActiveUse *use = findUse(piece, pi.owner)) {
          auto it = unionRet.find(p);
          if (it != unionRet.end()) {
            use->payloads = it->second;
            if (piece.source.node == n)
              piece.source.payloads = it->second;
          }
        }
      }
      if (!infos.empty()) {
        tokens.clear();
        if (auto owner = uniformPieceOwner(n); owner && owner->has_value()) {
          SmallVector<AsyncOp, 1> none{AsyncOp::NONE};
          tokens.record(*owner, n, none);
        }
      }
      break;
    }
    case Node::Exit: {
      for (auto &[p, pi] : sortedPieceInfo(n)) {
        auto it = st.find(p);
        if (it == st.end())
          continue;
        PieceState &piece = it->second;
        bool needed = underFor || pieceTouchedAfter(g, n->parent, p);
        if (needed)
          for (ActiveUse &use : piece.uses) {
            if (sameOwner(use.owner, pi.owner))
              continue;
            if (llvm::is_contained(use.orderedBefore, ownerKey(pi.owner)))
              continue; // carried owner already synchronized — no close
            edges.push_back(EdgeRec{use.node, n, use.owner, pi.owner,
                                    use.payloads, {p}});
          }
        ActiveUse keep;
        if (ActiveUse *carriedUse = findUse(piece, pi.owner))
          keep = *carriedUse;
        else
          keep = ActiveUse{pi.owner, n, {AsyncOp::NONE}, {}};
        piece.uses.assign(1, keep);
      }
      break;
    }
    case Node::Acquire:
    case Node::Release:
    case Node::Func:
      break; // not present during the walk
    }
  }
  std::map<PieceId, SmallVector<AsyncOp, 1>> result;
  for (auto &[p, who] : carried) {
    auto it = st.find(p);
    SmallVector<AsyncOp, 1> pay;
    if (it != st.end())
      if (ActiveUse *use = findUse(it->second, who))
        pay = use->payloads;
    if (pay.empty())
      pay.push_back(AsyncOp::NONE);
    result.emplace(p, pay);
  }
  return result;
}
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
static Node *newProtocolNode(GroupDag &g, Node::Kind kind, Node *parent,
                             Owner owner, SemaId sema, unsigned count) {
  Node *node = g.newNode(kind, nullptr, parent);
  node->owner = owner;
  node->sema = sema;
  node->count = count;
  return node;
}

namespace {
using SyncVec = std::map<int64_t, unsigned>; // partitionKey -> node index

struct ChainIndex {
  DenseMap<Node *, unsigned> idx;   // node -> position within its chain
  DenseMap<Node *, Node *> chainOf; // node -> chain head
};
static void indexChains(Node *head, ChainIndex &ci) {
  unsigned i = 0;
  for (Node *n = head; n; n = n->next) {
    ci.idx[n] = i++;
    ci.chainOf[n] = head;
    if (n->isRegion())
      for (Node *child : n->children)
        indexChains(child, ci);
  }
}
static bool covers(const SyncVec &v, int64_t key, unsigned srcIdx) {
  auto it = v.find(key);
  return it != v.end() && it->second >= srcIdx;
}
static LogicalResult sweepChain(GroupDag &g, Node *head, ChainIndex &ci,
                                SmallVector<EdgeRec> &edges, std::vector<bool> &drop, bool reduce,
                                ArrayRef<unsigned> checkIdxs) {
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  for (auto [i, e] : llvm::enumerate(edges))
    if (ci.chainOf.lookup(e.src) == head && ci.chainOf.lookup(e.dst) == head)
      atDst[e.dst].push_back(i);
  for (auto &[d, v] : atDst)
    llvm::stable_sort(v, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snapshotAtNode;
  SmallVector<int64_t, 2> tokenOwners;
  auto recordToken = [&](int64_t owner) {
    auto it = llvm::find(tokenOwners, owner);
    if (it != tokenOwners.end())
      tokenOwners.erase(it);
    tokenOwners.push_back(owner);
  };
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner)
        recordToken(ownerKey(pi.owner));
  auto ownerOfNode = [&](Node *n) -> Owner {
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
        bool destinationHasToken =
            !tokenOwners.empty() && tokenOwners.back() == dk;
        if (reduce && !drop[ei] && implied && destinationHasToken &&
            e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        if (drop[ei])
          continue;
        recordToken(dk); // the kept acquire supplies Q's token
        SyncVec &dv = behind[dk];
        auto snap = snapshotAtNode.find(e.src);
        if (snap != snapshotAtNode.end())
          for (auto &[k, v] : snap->second)
            if (!covers(dv, k, v))
              dv[k] = std::max(dv[k], v);
        dv[sk] = std::max(dv[sk], srcIdx);
      }
    if (Owner o = ownerOfNode(n)) {
      behind[ownerKey(o)][ownerKey(o)] = ci.idx.lookup(n);
      snapshotAtNode[n] = behind[ownerKey(o)];
    }
  }
  if (!reduce)
    for (unsigned ei : checkIdxs) {
      EdgeRec &e = edges[ei];
      if (ci.chainOf.lookup(e.src) != head)
        continue;
      if (e.dst->kind == Node::Exit)
        continue; // EXIT closes are re-verified by sweepTraversalClosure
      if (!covers(behind[ownerKey(e.dstOwner)], ownerKey(e.srcOwner), ci.idx.lookup(e.src)))
        return semaError(e.dst->op ? e.dst->op : g.root->op)
               << "transitive-reduction closure violation: dropped edge is "
                  "not implied by the final edge set";
    }
  return success();
}
static LogicalResult sweepTraversalClosure(GroupDag &g, Node *head, ChainIndex &ci,
                                   SmallVector<EdgeRec> &edges, std::vector<bool> &drop, bool reduce,
                                   ArrayRef<unsigned> checkIdxs) {
  Owner firstAccessOwner;
  for (Node *n = head; n && !firstAccessOwner; n = n->next)
    if (n->kind == Node::Access && n->owner)
      firstAccessOwner = n->owner;
  if (!firstAccessOwner)
    return success();
  constexpr unsigned kPass2 = 1u << 20;
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  SmallVector<unsigned, 4> closes;
  for (auto [i, e] : llvm::enumerate(edges)) {
    if (drop[i] || ci.chainOf.lookup(e.src) != head || ci.chainOf.lookup(e.dst) != head)
      continue;
    if (e.dst->kind == Node::Exit && e.src->kind == Node::Access && e.srcOwner && e.dstOwner)
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
  std::map<int64_t, unsigned> tokenAvailableAt; // ownerKey -> pass-2 node
  auto applyKept = [&](EdgeRec &e, unsigned srcIdx, DenseMap<Node *, SyncVec> &snaps) {
    SyncVec &dv = behind[ownerKey(e.dstOwner)];
    auto sn = snaps.find(e.src);
    if (sn != snaps.end())
      for (auto &[k, v] : sn->second)
        dv[k] = std::max(dv[k], v);
    int64_t sk = ownerKey(e.srcOwner);
    dv[sk] = std::max(dv[sk], srcIdx);
  };
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
        tokenAvailableAt.try_emplace(ownerKey(edges[ei].dstOwner),
                                     kPass2 + ci.idx.lookup(n));
      }
    auto ct = closeAt.find(n);
    if (ct != closeAt.end())
      for (unsigned ei : ct->second) {
        EdgeRec &e = edges[ei];
        int64_t dk = ownerKey(e.dstOwner);
        bool covered = covers(behind[dk], ownerKey(e.srcOwner), ci.idx.lookup(e.src));
        bool hasToken = tokenAvailableAt.count(dk);
        bool isInitialTokenClose = sameOwner(e.dstOwner, firstAccessOwner);
        if (reduce) {
          if (!drop[ei] && covered && hasToken && !isInitialTokenClose)
            drop[ei] = true;
          if (!drop[ei]) // kept close: provides its ordering at dst
            applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (!drop[ei]) {
          applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (llvm::is_contained(checkIdxs, ei) &&
                   !(covered && hasToken)) {
          result = semaError(e.src->op ? e.src->op : g.root->op)
                   << "traversal-closure violation: dropped close not implied";
        }
      }
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = kPass2 + ci.idx.lookup(n);
      snap2[n] = behind[ownerKey(n->owner)];
    }
  }
  return result;
}

// Remove only edges re-proved by kept waits, program order, and loop closure.
static LogicalResult reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (g.root->children.empty() || edges.empty())
    return success();
  ChainIndex ci;
  indexChains(g.root->children[0], ci);
  std::vector<bool> drop(edges.size(), false);
  SmallVector<Node *, 8> heads;
  DenseSet<Node *> seen;
  for (auto &[node, h] : ci.chainOf)
    if (seen.insert(h).second)
      heads.push_back(h);
  llvm::sort(heads, [&](Node *a, Node *b) {
    return ci.idx.lookup(a) < ci.idx.lookup(b);
  });
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, edges, drop, /*reduce=*/true, {})))
      return failure();
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, edges, drop, /*reduce=*/true, {})))
        return failure();
  SmallVector<unsigned, 8> dropped;
  for (auto [i, d] : llvm::enumerate(drop))
    if (d)
      dropped.push_back(i);
  if (dropped.empty())
    return success();
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, edges, drop, /*reduce=*/false, dropped)))
      return failure();
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, edges, drop, /*reduce=*/false, dropped)))
        return failure();
  SmallVector<EdgeRec> kept;
  for (auto [i, e] : llvm::enumerate(edges))
    if (!drop[i])
      kept.push_back(e);
  edges = std::move(kept);
  return success();
}
} // namespace
static void absorbEdge(EdgeRec &dst, const EdgeRec &src) {
  unionPayloads(dst.payloads, src.payloads);
  for (PieceId piece : src.pieces)
    if (!llvm::is_contained(dst.pieces, piece))
      dst.pieces.push_back(piece);
  llvm::sort(dst.pieces);
}
static bool followsInChain(Node *node, Node *other) {
  for (Node *next = other->next; next; next = next->next)
    if (next == node)
      return true;
  return false;
}
template <typename KeyFn>
static SmallVector<EdgeRec> mergeEdges(ArrayRef<EdgeRec> edges, KeyFn key, bool keepLatestSource) {
  using Key = decltype(key(std::declval<const EdgeRec &>()));
  DenseMap<Key, unsigned> index;
  SmallVector<EdgeRec> merged;
  for (const EdgeRec &edge : edges) {
    auto [it, inserted] = index.try_emplace(key(edge), merged.size());
    if (inserted) {
      merged.push_back(edge);
      llvm::sort(merged.back().payloads, [](AsyncOp a, AsyncOp b) {
        return static_cast<int>(a) < static_cast<int>(b);
      });
      llvm::sort(merged.back().pieces);
      continue;
    }
    EdgeRec &dst = merged[it->second];
    if (keepLatestSource && followsInChain(edge.src, dst.src))
      dst.src = edge.src;
    absorbEdge(dst, edge);
  }
  return merged;
}

static LogicalResult buildEdgesAndSemas(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (failed(reduceEdges(g, edges)))
    return failure();
  auto deduped = mergeEdges(edges, [&](const EdgeRec &edge) {
    return std::make_tuple(edge.src, edge.dst, ownerKey(edge.srcOwner));
  }, false);
  auto collapsed = mergeEdges(deduped, [&](const EdgeRec &edge) {
    return std::make_tuple(edge.dst, ownerKey(edge.srcOwner));
  }, true);
  struct DstGroup {
    Node *dst;
    SmallVector<unsigned, 2> idxs;
    int sema = -1;
  };
  llvm::MapVector<std::tuple<Node *, int64_t>, unsigned> dstIndex;
  SmallVector<DstGroup> groups;
  for (auto [i, e] : llvm::enumerate(collapsed)) {
    auto key = std::make_tuple(e.dst, ownerKey(e.dstOwner));
    auto it = dstIndex.find(key);
    if (it == dstIndex.end()) {
      dstIndex.try_emplace(key, groups.size());
      groups.push_back(DstGroup{e.dst, {static_cast<unsigned>(i)}, -1});
    } else {
      groups[it->second].idxs.push_back(i);
    }
  }
  auto groupAcquirer = [&](const DstGroup &grp) {
    return collapsed[grp.idxs.front()].dstOwner;
  };
  auto findRegainGroup = [&](Node *forNode, const Owner &acq) -> int {
    int best = -1;
    for (Node *m = forNode->children[0]; m; m = m->next) {
      auto it = dstIndex.find(std::make_tuple(m, ownerKey(acq)));
      if (it == dstIndex.end())
        continue;
      best = static_cast<int>(it->second);
    }
    return best;
  };
  auto createSema = [&](DstGroup &grp) -> LogicalResult {
    SemaId sid = g.semas.size();
    Sema s;
    s.name = "S" + std::to_string(sid);
    s.count = grp.idxs.size();
    for (unsigned idx : grp.idxs)
      for (PieceId p : collapsed[idx].pieces)
        if (!llvm::is_contained(s.pieces, p))
          s.pieces.push_back(p);
    llvm::sort(s.pieces);
    grp.sema = static_cast<int>(sid);
    g.semas.push_back(std::move(s));
    return success();
  };
  for (DstGroup &grp : groups) {
    if (grp.sema != -1)
      continue;
    for (unsigned i = 0; i < grp.idxs.size(); ++i)
      for (unsigned j = i + 1; j < grp.idxs.size(); ++j)
        if (sameOwner(collapsed[grp.idxs[i]].srcOwner, collapsed[grp.idxs[j]].srcOwner))
          return semaError(grp.dst->op ? grp.dst->op : g.root->op)
                 << "fan-in sources share a partition — not expressible as one semaphore";
    if (grp.dst->kind == Node::For) {
      int t = findRegainGroup(grp.dst, groupAcquirer(grp));
      if (t >= 0) {
        if (groups[t].sema == -1)
          if (failed(createSema(groups[t])))
            return failure();
        grp.sema = groups[t].sema;
        Sema &s = g.semas[grp.sema];
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
  DenseMap<Node *, Node *> lastAfter; // release insertion cursor per source
  for (DstGroup &grp : groups) {
    Sema &s = g.semas[grp.sema];
    unsigned m = grp.idxs.size();
    unsigned relCount = 1;
    if (m != s.count) {
      if (m == 1)
        relCount = s.count;
      else
        return semaError(grp.dst->op ? grp.dst->op : g.root->op) << "destination group with "
               << m << " sources cannot meet semaphore " << s.name << " pending count " << s.count;
    }
    Node *acq = newProtocolNode(g, Node::Acquire, grp.dst->parent, groupAcquirer(grp), grp.sema, s.count);
    Node *dstAnchor = grp.dst;
    if (grp.dst->kind == Node::Exit && acq->owner) {
      Node *head = grp.dst;
      while (head->prev)
        head = head->prev;
      Owner firstAccessOwner;
      Node *firstTouch = nullptr;
      for (Node *r = head; r; r = r->next) {
        if (r->kind != Node::Access || !r->owner)
          continue;
        if (!firstAccessOwner)
          firstAccessOwner = r->owner;
        if (!firstTouch && sameOwner(r->owner, acq->owner))
          firstTouch = r;
      }
      if (firstAccessOwner && !sameOwner(firstAccessOwner, acq->owner) &&
          firstTouch) {
        dstAnchor = firstTouch;
        s.isEntry = true; // initially released; no pre-loop entry instance
        s.entryTokenOwner = acq->owner;
      }
    }
    acq->scheduleAnchor = dstAnchor;
    spliceBefore(acq, dstAnchor);
    s.expectedReleases += m * relCount;
    for (unsigned idx : grp.idxs) {
      EdgeRec &e = collapsed[idx];
      Node *rel = newProtocolNode(g, Node::Release, e.src->parent, e.srcOwner, grp.sema, relCount);
      rel->payloads = e.payloads;
      rel->sat = acq;
      rel->scheduleAnchor = e.src;
      if (nodeReusesToken(e.src, e.srcOwner))
        markTokenReuse(rel, e.srcOwner);
      Node *anchor = lastAfter.lookup(e.src);
      spliceAfter(rel, anchor ? anchor : e.src);
      lastAfter[e.src] = rel;
    }
  }
  return success();
}
static bool nodeInvolvesComp(GroupDag &g, Node *n) {
  if (n->kind == Node::Access)
    return nodeTouchesGroup(g, n);
  if (n->isRegion())
    return !n->pieceInfo.empty();
  return false;
}
static Node *lastAcquireOfCompInChain(GroupDag &g, Node *head) {
  Node *found = nullptr;
  forEachNode(head, [&](Node *n) {
    if (n->kind == Node::Acquire)
      found = n;
  });
  return found;
}
static Owner firstAccessOwnerOfComp(GroupDag &g, Node *head, bool &found) {
  Owner owner;
  forEachNode(head, [&](Node *n) {
    if (!found && n->kind == Node::Access && nodeTouchesGroup(g, n)) {
      found = true;
      owner = n->owner;
    }
  });
  return owner;
}

static LogicalResult insertEntryAcquires(GroupDag &g) {
  Node *top = g.root->children.empty() ? nullptr : g.root->children[0];
  if (!top)
    return success();
  {
    if (g.semas.empty())
      return success();
    Node *chainHead = top;
    SmallVector<Node *, 4> nodes;
    auto collectNodes = [&](Node *head) {
      nodes.clear();
      for (Node *n = head; n; n = n->next)
        if (nodeInvolvesComp(g, n))
          nodes.push_back(n);
    };
    collectNodes(chainHead);
    while (nodes.size() == 1 && nodes[0]->kind == Node::If) {
      Node *onlyChild = nullptr;
      int cnt = 0;
      for (Node *child : nodes[0]->children) {
        bool involves = false;
        for (Node *n = child; n; n = n->next)
          if (nodeInvolvesComp(g, n))
            involves = true;
        if (involves) {
          onlyChild = child;
          ++cnt;
        }
      }
      if (cnt != 1)
        break;
      chainHead = onlyChild;
      collectNodes(chainHead);
    }
    if (nodes.empty())
      return semaError(g.root->op) << "group with sync but no placement nodes";
    bool foundEntryTokenOwner = false;
    Owner entryTokenOwner =
        firstAccessOwnerOfComp(g, top, foundEntryTokenOwner);
    if (!foundEntryTokenOwner)
      return semaError(g.root->op) << "group has no access nodes";
    Node *regain = nullptr;
    for (Node *node : llvm::reverse(nodes))
      if (node->kind == Node::For) {
        regain = lastAcquireOfCompInChain(g, node->children[0]);
        if (regain)
          break;
      }
    if (regain) {
      Sema &s = getSema(g, regain);
      s.isEntry = true; // first event in chain order is an acquire
      s.entryTokenOwner = entryTokenOwner;
      Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent,
                                  std::nullopt, regain->sema, regain->count);
      spliceBefore(acq, nodes.front());
    } else {
      SemaId sid = g.semas.size();
      Sema s;
      s.name = "E" + std::to_string(sid);
      for (auto [p, piece] : llvm::enumerate(g.pieceTable.pieces))
        s.pieces.push_back(static_cast<PieceId>(p));
      s.count = 1;
      s.isEntry = true;
      s.expectedReleases = 1; // the terminal release
      s.entryTokenOwner = entryTokenOwner;
      Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent,
                                  std::nullopt, sid, 1);
      spliceBefore(acq, nodes.front());
      Node *terminal = nodes.back();
      Owner owner = terminal->kind == Node::Access ? terminal->owner
                        : sortedPieceInfo(terminal).front().second.owner;
      Node *rel = newProtocolNode(g, Node::Release, terminal->parent, owner, sid, 1);
      rel->payloads.push_back(AsyncOp::NONE);
      Node *anchor = terminal;
      while (anchor->next && anchor->next->kind == Node::Release)
        anchor = anchor->next;
      spliceAfter(rel, anchor);
      g.semas.push_back(std::move(s));
    }
  }
  return success();
}
static Node *chainFinalForComp(GroupDag &g, Node *head) {
  Node *final = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire)
      final = n;
    if (n->isRegion() && !n->crossings.empty())
      final = n;
  }
  return final;
}
static Owner finalOwner(GroupDag &g, Node *final) {
  if (final->kind == Node::Acquire)
    return final->owner;
  if (!final->crossings.empty())
    return final->crossings.front().tokenOwner;
  return std::nullopt;
}

static void computeCrossings(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    Crossing cr;
    bool any = false;
    for (Node *child : n->children) {
      Node *f = chainFinalForComp(g, child);
      cr.finals.push_back(f);
      if (f) {
        any = true;
        cr.tokenOwner = finalOwner(g, f);
      }
    }
    if (any)
      n->crossings.push_back(std::move(cr));
  });
}
static bool tokenUsedBeforeNextAcquire(GroupDag &g, Node *start) {
  for (Node *n = start; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      return false; // a fresh token supersedes the earlier one
    case Node::Release:
      return true;
    case Node::Access:
      if (nodeTouchesGroup(g, n))
        return true;
      break;
    case Node::For:
    case Node::If:
      if (!n->crossings.empty())
        return true;
      break;
    default:
      break;
    }
  }
  return false;
}
static bool regionLiveFor(const Node *region) {
  if (!region)
    return false; // function chain
  return !region->crossings.empty();
}

static void pruneDeadIfCrossings(GroupDag &g, Node *head, Node *region) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes))
    if (n->kind == Node::If)
      llvm::erase_if(n->crossings, [&](const Crossing &c) {
        return !tokenUsedBeforeNextAcquire(g, n->next) &&
               !regionLiveFor(region);
      });
  for (Node *n : nodes)
    if (n->isRegion())
      for (Node *child : n->children)
        pruneDeadIfCrossings(g, child, n);
}
static void collectParts(Node *head, SmallVector<int, 4> &parts) {
  forEachNode(head, [&](Node *n) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire || n->kind == Node::Release) &&
        n->owner.has_value())
      if (!llvm::is_contained(parts, n->owner->first))
        parts.push_back(n->owner->first);
  });
}
static bool crossesComp(const Node *n) {
  return !n->crossings.empty();
}
static scf::ForOp outerWSLoop(scf::ForOp loop);
static const Crossing *findCrossing(const Node *n) {
  return n->crossings.empty() ? nullptr : &n->crossings.front();
}
static bool allEnclosersCanDrop(const Node *F) {
  if (F->op && gpu::hasWarpSpecializeTag(F->op))
    return true;
  for (const Node *p = F->parent; p; p = p->parent) {
    if (p->kind == Node::Func)
      return true;
    if (p->kind == Node::If)
      return false;
    if (p->kind != Node::For)
      continue;
    if (p->op && gpu::hasWarpSpecializeTag(p->op))
      return true;
  }
  return true;
}
static bool regionResultConsumedAfter(GroupDag &g, Node *region) {
  for (Node *m = region->next; m; m = m->next) {
    if (m->kind == Node::Acquire)
      return false;
    if (m->kind == Node::Release)
      return true;
    if (m->kind == Node::Access && nodeInvolvesComp(g, m))
      return true;
    if (m->isRegion() && crossesComp(m))
      return true;
  }
  Node *p = region->parent;
  if (!p || !p->isRegion())
    return false;
  for (const Crossing &x : p->crossings)
    if (llvm::any_of(x.finals, [&](Node *f) { return f == region; }))
      return regionResultConsumedAfter(g, p);
  return false;
}
static bool prefixNodeIsSingleBufferView(Node *F, Node *bufferNode) {
  if (!F || !F->op || gpu::hasWarpSpecializeTag(F->op))
    return true;
  if (!bufferNode || !bufferNode->op)
    return false;
  auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(bufferNode->op);
  return !alloc || !alloc.getSrc();
}
static bool isAcquireForComp(GroupDag &g, const Node *n) {
  return n->kind == Node::Acquire;
}
static bool isReleaseForComp(GroupDag &g, const Node *n) {
  return n->kind == Node::Release;
}
static bool isAccessForComp(GroupDag &g, Node *n) {
  return n->kind == Node::Access && nodeInvolvesComp(g, n);
}
static bool isRegionCrossingForComp(const Node *n) {
  return n && n->isRegion() && crossesComp(n);
}
static bool nodeHasCompEvent(GroupDag &g, Node *n) {
  return isAcquireForComp(g, n) || isReleaseForComp(g, n) || nodeInvolvesComp(g, n);
}
static bool isTokenEvent(GroupDag &g, Node *n) {
  return isAcquireForComp(g, n) || isReleaseForComp(g, n) ||
         isAccessForComp(g, n) || isRegionCrossingForComp(n);
}
static bool precedesInChain(Node *src, Node *dst) {
  for (Node *n = src->next; n; n = n->next)
    if (n == dst)
      return true;
  return false;
}
static bool regionEntryOwner(GroupDag &g, Node *region, Owner &owner) {
  bool found = false;
  for (auto &[p, pi] : sortedPieceInfo(region)) {
    if (!found) {
      owner = pi.owner;
      found = true;
      continue;
    }
    if (!sameOwner(owner, pi.owner))
      return false;
  }
  return found;
}
static bool chainHasCompEvent(GroupDag &g, Node *head) {
  for (Node *n = head; n; n = n->next)
    if (nodeHasCompEvent(g, n))
      return true;
  return false;
}
static Owner returnedOwnerForFinal(GroupDag &g, Node *final, Owner incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->owner;
  if (final->isRegion())
    if (const Crossing *c = findCrossing(final))
      return c->tokenOwner;
  return std::nullopt;
}
static std::optional<SemaId>
returnedSemaForFinal(GroupDag &g, Node *final, SemaId incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->sema;
  if (!final->isRegion())
    return std::nullopt;
  const Crossing *c = findCrossing(final);
  if (!c)
    return std::nullopt;
  std::optional<SemaId> common;
  if (final->kind == Node::For)
    common = incoming; // zero-trip path.
  for (unsigned i = 0, e = final->children.size(); i < e; ++i) {
    Node *childFinal = i < c->finals.size() ? c->finals[i] : nullptr;
    std::optional<SemaId> child = returnedSemaForFinal(g, childFinal, incoming);
    if (!child)
      return std::nullopt;
    if (!common)
      common = child;
    else if (*common != *child)
      return std::nullopt;
  }
  return common;
}
static bool isHoldTransparentRegion(GroupDag &g, Node *region, Owner holdOwner) {
  const Crossing *rc = findCrossing(region);
  if (!rc)
    return false;
  Owner entryOwner;
  if (!regionEntryOwner(g, region, entryOwner) || !sameOwner(entryOwner, holdOwner))
    return false;
  if (rc->finals.size() > region->children.size())
    return false;
  for (unsigned i = 0, e = region->children.size(); i < e; ++i) {
    Node *childFinal = i < rc->finals.size() ? rc->finals[i] : nullptr;
    if (!childFinal && chainHasCompEvent(g, region->children[i]))
      return false;
    if (childFinal && chainHasCompEvent(g, childFinal->next))
      return false;
    Owner returned = returnedOwnerForFinal(g, childFinal, entryOwner);
    if (!sameOwner(returned, holdOwner))
      return false;
  }
  return true;
}
static Hold threadedTokenHold(Node *regain = nullptr,
                              Hold::Blocker blocker = Hold::Blocker::NONE) {
  Hold h;
  h.outcome = Hold::Outcome::THREADED;
  h.blocker = blocker;
  h.regain = regain;
  return h;
}
static Hold childOwnsHold(Node *regain) {
  Hold h;
  h.outcome = Hold::Outcome::CHILD_OWNS;
  h.regain = regain;
  return h;
}
static bool childOwnsToken(const Node *region) {
  if (const Crossing *child = findCrossing(region))
    return !child->hold.threadsToken();
  return false;
}
static bool hasTrailingCompUse(GroupDag &g, Node *regain) {
  for (Node *m = regain->next; m; m = m->next)
    if (isAccessForComp(g, m) || isRegionCrossingForComp(m) || isReleaseForComp(g, m))
      return true;
  return false;
}
static Node *findBridgeAcquireAfter(GroupDag &g, Node *F, SemaId feedSema, Owner owner,
                                    Node *existingBridgeRelease) {
  for (Node *m = F->next; m; m = m->next) {
    if (!isAcquireForComp(g, m) || m->sema != feedSema || !sameOwner(m->owner, owner))
      continue;
    for (Node *tail = m->next; tail; tail = tail->next)
      if (isTokenEvent(g, tail) && tail != existingBridgeRelease)
        return nullptr;
    return m;
  }
  return nullptr;
}

static Node *findHoldFeedAcquire(GroupDag &g, Node *F) {
  for (Node *cur = F;; cur = cur->parent) {
    for (Node *m = cur->prev; m; m = m->prev) {
      if (isAcquireForComp(g, m))
        return m;
      if (isTokenEvent(g, m))
        return nullptr;
    }
    if (!cur->parent || !cur->parent->isRegion())
      return nullptr;
  }
}

struct HoldPrefix {
  Node *firstToucher = nullptr;
  Node *closingRelease = nullptr;
  SmallVector<Node *, 4> nodes;
};
static std::optional<HoldPrefix>
matchHoldPrefix(GroupDag &g, Node *F, Owner tokenOwner, Node *regain,
                bool regionTail) {
  HoldPrefix p;
  unsigned releases = 0;
  bool releaseBeforeFirstToucher = false;
  for (Node *m = F->children[0]; m; m = m->next) {
    if (regionTail && m == regain)
      break;
    if (isAcquireForComp(g, m))
      break;
    if (isRegionCrossingForComp(m)) {
      if (!p.firstToucher || !isHoldTransparentRegion(g, m, tokenOwner))
        return std::nullopt;
      p.nodes.push_back(m);
      continue;
    }
    if (isAccessForComp(g, m)) {
      if (!p.firstToucher) {
        p.firstToucher = m;
        releaseBeforeFirstToucher = releases != 0;
      }
      p.nodes.push_back(m);
    }
    if (isReleaseForComp(g, m)) {
      releases += std::max(1u, m->count);
      if (!p.closingRelease)
        p.closingRelease = m;
    }
  }
  unsigned expectedReleases = regionTail ? 0 : 1;
  if (!p.firstToucher || releases != expectedReleases ||
      releaseBeforeFirstToucher)
    return std::nullopt;
  return p;
}

struct OwnerScheduleState {
  bool present = false;
  gpu::StageCluster stageCluster;
};

static OwnerScheduleState ownerScheduleAfterChain(Node *head, const Owner &owner,
                                                  OwnerScheduleState state);

static OwnerScheduleState ownerScheduleAfterRegion(Node *region, const Owner &owner,
                                                   OwnerScheduleState state) {
  Node *firstChild = region->children.empty() ? nullptr : region->children[0];
  if (auto forOp = dyn_cast_or_null<scf::ForOp>(region->op)) {
    OwnerScheduleState body = ownerScheduleAfterChain(firstChild, owner, state);
    return gpu::hasWarpSpecializeTag(forOp) ? state : body;
  }

  OwnerScheduleState thenState = ownerScheduleAfterChain(firstChild, owner, state);
  Node *elseChild = region->children.size() > 1 ? region->children[1] : nullptr;
  OwnerScheduleState elseState = ownerScheduleAfterChain(elseChild, owner, state);
  return thenState.present ? thenState : elseState;
}

static OwnerScheduleState ownerScheduleAfterChain(Node *head, const Owner &owner,
                                                  OwnerScheduleState state) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access && sameOwner(n->owner, owner)) {
      Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
      state.present = true;
      state.stageCluster = gpu::getStageCluster(completion);
      continue;
    }
    if (n->isRegion())
      state = ownerScheduleAfterRegion(n, owner, state);
  }
  return state;
}

static gpu::StageCluster ownerCompletionScheduleAtLoopExit(Node *F,
                                                           const Owner &owner) {
  Node *body = F->children.empty() ? nullptr : F->children[0];
  OwnerScheduleState state = ownerScheduleAfterChain(body, owner, {});
  return state.present ? state.stageCluster : gpu::StageCluster{};
}

static std::optional<Hold>
matchPointOfUse(GroupDag &g, Node *F, scf::ForOp forOp, const Crossing &c,
                Node *regain, Node *entryAcquire, SemaId recurrenceSema,
                bool regionTail, bool needsPostLoopAcquire) {
  bool keepsEntryAcquire = entryAcquire->sema != recurrenceSema;
  Node *bridgeAcquire = nullptr;
  if (keepsEntryAcquire) {
    if (regionTail || !needsPostLoopAcquire)
      return std::nullopt;
    const Sema &entrySema = getSema(g, entryAcquire);
    Owner entryOwner = entryAcquire->owner ? entryAcquire->owner
                                           : entrySema.entryTokenOwner;
    bridgeAcquire = c.bridgeAcquire;
    if (!bridgeAcquire)
      bridgeAcquire = findBridgeAcquireAfter(
          g, F, entryAcquire->sema, c.tokenOwner,
          c.bridgeRelease);
    if (!sameOwner(entryOwner, c.tokenOwner) || !bridgeAcquire)
      return std::nullopt;
  }

  std::optional<HoldPrefix> prefix =
      matchHoldPrefix(g, F, c.tokenOwner, regain, regionTail);
  if (!prefix || !prefixNodeIsSingleBufferView(F, prefix->firstToucher))
    return std::nullopt;
  if (needsPostLoopAcquire && !gpu::hasWarpSpecializeTag(forOp) &&
      sameOwner(prefix->firstToucher->owner, c.tokenOwner)) {
    gpu::StageCluster point = gpu::getStageCluster(prefix->firstToucher->op);
    gpu::StageCluster exit = ownerCompletionScheduleAtLoopExit(F, c.tokenOwner);
    if (point && exit && point->first != exit->first)
      return std::nullopt;
  }

  Hold h;
  h.outcome = Hold::Outcome::POINT_OF_USE;
  h.nodes = std::move(prefix->nodes);
  h.entryAcquire = entryAcquire;
  h.closingRelease = prefix->closingRelease;
  h.regain = regain;
  h.firstToucher = prefix->firstToucher;
  h.bridgeAcquire = bridgeAcquire;
  h.needsPostLoopAcquire = needsPostLoopAcquire;
  h.keepsEntryAcquire = keepsEntryAcquire;
  h.regionTail = regionTail;
  return h;
}

// Select threaded, point-of-use, or child-owned token handling for one loop
// crossing.  Structural mismatches retain the token silently; only uses after
// the recurrence acquire or after the loop are named blockers.
static Hold buildUniformHold(GroupDag &g, Node *F, const Crossing &c) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(F->op);
  if (!forOp || !gpu::hasWarpSpecializeTag(outerWSLoop(forOp)) ||
      !allEnclosersCanDrop(F))
    return threadedTokenHold();

  Node *regain = c.finals.empty() ? nullptr : c.finals[0];
  if (!regain || regain->postLoopAcquire)
    return threadedTokenHold(regain);
  if (regain->kind == Node::For && childOwnsToken(regain))
    return childOwnsHold(regain);
  bool regionTail = regain->isRegion();
  if ((regionTail && !isHoldTransparentRegion(g, regain, c.tokenOwner)) ||
      (!regionTail && regain->kind != Node::Acquire))
    return threadedTokenHold(regain);
  if (hasTrailingCompUse(g, regain))
    return threadedTokenHold(regain, Hold::Blocker::TRAILING_USE);

  Node *entryAcquire = findHoldFeedAcquire(g, F);
  if (!entryAcquire)
    return threadedTokenHold(regain);
  std::optional<SemaId> recurrenceSema =
      regionTail
          ? returnedSemaForFinal(g, regain, entryAcquire->sema)
          : std::optional<SemaId>(regain->sema);
  if (!recurrenceSema)
    return threadedTokenHold(regain);

  bool needsPostLoopAcquire = c.postLoopAcquire || regionResultConsumedAfter(g, F);
  bool sameSemaphore = entryAcquire->sema == *recurrenceSema;
  if (sameSemaphore && needsPostLoopAcquire)
    return threadedTokenHold(regain, Hold::Blocker::RESULT_CONSUMED);

  // A nested boundary may normalize a different incoming semaphore through
  // the existing post-loop acquire/bridge composition.  The matcher keeps
  // that realization separate from the direct same-semaphore rule above.
  std::optional<Hold> pointOfUse = matchPointOfUse(
      g, F, forOp, c, regain, entryAcquire, *recurrenceSema, regionTail,
      needsPostLoopAcquire);
  return pointOfUse ? std::move(*pointOfUse) : threadedTokenHold(regain);
}

static LogicalResult computeHoldRules(GroupDag &g, Node *head) {
  LogicalResult result = success();
  forEachRegionPostOrder(head, [&](Node *n) {
    if (failed(result))
      return;
    if (n->kind == Node::For) {
      for (Crossing &c : n->crossings) {
        Node *postLoopAcquire = c.postLoopAcquire;
        Hold next = buildUniformHold(g, n, c);
        if (postLoopAcquire &&
            (!next.isPointOfUse() || !next.needsPostLoopAcquire)) {
          result = semaError(n->op) << "post-loop acquire invalidated its point-of-use hold";
          return;
        }
        c.hold = std::move(next);
      }
    }
  });
  return result;
}
static void unlinkFromChain(Node *n) {
  if (n->prev)
    n->prev->next = n->next;
  else if (n->parent)
    for (Node *&child : n->parent->children)
      if (child == n)
        child = n->next;
  if (n->next)
    n->next->prev = n->prev;
  n->prev = n->next = nullptr;
}

static bool materializeHoldHandoffs(GroupDag &g, Node *head) {
  bool changed = false;
  forEachRegionPostOrder(head, [&](Node *n) {
    if (n->kind != Node::For)
      return;
    Node *anchor = n;
    for (Crossing &c : n->crossings) {
      if (!c.hold.isPointOfUse() || !c.hold.needsPostLoopAcquire)
        continue;
      Node *recurrenceAcquire =
          c.hold.regionTail ? c.hold.entryAcquire : c.hold.regain;
      if (!c.postLoopAcquire) {
        Node *postLoopAcquire = newProtocolNode(
            g, Node::Acquire, n->parent, c.tokenOwner,
            recurrenceAcquire->sema, recurrenceAcquire->count);
        postLoopAcquire->postLoopAcquire = true;
        spliceAfter(postLoopAcquire, anchor);
        anchor = postLoopAcquire;
        c.postLoopAcquire = postLoopAcquire;
        changed = true;
      } else {
        anchor = c.postLoopAcquire;
      }
      if (c.hold.keepsEntryAcquire && !c.bridgeAcquire)
        c.bridgeAcquire = c.hold.bridgeAcquire;
      if (c.hold.keepsEntryAcquire && !c.bridgeRelease) {
        Node *bridge = newProtocolNode(g, Node::Release,
                                      c.bridgeAcquire->parent,
                                      c.tokenOwner,
                                      recurrenceAcquire->sema,
                                      recurrenceAcquire->count);
        bridge->payloads.push_back(AsyncOp::NONE);
        spliceAfter(bridge, c.bridgeAcquire);
        getSema(g, bridge).expectedReleases += bridge->count;
        c.bridgeRelease = bridge;
        changed = true;
      }
    }
  });
  return changed;
}
static void refreshCrossingFinals(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    for (Crossing &c : n->crossings) {
      c.finals.clear();
      for (Node *child : n->children) {
        Node *final = chainFinalForComp(g, child);
        c.finals.push_back(final);
        if (final)
          c.tokenOwner = finalOwner(g, final);
      }
    }
  });
}

static void applyHoldRulePlacement(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    if (n->kind != Node::For)
      return;
    for (Crossing &c : n->crossings) {
      if (!c.hold.isPointOfUse())
        continue;
      if (c.hold.regionTail) {
        Node *tail = c.finals[0];
        Node *pointAcquire = c.hold.entryAcquire;
        Sema &s = getSema(g, pointAcquire);
        pointAcquire->owner = c.tokenOwner;
        unlinkFromChain(pointAcquire);
        pointAcquire->scheduleAnchor = c.hold.firstToucher;
        spliceBefore(pointAcquire, c.hold.firstToucher);
        Node *closing = newProtocolNode(g, Node::Release, tail->parent,
                                        c.tokenOwner, pointAcquire->sema, 1);
        closing->payloads.push_back(AsyncOp::NONE);
        closing->sat = pointAcquire;
        closing->scheduleAnchor = tail;
        spliceAfter(closing, tail);
        s.expectedReleases += closing->count;
        c.hold.closingRelease = closing;
        continue;
      }
      Node *regain = c.finals[0];
      unlinkFromChain(regain);
      regain->scheduleAnchor = c.hold.firstToucher;
      spliceBefore(regain, c.hold.firstToucher);
      if (c.hold.keepsEntryAcquire) {
        Sema &recurrenceSema = getSema(g, regain);
        recurrenceSema.isEntry = true;
        recurrenceSema.entryTokenOwner = c.tokenOwner;
      } else {
        unlinkFromChain(c.hold.entryAcquire);
      }
    }
  });
}
static void computeRequiredParts(Node *head) {
  forEachNode(head, [&](Node *n) {
    if (n->isRegion()) {
      SmallVector<int, 4> parts;
      for (Node *child : n->children)
        collectParts(child, parts);
      llvm::sort(parts);
      n->requiredParts.assign(parts.begin(), parts.end());
    }
  });
}
static bool canDoubleBufferAcc(nvidia_gpu::MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  int64_t blockM = tmemDesc.getShape()[0];
  int64_t blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + blockM * blockN * 2 > numTMEMRows * numTMEMColumns)
    return false;
  if (isa<nvidia_gpu::TCGen5MMAScaledOp>(mmaOp.getOperation()) && blockN == 256)
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
static bool isMultiBufferedGroup(GroupDag &g, int numTmemBlocks) {
  bool isMultiBuffered = true;
  for (const Member &m : g.pieceTable.members) {
    for (Operation *user : m.allocOp->getResult(0).getUsers()) {
      if (auto mmaOp = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          scf::ForOp wsLoop = outerWSLoop(loop);
          bool accIsMultiBuffered = !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              nvidia_gpu::isAccMultibufferingPossible(mmaOp, loop) &&
              !getDisallowAccMultiBuffer(wsLoop) && canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiBuffered = isMultiBuffered && accIsMultiBuffered;
        }
      }
    }
  }
  return isMultiBuffered;
}
static LogicalResult getPlannedBufferCopy(GroupDag &g, std::optional<int> &plannedCopy) {
  bool sawMissing = false;
  for (const Member &m : g.pieceTable.members) {
    auto copyAttr = m.allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
    if (!copyAttr) {
      sawMissing = true;
      continue;
    }
    int copy = copyAttr.getInt();
    if (copy < 1)
      return semaError(m.allocOp) << "planned buffer.copy must be positive";
    if (plannedCopy && *plannedCopy != copy)
      return semaError(m.allocOp) << "allocs in one planned reuse group have inconsistent buffer.copy values";
    plannedCopy = copy;
  }
  if (plannedCopy && sawMissing)
    return semaError(g.pieceTable.members.front().allocOp)
           << "planned reuse group mixes buffer.copy and non-buffer.copy allocs";
  return success();
}

LogicalResult computeBackingPlan(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages,
                                 int &numTmemBlocks) {
  g.numCopies = 1;
  bool untouched = g.semas.empty();
  std::optional<int> plannedBufferCopy;
  if (failed(getPlannedBufferCopy(g, plannedBufferCopy)))
    return failure();
  if (!untouched && plannedBufferCopy) {
    g.numCopies = *plannedBufferCopy;
  } else if (g.isTmem() && !untouched && !useMetaPartitioner &&
             isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  g.numSemaphoreCopies = g.numCopies;
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad))
      hasProducerLoad = true;
  });
  if (!untouched && g.isLocal() && !plannedBufferCopy && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
  if (g.isTmem() && !untouched)
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return success();
}

static LogicalResult verifyTokenLocality(GroupDag &g, Node *head) {
  SmallVector<Owner, 2> tokens;
  auto hasToken = [&](const Owner &owner) {
    return owner && llvm::any_of(tokens, [&](const Owner &tokenOwner) {
             return sameOwner(tokenOwner, owner);
           });
  };
  auto recordToken = [&](const Owner &owner) {
    auto it = llvm::find_if(tokens, [&](const Owner &tokenOwner) {
      return sameOwner(tokenOwner, owner);
    });
    if (it != tokens.end())
      tokens.erase(it);
    tokens.push_back(owner);
  };
  auto canReuseToken = [&](Node *n, const Owner &owner) {
    return nodeReusesToken(n, owner) && hasToken(owner);
  };
  auto seedFromPieces = [&](Node *n) {
    if (n->pieceInfo.empty())
      return;
    tokens.clear();
    if (auto owner = uniformPieceOwner(n)) {
      tokens.push_back(*owner);
    }
  };
  if (head->kind == Node::Enter)
    seedFromPieces(head);
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      const Sema &sm = getSema(g, n);
      Owner owner =
          sm.isEntry && !n->owner ? sm.entryTokenOwner : n->owner;
      recordToken(owner);
      break;
    }
    case Node::Release: {
      if (n->reuseTokenOwner &&
          (!n->owner || *n->reuseTokenOwner != ownerKey(n->owner) ||
           !hasToken(n->owner)))
        return semaError(n->sat && n->sat->op ? n->sat->op : g.root->op)
               << "token-reuse release names no token for its owner";
      const Owner *token = tokens.empty() ? nullptr : &tokens.back();
      if (n->owner && token && *token && !sameOwner(*token, n->owner) &&
          !canReuseToken(n, n->owner))
        return semaError(n->sat && n->sat->op ? n->sat->op : g.root->op)
               << "token-locality violation: release owned by partition "
               << n->owner->first
               << " consumes a token owned by partition " << (*token)->first;
      break;
    }
    case Node::Access: {
      if (!n->owner)
        break; // root nodes: outside partition discipline
      if (n->reuseTokenOwner &&
          (*n->reuseTokenOwner != ownerKey(n->owner) ||
           !hasToken(n->owner)))
        return semaError(n->op)
               << "token-reuse access names no token for its owner";
      const Owner *token = tokens.empty() ? nullptr : &tokens.back();
      if (nodeTouchesGroup(g, n) && token && *token &&
          !sameOwner(*token, n->owner) && !canReuseToken(n, n->owner))
        return semaError(n->op)
               << "token-locality violation: access owned by partition "
               << n->owner->first
               << " uses a token owned by partition " << (*token)->first;
      break;
    }
    case Node::For:
    case Node::If: {
      for (Node *child : n->children)
        if (failed(verifyTokenLocality(g, child)))
          return failure();
      seedFromPieces(n); // EXIT regain: select the boundary owner's token
      break;
    }
    default:
      break;
    }
  }
  if (head->parent && head->parent->kind == Node::For) {
    auto tokenConsumed = [&]() {
      for (Node *n = head; n; n = n->next) {
        if (n->isProtocol())
          return n->kind == Node::Release;
        if ((n->kind == Node::Access && nodeTouchesGroup(g, n)) ||
            (n->isRegion() && crossesComp(n)))
          return true;
      }
      return false;
    };
    bool checked = false;
    for (const Crossing &c : head->parent->crossings)
      if (c.hold.threadsToken() && tokenConsumed())
        checked = true;
    auto nodeCarriesToken = [&](Node *n) {
      if (n->isProtocol())
        return checked;
      if (n->kind == Node::Access)
        return checked && nodeTouchesGroup(g, n);
      return false;
    };
    std::function<Owner(Node *)> firstTokenOwnerOf = [&](Node *h) -> Owner {
      for (Node *n = h; n; n = n->next) {
        if ((n->kind == Node::Access || n->kind == Node::Acquire) &&
            n->owner && nodeCarriesToken(n))
          return n->owner;
        if (n->isRegion() && !n->children.empty())
          if (Owner o = firstTokenOwnerOf(n->children[0]))
            return o;
      }
      return std::nullopt;
    };
    Owner initialTokenOwner = firstTokenOwnerOf(head);
    Owner finalTokenOwner;
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Acquire && n->owner && nodeCarriesToken(n))
        finalTokenOwner = n->owner;
    if (initialTokenOwner && finalTokenOwner &&
        !sameOwner(initialTokenOwner, finalTokenOwner))
      return semaError(g.root->op ? g.root->op : head->parent->op)
             << "traversal-boundary token-locality violation: loop body's "
                "final token owner "
             << finalTokenOwner->first << " differs from its initial token owner "
             << initialTokenOwner->first;
  }
  return success();
}

static LogicalResult verifyPointOfUseTransparency(GroupDag &g, Node *F, const Crossing &c) {
  const Hold &h = c.hold;
  auto errorAt = [&](Node *node) {
    return semaError(node && node->op ? node->op : F->op);
  };
  Node *pointAcquire = h.regionTail ? h.entryAcquire : h.regain;
  Node *recurrenceAcquire = pointAcquire;
  if (pointAcquire->next != h.firstToucher)
    return errorAt(F) << "point-of-use acquire is not adjacent to its first buffer use";
  for (Node *m = F->children[0]; m && m != pointAcquire; m = m->next) {
    if (isTokenEvent(g, m))
      return errorAt(m) << "token event before point-of-use acquire";
  }
  auto verifyTransparentRegion = [&](Node *region) -> LogicalResult {
    if (!isHoldTransparentRegion(g, region, c.tokenOwner))
      return errorAt(region) << "non-transparent region reached point-of-use hold";
    return success();
  };
  bool sawFirstToucher = false;
  for (Node *node : h.nodes) {
    if (node == h.firstToucher)
      sawFirstToucher = true;
    if (isAccessForComp(g, node))
      continue;
    if (isRegionCrossingForComp(node)) {
      if (failed(verifyTransparentRegion(node)))
        return failure();
      continue;
    }
    return errorAt(node) << "invalid node recorded in point-of-use hold";
  }
  if (!sawFirstToucher)
    return errorAt(h.firstToucher)
           << "point-of-use hold first toucher is not a recorded hold node";
  if (h.regionTail)
    if (failed(verifyTransparentRegion(h.regain)))
      return failure();
  if (h.regionTail) {
    Node *closing = h.regain->next;
    if (!closing || closing->kind != Node::Release ||
        closing->sema != h.entryAcquire->sema || closing != h.closingRelease)
      return errorAt(F) << "regionTail point-of-use lacks closing release after region result";
  } else if (!h.closingRelease || std::max(1u, h.closingRelease->count) != 1) {
    Operation *op = h.closingRelease && h.closingRelease->op ? h.closingRelease->op : F->op;
    return semaError(op) << "point-of-use hold requires exactly one closing release";
  }
  if (h.needsPostLoopAcquire) {
    Node *postLoopAcquire = c.postLoopAcquire;
    if (!postLoopAcquire || postLoopAcquire->kind != Node::Acquire ||
        !postLoopAcquire->postLoopAcquire || postLoopAcquire->parent != F->parent ||
        postLoopAcquire->sema != recurrenceAcquire->sema ||
        postLoopAcquire->count != recurrenceAcquire->count ||
        !sameOwner(postLoopAcquire->owner, c.tokenOwner))
      return errorAt(F) << "malformed post-loop acquire";
    bool reachedPostLoopAcquire = false;
    for (Node *m = F->next; m; m = m->next) {
      if (m == postLoopAcquire) {
        reachedPostLoopAcquire = true;
        break;
      }
      if (isTokenEvent(g, m))
        return errorAt(m) << "group use precedes its post-loop acquire";
    }
    if (!reachedPostLoopAcquire)
      return errorAt(F) << "post-loop acquire is not after its loop";
  } else if (c.postLoopAcquire) {
    return errorAt(F) << "unexpected post-loop acquire";
  }
  if (h.keepsEntryAcquire) {
    const Sema &recurrenceSema = getSema(g, recurrenceAcquire);
    Node *bridgeAcquire = c.bridgeAcquire;
    Node *bridgeRelease = c.bridgeRelease;
    if (!recurrenceSema.isEntry || !bridgeAcquire ||
        h.bridgeAcquire != bridgeAcquire || !bridgeRelease ||
        bridgeAcquire->kind != Node::Acquire || bridgeAcquire->sema != h.entryAcquire->sema ||
        bridgeAcquire->parent != F->parent ||
        !sameOwner(bridgeAcquire->owner, c.tokenOwner) ||
        bridgeRelease->kind != Node::Release || bridgeRelease->parent != bridgeAcquire->parent ||
        bridgeRelease->sema != recurrenceAcquire->sema || bridgeRelease->count != recurrenceAcquire->count ||
        !sameOwner(bridgeRelease->owner, c.tokenOwner) || bridgeRelease->sat)
      return errorAt(F) << "malformed outer-to-local semaphore bridge";
    if (!precedesInChain(F, bridgeAcquire) || !precedesInChain(bridgeAcquire, bridgeRelease))
      return errorAt(F) << "semaphore bridge is not after its loop";
  } else if (h.bridgeAcquire || c.bridgeAcquire || c.bridgeRelease) {
    return errorAt(F) << "unexpected outer-to-local semaphore bridge";
  }
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (!g.semas.empty() && !g.root->children.empty())
    if (failed(verifyTokenLocality(g, g.root->children[0])))
      return failure();
  auto verifyHold = [&](Node *n) -> LogicalResult {
    if (!n->isRegion())
      return success();
    for (const Crossing &c : n->crossings) {
      if (c.hold.threadsToken())
        continue;
      if (c.hold.isPointOfUse()) {
        if (c.finals.empty() || !c.finals[0] || !c.hold.regain ||
            c.hold.regain != c.finals[0] || !c.hold.firstToucher || !c.hold.entryAcquire)
          return semaError(n->op) << "malformed point-of-use hold crossing";
        if (!c.hold.regionTail && c.hold.regain->kind != Node::Acquire)
          return semaError(n->op) << "point-of-use hold without acquire regain";
        if (c.hold.regionTail && !c.hold.regain->isRegion())
          return semaError(n->op) << "regionTail point-of-use without region regain";
        if (failed(verifyPointOfUseTransparency(g, n, c)))
          return failure();
        continue;
      }
      if (c.finals.empty() || !c.finals[0] ||
          c.finals[0]->kind != Node::For || c.hold.firstToucher || c.hold.entryAcquire)
        return semaError(n->op) << "malformed child-owned hold crossing";
      const Crossing *child = findCrossing(c.finals[0]);
      if (!child || child->hold.threadsToken())
        return semaError(n->op) << "child-owned hold without native child";
    }
    return success();
  };
  if (!g.root->children.empty())
    if (failed(forEachNodeChecked(g.root->children[0], verifyHold)))
      return failure();
  SmallVector<unsigned> releaseCount(g.semas.size(), 0);
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifySemaNode = [&](Node *n) -> LogicalResult {
      if (n->kind == Node::Release) {
        releaseCount[n->sema] += std::max(1u, n->count);
        if (n->payloads.empty())
          return semaError(g.root->op) << "release without payload record";
        if (n->sat) {
          if (n->sat->parent != n->parent)
            return semaError(g.root->op) << "release and its acquire are in different chains";
          bool forward = false;
          for (Node *m = n->next; m; m = m->next)
            if (m == n->sat)
              forward = true;
          if (!forward && !getSema(g, n).isEntry)
            return semaError(g.root->op) << "release does not precede its acquire";
        }
      }
      if (n->kind == Node::Acquire && n->count != getSema(g, n).count)
        return semaError(g.root->op) << "semaphore " << getSema(g, n).name
               << " acquired with non-uniform pending count";
      if (n->kind == Node::Acquire && n->owner.has_value()) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema] && *acqClass[n->sema] != k)
          return semaError(g.root->op) << "semaphore " << getSema(g, n).name
                 << " acquired by two distinct partitions (M3 violation)";
        acqClass[n->sema] = k;
      }
      return success();
  };
  if (!g.root->children.empty())
    if (failed(forEachNodeChecked(g.root->children[0], verifySemaNode)))
      return failure();
  for (auto [sid, s] : llvm::enumerate(g.semas)) {
    if (releaseCount[sid] != s.expectedReleases)
      return semaError(g.root->op) << "semaphore "
             << s.name << " has " << releaseCount[sid] << " releases, expected " << s.expectedReleases;
  }
  return success();
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;

struct ScheduleEdge {
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
};

struct SlotSchedule {
  int64_t advancesPerIteration = 0;
  DenseMap<Node *, int64_t> ordinalByAccess;
  bool complete = true;
};
static Effect accessEffect(const Node *n) {
  Effect effect = Effect::R;
  for (const Touch &touch : n->touches)
    effect = joinEffect(effect, touch.effect);
  return effect;
}

struct PhysicalStage {
  int64_t requiredOrdinal;
  int64_t cursorOrdinal;
};

static std::optional<PhysicalStage> recordPhysicalStage(
    SlotSchedule &result,
    DenseMap<GroupDag *, int64_t> &lastProducedOrdinal,
    int64_t &cursorOrdinal, GroupDag *group, Node *access,
    unsigned advances) {
  int64_t requiredOrdinal;
  if (accessEffect(access) == Effect::W) {
    if (advances > 1)
      result.complete = false;
    cursorOrdinal += advances;
    result.advancesPerIteration += advances;
    if (cursorOrdinal < 0) {
      result.complete = false;
      return std::nullopt;
    }
    requiredOrdinal = lastProducedOrdinal[group] = cursorOrdinal;
  } else {
    auto it = lastProducedOrdinal.find(group);
    if (it == lastProducedOrdinal.end()) {
      result.complete = false;
      return std::nullopt;
    }
    requiredOrdinal = it->second;
  }
  result.ordinalByAccess[access] = requiredOrdinal;
  return PhysicalStage{requiredOrdinal, cursorOrdinal};
}

// Derive physical ring displacements before EMIT so protocol and views agree.
static LogicalResult assignCircularStageOffsets(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 4>> sets;
  for (GroupDag &g : groups)
    if (g.isCircular() && !g.semas.empty())
      sets[g.bufferId].push_back(&g);
  struct Event {
    GroupDag *group;
    Node *access;
  };
  for (auto &[id, set] : sets) {
    (void)id;
    auto type = set.front()->pieceTable.members.front().type;
    int64_t numCopies = set.front()->numCopies;
    DenseSet<int64_t> starts;
    DenseMap<Operation *, SmallVector<Event, 1>> eventsByOp;
    for (GroupDag *g : set) {
      const Member &member = g->pieceTable.members.front();
      if (g->pieceTable.members.size() != 1)
        return semaError(g->root->op)
               << "malformed circular local logical group";
      if (member.type != type)
        return semaError(member.allocOp) << "circular local group has mismatched member types";
      if (g->numCopies != numCopies)
        return semaError(member.allocOp) << "circular local group has mismatched buffer.copy";
      if (member.circularStart < 0 || member.circularStart >= numCopies)
        return semaError(member.allocOp) << "circular buffer.start is outside buffer.copy";
      if (!starts.insert(member.circularStart).second)
        return semaError(member.allocOp) << "duplicate circular buffer.start in one group";
      forEachNode(*g, [&](Node *n) {
        if (n->kind == Node::Access)
          eventsByOp[n->op].push_back(Event{g, n});
      });
    }
    SmallVector<Event> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end())
        ordered.append(it->second.begin(), it->second.end());
    });
    SlotSchedule slots;
    int64_t cursorOrdinal = -1;
    DenseMap<GroupDag *, int64_t> lastProducedOrdinal;
    for (const Event &event : ordered) {
      const Member &member = event.group->pieceTable.members.front();
      auto stage = recordPhysicalStage(
          slots, lastProducedOrdinal, cursorOrdinal, event.group,
          event.access, accessEffect(event.access) == Effect::W);
      if (accessEffect(event.access) == Effect::W) {
        assert(stage);
        if (member.circularStart != stage->requiredOrdinal % numCopies)
          return semaError(member.allocOp)
                 << "circular producer order expects buffer.start "
                 << stage->requiredOrdinal % numCopies << ", got "
                 << member.circularStart;
      } else if (!stage) {
        return semaError(member.allocOp)
               << "circular consumer appears before producer";
      }
      event.access->bufferStageOffset =
          stage->requiredOrdinal - stage->cursorOrdinal;
    }
    for (GroupDag *g : set)
      forEachNode(*g, [&](Node *n) {
        Node *access;
        if (n->kind == Node::Acquire) {
          access = n;
          while ((access = access->next))
            if (access->kind == Node::Access && access->bufferStageOffset)
              break;
        } else if (n->kind == Node::Release) {
          access = n;
          while ((access = access->prev))
            if (access->kind == Node::Access && access->bufferStageOffset)
              break;
        } else {
          return;
        }
        if (access)
          n->stageOffset = access->bufferStageOffset;
      });
  }
  return success();
}

static Operation *realScheduleAnchor(Node *anchor, bool producer) {
  for (Node *n = anchor; n; n = producer ? n->prev : n->next) {
    if (n->kind == Node::Access)
      return producer && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion() && n->op)
      return n->op;
  }
  return nullptr;
}

struct LoopAnchorPair {
  scf::ForOp loop;
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
};
static std::optional<LoopAnchorPair>
findCommonScheduledLoop(Operation *producer, Operation *consumer) {
  for (Operation *parent = producer->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName))
      continue;
    Operation *producerInLoop =
        loop.getBody()->findAncestorOpInBlock(*producer);
    Operation *consumerInLoop =
        loop.getBody()->findAncestorOpInBlock(*consumer);
    if (producerInLoop && consumerInLoop)
      return LoopAnchorPair{loop, producerInLoop, consumerInLoop};
  }
  return std::nullopt;
}

static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet, scf::ForOp loop) {
  struct Event {
    GroupDag *group = nullptr;
    Node *node = nullptr;
    Operation *op = nullptr;
    unsigned groupOrder = 0;
  };
  DenseMap<Operation *, unsigned> operationOrder;
  for (auto [index, op] :
       llvm::enumerate(loop.getBody()->without_terminator()))
    operationOrder[&op] = index;
  SlotSchedule result;
  SmallVector<Event, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (auto [groupOrder, group] : llvm::enumerate(physicalSet)) {
    forEachNode(*group, [&](Node *node) {
      if (node->kind != Node::Acquire || !node->scheduleAnchor ||
          node->scheduleAnchor->kind != Node::Access || accessEffect(node->scheduleAnchor) != Effect::W)
        return;
      Operation *direct = loop.getBody()->findAncestorOpInBlock(*node->scheduleAnchor->op);
      if (direct == node->scheduleAnchor->op)
        ++advancesByAccess[node->scheduleAnchor];
    });
    forEachNode(*group, [&](Node *node) {
      if (node->kind != Node::Access || !node->op)
        return;
      Operation *direct = loop.getBody()->findAncestorOpInBlock(*node->op);
      if (!direct)
        return;
      if (direct != node->op) {
        result.complete = false;
        return;
      }
      events.push_back(Event{group, node, direct, static_cast<unsigned>(groupOrder)});
    });
  }
  llvm::sort(events, [&](const Event &lhs, const Event &rhs) {
    unsigned lhsOrder = operationOrder.lookup(lhs.op);
    unsigned rhsOrder = operationOrder.lookup(rhs.op);
    return lhsOrder != rhsOrder ? lhsOrder < rhsOrder : lhs.groupOrder < rhs.groupOrder;
  });
  int64_t cursorOrdinal = -1;
  DenseMap<GroupDag *, int64_t> lastProducedOrdinal;
  for (const Event &event : events)
    recordPhysicalStage(result, lastProducedOrdinal, cursorOrdinal,
                        event.group, event.node,
                        advancesByAccess.lookup(event.node));
  return result;
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}
static std::optional<int64_t>
computeLoopCarriedDistance(const SlotSchedule &slots,
                           int64_t numSemaphoreCopies,
                           Node *producer, Node *consumer) {
  auto producerIt = slots.ordinalByAccess.find(producer);
  auto consumerIt = slots.ordinalByAccess.find(consumer);
  if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
      consumerIt == slots.ordinalByAccess.end() ||
      slots.advancesPerIteration <= 0)
    return std::nullopt;
  int64_t orbit = numSemaphoreCopies /
                  std::gcd(numSemaphoreCopies, slots.advancesPerIteration);
  for (int64_t distance = 1; distance <= orbit; ++distance)
    if (positiveMod(consumerIt->second +
                        distance * slots.advancesPerIteration,
                    numSemaphoreCopies) ==
        positiveMod(producerIt->second, numSemaphoreCopies))
      return distance;
  return std::nullopt;
}
static bool isExactAliasMultibufferedGroup(const GroupDag &group) {
  if (group.isCircular() ||
      group.pieceTable.members.size() < 2 || group.numSemaphoreCopies <= 1)
    return false;
  bool hasPlannedCopy =
      llvm::all_of(group.pieceTable.members, [](const Member &member) {
        return member.allocOp->hasAttr(kBufferCopyAttrName);
      });
  // A planned copy depth or a separately staged semaphore provides the
  // stable stage domain required for authored offsets.
  bool authoredMulticopy = hasPlannedCopy && group.numCopies > 1;
  bool extraSemaphoreStages = group.numSemaphoreCopies > group.numCopies;
  if (!authoredMulticopy && !extraSemaphoreStages)
    return false;
  const Member &first = group.pieceTable.members.front();
  return llvm::all_of(group.pieceTable.members, [&](const Member &member) {
    return member.offset == first.offset && member.extent == first.extent && member.type == first.type;
  });
}

static LogicalResult assignAliasedHandoffStageOffsets(GroupDag &group) {
  if (!isExactAliasMultibufferedGroup(group))
    return success();
  DenseMap<Operation *, SlotSchedule> slotsByLoop;
  bool hasShiftedRelease = false;
  LogicalResult result = success();
  forEachNode(group, [&](Node *release) {
    if (failed(result) || release->kind != Node::Release || !release->sat)
      return;
    Node *producer = release->scheduleAnchor;
    Node *consumer = release->sat->scheduleAnchor;
    if (!producer || !consumer || producer->kind != Node::Access ||
        consumer->kind != Node::Access || producer->parent != consumer->parent ||
        !producer->parent || producer->parent->kind != Node::For) {
      result = semaError(producer && producer->op ? producer->op : group.root->op)
               << "multibuffered exact-alias handoff requires direct accesses "
                  "in one loop body";
      return;
    }
    auto loop = dyn_cast<scf::ForOp>(producer->parent->op);
    if (!loop ||
        loop.getBody()->findAncestorOpInBlock(*producer->op) != producer->op ||
        loop.getBody()->findAncestorOpInBlock(*consumer->op) != consumer->op) {
      result = semaError(producer->op)
               << "multibuffered exact-alias handoff is not directly "
                  "represented in its ownership loop";
      return;
    }
    auto [it, inserted] = slotsByLoop.try_emplace(loop.getOperation());
    if (inserted)
      it->second = computeSlotSchedule(ArrayRef<GroupDag *>{&group}, loop);
    const SlotSchedule &slots = it->second;
    auto producerIt = slots.ordinalByAccess.find(producer);
    auto consumerIt = slots.ordinalByAccess.find(consumer);
    if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
        consumerIt == slots.ordinalByAccess.end() ||
        slots.advancesPerIteration <= 0) {
      result = semaError(producer->op)
               << "cannot derive multibuffered exact-alias handoff slots";
      return;
    }
    int64_t numSemaphoreCopies = group.numSemaphoreCopies;
    int64_t offset = 0;
    if (precedesInChain(release, release->sat)) {
      offset = positiveMod(consumerIt->second - producerIt->second,
                           numSemaphoreCopies);
    } else if (!computeLoopCarriedDistance(
                   slots, numSemaphoreCopies, producer, consumer)) {
      int64_t nextConsumer =
          consumerIt->second + slots.advancesPerIteration;
      offset =
          positiveMod(nextConsumer - producerIt->second, numSemaphoreCopies);
    }
    release->stageOffset = offset;
    hasShiftedRelease |= offset != 0;
  });
  if (failed(result))
    return result;
  if (!hasShiftedRelease) {
    forEachNode(group, [&](Node *node) {
      if (node->kind == Node::Release)
        node->stageOffset.reset();
    });
    return success();
  }
  forEachNode(group, [&](Node *node) {
    if (node->kind == Node::Acquire)
      node->stageOffset = 0;
  });
  return success();
}

static LogicalResult
assignBufferStageOffsets(MutableArrayRef<GroupDag> groups) {
  if (failed(assignCircularStageOffsets(groups)))
    return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(group)))
      return failure();
  return success();
}

static LogicalResult addSyncScheduleEdges(MutableArrayRef<GroupDag> groups,
    llvm::MapVector<Operation *, SmallVector<ScheduleEdge, 4>> &edgesByLoop) {
  SmallVector<SmallVector<GroupDag *, 2>, 8> physicalSets;
  DenseMap<GroupDag *, unsigned> setByGroup;
  llvm::MapVector<int64_t, unsigned> circularSetByBuffer;
  for (GroupDag &group : groups) {
    unsigned setIndex;
    if (group.isCircular()) {
      auto [it, inserted] = circularSetByBuffer.insert(
          {group.bufferId, static_cast<unsigned>(physicalSets.size())});
      if (inserted)
        physicalSets.emplace_back();
      setIndex = it->second;
    } else {
      setIndex = physicalSets.size();
      physicalSets.emplace_back();
    }
    physicalSets[setIndex].push_back(&group);
    setByGroup[&group] = setIndex;
  }
  std::map<std::pair<unsigned, Operation *>, SlotSchedule> slotCache;
  for (GroupDag &group : groups) {
    LogicalResult result = success();
    forEachNode(group, [&](Node *release) {
      if (failed(result) || release->kind != Node::Release || !release->sat)
        return;
      Node *acquire = release->sat;
      Operation *producer =
          realScheduleAnchor(release->scheduleAnchor, /*producer=*/true);
      Operation *consumer =
          realScheduleAnchor(acquire->scheduleAnchor, /*producer=*/false);
      if (!producer || !consumer)
        return;
      std::optional<LoopAnchorPair> anchors =
          findCommonScheduledLoop(producer, consumer);
      if (!anchors || anchors->producer == anchors->consumer)
        return;
      gpu::StageCluster producerSchedule =
          gpu::getStageCluster(anchors->producer);
      gpu::StageCluster consumerSchedule =
          gpu::getStageCluster(anchors->consumer);
      if (!producerSchedule || !consumerSchedule)
        return;
      int64_t distance = 0;
      if (!precedesInChain(release, acquire)) {
        unsigned setIndex = setByGroup.lookup(&group);
        auto key = std::make_pair(setIndex, anchors->loop.getOperation());
        auto it = slotCache.find(key);
        if (it == slotCache.end())
          it = slotCache
                   .emplace(key, computeSlotSchedule(physicalSets[setIndex], anchors->loop))
                   .first;
        std::optional<int64_t> loopCarriedDistance = computeLoopCarriedDistance(
            it->second, group.numSemaphoreCopies, release->scheduleAnchor,
            acquire->scheduleAnchor);
        if (!loopCarriedDistance) {
          InFlightDiagnostic diag = semaError(anchors->producer)
              << "cannot determine loop-carried dependency distance for a "
                 "physical buffer slot";
          diag.attachNote(anchors->consumer->getLoc())
              << "next token ownership starts here";
          result = failure();
          return;
        }
        distance = *loopCarriedDistance;
      }
      if (producerSchedule->first > consumerSchedule->first + distance) {
        InFlightDiagnostic diag = semaError(anchors->producer) << "fixed loop.stage assignment cannot "
                                     "satisfy semaphore handoff";
        diag << " (producer loop.stage " << producerSchedule->first
             << ", consumer loop.stage " << consumerSchedule->first
             << ", loop-carried dependency distance " << distance << ")";
        diag.attachNote(anchors->consumer->getLoc())
            << "consumer would execute before the released slot can be "
               "reacquired";
        result = failure();
        return;
      }
      if (producerSchedule->first == consumerSchedule->first + distance)
        edgesByLoop[anchors->loop.getOperation()].push_back(
            ScheduleEdge{anchors->producer, anchors->consumer});
    });
    if (failed(result))
      return failure();
  }
  return success();
}
static void addSSAClusterConstraints(scf::ForOp loop,
                                     SmallVectorImpl<ScheduleEdge> &edges) {
  for (Operation &consumer : loop.getBody()->without_terminator()) {
    gpu::StageCluster consumerSchedule = gpu::getStageCluster(&consumer);
    if (!consumerSchedule)
      continue;
    for (Value operand : getNestedOperands(&consumer)) {
      auto [producer, distance] = triton::getDefiningOpAndDistance(loop, operand);
      if (!producer)
        continue;
      producer = loop.getBody()->findAncestorOpInBlock(*producer);
      if (!producer || producer == &consumer)
        continue;
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producer);
      if (!producerSchedule)
        continue;
      if (producerSchedule->first == consumerSchedule->first + distance)
        edges.push_back(ScheduleEdge{producer, &consumer});
    }
  }
}

static LogicalResult legalizeLoopSchedule(scf::ForOp loop, ArrayRef<ScheduleEdge> edges) {
  SmallVector<Operation *, 32> scheduledOps;
  DenseMap<Operation *, int64_t> cluster;
  for (Operation &op : loop.getBody()->without_terminator()) {
    gpu::StageCluster schedule = gpu::getStageCluster(&op);
    if (!schedule)
      continue;
    scheduledOps.push_back(&op);
    cluster[&op] = schedule->second;
  }
  bool changed = false;
  for (unsigned iteration = 0; iteration <= scheduledOps.size(); ++iteration) {
    changed = false;
    for (const ScheduleEdge &edge : edges) {
      if (!cluster.contains(edge.producer) || !cluster.contains(edge.consumer))
        continue;
      int64_t separation =
          edge.producer->isBeforeInBlock(edge.consumer) ? 0 : 1;
      int64_t required = cluster.lookup(edge.producer) + separation;
      if (cluster.lookup(edge.consumer) >= required)
        continue;
      cluster[edge.consumer] = required;
      changed = true;
    }
    if (!changed)
      break;
    if (iteration == scheduledOps.size())
      return semaError(loop) << "cyclic loop.cluster constraints";
  }
  OpBuilder builder(loop.getContext());
  for (Operation *op : scheduledOps) {
    gpu::StageCluster oldSchedule = gpu::getStageCluster(op);
    int64_t newCluster = cluster.lookup(op);
    if (newCluster == oldSchedule->second)
      continue;
    if (newCluster > std::numeric_limits<int32_t>::max())
      return semaError(op) << "legalized loop.cluster exceeds i32 range";
    gpu::setStageCluster(builder, op, std::make_pair(oldSchedule->first, static_cast<int>(newCluster)));
  }
  return success();
}
static Operation *nextScheduleAnchor(const Node *n) {
  for (const Node *m = n; m; m = m->next)
    if ((m->kind == Node::Access || m->kind == Node::For || m->kind == Node::If) && m->op)
      return m->op;
  return nullptr;
}
static gpu::StageCluster cachedSchedule(const ScheduleCache &cache, const Owner &owner) {
  auto it = cache.find(ownerKey(owner));
  return it == cache.end() ? gpu::StageCluster{} : it->second;
}
static gpu::StageCluster scheduleAtOwnerBoundary(
    const Node *n, gpu::StageCluster schedule) {
  if (!schedule || nextScheduleAnchor(n->next))
    return schedule;
  auto forOp = dyn_cast_or_null<scf::ForOp>(n->parent ? n->parent->op : nullptr);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName))
    return schedule;
  auto [stage, cluster] = *schedule;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate || candidate->first != stage || !gpu::hasPartition(&op))
      continue;
    SetVector<int> partitions = gpu::getPartitionIds(&op);
    if (partitions.contains(n->owner->first))
      cluster = std::max(cluster, candidate->second);
  }
  return std::make_pair(stage, cluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache);
static void assignSyncScheduleRegion(Node *n, ScheduleCache &cache) {
  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    ScheduleCache body = cache;
    assignSyncScheduleChain(n->children[0], body);
    if (!gpu::hasWarpSpecializeTag(forOp))
      cache = std::move(body);
    return;
  }
  ScheduleCache thenCache = cache;
  ScheduleCache elseCache = cache;
  assignSyncScheduleChain(n->children[0], thenCache);
  if (n->children.size() > 1 && n->children[1])
    assignSyncScheduleChain(n->children[1], elseCache);
  cache = std::move(thenCache);
  for (auto &[key, stageCluster] : elseCache)
    cache.try_emplace(key, stageCluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      if (n->owner) {
        Operation *anchor = n->postLoopAcquire ? nullptr
                                                : nextScheduleAnchor(n->next);
        n->stageCluster =
            anchor ? gpu::getStageCluster(anchor)
                   : scheduleAtOwnerBoundary(
                         n, cachedSchedule(cache, n->owner));
      }
      break;
    case Node::Release:
      if (n->owner)
        n->stageCluster = cachedSchedule(cache, n->owner);
      break;
    case Node::Access:
      if (n->owner) {
        Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
        cache[ownerKey(n->owner)] = gpu::getStageCluster(completion);
      }
      break;
    case Node::For:
    case Node::If:
      assignSyncScheduleRegion(n, cache);
      break;
    case Node::Enter:
    case Node::Exit:
    case Node::Func:
      break;
    }
  }
}

LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, SmallVector<ScheduleEdge, 4>> edgesByLoop;
  if (failed(assignBufferStageOffsets(groups)))
    return failure();
  if (failed(addSyncScheduleEdges(groups, edgesByLoop)))
    return failure();
  for (auto &[loopOp, edges] : edgesByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    addSSAClusterConstraints(loop, edges);
    if (failed(legalizeLoopSchedule(loop, edges)))
      return failure();
  }
  for (GroupDag &g : groups) {
    if (g.root->children.empty())
      continue;
    ScheduleCache cache;
    assignSyncScheduleChain(g.root->children[0], cache);
  }
  return success();
}

LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages,
                           int &numTmemBlocks) {
  SmallVector<EdgeRec> edges;
  if (!g.root->children.empty()) {
    ChainState top; // function chain: games start at bottom (first-touch)
    walkChain(g, g.root->children[0], top, edges, /*underFor=*/false);
  }
  if (failed(buildEdgesAndSemas(g, edges)))
    return failure();
  if (failed(insertEntryAcquires(g)))
    return failure();
  if (!g.root->children.empty()) {
    computeCrossings(g, g.root->children[0]);
    pruneDeadIfCrossings(g, g.root->children[0], /*region=*/nullptr);
    if (failed(computeHoldRules(g, g.root->children[0])))
      return failure();
    while (materializeHoldHandoffs(g, g.root->children[0])) {
      refreshCrossingFinals(g, g.root->children[0]);
      if (failed(computeHoldRules(g, g.root->children[0])))
        return failure();
    }
    applyHoldRulePlacement(g, g.root->children[0]); // plan M2: native shape
    computeRequiredParts(g.root->children[0]);
  }
  if (failed(computeBackingPlan(g, useMetaPartitioner, lowerSemaphoreNumStages, numTmemBlocks)))
    return failure();
  if (!g.semas.empty())
    for (Operation *alloc : g.ttDescriptorFedMembers)
      return semaError(alloc) << "managed local_alloc sourced from a tt-form descriptor load — "
                "nvws-insert-allocas must convert this upstream (pipeline " "invariant violated)";
  return verifySyncDag(g);
}
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
static std::string syncNodeLabel(GroupDag &g, const Node *n) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (n->kind == Node::Acquire)
    os << "a " << getSema(g, n).name;
  else if (n->kind == Node::Release)
    os << "r " << getSema(g, n).name;
  else if (n->kind == Node::For)
    os << "scf.for";
  else if (n->kind == Node::If)
    os << "scf.if";
  return s;
}
static void printThreadInfo(llvm::raw_ostream &os, GroupDag &g, const Node *n) {
  if (!n->requiredParts.empty()) {
    os << " parts{";
    llvm::interleaveComma(n->requiredParts, os);
    os << "}";
  }
  if (!n->crossings.empty()) {
    os << " thread{";
    llvm::interleaveComma(n->crossings, os, [&](const Crossing &c) {
      os << ownerStr(n->op, c.tokenOwner);
    });
    os << "}";
  }
  if (n->kind == Node::For && !n->crossings.empty()) {
    os << " holdrule{";
    llvm::interleaveComma(n->crossings, os, [&](const Crossing &c) {
      if (c.hold.outcome == Hold::Outcome::THREADED) {
        os << "gated";
        if (c.hold.blocker == Hold::Blocker::TRAILING_USE)
          os << "(trailing-use)";
        else if (c.hold.blocker == Hold::Blocker::RESULT_CONSUMED)
          os << "(result-consumed)";
      } else if (c.hold.outcome == Hold::Outcome::POINT_OF_USE) {
        os << "pointofuse->";
        if (c.hold.firstToucher && c.hold.firstToucher->op)
          os << c.hold.firstToucher->op->getName().getStringRef();
        else
          os << "?";
      } else
        os << "passthrough-drop";
      if (c.hold.regionTail)
        os << ":regionTail";
      if (c.postLoopAcquire)
        os << ":postLoopAcquire";
      if (c.bridgeRelease)
        os << ":entryBridge";
    });
    os << "}";
  }
}
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g, const Node *exit, const Node *region,
                           unsigned chainIdx) {
  if (!region || region->crossings.empty())
    return;
  os << " yield{";
  llvm::interleaveComma(region->crossings, os, [&](const Crossing &c) {
    if (!c.hold.threadsToken()) { // native/child-owned: no slot
      os << (c.hold.isChildOwns() ? "drop" : "native");
      return;
    }
    Node *f = chainIdx < c.finals.size() ? c.finals[chainIdx] : nullptr;
    os << (f ? syncNodeLabel(g, f) : std::string("pass"));
  });
  os << "}";
}
static void printEffects(llvm::raw_ostream &os, const Node *n) {
  if (n->pieceInfo.empty())
    return;
  os << " effects{";
  llvm::interleaveComma(sortedPieceInfo(n), os, [&](const auto &item) {
    os << "P" << item.first << ":" << (item.second.effect == Effect::W ? "W" : "R");
  });
  os << "}";
}
static void dumpDagChain(GroupDag &g, const Node *head, unsigned depth,
                         const Node *region, unsigned chainIdx, DumpStage stage) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    Operation *anchor = n->parent ? n->parent->op : nullptr;
    switch (n->kind) {
    case Node::Access: {
      if (stage != DumpStage::Sync) {
        for (const Touch &t : n->touches) {
          os << treePrefix(depth) << "|- " << (t.effect == Effect::W ? "W" : "R") << "  m" << t.member
             << "  " << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner);
          if (stage == DumpStage::Access && n->completionAnchor != n->op)
            os << " complete=" << n->completionAnchor->getName().getStringRef();
          os << "\n";
        }
        break;
      }
      os << treePrefix(depth) << "|- ";
      llvm::interleaveComma(n->touches, os, [&](const Touch &t) {
        os << (t.effect == Effect::W ? "W" : "R") << " m" << t.member;
      });
      os << "  " << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner) << "\n";
      break;
    }
    case Node::Acquire:
    case Node::Release: {
      if (stage != DumpStage::Sync)
        break;
      bool acquire = n->kind == Node::Acquire;
      const Sema &s = getSema(g, n);
      os << treePrefix(depth) << "|- " << (acquire ? "a" : "r") << "  " << s.name;
      if (n->count > 1)
        os << "(" << n->count << ")";
      os << "  " << ownerStr(anchor, n->owner);
      if (acquire && s.isEntry && !n->owner)
        os << "  ; entry";
      if (!acquire) {
        os << " [";
        llvm::interleaveComma(n->payloads, os, [&](AsyncOp p) { os << asyncOpStr(p); });
        os << "]";
      }
      if (n->stageOffset)
        os << "  stage-offset=" << *n->stageOffset;
      os << "\n";
      break;
    }
    case Node::For:
    case Node::If: {
      bool loop = n->kind == Node::For;
      os << treePrefix(depth) << "|- " << (loop ? "scf.for" : "scf.if");
      if (loop && gpu::hasWarpSpecializeTag(n->op))
        os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      if (stage == DumpStage::Access)
        printEffects(os, n);
      else
        printPieceRecord(os, n, loop ? n->op : anchor);
      if (stage == DumpStage::Sync)
        printThreadInfo(os, g, n);
      os << "\n";
      if (loop) {
        dumpDagChain(g, n->children[0], depth + 1, n, 0, stage);
        break;
      }
      os << treePrefix(depth + 1) << "|- then\n";
      dumpDagChain(g, n->children[0], depth + 2, n, 0, stage);
      bool virtualElse = !cast<scf::IfOp>(n->op).elseBlock();
      if (stage != DumpStage::Access || !virtualElse) {
        os << treePrefix(depth + 1) << "|- else" << (virtualElse ? " (virtual)" : "") << "\n";
        if (n->children.size() > 1)
          dumpDagChain(g, n->children[1], depth + 2, n, 1, stage);
      }
      break;
    }
    case Node::Enter:
    case Node::Exit:
      if (stage == DumpStage::Access)
        break;
      os << treePrefix(depth) << (n->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
      printPieceRecord(os, n, anchor);
      if (n->kind == Node::Exit && stage == DumpStage::Sync)
        printYieldInfo(os, g, n, region, chainIdx);
      os << "\n";
      break;
    case Node::Func:
      break;
    }
  }
}

void dumpDagTree(GroupDag &g, DumpStage stage) {
  if (!g.root->children.empty())
    dumpDagChain(g, g.root->children[0], 1, nullptr, 0, stage);
}

void dumpGroupSyncDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "SYNC-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  dumpDagTree(g, DumpStage::Sync);
  {
    std::string line;
    llvm::raw_string_ostream ls(line);
    bool any = false;
    for (const Sema &s : g.semas) {
      if (any)
        ls << " ";
      any = true;
      ls << s.name << "{count=" << s.count;
      if (s.isEntry)
        ls << " entry inherit=" << ownerStr(nullptr, s.entryTokenOwner);
      ls << "}";
    }
    if (any)
      os << "  SEMAS: " << ls.str() << "\n";
  }
  if (g.semas.empty()) {
    os << "  BACKING: untouched (no semaphores)\n";
    return;
  }
  os << "  BACKING: numCopies=" << g.numCopies << "\n";
}
} // namespace mlir::triton::nvws_semas
