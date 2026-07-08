// SYNC analysis and scheduling. Doc section references below point into
// sema-docs/insert-semas/sync-dag.md.
#include "InsertSemas.h"
#include <limits>
#include <numeric>

namespace mlir::triton::nvws_semas {

using Payloads = SmallVector<AsyncOp, 1>;
using PieceEffects = std::map<PieceId, Effect>;

struct PieceExitFacts {
  Payloads payloads;
  SmallVector<int64_t, 2> mustOrderedBefore;
};
using ExitFacts = std::map<PieceId, PieceExitFacts>;

// Doc: sync-dag.md#the-walk-accesses-to-edges
struct ActiveUse {
  Owner owner;
  Node *node = nullptr;
  Payloads payloads;
  // Owners whose existing dependency already orders this node before them.
  SmallVector<int64_t, 2> orderedBefore;
};

struct VersionSource {
  Owner producer;    // logical producer of the current version
  Owner sourceOwner; // owner of the chain-local source node
  Node *node = nullptr;
  Payloads payloads;
};

struct PieceState {
  // Reads move their active use but not this source. New readers fan out from
  // the write, or from ENTER when the version was established outside.
  VersionSource source;
  SmallVector<ActiveUse, 2> uses;

  bool initialized() const { return source.node != nullptr; }
  ActiveUse *useFor(const Owner &owner) {
    auto it = llvm::find_if(uses, [&](const ActiveUse &use) {
      return sameOwner(use.owner, owner);
    });
    return it == uses.end() ? nullptr : &*it;
  }
  void startVersion(const Owner &producer, const Owner &sourceOwner, Node *node,
                    const Payloads &payloads) {
    source = VersionSource{producer, sourceOwner, node, payloads};
    uses.assign(1, ActiveUse{sourceOwner, node, payloads, {}});
  }
};
using ChainState = std::map<PieceId, PieceState>;

// Live tokens remain in deterministic handoff order. The last source-bearing
// token supplies a handoff when no memory edge or owner-token reuse does.
struct Tokens {
  struct Token {
    Owner owner;
    Node *node = nullptr;
    Payloads payloads;
  };
  SmallVector<Token, 2> live;

  const Token *find(const Owner &owner) const {
    if (!owner)
      return nullptr;
    auto it = llvm::find_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
    return it == live.end() ? nullptr : &*it;
  }
  const Token *last() const {
    return !live.empty() && live.back().node ? &live.back() : nullptr;
  }
  void remember(const Owner &owner) {
    if (owner && !find(owner))
      live.push_back(Token{owner, nullptr, {}});
  }
  void record(const Owner &owner, Node *node, const Payloads &payloads) {
    auto it = llvm::find_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
    if (it != live.end())
      live.erase(it);
    live.push_back(Token{owner, node, payloads});
  }
};

struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  Payloads payloads;
  SmallVector<PieceId, 2> pieces;
};
using EdgeList = SmallVector<EdgeRec>;

static void unionPayloads(Payloads &into, const Payloads &from) {
  for (AsyncOp payload : from)
    if (!llvm::is_contained(into, payload))
      into.push_back(payload);
  llvm::sort(into, [](AsyncOp a, AsyncOp b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
}

static SmallVector<int64_t, 2>
intersectOrderFacts(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t, 2> result;
  for (int64_t owner : lhs)
    if (llvm::is_contained(rhs, owner))
      result.push_back(owner);
  return result;
}

static bool knownNonEmptyLoop(Node *node) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(node->op);
  if (!forOp)
    return false;
  if (forOp->hasAttr("ttg.must-execute"))
    return true;
  std::optional<APInt> tripCount = forOp.getStaticTripCount();
  if (!tripCount)
    return false;
  return forOp.getUnsignedCmp() ? tripCount->ugt(0) : tripCount->sgt(0);
}

// Doc: sync-dag.md#the-per-access-rules-in-full
// Apply one piece's RAW/WAR rules. Token supply is handled after all pieces of
// an access have advanced.
static void applyTouch(PieceState &piece, PieceId id, const Owner &owner,
                       Effect effect, Node *node, const Payloads &payloads,
                       EdgeList &edges, bool wsAdopt) {
  if (!piece.initialized()) {
    piece.startVersion(owner, owner, node, payloads);
    return;
  }
  if (effect == Effect::W) {
    for (const ActiveUse &use : piece.uses) {
      bool adoptedRoot = wsAdopt && !use.owner;
      bool alreadyOrdered =
          llvm::is_contained(use.orderedBefore, ownerKey(owner));
      if (!sameOwner(use.owner, owner) && !adoptedRoot && !alreadyOrdered)
        edges.push_back(
            EdgeRec{use.node, node, use.owner, owner, use.payloads, {id}});
    }
    piece.startVersion(owner, owner, node, payloads);
    return;
  }
  if (ActiveUse *use = piece.useFor(owner)) {
    use->node = node;
    use->payloads = payloads;
    use->orderedBefore.clear();
    return;
  }

  assert(piece.source.node && "initialized piece without a version source");
  bool adoptedRoot = wsAdopt && !piece.source.sourceOwner;
  if (!adoptedRoot && !sameOwner(piece.source.sourceOwner, owner)) {
    edges.push_back(EdgeRec{piece.source.node,
                            node,
                            piece.source.sourceOwner,
                            owner,
                            piece.source.payloads,
                            {id}});
    // The source edge covers the source use only while it still names that
    // node; a later reread must retain its own WAR obligation.
    if (ActiveUse *source = piece.useFor(piece.source.sourceOwner))
      if (source->node == piece.source.node &&
          !llvm::is_contained(source->orderedBefore, ownerKey(owner)))
        source->orderedBefore.push_back(ownerKey(owner));
  }
  piece.uses.push_back(ActiveUse{owner, node, payloads, {}});
}

static bool canReuseTokenForPiece(ChainState &state, PieceId id,
                                  const Owner &owner, Effect effect) {
  auto it = state.find(id);
  if (it == state.end() || !it->second.initialized())
    return false;
  PieceState &piece = it->second;
  if (effect == Effect::R)
    return piece.useFor(owner);
  return llvm::all_of(piece.uses, [&](const ActiveUse &use) {
    return sameOwner(use.owner, owner) ||
           llvm::is_contained(use.orderedBefore, ownerKey(owner));
  });
}

static bool nodeTouchesPiece(GroupDag &g, Node *node, PieceId piece) {
  if (node->kind == Node::Access)
    return touchesPiece(g, node, piece);
  return node->isRegion() && node->pieceInfo.count(piece);
}
static bool pieceTouchedAfter(GroupDag &g, Node *region, PieceId piece) {
  for (Node *scope = region; scope && scope->kind != Node::Func;
       scope = scope->parent)
    for (Node *node = scope->next; node; node = node->next)
      if (nodeTouchesPiece(g, node, piece))
        return true;
  return false;
}

// Doc: sync-dag.md#the-walk-accesses-to-edges
class ChainWalker {
public:
  ChainWalker(GroupDag &group, ChainState &state, EdgeList &edges,
              bool underFor)
      : group(group), state(state), edges(edges), underFor(underFor) {}

  ExitFacts run(Node *head) {
    if (head->kind == Node::Enter)
      if (auto owner = uniformPieceOwner(head); owner && owner->has_value())
        tokens.remember(*owner);
    for (Node *node = head; node; node = node->next) {
      switch (node->kind) {
      case Node::Access:
        visitAccess(node);
        break;
      case Node::For:
      case Node::If:
        visitRegion(node);
        break;
      case Node::Exit:
        visitExit(node);
        break;
      case Node::Enter:
      case Node::Acquire:
      case Node::Release:
      case Node::Func:
        break;
      }
    }
    return returnedExitFacts(head);
  }

private:
  // Doc: sync-dag.md#memory-edges-and-token-supply
  void visitAccess(Node *node) {
    PieceEffects effects;
    forEachTouchedPiece(group, node, [&](PieceId id, Effect effect) {
      mergeEffect(effects, id, effect);
    });
    const Tokens::Token *last = tokens.last();
    bool ownerDiffers =
        last && node->owner && !sameOwner(last->owner, node->owner);
    bool canReuse = tokens.find(node->owner) &&
                    llvm::all_of(effects, [&](const auto &item) {
                      return canReuseTokenForPiece(state, item.first,
                                                   node->owner, item.second);
                    });
    size_t edgeStart = edges.size();
    Payloads payloads{asyncPayloadOf(node->op)};

    // A release describes every completion signal produced during one
    // ownership wave, not just the last access in that wave.  Members can be
    // written consecutively by the same owner before another owner consumes
    // either member.  Keep earlier async completions in that case; otherwise a
    // later synchronous write would hide (for example) a descriptor load from
    // LowerAref.
    //
    // A foreign active use marks an ownership handoff.  Its dependency has
    // already consumed the earlier wave, so a write after that handoff starts
    // a fresh payload set rather than carrying completed work forward.
    // Use the group-wide token-reuse proof rather than member geometry.  It
    // requires one reusable owner token and proves that no new handoff is
    // needed on any touched piece, so a handoff forced by an overlapping piece
    // starts a fresh payload set.
    bool synchronousWrite = payloads.front() == AsyncOp::NONE;
    if (group.pieceTable.members.size() > 1 && canReuse &&
        synchronousWrite) {
      for (auto [id, effect] : effects) {
        if (effect != Effect::W)
          continue;
        auto it = state.find(id);
        if (it == state.end() || !it->second.initialized())
          continue;
        PieceState &piece = it->second;
        if (!sameOwner(piece.source.sourceOwner, node->owner))
          continue;
        bool sameOwnerWave = llvm::all_of(piece.uses, [&](const ActiveUse &use) {
          return sameOwner(use.owner, node->owner);
        });
        if (sameOwnerWave)
          unionPayloads(payloads, piece.source.payloads);
      }
    }

    for (auto [id, effect] : effects)
      applyTouch(state[id], id, node->owner, effect, node, payloads, edges,
                 /*wsAdopt=*/false);

    bool noDataEdge = edges.size() == edgeStart;
    bool reusesToken = noDataEdge && canReuse;
    if (reusesToken) {
      markTokenReuse(node, node->owner);
    } else if (ownerDiffers && noDataEdge) {
      assert(last && last->node && "last token without a source node");
      edges.push_back(EdgeRec{
          last->node, node, last->owner, node->owner, last->payloads, {}});
    }
    if (node->owner && !(ownerDiffers && reusesToken))
      tokens.record(node->owner, node, payloads);
  }

  // Doc: sync-dag.md#composition-nested-regions-in-the-walk
  void visitRegion(Node *node) {
    auto infos = sortedPieceInfo(node);
    std::map<PieceId, VersionSource> incoming;
    std::map<PieceId, SmallVector<int64_t, 2>> incomingOrder;
    for (auto [id, info] : infos) {
      auto it = state.find(id);
      if (it != state.end() && it->second.initialized()) {
        incoming.emplace(id, it->second.source);
        if (ActiveUse *use = it->second.useFor(info.owner))
          incomingOrder[id] = use->orderedBefore;
      }
    }

    Payloads none{AsyncOp::NONE};
    bool wsAdopt =
        node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [id, info] : infos)
      applyTouch(state[id], id, info.owner, info.effect, node, none, edges,
                 wsAdopt);

    ExitFacts returned;
    bool firstChild = true;
    for (Node *childHead : node->children) {
      ChainState child;
      for (auto [id, info] : sortedPieceInfo(childHead)) {
        auto before = incoming.find(id);
        Owner producer =
            before == incoming.end() ? info.owner : before->second.producer;
        Payloads seed{AsyncOp::NONE};
        if (before != incoming.end() &&
            sameOwner(info.owner, before->second.producer))
          seed = before->second.payloads;
        child[id].startVersion(producer, info.owner, childHead, seed);
      }
      ChainWalker nested(group, child, edges,
                         underFor || node->kind == Node::For);
      ExitFacts childFacts = nested.run(childHead);
      for (auto &[id, facts] : childFacts)
        unionPayloads(returned[id].payloads, facts.payloads);

      for (auto [id, info] : infos) {
        SmallVector<int64_t, 2> branchOrder = incomingOrder[id];
        if (auto it = childFacts.find(id); it != childFacts.end())
          branchOrder = it->second.mustOrderedBefore;
        if (firstChild)
          returned[id].mustOrderedBefore = std::move(branchOrder);
        else
          returned[id].mustOrderedBefore = intersectOrderFacts(
              returned[id].mustOrderedBefore, branchOrder);
      }
      firstChild = false;
    }

    // A loop body does not establish a fact on its zero-trip path. Likewise,
    // an absent else branch carries the incoming fact unchanged.
    if (node->kind == Node::For && !knownNonEmptyLoop(node)) {
      for (auto [id, info] : infos)
        returned[id].mustOrderedBefore = intersectOrderFacts(
            returned[id].mustOrderedBefore, incomingOrder[id]);
    } else if (node->kind == Node::If && node->children.size() < 2) {
      for (auto [id, info] : infos)
        returned[id].mustOrderedBefore = intersectOrderFacts(
            returned[id].mustOrderedBefore, incomingOrder[id]);
    }

    for (auto [id, info] : infos) {
      auto facts = returned.find(id);
      ActiveUse *use = state[id].useFor(info.owner);
      if (facts == returned.end() || !use)
        continue;
      if (!facts->second.payloads.empty())
        use->payloads = facts->second.payloads;
      if (use->node == node)
        for (int64_t owner : facts->second.mustOrderedBefore)
          if (!llvm::is_contained(use->orderedBefore, owner))
            use->orderedBefore.push_back(owner);
      if (state[id].source.node == node &&
          !facts->second.payloads.empty())
        state[id].source.payloads = facts->second.payloads;
    }
    if (infos.empty())
      return;
    tokens.live.clear();
    if (auto owner = uniformPieceOwner(node); owner && owner->has_value())
      tokens.record(*owner, node, none);
  }

  void visitExit(Node *node) {
    for (auto [id, info] : sortedPieceInfo(node)) {
      auto it = state.find(id);
      if (it == state.end())
        continue;
      PieceState &piece = it->second;
      if (underFor || pieceTouchedAfter(group, node->parent, id))
        for (const ActiveUse &use : piece.uses)
          if (!sameOwner(use.owner, info.owner) &&
              !llvm::is_contained(use.orderedBefore, ownerKey(info.owner)))
            edges.push_back(EdgeRec{
                use.node, node, use.owner, info.owner, use.payloads, {id}});
      ActiveUse carried{info.owner, node, {AsyncOp::NONE}, {}};
      if (ActiveUse *use = piece.useFor(info.owner))
        carried = *use;
      piece.uses.assign(1, carried);
    }
  }

  ExitFacts returnedExitFacts(Node *head) {
    ExitFacts result;
    if (head->kind != Node::Enter)
      return result;
    for (auto [id, info] : sortedPieceInfo(head)) {
      PieceExitFacts facts;
      facts.payloads.push_back(AsyncOp::NONE);
      auto it = state.find(id);
      if (it != state.end())
        if (ActiveUse *use = it->second.useFor(info.owner)) {
          if (!use->payloads.empty())
            facts.payloads = use->payloads;
          facts.mustOrderedBefore = use->orderedBefore;
        }
      result.emplace(id, std::move(facts));
    }
    return result;
  }

  GroupDag &group;
  ChainState &state;
  EdgeList &edges;
  bool underFor;
  Tokens tokens;
};

static ExitFacts walkChain(GroupDag &g, Node *head, ChainState &state,
                           EdgeList &edges, bool underFor) {
  return ChainWalker(g, state, edges, underFor).run(head);
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
using EdgeBuckets = DenseMap<Node *, SmallVector<unsigned, 2>>;

struct ChainIndex {
  DenseMap<Node *, unsigned> idx;   // node -> position within its chain
  DenseMap<Node *, Node *> chainOf; // node -> chain head
  SmallVector<Node *, 8> heads;
};
static void indexChains(Node *head, ChainIndex &ci) {
  ci.heads.push_back(head);
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
static void sortLatestSourceFirst(EdgeBuckets &buckets, ArrayRef<EdgeRec> edges,
                                  const ChainIndex &ci) {
  for (auto &item : buckets)
    llvm::stable_sort(item.second, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
}
static void recordTokenOwner(SmallVectorImpl<int64_t> &owners, int64_t owner) {
  auto it = llvm::find(owners, owner);
  if (it != owners.end())
    owners.erase(it);
  owners.push_back(owner);
}
static void applyKnownEdge(const EdgeRec &edge, unsigned sourceIdx,
                           const DenseMap<Node *, SyncVec> &snapshots,
                           std::map<int64_t, SyncVec> &behind) {
  SyncVec &known = behind[ownerKey(edge.dstOwner)];
  auto snapshot = snapshots.find(edge.src);
  if (snapshot != snapshots.end())
    for (auto &[owner, idx] : snapshot->second)
      known[owner] = std::max(known[owner], idx);
  int64_t sourceOwner = ownerKey(edge.srcOwner);
  known[sourceOwner] = std::max(known[sourceOwner], sourceIdx);
}

// Doc: sync-dag.md#1-implied-ordering-reduceedges
// Each dropped edge is proved by program order and edges already committed to
// the kept set. Dropped edges never update the state, and a kept edge is never
// reconsidered, so later reductions cannot invalidate an earlier proof.
static void reduceStraightEdges(Node *head, const ChainIndex &ci,
                                ArrayRef<EdgeRec> edges,
                                std::vector<bool> &drop) {
  EdgeBuckets atDst;
  for (auto [i, e] : llvm::enumerate(edges))
    if (!drop[i] && ci.chainOf.lookup(e.src) == head &&
        ci.chainOf.lookup(e.dst) == head)
      atDst[e.dst].push_back(i);
  sortLatestSourceFirst(atDst, edges, ci);
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snapshotAtNode;
  SmallVector<int64_t, 2> tokenOwners;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner)
        recordTokenOwner(tokenOwners, ownerKey(pi.owner));
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        const EdgeRec &e = edges[ei];
        if (!e.srcOwner || !e.dstOwner || e.src->kind != Node::Access)
          continue; // region/root endpoints: never reduced
        int64_t sk = ownerKey(e.srcOwner), dk = ownerKey(e.dstOwner);
        unsigned srcIdx = ci.idx.lookup(e.src);
        bool implied = covers(behind[dk], sk, srcIdx);
        bool destinationHasToken =
            !tokenOwners.empty() && tokenOwners.back() == dk;
        if (implied && destinationHasToken && e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        recordTokenOwner(tokenOwners, dk); // Kept acquire supplies Q's token.
        applyKnownEdge(e, srcIdx, snapshotAtNode, behind);
      }
    if (n->kind != Node::Exit && n->owner) {
      int64_t owner = ownerKey(n->owner);
      behind[owner][owner] = ci.idx.lookup(n);
      snapshotAtNode[n] = behind[owner];
    }
  }
}

static bool isLoopClose(const EdgeRec &edge) {
  return edge.dst->kind == Node::Exit && edge.src->kind == Node::Access &&
         edge.srcOwner && edge.dstOwner;
}
// Doc: sync-dag.md#example-a-loop-closing-edge-is-dropped
static void reduceLoopCloses(GroupDag &g, Node *head, const ChainIndex &ci,
                             ArrayRef<EdgeRec> edges, std::vector<bool> &drop) {
  Owner firstAccessOwner;
  for (Node *n = head; n && !firstAccessOwner; n = n->next)
    if (n->kind == Node::Access && n->owner)
      firstAccessOwner = n->owner;
  if (!firstAccessOwner)
    return;
  constexpr unsigned kPass2 = 1u << 20;
  EdgeBuckets atDst;
  SmallVector<unsigned, 4> closes;
  for (auto [i, e] : llvm::enumerate(edges)) {
    if (drop[i] || ci.chainOf.lookup(e.src) != head ||
        ci.chainOf.lookup(e.dst) != head)
      continue;
    if (isLoopClose(e))
      closes.push_back(i);
    else
      atDst[e.dst].push_back(i);
  }
  if (closes.empty())
    return;
  sortLatestSourceFirst(atDst, edges, ci);
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
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        applyKnownEdge(edges[ei], ci.idx.lookup(edges[ei].src), snap1, behind);
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = ci.idx.lookup(n);
      snap1[n] = behind[ownerKey(n->owner)];
    }
  }
  DenseMap<Node *, SmallVector<unsigned, 2>> closeAt;
  for (unsigned ei : closes) {
    const EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    for (PieceId pc : e.pieces)
      if (Node *ft = firstTouch(e.dstOwner, pc))
        if (!latest || ci.idx.lookup(ft) > ci.idx.lookup(latest))
          latest = ft;
    if (latest)
      closeAt[latest].push_back(ei);
  }
  // Non-close edges were permanently decided by reduceStraightEdges. During
  // pass 2, each close sees only those edges and earlier closes that were kept.
  DenseSet<int64_t> tokenAvailable;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        applyKnownEdge(edges[ei], kPass2 + ci.idx.lookup(edges[ei].src), snap2,
                       behind);
        tokenAvailable.insert(ownerKey(edges[ei].dstOwner));
      }
    auto ct = closeAt.find(n);
    if (ct != closeAt.end())
      for (unsigned ei : ct->second) {
        const EdgeRec &e = edges[ei];
        int64_t dk = ownerKey(e.dstOwner);
        bool covered =
            covers(behind[dk], ownerKey(e.srcOwner), ci.idx.lookup(e.src));
        if (covered && tokenAvailable.contains(dk) &&
            !sameOwner(e.dstOwner, firstAccessOwner)) {
          drop[ei] = true;
          continue;
        }
        // A kept close becomes permanent before the next candidate is tested.
        applyKnownEdge(e, ci.idx.lookup(e.src), snap1, behind);
      }
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = kPass2 + ci.idx.lookup(n);
      snap2[n] = behind[ownerKey(n->owner)];
    }
  }
}

// Remove only edges proved by permanently kept waits, program order, and loop
// closure. Filtering preserves the original order of every surviving edge.
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (g.root->children.empty() || edges.empty())
    return;
  ChainIndex ci;
  indexChains(g.root->children[0], ci);
  std::vector<bool> drop(edges.size(), false);
  for (Node *h : ci.heads)
    reduceStraightEdges(h, ci, edges, drop);
  for (Node *h : ci.heads)
    if (h->parent && h->parent->kind == Node::For)
      reduceLoopCloses(g, h, ci, edges, drop);
  if (llvm::none_of(drop, [](bool dropped) { return dropped; }))
    return;
  SmallVector<EdgeRec> kept;
  kept.reserve(edges.size());
  for (auto [i, e] : llvm::enumerate(edges))
    if (!drop[i])
      kept.push_back(e);
  edges = std::move(kept);
}
} // namespace
static bool precedesInChain(Node *before, Node *after) {
  for (Node *next = before->next; next; next = next->next)
    if (next == after)
      return true;
  return false;
}

// Doc: sync-dag.md#2-repeats-from-one-sender-collapseedges
static SmallVector<EdgeRec> collapseEdges(ArrayRef<EdgeRec> edges) {
  DenseMap<std::tuple<Node *, int64_t>, unsigned> index;
  SmallVector<EdgeRec> merged;
  for (const EdgeRec &edge : edges) {
    auto key = std::make_tuple(edge.dst, ownerKey(edge.srcOwner));
    auto [it, inserted] = index.try_emplace(key, merged.size());
    if (inserted) {
      merged.push_back(edge);
      llvm::sort(merged.back().payloads, [](AsyncOp a, AsyncOp b) {
        return static_cast<int>(a) < static_cast<int>(b);
      });
      continue;
    }
    EdgeRec &dst = merged[it->second];
    if (precedesInChain(dst.src, edge.src))
      dst.src = edge.src;
    unionPayloads(dst.payloads, edge.payloads);
  }
  return merged;
}

static unsigned arrivalContribution(const EdgeRec &edge) {
  // Payload kinds are the distinct completion mechanisms emitted by one
  // release.  Each kind contributes one arrival when arrive_count is one.
  return std::max(1u, static_cast<unsigned>(edge.payloads.size()));
}

static unsigned arrivalContribution(ArrayRef<EdgeRec> edges) {
  return std::accumulate(edges.begin(), edges.end(), 0u,
                         [](unsigned total, const EdgeRec &edge) {
                           return total + arrivalContribution(edge);
                         });
}

// Doc: sync-dag.md#one-destination-node-one-semaphore
static LogicalResult buildEdgesAndSemas(GroupDag &g,
                                        SmallVector<EdgeRec> &edges) {
  reduceEdges(g, edges);
  SmallVector<EdgeRec> collapsed = collapseEdges(edges);
  struct Handoff {
    Node *dst = nullptr;
    Owner owner;
    SmallVector<EdgeRec, 2> incoming;
    std::optional<SemaId> sema;
  };
  llvm::MapVector<std::tuple<Node *, int64_t>, unsigned> dstIndex;
  SmallVector<Handoff> handoffs;
  for (const EdgeRec &edge : collapsed) {
    auto key = std::make_tuple(edge.dst, ownerKey(edge.dstOwner));
    auto [it, inserted] = dstIndex.try_emplace(key, handoffs.size());
    if (inserted)
      handoffs.push_back(Handoff{edge.dst, edge.dstOwner});
    handoffs[it->second].incoming.push_back(edge);
  }
  auto findRegain = [&](Node *forNode, const Owner &owner) -> Handoff * {
    Handoff *regain = nullptr;
    for (Node *m = forNode->children[0]; m; m = m->next) {
      auto it = dstIndex.find(std::make_tuple(m, ownerKey(owner)));
      if (it != dstIndex.end())
        regain = &handoffs[it->second];
    }
    return regain;
  };
  auto createSema = [&](Handoff &handoff) {
    SemaId sid = g.semas.size();
    Sema s;
    s.name = "S" + std::to_string(sid);
    s.count = arrivalContribution(handoff.incoming);
    handoff.sema = sid;
    g.semas.push_back(std::move(s));
  };
  for (Handoff &handoff : handoffs) {
    if (handoff.sema)
      continue;
    if (handoff.dst->kind == Node::For)
      if (Handoff *regain = findRegain(handoff.dst, handoff.owner)) {
        if (!regain->sema)
          createSema(*regain);
        handoff.sema = regain->sema;
        continue;
      }
    createSema(handoff);
  }
  DenseMap<Node *, Node *> lastAfter; // release insertion cursor per source
  for (Handoff &handoff : handoffs) {
    Sema &sema = g.semas[*handoff.sema];
    unsigned sources = handoff.incoming.size();
    unsigned arrivals = arrivalContribution(handoff.incoming);
    if (arrivals != sema.count &&
        !(sources == 1 && arrivals == 1))
      return semaError(handoff.dst->op ? handoff.dst->op : g.root->op)
             << "destination group with " << arrivals
             << " arrival contributions from " << sources
             << " sources cannot meet semaphore " << sema.name
             << " pending count " << sema.count;
    // A lone generic release may be scaled to satisfy a reused semaphore's
    // fan-in count.  A release with multiple completion kinds already emits
    // one arrival per kind and must keep arrive_count equal to one.
    unsigned releaseCount = arrivals == sema.count ? 1 : sema.count;
    Node *acquire = newProtocolNode(g, Node::Acquire, handoff.dst->parent,
                                    handoff.owner, *handoff.sema, sema.count);
    Node *destination = handoff.dst;
    if (handoff.dst->kind == Node::Exit && acquire->owner) {
      Node *head = handoff.dst;
      while (head->prev)
        head = head->prev;
      Owner firstAccessOwner;
      Node *firstTouch = nullptr;
      for (Node *r = head; r; r = r->next) {
        if (r->kind != Node::Access || !r->owner)
          continue;
        if (!firstAccessOwner)
          firstAccessOwner = r->owner;
        if (!firstTouch && sameOwner(r->owner, acquire->owner))
          firstTouch = r;
      }
      if (firstAccessOwner && !sameOwner(firstAccessOwner, acquire->owner) &&
          firstTouch) {
        destination = firstTouch;
        sema.isEntry = true; // initially released; no pre-loop entry instance
        sema.entryTokenOwner = acquire->owner;
      }
    }
    acquire->scheduleAnchor = destination;
    spliceBefore(acquire, destination);
    sema.expectedArrivals += arrivals * releaseCount;
    for (const EdgeRec &edge : handoff.incoming) {
      Node *release =
          newProtocolNode(g, Node::Release, edge.src->parent, edge.srcOwner,
                          *handoff.sema, releaseCount);
      release->payloads = edge.payloads;
      release->sat = acquire;
      release->scheduleAnchor = edge.src;
      if (nodeReusesToken(edge.src, edge.srcOwner))
        markTokenReuse(release, edge.srcOwner);
      Node *anchor = lastAfter.lookup(edge.src);
      spliceAfter(release, anchor ? anchor : edge.src);
      lastAfter[edge.src] = release;
    }
  }
  return success();
}
static bool nodeInvolvesComp(GroupDag &g, Node *n) {
  if (n->kind == Node::Access)
    return nodeTouchesGroup(g, n);
  return n->isRegion() && !n->pieceInfo.empty();
}
static SmallVector<Node *, 4> compNodesInChain(GroupDag &g, Node *head) {
  SmallVector<Node *, 4> nodes;
  for (Node *n = head; n; n = n->next)
    if (nodeInvolvesComp(g, n))
      nodes.push_back(n);
  return nodes;
}
static Node *singleCompChild(GroupDag &g, Node *region) {
  Node *only = nullptr;
  for (Node *child : region->children) {
    if (compNodesInChain(g, child).empty())
      continue;
    if (only)
      return nullptr;
    only = child;
  }
  return only;
}
static Node *lastAcquireOfCompInChain(Node *head) {
  Node *found = nullptr;
  forEachNode(head, [&](Node *n) {
    if (n->kind == Node::Acquire)
      found = n;
  });
  return found;
}
static std::optional<Owner> firstAccessOwnerOfComp(GroupDag &g, Node *head) {
  std::optional<Owner> owner;
  forEachNode(head, [&](Node *n) {
    if (!owner && n->kind == Node::Access && nodeTouchesGroup(g, n))
      owner.emplace(n->owner);
  });
  return owner;
}

// Doc:
// sync-dag.md#composition-why-loop-entry-and-loop-recurrence-share-one-semaphore
static LogicalResult insertEntryAcquires(GroupDag &g) {
  Node *top = g.root->children.empty() ? nullptr : g.root->children[0];
  if (!top || g.semas.empty())
    return success();
  SmallVector<Node *, 4> nodes = compNodesInChain(g, top);
  while (nodes.size() == 1 && nodes.front()->kind == Node::If) {
    Node *child = singleCompChild(g, nodes.front());
    if (!child)
      break;
    nodes = compNodesInChain(g, child);
  }
  if (nodes.empty())
    return semaError(g.root->op) << "group with sync but no placement nodes";

  std::optional<Owner> entryTokenOwner = firstAccessOwnerOfComp(g, top);
  if (!entryTokenOwner)
    return semaError(g.root->op) << "group has no access nodes";

  Node *regain = nullptr;
  for (Node *node : llvm::reverse(nodes))
    if (node->kind == Node::For &&
        (regain = lastAcquireOfCompInChain(node->children[0])))
      break;
  if (regain) {
    Sema &s = getSema(g, regain);
    s.isEntry = true; // first event in chain order is an acquire
    s.entryTokenOwner = *entryTokenOwner;
    Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent,
                                std::nullopt, regain->sema, regain->count);
    spliceBefore(acq, nodes.front());
    return success();
  }

  SemaId sid = g.semas.size();
  Sema s;
  s.name = "E" + std::to_string(sid);
  s.count = 1;
  s.isEntry = true;
  s.expectedArrivals = 1; // the terminal release
  s.entryTokenOwner = *entryTokenOwner;
  Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent,
                              std::nullopt, sid, 1);
  spliceBefore(acq, nodes.front());
  Node *terminal = nodes.back();
  Owner owner = terminal->kind == Node::Access
                    ? terminal->owner
                    : sortedPieceInfo(terminal).front().second.owner;
  Node *rel =
      newProtocolNode(g, Node::Release, terminal->parent, owner, sid, 1);
  rel->payloads.push_back(AsyncOp::NONE);
  Node *anchor = terminal;
  while (anchor->next && anchor->next->kind == Node::Release)
    anchor = anchor->next;
  spliceAfter(rel, anchor);
  g.semas.push_back(std::move(s));
  return success();
}
// Doc: sync-dag.md#the-decision-per-region
using SemaTransfer = RegionFlow::SemaTransfer;
struct SemaExpr {
  SemaTransfer kind = SemaTransfer::INVALID;
  SemaId sema = 0;
};

static SemaExpr inputSema() { return {SemaTransfer::INPUT, 0}; }
static SemaExpr concreteSema(SemaId sema) {
  return {SemaTransfer::CONCRETE, sema};
}
static bool sameSemaExpr(const SemaExpr &a, const SemaExpr &b) {
  return a.kind == b.kind &&
         (a.kind == SemaTransfer::INPUT || a.kind == SemaTransfer::INVALID ||
          a.sema == b.sema);
}
static SemaExpr joinSemaExpr(SemaExpr a, SemaExpr b) {
  if (a.kind == SemaTransfer::INVALID || b.kind == SemaTransfer::INVALID)
    return {};
  if (sameSemaExpr(a, b))
    return a;
  auto concreteOf = [](SemaExpr value) -> std::optional<SemaId> {
    if (value.kind == SemaTransfer::CONCRETE ||
        value.kind == SemaTransfer::INPUT_OR_CONCRETE)
      return value.sema;
    return std::nullopt;
  };
  std::optional<SemaId> lhs = concreteOf(a), rhs = concreteOf(b);
  if (a.kind == SemaTransfer::INPUT && rhs)
    return {SemaTransfer::INPUT_OR_CONCRETE, *rhs};
  if (b.kind == SemaTransfer::INPUT && lhs)
    return {SemaTransfer::INPUT_OR_CONCRETE, *lhs};
  if (lhs && rhs && *lhs == *rhs)
    return {SemaTransfer::INPUT_OR_CONCRETE, *lhs};
  return {};
}
static SemaExpr applyFlow(const RegionFlow &flow, SemaExpr input) {
  switch (flow.semaTransfer) {
  case SemaTransfer::INPUT:
    return input;
  case SemaTransfer::CONCRETE:
    return concreteSema(flow.concreteSema);
  case SemaTransfer::INPUT_OR_CONCRETE:
    if (input.kind == SemaTransfer::INPUT)
      return {SemaTransfer::INPUT_OR_CONCRETE, flow.concreteSema};
    if ((input.kind == SemaTransfer::CONCRETE ||
         input.kind == SemaTransfer::INPUT_OR_CONCRETE) &&
        input.sema == flow.concreteSema)
      return input;
    return {};
  case SemaTransfer::INVALID:
    return {};
  }
  llvm_unreachable("unknown semaphore transfer");
}
static std::optional<SemaId> resolveFlowSema(const RegionFlow &flow,
                                             SemaId input) {
  SemaExpr result = applyFlow(flow, concreteSema(input));
  if (result.kind != SemaTransfer::CONCRETE)
    return std::nullopt;
  return result.sema;
}

static Node *chainFinalForComp(Node *head) {
  Node *final = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire)
      final = n;
    if (n->isRegion() && n->flow)
      final = n;
  }
  return final;
}
static SemaExpr chainSemaExpr(Node *head) {
  SemaExpr current = inputSema();
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire)
      current = concreteSema(n->sema);
    else if (n->isRegion() && n->flow)
      current = applyFlow(*n->flow, current);
  }
  return current;
}
static Owner finalOwner(Node *final) {
  if (final->kind == Node::Acquire)
    return final->owner;
  if (final->flow)
    return final->flow->owner;
  return std::nullopt;
}
static bool exportsToken(const Node *final) {
  if (!final)
    return false;
  if (final->kind == Node::Acquire)
    return true;
  return final->isRegion() && final->flow &&
         (final->kind != Node::For || final->flow->threadsToken());
}
struct CompletionExpr {
  bool usesInput = true, hasFallback = false;
  gpu::StageCluster schedule;
};
static CompletionExpr concreteCompletion(gpu::StageCluster schedule) {
  return {false, true, schedule};
}
static CompletionExpr applyCompletion(const RegionFlow &flow,
                                      CompletionExpr input) {
  if (!flow.completionUsesInput)
    return concreteCompletion(flow.completionSchedule);
  if (!flow.completionHasFallback || input.hasFallback)
    return input;
  return {true, true, flow.completionSchedule};
}
static CompletionExpr preferCompletion(CompletionExpr first,
                                       CompletionExpr second) {
  if (!first.usesInput || first.hasFallback || !second.hasFallback)
    return first;
  return {true, true, second.schedule};
}
static CompletionExpr completionAfterChain(Node *head, const Owner &owner,
                                           CompletionExpr state = {}) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access && sameOwner(n->owner, owner)) {
      Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
      state = concreteCompletion(gpu::getStageCluster(completion));
      continue;
    }
    if (!n->isRegion() || !n->flow ||
        (n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op)))
      continue;
    const RegionFlow &nested = *n->flow;
    if (sameOwner(nested.owner, owner))
      state = applyCompletion(nested, state);
  }
  return state;
}

static bool chainHasCompEvent(GroupDag &g, Node *head);
static bool summarizeRegionFlow(GroupDag &g, Node *region,
                              RegionFlow &crossing) {
  crossing.exits.clear();
  crossing.exitTokens.clear();
  crossing.owner = std::nullopt;
  crossing.ownersCompatible = true;
  bool live = false;
  bool sawOwner = false;
  bool transparent = true;
  std::optional<Owner> entryOwner = uniformPieceOwner(region);
  std::optional<SemaExpr> joined;
  auto joinOwner = [&](const Owner &owner) {
    if (!sawOwner) {
      crossing.owner = owner;
      sawOwner = true;
    } else if (!sameOwner(crossing.owner, owner)) {
      crossing.ownersCompatible = false;
      transparent = false;
    }
  };
  for (Node *child : region->children) {
    Node *final = chainFinalForComp(child);
    crossing.exits.push_back(final);
    crossing.exitTokens.push_back(
        !final ? RegionFlow::ExitToken::INPUT
               : exportsToken(final) ? RegionFlow::ExitToken::CURRENT
                                     : RegionFlow::ExitToken::NONE);
    SemaExpr branch = chainSemaExpr(child);
    joined = joined ? std::optional<SemaExpr>(joinSemaExpr(*joined, branch))
                    : std::optional<SemaExpr>(branch);
    if ((!final && chainHasCompEvent(g, child)) ||
        (final && chainHasCompEvent(g, final->next)))
      transparent = false;
    if (!final) {
      if (entryOwner)
        joinOwner(*entryOwner);
      continue;
    }
    live = true;
    joinOwner(finalOwner(final));
  }
  if (region->kind == Node::For) {
    joined = joined ? std::optional<SemaExpr>(joinSemaExpr(*joined, inputSema()))
                    : std::optional<SemaExpr>(inputSema());
    if (entryOwner)
      joinOwner(*entryOwner);
  }
  if (region->kind == Node::If && region->children.size() < 2) {
    joined = joined ? std::optional<SemaExpr>(joinSemaExpr(*joined, inputSema()))
                    : std::optional<SemaExpr>(inputSema());
    if (entryOwner)
      joinOwner(*entryOwner);
  }
  SemaExpr output = joined.value_or(SemaExpr{});
  crossing.semaTransfer = output.kind;
  crossing.concreteSema = output.sema;
  crossing.transparent =
      transparent && entryOwner && crossing.owner &&
      sameOwner(*entryOwner, crossing.owner);
  CompletionExpr completion;
  if (!region->children.empty())
    completion = completionAfterChain(region->children[0], crossing.owner);
  if (region->kind == Node::If) {
    CompletionExpr elseCompletion;
    if (region->children.size() > 1)
      elseCompletion =
          completionAfterChain(region->children[1], crossing.owner);
    completion = preferCompletion(completion, elseCompletion);
  }
  crossing.completionUsesInput = completion.usesInput;
  crossing.completionHasFallback = completion.hasFallback;
  crossing.completionSchedule = completion.schedule;
  return live;
}
static void buildRegionFlows(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    RegionFlow crossing;
    if (summarizeRegionFlow(g, n, crossing))
      n->flow.emplace(std::move(crossing));
  });
}
static bool tokenUsedBeforeNextAcquire(GroupDag &g, Node *start) {
  for (Node *n = start; n; n = n->next) {
    if (n->kind == Node::Acquire)
      return false; // a fresh token supersedes the earlier one
    if (n->kind == Node::Release ||
        (n->kind == Node::Access && nodeTouchesGroup(g, n)) ||
        (n->isRegion() && n->flow))
      return true;
  }
  return false;
}

static void pruneDeadIfFlows(GroupDag &g, Node *head, Node *region) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes))
    if (n->kind == Node::If && !tokenUsedBeforeNextAcquire(g, n->next) &&
        (!region || !region->flow))
      n->flow.reset();
  for (Node *n : nodes)
    if (n->isRegion())
      for (Node *child : n->children)
        pruneDeadIfFlows(g, child, n);
}
static scf::ForOp outerWSLoop(scf::ForOp loop);
static const RegionFlow *findFlow(const Node *n) {
  return n->flow ? &*n->flow : nullptr;
}
static bool exitsThroughRegion(const RegionFlow &flow) {
  return flow.exit() && flow.exit()->isRegion();
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
    if (m->isRegion() && m->flow)
      return true;
  }
  Node *p = region->parent;
  if (!p || !p->isRegion())
    return false;
  if (p->flow && llvm::is_contained(p->flow->exits, region))
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
static bool isAccessForComp(GroupDag &g, Node *n) {
  return n->kind == Node::Access && nodeInvolvesComp(g, n);
}
static bool isRegionFlowForComp(const Node *n) {
  return n && n->isRegion() && n->flow;
}
static bool nodeHasCompEvent(GroupDag &g, Node *n) {
  return n->isProtocol() || nodeInvolvesComp(g, n);
}
static bool isTokenEvent(GroupDag &g, Node *n) {
  return n->isProtocol() || isAccessForComp(g, n) || isRegionFlowForComp(n);
}
static bool chainHasCompEvent(GroupDag &g, Node *head) {
  for (Node *n = head; n; n = n->next)
    if (nodeHasCompEvent(g, n))
      return true;
  return false;
}
static std::optional<SemaId>
resolveExitSema(Node *final, SemaId incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->sema;
  if (!final->isRegion())
    return std::nullopt;
  const RegionFlow *c = findFlow(final);
  if (!c)
    return std::nullopt;
  return resolveFlowSema(*c, incoming);
}
static bool isTransparentFlow(Node *region, Owner owner) {
  const RegionFlow *rc = findFlow(region);
  return rc && rc->transparent && sameOwner(rc->owner, owner);
}
static bool childOwnsToken(const Node *region) {
  const RegionFlow *child = findFlow(region);
  return child && !child->threadsToken();
}
static bool hasTrailingCompUse(GroupDag &g, Node *regain) {
  for (Node *m = regain->next; m; m = m->next)
    if (isAccessForComp(g, m) || isRegionFlowForComp(m) ||
        m->kind == Node::Release)
      return true;
  return false;
}
static Node *findBridgeAcquireAfter(GroupDag &g, Node *F, SemaId feedSema,
                                    Owner owner) {
  for (Node *m = F->next; m; m = m->next) {
    if (m->kind != Node::Acquire || m->sema != feedSema ||
        !sameOwner(m->owner, owner))
      continue;
    for (Node *tail = m->next; tail; tail = tail->next)
      if (isTokenEvent(g, tail))
        return nullptr;
    return m;
  }
  return nullptr;
}

struct InputCapability {
  Node *acquire = nullptr;
  bool retainForPrefix = false;
};

static bool carrierPreservesFeed(GroupDag &g, Node *carrier,
                                 const Owner &owner, SemaId sema) {
  if (!carrier || carrier->kind != Node::For)
    return false;
  std::optional<Owner> boundaryOwner = uniformPieceOwner(carrier);
  const RegionFlow *crossing = findFlow(carrier);
  if (!boundaryOwner || !sameOwner(*boundaryOwner, owner) || !crossing ||
      !sameOwner(crossing->owner, owner))
    return false;
  std::optional<SemaId> returned = resolveFlowSema(*crossing, sema);
  return returned && *returned == sema;
}

static bool feedDominatesThroughCarriers(GroupDag &g, Node *F, Node *feed,
                                         const Owner &owner, SemaId sema) {
  if (!feed || feed->kind != Node::Acquire || feed->sema != sema)
    return false;
  const Sema &feedSema = getSema(g, feed);
  Owner feedOwner = feed->owner ? feed->owner : feedSema.entryTokenOwner;
  if (!sameOwner(feedOwner, owner))
    return false;

  for (Node *cursor = F; cursor;) {
    if (feed->parent == cursor->parent)
      return precedesInChain(feed, cursor);
    Node *carrier = cursor->parent;
    if (!carrierPreservesFeed(g, carrier, owner, sema))
      return false;
    cursor = carrier;
  }
  return false;
}

static std::optional<InputCapability>
findInputAcquire(GroupDag &g, Node *F, const RegionFlow &crossing) {
  bool crossedPrefixUse = false;
  for (Node *cur = F;; cur = cur->parent) {
    bool originalChain = cur == F;
    for (Node *m = cur->prev; m; m = m->prev) {
      if (m->kind == Node::Acquire) {
        const Sema &sema = getSema(g, m);
        Owner owner = m->owner ? m->owner : sema.entryTokenOwner;
        if (!sameOwner(owner, crossing.owner) ||
            !feedDominatesThroughCarriers(g, F, m, crossing.owner,
                                          m->sema))
          return std::nullopt;
        return InputCapability{m, crossedPrefixUse};
      }
      if (m->isRegion() && nodeInvolvesComp(g, m))
        return std::nullopt;
      if (!isTokenEvent(g, m))
        continue;
      if (m->kind == Node::Access &&
          originalChain &&
          sameOwner(m->owner, crossing.owner) &&
          (!m->reuseTokenOwner ||
           *m->reuseTokenOwner == ownerKey(crossing.owner)) &&
          asyncPayloadOf(m->op) == AsyncOp::NONE) {
        crossedPrefixUse = true;
        continue;
      }
      return std::nullopt;
    }
    if (!cur->parent || !cur->parent->isRegion())
      return std::nullopt;
    if (crossedPrefixUse && cur->parent->kind != Node::For)
      return std::nullopt;
  }
}

struct DemandPrefix {
  Node *firstToucher = nullptr;
  Node *closingRelease = nullptr;
};
static std::optional<DemandPrefix>
matchDemandPrefix(GroupDag &g, Node *F, Owner tokenOwner, Node *regain,
                bool nestedExit) {
  DemandPrefix p;
  unsigned releases = 0;
  for (Node *m = F->children[0]; m; m = m->next) {
    if (nestedExit && m == regain)
      break;
    if (m->kind == Node::Acquire)
      break;
    if (isRegionFlowForComp(m)) {
      if (!p.firstToucher || !isTransparentFlow(m, tokenOwner))
        return std::nullopt;
      continue;
    }
    if (isAccessForComp(g, m)) {
      if (!p.firstToucher) {
        if (releases)
          return std::nullopt;
        p.firstToucher = m;
      }
    }
    if (m->kind == Node::Release) {
      releases += std::max(1u, m->count);
      if (!p.closingRelease)
        p.closingRelease = m;
    }
  }
  unsigned expectedReleases = nestedExit ? 0 : 1;
  if (!p.firstToucher || releases != expectedReleases)
    return std::nullopt;
  return p;
}

static bool hasPlannedSingleCopy(const GroupDag &g) {
  return !g.pieceTable.members.empty() &&
         llvm::all_of(g.pieceTable.members, [](const Member &member) {
           auto copy =
               member.allocOp->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
           return copy && copy.getInt() == 1;
         });
}

// Select the loop transport directly from its opaque boundary transfer.  The
// child body has already been compiled; this function reads no child internals.
static LogicalResult planLoopFlow(GroupDag &g, Node *F, RegionFlow &flow,
                                  bool &needsPostLoopAcquire) {
  auto carry = [&](RegionFlow::Blocker blocker = RegionFlow::Blocker::NONE) {
    flow.mode = RegionFlow::Mode::CARRIED;
    flow.blocker = blocker;
    return success();
  };
  bool outputUsed = regionResultConsumedAfter(g, F);
  auto forOp = dyn_cast_or_null<scf::ForOp>(F->op);
  if (!forOp || !gpu::hasWarpSpecializeTag(outerWSLoop(forOp)) ||
      !allEnclosersCanDrop(F))
    return carry();

  Node *regain = flow.exits.empty() ? nullptr : flow.exits[0];
  if (!regain || regain->postLoopAcquire)
    return carry();
  if (regain->kind == Node::For && childOwnsToken(regain)) {
    flow.mode = RegionFlow::Mode::CHILD_OWNS;
    return success();
  }
  bool nestedExit = regain->isRegion();
  if ((nestedExit && !isTransparentFlow(regain, flow.owner)) ||
      (!nestedExit && regain->kind != Node::Acquire))
    return carry();
  if (hasTrailingCompUse(g, regain))
    return carry(RegionFlow::Blocker::TRAILING_USE);

  std::optional<InputCapability> feed = findInputAcquire(g, F, flow);
  if (!feed)
    return carry();
  // Moving an inherited regain across the backedge is currently modeled only
  // for a planner-authored single physical slot. Unknown or multibuffered
  // groups retain the existing threaded form so physical-slot distance remains
  // derived by the normal schedule analysis.
  if (feed->retainForPrefix && !hasPlannedSingleCopy(g))
    return carry();
  Node *entryAcquire = feed->acquire;
  std::optional<SemaId> recurrenceSema =
      nestedExit
          ? resolveExitSema(regain, entryAcquire->sema)
          : std::optional<SemaId>(regain->sema);
  if (!recurrenceSema)
    return carry();

  needsPostLoopAcquire = outputUsed || feed->retainForPrefix;
  bool sameSemaphore = entryAcquire->sema == *recurrenceSema;
  if (sameSemaphore && needsPostLoopAcquire && nestedExit)
    return carry(RegionFlow::Blocker::RESULT_CONSUMED);

  bool differentEntrySemaphore = entryAcquire->sema != *recurrenceSema;
  Node *bridgeAcquire = feed->retainForPrefix ? entryAcquire : nullptr;
  if (feed->retainForPrefix && nestedExit)
    return carry();
  if (differentEntrySemaphore) {
    if (nestedExit || !needsPostLoopAcquire)
      return carry();
    const Sema &entrySema = getSema(g, entryAcquire);
    Owner entryOwner = entryAcquire->owner ? entryAcquire->owner
                                           : entrySema.entryTokenOwner;
    bridgeAcquire =
        findBridgeAcquireAfter(g, F, entryAcquire->sema, flow.owner);
    if (!sameOwner(entryOwner, flow.owner) || !bridgeAcquire)
      return carry();
  }
  std::optional<DemandPrefix> prefix =
      matchDemandPrefix(g, F, flow.owner, regain, nestedExit);
  if (!prefix || !prefixNodeIsSingleBufferView(F, prefix->firstToucher))
    return carry();
  if (needsPostLoopAcquire && !gpu::hasWarpSpecializeTag(forOp) &&
      sameOwner(prefix->firstToucher->owner, flow.owner)) {
    gpu::StageCluster point = gpu::getStageCluster(prefix->firstToucher->op);
    gpu::StageCluster exit = flow.completionSchedule;
    if (point && exit && point->first != exit->first)
      return carry();
  }
  flow.mode = RegionFlow::Mode::POINT_OF_USE;
  flow.entryAcquire = entryAcquire;
  flow.closingRelease = prefix->closingRelease;
  flow.firstToucher = prefix->firstToucher;
  flow.bridgeAcquire = bridgeAcquire;
  return success();
}
static void detachFromChain(Node *n) {
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

static void lowerFlowHandoffs(GroupDag &g, Node *loop, RegionFlow &flow,
                              Node *&afterLoop, bool needsPostLoopAcquire) {
  if (!flow.isPointOfUse() || !needsPostLoopAcquire)
    return;
  Node *recurrenceAcquire =
      exitsThroughRegion(flow) ? flow.entryAcquire : flow.exit();
  flow.postLoopAcquire = newProtocolNode(
      g, Node::Acquire, loop->parent, flow.owner, recurrenceAcquire->sema,
      recurrenceAcquire->count);
  flow.postLoopAcquire->postLoopAcquire = true;
  spliceAfter(flow.postLoopAcquire, afterLoop);
  afterLoop = flow.postLoopAcquire;
  bool prefixCut = flow.bridgeAcquire == flow.entryAcquire &&
                   flow.entryAcquire->sema == recurrenceAcquire->sema;
  if (prefixCut) {
    Node *cut = newProtocolNode(g, Node::Release, loop->parent, flow.owner,
                                recurrenceAcquire->sema,
                                recurrenceAcquire->count);
    cut->payloads.push_back(AsyncOp::NONE);
    if (flow.owner)
      markTokenReuse(cut, flow.owner);
    spliceBefore(cut, loop);
    getSema(g, cut).expectedArrivals += cut->count;
    flow.bridgeRelease = cut;
    if (!exitsThroughRegion(flow) && flow.exit())
      flow.exit()->recurrenceDistance = 1;
  } else if (flow.bridgeAcquire) {
    Node *bridge = newProtocolNode(
        g, Node::Release, flow.bridgeAcquire->parent, flow.owner,
        recurrenceAcquire->sema, recurrenceAcquire->count);
    bridge->payloads.push_back(AsyncOp::NONE);
    spliceAfter(bridge, flow.bridgeAcquire);
    getSema(g, bridge).expectedArrivals += bridge->count;
    flow.bridgeRelease = bridge;
  }
}

static void lowerFlowPlacement(GroupDag &g, Node *loop, RegionFlow &flow) {
  if (!flow.isPointOfUse())
    return;
  if (exitsThroughRegion(flow)) {
    Node *tail = flow.exits[0];
    Node *pointAcquire = flow.entryAcquire;
    Sema &s = getSema(g, pointAcquire);
    pointAcquire->owner = flow.owner;
    detachFromChain(pointAcquire);
    pointAcquire->scheduleAnchor = flow.firstToucher;
    spliceBefore(pointAcquire, flow.firstToucher);
    Node *closing = newProtocolNode(g, Node::Release, tail->parent,
                                    flow.owner, pointAcquire->sema, 1);
    closing->payloads.push_back(AsyncOp::NONE);
    closing->sat = pointAcquire;
    closing->scheduleAnchor = tail;
    spliceAfter(closing, tail);
    s.expectedArrivals += closing->count;
    flow.closingRelease = closing;
    return;
  }
  Node *regain = flow.exits[0];
  detachFromChain(regain);
  regain->scheduleAnchor = flow.firstToucher;
  spliceBefore(regain, flow.firstToucher);
  if (flow.bridgeAcquire) {
    Sema &recurrenceSema = getSema(g, regain);
    recurrenceSema.isEntry = true;
    recurrenceSema.entryTokenOwner = flow.owner;
  } else {
    detachFromChain(flow.entryAcquire);
  }
}

// Regions are lowered from right to left and inner to outer.  A later sibling
// therefore materializes its boundary supply before an earlier sibling asks
// whether its result is live.  Child summaries are refreshed once after their
// final placement; no mutation-driven global reclassification is required.
static LogicalResult planRegionFlows(GroupDag &g, Node *head) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes)) {
    if (!n->isRegion())
      continue;
    for (Node *child : n->children)
      if (failed(planRegionFlows(g, child)))
        return failure();
    if (n->flow)
      summarizeRegionFlow(g, n, *n->flow);
    if (n->kind != Node::For || !n->flow)
      continue;
    Node *afterLoop = n;
    RegionFlow &flow = *n->flow;
    bool needsPostLoopAcquire = false;
    if (failed(planLoopFlow(g, n, flow, needsPostLoopAcquire)))
      return failure();
    lowerFlowHandoffs(g, n, flow, afterLoop, needsPostLoopAcquire);
    lowerFlowPlacement(g, n, flow);
  }
  return success();
}
static void addPart(SmallVectorImpl<int> &parts, int part) {
  if (!llvm::is_contained(parts, part))
    parts.push_back(part);
}
// Doc: sync-dag.md#notation
static SmallVector<int, 4> computeRequiredParts(Node *head) {
  SmallVector<int, 4> chainParts;
  for (Node *n = head; n; n = n->next) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire ||
         n->kind == Node::Release) &&
        n->owner)
      addPart(chainParts, n->owner->first);
    if (!n->isRegion())
      continue;

    SmallVector<int, 4> regionParts;
    for (Node *child : n->children)
      for (int part : computeRequiredParts(child))
        addPart(regionParts, part);
    llvm::sort(regionParts);
    n->requiredParts.assign(regionParts.begin(), regionParts.end());
    for (int part : regionParts)
      addPart(chainParts, part);
  }
  llvm::sort(chainParts);
  return chainParts;
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
  for (const Member &member : g.pieceTable.members)
    for (Operation *user : member.allocOp->getResult(0).getUsers()) {
      auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user);
      auto loop = dyn_cast<scf::ForOp>(user->getParentOp());
      if (!mma || !loop)
        continue;
      if (nvidia_gpu::hasAccReadModifyWrite(mma, loop) ||
          !nvidia_gpu::isAccMultibufferingPossible(mma, loop) ||
          getDisallowAccMultiBuffer(outerWSLoop(loop)) ||
          !canDoubleBufferAcc(mma, numTmemBlocks))
        return false;
    }
  return true;
}

static FailureOr<std::optional<int>> getPlannedBufferCopy(GroupDag &g) {
  std::optional<int> plannedCopy;
  bool sawMissing = false;
  for (const Member &m : g.pieceTable.members) {
    auto copyAttr = m.allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
    if (!copyAttr) {
      sawMissing = true;
      continue;
    }
    int copy = copyAttr.getInt();
    if (copy < 1) {
      semaError(m.allocOp) << "planned buffer.copy must be positive";
      return failure();
    }
    if (plannedCopy && *plannedCopy != copy) {
      semaError(m.allocOp) << "allocs in one planned reuse group have "
                              "inconsistent buffer.copy values";
      return failure();
    }
    plannedCopy = copy;
  }
  if (plannedCopy && sawMissing) {
    semaError(g.pieceTable.members.front().allocOp)
        << "planned reuse group mixes buffer.copy and non-buffer.copy allocs";
    return failure();
  }
  return plannedCopy;
}

// Doc: sync-dag.md#backing-copies
LogicalResult computeBackingPlan(GroupDag &g, bool useMetaPartitioner,
                                 int lowerSemaphoreNumStages,
                                 int &numTmemBlocks) {
  g.numCopies = 1;
  bool synchronized = !g.semas.empty();
  FailureOr<std::optional<int>> planned = getPlannedBufferCopy(g);
  if (failed(planned))
    return failure();
  std::optional<int> plannedCopy = *planned;
  if (synchronized && plannedCopy)
    g.numCopies = *plannedCopy;
  else if (synchronized && g.isTmem() && !useMetaPartitioner &&
           isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  g.numSemaphoreCopies = g.numCopies;
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad))
      hasProducerLoad = true;
  });
  if (synchronized && g.isLocal() && !plannedCopy && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
  if (synchronized && g.isTmem())
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return success();
}

// Docs: sync-dag.md#edges-to-semaphores
//       sync-dag.md#region-flows
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
            isRegionFlowForComp(n))
          return true;
      }
      return false;
    };
    bool checked = head->parent->flow &&
                   head->parent->flow->threadsToken() && tokenConsumed();
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

static LogicalResult verifyPointOfUseFlow(GroupDag &g, Node *F,
                                          const RegionFlow &c) {
  auto errorAt = [&](Node *node) {
    return semaError(node && node->op ? node->op : F->op);
  };
  Node *pointAcquire = exitsThroughRegion(c) ? c.entryAcquire : c.exit();
  Node *recurrenceAcquire = pointAcquire;
  if (pointAcquire->next != c.firstToucher)
    return errorAt(F) << "point-of-use acquire is not adjacent to its first buffer use";
  for (Node *m = F->children[0]; m && m != pointAcquire; m = m->next) {
    if (isTokenEvent(g, m))
      return errorAt(m) << "token event before point-of-use acquire";
  }
  if (exitsThroughRegion(c) && !isTransparentFlow(c.exit(), c.owner))
    return errorAt(c.exit())
           << "non-transparent region supplies a point-of-use flow";
  if (exitsThroughRegion(c)) {
    Node *closing = c.exit()->next;
    if (!closing || closing->kind != Node::Release ||
        closing->sema != c.entryAcquire->sema || closing != c.closingRelease)
      return errorAt(F) << "nestedExit point-of-use lacks closing release after region result";
  } else if (!c.closingRelease || std::max(1u, c.closingRelease->count) != 1) {
    Operation *op = c.closingRelease && c.closingRelease->op ? c.closingRelease->op : F->op;
    return semaError(op) << "point-of-use flow requires exactly one closing release";
  }
  if (c.postLoopAcquire) {
    Node *postLoopAcquire = c.postLoopAcquire;
    if (!postLoopAcquire || postLoopAcquire->kind != Node::Acquire ||
        !postLoopAcquire->postLoopAcquire || postLoopAcquire->parent != F->parent ||
        postLoopAcquire->sema != recurrenceAcquire->sema ||
        postLoopAcquire->count != recurrenceAcquire->count ||
        !sameOwner(postLoopAcquire->owner, c.owner))
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
  }
  if (c.bridgeAcquire) {
    Node *bridgeAcquire = c.bridgeAcquire;
    Node *bridgeRelease = c.bridgeRelease;
    if (!getSema(g, recurrenceAcquire).isEntry || !bridgeAcquire ||
        !bridgeRelease || bridgeAcquire->kind != Node::Acquire ||
        bridgeRelease->kind != Node::Release ||
        bridgeRelease->sema != recurrenceAcquire->sema ||
        bridgeRelease->count != recurrenceAcquire->count ||
        !sameOwner(bridgeRelease->owner, c.owner) || bridgeRelease->sat)
      return errorAt(F) << "malformed capability bridge";
    bool prefixCut = bridgeAcquire == c.entryAcquire &&
                     c.entryAcquire->sema == recurrenceAcquire->sema;
    if (prefixCut) {
      if (bridgeRelease->parent != F->parent || bridgeRelease->next != F ||
          (c.owner && !nodeReusesToken(bridgeRelease, c.owner)))
        return errorAt(F) << "malformed prefix capability cut";
    } else if (!precedesInChain(F, bridgeAcquire) ||
               !precedesInChain(bridgeAcquire, bridgeRelease)) {
      return errorAt(F) << "capability bridge is not after its loop";
    }
  } else if (c.bridgeAcquire || c.bridgeRelease) {
    return errorAt(F) << "unexpected outer-to-local semaphore bridge";
  }
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (!g.semas.empty() && !g.root->children.empty())
    if (failed(verifyTokenLocality(g, g.root->children[0])))
      return failure();
  auto verifyFlow = [&](Node *n) -> LogicalResult {
    if (!n->isRegion() || !n->flow)
      return success();
    const RegionFlow &c = *n->flow;
    if (!c.ownersCompatible)
      return semaError(n->op)
             << "region paths export capabilities owned by different partitions";
    if (c.exits.size() != n->children.size() ||
        c.exitTokens.size() != n->children.size())
      return semaError(n->op) << "region flow does not cover every exit path";
    for (auto [final, source] : llvm::zip(c.exits, c.exitTokens)) {
      RegionFlow::ExitToken expected =
          !final ? RegionFlow::ExitToken::INPUT
                 : exportsToken(final) ? RegionFlow::ExitToken::CURRENT
                                       : RegionFlow::ExitToken::NONE;
      if (source != expected)
        return semaError(n->op) << "stale region exit capability source";
      if (c.threadsToken() && source == RegionFlow::ExitToken::NONE)
        return semaError(n->op)
               << "carried region path exports no SSA capability";
    }
    if (c.threadsToken() &&
        c.semaTransfer == RegionFlow::SemaTransfer::INVALID)
      return semaError(n->op)
             << "region paths export incompatible semaphore capabilities";
    if (c.threadsToken())
      return success();
    if (c.isPointOfUse()) {
      if (!c.exit() || !c.firstToucher || !c.entryAcquire)
        return semaError(n->op) << "malformed point-of-use flow crossing";
      if (!exitsThroughRegion(c) && c.exit()->kind != Node::Acquire)
        return semaError(n->op) << "point-of-use flow without acquire regain";
      if (exitsThroughRegion(c) && !c.exit()->isRegion())
        return semaError(n->op)
               << "nestedExit point-of-use without region regain";
      return verifyPointOfUseFlow(g, n, c);
    }
    if (c.exits.empty() || !c.exits[0] ||
        c.exits[0]->kind != Node::For || c.firstToucher || c.entryAcquire)
      return semaError(n->op) << "malformed child-owned region flow";
    const RegionFlow *child = findFlow(c.exits[0]);
    if (!child || child->threadsToken())
      return semaError(n->op) << "child-owned flow without native child";
    return success();
  };
  if (!g.root->children.empty())
    if (failed(forEachNodeChecked(g.root->children[0], verifyFlow)))
      return failure();
  SmallVector<unsigned> releaseArrivals(g.semas.size(), 0);
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifySemaNode = [&](Node *n) -> LogicalResult {
      if (n->kind == Node::Release) {
        if (n->payloads.empty())
          return semaError(g.root->op) << "release without payload record";
        releaseArrivals[n->sema] +=
            std::max(1u, n->count) *
            std::max(1u, static_cast<unsigned>(n->payloads.size()));
        if (n->sat) {
          if (n->sat->parent != n->parent)
            return semaError(g.root->op) << "release and its acquire are in different chains";
          if (!precedesInChain(n, n->sat) && !getSema(g, n).isEntry)
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
    if (releaseArrivals[sid] != s.expectedArrivals)
      return semaError(g.root->op) << "semaphore "
             << s.name << " has " << releaseArrivals[sid]
             << " release arrivals, expected " << s.expectedArrivals;
  }
  return success();
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;

struct ScheduleEdge {
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
};

// A handoff constrains the whole-iteration skew between its two owners. If a
// release by `producerOwner` at stage P supplies an acquire by `consumerOwner`
// at stage C and loop distance D, their owner offsets must satisfy
//
//   offset[consumerOwner] >= offset[producerOwner] + P - C - D.
//
// One positive edge is legal backpressure: the consumer can wait while the
// producer advances. Only a positive-weight owner cycle is impossible.
struct OwnerScheduleConstraint {
  Owner producerOwner;
  Owner consumerOwner;
  Operation *producer = nullptr;
  Operation *consumer = nullptr;
  int64_t producerStage = 0;
  int64_t consumerStage = 0;
  int64_t distance = 0;

  int64_t requiredDelay() const {
    return producerStage - consumerStage - distance;
  }
};

struct LoopScheduleModel {
  SmallVector<ScheduleEdge, 4> clusterEdges;
  SmallVector<OwnerScheduleConstraint, 4> ownerConstraints;
};

struct SlotSchedule {
  int64_t advancesPerIteration = 0;
  DenseMap<Node *, int64_t> ordinalByAccess;
  bool complete = true;
};

// A scheduled region is one atomic event in its enclosing loop.  ACCESS-DAG
// has already summarized its children in pieceInfo, so slot replay uses that
// summary rather than descending into the region.
static Effect slotEventEffect(const Node *n) {
  Effect effect = Effect::R;
  if (n->kind == Node::Access)
    for (const Touch &touch : n->touches)
      effect = joinEffect(effect, touch.effect);
  else
    for (const auto &[_, info] : n->pieceInfo)
      effect = joinEffect(effect, info.effect);
  return effect;
}

static bool isSlotEvent(const Node *n) {
  return n->kind == Node::Access ||
         (n->isRegion() && !n->pieceInfo.empty());
}
static bool isLinkedInChain(const Node *n) {
  return n->parent &&
         (n->prev || n->next || llvm::is_contained(n->parent->children, n));
}
static bool isDirectLoopNode(const Node *n) {
  return isLinkedInChain(n) && n->parent->kind == Node::For;
}

// Doc: sync-dag.md#authored-buffer-stage-offsets
static std::optional<int64_t> recordPhysicalStage(
    SlotSchedule &result, DenseMap<GroupDag *, int64_t> &lastProducedOrdinal,
    int64_t &cursorOrdinal, GroupDag *group, Node *access, unsigned advances) {
  int64_t requiredOrdinal;
  if (slotEventEffect(access) == Effect::W) {
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
  return requiredOrdinal;
}

// Doc: sync-dag.md#circular-groups
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
      if (g->pieceTable.members.size() != 1)
        return semaError(g->root->op)
               << "malformed circular local logical group";
      const Member &member = g->pieceTable.members.front();
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
      bool produces = slotEventEffect(event.access) == Effect::W;
      auto stage =
          recordPhysicalStage(slots, lastProducedOrdinal, cursorOrdinal,
                              event.group, event.access, produces);
      if (produces) {
        assert(stage);
        if (member.circularStart != *stage % numCopies)
          return semaError(member.allocOp)
                 << "circular producer order expects buffer.start "
                 << *stage % numCopies << ", got " << member.circularStart;
      } else if (!stage)
        return semaError(member.allocOp)
               << "circular consumer appears before producer";
      event.access->bufferStageOffset = *stage - cursorOrdinal;
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

// Doc: sync-dag.md#pipeline-schedule
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

// Doc: sync-dag.md#authored-buffer-stage-offsets
static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet,
                                        scf::ForOp loop) {
  struct Event {
    GroupDag *group = nullptr;
    Node *node = nullptr;
  };
  SlotSchedule result;
  SmallVector<Event, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (GroupDag *group : physicalSet) {
    // Select only nodes in this loop's direct SYNC-DAG chain.  In particular,
    // do not recurse into scf.if/scf.for children: the scheduled region op is
    // the enclosing loop's event and already carries their ownership/effect
    // summary.
    for (const std::unique_ptr<Node> &storage : group->nodes) {
      Node *node = storage.get();
      if (!isLinkedInChain(node) || node->parent->kind != Node::For ||
          node->parent->op != loop.getOperation())
        continue;
      if (node->kind == Node::Acquire && node->scheduleAnchor &&
          isSlotEvent(node->scheduleAnchor) &&
          slotEventEffect(node->scheduleAnchor) == Effect::W) {
        ++advancesByAccess[node->scheduleAnchor];
        continue;
      }
      if (!isSlotEvent(node) || !node->op)
        continue;
      events.push_back(Event{group, node});
    }
  }
  llvm::stable_sort(events, [](const Event &lhs, const Event &rhs) {
    return lhs.node->op != rhs.node->op &&
           lhs.node->op->isBeforeInBlock(rhs.node->op);
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
// Doc: sync-dag.md#finalizing-one-handoff
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
// Doc: sync-dag.md#non-circular-alias-handoffs
static bool isAliasedMultibufferedGroup(const GroupDag &group) {
  if (group.isCircular() ||
      group.pieceTable.members.size() < 2 || group.numSemaphoreCopies <= 1)
    return false;
  // Planner-authored copies give every member of this buffer.id group one
  // shared physical stage domain, regardless of view geometry.
  bool authored = group.numCopies > 1 &&
                  llvm::all_of(group.pieceTable.members, [](const Member &m) {
                    return m.allocOp->hasAttr(kBufferCopyAttrName);
                  });
  if (authored || group.numSemaphoreCopies <= group.numCopies)
    return authored;
  // Keep the exact-alias fallback for separately staged semaphores.
  const Member &first = group.pieceTable.members.front();
  return llvm::all_of(group.pieceTable.members, [&](const Member &member) {
    return member.offset == first.offset && member.extent == first.extent && member.type == first.type;
  });
}

static LogicalResult assignAliasedHandoffStageOffsets(GroupDag &group) {
  if (!isAliasedMultibufferedGroup(group))
    return success();
  DenseMap<Operation *, SlotSchedule> slotsByLoop;
  bool hasShiftedRelease = false;
  auto assignRelease = [&](Node *release) -> LogicalResult {
    if (release->kind != Node::Release || !release->sat)
      return success();
    Node *producer = release->scheduleAnchor;
    Node *consumer = release->sat->scheduleAnchor;
    if (!producer || !consumer || !isSlotEvent(producer) ||
        !isSlotEvent(consumer) || producer->parent != consumer->parent ||
        !producer->parent || producer->parent->kind != Node::For)
      return semaError(producer && producer->op ? producer->op : group.root->op)
             << "multibuffered alias handoff requires direct scheduled events "
                "in one loop body";
    auto loop = cast<scf::ForOp>(producer->parent->op);
    auto [it, inserted] = slotsByLoop.try_emplace(loop.getOperation());
    if (inserted)
      it->second = computeSlotSchedule(ArrayRef<GroupDag *>{&group}, loop);
    const SlotSchedule &slots = it->second;
    auto producerIt = slots.ordinalByAccess.find(producer);
    auto consumerIt = slots.ordinalByAccess.find(consumer);
    if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
        consumerIt == slots.ordinalByAccess.end() ||
        slots.advancesPerIteration <= 0)
      return semaError(producer->op)
             << "cannot derive multibuffered alias handoff slots";
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
    return success();
  };
  // The flat node store avoids descending through scheduled region events.
  for (const auto &node : group.nodes)
    if (isDirectLoopNode(node.get()) && failed(assignRelease(node.get())))
      return failure();
  for (const auto &node : group.nodes) {
    if (!isDirectLoopNode(node.get()))
      continue;
    if (!hasShiftedRelease && node->kind == Node::Release)
      node->stageOffset.reset();
    if (hasShiftedRelease && node->kind == Node::Acquire)
      node->stageOffset = 0;
  }
  return success();
}

// Doc: sync-dag.md#finalizing-one-handoff
// Solve the owner-skew difference constraints for one scheduled loop. Longest
// path relaxation converges exactly when every owner cycle has non-positive
// total required delay. A change on the |V|th pass proves a positive cycle.
//
// Edges on a zero-delay cycle execute in the same expanded pipeline wave after
// applying the solved owner offsets. They need the same loop.cluster ordering
// repair as a directly zero-delay handoff. Direct zero-delay edges retain the
// existing repair even when they are not part of a cycle.
static LogicalResult solveOwnerScheduleConstraints(LoopScheduleModel &model) {
  if (model.ownerConstraints.empty())
    return success();

  DenseMap<int64_t, unsigned> vertexByOwner;
  auto getVertex = [&](const Owner &owner) {
    int64_t key = ownerKey(owner);
    auto [it, inserted] =
        vertexByOwner.try_emplace(key, vertexByOwner.size());
    return it->second;
  };
  for (const OwnerScheduleConstraint &constraint : model.ownerConstraints) {
    getVertex(constraint.producerOwner);
    getVertex(constraint.consumerOwner);
  }

  const unsigned numVertices = vertexByOwner.size();
  SmallVector<int64_t, 8> offset(numVertices, 0);
  SmallVector<std::optional<unsigned>, 8> predecessor(numVertices);
  std::optional<unsigned> lastUpdated;
  for (unsigned iteration = 0; iteration < numVertices; ++iteration) {
    lastUpdated.reset();
    for (auto [edgeIndex, constraint] :
         llvm::enumerate(model.ownerConstraints)) {
      unsigned producer = getVertex(constraint.producerOwner);
      unsigned consumer = getVertex(constraint.consumerOwner);
      int64_t candidate =
          offset[producer] + constraint.requiredDelay();
      if (offset[consumer] >= candidate)
        continue;
      offset[consumer] = candidate;
      predecessor[consumer] = edgeIndex;
      lastUpdated = edgeIndex;
    }
    if (!lastUpdated)
      break;
  }

  if (lastUpdated) {
    unsigned vertex = getVertex(
        model.ownerConstraints[*lastUpdated].consumerOwner);
    for (unsigned i = 0; i < numVertices; ++i) {
      if (!predecessor[vertex])
        break;
      vertex = getVertex(
          model.ownerConstraints[*predecessor[vertex]].producerOwner);
    }

    SmallVector<unsigned, 4> cycle;
    unsigned cycleStart = vertex;
    do {
      if (!predecessor[vertex]) {
        cycle.clear();
        break;
      }
      unsigned edgeIndex = *predecessor[vertex];
      cycle.push_back(edgeIndex);
      vertex = getVertex(
          model.ownerConstraints[edgeIndex].producerOwner);
    } while (vertex != cycleStart && cycle.size() <= numVertices);

    if (cycle.empty())
      cycle.push_back(*lastUpdated);
    int64_t cycleDelay = 0;
    for (unsigned edgeIndex : cycle)
      cycleDelay += model.ownerConstraints[edgeIndex].requiredDelay();

    const OwnerScheduleConstraint &first =
        model.ownerConstraints[cycle.front()];
    InFlightDiagnostic diag = semaError(first.producer)
                              << "fixed loop.stage assignments form an "
                                 "unsatisfiable semaphore handoff cycle";
    if (cycleDelay > 0)
      diag << " (cycle requires " << cycleDelay
           << " additional pipeline iteration"
           << (cycleDelay == 1 ? "" : "s") << ")";
    for (unsigned edgeIndex : llvm::reverse(cycle)) {
      const OwnerScheduleConstraint &constraint =
          model.ownerConstraints[edgeIndex];
      diag.attachNote(constraint.consumer->getLoc())
          << "handoff "
          << ownerStr(constraint.producer, constraint.producerOwner) << " -> "
          << ownerStr(constraint.consumer, constraint.consumerOwner)
          << " has producer loop.stage " << constraint.producerStage
          << ", consumer loop.stage " << constraint.consumerStage
          << ", loop-carried dependency distance " << constraint.distance
          << ", and required delay " << constraint.requiredDelay();
    }
    return failure();
  }

  auto isTight = [&](const OwnerScheduleConstraint &constraint) {
    return offset[getVertex(constraint.consumerOwner)] ==
           offset[getVertex(constraint.producerOwner)] +
               constraint.requiredDelay();
  };
  auto hasTightPath = [&](unsigned from, unsigned to) {
    SmallVector<unsigned, 8> stack{from};
    SmallVector<bool, 8> seen(numVertices, false);
    while (!stack.empty()) {
      unsigned vertex = stack.pop_back_val();
      if (vertex == to)
        return true;
      if (seen[vertex])
        continue;
      seen[vertex] = true;
      for (const OwnerScheduleConstraint &constraint : model.ownerConstraints)
        if (isTight(constraint) &&
            getVertex(constraint.producerOwner) == vertex)
          stack.push_back(getVertex(constraint.consumerOwner));
    }
    return false;
  };

  for (const OwnerScheduleConstraint &constraint : model.ownerConstraints) {
    bool directlySameWave = constraint.requiredDelay() == 0;
    bool onZeroDelayCycle = isTight(constraint) &&
                            hasTightPath(getVertex(constraint.consumerOwner),
                                         getVertex(constraint.producerOwner));
    if (directlySameWave || onZeroDelayCycle)
      model.clusterEdges.push_back(
          ScheduleEdge{constraint.producer, constraint.consumer});
  }
  return success();
}

// Doc: sync-dag.md#finalizing-one-handoff
static LogicalResult addSyncScheduleEdges(MutableArrayRef<GroupDag> groups,
    llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
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
    auto addReleaseEdge = [&](Node *release) -> LogicalResult {
      if (release->kind != Node::Release || !release->sat)
        return success();
      Node *acquire = release->sat;
      Operation *producer =
          realScheduleAnchor(release->scheduleAnchor, /*producer=*/true);
      Operation *consumer =
          realScheduleAnchor(acquire->scheduleAnchor, /*producer=*/false);
      if (!producer || !consumer)
        return success();
      std::optional<LoopAnchorPair> anchors =
          findCommonScheduledLoop(producer, consumer);
      if (!anchors || anchors->producer == anchors->consumer)
        return success();
      gpu::StageCluster producerSchedule =
          gpu::getStageCluster(anchors->producer);
      gpu::StageCluster consumerSchedule =
          gpu::getStageCluster(anchors->consumer);
      if (!producerSchedule || !consumerSchedule)
        return success();
      int64_t distance = 0;
      if (!precedesInChain(release, acquire)) {
        bool exactPlannedRecurrence =
            acquire->recurrenceDistance &&
            release->parent == acquire->parent && acquire->parent &&
            acquire->parent->op == anchors->loop.getOperation();
        if (exactPlannedRecurrence) {
          distance = *acquire->recurrenceDistance;
        } else {
          unsigned setIndex = setByGroup.lookup(&group);
          auto key = std::make_pair(setIndex, anchors->loop.getOperation());
          auto it = slotCache.find(key);
          if (it == slotCache.end())
            it = slotCache
                     .emplace(key, computeSlotSchedule(
                                       physicalSets[setIndex], anchors->loop))
                     .first;
          std::optional<int64_t> loopCarriedDistance =
              computeLoopCarriedDistance(
                  it->second, group.numSemaphoreCopies,
                  release->scheduleAnchor, acquire->scheduleAnchor);
          if (!loopCarriedDistance) {
            InFlightDiagnostic diag = semaError(anchors->producer)
                << "cannot determine loop-carried dependency distance for a "
                   "physical buffer slot";
            diag.attachNote(anchors->consumer->getLoc())
                << "next token ownership starts here";
            return failure();
          }
          distance = *loopCarriedDistance;
        }
      }
      modelsByLoop[anchors->loop.getOperation()]
          .ownerConstraints.push_back(OwnerScheduleConstraint{
              release->owner, acquire->owner, anchors->producer,
              anchors->consumer, producerSchedule->first,
              consumerSchedule->first, distance});
      return success();
    };
    for (const auto &node : group.nodes)
      if (isDirectLoopNode(node.get()) && failed(addReleaseEdge(node.get())))
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
// Doc: sync-dag.md#finalizing-one-handoff
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

static LogicalResult analyzeSyncSchedule(
    MutableArrayRef<GroupDag> groups,
    llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
  if (failed(assignCircularStageOffsets(groups)))
    return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(group)))
      return failure();
  if (failed(addSyncScheduleEdges(groups, modelsByLoop)))
    return failure();
  for (auto &[loopOp, model] : modelsByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    if (failed(solveOwnerScheduleConstraints(model)))
      return failure();
    if (model.clusterEdges.empty())
      continue;
    addSSAClusterConstraints(loop, model.clusterEdges);
    if (failed(legalizeLoopSchedule(loop, model.clusterEdges)))
      return failure();
  }
  return success();
}

static LogicalResult validateSyncSchedule(LogicalResult result,
                                          std::optional<Diagnostic> &error) {
  if (succeeded(result) || !error)
    return result;
  error->getLocation().getContext()->getDiagEngine().emit(std::move(*error));
  return failure();
}

LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  if (groups.empty())
    return success();
  llvm::MapVector<Operation *, LoopScheduleModel> modelsByLoop;
  std::optional<Diagnostic> scheduleError;
  LogicalResult result = success();
  {
    ScopedDiagnosticHandler defer(
        groups.front().root->op->getContext(),
        [&](Diagnostic &diag) { scheduleError.emplace(std::move(diag)); });
    result = analyzeSyncSchedule(groups, modelsByLoop);
  }
  // The sole rejection gate; analysis and its diagnostics remain available if
  // this policy check is temporarily disabled.
  if (failed(validateSyncSchedule(result, scheduleError)))
    return failure();
  for (GroupDag &g : groups) {
    if (g.root->children.empty())
      continue;
    ScheduleCache cache;
    assignSyncScheduleChain(g.root->children[0], cache);
  }
  return success();
}

// Doc: sync-dag.md#purpose
LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner,
                           int lowerSemaphoreNumStages, int &numTmemBlocks) {
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
    buildRegionFlows(g, g.root->children[0]);
    pruneDeadIfFlows(g, g.root->children[0], /*region=*/nullptr);
    if (failed(planRegionFlows(g, g.root->children[0])))
      return failure();
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
// Doc: sync-dag.md#notation
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
  if (n->flow)
    os << " thread{" << ownerStr(n->op, n->flow->owner) << "}";
  if (n->kind == Node::For && n->flow) {
    os << " holdrule{";
    const RegionFlow &c = *n->flow;
    if (c.mode == RegionFlow::Mode::CARRIED) {
        os << "gated";
        if (c.blocker == RegionFlow::Blocker::TRAILING_USE)
          os << "(trailing-use)";
        else if (c.blocker == RegionFlow::Blocker::RESULT_CONSUMED)
          os << "(result-consumed)";
    } else if (c.mode == RegionFlow::Mode::POINT_OF_USE) {
      os << "pointofuse->";
      if (c.firstToucher && c.firstToucher->op)
        os << c.firstToucher->op->getName().getStringRef();
      else
        os << "?";
    } else
      os << "passthrough-drop";
    if (exitsThroughRegion(c))
      os << ":nestedExit";
    if (c.postLoopAcquire)
      os << ":postLoopAcquire";
    if (c.bridgeRelease)
      os << ":entryBridge";
    os << "}";
  }
}
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g, const Node *exit, const Node *region,
                           unsigned chainIdx) {
  if (!region || !region->flow)
    return;
  os << " yield{";
  const RegionFlow &c = *region->flow;
  if (!c.threadsToken()) { // native/child-owned: no slot
    os << (c.isChildOwned() ? "drop" : "native");
  } else {
    Node *f = chainIdx < c.exits.size() ? c.exits[chainIdx] : nullptr;
    os << (f ? syncNodeLabel(g, f) : std::string("pass"));
  }
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
