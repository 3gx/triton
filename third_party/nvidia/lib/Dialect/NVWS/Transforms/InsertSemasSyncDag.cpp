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
    llvm::erase_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
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
      bool alreadyOrdered = llvm::is_contained(use.orderedBefore, ownerKey(owner));
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
                            node, piece.source.sourceOwner, owner, piece.source.payloads, {id}});
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
    return sameOwner(use.owner, owner) || llvm::is_contained(use.orderedBefore, ownerKey(owner));
  });
}
static bool nodeTouchesPiece(GroupDag &g, Node *node, PieceId piece) {
  if (node->kind == Node::Access)
    return touchesPiece(g, node, piece);
  return node->isRegion() && node->pieceInfo.count(piece);
}
static bool pieceTouchedAfter(GroupDag &g, Node *region, PieceId piece) {
  for (Node *scope = region; scope && scope->kind != Node::Func; scope = scope->parent)
    for (Node *node = scope->next; node; node = node->next)
      if (nodeTouchesPiece(g, node, piece))
        return true;
  return false;
}

// Doc: sync-dag.md#the-walk-accesses-to-edges
class ChainWalker {
public:
  ChainWalker(GroupDag &group, ChainState &state, EdgeList &edges, bool underFor)
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
    bool ownerDiffers = last && node->owner && !sameOwner(last->owner, node->owner);
    bool canReuse = tokens.find(node->owner) && llvm::all_of(effects, [&](const auto &item) {
                      return canReuseTokenForPiece(state, item.first, node->owner, item.second);
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
    if (group.pieceTable.members.size() > 1 && canReuse && synchronousWrite) {
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
      applyTouch(state[id], id, node->owner, effect, node, payloads, edges, /*wsAdopt=*/false);
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
    bool wsAdopt = node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [id, info] : infos)
      applyTouch(state[id], id, info.owner, info.effect, node, none, edges, wsAdopt);
    ExitFacts returned;
    bool firstChild = true;
    for (Node *childHead : node->children) {
      ChainState child;
      for (auto [id, info] : sortedPieceInfo(childHead)) {
        auto before = incoming.find(id);
        Owner producer = before == incoming.end() ? info.owner : before->second.producer;
        Payloads seed{AsyncOp::NONE};
        if (before != incoming.end() && sameOwner(info.owner, before->second.producer))
          seed = before->second.payloads;
        child[id].startVersion(producer, info.owner, childHead, seed);
      }
      ChainWalker nested(group, child, edges, underFor || node->kind == Node::For);
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
      if (state[id].source.node == node && !facts->second.payloads.empty())
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
using Snapshots = DenseMap<Node *, SyncVec>;
using Positions = DenseMap<Node *, unsigned>;
static void recordTokenOwner(SmallVectorImpl<int64_t> &owners, int64_t owner) {
  llvm::erase(owners, owner);
  owners.push_back(owner);
}

struct KnownOrder {
  std::map<int64_t, SyncVec> behind;
  void apply(const EdgeRec &edge, unsigned sourceIdx, const Snapshots &snapshots) {
    SyncVec &known = behind[ownerKey(edge.dstOwner)];
    if (auto it = snapshots.find(edge.src); it != snapshots.end())
      for (auto [owner, idx] : it->second)
        known[owner] = std::max(known[owner], idx);
    auto &source = known[ownerKey(edge.srcOwner)];
    source = std::max(source, sourceIdx);
  }
  void record(Node *node, unsigned position, Snapshots &snapshots) {
    int64_t owner = ownerKey(node->owner);
    behind[owner][owner] = position;
    snapshots[node] = behind[owner];
  }
  bool covers(const EdgeRec &edge, unsigned sourceIdx) const {
    auto known = behind.find(ownerKey(edge.dstOwner));
    if (known == behind.end())
      return false;
    auto source = known->second.find(ownerKey(edge.srcOwner));
    return source != known->second.end() && source->second >= sourceIdx;
  }
};

static bool isLoopClose(const EdgeRec &edge) {
  return edge.dst->kind == Node::Exit && edge.src->kind == Node::Access &&
         edge.srcOwner && edge.dstOwner;
}
static EdgeBuckets collectEdges(const Positions &positions, ArrayRef<EdgeRec> edges,
                                const std::vector<bool> &drop, SmallVectorImpl<unsigned> &closes) {
  EdgeBuckets buckets;
  for (auto [i, edge] : llvm::enumerate(edges)) {
    if (drop[i] || !positions.contains(edge.src) || !positions.contains(edge.dst))
      continue;
    buckets[edge.dst].push_back(i);
    if (isLoopClose(edge))
      closes.push_back(i);
  }
  for (auto &bucket : buckets)
    llvm::stable_sort(bucket.second, [&](unsigned a, unsigned b) {
      return positions.lookup(edges[a].src) > positions.lookup(edges[b].src);
    });
  return buckets;
}
// Doc: sync-dag.md#1-implied-ordering-reduceedges. Drops use only kept edges.
static void reduceStraightEdges(Node *head, const Positions &positions,
                                ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst,
                                std::vector<bool> &drop) {
  KnownOrder order;
  Snapshots snapshots;
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
        int64_t dk = ownerKey(e.dstOwner);
        unsigned srcIdx = positions.lookup(e.src);
        bool hasToken = !tokenOwners.empty() && tokenOwners.back() == dk;
        if (order.covers(e, srcIdx) && hasToken && e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        recordTokenOwner(tokenOwners, dk); // Kept acquire supplies Q's token.
        order.apply(e, srcIdx, snapshots);
      }
    if (n->kind != Node::Exit && n->owner)
      order.record(n, positions.lookup(n), snapshots);
  }
}
static void reduceLoopCloses(GroupDag &g, Node *head, const Positions &positions,
                             ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst,
                             ArrayRef<unsigned> closes, std::vector<bool> &drop) {
  if (closes.empty())
    return;
  constexpr unsigned kPass2 = 1u << 20;
  KnownOrder order;
  Snapshots snap1, snap2;
  Owner firstAccessOwner;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        if (!drop[ei] && !isLoopClose(edges[ei]))
          order.apply(edges[ei], positions.lookup(edges[ei].src), snap1);
    if (n->owner && n->kind == Node::Access) {
      if (!firstAccessOwner)
        firstAccessOwner = n->owner;
      order.record(n, positions.lookup(n), snap1);
    }
  }
  if (!firstAccessOwner)
    return;
  EdgeBuckets closeAt;
  for (unsigned ei : closes) {
    const EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    llvm::SmallDenseSet<PieceId> seen;
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Access && n->owner && sameOwner(n->owner, e.dstOwner))
        for (const Touch &touch : n->touches)
          for (PieceId pc : g.pieceTable.footprint[touch.member])
            if (llvm::is_contained(e.pieces, pc) && seen.insert(pc).second)
              latest = n;
    if (latest)
      closeAt[latest].push_back(ei);
  }
  DenseSet<int64_t> tokenAvailable;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        if (drop[ei] || isLoopClose(edges[ei]))
          continue;
        order.apply(edges[ei], kPass2 + positions.lookup(edges[ei].src), snap2);
        tokenAvailable.insert(ownerKey(edges[ei].dstOwner));
      }
    auto ct = closeAt.find(n);
    if (ct != closeAt.end())
      for (unsigned ei : ct->second) {
        const EdgeRec &e = edges[ei];
        int64_t dk = ownerKey(e.dstOwner);
        if (order.covers(e, positions.lookup(e.src)) && tokenAvailable.contains(dk) &&
            !sameOwner(e.dstOwner, firstAccessOwner)) {
          drop[ei] = true;
          continue;
        }
        order.apply(e, positions.lookup(e.src), snap1);
      }
    if (n->owner && n->kind == Node::Access)
      order.record(n, kPass2 + positions.lookup(n), snap2);
  }
}
static void reduceChain(GroupDag &g, Node *head, ArrayRef<EdgeRec> edges, std::vector<bool> &drop) {
  Positions positions;
  unsigned position = 0;
  for (Node *n = head; n; n = n->next)
    positions[n] = position++;
  SmallVector<unsigned, 4> closes;
  EdgeBuckets atDst = collectEdges(positions, edges, drop, closes);
  reduceStraightEdges(head, positions, edges, atDst, drop);
  if (head->parent && head->parent->kind == Node::For)
    reduceLoopCloses(g, head, positions, edges, atDst, closes, drop);
  for (Node *n = head; n; n = n->next)
    if (n->isRegion())
      for (Node *child : n->children)
        reduceChain(g, child, edges, drop);
}
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (g.root->children.empty() || edges.empty())
    return;
  std::vector<bool> drop(edges.size(), false);
  reduceChain(g, g.root->children[0], edges, drop);
  unsigned i = 0;
  llvm::erase_if(edges, [&](const EdgeRec &) { return drop[i++]; });
}
} // namespace

static bool precedesInChain(Node *before, Node *after) {
  for (Node *next = before->next; next; next = next->next)
    if (next == after)
      return true;
  return false;
}
static unsigned arrivalContribution(ArrayRef<EdgeRec> edges) {
  return std::accumulate(
      edges.begin(), edges.end(), 0u, [](unsigned total, const EdgeRec &edge) {
        return total +
               std::max(1u, static_cast<unsigned>(edge.payloads.size()));
      });
}

// Consumes the already-reduced edge list; edge reduction happens in
// buildSyncDag before any backing or protocol decision.
static LogicalResult buildEdgesAndSemas(GroupDag &g, SmallVector<EdgeRec> &edges) {
  struct Handoff {
    Node *dst = nullptr;
    Owner owner;
    SmallVector<EdgeRec, 2> incoming;
    std::optional<SemaId> sema;
  };
  llvm::MapVector<std::tuple<Node *, int64_t>, unsigned> dstIndex;
  SmallVector<Handoff> handoffs;
  for (const EdgeRec &edge : edges) {
    auto key = std::make_tuple(edge.dst, ownerKey(edge.dstOwner));
    auto [it, inserted] = dstIndex.try_emplace(key, handoffs.size());
    if (inserted)
      handoffs.push_back(Handoff{edge.dst, edge.dstOwner});
    Handoff &handoff = handoffs[it->second];
    auto source = llvm::find_if(handoff.incoming, [&](const EdgeRec &prior) {
      return ownerKey(prior.srcOwner) == ownerKey(edge.srcOwner);
    });
    if (source == handoff.incoming.end()) {
      handoff.incoming.push_back(edge);
      llvm::sort(handoff.incoming.back().payloads);
    } else {
      if (precedesInChain(source->src, edge.src))
        source->src = edge.src;
      unionPayloads(source->payloads, edge.payloads);
    }
  }
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
    Handoff *source = &handoff;
    if (handoff.dst->kind == Node::For)
      for (Node *n = handoff.dst->children[0]; n; n = n->next)
        if (auto it = dstIndex.find(std::make_tuple(n, ownerKey(handoff.owner)));
            it != dstIndex.end())
          source = &handoffs[it->second];
    if (!source->sema)
      createSema(*source);
    handoff.sema = source->sema;
  }
  DenseMap<Node *, Node *> lastAfter; // release insertion cursor per source
  for (Handoff &handoff : handoffs) {
    Sema &sema = g.semas[*handoff.sema];
    unsigned sources = handoff.incoming.size();
    unsigned arrivals = arrivalContribution(handoff.incoming);
    if (arrivals != sema.count && !(sources == 1 && arrivals == 1))
      return semaError(handoff.dst->op ? handoff.dst->op : g.root->op)
             << "destination group with " << arrivals << " arrival contributions from " << sources
             << " sources cannot meet semaphore " << sema.name << " pending count " << sema.count;
    // Only a lone generic release can be scaled for a reused semaphore.
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
      for (Node *r = head; r; r = r->next)
        if (r->kind == Node::Access && r->owner) {
          if (!firstAccessOwner)
            firstAccessOwner = r->owner;
          if (!firstTouch && sameOwner(r->owner, acquire->owner))
            firstTouch = r;
        }
      if (firstAccessOwner && !sameOwner(firstAccessOwner, acquire->owner) && firstTouch) {
        destination = firstTouch;
        sema.isEntry = true; // initially released; no pre-loop entry instance
        sema.entryTokenOwner = acquire->owner;
      }
    }
    acquire->scheduleAnchor = destination;
    spliceBefore(acquire, destination);
    sema.expectedArrivals += arrivals * releaseCount;
    for (const EdgeRec &edge : handoff.incoming) {
      Node *release = newProtocolNode(g, Node::Release, edge.src->parent, edge.srcOwner,
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
    if (node->kind == Node::For && (regain = lastAcquireOfCompInChain(node->children[0])))
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
  Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent, std::nullopt, sid, 1);
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
  return success();
}
// Doc: sync-dag.md#the-decision-per-region
using SemaExpr = SemaTransfer;
static SemaExpr inputSema() { return {true, true, std::nullopt}; }
static SemaExpr concreteSema(SemaId sema) { return {true, false, sema}; }
static SemaExpr joinSemaExpr(SemaExpr a, SemaExpr b) {
  if (!a.valid || !b.valid)
    return {};
  bool passesInput = a.passesInput || b.passesInput;
  return {true, passesInput, a.concrete ? a.concrete : b.concrete};
}
static SemaExpr applyFlow(const RegionFlow &flow, SemaExpr input) {
  const SemaExpr &transfer = flow.semaTransfer;
  if (!transfer.valid || !input.valid)
    return {};
  return transfer.passesInput ? input : transfer;
}
static std::optional<SemaId> resolveFlowSema(const RegionFlow &flow, SemaId input) {
  return flow.semaTransfer.valid ? std::optional<SemaId>(input) : std::nullopt;
}

struct ChainBoundary {
  Node *final = nullptr;
  Owner owner;
  SemaExpr sema = inputSema();
  bool eventAfterFinal = false;
};
static ChainBoundary summarizeChainBoundary(GroupDag &g, Node *head) {
  ChainBoundary result;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire) {
      result.final = n;
      result.owner = n->owner;
      result.sema = concreteSema(n->sema);
      result.eventAfterFinal = false;
    } else if (n->isRegion() && n->flow) {
      result.final = n;
      result.owner = n->flow->owner;
      result.sema = applyFlow(*n->flow, result.sema);
      result.eventAfterFinal = false;
    } else if (n->isProtocol() || nodeInvolvesComp(g, n)) {
      result.eventAfterFinal = true;
    }
  }
  return result;
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
  bool valid = true;
  bool usesInput = true, hasFallback = false;
  gpu::StageCluster schedule;
};
static CompletionExpr concreteCompletion(gpu::StageCluster schedule) {
  return {true, false, true, schedule};
}
static CompletionExpr applyCompletion(const RegionFlow &flow, CompletionExpr input) {
  if (!input.valid || !flow.completionUniform)
    return {false};
  if (!flow.completionUsesInput)
    return concreteCompletion(flow.completionSchedule);
  if (flow.completionHasFallback && input.hasFallback && flow.completionSchedule != input.schedule)
    return {false};
  return {true, input.usesInput,
          flow.completionHasFallback || input.hasFallback,
          input.hasFallback ? input.schedule : flow.completionSchedule};
}
static CompletionExpr joinCompletion(CompletionExpr a, CompletionExpr b) {
  if (!a.valid || !b.valid || (a.hasFallback && b.hasFallback && a.schedule != b.schedule))
    return {false};
  return {true, a.usesInput || b.usesInput,
          a.hasFallback || b.hasFallback, a.hasFallback ? a.schedule : b.schedule};
}
static CompletionExpr completionAfterChain(Node *head, const Owner &owner,
                                           CompletionExpr state = {}) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access && sameOwner(n->owner, owner)) {
      Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
      state = concreteCompletion(gpu::getStageCluster(completion));
      continue;
    }
    if (!n->isRegion() || !n->flow || (n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op)))
      continue;
    const RegionFlow &nested = *n->flow;
    if (sameOwner(nested.owner, owner))
      state = applyCompletion(nested, state);
  }
  return state;
}
static bool summarizeRegionFlow(GroupDag &g, Node *region, RegionFlow &crossing) {
  crossing.exits.clear();
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
    ChainBoundary branch = summarizeChainBoundary(g, child);
    Node *final = branch.final;
    crossing.exits.push_back(final);
    joined = joined ? std::optional(joinSemaExpr(*joined, branch.sema))
                    : std::optional(branch.sema);
    if (branch.eventAfterFinal)
      transparent = false;
    if (!final) {
      if (entryOwner)
        joinOwner(*entryOwner);
      continue;
    }
    live = true;
    joinOwner(branch.owner);
  }
  bool hasImplicitInputPath =
      region->kind == Node::For ||
      (region->kind == Node::If && region->children.size() < 2);
  if (hasImplicitInputPath) {
    joined = joined ? std::optional<SemaExpr>(joinSemaExpr(*joined, inputSema()))
                    : std::optional<SemaExpr>(inputSema());
    if (entryOwner)
      joinOwner(*entryOwner);
  }
  SemaExpr output = joined.value_or(SemaExpr{});
  crossing.semaTransfer = output;
  crossing.transparent = transparent && entryOwner && crossing.owner &&
      sameOwner(*entryOwner, crossing.owner);
  CompletionExpr completion;
  bool firstCompletion = true;
  for (Node *child : region->children) {
    CompletionExpr branch = completionAfterChain(child, crossing.owner);
    completion = firstCompletion ? branch : joinCompletion(completion, branch);
    firstCompletion = false;
  }
  if (hasImplicitInputPath)
    completion = firstCompletion ? CompletionExpr{}
                                 : joinCompletion(completion, {});
  crossing.completionUniform = completion.valid;
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
    if (n->kind == Node::Release || (n->kind == Node::Access && nodeTouchesGroup(g, n)) ||
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
static bool exitsThroughRegion(const RegionFlow &flow) {
  return flow.exit() && flow.exit()->isRegion();
}
static bool allEnclosersCanDrop(const Node *F) {
  for (const Node *p = F; p; p = p->parent) {
    if (p->kind == Node::Func)
      return true;
    if (p != F && p->kind == Node::If)
      return false;
    if (p->kind == Node::For && p->op &&
        gpu::hasWarpSpecializeTag(p->op))
      return true;
  }
  return true;
}
static bool regionResultConsumedAfter(GroupDag &g, Node *region) {
  if (tokenUsedBeforeNextAcquire(g, region->next))
    return true;
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
static bool isTokenEvent(GroupDag &g, Node *n) {
  return n->isProtocol() || isAccessForComp(g, n) || isRegionFlowForComp(n);
}
static std::optional<SemaId>
resolveExitSema(Node *final, SemaId incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->sema;
  if (!final->isRegion())
    return std::nullopt;
  const RegionFlow *c = final->flow ? &*final->flow : nullptr;
  if (!c)
    return std::nullopt;
  return resolveFlowSema(*c, incoming);
}
static bool isTransparentFlow(Node *region, Owner owner) {
  const RegionFlow *rc = region->flow ? &*region->flow : nullptr;
  return rc && rc->transparent && sameOwner(rc->owner, owner);
}
static bool childOwnsToken(const Node *region) {
  const RegionFlow *child = region->flow ? &*region->flow : nullptr;
  return child && !child->threadsToken();
}
static bool hasTrailingCompUse(GroupDag &g, Node *regain) {
  for (Node *m = regain->next; m; m = m->next)
    if (isAccessForComp(g, m) || isRegionFlowForComp(m) || m->kind == Node::Release)
      return true;
  return false;
}
static Node *findBridgeAcquireAfter(GroupDag &g, Node *F, SemaId feedSema, Owner owner) {
  for (Node *m = F->next; m; m = m->next) {
    if (m->kind != Node::Acquire || m->sema != feedSema || !sameOwner(m->owner, owner))
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
static bool carrierPreservesFeed(Node *carrier, const Owner &owner,
                                 SemaId sema) {
  if (!carrier || carrier->kind != Node::For)
    return false;
  std::optional<Owner> boundaryOwner = uniformPieceOwner(carrier);
  const RegionFlow *crossing = carrier->flow ? &*carrier->flow : nullptr;
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
    if (!carrierPreservesFeed(carrier, owner, sema))
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
            !feedDominatesThroughCarriers(g, F, m, crossing.owner, m->sema))
          return std::nullopt;
        return InputCapability{m, crossedPrefixUse};
      }
      if (m->isRegion() && nodeInvolvesComp(g, m))
        return std::nullopt;
      if (!isTokenEvent(g, m))
        continue;
      if (m->kind == Node::Access && originalChain && sameOwner(m->owner, crossing.owner) &&
          (!m->reuseTokenOwner || *m->reuseTokenOwner == ownerKey(crossing.owner)) &&
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
matchDemandPrefix(GroupDag &g, Node *F, Owner tokenOwner, Node *regain, bool nestedExit) {
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
           auto copy = member.allocOp->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
           return copy && copy.getInt() == 1;
         });
}
// Select the loop transport directly from its opaque boundary transfer.  The
// child body has already been compiled; this function reads no child internals.
static void planLoopFlow(GroupDag &g, Node *F, RegionFlow &flow, bool &needsPostLoopAcquire) {
  auto carry = [&](RegionFlow::Blocker blocker = RegionFlow::Blocker::NONE) {
    flow.mode = RegionFlow::Mode::CARRIED;
    flow.blocker = blocker;
  };
  bool outputUsed = regionResultConsumedAfter(g, F);
  auto forOp = dyn_cast_or_null<scf::ForOp>(F->op);
  if (!forOp || !gpu::hasWarpSpecializeTag(outerWSLoop(forOp)) || !allEnclosersCanDrop(F))
    return carry();
  Node *regain = flow.exits.empty() ? nullptr : flow.exits[0];
  if (!regain || regain->postLoopAcquire)
    return carry();
  if (regain->kind == Node::For && childOwnsToken(regain)) {
    flow.mode = RegionFlow::Mode::CHILD_OWNS;
    return;
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
      nestedExit ? resolveExitSema(regain, entryAcquire->sema)
                 : std::optional<SemaId>(regain->sema);
  if (!recurrenceSema)
    return carry();
  needsPostLoopAcquire = outputUsed || feed->retainForPrefix;
  bool sameSemaphore = entryAcquire->sema == *recurrenceSema;
  if (sameSemaphore && needsPostLoopAcquire && nestedExit)
    return carry(RegionFlow::Blocker::RESULT_CONSUMED);
  Node *bridgeAcquire = feed->retainForPrefix ? entryAcquire : nullptr;
  if (feed->retainForPrefix && nestedExit)
    return carry();
  if (!sameSemaphore) {
    if (nestedExit || !needsPostLoopAcquire)
      return carry();
    const Sema &entrySema = getSema(g, entryAcquire);
    Owner entryOwner = entryAcquire->owner ? entryAcquire->owner : entrySema.entryTokenOwner;
    bridgeAcquire = findBridgeAcquireAfter(g, F, entryAcquire->sema, flow.owner);
    if (!sameOwner(entryOwner, flow.owner) || !bridgeAcquire)
      return carry();
  }
  std::optional<DemandPrefix> prefix = matchDemandPrefix(g, F, flow.owner, regain, nestedExit);
  if (!prefix || !prefixNodeIsSingleBufferView(F, prefix->firstToucher))
    return carry();
  if (needsPostLoopAcquire && !gpu::hasWarpSpecializeTag(forOp) &&
      sameOwner(prefix->firstToucher->owner, flow.owner)) {
    if (!flow.completionUniform)
      return carry();
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
  return;
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
static void lowerPointOfUse(GroupDag &g, Node *loop, RegionFlow &flow, bool needsPostLoopAcquire) {
  bool nestedExit = exitsThroughRegion(flow);
  Node *recurrenceAcquire = nestedExit ? flow.entryAcquire : flow.exit();
  if (needsPostLoopAcquire) {
    flow.postLoopAcquire = newProtocolNode(
        g, Node::Acquire, loop->parent, flow.owner, recurrenceAcquire->sema,
        recurrenceAcquire->count);
    flow.postLoopAcquire->postLoopAcquire = true;
    spliceAfter(flow.postLoopAcquire, loop);
    bool prefixCut = flow.bridgeAcquire == flow.entryAcquire &&
                     flow.entryAcquire->sema == recurrenceAcquire->sema;
    if (prefixCut) {
      Node *cut = newProtocolNode(g, Node::Release, loop->parent, flow.owner,
                                  recurrenceAcquire->sema, recurrenceAcquire->count);
      cut->payloads.push_back(AsyncOp::NONE);
      if (flow.owner)
        markTokenReuse(cut, flow.owner);
      spliceBefore(cut, loop);
      getSema(g, cut).expectedArrivals += cut->count;
      flow.bridgeRelease = cut;
      if (!nestedExit)
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
  if (nestedExit) {
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
  } else {
    detachFromChain(recurrenceAcquire);
    recurrenceAcquire->scheduleAnchor = flow.firstToucher;
    spliceBefore(recurrenceAcquire, flow.firstToucher);
    Sema &recurrenceSema = getSema(g, recurrenceAcquire);
    if (flow.bridgeAcquire) {
      recurrenceSema.isEntry = true;
      recurrenceSema.entryTokenOwner = flow.owner;
    } else {
      detachFromChain(flow.entryAcquire);
    }
  }
}
// Regions are lowered from right to left and inner to outer.  A later sibling
// therefore materializes its boundary supply before an earlier sibling asks
// whether its result is live.  Child summaries are refreshed once after their
// final placement; no mutation-driven global reclassification is required.
static void planRegionFlows(GroupDag &g, Node *head) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes)) {
    if (!n->isRegion())
      continue;
    for (Node *child : n->children)
      planRegionFlows(g, child);
    if (n->flow)
      summarizeRegionFlow(g, n, *n->flow);
    if (n->kind != Node::For || !n->flow)
      continue;
    RegionFlow &flow = *n->flow;
    bool needsPostLoopAcquire = false;
    planLoopFlow(g, n, flow, needsPostLoopAcquire);
    if (flow.isPointOfUse())
      lowerPointOfUse(g, n, flow, needsPostLoopAcquire);
  }
}

// Doc: sync-dag.md#the-walk-accesses-to-edges
// Pass tokens through chains, ENTER, and EXIT once every acquire and release
// has its final place, and record on each access and release the exact node
// whose token it consumes. Producers are acquires and token-returning
// regions; region entry inherits the incoming producer unchanged.
struct TokenEnv {
  struct Record {
    Owner owner;
    Node *producer = nullptr;
  };
  SmallVector<Record, 2> live;
  const Record *last() const { return live.empty() ? nullptr : &live.back(); }
  Node *findOwner(const Owner &owner) {
    for (Record &r : live)
      if (sameOwner(r.owner, owner))
        return r.producer;
    return nullptr;
  }
  void record(const Owner &owner, Node *producer) {
    llvm::erase_if(live, [&](const Record &r) {
      return !r.owner || sameOwner(r.owner, owner);
    });
    live.push_back(Record{owner, producer});
  }
  void keepOnly(const Record &r) { live.assign(1, r); }
  void clear() { live.clear(); }
};
// Mirror of EmitIR's per-node token selection: the last live token by
// default; the owner's retained token when the DAG proved reuse.
static Node *selectTokenSource(TokenEnv &env, Node *n) {
  const TokenEnv::Record *last = env.last();
  if (n->owner && nodeReusesToken(n, n->owner) &&
      (!last || !sameOwner(last->owner, n->owner)))
    return env.findOwner(n->owner);
  return last ? last->producer : nullptr;
}
static void seedEntryTokens(TokenEnv &env, Node *head) {
  if (!head || head->pieceInfo.empty())
    return;
  std::optional<Owner> owner = uniformPieceOwner(head);
  if (!owner) {
    env.clear();
    return;
  }
  const TokenEnv::Record *last = env.last();
  if (owner->has_value() && last && !last->owner)
    env.keepOnly(TokenEnv::Record{*owner, last->producer}); // adopt root token
  else if (Node *producer = env.findOwner(*owner))
    env.keepOnly(TokenEnv::Record{*owner, producer});
  else
    env.clear();
}
static void assignTokenChain(GroupDag &g, Node *head, TokenEnv &env);
static void assignTokenRegion(GroupDag &g, Node *n, TokenEnv &env) {
  if (n->kind == Node::For) {
    TokenEnv body = env;
    if (n->flow) {
      if (!n->flow->threadsToken())
        body.clear();
      else // the carrier: the loop region node is the body's producer
        body.keepOnly(TokenEnv::Record{n->flow->owner, n});
    } else {
      seedEntryTokens(body, n->children[0]);
    }
    assignTokenChain(g, n->children[0], body);
    if (n->flow) {
      if (!n->flow->threadsToken())
        env.clear();
      else
        env.keepOnly(TokenEnv::Record{n->flow->owner, n});
    }
    return;
  }
  TokenEnv thenEnv = env, elseEnv = env;
  seedEntryTokens(thenEnv, n->children[0]);
  if (n->children.size() > 1 && n->children[1])
    seedEntryTokens(elseEnv, n->children[1]);
  assignTokenChain(g, n->children[0], thenEnv);
  if (n->children.size() > 1 && n->children[1])
    assignTokenChain(g, n->children[1], elseEnv);
  if (n->flow)
    env.keepOnly(TokenEnv::Record{n->flow->owner, n});
}
static void assignTokenChain(GroupDag &g, Node *head, TokenEnv &env) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      const Sema &s = getSema(g, n);
      Owner tokenOwner = s.isEntry && !n->owner ? s.entryTokenOwner : n->owner;
      env.record(tokenOwner, n);
      break;
    }
    case Node::Release:
      n->tokenSource = selectTokenSource(env, n);
      break;
    case Node::Access:
      if (nodeTouchesGroup(g, n))
        n->tokenSource = selectTokenSource(env, n);
      break;
    case Node::For:
    case Node::If:
      assignTokenRegion(g, n, env);
      break;
    case Node::Enter:
    case Node::Exit:
    case Node::Func:
      break;
    }
  }
}
static void assignTokenSources(GroupDag &g) {
  if (g.semas.empty() || g.root->children.empty())
    return;
  TokenEnv env;
  assignTokenChain(g, g.root->children[0], env);
}

static void addPart(SmallVectorImpl<int> &parts, int part) {
  if (!llvm::is_contained(parts, part))
    parts.push_back(part);
}
// Doc: sync-dag.md#notation
static SmallVector<int, 4> computeRequiredParts(Node *head) {
  SmallVector<int, 4> chainParts;
  for (Node *n = head; n; n = n->next) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire || n->kind == Node::Release) &&
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
          getDisallowAccMultiBuffer(outerWSLoop(loop)) || !canDoubleBufferAcc(mma, numTmemBlocks))
        return false;
    }
  return true;
}
static FailureOr<std::optional<int>> getPlannedBufferCopy(GroupDag &g) {
  std::optional<int> plannedCopy;
  bool sawMissing = false;
  for (const Member &m : g.pieceTable.members) {
    auto copyAttr =
        m.allocOp->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
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
// Early half: the buffer copy count depends only on the reduced edges (is the
// group synchronized at all?) and the members, so it is decided before any
// acquire or release exists.
static LogicalResult computeBackingCopies(GroupDag &g, ArrayRef<EdgeRec> edges,
                                          bool useMetaPartitioner, int &numTmemBlocks) {
  g.numCopies = 1;
  bool synchronized = !edges.empty();
  FailureOr<std::optional<int>> planned = getPlannedBufferCopy(g);
  if (failed(planned))
    return failure();
  std::optional<int> plannedCopy = *planned;
  if (synchronized && plannedCopy)
    g.numCopies = *plannedCopy;
  else if (synchronized && g.isTmem() && !useMetaPartitioner &&
           isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  if (synchronized && g.isTmem())
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return success();
}
// Late half: semaphore copies read the placed releases (producer-load payload)
// and so run after protocol creation.
static LogicalResult computeSemaphoreCopies(GroupDag &g, int lowerSemaphoreNumStages) {
  g.numSemaphoreCopies = g.numCopies;
  FailureOr<std::optional<int>> planned = getPlannedBufferCopy(g);
  if (failed(planned))
    return failure();
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad))
      hasProducerLoad = true;
  });
  if (!g.semas.empty() && g.isLocal() && !*planned && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
  return success();
}

// Docs: sync-dag.md#edges-to-semaphores
//       sync-dag.md#region-flows
static LogicalResult verifyPointOfUseFlow(GroupDag &g, Node *F, const RegionFlow &c) {
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
    return errorAt(c.exit()) << "non-transparent region supplies a point-of-use flow";
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
    if (postLoopAcquire->kind != Node::Acquire ||
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
    bool prefixCut = bridgeAcquire == c.entryAcquire &&
                     c.entryAcquire->sema == recurrenceAcquire->sema;
    const Sema &entrySema = getSema(g, bridgeAcquire);
    Owner entryOwner = bridgeAcquire->owner ? bridgeAcquire->owner : entrySema.entryTokenOwner;
    bool common = getSema(g, recurrenceAcquire).isEntry && bridgeRelease &&
                  bridgeAcquire->kind == Node::Acquire && sameOwner(entryOwner, c.owner) &&
                  bridgeRelease->kind == Node::Release &&
                  bridgeRelease->sema == recurrenceAcquire->sema &&
                  bridgeRelease->count == recurrenceAcquire->count &&
                  sameOwner(bridgeRelease->owner, c.owner) &&
                  bridgeRelease->payloads == Payloads{AsyncOp::NONE} &&
                  !bridgeRelease->sat;
    if (!common)
      return errorAt(F) << "malformed capability bridge";
    if (prefixCut) {
      if (!feedDominatesThroughCarriers(g, F, bridgeAcquire, c.owner, recurrenceAcquire->sema) ||
          bridgeRelease->parent != F->parent || bridgeRelease->next != F ||
          (c.owner && !nodeReusesToken(bridgeRelease, c.owner)))
        return errorAt(F) << "malformed prefix capability cut";
    } else {
      if (bridgeAcquire->sema != c.entryAcquire->sema || bridgeAcquire->parent != F->parent ||
          !sameOwner(bridgeAcquire->owner, c.owner) ||
          bridgeRelease->parent != bridgeAcquire->parent || !precedesInChain(F, bridgeAcquire) ||
          !precedesInChain(bridgeAcquire, bridgeRelease))
        return errorAt(F) << "capability bridge is not after its loop";
    }
  } else if (c.bridgeRelease) {
    return errorAt(F) << "unexpected outer-to-local semaphore bridge";
  }
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  SmallVector<unsigned> releaseArrivals(g.semas.size(), 0);
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifyFlow = [&](Node *n) -> LogicalResult {
    if (!n->isRegion() || !n->flow)
      return success();
    const RegionFlow &c = *n->flow;
    if (!c.ownersCompatible)
      return semaError(n->op) << "region paths export capabilities owned by different partitions";
    if (c.exits.size() != n->children.size())
      return semaError(n->op) << "region flow does not cover every exit path";
    for (Node *final : c.exits)
      if (c.threadsToken() && final && !exportsToken(final))
        return semaError(n->op) << "carried region path exports no SSA capability";
    if (c.threadsToken() && !c.semaTransfer.valid)
      return semaError(n->op) << "region paths export incompatible semaphore capabilities";
    if (c.threadsToken())
      return success();
    if (c.isPointOfUse()) {
      if (!c.exit() || !c.firstToucher || !c.entryAcquire)
        return semaError(n->op) << "malformed point-of-use flow crossing";
      if (!exitsThroughRegion(c) && c.exit()->kind != Node::Acquire)
        return semaError(n->op) << "point-of-use flow without acquire regain";
      return verifyPointOfUseFlow(g, n, c);
    }
    if (c.exits.empty() || !c.exits[0] ||
        c.exits[0]->kind != Node::For || c.firstToucher || c.entryAcquire)
      return semaError(n->op) << "malformed child-owned region flow";
    const RegionFlow *child = c.exits[0]->flow ? &*c.exits[0]->flow : nullptr;
    if (!child || child->threadsToken())
      return semaError(n->op) << "child-owned flow without native child";
    return success();
  };
  auto verifyNode = [&](Node *n) -> LogicalResult {
      if (failed(verifyFlow(n)))
        return failure();
      if (n->kind == Node::Release) {
        if (n->payloads.empty())
          return semaError(g.root->op) << "release without payload record";
        releaseArrivals[n->sema] += std::max(1u, n->count) *
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
    if (failed(forEachNodeChecked(g.root->children[0], verifyNode)))
      return failure();
  for (auto [sid, s] : llvm::enumerate(g.semas)) {
    if (releaseArrivals[sid] != s.expectedArrivals)
      return semaError(g.root->op) << "semaphore " << s.name << " has " << releaseArrivals[sid]
             << " release arrivals, expected " << s.expectedArrivals;
  }
  return success();
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;
struct ScheduleEdge { Operation *producer, *consumer; };
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

struct SlotEvent {
  GroupDag *group = nullptr;
  Node *node = nullptr;
  unsigned advances = 0;
};
using PhysicalKey = std::pair<int64_t, GroupDag *>;
static PhysicalKey physicalKey(GroupDag &group) {
  return group.isCircular() ? PhysicalKey{group.bufferId, nullptr}
                            : PhysicalKey{0, &group};
}
struct PhysicalSchedules {
  MutableArrayRef<GroupDag> groups;
  llvm::MapVector<PhysicalKey, SmallVector<GroupDag *, 2>> sets;
  std::map<std::pair<PhysicalKey, Operation *>, SlotSchedule> loopSlots;
  explicit PhysicalSchedules(MutableArrayRef<GroupDag> groups) : groups(groups) {
    for (GroupDag &group : groups)
      sets[physicalKey(group)].push_back(&group);
  }
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
  return n->kind == Node::Access || (n->isRegion() && !n->pieceInfo.empty());
}
static bool isDirectLoopNode(const Node *n) {
  return n->parent && n->parent->kind == Node::For &&
         (n->prev || n->next || llvm::is_contained(n->parent->children, n));
}

// Doc: sync-dag.md#authored-buffer-stage-offsets
static SlotSchedule replaySlots(ArrayRef<SlotEvent> events, bool assignOffsets = false) {
  SlotSchedule result;
  DenseMap<GroupDag *, int64_t> lastProduced;
  int64_t cursor = -1;
  for (const SlotEvent &event : events) {
    std::optional<int64_t> required;
    if (slotEventEffect(event.node) == Effect::W) {
      if (event.advances > 1)
        result.complete = false;
      cursor += event.advances;
      result.advancesPerIteration += event.advances;
      if (cursor >= 0)
        required = lastProduced[event.group] = cursor;
    } else if (auto it = lastProduced.find(event.group); it != lastProduced.end()) {
      required = it->second;
    }
    if (!required) {
      result.complete = false;
      continue;
    }
    result.ordinalByAccess[event.node] = *required;
    if (assignOffsets)
      event.node->bufferStageOffset = *required - cursor;
  }
  return result;
}

// Doc: sync-dag.md#circular-groups
// Derive physical ring displacements before EMIT so protocol and views agree.
static LogicalResult assignCircularStageOffsets(PhysicalSchedules &physical) {
  for (auto &[_, physicalSet] : physical.sets) {
    if (!physicalSet.front()->isCircular())
      continue;
    SmallVector<GroupDag *, 4> set;
    llvm::copy_if(physicalSet, std::back_inserter(set),
                  [](GroupDag *g) { return !g->semas.empty(); });
    if (set.empty())
      continue;
    auto type = set.front()->pieceTable.members.front().type;
    int64_t numCopies = set.front()->numCopies;
    DenseSet<int64_t> starts;
    DenseMap<Operation *, SmallVector<SlotEvent, 1>> eventsByOp;
    for (GroupDag *g : set) {
      if (g->pieceTable.members.size() != 1)
        return semaError(g->root->op) << "malformed circular local logical group";
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
          eventsByOp[n->op].push_back(
              SlotEvent{g, n, slotEventEffect(n) == Effect::W});
      });
    }
    SmallVector<SlotEvent> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end())
        ordered.append(it->second.begin(), it->second.end());
    });
    SlotSchedule slots = replaySlots(ordered, /*assignOffsets=*/true);
    for (const SlotEvent &event : ordered) {
      const Member &member = event.group->pieceTable.members.front();
      auto stage = slots.ordinalByAccess.find(event.node);
      if (event.advances) {
        assert(stage != slots.ordinalByAccess.end());
        if (member.circularStart != stage->second % numCopies)
          return semaError(member.allocOp) << "circular producer order expects buffer.start "
                 << stage->second % numCopies << ", got " << member.circularStart;
      } else if (stage == slots.ordinalByAccess.end()) {
        return semaError(member.allocOp) << "circular consumer appears before producer";
      }
    }
    for (GroupDag *g : set)
      forEachNode(*g, [&](Node *n) {
        bool forward = n->kind == Node::Acquire;
        if (!forward && n->kind != Node::Release)
          return;
        Node *access = n;
        do
          access = forward ? access->next : access->prev;
        while (access && (access->kind != Node::Access || !access->bufferStageOffset));
        if (access)
          n->stageOffset = access->bufferStageOffset;
      });
  }
  return success();
}

// Doc: sync-dag.md#pipeline-schedule
static Operation *findScheduleAnchor(const Node *anchor, bool producer = false) {
  for (const Node *n = anchor; n; n = producer ? n->prev : n->next) {
    if (n->kind == Node::Access)
      return producer && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion() && n->op)
      return n->op;
  }
  return nullptr;
}
struct LoopAnchors { scf::ForOp loop; Operation *producer, *consumer; };
static std::optional<LoopAnchors>
findCommonScheduledLoop(Operation *producer, Operation *consumer) {
  for (Operation *parent = producer->getParentOp(); parent; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName))
      continue;
    Operation *producerInLoop = loop.getBody()->findAncestorOpInBlock(*producer);
    Operation *consumerInLoop = loop.getBody()->findAncestorOpInBlock(*consumer);
    if (producerInLoop && consumerInLoop)
      return LoopAnchors{loop, producerInLoop, consumerInLoop};
  }
  return std::nullopt;
}

// Doc: sync-dag.md#authored-buffer-stage-offsets
static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet, scf::ForOp loop) {
  SmallVector<SlotEvent, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (GroupDag *group : physicalSet) {
    // Select only nodes in this loop's direct SYNC-DAG chain.  In particular,
    // do not recurse into scf.if/scf.for children: the scheduled region op is
    // the enclosing loop's event and already carries their ownership/effect
    // summary.
    for (const std::unique_ptr<Node> &storage : group->nodes) {
      Node *node = storage.get();
      if (!isDirectLoopNode(node) || node->parent->op != loop.getOperation())
        continue;
      if (node->kind == Node::Acquire && node->scheduleAnchor &&
          isSlotEvent(node->scheduleAnchor) && slotEventEffect(node->scheduleAnchor) == Effect::W) {
        ++advancesByAccess[node->scheduleAnchor];
        continue;
      }
      if (!isSlotEvent(node) || !node->op)
        continue;
      events.push_back(SlotEvent{group, node});
    }
  }
  llvm::stable_sort(events, [](const SlotEvent &lhs, const SlotEvent &rhs) {
    return lhs.node->op != rhs.node->op && lhs.node->op->isBeforeInBlock(rhs.node->op);
  });
  for (SlotEvent &event : events)
    event.advances = advancesByAccess.lookup(event.node);
  return replaySlots(events);
}
static const SlotSchedule &getSlotSchedule(PhysicalSchedules &physical,
                                           GroupDag &group, scf::ForOp loop) {
  PhysicalKey set = physicalKey(group);
  auto [it, inserted] = physical.loopSlots.try_emplace(std::make_pair(set, loop.getOperation()));
  if (inserted)
    it->second = computeSlotSchedule(physical.sets[set], loop);
  return it->second;
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}
// Doc: sync-dag.md#finalizing-one-handoff
static std::optional<int64_t>
computeLoopCarriedDistance(const SlotSchedule &slots, int64_t numSemaphoreCopies,
                           Node *producer, Node *consumer) {
  auto producerIt = slots.ordinalByAccess.find(producer);
  auto consumerIt = slots.ordinalByAccess.find(consumer);
  if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
      consumerIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0)
    return std::nullopt;
  int64_t orbit = numSemaphoreCopies / std::gcd(numSemaphoreCopies, slots.advancesPerIteration);
  for (int64_t distance = 1; distance <= orbit; ++distance)
    if (positiveMod(consumerIt->second + distance * slots.advancesPerIteration,
                    numSemaphoreCopies) == positiveMod(producerIt->second, numSemaphoreCopies))
      return distance;
  return std::nullopt;
}
// Doc: sync-dag.md#non-circular-alias-handoffs
static bool isAliasedMultibufferedGroup(const GroupDag &group) {
  if (group.isCircular() || group.pieceTable.members.size() < 2 || group.numSemaphoreCopies <= 1)
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
static LogicalResult
assignAliasedHandoffStageOffsets(PhysicalSchedules &physical, GroupDag &group) {
  if (!isAliasedMultibufferedGroup(group))
    return success();
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
    const SlotSchedule &slots = getSlotSchedule(physical, group, loop);
    auto producerIt = slots.ordinalByAccess.find(producer);
    auto consumerIt = slots.ordinalByAccess.find(consumer);
    if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
        consumerIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0)
      return semaError(producer->op) << "cannot derive multibuffered alias handoff slots";
    int64_t numSemaphoreCopies = group.numSemaphoreCopies;
    int64_t offset = 0;
    if (precedesInChain(release, release->sat)) {
      offset = positiveMod(consumerIt->second - producerIt->second, numSemaphoreCopies);
    } else if (!computeLoopCarriedDistance(
                   slots, numSemaphoreCopies, producer, consumer)) {
      int64_t nextConsumer = consumerIt->second + slots.advancesPerIteration;
      offset = positiveMod(nextConsumer - producerIt->second, numSemaphoreCopies);
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
    auto [it, inserted] = vertexByOwner.try_emplace(key, vertexByOwner.size());
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
    for (auto [edgeIndex, constraint] : llvm::enumerate(model.ownerConstraints)) {
      unsigned producer = getVertex(constraint.producerOwner);
      unsigned consumer = getVertex(constraint.consumerOwner);
      int64_t candidate = offset[producer] + constraint.requiredDelay();
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
    unsigned vertex = getVertex(model.ownerConstraints[*lastUpdated].consumerOwner);
    for (unsigned i = 0; i < numVertices; ++i) {
      if (!predecessor[vertex])
        break;
      vertex = getVertex(model.ownerConstraints[*predecessor[vertex]].producerOwner);
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
      vertex = getVertex(model.ownerConstraints[edgeIndex].producerOwner);
    } while (vertex != cycleStart && cycle.size() <= numVertices);
    if (cycle.empty())
      cycle.push_back(*lastUpdated);
    int64_t cycleDelay = 0;
    for (unsigned edgeIndex : cycle)
      cycleDelay += model.ownerConstraints[edgeIndex].requiredDelay();
    const OwnerScheduleConstraint &first = model.ownerConstraints[cycle.front()];
    InFlightDiagnostic diag =
        semaError(first.producer)
        << "fixed loop.stage assignments form an unsatisfiable semaphore handoff cycle";
    if (cycleDelay > 0)
      diag << " (cycle requires " << cycleDelay << " additional pipeline iteration"
           << (cycleDelay == 1 ? "" : "s") << ")";
    for (unsigned edgeIndex : llvm::reverse(cycle)) {
      const OwnerScheduleConstraint &constraint = model.ownerConstraints[edgeIndex];
      diag.attachNote(constraint.consumer->getLoc()) << "handoff "
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
           offset[getVertex(constraint.producerOwner)] + constraint.requiredDelay();
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
        if (isTight(constraint) && getVertex(constraint.producerOwner) == vertex)
          stack.push_back(getVertex(constraint.consumerOwner));
    }
    return false;
  };
  for (const OwnerScheduleConstraint &constraint : model.ownerConstraints) {
    bool directlySameWave = constraint.requiredDelay() == 0;
    bool onZeroDelayCycle =
        isTight(constraint) &&
        hasTightPath(getVertex(constraint.consumerOwner),
                     getVertex(constraint.producerOwner));
    if (directlySameWave || onZeroDelayCycle)
      model.clusterEdges.push_back(
          ScheduleEdge{constraint.producer, constraint.consumer});
  }
  return success();
}

// Doc: sync-dag.md#finalizing-one-handoff
static LogicalResult addSyncScheduleEdges(PhysicalSchedules &physical,
                                          llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
  for (GroupDag &group : physical.groups) {
    auto addReleaseEdge = [&](Node *release) -> LogicalResult {
      if (release->kind != Node::Release || !release->sat)
        return success();
      Node *acquire = release->sat;
      Operation *producer = findScheduleAnchor(release->scheduleAnchor, /*producer=*/true);
      Operation *consumer = findScheduleAnchor(acquire->scheduleAnchor);
      if (!producer || !consumer)
        return success();
      std::optional<LoopAnchors> anchors = findCommonScheduledLoop(producer, consumer);
      if (!anchors)
        return success();
      auto [loop, producerAnchor, consumerAnchor] = *anchors;
      if (producerAnchor == consumerAnchor)
        return success();
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producerAnchor);
      gpu::StageCluster consumerSchedule = gpu::getStageCluster(consumerAnchor);
      if (!producerSchedule || !consumerSchedule)
        return success();
      int64_t distance = 0;
      if (!precedesInChain(release, acquire)) {
        bool exactPlannedRecurrence = acquire->recurrenceDistance &&
            release->parent == acquire->parent && acquire->parent &&
            acquire->parent->op == loop.getOperation();
        if (exactPlannedRecurrence) {
          distance = *acquire->recurrenceDistance;
        } else {
          const SlotSchedule &slots = getSlotSchedule(physical, group, loop);
          std::optional<int64_t> loopCarriedDistance = computeLoopCarriedDistance(
                  slots, group.numSemaphoreCopies, release->scheduleAnchor,
                  acquire->scheduleAnchor);
          if (!loopCarriedDistance) {
            InFlightDiagnostic diag = semaError(producerAnchor)
                << "cannot determine loop-carried dependency distance for a "
                   "physical buffer slot";
            diag.attachNote(consumerAnchor->getLoc()) << "next token ownership starts here";
            return failure();
          }
          distance = *loopCarriedDistance;
        }
      }
      modelsByLoop[loop.getOperation()].ownerConstraints.push_back(
          OwnerScheduleConstraint{release->owner, acquire->owner,
                                  producerAnchor, consumerAnchor,
                                  producerSchedule->first,
                                  consumerSchedule->first, distance});
      return success();
    };
    for (const auto &node : group.nodes)
      if (isDirectLoopNode(node.get()) && failed(addReleaseEdge(node.get())))
        return failure();
  }
  return success();
}
static void addSSAClusterConstraints(scf::ForOp loop, SmallVectorImpl<ScheduleEdge> &edges) {
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
      auto [producer, consumer] = edge;
      if (!cluster.contains(producer) || !cluster.contains(consumer))
        continue;
      int64_t required = cluster.lookup(producer) + (producer->isBeforeInBlock(consumer) ? 0 : 1);
      if (cluster.lookup(consumer) >= required)
        continue;
      cluster[consumer] = required;
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
static gpu::StageCluster scheduleAtOwnerBoundary(
    const Node *n, gpu::StageCluster schedule) {
  if (!schedule || findScheduleAnchor(n->next))
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
        Operation *anchor = n->postLoopAcquire ? nullptr : findScheduleAnchor(n->next);
        n->stageCluster =
            anchor ? gpu::getStageCluster(anchor)
                   : scheduleAtOwnerBoundary(
                         n, cache.lookup(ownerKey(n->owner)));
      }
      break;
    case Node::Release:
      if (n->owner)
        n->stageCluster = cache.lookup(ownerKey(n->owner));
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
static LogicalResult analyzeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, LoopScheduleModel> modelsByLoop;
  PhysicalSchedules physical(groups);
  if (failed(assignCircularStageOffsets(physical)))
    return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(physical, group)))
      return failure();
  if (failed(addSyncScheduleEdges(physical, modelsByLoop)))
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
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  if (groups.empty())
    return success();
  if (failed(analyzeSyncSchedule(groups)))
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
    ChainWalker(g, top, edges, /*underFor=*/false).run(g.root->children[0]);
  }
  reduceEdges(g, edges);
  if (failed(computeBackingCopies(g, edges, useMetaPartitioner, numTmemBlocks)))
    return failure();
  if (failed(buildEdgesAndSemas(g, edges)))
    return failure();
  if (failed(insertEntryAcquires(g)))
    return failure();
  if (!g.root->children.empty()) {
    buildRegionFlows(g, g.root->children[0]);
    pruneDeadIfFlows(g, g.root->children[0], /*region=*/nullptr);
    planRegionFlows(g, g.root->children[0]);
    computeRequiredParts(g.root->children[0]);
    assignTokenSources(g);
  }
  if (failed(computeSemaphoreCopies(g, lowerSemaphoreNumStages)))
    return failure();
  if (!g.semas.empty())
    for (Operation *alloc : g.ttDescriptorFedMembers)
      return semaError(alloc) << "managed local_alloc sourced from a tt-form descriptor load — "
                "nvws-insert-allocas must convert this upstream "
                "(pipeline invariant violated)";
  return verifySyncDag(g);
}

static void printThreadInfo(llvm::raw_ostream &os, const Node *n) {
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
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g,
                           const Node *region, unsigned chainIdx) {
  if (!region || !region->flow)
    return;
  os << " yield{";
  const RegionFlow &c = *region->flow;
  if (!c.threadsToken()) { // native/child-owned: no slot
    os << (c.isChildOwned() ? "drop" : "native");
  } else {
    Node *f = chainIdx < c.exits.size() ? c.exits[chainIdx] : nullptr;
    if (!f)
      os << "pass";
    else if (f->kind == Node::Acquire || f->kind == Node::Release)
      os << (f->kind == Node::Acquire ? "a " : "r ") << getSema(g, f).name;
    else
      os << (f->kind == Node::For ? "scf.for" : "scf.if");
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
        llvm::interleaveComma(n->payloads, os, [&](AsyncOp p) {
          os << nvws::stringifyAsyncOp(p);
        });
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
        printThreadInfo(os, n);
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
        printYieldInfo(os, g, region, chainIdx);
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
  if (!g.semas.empty()) {
    os << "  SEMAS: ";
    llvm::interleave(g.semas, os, [&](const Sema &s) {
      os << s.name << "{count=" << s.count;
      if (s.isEntry)
        os << " entry inherit=" << ownerStr(nullptr, s.entryTokenOwner);
      os << "}";
    }, " ");
    os << "\n";
  }
  if (g.semas.empty()) {
    os << "  BACKING: untouched (no semaphores)\n";
    return;
  }
  os << "  BACKING: numCopies=" << g.numCopies << "\n";
}
} // namespace mlir::triton::nvws_semas
