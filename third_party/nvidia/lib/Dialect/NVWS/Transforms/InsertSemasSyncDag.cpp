// SYNC analysis and scheduling. Doc section references below point into
// sema-docs/insert-semas/sync-dag.md.
#include "InsertSemas.h"
#include <limits>
#include <numeric>

namespace mlir::triton::nvws_semas {
using Payloads = SmallVector<AsyncOp, 1>;
using PieceEffects = std::map<PieceId, Effect>;

struct ActiveUse {
  Owner owner;
  Node *node = nullptr;
  Payloads payloads;
  // Owners whose existing dependency already orders this node before them.
  SmallVector<int64_t, 2> orderedBefore;
};
using ExitFacts = std::map<PieceId, ActiveUse>;

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
    for (ActiveUse &use : uses)
      if (sameOwner(use.owner, owner))
        return &use;
    return nullptr;
  }
  bool canReuseToken(const Owner &owner, Effect effect) {
    if (!initialized())
      return false;
    if (effect == Effect::R)
      return useFor(owner);
    return llvm::all_of(uses, [&](const ActiveUse &use) {
      return sameOwner(use.owner, owner) ||
             llvm::is_contained(use.orderedBefore, ownerKey(owner));
    });
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
    for (const Token &token : live)
      if (sameOwner(token.owner, owner))
        return &token;
    return nullptr;
  }
  const Token *last() const {
    return !live.empty() && live.back().node ? &live.back() : nullptr;
  }
  void record(const Owner &owner, Node *node, const Payloads &payloads) {
    llvm::erase_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
    live.push_back(Token{owner, node, payloads});
  }
};

// Sentinel for EdgeRec::coveredVia: no alternate live use is known to impose
// the same ordering.
constexpr int64_t kUncovered = std::numeric_limits<int64_t>::min();
struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  Payloads payloads;
  SmallVector<PieceId, 2> pieces;
  // Owner key of a live use through which the same ordering can be imposed,
  // or kUncovered. Doc: sync-dag.md#removing-a-release-when-another-path-imposes-the-same-wait
  int64_t coveredVia = kUncovered;
  // An exact-source async handoff carries completion, not just ordering.
  bool preserve = false;
};
using EdgeList = SmallVector<EdgeRec>;
static bool hasAsyncPayload(ArrayRef<AsyncOp> payloads) {
  return llvm::any_of(payloads,
                      [](AsyncOp payload) { return payload != AsyncOp::NONE; });
}
// Returns the owner key of a live use that already orders `use` before
// itself, or kUncovered. Only the version-source use can use this alternate
// path: a later reread carries no dependency other uses are known to inherit.
static int64_t coveringLiveUse(const PieceState &piece, const ActiveUse &use) {
  if (use.node != piece.source.node)
    return kUncovered;
  for (const ActiveUse &other : piece.uses)
    if (&other != &use &&
        llvm::is_contained(use.orderedBefore, ownerKey(other.owner)))
      return ownerKey(other.owner);
  return kUncovered;
}
static void unionPayloads(Payloads &into, const Payloads &from) {
  for (AsyncOp payload : from)
    if (!llvm::is_contained(into, payload))
      into.push_back(payload);
  llvm::sort(into);
}
static SmallVector<int64_t, 2> intersectOrderFacts(ArrayRef<int64_t> lhs,
                                                   ArrayRef<int64_t> rhs) {
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
  auto count = forOp.getStaticTripCount();
  return count && (forOp.getUnsignedCmp() ? count->ugt(0) : count->sgt(0));
}

// Raise ordering edges into `node` from every foreign live use of `piece`
// that is not already ordered before `owner` (wsAdopt also skips root uses).
static void raiseForeignUseEdges(PieceState &piece, PieceId id,
                                 const Owner &owner, Node *node,
                                 EdgeList &edges, bool wsAdopt) {
  for (const ActiveUse &use : piece.uses) {
    if (sameOwner(use.owner, owner) || (wsAdopt && !use.owner) ||
        llvm::is_contained(use.orderedBefore, ownerKey(owner)))
      continue;
    bool exactSource = use.node == piece.source.node;
    edges.push_back(EdgeRec{use.node,
                            node,
                            use.owner,
                            owner,
                            use.payloads,
                            {id},
                            coveringLiveUse(piece, use),
                            exactSource && hasAsyncPayload(use.payloads)});
  }
}
// Apply one piece's RAW/WAR rules. Token supply is handled after all pieces of
// an access have advanced.
static void applyTouch(PieceState &piece, PieceId id, const Owner &owner,
                       Effect effect, Node *node, const Payloads &payloads,
                       EdgeList &edges, bool wsAdopt) {
  if (!piece.initialized() || effect == Effect::W) {
    if (piece.initialized())
      raiseForeignUseEdges(piece, id, owner, node, edges, wsAdopt);
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
    ActiveUse *source = piece.useFor(piece.source.sourceOwner);
    bool exactSource = source && source->node == piece.source.node;
    edges.push_back(
        EdgeRec{piece.source.node,
                node,
                piece.source.sourceOwner,
                owner,
                piece.source.payloads,
                {id},
                kUncovered,
                exactSource && hasAsyncPayload(piece.source.payloads)});
    // The source edge orders the source use only while it still names that
    // node; a later reread must retain its own WAR obligation.
    if (exactSource &&
        !llvm::is_contained(source->orderedBefore, ownerKey(owner)))
      source->orderedBefore.push_back(ownerKey(owner));
  }
  piece.uses.push_back(ActiveUse{owner, node, payloads, {}});
}
static bool pieceTouchedAfter(GroupDag &g, Node *region, PieceId piece) {
  for (Node *scope = region; scope && scope->kind != Node::Func;
       scope = scope->parent)
    for (Node *node = scope->next; node; node = node->next)
      if ((node->kind == Node::Access && touchesPiece(g, node, piece)) ||
          (node->isRegion() && node->pieceInfo.count(piece)))
        return true;
  return false;
}

class ChainWalker {
public:
  ChainWalker(GroupDag &group, ChainState &state, EdgeList &edges,
              bool underFor)
      : group(group), state(state), edges(edges), underFor(underFor) {}
  ExitFacts run(Node *head) {
    if (head->kind == Node::Enter)
      if (auto owner = uniformPieceOwner(head); owner && owner->has_value())
        tokens.live.push_back({*owner, nullptr, {}});
    for (Node *node = head; node; node = node->next) {
      if (node->kind == Node::Access)
        visitAccess(node);
      else if (node->isRegion())
        visitRegion(node);
      else if (node->kind == Node::Exit)
        visitExit(node);
    }
    ExitFacts result;
    if (head->kind == Node::Enter)
      for (auto [id, info] : sortedPieceInfo(head)) {
        ActiveUse &facts = result[id];
        if (ActiveUse *use = state[id].useFor(info.owner))
          facts = *use;
        if (facts.payloads.empty())
          facts.payloads.push_back(AsyncOp::NONE);
      }
    return result;
  }

private:
  void visitAccess(Node *node) {
    PieceEffects effects;
    forEachTouchedPiece(group, node, [&](PieceId id, Effect effect) {
      mergeEffect(effects, id, effect);
    });
    const Tokens::Token *last = tokens.last();
    bool ownerDiffers =
        last && node->owner && !sameOwner(last->owner, node->owner);
    bool canReuse =
        tokens.find(node->owner) &&
        llvm::all_of(effects, [&](const auto &item) {
          return state[item.first].canReuseToken(node->owner, item.second);
        });
    size_t edgeStart = edges.size();
    Payloads payloads{asyncPayloadOf(node->op)};
    // A release describes every completion signal produced during one
    // ownership wave: consecutive same-owner writes keep earlier async
    // completions, else a later synchronous write would hide (for example) a
    // descriptor load from LowerAref. The group-wide token-reuse proof
    // certifies that no handoff intervened on any touched piece, so a forced
    // handoff starts a fresh payload set.
    bool synchronousWrite = payloads.front() == AsyncOp::NONE;
    if (group.pieceTable.members.size() > 1 && canReuse && synchronousWrite) {
      for (auto [id, effect] : effects) {
        if (effect != Effect::W)
          continue;
        PieceState &piece = state[id];
        if (!sameOwner(piece.source.sourceOwner, node->owner))
          continue;
        bool sameOwnerWave =
            llvm::all_of(piece.uses, [&](const ActiveUse &use) {
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
    for (auto [branch, childHead] : llvm::enumerate(node->children)) {
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
          branchOrder = it->second.orderedBefore;
        auto &order = returned[id].orderedBefore;
        order = branch == 0 ? std::move(branchOrder)
                            : intersectOrderFacts(order, branchOrder);
      }
    }
    // A loop body does not establish a fact on its zero-trip path. Likewise,
    // an absent else branch carries the incoming fact unchanged.
    bool hasBypass = (node->kind == Node::For && !knownNonEmptyLoop(node)) ||
                     (node->kind == Node::If && node->children.size() < 2);
    if (hasBypass)
      for (auto [id, info] : infos)
        returned[id].orderedBefore =
            intersectOrderFacts(returned[id].orderedBefore, incomingOrder[id]);
    for (auto [id, info] : infos) {
      auto facts = returned.find(id);
      ActiveUse *use = state[id].useFor(info.owner);
      if (facts == returned.end() || !use)
        continue;
      if (!facts->second.payloads.empty())
        use->payloads = facts->second.payloads;
      if (use->node == node)
        for (int64_t owner : facts->second.orderedBefore)
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
        raiseForeignUseEdges(piece, id, info.owner, node, edges,
                             /*wsAdopt=*/false);
      ActiveUse carried{info.owner, node, {AsyncOp::NONE}, {}};
      if (ActiveUse *use = piece.useFor(info.owner))
        carried = *use;
      piece.uses.assign(1, carried);
    }
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
static void appendAfterReleases(Node *node, Node *anchor) {
  while (anchor->next && anchor->next->kind == Node::Release)
    anchor = anchor->next;
  spliceAfter(node, anchor);
}
static Node *newProtocolNode(GroupDag &g, Node::Kind kind, Node *parent,
                             Owner owner, SemaId sema, unsigned count) {
  Node *node = g.newNode(kind, nullptr, parent);
  node->owner = owner;
  node->sema = sema;
  node->count = count;
  return node;
}
// First chain-local (no region descent) owned access, optionally restricted
// to owner `match`.
static Node *firstOwnedAccess(Node *head, const Owner &match = std::nullopt) {
  for (Node *n = head; n; n = n->next)
    if (n->kind == Node::Access && n->owner &&
        (!match || sameOwner(n->owner, match)))
      return n;
  return nullptr;
}
namespace {
using SyncVec = std::map<int64_t, unsigned>; // partitionKey -> node index
using EdgeBuckets = DenseMap<Node *, SmallVector<unsigned, 2>>;
using Snapshots = DenseMap<Node *, SyncVec>;
using Positions = DenseMap<Node *, unsigned>;
struct KnownOrder {
  std::map<int64_t, SyncVec> behind;
  void apply(const EdgeRec &edge, unsigned sourceIdx,
             const Snapshots &snapshots) {
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
static void reduceStraightEdges(Node *head, const Positions &positions,
                                ArrayRef<EdgeRec> edges,
                                const EdgeBuckets &atDst,
                                std::vector<bool> &drop) {
  KnownOrder order;
  Snapshots snapshots;
  SmallVector<int64_t, 2> tokenOwners;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner) {
        llvm::erase(tokenOwners, ownerKey(pi.owner));
        tokenOwners.push_back(ownerKey(pi.owner));
      }
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
        if (!e.preserve && order.covers(e, srcIdx) && hasToken &&
            e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        llvm::erase(tokenOwners, dk);
        tokenOwners.push_back(dk); // Kept acquire supplies Q's token.
        order.apply(e, srcIdx, snapshots);
      }
    if (n->owner) // reduction runs before protocol nodes: owned => Access
      order.record(n, positions.lookup(n), snapshots);
  }
}
static void reduceLoopCloses(GroupDag &g, Node *head,
                             const Positions &positions,
                             ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst,
                             ArrayRef<unsigned> closes,
                             std::vector<bool> &drop) {
  if (closes.empty())
    return;
  Node *firstAccess = firstOwnedAccess(head);
  if (!firstAccess)
    return;
  constexpr unsigned kPass2 = 1u << 20;
  KnownOrder order;
  Snapshots snap1, snap2;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        if (!drop[ei] && !isLoopClose(edges[ei]))
          order.apply(edges[ei], positions.lookup(edges[ei].src), snap1);
    if (n->owner)
      order.record(n, positions.lookup(n), snap1);
  }
  EdgeBuckets closeAt;
  for (unsigned ei : closes) {
    const EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    llvm::SmallDenseSet<PieceId> seen;
    for (Node *n = head; n; n = n->next)
      if (sameOwner(n->owner, e.dstOwner))
        forEachTouchedPiece(g, n, [&](PieceId pc, Effect) {
          if (llvm::is_contained(e.pieces, pc) && seen.insert(pc).second)
            latest = n;
        });
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
        if (!e.preserve && order.covers(e, positions.lookup(e.src)) &&
            tokenAvailable.contains(dk) &&
            !sameOwner(e.dstOwner, firstAccess->owner)) {
          drop[ei] = true;
          continue;
        }
        order.apply(e, positions.lookup(e.src), snap1);
      }
    if (n->owner)
      order.record(n, kPass2 + positions.lookup(n), snap2);
  }
}
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (g.root->children.empty() || edges.empty())
    return;
  std::vector<bool> drop(edges.size(), false);
  auto reduce = [&](auto &&self, Node *head) -> void {
    Positions positions;
    unsigned position = 0;
    for (Node *n = head; n; n = n->next)
      positions[n] = position++;
    SmallVector<unsigned, 4> closes;
    EdgeBuckets atDst;
    for (auto [i, edge] : llvm::enumerate(edges)) {
      if (!positions.contains(edge.src) || !positions.contains(edge.dst))
        continue;
      atDst[edge.dst].push_back(i);
      if (isLoopClose(edge))
        closes.push_back(i);
    }
    for (auto &bucket : atDst)
      llvm::stable_sort(bucket.second, [&](unsigned a, unsigned b) {
        return positions.lookup(edges[a].src) > positions.lookup(edges[b].src);
      });
    reduceStraightEdges(head, positions, edges, atDst, drop);
    if (head->parent && head->parent->kind == Node::For)
      reduceLoopCloses(g, head, positions, edges, atDst, closes, drop);
    for (Node *n = head; n; n = n->next)
      if (n->isRegion())
        for (Node *child : n->children)
          self(self, child);
  };
  reduce(reduce, g.root->children[0]);
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
// True when an anchor at `before` is guaranteed to have executed once the
// walk reaches `after`, across nesting levels. Both sides climb only through
// For parents: a loop body executes on every pass that raises obligations
// beyond it (a zero-trip loop raises none), while an If branch must never
// certify the other path.
static bool reachesForward(Node *before, Node *after) {
  auto forParent = [](Node *n) -> Node * {
    return n->parent && n->parent->kind == Node::For ? n->parent : nullptr;
  };
  for (Node *b = before; b; b = forParent(b))
    for (Node *a = after; a; a = forParent(a))
      if (a == b || (a->parent == b->parent && precedesInChain(b, a)))
        return true;
  return false;
}
static unsigned arrivalContribution(ArrayRef<EdgeRec> edges) {
  unsigned total = 0;
  for (const EdgeRec &edge : edges)
    total += std::max(1u, static_cast<unsigned>(edge.payloads.size()));
  return total;
}

static LogicalResult buildEdgesAndSemas(GroupDag &g,
                                        SmallVector<EdgeRec> &edges) {
  std::map<std::tuple<Node *, int64_t, int64_t>, Node *> releaseFloors;
  for (const EdgeRec &edge : edges) {
    Node *&floor = releaseFloors[{edge.dst, ownerKey(edge.dstOwner),
                                  ownerKey(edge.srcOwner)}];
    if (!floor || precedesInChain(floor, edge.src))
      floor = edge.src;
  }
  reduceEdges(g, edges);
  struct Handoff {
    Node *dst = nullptr;
    Owner owner;
    SmallVector<EdgeRec, 2> incoming;
    // Parallel to incoming: possible alternate paths for each merged sender.
    struct SenderFacts {
      bool allCovered = true;
      bool drop = false;
      SmallVector<std::pair<Node *, int64_t>, 2> covers; // (src, coverer key)
    };
    SmallVector<SenderFacts, 2> facts;
    std::optional<SemaId> sema;
  };
  auto recordCover = [](Handoff::SenderFacts &facts, const EdgeRec &edge) {
    if (edge.coveredVia == kUncovered)
      facts.allCovered = false;
    else
      facts.covers.push_back({edge.src, edge.coveredVia});
  };
  llvm::MapVector<std::tuple<Node *, int64_t>, unsigned> dstIndex;
  SmallVector<Handoff, 0> handoffs;
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
      handoff.incoming.back().src = releaseFloors[{
          edge.dst, ownerKey(edge.dstOwner), ownerKey(edge.srcOwner)}];
      llvm::sort(handoff.incoming.back().payloads);
      handoff.facts.emplace_back();
      recordCover(handoff.facts.back(), edge);
    } else {
      if (precedesInChain(source->src, edge.src))
        source->src = edge.src;
      unionPayloads(source->payloads, edge.payloads);
      recordCover(handoff.facts[source - handoff.incoming.begin()], edge);
    }
  }
  // Drop a whole sender only after shrinking candidates to a fixpoint. Every
  // edge still anchors its sender's release before this removal.
  for (Handoff &handoff : handoffs)
    for (Handoff::SenderFacts &facts : handoff.facts)
      facts.drop = facts.allCovered && !facts.covers.empty();
  auto survivingSender = [](const Handoff &handoff, int64_t owner) -> Node * {
    for (auto [si, sender] : llvm::enumerate(handoff.incoming))
      if (!handoff.facts[si].drop && ownerKey(sender.srcOwner) == owner)
        return sender.src;
    return nullptr;
  };
  auto coverHolds = [&](const Handoff &handoff, unsigned si) {
    int64_t self = ownerKey(handoff.incoming[si].srcOwner);
    for (auto [coveredSrc, coverer] : handoff.facts[si].covers) {
      // The intermediate owner must still release to this destination. If it
      // is the destination owner, its own program order carries its acquire
      // forward to this destination node.
      Node *covererSrc = coverer == ownerKey(handoff.owner)
                             ? handoff.dst
                             : survivingSender(handoff, coverer);
      if (!covererSrc)
        return false;
      // A surviving release from this sender must occur no earlier than the
      // original source and be acquired by the intermediate owner no later
      // than that owner's release into this handoff.
      bool leg1 = false;
      for (const Handoff &other : handoffs) {
        if (ownerKey(other.owner) != coverer)
          continue;
        if (other.dst != covererSrc && !reachesForward(other.dst, covererSrc))
          continue;
        Node *relSrc = survivingSender(other, self);
        if (relSrc &&
            (relSrc == coveredSrc || reachesForward(coveredSrc, relSrc))) {
          leg1 = true;
          break;
        }
      }
      if (!leg1)
        return false;
    }
    return true;
  };
  for (bool changed = true; changed;) {
    changed = false;
    for (Handoff &handoff : handoffs)
      for (unsigned si = 0; si < handoff.incoming.size(); ++si)
        if (handoff.facts[si].drop && !coverHolds(handoff, si)) {
          handoff.facts[si].drop = false;
          changed = true;
        }
  }
  auto survivingStats = [](const Handoff &handoff) {
    std::pair<unsigned, unsigned> stats{0, 0}; // sources, arrivals
    for (auto [si, sender] : llvm::enumerate(handoff.incoming)) {
      if (handoff.facts[si].drop)
        continue;
      ++stats.first;
      stats.second +=
          std::max(1u, static_cast<unsigned>(sender.payloads.size()));
    }
    return stats;
  };
  // Loop entry and recurrence sites share one fixed-count semaphore. This
  // removal is optional; retain the original senders when removing them would
  // make either site unable to meet the recurrence count.
  for (bool changed = true; changed;) {
    changed = false;
    for (Handoff &entry : handoffs) {
      Handoff *recurrence = nullptr;
      if (entry.dst->kind == Node::For)
        for (Node *n = entry.dst->children[0]; n; n = n->next)
          if (auto it =
                  dstIndex.find(std::make_tuple(n, ownerKey(entry.owner)));
              it != dstIndex.end() &&
              survivingStats(handoffs[it->second]).first)
            recurrence = &handoffs[it->second];
      if (!recurrence)
        continue;
      auto [entrySources, entryArrivals] = survivingStats(entry);
      unsigned recurrenceArrivals = survivingStats(*recurrence).second;
      bool entryCanMeet = entryArrivals == recurrenceArrivals ||
                          (entrySources == 1 && entryArrivals == 1);
      if (!entryCanMeet)
        for (Handoff *site : {&entry, recurrence})
          for (Handoff::SenderFacts &facts : site->facts)
            if (facts.drop) {
              facts.drop = false;
              changed = true;
            }
    }
  }
  for (Handoff &handoff : handoffs) {
    unsigned si = 0;
    llvm::erase_if(handoff.incoming,
                   [&](const EdgeRec &) { return handoff.facts[si++].drop; });
  }
  // Remove a destination group when no sender remains.
  llvm::erase_if(handoffs, [](const Handoff &h) { return h.incoming.empty(); });
  dstIndex.clear();
  for (auto [hi, handoff] : llvm::enumerate(handoffs))
    dstIndex.try_emplace(std::make_tuple(handoff.dst, ownerKey(handoff.owner)),
                         hi);
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
        if (auto it =
                dstIndex.find(std::make_tuple(n, ownerKey(handoff.owner)));
            it != dstIndex.end())
          source = &handoffs[it->second];
    if (!source->sema)
      createSema(*source);
    assert(sameOwner(source->owner, handoff.owner));
    handoff.sema = source->sema;
  }
  for (Handoff &handoff : handoffs) {
    Sema &sema = g.semas[*handoff.sema];
    unsigned sources = handoff.incoming.size();
    unsigned arrivals = arrivalContribution(handoff.incoming);
    if (arrivals != sema.count && !(sources == 1 && arrivals == 1))
      return semaError(handoff.dst->op ? handoff.dst->op : g.root->op)
             << "destination group with " << arrivals
             << " arrival contributions from " << sources
             << " sources cannot meet semaphore " << sema.name
             << " pending count " << sema.count;
    // Only a lone generic release can be scaled for a reused semaphore.
    unsigned releaseCount = arrivals == sema.count ? 1 : sema.count;
    Node *acquire = newProtocolNode(g, Node::Acquire, handoff.dst->parent,
                                    handoff.owner, *handoff.sema, sema.count);
    Node *destination = handoff.dst;
    if (handoff.dst->kind == Node::Exit && acquire->owner) {
      Node *head = handoff.dst;
      while (head->prev)
        head = head->prev;
      Node *firstAccess = firstOwnedAccess(head);
      Node *firstTouch = firstOwnedAccess(head, acquire->owner);
      if (firstTouch && firstAccess &&
          !sameOwner(firstAccess->owner, acquire->owner)) {
        destination = firstTouch;
        sema.isEntry = true;
        sema.entryTokenOwner = acquire->owner;
      }
    }
    spliceBefore(acquire, destination);
    for (const EdgeRec &edge : handoff.incoming) {
      Node *release =
          newProtocolNode(g, Node::Release, edge.src->parent, edge.srcOwner,
                          *handoff.sema, releaseCount);
      release->payloads = edge.payloads;
      assert(!release->payloads.empty());
      g.protocolArcs.push_back(
          {release, acquire, edge.src, destination, acquire});
      if (nodeReusesToken(edge.src, edge.srcOwner))
        markTokenReuse(release, edge.srcOwner);
      appendAfterReleases(release, edge.src);
      assert(release->parent == acquire->parent);
      assert(precedesInChain(release, acquire) || sema.isEntry);
    }
  }
  return success();
}
static SmallVector<Node *, 4> summarizeEntry(GroupDag &g, Node *top,
                                             std::optional<Owner> &firstOwner) {
  SmallVector<Node *, 4> placement;
  SmallVector<SmallVector<Node *, 4>, 2> branches;
  for (Node *n = top; n; n = n->next) {
    SmallVector<SmallVector<Node *, 4>, 2> nested;
    if (n->isRegion())
      for (Node *child : n->children)
        nested.push_back(summarizeEntry(g, child, firstOwner));
    bool access = n->kind == Node::Access && nodeTouchesGroup(g, n);
    bool comp = access || (n->isRegion() && !n->pieceInfo.empty());
    if (access && !firstOwner)
      firstOwner.emplace(n->owner);
    if (comp)
      placement.push_back(n);
    if (comp && n->kind == Node::If)
      branches = std::move(nested);
  }
  if (placement.size() == 1 && placement.front()->kind == Node::If) {
    SmallVector<Node *, 4> *only = nullptr;
    for (SmallVector<Node *, 4> &branch : branches)
      if (!branch.empty()) {
        if (only)
          return placement;
        only = &branch;
      }
    if (only)
      placement = std::move(*only);
  }
  return placement;
}
static bool summarizeRegionFlow(GroupDag &g, Node *region,
                                RegionFlow &crossing);
static LogicalResult insertEntryAcquires(GroupDag &g) {
  Node *top = g.root->children.empty() ? nullptr : g.root->children[0];
  if (!top || g.semas.empty())
    return success();
  std::optional<Owner> firstOwner;
  SmallVector<Node *, 4> nodes = summarizeEntry(g, top, firstOwner);
  if (nodes.empty())
    return semaError(g.root->op) << "group with sync but no placement nodes";
  if (!firstOwner)
    return semaError(g.root->op) << "group has no access nodes";
  Owner tokenOwner = *firstOwner;
  auto lastBodyAcquireSema = [](Node *loop) -> std::optional<SemaId> {
    std::optional<SemaId> sema;
    forEachNode(loop->children[0], [&](Node *m) {
      if (m->kind == Node::Acquire)
        sema = m->sema;
    });
    return sema;
  };
  std::optional<SemaId> channel;
  for (Node *n : llvm::reverse(nodes))
    if (n->kind == Node::For) {
      channel = lastBodyAcquireSema(n);
      if (channel)
        break;
    }
  if (channel) {
    SemaId sid = *channel;
    Sema &s = g.semas[sid];
    s.isEntry = true; // first event in chain order is this acquire
    s.entryTokenOwner = tokenOwner;
    Node *acq = newProtocolNode(g, Node::Acquire, nodes.front()->parent,
                                std::nullopt, sid, s.count);
    spliceBefore(acq, nodes.front());
    return success();
  }
  SemaId sid = g.semas.size();
  Sema s;
  s.name = "E" + std::to_string(sid);
  s.count = 1;
  s.isEntry = true;
  s.entryTokenOwner = tokenOwner;
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
  appendAfterReleases(rel, terminal);
  assert(rel->parent == acq->parent);
  g.semas.push_back(std::move(s));
  return success();
}
static bool summarizeRegionFlow(GroupDag &g, Node *region,
                                RegionFlow &crossing) {
  crossing.exits.clear();
  crossing.owner = std::nullopt;
  crossing.concreteSema.reset();
  std::optional<Owner> entryOwner = uniformPieceOwner(region);
  bool live = false, sawOwner = false, compatible = true;
  auto joinOwner = [&](const Owner &owner) {
    if (!sawOwner) {
      crossing.owner = owner;
      sawOwner = true;
    } else if (!sameOwner(crossing.owner, owner)) {
      compatible = false;
    }
  };
  for (Node *child : region->children) {
    Node *final = nullptr, *preferred = nullptr;
    Owner owner;
    for (Node *n = child; n; n = n->next) {
      if (n->kind == Node::Acquire) {
        final = n;
        owner = n->owner;
        if (entryOwner) {
          const Sema &s = getSema(g, n);
          Owner tokenOwner =
              s.isEntry && !n->owner ? s.entryTokenOwner : n->owner;
          if (sameOwner(tokenOwner, *entryOwner))
            preferred = n;
        }
      } else if (n->isRegion() && n->flow) {
        final = n;
        owner = n->flow->owner;
        if (entryOwner && sameOwner(owner, *entryOwner))
          preferred = n;
      }
    }
    if (preferred && final != preferred) {
      final = preferred;
      owner = *entryOwner;
    }
    if (!crossing.concreteSema && final) {
      if (final->kind == Node::Acquire)
        crossing.concreteSema = final->sema;
      else if (final->isRegion() && final->flow && final->flow->concreteSema)
        crossing.concreteSema = final->flow->concreteSema;
    }
    crossing.exits.push_back(final);
    if (!final) {
      if (entryOwner)
        joinOwner(*entryOwner);
      continue;
    }
    live = true;
    joinOwner(owner);
  }
  bool hasImplicitInputPath =
      region->kind == Node::For ||
      (region->kind == Node::If && region->children.size() < 2);
  if (hasImplicitInputPath && entryOwner)
    joinOwner(*entryOwner);
  if (!live || !compatible) {
    region->completionOwner.reset();
    return false;
  }
  CompletionSummary completion{true, false, false, {}};
  for (Node *child : region->children)
    completion =
        joinCompletion(completion, completionAfterChain(child, crossing.owner));
  if (hasImplicitInputPath)
    completion = joinCompletion(completion, {});
  region->completionOwner.emplace(crossing.owner);
  region->completion = completion;
  return true;
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
        (n->kind == Node::Access && nodeTouchesGroup(g, n)) || n->flow)
      return true;
  }
  return false;
}
static void pruneDeadIfFlows(GroupDag &g, Node *head, Node *region) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes))
    if (n->kind == Node::If && n->flow &&
        !tokenUsedBeforeNextAcquire(g, n->next) && (!region || !region->flow)) {
      n->flow.reset();
      n->completionOwner.reset();
    }
  for (Node *n : nodes)
    if (n->isRegion())
      for (Node *child : n->children)
        pruneDeadIfFlows(g, child, n);
}
static bool boundaryEvent(GroupDag &g, Node *n) {
  return n->isProtocol() ||
         (n->kind == Node::Access && nodeTouchesGroup(g, n)) ||
         (n->isRegion() && !n->pieceInfo.empty());
}
static bool hasBoundaryEvent(GroupDag &g, Node *head) {
  for (Node *n = head; n; n = n->next)
    if (boundaryEvent(g, n))
      return true;
  return false;
}
static bool transparentRegion(GroupDag &g, Node *region, const Owner &owner) {
  auto entry = uniformPieceOwner(region);
  if (!region->flow || !entry || !sameOwner(*entry, owner) ||
      !sameOwner(region->flow->owner, owner))
    return false;
  for (auto [i, child] : llvm::enumerate(region->children)) {
    Node *final =
        i < region->flow->exits.size() ? region->flow->exits[i] : nullptr;
    if (hasBoundaryEvent(g, final ? final->next : child))
      return false;
  }
  return true;
}
static bool canDropLoop(Node *loop) {
  for (Node *p = loop; p && p->kind != Node::Func; p = p->parent) {
    if (p->kind == Node::For && gpu::hasWarpSpecializeTag(p->op))
      return true;
    if (p != loop && p->kind == Node::If)
      return false;
  }
  return false;
}
struct Feed {
  Node *acquire = nullptr;
  bool retainedPrefix = false;
};
static std::optional<Feed> findFeed(GroupDag &g, Node *loop,
                                    const Owner &owner) {
  bool prefix = false;
  for (Node *cur = loop;; cur = cur->parent) {
    bool direct = cur == loop;
    for (Node *n = cur->prev; n; n = n->prev) {
      if (n->kind == Node::Acquire) {
        const Sema &s = getSema(g, n);
        Owner tokenOwner = n->owner ? n->owner : s.entryTokenOwner;
        return sameOwner(tokenOwner, owner)
                   ? std::optional<Feed>(Feed{n, prefix})
                   : std::nullopt;
      }
      if (n->kind == Node::Release || (n->isRegion() && !n->pieceInfo.empty()))
        return std::nullopt;
      if (n->kind == Node::Access && nodeTouchesGroup(g, n)) {
        bool reusable =
            direct && sameOwner(n->owner, owner) &&
            (!n->reuseTokenOwner || *n->reuseTokenOwner == ownerKey(owner)) &&
            asyncPayloadOf(n->op) == AsyncOp::NONE;
        if (!reusable)
          return std::nullopt;
        prefix = true;
      }
    }
    if (!cur->parent || cur->parent->kind != Node::For || !cur->parent->flow ||
        !sameOwner(cur->parent->flow->owner, owner))
      return std::nullopt;
  }
}
static bool resultConsumed(GroupDag &g, Node *region) {
  if (tokenUsedBeforeNextAcquire(g, region->next))
    return true;
  Node *p = region->parent;
  return p && p->flow && llvm::is_contained(p->flow->exits, region) &&
         resultConsumed(g, p);
}
static Node *matchDemand(GroupDag &g, Node *loop, Owner owner, Node *regain,
                         bool nested) {
  Node *first = nullptr;
  unsigned releases = 0;
  for (Node *n = loop->children[0]; n; n = n->next) {
    if ((nested && n == regain) || n->kind == Node::Acquire)
      break;
    if (n->isRegion() && !n->pieceInfo.empty()) {
      auto nestedOwner = uniformPieceOwner(n);
      if (!first || !nestedOwner || !sameOwner(*nestedOwner, owner) ||
          (n->flow && !transparentRegion(g, n, owner)))
        return nullptr;
      continue;
    }
    if (n->kind == Node::Access && nodeTouchesGroup(g, n) && !first) {
      if (releases)
        return nullptr;
      first = n;
    }
    if (n->kind == Node::Release)
      releases += std::max(1u, n->count);
  }
  return first && releases == (nested ? 0u : 1u) ? first : nullptr;
}
static void detach(Node *n) {
  if (n->prev)
    n->prev->next = n->next;
  else
    for (Node *&head : n->parent->children)
      if (head == n)
        head = n->next;
  if (n->next)
    n->next->prev = n->prev;
  n->prev = n->next = nullptr;
}
static void fixupProtocolArcs(GroupDag &g, Node *acquire, Node *candidate) {
  for (ProtocolArc &arc : g.protocolArcs) {
    if (arc.wait != acquire)
      continue;
    if (acquire->isLinked() && acquire->parent == arc.release->parent &&
        (precedesInChain(arc.release, acquire) ||
         getSema(g, arc.release).isEntry))
      continue;
    if (candidate && candidate->sema == arc.release->sema &&
        candidate->parent == arc.release->parent &&
        precedesInChain(arc.release, candidate))
      arc.wait = candidate;
    else
      arc.wait = nullptr;
  }
}
static Node *findBridge(GroupDag &g, Node *loop, SemaId sema, Owner owner) {
  for (Node *n = loop->next; n; n = n->next) {
    if (n->kind != Node::Acquire || n->sema != sema ||
        !sameOwner(n->owner, owner))
      continue;
    if (hasBoundaryEvent(g, n->next))
      return nullptr;
    return n;
  }
  return nullptr;
}
static bool authoredSingleCopy(GroupDag &g) {
  return llvm::all_of(g.pieceTable.members, [](const Member &m) {
    auto a = m.allocOp->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
    return a && a.getInt() == 1;
  });
}
static void planLoop(GroupDag &g, Node *loop) {
  RegionFlow &flow = *loop->flow;
  Node *regain = flow.exits.empty() ? nullptr : flow.exits.front();
  bool nested = regain && regain->isRegion();
  if (!canDropLoop(loop) || !regain || regain->postLoopAcquire ||
      (nested &&
       (!regain->flow || !transparentRegion(g, regain, flow.owner))) ||
      (!nested && regain->kind != Node::Acquire))
    return;
  if (hasBoundaryEvent(g, regain->next))
    return;
  auto feed = findFeed(g, loop, flow.owner);
  if (!feed || (feed->retainedPrefix && !authoredSingleCopy(g)))
    return;
  bool output = resultConsumed(g, loop);
  if (nested && output)
    return;
  Node *recurrence = nested ? feed->acquire : regain;
  bool needsPost = output || feed->retainedPrefix;
  Node *bridgeAcquire = feed->retainedPrefix ? feed->acquire : nullptr;
  if (!nested && feed->acquire->sema != recurrence->sema) {
    if (!needsPost ||
        !(bridgeAcquire = findBridge(g, loop, feed->acquire->sema, flow.owner)))
      return;
  }
  Node *demand = matchDemand(g, loop, flow.owner, regain, nested);
  if (!demand || (!gpu::hasWarpSpecializeTag(loop->op) &&
                  isa_and_nonnull<nvidia_gpu::TMEMAllocOp>(demand->op) &&
                  cast<nvidia_gpu::TMEMAllocOp>(demand->op).getSrc()))
    return;
  if (needsPost && !gpu::hasWarpSpecializeTag(loop->op) &&
      sameOwner(demand->owner, flow.owner)) {
    if (!loop->completion.valid)
      return;
    gpu::StageCluster point = gpu::getStageCluster(demand->op);
    if (point && loop->completion.schedule &&
        point->first != loop->completion.schedule->first)
      return;
  }
  Node *post = nullptr;
  if (needsPost) {
    post = newProtocolNode(g, Node::Acquire, loop->parent, flow.owner,
                           recurrence->sema, recurrence->count);
    post->postLoopAcquire = true;
    spliceAfter(post, loop);
    if (bridgeAcquire) {
      Node *bridge = newProtocolNode(
          g, Node::Release,
          bridgeAcquire == feed->acquire ? loop->parent : bridgeAcquire->parent,
          flow.owner, recurrence->sema, recurrence->count);
      bridge->payloads.push_back(AsyncOp::NONE);
      if (bridgeAcquire == feed->acquire) {
        markTokenReuse(bridge, flow.owner);
        spliceBefore(bridge, loop);
      } else {
        appendAfterReleases(bridge, bridgeAcquire);
      }
      g.protocolArcs.push_back(
          {bridge, recurrence, bridgeAcquire, demand, nullptr});
    }
  }
  if (nested)
    recurrence->owner = flow.owner;
  detach(recurrence);
  spliceBefore(recurrence, demand);
  for (ProtocolArc &arc : g.protocolArcs)
    if (arc.acquire == recurrence)
      arc.consumer = demand;
  if (nested) {
    Node *close = newProtocolNode(g, Node::Release, regain->parent, flow.owner,
                                  recurrence->sema, 1);
    close->payloads.push_back(AsyncOp::NONE);
    appendAfterReleases(close, regain);
    g.protocolArcs.push_back({close, recurrence, regain, demand, recurrence});
  } else if (bridgeAcquire) {
    Sema &s = getSema(g, recurrence);
    s.isEntry = true;
    s.entryTokenOwner = flow.owner;
  } else {
    detach(feed->acquire);
    fixupProtocolArcs(g, feed->acquire, post);
  }
  fixupProtocolArcs(g, recurrence, post);
  loop->flow.reset();
}
static void planRegionFlows(GroupDag &g, Node *head) {
  SmallVector<Node *, 8> nodes;
  for (Node *n = head; n; n = n->next)
    nodes.push_back(n);
  for (Node *n : llvm::reverse(nodes)) {
    if (!n->isRegion())
      continue;
    for (Node *child : n->children)
      planRegionFlows(g, child);
    if (n->flow) {
      RegionFlow updated;
      if (summarizeRegionFlow(g, n, updated))
        *n->flow = std::move(updated);
      else
        n->flow.reset();
    }
    if (n->kind == Node::For && n->flow)
      planLoop(g, n);
  }
}
static void addPart(SmallVectorImpl<int> &parts, int part) {
  if (!llvm::is_contained(parts, part))
    parts.push_back(part);
}
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
static bool canDoubleBufferAcc(nvidia_gpu::MMAv5OpInterface mmaOp,
                               int numTmemBlocks) {
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
    auto copyAttr = m.allocOp->getAttrOfType<IntegerAttr>(kBufferCopyAttrName);
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
static LogicalResult computeBackingCopies(GroupDag &g, ArrayRef<EdgeRec> edges,
                                          bool useMetaPartitioner,
                                          int &numTmemBlocks) {
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
static void computeSemaphoreCopies(GroupDag &g, int lowerSemaphoreNumStages) {
  g.numSemaphoreCopies = g.numCopies;
  bool authored =
      g.pieceTable.members.front().allocOp->hasAttr(kBufferCopyAttrName);
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release &&
        llvm::is_contained(node->payloads, AsyncOp::TMALoad))
      hasProducerLoad = true;
  });
  if (g.isLocal() && !authored && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
}

// Schedule finalization. Doc: sync-dag.md#pipeline-schedule
using ScheduleEdge = std::pair<Operation *, Operation *>;
struct OwnerScheduleConstraint {
  Owner producerOwner, consumerOwner;
  Operation *producer = nullptr, *consumer = nullptr;
  int64_t producerStage = 0, consumerStage = 0, distance = 0;
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
  DenseMap<Node *, int64_t> ordinal;
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
using PhysicalSets = llvm::MapVector<PhysicalKey, SmallVector<GroupDag *, 2>>;
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
static SlotSchedule replaySlots(ArrayRef<SlotEvent> events,
                                bool assignOffsets = false) {
  SlotSchedule result;
  DenseMap<GroupDag *, int64_t> lastProduced;
  int64_t cursor = -1;
  for (const SlotEvent &event : events) {
    int64_t required;
    if (slotEventEffect(event.node) == Effect::W) {
      cursor += event.advances;
      result.advancesPerIteration += event.advances;
      if (cursor < 0) {
        result.complete = false;
        continue;
      }
      required = lastProduced[event.group] = cursor;
    } else {
      auto it = lastProduced.find(event.group);
      if (it == lastProduced.end()) {
        result.complete = false;
        continue;
      }
      required = it->second;
    }
    result.ordinal[event.node] = required;
    if (assignOffsets)
      event.node->bufferStageOffset = required - cursor;
  }
  return result;
}

static LogicalResult assignCircularStageOffsets(PhysicalSets &physicalSets) {
  for (auto &[_, physicalSet] : physicalSets) {
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
        return semaError(g->root->op)
               << "malformed circular local logical group";
      const Member &member = g->pieceTable.members.front();
      if (member.type != type)
        return semaError(member.allocOp)
               << "circular local group has mismatched member types";
      if (g->numCopies != numCopies)
        return semaError(member.allocOp)
               << "circular local group has mismatched buffer.copy";
      if (member.circularStart < 0 || member.circularStart >= numCopies)
        return semaError(member.allocOp)
               << "circular buffer.start is outside buffer.copy";
      if (!starts.insert(member.circularStart).second)
        return semaError(member.allocOp)
               << "duplicate circular buffer.start in one group";
      for (const std::unique_ptr<Node> &storage : g->nodes) {
        Node *n = storage.get();
        if (n->kind == Node::Access)
          eventsByOp[n->op].push_back(
              SlotEvent{g, n, slotEventEffect(n) == Effect::W});
      }
    }
    SmallVector<SlotEvent> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end())
        ordered.append(it->second.begin(), it->second.end());
    });
    SlotSchedule slots = replaySlots(ordered, /*assignOffsets=*/true);
    for (const SlotEvent &event : ordered) {
      const Member &member = event.group->pieceTable.members.front();
      if (event.advances) {
        int64_t stage = slots.ordinal.lookup(event.node);
        if (member.circularStart != stage % numCopies)
          return semaError(member.allocOp)
                 << "circular producer order expects buffer.start "
                 << stage % numCopies << ", got " << member.circularStart;
      } else if (!slots.ordinal.contains(event.node)) {
        return semaError(member.allocOp)
               << "circular consumer appears before producer";
      }
    }
    for (GroupDag *g : set) {
      for (const std::unique_ptr<Node> &storage : g->nodes) {
        Node *n = storage.get();
        bool forward = n->kind == Node::Acquire;
        if (!forward && n->kind != Node::Release)
          continue;
        Node *access = n;
        do
          access = forward ? access->next : access->prev;
        while (access &&
               (access->kind != Node::Access || !access->bufferStageOffset));
        if (access)
          n->stageOffset = access->bufferStageOffset;
      }
    }
  }
  return success();
}

static Operation *findScheduleAnchor(const Node *anchor,
                                     bool producer = false) {
  for (const Node *n = anchor; n; n = producer ? n->prev : n->next) {
    if (n->kind == Node::Access)
      return producer && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion() && n->op)
      return n->op;
  }
  return nullptr;
}
static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet,
                                        scf::ForOp loop) {
  SmallVector<SlotEvent, 8> events;
  for (GroupDag *group : physicalSet) {
    for (const std::unique_ptr<Node> &storage : group->nodes) {
      Node *node = storage.get();
      if (!node->isDirectLoopNode() || node->parent->op != loop.getOperation())
        continue;
      if (!node->isSlotEvent() || !node->op)
        continue;
      unsigned advances =
          slotEventEffect(node) == Effect::W &&
          llvm::any_of(group->protocolArcs, [&](const auto &arc) {
            return arc.consumer == node && arc.acquire->isDirectLoopNode() &&
                   arc.acquire->parent->op == loop.getOperation();
          });
      events.push_back(SlotEvent{group, node, advances});
    }
  }
  llvm::stable_sort(events, [](const SlotEvent &lhs, const SlotEvent &rhs) {
    return lhs.node->op != rhs.node->op &&
           lhs.node->op->isBeforeInBlock(rhs.node->op);
  });
  return replaySlots(events);
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}
static std::optional<int64_t>
computeLoopCarriedDistance(const SlotSchedule &slots,
                           int64_t numSemaphoreCopies, Node *producer,
                           Node *consumer) {
  if (numSemaphoreCopies == 1)
    return 1; // one slot: a loop-carried pair spans exactly one iteration
  auto producerIt = slots.ordinal.find(producer);
  auto consumerIt = slots.ordinal.find(consumer);
  if (!slots.complete || producerIt == slots.ordinal.end() ||
      consumerIt == slots.ordinal.end() || slots.advancesPerIteration <= 0)
    return std::nullopt;
  int64_t orbit = numSemaphoreCopies /
                  std::gcd(numSemaphoreCopies, slots.advancesPerIteration);
  for (int64_t distance = 1; distance <= orbit; ++distance)
    if (positiveMod(consumerIt->second + distance * slots.advancesPerIteration,
                    numSemaphoreCopies) ==
        positiveMod(producerIt->second, numSemaphoreCopies))
      return distance;
  return std::nullopt;
}
static LogicalResult
assignAliasedHandoffStageOffsets(PhysicalSets &physicalSets, GroupDag &group) {
  if (group.isCircular() || group.pieceTable.members.size() < 2 ||
      group.numSemaphoreCopies <= 1)
    return success();
  bool authored = group.numCopies > 1 &&
                  llvm::all_of(group.pieceTable.members, [](const Member &m) {
                    return m.allocOp->hasAttr(kBufferCopyAttrName);
                  });
  const Member &first = group.pieceTable.members.front();
  bool exactAlias =
      llvm::all_of(group.pieceTable.members, [&](const Member &member) {
        return member.offset == first.offset && member.extent == first.extent &&
               member.type == first.type;
      });
  if (!authored && (group.numSemaphoreCopies <= group.numCopies || !exactAlias))
    return success();
  bool hasShiftedRelease = false;
  for (const ProtocolArc &arc : group.protocolArcs) {
    if (!arc.wait || !arc.release->isDirectLoopNode())
      continue;
    Node *producer = arc.producer;
    Node *consumer = arc.wait == arc.acquire ? arc.consumer : nullptr;
    if (!producer->isSlotEvent() || !consumer || !consumer->isSlotEvent() ||
        producer->parent != consumer->parent ||
        producer->parent->kind != Node::For)
      return semaError(producer->op)
             << "multibuffered alias handoff requires direct scheduled events "
                "in one loop body";
    auto loop = cast<scf::ForOp>(producer->parent->op);
    SlotSchedule slots =
        computeSlotSchedule(physicalSets[physicalKey(group)], loop);
    auto producerIt = slots.ordinal.find(producer);
    auto consumerIt = slots.ordinal.find(consumer);
    if (!slots.complete || producerIt == slots.ordinal.end() ||
        consumerIt == slots.ordinal.end() || slots.advancesPerIteration <= 0)
      return semaError(producer->op)
             << "cannot derive multibuffered alias handoff slots";
    int64_t numSemaphoreCopies = group.numSemaphoreCopies;
    int64_t offset = 0;
    if (precedesInChain(arc.release, arc.wait)) {
      offset = positiveMod(consumerIt->second - producerIt->second,
                           numSemaphoreCopies);
    } else if (!computeLoopCarriedDistance(slots, numSemaphoreCopies, producer,
                                           consumer)) {
      int64_t nextConsumer = consumerIt->second + slots.advancesPerIteration;
      offset =
          positiveMod(nextConsumer - producerIt->second, numSemaphoreCopies);
    }
    arc.release->stageOffset = offset;
    hasShiftedRelease |= offset != 0;
  }
  for (const auto &storage : group.nodes) {
    Node *node = storage.get();
    if (!node->isDirectLoopNode())
      continue;
    if (hasShiftedRelease && node->kind == Node::Acquire)
      node->stageOffset = 0;
    if (!hasShiftedRelease && node->kind == Node::Release)
      node->stageOffset.reset();
  }
  return success();
}

static LogicalResult solveOwnerScheduleConstraints(LoopScheduleModel &model) {
  auto &constraints = model.ownerConstraints;
  DenseMap<int64_t, int64_t> offset;
  for (const OwnerScheduleConstraint &constraint : constraints) {
    offset.try_emplace(ownerKey(constraint.producerOwner), 0);
    offset.try_emplace(ownerKey(constraint.consumerOwner), 0);
  }
  DenseMap<int64_t, unsigned> predecessor;
  const unsigned numVertices = offset.size();
  std::optional<unsigned> lastUpdated;
  for (unsigned iteration = 0; iteration < numVertices; ++iteration) {
    lastUpdated.reset();
    for (auto [edgeIndex, constraint] : llvm::enumerate(constraints)) {
      int64_t producer = ownerKey(constraint.producerOwner);
      int64_t consumer = ownerKey(constraint.consumerOwner);
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
    int64_t vertex = ownerKey(constraints[*lastUpdated].consumerOwner);
    for (unsigned i = 0; i < numVertices; ++i)
      vertex = ownerKey(constraints[predecessor.lookup(vertex)].producerOwner);
    SmallVector<unsigned, 4> cycle;
    int64_t cycleStart = vertex;
    do {
      unsigned edgeIndex = predecessor.lookup(vertex);
      cycle.push_back(edgeIndex);
      vertex = ownerKey(constraints[edgeIndex].producerOwner);
    } while (vertex != cycleStart);
    int64_t cycleDelay = 0;
    for (unsigned edgeIndex : cycle)
      cycleDelay += constraints[edgeIndex].requiredDelay();
    const OwnerScheduleConstraint &first = constraints[cycle.front()];
    InFlightDiagnostic diag = semaError(first.producer)
                              << "fixed loop.stage assignments form an "
                                 "unsatisfiable semaphore handoff cycle";
    diag << " (cycle requires " << cycleDelay
         << " additional pipeline iteration" << (cycleDelay == 1 ? "" : "s")
         << ")";
    for (unsigned edgeIndex : llvm::reverse(cycle)) {
      const OwnerScheduleConstraint &constraint = constraints[edgeIndex];
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
    return offset[ownerKey(constraint.consumerOwner)] ==
           offset[ownerKey(constraint.producerOwner)] +
               constraint.requiredDelay();
  };
  auto hasTightPath = [&](int64_t from, int64_t to) {
    SmallVector<int64_t, 8> stack{from};
    DenseSet<int64_t> seen;
    while (!stack.empty()) {
      int64_t vertex = stack.pop_back_val();
      if (vertex == to)
        return true;
      if (!seen.insert(vertex).second)
        continue;
      for (const OwnerScheduleConstraint &constraint : constraints)
        if (isTight(constraint) && ownerKey(constraint.producerOwner) == vertex)
          stack.push_back(ownerKey(constraint.consumerOwner));
    }
    return false;
  };
  for (const OwnerScheduleConstraint &constraint : constraints) {
    if (constraint.requiredDelay() == 0 ||
        (isTight(constraint) &&
         hasTightPath(ownerKey(constraint.consumerOwner),
                      ownerKey(constraint.producerOwner))))
      model.clusterEdges.emplace_back(constraint.producer, constraint.consumer);
  }
  return success();
}

static LogicalResult legalizeLoopSchedule(scf::ForOp loop,
                                          ArrayRef<ScheduleEdge> edges) {
  SmallVector<ScheduleEdge> constraints(edges);
  DenseMap<Operation *, int64_t> cluster;
  for (Operation &consumer : loop.getBody()->without_terminator()) {
    gpu::StageCluster after = gpu::getStageCluster(&consumer);
    if (!after)
      continue;
    cluster[&consumer] = after->second;
    for (Value operand : getNestedOperands(&consumer)) {
      auto [producer, distance] =
          triton::getDefiningOpAndDistance(loop, operand);
      if (!producer)
        continue;
      producer = loop.getBody()->findAncestorOpInBlock(*producer);
      if (!producer || producer == &consumer)
        continue;
      gpu::StageCluster before = gpu::getStageCluster(producer);
      if (before && before->first == after->first + distance)
        constraints.emplace_back(producer, &consumer);
    }
  }
  bool changed = false;
  for (unsigned iteration = 0; iteration <= cluster.size(); ++iteration) {
    changed = false;
    for (const ScheduleEdge &edge : constraints) {
      auto [producer, consumer] = edge;
      if (!cluster.contains(producer) || !cluster.contains(consumer))
        continue;
      int64_t required = cluster.lookup(producer) +
                         (producer->isBeforeInBlock(consumer) ? 0 : 1);
      if (cluster.lookup(consumer) >= required)
        continue;
      cluster[consumer] = required;
      changed = true;
    }
    if (!changed)
      break;
    if (iteration == cluster.size())
      return semaError(loop) << "cyclic loop.cluster constraints";
  }
  OpBuilder builder(loop.getContext());
  for (Operation &op : loop.getBody()->without_terminator()) {
    gpu::StageCluster oldSchedule = gpu::getStageCluster(&op);
    if (!oldSchedule)
      continue;
    int64_t newCluster = cluster.lookup(&op);
    if (newCluster == oldSchedule->second)
      continue;
    if (newCluster > std::numeric_limits<int32_t>::max())
      return semaError(&op) << "legalized loop.cluster exceeds i32 range";
    gpu::setStageCluster(
        builder, &op,
        std::make_pair(oldSchedule->first, static_cast<int>(newCluster)));
  }
  return success();
}
static gpu::StageCluster scheduleAtOwnerBoundary(const Node *n,
                                                 gpu::StageCluster schedule) {
  if (!schedule || findScheduleAnchor(n->next))
    return schedule;
  auto forOp =
      dyn_cast_or_null<scf::ForOp>(n->parent ? n->parent->op : nullptr);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName))
    return schedule;
  auto [stage, cluster] = *schedule;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate || candidate->first != stage || !gpu::hasPartition(&op))
      continue;
    if (gpu::getPartitionIds(&op).contains(n->owner->first))
      cluster = std::max(cluster, candidate->second);
  }
  return std::make_pair(stage, cluster);
}
using OwnerSchedules = DenseMap<int64_t, gpu::StageCluster>;
static void assignSyncSchedules(Node *head, OwnerSchedules &cache) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire && n->owner) {
      Operation *anchor =
          n->postLoopAcquire ? nullptr : findScheduleAnchor(n->next);
      n->stageCluster =
          anchor ? gpu::getStageCluster(anchor)
                 : scheduleAtOwnerBoundary(n, cache.lookup(ownerKey(n->owner)));
    } else if (n->kind == Node::Release && n->owner) {
      n->stageCluster = cache.lookup(ownerKey(n->owner));
    } else if (n->kind == Node::Access && n->owner) {
      Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
      cache[ownerKey(n->owner)] = gpu::getStageCluster(completion);
    } else if (n->kind == Node::For) {
      OwnerSchedules body = cache;
      assignSyncSchedules(n->children[0], body);
      if (!gpu::hasWarpSpecializeTag(n->op))
        cache = std::move(body);
    } else if (n->kind == Node::If) {
      OwnerSchedules thenCache = cache, elseCache = cache;
      assignSyncSchedules(n->children[0], thenCache);
      if (n->children.size() > 1 && n->children[1])
        assignSyncSchedules(n->children[1], elseCache);
      cache = std::move(thenCache);
      for (auto [owner, schedule] : elseCache)
        cache.try_emplace(owner, schedule);
    }
  }
}
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, LoopScheduleModel> modelsByLoop;
  PhysicalSets physicalSets;
  for (GroupDag &group : groups)
    physicalSets[physicalKey(group)].push_back(&group);
  if (failed(assignCircularStageOffsets(physicalSets)))
    return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(physicalSets, group)))
      return failure();
  for (GroupDag &group : groups)
    for (const ProtocolArc &arc : group.protocolArcs) {
      if (!arc.wait || !arc.release->isDirectLoopNode())
        continue;
      Operation *source = findScheduleAnchor(arc.producer, true);
      Operation *destination =
          arc.wait == arc.acquire ? findScheduleAnchor(arc.consumer) : nullptr;
      if (!source || !destination)
        continue;
      for (Operation *parent = source->getParentOp(); parent;
           parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName))
          continue;
        Operation *producer = loop.getBody()->findAncestorOpInBlock(*source);
        Operation *consumer =
            loop.getBody()->findAncestorOpInBlock(*destination);
        if (!producer || !consumer)
          continue;
        if (producer == consumer)
          break;
        gpu::StageCluster before = gpu::getStageCluster(producer);
        gpu::StageCluster after = gpu::getStageCluster(consumer);
        if (!before || !after)
          break;
        int64_t distance = 0;
        if (!precedesInChain(arc.release, arc.wait)) {
          SlotSchedule slots =
              computeSlotSchedule(physicalSets[physicalKey(group)], loop);
          std::optional<int64_t> carried = computeLoopCarriedDistance(
              slots, group.numSemaphoreCopies, arc.producer, arc.consumer);
          if (!carried) {
            InFlightDiagnostic diag =
                semaError(producer)
                << "cannot determine loop-carried dependency distance for a "
                   "physical buffer slot";
            diag.attachNote(consumer->getLoc())
                << "next token ownership starts here";
            return failure();
          }
          distance = *carried;
        }
        modelsByLoop[loop.getOperation()].ownerConstraints.push_back(
            {arc.release->owner, arc.wait->owner, producer, consumer,
             before->first, after->first, distance});
        break;
      }
    }
  for (auto &[loopOp, model] : modelsByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    if (failed(solveOwnerScheduleConstraints(model)))
      return failure();
    if (model.clusterEdges.empty())
      continue;
    if (failed(legalizeLoopSchedule(loop, model.clusterEdges)))
      return failure();
  }
  for (GroupDag &g : groups) {
    OwnerSchedules cache;
    assignSyncSchedules(
        g.root->children.empty() ? nullptr : g.root->children[0], cache);
  }
  return success();
}

LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner,
                           int lowerSemaphoreNumStages, int &numTmemBlocks) {
  SmallVector<EdgeRec> edges;
  if (!g.root->children.empty()) {
    ChainState top; // function chain: games start at bottom (first-touch)
    ChainWalker(g, top, edges, /*underFor=*/false).run(g.root->children[0]);
  }
  if (failed(computeBackingCopies(g, edges, useMetaPartitioner, numTmemBlocks)))
    return failure();
  if (failed(buildEdgesAndSemas(g, edges)))
    return failure();
  if (failed(insertEntryAcquires(g)))
    return failure();
  if (!g.root->children.empty()) {
    Node *head = g.root->children[0];
    buildRegionFlows(g, head);
    planRegionFlows(g, head);
    pruneDeadIfFlows(g, head, /*region=*/nullptr);
    computeRequiredParts(head);
  }
  computeSemaphoreCopies(g, lowerSemaphoreNumStages);
  if (!g.semas.empty())
    for (Operation *alloc : g.ttDescriptorFedMembers)
      return semaError(alloc)
             << "managed local_alloc sourced from a tt-form descriptor load — "
                "nvws-insert-allocas must convert this upstream "
                "(pipeline invariant violated)";
  return success();
}

} // namespace mlir::triton::nvws_semas
