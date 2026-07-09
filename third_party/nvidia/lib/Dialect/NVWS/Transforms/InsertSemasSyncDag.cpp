// SYNC analysis and scheduling; section links refer to sync-dag.md.
#include "InsertSemas.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

namespace mlir::triton::nvws_semas {
using Payloads = SmallVector<AsyncOp, 1>;
using PieceEffects = std::map<PieceId, Effect>;
using OrderFacts = SmallVector<int64_t, 2>;
using ExitFacts = std::map<PieceId, OrderFacts>;
static void dumpDagTree(GroupDag &g);
// Doc: sync-dag.md#the-walk-accesses-to-edges
struct ActiveUse {
  Owner owner;
  Node *node = nullptr;
  Payloads payloads;
  SmallVector<int64_t, 2> orderedBefore; // Owners already ordered after node.
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
    auto it = llvm::find_if(uses, [&](const ActiveUse &use) { return sameOwner(use.owner, owner); });
    return it == uses.end() ? nullptr : &*it;
  }
  void startVersion(const Owner &producer, const Owner &sourceOwner, Node *node, const Payloads &payloads) {
    source = VersionSource{producer, sourceOwner, node, payloads};
    uses.assign(1, ActiveUse{sourceOwner, node, payloads, {}});
  }
};
using ChainState = std::map<PieceId, PieceState>;
struct Tokens {
  struct Token {
    Owner owner;
    Node *producer = nullptr;
    Node *last = nullptr;
    Payloads payloads;
    Node *closedBy = nullptr;
  };
  SmallVector<Token, 2> live;
  const Token *find(const Owner &owner) const {
    auto it = llvm::find_if(live, [&](const Token &token) { return sameOwner(token.owner, owner); });
    return it == live.end() ? nullptr : &*it;
  }
  const Token *last() const {
    auto it = llvm::find_if(llvm::reverse(live), [](const Token &token) { return token.producer && !token.closedBy; });
    return it == live.rend() ? nullptr : &*it;
  }
  Token *find(const Owner &owner) { return const_cast<Token *>(std::as_const(*this).find(owner)); }
  const Token *findOpen(const Owner &owner) const {
    const Token *token = find(owner);
    return token && !token->closedBy ? token : nullptr;
  }
  const Token *findProducer(Node *producer) const {
    auto it = llvm::find_if(live, [&](const Token &token) { return token.producer == producer; });
    return it == live.end() ? nullptr : &*it;
  }
  void record(const Owner &owner, Node *producer, Node *last, const Payloads &payloads) {
    llvm::erase_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner) || (owner.has_value() && !token.owner.has_value());
    });
    live.push_back(Token{owner, producer, last, payloads, nullptr});
  }
  void eraseOwner(const Owner &owner) { llvm::erase_if(live, [&](const Token &t) { return sameOwner(t.owner, owner); }); }
  void eraseProducer(Node *producer) {
    assert(producer && "cannot erase a token without its exact producer");
    llvm::erase_if(live, [&](const Token &t) { return t.producer == producer; });
  }
  void close(Node *producer, Node *release) {
    for (Token &token : live)
      if (token.producer == producer && !token.closedBy) token.closedBy = release;
  }
};
struct EdgeRec {
  Node *src = nullptr, *dst = nullptr;
  Owner srcOwner, dstOwner;
  Payloads payloads;
  SmallVector<PieceId, 2> pieces;
};
using EdgeList = SmallVector<EdgeRec>;
static void unionPayloads(Payloads &into, const Payloads &from) {
  for (AsyncOp payload : from)
    if (!llvm::is_contained(into, payload)) into.push_back(payload);
  llvm::sort(into);
}
static SmallVector<int64_t, 2> intersectOrderFacts(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t, 2> result;
  for (int64_t owner : lhs)
    if (llvm::is_contained(rhs, owner)) result.push_back(owner);
  return result;
}
static bool knownNonEmptyLoop(Node *node) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(node->op);
  if (!forOp) return false;
  if (forOp->hasAttr("ttg.must-execute")) return true;
  std::optional<APInt> count = forOp.getStaticTripCount();
  return count && (forOp.getUnsignedCmp() ? count->ugt(0) : count->sgt(0));
}
static void raiseForeignUseEdges(PieceState &piece, PieceId id, const Owner &owner, Node *node, EdgeList &edges, bool wsAdopt) {
  for (const ActiveUse &use : piece.uses)
    if (!sameOwner(use.owner, owner) && !(wsAdopt && !use.owner) &&
        !llvm::is_contained(use.orderedBefore, ownerKey(owner)))
      edges.push_back(
          EdgeRec{use.node, node, use.owner, owner, use.payloads, {id}});
}
// Doc: sync-dag.md#the-per-access-rules-in-full
static void applyTouch(PieceState &piece, PieceId id, const Owner &owner, Effect effect, Node *node, const Payloads &payloads, EdgeList &edges, bool wsAdopt) {
  if (!piece.initialized() || effect == Effect::W) {
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
    edges.push_back(EdgeRec{piece.source.node, node, piece.source.sourceOwner, owner, piece.source.payloads, {id}});
    if (ActiveUse *source = piece.useFor(piece.source.sourceOwner);
        source && source->node == piece.source.node &&
        !llvm::is_contained(source->orderedBefore, ownerKey(owner)))
      source->orderedBefore.push_back(ownerKey(owner));
  }
  piece.uses.push_back(ActiveUse{owner, node, payloads, {}});
}
static bool canReuseTokenForPiece(ChainState &state, PieceId id, const Owner &owner, Effect effect) {
  auto it = state.find(id);
  if (it == state.end() || !it->second.initialized()) return false;
  PieceState &piece = it->second;
  if (effect == Effect::R) return piece.useFor(owner);
  return llvm::all_of(piece.uses, [&](const ActiveUse &use) {
    return sameOwner(use.owner, owner) || llvm::is_contained(use.orderedBefore, ownerKey(owner));
  });
}
static bool pieceTouchedAfter(GroupDag &g, Node *region, PieceId piece) {
  for (Node *scope = region; scope && scope->kind != Node::Func; scope = scope->parent)
    for (Node *node = scope->next; node; node = node->next)
      if ((node->kind == Node::Access && touchesPiece(g, node, piece)) ||
          (node->isRegion() && node->pieceInfo.count(piece)))
        return true;
  return false;
}

// Doc: sync-dag.md#the-walk-accesses-to-edges
class ChainWalker {
public:
  ChainWalker(GroupDag &group, ChainState &state, EdgeList &edges, DenseSet<Node *> &reusable, bool underFor)
      : group(group), state(state), edges(edges), reusable(reusable), underFor(underFor) {}
  ExitFacts run(Node *head) {
    if (head->kind == Node::Enter)
      for (auto [piece, info] : sortedPieceInfo(head))
        if (info.owner && !chainTokens.find(info.owner)) chainTokens.record(info.owner, head, head, {AsyncOp::NONE});
    for (Node *node = head; node; node = node->next) {
      switch (node->kind) {
      case Node::Access: visitAccess(node); break;
      case Node::For: case Node::If: visitRegion(node); break;
      case Node::Exit: visitExit(node); break;
      default: break;
      }
    }
    ExitFacts result;
    if (head->kind == Node::Enter)
      for (auto [id, info] : sortedPieceInfo(head)) {
        OrderFacts facts;
        auto it = state.find(id);
        if (it != state.end())
          if (ActiveUse *use = it->second.useFor(info.owner))
            facts = use->orderedBefore;
        result.emplace(id, std::move(facts));
      }
    return result;
  }
private:
  // Doc: sync-dag.md#memory-edges-and-token-supply
  void visitAccess(Node *node) {
    PieceEffects effects;
    forEachTouchedPiece(group, node, [&](PieceId id, Effect effect) {
      mergeEffect(effects, id, effect);
    });
    const Tokens::Token *last = chainTokens.last();
    bool ownerDiffers = last && node->owner && !sameOwner(last->owner, node->owner);
    bool canReuse = chainTokens.find(node->owner) && llvm::all_of(effects, [&](const auto &item) {
                      return canReuseTokenForPiece( state, item.first, node->owner, item.second);
                    });
    size_t edgeStart = edges.size();
    Payloads payloads{asyncPayloadOf(node->op)};
    // Preserve prior async completions in a same-owner synchronous write wave.
    bool synchronousWrite = payloads.front() == AsyncOp::NONE;
    if (group.pieceTable.members.size() > 1 && canReuse && synchronousWrite) {
      for (auto [id, effect] : effects) {
        if (effect != Effect::W) continue;
        PieceState &piece = state[id];
        if (!sameOwner(piece.source.sourceOwner, node->owner)) continue;
        bool sameOwnerWave = llvm::all_of(piece.uses, [&](const ActiveUse &use) {
          return sameOwner(use.owner, node->owner);
        });
        if (sameOwnerWave) unionPayloads(payloads, piece.source.payloads);
      }
    }
    for (auto [id, effect] : effects)
      applyTouch(state[id], id, node->owner, effect, node, payloads, edges, /*wsAdopt=*/false);
    bool noDataEdge = edges.size() == edgeStart;
    bool reusesToken = noDataEdge && canReuse;
    if (reusesToken) {
      reusable.insert(node);
    } else if (ownerDiffers && noDataEdge) {
      edges.push_back(EdgeRec{last->last, node, last->owner, node->owner, last->payloads, {}});
    }
    if (node->owner && !(ownerDiffers && reusesToken)) chainTokens.record(node->owner, node, node, payloads);
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
        if (ActiveUse *use = it->second.useFor(info.owner)) incomingOrder[id] = use->orderedBefore;
      }
    }
    Payloads none{AsyncOp::NONE};
    bool wsAdopt = node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [id, info] : infos) applyTouch(state[id], id, info.owner, info.effect, node, none, edges, wsAdopt);
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
      ChainWalker nested(group, child, edges, reusable, underFor || node->kind == Node::For);
      ExitFacts childFacts = nested.run(childHead);
      for (auto [id, info] : infos) {
        SmallVector<int64_t, 2> branchOrder = incomingOrder[id];
        if (auto it = childFacts.find(id); it != childFacts.end())
          branchOrder = it->second;
        if (firstChild) returned[id] = std::move(branchOrder);
        else
          returned[id] = intersectOrderFacts(returned[id], branchOrder);
      }
      firstChild = false;
    }
    if (node->kind == Node::For && !knownNonEmptyLoop(node))
      for (auto [id, info] : infos)
        returned[id] = intersectOrderFacts(returned[id], incomingOrder[id]);
    for (auto [id, info] : infos) {
      auto facts = returned.find(id);
      ActiveUse *use = state[id].useFor(info.owner);
      if (facts == returned.end() || !use) continue;
      if (use->node == node)
        for (int64_t owner : facts->second)
          if (!llvm::is_contained(use->orderedBefore, owner)) use->orderedBefore.push_back(owner);
    }
    chainTokens.live.clear();
    if (auto owner = uniformPieceOwner(node); owner && owner->has_value())
      chainTokens.record(*owner, node, node, {AsyncOp::NONE});
  }
  void visitExit(Node *node) {
    for (auto [id, info] : sortedPieceInfo(node)) {
      auto it = state.find(id);
      if (it == state.end()) continue;
      PieceState &piece = it->second;
      if (underFor || pieceTouchedAfter(group, node->parent, id))
        raiseForeignUseEdges(piece, id, info.owner, node, edges,
                             /*wsAdopt=*/false);
      ActiveUse carried{info.owner, node, {AsyncOp::NONE}, {}};
      if (ActiveUse *use = piece.useFor(info.owner)) carried = *use;
      piece.uses.assign(1, carried);
    }
  }
  GroupDag &group;
  ChainState &state;
  EdgeList &edges;
  DenseSet<Node *> &reusable;
  bool underFor;
  Tokens chainTokens;
};
static void spliceBefore(Node *node, Node *before) {
  node->parent = before->parent;
  node->prev = before->prev;
  node->next = before;
  if (before->prev) before->prev->next = node;
  else if (node->parent) // chain head: repoint the parent's children slot
    for (Node *&slot : node->parent->children)
      if (slot == before) slot = node;
  before->prev = node;
}
static void spliceAfter(Node *node, Node *after) {
  node->parent = after->parent;
  node->next = after->next;
  node->prev = after;
  if (after->next) after->next->prev = node;
  after->next = node;
}
static Node *newProtocolNode(GroupDag &g, Node::Kind kind, Node *parent, Owner owner) {
  Node *node = g.newNode(kind, nullptr, parent);
  node->owner = owner;
  node->count = kind == Node::Release;
  return node;
}
namespace {
using SyncVec = std::map<int64_t, unsigned>; // partitionKey -> node index
using EdgeBuckets = DenseMap<Node *, SmallVector<unsigned, 2>>;
using Snapshots = DenseMap<Node *, SyncVec>;
using Positions = DenseMap<Node *, unsigned>;
struct KnownOrder {
  std::map<int64_t, SyncVec> behind;
  void apply(const EdgeRec &edge, unsigned sourceIdx, const Snapshots &snapshots) {
    SyncVec &known = behind[ownerKey(edge.dstOwner)];
    if (auto it = snapshots.find(edge.src); it != snapshots.end())
      for (auto [owner, idx] : it->second) known[owner] = std::max(known[owner], idx);
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
    if (known == behind.end()) return false;
    auto source = known->second.find(ownerKey(edge.srcOwner));
    return source != known->second.end() && source->second >= sourceIdx;
  }
};

static bool isLoopClose(const EdgeRec &edge) {
  return edge.dst->kind == Node::Exit && edge.src->kind == Node::Access && edge.srcOwner && edge.dstOwner;
}
static ArrayRef<unsigned> edgesAt(const EdgeBuckets &buckets, Node *node) {
  auto it = buckets.find(node);
  return it == buckets.end() ? ArrayRef<unsigned>{} : it->second;
}
// Doc: sync-dag.md#1-implied-ordering-reduceedges. Drops use only kept edges.
static void reduceStraightEdges(Node *head, const Positions &positions, ArrayRef<EdgeRec> edges,
                                const EdgeBuckets &atDst, std::vector<bool> &drop, DenseSet<Node *> &reusable) {
  KnownOrder order;
  Snapshots snapshots;
  std::optional<int64_t> tokenOwner;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner) tokenOwner = ownerKey(pi.owner);
  for (Node *n = head; n; n = n->next) {
    for (unsigned ei : edgesAt(atDst, n)) {
      const EdgeRec &e = edges[ei];
      if (e.srcOwner && e.dstOwner && e.src->kind == Node::Access) {
        int64_t dk = ownerKey(e.dstOwner);
        unsigned srcIdx = positions.lookup(e.src);
        bool hasToken = tokenOwner == dk;
        if (order.covers(e, srcIdx) && hasToken &&
            e.dst->kind == Node::Access) {
          drop[ei] = true;
          reusable.insert(e.dst);
          continue;
        }
        tokenOwner = dk; // Kept acquire supplies Q's token.
        order.apply(e, srcIdx, snapshots);
      }
    }
    if (n->owner)
      order.record(n, positions.lookup(n), snapshots);
  }
}
static void reduceLoopCloses(GroupDag &g, Node *head, const Positions &positions,
                             ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst,
                             ArrayRef<unsigned> closes, Node *firstAccess,
                             std::vector<bool> &drop) {
  if (closes.empty() || !firstAccess) return;
  constexpr unsigned kPass2 = 1u << 20;
  KnownOrder order;
  Snapshots snap1, snap2;
  for (Node *n = head; n; n = n->next) {
    for (unsigned ei : edgesAt(atDst, n)) {
      if (drop[ei] || isLoopClose(edges[ei])) continue;
      order.apply(edges[ei], positions.lookup(edges[ei].src), snap1);
    }
    if (n->owner) order.record(n, positions.lookup(n), snap1);
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
    if (latest) closeAt[latest].push_back(ei);
  }
  DenseSet<int64_t> tokenAvailable;
  for (Node *n = head; n; n = n->next) {
    for (unsigned ei : edgesAt(atDst, n)) {
      if (drop[ei] || isLoopClose(edges[ei])) continue;
      order.apply(edges[ei], kPass2 + positions.lookup(edges[ei].src), snap2);
      tokenAvailable.insert(ownerKey(edges[ei].dstOwner));
    }
    for (unsigned ei : edgesAt(closeAt, n)) {
      const EdgeRec &e = edges[ei];
      int64_t dk = ownerKey(e.dstOwner);
      if (order.covers(e, positions.lookup(e.src)) && tokenAvailable.contains(dk) &&
          !sameOwner(e.dstOwner, firstAccess->owner)) {
        drop[ei] = true;
        continue;
      }
      order.apply(e, positions.lookup(e.src), snap1);
    }
    if (n->owner) order.record(n, kPass2 + positions.lookup(n), snap2);
  }
}
static void reduceChain(GroupDag &g, Node *head, ArrayRef<EdgeRec> edges, std::vector<bool> &drop,
                        DenseSet<Node *> &reusable) {
  Positions positions;
  unsigned position = 0;
  Node *firstAccess = nullptr;
  for (Node *n = head; n; n = n->next) {
    positions[n] = position++;
    if (!firstAccess && n->kind == Node::Access && n->owner)
      firstAccess = n;
  }
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
      return positions.lookup(edges[a].src) >
             positions.lookup(edges[b].src);
    });
  reduceStraightEdges(head, positions, edges, atDst, drop, reusable);
  if (head->parent && head->parent->kind == Node::For)
    reduceLoopCloses(g, head, positions, edges, atDst, closes, firstAccess,
                     drop);
  for (Node *n = head; n; n = n->next)
    for (Node *child : n->children)
      reduceChain(g, child, edges, drop, reusable);
}
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges, DenseSet<Node *> &reusable) {
  if (edges.empty()) return;
  std::vector<bool> drop(edges.size(), false);
  reduceChain(g, g.root->children[0], edges, drop, reusable);
  unsigned i = 0;
  llvm::erase_if(edges, [&](const EdgeRec &) { return drop[i++]; });
}
} // namespace

static bool precedesInChain(Node *before, Node *after) {
  for (Node *next = before->next; next; next = next->next)
    if (next == after) return true;
  return false;
}
// Direct protocol placement.  Acquires and releases are first placed with
// exact token sources and completion anchors.  Semaphore channels and counts
// are formed only after no placement can change.
class DirectBuilder {
public:
  DirectBuilder(GroupDag &group, ArrayRef<EdgeRec> edges, const DenseSet<Node *> &reuse) : g(group), reusable(reuse) {
    for (const EdgeRec &edge : edges) {
      atDst[edge.dst].push_back(&edge);
      atSrc[edge.src].push_back(&edge);
      if (edge.dst->kind == Node::Exit)
        exitSources.insert(edge.src);
    }
  }
  LogicalResult run() {
    assert(!g.root->children.empty() && "edges require an access DAG");
    auto placed = placeChain(g.root->children[0], {});
    if (hadError) return failure();
    for (const auto &[dst, refs] : atDst)
      for (const EdgeRec *edge : refs)
        if (!handledEdges.contains(edge)) {
          if (shouldDumpDag()) dumpDagTree(g);
          return semaError(dst->op ? dst->op : g.root->op) <<
                 "reduced dependency edge was not placed (source kind "
                 << static_cast<unsigned>(edge->src->kind) << ", source owner "
                 << ownerKey(edge->srcOwner) << ", target kind "
                 << static_cast<unsigned>(dst->kind) << ", target owner "
                 << ownerKey(edge->dstOwner) << ")";
        }
    if (!placed.supplies.empty()) {
      if (shouldDumpDag()) dumpDagTree(g);
      return semaError(g.root->op) << "conditional release path was not consumed (" << placed.supplies.size() << ")";
    }
    return formSemaphores();
  }
private:
  using EdgeRefs = SmallVector<const EdgeRec *, 2>;
  struct Supply {
    SmallVector<Node *, 2> releases;
    unsigned arrivals = 0;
    bool present = false, multiple = false, mismatch = false;
    // The same pre-if releases may feed acquire sites in mutually exclusive
    // branches.  No other release-to-multiple-acquires shape is legal.
    bool mayBindAlternatives = false;
    bool empty() const { return !present; }
    void append(Supply other, bool alternative) {
      if (other.empty()) return;
      if (empty()) {
        *this = std::move(other);
        return;
      }
      mismatch |= other.mismatch ||
                  (alternative && arrivals != other.arrivals);
      arrivals += alternative ? 0 : other.arrivals;
      multiple |= other.multiple || alternative;
      mayBindAlternatives |= other.mayBindAlternatives;
      releases.append(other.releases.begin(), other.releases.end());
    }
    void appendExecuted(Supply other) { append(std::move(other), false); }
    void appendExecuted(Node *release) {
      if (!release) return;
      present = true;
      arrivals += std::max(1u, release->count) *
                  std::max(1u, unsigned(release->payloads.size()));
      releases.push_back(release);
    }
    void appendAlternative(Supply other) { append(std::move(other), true); }
  };
  using BoundaryKey = std::pair<Node *, Owner>;
  struct PendingSupply { Supply paths; SmallVector<Node *, 1> sources; };
  using PendingSupplies = std::map<BoundaryKey, PendingSupply>;
  using PendingList = SmallVector<std::pair<Owner, PendingSupply>, 2>;
  struct Chain {
    Tokens tokens, entryTokens;
    EdgeRefs pending;
    PendingSupplies supplies;
    SmallVector<std::tuple<Owner, Node *, Node *>, 1> pendingChannels;
    Node *guard = nullptr;
    std::optional<Owner> watchedOwner;
    bool watchedHasRealInput = false;
    bool returnDemand = false;
    Node *lastCompletion = nullptr;
  };
  struct LoopSupply {
    EdgeRefs entry, closes; Supply entryPaths, closePaths;
    SmallVector<Node *, 1> sources;
  };
  static BoundaryKey boundaryKey(Node *region, const Owner &owner) { return {region, owner}; }
  EdgeRefs unhandled(ArrayRef<const EdgeRec *> refs) const {
    EdgeRefs result;
    llvm::copy_if(refs, std::back_inserter(result), [&](const EdgeRec *edge) { return !handledEdges.contains(edge); });
    return result;
  }
  EdgeRefs unhandledAt(Node *node) const { return unhandled(atDst.lookup(node)); }
  EdgeRefs unhandledAt(Node *node, const Owner &owner) const {
    EdgeRefs refs = unhandledAt(node);
    llvm::erase_if(refs, [&](const EdgeRec *edge) {
      return !sameOwner(edge->dstOwner, owner);
    });
    return refs;
  }
  void consumeRouted(Node *target, const Owner &owner) {
    for (const EdgeRec *edge : atDst.lookup(target))
      if (sameOwner(edge->dstOwner, owner) && routedEdges.erase(edge))
        handledEdges.insert(edge);
  }
  Supply takeSupply(Chain &chain, Node *target, const Owner &owner) {
    auto entry = chain.supplies.extract(boundaryKey(target, owner));
    if (entry.empty()) return {};
    consumeRouted(target, owner);
    return std::move(entry.mapped().paths);
  }
  PendingList takeSupplies(Chain &chain, Node *target) {
    PendingList result;
    auto it = chain.supplies.lower_bound(boundaryKey(target, {}));
    while (it != chain.supplies.end() && it->first.first == target) {
      auto current = it++;
      result.emplace_back(current->first.second, std::move(current->second));
      chain.supplies.erase(current);
    }
    return result;
  }
  static EdgeRefs takePending(Chain &chain, const Owner &owner) {
    EdgeRefs result;
    llvm::erase_if(chain.pending, [&](const EdgeRec *edge) {
      if (!sameOwner(edge->dstOwner, owner)) return false;
      result.push_back(edge);
      return true;
    });
    return result;
  }
  static bool hasPending(const Chain &chain, const Owner &owner) {
    return llvm::any_of(chain.pending, [&](const EdgeRec *edge) {
      return sameOwner(edge->dstOwner, owner);
    });
  }
  static bool hasResidual(const Chain &chain) {
    return !chain.pending.empty() || !chain.supplies.empty() || !chain.pendingChannels.empty();
  }
  Node *regionChannelFor(Node *region, const Owner &owner) const {
    auto it = regionChannels.find(boundaryKey(region, owner));
    return it == regionChannels.end() ? nullptr : it->second;
  }
  bool isSoleRegionOutput(const EdgeRec &edge) const {
    return llvm::count_if(atSrc.lookup(edge.src), [&](const EdgeRec *other) {
      return sameOwner(other->srcOwner, edge.srcOwner);
    }) == 1;
  }
  void fail(Node *node, StringRef message) {
    semaError(node && node->op ? node->op : g.root->op) << message;
    hadError = true;
  }
  Node *concreteAcquire(Node *producer) const {
    if (!producer) return nullptr;
    switch (producer->kind) {
    case Node::Acquire: return producer;
    case Node::For: case Node::If: break;
    default: return nullptr;
    }
    if (!producer->flow) return nullptr;
    for (Node *exit : producer->flow->exits) if (Node *acquire = concreteAcquire(exit)) return acquire;
    return nullptr;
  }
  static bool scalableRelease(const Node *release) {
    return llvm::all_of(release->payloads, [](AsyncOp p) { return p == AsyncOp::NONE || p == AsyncOp::WGMMA; });
  }
  static Node *chainExit(Node *head) {
    while (head && head->next)
      head = head->next;
    return head && head->kind == Node::Exit ? head : nullptr;
  }
  static bool touchesOwner(Node *node, const Owner &owner) {
    if (node->kind == Node::Access) return sameOwner(node->owner, owner);
    return node->isRegion() && llvm::any_of(node->pieceInfo, [&](const auto &piece) {
             return sameOwner(piece.second.owner, owner);
           });
  }
  std::pair<Node *, Node *> firstLoopDemand(Node *head, const Owner &owner, ArrayRef<Node *> deferred) const {
    for (Node *node = head; node && node->kind != Node::Exit; node = node->next) {
      Node *channel = node->isRegion() ? regionChannelFor(node, owner) : nullptr;
      bool deferredUse = node->isRegion() &&
                         llvm::is_contained(deferred, node);
      bool usableRegion = node->isRegion() && !node->pieceInfo.empty() &&
                          (node->flow || node->tokenSource || channel || deferredUse);
      bool touches = touchesOwner(node, owner) &&
                     (node->kind == Node::Access || usableRegion);
      if (channel || touches) return {node, channel};
    }
    return {};
  }
  Node *nextGroupUse(Node *region) const {
    for (Node *node = region->next; node; node = node->next)
      if (node->kind == Node::Exit || node->kind == Node::Access ||
          (node->isRegion() && !node->pieceInfo.empty()))
        return node->kind == Node::Exit ? nullptr : node;
    return nullptr;
  }
  Node *firstOwnerUse(Node *head, const Owner &owner) const {
    for (Node *node = head; node && node->kind != Node::Exit;
         node = node->next)
      if (touchesOwner(node, owner)) return node;
    return chainExit(head);
  }
  bool reusableUse(Node *node, const Owner &owner) const {
    if (!node || !unhandledAt(node).empty()) return false;
    switch (node->kind) {
    case Node::Access: return sameOwner(node->owner, owner) && reusable.contains(node);
    case Node::For: case Node::If: {
      std::optional<Owner> nested = uniformPieceOwner(node);
      return nested && sameOwner(*nested, owner);
    }
    default: return false;
    }
  }
  static bool usesTokenSource(Node *head, Node *source) {
    bool used = false;
    forEachNode(head, [&](Node *node) { used |= node->tokenSource == source; });
    return used;
  }
  Node *findChannel(Node *acquire) {
    Node *parent = channelParent.lookup(acquire);
    if (!parent || parent == acquire)
      return acquire;
    return channelParent[acquire] = findChannel(parent);
  }
  void uniteChannels(Node *lhs, Node *rhs) {
    lhs = findChannel(lhs);
    rhs = findChannel(rhs);
    if (lhs != rhs) channelParent[rhs] = lhs;
  }
  void spliceAfterLast(Node *node, Node *anchor) {
    Node *last = lastAfter.lookup(anchor);
    spliceAfter(node, last ? last : anchor);
    lastAfter[anchor] = node;
  }
  Node *makeAcquire(Node *before, const Owner &owner) {
    Node *acquire = newProtocolNode(g, Node::Acquire, before->parent, owner);
    acquire->scheduleAnchor = before;
    spliceBefore(acquire, before);
    channelParent[acquire] = acquire;
    return acquire;
  }
  Node *makeAcquireAfter(Node *after, const Owner &owner) {
    Node *acquire = newProtocolNode(g, Node::Acquire, after->parent, owner);
    acquire->scheduleAnchor = after;
    spliceAfterLast(acquire, after);
    channelParent[acquire] = acquire;
    return acquire;
  }
  Node *regionDrain(Node *region, const Owner &owner, Node *channel, Node *guard = nullptr) {
    auto key = std::make_pair(guard ? guard : region, channel);
    auto [it, inserted] = regionDrains.try_emplace(key, nullptr);
    if (!inserted) return it->second;
    Node *drain = makeAcquireAfter(key.first, owner);
    drain->count = channel->count;
    uniteChannels(drain, channel);
    return it->second = drain;
  }
  void materializePendingChannels(Chain &chain) {
    for (auto &[owner, source, channel] : chain.pendingChannels) {
      Node *drain = regionDrain(source, owner, channel);
      chain.tokens.record(owner, drain, drain, {AsyncOp::NONE});
    }
    chain.pendingChannels.clear();
  }
  Node *localDrainGuard(Node *source, const Chain &chain, Node *guard = nullptr) const {
    guard = guard ? guard : chain.guard;
    return !chain.returnDemand && guard && source->parent != guard->parent ? guard : nullptr;
  }
  void materializePendingChannel(Chain &chain, Tokens &tokens, const Owner &owner, Node *guard) {
    auto it = llvm::find_if(chain.pendingChannels, [&](const auto &p) { return sameOwner(std::get<0>(p), owner); });
    if (it == chain.pendingChannels.end()) return;
    auto [actual, source, channel] = *it;
    Node *drain = regionDrain(source, actual, channel, localDrainGuard(source, chain, guard));
    tokens.record(actual, drain, drain, {AsyncOp::NONE});
    chain.entryTokens.record(actual, drain, drain, {AsyncOp::NONE});
    chain.pendingChannels.erase(it);
  }
  Tokens::Token sourceToken(const EdgeRec &edge, const Tokens &tokens, const Chain &chain) const {
    switch (edge.src->kind) {
    case Node::Access: return {edge.srcOwner, edge.src->tokenSource, edge.src, edge.payloads};
    case Node::For: case Node::If: {
      Node *producer = edge.src->flow ? edge.src : edge.src->tokenSource;
      if (!producer) return {};
      if (const Tokens::Token *token = tokens.findProducer(producer)) return *token;
      return {edge.srcOwner, producer, edge.src, edge.payloads};
    }
    case Node::Acquire:
      if (const Tokens::Token *token = tokens.findProducer(edge.src)) return *token;
      return {};
    case Node::Enter:
      if (const Tokens::Token *token = chain.entryTokens.findOpen(edge.srcOwner)) return *token;
      return {};
    case Node::Exit: case Node::Release: case Node::Func: return {};
    }
    llvm_unreachable("unknown access DAG node kind");
  }
  void routeSupply(Node *region, Node *input, Node *target, const Owner &sourceOwner, Supply paths, Chain &chain) {
    if (!target)
      return fail(region, "conditional release path has no downstream use");
    std::optional<Owner> targetOwner = target->kind == Node::Access
                                           ? std::optional<Owner>(target->owner)
                                           : uniformPieceOwner(target);
    for (const EdgeRec *edge : unhandledAt(target)) {
      Tokens::Token source = sourceToken(*edge, chain.tokens, chain);
      bool matches = edge->src == region || (input && edge->src == input) ||
                     source.producer == region || (input && source.producer == input);
      if (!matches || !sameOwner(edge->srcOwner, sourceOwner)) continue;
      if (targetOwner && !sameOwner(*targetOwner, edge->dstOwner))
        return fail(region, "conditional release path has conflicting target owners");
      targetOwner.emplace(edge->dstOwner);
      routedEdges.insert(edge);
    }
    if (!targetOwner)
      return fail(region, "conditional release path has no exact target owner");
    PendingSupply &pending = chain.supplies[boundaryKey(target, *targetOwner)];
    pending.paths.appendExecuted(std::move(paths));
    pending.sources.push_back(region);
  }
  Node *materializeRelease(Tokens::Token &token, const Owner &owner, Node *acquire, Node *guard) {
    Node *release = newProtocolNode(g, Node::Release, guard ? guard->parent : token.last->parent, owner);
    release->payloads = token.payloads;
    assert(!release->payloads.empty() && "token must retain completion payload");
    release->tokenSource = token.producer;
    if (acquire) release->consumers.push_back(acquire);
    release->scheduleAnchor = token.last;
    if (guard)
      spliceAfter(release, guard);
    else
      spliceAfterLast(release, token.last);
    return release;
  }
  Node *insertRelease(const EdgeRec &edge, Node *acquire, Tokens &tokens, const Chain &chain, Node *guard = nullptr) {
    Tokens::Token source = sourceToken(edge, tokens, chain);
    if (!source.producer || !source.last) {
      semaError(edge.src->op ? edge.src->op : g.root->op)
          << "release has no exact token for source kind " << unsigned(edge.src->kind)
          << ", source owner " << ownerKey(edge.srcOwner) << ", target kind "
          << unsigned(edge.dst->kind) << ", target owner " << ownerKey(edge.dstOwner);
      hadError = true;
      return nullptr;
    }
    if (edge.src->kind == Node::Enter)
      guard = source.last->parent == edge.src->parent ? source.last : edge.src;
    Node *release = materializeRelease(source, edge.srcOwner, acquire, guard);
    tokens.close(source.producer, release);
    return release;
  }
  void applySupply(Node *acquire, const Supply &supply, std::optional<unsigned> requiredCount = std::nullopt) {
    if (supply.mismatch || (!supply.empty() && !supply.arrivals))
      return fail(acquire, "conditional paths provide incompatible semaphore counts");
    unsigned expected = supply.arrivals;
    if (expected && requiredCount && *requiredCount && expected != *requiredCount) {
      if (supply.multiple || supply.releases.size() != 1 ||
          !scalableRelease(supply.releases.front()) ||
          supply.releases.front()->payloads.empty() ||
          *requiredCount % supply.releases.front()->payloads.size())
        return fail(acquire, "one execution path does not supply the acquire pending count");
      Node *release = supply.releases.front();
      unsigned payloads = release->payloads.size();
      release->count = *requiredCount / payloads;
      expected = *requiredCount;
    }
    unsigned pending = std::max(acquire->count, expected);
    auto conflict = llvm::find_if(supply.releases, [&](Node *release) {
      return llvm::any_of(release->consumers, [&](Node *bound) {
        return bound != acquire &&
               (!supply.mayBindAlternatives ||
                !sameOwner(bound->owner, acquire->owner) ||
                bound->count != pending ||
                bound->recurrenceDistance != acquire->recurrenceDistance ||
                seeded.contains(bound) != seeded.contains(acquire));
      });
    });
    if (conflict != supply.releases.end())
      return fail(*conflict,
                  "alternative acquire sites have incompatible protocol facts");
    for (Node *release : supply.releases) {
      if (!release->consumers.empty())
        uniteChannels(release->consumers.front(), acquire);
      if (!llvm::is_contained(release->consumers, acquire))
        release->consumers.push_back(acquire);
    }
    acquire->count = pending;
  }
  Supply collectSupply(Node *acquire, ArrayRef<const EdgeRec *> refs, Tokens &tokens, Chain &chain, Node *guard = nullptr, bool reuseRegionChannel = false) {
    SmallVector<EdgeRec, 2> incoming;
    for (const EdgeRec *edgeRef : refs) {
      EdgeRec edge = *edgeRef;
      if (edge.src->kind == Node::Enter &&
          !chain.entryTokens.findOpen(edge.srcOwner))
        materializePendingChannel(chain, tokens, edge.srcOwner,
                                  guard ? guard : chain.guard);
      Tokens::Token source = sourceToken(edge, tokens, chain);
      auto prior = source.producer
                       ? llvm::find_if(incoming, [&](const EdgeRec &item) {
                           Tokens::Token other = sourceToken(item, tokens, chain);
                           return other.producer == source.producer &&
                                  (item.src == edge.src ||
                                   precedesInChain(item.src, edge.src) ||
                                   precedesInChain(edge.src, item.src));
                         })
                       : incoming.end();
      if (prior == incoming.end()) {
        incoming.push_back(std::move(edge));
        continue;
      }
      unionPayloads(prior->payloads, edge.payloads);
      for (PieceId piece : edge.pieces)
        if (!llvm::is_contained(prior->pieces, piece)) prior->pieces.push_back(piece);
      if (precedesInChain(prior->src, edge.src)) prior->src = edge.src;
    }
    Supply supply;
    for (const EdgeRec &edge : incoming) {
      if (edge.src->isRegion() && !edge.src->flow) {
        if (Node *channel = regionChannelFor(edge.src, edge.srcOwner)) {
          if (reuseRegionChannel && isSoleRegionOutput(edge) && acquire &&
              !acquire->count && sameOwner(edge.srcOwner, edge.dstOwner)) {
            uniteChannels(acquire, channel);
            acquire->count = channel->count;
          } else {
            Node *drainGuard = localDrainGuard(edge.src, chain, guard);
            Node *drain = regionDrain(edge.src, edge.srcOwner, channel, drainGuard);
            if (!tokens.findProducer(drain))
              tokens.record(edge.srcOwner, drain, drain, {AsyncOp::NONE});
            EdgeRec handoff{drain, acquire, edge.srcOwner, edge.dstOwner,
                            {AsyncOp::NONE}, edge.pieces};
            supply.appendExecuted(insertRelease(
                handoff, acquire, tokens, chain, drainGuard ? drain : guard));
            if (acquire) acquire->count = std::max(acquire->count, 1u);
          }
          continue;
        }
        if (!edge.src->tokenSource) {
          fail(edge.src, "no-flow region has no exact routed token");
          continue;
        }
      }
      supply.appendExecuted(insertRelease(edge, acquire, tokens, chain, guard));
    }
    for (const EdgeRec *edge : refs) handledEdges.insert(edge);
    return supply;
  }
  Node *insertTokenRelease(Tokens::Token &token, Node *acquire, Node *guard = nullptr) {
    if (!token.producer || !token.last) {
      fail(token.last, "token handoff has no exact completion");
      return nullptr;
    }
    return materializeRelease(token, token.owner, acquire, guard);
  }
  static void dropOwner(Tokens &tokens, Tokens incoming, const Owner &owner) {
    tokens = std::move(incoming);
    tokens.eraseOwner(owner);
  }
  void publishRegionFlow(Node *region, Node *input, const Owner &owner, ArrayRef<Node *> exits, const Payloads &payloads, Tokens incoming, Tokens &tokens) {
    Node *common = nullptr;
    for (Node *output : exits) {
      Node *site = concreteAcquire(output ? output : input);
      if (!site) continue;
      if (common && common != site && common->count != site->count)
        fail(region, "region paths return incompatible semaphore counts");
      if (common) uniteChannels(common, site);
      else common = site;
    }
    RegionFlow flow;
    flow.owner = owner;
    flow.exits.append(exits.begin(), exits.end());
    region->tokenSource = input;
    region->flow.emplace(std::move(flow));
    tokens = std::move(incoming);
    if (input) tokens.eraseProducer(input);
    tokens.record(owner, region, region, payloads);
  }
  Chain enterChain(Chain chain, const Tokens &incoming, Node *enter) {
    chain.tokens = incoming;
    Tokens &tokens = chain.tokens;
    std::optional<Owner> owner = uniformPieceOwner(enter);
    if (owner && owner->has_value() && tokens.last() && !tokens.last()->owner) {
      Tokens::Token adopted = *tokens.last();
      adopted.owner = *owner;
      tokens.record(adopted.owner, adopted.producer, adopted.last, adopted.payloads);
    }
    chain.entryTokens = tokens;
    if (owner)
      if (const Tokens::Token *token = tokens.findOpen(*owner))
        enter->tokenSource = token->producer;
    return chain;
  }
  void placeAccess(Node *node, Chain &chain) {
    Tokens &tokens = chain.tokens;
    Payloads payloads{asyncPayloadOf(node->op)};
    Supply routed = takeSupply(chain, node, node->owner);
    EdgeRefs direct = unhandledAt(node);
    EdgeRefs inherited;
    llvm::erase_if(chain.pending, [&](const EdgeRec *edge) {
      if (!sameOwner(edge->dstOwner, node->owner)) return false;
      inherited.push_back(edge);
      return true;
    });
    bool noIncoming = direct.empty() && inherited.empty() && routed.empty();
    llvm::erase_if(chain.pendingChannels, [&](const auto &pending) {
      auto [owner, source, channel] = pending;
      if (!sameOwner(owner, node->owner)) return false;
      if (noIncoming && reusable.contains(node)) {
        Node *acquire = node->tokenSource;
        if (!acquire) node->tokenSource = acquire = makeAcquire(node, node->owner);
        acquire->count = std::max(acquire->count, channel->count);
        uniteChannels(acquire, channel);
      } else {
        Node *drain = regionDrain(source, owner, channel,
                                  localDrainGuard(source, chain));
        Tokens::Token token{owner, drain, drain, {AsyncOp::NONE}};
        routed.appendExecuted(insertTokenRelease(token, nullptr));
      }
      return true;
    });
    noIncoming &= routed.empty();
    auto supply = [&](Node *acquire) {
      Supply combined;
      bool oneRaw = direct.size() + inherited.size() == 1 && routed.empty();
      combined.appendExecuted(collectSupply(acquire, direct, tokens, chain, nullptr,
          oneRaw && inherited.empty() && !chain.guard));
      combined.appendExecuted(
          collectSupply(acquire, inherited, tokens, chain, chain.guard));
      combined.appendExecuted(std::move(routed));
      applySupply(acquire, combined);
    };
    if (Node *acquire = node->tokenSource) {
      assert(acquire->kind == Node::Acquire && "preplaced non-acquire token");
      supply(acquire);
      node->tokenSource = acquire;
      tokens.record(node->owner, acquire, node, payloads);
      return;
    }
    Tokens::Token *owned = tokens.find(node->owner);
    if (noIncoming && owned && owned->producer && reusable.contains(node)) {
      node->tokenSource = owned->producer;
      owned->last = node;
      owned->payloads = payloads;
      // A read fan-out release does not invalidate the releasing owner's
      // compatible token.  The per-piece reuse proof is what permits
      // this access to remain in the same ownership wave.
      owned->closedBy = nullptr;
      return;
    }
    Node *acquire = makeAcquire(node, node->owner);
    supply(acquire);
    if (noIncoming) {
      seeded.insert(acquire);
      acquire->count = 1;
    }
    node->tokenSource = acquire;
    tokens.record(node->owner, acquire, node, payloads);
  }
  void placeIf(Node *region, Chain &chain) {
    Node *next = nextGroupUse(region); bool returnChannels = next || chain.returnDemand || exitSources.contains(region);
    if (returnChannels) materializePendingChannels(chain);
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    PendingList routed = takeSupplies(chain, region);
    Tokens incoming = chain.tokens;
    Chain adopted = enterChain({}, incoming, region->children.front());
    const Tokens::Token *input =
        boundaryOwner ? adopted.tokens.findOpen(*boundaryOwner) : nullptr;
    if (input) region->tokenSource = input->producer;
    SmallVector<Chain, 2> branches;
    for (Node *child : region->children) {
      Chain branch = enterChain(chain, incoming, child);
      branch.returnDemand = returnChannels;
      branch.supplies.clear();
      if (input) {
        branch.tokens.record(*boundaryOwner, region, child, input->payloads);
        branch.entryTokens = branch.tokens;
        child->tokenSource = region;
      }
      branch.pending = atDst.lookup(region);
      branch.pending.append(chain.pending.begin(), chain.pending.end());
      llvm::erase_if(branch.pending, [&](const EdgeRec *edge) {
        return routedEdges.contains(edge);
      });
      for (const auto &[owner, pending] : routed) {
        PendingSupply copy = pending;
        copy.paths.mayBindAlternatives = true;
        branch.supplies[boundaryKey(child, owner)] = std::move(copy);
      }
      branch.guard = child;
      branch = placeChain(child, std::move(branch));
      if (returnChannels) materializePendingChannels(branch);
      else branch.pendingChannels.clear();
      branches.push_back(std::move(branch));
    }
    SmallVector<Owner, 4> activeOwners;
    auto rememberOwner = [&](const Owner &owner) {
      if (!llvm::is_contained(activeOwners, owner)) activeOwners.push_back(owner);
    };
    for (const auto &[owner, _] : routed) rememberOwner(owner);
    for (const Chain &branch : branches) {
      Node *exit = chainExit(branch.guard);
      for (const EdgeRec *edge : branch.pending)
        rememberOwner(edge->dstOwner);
      for (const auto &[key, _] : branch.supplies) {
        if (key.first != exit)
          fail(region, "if branch left a conditional release before EXIT");
        rememberOwner(key.second);
      }
      for (const EdgeRec *edge : unhandledAt(exit))
        rememberOwner(edge->dstOwner);
    }
    if (hadError) return;
    for (const auto &[owner, _] : routed) consumeRouted(region, owner);
    chain.pending.clear();
    chain.pendingChannels.clear();
    Node *target = next ? next : chainExit(region);
    bool terminal = !next && !chain.watchedOwner && !chain.returnDemand &&
                    !exitSources.contains(region);
    auto closeOwner = [&](const Owner &owner, Node *inputProducer) {
      Supply paths;
      for (Chain &branch : branches) {
        if (inputProducer) replaceTokenSource(branch.guard, region, inputProducer);
        Node *exit = chainExit(branch.guard);
        Supply branchPaths = takeSupply(branch, exit, owner);
        EdgeRefs pending = takePending(branch, owner);
        branchPaths.appendExecuted(
            collectSupply(nullptr, pending, branch.tokens, branch, branch.guard));
        EdgeRefs edges = unhandledAt(exit, owner);
        branchPaths.appendExecuted(
            collectSupply(nullptr, edges, branch.tokens, branch));
        Node *guard = exit && exit->prev ? exit->prev : branch.guard;
        if (const Tokens::Token *open = branch.tokens.findOpen(owner)) {
          Tokens::Token exact = *open;
          if (inputProducer && exact.producer == region)
            exact.producer = inputProducer;
          branchPaths.appendExecuted(insertTokenRelease(exact, nullptr, guard));
        }
        if (branchPaths.empty()) branchPaths.present = true;
        paths.appendAlternative(std::move(branchPaths));
      }
      return paths;
    };
    auto discardOwner = [&](const Owner &owner) {
      for (Chain &branch : branches) {
        Node *exit = chainExit(branch.guard);
        (void)takeSupply(branch, exit, owner);
        for (const EdgeRec *edge : takePending(branch, owner)) handledEdges.insert(edge);
        for (const EdgeRec *edge : unhandledAt(exit, owner)) handledEdges.insert(edge);
      }
    };
    for (const Owner &owner : activeOwners) {
      if (boundaryOwner && sameOwner(owner, *boundaryOwner)) continue;
      if (terminal) discardOwner(owner);
      else routeSupply(region, nullptr, target, owner, closeOwner(owner, nullptr), chain);
      incoming.eraseOwner(owner);
    }
    if (!boundaryOwner) {
      if (llvm::any_of(branches, hasResidual))
        fail(region, "mixed conditional flow retains a deferred dependency");
      chain.tokens = std::move(incoming);
      return;
    }

    Owner owner = *boundaryOwner;
    auto output = [&](const Chain &branch) { return branch.tokens.findOpen(owner); };
    auto hasDeferred = [&](const Chain &branch) {
      Node *exit = chainExit(branch.guard);
      return branch.supplies.count(boundaryKey(exit, owner)) ||
             hasPending(branch, owner) ||
             llvm::any_of(unhandledAt(exit), [&](const EdgeRec *edge) {
               return sameOwner(edge->dstOwner, owner);
             });
    };
    auto final = [&](const Chain &branch) { return output(branch) && !hasDeferred(branch); };
    auto isPass = [&](const Chain &branch) {
      const Tokens::Token *token = output(branch);
      return input && token &&
             (token->producer == region || token->producer == input->producer) &&
             !hasDeferred(branch);
    };
    bool changed = llvm::any_of(branches, [&](const Chain &branch) { return !isPass(branch); });
    if (!changed) {
      chain.tokens = adopted.tokens;
      if (input) {
        chain.tokens.eraseProducer(input->producer);
        chain.tokens.record(owner, input->producer, region, input->payloads);
      }
      return;
    }
    bool complete = llvm::all_of(branches, final);
    bool outgoing = next && llvm::any_of(unhandledAt(next), [&](const EdgeRec *edge) {
      return edge->src == region && sameOwner(edge->srcOwner, owner);
    });
    std::optional<Owner> parentOwner =
        chain.guard ? uniformPieceOwner(chain.guard) : std::nullopt;
    bool allowCompletion = !next && chain.watchedOwner;
    bool publish = next
                       ? reusableUse(next, owner) || (complete && outgoing)
                       : chain.watchedOwner
                             ? sameOwner(owner, *chain.watchedOwner) &&
                                   (g.numCopies == 1 || chain.watchedHasRealInput)
                       : chain.returnDemand ? parentOwner && sameOwner(owner, *parentOwner)
                                            : exitSources.contains(region);
    if (publish && allowCompletion && !complete) {
      complete = true;
      for (Chain &branch : branches) {
        Node *exit = chainExit(branch.guard);
        if (final(branch)) continue;
        if (output(branch)) {
          complete = false;
          break;
        }
        Supply routed = takeSupply(branch, exit, owner);
        EdgeRefs edges = unhandledAt(exit, owner);
        EdgeRefs pending = takePending(branch, owner);
        if (edges.empty() && pending.empty() && routed.empty()) {
          complete = false;
          break;
        }
        Node *acquire = makeAcquire(exit, owner);
        bool oneRaw = edges.size() + pending.size() == 1 && routed.empty();
        Supply supply = collectSupply(acquire, edges, branch.tokens, branch, nullptr,
                                      oneRaw && pending.empty() && !branch.guard);
        supply.appendExecuted(collectSupply(
            acquire, pending, branch.tokens, branch, branch.guard));
        supply.appendExecuted(std::move(routed));
        applySupply(acquire, supply);
        branch.tokens.record(owner, acquire, acquire, {AsyncOp::NONE});
      }
    }
    if (publish && complete) {
      SmallVector<Node *, 2> exits;
      for (const Chain &branch : branches) {
        exits.push_back(isPass(branch) ? nullptr : output(branch)->producer);
        if (!branch.pending.empty() || !branch.supplies.empty() ||
            !unhandledAt(chainExit(branch.guard)).empty())
          fail(region, "complete conditional flow retains an unrepresented dependency");
      }
      if (hadError) return;
      publishRegionFlow(region, input ? input->producer : nullptr, owner, exits,
                        {AsyncOp::NONE}, std::move(incoming), chain.tokens);
      return;
    }
    if ((publish && allowCompletion) || terminal) {
      if (terminal) discardOwner(owner);
      if (llvm::any_of(branches, hasResidual))
        fail(region, "conditional handoff is not consumed on every continuation path");
      region->tokenSource = nullptr;
      dropOwner(chain.tokens, std::move(incoming), owner);
      return;
    }
    Supply paths = closeOwner(owner, input ? input->producer : nullptr);
    if (llvm::any_of(branches, hasResidual)) {
      fail(region, "conditional close retains a deferred dependency");
      return;
    }
    routeSupply(region, input ? input->producer : nullptr, target, owner, std::move(paths), chain);
    dropOwner(chain.tokens, std::move(incoming), owner);
    region->tokenSource = nullptr;
  }
  void replaceTokenSource(Node *head, Node *from, Node *to) {
    forEachNode(head, [&](Node *node) {
      if (node->tokenSource == from) node->tokenSource = to;
    });
  }
  void finishLoopToken(Node *region, Node *input, Node *output, const Owner &owner, const Payloads &payloads, Tokens incoming, Tokens &tokens) {
    if (output != region) {
      publishRegionFlow(region, input, owner, {output}, payloads, std::move(incoming), tokens);
      return;
    }
    region->tokenSource = input;
    tokens = std::move(incoming);
    tokens.eraseProducer(input);
    tokens.record(owner, input, region, payloads);
  }
  void placeFor(Node *region, Chain &chain, Node *&lastOwnerAccess) {
    materializePendingChannels(chain);
    Node *body = region->children.front(), *exit = chainExit(body);
    Node *next = nextGroupUse(region); bool returnBody = next || chain.returnDemand;
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    EdgeRefs rawEntry = atDst.lookup(region);
    rawEntry.append(chain.pending.begin(), chain.pending.end());
    Tokens incoming = chain.tokens;
    Chain bodyState = enterChain({}, incoming, body);
    Tokens loopInputs = bodyState.tokens;
    bodyState.watchedOwner = boundaryOwner;
    bodyState.returnDemand = returnBody;
    const Tokens::Token *held =
        boundaryOwner ? loopInputs.findOpen(*boundaryOwner) : nullptr;
    bodyState.watchedHasRealInput =
        held && held->last && held->last->kind != Node::Enter;
    if (boundaryOwner) {
      bodyState.tokens.record(*boundaryOwner, region, body, {AsyncOp::NONE});
      bodyState.entryTokens = bodyState.tokens;
      body->tokenSource = region;
    }
    bodyState = placeChain(body, std::move(bodyState));
    if (returnBody) materializePendingChannels(bodyState); else bodyState.pendingChannels.clear();
    lastOwnerAccess = bodyState.lastCompletion;
    Tokens &bodyTokens = bodyState.tokens;
    chain.pending.clear();
    llvm::MapVector<Owner, LoopSupply> supplies;
    auto takePaths = [&](Chain &state, Node *target, bool entry) {
      for (auto &[owner, pending] : takeSupplies(state, target)) {
        LoopSupply &supply = supplies[owner];
        (entry ? supply.entryPaths : supply.closePaths)
            .appendExecuted(std::move(pending.paths));
        supply.sources.append(pending.sources);
        consumeRouted(target, owner);
      }
    };
    takePaths(chain, region, true);
    takePaths(bodyState, exit, false);
    if (!bodyState.supplies.empty()) {
      fail(region, "loop body left a conditional release before EXIT");
      return;
    }
    for (const EdgeRec *edge : exit ? unhandledAt(exit) : EdgeRefs{})
      supplies[edge->dstOwner].closes.push_back(edge);
    for (const EdgeRec *edge : unhandled(rawEntry))
      supplies[edge->dstOwner].entry.push_back(edge);
    if (boundaryOwner && llvm::any_of(supplies, [&](const auto &item) {
          return !sameOwner(item.first, *boundaryOwner);
        })) {
      fail(region, "uniform loop has mixed-owner deferred supply");
      return;
    }
    if (boundaryOwner) supplies[*boundaryOwner];
    auto supplyEntry = [&](Node *acquire, const Owner &owner,
                           LoopSupply &supply, bool seed) {
      Supply entry = std::move(supply.entryPaths);
      entry.appendExecuted(
          collectSupply(acquire, supply.entry, incoming, chain, chain.guard));
      if (!entry.empty()) {
        seeded.erase(acquire);
        applySupply(acquire, entry, /*requiredCount=*/acquire->count);
        return;
      }
      const Tokens::Token *initial = loopInputs.findOpen(owner);
      if (!initial || !initial->producer || !initial->last ||
          initial->last->kind == Node::Enter) {
        if (seed) seeded.insert(acquire);
        return;
      }
      Tokens::Token token = *initial;
      Owner releaseOwner = token.producer->kind == Node::Acquire ? token.producer->owner : token.owner;
      Node *guard = token.last->parent == region->parent ? nullptr : chain.guard;
      seeded.erase(acquire);
      Node *release = materializeRelease(token, releaseOwner, acquire, guard);
      Supply initialSupply;
      initialSupply.appendExecuted(release);
      applySupply(acquire, initialSupply,
                  /*requiredCount=*/acquire->count);
    };
    if (!boundaryOwner) {
      for (auto &[owner, supply] : supplies) {
        auto [first, acquire] =
            firstLoopDemand(body, owner, supply.sources);
        if (!first) {
          fail(region, "mixed-owner loop close has no point-of-use demand");
          continue;
        }
        if (!acquire) acquire = first->tokenSource;
        if (!acquire || acquire->kind != Node::Acquire) {
          Node *old = acquire;
          acquire = makeAcquire(first, owner);
          first->tokenSource = acquire;
          if (old) replaceTokenSource(first, old, acquire);
        }
        Supply paths = collectSupply(acquire, supply.closes, bodyTokens, bodyState);
        paths.appendExecuted(std::move(supply.closePaths));
        applySupply(acquire, paths);
        supplyEntry(acquire, owner, supply, true);
        acquire->count = std::max(acquire->count, 1u);
        regionChannels[boundaryKey(region, owner)] = acquire;
        incoming.eraseOwner(owner);
      }
      chain.tokens = std::move(incoming);
      return;
    }
    auto closeSupply = [&](Node *acquire, LoopSupply &supply) {
      Supply paths = collectSupply(acquire, supply.closes, bodyTokens, bodyState);
      paths.appendExecuted(std::move(supply.closePaths));
      applySupply(acquire, paths);
    };
    Owner owner = *boundaryOwner;
    LoopSupply &supply = supplies[owner];
    auto [demand, childChannel] =
        firstLoopDemand(body, owner, supply.sources);
    const Tokens::Token *initial = loopInputs.findOpen(owner);
    Node *demandAnchor = demand;
    while (demandAnchor && demandAnchor->isRegion() &&
           demandAnchor->scheduleAnchor)
      demandAnchor = demandAnchor->scheduleAnchor;
    gpu::StageCluster firstSchedule = demandAnchor && demandAnchor->op
                                          ? gpu::getStageCluster(demandAnchor->op)
                                          : std::nullopt;
    gpu::StageCluster lastSchedule = bodyState.lastCompletion && bodyState.lastCompletion->op
                                         ? gpu::getStageCluster(bodyState.lastCompletion->op)
                                         : std::nullopt;
    bool crossStage = g.numCopies == 1 && demand &&
                      demand->tokenSource == region && firstSchedule &&
                      lastSchedule &&
                      firstSchedule->first != lastSchedule->first;
    const Tokens::Token *output = bodyTokens.findOpen(owner);
    bool hasEntry = !supply.entry.empty() || !supply.entryPaths.empty();
    bool hasSupply = !supply.closes.empty() || !supply.closePaths.empty();
    bool returnsToken = output && !hasSupply;
    auto loopInput = [&]() -> Node * {
      if (!hasEntry && initial) {
        incoming.record(initial->owner, initial->producer, initial->last,
                        initial->payloads);
        return initial->producer;
      }
      Node *acquire = makeAcquire(region, owner);
      supplyEntry(acquire, owner, supply, true);
      if (seeded.contains(acquire)) acquire->count = 1;
      incoming.record(owner, acquire, acquire, {AsyncOp::NONE});
      return acquire;
    };
    if (returnsToken) {
      Node *input = loopInput();
      assert(input && "loop input must have an exact producer");
      Node *outputChannel = concreteAcquire(output->producer);
      if (seeded.contains(input) && outputChannel && !crossStage)
        uniteChannels(input, outputChannel);
      finishLoopToken(region, input, output->producer, owner, output->payloads, std::move(incoming), chain.tokens);
      return;
    }
    auto closeAtExit = [&](Node *input, bool distanceOne) {
      assert(input && exit && "loop close requires an input and EXIT");
      Node *tail = makeAcquire(exit, owner);
      tail->scheduleAnchor = demandAnchor;
      if (distanceOne) tail->recurrenceDistance = 1;
      closeSupply(tail, supply);
      if (input->kind == Node::Acquire) uniteChannels(tail, input);
      return tail;
    };
    crossStage &= hasSupply;
    bool nestedInput = false;
    for (Node *node = body; node && node != demand; node = node->next)
      nestedInput |= llvm::any_of(node->children, [&](Node *child) { return usesTokenSource(child, region); });
    bool forwardsInput = nestedInput || (demand && demand->isRegion() && !demand->flow &&
                                         demand->tokenSource == region);
    bool carryInput = hasSupply &&
                      (bodyState.watchedHasRealInput || forwardsInput) &&
                      !reusableUse(next, owner);
    if (crossStage || carryInput) {
      Node *input = loopInput();
      Node *tail = closeAtExit(input, true);
      region->scheduleAnchor = demandAnchor;
      finishLoopToken(region, input, tail, owner, {AsyncOp::NONE}, std::move(incoming), chain.tokens);
      return;
    }
    if (!demand)
      return hasEntry || hasSupply
                 ? fail(region, "loop has no recorded point-of-use demand")
                 : dropOwner(chain.tokens, std::move(incoming), owner);
    if (childChannel) {
      if (!hasSupply) {
        supplyEntry(childChannel, owner, supply, false);
        region->scheduleAnchor = demandAnchor;
      } else {
        Node *input = loopInput();
        Node *tail = closeAtExit(input, g.numCopies == 1);
        Tokens::Token bridge{owner, tail, tail, {AsyncOp::NONE}};
        insertTokenRelease(bridge, childChannel)->count =
            std::max(1u, childChannel->count);
      }
      regionChannels[boundaryKey(region, owner)] = childChannel;
      dropOwner(chain.tokens, std::move(incoming), owner);
      return;
    }
    Node *firstUse = demand;
    for (Node *node = body; node && node != demand; node = node->next)
      if (node->kind != Node::Enter && node->tokenSource == region) {
        firstUse = node;
        break;
      }
    Node *recurrence = demand->tokenSource;
    if (recurrence == region) {
      recurrence = makeAcquire(firstUse, owner);
      replaceTokenSource(body, region, recurrence);
    }
    if (!recurrence || recurrence->kind != Node::Acquire) {
      fail(region, "loop recurrence has no acquire at its recorded demand");
      return;
    }
    closeSupply(recurrence, supply);
    supplyEntry(recurrence, owner, supply, true);
    region->scheduleAnchor = demandAnchor;
    regionChannels[boundaryKey(region, owner)] = recurrence;
    chain.pendingChannels.emplace_back(owner, region, recurrence);
    for (const EdgeRec *edge : supply.entry)
      incoming.eraseOwner(edge->srcOwner);
    dropOwner(chain.tokens, std::move(incoming), owner);
  }
  Chain placeChain(Node *head, Chain chain) {
    for (auto &[owner, pending] : takeSupplies(chain, head)) {
      PendingSupply &target =
          chain.supplies[boundaryKey(firstOwnerUse(head, owner), owner)];
      target.paths.appendExecuted(std::move(pending.paths));
      target.sources.append(pending.sources);
    }
    for (Node *node = head; node;) {
      Node *next = node->next;
      std::optional<Owner> observedOwner;
      Node *completion = nullptr;
      switch (node->kind) {
      case Node::Access:
        placeAccess(node, chain);
        observedOwner.emplace(node->owner);
        completion = node;
        break;
      case Node::If: placeIf(node, chain); break;
      case Node::For:
        placeFor(node, chain, completion);
        observedOwner = uniformPieceOwner(node);
        break;
      default: break;
      }
      bool watched = chain.watchedOwner && observedOwner &&
                     sameOwner(*chain.watchedOwner, *observedOwner);
      if (watched && completion) chain.lastCompletion = completion;
      node = next;
    }
    return chain;
  }
  std::optional<SemaId> channelOfProducer(Node *producer) const {
    if (!producer) return std::nullopt;
    switch (producer->kind) {
    case Node::Acquire: return producer->sema;
    case Node::For: case Node::If:
      if (!producer->flow) return std::nullopt;
      return producer->flow->sema;
    default: return std::nullopt;
    }
  }
  LogicalResult formSemaphores() {
    llvm::MapVector<Node *, SmallVector<Node *, 2>> classes;
    DenseMap<Node *, SmallVector<Node *, 2>> releasesAt;
    for (const auto &storage : g.nodes) {
      Node *node = storage.get();
      switch (node->kind) {
      case Node::Acquire: classes[findChannel(node)].push_back(node); break;
      case Node::Release: {
        if (node->consumers.empty())
          return semaError(g.root->op) << "placed release has no acquire consumer";
        Node *primary = node->consumers.front();
        bool incompatible = llvm::any_of(ArrayRef<Node *>(node->consumers).drop_front(),
            [&](Node *consumer) {
              return consumer->kind != Node::Acquire ||
                     findChannel(consumer) != findChannel(primary) ||
                     !sameOwner(consumer->owner, primary->owner) ||
                     consumer->count != primary->count ||
                     consumer->recurrenceDistance != primary->recurrenceDistance ||
                     seeded.contains(consumer) != seeded.contains(primary);
            });
        if (primary->kind != Node::Acquire || incompatible)
          return semaError(g.root->op) << "alternative acquire consumers have incompatible final protocol facts";
        releasesAt[primary].push_back(node);
        break;
      }
      default: break;
      }
    }
    for (auto &[root, sites] : classes) {
      unsigned count = 1;
      for (Node *site : sites) count = std::max(count, site->count);
      for (Node *site : sites) {
        SmallVector<Node *, 2> &releases = releasesAt[site];
        if (site->count == count) continue;
        if (releases.empty() && seeded.contains(site)) {
          site->count = count;
          continue;
        }
        bool scalable = releases.size() == 1 && scalableRelease(releases.front());
        unsigned payloads = scalable ? releases.front()->payloads.size() : 0;
        if (!scalable || count % payloads)
          return semaError(site->op ? site->op : g.root->op) << "incompatible path counts for one semaphore channel";
        releases.front()->count = count / payloads;
        site->count = count;
      }
      SemaId sid = g.semas.size();
      Sema sema{"S" + std::to_string(sid), count};
      auto entry = llvm::find_if(sites, [&](Node *site) { return seeded.contains(site); });
      if (entry != sites.end()) sema.entryOwner.emplace((*entry)->owner);
      g.semas.push_back(std::move(sema));
      for (Node *site : sites) {
        site->sema = sid;
        site->count = count;
        for (Node *release : releasesAt[site]) release->sema = sid;
      }
    }
    forEachRegionPostOrder(g.root->children[0], [&](Node *node) {
      if (!node->flow) return;
      std::optional<SemaId> channel;
      for (Node *exit : node->flow->exits)
        if (exit && (channel = channelOfProducer(exit))) break;
      if (!channel) channel = channelOfProducer(node->tokenSource);
      node->flow->sema = channel;
    });
    return success();
  }
  GroupDag &g;
  const DenseSet<Node *> &reusable;
  DenseMap<Node *, EdgeRefs> atDst, atSrc;
  DenseSet<Node *> exitSources, seeded;
  DenseMap<Node *, Node *> lastAfter, channelParent;
  std::map<BoundaryKey, Node *> regionChannels;
  std::map<std::pair<Node *, Node *>, Node *> regionDrains;
  DenseSet<const EdgeRec *> routedEdges, handledEdges;
  bool hadError = false;
};

using RequiredParts = llvm::SmallSetVector<int, 4>;
static RequiredParts computeRequiredParts(Node *head) {
  RequiredParts chainParts;
  for (Node *n = head; n; n = n->next) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire || n->kind == Node::Release) && n->owner)
      chainParts.insert(n->owner->first);
    RequiredParts regionParts;
    for (Node *child : n->children) regionParts.insert_range(computeRequiredParts(child));
    SmallVector<int, 4> sorted = regionParts.takeVector();
    llvm::sort(sorted);
    n->requiredParts.assign(sorted.begin(), sorted.end());
    chainParts.insert_range(n->requiredParts);
  }
  return chainParts;
}
static bool isMultiBufferedGroup(GroupDag &g, int numTmemBlocks) {
  for (const Member &member : g.pieceTable.members)
    for (Operation *user : member.allocOp->getResult(0).getUsers()) {
      auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user);
      auto loop = dyn_cast<scf::ForOp>(user->getParentOp());
      if (!mma || !loop) continue;
      scf::ForOp ws = loop;
      for (Operation *parent = loop; parent; parent = parent->getParentOp())
        if (auto outer = dyn_cast<scf::ForOp>(parent);
            outer && gpu::hasWarpSpecializeTag(outer))
          ws = outer;
      auto shape = mma.getAccumulator().getType().getShape();
      int64_t blockM = shape[0], blockN = shape[1];
      if (nvidia_gpu::hasAccReadModifyWrite(mma, loop) || !nvidia_gpu::isAccMultibufferingPossible(mma, loop) ||
          getDisallowAccMultiBuffer(ws) ||
          numTmemBlocks + blockM * blockN * 2 > 128 * 512 ||
          (isa<nvidia_gpu::TCGen5MMAScaledOp>(mma.getOperation()) &&
           blockN == 256))
        return false;
    }
  return true;
}
static FailureOr<std::optional<int>>
computeBackingCopies(GroupDag &g, ArrayRef<EdgeRec> edges,
                     bool useMetaPartitioner, int &numTmemBlocks) {
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
      semaError(m.allocOp) << "allocs in one planned reuse group have inconsistent buffer.copy values";
      return failure();
    }
    plannedCopy = copy;
  }
  if (plannedCopy && sawMissing) {
    semaError(g.pieceTable.members.front().allocOp)
        << "planned reuse group mixes buffer.copy and non-buffer.copy allocs";
    return failure();
  }
  g.numCopies = 1;
  bool synchronized = !edges.empty();
  if (synchronized && plannedCopy) g.numCopies = *plannedCopy;
  else if (synchronized && g.isTmem() && !useMetaPartitioner && isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  if (synchronized && g.isTmem())
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2) numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return plannedCopy;
}
static void computeSemaphoreCopies(GroupDag &g, int lowerSemaphoreNumStages, std::optional<int> plannedCopy) {
  g.numSemaphoreCopies = g.numCopies;
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad)) hasProducerLoad = true;
  });
  if (!g.semas.empty() && !g.isTmem() && !plannedCopy && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (g.semas.empty()) return success();
  DenseSet<Node *> used;
  SmallVector<SmallVector<Node *, 2>, 4> releases(g.semas.size());
  forEachNode(g, [&](Node *n) {
    used.insert(n->tokenSource);
    if (n->flow)
      for (Node *exit : n->flow->exits) used.insert(exit);
    if (n->kind == Node::Release && n->sema < releases.size())
      releases[n->sema].push_back(n);
  });

  auto resolveTokenOwner = [&](Node *producer) -> std::optional<Owner> {
    DenseSet<Node *> seen;
    while (producer && seen.insert(producer).second) {
      switch (producer->kind) {
      case Node::Acquire: {
        if (producer->sema >= g.semas.size()) return std::nullopt;
        Owner owner = producer->owner;
        if (!owner && g.semas[producer->sema].entryOwner)
          owner = *g.semas[producer->sema].entryOwner;
        return std::optional<Owner>(std::in_place, owner);
      }
      case Node::For: case Node::If:
        if (producer->flow) return std::optional<Owner>(std::in_place, producer->flow->owner);
        producer = producer->tokenSource;
        break;
      default:
        return std::nullopt;
      }
    }
    return std::nullopt;
  };
  auto compatible = [&](Node *producer, const Owner &owner) {
    std::optional<Owner> actual = resolveTokenOwner(producer);
    return actual && (!actual->has_value() || sameOwner(*actual, owner));
  };
  auto nestedIn = [](Node *node, Node *region) {
    for (Node *parent = node->parent; parent; parent = parent->parent)
      if (parent == region) return true;
    return false;
  };
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifyFlow = [&](Node *n) -> LogicalResult {
    if (!n->flow) return success();
    const RegionFlow &c = *n->flow;
    bool needsInput = n->kind == Node::For || llvm::is_contained(c.exits, nullptr);
    bool malformed = (needsInput && !n->tokenSource) ||
                     (n->tokenSource && !compatible(n->tokenSource, c.owner)) ||
                     !c.sema || *c.sema >= g.semas.size() ||
                     c.exits.size() != n->children.size() || !used.contains(n);
    if (malformed) return semaError(n->op) << "malformed region token flow";
    for (auto [index, final] : llvm::enumerate(c.exits)) {
      if (!final) continue;
      std::optional<Owner> owner = resolveTokenOwner(final);
      DenseSet<Node *> childNodes;
      forEachNode(n->children[index], [&](Node *c) { childNodes.insert(c); });
      bool producer = final->kind == Node::Acquire ||
                      (final->isRegion() && final->flow);
      if (!producer || !owner || !sameOwner(*owner, c.owner) ||
          !childNodes.contains(final))
        return semaError(n->op)
               << "region path exports no exact compatible token";
    }
    return success();
  };
  auto verifyNode = [&](Node *n) -> LogicalResult {
    switch (n->kind) {
    case Node::For: case Node::If:
      return verifyFlow(n);
    case Node::Release: {
      if (n->sema >= g.semas.size() || n->consumers.empty() ||
          llvm::any_of(n->consumers, [](Node *consumer) {
            return !consumer || consumer->kind != Node::Acquire;
          }) || !n->scheduleAnchor || n->payloads.empty())
        return semaError(g.root->op) << "release has incomplete protocol facts";
      const Sema &sema = g.semas[n->sema];
      if (!n->count || !n->tokenSource || !compatible(n->tokenSource, n->owner))
        return semaError(g.root->op) << "release has no exact token source (owner "
               << ownerKey(n->owner) << ", source kind "
               << (n->tokenSource ? static_cast<int>(n->tokenSource->kind) : -1)
               << ", count " << n->count << ")";
      bool exactAnchor = n->scheduleAnchor == n->tokenSource || n->scheduleAnchor->tokenSource == n->tokenSource ||
                         (n->scheduleAnchor->kind == Node::Enter && n->tokenSource->isRegion() &&
                          n->scheduleAnchor->parent == n->tokenSource);
      bool badConsumer = llvm::any_of(n->consumers, [&](Node *consumer) {
        bool sameChain = consumer->parent == n->parent;
        bool unordered = sameChain && !precedesInChain(n, consumer) &&
                         !sema.entryOwner && !consumer->recurrenceDistance &&
                         (!n->parent || n->parent->kind != Node::For);
        return consumer->sema != n->sema || unordered;
      });
      if (!exactAnchor || badConsumer || n->count * n->payloads.size() > sema.count)
        return semaError(g.root->op) << "release has incompatible protocol facts";
      return success();
    }
    case Node::Access:
      if (!n->tokenSource || !compatible(n->tokenSource, n->owner))
        return semaError(n->op) << "buffer access has no valid owner token";
      return success();
    case Node::Acquire: {
      if (n->sema >= g.semas.size() || !n->scheduleAnchor || !n->count ||
          (!g.semas[n->sema].entryOwner && releases[n->sema].empty()))
        return semaError(g.root->op) << "acquire has no valid supply";
      const Sema &sema = g.semas[n->sema];
      if (n->count != sema.count ||
          (n->recurrenceDistance && *n->recurrenceDistance <= 0))
        return semaError(g.root->op) << "acquire has incompatible protocol facts";
      if (sema.entryOwner)
        for (Node *loop = n->parent; loop; loop = loop->parent)
          if (loop->kind == Node::For && llvm::none_of(releases[n->sema], [&](Node *release) {
                return nestedIn(release, loop);
              })) {
            if (shouldDumpDag()) dumpDagTree(g);
            return semaError(g.root->op) << "repeated entry acquire has no per-loop release";
          }
      if (n->owner) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema] && *acqClass[n->sema] != k) {
          if (shouldDumpDag()) dumpDagTree(g);
          return semaError(g.root->op) << "semaphore acquired by two partitions (M3 violation)";
        }
        acqClass[n->sema] = k;
      }
      return success();
    }
    default:
      return success();
    }
  };
  LogicalResult result = success();
  forEachNode(g, [&](Node *node) {
    if (succeeded(result)) result = verifyNode(node);
  });
  return result;
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;
using ScheduleEdge = std::pair<Operation *, Operation *>;
struct OwnerScheduleConstraint {
  Owner producerOwner, consumerOwner;
  Operation *producer = nullptr, *consumer = nullptr;
  int64_t producerStage = 0, consumerStage = 0, distance = 0;
  int64_t requiredDelay() const { return producerStage - consumerStage - distance; }
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
  GroupDag *group = nullptr; Node *node = nullptr;
  unsigned advances = 0;
};
using PhysicalKey = std::pair<int64_t, GroupDag *>;
static PhysicalKey physicalKey(GroupDag &group) {
  return group.isCircular() ? PhysicalKey{group.bufferId, nullptr} : PhysicalKey{0, &group};
}
struct PhysicalSchedules {
  llvm::MapVector<PhysicalKey, SmallVector<GroupDag *, 2>> sets;
  std::map<std::pair<PhysicalKey, Operation *>, SlotSchedule> loopSlots;
  explicit PhysicalSchedules(MutableArrayRef<GroupDag> groups) {
    for (GroupDag &group : groups)
      sets[physicalKey(group)].push_back(&group);
  }
};
static Effect slotEventEffect(const Node *n) {
  Effect effect = Effect::R;
  if (n->kind == Node::Access)
    for (const Touch &touch : n->touches) effect = joinEffect(effect, touch.effect);
  else
    for (const auto &[_, info] : n->pieceInfo) effect = joinEffect(effect, info.effect);
  return effect;
}
static bool isSlotEvent(const Node *n) { return n->kind == Node::Access || (n->isRegion() && !n->pieceInfo.empty()); }
static bool isDirectLoopNode(const Node *n) {
  return n->parent && n->parent->kind == Node::For &&
         (n->prev || n->next || llvm::is_contained(n->parent->children, n));
}

static SlotSchedule replaySlots(ArrayRef<SlotEvent> events, bool assignOffsets = false) {
  SlotSchedule result;
  DenseMap<GroupDag *, int64_t> lastProduced; // ordinal + 1; zero means absent
  int64_t cursor = -1;
  for (const SlotEvent &event : events) {
    int64_t required = 0;
    if (slotEventEffect(event.node) == Effect::W) {
      if (event.advances > 1) result.complete = false;
      cursor += event.advances;
      result.advancesPerIteration += event.advances;
      required = lastProduced[event.group] = cursor + 1;
    } else
      required = lastProduced.lookup(event.group);
    if (!required) {
      result.complete = false;
      continue;
    }
    --required;
    result.ordinalByAccess[event.node] = required;
    if (assignOffsets) event.node->bufferStageOffset = required - cursor;
  }
  return result;
}

static LogicalResult assignCircularStageOffsets(PhysicalSchedules &physical) {
  for (auto &[_, physicalSet] : physical.sets) {
    if (!physicalSet.front()->isCircular()) continue;
    SmallVector<GroupDag *, 4> set;
    llvm::copy_if(physicalSet, std::back_inserter(set), [](GroupDag *g) { return !g->semas.empty(); });
    if (set.empty()) continue;
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
        if (n->kind == Node::Access) eventsByOp[n->op].push_back(SlotEvent{g, n, slotEventEffect(n) == Effect::W});
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
        if (!forward && n->kind != Node::Release) return;
        if (n->scheduleAnchor && n->scheduleAnchor->bufferStageOffset) {
          n->stageOffset = n->scheduleAnchor->bufferStageOffset;
          return;
        }
        Node *access = n;
        do access = forward ? access->next : access->prev;
        while (access && (access->kind != Node::Access || !access->bufferStageOffset));
        if (access) n->stageOffset = access->bufferStageOffset;
      });
  }
  return success();
}

static Operation *findScheduleAnchor(const Node *anchor, bool producer = false) {
  for (const Node *n = anchor; n; n = producer ? n->prev : n->next) {
    if (n->kind == Node::Access) return producer && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion() && n->op) return n->op;
  }
  return nullptr;
}
static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet, scf::ForOp loop) {
  SmallVector<SlotEvent, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (GroupDag *group : physicalSet) {
    for (const std::unique_ptr<Node> &storage : group->nodes) {
      Node *node = storage.get();
      if (!isDirectLoopNode(node) || node->parent->op != loop.getOperation()) continue;
      if (node->kind == Node::Acquire && node->scheduleAnchor &&
          isSlotEvent(node->scheduleAnchor) && slotEventEffect(node->scheduleAnchor) == Effect::W) {
        ++advancesByAccess[node->scheduleAnchor];
        continue;
      }
      if (!isSlotEvent(node) || !node->op) continue;
      events.push_back(SlotEvent{group, node});
    }
  }
  llvm::stable_sort(events, [](const SlotEvent &lhs, const SlotEvent &rhs) {
    return lhs.node->op != rhs.node->op && lhs.node->op->isBeforeInBlock(rhs.node->op);
  });
  for (SlotEvent &event : events) event.advances = advancesByAccess.lookup(event.node);
  return replaySlots(events);
}
static const SlotSchedule &getSlotSchedule(PhysicalSchedules &physical, GroupDag &group, scf::ForOp loop) {
  PhysicalKey set = physicalKey(group);
  auto [it, inserted] = physical.loopSlots.try_emplace(std::make_pair(set, loop.getOperation()));
  if (inserted) it->second = computeSlotSchedule(physical.sets[set], loop);
  return it->second;
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}
static std::optional<int64_t> computeLoopCarriedDistance(const SlotSchedule &slots, int64_t numSemaphoreCopies,
                                                         Node *producer, Node *consumer) {
  if (numSemaphoreCopies == 1) return 1; // one slot: a loop-carried pair spans exactly one iteration
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
static bool isAliasedMultibufferedGroup(const GroupDag &group) {
  bool authored = group.numCopies > 1 &&
      llvm::all_of(group.pieceTable.members, [](const Member &m) { return m.allocOp->hasAttr(kBufferCopyAttrName); });
  const Member &first = group.pieceTable.members.front();
  bool aliases = llvm::all_of(group.pieceTable.members, [&](const Member &member) {
    return member.offset == first.offset && member.extent == first.extent &&
           member.type == first.type;
  });
  return !group.isCircular() && group.pieceTable.members.size() > 1 &&
         group.numSemaphoreCopies > 1 &&
         (authored || (group.numSemaphoreCopies > group.numCopies && aliases));
}
static LogicalResult assignAliasedHandoffStageOffsets(PhysicalSchedules &physical, GroupDag &group) {
  if (!isAliasedMultibufferedGroup(group)) return success();
  bool hasShiftedRelease = false;
  for (const auto &storage : group.nodes) {
    Node *release = storage.get();
    if (!isDirectLoopNode(release) || release->kind != Node::Release)
      continue;
    Node *producer = release->scheduleAnchor;
    std::optional<int64_t> releaseOffset;
    for (Node *acquire : release->consumers) {
      Node *consumer = acquire->scheduleAnchor;
      if (!producer || !consumer || !isSlotEvent(producer) ||
          !isSlotEvent(consumer) || producer->parent != consumer->parent ||
          !producer->parent || producer->parent->kind != Node::For)
        return semaError(producer && producer->op ? producer->op : group.root->op)
               << "multibuffered alias handoff requires direct scheduled events in one loop body";
      auto loop = cast<scf::ForOp>(producer->parent->op);
      const SlotSchedule &slots = getSlotSchedule(physical, group, loop);
      auto producerIt = slots.ordinalByAccess.find(producer);
      auto consumerIt = slots.ordinalByAccess.find(consumer);
      if (!slots.complete || producerIt == slots.ordinalByAccess.end() ||
          consumerIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0) {
        if (shouldDumpDag()) dumpDagTree(group);
        return semaError(producer->op) << "cannot derive multibuffered alias handoff slots (complete "
               << slots.complete << ", advances " << slots.advancesPerIteration
               << ", producer " << (producerIt != slots.ordinalByAccess.end()) << ", consumer "
               << (consumerIt != slots.ordinalByAccess.end()) << ")";
      }
      int64_t offset = 0;
      if (acquire->recurrenceDistance) {
        int64_t nextConsumer = consumerIt->second +
            *acquire->recurrenceDistance * slots.advancesPerIteration;
        offset = positiveMod(nextConsumer - producerIt->second, group.numSemaphoreCopies);
      } else if (precedesInChain(release, acquire)) {
        offset = positiveMod(consumerIt->second - producerIt->second, group.numSemaphoreCopies);
      } else if (!computeLoopCarriedDistance(slots, group.numSemaphoreCopies,
                                             producer, consumer)) {
        int64_t nextConsumer = consumerIt->second + slots.advancesPerIteration;
        offset = positiveMod(nextConsumer - producerIt->second, group.numSemaphoreCopies);
      }
      if (releaseOffset && *releaseOffset != offset)
        return semaError(producer->op) << "alternative acquire consumers require incompatible alias stage offsets";
      releaseOffset = offset;
    }
    if (!releaseOffset) return semaError(group.root->op) << "multibuffered alias release has no acquire consumer";
    release->stageOffset = *releaseOffset;
    hasShiftedRelease |= *releaseOffset != 0;
  }
  for (const auto &node : group.nodes) {
    if (!isDirectLoopNode(node.get())) continue;
    if (!hasShiftedRelease && node->kind == Node::Release) node->stageOffset.reset();
    if (hasShiftedRelease && node->kind == Node::Acquire) node->stageOffset = 0;
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
      if (offset[consumer] >= candidate) continue;
      offset[consumer] = candidate;
      predecessor[consumer] = edgeIndex;
      lastUpdated = edgeIndex;
    }
    if (!lastUpdated) break;
  }
  if (lastUpdated) {
    // A relaxation on pass V proves a positive cycle.  Every V-step
    // predecessor walk therefore reaches that cycle without an unset edge.
    int64_t vertex = ownerKey(constraints[*lastUpdated].consumerOwner);
    for (unsigned i = 0; i < numVertices; ++i) {
      unsigned edgeIndex = predecessor.lookup(vertex);
      vertex = ownerKey(constraints[edgeIndex].producerOwner);
    }
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
        << "fixed loop.stage assignments form an unsatisfiable semaphore handoff cycle";
    diag << " (cycle requires " << cycleDelay << " additional pipeline iteration"
         << (cycleDelay == 1 ? "" : "s") << ")";
    for (unsigned edgeIndex : llvm::reverse(cycle)) {
      const OwnerScheduleConstraint &constraint = constraints[edgeIndex];
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
    return offset[ownerKey(constraint.consumerOwner)] ==
           offset[ownerKey(constraint.producerOwner)] +
               constraint.requiredDelay();
  };
  auto hasTightPath = [&](int64_t from, int64_t to) {
    SmallVector<int64_t, 8> stack{from};
    DenseSet<int64_t> seen;
    while (!stack.empty()) {
      int64_t vertex = stack.pop_back_val();
      if (vertex == to) return true;
      if (!seen.insert(vertex).second) continue;
      for (const OwnerScheduleConstraint &constraint : constraints)
        if (isTight(constraint) &&
            ownerKey(constraint.producerOwner) == vertex)
          stack.push_back(ownerKey(constraint.consumerOwner));
    }
    return false;
  };
  auto alreadySSAOrdered = [](Operation *producer, Operation *consumer) {
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.omitUsesFromAbove = true;
    options.filter = [&](Operation *op) { return op->getBlock() == consumer->getBlock(); };
    (void)getBackwardSlice(consumer, &slice, options);
    return slice.contains(producer);
  };
  for (const OwnerScheduleConstraint &constraint : constraints) {
    bool directlySameWave = constraint.requiredDelay() == 0;
    bool onZeroDelayCycle =
        isTight(constraint) &&
        hasTightPath(ownerKey(constraint.consumerOwner),
                     ownerKey(constraint.producerOwner));
    if ((directlySameWave || onZeroDelayCycle) && !alreadySSAOrdered(constraint.producer, constraint.consumer))
      model.clusterEdges.emplace_back(constraint.producer,
                                      constraint.consumer);
  }
  return success();
}

static LogicalResult addSyncScheduleEdges(MutableArrayRef<GroupDag> groups,
                                          PhysicalSchedules &physical,
                                          llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
  for (GroupDag &group : groups)
    for (const auto &storage : group.nodes) {
      Node *release = storage.get();
      if (release->kind == Node::For && release->op &&
          release->op->hasAttr(triton::kScheduledMaxStageAttrName))
        modelsByLoop.try_emplace(release->op);
      if (!isDirectLoopNode(release) || release->kind != Node::Release)
        continue;
      for (Node *acquire : release->consumers) {
        Operation *producer =
                      findScheduleAnchor(release->scheduleAnchor,
                                         /*producer=*/true),
                  *consumer = findScheduleAnchor(acquire->scheduleAnchor);
        if (!producer || !consumer) continue;
        for (Operation *parent = producer->getParentOp(); parent;
             parent = parent->getParentOp()) {
          auto loop = dyn_cast<scf::ForOp>(parent);
          if (!loop ||
              !loop->hasAttr(triton::kScheduledMaxStageAttrName))
            continue;
          Operation *producerAnchor =
              loop.getBody()->findAncestorOpInBlock(*producer);
          Operation *consumerAnchor =
              loop.getBody()->findAncestorOpInBlock(*consumer);
          if (!producerAnchor || !consumerAnchor)
            continue;
          if (producerAnchor == consumerAnchor)
            break;
          gpu::StageCluster producerSchedule =
              gpu::getStageCluster(producerAnchor);
          gpu::StageCluster consumerSchedule =
              gpu::getStageCluster(consumerAnchor);
          if (!producerSchedule || !consumerSchedule)
            break;
          int64_t distance = acquire->recurrenceDistance.value_or(0);
          if (!acquire->recurrenceDistance &&
              !precedesInChain(release, acquire)) {
            const SlotSchedule &slots = getSlotSchedule(physical, group, loop);
            std::optional<int64_t> loopCarriedDistance =
                computeLoopCarriedDistance(
                    slots, group.numSemaphoreCopies, release->scheduleAnchor,
                    acquire->scheduleAnchor);
            if (!loopCarriedDistance) {
              InFlightDiagnostic diag =
                  semaError(producerAnchor)
                  << "cannot determine loop-carried dependency distance for a "
                     "physical buffer slot";
              diag.attachNote(consumerAnchor->getLoc())
                  << "next token ownership starts here";
              return failure();
            }
            distance = *loopCarriedDistance;
          }
          modelsByLoop[loop.getOperation()].ownerConstraints.push_back(
              OwnerScheduleConstraint{
                  release->owner, acquire->owner, producerAnchor,
                  consumerAnchor, producerSchedule->first,
                  consumerSchedule->first, distance});
          break;
        }
      }
    }
  return success();
}
static LogicalResult legalizeLoopSchedule(scf::ForOp loop, ArrayRef<ScheduleEdge> edges) {
  SmallVector<ScheduleEdge> constraints(edges.begin(), edges.end());
  DenseMap<Operation *, int64_t> cluster;
  for (Operation &consumer : loop.getBody()->without_terminator()) {
    gpu::StageCluster schedule = gpu::getStageCluster(&consumer);
    if (!schedule) continue;
    cluster[&consumer] = schedule->second;
    for (Value operand : getNestedOperands(&consumer)) {
      auto [producer, distance] = triton::getDefiningOpAndDistance(loop, operand);
      if (!producer) continue;
      producer = loop.getBody()->findAncestorOpInBlock(*producer);
      if (!producer || producer == &consumer) continue;
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producer);
      if (producerSchedule && producerSchedule->first == schedule->first + distance)
        constraints.emplace_back(producer, &consumer);
    }
  }
  for (auto [producer, consumer] : constraints)
    if (!producer->isBeforeInBlock(consumer) && cluster.contains(producer) &&
        cluster.contains(consumer))
      cluster[producer] =
          std::min(cluster.lookup(producer), cluster.lookup(consumer) - 1);
  for (unsigned iteration = 0; iteration <= cluster.size(); ++iteration) {
    bool changed = false;
    for (auto [producer, consumer] : constraints) {
      if (!cluster.contains(producer) || !cluster.contains(consumer)) continue;
      int64_t required = cluster.lookup(producer) + (producer->isBeforeInBlock(consumer) ? 0 : 1);
      if (cluster.lookup(consumer) >= required) continue;
      cluster[consumer] = required;
      changed = true;
    }
    if (!changed) break;
    if (iteration == cluster.size()) {
      InFlightDiagnostic diag = semaError(loop) << "cyclic loop.cluster constraints";
      if (shouldDumpDag())
        for (auto [producer, consumer] : constraints)
          diag.attachNote(producer->getLoc()) << producer->getName()
              << " (cluster " << cluster.lookup(producer) << ") -> "
              << consumer->getName() << " (cluster "
              << cluster.lookup(consumer) << ")";
      return failure();
    }
  }
  int64_t rebase = 0;
  for (auto [op, legalized] : cluster)
    rebase = std::max(rebase,
                      gpu::getStageCluster(op)->second - legalized);
  OpBuilder builder(loop.getContext());
  for (auto [op, legalized] : cluster) {
    gpu::StageCluster oldSchedule = gpu::getStageCluster(op);
    if (legalized > std::numeric_limits<int32_t>::max() - rebase)
      return semaError(op) << "legalized loop.cluster exceeds i32 range";
    int64_t newCluster = legalized + rebase;
    if (newCluster < oldSchedule->second)
      return semaError(op) << "legalization lowered an authored loop.cluster";
    if (newCluster == oldSchedule->second) continue;
    gpu::setStageCluster(builder, op, std::make_pair(oldSchedule->first, static_cast<int>(newCluster)));
  }
  return success();
}
static gpu::StageCluster scheduleAtOwnerBoundary(const Node *n, gpu::StageCluster schedule) {
  if (!schedule || findScheduleAnchor(n->next)) return schedule;
  auto forOp = dyn_cast_or_null<scf::ForOp>(n->parent ? n->parent->op : nullptr);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName)) return schedule;
  auto [stage, cluster] = *schedule;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate || candidate->first != stage || !gpu::hasPartition(&op)) continue;
    if (gpu::getPartitionIds(&op).contains(n->owner->first)) cluster = std::max(cluster, candidate->second);
  }
  return std::make_pair(stage, cluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      if (!n->owner) break;
      gpu::StageCluster boundary =
          scheduleAtOwnerBoundary(n, cache.lookup(ownerKey(n->owner)));
      if (n->recurrenceDistance && n->next && n->next->kind == Node::Exit)
        n->stageCluster = boundary;
      else {
        Operation *anchor = findScheduleAnchor(
            n->scheduleAnchor ? n->scheduleAnchor : n->next);
        n->stageCluster = anchor ? gpu::getStageCluster(anchor) : boundary;
      }
      break;
    }
    case Node::Release:
      if (n->owner) n->stageCluster = cache.lookup(ownerKey(n->owner));
      break;
    case Node::Access:
      if (n->owner)
        cache[ownerKey(n->owner)] = gpu::getStageCluster(
            n->completionAnchor ? n->completionAnchor : n->op);
      break;
    case Node::For: {
      ScheduleCache body = cache;
      assignSyncScheduleChain(n->children[0], body);
      if (!gpu::hasWarpSpecializeTag(n->op)) cache = std::move(body);
      break;
    }
    case Node::If: {
      ScheduleCache thenCache = cache, elseCache = cache;
      assignSyncScheduleChain(n->children[0], thenCache);
      assignSyncScheduleChain(n->children[1], elseCache);
      cache = std::move(thenCache);
      for (auto &[key, stageCluster] : elseCache)
        cache.try_emplace(key, stageCluster);
      break;
    }
    default: break;
    }
  }
}
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, LoopScheduleModel> modelsByLoop;
  PhysicalSchedules physical(groups);
  if (failed(assignCircularStageOffsets(physical))) return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(physical, group))) return failure();
  if (failed(addSyncScheduleEdges(groups, physical, modelsByLoop))) return failure();
  auto scheduleFailure = [&]() -> LogicalResult {
    if (shouldDumpDag())
      for (GroupDag &group : groups) dumpDagTree(group);
    return failure();
  };
  for (auto &[loopOp, model] : modelsByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    if (failed(solveOwnerScheduleConstraints(model))) return scheduleFailure();
    if (failed(legalizeLoopSchedule(loop, model.clusterEdges))) return scheduleFailure();
  }
  for (GroupDag &g : groups)
    for (Node *head : g.root->children) {
      ScheduleCache cache;
      assignSyncScheduleChain(head, cache);
    }
  return success();
}

LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages, int &numTmemBlocks) {
  SmallVector<EdgeRec> edges;
  DenseSet<Node *> reusable;
  for (Node *head : g.root->children) {
    ChainState top; // function chain starts at bottom (first-touch)
    ChainWalker(g, top, edges, reusable, /*underFor=*/false).run(head);
  }
  reduceEdges(g, edges, reusable);
  FailureOr<std::optional<int>> plannedCopy = computeBackingCopies(g, edges, useMetaPartitioner, numTmemBlocks);
  if (failed(plannedCopy)) return failure();
  if (!edges.empty() && failed(DirectBuilder(g, edges, reusable).run())) return failure();
  for (Node *head : g.root->children) computeRequiredParts(head);
  computeSemaphoreCopies(g, lowerSemaphoreNumStages, *plannedCopy);
  if (!g.semas.empty() && !g.ttDescriptorFedMembers.empty())
    return semaError(g.ttDescriptorFedMembers.front()) <<
           "managed local_alloc sourced from a tt-form descriptor load — "
           "nvws-insert-allocas must convert this upstream (pipeline invariant violated)";
  return verifySyncDag(g);
}
static void printPieceRecord(llvm::raw_ostream &os, const Node *n, Operation *anchor) {
  if (n->pieceInfo.empty()) return;
  os << " pieces{";
  llvm::interleaveComma(sortedPieceInfo(n), os, [&](const auto &item) {
    os << "P" << item.first << ":" << (item.second.effect == Effect::W ? "W" : "R")
       << ":" << ownerStr(anchor, item.second.owner);
  });
  os << "}";
}
static StringRef semaName(const GroupDag &g, const Node *node) {
  if (node->sema < g.semas.size()) return g.semas[node->sema].name;
  return "<unformed>";
}
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g, const Node *region, unsigned chainIdx) {
  if (!region || !region->flow) return;
  os << " yield{";
  const RegionFlow &c = *region->flow;
  Node *f = chainIdx < c.exits.size() ? c.exits[chainIdx] : nullptr;
  if (!f) os << "pass";
  else if (f->kind == Node::Acquire || f->kind == Node::Release)
    os << (f->kind == Node::Acquire ? "a " : "r ") << semaName(g, f);
  else os << (f->kind == Node::For ? "scf.for" : "scf.if");
  os << "}";
}
static void dumpDagChain(GroupDag &g, const Node *head, unsigned depth, const Node *region, unsigned chainIdx) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    Operation *anchor = n->parent ? n->parent->op : nullptr;
    switch (n->kind) {
    case Node::Access: {
      os << treePrefix(depth) << "|- ";
      llvm::interleaveComma(n->touches, os, [&](const Touch &t) {
        os << (t.effect == Effect::W ? "W" : "R") << " m" << t.member; });
      os << "  " << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner) << "\n";
      break;
    }
    case Node::Acquire: case Node::Release: {
      bool acquire = n->kind == Node::Acquire;
      os << treePrefix(depth) << "|- " << (acquire ? "a" : "r") << "  " << semaName(g, n);
      if (n->count > 1) os << "(" << n->count << ")";
      os << "  " << ownerStr(anchor, n->owner);
      if (acquire && n->sema < g.semas.size() && g.semas[n->sema].entryOwner && !n->owner)
        os << "  ; entry";
      if (!acquire) {
        os << " [";
        llvm::interleaveComma(n->payloads, os, [&](AsyncOp p) { os << nvws::stringifyAsyncOp(p); });
        os << "]"; }
      if (n->stageOffset) os << "  stage-offset=" << *n->stageOffset;
      os << "\n";
      break;
    }
    case Node::For: case Node::If: {
      bool loop = n->kind == Node::For;
      os << treePrefix(depth) << "|- " << (loop ? "scf.for" : "scf.if");
      if (loop && gpu::hasWarpSpecializeTag(n->op)) os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      printPieceRecord(os, n, loop ? n->op : anchor);
      if (!n->requiredParts.empty()) {
        os << " parts{";
        llvm::interleaveComma(n->requiredParts, os);
        os << "}"; }
      if (n->flow) os << " thread{" << ownerStr(n->op, n->flow->owner) << "}";
      os << "\n";
      if (loop) { dumpDagChain(g, n->children[0], depth + 1, n, 0); break; }
      os << treePrefix(depth + 1) << "|- then\n";
      dumpDagChain(g, n->children[0], depth + 2, n, 0);
      bool virtualElse = !cast<scf::IfOp>(n->op).elseBlock();
      os << treePrefix(depth + 1) << "|- else" << (virtualElse ? " (virtual)" : "") << "\n";
      dumpDagChain(g, n->children[1], depth + 2, n, 1);
      break;
    }
    case Node::Enter: case Node::Exit:
      os << treePrefix(depth) << (n->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
      printPieceRecord(os, n, anchor);
      if (n->kind == Node::Exit) printYieldInfo(os, g, region, chainIdx);
      os << "\n";
      break;
    case Node::Func: break;
    }
  }
}
static void dumpDagTree(GroupDag &g) { for (Node *head : g.root->children) dumpDagChain(g, head, 1, nullptr, 0); }
void dumpGroupSyncDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "SYNC-DAG\n|- func @" << funcOp.getName() << "\n";
  dumpDagTree(g);
  if (g.semas.empty()) {
    os << "  BACKING: untouched (no semaphores)\n";
    return;
  }
  os << "  SEMAS: ";
  llvm::interleave(g.semas, os, [&](const Sema &s) {
    os << s.name << "{count=" << s.count;
    if (s.entryOwner) os << " entry inherit=" << ownerStr(nullptr, *s.entryOwner);
    os << "}";
  }, " ");
  os << "\n" << "  BACKING: numCopies=" << g.numCopies << "\n";
}
} // namespace mlir::triton::nvws_semas
