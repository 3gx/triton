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
  llvm::MapVector<Owner, ActiveUse> uses;
  bool initialized() const { return source.node != nullptr; }
  ActiveUse *useFor(const Owner &owner) {
    auto it = uses.find(owner);
    return it == uses.end() ? nullptr : &it->second;
  }
  bool canReuseToken(const Owner &owner, Effect effect) {
    if (!initialized()) return false;
    if (effect == Effect::R) return useFor(owner);
    return llvm::all_of(uses, [&](const auto &entry) {
      const ActiveUse &use = entry.second;
      return sameOwner(use.owner, owner) || llvm::is_contained(use.orderedBefore, ownerKey(owner));
    });
  }
  void startVersion(const Owner &producer, const Owner &sourceOwner, Node *node, const Payloads &payloads) {
    source = VersionSource{producer, sourceOwner, node, payloads};
    uses.clear();
    uses.insert({sourceOwner,
                 ActiveUse{sourceOwner, node, payloads, {}}});
  }
};
using ChainState = std::map<PieceId, PieceState>;
struct Tokens {
  struct Token {
    Owner owner;
    Node *producer = nullptr;
    Node *last = nullptr;
    Payloads payloads;
    bool closed = false;
  };
  llvm::MapVector<Owner, Token> live;
  const Token *find(const Owner &owner) const {
    auto it = live.find(owner);
    return it == live.end() ? nullptr : &it->second;
  }
  const Token *last() const {
    for (const auto &[_, token] : llvm::reverse(live))
      if (!token.closed) return &token;
    return nullptr;
  }
  Token *find(const Owner &owner) { return const_cast<Token *>(std::as_const(*this).find(owner)); }
  const Token *findOpen(const Owner &owner) const {
    const Token *token = find(owner);
    return token && !token->closed ? token : nullptr;
  }
  const Token *findProducer(Node *producer) const {
    auto it = llvm::find_if(live, [&](const auto &entry) {
      return entry.second.producer == producer;
    });
    return it == live.end() ? nullptr : &it->second;
  }
  void record(const Owner &owner, Node *producer, Node *last, const Payloads &payloads) {
    assert(producer && "recorded token needs a producer");
    assert(last && "recorded token needs a completion");
    live.erase(owner);
    if (owner.has_value()) live.erase(Owner{});
    live.insert({owner, Token{owner, producer, last, payloads, false}});
  }
  void eraseOwner(const Owner &owner) { live.erase(owner); }
  void eraseProducer(Node *producer) {
    assert(producer && "cannot erase a token without its exact producer");
    while (const Token *token = findProducer(producer)) {
      Owner owner = token->owner;
      live.erase(owner);
    }
  }
  void close(Node *producer) {
    for (auto &[_, token] : live)
      if (token.producer == producer) token.closed = true;
  }
  void clear() { live.clear(); }
};

struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  Payloads payloads;
  SmallVector<PieceId, 2> pieces;
  bool preserve = false;
  bool handled = false;
  SmallVector<Node *, 2> rawSources;
};
using EdgeList = SmallVector<EdgeRec>;
static bool hasAsyncPayload(ArrayRef<AsyncOp> payloads) {
  assert(!payloads.empty() && "completion payloads are never empty");
  // unionPayloads keeps the enum sorted, and NONE is its lowest value.
  return payloads.back() != AsyncOp::NONE;
}
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
  auto forOp = cast<scf::ForOp>(node->op);
  if (forOp->hasAttr("ttg.must-execute")) return true;
  std::optional<APInt> count = forOp.getStaticTripCount();
  return count && (forOp.getUnsignedCmp() ? count->ugt(0) : count->sgt(0));
}

static void raiseForeignUseEdges(PieceState &piece, PieceId id, const Owner &owner, Node *node,
                                 EdgeList &edges, bool wsAdopt) {
  for (const auto &[_, use] : piece.uses)
    if (!sameOwner(use.owner, owner) && !(wsAdopt && !use.owner) &&
        !llvm::is_contained(use.orderedBefore, ownerKey(owner)))
      edges.push_back(EdgeRec{use.node, node, use.owner, owner, use.payloads, {id},
                              use.node == piece.source.node && hasAsyncPayload(use.payloads)});
}

// Doc: sync-dag.md#the-per-access-rules-in-full
static void applyTouch(PieceState &piece, PieceId id, const Owner &owner, Effect effect, Node *node,
                       const Payloads &payloads, EdgeList &edges, bool wsAdopt) {
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
    ActiveUse *source = piece.useFor(piece.source.sourceOwner);
    bool exact = source && source->node == piece.source.node;
    edges.push_back(EdgeRec{piece.source.node, node, piece.source.sourceOwner,
                            owner, piece.source.payloads, {id},
                            exact && hasAsyncPayload(piece.source.payloads)});
    if (exact && !llvm::is_contained(source->orderedBefore, ownerKey(owner)))
      source->orderedBefore.push_back(ownerKey(owner));
  }
  piece.uses.insert({owner, ActiveUse{owner, node, payloads, {}}});
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
          if (ActiveUse *use = it->second.useFor(info.owner)) facts = use->orderedBefore;
        result.emplace(id, std::move(facts));
      }
    return result;
  }
private:
  // Doc: sync-dag.md#memory-edges-and-token-supply
  void visitAccess(Node *node) {
    PieceEffects effects;
    forEachTouchedPiece(group, node, [&](PieceId id, Effect effect) { mergeEffect(effects, id, effect); });
    const Tokens::Token *last = chainTokens.last();
    bool ownerDiffers = last && node->owner && !sameOwner(last->owner, node->owner);
    bool canReuse = chainTokens.find(node->owner);
    for (auto [id, effect] : effects)
      canReuse &= state[id].canReuseToken(node->owner, effect);
    size_t edgeStart = edges.size();
    Payloads payloads{asyncPayloadOf(node->op)};
    // Preserve prior async completions in a same-owner synchronous write wave.
    bool synchronousWrite = payloads.front() == AsyncOp::NONE;
    if (group.pieceTable.members.size() > 1 && canReuse && synchronousWrite) {
      for (auto [id, effect] : effects) {
        if (effect != Effect::W) continue;
        PieceState &piece = state[id];
        if (!sameOwner(piece.source.sourceOwner, node->owner)) continue;
        bool sameOwnerWave = true;
        for (const auto &[_, use] : piece.uses)
          sameOwnerWave &= sameOwner(use.owner, node->owner);
        if (sameOwnerWave) unionPayloads(payloads, piece.source.payloads);
      }
    }
    for (auto [id, effect] : effects)
      applyTouch(state[id], id, node->owner, effect, node, payloads, edges, /*wsAdopt=*/false);
    bool noDataEdge = edges.size() == edgeStart;
    bool reusesToken = noDataEdge && canReuse;
    if (reusesToken)
      reusable.insert(node);
    else if (ownerDiffers && noDataEdge)
      edges.push_back(EdgeRec{last->last, node, last->owner, node->owner, last->payloads, {},
                              hasAsyncPayload(last->payloads)});
    if (node->owner && !(ownerDiffers && reusesToken)) chainTokens.record(node->owner, node, node, payloads);
  }

  // Doc: sync-dag.md#composition-nested-regions-in-the-walk
  void visitRegion(Node *node) {
    struct RegionInput { std::optional<VersionSource> source; OrderFacts order; };
    auto infos = sortedPieceInfo(node);
    std::map<PieceId, RegionInput> incoming;
    for (auto [id, info] : infos) {
      RegionInput &input = incoming[id];
      if (auto it = state.find(id); it != state.end() && it->second.initialized()) {
        input.source = it->second.source;
        if (ActiveUse *use = it->second.useFor(info.owner))
          input.order = use->orderedBefore;
      }
    }
    Payloads none{AsyncOp::NONE};
    bool wsAdopt = node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [id, info] : infos) applyTouch(state[id], id, info.owner, info.effect, node, none, edges, wsAdopt);
    ExitFacts returned;
    for (auto [childIndex, childHead] : llvm::enumerate(node->children)) {
      ChainState child;
      for (auto [id, info] : sortedPieceInfo(childHead)) {
        const std::optional<VersionSource> &before = incoming[id].source;
        Owner producer = before ? before->producer : info.owner;
        Payloads seed = before && sameOwner(info.owner, before->producer) ? before->payloads
                                                                          : Payloads{AsyncOp::NONE};
        child[id].startVersion(producer, info.owner, childHead, seed);
      }
      ChainWalker nested(group, child, edges, reusable, underFor || node->kind == Node::For);
      ExitFacts childFacts = nested.run(childHead);
      for (auto [id, info] : infos) {
        OrderFacts branchOrder = incoming[id].order;
        if (auto it = childFacts.find(id); it != childFacts.end()) branchOrder = it->second;
        if (childIndex == 0) returned[id] = std::move(branchOrder);
        else returned[id] = intersectOrderFacts(returned[id], branchOrder);
      }
    }
    if (node->kind == Node::For && !knownNonEmptyLoop(node))
      for (auto [id, info] : infos)
        returned[id] = intersectOrderFacts(returned[id], incoming[id].order);
    for (auto [id, info] : infos) {
      auto facts = returned.find(id);
      ActiveUse *use = state[id].useFor(info.owner);
      if (facts == returned.end() || !use || use->node != node) continue;
      for (int64_t owner : facts->second)
        if (!llvm::is_contained(use->orderedBefore, owner)) use->orderedBefore.push_back(owner);
    }
    assert(!infos.empty() && "retained region must touch the group");
    chainTokens.clear();
    if (auto owner = uniformPieceOwner(node); owner && owner->has_value())
      chainTokens.record(*owner, node, node, {AsyncOp::NONE});
  }
  void visitExit(Node *node) {
    for (auto [id, info] : sortedPieceInfo(node)) {
      auto it = state.find(id);
      if (it == state.end()) continue;
      PieceState &piece = it->second;
      if (underFor || pieceTouchedAfter(group, node->parent, id))
        raiseForeignUseEdges(piece, id, info.owner, node, edges, /*wsAdopt=*/false);
      ActiveUse carried{info.owner, node, {AsyncOp::NONE}, {}};
      if (ActiveUse *use = piece.useFor(info.owner)) carried = *use;
      piece.uses.clear();
      piece.uses.insert({carried.owner, carried});
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
  return edge.dst->kind == Node::Exit && edge.src->kind == Node::Access && edge.srcOwner &&
         edge.dstOwner;
}
static ArrayRef<unsigned> edgesAt(const EdgeBuckets &buckets, Node *node) {
  auto it = buckets.find(node);
  return it == buckets.end() ? ArrayRef<unsigned>{} : ArrayRef<unsigned>(it->second);
}
// Doc: sync-dag.md#1-implied-ordering-reduceedges. Drops use only kept edges.
static void reduceStraightEdges(Node *head, const Positions &positions, ArrayRef<EdgeRec> edges,
                                const EdgeBuckets &atDst, std::vector<bool> &drop,
                                DenseSet<Node *> &reusable) {
  KnownOrder order;
  Snapshots snapshots;
  std::optional<int64_t> tokenOwner;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner) tokenOwner = ownerKey(pi.owner);
  for (Node *n = head; n; n = n->next) {
    for (unsigned ei : edgesAt(atDst, n)) {
      const EdgeRec &e = edges[ei];
      if (!e.srcOwner || !e.dstOwner || e.src->kind != Node::Access) continue;
      int64_t dk = ownerKey(e.dstOwner);
      unsigned srcIdx = positions.lookup(e.src);
      bool hasToken = tokenOwner == dk;
      if (!e.preserve && order.covers(e, srcIdx) && hasToken && e.dst->kind == Node::Access) {
        drop[ei] = true;
        reusable.insert(e.dst);
        continue;
      }
      tokenOwner = dk; // Kept acquire supplies Q's token.
      order.apply(e, srcIdx, snapshots);
    }
    if (n->owner) order.record(n, positions.lookup(n), snapshots);
  }
}
static void reduceLoopCloses(GroupDag &g, Node *head, const Positions &positions, ArrayRef<EdgeRec> edges,
                             const EdgeBuckets &atDst, ArrayRef<unsigned> closes, Node *firstAccess,
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
      if (!e.preserve && order.covers(e, positions.lookup(e.src)) &&
          tokenAvailable.contains(dk) &&
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
      return positions.lookup(edges[a].src) > positions.lookup(edges[b].src);
    });
  reduceStraightEdges(head, positions, edges, atDst, drop, reusable);
  if (head->parent && head->parent->kind == Node::For)
    reduceLoopCloses(g, head, positions, edges, atDst, closes, firstAccess, drop);
  for (Node *n = head; n; n = n->next)
    for (Node *child : n->children) reduceChain(g, child, edges, drop, reusable);
}
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges, DenseSet<Node *> &reusable) {
  if (edges.empty()) return;
  using Key = std::tuple<Node *, Owner, Owner>;
  std::map<Key, llvm::SmallSetVector<Node *, 2>> rawSources;
  auto key = [](const EdgeRec &edge) { return Key{edge.dst, edge.srcOwner, edge.dstOwner}; };
  for (const EdgeRec &edge : edges) rawSources[key(edge)].insert(edge.src);
  for (EdgeRec &edge : edges)
    edge.rawSources.append(rawSources[key(edge)].begin(), rawSources[key(edge)].end());
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
  DirectBuilder(GroupDag &group, MutableArrayRef<EdgeRec> edges, const DenseSet<Node *> &reuse)
      : g(group), edges(edges), reusable(reuse) {
    for (EdgeRec &edge : edges) {
      atDst[edge.dst].push_back(&edge);
      remainingEdges.insert(&edge);
      if (edge.dst->kind == Node::Exit) exitSources.insert(edge.src);
    }
  }
  LogicalResult run() {
    assert(!g.root->children.empty() && "edges require an access DAG");
    Chain placed;
    placeChain(g.root->children[0], placed);
    if (hadError) return failure();
    if (remainingEdges.empty() && placed.supplies.empty())
      return formSemaphores();
    dumpSyncDagTree(g);
    if (!remainingEdges.empty()) {
      EdgeRec *missing = remainingEdges.front();
      return semaError(missing->dst->op ? missing->dst->op : g.root->op)
             << "reduced dependency edge was not placed";
    }
    return semaError(g.root->op) << "conditional release path was not consumed ("
                                 << placed.supplies.size() << ")";
  }
private:
  using EdgeRefs = SmallVector<EdgeRec *, 2>;
  // Alternatives agree on arrivals; coexecuting supplies add arrivals.
  struct Supply {
    SmallVector<Node *, 2> releases;
    unsigned arrivals = 0;
    bool empty() const { return releases.empty(); }
    void appendExecuted(Supply other) {
      arrivals += other.arrivals;
      releases.append(other.releases.begin(), other.releases.end());
    }
    void appendExecuted(Node *release) {
      if (!release) return;
      arrivals += release->count * release->payloads.size();
      releases.push_back(release);
    }
  };
  using BoundaryKey = std::pair<Node *, Owner>;
  using PendingSupplies = std::map<BoundaryKey, Supply>;
  struct Watch {
    struct Demand {
      Node *node = nullptr, *channel = nullptr, *anchor = nullptr;
    };
    std::optional<Owner> owner;
    bool hasRealInput = false;
    Node *lastCompletion = nullptr;
    std::map<Owner, Demand> demands;
    void observe(std::optional<Owner> candidate, Node *completion) {
      if (owner && candidate && completion && sameOwner(*owner, *candidate)) lastCompletion = completion;
    }
  };
  struct Chain {
    Tokens tokens;
    EdgeRefs pending;
    PendingSupplies supplies;
    Node *guard = nullptr;
    Watch watch;
  };
  struct LoopSupply { EdgeRefs entry, closes; };
  using LoopSupplies = llvm::MapVector<Owner, LoopSupply>;
  struct BranchExit {
    Chain chain;
    Node *exit = nullptr;
    EdgeRefs exitEdges;
    std::optional<Tokens::Token> token;
    bool passesInput = false;
  };
  static BoundaryKey boundaryKey(Node *region, const Owner &owner) { return {region, owner}; }
  static void setEntryToken(Node *enter, const Tokens::Token &token) {
    enter->tokenSource = token.producer;
    enter->scheduleAnchor = token.last;
    enter->payloads = token.payloads;
  }
  EdgeRefs unhandled(ArrayRef<EdgeRec *> refs) const {
    EdgeRefs result;
    llvm::copy_if(refs, std::back_inserter(result), [](const EdgeRec *edge) { return !edge->handled; });
    return result;
  }
  EdgeRefs unhandledAt(Node *node) const { return unhandled(atDst.lookup(node)); }
  void markHandled(EdgeRec *edge) {
    edge->handled = true;
    remainingEdges.remove(edge);
  }
  static Supply takeSupply(Chain &chain, Node *source, const Owner &owner) {
    auto entry = chain.supplies.extract(boundaryKey(source, owner));
    return entry.empty() ? Supply{} : std::move(entry.mapped());
  }
  Node *regionChannelFor(Node *region, const Owner &owner) const {
    auto it = regionChannels.find(boundaryKey(region, owner));
    return it == regionChannels.end() ? nullptr : it->second;
  }
  void fail(Node *node, StringRef message) {
    semaError(node && node->op ? node->op : g.root->op) << message;
    hadError = true;
  }
  void appendAlternative(Supply &paths, Supply branch, Node *region) {
    if (branch.empty()) {
      fail(region, "conditional path provides no semaphore arrival");
      return;
    }
    if (!paths.empty() && paths.arrivals != branch.arrivals) {
      fail(region, "conditional paths provide incompatible semaphore counts");
      return;
    }
    paths.arrivals = branch.arrivals;
    paths.releases.append(branch.releases.begin(), branch.releases.end());
  }
  Node *concreteAcquire(Node *producer) const {
    if (!producer) return nullptr;
    switch (producer->kind) {
    case Node::Acquire: return producer;
    case Node::For: case Node::If:
      if (producer->flow)
        for (Node *exit : producer->flow->exits)
          if (Node *acquire = concreteAcquire(exit)) return acquire;
      break;
    default: break;
    }
    return nullptr;
  }
  static bool scalableRelease(const Node *release) {
    return llvm::all_of(release->payloads, [](AsyncOp p) { return p == AsyncOp::NONE || p == AsyncOp::WGMMA; });
  }
  static Node *chainExit(Node *head) {
    while (head->next) head = head->next;
    assert(head->kind == Node::Exit && "structured chain must end at EXIT");
    return head;
  }
  void observeRegionDemand(Node *region, Chain &chain) const {
    Watch &watch = chain.watch;
    std::optional<Owner> owner = uniformPieceOwner(region);
    if (!owner) return;
    Node *channel = regionChannelFor(region, *owner);
    bool deferred = chain.supplies.count(boundaryKey(region, *owner));
    if (!region->flow && !region->tokenSource && !channel && !deferred) return;
    Node *anchor = region->scheduleAnchor ? region->scheduleAnchor : region;
    watch.demands.try_emplace(*owner,
                              Watch::Demand{region, channel, anchor});
  }
  Node *precedingChannel(Node *node, const Owner &owner) const {
    for (Node *cursor = node; cursor && cursor->parent; cursor = cursor->parent)
      for (Node *prev = cursor->prev; prev; prev = prev->prev)
        if (prev->op) return regionChannelFor(prev, owner);
    return nullptr;
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
    assert(parent && "every acquire must belong to a channel");
    if (parent == acquire) return acquire;
    return channelParent[acquire] = findChannel(parent);
  }
  void uniteChannels(Node *lhs, Node *rhs) {
    Node *left = findChannel(lhs), *right = findChannel(rhs);
    channelParent[right] = left;
  }
  void spliceAfterLast(Node *node, Node *anchor) {
    Node *last = anchor;
    while (last->next && last->next->scheduleAnchor == anchor) last = last->next;
    spliceAfter(node, last);
  }
  Node *makeAcquire(Node *anchor, const Owner &owner, bool after = false) {
    Node *acquire = newProtocolNode(g, Node::Acquire, anchor->parent, owner);
    acquire->scheduleAnchor = anchor;
    after ? spliceAfterLast(acquire, anchor) : spliceBefore(acquire, anchor);
    channelParent[acquire] = acquire;
    return acquire;
  }
  Tokens::Token sourceToken(const EdgeRec &edge, const Tokens &tokens) const {
    switch (edge.src->kind) {
    case Node::Access:
      return {edge.srcOwner, edge.src->tokenSource, edge.src, edge.payloads};
    case Node::For:
    case Node::If: {
      Node *producer = edge.src->flow ? edge.src : edge.src->tokenSource;
      if (!producer) return {};
      if (const Tokens::Token *token = tokens.findProducer(producer)) return *token;
      return {edge.srcOwner, producer, edge.src, edge.payloads};
    }
    case Node::Acquire: {
      const Tokens::Token *token = tokens.findProducer(edge.src);
      assert(token && "synthetic acquire edge must retain its token");
      return *token;
    }
    case Node::Enter:
      return {edge.srcOwner, edge.src->tokenSource, edge.src->scheduleAnchor, edge.src->payloads};
    default:
      return {};
    }
  }
  Node *materializeRelease(Tokens::Token &token, const Owner &owner, Node *acquire, Node *guard) {
    if (!token.producer || !token.last) {
      fail(token.last, "release has no exact source token");
      return nullptr;
    }
    Node *release = newProtocolNode(g, Node::Release, guard ? guard->parent : token.last->parent, owner);
    release->payloads = token.payloads;
    assert(!release->payloads.empty() && "token must retain completion payload");
    release->tokenSource = token.producer;
    release->sat = acquire;
    release->scheduleAnchor = token.last;
    guard ? spliceAfter(release, guard) : spliceAfterLast(release, token.last);
    return release;
  }
  Node *insertRelease(const EdgeRec &edge, Node *acquire, Tokens &tokens, Node *guard = nullptr) {
    Tokens::Token source = sourceToken(edge, tokens);
    switch (edge.src->kind) {
    case Node::Enter:
      guard = edge.src;
      break;
    default: break;
    }
    Node *release = materializeRelease(source, edge.srcOwner, acquire, guard);
    if (release) tokens.close(source.producer);
    return release;
  }
  void applySupply(Node *acquire, const Supply &supply,
                   unsigned requiredCount = 0) {
    unsigned expected = supply.arrivals;
    if (expected && requiredCount && expected != requiredCount) {
      if (supply.releases.size() != 1 ||
          !scalableRelease(supply.releases.front()) ||
          supply.releases.front()->payloads.empty() ||
          requiredCount % supply.releases.front()->payloads.size())
        return fail(acquire, "one execution path does not supply the acquire pending count");
      Node *release = supply.releases.front();
      unsigned payloads = release->payloads.size();
      release->count = requiredCount / payloads;
      expected = requiredCount;
    }
    for (Node *release : supply.releases) {
      if (release->sat && release->sat != acquire) {
        this->fail(release, "one release cannot satisfy two acquire sites");
        return;
      }
      release->sat = acquire;
    }
    acquire->count = std::max(acquire->count, expected);
  }
  Supply collectSupply(Node *acquire, ArrayRef<EdgeRec *> refs, Tokens &tokens,
                       Chain &chain, Node *guard = nullptr) {
    llvm::MapVector<std::pair<Node *, Node *>, EdgeRec> incoming;
    for (const EdgeRec *edgeRef : refs) {
      EdgeRec edge = *edgeRef;
      Tokens::Token source = sourceToken(edge, tokens);
      for (Node *candidate : edge.rawSources) {
        EdgeRec raw = edge;
        raw.src = candidate;
        if (sourceToken(raw, tokens).producer == source.producer &&
            precedesInChain(edge.src, candidate))
          edge.src = candidate;
      }
      Node *producer = source.producer ? source.producer : edge.src;
      auto [entry, inserted] = incoming.try_emplace({producer, edge.src->parent}, edge);
      if (inserted) continue;
      EdgeRec &prior = entry->second;
      unionPayloads(prior.payloads, edge.payloads);
      if (precedesInChain(prior.src, edge.src)) prior.src = edge.src;
    }
    Supply supply;
    for (const auto &[_, edge] : incoming) {
      if (edge.src->isRegion() && !edge.src->flow) {
        if (Node *channel = regionChannelFor(edge.src, edge.srcOwner);
            channel && acquire) {
          if (sameOwner(edge.srcOwner, edge.dstOwner)) {
            uniteChannels(acquire, channel);
            acquire->count = std::max(acquire->count, channel->count);
          } else {
            Node *drain = makeAcquire(edge.src, edge.srcOwner, true);
            drain->count = channel->count;
            uniteChannels(drain, channel);
            tokens.record(edge.srcOwner, drain, drain, {AsyncOp::NONE});
            EdgeRec handoff{drain, acquire, edge.srcOwner, edge.dstOwner, {AsyncOp::NONE}, {}};
            supply.appendExecuted(insertRelease(handoff, acquire, tokens, guard));
          }
          continue;
        }
        Supply alternatives = takeSupply(chain, edge.dst, edge.dstOwner);
        if (!alternatives.empty()) {
          supply.appendExecuted(std::move(alternatives));
          continue;
        }
      }
      supply.appendExecuted(insertRelease(edge, acquire, tokens, guard));
    }
    for (EdgeRec *edge : refs) markHandled(edge);
    return supply;
  }
  static void dropOwner(Tokens &tokens, Tokens incoming, const Owner &owner) {
    tokens = std::move(incoming);
    tokens.eraseOwner(owner);
  }
  static void publishRegionFlow(Node *region, Node *input, const Owner &owner, ArrayRef<Node *> exits,
                                const Payloads &payloads, Tokens incoming, Tokens &tokens) {
    assert(!exits.empty() && "region flow needs an exit per branch");
    bool passthrough = exits.front() == region;
    region->tokenSource = input;
    region->producedTokenOwner.emplace(owner);
    if (!passthrough) {
      RegionFlow flow;
      flow.owner = owner;
      flow.exits.append(exits.begin(), exits.end());
      region->flow.emplace(std::move(flow));
    }
    tokens = std::move(incoming);
    if (input) tokens.eraseProducer(input);
    tokens.record(owner, passthrough ? input : region, region, payloads);
  }
  Chain enterChain(Chain chain, const Tokens &incoming, Node *enter) {
    chain.tokens = incoming;
    std::optional<Owner> owner = uniformPieceOwner(enter);
    const Tokens::Token *last = chain.tokens.last();
    if (owner && owner->has_value() && last && !last->owner) {
      Tokens::Token adopted = *last;
      adopted.owner = *owner;
      chain.tokens.record(adopted.owner, adopted.producer, adopted.last, adopted.payloads);
    }
    if (const Tokens::Token *token = owner ? chain.tokens.findOpen(*owner) : nullptr)
      setEntryToken(enter, *token);
    return chain;
  }
  BranchExit normalizeBranch(Chain branch, Node *region, const Owner &owner) {
    BranchExit result;
    result.exit = chainExit(branch.guard);
    result.exitEdges = unhandledAt(result.exit);
    if (const Tokens::Token *open = branch.tokens.findOpen(owner))
      result.token = *open;
    result.passesInput = result.token && result.token->producer == region;
    result.chain = std::move(branch);
    return result;
  }
  Supply materializeExitSupply(BranchExit &branch, Node *acquire,
                               const Owner &owner) {
    Supply supply = collectSupply(acquire, branch.exitEdges,
                                  branch.chain.tokens, branch.chain);
    supply.appendExecuted(takeSupply(branch.chain, branch.exit, owner));
    return supply;
  }
  void placeAccess(Node *node, Chain &chain) {
    Tokens &tokens = chain.tokens;
    EdgeRefs direct = unhandledAt(node);
    EdgeRefs inherited;
    llvm::erase_if(chain.pending, [&](EdgeRec *edge) {
      if (!sameOwner(edge->dstOwner, node->owner)) return false;
      inherited.push_back(edge);
      return true;
    });
    bool noIncoming = direct.empty() && inherited.empty();
    bool suppliedByRegion = chain.supplies.count(boundaryKey(node, node->owner));
    if (noIncoming && reusable.contains(node))
      if (Node *channel = precedingChannel(node, node->owner)) {
        Node *&acquire = node->tokenSource;
        if (!acquire) acquire = makeAcquire(node, node->owner);
        acquire->count = std::max(acquire->count, channel->count);
        uniteChannels(acquire, channel);
      }
    Tokens::Token *owned = tokens.find(node->owner);
    if (!node->tokenSource && noIncoming && owned &&
        reusable.contains(node)) {
      node->tokenSource = owned->producer;
      owned->last = node;
      owned->payloads = {asyncPayloadOf(node->op)};
      // Per-piece reuse keeps a read fan-out in the releasing owner's wave.
      owned->closed = false;
      return;
    }
    bool preplaced = node->tokenSource;
    Node *&acquire = node->tokenSource;
    if (!acquire) acquire = makeAcquire(node, node->owner);
    assert(acquire->kind == Node::Acquire && "preplaced non-acquire token");
    Supply supply = collectSupply(acquire, direct, tokens, chain);
    supply.appendExecuted(collectSupply(acquire, inherited, tokens, chain, chain.guard));
    supply.appendExecuted(takeSupply(chain, node, node->owner));
    applySupply(acquire, supply);
    if (!preplaced && noIncoming && !suppliedByRegion) {
      seeded.insert(acquire);
      acquire->count = 1;
    }
    tokens.record(node->owner, acquire, node, {asyncPayloadOf(node->op)});
  }
  void placeIf(Node *region, Chain &chain, Node *next) {
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    Tokens incoming = chain.tokens;
    Chain adopted = enterChain(Chain{}, incoming, region->children.front());
    const Tokens::Token *input = boundaryOwner ? adopted.tokens.findOpen(*boundaryOwner) : nullptr;
    if (input) {
      region->tokenSource = input->producer;
      region->producedTokenOwner.emplace(*boundaryOwner);
    }
    SmallVector<Chain, 2> branches;
    for (Node *child : region->children) {
      Chain branch = enterChain(chain, incoming, child);
      if (input) {
        branch.tokens.record(*boundaryOwner, region, child, input->payloads);
        setEntryToken(child, {*boundaryOwner, region, child, input->payloads});
      }
      branch.pending = atDst.lookup(region);
      branch.pending.append(chain.pending.begin(), chain.pending.end());
      branch.guard = child;
      placeChain(child, branch);
      branches.push_back(std::move(branch));
    }
    chain.pending.clear();
    if (!boundaryOwner) {
      chain.tokens = std::move(incoming);
      return;
    }
    Owner owner = *boundaryOwner;
    SmallVector<BranchExit, 2> exits;
    bool allPass = true, complete = true;
    for (Chain &branch : branches) {
      exits.push_back(
          normalizeBranch(std::move(branch), region, owner));
      allPass &= exits.back().passesInput;
      complete &= exits.back().token.has_value();
    }
    if (allPass) {
      assert(input && "pass-through region must have an input token");
      chain.tokens = adopted.tokens;
      chain.tokens.eraseProducer(input->producer);
      chain.tokens.record(owner, input->producer, region, input->payloads);
      return;
    }
    bool publish = reusableUse(next, owner), allowCompletion = false, discard = false;
    if (!next) {
      if (chain.watch.owner) {
        publish = sameOwner(owner, *chain.watch.owner) && (g.numCopies == 1 || chain.watch.hasRealInput);
        allowCompletion = publish;
      } else if (chain.guard) {
        std::optional<Owner> parentOwner = uniformPieceOwner(chain.guard);
        publish = parentOwner && sameOwner(owner, *parentOwner);
      } else {
        publish = exitSources.contains(region);
        discard = !publish;
      }
    }
    if (allowCompletion && !complete) {
      complete = true;
      for (BranchExit &branch : exits) {
        if (branch.token) continue;
        bool hasPending = branch.chain.supplies.count(
            boundaryKey(branch.exit, owner));
        if (branch.exitEdges.empty() && !hasPending) {
          complete = false;
          break;
        }
        Node *acquire = makeAcquire(branch.exit, owner);
        Supply supply = materializeExitSupply(branch, acquire, owner);
        applySupply(acquire, supply);
        branch.chain.tokens.record(owner, acquire, acquire, {AsyncOp::NONE});
        branch.token = *branch.chain.tokens.findOpen(owner);
        branch.passesInput = false;
      }
    }
    if (publish && complete) {
      SmallVector<Node *, 2> producers;
      for (BranchExit &branch : exits) {
        producers.push_back(branch.passesInput ? nullptr
                                               : branch.token->producer);
        for (EdgeRec *edge : branch.exitEdges) markHandled(edge);
      }
      publishRegionFlow(region, input ? input->producer : nullptr, owner,
                        producers, {AsyncOp::NONE}, std::move(incoming),
                        chain.tokens);
      return;
    }
    if (allowCompletion || discard) {
      region->tokenSource = nullptr;
      dropOwner(chain.tokens, std::move(incoming), owner);
      return;
    }
    Supply paths;
    for (BranchExit &branch : exits) {
      if (input)
        replaceTokenSource(branch.chain.guard, region, input->producer);
      Supply branchPaths = materializeExitSupply(branch, nullptr, owner);
      Node *guard = branch.exit->prev ? branch.exit->prev
                                      : branch.chain.guard;
      if (const Tokens::Token *open =
              branch.chain.tokens.findOpen(owner)) {
        Tokens::Token exact = *open;
        if (input && exact.producer == region) exact.producer = input->producer;
        branchPaths.appendExecuted(
            materializeRelease(exact, exact.owner, nullptr, guard));
      }
      appendAlternative(paths, std::move(branchPaths), region);
    }
    Node *target = next;
    if (!target && (chain.guard || chain.watch.owner))
      target = chainExit(chain.guard ? chain.guard : region);
    if (!target) {
      fail(region, "conditional release path has no exact continuation");
      return;
    }
    std::optional<Owner> targetOwner = target->kind == Node::Access ? std::optional<Owner>(target->owner)
                                                                    : uniformPieceOwner(target);
    Owner routedOwner = targetOwner.value_or(owner);
    chain.supplies[boundaryKey(target, routedOwner)].appendExecuted(std::move(paths));
    dropOwner(chain.tokens, std::move(incoming), owner);
    region->tokenSource = nullptr;
    SmallVector<Node *, 2> sources{region};
    if (input) sources.push_back(input->producer);
    for (EdgeRec *edge : unhandledAt(target)) {
      Tokens::Token source = sourceToken(*edge, chain.tokens);
      if (llvm::is_contained(sources, edge->src) ||
          llvm::is_contained(sources, source.producer))
        markHandled(edge);
    }
  }
  void replaceTokenSource(Node *head, Node *from, Node *to) {
    forEachNode(head, [&](Node *node) {
      if (node->tokenSource == from) node->tokenSource = to;
    });
  }
  void supplyLoopEntry(Node *region, Chain &chain, Tokens &incoming, const Tokens &loopInputs,
                       Node *acquire, const Owner &owner, LoopSupply &supply, bool seed) {
    if (!supply.entry.empty()) {
      seeded.erase(acquire);
      Supply entry = collectSupply(acquire, supply.entry, incoming, chain, chain.guard);
      applySupply(acquire, entry, /*requiredCount=*/acquire->count);
      return;
    }
    const Tokens::Token *initial = loopInputs.findOpen(owner);
    if (!initial || initial->last->kind == Node::Enter) {
      if (seed) seeded.insert(acquire);
      return;
    }
    Tokens::Token token = *initial;
    Owner releaseOwner = token.producer->kind == Node::Acquire ? token.producer->owner : token.owner;
    Node *guard = token.last->parent == region->parent ? nullptr : chain.guard;
    seeded.erase(acquire);
    Node *release = materializeRelease(token, releaseOwner, acquire, guard);
    Supply entry;
    entry.appendExecuted(release);
    applySupply(acquire, entry, /*requiredCount=*/acquire->count);
  }
  LoopSupplies indexLoopSupplies(Node *exit, ArrayRef<EdgeRec *> rawEntry) {
    LoopSupplies supplies;
    for (EdgeRec *edge : unhandledAt(exit))
      supplies[edge->dstOwner].closes.push_back(edge);
    for (EdgeRec *edge : unhandled(rawEntry))
      supplies[edge->dstOwner].entry.push_back(edge);
    return supplies;
  }
  void placeMixedOwnerFor(Node *region, Chain &chain,
                          ArrayRef<EdgeRec *> rawEntry, Tokens incoming) {
    Node *body = region->children.front(), *exit = chainExit(body);
    Chain bodyState = enterChain(Chain{}, incoming, body);
    Tokens loopInputs = bodyState.tokens;
    placeChain(body, bodyState);
    Tokens &bodyTokens = bodyState.tokens;
    LoopSupplies supplies = indexLoopSupplies(exit, rawEntry);
    for (const auto &[key, _] : bodyState.supplies) supplies[key.second];
    for (auto &[owner, supply] : supplies) {
      auto demandIt = bodyState.watch.demands.find(owner);
      if (demandIt == bodyState.watch.demands.end()) {
        fail(region, "mixed-owner loop close has no point-of-use demand");
        continue;
      }
      Node *first = demandIt->second.node;
      Node *acquire = demandIt->second.channel;
      if (!acquire) acquire = first->tokenSource;
      if (!acquire || acquire->kind != Node::Acquire) {
        Node *old = acquire;
        acquire = makeAcquire(first, owner);
        first->tokenSource = acquire;
        if (old) replaceTokenSource(first, old, acquire);
      }
      Supply paths =
          collectSupply(acquire, supply.closes, bodyTokens, bodyState);
      paths.appendExecuted(takeSupply(bodyState, exit, owner));
      applySupply(acquire, paths);
      supplyLoopEntry(region, chain, incoming, loopInputs, acquire, owner,
                      supply, true);
      acquire->count = std::max(acquire->count, 1u);
      regionChannels[boundaryKey(region, owner)] = acquire;
      incoming.eraseOwner(owner);
    }
    chain.tokens = std::move(incoming);
  }
  void placeUniformOwnerFor(Node *region, Chain &chain,
                            Node *&lastOwnerAccess, const Owner &owner,
                            ArrayRef<EdgeRec *> rawEntry, Tokens incoming,
                            Node *continuation) {
    Node *body = region->children.front(), *exit = chainExit(body);
    Chain bodyState = enterChain(Chain{}, incoming, body);
    Tokens loopInputs = bodyState.tokens;
    bodyState.watch.owner.emplace(owner);
    if (const Tokens::Token *watchedInput = loopInputs.findOpen(owner))
      bodyState.watch.hasRealInput = watchedInput->last->kind != Node::Enter;
    region->producedTokenOwner.emplace(owner);
    bodyState.tokens.record(owner, region, body, {AsyncOp::NONE});
    setEntryToken(body, {owner, region, body, {AsyncOp::NONE}});
    placeChain(body, bodyState);
    lastOwnerAccess = bodyState.watch.lastCompletion;
    Watch::Demand boundaryDemand;
    auto demandIt = bodyState.watch.demands.find(owner);
    if (demandIt != bodyState.watch.demands.end())
      boundaryDemand = demandIt->second;
    Node *demand = boundaryDemand.node;
    Node *childChannel = boundaryDemand.channel;
    Tokens &bodyTokens = bodyState.tokens;
    LoopSupplies supplies = indexLoopSupplies(exit, rawEntry);
    auto closeSupply = [&](Node *acquire, LoopSupply &supply) {
      Supply paths = collectSupply(acquire, supply.closes, bodyTokens, bodyState);
      paths.appendExecuted(takeSupply(bodyState, exit, acquire->owner));
      applySupply(acquire, paths);
    };
    LoopSupply &supply = supplies[owner];
    const Tokens::Token *initial = loopInputs.findOpen(owner);
    Node *demandAnchor = boundaryDemand.anchor;
    auto scheduleOf = [](Node *node) -> gpu::StageCluster {
      return node && node->op ? gpu::getStageCluster(node->op) : std::nullopt;
    };
    gpu::StageCluster firstSchedule = scheduleOf(demandAnchor);
    gpu::StageCluster lastSchedule = scheduleOf(bodyState.watch.lastCompletion);
    bool crossStage = g.numCopies == 1 && demand && demand->tokenSource == region && firstSchedule && lastSchedule &&
                      firstSchedule->first != lastSchedule->first;
    const Tokens::Token *output = bodyTokens.findOpen(owner);
    bool hasSupply = !supply.closes.empty() || bodyState.supplies.count(boundaryKey(exit, owner));
    bool returnsToken = output && !hasSupply;
    auto loopInput = [&]() -> Node * {
      if (supply.entry.empty() && initial) {
        incoming.record(initial->owner, initial->producer, initial->last, initial->payloads);
        return initial->producer;
      }
      Node *acquire = makeAcquire(region, owner);
      supplyLoopEntry(region, chain, incoming, loopInputs, acquire, owner, supply, true);
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
      publishRegionFlow(region, input, owner, {output->producer}, output->payloads,
                        std::move(incoming), chain.tokens);
      return;
    }
    auto closeAtExit = [&](Node *input, bool distanceOne) {
      assert(input && "loop close requires an input");
      Node *tail = makeAcquire(exit, owner);
      tail->scheduleAnchor = demandAnchor;
      if (distanceOne) tail->recurrenceDistance = 1;
      closeSupply(tail, supply);
      if (input->kind == Node::Acquire) uniteChannels(tail, input);
      return tail;
    };
    crossStage &= hasSupply;
    bool nestedInput = false;
    Node *firstUse = demand;
    for (Node *node = body; node && node != demand; node = node->next) {
      nestedInput |= llvm::any_of(node->children, [&](Node *child) { return usesTokenSource(child, region); });
      if (firstUse == demand && node->kind != Node::Enter && node->tokenSource == region)
        firstUse = node;
    }
    bool forwardsInput = nestedInput ||
                         (demand && demand->isRegion() && !demand->flow && demand->tokenSource == region);
    bool carryInput = hasSupply && (bodyState.watch.hasRealInput || forwardsInput) &&
                      !reusableUse(continuation, owner);
    if (crossStage || carryInput) {
      Node *input = loopInput();
      Node *tail = closeAtExit(input, true);
      region->scheduleAnchor = demandAnchor;
      publishRegionFlow(region, input, owner, {tail}, {AsyncOp::NONE}, std::move(incoming), chain.tokens);
      return;
    }
    if (!demand) {
      if (supply.entry.empty() && !hasSupply) dropOwner(chain.tokens, std::move(incoming), owner);
      else fail(region, "loop has no recorded point-of-use demand");
      return;
    }
    if (childChannel) {
      if (!hasSupply) {
        supplyLoopEntry(region, chain, incoming, loopInputs, childChannel, owner, supply, false);
        region->scheduleAnchor = demandAnchor;
      } else {
        Node *input = loopInput();
        Node *tail = closeAtExit(input, g.numCopies == 1);
        Tokens::Token bridge{owner, tail, tail, {AsyncOp::NONE}};
        materializeRelease(bridge, bridge.owner, childChannel, nullptr)->count =
            std::max(1u, childChannel->count);
      }
      regionChannels[boundaryKey(region, owner)] = childChannel;
      dropOwner(chain.tokens, std::move(incoming), owner);
      return;
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
    supplyLoopEntry(region, chain, incoming, loopInputs, recurrence, owner, supply, true);
    region->scheduleAnchor = demandAnchor;
    regionChannels[boundaryKey(region, owner)] = recurrence;
    for (const EdgeRec *edge : supply.entry) incoming.eraseOwner(edge->srcOwner);
    dropOwner(chain.tokens, std::move(incoming), owner);
  }
  void placeFor(Node *region, Chain &chain, Node *&lastOwnerAccess,
                Node *continuation) {
    EdgeRefs rawEntry = atDst.lookup(region);
    rawEntry.append(chain.pending.begin(), chain.pending.end());
    chain.pending.clear();
    Tokens incoming = chain.tokens;
    lastOwnerAccess = nullptr;
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    if (!boundaryOwner) {
      placeMixedOwnerFor(region, chain, rawEntry, std::move(incoming));
      return;
    }
    placeUniformOwnerFor(region, chain, lastOwnerAccess, *boundaryOwner,
                         rawEntry, std::move(incoming), continuation);
  }
  void placeChain(Node *head, Chain &chain) {
    for (Node *node = head; node;) {
      Node *next = node->next;
      Node *continuation =
          next && next->kind != Node::Exit ? next : nullptr;
      switch (node->kind) {
      case Node::Access:
        placeAccess(node, chain);
        chain.watch.observe(std::optional<Owner>(std::in_place, node->owner), node);
        chain.watch.demands.try_emplace(
            node->owner, Watch::Demand{node, nullptr, node});
        break;
      case Node::If:
        placeIf(node, chain, continuation);
        observeRegionDemand(node, chain);
        break;
      case Node::For: {
        Node *completion = nullptr;
        placeFor(node, chain, completion, continuation);
        chain.watch.observe(uniformPieceOwner(node), completion);
        observeRegionDemand(node, chain);
        break;
      }
      default: break;
      }
      node = next;
    }
  }
  std::optional<SemaId> channelOfProducer(Node *producer) const {
    if (Node *acquire = concreteAcquire(producer)) return acquire->sema;
    return producer && producer->flow ? producer->flow->sema : std::nullopt;
  }
  LogicalResult formSemaphores() {
    llvm::MapVector<Node *, SmallVector<Node *, 2>> classes;
    DenseMap<Node *, SmallVector<Node *, 2>> releasesAt;
    DenseMap<Node *, Owner> entryOwners;
    for (Node *acquire : g.nodesOfKind(Node::Acquire)) {
      Node *root = findChannel(acquire);
      classes[root].push_back(acquire);
      if (seeded.contains(acquire)) entryOwners.try_emplace(root, acquire->owner);
    }
    for (Node *release : g.nodesOfKind(Node::Release)) {
      if (release->sat) releasesAt[release->sat].push_back(release);
    }
    for (auto &[root, sites] : classes) {
      unsigned count = 1;
      for (Node *site : sites) count = std::max(count, site->count);
      for (Node *site : sites) {
        SmallVector<Node *, 2> &releases = releasesAt[site];
        if (site->count == count) continue;
        if (releases.empty() && seeded.contains(site)) continue;
        bool scalable = releases.size() == 1 && scalableRelease(releases.front());
        unsigned payloads = scalable ? releases.front()->payloads.size() : 0;
        if (!scalable || count % payloads)
          return semaError(site->op ? site->op : g.root->op) << "incompatible path counts for one semaphore channel";
        releases.front()->count = count / payloads;
      }
      SemaId sid = g.semas.size();
      Sema sema{"S" + std::to_string(sid), count};
      if (auto entry = entryOwners.find(root); entry != entryOwners.end())
        sema.entryOwner.emplace(entry->second);
      g.semas.push_back(std::move(sema));
      for (Node *site : sites) {
        site->sema = sid;
        site->count = count;
        site->producedTokenOwner.emplace(
            site->owner ? site->owner : entryOwners.lookup(root));
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
  MutableArrayRef<EdgeRec> edges;
  const DenseSet<Node *> &reusable;
  DenseMap<Node *, EdgeRefs> atDst;
  DenseMap<Node *, Node *> channelParent;
  std::map<BoundaryKey, Node *> regionChannels;
  DenseSet<Node *> seeded;
  DenseSet<Node *> exitSources;
  llvm::SmallSetVector<EdgeRec *, 8> remainingEdges;
  bool hadError = false;
};

using RequiredParts = llvm::SmallSetVector<int, 4>;
static RequiredParts computeRequiredParts(Node *head) {
  RequiredParts chainParts;
  for (Node *n = head; n; n = n->next) {
    if (n->owner)
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
static FailureOr<std::optional<int>> computeBackingCopies(GroupDag &g, ArrayRef<EdgeRec> edges,
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
  if (edges.empty()) return plannedCopy;
  if (plannedCopy) g.numCopies = *plannedCopy;
  else if (g.isTmem() && !useMetaPartitioner && isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  if (g.isTmem())
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2) numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return plannedCopy;
}
static void computeSemaphoreCopies(GroupDag &g, int lowerSemaphoreNumStages, std::optional<int> plannedCopy) {
  bool hasProducerLoad = false;
  for (Node *release : g.nodesOfKind(Node::Release))
    hasProducerLoad |= llvm::is_contained(release->payloads,
                                         AsyncOp::TMALoad);
  bool stageSemaphores = !g.semas.empty() && !g.isTmem() && !plannedCopy &&
                         hasProducerLoad;
  g.numSemaphoreCopies = stageSemaphores ? std::max(1, lowerSemaphoreNumStages) : g.numCopies;
}
static LogicalResult verifySyncDag(GroupDag &g) {
  if (g.semas.empty()) return success();
  DenseSet<Node *> used;
  SmallVector<SmallVector<Node *, 2>, 4> releases(g.semas.size());
  forEachNode(g, [&](Node *n) {
    used.insert(n->tokenSource);
    if (n->flow) for (Node *exit : n->flow->exits) used.insert(exit);
    if (n->kind == Node::Release && n->sema < releases.size()) releases[n->sema].push_back(n);
  });
  auto compatible = [&](Node *producer, const Owner &owner) {
    assert(producer && "token-owner queries require an exact producer");
    const std::optional<Owner> &actual = producer->producedTokenOwner;
    return actual && (!actual->has_value() || sameOwner(*actual, owner));
  };
  auto childOf = [](Node *n, Node *region) { while (n && n->parent != region) n = n->parent; return n; };
  auto contains = [](Node *head, Node *target) {
    bool found = false;
    forEachNode(head, [&](Node *node) { found |= node == target; });
    return found;
  };
  auto structuralError = [&](StringRef message) -> LogicalResult {
    dumpSyncDagTree(g);
    return semaError(g.root->op) << message;
  };
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifyFlow = [&](Node *n) -> LogicalResult {
    if (!n->flow) return success();
    const RegionFlow &c = *n->flow;
    bool needsInput = n->kind == Node::For || llvm::is_contained(c.exits, nullptr);
    if ((needsInput && !n->tokenSource) ||
        (n->tokenSource && !compatible(n->tokenSource, c.owner)) || !c.sema ||
        c.exits.size() != n->children.size() || !used.contains(n))
      return semaError(n->op) << "region has incomplete exact token flow";
    if (*c.sema >= g.semas.size()) return semaError(n->op) << "region has incomplete exact token flow";
    for (auto [index, final] : llvm::enumerate(c.exits)) {
      if (!final) continue;
      const std::optional<Owner> &owner = final->producedTokenOwner;
      if (!owner) return semaError(n->op) << "region path exports no compatible token";
      if (!sameOwner(*owner, c.owner) || !contains(n->children[index], final))
        return semaError(n->op) << "region path exports no compatible token";
    }
    return success();
  };
  auto verifyNode = [&](Node *n) -> LogicalResult {
    switch (n->kind) {
    case Node::For: case Node::If: return verifyFlow(n);
    case Node::Release: {
      if (n->sema >= g.semas.size() || !n->sat || !n->scheduleAnchor ||
          n->payloads.empty() || !n->count || !n->tokenSource ||
          !compatible(n->tokenSource, n->owner))
        return semaError(g.root->op) << "release has no exact token source";
      if (n->sat->kind != Node::Acquire) return semaError(g.root->op) << "release has no exact token source";
      const Sema &sema = g.semas[n->sema];
      bool exactAnchor = n->scheduleAnchor == n->tokenSource ||
                         n->scheduleAnchor->tokenSource == n->tokenSource ||
                         (n->scheduleAnchor->kind == Node::Enter && n->tokenSource->isRegion() &&
                          n->scheduleAnchor->parent == n->tokenSource);
      if (!exactAnchor) return semaError(g.root->op) << "release lost its exact completion";
      if (n->sat->sema != n->sema || n->count * n->payloads.size() > sema.count)
        return semaError(g.root->op) << "release has incompatible acquire";
      bool mustPrecede = n->sat->parent == n->parent && !sema.entryOwner &&
                         !n->sat->recurrenceDistance &&
                         (!n->parent || n->parent->kind != Node::For);
      if (mustPrecede && !precedesInChain(n, n->sat))
        return semaError(g.root->op) << "release does not precede its acquire";
      return success();
    }
    case Node::Access:
      if (!n->tokenSource || !compatible(n->tokenSource, n->owner))
        return semaError(n->op) << "buffer access has no valid owner token";
      return success();
    case Node::Acquire: {
      if (n->sema >= g.semas.size() || !n->scheduleAnchor || !n->count)
        return semaError(g.root->op) << "acquire has no valid supply";
      if ((!g.semas[n->sema].entryOwner && releases[n->sema].empty()) ||
          n->count != g.semas[n->sema].count)
        return semaError(g.root->op) << "acquire has no valid supply";
      const Sema &sema = g.semas[n->sema];
      if (sema.entryOwner)
        for (Node *loop = n->parent; loop; loop = loop->parent)
          if (loop->kind == Node::For && llvm::none_of(releases[n->sema], [&](Node *release) {
                return childOf(release, loop);
              }))
            return structuralError("repeated entry acquire has no per-loop release");
      if (n->owner) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema].value_or(k) != k)
          return structuralError("semaphore acquired by two partitions (M3 violation)");
        acqClass[n->sema] = k;
      }
      if (n->recurrenceDistance.value_or(1) <= 0)
        return semaError(g.root->op) << "recurrence acquire has non-positive distance";
      return success();
    }
    default: return success();
    }
  };
  LogicalResult result = success();
  forEachNode(g, [&](Node *node) { if (succeeded(result)) result = verifyNode(node); });
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
  GroupDag *group = nullptr;
  Node *node = nullptr;
  unsigned advances = 0;
};
using PhysicalKey = std::pair<int64_t, GroupDag *>;
static PhysicalKey physicalKey(GroupDag &group) {
  return group.isCircular() ? PhysicalKey{group.bufferId, nullptr} : PhysicalKey{0, &group};
}
struct PhysicalSet {
  SmallVector<GroupDag *, 2> groups;
  DenseMap<Operation *, SlotSchedule> loopSlots;
};
using PhysicalSets = llvm::MapVector<PhysicalKey, PhysicalSet>;
static bool isDirectLoopNode(const Node *n) { return n->parent && n->parent->kind == Node::For; }
static SlotSchedule replaySlots(ArrayRef<SlotEvent> events, bool assignOffsets = false) {
  SlotSchedule result;
  DenseMap<GroupDag *, int64_t> lastProduced; // ordinal + 1; zero means absent
  int64_t cursor = -1;
  for (const SlotEvent &event : events) {
    int64_t required = 0;
    if (*event.node->slotEffect == Effect::W) {
      result.complete &= event.advances <= 1;
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
static LogicalResult assignCircularStageOffsets(PhysicalSets &physical) {
  for (auto &[_, physicalSet] : physical) {
    if (!physicalSet.groups.front()->isCircular()) continue;
    SmallVector<GroupDag *, 4> set;
    llvm::copy_if(physicalSet.groups, std::back_inserter(set),
                  [](GroupDag *g) { return !g->semas.empty(); });
    if (set.empty()) continue;
    auto type = set.front()->pieceTable.members.front().type;
    int64_t numCopies = set.front()->numCopies;
    DenseSet<int64_t> starts;
    DenseMap<Operation *, SmallVector<SlotEvent, 1>> eventsByOp;
    for (GroupDag *g : set) {
      if (g->pieceTable.members.size() != 1) return semaError(g->root->op) << "malformed circular local logical group";
      const Member &member = g->pieceTable.members.front();
      if (member.type != type) return semaError(member.allocOp) << "circular local group has mismatched member types";
      if (g->numCopies != numCopies)
        return semaError(member.allocOp) << "circular local group has mismatched buffer.copy";
      if (member.circularStart < 0 || member.circularStart >= numCopies)
        return semaError(member.allocOp) << "circular buffer.start is outside buffer.copy";
      if (!starts.insert(member.circularStart).second)
        return semaError(member.allocOp) << "duplicate circular buffer.start in one group";
      for (Node *access : g->nodesOfKind(Node::Access))
        eventsByOp[access->op].push_back(
            SlotEvent{g, access, *access->slotEffect == Effect::W});
    }
    SmallVector<SlotEvent> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end()) ordered.append(it->second.begin(), it->second.end());
    });
    SlotSchedule slots = replaySlots(ordered, /*assignOffsets=*/true);
    for (const SlotEvent &event : ordered) {
      const Member &member = event.group->pieceTable.members.front();
      if (event.advances) {
        assert(slots.ordinalByAccess.contains(event.node));
        int64_t stage = slots.ordinalByAccess.lookup(event.node);
        if (member.circularStart != stage % numCopies)
          return semaError(member.allocOp) << "circular producer order expects buffer.start "
                 << stage % numCopies << ", got " << member.circularStart;
      } else if (!slots.ordinalByAccess.contains(event.node))
        return semaError(member.allocOp) << "circular consumer appears before producer";
    }
    auto assignProtocolOffset = [](Node *n, Node *Node::*step) {
      if (n->scheduleAnchor->bufferStageOffset) {
        n->stageOffset = n->scheduleAnchor->bufferStageOffset;
        return;
      }
      Node *access = n;
      do access = access->*step;
      while (access &&
             (access->kind != Node::Access || !access->bufferStageOffset));
      if (access) n->stageOffset = access->bufferStageOffset;
    };
    for (GroupDag *g : set) {
      for (Node *acquire : g->nodesOfKind(Node::Acquire))
        assignProtocolOffset(acquire, &Node::next);
      for (Node *release : g->nodesOfKind(Node::Release))
        assignProtocolOffset(release, &Node::prev);
    }
  }
  return success();
}
static Operation *findScheduleAnchor(const Node *anchor, bool producer = false) {
  for (const Node *n = anchor; n; n = producer ? n->prev : n->next) {
    if (n->kind == Node::Access) return producer && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion()) return n->op;
  }
  return nullptr;
}
static const SlotSchedule &computeSlotSchedule(PhysicalSet &physicalSet, scf::ForOp loop) {
  auto [cached, inserted] = physicalSet.loopSlots.try_emplace(loop.getOperation());
  if (!inserted) return cached->second;
  SmallVector<SlotEvent, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (GroupDag *group : physicalSet.groups) {
    for (const std::unique_ptr<Node> &storage : group->nodes) {
      Node *node = storage.get();
      if (!isDirectLoopNode(node) || node->parent->op != loop.getOperation()) continue;
      switch (node->kind) {
      case Node::Acquire:
        if (node->scheduleAnchor->slotEffect &&
            *node->scheduleAnchor->slotEffect == Effect::W)
          ++advancesByAccess[node->scheduleAnchor];
        break;
      case Node::Access:
        events.push_back(SlotEvent{group, node});
        break;
      case Node::For:
      case Node::If:
        assert(node->slotEffect && "retained region must be a slot event");
        events.push_back(SlotEvent{group, node});
        break;
      default:
        break;
      }
    }
  }
  llvm::stable_sort(events, [](const SlotEvent &lhs, const SlotEvent &rhs) {
    return lhs.node->op != rhs.node->op && lhs.node->op->isBeforeInBlock(rhs.node->op);
  });
  for (SlotEvent &event : events) event.advances = advancesByAccess.lookup(event.node);
  return cached->second = replaySlots(events);
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus; return remainder < 0 ? remainder + modulus : remainder;
}
struct SlotPair {
  int64_t producer, consumer, advances;
};
static std::optional<SlotPair> findSlotPair(const SlotSchedule &slots, Node *producer,
                                            Node *consumer) {
  auto producerIt = slots.ordinalByAccess.find(producer);
  auto consumerIt = slots.ordinalByAccess.find(consumer);
  if (!slots.complete || slots.advancesPerIteration <= 0 ||
      producerIt == slots.ordinalByAccess.end() ||
      consumerIt == slots.ordinalByAccess.end())
    return std::nullopt;
  return SlotPair{producerIt->second, consumerIt->second,
                  slots.advancesPerIteration};
}
static std::optional<int64_t> computeLoopCarriedDistance(const SlotSchedule &slots, int64_t numSemaphoreCopies,
                                                         Node *producer, Node *consumer) {
  if (numSemaphoreCopies == 1) return 1; // one slot: a loop-carried pair spans exactly one iteration
  std::optional<SlotPair> pair = findSlotPair(slots, producer, consumer);
  if (!pair) return std::nullopt;
  int64_t orbit = numSemaphoreCopies / std::gcd(numSemaphoreCopies, pair->advances);
  for (int64_t distance = 1; distance <= orbit; ++distance)
    if (positiveMod(pair->consumer + distance * pair->advances, numSemaphoreCopies) ==
        positiveMod(pair->producer, numSemaphoreCopies))
      return distance;
  return std::nullopt;
}
static bool isAliasedMultibufferedGroup(const GroupDag &group) {
  if (group.isCircular() || group.pieceTable.members.size() < 2 ||
      group.numSemaphoreCopies <= 1)
    return false;
  bool authored = group.numCopies > 1, aliased = true;
  const Member &first = group.pieceTable.members.front();
  for (const Member &member : group.pieceTable.members) {
    authored &= member.allocOp->hasAttr(kBufferCopyAttrName);
    aliased &= member.offset == first.offset && member.extent == first.extent &&
               member.type == first.type;
  }
  if (authored || group.numSemaphoreCopies <= group.numCopies) return authored;
  return aliased;
}
static LogicalResult assignAliasedHandoffStageOffsets(PhysicalSets &physical, GroupDag &group) {
  if (!isAliasedMultibufferedGroup(group)) return success();
  bool hasShiftedRelease = false;
  for (Node *release : group.nodesOfKind(Node::Release)) {
    if (!isDirectLoopNode(release)) continue;
    Node *producer = release->scheduleAnchor;
    Node *consumer = release->sat->scheduleAnchor;
    bool valid = producer->slotEffect && consumer->slotEffect &&
                 producer->parent == consumer->parent &&
                 producer->parent->kind == Node::For;
    if (!valid)
      return semaError(producer->op ? producer->op : group.root->op)
             << "multibuffered alias handoff requires direct scheduled events "
                "in one loop body";
    auto loop = cast<scf::ForOp>(producer->parent->op);
    const SlotSchedule &slots = computeSlotSchedule(physical[physicalKey(group)], loop);
    std::optional<SlotPair> pair = findSlotPair(slots, producer, consumer);
    if (!pair) {
      dumpSyncDagTree(group);
      return semaError(producer->op) << "cannot derive multibuffered alias handoff slots (complete "
             << slots.complete << ", advances " << slots.advancesPerIteration
             << ", producer " << slots.ordinalByAccess.contains(producer)
             << ", consumer " << slots.ordinalByAccess.contains(consumer) << ")";
    }
    int64_t numSemaphoreCopies = group.numSemaphoreCopies;
    int64_t offset = 0;
    if (release->sat->recurrenceDistance) {
      int64_t nextConsumer = pair->consumer + *release->sat->recurrenceDistance * pair->advances;
      offset = positiveMod(nextConsumer - pair->producer, numSemaphoreCopies);
    } else if (precedesInChain(release, release->sat)) {
      offset = positiveMod(pair->consumer - pair->producer, numSemaphoreCopies);
    } else if (!computeLoopCarriedDistance(slots, numSemaphoreCopies, producer, consumer)) {
      int64_t nextConsumer = pair->consumer + pair->advances;
      offset = positiveMod(nextConsumer - pair->producer, numSemaphoreCopies);
    }
    release->stageOffset = offset;
    hasShiftedRelease |= offset != 0;
  }
  ArrayRef<Node *> normalized = group.nodesOfKind(
      hasShiftedRelease ? Node::Acquire : Node::Release);
  std::optional<int64_t> normalizedOffset =
      hasShiftedRelease ? std::optional<int64_t>(0) : std::nullopt;
  for (Node *node : normalized)
    if (isDirectLoopNode(node)) node->stageOffset = normalizedOffset;
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
    for (unsigned edgeIndex : cycle) cycleDelay += constraints[edgeIndex].requiredDelay();
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
           offset[ownerKey(constraint.producerOwner)] + constraint.requiredDelay();
  };
  auto hasTightPath = [&](int64_t from, int64_t to) {
    SmallVector<int64_t, 8> stack{from};
    DenseSet<int64_t> seen;
    while (!stack.empty()) {
      int64_t vertex = stack.pop_back_val();
      if (vertex == to) return true;
      if (!seen.insert(vertex).second) continue;
      for (const OwnerScheduleConstraint &constraint : constraints)
        if (isTight(constraint) && ownerKey(constraint.producerOwner) == vertex)
          stack.push_back(ownerKey(constraint.consumerOwner));
    }
    return false;
  };
  auto alreadySSAOrdered = [](Operation *producer, Operation *consumer) {
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) { return op->getBlock() == consumer->getBlock(); };
    (void)getBackwardSlice(consumer, &slice, options);
    return slice.contains(producer);
  };
  for (const OwnerScheduleConstraint &constraint : constraints) {
    bool orders = constraint.requiredDelay() == 0;
    if (!orders && isTight(constraint))
      orders = hasTightPath(ownerKey(constraint.consumerOwner), ownerKey(constraint.producerOwner));
    if (orders && !alreadySSAOrdered(constraint.producer, constraint.consumer))
      model.clusterEdges.emplace_back(constraint.producer, constraint.consumer);
  }
  return success();
}
static LogicalResult addSyncScheduleEdges(MutableArrayRef<GroupDag> groups, PhysicalSets &physical,
                                          llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
  for (GroupDag &group : groups) {
    for (Node *loop : group.nodesOfKind(Node::For))
      if (loop->op->hasAttr(triton::kScheduledMaxStageAttrName))
        modelsByLoop.try_emplace(loop->op);
    for (Node *release : group.nodesOfKind(Node::Release)) {
      if (!isDirectLoopNode(release)) continue;
      Node *acquire = release->sat;
      Operation *producer = findScheduleAnchor(release->scheduleAnchor, /*producer=*/true),
                *consumer = findScheduleAnchor(acquire->scheduleAnchor);
      if (!producer || !consumer) continue;
      for (Operation *parent = producer->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName)) continue;
        Operation *producerAnchor = loop.getBody()->findAncestorOpInBlock(*producer);
        Operation *consumerAnchor = loop.getBody()->findAncestorOpInBlock(*consumer);
        if (!producerAnchor || !consumerAnchor) continue;
        if (producerAnchor == consumerAnchor) break;
        gpu::StageCluster producerSchedule = gpu::getStageCluster(producerAnchor);
        gpu::StageCluster consumerSchedule = gpu::getStageCluster(consumerAnchor);
        if (!producerSchedule || !consumerSchedule) break;
        int64_t distance = acquire->recurrenceDistance.value_or(0);
        if (!acquire->recurrenceDistance && !precedesInChain(release, acquire)) {
          const SlotSchedule &slots = computeSlotSchedule(physical[physicalKey(group)], loop);
          std::optional<int64_t> loopCarriedDistance =
              computeLoopCarriedDistance(slots, group.numSemaphoreCopies, release->scheduleAnchor,
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
        modelsByLoop[loop.getOperation()].ownerConstraints.push_back(
            OwnerScheduleConstraint{release->owner, acquire->owner, producerAnchor, consumerAnchor,
                                    producerSchedule->first, consumerSchedule->first, distance});
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
  for (auto [producer, consumer] : constraints) {
    if (producer->isBeforeInBlock(consumer)) continue;
    cluster[producer] =
        std::min(cluster.lookup(producer), cluster.lookup(consumer) - 1);
  }
  for (unsigned iteration = 0; iteration <= cluster.size(); ++iteration) {
    bool changed = false;
    for (auto [producer, consumer] : constraints) {
      int64_t required = cluster.lookup(producer) + !producer->isBeforeInBlock(consumer);
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
  auto forOp = dyn_cast<scf::ForOp>(n->parent->op);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName)) return schedule;
  auto [stage, cluster] = *schedule;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate) continue;
    if (candidate->first != stage || !gpu::hasPartition(&op)) continue;
    if (gpu::getPartitionIds(&op).contains(n->owner->first)) cluster = std::max(cluster, candidate->second);
  }
  return std::make_pair(stage, cluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      if (!n->owner) break;
      gpu::StageCluster boundary = scheduleAtOwnerBoundary(n, cache.lookup(ownerKey(n->owner)));
      if (n->recurrenceDistance && n->next &&
          n->next->kind == Node::Exit) {
        n->stageCluster = boundary;
      } else {
        Operation *anchor = findScheduleAnchor(n->scheduleAnchor);
        n->stageCluster = anchor ? gpu::getStageCluster(anchor) : boundary;
      }
      break;
    }
    case Node::Release:
      if (n->owner) n->stageCluster = cache.lookup(ownerKey(n->owner));
      break;
    case Node::Access:
      if (n->owner)
        cache[ownerKey(n->owner)] = gpu::getStageCluster(n->completionAnchor ? n->completionAnchor : n->op);
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
      for (auto &[key, stageCluster] : elseCache) cache.try_emplace(key, stageCluster);
      break;
    }
    default: break;
    }
  }
}
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, LoopScheduleModel> modelsByLoop;
  PhysicalSets physical;
  for (GroupDag &group : groups) physical[physicalKey(group)].groups.push_back(&group);
  if (failed(assignCircularStageOffsets(physical))) return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(physical, group))) return failure();
  if (failed(addSyncScheduleEdges(groups, physical, modelsByLoop))) return failure();
  for (auto &[loopOp, model] : modelsByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    if (failed(solveOwnerScheduleConstraints(model)) ||
        failed(legalizeLoopSchedule(loop, model.clusterEdges))) {
      dumpSyncDagTrees(groups);
      return failure();
    }
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
  FailureOr<std::optional<int>> plannedCopy = computeBackingCopies(
      g, edges, useMetaPartitioner, numTmemBlocks);
  if (failed(plannedCopy)) return failure();
  if (!edges.empty() &&
      failed(DirectBuilder(g, edges, reusable).run()))
    return failure();
  for (Node *head : g.root->children) computeRequiredParts(head);
  computeSemaphoreCopies(g, lowerSemaphoreNumStages, *plannedCopy);
  if (!g.semas.empty() && !g.ttDescriptorFedMembers.empty())
    return semaError(g.ttDescriptorFedMembers.front())
           << "managed local_alloc sourced from a tt-form descriptor load — "
           "nvws-insert-allocas must convert this upstream (pipeline invariant violated)";
  return verifySyncDag(g);
}

} // namespace mlir::triton::nvws_semas
