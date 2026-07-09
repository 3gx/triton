// SYNC analysis and scheduling; section links refer to sync-dag.md.
#include "InsertSemas.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include <limits>

namespace mlir::triton::nvws_semas {
using Payloads = SmallVector<AsyncOp, 1>;
using PieceEffects = std::map<PieceId, Effect>;

struct PieceExitFacts {
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
    auto it = llvm::find_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner);
    });
    return it == live.end() ? nullptr : &*it;
  }
  const Token *last() const {
    auto it = llvm::find_if(llvm::reverse(live), [](const Token &token) {
      return token.producer && !token.closedBy;
    });
    return it == live.rend() ? nullptr : &*it;
  }
  Token *find(const Owner &owner) {
    return const_cast<Token *>(std::as_const(*this).find(owner));
  }
  const Token *findOpen(const Owner &owner) const {
    const Token *token = find(owner);
    return token && !token->closedBy ? token : nullptr;
  }
  Token *findOpen(const Owner &owner) {
    return const_cast<Token *>(std::as_const(*this).findOpen(owner));
  }
  const Token *findProducer(Node *producer) const {
    auto it = llvm::find_if(live, [&](const Token &token) {
      return token.producer == producer;
    });
    return it == live.end() ? nullptr : &*it;
  }
  void record(const Owner &owner, Node *producer, Node *last, const Payloads &payloads) {
    llvm::erase_if(live, [&](const Token &token) {
      return sameOwner(token.owner, owner) || (owner.has_value() && !token.owner.has_value());
    });
    live.push_back(Token{owner, producer, last, payloads, nullptr});
  }
  void eraseOwner(const Owner &owner) {
    llvm::erase_if(live, [&](const Token &t) { return sameOwner(t.owner, owner); });
  }
  void eraseProducer(Node *producer) {
    if (!producer) return;
    llvm::erase_if(live, [&](const Token &t) { return t.producer == producer; });
  }
  void close(Node *producer, Node *release) {
    for (Token &token : live)
      if (token.producer == producer && !token.closedBy) token.closedBy = release;
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
    if (!llvm::is_contained(into, payload)) into.push_back(payload);
  llvm::sort(into, [](AsyncOp a, AsyncOp b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
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
  std::optional<APInt> tripCount = forOp.getStaticTripCount();
  if (!tripCount) return false;
  return forOp.getUnsignedCmp() ? tripCount->ugt(0) : tripCount->sgt(0);
}

// Doc: sync-dag.md#the-per-access-rules-in-full
static void applyTouch(PieceState &piece, PieceId id, const Owner &owner,
                       Effect effect, Node *node, const Payloads &payloads, EdgeList &edges, bool wsAdopt) {
  if (!piece.initialized()) {
    piece.startVersion(owner, owner, node, payloads);
    return;
  }
  if (effect == Effect::W) {
    for (const ActiveUse &use : piece.uses) {
      bool adoptedRoot = wsAdopt && !use.owner;
      bool alreadyOrdered = llvm::is_contained(use.orderedBefore, ownerKey(owner));
      if (!sameOwner(use.owner, owner) && !adoptedRoot && !alreadyOrdered)
        edges.push_back( EdgeRec{use.node, node, use.owner, owner, use.payloads, {id}});
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
    edges.push_back(EdgeRec{piece.source.node, node, piece.source.sourceOwner, owner, piece.source.payloads, {id}});
    if (ActiveUse *source = piece.useFor(piece.source.sourceOwner))
      if (source->node == piece.source.node && !llvm::is_contained(source->orderedBefore, ownerKey(owner)))
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
static bool nodeTouchesPiece(GroupDag &g, Node *node, PieceId piece) {
  if (node->kind == Node::Access) return touchesPiece(g, node, piece);
  return node->isRegion() && node->pieceInfo.count(piece);
}
static bool pieceTouchedAfter(GroupDag &g, Node *region, PieceId piece) {
  for (Node *scope = region; scope && scope->kind != Node::Func; scope = scope->parent)
    for (Node *node = scope->next; node; node = node->next)
      if (nodeTouchesPiece(g, node, piece)) return true;
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
        auto it = state.find(id);
        if (it == state.end() || !it->second.initialized()) continue;
        PieceState &piece = it->second;
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
        if (auto it = childFacts.find(id); it != childFacts.end()) branchOrder = it->second.mustOrderedBefore;
        if (firstChild) returned[id].mustOrderedBefore = std::move(branchOrder);
        else
          returned[id].mustOrderedBefore = intersectOrderFacts( returned[id].mustOrderedBefore, branchOrder);
      }
      firstChild = false;
    }
    if (node->kind == Node::For && !knownNonEmptyLoop(node)) {
      for (auto [id, info] : infos) returned[id].mustOrderedBefore = intersectOrderFacts(
            returned[id].mustOrderedBefore, incomingOrder[id]);
    } else if (node->kind == Node::If && node->children.size() < 2) {
      for (auto [id, info] : infos) returned[id].mustOrderedBefore = intersectOrderFacts(
            returned[id].mustOrderedBefore, incomingOrder[id]);
    }
    for (auto [id, info] : infos) {
      auto facts = returned.find(id);
      ActiveUse *use = state[id].useFor(info.owner);
      if (facts == returned.end() || !use) continue;
      if (use->node == node)
        for (int64_t owner : facts->second.mustOrderedBefore)
          if (!llvm::is_contained(use->orderedBefore, owner)) use->orderedBefore.push_back(owner);
    }
    if (infos.empty()) return;
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
        for (const ActiveUse &use : piece.uses)
          if (!sameOwner(use.owner, info.owner) && !llvm::is_contained(use.orderedBefore, ownerKey(info.owner)))
            edges.push_back(EdgeRec{
                use.node, node, use.owner, info.owner, use.payloads, {id}});
      ActiveUse carried{info.owner, node, {AsyncOp::NONE}, {}};
      if (ActiveUse *use = piece.useFor(info.owner)) carried = *use;
      piece.uses.assign(1, carried);
    }
  }
  ExitFacts returnedExitFacts(Node *head) {
    ExitFacts result;
    if (head->kind != Node::Enter) return result;
    for (auto [id, info] : sortedPieceInfo(head)) {
      PieceExitFacts facts;
      auto it = state.find(id);
      if (it != state.end())
        if (ActiveUse *use = it->second.useFor(info.owner)) {
          facts.mustOrderedBefore = use->orderedBefore;
        }
      result.emplace(id, std::move(facts));
    }
    return result;
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
static Node *newProtocolNode(GroupDag &g, Node::Kind kind, Node *parent, Owner owner, SemaId sema, unsigned count) {
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
static EdgeBuckets collectEdges(const Positions &positions, ArrayRef<EdgeRec> edges,
                                const std::vector<bool> &drop, SmallVectorImpl<unsigned> &closes) {
  EdgeBuckets buckets;
  for (auto [i, edge] : llvm::enumerate(edges)) {
    if (drop[i] || !positions.contains(edge.src) || !positions.contains(edge.dst)) continue;
    buckets[edge.dst].push_back(i);
    if (isLoopClose(edge)) closes.push_back(i);
  }
  for (auto &bucket : buckets)
    llvm::stable_sort(bucket.second, [&](unsigned a, unsigned b) {
      return positions.lookup(edges[a].src) > positions.lookup(edges[b].src);
    });
  return buckets;
}
// Doc: sync-dag.md#1-implied-ordering-reduceedges. Drops use only kept edges.
static void reduceStraightEdges(Node *head, const Positions &positions,
                                ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst, std::vector<bool> &drop,
                                DenseSet<Node *> &reusable) {
  KnownOrder order;
  Snapshots snapshots;
  SmallVector<int64_t, 2> tokenOwners;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner) recordTokenOwner(tokenOwners, ownerKey(pi.owner));
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
          reusable.insert(e.dst);
          continue;
        }
        recordTokenOwner(tokenOwners, dk); // Kept acquire supplies Q's token.
        order.apply(e, srcIdx, snapshots);
      }
    if (n->kind != Node::Exit && n->owner) order.record(n, positions.lookup(n), snapshots);
  }
}
static void reduceLoopCloses(GroupDag &g, Node *head, const Positions &positions,
                             ArrayRef<EdgeRec> edges, const EdgeBuckets &atDst,
                             ArrayRef<unsigned> closes, std::vector<bool> &drop) {
  if (closes.empty()) return;
  constexpr unsigned kPass2 = 1u << 20;
  KnownOrder order;
  Snapshots snap1, snap2;
  Owner firstAccessOwner;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        if (!drop[ei] && !isLoopClose(edges[ei])) order.apply(edges[ei], positions.lookup(edges[ei].src), snap1);
    if (n->owner && n->kind == Node::Access) {
      if (!firstAccessOwner) firstAccessOwner = n->owner;
      order.record(n, positions.lookup(n), snap1);
    }
  }
  if (!firstAccessOwner) return;
  EdgeBuckets closeAt;
  for (unsigned ei : closes) {
    const EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    llvm::SmallDenseSet<PieceId> seen;
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Access && n->owner && sameOwner(n->owner, e.dstOwner))
        for (const Touch &touch : n->touches)
          for (PieceId pc : g.pieceTable.footprint[touch.member])
            if (llvm::is_contained(e.pieces, pc) && seen.insert(pc).second) latest = n;
    if (latest) closeAt[latest].push_back(ei);
  }
  DenseSet<int64_t> tokenAvailable;
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        if (drop[ei] || isLoopClose(edges[ei])) continue;
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
    if (n->owner && n->kind == Node::Access) order.record(n, kPass2 + positions.lookup(n), snap2);
  }
}
static void reduceChain(GroupDag &g, Node *head, ArrayRef<EdgeRec> edges, std::vector<bool> &drop,
                        DenseSet<Node *> &reusable) {
  Positions positions;
  unsigned position = 0;
  for (Node *n = head; n; n = n->next) positions[n] = position++;
  SmallVector<unsigned, 4> closes;
  EdgeBuckets atDst = collectEdges(positions, edges, drop, closes);
  reduceStraightEdges(head, positions, edges, atDst, drop, reusable);
  if (head->parent && head->parent->kind == Node::For)
    reduceLoopCloses(g, head, positions, edges, atDst, closes, drop);
  for (Node *n = head; n; n = n->next)
    if (n->isRegion())
      for (Node *child : n->children) reduceChain(g, child, edges, drop, reusable);
}
static void reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges, DenseSet<Node *> &reusable) {
  if (g.root->children.empty() || edges.empty()) return;
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
  DirectBuilder(GroupDag &group, ArrayRef<EdgeRec> edges, const DenseSet<Node *> &reusable)
      : g(group), reusable(reusable) {
    for (const EdgeRec &edge : edges) {
      atDst[edge.dst].push_back(&edge);
      atSrc[edge.src].push_back(&edge);
    }
  }

  LogicalResult run() {
    if (g.root->children.empty()) return success();
    Tokens tokens;
    placeChain(g.root->children[0], tokens);
    if (hadError) return failure();
    for (const auto &[dst, refs] : atDst)
      for (const EdgeRec *edge : refs)
        if (!handledEdges.contains(edge)) {
          if (shouldDumpDag()) dumpDagTree(g, DumpStage::Owner);
          return semaError(dst->op ? dst->op : g.root->op) << "reduced dependency edge was not placed (source kind "
                 << static_cast<unsigned>(edge->src->kind) << ", source op " << (edge->src->op
                         ? edge->src->op->getName().getStringRef() : StringRef("boundary"))
                 << ", target kind " << static_cast<unsigned>(dst->kind) << ")";
        }
    if (!deferredPaths.empty() || !loopPaths.empty()) {
      if (shouldDumpDag()) dumpDagTree(g, DumpStage::Owner);
      return semaError(g.root->op) << "conditional release path was not consumed (deferred "
             << deferredPaths.size() << ", loop " << loopPaths.size() << ")";
    }
    for (const auto &storage : g.nodes)
      if (storage->kind == Node::Release && !storage->sat)
        return semaError(g.root->op) << "release has no consuming acquire";
    return formSemaphores();
  }

private:
  using EdgeRefs = SmallVector<const EdgeRec *, 2>;
  using ReleasePath = SmallVector<Node *, 2>;
  using ReleasePaths = SmallVector<ReleasePath, 2>;
  using BoundaryKey = std::pair<Node *, int64_t>;
  static BoundaryKey boundaryKey(Node *region, const Owner &owner) { return {region, ownerKey(owner)}; }
  Node *regionChannelFor(Node *region, const Owner &owner) const {
    auto it = regionChannels.find(boundaryKey(region, owner));
    return it == regionChannels.end() ? nullptr : it->second;
  }
  Node *concreteAcquire(Node *producer) const {
    if (!producer) return nullptr;
    if (producer->kind == Node::Acquire) return producer;
    if (!producer->isRegion() || !producer->flow) return nullptr;
    for (Node *exit : producer->flow->exits) if (Node *acquire = concreteAcquire(exit)) return acquire;
    return nullptr;
  }
  static void appendAlternatives(ReleasePaths &prefixes, const ReleasePaths &suffixes) {
    if (suffixes.empty()) return;
    if (prefixes.empty()) {
      prefixes = suffixes;
      return;
    }
    ReleasePaths joined;
    for (const ReleasePath &prefix : prefixes)
      for (const ReleasePath &suffix : suffixes) {
        ReleasePath path = prefix;
        path.append(suffix.begin(), suffix.end());
        joined.push_back(std::move(path));
      }
    prefixes = std::move(joined);
  }
  static void appendRelease(ReleasePaths &paths, Node *release) {
    if (!release) return;
    if (paths.empty()) paths.emplace_back();
    for (ReleasePath &path : paths) path.push_back(release);
  }
  static Node *chainExit(Node *head) {
    Node *tail = head;
    while (tail && tail->next) tail = tail->next;
    return tail && tail->kind == Node::Exit ? tail : nullptr;
  }
  Node *nextGroupUse(Node *region) const {
    for (Node *node = region->next; node; node = node->next) {
      if (node->kind == Node::Exit) return nullptr;
      if (node->kind == Node::Access && nodeTouchesGroup(g, node)) return node;
      if (node->isRegion() && !node->pieceInfo.empty()) return node;
    }
    return nullptr;
  }
  Node *nextReusableUse(Node *region, const Owner &owner) const {
    Node *node = nextGroupUse(region);
    if (!node) return nullptr;
    if (node->kind == Node::Access) return sameOwner(node->owner, owner) && reusable.contains(node) &&
                     llvm::none_of(atDst.lookup(node), [&](const EdgeRec *edge) {
                       return !handledEdges.contains(edge);
                     }) ? node : nullptr;
    std::optional<Owner> nestedOwner = uniformPieceOwner(node);
    bool hasIncoming = llvm::any_of(atDst.lookup(node), [&](const EdgeRec *edge) {
      return !handledEdges.contains(edge);
    });
    return nestedOwner && sameOwner(*nestedOwner, owner) && !hasIncoming ? node : nullptr;
  }
  Node *firstDemandAnchor(Node *node) const {
    while (node && node->isRegion() && node->scheduleAnchor) node = node->scheduleAnchor;
    return node;
  }
  bool crossesLoopStage(Node *region, Node *demand) const {
    Node *first = firstDemandAnchor(demand);
    Node *last = loopLastOwnerAccess.lookup(region);
    gpu::StageCluster firstSchedule = first && first->op ? gpu::getStageCluster(first->op) : std::nullopt;
    gpu::StageCluster lastSchedule = last && last->op ? gpu::getStageCluster(last->op) : std::nullopt;
    return g.numCopies == 1 && demand && demand->tokenSource == region && firstSchedule && lastSchedule &&
           firstSchedule->first != lastSchedule->first;
  }
  SmallVector<EdgeRec, 2> collapse(ArrayRef<const EdgeRec *> refs) const {
    SmallVector<EdgeRec, 2> result;
    for (const EdgeRec *edge : refs) {
      auto prior = llvm::find_if(result, [&](const EdgeRec &item) {
        return item.src == edge->src && sameOwner(item.srcOwner, edge->srcOwner);
      });
      if (prior == result.end()) {
        result.push_back(*edge);
        llvm::sort(result.back().payloads);
        continue;
      }
      unionPayloads(prior->payloads, edge->payloads);
      for (PieceId piece : edge->pieces)
        if (!llvm::is_contained(prior->pieces, piece)) prior->pieces.push_back(piece);
    }
    return result;
  }
  Node *findChannel(Node *acquire) {
    Node *parent = channelParent.lookup(acquire);
    if (!parent || parent == acquire) return acquire;
    return channelParent[acquire] = findChannel(parent);
  }
  void uniteChannels(Node *lhs, Node *rhs) {
    lhs = findChannel(lhs);
    rhs = findChannel(rhs);
    if (lhs != rhs) channelParent[rhs] = lhs;
  }
  Node *makeAcquire(Node *before, const Owner &owner) {
    Node *acquire = newProtocolNode(g, Node::Acquire, before->parent, owner, /*sema=*/0, /*count=*/0);
    acquire->scheduleAnchor = before;
    spliceBefore(acquire, before);
    acquires.push_back(acquire);
    channelParent[acquire] = acquire;
    return acquire;
  }
  Node *makeAcquireAfter(Node *after, const Owner &owner) {
    Node *acquire = newProtocolNode(g, Node::Acquire, after->parent, owner, /*sema=*/0, /*count=*/0);
    acquire->scheduleAnchor = after;
    Node *anchor = lastAfter.lookup(after);
    spliceAfter(acquire, anchor ? anchor : after);
    lastAfter[after] = acquire;
    acquires.push_back(acquire);
    channelParent[acquire] = acquire;
    return acquire;
  }
  Tokens::Token sourceToken(const EdgeRec &edge, const Tokens &tokens) const {
    if (edge.src->kind == Node::Access) return edge.src->tokenSource
                 ? Tokens::Token{edge.srcOwner, edge.src->tokenSource, edge.src, edge.payloads} : Tokens::Token{};
    if (edge.src->isRegion()) {
      Node *producer = edge.src->flow ? edge.src : edge.src->tokenSource;
      if (producer)
        if (const Tokens::Token *token = tokens.findProducer(producer)) return *token;
      if (producer) return Tokens::Token{edge.srcOwner, producer, edge.src, edge.payloads};
      return {};
    }
    if (edge.src->kind == Node::Acquire)
      if (const Tokens::Token *token = tokens.findProducer(edge.src)) return *token;
    if (edge.src->kind == Node::Enter) {
      auto it = boundaryTokens.find(boundaryKey(edge.src, edge.srcOwner));
      if (it != boundaryTokens.end()) return it->second;
    }
    return {};
  }

  Node *materializeRelease(Tokens::Token &token, const Owner &owner, Node *acquire, Node *guard,
                           Tokens *live = nullptr) {
    Node *release = newProtocolNode( g, Node::Release, guard ? guard->parent : token.last->parent, owner,
        /*sema=*/0, /*count=*/1);
    release->payloads = token.payloads;
    if (release->payloads.empty()) release->payloads.push_back(AsyncOp::NONE);
    release->tokenSource = token.producer;
    release->sat = acquire;
    release->scheduleAnchor = token.last;
    if (live) live->close(token.producer, release);
    else
      token.closedBy = release;
    if (guard) spliceAfter(release, guard);
    else {
      Node *anchor = lastAfter.lookup(token.last);
      spliceAfter(release, anchor ? anchor : token.last);
      lastAfter[token.last] = release;
    }
    return release;
  }

  Node *insertRelease(const EdgeRec &edge, Node *acquire, Tokens &tokens, Node *guard = nullptr) {
    Tokens::Token source = sourceToken(edge, tokens);
    if (!source.producer || !source.last) {
      semaError(edge.src->op ? edge.src->op : g.root->op) << "release has no exact token for source kind "
          << static_cast<unsigned>(edge.src->kind) << " and target kind "
          << static_cast<unsigned>(edge.dst->kind) << " (flow " << edge.src->flow.has_value() << ", source token "
          << (edge.src->tokenSource ? static_cast<int>(edge.src->tokenSource->kind) : -1) << ")";
      hadError = true;
      return nullptr;
    }
    if (edge.src->kind == Node::Enter) guard = edge.src;
    return materializeRelease(source, edge.srcOwner, acquire, guard, &tokens);
  }

  bool attachPaths(Node *acquire, const ReleasePaths &paths) {
    unsigned expected = 0;
    for (const ReleasePath &path : paths) {
      unsigned count = 0;
      for (Node *release : path)
        count += std::max(1u, release->count) * std::max(1u, static_cast<unsigned>(release->payloads.size()));
      if (!count || (expected && count != expected)) {
        semaError(acquire->op ? acquire->op : g.root->op)
            << "conditional paths provide incompatible semaphore counts";
        hadError = true;
        return false;
      }
      expected = count;
      for (Node *release : path) {
        if (release->sat && release->sat != acquire) {
          semaError(g.root->op) << "one release cannot satisfy two acquire sites";
          hadError = true;
          return false;
        }
        release->sat = acquire;
      }
    }
    acquire->count = std::max(acquire->count, expected);
    return true;
  }

  SmallVector<Node *, 2> supplyAcquire(Node *acquire, ArrayRef<const EdgeRec *> refs,
                Tokens &tokens, Node *guard = nullptr, ReleasePaths *collectedPaths = nullptr) {
    SmallVector<EdgeRec, 2> incoming;
    for (EdgeRec edge : collapse(refs)) {
      Tokens::Token source = sourceToken(edge, tokens);
      auto prior = source.producer ? llvm::find_if(incoming, [&](const EdgeRec &item) {
                           Tokens::Token other = sourceToken(item, tokens);
                           return other.producer == source.producer && (precedesInChain(item.src, edge.src) ||
                                   precedesInChain(edge.src, item.src));
                         }) : incoming.end();
      if (prior == incoming.end()) {
        incoming.push_back(std::move(edge));
        continue;
      }
      unionPayloads(prior->payloads, edge.payloads);
      for (PieceId piece : edge.pieces)
        if (!llvm::is_contained(prior->pieces, piece)) prior->pieces.push_back(piece);
      if (precedesInChain(prior->src, edge.src)) prior->src = edge.src;
    }
    SmallVector<Node *, 2> releases;
    ReleasePaths localPaths;
    ReleasePaths &paths = collectedPaths ? *collectedPaths : localPaths;
    for (const EdgeRec &edge : incoming) {
      if (edge.src->isRegion() && !edge.src->flow) {
        if (Node *channel = regionChannelFor(edge.src, edge.srcOwner)) {
          if (sameOwner(edge.srcOwner, edge.dstOwner)) {
            uniteChannels(acquire, channel);
            acquire->count = std::max(acquire->count, channel->count);
          } else {
            Node *drain = makeAcquireAfter(edge.src, edge.srcOwner);
            drain->count = channel->count;
            uniteChannels(drain, channel);
            tokens.record(edge.srcOwner, drain, drain, {AsyncOp::NONE});
            EdgeRec handoff{drain, acquire, edge.srcOwner, edge.dstOwner, {AsyncOp::NONE}, edge.pieces};
            if (Node *release = insertRelease(handoff, acquire, tokens, guard)) releases.push_back(release);
            acquire->count = std::max(acquire->count, 1u);
          }
          continue;
        }
        if (auto it = deferredPaths.find(boundaryKey(edge.src, edge.srcOwner));
            it != deferredPaths.end()) {
          ReleasePaths alternatives = std::move(it->second);
          deferredPaths.erase(it);
          appendAlternatives(paths, alternatives);
          continue;
        }
      }
      if (Node *release = insertRelease(edge, acquire, tokens, guard)) releases.push_back(release);
    }
    for (Node *release : releases) appendRelease(paths, release);
    if (!paths.empty()) attachPaths(acquire, paths);
    for (const EdgeRec *edge : refs) handledEdges.insert(edge);
    return releases;
  }

  void supplyAcquireWithPaths(Node *acquire, ArrayRef<const EdgeRec *> refs,
                              const ReleasePaths &paths, Tokens &tokens) {
    ReleasePaths combined = paths;
    supplyAcquire(acquire, refs, tokens, nullptr, &combined);
  }

  void requirePathCount(Node *acquire, ArrayRef<Node *> releases) {
    unsigned count = 0;
    for (Node *release : releases)
      count += std::max(1u, release->count) * std::max(1u, static_cast<unsigned>(release->payloads.size()));
    if (!count || count == acquire->count) return;
    bool scalable = releases.size() == 1 && llvm::all_of(releases.front()->payloads, [](AsyncOp payload) {
          return payload == AsyncOp::NONE || payload == AsyncOp::WGMMA;
        });
    if (scalable && acquire->count % releases.front()->payloads.size() == 0) {
      releases.front()->count = acquire->count / releases.front()->payloads.size();
      return;
    }
    semaError(acquire->op ? acquire->op : g.root->op)
        << "one execution path does not supply the acquire pending count";
    hadError = true;
  }

  Node *insertTokenRelease(Tokens::Token &token, Node *acquire, Node *guard = nullptr) {
    if (!token.producer || !token.last) {
      semaError(g.root->op) << "token handoff has no exact completion";
      hadError = true;
      return nullptr;
    }
    return materializeRelease(token, token.owner, acquire, guard);
  }

  Tokens childInput(const Tokens &incoming, Node *enter) {
    Tokens result = incoming;
    std::optional<Owner> owner = uniformPieceOwner(enter);
    if (owner && owner->has_value() && result.last() && !result.last()->owner) {
      Tokens::Token adopted = *result.last();
      adopted.owner = *owner;
      result.record(adopted.owner, adopted.producer, adopted.last, adopted.payloads);
    }
    DenseSet<int64_t> recorded;
    for (auto [piece, info] : sortedPieceInfo(enter))
      if (recorded.insert(ownerKey(info.owner)).second)
        if (const Tokens::Token *token = result.findOpen(info.owner))
          boundaryTokens[boundaryKey(enter, info.owner)] = *token;
    if (owner)
      if (const Tokens::Token *token = result.findOpen(*owner)) enter->tokenSource = token->producer;
    return result;
  }

  void placeAccess(Node *node, Tokens &tokens, EdgeRefs *pending = nullptr, Node *pendingGuard = nullptr) {
    Payloads payloads{asyncPayloadOf(node->op)};
    EdgeRefs incoming;
    for (const EdgeRec *edge : atDst.lookup(node))
      if (!handledEdges.contains(edge)) incoming.push_back(edge);
    EdgeRefs inherited;
    if (pending)
      llvm::erase_if(*pending, [&](const EdgeRec *edge) {
        if (!sameOwner(edge->dstOwner, node->owner)) return false;
        inherited.push_back(edge);
        return true;
      });
    if (Node *acquire = node->tokenSource) {
      assert(acquire->kind == Node::Acquire && "preplaced non-acquire token");
      if (!incoming.empty()) supplyAcquire(acquire, incoming, tokens);
      if (!inherited.empty()) supplyAcquire(acquire, inherited, tokens, pendingGuard);
      node->tokenSource = acquire;
      tokens.record(node->owner, acquire, node, payloads);
      recordLoopDemand(node);
      return;
    }
    Tokens::Token *owned = tokens.find(node->owner);
    if (incoming.empty() && inherited.empty() && owned && owned->producer && reusable.contains(node)) {
      node->tokenSource = owned->producer;
      owned->last = node;
      owned->payloads = payloads;
      // A read fan-out release does not invalidate the releasing owner's
      // compatible token.  The per-piece reuse proof is what permits
      // this access to remain in the same ownership wave.
      owned->closedBy = nullptr;
      recordLoopDemand(node);
      return;
    }
    Node *acquire = makeAcquire(node, node->owner);
    if (!incoming.empty()) {
      supplyAcquire(acquire, incoming, tokens);
    }
    if (!inherited.empty()) supplyAcquire(acquire, inherited, tokens, pendingGuard);
    if (incoming.empty() && inherited.empty()) {
      seeded.insert(acquire);
      acquire->count = 1;
    }
    node->tokenSource = acquire;
    tokens.record(node->owner, acquire, node, payloads);
    recordLoopDemand(node);
  }

  void recordLoopDemand(Node *node) {
    if (activeLoops.empty()) return;
    Node *loop = activeLoops.back();
    if (node->parent != loop) return;
    std::optional<Owner> owner = node->kind == Node::Access ? std::optional<Owner>(node->owner)
                                   : uniformPieceOwner(node);
    if (!owner || !sameOwner(loopOwner.lookup(loop), *owner)) return;
    Node *completion = node->isRegion() ? loopLastOwnerAccess.lookup(node) : node;
    if (completion) loopLastOwnerAccess[loop] = completion;
    if (loopDemand.count(loop)) return;
    if (node->isRegion() && !node->flow && !node->tokenSource && !regionChannelFor(node, *owner) &&
        !deferredPaths.count(boundaryKey(node, *owner)))
      return;
    loopDemand[loop] = node;
  }

  void placeIf(Node *region, Tokens &tokens, EdgeRefs *outerPending) {
    Tokens incoming = tokens;
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    Tokens adoptedIncoming;
    const Tokens::Token *input = nullptr;
    if (boundaryOwner && !region->children.empty()) {
      adoptedIncoming = childInput(incoming, region->children.front());
      input = adoptedIncoming.findOpen(*boundaryOwner);
      if (input) region->tokenSource = input->producer;
    }
    SmallVector<Tokens, 2> branches;
    SmallVector<std::optional<Tokens::Token>, 2> outputs;
    SmallVector<EdgeRefs, 2> exitEdges;
    for (Node *child : region->children) {
      Tokens branch = childInput(incoming, child);
      if (input) {
        branch.record(*boundaryOwner, region, child, input->payloads);
        child->tokenSource = region;
        boundaryTokens[boundaryKey(child, *boundaryOwner)] = *branch.findOpen(*boundaryOwner);
      }
      EdgeRefs pending = atDst.lookup(region);
      if (outerPending) pending.append(outerPending->begin(), outerPending->end());
      placeChain(child, branch, &pending, child);

      EdgeRefs exits;
      if (Node *exit = chainExit(child))
        for (const EdgeRec *edge : atDst.lookup(exit))
          if (!handledEdges.contains(edge)) exits.push_back(edge);
      exitEdges.push_back(std::move(exits));
      if (boundaryOwner) {
        const Tokens::Token *output = branch.findOpen(*boundaryOwner);
        outputs.emplace_back(output ? std::optional<Tokens::Token>(*output) : std::nullopt);
      }
      branches.push_back(std::move(branch));
    }
    if (outerPending) outerPending->clear();
    if (!boundaryOwner || branches.empty()) {
      tokens = incoming;
      return;
    }

    Owner owner = *boundaryOwner;
    auto exitPaths = [&](unsigned index) {
      ReleasePaths result;
      EdgeRefs pending;
      for (const EdgeRec *edge : exitEdges[index]) if (!handledEdges.contains(edge)) pending.push_back(edge);
      for (const EdgeRec &edge : collapse(pending)) {
        if (edge.src->isRegion() && !edge.src->flow) {
          auto nested = deferredPaths.find(boundaryKey(edge.src, edge.srcOwner));
          if (nested != deferredPaths.end()) {
            appendAlternatives(result, nested->second);
            deferredPaths.erase(nested);
            continue;
          }
        }
        appendRelease(result, insertRelease(edge, nullptr, branches[index]));
      }
      for (const EdgeRec *edge : pending) handledEdges.insert(edge);
      return result;
    };

    Node *nextUse = nextGroupUse(region);
    Node *laterUse = nextReusableUse(region, owner);
    Node *directLoop = !activeLoops.empty() && region->parent == activeLoops.back() ? activeLoops.back() : nullptr;
    Node *nextEntryDemand = directLoop ? loopDemand.lookup(directLoop) : nullptr;
    bool branchCloses = llvm::any_of( exitEdges, [](const EdgeRefs &refs) { return !refs.empty(); });
    bool realEntry = directLoop && loopRealInputs.count(boundaryKey(directLoop, owner));
    bool nextEntryUse = directLoop && !nextUse && input && (nextEntryDemand || branchCloses) &&
                        (g.numCopies == 1 || realEntry) &&
                        sameOwner(loopOwner.lookup(directLoop), owner);
    bool outgoingUse = false, outgoingExitUse = false;
    for (const EdgeRec *edge : atSrc.lookup(region)) {
      if (handledEdges.contains(edge)) continue;
      outgoingUse = true;
      outgoingExitUse |= edge->dst->kind == Node::Exit;
    }
    if (nextEntryUse)
      for (auto [index, output] : llvm::enumerate(outputs)) {
        if (output) continue;
        ReleasePaths branchPaths = exitPaths(index);
        Node *exit = chainExit(region->children[index]);
        if (!exit || branchPaths.empty()) {
          semaError(region->op) << "next loop entry path has no exact release";
          hadError = true;
          continue;
        }
        Node *acquire = makeAcquire(exit, owner);
        attachPaths(acquire, branchPaths);
        branches[index].record(owner, acquire, acquire, {AsyncOp::NONE});
        outputs[index] = *branches[index].findOpen(owner);
      }
    bool hasNone = llvm::any_of( outputs, [](const auto &output) { return !output.has_value(); });
    auto isPass = [&](const std::optional<Tokens::Token> &output) {
      return input && output && (output->producer == region || output->producer == input->producer);
    };
    bool changed = llvm::any_of( outputs, [&](const auto &output) { return !isPass(output); });
    bool returnToken = !hasNone && (laterUse || outgoingExitUse || nextEntryUse);
    bool laterPointOfUse = changed && nextUse && (!laterUse || hasNone) && nextUse->kind == Node::Access;
    bool loopPointOfUse = directLoop && !nextUse && !returnToken && (nextEntryDemand || branchCloses) && changed;
    bool parentNeedsPaths = region->parent && region->parent->kind == Node::If && changed;
    bool deferToAcquire = !returnToken && (laterPointOfUse || loopPointOfUse || outgoingUse || parentNeedsPaths);

    if (deferToAcquire) {
      ReleasePaths paths;
      for (auto [index, output] : llvm::enumerate(outputs)) {
        ReleasePaths branchPaths = exitPaths(index);
        Node *exit = chainExit(region->children[index]);
        Node *guard = exit && exit->prev ? exit->prev : region->children[index];
        if (output)
          if (Tokens::Token *open = branches[index].findOpen(owner))
            appendRelease(branchPaths, insertTokenRelease(*open, nullptr, guard));
        if (branchPaths.empty()) {
          semaError(region->op) << "region path has no exact release for its later acquire";
          hadError = true;
          continue;
        }
        for (ReleasePath &path : branchPaths) paths.push_back(std::move(path));
      }
      if (laterPointOfUse) {
        Node *acquire = makeAcquire(nextUse, nextUse->owner);
        EdgeRefs common;
        for (const EdgeRec *edge : atDst.lookup(nextUse)) {
          if (handledEdges.contains(edge)) continue;
          Tokens::Token source = sourceToken(*edge, incoming);
          if (input && source.producer == input->producer && sameOwner(edge->srcOwner, owner))
            handledEdges.insert(edge);
          else
            common.push_back(edge);
        }
        if (!common.empty())
          for (Node *release : supplyAcquire(acquire, common, incoming)) appendRelease(paths, release);
        attachPaths(acquire, paths);
        nextUse->tokenSource = acquire;
      } else if (loopPointOfUse) {
        Owner target = loopOwner.count(directLoop) ? loopOwner.lookup(directLoop) : owner;
        ReleasePaths &loop = loopPaths[boundaryKey(directLoop, target)];
        loop.append(paths.begin(), paths.end());
        for (const EdgeRec *edge : atSrc.lookup(region))
          if (edge->dst->kind == Node::Exit && edge->dst->parent == directLoop) handledEdges.insert(edge);
        loopDemand.try_emplace(directLoop, region);
      } else
        deferredPaths[boundaryKey(region, owner)] = std::move(paths);
      tokens = incoming;
      tokens.eraseOwner(owner);
      return;
    }

    if (!changed) {
      tokens = adoptedIncoming;
      if (input) {
        tokens.eraseProducer(input->producer);
        tokens.record(owner, input->producer, region, input->payloads);
      }
      return;
    }
    if (hasNone || (!laterUse && !outgoingUse && !nextEntryUse)) {
      if (hasNone) region->tokenSource = nullptr;
      tokens = incoming;
      tokens.eraseOwner(owner);
      return;
    }
    RegionFlow flow;
    flow.owner = owner;
    for (const std::optional<Tokens::Token> &output : outputs)
      flow.exits.push_back(isPass(output) ? nullptr : output->producer);
    for (const EdgeRefs &refs : exitEdges)
      for (const EdgeRec *edge : refs) handledEdges.insert(edge);
    region->flow.emplace(std::move(flow));
    tokens = incoming;
    if (input) tokens.eraseProducer(input->producer);
    tokens.record(owner, region, region, {AsyncOp::NONE});
  }

  void replaceTokenSource(Node *head, Node *from, Node *to) {
    for (Node *node = head; node; node = node->next)
      if (node->tokenSource == from) node->tokenSource = to;
  }

  Node *bindLoopInput(Node *region, Node *body, const Owner &owner, ArrayRef<const EdgeRec *> entry, Tokens &incoming,
                      Node *guard = nullptr) {
    Tokens adopted = childInput(incoming, body);
    if (entry.empty())
      if (const Tokens::Token *held = adopted.findOpen(owner)) {
        Node *producer = held->producer;
        incoming = std::move(adopted);
        return producer;
      }
    Node *acquire = makeAcquire(region, owner);
    if (entry.empty()) {
      seeded.insert(acquire);
      acquire->count = 1;
    } else {
      supplyAcquire(acquire, entry, incoming, guard);
    }
    incoming.record(owner, acquire, acquire, {AsyncOp::NONE});
    return acquire;
  }

  void placeFor(Node *region, Tokens &tokens, EdgeRefs *outerPending, Node *outerGuard) {
    Node *body = region->children.front();
    Node *exit = chainExit(body);
    std::optional<Owner> boundaryOwner = uniformPieceOwner(region);
    EdgeRefs entry = atDst.lookup(region);
    if (outerPending) entry.append(outerPending->begin(), outerPending->end());
    Tokens incoming = tokens;
    Tokens bodyTokens = childInput(incoming, body);
    std::map<int64_t, Tokens::Token> loopInputs;
    for (auto [piece, info] : sortedPieceInfo(body))
      if (!loopInputs.count(ownerKey(info.owner)))
        if (const Tokens::Token *token = bodyTokens.findOpen(info.owner)) {
          loopInputs.emplace(ownerKey(info.owner), *token);
          if (token->last && token->last->kind != Node::Enter) loopRealInputs[boundaryKey(region, info.owner)] = true;
        }
    if (boundaryOwner) {
      loopOwner[region] = *boundaryOwner;
      bodyTokens.record(*boundaryOwner, region, body, {AsyncOp::NONE});
      body->tokenSource = region;
      boundaryTokens[boundaryKey(body, *boundaryOwner)] = *bodyTokens.findOpen(*boundaryOwner);
    }
    activeLoops.push_back(region);
    placeChain(body, bodyTokens);
    activeLoops.pop_back();
    if (outerPending) outerPending->clear();
    EdgeRefs closes;
    if (exit)
      for (const EdgeRec *edge : atDst.lookup(exit))
        if (!handledEdges.contains(edge)) closes.push_back(edge);
    Node *demand = loopDemand.lookup(region);
    auto takePaths = [&](const Owner &owner) {
      ReleasePaths result;
      auto it = loopPaths.find(boundaryKey(region, owner));
      if (it != loopPaths.end()) {
        result = std::move(it->second);
        loopPaths.erase(it);
      }
      return result;
    };
    auto pendingEntry = [&](const Owner &owner) {
      EdgeRefs result;
      for (const EdgeRec *edge : entry)
        if (!handledEdges.contains(edge) && sameOwner(edge->dstOwner, owner)) result.push_back(edge);
      return result;
    };
    auto supplyEntry = [&](Node *acquire, const Owner &owner, bool seed) {
      EdgeRefs refs = pendingEntry(owner);
      if (refs.empty()) {
        auto held = loopInputs.find(ownerKey(owner));
        if (held != loopInputs.end() && held->second.last && held->second.last->kind != Node::Enter) {
          Tokens::Token token = held->second;
          Owner releaseOwner = token.producer && token.producer->kind == Node::Acquire
                                   ? token.producer->owner : token.owner;
          Node *guard = token.last->parent == region->parent ? nullptr : outerGuard;
          if (token.producer) {
            seeded.erase(acquire);
            if (Node *release = materializeRelease(token, releaseOwner, acquire, guard))
              requirePathCount(acquire, {release});
            return;
          }
        }
        if (seed && !seeded.contains(acquire)) seeded.insert(acquire);
        return;
      }
      seeded.erase(acquire);
      SmallVector<Node *, 2> releases = supplyAcquire(acquire, refs, incoming, outerGuard);
      requirePathCount(acquire, releases);
    };
    if (!boundaryOwner) {
      SmallVector<std::pair<Owner, EdgeRefs>, 2> byOwner;
      auto find = [&](const Owner &owner) {
        return llvm::find_if(byOwner, [&](const auto &group) {
          return sameOwner(group.first, owner);
        });
      };
      auto add = [&](const Owner &owner, const EdgeRec *close = nullptr) {
        auto it = find(owner);
        if (it == byOwner.end()) {
          byOwner.push_back({owner, {}});
          it = std::prev(byOwner.end());
        }
        if (close) it->second.push_back(close);
      };
      for (const EdgeRec *edge : closes) add(edge->dstOwner, edge);
      for (const EdgeRec *edge : entry)
        if (!handledEdges.contains(edge)) add(edge->dstOwner);
      for (const auto &[key, unused] : loopPaths)
        if (key.first == region)
          add(key.second == -1 ? Owner{} : Owner{PartitionId{static_cast<int>(key.second),
                                      static_cast<int>(key.second >> 32)}});
      for (auto &[owner, ownerCloses] : byOwner) {
        Node *first = nullptr, *acquire = nullptr;
        for (Node *node = body; node && node->kind != Node::Exit;
             node = node->next) {
          if (node->isRegion())
            if (Node *channel = regionChannelFor(node, owner)) {
              first = node;
              acquire = channel;
              break;
            }
          std::optional<Owner> nodeOwner = node->kind == Node::Access ? std::optional<Owner>(node->owner)
                  : (node->isRegion() ? uniformPieceOwner(node) : std::nullopt);
          if (nodeOwner && sameOwner(*nodeOwner, owner) && (node->kind == Node::Access || !node->pieceInfo.empty())) {
            first = node;
            break;
          }
        }
        if (!first) {
          semaError(region->op) << "mixed-owner loop close has no point-of-use demand";
          hadError = true;
          continue;
        }
        if (!acquire) acquire = first->tokenSource;
        if (!acquire || acquire->kind != Node::Acquire) {
          Node *old = acquire;
          acquire = makeAcquire(first, owner);
          first->tokenSource = acquire;
          if (old) replaceTokenSource(first, old, acquire);
        }
        ReleasePaths ownerPaths = takePaths(owner);
        supplyAcquireWithPaths(acquire, ownerCloses, ownerPaths, bodyTokens);
        supplyEntry(acquire, owner, true);
        acquire->count = std::max(acquire->count, 1u);
        regionChannels[boundaryKey(region, owner)] = acquire;
        incoming.eraseOwner(owner);
      }
      tokens = std::move(incoming);
      return;
    }
    Owner owner = *boundaryOwner;
    ReleasePaths paths = takePaths(owner);
    const Tokens::Token *output = bodyTokens.findOpen(owner);
    bool returnsToken = output && closes.empty() && paths.empty();
    if (returnsToken) {
      Node *input = bindLoopInput(region, body, owner, entry, incoming, outerGuard);
      if (!input) return;
      if (seeded.contains(input) && !crossesLoopStage(region, demand))
        if (Node *channel = concreteAcquire(output->producer)) uniteChannels(input, channel);
      if (output->producer == region) {
        // The body captures the exact incoming token without adding a loop
        // argument.  Keep the child boundary as the body's symbolic producer;
        // EmitIR aliases it to this declared capture mechanically.
        region->tokenSource = input;
        tokens = incoming;
        tokens.eraseProducer(input);
        tokens.record(owner, input, region, output->payloads);
        return;
      }
      RegionFlow flow;
      flow.owner = owner;
      flow.exits.push_back(output->producer);
      region->tokenSource = input;
      region->flow.emplace(std::move(flow));
      tokens = incoming;
      tokens.eraseProducer(input);
      tokens.record(owner, region, region, output->payloads);
      return;
    }
    bool hasSupply = !closes.empty() || !paths.empty();
    Node *continuation = nextReusableUse(region, owner);
    auto closeAtExit = [&](Node *input, bool distanceOne) {
      if (!input || !exit) return static_cast<Node *>(nullptr);
      Node *tail = makeAcquire(exit, owner);
      tail->scheduleAnchor = firstDemandAnchor(demand);
      if (distanceOne) tail->recurrenceDistance = 1;
      supplyAcquireWithPaths(tail, closes, paths, bodyTokens);
      if (input->kind == Node::Acquire) uniteChannels(tail, input);
      return tail;
    };
    bool crossStage = hasSupply && crossesLoopStage(region, demand);
    auto initial = loopInputs.find(ownerKey(owner));
    bool hasRealInput = initial != loopInputs.end() && initial->second.last &&
                        initial->second.last->kind != Node::Enter;
    bool forwardsInput = demand && demand->isRegion() && !demand->flow && demand->tokenSource == region;
    bool carryInput = hasSupply && (hasRealInput || forwardsInput) && !continuation;
    if (crossStage || carryInput) {
      Node *input = bindLoopInput(region, body, owner, entry, incoming, outerGuard);
      Node *tail = closeAtExit(input, true);
      if (!tail) return;
      RegionFlow flow;
      flow.owner = owner;
      flow.exits.push_back(tail);
      region->tokenSource = input;
      region->scheduleAnchor = firstDemandAnchor(demand);
      region->flow.emplace(std::move(flow));
      tokens = incoming;
      tokens.eraseProducer(input);
      tokens.record(owner, region, region, {AsyncOp::NONE});
      return;
    }
    if (!demand && entry.empty() && closes.empty() && paths.empty()) {
      tokens = incoming;
      tokens.eraseOwner(owner);
      return;
    }
    if (!demand) {
      semaError(region->op) << "loop has no recorded point-of-use demand";
      hadError = true;
      return;
    }
    if (demand->isRegion() && !demand->flow) {
      if (Node *childChannel = regionChannelFor(demand, owner)) {
        if (!hasSupply) {
          supplyEntry(childChannel, owner, false);
          region->scheduleAnchor = firstDemandAnchor(demand);
          regionChannels[boundaryKey(region, owner)] = childChannel;
          tokens = incoming;
          tokens.eraseOwner(owner);
          return;
        }
        Node *input = bindLoopInput(region, body, owner, entry, incoming, outerGuard);
        Node *tail = closeAtExit(input, g.numCopies == 1);
        if (!tail) return;
        Tokens bridge;
        bridge.record(owner, tail, tail, {AsyncOp::NONE});
        Node *release = insertTokenRelease(*bridge.findOpen(owner), childChannel);
        if (release) release->count = std::max(1u, childChannel->count);
        regionChannels[boundaryKey(region, owner)] = childChannel;
        tokens = incoming;
        tokens.eraseOwner(owner);
        return;
      }
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
      if (demand->isRegion()) demand->tokenSource = recurrence;
      replaceTokenSource(body, region, recurrence);
    }
    if (!recurrence || recurrence->kind != Node::Acquire) {
      semaError(region->op) << "loop recurrence has no acquire at its recorded demand";
      hadError = true;
      return;
    }
    supplyAcquireWithPaths(recurrence, closes, paths, bodyTokens);
    if (continuation && continuation->kind == Node::Access) {
      Node *resume = continuation->tokenSource;
      if (!resume) {
        resume = makeAcquire(continuation, owner);
        continuation->tokenSource = resume;
      }
      if (resume->kind != Node::Acquire) {
        semaError(continuation->op) << "post-loop point of use has no acquire";
        hadError = true;
        return;
      }
      resume->count = recurrence->count;
      uniteChannels(resume, recurrence);
    }
    supplyEntry(recurrence, owner, true);
    region->scheduleAnchor = firstDemandAnchor(demand);
    regionChannels[boundaryKey(region, owner)] = recurrence;
    for (const EdgeRec *edge : entry) incoming.eraseOwner(edge->srcOwner);
    tokens = incoming;
    tokens.eraseOwner(owner);
  }

  void placeChain(Node *head, Tokens &tokens, EdgeRefs *pending = nullptr, Node *pendingGuard = nullptr) {
    for (Node *node = head; node;) {
      Node *next = node->next;
      switch (node->kind) {
      case Node::Access:
        if (nodeTouchesGroup(g, node)) placeAccess(node, tokens, pending, pendingGuard);
        break;
      case Node::If:
        placeIf(node, tokens, pending);
        recordLoopDemand(node);
        break;
      case Node::For:
        placeFor(node, tokens, pending, pendingGuard);
        recordLoopDemand(node);
        break;
      case Node::Enter:
      case Node::Exit:
      case Node::Release:
      case Node::Func:
        break;
      case Node::Acquire:
        tokens.record(node->owner, node, node, {AsyncOp::NONE});
        break;
      }
      node = next;
    }
  }

  std::optional<SemaId> channelOfProducer(Node *producer) const {
    if (!producer) return std::nullopt;
    if (producer->kind == Node::Acquire) return producer->sema;
    if (producer->isRegion() && producer->flow) return producer->flow->sema;
    return std::nullopt;
  }

  LogicalResult formSemaphores() {
    llvm::MapVector<Node *, SmallVector<Node *, 2>> classes;
    for (Node *acquire : acquires) classes[findChannel(acquire)].push_back(acquire);
    for (auto &[root, sites] : classes) {
      unsigned count = 1;
      for (Node *site : sites) count = std::max(count, site->count);
      for (Node *site : sites) {
        SmallVector<Node *, 2> releases;
        for (const auto &storage : g.nodes)
          if (storage->kind == Node::Release && storage->sat == site) releases.push_back(storage.get());
        if (site->count == count) continue;
        if (releases.empty() && seeded.contains(site)) {
          site->count = count;
          continue;
        }
        bool scalable = releases.size() == 1 && llvm::all_of(releases.front()->payloads, [](AsyncOp payload) {
              return payload == AsyncOp::NONE || payload == AsyncOp::WGMMA;
            });
        unsigned payloads = scalable ? releases.front()->payloads.size() : 0;
        if (!scalable || count % payloads) return semaError(site->op ? site->op : g.root->op)
                 << "incompatible path counts for one semaphore channel";
        releases.front()->count = count / payloads;
        site->count = count;
      }
      SemaId sid = g.semas.size();
      Sema sema;
      sema.name = "S" + std::to_string(sid);
      sema.count = count;
      for (Node *site : sites)
        if (seeded.contains(site)) {
          sema.isEntry = true;
          sema.entryTokenOwner = site->owner;
        }
      g.semas.push_back(std::move(sema));
      for (Node *site : sites) {
        site->sema = sid;
        site->count = count;
      }
    }
    for (const auto &storage : g.nodes) if (Node *node = storage.get(); node->kind == Node::Release && node->sat)
        node->sema = node->sat->sema;
    if (!g.root->children.empty())
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
  DenseMap<Node *, EdgeRefs> atDst;
  DenseMap<Node *, EdgeRefs> atSrc;
  DenseMap<Node *, Node *> lastAfter;
  DenseMap<Node *, Node *> channelParent;
  std::map<BoundaryKey, ReleasePaths> deferredPaths, loopPaths;
  std::map<BoundaryKey, Node *> regionChannels;
  std::map<BoundaryKey, bool> loopRealInputs;
  std::map<BoundaryKey, Tokens::Token> boundaryTokens;
  DenseSet<const EdgeRec *> handledEdges;
  SmallVector<Node *, 4> activeLoops;
  DenseMap<Node *, Owner> loopOwner;
  DenseMap<Node *, Node *> loopDemand;
  DenseMap<Node *, Node *> loopLastOwnerAccess;
  DenseSet<Node *> seeded;
  SmallVector<Node *, 8> acquires;
  bool hadError = false;
};

static void addPart(SmallVectorImpl<int> &parts, int part) {
  if (!llvm::is_contained(parts, part)) parts.push_back(part);
}
static SmallVector<int, 4> computeRequiredParts(Node *head) {
  SmallVector<int, 4> chainParts;
  for (Node *n = head; n; n = n->next) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire || n->kind == Node::Release) && n->owner)
      addPart(chainParts, n->owner->first);
    if (!n->isRegion()) continue;
    SmallVector<int, 4> regionParts;
    for (Node *child : n->children)
      for (int part : computeRequiredParts(child)) addPart(regionParts, part);
    llvm::sort(regionParts);
    n->requiredParts.assign(regionParts.begin(), regionParts.end());
    for (int part : regionParts) addPart(chainParts, part);
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
  if (numTmemBlocks + blockM * blockN * 2 > numTMEMRows * numTMEMColumns) return false;
  return !(isa<nvidia_gpu::TCGen5MMAScaledOp>(mmaOp.getOperation()) && blockN == 256);
}
static scf::ForOp outerWSLoop(scf::ForOp loop) {
  scf::ForOp ws = loop;
  for (Operation *p = loop; p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      if (gpu::hasWarpSpecializeTag(f)) ws = f;
  return ws;
}
static bool isMultiBufferedGroup(GroupDag &g, int numTmemBlocks) {
  for (const Member &member : g.pieceTable.members)
    for (Operation *user : member.allocOp->getResult(0).getUsers()) {
      auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user);
      auto loop = dyn_cast<scf::ForOp>(user->getParentOp());
      if (!mma || !loop) continue;
      if (nvidia_gpu::hasAccReadModifyWrite(mma, loop) || !nvidia_gpu::isAccMultibufferingPossible(mma, loop) ||
          getDisallowAccMultiBuffer(outerWSLoop(loop)) || !canDoubleBufferAcc(mma, numTmemBlocks))
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

static LogicalResult computeBackingCopies(GroupDag &g, ArrayRef<EdgeRec> edges, bool useMetaPartitioner,
                                          int &numTmemBlocks) {
  g.numCopies = 1;
  bool synchronized = !edges.empty();
  FailureOr<std::optional<int>> planned = getPlannedBufferCopy(g);
  if (failed(planned)) return failure();
  std::optional<int> plannedCopy = *planned;
  if (synchronized && plannedCopy) g.numCopies = *plannedCopy;
  else if (synchronized && g.isTmem() && !useMetaPartitioner && isMultiBufferedGroup(g, numTmemBlocks))
    g.numCopies = 2;
  if (synchronized && g.isTmem())
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.numCopies;
    }
  return success();
}
static LogicalResult computeSemaphoreCopies(GroupDag &g, int lowerSemaphoreNumStages) {
  g.numSemaphoreCopies = g.numCopies;
  FailureOr<std::optional<int>> planned = getPlannedBufferCopy(g);
  if (failed(planned)) return failure();
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad)) hasProducerLoad = true;
  });
  if (!g.semas.empty() && g.isLocal() && !*planned && hasProducerLoad)
    g.numSemaphoreCopies = std::max(1, lowerSemaphoreNumStages);
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (g.semas.empty()) return success();
  DenseSet<Node *> used;
  SmallVector<bool> supplied(g.semas.size());
  SmallVector<SmallVector<Node *, 2>, 4> releases(g.semas.size());
  forEachNode(g, [&](Node *n) {
    if (n->tokenSource) used.insert(n->tokenSource);
    if (n->flow)
      for (Node *exit : n->flow->exits)
        if (exit) used.insert(exit);
    if (n->kind == Node::Release && n->sema < supplied.size()) {
      supplied[n->sema] = true;
      releases[n->sema].push_back(n);
    }
  });
  for (auto [sid, sema] : llvm::enumerate(g.semas)) supplied[sid] = supplied[sid] || sema.isEntry;

  auto resolveTokenOwner = [&](Node *producer) -> std::optional<Owner> {
    DenseSet<Node *> seen;
    while (producer && seen.insert(producer).second) {
      if (producer->kind == Node::Acquire) {
        if (producer->sema >= g.semas.size()) return std::nullopt;
        Owner owner = producer->owner;
        if (!owner && g.semas[producer->sema].isEntry) owner = g.semas[producer->sema].entryTokenOwner;
        return std::optional<Owner>(std::in_place, owner);
      }
      if (!producer->isRegion()) return std::nullopt;
      if (producer->flow) return std::optional<Owner>(std::in_place, producer->flow->owner);
      producer = producer->tokenSource;
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
    if (!n->isRegion() || !n->flow) return success();
    const RegionFlow &c = *n->flow;
    bool needsInput = n->kind == Node::For || n->children.size() < 2 || llvm::is_contained(c.exits, nullptr);
    if (needsInput && !n->tokenSource) return semaError(n->op) << "region pass path has no exact input token";
    if (n->tokenSource && !compatible(n->tokenSource, c.owner))
      return semaError(n->op) << "region input token has incompatible owner";
    if (!c.sema || *c.sema >= g.semas.size()) return semaError(n->op) << "region token result has no render channel";
    if (c.exits.size() != n->children.size()) return semaError(n->op) << "region flow does not cover every exit path";
    if (!used.contains(n)) return semaError(n->op) << "region token result has no consumer";
    for (auto [index, final] : llvm::enumerate(c.exits)) {
      if (!final) continue;
      if ((final->kind != Node::Acquire || !resolveTokenOwner(final)) && !(final->isRegion() && final->flow))
        return semaError(n->op) << "region path exports no token";
      std::optional<Owner> owner = resolveTokenOwner(final);
      if (!owner || !sameOwner(*owner, c.owner))
        return semaError(n->op) << "region path exports another owner's token";
      DenseSet<Node *> childNodes;
      forEachNode(n->children[index], [&](Node *c) { childNodes.insert(c); });
      if (!childNodes.contains(final)) return semaError(n->op) << "region path exports no token";
    }
    return success();
  };
  auto verifyNode = [&](Node *n) -> LogicalResult {
    if (failed(verifyFlow(n))) return failure();
    if (n->kind == Node::Release) {
      if (n->sema >= g.semas.size() || !n->sat || n->sat->kind != Node::Acquire || !n->scheduleAnchor)
        return semaError(g.root->op) << "release has incomplete protocol facts";
      if (n->payloads.empty()) return semaError(g.root->op) << "release without payload record";
      if (!n->count || !n->tokenSource || !compatible(n->tokenSource, n->owner))
        return semaError(g.root->op) << "release has no exact token source";
      bool exactAnchor = n->scheduleAnchor == n->tokenSource || n->scheduleAnchor->tokenSource == n->tokenSource ||
                         (n->scheduleAnchor->kind == Node::Enter && n->tokenSource->isRegion() &&
                          n->scheduleAnchor->parent == n->tokenSource);
      if (!exactAnchor) return semaError(g.root->op) << "release lost its exact completion";
      if (n->sat->sema != n->sema || n->count * n->payloads.size() > getSema(g, n).count)
        return semaError(g.root->op) << "release has incompatible acquire";
      bool sameChain = n->sat->parent == n->parent;
      bool recurrence = n->sat->recurrenceDistance.has_value();
      if (sameChain && !precedesInChain(n, n->sat) && !getSema(g, n).isEntry && !recurrence &&
          (!n->parent || n->parent->kind != Node::For))
        return semaError(g.root->op) << "release does not precede its acquire";
    }
    if (n->kind == Node::Access && nodeTouchesGroup(g, n) &&
        (!n->tokenSource || !compatible(n->tokenSource, n->owner)))
      return semaError(n->op) << "buffer access has no valid owner token";
    if (n->kind == Node::Acquire) {
      if (n->sema >= g.semas.size() || !n->scheduleAnchor || !n->count || !supplied[n->sema])
        return semaError(g.root->op) << "acquire has no valid supply";
      if (n->count != getSema(g, n).count) return semaError(g.root->op)
               << "semaphore acquired with non-uniform pending count";
      if (getSema(g, n).isEntry)
        for (Node *loop = n->parent; loop; loop = loop->parent)
          if (loop->kind == Node::For && llvm::none_of(releases[n->sema], [&](Node *release) {
                return nestedIn(release, loop);
              })) return semaError(g.root->op) << "repeated entry acquire has no per-loop release";
      if (n->owner) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema] && *acqClass[n->sema] != k) {
          if (shouldDumpDag()) dumpDagTree(g, DumpStage::Sync);
          return semaError(g.root->op) << "semaphore acquired by two partitions (M3 violation)";
        }
        acqClass[n->sema] = k;
      }
      if (n->recurrenceDistance && *n->recurrenceDistance <= 0) return semaError(g.root->op)
               << "recurrence acquire has non-positive distance";
    }
    return success();
  };
  return g.root->children.empty() ? success() : forEachNodeChecked(g.root->children[0], verifyNode);
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;
struct ScheduleEdge { Operation *producer, *consumer; };
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
  return group.isCircular() ? PhysicalKey{group.bufferId, nullptr} : PhysicalKey{0, &group};
}
struct PhysicalSchedules {
  MutableArrayRef<GroupDag> groups;
  llvm::MapVector<PhysicalKey, SmallVector<GroupDag *, 2>> sets;
  std::map<std::pair<PhysicalKey, Operation *>, SlotSchedule> loopSlots;
  explicit PhysicalSchedules(MutableArrayRef<GroupDag> groups) : groups(groups) {
    for (GroupDag &group : groups) sets[physicalKey(group)].push_back(&group);
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
static bool isSlotEvent(const Node *n) {
  return n->kind == Node::Access || (n->isRegion() && !n->pieceInfo.empty());
}
static bool isDirectLoopNode(const Node *n) {
  return n->parent && n->parent->kind == Node::For &&
         (n->prev || n->next || llvm::is_contained(n->parent->children, n));
}

static SlotSchedule replaySlots(ArrayRef<SlotEvent> events, bool assignOffsets = false) {
  SlotSchedule result;
  DenseMap<GroupDag *, int64_t> lastProduced;
  int64_t cursor = -1;
  for (const SlotEvent &event : events) {
    std::optional<int64_t> required;
    if (slotEventEffect(event.node) == Effect::W) {
      if (event.advances > 1) result.complete = false;
      cursor += event.advances;
      result.advancesPerIteration += event.advances;
      if (cursor >= 0) required = lastProduced[event.group] = cursor;
    } else if (auto it = lastProduced.find(event.group); it != lastProduced.end()) {
      required = it->second;
    }
    if (!required) {
      result.complete = false;
      continue;
    }
    result.ordinalByAccess[event.node] = *required;
    if (assignOffsets) event.node->bufferStageOffset = *required - cursor;
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
      if (member.type != type) return semaError(member.allocOp) << "circular local group has mismatched member types";
      if (g->numCopies != numCopies)
        return semaError(member.allocOp) << "circular local group has mismatched buffer.copy";
      if (member.circularStart < 0 || member.circularStart >= numCopies)
        return semaError(member.allocOp) << "circular buffer.start is outside buffer.copy";
      if (!starts.insert(member.circularStart).second)
        return semaError(member.allocOp) << "duplicate circular buffer.start in one group";
      forEachNode(*g, [&](Node *n) {
        if (n->kind == Node::Access)
          eventsByOp[n->op].push_back( SlotEvent{g, n, slotEventEffect(n) == Effect::W});
      });
    }
    SmallVector<SlotEvent> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end()) ordered.append(it->second.begin(), it->second.end());
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
        do
          access = forward ? access->next : access->prev;
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
struct LoopAnchors { scf::ForOp loop; Operation *producer, *consumer; };
static std::optional<LoopAnchors> findCommonScheduledLoop(Operation *producer, Operation *consumer) {
  for (Operation *parent = producer->getParentOp(); parent; parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName)) continue;
    Operation *producerInLoop = loop.getBody()->findAncestorOpInBlock(*producer);
    Operation *consumerInLoop = loop.getBody()->findAncestorOpInBlock(*consumer);
    if (producerInLoop && consumerInLoop) return LoopAnchors{loop, producerInLoop, consumerInLoop};
  }
  return std::nullopt;
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
  if (group.isCircular() || group.pieceTable.members.size() < 2 || group.numSemaphoreCopies <= 1) return false;
  bool authored = group.numCopies > 1 && llvm::all_of(group.pieceTable.members, [](const Member &m) {
                    return m.allocOp->hasAttr(kBufferCopyAttrName);
                  });
  if (authored || group.numSemaphoreCopies <= group.numCopies) return authored;
  const Member &first = group.pieceTable.members.front();
  return llvm::all_of(group.pieceTable.members, [&](const Member &member) {
    return member.offset == first.offset && member.extent == first.extent && member.type == first.type;
  });
}
static LogicalResult
assignAliasedHandoffStageOffsets(PhysicalSchedules &physical, GroupDag &group) {
  if (!isAliasedMultibufferedGroup(group)) return success();
  bool hasShiftedRelease = false;
  auto assignRelease = [&](Node *release) -> LogicalResult {
    if (release->kind != Node::Release || !release->sat) return success();
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
        consumerIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0) {
      if (shouldDumpDag()) dumpDagTree(group, DumpStage::Sync);
      return semaError(producer->op) << "cannot derive multibuffered alias handoff slots (complete "
             << slots.complete << ", advances " << slots.advancesPerIteration
             << ", producer " << (producerIt != slots.ordinalByAccess.end())
             << ", consumer " << (consumerIt != slots.ordinalByAccess.end()) << ")";
    }
    int64_t numSemaphoreCopies = group.numSemaphoreCopies;
    int64_t offset = 0;
    if (release->sat->recurrenceDistance) {
      int64_t nextConsumer = consumerIt->second + *release->sat->recurrenceDistance * slots.advancesPerIteration;
      offset = positiveMod(nextConsumer - producerIt->second, numSemaphoreCopies);
    } else if (precedesInChain(release, release->sat)) {
      offset = positiveMod(consumerIt->second - producerIt->second, numSemaphoreCopies);
    } else if (!computeLoopCarriedDistance( slots, numSemaphoreCopies, producer, consumer)) {
      int64_t nextConsumer = consumerIt->second + slots.advancesPerIteration;
      offset = positiveMod(nextConsumer - producerIt->second, numSemaphoreCopies);
    }
    release->stageOffset = offset;
    hasShiftedRelease |= offset != 0;
    return success();
  };
  for (const auto &node : group.nodes)
    if (isDirectLoopNode(node.get()) && failed(assignRelease(node.get()))) return failure();
  for (const auto &node : group.nodes) {
    if (!isDirectLoopNode(node.get())) continue;
    if (!hasShiftedRelease && node->kind == Node::Release) node->stageOffset.reset();
    if (hasShiftedRelease && node->kind == Node::Acquire) node->stageOffset = 0;
  }
  return success();
}

static LogicalResult solveOwnerScheduleConstraints(LoopScheduleModel &model) {
  if (model.ownerConstraints.empty()) return success();
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
      if (offset[consumer] >= candidate) continue;
      offset[consumer] = candidate;
      predecessor[consumer] = edgeIndex;
      lastUpdated = edgeIndex;
    }
    if (!lastUpdated) break;
  }
  if (lastUpdated) {
    unsigned vertex = getVertex(model.ownerConstraints[*lastUpdated].consumerOwner);
    for (unsigned i = 0; i < numVertices; ++i) {
      if (!predecessor[vertex]) break;
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
    if (cycle.empty()) cycle.push_back(*lastUpdated);
    int64_t cycleDelay = 0;
    for (unsigned edgeIndex : cycle)
      cycleDelay += model.ownerConstraints[edgeIndex].requiredDelay();
    const OwnerScheduleConstraint &first = model.ownerConstraints[cycle.front()];
    InFlightDiagnostic diag = semaError(first.producer)
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
      if (vertex == to) return true;
      if (seen[vertex]) continue;
      seen[vertex] = true;
      for (const OwnerScheduleConstraint &constraint : model.ownerConstraints)
        if (isTight(constraint) && getVertex(constraint.producerOwner) == vertex)
          stack.push_back(getVertex(constraint.consumerOwner));
    }
    return false;
  };
  auto alreadySSAOrdered = [](Operation *producer, Operation *consumer) {
    SetVector<Operation *> slice;
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.omitUsesFromAbove = true;
    options.filter = [&](Operation *op) {
      return op->getBlock() == consumer->getBlock();
    };
    (void)getBackwardSlice(consumer, &slice, options);
    return slice.contains(producer);
  };
  for (const OwnerScheduleConstraint &constraint : model.ownerConstraints) {
    bool directlySameWave = constraint.requiredDelay() == 0;
    bool onZeroDelayCycle = isTight(constraint) && hasTightPath(getVertex(constraint.consumerOwner),
                     getVertex(constraint.producerOwner));
    if ((directlySameWave || onZeroDelayCycle) && !alreadySSAOrdered(constraint.producer, constraint.consumer))
      model.clusterEdges.push_back( ScheduleEdge{constraint.producer, constraint.consumer});
  }
  return success();
}

static LogicalResult addSyncScheduleEdges(PhysicalSchedules &physical,
                                          llvm::MapVector<Operation *, LoopScheduleModel> &modelsByLoop) {
  for (GroupDag &group : physical.groups) {
    auto addReleaseEdge = [&](Node *release) -> LogicalResult {
      if (release->kind != Node::Release || !release->sat) return success();
      Node *acquire = release->sat;
      Operation *producer = findScheduleAnchor(release->scheduleAnchor, /*producer=*/true);
      Operation *consumer = findScheduleAnchor(acquire->scheduleAnchor);
      if (!producer || !consumer) return success();
      std::optional<LoopAnchors> anchors = findCommonScheduledLoop(producer, consumer);
      if (!anchors) return success();
      auto [loop, producerAnchor, consumerAnchor] = *anchors;
      if (producerAnchor == consumerAnchor) return success();
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producerAnchor);
      gpu::StageCluster consumerSchedule = gpu::getStageCluster(consumerAnchor);
      if (!producerSchedule || !consumerSchedule) return success();
      int64_t distance = acquire->recurrenceDistance.value_or(0);
      if (!acquire->recurrenceDistance && !precedesInChain(release, acquire)) {
        const SlotSchedule &slots = getSlotSchedule(physical, group, loop);
        std::optional<int64_t> loopCarriedDistance = computeLoopCarriedDistance(
                slots, group.numSemaphoreCopies, release->scheduleAnchor, acquire->scheduleAnchor);
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
      return success();
    };
    for (const auto &node : group.nodes)
      if (isDirectLoopNode(node.get()) && failed(addReleaseEdge(node.get()))) return failure();
  }
  return success();
}
static void addSSAClusterConstraints(scf::ForOp loop, SmallVectorImpl<ScheduleEdge> &edges) {
  for (Operation &consumer : loop.getBody()->without_terminator()) {
    gpu::StageCluster consumerSchedule = gpu::getStageCluster(&consumer);
    if (!consumerSchedule) continue;
    for (Value operand : getNestedOperands(&consumer)) {
      auto [producer, distance] = triton::getDefiningOpAndDistance(loop, operand);
      if (!producer) continue;
      producer = loop.getBody()->findAncestorOpInBlock(*producer);
      if (!producer || producer == &consumer) continue;
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producer);
      if (!producerSchedule) continue;
      if (producerSchedule->first == consumerSchedule->first + distance)
        edges.push_back(ScheduleEdge{producer, &consumer});
    }
  }
}
static LogicalResult legalizeLoopSchedule(scf::ForOp loop, ArrayRef<ScheduleEdge> edges) {
  SmallVector<Operation *, 32> scheduledOps;
  DenseMap<Operation *, int64_t> original, cluster;
  for (Operation &op : loop.getBody()->without_terminator()) {
    gpu::StageCluster schedule = gpu::getStageCluster(&op);
    if (!schedule) continue;
    scheduledOps.push_back(&op);
    original[&op] = cluster[&op] = schedule->second;
  }
  for (const ScheduleEdge &edge : edges)
    if (!edge.producer->isBeforeInBlock(edge.consumer) &&
        cluster.contains(edge.producer) && cluster.contains(edge.consumer))
      cluster[edge.producer] = std::min(cluster.lookup(edge.producer), cluster.lookup(edge.consumer) - 1);
  bool changed = false;
  for (unsigned iteration = 0; iteration <= scheduledOps.size(); ++iteration) {
    changed = false;
    for (const ScheduleEdge &edge : edges) {
      auto [producer, consumer] = edge;
      if (!cluster.contains(producer) || !cluster.contains(consumer)) continue;
      int64_t required = cluster.lookup(producer) + (producer->isBeforeInBlock(consumer) ? 0 : 1);
      if (cluster.lookup(consumer) >= required) continue;
      cluster[consumer] = required;
      changed = true;
    }
    if (!changed) break;
    if (iteration == scheduledOps.size()) {
      InFlightDiagnostic diag = semaError(loop) << "cyclic loop.cluster constraints";
      if (shouldDumpDag())
        for (const ScheduleEdge &edge : edges)
          diag.attachNote(edge.producer->getLoc()) << edge.producer->getName() << " (cluster "
              << cluster.lookup(edge.producer) << ") -> " << edge.consumer->getName() << " (cluster "
              << cluster.lookup(edge.consumer) << ")";
      return failure();
    }
  }
  int64_t rebase = 0;
  for (Operation *op : scheduledOps) rebase = std::max(rebase, original.lookup(op) - cluster.lookup(op));
  OpBuilder builder(loop.getContext());
  for (Operation *op : scheduledOps) {
    gpu::StageCluster oldSchedule = gpu::getStageCluster(op);
    if (cluster.lookup(op) > std::numeric_limits<int32_t>::max() - rebase)
      return semaError(op) << "legalized loop.cluster exceeds i32 range";
    int64_t newCluster = cluster.lookup(op) + rebase;
    if (newCluster < original.lookup(op)) return semaError(op) << "legalization lowered an authored loop.cluster";
    if (newCluster == oldSchedule->second) continue;
    gpu::setStageCluster(builder, op, std::make_pair(oldSchedule->first, static_cast<int>(newCluster)));
  }
  return success();
}
static gpu::StageCluster scheduleAtOwnerBoundary( const Node *n, gpu::StageCluster schedule) {
  if (!schedule || findScheduleAnchor(n->next)) return schedule;
  auto forOp = dyn_cast_or_null<scf::ForOp>(n->parent ? n->parent->op : nullptr);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName)) return schedule;
  auto [stage, cluster] = *schedule;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate || candidate->first != stage || !gpu::hasPartition(&op)) continue;
    SetVector<int> partitions = gpu::getPartitionIds(&op);
    if (partitions.contains(n->owner->first)) cluster = std::max(cluster, candidate->second);
  }
  return std::make_pair(stage, cluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache);
static void assignSyncScheduleRegion(Node *n, ScheduleCache &cache) {
  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    ScheduleCache body = cache;
    assignSyncScheduleChain(n->children[0], body);
    if (!gpu::hasWarpSpecializeTag(forOp)) cache = std::move(body);
    return;
  }
  ScheduleCache thenCache = cache;
  ScheduleCache elseCache = cache;
  assignSyncScheduleChain(n->children[0], thenCache);
  if (n->children.size() > 1 && n->children[1]) assignSyncScheduleChain(n->children[1], elseCache);
  cache = std::move(thenCache);
  for (auto &[key, stageCluster] : elseCache) cache.try_emplace(key, stageCluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      if (n->owner) {
        gpu::StageCluster boundary = scheduleAtOwnerBoundary(n, cache.lookup(ownerKey(n->owner)));
        if (n->recurrenceDistance && n->next && n->next->kind == Node::Exit) {
          n->stageCluster = boundary;
        } else {
          Operation *anchor = findScheduleAnchor( n->scheduleAnchor ? n->scheduleAnchor : n->next);
          n->stageCluster = anchor ? gpu::getStageCluster(anchor) : boundary;
        }
      }
      break;
    case Node::Release:
      if (n->owner) n->stageCluster = cache.lookup(ownerKey(n->owner));
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
  for (GroupDag &group : groups)
    forEachNode(group, [&](Node *node) {
      if (node->kind == Node::For && node->op && node->op->hasAttr(triton::kScheduledMaxStageAttrName))
        modelsByLoop.try_emplace(node->op);
    });
  if (failed(assignCircularStageOffsets(physical))) return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(physical, group))) return failure();
  if (failed(addSyncScheduleEdges(physical, modelsByLoop))) return failure();
  for (auto &[loopOp, model] : modelsByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    if (failed(solveOwnerScheduleConstraints(model))) {
      if (shouldDumpDag())
        for (GroupDag &group : groups) dumpDagTree(group, DumpStage::Sync);
      return failure();
    }
    addSSAClusterConstraints(loop, model.clusterEdges);
    if (model.clusterEdges.empty()) continue;
    if (failed(legalizeLoopSchedule(loop, model.clusterEdges))) {
      if (shouldDumpDag())
        for (GroupDag &group : groups) dumpDagTree(group, DumpStage::Sync);
      return failure();
    }
  }
  return success();
}
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  if (groups.empty()) return success();
  if (failed(analyzeSyncSchedule(groups))) return failure();
  for (GroupDag &g : groups) {
    if (g.root->children.empty()) continue;
    ScheduleCache cache;
    assignSyncScheduleChain(g.root->children[0], cache);
  }
  return success();
}

LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages, int &numTmemBlocks) {
  SmallVector<EdgeRec> edges;
  DenseSet<Node *> reusable;
  if (!g.root->children.empty()) {
    ChainState top; // function chain: games start at bottom (first-touch)
    ChainWalker(g, top, edges, reusable, /*underFor=*/false) .run(g.root->children[0]);
  }
  reduceEdges(g, edges, reusable);
  if (failed(computeBackingCopies(g, edges, useMetaPartitioner, numTmemBlocks))) return failure();
  if (!edges.empty() && failed(DirectBuilder(g, edges, reusable).run())) return failure();
  if (!g.root->children.empty()) computeRequiredParts(g.root->children[0]);
  if (failed(computeSemaphoreCopies(g, lowerSemaphoreNumStages))) return failure();
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
}
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g, const Node *region, unsigned chainIdx) {
  if (!region || !region->flow) return;
  os << " yield{";
  const RegionFlow &c = *region->flow;
  Node *f = chainIdx < c.exits.size() ? c.exits[chainIdx] : nullptr;
  if (!f)
    os << "pass";
  else if (f->kind == Node::Acquire || f->kind == Node::Release)
    os << (f->kind == Node::Acquire ? "a " : "r ") << getSema(g, f).name;
  else
    os << (f->kind == Node::For ? "scf.for" : "scf.if");
  os << "}";
}
static void printEffects(llvm::raw_ostream &os, const Node *n) {
  if (n->pieceInfo.empty()) return;
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
      if (stage != DumpStage::Sync) break;
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
      if (stage == DumpStage::Access) printEffects(os, n);
      else
        printPieceRecord(os, n, loop ? n->op : anchor);
      if (stage == DumpStage::Sync) printThreadInfo(os, n);
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
        if (n->children.size() > 1) dumpDagChain(g, n->children[1], depth + 2, n, 1, stage);
      }
      break;
    }
    case Node::Enter:
    case Node::Exit:
      if (stage == DumpStage::Access) break;
      os << treePrefix(depth) << (n->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
      printPieceRecord(os, n, anchor);
      if (n->kind == Node::Exit && stage == DumpStage::Sync) printYieldInfo(os, g, region, chainIdx);
      os << "\n";
      break;
    case Node::Func:
      break;
    }
  }
}
void dumpDagTree(GroupDag &g, DumpStage stage) {
  if (!g.root->children.empty()) dumpDagChain(g, g.root->children[0], 1, nullptr, 0, stage);
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
