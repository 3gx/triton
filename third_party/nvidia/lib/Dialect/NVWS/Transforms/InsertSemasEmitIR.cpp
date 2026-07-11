// Protocol materialization; see sema-docs/insert-semas/emit-ir.md.
#include "InsertSemas.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::triton::nvws_semas {

namespace {
class SyncDagDumper {
public:
  SyncDagDumper(GroupDag &group, llvm::raw_ostream &stream)
      : g(group), os(stream) {}

  void printTree() {
    for (Node *head : g.root->children)
      printChain(head, 1, nullptr, 0);
  }

  void print(triton::FuncOp func) {
    os << "SYNC-DAG\n|- func @" << func.getName() << "\n";
    printTree();
    if (g.semas.empty()) {
      os << "  BACKING: untouched (no semaphores)\n";
      return;
    }
    os << "  SEMAS: ";
    llvm::interleave(g.semas, os, [&](const Sema &sema) {
      os << sema.name << "{count=" << sema.count;
      if (sema.entryOwner)
        os << " entry inherit=" << ownerStr(nullptr, *sema.entryOwner);
      os << "}";
    }, " ");
    os << "\n  BACKING: numCopies=" << g.numCopies << "\n";
  }

private:
  void printPrefix(unsigned depth) {
    for (unsigned i = 0; i < depth; ++i)
      os << "|  ";
  }

  StringRef semaName(const Node *node) const {
    if (node->sema < g.semas.size())
      return g.semas[node->sema].name;
    return "<unformed>";
  }

  void printPieces(const Node *node, Operation *anchor) {
    if (node->pieceInfo.empty())
      return;
    os << " pieces{";
    llvm::interleaveComma(sortedPieceInfo(node), os, [&](const auto &entry) {
      os << "P" << entry.first << ":"
         << (entry.second.effect == Effect::W ? "W" : "R") << ":"
         << ownerStr(anchor, entry.second.owner);
    });
    os << "}";
  }

  void printYield(const Node *region, unsigned index) {
    if (!region || !region->flow)
      return;
    Node *final = index < region->flow->exits.size()
                      ? region->flow->exits[index]
                      : nullptr;
    os << " yield{";
    if (!final) {
      os << "pass";
    } else {
      switch (final->kind) {
      case Node::Acquire:
      case Node::Release:
        os << (final->kind == Node::Acquire ? "a " : "r ")
           << semaName(final);
        break;
      case Node::For:
      case Node::If:
        os << (final->kind == Node::For ? "scf.for" : "scf.if");
        break;
      default:
        llvm_unreachable("invalid region result");
      }
    }
    os << "}";
  }

  void printRegion(const Node *node, Operation *anchor) {
    printPieces(node, anchor);
    if (!node->requiredParts.empty()) {
      os << " parts{";
      llvm::interleaveComma(node->requiredParts, os);
      os << "}";
    }
    if (node->flow)
      os << " thread{" << ownerStr(node->op, node->flow->owner) << "}";
    os << "\n";
  }

  void printAccess(const Node *node, unsigned depth) {
    printPrefix(depth);
    os << "|- ";
    llvm::interleaveComma(node->touches, os, [&](const Touch &touch) {
      os << (touch.effect == Effect::W ? "W" : "R") << " m"
         << touch.member;
    });
    os << "  " << node->op->getName().getStringRef() << " "
       << ownerStr(node->op, node->owner) << "\n";
  }

  void printProtocol(const Node *node, unsigned depth, Operation *anchor) {
    printPrefix(depth);
    os << "|- " << (node->kind == Node::Acquire ? "a" : "r") << "  "
       << semaName(node);
    if (node->count > 1)
      os << "(" << node->count << ")";
    os << "  " << ownerStr(anchor, node->owner);
    if (node->kind == Node::Acquire) {
      if (node->sema < g.semas.size() && g.semas[node->sema].entryOwner &&
          !node->owner)
        os << "  ; entry";
    } else {
      os << " [";
      llvm::interleaveComma(node->payloads, os, [&](AsyncOp payload) {
        os << nvws::stringifyAsyncOp(payload);
      });
      os << "]";
    }
    if (node->stageOffset)
      os << "  stage-offset=" << *node->stageOffset;
    os << "\n";
  }

  void printChain(const Node *head, unsigned depth, const Node *parent,
                  unsigned index) {
    for (const Node *node = head; node; node = node->next) {
      Operation *anchor = node->parent ? node->parent->op : nullptr;
      switch (node->kind) {
      case Node::Access:
        printAccess(node, depth);
        break;
      case Node::Acquire:
      case Node::Release:
        printProtocol(node, depth, anchor);
        break;
      case Node::For:
        printPrefix(depth);
        os << "|- scf.for";
        if (gpu::hasWarpSpecializeTag(node->op))
          os << " (WS, tag=" << *gpu::getWarpSpecializeTag(node->op) << ")";
        printRegion(node, node->op);
        printChain(node->children[0], depth + 1, node, 0);
        break;
      case Node::If:
        printPrefix(depth);
        os << "|- scf.if";
        printRegion(node, anchor);
        printPrefix(depth + 1);
        os << "|- then\n";
        printChain(node->children[0], depth + 2, node, 0);
        printPrefix(depth + 1);
        os << "|- else"
           << (cast<scf::IfOp>(node->op).elseBlock() ? "" : " (virtual)")
           << "\n";
        printChain(node->children[1], depth + 2, node, 1);
        break;
      case Node::Enter:
      case Node::Exit:
        printPrefix(depth);
        os << (node->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
        printPieces(node, anchor);
        if (node->kind == Node::Exit)
          printYield(parent, index);
        os << "\n";
        break;
      case Node::Func:
        break;
      }
    }
  }

  GroupDag &g;
  llvm::raw_ostream &os;
};
} // namespace

bool shouldDumpDag() {
  const char *env = ::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  return env && StringRef(env) == "1";
}

void dumpSyncDagTree(GroupDag &g) {
  if (shouldDumpDag())
    SyncDagDumper(g, llvm::errs()).printTree();
}

void dumpSyncDagTrees(MutableArrayRef<GroupDag> groups) {
  if (!shouldDumpDag())
    return;
  for (GroupDag &g : groups)
    SyncDagDumper(g, llvm::errs()).printTree();
}

void dumpSyncDags(MutableArrayRef<GroupDag> groups, triton::FuncOp func) {
  if (!shouldDumpDag())
    return;
  llvm::errs() << "==== NVWS InsertSemas SYNC-DAG ====\n";
  llvm::errs() << "function: @" << func.getName() << "\n";
  llvm::errs() << "groups: " << groups.size() << "\n";
  for (GroupDag &g : groups)
    SyncDagDumper(g, llvm::errs()).print(func);
}

struct EmitCtx {
  triton::FuncOp func;
  Value poison; // the single function-level ub.poison token (contract E)
  Type tokenType;
  struct Slot {
    GroupDag *g;
    unsigned index; // absolute result / iter_arg index in the NEW op
  };
  llvm::MapVector<Operation *, SmallVector<Slot, 2>> slots;
  DenseSet<Operation *> exactReuseBufferOps;
  struct CachedReuseContract {
    Value view;
    Value token;
  };
  SmallVector<CachedReuseContract, 2> cachedReuseContracts;
};

static bool sameViewType(Type lhs, Type rhs);

struct RenderState {
  struct Token {
    Value value;
    Value sema;
    TokenRef ref;
  };
  // One exact record per token known in this chain. Consumers always name
  // the producer selected by the SYNC-DAG; rendering never guesses by owner or
  // list position.
  SmallVector<Token, 2> tokens;
  DenseSet<Node *> releasedSources;
  struct ViewBundle {
    Node *producer = nullptr;
    SemaId channel = 0;
    Value token;
    Value semaphore;
    Owner owner;
    std::optional<int64_t> bufferStageOffset;
    SmallVector<Value, 2> buffers;
  };
  // Keep the current capability as an explicit exact key rather than a lossy
  // owner-only hash surrogate.
  std::optional<ViewBundle> view;
  void clearViews() { view.reset(); }
  ViewBundle *findViewBundle(const Token &source, const Owner &owner,
                             std::optional<int64_t> bufferStageOffset,
                             MemberId member, Type resultType) {
    if (!view || view->producer != source.ref.producer ||
        view->channel != source.ref.sema || view->token != source.value ||
        view->semaphore != source.sema || !sameOwner(view->owner, owner) ||
        view->bufferStageOffset != bufferStageOffset ||
        !sameViewType(view->buffers[member].getType(), resultType))
      return nullptr;
    return &*view;
  }
  // Exact routing: the record whose producer is the node's recorded token
  // source. No owner guessing.
  const Token *tokenForSource(const Node *producer) const {
    for (const Token &token : tokens)
      if (token.ref.producer == producer)
        return &token;
    return nullptr;
  }
  void recordToken(Value value, Value sema, const TokenRef &ref) {
    llvm::erase_if(tokens, [&](const Token &token) {
      return token.ref.producer == ref.producer;
    });
    releasedSources.erase(ref.producer);
    tokens.push_back(Token{value, sema, ref});
  }
  RenderState nested() const {
    RenderState copy = *this;
    copy.clearViews();
    return copy;
  }
};
template <typename OpT, typename... Args>
static OpT emitInto(OpBuilder &b, Location loc, const Owner &owner,
                    gpu::StageCluster stageCluster, Args &&...args) {
  std::optional<SetVector<int>> ids = SetVector<int>();
  if (owner)
    ids->insert(owner->first);
  else
    ids = std::nullopt;
  auto op = gpu::createInto<OpT>(b, loc, ids, stageCluster, std::forward<Args>(args)...);
  if (owner) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !gpu::hasWarpSpecializeTag(forOp))
      forOp = forOp->template getParentOfType<scf::ForOp>();
    if (!forOp) {
      if (owner->first == 0)
        op->removeAttr(gpu::kPartitionAttrName);
      else
        gpu::setWarpSpecializeTag(op, owner->second);
    }
  }
  return op;
}
static Operation *nextRealOp(const Node *n) {
  for (const Node *m = n; m; m = m->next)
    if ((m->kind == Node::Access || m->kind == Node::For || m->kind == Node::If) && m->op)
      return m->op;
  return nullptr;
}
static Block *chainBlock(Node *node) {
  if (!node || !node->parent || !node->parent->op)
    return nullptr;
  Node *head = node;
  while (head->prev)
    head = head->prev;
  Node *parent = node->parent;
  if (parent->kind == Node::Func)
    return &parent->op->getRegion(0).front();
  for (auto [index, child] : llvm::enumerate(parent->children))
    if (child == head && index < parent->op->getNumRegions())
      return &parent->op->getRegion(index).front();
  return nullptr;
}
static SetVector<int> partitionIdsOfFwd(Operation *op) {
  SetVector<int> s;
  if (gpu::hasPartition(op))
    for (int p : gpu::getPartitionIds(op))
      s.insert(p);
  return s;
}
static ArrayAttr asyncOpsAttr(MLIRContext *ctx, const Node *rel) {
  SmallVector<Attribute, 2> elems;
  for (AsyncOp p : rel->payloads)
    elems.push_back(nvws::AsyncOpAttr::get(ctx, p));
  return ArrayAttr::get(ctx, elems);
}
static Value materializeI32Before(Operation *op, int64_t value);
static void nukeGroupTokens(EmitCtx &ctx, GroupDag &g) {
  auto nukeOp = [&](Operation *op) {
    Value token;
    if (auto load = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      load.getDepMutable().clear();
      token = load.getToken();
    } else if (auto store = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
      store.getDepMutable().clear();
      token = store.getToken();
    } else if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (g.aliases.count(mma.getAccumulator())) {
        mma.getAccDepMutable().clear();
        token = mma.getToken();
      }
    } else if (auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      token = alloc.getToken();
    }
    if (token) token.replaceAllUsesWith(ctx.poison);
  };
  for (const Member &m : g.pieceTable.members)
    nukeOp(m.allocOp);
  for (Operation *op : g.accessNodeOps)
    nukeOp(op);
}
static bool isScalesEnc(gpu::MemDescType t) {
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(t.getEncoding());
}
static gpu::MemDescType backingType(const GroupDag &g, const Member &m) {
  auto t = m.type;
  SmallVector<int64_t> shape(t.getShape());
  if (!isScalesEnc(t))
    shape.insert(shape.begin(), g.numCopies);
  return gpu::MemDescType::get(shape, t.getElementType(), t.getEncoding(),
                               t.getMemorySpace(), /*mutableMemory=*/true);
}
static gpu::MemDescType withMutable(gpu::MemDescType t, bool m) {
  if (t.getMutableMemory() == m)
    return t;
  return gpu::MemDescType::get(t.getShape(), t.getElementType(),
                               t.getEncoding(), t.getMemorySpace(), m, t.getAllocShape());
}
static gpu::MemDescType genericViewType(gpu::MemDescType backing) {
  auto shape = backing.getShape();
  return gpu::MemDescType::get(isScalesEnc(backing) ? shape : shape.drop_front(),
                               backing.getElementType(), backing.getEncoding(), backing.getMemorySpace(),
                               /*mutableMemory=*/true, backing.getShape());
}
static bool sameViewType(Type a, Type b) {
  auto x = cast<gpu::MemDescType>(a), y = cast<gpu::MemDescType>(b);
  return x.getShape() == y.getShape() && x.getElementType() == y.getElementType() &&
         x.getEncoding() == y.getEncoding() && x.getMemorySpace() == y.getMemorySpace() &&
         x.getMutableMemory() == y.getMutableMemory();
}
static gpu::MemDescType viewType(const GroupDag &g, MemberId member,
                                 const Touch &touch,
                                 gpu::MemDescType backing) {
  if (g.isTmem() || touch.member != member) return genericViewType(backing);
  Type type = g.pieceTable.members[member].type;
  if (touch.alias.empty()) type = touch.accessType;
  for (const AliasStep &step : touch.alias) {
    if (step.op->getName().getStringRef() != "ttg.memdesc_index") break;
    type = step.resultType;
  }
  return withMutable(cast<gpu::MemDescType>(type), true);
}

static Operation *backingAnchor(GroupDag &g) {
  Operation *anchor = g.pieceTable.members.front().allocOp;
  for (const Member &m : g.pieceTable.members)
    if (m.allocOp->getBlock() == anchor->getBlock() && m.allocOp->isBeforeInBlock(anchor))
      anchor = m.allocOp;
  while (isa<scf::ForOp>(anchor->getParentOp()))
    anchor = anchor->getParentOp();
  return anchor;
}
static Value emitBacking(OpBuilder &b, Location loc, GroupDag &g,
                         const Member &member) {
  auto type = backingType(g, member);
  Value backing;
  if (g.isTmem())
    backing =
        nvidia_gpu::TMEMAllocOp::create(b, loc, type, Value()).getResult();
  else
    backing = gpu::LocalAllocOp::create(b, loc, type).getResult();
  for (StringRef name :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName,
        kBufferCircularAttrName, kBufferStartAttrName})
    if (Attribute attr = member.allocOp->getAttr(name))
      backing.getDefiningOp()->setAttr(name, attr);
  return backing;
}
static bool sharesCover(const GroupDag &g, unsigned cover, unsigned member) {
  if (member == cover ||
      (!g.isTmem() && llvm::none_of(g.pieceTable.members, [](const Member &m) {
        return m.allocOp->hasAttr(kBufferCopyAttrName);
      })))
    return false;
  const Member &owner = g.pieceTable.members[cover];
  const Member &reuser = g.pieceTable.members[member];
  if (reuser.offset < owner.offset ||
      reuser.offset + reuser.extent > owner.offset + owner.extent)
    return false;
  return g.isTmem() ||
         (reuser.offset == owner.offset &&
          backingType(g, reuser) == backingType(g, owner));
}
static Value emitTmemView(OpBuilder &b, Location loc, Value owner,
                          gpu::MemDescType target, int64_t offset,
                          int64_t sizeHint, bool reinterpret = false) {
  Value view = nvidia_gpu::TMEMSubSliceOp::create(
      b, loc, owner, static_cast<int32_t>(offset),
      static_cast<int32_t>(sizeHint));
  if (reinterpret || view.getType() != target)
    view = gpu::MemDescReinterpretOp::create(b, loc, target, view);
  return view;
}
static void materializeLogicalBacking(GroupDag &g) {
  auto &members = g.pieceTable.members;
  unsigned cover = 0;
  for (auto [i, member] : llvm::enumerate(members))
    if (const Member &current = members[cover];
        member.offset <= current.offset &&
        member.offset + member.extent >= current.offset + current.extent)
      cover = i;
  Operation *anchor = backingAnchor(g);
  OpBuilder b(anchor);
  Location loc = anchor->getLoc();
  g.backing.resize(members.size());
  for (auto [i, member] : llvm::enumerate(members)) {
    if (sharesCover(g, cover, i))
      continue;
    g.backing[i] = emitBacking(b, loc, g, member);
    if (i != cover)
      continue;
    for (int j = members.size() - 1; j >= 0; --j)
      if (sharesCover(g, cover, j)) {
        auto target = backingType(g, members[j]);
        if (members[j].offset == member.offset &&
            target == g.backing[i].getType())
          g.backing[j] = g.backing[i];
        else
          g.backing[j] = emitTmemView(
              b, loc, g.backing[i], target,
              members[j].offset - member.offset, target.getShape().back());
      }
  }
}
static LogicalResult materializeMixedDepth(ArrayRef<GroupDag *> set) {
  if (set.size() != 2)
    return semaError(set.front()->pieceTable.members.front().allocOp)
           << "mixed-depth TMEM reuse requires exactly two logical channels";
  bool firstOwns = canOwnMixedDepthTmem(*set[0], *set[1]);
  bool secondOwns = canOwnMixedDepthTmem(*set[1], *set[0]);
  if (firstOwns == secondOwns)
    return semaError(set.front()->pieceTable.members.front().allocOp)
           << "mixed-depth TMEM reuse has no unique physical owner by span and "
              "element width";
  GroupDag &owner = *set[static_cast<unsigned>(secondOwns)];
  GroupDag &reuser = *set[static_cast<unsigned>(firstOwns)];
  Operation *ownerAnchor = backingAnchor(owner);
  Operation *reuserAnchor = backingAnchor(reuser);
  bool dominates = ownerAnchor->getBlock() == reuserAnchor->getBlock() &&
                   ((ownerAnchor == reuserAnchor && firstOwns) ||
                    (ownerAnchor != reuserAnchor &&
                     ownerAnchor->isBeforeInBlock(reuserAnchor)));
  if (!dominates)
    return semaError(reuserAnchor)
           << "mixed-depth TMEM physical owner does not dominate its reuser";
  OpBuilder b(ownerAnchor);
  Value backing = emitBacking(b, ownerAnchor->getLoc(), owner,
                              owner.pieceTable.members.front());
  auto ownerType = backingType(owner, owner.pieceTable.members.front());
  auto reuserType = backingType(reuser, reuser.pieceTable.members.front());
  auto ownerShape = ownerType.getShape(), reuserShape = reuserType.getShape();
  if (ownerShape.empty() || reuserShape.empty())
    return semaError(reuserAnchor)
           << "mixed-depth TMEM backing has empty shape";
  int64_t offset = reuser.pieceTable.members.front().offset -
                   owner.pieceTable.members.front().offset;
  int64_t ownerN = ownerShape.back(), reuserN = reuserShape.back();
  if (ownerN < reuserN || ownerN % reuserN || offset < 0 ||
      offset + reuserN > ownerN)
    return semaError(reuserAnchor)
           << "mixed-depth TMEM reuser is outside its physical owner";
  unsigned ownerWidth = ownerType.getElementTypeBitWidth();
  unsigned reuserWidth = reuserType.getElementTypeBitWidth();
  int64_t size = reuserN / (ownerWidth / reuserWidth);
  if (size <= 0)
    return semaError(reuserAnchor)
           << "invalid mixed-depth TMEM subslice width";
  Value view = emitTmemView(b, reuserAnchor->getLoc(), backing, reuserType,
                           offset, size, /*reinterpret=*/true);
  owner.backing.assign(1, backing);
  reuser.backing.assign(1, view);
  return success();
}

static LogicalResult materializeCircular(ArrayRef<GroupDag *> set) {
  GroupDag *owner = set.front();
  for (GroupDag *g : set)
    if (g->pieceTable.members.front().circularStart == 0)
      owner = g;
  Operation *ownerAnchor = backingAnchor(*owner);
  Operation *earliest = backingAnchor(*set.front());
  auto type = backingType(*owner, owner->pieceTable.members.front());
  for (GroupDag *g : set) {
    Operation *anchor = backingAnchor(*g);
    if (anchor->getBlock() != earliest->getBlock())
      return semaError(ownerAnchor)
             << "circular folded backings must be defined in one block";
    if (backingType(*g, g->pieceTable.members.front()) != type)
      return semaError(anchor) << "circular logical backing type mismatch";
    if (anchor->isBeforeInBlock(earliest))
      earliest = anchor;
  }
  OpBuilder b(earliest);
  Value backing = emitBacking(b, ownerAnchor->getLoc(), *owner,
                              owner->pieceTable.members.front());
  for (GroupDag *g : set)
    g->backing.assign(1, backing);
  return success();
}
static LogicalResult emitPhysicalIR(EmitCtx &ctx, ArrayRef<GroupDag *> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 2>> mixed, circular;
  for (GroupDag *group : groups) {
    GroupDag &g = *group;
    if (g.mixedDepthPhysicalAlias)
      mixed[g.bufferId].push_back(group);
    else if (g.isCircular())
      circular[g.bufferId].push_back(group);
  }
  std::map<std::pair<int64_t, bool>, Sema *> circularPrimary;
  for (GroupDag *group : groups) {
    GroupDag &g = *group;
    if (g.backing.empty()) {
      if (g.mixedDepthPhysicalAlias) {
        if (failed(materializeMixedDepth(mixed[g.bufferId])))
          return failure();
      } else if (g.isCircular()) {
        if (failed(materializeCircular(circular[g.bufferId])))
          return failure();
      } else {
        materializeLogicalBacking(g);
      }
    }
    OpBuilder b(ctx.func);
    Operation *anchor = backingAnchor(g);
    b.setInsertionPoint(anchor);
    SmallVector<Type> baseTypes;
    for (const Member &member : g.pieceTable.members)
      baseTypes.push_back(backingType(g, member));
    auto semaTy = nvws::SemaphoreType::get(
        b.getContext(), nvws::TypeArrayAttr::get(b.getContext(), baseTypes));
    for (bool entry : {true, false})
      for (Sema &s : g.semas) {
        if (s.entryOwner.has_value() != entry)
          continue;
        if (g.isCircular()) {
          auto [primary, inserted] = circularPrimary.try_emplace(
              std::make_pair(g.bufferId, entry), &s);
          if (!inserted) {
            if (primary->second->count != s.count)
              return semaError(anchor)
                     << "circular folded semaphores disagree on pending_count";
            s.create = primary->second->create;
            continue;
          }
        }
        auto create = nvws::SemaphoreCreateOp::create(
            b, anchor->getLoc(), semaTy, g.backing, entry);
        create.setPendingCountAttr(b.getI32IntegerAttr(s.count));
        s.create = create.getResult();
      }
  }
  return success();
}

static void fixupAnchors(MutableArrayRef<GroupDag> groups, Operation *oldOp, Operation *newOp) {
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->op == oldOp)
        n->op = newOp;
    });
  }
}
static void finishResultFilter(Operation *oldOp, Operation *newOp,
                               const llvm::BitVector &drop,
                               MutableArrayRef<GroupDag> groups) {
  auto attr = newOp->getAttrOfType<ArrayAttr>(gpu::kPartitionOutputsAttrName);
  if (attr && attr.size() == drop.size()) {
    SmallVector<Attribute> kept;
    for (auto [i, value] : llvm::enumerate(attr.getValue()))
      if (!drop.test(i))
        kept.push_back(value);
    newOp->setAttr(gpu::kPartitionOutputsAttrName,
                   ArrayAttr::get(newOp->getContext(), kept));
  }
  unsigned next = 0;
  for (auto [i, result] : llvm::enumerate(oldOp->getResults()))
    if (!drop.test(i))
      result.replaceAllUsesWith(newOp->getResult(next++));
  fixupAnchors(groups, oldOp, newOp);
  oldOp->erase();
}
static void eraseDroppedYields(Operation *op, const llvm::BitVector &drop) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    region.front().getTerminator()->eraseOperands(drop);
  }
}

static bool eraseDeadTokenSlots(EmitCtx &ctx, MutableArrayRef<GroupDag> groups) {
  std::function<bool(Value, DenseSet<Value> &)> hasRealUse =
      [&](Value value, DenseSet<Value> &seen) -> bool {
    if (!seen.insert(value).second)
      return false;
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (auto yield = dyn_cast<scf::YieldOp>(owner)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (index < parent->getNumResults() &&
            hasRealUse(parent->getResult(index), seen))
          return true;
        continue;
      }
      if (auto nested = dyn_cast<scf::ForOp>(owner)) {
        unsigned operand = use.getOperandNumber();
        if (operand >= 3) {
          unsigned index = operand - 3;
          if (index < nested.getNumRegionIterArgs() &&
              (hasRealUse(nested.getRegionIterArg(index), seen) ||
               hasRealUse(nested.getResult(index), seen)))
            return true;
          continue;
        }
      }
      return true;
    }
    return false;
  };
  bool changed = false;
  SmallVector<scf::ForOp> fors;
  SmallVector<scf::IfOp> ifs;
  ctx.func.walk([&](scf::ForOp op) { fors.push_back(op); });
  ctx.func.walk([&](scf::IfOp op) { ifs.push_back(op); });
  for (scf::IfOp ifOp : ifs) {
    llvm::BitVector drop(ifOp.getNumResults());
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (res.getType() == ctx.tokenType && res.use_empty())
        drop.set(i);
    if (drop.none())
      continue;
    SmallVector<Type> keptTypes;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (!drop.test(i))
        keptTypes.push_back(res.getType());
    OpBuilder b(ifOp);
    auto newIf = scf::IfOp::create(b, ifOp.getLoc(), keptTypes, ifOp.getCondition(),
                                   /*withElseRegion=*/!ifOp.getElseRegion()
                                       .empty());
    newIf->setAttrs(ifOp->getAttrs());
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    eraseDroppedYields(newIf, drop);
    finishResultFilter(ifOp, newIf, drop, groups);
    changed = true;
  }
  for (scf::ForOp forOp : fors) {
    llvm::BitVector drop(forOp.getNumResults());
    for (auto [i, res] : llvm::enumerate(forOp.getResults())) {
      DenseSet<Value> seen;
      BlockArgument arg = forOp.getRegionIterArg(i);
      if (res.getType() == ctx.tokenType && res.use_empty() &&
          !hasRealUse(arg, seen)) {
        arg.replaceAllUsesWith(ctx.poison);
        drop.set(i);
      }
    }
    if (drop.none())
      continue;
    SmallVector<Value> keptInits;
    for (auto [i, init] : llvm::enumerate(forOp.getInits()))
      if (!drop.test(i))
        keptInits.push_back(init);
    OpBuilder b(forOp);
    auto newFor = scf::ForOp::create(
        b, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), keptInits);
    newFor->setAttrs(forOp->getAttrs());
    newFor.getRegion().takeBody(forOp.getRegion());
    Block &body = newFor.getRegion().front();
    eraseDroppedYields(newFor, drop);
    llvm::BitVector dropArgs(body.getNumArguments());
    for (unsigned i : drop.set_bits())
      dropArgs.set(1 + i); // +1: induction variable
    body.eraseArguments(dropArgs);
    finishResultFilter(forOp, newFor, drop, groups);
    changed = true;
  }
  return changed;
}

// Aggregate every group's token slots before rebuilding a region exactly once.
static void rewriteSignatures(EmitCtx &ctx, MutableArrayRef<GroupDag> groups) {
  struct Want {
    GroupDag *g;
    Owner owner;
  };
  llvm::MapVector<Operation *, SmallVector<Want, 2>> wanted;
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->flow)
        wanted[n->op].push_back(Want{&g, n->flow->owner});
    });
  }
  SmallVector<Operation *> ops;
  for (auto &[op, _] : wanted)
    ops.push_back(op);
  llvm::stable_sort(ops, [](Operation *a, Operation *b) {
    auto depth = [](Operation *op) {
      unsigned d = 0;
      while ((op = op->getParentOp()))
        ++d;
      return d;
    };
    return depth(a) < depth(b);
  });
  OpBuilder b(ctx.func);
  for (Operation *op : ops) {
    auto &list = wanted[op];
    unsigned nSlots = list.size();
    SmallVector<Value> poisons(nSlots, ctx.poison);
    Operation *newOp = nullptr;
    unsigned base = 0;
    SmallVector<SetVector<int>, 4> outputs;
    if (op->getNumResults() == 0 || op->hasAttr(gpu::kPartitionOutputsAttrName))
      outputs = gpu::getPartitionOutputs(op);
    else
      outputs.assign(op->getNumResults(), partitionIdsOfFwd(op));
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      base = forOp.getNumRegionIterArgs();
      b.setInsertionPoint(forOp);
      auto newLoop = addIterArgsToLoop(b, forOp, poisons);
      appendToForOpYield(newLoop, poisons);
      newOp = newLoop;
    } else {
      auto ifOp = cast<scf::IfOp>(op);
      base = ifOp.getNumResults();
      SmallVector<Type> types(nSlots, ctx.tokenType);
      b.setInsertionPoint(ifOp);
      auto newIf = replaceIfOpWithNewSignature(b, ifOp, types);
      for (Block *blk : {newIf.thenBlock(), newIf.elseBlock()}) {
        auto yield = cast<scf::YieldOp>(blk->getTerminator());
        SmallVector<Value> operands(yield.getOperands());
        operands.append(poisons);
        OpBuilder yb(yield);
        auto newYield = scf::YieldOp::create(yb, yield.getLoc(), operands);
        newYield->setAttrs(yield->getAttrs());
        yield.erase();
      }
      ifOp.erase(); // the husk (utility does not erase)
      newOp = newIf;
    }
    for (const Want &w : list) {
      SetVector<int> set;
      if (w.owner)
        set.insert(w.owner->first);
      else
        set = partitionIdsOfFwd(newOp);
      outputs.push_back(set);
    }
    if (gpu::hasPartition(newOp)) {
      gpu::setPartitionOutputs(newOp, outputs);
      auto ids = gpu::getPartitionIds(newOp);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r)
          if (Operation *term = blk.getTerminator();
              term && !gpu::hasPartition(term))
            term->setAttr(
                gpu::kPartitionAttrName,
                DenseI32ArrayAttr::get(
                    newOp->getContext(),
                    SmallVector<int>(ids.begin(), ids.end())));
    }
    fixupAnchors(groups, op, newOp);
    auto &rec = ctx.slots[newOp];
    for (auto [i, w] : llvm::enumerate(list))
      rec.push_back(EmitCtx::Slot{w.g, base + static_cast<unsigned>(i)});
  }
}
static unsigned slotIndexFor(EmitCtx &ctx, Operation *op, GroupDag *g) {
  for (const EmitCtx::Slot &s : ctx.slots[op])
    if (s.g == g)
      return s.index;
  llvm_unreachable("missing slot");
}
static void refreshAliasResultTypes(Operation *op, Value source) {
  // A mapped semaphore view can carry a staged allocShape that was absent from
  // the original alias operand. Let each alias op derive its result from the
  // cloned operands before falling back to the old mutability-only adjustment.
  if (auto typeInfer = dyn_cast<InferTypeOpInterface>(op)) {
    SmallVector<Type> inferredTypes;
    if (succeeded(typeInfer.inferReturnTypes(
            op->getContext(), op->getLoc(), op->getOperands(),
            op->getAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), inferredTypes)) &&
        inferredTypes.size() == op->getNumResults()) {
      for (auto [result, type] : llvm::zip(op->getResults(), inferredTypes))
        result.setType(type);
      return;
    }
  }
  op->getResult(0).setType(withMutable(
      cast<gpu::MemDescType>(op->getResult(0).getType()),
      cast<gpu::MemDescType>(source.getType()).getMutableMemory()));
}
static Value getView(EmitCtx &ctx, GroupDag &g, RenderState &rs, Node *node,
                     const Touch &touch, Operation *accessOp,
                     const Owner &owner,
                     const RenderState::Token &source) {
  SmallVector<Type, 2> types;
  for (auto [mi, m] : llvm::enumerate(g.pieceTable.members)) {
    auto bt = cast<gpu::MemDescType>(g.backing[mi].getType());
    types.push_back(viewType(g, static_cast<MemberId>(mi), touch, bt));
  }
  gpu::StageCluster stageCluster = gpu::getStageCluster(accessOp);
  RenderState::ViewBundle *bundle = rs.findViewBundle(
      source, owner, node->bufferStageOffset, touch.member,
      types[touch.member]);
  if (bundle && rs.releasedSources.contains(source.ref.producer)) {
    Value buffer = bundle->buffers[touch.member];
    if (!ctx.exactReuseBufferOps.contains(buffer.getDefiningOp()))
      ctx.cachedReuseContracts.push_back({buffer, source.value});
  }
  if (!bundle) {
    OpBuilder b(accessOp);
    auto buf = emitInto<nvws::SemaphoreBufferOp>(
        b, accessOp->getLoc(), owner, stageCluster, source.sema,
        TypeRange(types), source.value);
    if (rs.releasedSources.contains(node->tokenSource))
      ctx.exactReuseBufferOps.insert(buf.getOperation());
    if (node->bufferStageOffset)
      buf.setStage(materializeI32Before(buf, *node->bufferStageOffset));
    // Keep one current bundle, matching the emitter's established locality
    // behavior while making every reuse an exact-capability comparison.
    rs.view = RenderState::ViewBundle{
        source.ref.producer, source.ref.sema, source.value, source.sema, owner,
        node->bufferStageOffset,
        SmallVector<Value, 2>(buf.getBuffers().begin(), buf.getBuffers().end())};
    bundle = &*rs.view;
  }
  Value base = bundle->buffers[touch.member];
  Value cur = base;
  OpBuilder b(accessOp);
  for (const AliasStep &step : touch.alias) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" &&
        sameViewType(step.resultType, cur.getType()))
      continue;
    IRMapping mapping;
    for (auto [idx, operand] : llvm::enumerate(old->getOperands()))
      mapping.map(operand, idx == step.operandIdx ? cur : operand);
    Value source = cur;
    Operation *cloned = b.clone(*old, mapping);
    refreshAliasResultTypes(cloned, source);
    cur = cloned->getResult(0);
  }
  return cur;
}

static LogicalResult renderChain(EmitCtx &ctx, GroupDag &g, Node *head,
                                 RenderState &rs);
static Operation *renderAccess(EmitCtx &ctx, GroupDag &g, Node *n,
                               RenderState &rs,
                               const RenderState::Token &source) {
  Operation *op = n->op;
  Operation *anchor = n->completionAnchor ? n->completionAnchor : op;
  for (const Touch &touch : n->touches) {
    Value view = getView(ctx, g, rs, n, touch, op, n->owner, source);
    if (auto ta = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      OpBuilder b(op);
      auto pidsc = std::make_pair(n->owner, gpu::getStageCluster(op));
      auto vTrue = emitInto<arith::ConstantOp>(b, op->getLoc(), n->owner, pidsc.second, b.getBoolAttr(true));
      anchor = emitInto<nvidia_gpu::TMEMStoreOp>(b, op->getLoc(), n->owner, pidsc.second, Type(), view,
                                                 Value(), ta.getSrc(), vTrue);
      ta.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               use.getOwner() != view.getDefiningOp() &&
               !g.accessNodeOps.contains(use.getOwner());
      });
      return anchor;
    }
    if (auto la = dyn_cast<gpu::LocalAllocOp>(op)) {
      OpBuilder b(op);
      Value src = la.getSrc();
      if (src && !isa<RankedTensorType>(src.getType())) {
        auto splat = emitInto<triton::SplatOp>(b, op->getLoc(), n->owner, gpu::getStageCluster(op),
            RankedTensorType::get(cast<gpu::MemDescType>(view.getType()).getShape(), src.getType()), src);
        src = splat.getResult();
      }
      anchor = emitInto<gpu::LocalStoreOp>(b, op->getLoc(), n->owner, gpu::getStageCluster(op), src, view);
      la.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               !g.accessNodeOps.contains(use.getOwner());
      });
      return anchor;
    }
    for (OpOperand &o : op->getOpOperands())
      if (o.get() == touch.accessValue)
        o.set(view);
  }
  return anchor;
}

static void recordProducerAlias(RenderState &state, Node *producer,
                                Node *except,
                                const RenderState::Token &token) {
  if (producer && producer != except)
    state.recordToken(token.value, token.sema,
                      TokenRef{producer, token.ref.sema, token.ref.owner});
}
static const RenderState::Token *
regionExitToken(const RegionFlow &flow, unsigned branch, RenderState &state,
                const RenderState::Token *passThrough) {
  Node *final = flow.exits[branch];
  if (!final)
    return passThrough;
  return state.tokenForSource(final);
}
static LogicalResult renderPlainLoop(EmitCtx &ctx, GroupDag &g, Node *node,
                                     RenderState &state,
                                     const std::optional<RenderState::Token> &incoming) {
  RenderState body = state.nested();
  if (incoming)
    body.recordToken(incoming->value, incoming->sema,
                     TokenRef{node, incoming->ref.sema,
                              incoming->ref.owner});
  if (failed(renderChain(ctx, g, node->children[0], body)))
    return failure();
  if (incoming)
    state.recordToken(incoming->value, incoming->sema,
                      TokenRef{node, incoming->ref.sema,
                               incoming->ref.owner});
  state.clearViews();
  return success();
}
static LogicalResult
renderCarriedLoop(EmitCtx &ctx, GroupDag &g, Node *node, scf::ForOp forOp,
                  RenderState &state, const RenderState::Token &incoming,
                  const TokenRef &resultRef) {
  unsigned index = slotIndexFor(ctx, node->op, &g);
  forOp.getInitsMutable()[index].assign(incoming.value);
  RenderState body = state.nested();
  RenderState::Token carrier{forOp.getRegionIterArg(index),
                             g.semas[resultRef.sema].create, resultRef};
  body.recordToken(carrier.value, carrier.sema, carrier.ref);
  recordProducerAlias(body, node->tokenSource, node, carrier);
  if (failed(renderChain(ctx, g, node->children[0], body)))
    return failure();
  const RegionFlow &flow = *node->flow;
  const RenderState::Token *bodyToken =
      regionExitToken(flow, 0, body,
                      body.tokenForSource(resultRef.producer));
  if (!bodyToken)
    return semaError(node->op) << "loop body exports no exact carried token";
  cast<scf::YieldOp>(forOp.getBody()->getTerminator())
      .setOperand(index, bodyToken->value);
  RenderState::Token result{forOp.getResult(index),
                            g.semas[resultRef.sema].create, resultRef};
  state.recordToken(result.value, result.sema, result.ref);
  recordProducerAlias(state, node->tokenSource, node, result);
  recordProducerAlias(state, flow.exits.front(), node, result);
  state.clearViews();
  return success();
}

static LogicalResult renderRegion(EmitCtx &ctx, GroupDag &g, Node *n,
                                  RenderState &rs) {
  std::optional<RenderState::Token> incoming;
  if (n->tokenSource) {
    const RenderState::Token *source = rs.tokenForSource(n->tokenSource);
    if (!source)
      return semaError(n->op)
             << "region cannot resolve its incoming token producer";
    incoming = *source;
  }
  TokenRef resultRef;
  if (n->flow) {
    bool needsIncoming = n->kind == Node::For ||
                         llvm::is_contained(n->flow->exits, nullptr);
    if (incoming && incoming->ref.owner &&
        !sameOwner(incoming->ref.owner, n->flow->owner))
      return semaError(n->op)
             << "threaded region input belongs to another partition";
    if (needsIncoming && !incoming)
      return semaError(n->op)
             << "pass-through region has no exact incoming token producer";
    std::optional<SemaId> channel =
        incoming ? std::optional<SemaId>(incoming->ref.sema) : n->flow->sema;
    if (!channel)
      return semaError(n->op)
             << "region has no statically selected semaphore channel";
    resultRef = TokenRef{n, *channel, n->flow->owner};
  }
  if (!n->requiredParts.empty() && gpu::hasPartition(n->op)) {
    SetVector<int> set = gpu::getPartitionIds(n->op);
    unsigned before = set.size();
    for (int p : n->requiredParts)
      set.insert(p);
    if (set.size() != before)
      gpu::setPartition(n->op, set.getArrayRef());
  }
  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    if (!n->flow)
      return renderPlainLoop(ctx, g, n, rs, incoming);
    return renderCarriedLoop(ctx, g, n, forOp, rs, *incoming, resultRef);
  }
  auto ifOp = cast<scf::IfOp>(n->op);
  if (!ifOp.elseBlock()) {
    Block &block = ifOp.getElseRegion().emplaceBlock();
    OpBuilder b = OpBuilder::atBlockEnd(&block);
    scf::YieldOp::create(b, ifOp.getLoc());
  }
  RenderState thenSt = rs.nested(), elseSt = rs.nested();
  if (incoming) {
    TokenRef boundary{n, incoming->ref.sema, incoming->ref.owner};
    thenSt.recordToken(incoming->value, incoming->sema, boundary);
    elseSt.recordToken(incoming->value, incoming->sema, boundary);
  }
  if (failed(renderChain(ctx, g, n->children[0], thenSt)))
    return failure();
  if (failed(renderChain(ctx, g, n->children[1], elseSt)))
    return failure();
  if (n->flow) {
    const RegionFlow &c = *n->flow;
    unsigned idx = slotIndexFor(ctx, n->op, &g);
    const RenderState::Token *passThrough = incoming ? &*incoming : nullptr;
    const RenderState::Token *thenToken =
        regionExitToken(c, 0, thenSt, passThrough);
    const RenderState::Token *elseToken =
        regionExitToken(c, 1, elseSt, passThrough);
    if (!thenToken || !elseToken)
      return semaError(n->op) << "if path exports no exact compatible token";
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    thenYield->setOperand(idx, thenToken->value);
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    elseYield->setOperand(idx, elseToken->value);
    RenderState::Token result{ifOp.getResult(idx),
                              g.semas[resultRef.sema].create, resultRef};
    rs.recordToken(result.value, result.sema, result.ref);
    for (Node *final : c.exits)
      recordProducerAlias(rs, final, n, result);
  }
  rs.clearViews();
  return success();
}

static LogicalResult renderChain(EmitCtx &ctx, GroupDag &g, Node *head,
                                 RenderState &rs) {
  Operation *lastReal = nullptr;
  for (Node *n = head; n; n = n->next) {
    const RenderState::Token *source = nullptr;
    if (n->kind == Node::Release || n->kind == Node::Access) {
      source = n->tokenSource ? rs.tokenForSource(n->tokenSource) : nullptr;
      if (!source)
        return semaError(n->op ? n->op : g.root->op)
               << "buffer consumer cannot resolve its exact token producer";
      if (source->ref.owner &&
          !sameOwner(n->owner, source->ref.owner))
        return semaError(n->op ? n->op : g.root->op)
               << "buffer consumer token belongs to another partition";
    }
    switch (n->kind) {
    case Node::Enter:
    case Node::Exit:
      break; // markers; yield wiring is the parent's job
    case Node::Acquire: {
      const Sema &sema = g.semas[n->sema];
      assert(n->producedTokenOwner && "acquire token owner must be sealed");
      Operation *before = nextRealOp(n->next);
      OpBuilder b(ctx.func);
      if (before) {
        b.setInsertionPoint(before);
      } else if (lastReal && !isa<triton::FuncOp>(lastReal->getBlock()->getParentOp())) {
        b.setInsertionPoint(lastReal->getBlock()->getTerminator());
      } else if (lastReal) {
        b.setInsertionPointAfter(lastReal);
      } else if (Block *block = chainBlock(n)) {
        b.setInsertionPoint(block->getTerminator());
      } else {
        return semaError(ctx.func)
               << "acquire has no exact containing block";
      }
      auto acq = emitInto<nvws::SemaphoreAcquireOp>(
          b, before ? before->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, sema.create, ctx.tokenType);
      if (n->stageOffset)
        acq.setStage(materializeI32Before(acq, *n->stageOffset));
      rs.recordToken(acq.getToken(), sema.create,
                     TokenRef{n, n->sema, *n->producedTokenOwner});
      rs.clearViews();
      lastReal = acq;
      break;
    }
    case Node::Release: {
      Value tok = source->value;
      rs.releasedSources.insert(n->tokenSource);
      OpBuilder b(ctx.func);
      if (lastReal)
        b.setInsertionPointAfter(lastReal);
      else if (Block *block = chainBlock(n))
        b.setInsertionPointToStart(block);
      else
        return semaError(ctx.func)
               << "release has no exact containing block";
      auto rel = emitInto<nvws::SemaphoreReleaseOp>(
          b, lastReal ? lastReal->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, g.semas[n->sema].create, tok,
          asyncOpsAttr(b.getContext(), n));
      if (n->stageOffset)
        rel.setStage(materializeI32Before(rel, *n->stageOffset));
      rel.setArriveCountAttr(b.getI32IntegerAttr(n->count));
      lastReal = rel;
      break;
    }
    case Node::Access: {
      lastReal = renderAccess(ctx, g, n, rs, *source);
      break;
    }
    case Node::For:
    case Node::If:
      if (failed(renderRegion(ctx, g, n, rs)))
        return failure();
      lastReal = n->op;
      break;
    case Node::Func:
      break;
    }
  }
  return success();
}
static Value materializeI32Before(Operation *op, int64_t value) {
  OpBuilder b(op);
  auto cst = emitInto<arith::ConstantOp>(b, op->getLoc(), resolveOwner(op),
                                         gpu::getStageCluster(op), b.getI32IntegerAttr(value));
  return cst.getResult();
}
static LogicalResult verifyTokenLocality(triton::FuncOp func) {
  auto idsOf = [](Operation *op) -> std::optional<SmallVector<int, 2>> {
    if (!gpu::hasPartition(op))
      return std::nullopt;
    auto set = gpu::getPartitionIds(op);
    SmallVector<int, 2> v(set.begin(), set.end());
    llvm::sort(v);
    return v;
  };
  auto trace = [&](Operation *consumer, Value root) -> LogicalResult {
    DenseSet<Value> seen;
    SmallVector<Value, 8> pending{root};
    auto appendLoopInputs = [&](scf::ForOp loop, unsigned index) {
      pending.push_back(loop.getBody()->getTerminator()->getOperand(index));
      pending.push_back(loop.getInits()[index]);
    };
    while (!pending.empty()) {
      Value token = pending.pop_back_val();
      if (!seen.insert(token).second)
        continue;
      if (auto arg = dyn_cast<BlockArgument>(token)) {
        auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
        if (loop)
          appendLoopInputs(loop, arg.getArgNumber() - 1);
        continue; // Other block arguments are outside this verifier's scope.
      }
      Operation *def = token.getDefiningOp();
      if (!def || isa<ub::PoisonOp>(def))
        continue;
      if (auto acquire = dyn_cast<nvws::SemaphoreAcquireOp>(def)) {
        auto acquireParts = idsOf(acquire);
        if (!acquireParts)
          continue; // Root entry seed exemption.
        auto consumerParts = idsOf(consumer);
        if (consumerParts && *acquireParts != *consumerParts) {
          InFlightDiagnostic diag = semaError(consumer)
              << "token-locality violation: token acquired in partition set "
                 "differs from consumer's";
          diag.attachNote(acquire.getLoc()) << "acquired here";
          return failure();
        }
        continue;
      }
      unsigned index = cast<OpResult>(token).getResultNumber();
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        pending.push_back(ifOp.elseYield().getOperand(index));
        pending.push_back(ifOp.thenYield().getOperand(index));
      } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        appendLoopInputs(forOp, index);
      }
    }
    return success();
  };
  LogicalResult result = success();
  auto traceConsumer = [&](Operation *op, Value token) {
    if (failed(trace(op, token)))
      result = failure();
  };
  func.walk([&](nvws::SemaphoreReleaseOp release) {
    traceConsumer(release.getOperation(), release.getToken());
  });
  func.walk([&](nvws::SemaphoreBufferOp buffer) {
    traceConsumer(buffer.getOperation(), buffer.getToken());
    auto bp = idsOf(buffer.getOperation());
    if (!bp)
      return;
    for (Value view : buffer->getResults())
      for (Operation *user : view.getUsers()) {
        auto up = idsOf(user);
        if (up && *up != *bp) {
          InFlightDiagnostic diag = semaError(user)
              << "view-locality violation: view consumed outside its partition";
          diag.attachNote(buffer.getLoc()) << "view materialized here";
          result = failure();
        }
      }
  });
  return result;
}
static nvws::SemaphoreAcquireOp resolveAcquireThroughIfs(Value v) {
  for (int fuel = 0; fuel < 8; ++fuel) {
    if (auto acq = v.getDefiningOp<nvws::SemaphoreAcquireOp>())
      return acq;
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return nullptr;
    unsigned idx = cast<OpResult>(v).getResultNumber();
    Value next = ifOp.thenYield()->getOperand(idx);
    for (Value yielded : {next, ifOp.elseYield()->getOperand(idx)})
      if (auto acquire =
              yielded.getDefiningOp<nvws::SemaphoreAcquireOp>())
        return acquire;
    v = next;
  }
  return nullptr;
}
static LogicalResult
verifyEmittedIR(triton::FuncOp func,
                const DenseSet<Operation *> &exactReuseBufferOps,
                ArrayRef<EmitCtx::CachedReuseContract> cachedReuseContracts) {
  for (const EmitCtx::CachedReuseContract &contract : cachedReuseContracts) {
    auto buffer = contract.view.getDefiningOp<nvws::SemaphoreBufferOp>();
    if (!buffer || buffer.getToken() != contract.token)
      return semaError(func)
             << "emitter contract: malformed exact cached-view reuse";
    Operation *bufferOp = buffer.getOperation();
    bool witnessed = false;
    for (Operation *tokenUser : contract.token.getUsers()) {
      if (!isa<nvws::SemaphoreReleaseOp>(tokenUser) ||
          tokenUser->getBlock() != bufferOp->getBlock() ||
          !bufferOp->isBeforeInBlock(tokenUser))
        continue;
      witnessed |=
          llvm::any_of(contract.view.getUsers(), [&](Operation *viewUser) {
            return viewUser->getBlock() == tokenUser->getBlock() &&
                   tokenUser->isBeforeInBlock(viewUser);
          });
    }
    if (!witnessed)
      return semaError(buffer)
             << "emitter contract: exact cached view has no use after its release";
  }
  auto checkPartitionOutputs = [](Operation *op) -> LogicalResult {
    if (!isa<scf::ForOp, scf::IfOp>(op) ||
        !op->hasAttr(gpu::kPartitionOutputsAttrName))
      return success();
    auto outputs = gpu::getPartitionOutputs(op);
    if (outputs.size() != op->getNumResults())
      return semaError(op) << "partition-outputs verifier: attribute has "
                           << outputs.size() << " entries for "
                           << op->getNumResults() << " results";
    SmallVector<Operation *, 2> terms;
    for (Region &region : op->getRegions())
      if (!region.empty())
        terms.push_back(region.front().getTerminator());
    for (auto [i, output] : llvm::enumerate(outputs))
      for (Operation *term : terms) {
        Operation *def = term->getOperand(i).getDefiningOp();
        if (!def || isa<ub::PoisonOp>(def) ||
            def->hasTrait<OpTrait::ConstantLike>() || !gpu::hasPartition(def))
          continue;
        SetVector<int> producer = gpu::getPartitionIds(def);
        if (producer.empty() || llvm::any_of(producer, [&](int p) {
              return output.contains(p);
            }))
          continue;
        std::string have;
        llvm::raw_string_ostream os(have);
        for (int p : producer)
          os << p << " ";
        return semaError(op) << "partition-outputs verifier: result " << i
                             << " is produced by partition(s) " << os.str()
                             << "but ttg.partition.outputs names none of them";
      }
    return success();
  };
  auto partitionResult = func.walk([&](Operation *op) -> WalkResult {
    if (failed(checkPartitionOutputs(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (partitionResult.wasInterrupted() || failed(verifyTokenLocality(func)))
    return failure();

  auto checkToken = [&](Value token) -> LogicalResult {
    llvm::SmallDenseMap<Block *, SmallVector<Operation *, 4>> usersByBlock;
    for (Operation *user : token.getUsers())
      usersByBlock[user->getBlock()].push_back(user);
    for (auto &[block, users] : usersByBlock) {
      (void)block;
      llvm::sort(users, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      bool released = false;
      for (Operation *user : users) {
        released |= isa<nvws::SemaphoreReleaseOp>(user);
        if (released && isa<nvws::SemaphoreBufferOp>(user) &&
            !exactReuseBufferOps.contains(user))
          return semaError(user)
                 << "token has a buffer view after its release "
                    "(use-after-release; spec fable/semas-report3.md "
                    "Addendum B.3(b))";
      }
    }
    return success();
  };
  auto checkLoop = [&](scf::ForOp forOp) -> LogicalResult {
    llvm::SmallDenseMap<Value, unsigned> slotsPerBacking;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      if (!isa<gpu::AsyncTokenType>(arg.getType()))
        continue;
      if (failed(checkToken(arg)))
        return failure();
      unsigned idx = arg.getArgNumber() - 1; // skip induction variable
      nvws::SemaphoreAcquireOp acq =
          resolveAcquireThroughIfs(yieldOp.getOperand(idx));
      if (!acq)
        continue;
      auto create = acq.getSemaphore().getDefiningOp<nvws::SemaphoreCreateOp>();
      if (!create || create.getBuffers().empty())
        continue;
      Value backing = create.getBuffers().front();
      if (auto alloc = backing.getDefiningOp<gpu::LocalAllocOp>();
          alloc && alloc->hasAttr(kBufferCircularAttrName))
        continue;
      if (++slotsPerBacking[backing] > 1)
        return semaError(forOp)
               << "two token slots for one semaphore group in a single loop "
                  "(spec fable/semas-report3.md Addendum B.3(a)); "
                  "AssignStagePhase cannot thread this";
    }
    return success();
  };
  auto result = func.walk([&](Operation *op) -> WalkResult {
    if (auto acquire = dyn_cast<nvws::SemaphoreAcquireOp>(op);
        acquire && failed(checkToken(acquire.getToken())))
      return WalkResult::interrupt();
    if (auto forOp = dyn_cast<scf::ForOp>(op);
        forOp && failed(checkLoop(forOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult emitIR(triton::FuncOp funcOp, MutableArrayRef<GroupDag> groups) {
  EmitCtx ctx;
  ctx.func = funcOp;
  ctx.tokenType = gpu::AsyncTokenType::get(funcOp.getContext());
  {
    OpBuilder b(&funcOp.getBody().front(), funcOp.getBody().front().begin());
    ctx.poison = ub::PoisonOp::create(b, funcOp.getLoc(), ctx.tokenType).getResult();
  }
  SmallVector<GroupDag *> activeGroups;
  for (GroupDag &g : groups)
    if (!g.semas.empty())
      activeGroups.push_back(&g);
  for (GroupDag *group : activeGroups)
    nukeGroupTokens(ctx, *group);
  while (eraseDeadTokenSlots(ctx, groups)) {
  }
  if (failed(emitPhysicalIR(ctx, activeGroups)))
    return failure();
  rewriteSignatures(ctx, groups);
  for (GroupDag *group : activeGroups) {
    RenderState rs;
    if (failed(renderChain(ctx, *group, group->root->children[0], rs)))
      return failure();
  }
  SmallVector<Operation *> aliasOps;
  ctx.func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isSupportedAliasOp(op))
      aliasOps.push_back(op);
  });
  for (Operation *op : llvm::reverse(aliasOps))
    if (llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); }))
      op->erase();
  for (GroupDag *group : activeGroups)
    for (const Member &m : group->pieceTable.members)
      if (m.allocOp && m.allocOp->getBlock() && m.allocOp->use_empty())
        m.allocOp->erase();
  // Managed buffer dependencies have all been rebuilt from exact semaphore
  // tokens.  A poison can remain only on detached legacy token plumbing whose
  // result is consumed by an unrelated opaque operation; keep the valid IR as
  // the pre-refactor emitter did.  Structural dead slots are removed above.
  if (ctx.poison.use_empty())
    ctx.poison.getDefiningOp()->erase();
  return verifyEmittedIR(funcOp, ctx.exactReuseBufferOps,
                         ctx.cachedReuseContracts);
}
} // namespace mlir::triton::nvws_semas
