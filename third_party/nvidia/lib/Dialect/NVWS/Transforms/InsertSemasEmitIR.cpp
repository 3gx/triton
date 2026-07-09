// Protocol materialization; see sema-docs/insert-semas/emit-ir.md.
#include "InsertSemas.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir::triton::nvws_semas {

struct EmitCtx {
  triton::FuncOp func;
  Value poison; // the single function-level ub.poison token (contract E)
  Type tokenType;
  struct Slot {
    GroupDag *g;
    unsigned index; // absolute result / iter_arg index in the NEW op
  };
  llvm::MapVector<Operation *, SmallVector<Slot, 2>> slots;
  DenseSet<Operation *> reusedTokenBufferOps;
  DenseMap<Operation *, unsigned> remainingIfRenders;
};

struct RenderState {
  struct Token {
    Value value;
    Value sema;
    TokenRef ref;
  };
  // One exact capability per token known in this chain. The last record is
  // used by default; owner-marked nodes may instead use their owner's record.
  SmallVector<Token, 2> tokens;
  DenseMap<MemberId, std::pair<Value, int64_t>> view; // member view + owner
  void clearViews() { view.clear(); }
  const Token *lastToken() const {
    return tokens.empty() ? nullptr : &tokens.back();
  }
  Token *tokenForOwner(const Owner &owner) {
    for (Token &token : tokens)
      if (sameOwner(token.ref.owner, owner) && token.value && token.sema)
        return &token;
    return nullptr;
  }
  const Token *tokenFor(const TokenRef &ref) const {
    for (const Token &token : tokens)
      if (token.ref.producer == ref.producer && token.ref.sema == ref.sema &&
          token.value && token.sema)
        return &token;
    return nullptr;
  }
  const Token *tokenForNode(const Node *node, const Owner &owner) {
    const Token *active = lastToken();
    if (nodeReusesToken(node, owner) &&
        (!active || !sameOwner(active->ref.owner, owner)))
      return tokenForOwner(owner);
    return active;
  }
  Token *tokenFor(const TokenRef &ref) {
    return const_cast<Token *>(std::as_const(*this).tokenFor(ref));
  }
  Token *tokenForProducer(Node *producer, const Owner &owner) {
    // Producer selects phase/slot lineage. The region result separately names
    // its static render channel, so the source acquire's semaphore may differ.
    for (Token &token : tokens)
      if (token.ref.producer == producer && sameOwner(token.ref.owner, owner))
        return &token;
    return nullptr;
  }
  void recordToken(Value value, Value sema, const TokenRef &ref) {
    for (auto it = tokens.begin(); it != tokens.end();) {
      if (!it->ref.owner || sameOwner(it->ref.owner, ref.owner))
        it = tokens.erase(it);
      else
        ++it;
    }
    tokens.push_back(Token{value, sema, ref});
  }
  void keepOnly(const Token &token) {
    Token copy = token;
    tokens.clear();
    tokens.push_back(copy);
  }
  void clearTokens() {
    tokens.clear();
    clearViews();
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
  auto op = gpu::createInto<OpT>(b, loc, ids, stageCluster,
                                 std::forward<Args>(args)...);
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
    if ((m->kind == Node::Access || m->kind == Node::For ||
         m->kind == Node::If) &&
        m->op)
      return m->op;
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
    if (auto l = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      l.getDepMutable().clear();
      if (l.getToken())
        l.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
    if (auto st = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
      st.getDepMutable().clear();
      if (st.getToken())
        st.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
    if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (g.aliases.count(mma.getAccumulator())) {
        mma.getAccDepMutable().clear();
        if (mma.getToken())
          mma.getToken().replaceAllUsesWith(ctx.poison);
      }
      return;
    }
    if (auto a = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      if (a.getToken())
        a.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
  };
  for (const Member &m : g.pieceTable.members)
    nukeOp(m.allocOp);
  forEachNode(g, [&](Node *n) {
    if (n->kind == Node::Access)
      nukeOp(n->op);
  });
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
                               t.getEncoding(), t.getMemorySpace(), m,
                               t.getAllocShape());
}
static gpu::MemDescType genericViewType(gpu::MemDescType backing) {
  auto shape = backing.getShape();
  return gpu::MemDescType::get(
      isScalesEnc(backing) ? shape : shape.drop_front(),
      backing.getElementType(), backing.getEncoding(), backing.getMemorySpace(),
      /*mutableMemory=*/true, backing.getShape());
}
static bool sameViewType(Type a, Type b) {
  auto x = dyn_cast<gpu::MemDescType>(a), y = dyn_cast<gpu::MemDescType>(b);
  if (!x || !y)
    return false;
  return x.getShape() == y.getShape() &&
         x.getElementType() == y.getElementType() &&
         x.getEncoding() == y.getEncoding() &&
         x.getMemorySpace() == y.getMemorySpace() &&
         x.getMutableMemory() == y.getMutableMemory();
}
static gpu::MemDescType localViewType(const GroupDag &g, MemberId member,
                                      ArrayRef<const Touch *> touches,
                                      gpu::MemDescType backing) {
  for (const Touch *t : touches) {
    if (t->member != member)
      continue;
    Type ty = g.pieceTable.members[member].type;
    if (t->alias.empty())
      ty = t->accessType;
    for (const AliasStep &step : t->alias) {
      if (step.op->getName().getStringRef() != "ttg.memdesc_index")
        break;
      ty = step.resultType;
    }
    if (auto at = dyn_cast<gpu::MemDescType>(ty))
      return withMutable(at, true);
  }
  return withMutable(genericViewType(backing), true);
}

static Operation *backingAnchor(GroupDag &g) {
  Operation *anchor = g.pieceTable.members.front().allocOp;
  for (const Member &m : g.pieceTable.members)
    if (m.allocOp->getBlock() == anchor->getBlock() &&
        m.allocOp->isBeforeInBlock(anchor))
      anchor = m.allocOp;
  while (isa<scf::ForOp>(anchor->getParentOp()))
    anchor = anchor->getParentOp();
  return anchor;
}
static Value emitBacking(OpBuilder &b, Location loc, GroupDag &g,
                         const Member &member) {
  auto type = backingType(g, member);
  Value backing =
      g.isTmem()
          ? nvidia_gpu::TMEMAllocOp::create(b, loc, type, Value()).getResult()
          : gpu::LocalAllocOp::create(b, loc, type).getResult();
  for (StringRef name :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName,
        kBufferCircularAttrName, kBufferStartAttrName})
    if (Attribute attr = member.allocOp->getAttr(name))
      backing.getDefiningOp()->setAttr(name, attr);
  return backing;
}

static bool sharesCover(const GroupDag &g, unsigned cover, unsigned member) {
  if (member == cover || g.pieceTable.members.size() < 2 ||
      (!g.isTmem() && llvm::none_of(g.pieceTable.members, [](const Member &m) {
        return m.allocOp->hasAttr(kBufferCopyAttrName);
      })))
    return false;
  const Member &owner = g.pieceTable.members[cover];
  const Member &reuser = g.pieceTable.members[member];
  if (reuser.offset < owner.offset ||
      reuser.offset + reuser.extent > owner.offset + owner.extent)
    return false;
  return g.isTmem() || (reuser.offset == owner.offset &&
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

static FailureOr<Value> emitMixedDepthView(OpBuilder &b, Location loc,
                                           GroupDag &owner, GroupDag &reuser,
                                           Value backing) {
  auto ownerType = backingType(owner, owner.pieceTable.members.front());
  auto reuserType = backingType(reuser, reuser.pieceTable.members.front());
  auto ownerShape = ownerType.getShape(), reuserShape = reuserType.getShape();
  Operation *errorOp = backingAnchor(reuser);
  if (ownerShape.empty() || reuserShape.empty())
    return semaError(errorOp) << "mixed-depth TMEM backing has empty shape";
  int64_t offset = reuser.pieceTable.members.front().offset -
                   owner.pieceTable.members.front().offset;
  int64_t ownerN = ownerShape.back(), reuserN = reuserShape.back();
  if (ownerN < reuserN || ownerN % reuserN || offset < 0 ||
      offset + reuserN > ownerN)
    return semaError(errorOp)
           << "mixed-depth TMEM reuser is outside its physical owner";
  unsigned ownerWidth = ownerType.getElementTypeBitWidth();
  unsigned reuserWidth = reuserType.getElementTypeBitWidth();
  if (ownerWidth != reuserWidth && ownerWidth != 2 * reuserWidth)
    return semaError(errorOp)
           << "unsupported mixed-depth TMEM element-width reinterpretation";
  int64_t size = ownerWidth == reuserWidth ? reuserN : reuserN / 2;
  if (size <= 0)
    return semaError(errorOp) << "invalid mixed-depth TMEM subslice width";
  return emitTmemView(b, loc, backing, reuserType, offset, size,
                      /*reinterpret=*/true);
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
        g.backing[j] = members[j].offset == member.offset &&
                               target == g.backing[i].getType()
                           ? g.backing[i]
                           : emitTmemView(b, loc, g.backing[i], target,
                                          members[j].offset - member.offset,
                                          target.getShape().back());
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
  GroupDag &owner = *set[!firstOwns], &reuser = *set[firstOwns];
  Operation *ownerAnchor = backingAnchor(owner);
  Operation *reuserAnchor = backingAnchor(reuser);
  if (ownerAnchor->getBlock() != reuserAnchor->getBlock() ||
      (ownerAnchor != reuserAnchor ? !ownerAnchor->isBeforeInBlock(reuserAnchor)
                                   : !firstOwns))
    return semaError(reuserAnchor)
           << "mixed-depth TMEM physical owner does not dominate its reuser";
  OpBuilder b(ownerAnchor);
  Value backing = emitBacking(b, ownerAnchor->getLoc(), owner,
                              owner.pieceTable.members.front());
  FailureOr<Value> view =
      emitMixedDepthView(b, reuserAnchor->getLoc(), owner, reuser, backing);
  if (failed(view))
    return failure();
  owner.backing.assign(1, backing);
  reuser.backing.assign(1, *view);
  return success();
}

static LogicalResult materializeCircular(ArrayRef<GroupDag *> set) {
  if (set.size() == 1) {
    materializeLogicalBacking(*set.front());
    return success();
  }
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

static LogicalResult emitPhysicalIR(EmitCtx &ctx,
                                    MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 2>> mixed, circular;
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    if (g.mixedDepthPhysicalAlias)
      mixed[g.bufferId].push_back(&g);
    else if (g.isCircular())
      circular[g.bufferId].push_back(&g);
  }
  std::map<std::pair<int64_t, bool>, Sema *> circularPrimary;
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    if (g.backing.empty())
      if (g.mixedDepthPhysicalAlias) {
        if (failed(materializeMixedDepth(mixed[g.bufferId])))
          return failure();
      } else if (g.isCircular()) {
        if (failed(materializeCircular(circular[g.bufferId])))
          return failure();
      } else {
        materializeLogicalBacking(g);
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
        if (s.isEntry != entry)
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
            b, anchor->getLoc(), semaTy, g.backing, s.isEntry);
        create.setPendingCountAttr(b.getI32IntegerAttr(s.count));
        s.create = create.getResult();
      }
  }
  return success();
}
static void emitEntryAcquires(EmitCtx &ctx, GroupDag &g,
                              DenseMap<Node *, Value> &emitted) {
  forEachNode(g, [&](Node *n) {
    if (n->kind != Node::Acquire || n->owner || !getSema(g, n).isEntry ||
        emitted.count(n))
      return;
    Operation *before = nextRealOp(n->next);
    OpBuilder b(ctx.func);
    if (before)
      b.setInsertionPoint(before);
    else
      b.setInsertionPoint(
          n->parent && n->parent->op
              ? n->parent->op->getRegion(0).front().getTerminator()
              : &ctx.func.getBody().front().back());
    const Sema &s = getSema(g, n);
    auto acq = emitInto<nvws::SemaphoreAcquireOp>(
        b, before ? before->getLoc() : ctx.func.getLoc(), Owner(), {}, s.create,
        ctx.tokenType);
    if (n->stageOffset)
      acq.setStage(materializeI32Before(acq, *n->stageOffset));
    emitted[n] = acq.getToken();
  });
}
static void fixupAnchors(MutableArrayRef<GroupDag> groups, Operation *oldOp,
                         Operation *newOp) {
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->op == oldOp)
        n->op = newOp;
    });
  }
}
static void filterPartitionOutputs(Operation *op, ArrayRef<bool> keep) {
  auto attr = op->getAttrOfType<ArrayAttr>(gpu::kPartitionOutputsAttrName);
  if (!attr || attr.size() != keep.size())
    return;
  SmallVector<Attribute> kept;
  for (auto [a, k] : llvm::zip(attr.getValue(), keep))
    if (k)
      kept.push_back(a);
  op->setAttr(gpu::kPartitionOutputsAttrName,
              ArrayAttr::get(op->getContext(), kept));
}
static void finishResultFilter(Operation *oldOp, Operation *newOp,
                               ArrayRef<bool> keep,
                               MutableArrayRef<GroupDag> groups) {
  filterPartitionOutputs(newOp, keep);
  unsigned next = 0;
  for (auto [i, result] : llvm::enumerate(oldOp->getResults()))
    if (keep[i])
      result.replaceAllUsesWith(newOp->getResult(next++));
  fixupAnchors(groups, oldOp, newOp);
  oldOp->erase();
}
static void eraseDroppedYields(Operation *op, ArrayRef<bool> keep) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    Operation *yield = region.front().getTerminator();
    for (int i = keep.size() - 1; i >= 0; --i)
      if (!keep[i])
        yield->eraseOperand(i);
  }
}

static bool eraseDeadTokenSlots(EmitCtx &ctx,
                                MutableArrayRef<GroupDag> groups) {
  bool changed = false;
  SmallVector<scf::ForOp> fors;
  SmallVector<scf::IfOp> ifs;
  ctx.func.walk([&](Operation *op) {
    if (auto f = dyn_cast<scf::ForOp>(op))
      fors.push_back(f);
    else if (auto i = dyn_cast<scf::IfOp>(op))
      ifs.push_back(i);
  });
  for (scf::IfOp ifOp : ifs) {
    SmallVector<bool> keep(ifOp.getNumResults(), true);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (res.getType() == ctx.tokenType && res.use_empty()) {
        keep[i] = false;
        any = true;
      }
    if (!any)
      continue;
    SmallVector<Type> keptTypes;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (keep[i])
        keptTypes.push_back(res.getType());
    OpBuilder b(ifOp);
    auto newIf =
        scf::IfOp::create(b, ifOp.getLoc(), keptTypes, ifOp.getCondition(),
                          /*withElseRegion=*/!ifOp.getElseRegion().empty());
    newIf->setAttrs(ifOp->getAttrs());
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    eraseDroppedYields(newIf, keep);
    finishResultFilter(ifOp, newIf, keep, groups);
    changed = true;
  }
  for (scf::ForOp forOp : fors) {
    SmallVector<bool> keep(forOp.getNumResults(), true);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(forOp.getResults()))
      if (res.getType() == ctx.tokenType && res.use_empty() &&
          forOp.getRegionIterArg(i).use_empty()) {
        keep[i] = false;
        any = true;
      }
    if (!any)
      continue;
    SmallVector<Value> keptInits;
    for (auto [i, init] : llvm::enumerate(forOp.getInits()))
      if (keep[i])
        keptInits.push_back(init);
    OpBuilder b(forOp);
    auto newFor =
        scf::ForOp::create(b, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), keptInits);
    newFor->setAttrs(forOp->getAttrs());
    newFor.getRegion().takeBody(forOp.getRegion());
    Block &body = newFor.getRegion().front();
    for (int i = keep.size() - 1; i >= 0; --i)
      if (!keep[i]) {
        body.getTerminator()->eraseOperand(i);
        body.eraseArgument(1 + i); // +1: induction variable
      }
    finishResultFilter(forOp, newFor, keep, groups);
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
        if (yield.getNumOperands() < newIf.getNumResults()) {
          SmallVector<Value> operands(yield.getOperands());
          operands.append(newIf.getNumResults() - yield.getNumOperands(),
                          ctx.poison);
          OpBuilder yb(yield);
          auto ny = scf::YieldOp::create(yb, yield.getLoc(), operands);
          ny->setAttrs(yield->getAttrs());
          yield.erase();
        }
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
    if (gpu::hasPartition(newOp))
      gpu::setPartitionOutputs(newOp, outputs);
    if (gpu::hasPartition(newOp)) {
      auto ids = gpu::getPartitionIds(newOp);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r)
          if (Operation *term = blk.getTerminator())
            if (!gpu::hasPartition(term))
              term->setAttr(gpu::kPartitionAttrName,
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
  if (op->getNumResults() == 1)
    if (auto resultType =
            dyn_cast<gpu::MemDescType>(op->getResult(0).getType()))
      op->getResult(0).setType(withMutable(
          resultType,
          cast<gpu::MemDescType>(source.getType()).getMutableMemory()));
}
static Value getView(EmitCtx &ctx, GroupDag &g, RenderState &rs, Node *node,
                     const Touch &t, Operation *accessOp, const Owner &owner) {
  auto it = rs.view.find(t.member);
  Value base;
  int64_t viewOwner = ownerKey(owner);
  if (it != rs.view.end() && it->second.second == viewOwner) {
    base = it->second.first;
  } else {
    OpBuilder b(accessOp);
    SmallVector<Type> types;
    for (auto [mi, m] : llvm::enumerate(g.pieceTable.members)) {
      auto bt = cast<gpu::MemDescType>(g.backing[mi].getType());
      if (g.isTmem())
        types.push_back(genericViewType(bt));
      else
        types.push_back(localViewType(g, static_cast<MemberId>(mi), {&t}, bt));
    }
    bool reusesToken = nodeReusesToken(node, owner);
    const RenderState::Token *token = rs.tokenForNode(node, owner);
    assert(token && "no capability for buffer view");
    Value tok = token->value;
    Value semaVal = token->sema;
    assert(tok && "no token for view");
    assert(semaVal && "no semaphore for token");
    auto buf = emitInto<nvws::SemaphoreBufferOp>(
        b, accessOp->getLoc(), owner, gpu::getStageCluster(accessOp), semaVal,
        TypeRange(types), tok);
    if (reusesToken)
      ctx.reusedTokenBufferOps.insert(buf.getOperation());
    if (node->bufferStageOffset)
      buf.setStage(materializeI32Before(buf, *node->bufferStageOffset));
    for (auto [mi, v] : llvm::enumerate(buf.getBuffers()))
      rs.view[static_cast<MemberId>(mi)] = {v, viewOwner};
    base = rs.view[t.member].first;
  }
  Value cur = base;
  OpBuilder b(accessOp);
  for (const AliasStep &step : t.alias) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" &&
        old->getNumResults() == 1 &&
        sameViewType(old->getResult(0).getType(), cur.getType()))
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
                                 RenderState &rs,
                                 DenseMap<Node *, Value> &emitted);
static void seedRegionEntry(RenderState &state, Node *head) {
  if (!head || head->pieceInfo.empty())
    return;
  std::optional<Owner> owner = uniformPieceOwner(head);
  if (!owner) {
    state.clearTokens();
    return;
  }
  if (owner->has_value() && state.lastToken() &&
      !state.lastToken()->ref.owner) {
    RenderState::Token adopted = *state.lastToken();
    adopted.ref.owner = *owner;
    state.keepOnly(adopted);
  } else if (RenderState::Token *token = state.tokenForOwner(*owner)) {
    state.keepOnly(*token);
  } else {
    state.clearTokens();
  }
}
static Operation *renderAccess(EmitCtx &ctx, GroupDag &g, Node *n,
                               RenderState &rs) {
  Operation *op = n->op;
  Operation *anchor = op;
  for (const Touch &t : n->touches) {
    Value view = getView(ctx, g, rs, n, t, op, n->owner);
    if (auto ta = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      OpBuilder b(op);
      auto pidsc = std::make_pair(n->owner, gpu::getStageCluster(op));
      auto vTrue = emitInto<arith::ConstantOp>(
          b, op->getLoc(), n->owner, pidsc.second, b.getBoolAttr(true));
      anchor = emitInto<nvidia_gpu::TMEMStoreOp>(b, op->getLoc(), n->owner,
                                                 pidsc.second, Type(), view,
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
        auto splat = emitInto<triton::SplatOp>(
            b, op->getLoc(), n->owner, gpu::getStageCluster(op),
            RankedTensorType::get(
                cast<gpu::MemDescType>(view.getType()).getShape(),
                src.getType()),
            src);
        src = splat.getResult();
      }
      anchor = emitInto<gpu::LocalStoreOp>(b, op->getLoc(), n->owner,
                                           gpu::getStageCluster(op), src, view);
      la.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               !g.accessNodeOps.contains(use.getOwner());
      });
      return anchor;
    }
    if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (mma.getAccumulator() == t.accessValue)
        mma.setAccumulator(view);
      for (OpOperand &o : op->getOpOperands())
        if (o.get() == t.accessValue && o.get() != mma.getAccumulator())
          o.set(view);
      continue;
    }
    for (OpOperand &o : op->getOpOperands())
      if (o.get() == t.accessValue)
        o.set(view);
  }
  if (n->completionAnchor)
    anchor = n->completionAnchor;
  return anchor;
}

// Materialize the scheduler-safe boundary shape as the final group renders
// this if, while its exact release/acquire and token slot are still local.
static Operation *finishIfRender(EmitCtx &ctx, scf::IfOp ifOp,
                                 RenderState &state) {
  bool thenPath = true, releaseOnly = false;
  unsigned slot = 0;
  nvws::SemaphoreReleaseOp release;
  nvws::SemaphoreAcquireOp acquire;
  auto match = [&](bool inThen, bool only) {
    if (!inThen && ifOp.getElseRegion().empty())
      return false;
    Block *block = inThen ? ifOp.thenBlock() : ifOp.elseBlock();
    nvws::SemaphoreReleaseOp candidate;
    for (Operation &op : *block) {
      if ((candidate = dyn_cast<nvws::SemaphoreReleaseOp>(&op)))
        break;
      if (isa<scf::YieldOp, nvws::SemaphoreAcquireOp>(op) ||
          (!op.hasTrait<OpTrait::ConstantLike>() && !isSupportedAliasOp(&op)))
        return false;
    }
    if (!candidate)
      return false;
    auto ty = dyn_cast<nvws::SemaphoreType>(candidate.getSemaphore().getType());
    auto base = ty && !ty.getBaseType().empty()
                    ? dyn_cast<gpu::MemDescType>(ty.getBaseType()[0])
                    : gpu::MemDescType();
    bool tmem =
        base && isa<nvidia_gpu::TensorMemorySpaceAttr>(base.getMemorySpace());
    auto trailing = dyn_cast_or_null<nvws::SemaphoreAcquireOp>(
        block->getTerminator()->getPrevNode());
    if (only) {
      bool hasLaterAcquire = false;
      for (Operation *op = candidate->getNextNode();
           op && !isa<scf::YieldOp>(op); op = op->getNextNode())
        hasLaterAcquire |= isa<nvws::SemaphoreAcquireOp>(op);
      if (!tmem || !hasLaterAcquire)
        return false;
    } else {
      if (!trailing || (tmem && ty.getBaseType().size() > 1))
        return false;
      auto yield = inThen ? ifOp.thenYield() : ifOp.elseYield();
      auto it = llvm::find(yield.getOperands(), trailing.getToken());
      if (it == yield.getOperands().end())
        return false;
      slot = std::distance(yield.getOperands().begin(), it);
    }
    thenPath = inThen;
    releaseOnly = only;
    release = candidate;
    acquire = trailing;
    return true;
  };
  if (!match(true, false) && !match(false, false) && !match(true, true) &&
      !match(false, true)) {
    acquire =
        dyn_cast_or_null<nvws::SemaphoreAcquireOp>(&ifOp.thenBlock()->front());
    Operation *prev = ifOp->getPrevNode();
    if (prev && ifOp.getCondition().getDefiningOp() == prev)
      prev = prev->getPrevNode();
    release = dyn_cast_or_null<nvws::SemaphoreReleaseOp>(prev);
    auto it =
        acquire ? llvm::find(ifOp.thenYield().getOperands(), acquire.getToken())
                : ifOp.thenYield().getOperands().end();
    if (!release || it == ifOp.thenYield().getOperands().end())
      return ifOp;
    slot = std::distance(ifOp.thenYield().getOperands().begin(), it);
  }
  OpBuilder b(ifOp);
  Location loc = ifOp.getLoc();
  auto exitIf = scf::IfOp::create(b, loc, TypeRange{}, ifOp.getCondition(),
                                  /*withElseRegion=*/!thenPath);
  Block *exitBlock = thenPath ? exitIf.thenBlock() : exitIf.elseBlock();
  release->moveBefore(exitBlock, exitBlock->begin());
  exitIf->setAttrs(ifOp->getAttrs());
  gpu::StageCluster releaseStage = gpu::getStageCluster(release);
  if (!releaseStage)
    for (Operation *op = ifOp->getPrevNode(); op; op = op->getPrevNode())
      if (isa<nvidia_gpu::MMAv5OpInterface>(op)) {
        releaseStage = gpu::getStageCluster(op);
        break;
      }
  if (releaseStage) {
    gpu::setStageCluster(b, release, releaseStage);
    gpu::setStageCluster(b, exitIf, releaseStage);
  }
  SetVector<int> ids = partitionIdsOfFwd(release);
  if (ids.empty())
    ids = partitionIdsOfFwd(ifOp);
  if (!ids.empty())
    gpu::setPartition(exitIf, ids.getArrayRef());
  gpu::setPartitionOutputs(exitIf, {});
  if (releaseOnly)
    return ifOp;

  b.setInsertionPointAfter(ifOp);
  auto enterIf = scf::IfOp::create(b, loc, TypeRange{ctx.tokenType},
                                   ifOp.getCondition(), true);
  Block *acquireBlock = thenPath ? enterIf.thenBlock() : enterIf.elseBlock();
  acquire->moveBefore(acquireBlock, acquireBlock->begin());
  Value oldResult = ifOp.getResult(slot);
  oldResult.replaceAllUsesWith(enterIf.getResult(0));
  b.setInsertionPointToEnd(enterIf.thenBlock());
  scf::YieldOp::create(b, loc,
                       thenPath ? acquire.getToken()
                                : ifOp.thenYield().getOperand(slot));
  b.setInsertionPointToEnd(enterIf.elseBlock());
  scf::YieldOp::create(b, loc,
                       thenPath ? ifOp.elseYield().getOperand(slot)
                                : acquire.getToken());
  b.setInsertionPoint(ifOp);
  Value poison = ub::PoisonOp::create(b, loc, ctx.tokenType).getResult();
  ifOp.thenYield().setOperand(slot, poison);
  ifOp.elseYield().setOperand(slot, poison);
  enterIf->setAttrs(ifOp->getAttrs());
  if (auto stage = gpu::getStageCluster(acquire))
    gpu::setStageCluster(b, enterIf, stage);
  ids = partitionIdsOfFwd(release);
  for (int p : partitionIdsOfFwd(acquire))
    ids.insert(p);
  if (!ids.empty()) {
    gpu::setPartition(exitIf, ids.getArrayRef());
    gpu::setPartition(enterIf, ids.getArrayRef());
    gpu::setPartitionOutputs(enterIf, SmallVector<SetVector<int>, 1>{ids});
  }
  SetVector<int> middleIds;
  for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()})
    if (!region->empty())
      for (Operation &op : region->front())
        if (!isa<scf::YieldOp>(op))
          for (int p : partitionIdsOfFwd(&op))
            middleIds.insert(p);
  for (auto [i, output] : llvm::enumerate(gpu::getPartitionOutputs(ifOp)))
    if (i != slot)
      for (int p : output)
        middleIds.insert(p);
  if (!middleIds.empty())
    gpu::setPartition(ifOp, middleIds.getArrayRef());
  for (RenderState::Token &token : state.tokens)
    if (token.value == oldResult)
      token.value = enterIf.getResult(0);
  return enterIf;
}

static FailureOr<Operation *> renderRegion(EmitCtx &ctx, GroupDag &g, Node *n,
                                           RenderState &rs,
                                           DenseMap<Node *, Value> &emitted) {
  std::optional<RenderState::Token> incoming;
  if (n->flow) {
    if (n->flow->owner && rs.lastToken() && !rs.lastToken()->ref.owner) {
      incoming = *rs.lastToken();
      incoming->ref.owner = n->flow->owner;
    } else if (auto *token = rs.tokenForOwner(n->flow->owner)) {
      incoming = *token;
    }
  }
  std::optional<TokenRef> resultRef;
  if (n->flow) {
    std::optional<SemaId> sema;
    if (incoming &&
        (n->kind == Node::For || !g.semas[incoming->ref.sema].isEntry ||
         incoming->ref.producer->kind != Node::Acquire ||
         incoming->ref.producer->owner))
      sema = incoming->ref.sema;
    else if (n->flow->concreteSema)
      sema = n->flow->concreteSema;
    else if (incoming)
      sema = incoming->ref.sema;
    if (!sema)
      return semaError(n->op)
             << "region has no statically selected semaphore channel";
    resultRef = TokenRef{n, *sema, n->flow->owner};
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
    RenderState body = rs.nested();
    if (n->flow) {
      if (!resultRef || !incoming)
        return semaError(n->op)
               << "carried loop lacks an exact zero-trip token";
      unsigned idx = slotIndexFor(ctx, n->op, &g);
      forOp.getInitsMutable()[idx].assign(incoming->value);
      RenderState::Token carrier{forOp.getRegionIterArg(idx),
                                 g.semas[resultRef->sema].create, *resultRef};
      body.keepOnly(carrier);
    } else {
      seedRegionEntry(body, n->children[0]);
    }
    if (failed(renderChain(ctx, g, n->children[0], body, emitted)))
      return failure();
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (n->flow) {
      const RegionFlow &c = *n->flow;
      unsigned idx = slotIndexFor(ctx, n->op, &g);
      const RenderState::Token *bodyToken = nullptr;
      Node *final = c.exits.empty() ? nullptr : c.exits.front();
      if (!final)
        bodyToken = body.tokenFor(*resultRef);
      else if (final->kind == Node::Acquire ||
               (final->isRegion() && final->flow))
        bodyToken = body.tokenForProducer(final, resultRef->owner);
      if (!bodyToken)
        return semaError(n->op) << "loop body exports no exact carried token";
      yield->setOperand(idx, bodyToken->value);
      RenderState::Token result{forOp.getResult(idx),
                                g.semas[resultRef->sema].create, *resultRef};
      rs.keepOnly(result);
    }
    rs.clearViews();
    return forOp.getOperation();
  }
  auto ifOp = cast<scf::IfOp>(n->op);
  RenderState thenSt = rs.nested(), elseSt = rs.nested();
  seedRegionEntry(thenSt, n->children[0]);
  if (n->children.size() > 1 && n->children[1])
    seedRegionEntry(elseSt, n->children[1]);
  if (failed(renderChain(ctx, g, n->children[0], thenSt, emitted)))
    return failure();
  if (n->children.size() > 1 && n->children[1])
    if (failed(renderChain(ctx, g, n->children[1], elseSt, emitted)))
      return failure();
  if (n->flow) {
    const RegionFlow &c = *n->flow;
    if (!resultRef)
      return semaError(n->op) << "threaded if lacks an exact result capability";
    unsigned idx = slotIndexFor(ctx, n->op, &g);
    auto exitCapability =
        [&](unsigned branch, RenderState &state) -> const RenderState::Token * {
      Node *final = branch < c.exits.size() ? c.exits[branch] : nullptr;
      if (!final)
        return incoming ? &*incoming : nullptr;
      if (final->kind != Node::Acquire && !(final->isRegion() && final->flow))
        return nullptr;
      return state.tokenForProducer(final, resultRef->owner);
    };
    const RenderState::Token *thenToken = exitCapability(0, thenSt);
    const RenderState::Token *elseToken = exitCapability(1, elseSt);
    if (!thenToken || !elseToken)
      return semaError(n->op)
             << "if path exports no exact compatible capability";
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    thenYield->setOperand(idx, thenToken->value);
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    elseYield->setOperand(idx, elseToken->value);
    RenderState::Token result{ifOp.getResult(idx),
                              g.semas[resultRef->sema].create, *resultRef};
    rs.keepOnly(result);
  }
  rs.clearViews();
  auto it = ctx.remainingIfRenders.find(ifOp);
  assert(it != ctx.remainingIfRenders.end() && it->second);
  if (--it->second)
    return ifOp.getOperation();
  return finishIfRender(ctx, ifOp, rs);
}

static LogicalResult renderChain(EmitCtx &ctx, GroupDag &g, Node *head,
                                 RenderState &rs,
                                 DenseMap<Node *, Value> &emitted) {
  Operation *lastReal = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access || n->kind == Node::Release) {
      RenderState::Token *owned =
          n->owner ? rs.tokenForOwner(n->owner) : nullptr;
      const RenderState::Token *active = rs.lastToken();
      bool consumes = n->kind == Node::Release || nodeTouchesGroup(g, n);
      if (n->reuseTokenOwner && !owned)
        return semaError(n->op ? n->op : g.root->op)
               << "token-reuse node names no live capability for its owner";
      if (consumes && !active && !owned)
        return semaError(n->op ? n->op : g.root->op)
               << "buffer use has no live semaphore capability";
      if (consumes && n->owner && active && active->ref.owner &&
          !sameOwner(active->ref.owner, n->owner) &&
          !(nodeReusesToken(n, n->owner) && owned))
        return semaError(n->op ? n->op : g.root->op)
               << "buffer use consumes another partition's capability";
    }
    switch (n->kind) {
    case Node::Enter:
    case Node::Exit:
      break; // markers; yield wiring is the parent's job
    case Node::Acquire: {
      const Sema &sema = getSema(g, n);
      Owner tokenOwner =
          sema.isEntry && !n->owner ? sema.entryTokenOwner : n->owner;
      if (Value v = emitted.lookup(n)) { // pre-rendered entry instance
        rs.recordToken(v, sema.create, TokenRef{n, n->sema, tokenOwner});
        rs.clearViews();
        break;
      }
      Operation *before = nextRealOp(n->next);
      OpBuilder b(ctx.func);
      if (before) {
        b.setInsertionPoint(before);
      } else if (lastReal &&
                 !isa<triton::FuncOp>(lastReal->getBlock()->getParentOp())) {
        b.setInsertionPoint(lastReal->getBlock()->getTerminator());
      } else if (lastReal) {
        b.setInsertionPointAfter(lastReal);
      } else if (n->parent && n->parent->op) {
        Region &region = n->parent->op->getRegion(0);
        b.setInsertionPoint(region.front().getTerminator());
      }
      auto acq = emitInto<nvws::SemaphoreAcquireOp>(
          b, before ? before->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, sema.create, ctx.tokenType);
      if (n->stageOffset)
        acq.setStage(materializeI32Before(acq, *n->stageOffset));
      emitted[n] = acq.getToken();
      rs.recordToken(acq.getToken(), sema.create,
                     TokenRef{n, n->sema, tokenOwner});
      rs.clearViews();
      lastReal = acq;
      break;
    }
    case Node::Release: {
      const RenderState::Token *source = rs.tokenForNode(n, n->owner);
      assert(source && "release without a live token");
      Value tok = source->value;
      assert(tok && "release without token");
      OpBuilder b(ctx.func);
      if (lastReal)
        b.setInsertionPointAfter(lastReal);
      else if (n->parent && n->parent->op)
        b.setInsertionPointToStart(&n->parent->op->getRegion(0).front());
      else
        b.setInsertionPointToStart(&ctx.func.getBody().front());
      auto rel = emitInto<nvws::SemaphoreReleaseOp>(
          b, lastReal ? lastReal->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, getSema(g, n).create, tok,
          asyncOpsAttr(b.getContext(), n));
      if (n->stageOffset)
        rel.setStage(materializeI32Before(rel, *n->stageOffset));
      rel.setArriveCountAttr(b.getI32IntegerAttr(n->count));
      emitted[n] = Value();
      lastReal = rel;
      break;
    }
    case Node::Access: {
      if (Operation *anchor = renderAccess(ctx, g, n, rs))
        lastReal = anchor;
      break;
    }
    case Node::For:
    case Node::If: {
      FailureOr<Operation *> tail = renderRegion(ctx, g, n, rs, emitted);
      if (failed(tail))
        return failure();
      lastReal = *tail;
      break;
    }
    case Node::Func:
      break;
    }
  }
  return success();
}
static Value materializeI32Before(Operation *op, int64_t value) {
  OpBuilder b(op);
  auto cst = emitInto<arith::ConstantOp>(b, op->getLoc(), resolveOwner(op),
                                         gpu::getStageCluster(op),
                                         b.getI32IntegerAttr(value));
  return cst.getResult();
}
static nvws::SemaphoreAcquireOp resolveAcquireThroughIfs(Value v) {
  for (int fuel = 0; fuel < 8; ++fuel) {
    if (auto acq = v.getDefiningOp<nvws::SemaphoreAcquireOp>())
      return acq;
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return nullptr;
    unsigned idx = cast<OpResult>(v).getResultNumber();
    Value t = ifOp.thenYield()->getOperand(idx);
    if (auto acq = t.getDefiningOp<nvws::SemaphoreAcquireOp>())
      return acq;
    if (ifOp.elseBlock()) {
      Value e = ifOp.elseYield()->getOperand(idx);
      if (auto acq = e.getDefiningOp<nvws::SemaphoreAcquireOp>())
        return acq;
    }
    v = t;
  }
  return nullptr;
}

static LogicalResult
verifyEmittedIR(triton::FuncOp func,
                const DenseSet<Operation *> &reusedTokenBufferOps) {
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
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      terms.push_back(forOp.getBody()->getTerminator());
    else {
      auto ifOp = cast<scf::IfOp>(op);
      terms.push_back(ifOp.thenYield());
      if (!ifOp.getElseRegion().empty())
        terms.push_back(ifOp.elseYield());
    }
    for (auto [i, output] : llvm::enumerate(outputs))
      for (Operation *term : terms) {
        Operation *def = term->getOperand(i).getDefiningOp();
        if (!def || isa<ub::PoisonOp>(def) ||
            def->hasTrait<OpTrait::ConstantLike>() || !gpu::hasPartition(def))
          continue;
        SetVector<int> producer = gpu::getPartitionIds(def);
        if (producer.empty() ||
            llvm::any_of(producer, [&](int p) { return output.contains(p); }))
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
            !reusedTokenBufferOps.contains(user))
          return semaError(user) << "token has a buffer view after its release "
                                    "(use-after-release; spec "
                                    "fable/semas-report3.md Addendum B.3(b))";
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
      if (auto alloc = backing.getDefiningOp<gpu::LocalAllocOp>())
        if (alloc->hasAttr(kBufferCircularAttrName))
          continue;
      if (++slotsPerBacking[backing] > 1) {
        return semaError(forOp)
               << "two token slots for one semaphore group in a single loop "
                  "(spec fable/semas-report3.md Addendum B.3(a)); "
                  "AssignStagePhase cannot thread this";
      }
    }
    return success();
  };
  auto result = func.walk([&](Operation *op) -> WalkResult {
    if (failed(checkPartitionOutputs(op)))
      return WalkResult::interrupt();
    if (auto acq = dyn_cast<nvws::SemaphoreAcquireOp>(op))
      if (failed(checkToken(acq.getToken())))
        return WalkResult::interrupt();
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      if (failed(checkLoop(forOp)))
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
    ctx.poison =
        ub::PoisonOp::create(b, funcOp.getLoc(), ctx.tokenType).getResult();
  }
  for (GroupDag &g : groups)
    if (!g.semas.empty()) {
      nukeGroupTokens(ctx, g);
      forEachNode(g, [&](Node *n) {
        if (n->kind == Node::Access && n->op)
          g.accessNodeOps.insert(n->op);
      });
    }
  while (eraseDeadTokenSlots(ctx, groups)) {
  }
  DenseMap<Node *, Value> emitted;
  if (failed(emitPhysicalIR(ctx, groups)))
    return failure();
  for (GroupDag &g : groups)
    if (!g.semas.empty())
      emitEntryAcquires(ctx, g, emitted);
  rewriteSignatures(ctx, groups);
  for (GroupDag &g : groups)
    if (!g.semas.empty())
      forEachNode(g, [&](Node *n) {
        if (n->kind == Node::If)
          ++ctx.remainingIfRenders[n->op];
      });
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    RenderState rs;
    for (Node *n = g.root->children[0]; n; n = n->next)
      if (n->kind == Node::Acquire && emitted.count(n)) {
        const Sema &sema = getSema(g, n);
        Owner owner =
            sema.isEntry && !n->owner ? sema.entryTokenOwner : n->owner;
        rs.recordToken(emitted.lookup(n), sema.create,
                       TokenRef{n, n->sema, owner});
      }
    if (failed(renderChain(ctx, g, g.root->children[0], rs, emitted)))
      return failure();
  }
  while (eraseDeadTokenSlots(ctx, groups)) {
  }
  {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> aliasOps;
      ctx.func.walk([&](Operation *op) {
        if (isSupportedAliasOp(op))
          aliasOps.push_back(op);
      });
      for (Operation *op : llvm::reverse(aliasOps))
        if (llvm::all_of(op->getResults(),
                         [](Value v) { return v.use_empty(); })) {
          op->erase();
          changed = true;
        }
    }
  }
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    for (const Member &m : g.pieceTable.members)
      if (m.allocOp && m.allocOp->getBlock() && m.allocOp->use_empty())
        m.allocOp->erase();
  }
  if (ctx.poison.use_empty())
    ctx.poison.getDefiningOp()->erase();
  if (failed(verifyEmittedIR(funcOp, ctx.reusedTokenBufferOps)))
    return failure();
  return success();
}
} // namespace mlir::triton::nvws_semas
