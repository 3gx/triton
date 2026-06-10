#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_

// Stage 1 — ACCESS-DAG (spec fable/semas-report3.md section 3; plan
// fable/new-insert-semas-plan-2.md commit 1). Pure analysis: discovery of
// buffer groups, cut-point pieces/footprints, access events with R/W
// touches, region effect summaries, and the ACCESS-DAG dump. No IR mutation.

// ---------------------------------------------------------------------------
// Discovery: bucket allocs by buffer.id (synthetic id when absent), uniform
// over TMEM and local. TMEM = every ttng.tmem_alloc; local = every
// mutable-memdesc ttg.local_alloc.
// ---------------------------------------------------------------------------
static SmallVector<GroupDag, 0> collectGroups(triton::FuncOp funcOp) {
  llvm::MapVector<int64_t, SmallVector<Operation *, 2>> tmemBuckets,
      localBuckets;
  llvm::DenseSet<int64_t> syntheticIds; // negative keys = synthetic
  int64_t nextSynthetic = -1;

  funcOp.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
      int64_t key = id ? *id : nextSynthetic--;
      if (!id)
        syntheticIds.insert(key);
      tmemBuckets[key].push_back(op);
      return;
    }
    if (auto alloc = dyn_cast<gpu::LocalAllocOp>(op)) {
      auto type = cast<gpu::MemDescType>(alloc.getType());
      if (!type.getMutableMemory())
        return;
      std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
      int64_t key = id ? *id : nextSynthetic--;
      if (!id)
        syntheticIds.insert(key);
      localBuckets[key].push_back(op);
      return;
    }
  });

  SmallVector<GroupDag, 0> groups;
  auto makeGroup = [&](MemKind memory, int64_t id,
                       ArrayRef<Operation *> allocs) {
    groups.emplace_back();
    GroupDag &g = groups.back();
    g.groupIdx = static_cast<unsigned>(groups.size() - 1);
    g.bufferId = id;
    g.synthetic = syntheticIds.contains(id);
    g.memory = memory;
    for (Operation *allocOp : allocs) {
      Member m;
      m.allocOp = allocOp;
      m.type = cast<gpu::MemDescType>(allocOp->getResult(0).getType());
      m.offset = getI64Attr(allocOp, kBufferOffsetAttrName).value_or(0);
      m.extent = memberExtent(memory, m.type);
      MemberId idx = static_cast<MemberId>(g.pieceTable.members.size());
      g.pieceTable.members.push_back(m);
      g.aliases.try_emplace(allocOp->getResult(0),
                            std::make_pair(idx, SmallVector<AliasStep, 2>()));
    }
  };
  for (auto &[id, allocs] : tmemBuckets)
    makeGroup(MemKind::Tmem, id, allocs);
  for (auto &[id, allocs] : localBuckets)
    makeGroup(MemKind::Local, id, allocs);
  return groups;
}

// ---------------------------------------------------------------------------
// Pieces: the cut-point construction (spec section 3 item 2). Invariant:
// two members overlap <=> their footprints intersect.
// ---------------------------------------------------------------------------
static void buildPieces(PieceTable &pt) {
  SmallVector<int64_t, 8> cuts;
  for (const Member &m : pt.members) {
    cuts.push_back(m.offset);
    cuts.push_back(m.offset + m.extent);
  }
  llvm::sort(cuts);
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

  // Candidate intervals between adjacent cuts; cover sets by containment.
  SmallVector<Piece, 4> raw;
  for (size_t i = 0; i + 1 < cuts.size(); ++i) {
    Piece p;
    p.lo = cuts[i];
    p.hi = cuts[i + 1];
    for (auto [mIdx, m] : llvm::enumerate(pt.members))
      if (m.offset <= p.lo && p.hi <= m.offset + m.extent)
        p.cover.push_back(static_cast<MemberId>(mIdx));
    if (!p.cover.empty())
      raw.push_back(std::move(p));
  }
  // Merge adjacent intervals with identical cover (coarsest valid partition).
  for (Piece &p : raw) {
    if (!pt.pieces.empty() && pt.pieces.back().hi == p.lo &&
        pt.pieces.back().cover == p.cover) {
      pt.pieces.back().hi = p.hi;
      continue;
    }
    pt.pieces.push_back(std::move(p));
  }
  // Invert to footprints.
  pt.footprint.assign(pt.members.size(), {});
  for (auto [pIdx, piece] : llvm::enumerate(pt.pieces))
    for (MemberId m : piece.cover)
      pt.footprint[m].push_back(static_cast<PieceId>(pIdx));
  // Connected components: pieces sharing a member are one token game.
  SmallVector<unsigned, 4> parent(pt.pieces.size());
  for (auto [i, _] : llvm::enumerate(parent))
    parent[i] = i;
  std::function<unsigned(unsigned)> find = [&](unsigned x) -> unsigned {
    while (parent[x] != x)
      x = parent[x] = parent[parent[x]];
    return x;
  };
  for (const auto &fp : pt.footprint)
    for (size_t i = 1; i < fp.size(); ++i)
      parent[find(fp[i])] = find(fp[0]);
  // Renumber components in ascending first-piece order (determinism).
  pt.pieceComp.assign(pt.pieces.size(), 0);
  DenseMap<unsigned, CompId> compId; // lookup-only
  CompId next = 0;
  for (auto [pIdx, _] : llvm::enumerate(pt.pieces)) {
    unsigned rep = find(static_cast<unsigned>(pIdx));
    auto it = compId.find(rep);
    if (it == compId.end())
      it = compId.try_emplace(rep, next++).first;
    pt.pieceComp[pIdx] = it->second;
  }
}

// ---------------------------------------------------------------------------
// Access-event collection + node-tree construction. One recursive pass in
// program order: alias chains extend as they are encountered; every access
// becomes an Access node with per-member touches classified per spec
// section 1.1.
// ---------------------------------------------------------------------------

static bool isStructuralOp(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::YieldOp, triton::FuncOp,
             triton::ReturnOp>(op);
}

// Try to extend the alias map through `op`. Returns: failure on an
// unsupported forward of a tracked memdesc; aliased=true if `op` was
// consumed as a view step (then it is not an access event).
static FailureOr<bool> tryExtendAlias(GroupDag &g, Operation *op) {
  if (op->getNumResults() != 1 ||
      !isa<gpu::MemDescType>(op->getResult(0).getType()))
    return false;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    auto it = g.aliases.find(operand);
    if (it == g.aliases.end())
      continue;
    if (!isSupportedAliasOp(op))
      return op->emitError("nvws-insert-semas: unsupported memdesc alias use ")
             << op->getName();
    auto chain = it->second; // copy {member, steps}
    chain.second.push_back({op, static_cast<unsigned>(idx)});
    g.aliases.try_emplace(op->getResult(0), std::move(chain));
    return true;
  }
  return false;
}

static void addTouch(GroupDag &g, SmallVectorImpl<Touch> &touches, Value v,
                     Effect effect) {
  auto it = g.aliases.find(v);
  if (it == g.aliases.end())
    return;
  Touch t;
  t.member = it->second.first;
  t.effect = effect;
  t.accessValue = v;
  t.alias = it->second.second;
  touches.push_back(std::move(t));
}

// Classify `op`'s touches on this group's buffers (spec section 1.1 table).
static LogicalResult collectTouches(GroupDag &g, Operation *op,
                                    SmallVectorImpl<Touch> &touches) {
  // Sourceful allocs act as a store of their source into the new buffer.
  if (auto tmemAlloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    if (tmemAlloc.getSrc())
      addTouch(g, touches, tmemAlloc.getResult(), Effect::W);
    return success();
  }
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op)) {
    if (localAlloc.getSrc())
      addTouch(g, touches, localAlloc.getResult(), Effect::W);
    return success();
  }
  if (auto load = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
    addTouch(g, touches, load.getSrc(), Effect::R);
    return success();
  }
  if (auto store = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
    addTouch(g, touches, store.getDst(), Effect::W);
    return success();
  }
  if (auto load = dyn_cast<gpu::LocalLoadOp>(op)) {
    addTouch(g, touches, load.getSrc(), Effect::R);
    return success();
  }
  if (auto store = dyn_cast<gpu::LocalStoreOp>(op)) {
    addTouch(g, touches, store.getDst(), Effect::W);
    return success();
  }
  if (auto descLoad = dyn_cast<nvws::DescriptorLoadOp>(op)) {
    addTouch(g, touches, descLoad.getResult(), Effect::W);
    return success();
  }
  if (auto descGather = dyn_cast<nvws::DescriptorGatherOp>(op)) {
    addTouch(g, touches, descGather.getResult(), Effect::W);
    return success();
  }
  if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
    // MMA: accumulator touch is W; A/B operand touches are R (whether the
    // operand lives in SMEM or TMEM). Spec section 1.1 — no special cases.
    Value acc = mma.getAccumulator();
    bool accTouched = false;
    for (Value operand : op->getOperands()) {
      if (operand == acc) {
        if (!accTouched)
          addTouch(g, touches, operand, Effect::W);
        accTouched = true;
        continue;
      }
      addTouch(g, touches, operand, Effect::R);
    }
    return success();
  }
  // Tracked memdesc flowing into structured control flow (loop init args,
  // yields, ...) is unsupported: hard diagnostic, never a silent skip.
  if (isStructuralOp(op)) {
    for (Value operand : op->getOperands())
      if (g.aliases.contains(operand))
        return op->emitError("nvws-insert-semas: unsupported memdesc flow "
                             "through control-flow op ")
               << op->getName();
    return success();
  }
  // Everything else touching a tracked memdesc: not provably load-only,
  // classified W (spec section 1.1: R is the closed list above).
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      addTouch(g, touches, operand, Effect::W);
  return success();
}

// Build the chain of `block` under `parent`; returns the head via out-param.
// Structural For/If nodes are created only when their subtree contains
// accesses of this group (spec section 3 item 3).
static FailureOr<Node *> buildChainForBlock(GroupDag &g, Block &block,
                                            Node *parent);

static void appendNode(Node *parent, Node *&head, Node *&tail, Node *n) {
  n->parent = parent;
  n->prev = tail;
  if (tail)
    tail->next = n;
  else
    head = n;
  tail = n;
}

static FailureOr<Node *> buildChainForBlock(GroupDag &g, Block &block,
                                            Node *parent) {
  Node *head = nullptr, *tail = nullptr;
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      // Tracked memdesc as loop operand is unsupported (hard diagnostic).
      for (Value operand : op.getOperands())
        if (g.aliases.contains(operand))
          return op.emitError("nvws-insert-semas: unsupported memdesc flow "
                              "through control-flow op ")
                 << op.getName();
      Node *forNode = g.newNode(Node::For, &op, parent);
      auto body = buildChainForBlock(g, *forOp.getBody(), forNode);
      if (failed(body))
        return failure();
      if (!*body) {
        g.nodes.pop_back(); // empty subtree: no structural node
        continue;
      }
      forNode->children.push_back(*body);
      appendNode(parent, head, tail, forNode);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      for (Value operand : op.getOperands())
        if (g.aliases.contains(operand))
          return op.emitError("nvws-insert-semas: unsupported memdesc flow "
                              "through control-flow op ")
                 << op.getName();
      Node *ifNode = g.newNode(Node::If, &op, parent);
      auto thenChain = buildChainForBlock(g, *ifOp.thenBlock(), ifNode);
      if (failed(thenChain))
        return failure();
      FailureOr<Node *> elseChain((Node *)nullptr);
      if (ifOp.elseBlock()) {
        elseChain = buildChainForBlock(g, *ifOp.elseBlock(), ifNode);
        if (failed(elseChain))
          return failure();
      }
      if (!*thenChain && !*elseChain) {
        g.nodes.pop_back();
        continue;
      }
      ifNode->children.push_back(*thenChain); // may be null
      ifNode->children.push_back(*elseChain); // may be null
      appendNode(parent, head, tail, ifNode);
      continue;
    }
    // Alias steps are consumed by the chain, not access events.
    auto aliased = tryExtendAlias(g, &op);
    if (failed(aliased))
      return failure();
    if (*aliased)
      continue;
    SmallVector<Touch, 2> touches;
    if (failed(collectTouches(g, &op, touches)))
      return failure();
    if (touches.empty())
      continue;
    Node *access = g.newNode(Node::Access, &op, parent);
    access->owner = resolveOwner(&op);
    access->touches = std::move(touches);
    appendNode(parent, head, tail, access);
  }
  return head;
}

// Region effect summaries (spec section 3 item 4): per For/If node, the
// per-piece OR of subtree touch effects. Presence in the map IS the
// region's footprint.
static void computeEffectSummary(GroupDag &g, Node *n,
                                 DenseMap<PieceId, Effect> &out) {
  if (n->kind == Node::Access) {
    for (const Touch &t : n->touches)
      for (PieceId p : g.pieceTable.footprint[t.member]) {
        auto it = out.find(p);
        if (it == out.end())
          out.try_emplace(p, t.effect);
        else
          it->second = joinEffect(it->second, t.effect);
      }
    return;
  }
  DenseMap<PieceId, Effect> sub;
  for (Node *childHead : n->children)
    for (Node *c = childHead; c; c = c->next)
      computeEffectSummary(g, c, sub);
  if (n->kind == Node::For || n->kind == Node::If)
    for (const auto &[p, e] : sub)
      n->pieceInfo[p] = PieceInfo{std::nullopt, e};
  for (const auto &[p, e] : sub) {
    auto it = out.find(p);
    if (it == out.end())
      out.try_emplace(p, e);
    else
      it->second = joinEffect(it->second, e);
  }
}

// Stage-1 entry point for one group.
static LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp) {
  buildPieces(g.pieceTable);
  Node *func = g.newNode(Node::Func, funcOp, nullptr);
  auto chain = buildChainForBlock(g, funcOp.getBody().front(), func);
  if (failed(chain))
    return failure();
  if (*chain)
    func->children.push_back(*chain);
  g.root = func;
  DenseMap<PieceId, Effect> ignored;
  computeEffectSummary(g, func, ignored);
  return success();
}

// ---------------------------------------------------------------------------
// ACCESS-DAG dump (format: spec section 3 dump + section 5.5 conventions;
// FOR/IF rows annotated with per-piece effects).
// ---------------------------------------------------------------------------
static void printEffects(llvm::raw_ostream &os, const Node *n) {
  if (n->pieceInfo.empty())
    return;
  os << " effects{";
  bool first = true;
  for (const auto &[p, info] : sortedPieceInfo(n)) {
    if (!first)
      os << ",";
    first = false;
    os << "P" << p << ":" << (info.effect == Effect::W ? "W" : "R");
  }
  os << "}";
}

static void dumpAccessChain(GroupDag &g, const Node *head, unsigned depth) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    // The ACCESS view filters later-stage rows (the tree is extended in
    // place; each stage's dump shows only the kinds it owns).
    if (n->kind == Node::Enter || n->kind == Node::Exit ||
        n->kind == Node::Acquire || n->kind == Node::Release)
      continue;
    if (n->kind == Node::For) {
      os << treePrefix(depth) << "|- scf.for";
      if (gpu::hasWarpSpecializeTag(n->op))
        os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      printEffects(os, n);
      os << "\n";
      dumpAccessChain(g, n->children[0], depth + 1);
      continue;
    }
    if (n->kind == Node::If) {
      // Faithful rendering (spec section 3): `then` always renders (an
      // scf.if always has a then region), even when empty; `else` renders
      // iff the IR op has an else region — the VIRTUAL else enters the DAG
      // only at stage 2 and belongs to the OWNER view. Empty branches are
      // bare labels: absence means absent from the DAG, nothing else.
      os << treePrefix(depth) << "|- scf.if";
      printEffects(os, n);
      os << "\n";
      os << treePrefix(depth + 1) << "|- then\n";
      if (n->children[0])
        dumpAccessChain(g, n->children[0], depth + 2);
      if (cast<scf::IfOp>(n->op).elseBlock()) {
        os << treePrefix(depth + 1) << "|- else\n";
        if (n->children.size() > 1 && n->children[1])
          dumpAccessChain(g, n->children[1], depth + 2);
      }
      continue;
    }
    assert(n->kind == Node::Access);
    for (const Touch &t : n->touches) {
      os << treePrefix(depth) << "|- "
         << (t.effect == Effect::W ? "W" : "R") << "  m" << t.member << "  "
         << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner)
         << "\n";
    }
  }
}

static void dumpGroupAccessDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "GROUP ";
  if (g.synthetic)
    os << "buffer.id=none#" << -g.bufferId;
  else
    os << "buffer.id=" << g.bufferId;
  os << " memory=" << (g.isTmem() ? "tmem" : "local")
     << " members=" << g.pieceTable.members.size() << "\n";
  os << "  members:";
  for (auto [idx, m] : llvm::enumerate(g.pieceTable.members))
    os << " m" << idx << "[" << m.offset << "," << (m.offset + m.extent)
       << ")";
  os << "\n  pieces:";
  for (auto [idx, p] : llvm::enumerate(g.pieceTable.pieces)) {
    os << " P" << idx << "=[" << p.lo << "," << p.hi << "){";
    for (auto [k, m] : llvm::enumerate(p.cover))
      os << (k ? "," : "") << "m" << m;
    os << "}c" << g.pieceTable.pieceComp[idx];
  }
  os << "\n  footprints:";
  for (auto [idx, fp] : llvm::enumerate(g.pieceTable.footprint)) {
    os << " m" << idx << "={";
    for (auto [k, p] : llvm::enumerate(fp))
      os << (k ? "," : "") << "P" << p;
    os << "}";
  }
  os << "\nACCESS-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  if (!g.root->children.empty())
    dumpAccessChain(g, g.root->children[0], 1);
}

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
