// ACCESS analysis; see sema-docs/insert-semas/access-dag.md.
#include "InsertSemas.h"

namespace mlir::triton::nvws_semas {

FailureOr<SmallVector<GroupDag, 0>> collectGroups(triton::FuncOp funcOp) {
  using Buckets = llvm::MapVector<int64_t, SmallVector<Operation *, 2>>;
  Buckets tmemBuckets, localBuckets;
  SmallVector<Operation *, 4> circularLocals;
  DenseSet<int64_t> syntheticIds;
  int64_t nextSynthetic = -1;
  auto add = [&](Buckets &buckets, Operation *op, std::optional<int64_t> id) {
    int64_t key = id ? *id : nextSynthetic--;
    if (!id)
      syntheticIds.insert(key);
    buckets[key].push_back(op);
  };
  LogicalResult result = success();
  funcOp.walk([&](Operation *op) {
    std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
    if (isa<nvidia_gpu::TMEMAllocOp>(op)) {
      add(tmemBuckets, op, id);
      return;
    }
    auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
    if (!alloc || !cast<gpu::MemDescType>(alloc.getType()).getMutableMemory())
      return;
    if (!op->hasAttr(kBufferCircularAttrName)) {
      add(localBuckets, op, id);
      return;
    }
    if (!id) {
      result = semaError(op) << "circular local alloc requires buffer.id";
      return;
    }
    for (StringRef name : {kBufferCopyAttrName, kBufferStartAttrName})
      if (!op->hasAttr(name)) {
        result = semaError(op) << "circular local alloc requires " << name;
        return;
      }
    if (op->hasAttr(kBufferOffsetAttrName)) {
      result = semaError(op)
               << "circular local alloc must not carry buffer.offset";
      return;
    }
    circularLocals.push_back(op);
  });
  if (failed(result))
    return failure();

  SmallVector<GroupDag, 0> groups;
  auto makeGroup = [&](MemKind memory, int64_t id, ArrayRef<Operation *> allocs,
                       bool circular = false, bool mixedDepth = false) {
    GroupDag &g = groups.emplace_back();
    g.bufferId = id;
    g.synthetic = syntheticIds.contains(id);
    g.mixedDepthPhysicalAlias = mixedDepth;
    g.memory = memory;
    g.circular = circular;
    for (Operation *op : allocs) {
      auto type = cast<gpu::MemDescType>(op->getResult(0).getType());
      int64_t extent =
          memory == MemKind::Tmem
              ? static_cast<int64_t>(mlir::triton::getMemDescSize(type))
              : (type.getShape().empty() ? 1 : type.getShape().front());
      Member member{
          op, type,
          circular ? 0 : getI64Attr(op, kBufferOffsetAttrName).value_or(0),
          extent, getI64Attr(op, kBufferStartAttrName).value_or(0)};
      MemberId index = g.pieceTable.members.size();
      g.pieceTable.members.push_back(member);
      g.aliases.try_emplace(op->getResult(0),
                            std::make_pair(index, SmallVector<AliasStep, 2>()));
    }
  };
  for (auto &[id, allocs] : tmemBuckets) {
    auto firstCopy = getI64Attr(allocs.front(), kBufferCopyAttrName);
    bool split =
        firstCopy &&
        llvm::all_of(allocs,
                     [&](Operation *op) {
                       return getI64Attr(op, kBufferCopyAttrName).has_value();
                     }) &&
        llvm::any_of(allocs, [&](Operation *op) {
          return getI64Attr(op, kBufferCopyAttrName) != firstCopy;
        });
    if (!split) {
      makeGroup(MemKind::Tmem, id, allocs);
      continue;
    }
    for (Operation *op : allocs)
      makeGroup(MemKind::Tmem, id, ArrayRef<Operation *>(op), false, true);
  }
  for (auto &[id, allocs] : localBuckets)
    makeGroup(MemKind::Local, id, allocs);
  for (Operation *op : circularLocals)
    makeGroup(MemKind::Local, *getI64Attr(op, kBufferIdAttrName),
              ArrayRef<Operation *>(op), true);
  return groups;
}

static bool buildPieces(PieceTable &pt) {
  SmallVector<int64_t, 8> cuts;
  for (const Member &member : pt.members) {
    cuts.push_back(member.offset);
    cuts.push_back(member.offset + member.extent);
  }
  llvm::sort(cuts);
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  for (size_t i = 0; i + 1 < cuts.size(); ++i) {
    SmallVector<MemberId, 2> cover;
    for (auto [index, member] : llvm::enumerate(pt.members))
      if (member.offset <= cuts[i] &&
          cuts[i + 1] <= member.offset + member.extent)
        cover.push_back(static_cast<MemberId>(index));
    if (cover.empty())
      continue;
    if (!pt.pieces.empty() && pt.pieces.back().hi == cuts[i] &&
        pt.pieces.back().cover == cover) {
      pt.pieces.back().hi = cuts[i + 1];
      continue;
    }
    pt.pieces.push_back(Piece{cuts[i], cuts[i + 1], std::move(cover)});
  }
  pt.footprint.assign(pt.members.size(), {});
  for (auto [piece, info] : llvm::enumerate(pt.pieces))
    for (MemberId member : info.cover)
      pt.footprint[member].push_back(static_cast<PieceId>(piece));
  // Interval pieces form one component exactly when each adjacent pair
  // overlaps through a member.
  for (size_t i = 1; i < pt.pieces.size(); ++i)
    if (pt.pieces[i - 1].hi != pt.pieces[i].lo ||
        llvm::none_of(pt.pieces[i - 1].cover, [&](MemberId member) {
          return llvm::is_contained(pt.pieces[i].cover, member);
        }))
      return false;
  return true;
}
static LogicalResult rejectAliasOperands(GroupDag &g, Operation *op) {
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      return semaError(op)
             << "unsupported memdesc flow through control-flow op "
             << op->getName();
  return success();
}
static LogicalResult collectTouches(GroupDag &g, Operation *op,
                                    SmallVectorImpl<Touch> &touches) {
  if (op->getNumResults() == 1 &&
      isa<gpu::MemDescType>(op->getResult(0).getType()))
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto it = g.aliases.find(operand);
      if (it == g.aliases.end())
        continue;
      if (!isSupportedAliasOp(op))
        return semaError(op)
               << "unsupported memdesc alias use " << op->getName();
      auto alias = it->second;
      alias.second.push_back(
          {op, static_cast<unsigned>(index), op->getResult(0).getType()});
      g.aliases.try_emplace(op->getResult(0), std::move(alias));
      return success();
    }
  auto touch = [&](Value value, Effect effect) {
    auto it = g.aliases.find(value);
    if (it != g.aliases.end())
      touches.push_back(Touch{it->second.first, effect, value, value.getType(),
                              it->second.second});
  };
  if (auto tmemAlloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    if (tmemAlloc.getSrc())
      touch(tmemAlloc.getResult(), Effect::W);
    return success();
  }
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op)) {
    if (Value src = localAlloc.getSrc()) {
      if (Operation *def = src.getDefiningOp())
        if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(def) &&
            g.aliases.count(localAlloc.getResult())) // member of THIS group
          g.ttDescriptorFedMembers.push_back(localAlloc);
      touch(localAlloc.getResult(), Effect::W);
    }
    return success();
  }
  Value read, write;
  if (auto x = dyn_cast<nvidia_gpu::TMEMLoadOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<gpu::LocalLoadOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<nvidia_gpu::TMEMStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<gpu::LocalStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<nvws::DescriptorLoadOp>(op))
    write = x.getResult();
  else if (auto x = dyn_cast<nvws::DescriptorGatherOp>(op))
    write = x.getResult();
  if (read || write) {
    touch(read ? read : write, read ? Effect::R : Effect::W);
    return success();
  }
  if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
    Value acc = mma.getAccumulator();
    bool accTouched = false;
    for (Value operand : op->getOperands()) {
      if (operand == acc) {
        if (!accTouched)
          touch(operand, Effect::W);
        accTouched = true;
        continue;
      }
      touch(operand, Effect::R);
    }
    return success();
  }
  if (isa<scf::YieldOp, triton::FuncOp, triton::ReturnOp>(op)) {
    return rejectAliasOperands(g, op);
  }
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      touch(operand, Effect::W);
  return success();
}

struct Chain {
  Node *head = nullptr, *tail = nullptr;
  DenseMap<PieceId, Effect> effects;
  DenseMap<PieceId, Owner> firstOwners, lastOwners;
};
static void appendNode(GroupDag &g, Chain &chain, Node *node) {
  node->prev = chain.tail;
  if (chain.tail)
    chain.tail->next = node;
  else
    chain.head = node;
  chain.tail = node;
  auto record = [&](PieceId piece, Effect effect, const Owner &owner) {
    mergeEffect(chain.effects, piece, effect);
    chain.firstOwners.try_emplace(piece, owner);
    chain.lastOwners[piece] = owner;
  };
  if (node->kind == Node::Access) {
    forEachTouchedPiece(g, node, [&](PieceId piece, Effect effect) {
      record(piece, effect, node->owner);
    });
  } else if (node->isRegion()) {
    bool sealed =
        node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [piece, info] : node->pieceInfo)
      record(piece, info.effect, sealed ? Owner() : info.owner);
  }
}
static LogicalResult deriveCompletionAnchor(Node *access) {
  auto load = dyn_cast<gpu::LocalLoadOp>(access->op);
  if (!load)
    return success();
  Operation *forward = nullptr, *store = nullptr;
  unsigned paths = 0;
  for (Operation *user : load.getResult().getUsers()) {
    if (isa<triton::DescriptorStoreOp>(user)) {
      forward = nullptr;
      store = user;
      ++paths;
      continue;
    }
    if (auto convert = dyn_cast<gpu::ConvertLayoutOp>(user))
      for (Operation *convertUser : convert.getResult().getUsers())
        if (isa<triton::DescriptorStoreOp>(convertUser)) {
          forward = user;
          store = convertUser;
          ++paths;
        }
  }
  if (!paths)
    return success();
  if (paths != 1)
    return semaError(load) << "managed local_load reaches multiple descriptor "
                              "stores; ownership completion is ambiguous";
  auto onlyUser = [](Value value, Operation *expected) {
    return llvm::hasSingleElement(value.getUsers()) &&
           *value.getUsers().begin() == expected;
  };
  if (!onlyUser(load.getResult(), forward ? forward : store))
    return semaError(load) << "descriptor-store local_load path has fan-out";
  if (forward && !onlyUser(forward->getResult(0), store))
    return semaError(load)
           << "descriptor-store convert_layout path has fan-out";
  Block *block = load->getBlock();
  if (store->getBlock() != block || (forward && forward->getBlock() != block)) {
    InFlightDiagnostic diag =
        semaError(load) << "descriptor-store completion crosses control flow";
    diag.attachNote(store->getLoc()) << "descriptor store is here";
    return failure();
  }
  if (!load->isBeforeInBlock(store))
    return semaError(load) << "descriptor store must follow managed local_load";
  if (!sameOwner(access->owner, resolveOwner(store))) {
    InFlightDiagnostic diag = semaError(load)
                              << "descriptor-store completion owner differs "
                                 "from managed local_load owner";
    diag.attachNote(store->getLoc()) << "descriptor store is here";
    return failure();
  }
  access->completionAnchor = store;
  return success();
}

static FailureOr<Chain> buildChainForBlock(GroupDag &g, Block &block,
                                           Node *parent) {
  Chain chain;
  for (Operation &op : block) {
    Node::Kind kind = isa<scf::ForOp>(op)  ? Node::For
                      : isa<scf::IfOp>(op) ? Node::If
                                           : Node::Access;
    if (kind != Node::Access) {
      if (failed(rejectAliasOperands(g, &op)))
        return failure();
      Node *region = g.newNode(kind, &op, parent);
      SmallVector<Chain, 2> branches;
      for (Region &nested : op.getRegions()) {
        if (nested.empty()) {
          branches.emplace_back();
          continue;
        }
        auto branch = buildChainForBlock(g, nested.front(), region);
        if (failed(branch))
          return failure();
        branches.push_back(std::move(*branch));
      }
      if (llvm::all_of(branches,
                       [](const Chain &branch) { return !branch.head; })) {
        g.nodes.pop_back();
        continue;
      }
      for (const Chain &branch : branches)
        for (auto [piece, effect] : branch.effects) {
          auto [it, inserted] = region->pieceInfo.try_emplace(
              piece, PieceInfo{std::nullopt, effect});
          if (!inserted)
            it->second.effect = joinEffect(it->second.effect, effect);
        }
      auto lookup = [](const DenseMap<PieceId, Owner> &owners,
                       PieceId piece) -> const Owner * {
        auto it = owners.find(piece);
        return it == owners.end() ? nullptr : &it->second;
      };
      for (auto [piece, info] : sortedPieceInfo(region)) {
        const Owner *owner = nullptr;
        if (region->kind == Node::If)
          owner = lookup(chain.lastOwners, piece);
        for (const Chain &branch : branches)
          if (!owner)
            owner = lookup(branch.firstOwners, piece);
        if (!owner) {
          semaError(region->op)
              << "no toucher resolves the owner for a piece in this region's "
                 "summary (stage-1/stage-2 inconsistency)";
          return failure();
        }
        region->pieceInfo[piece].owner = *owner;
      }
      for (const Chain &branch : branches) {
        Node *enter = g.newNode(Node::Enter, nullptr, region);
        Node *exit = g.newNode(Node::Exit, nullptr, region);
        for (auto [piece, effect] : branch.effects)
          enter->pieceInfo[piece] = exit->pieceInfo[piece] =
              PieceInfo{region->pieceInfo[piece].owner, effect};
        Node *tail = branch.tail ? branch.tail : enter;
        if (branch.head) {
          enter->next = branch.head;
          branch.head->prev = enter;
        }
        tail->next = exit;
        exit->prev = tail;
        region->children.push_back(enter);
      }
      appendNode(g, chain, region);
      continue;
    }
    SmallVector<Touch, 2> touches;
    if (failed(collectTouches(g, &op, touches)))
      return failure();
    if (touches.empty())
      continue;
    Node *access = g.newNode(Node::Access, &op, parent);
    access->owner = resolveOwner(&op);
    access->touches = std::move(touches);
    if (failed(deriveCompletionAnchor(access)))
      return failure();
    appendNode(g, chain, access);
  }
  return chain;
}

// Return the block directly owned by the function body that contains `op`.
// InsertSemas models structured control flow recursively, but a Triton
// function may also contain top-level CFG blocks (for example an early-return
// diamond).  A managed allocation and all of its memdesc users must live in a
// single such block until the access DAG grows a CFG dataflow model.
static Block *getFunctionBlock(Operation *op, triton::FuncOp funcOp) {
  Block *block = op->getBlock();
  while (block && block->getParent() != &funcOp.getBody()) {
    Operation *parent = block->getParentOp();
    block = parent ? parent->getBlock() : nullptr;
  }
  return block;
}

static FailureOr<Block *> getAnalysisBlock(GroupDag &g,
                                           triton::FuncOp funcOp) {
  Block *analysisBlock = nullptr;
  SmallVector<Value, 4> worklist;
  DenseSet<Value> seen;
  for (const Member &member : g.pieceTable.members) {
    Block *block = getFunctionBlock(member.allocOp, funcOp);
    if (!block)
      return semaError(member.allocOp)
             << "managed allocation is not nested in the function body";
    if (analysisBlock && analysisBlock != block)
      return semaError(member.allocOp)
             << "one buffer group spans function CFG blocks";
    analysisBlock = block;
    worklist.push_back(member.allocOp->getResult(0));
  }

  // Do not silently miss a use in another CFG block.  Alias operations are
  // followed explicitly because their results carry the same managed
  // storage.  Structured-region users remain in the same function block and
  // are handled recursively by buildChainForBlock.
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (getFunctionBlock(user, funcOp) != analysisBlock)
        return semaError(user)
               << "managed memdesc flow across function CFG blocks is "
                  "unsupported";
      if (user->getNumResults() == 1 &&
          isa<gpu::MemDescType>(user->getResult(0).getType()) &&
          isSupportedAliasOp(user))
        worklist.push_back(user->getResult(0));
    }
  }
  return analysisBlock;
}

LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp) {
  // Single-component invariant: every buffer.id group's pieces connect
  // (through shared members) into one component -- the memory planner keeps
  // reusers stacked within their owner's columns. The rest of InsertSemas
  // relies on this (one group == one synchronization unit); reject anything
  // that violates it rather than mis-synchronizing.
  if (!buildPieces(g.pieceTable)) {
    Operation *at = g.pieceTable.members.front().allocOp;
    return semaError(at) << "buffer.id group has disjoint pieces (more than "
                            "one connected component); InsertSemas requires "
                            "one component per group";
  }
  Node *func = g.newNode(Node::Func, funcOp, nullptr);
  FailureOr<Block *> analysisBlock = getAnalysisBlock(g, funcOp);
  if (failed(analysisBlock))
    return failure();
  auto chain = buildChainForBlock(g, **analysisBlock, func);
  if (failed(chain))
    return failure();
  if (chain->head)
    func->children.push_back(chain->head);
  g.root = func;
  return success();
}

} // namespace mlir::triton::nvws_semas
