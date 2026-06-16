// Stage 4 of nvws-insert-semas: EMIT-IR (spec section 6; plan commit 4).
// Strict order: (1) token-nuke pre-process, (2) backings + creates + entry
// acquires, (3) aggregated signature rewrites, (4) render walk per group,
// (5) post-emit verifier, (6) TMEM backing coalescing, (7) loop-scheduler
// workaround. The emitter transcribes the SYNC-DAG; it decides nothing.

// ---------------------------------------------------------------------------
// Shared emission state.
// ---------------------------------------------------------------------------
#include "InsertSemasEmitIR.h"
#include "InsertSemasSyncDag.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

struct EmitCtx {
  triton::FuncOp func;
  Value poison; // the single function-level ub.poison token (contract E)
  Type tokenType;
  // Slot registry after step 3: region op -> ordered slots; slot index is
  // relative to the op's ORIGINAL result/iter count at rewrite time.
  struct Slot {
    GroupDag *g;
    CompId comp;
    unsigned index; // absolute result / iter_arg index in the NEW op
  };
  llvm::MapVector<Operation *, SmallVector<Slot, 2>> slots;
};

// Per-group render-walk state.
struct RenderState {
  DenseMap<CompId, Value> carrier;          // current carrier token per comp
  DenseMap<CompId, Value> carrierSema;      // the create of that carrier's
                                            // acquire (buffer ops pair the
                                            // token with ITS semaphore)
  DenseMap<MemberId, Value> view;           // member view cache
  DenseMap<int64_t, gpu::StageCluster> stageCache; // ownerKey -> stage/cluster
};

// ---------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------

// The ~10-LoC createInto wrapper re-derived per plan ground rule 3: stamp
// partition + stage/cluster; add the WS tag only when the op lands outside
// any WS-tagged loop (spec section 6 stamping rules).
template <typename OpT, typename... Args>
static OpT emitInto(OpBuilder &b, Location loc, const Owner &owner,
                    gpu::StageCluster stageCluster, Args &&...args) {
  std::optional<SetVector<int>> ids = SetVector<int>();
  if (owner)
    ids->insert(owner->first);
  else
    ids = std::nullopt;
  auto op =
      gpu::createInto<OpT>(b, loc, ids, stageCluster,
                           std::forward<Args>(args)...);
  if (owner) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !gpu::hasWarpSpecializeTag(forOp))
      forOp = forOp->template getParentOfType<scf::ForOp>();
    // ROOT-OUTSIDE rule (enabled 12jun26 — semaphore combine is disabled,
    // which removes LowerAref's stamped-acquire assumptions; see
    // fable/attr-less-acquire-release-handoff.md): outside any WS-tagged
    // loop, p0 == root cost domain emits attr-less; a non-zero partition
    // keeps {P} + tag so its stage/phase SSA chain stays inside that
    // warp-group region.
    if (!forOp) {
      if (owner->first == 0)
        op->removeAttr(gpu::kPartitionAttrName);
      else
        gpu::setWarpSpecializeTag(op, owner->second);
    }
  }
  return op;
}

static gpu::StageCluster stageFor(RenderState &rs, const Owner &owner,
                                  Operation *anchor) {
  if (anchor)
    return gpu::getStageCluster(anchor);
  auto it = rs.stageCache.find(ownerKey(owner));
  if (it != rs.stageCache.end())
    return it->second;
  return {};
}

// The next row at-or-after `n` that anchors a real op (insertion target for
// acquires). Returns nullptr when the chain ends (insert before terminator).
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

// Arrive multiplicity (r S(n)) is first-class: the payload union is
// emitted once per release with arrive_count = the DAG count, and the
// lowering passes it to the mbarrier arrive (count > 1 valid for
// none/wgmma kinds only; see fable/integrate-pending-count-plan.md).
static ArrayAttr asyncOpsAttr(MLIRContext *ctx, const Node *rel) {
  SmallVector<Attribute, 2> elems;
  for (AsyncOp p : rel->payloads)
    elems.push_back(nvws::AsyncOpAttr::get(ctx, p));
  return ArrayAttr::get(ctx, elems);
}

// ---------------------------------------------------------------------------
// Step 1 — token-nuke pre-process (contract E).
// ---------------------------------------------------------------------------
static void nukeGroupTokens(EmitCtx &ctx, GroupDag &g) {
  // Member allocs: kill token results (sourceless allocs keep the op; the
  // sourceful ones are replaced at render but their tokens die now too).
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
      // Only when the accumulator belongs to this group (operand-only MMAs
      // carry no token plumbing for this buffer).
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

// ---------------------------------------------------------------------------
// Step 2 — backings, creates, entry acquires.
// ---------------------------------------------------------------------------
// Re-derived from NVWS Utilities.cpp:29-48 (plan ground rule 3): scales
// encodings never gain/drop the depth dimension.
static bool isScalesEnc(gpu::MemDescType t) {
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(t.getEncoding());
}

static gpu::MemDescType backingType(const GroupDag &g, const Member &m) {
  auto t = m.type;
  SmallVector<int64_t> shape(t.getShape());
  if (!isScalesEnc(t))
    shape.insert(shape.begin(), g.backingPlan.numStages);
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

// Generic semaphore view of a backing: drop the depth dim (non-scales),
// keep the backing shape as allocShape.
static gpu::MemDescType genericViewType(gpu::MemDescType backing) {
  auto shape = backing.getShape();
  return gpu::MemDescType::get(isScalesEnc(backing) ? shape
                                                    : shape.drop_front(),
                               backing.getElementType(), backing.getEncoding(),
                               backing.getMemorySpace(),
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

// Local member view type: the ACCESS SITE's own memdesc type, advanced
// through leading memdesc_index alias steps, mutability forced true
// (mining gap 4; old emitter :636-657). Falls back to the generic view.
static gpu::MemDescType localViewType(const GroupDag &g, MemberId member,
                                      ArrayRef<const Touch *> touches,
                                      gpu::MemDescType backing) {
  for (const Touch *t : touches) {
    if (t->member != member)
      continue;
    // Type-walk over stage-1-captured types (values may dangle after
    // sourceful allocs were replaced and erased): start at the member's
    // alloc type, advance through leading memdesc_index steps.
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

static LogicalResult emitBackingsAndCreates(EmitCtx &ctx, GroupDag &g) {
  if (g.semaTable.semas.empty())
    return success(); // untouched group (contract H)
  OpBuilder b(ctx.func);
  // Anchor at the FIRST member alloc (program order): it dominates every
  // access, pre-loop entry acquire, and the WS loop (old pass behavior —
  // creates sit right after the original allocs at function level).
  Operation *anchor = g.pieceTable.members.front().allocOp;
  for (const Member &m : g.pieceTable.members)
    if (m.allocOp->getBlock() == anchor->getBlock() &&
        m.allocOp->isBeforeInBlock(anchor))
      anchor = m.allocOp;
  // Hoist out of enclosing scf.for ops only — above the outermost WS
  // loop — but NEVER across an scf.if: a guarded WS loop keeps backings,
  // creates, and entry acquires inside the involving branch (plan
  // contract A; gate-2 oracle fact 10jun26).
  while (isa<scf::ForOp>(anchor->getParentOp()))
    anchor = anchor->getParentOp();
  b.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();

  g.backingPlan.backing.clear();
  SmallVector<Type> baseTypes;
  for (const Member &m : g.pieceTable.members) {
    auto bt = backingType(g, m);
    Value backing;
    if (g.isTmem()) {
      auto alloc = nvidia_gpu::TMEMAllocOp::create(b, loc, bt, /*src=*/Value());
      backing = alloc.getResult();
    } else {
      auto alloc = gpu::LocalAllocOp::create(b, loc, bt);
      backing = alloc.getResult();
    }
    // Preserve buffer.* attrs verbatim (contract A).
    SmallVector<StringRef, 3> attrNames{kBufferIdAttrName,
                                        kBufferOffsetAttrName, "buffer.copy"};
    for (StringRef name : attrNames)
      if (Attribute a = m.allocOp->getAttr(name))
        backing.getDefiningOp()->setAttr(name, a);
    g.backingPlan.backing.push_back(backing);
    baseTypes.push_back(bt);
  }
  auto semaTy = nvws::SemaphoreType::get(
      b.getContext(), nvws::TypeArrayAttr::get(b.getContext(), baseTypes));
  // Emission order: entry creates (is_released=true) first — downstream
  // stage/phase assignment walks creates in program order (oracle-IR fact:
  // the released create precedes its paired one in the reference output).
  SmallVector<Sema *, 4> order;
  for (Sema &s : g.semaTable.semas)
    if (s.isEntry)
      order.push_back(&s);
  for (Sema &s : g.semaTable.semas)
    if (!s.isEntry)
      order.push_back(&s);
  for (Sema *sp : order) {
    Sema &s = *sp;
    auto create = nvws::SemaphoreCreateOp::create(
        b, loc, semaTy, g.backingPlan.backing, s.isEntry);
    // First-class pending count: the DAG fan-in, transcribed (and
    // verified against the pending-count analysis) by the lowering.
    create.setPendingCountAttr(b.getI32IntegerAttr(s.count));
    s.create = create.getResult();
  }
  return success();
}

// Emit the function-level entry-acquire instances (root-owned acquires of
// entry semaphores). Step 3 needs their tokens as init values.
static void emitEntryAcquires(EmitCtx &ctx, GroupDag &g,
                              DenseMap<Node *, Value> &emitted) {
  forEachNode(g, [&](Node *n) {
    {
      if (n->kind == Node::Acquire && !n->owner.has_value() &&
          g.semaTable.semas[n->sema].isEntry && !emitted.count(n)) {
        Operation *before = nextRealOp(n->next);
        OpBuilder b(ctx.func);
        if (before)
          b.setInsertionPoint(before);
        else
          b.setInsertionPoint(n->parent && n->parent->op
                                  ? n->parent->op->getRegion(0).front()
                                        .getTerminator()
                                  : &ctx.func.getBody().front().back());
        // ROOT-OUTSIDE rule (enabled 12jun26): entry acquires are phase
        // SOURCES — emitted attr-less so they stay in the root block
        // (matching the DAG's `root ; entry` rows); the token reaches the
        // partitioned loop clones by plain SSA capture. See
        // fable/attr-less-acquire-release-handoff.md.
        const Sema &s = g.semaTable.semas[n->sema];
        gpu::StageCluster sc = {};
        auto acq = emitInto<nvws::SemaphoreAcquireOp>(
            b, before ? before->getLoc() : ctx.func.getLoc(), Owner(),
            sc, s.create, ctx.tokenType);
        emitted[n] = acq.getToken();
      }
    }
  });
}



// ---------------------------------------------------------------------------
// Step 3 — aggregated signature rewrites (exactly once per op, outside-in),
// with poison placeholders; render sets the real operands.
// ---------------------------------------------------------------------------
static void fixupAnchors(MutableArrayRef<GroupDag> groups, Operation *oldOp,
                         Operation *newOp) {
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->op == oldOp)
        n->op = newOp;
    });
    if (g.backingPlan.hoistAnchor == oldOp)
      g.backingPlan.hoistAnchor = newOp;
  }
}

// ---------------------------------------------------------------------------
// Step 1b — erase dead token-typed signature slots left by the nuke (spec
// post-nuke row): a scf.for iter_arg whose region arg AND result are both
// unused, or a scf.if result that is unused. Fixpoint: slots feed each
// other across nesting. Inits/yield operands are dropped and matching
// ttg.partition.outputs entries filtered; DAG anchors are fixed up.
// ---------------------------------------------------------------------------
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
    auto newIf = scf::IfOp::create(b, ifOp.getLoc(), keptTypes,
                                   ifOp.getCondition(),
                                   /*withElseRegion=*/!ifOp.getElseRegion()
                                       .empty());
    newIf->setAttrs(ifOp->getAttrs());
    filterPartitionOutputs(newIf, keep);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    for (Region &r : newIf->getRegions()) {
      if (r.empty())
        continue;
      Operation *term = r.front().getTerminator();
      for (int i = keep.size() - 1; i >= 0; --i)
        if (!keep[i])
          term->eraseOperand(i);
    }
    unsigned next = 0;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (keep[i])
        res.replaceAllUsesWith(newIf.getResult(next++));
    fixupAnchors(groups, ifOp, newIf);
    ifOp.erase();
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
    filterPartitionOutputs(newFor, keep);
    newFor.getRegion().takeBody(forOp.getRegion());
    Block &body = newFor.getRegion().front();
    for (int i = keep.size() - 1; i >= 0; --i)
      if (!keep[i]) {
        body.getTerminator()->eraseOperand(i);
        body.eraseArgument(1 + i); // +1: induction variable
      }
    unsigned next = 0;
    for (auto [i, res] : llvm::enumerate(forOp.getResults()))
      if (keep[i])
        res.replaceAllUsesWith(newFor.getResult(next++));
    fixupAnchors(groups, forOp, newFor);
    forOp.erase();
    changed = true;
  }
  return changed;
}

static LogicalResult rewriteSignatures(EmitCtx &ctx,
                                       MutableArrayRef<GroupDag> groups) {
  // Collect per op: (group, comp, slotOwner) triples, group order.
  struct Want {
    GroupDag *g;
    CompId comp;
    Owner owner;
  };
  llvm::MapVector<Operation *, SmallVector<Want, 2>> wanted;
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->kind == Node::For || n->kind == Node::If)
        for (const Crossing &c : n->crossings) {
          // Native point-of-use crossings (plan M2) materialize no slot:
          // the token is born and dies inside the body.
          if (n->kind == Node::For && !c.hold.materializesCarrier())
            continue;
          wanted[n->op].push_back(Want{&g, c.comp, c.slotOwner});
        }
    });
  }
  // Outside-in: stable sort by nesting depth.
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
    // Outputs read from the OLD op (getPartitionOutputs hard-casts the
    // attr; a 0-result op returns empty safely; results without the attr
    // get the op's own partition set, the old pass's fallback).
    SmallVector<SetVector<int>, 4> outputs;
    if (op->getNumResults() == 0 ||
        op->hasAttr(gpu::kPartitionOutputsAttrName))
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
    // partition.outputs: one owner-stamped entry per new slot; a root
    // slot owner falls back to the op's own partition set (the old pass's
    // partitionSetForValue fallback — the attr may not be empty).
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
    // Rebuilt terminators (appendToForOpYield, materialized elses) may
    // lack ttg.partition, which partition-loops requires on every child
    // of a tt.warp_specialize loop. Gap-fill from the op's own array —
    // never overwrite an existing curated set.
    if (gpu::hasPartition(newOp)) {
      auto ids = gpu::getPartitionIds(newOp);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r)
          if (Operation *term = blk.getTerminator())
            if (!gpu::hasPartition(term))
              term->setAttr(
                  gpu::kPartitionAttrName,
                  DenseI32ArrayAttr::get(
                      newOp->getContext(),
                      SmallVector<int>(ids.begin(), ids.end())));
    }
    fixupAnchors(groups, op, newOp);
    auto &rec = ctx.slots[newOp];
    for (auto [i, w] : llvm::enumerate(list))
      rec.push_back(EmitCtx::Slot{w.g, w.comp,
                                  base + static_cast<unsigned>(i)});
  }
  return success();
}

// ---------------------------------------------------------------------------
// Step 4 — render walk (one per group).
// ---------------------------------------------------------------------------
static unsigned slotIndexFor(EmitCtx &ctx, Operation *op, GroupDag *g,
                             CompId comp) {
  for (const EmitCtx::Slot &s : ctx.slots[op])
    if (s.g == g && s.comp == comp)
      return s.index;
  llvm_unreachable("missing slot");
}

// Materialize (or fetch) the view of `member`, replaying the access's alias
// chain (mining gap 4 rules).
static Value getView(EmitCtx &ctx, GroupDag &g, RenderState &rs,
                     const Touch &t, Operation *accessOp, const Owner &owner) {
  auto it = rs.view.find(t.member);
  Value base;
  if (it != rs.view.end()) {
    base = it->second;
  } else {
    // One buffer op yields all member views at once.
    OpBuilder b(accessOp);
    SmallVector<Type> types;
    for (auto [mi, m] : llvm::enumerate(g.pieceTable.members)) {
      auto bt = cast<gpu::MemDescType>(g.backingPlan.backing[mi].getType());
      if (g.isTmem())
        types.push_back(genericViewType(bt));
      else
        types.push_back(localViewType(g, static_cast<MemberId>(mi),
                                      {&t}, bt));
    }
    CompId comp = compOfMember(g, t.member);
    Value tok = rs.carrier.lookup(comp);
    assert(tok && "no carrier for view");
    // Pair the token with ITS OWN semaphore (the acquire that produced the
    // carrier), like the old pass's ping/pong-side selection.
    Value semaVal = rs.carrierSema.lookup(comp);
    assert(semaVal && "no semaphore for carrier");
    auto buf = emitInto<nvws::SemaphoreBufferOp>(
        b, accessOp->getLoc(), owner, gpu::getStageCluster(accessOp), semaVal,
        TypeRange(types), tok);
    // Cache only the views of THIS component's members: the buffer op
    // yields views for every group member, but a view is bound to the
    // carrier token (and partition) it was minted under — serving another
    // component's member from this row would cross token games (the
    // post-emit view-locality verifier rejects exactly that).
    for (auto [mi, v] : llvm::enumerate(buf.getBuffers()))
      if (compOfMember(g, static_cast<MemberId>(mi)) == comp)
        rs.view[static_cast<MemberId>(mi)] = v;
    base = rs.view[t.member];
  }
  // Replay the alias chain (old emitter :763-788): skip memdesc_index
  // steps already folded into the view type; clone with mapping; force the
  // clone's result mutability to its source's.
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
    if (cloned->getNumResults() == 1)
      if (auto rt = dyn_cast<gpu::MemDescType>(
              cloned->getResult(0).getType()))
        cloned->getResult(0).setType(withMutable(
            rt, cast<gpu::MemDescType>(source.getType()).getMutableMemory()));
    cur = cloned->getResult(0);
  }
  return cur;
}

static LogicalResult renderChain(EmitCtx &ctx, GroupDag &g, Node *head,
                                 RenderState &rs,
                                 DenseMap<Node *, Value> &emitted);

// `anchor` reports the op that anchors this row in the emitted IR: the
// original access op, or the synthesized store replacing a sourceful
// alloc (the original is erased; its pointer must not be touched again).
static LogicalResult renderAccess(EmitCtx &ctx, GroupDag &g, Node *n,
                                  RenderState &rs, Operation *&anchor) {
  Operation *op = n->op;
  anchor = op;
  if (n->owner)
    rs.stageCache[ownerKey(n->owner)] = gpu::getStageCluster(op);
  for (const Touch &t : n->touches) {
    Value view = getView(ctx, g, rs, t, op, n->owner);
    // Sourceful allocs become an explicit store into the view (contract D).
    if (auto ta = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      OpBuilder b(op);
      auto pidsc = std::make_pair(n->owner, gpu::getStageCluster(op));
      auto vTrue = emitInto<arith::ConstantOp>(b, op->getLoc(), n->owner,
                                               pidsc.second,
                                               b.getBoolAttr(true));
      anchor = emitInto<nvidia_gpu::TMEMStoreOp>(b, op->getLoc(), n->owner,
                                                 pidsc.second, Type(), view,
                                                 Value(), ta.getSrc(), vTrue);
      // RAUW dominated uses, excluding creates and the new store.
      ta.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               use.getOwner() != view.getDefiningOp() &&
               !g.accessRowOps.contains(use.getOwner());
      });
      // Deferred erase: later access rows of this group still reference
      // the original result until THEY retarget (their own owner's view);
      // the final cleanup erases it once use-empty.
      return success();
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
                                           gpu::getStageCluster(op), src,
                                           view);
      la.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               !g.accessRowOps.contains(use.getOwner());
      });
      // Deferred erase (see TMEMAlloc branch above).
      return success();
    }
    // Plain access: retarget the matching operand(s).
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
  return success();
}

static LogicalResult renderRegion(EmitCtx &ctx, GroupDag &g, Node *n,
                                  RenderState &rs,
                                  DenseMap<Node *, Value> &emitted) {
  // Set this op's slot inits / record incoming carriers.
  for (const Crossing &c : n->crossings) {
    if (n->kind == Node::For && !c.hold.materializesCarrier())
      continue; // native point-of-use: no slot (plan M2)
    unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
    Value incoming = rs.carrier.lookup(c.comp);
    if (!incoming)
      incoming = ctx.poison; // component starts inside this region
    if (auto forOp = dyn_cast<scf::ForOp>(n->op))
      forOp.getInitsMutable()[idx].assign(incoming);
  }
  // requiredParts -> extend the op's partition array (C10), with condition
  // availability verified.
  if (!n->requiredParts.empty() && gpu::hasPartition(n->op)) {
    SetVector<int> set = gpu::getPartitionIds(n->op);
    unsigned before = set.size();
    for (int p : n->requiredParts)
      set.insert(p);
    // Only restamp when the set actually GREW (C10): setPartition also
    // overwrites every region terminator's partition attr, and the input
    // yields carry curated sets that partition-loops consumes — a no-op
    // union must not disturb them.
    if (set.size() != before)
      gpu::setPartition(n->op, set.getArrayRef());
  }

  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    RenderState body = rs; // stage cache flows in; views do not
    body.view.clear();
    for (const Crossing &c : n->crossings) {
      if (!c.hold.materializesCarrier()) {
        // Native point-of-use (plan M2): no incoming carrier — the moved
        // wrap acquire renders before the hold's first toucher and births
        // the body-local token there.
        body.carrier.erase(c.comp);
        continue;
      }
      unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
      body.carrier[c.comp] = forOp.getRegionIterArg(idx);
    }
    if (failed(renderChain(ctx, g, n->children[0], body, emitted)))
      return failure();
    // Yield wiring: body-final carrier per slot (terminator looked up now).
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (const Crossing &c : n->crossings) {
      if (!c.hold.materializesCarrier()) {
        rs.carrier.erase(c.comp); // token died in the body; nothing flows out
        continue;
      }
      unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
      yield->setOperand(idx, body.carrier.lookup(c.comp));
      rs.carrier[c.comp] = forOp.getResult(idx);
    }
    if (!gpu::hasWarpSpecializeTag(forOp))
      rs.stageCache = std::move(body.stageCache);
    // Stage facts flow out of inner (non-WS) loop bodies — the epilogue
    // release after an inner loop inherits its in-body stage (oracle fact,
    // gate-2 case 3) — but never escape the WS-tagged loop itself: outside
    // it, loop.stage/cluster are meaningless and must not be stamped
    // (oracle fact, m2/m3 outside-loop releases).
    rs.view.clear();
    return success();
  }

  auto ifOp = cast<scf::IfOp>(n->op);
  RenderState thenSt = rs, elseSt = rs;
  thenSt.view.clear();
  elseSt.view.clear();
  if (failed(renderChain(ctx, g, n->children[0], thenSt, emitted)))
    return failure();
  if (n->children.size() > 1 && n->children[1])
    if (failed(renderChain(ctx, g, n->children[1], elseSt, emitted)))
      return failure();
  for (const Crossing &c : n->crossings) {
    unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
    Value incoming = rs.carrier.lookup(c.comp);
    if (!incoming)
      incoming = ctx.poison;
    Value thenV = c.finals[0] ? thenSt.carrier.lookup(c.comp) : incoming;
    Value elseV = (c.finals.size() > 1 && c.finals[1])
                      ? elseSt.carrier.lookup(c.comp)
                      : incoming;
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    thenYield->setOperand(idx, thenV);
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    elseYield->setOperand(idx, elseV);
    rs.carrier[c.comp] = ifOp.getResult(idx);
  }
  rs.stageCache = std::move(thenSt.stageCache);
  for (auto &[k, v] : elseSt.stageCache)
    rs.stageCache.try_emplace(k, v);
  rs.view.clear();
  return success();
}

static LogicalResult renderChain(EmitCtx &ctx, GroupDag &g, Node *head,
                                 RenderState &rs,
                                 DenseMap<Node *, Value> &emitted) {
  Operation *lastReal = nullptr;
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
    case Node::Exit:
      break; // markers; yield wiring is the parent's job
    case Node::Acquire: {
      CompId comp = g.semaTable.semas[n->sema].component;
      if (Value v = emitted.lookup(n)) { // pre-rendered entry instance
        rs.carrier[comp] = v;
        rs.carrierSema[comp] = g.semaTable.semas[n->sema].create;
        rs.view.clear();
        break;
      }
      Operation *before = nextRealOp(n->next);
      OpBuilder b(ctx.func);
      Operation *stageAnchor = nullptr;
      if (before) {
        b.setInsertionPoint(before);
        stageAnchor = before;
      } else if (lastReal &&
                 !isa<triton::FuncOp>(lastReal->getBlock()->getParentOp())) {
        // Trailing acquire of a region chain (no following access row):
        // end-of-block, immediately before the terminator — the old
        // emitter's placement, and required by the loop-scheduler if-split
        // (findBranchTrailingAcquire matches only acquire-before-yield).
        b.setInsertionPoint(lastReal->getBlock()->getTerminator());
      } else if (lastReal) {
        // Function-level trailing acquire (post-loop chain): right after
        // the last emitted op (old pass behavior).
        b.setInsertionPointAfter(lastReal);
      } else if (n->parent && n->parent->op) {
        Region &region = n->parent->op->getRegion(0);
        b.setInsertionPoint(region.front().getTerminator());
      }
      auto acq = emitInto<nvws::SemaphoreAcquireOp>(
          b, before ? before->getLoc() : ctx.func.getLoc(), n->owner,
          stageFor(rs, n->owner, stageAnchor),
          g.semaTable.semas[n->sema].create, ctx.tokenType);
      emitted[n] = acq.getToken();
      rs.carrier[comp] = acq.getToken();
      rs.carrierSema[comp] = g.semaTable.semas[n->sema].create;
      rs.view.clear();
      lastReal = acq;
      break;
    }
    case Node::Release: {
      CompId comp = g.semaTable.semas[n->sema].component;
      Value tok = rs.carrier.lookup(comp);
      assert(tok && "release without carrier");
      OpBuilder b(ctx.func);
      if (lastReal)
        b.setInsertionPointAfter(lastReal);
      else if (n->parent && n->parent->op)
        b.setInsertionPointToStart(&n->parent->op->getRegion(0).front());
      else
        b.setInsertionPointToStart(&ctx.func.getBody().front());
      auto rel = emitInto<nvws::SemaphoreReleaseOp>(
          b, lastReal ? lastReal->getLoc() : ctx.func.getLoc(), n->owner,
          stageFor(rs, n->owner, nullptr), g.semaTable.semas[n->sema].create,
          tok, asyncOpsAttr(b.getContext(), n));
      // First-class arrive multiplicity: transcribe the DAG fact (r S(n));
      // the lowering passes it to the mbarrier arrive.
      rel.setArriveCountAttr(b.getI32IntegerAttr(n->count));
      emitted[n] = Value();
      lastReal = rel;
      break;
    }
    case Node::Access: {
      Operation *anchor = nullptr;
      if (failed(renderAccess(ctx, g, n, rs, anchor)))
        return failure();
      if (anchor) {
        lastReal = anchor;
        if (n->owner)
          rs.stageCache[ownerKey(n->owner)] = gpu::getStageCluster(anchor);
      }
      break;
    }
    case Node::For:
    case Node::If:
      if (failed(renderRegion(ctx, g, n, rs, emitted)))
        return failure();
      lastReal = n->op;
      break;
    case Node::Func:
      break;
    }
  }
  return success();
}

// ---------------------------------------------------------------------------
// Step 6 — coalesce planned backings (contract C).
// ---------------------------------------------------------------------------
static bool hasPlannedBufferCopy(const GroupDag &g) {
  return llvm::any_of(g.pieceTable.members, [](const Member &m) {
    return m.allocOp->hasAttr("buffer.copy");
  });
}

static void coalesceBackings(GroupDag &g) {
  if (g.semaTable.semas.empty() || g.pieceTable.members.size() < 2)
    return;
  if (!g.isTmem() && !hasPlannedBufferCopy(g))
    return;
  // Covering member: minimal offset, maximal end.
  unsigned cover = 0;
  for (auto [i, m] : llvm::enumerate(g.pieceTable.members)) {
    const Member &c = g.pieceTable.members[cover];
    if (m.offset <= c.offset && m.offset + m.extent >= c.offset + c.extent)
      cover = i;
  }
  const Member &cm = g.pieceTable.members[cover];
  Value coverBacking = g.backingPlan.backing[cover];
  for (auto [i, m] : llvm::enumerate(g.pieceTable.members)) {
    if (i == cover)
      continue;
    if (m.offset < cm.offset || m.offset + m.extent > cm.offset + cm.extent)
      continue; // not contained; leave as-is
    Value backing = g.backingPlan.backing[i];
    Operation *alloc = backing.getDefiningOp();
    if (!g.isTmem() &&
        (m.offset != cm.offset || backing.getType() != coverBacking.getType()))
      continue;
    // The covering backing may be defined after this member's backing in
    // member order; insert the replacement view AFTER the cover so it
    // dominates every use (creates and views all come later).
    OpBuilder b(coverBacking.getContext());
    b.setInsertionPointAfterValue(coverBacking);
    Value repl;
    if (m.offset == cm.offset &&
        backing.getType() == coverBacking.getType()) {
      repl = coverBacking;
    } else {
      auto sub = nvidia_gpu::TMEMSubSliceOp::create(
          b, alloc->getLoc(), coverBacking,
          static_cast<int32_t>(m.offset - cm.offset),
          /*sizeHint*/ cast<gpu::MemDescType>(backing.getType())
              .getShape()
              .back());
      repl = sub.getResult();
      if (repl.getType() != backing.getType()) {
        auto re = gpu::MemDescReinterpretOp::create(
            b, alloc->getLoc(), backing.getType(), repl);
        repl = re.getResult();
      }
    }
    backing.replaceAllUsesWith(repl);
    alloc->erase();
    g.backingPlan.backing[i] = repl;
  }
}

// ---------------------------------------------------------------------------
// Step 7 — loop-scheduler workaround (separate post-processor; implemented
// against the old pass's splitSemaphoreIfForLoopScheduler).
// ---------------------------------------------------------------------------
// Reimplemented from the OLD pass's splitSemaphoreIfForLoopScheduler
// (5cfe0ac6e7^ InsertSemasEmitter.h:2784-2997) — behavioral obligation:
// make the downstream loop scheduler able to stage conditional sync.
struct IfSplitCandidate {
  scf::IfOp ifOp;
  bool branchIsThen = true;
  nvws::SemaphoreReleaseOp releaseOp;
  nvws::SemaphoreAcquireOp acquireOp;
  unsigned tokenResultIdx = 0;
  bool releaseOnly = false;
};

static nvws::SemaphoreReleaseOp findBranchReleaseForSplit(Block *block) {
  for (Operation &op : *block) {
    if (isa<scf::YieldOp>(op))
      return nullptr;
    if (auto rel = dyn_cast<nvws::SemaphoreReleaseOp>(&op))
      return rel;
    if (isa<nvws::SemaphoreAcquireOp>(op))
      return nullptr;
    if (op.hasTrait<OpTrait::ConstantLike>() || isSupportedAliasOp(&op))
      continue;
    return nullptr;
  }
  return nullptr;
}

static nvws::SemaphoreAcquireOp findBranchTrailingAcquire(Block *block) {
  return dyn_cast_or_null<nvws::SemaphoreAcquireOp>(
      block->getTerminator()->getPrevNode());
}

static bool branchHasAcquireAfter(nvws::SemaphoreReleaseOp rel) {
  for (Operation *op = rel->getNextNode(); op; op = op->getNextNode()) {
    if (isa<scf::YieldOp>(op))
      return false;
    if (isa<nvws::SemaphoreAcquireOp>(op))
      return true;
  }
  return false;
}

static gpu::StageCluster inferPrecedingMmaStage(scf::IfOp ifOp) {
  for (Operation *op = ifOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<nvidia_gpu::MMAv5OpInterface>(op))
      return gpu::getStageCluster(op);
  return {};
}

static bool semaUsesTmem(Value sem) {
  auto ty = dyn_cast<nvws::SemaphoreType>(sem.getType());
  if (!ty || ty.getBaseType().empty())
    return false;
  auto md = dyn_cast<gpu::MemDescType>(ty.getBaseType()[0]);
  return md && isa<nvidia_gpu::TensorMemorySpaceAttr>(md.getMemorySpace());
}

static unsigned semaBaseTypeCount(Value sem) {
  auto ty = dyn_cast<nvws::SemaphoreType>(sem.getType());
  return ty ? ty.getBaseType().size() : 0;
}

static void assignStageIfKnown(OpBuilder &b, Operation *op,
                               gpu::StageCluster sc) {
  if (sc)
    gpu::setStageCluster(b, op, sc);
}

static SetVector<int> partitionSetForValue(Value v) {
  SetVector<int> s;
  if (auto res = dyn_cast<OpResult>(v)) {
    Operation *def = res.getOwner();
    auto outs = gpu::getPartitionOutputs(def);
    if (res.getResultNumber() < outs.size() &&
        !outs[res.getResultNumber()].empty())
      return outs[res.getResultNumber()];
    return partitionIdsOfFwd(def);
  }
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      auto outs = gpu::getPartitionOutputs(forOp);
      unsigned idx = arg.getArgNumber();
      if (idx >= 1 && idx - 1 < outs.size() && !outs[idx - 1].empty())
        return outs[idx - 1];
    }
    return partitionIdsOfFwd(parent);
  }
  return s;
}

static LogicalResult workaroundLoopScheduler(EmitCtx &ctx) {
  SmallVector<IfSplitCandidate> candidates;
  ctx.func.walk([&](scf::IfOp ifOp) {
    if (ifOp.thenBlock()->empty())
      return;
    auto makeCandidate =
        [&](bool branchIsThen,
            bool releaseOnly) -> std::optional<IfSplitCandidate> {
      Block *block = branchIsThen ? ifOp.thenBlock() : nullptr;
      if (!branchIsThen) {
        if (ifOp.getElseRegion().empty())
          return std::nullopt;
        block = ifOp.elseBlock();
      }
      auto rel = findBranchReleaseForSplit(block);
      if (!rel)
        return std::nullopt;
      if (releaseOnly) {
        if (!(semaUsesTmem(rel.getSemaphore()) && branchHasAcquireAfter(rel)))
          return std::nullopt;
        return IfSplitCandidate{ifOp, branchIsThen, rel, {}, 0, true};
      }
      auto acq = findBranchTrailingAcquire(block);
      if (!acq)
        return std::nullopt;
      if (semaUsesTmem(rel.getSemaphore()) &&
          semaBaseTypeCount(rel.getSemaphore()) > 1)
        return std::nullopt;
      auto yieldOp = branchIsThen ? ifOp.thenYield() : ifOp.elseYield();
      std::optional<int> pos;
      for (auto [i, v] : llvm::enumerate(yieldOp->getOperands()))
        if (v == acq.getToken()) {
          pos = static_cast<int>(i);
          break;
        }
      if (!pos)
        return std::nullopt;
      return IfSplitCandidate{ifOp,        branchIsThen,
                              rel,         acq,
                              unsigned(*pos), false};
    };
    for (bool releaseOnly : {false, true})
      for (bool branchIsThen : {true, false})
        if (auto c = makeCandidate(branchIsThen, releaseOnly)) {
          candidates.push_back(*c);
          return;
        }
    // Fallback: acquire-first then-block with release immediately before.
    if (auto acq =
            dyn_cast_or_null<nvws::SemaphoreAcquireOp>(&ifOp.thenBlock()->front())) {
      Operation *prev = ifOp->getPrevNode();
      if (prev && ifOp.getCondition().getDefiningOp() == prev)
        prev = prev->getPrevNode();
      if (auto rel = dyn_cast_or_null<nvws::SemaphoreReleaseOp>(prev)) {
        std::optional<int> pos;
        for (auto [i, v] : llvm::enumerate(ifOp.thenYield()->getOperands()))
          if (v == acq.getToken()) {
            pos = static_cast<int>(i);
            break;
          }
        if (pos)
          candidates.push_back(IfSplitCandidate{ifOp, true, rel, acq,
                                                unsigned(*pos), false});
      }
    }
  });

  for (IfSplitCandidate &c : candidates) {
    scf::IfOp ifOp = c.ifOp;
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();
    // Step 1 — exit-if (release split).
    auto exitIf = scf::IfOp::create(b, loc, TypeRange{}, ifOp.getCondition(),
                                    /*withElseRegion=*/!c.branchIsThen);
    Block *exitBlock = c.branchIsThen ? exitIf.thenBlock() : exitIf.elseBlock();
    c.releaseOp->moveBefore(exitBlock, exitBlock->begin());
    exitIf->setAttrs(ifOp->getAttrs());
    gpu::StageCluster releaseStage = gpu::getStageCluster(c.releaseOp);
    if (!releaseStage)
      releaseStage = inferPrecedingMmaStage(ifOp);
    assignStageIfKnown(b, c.releaseOp, releaseStage);
    assignStageIfKnown(b, exitIf, releaseStage);
    SetVector<int> exitIds = partitionIdsOfFwd(c.releaseOp);
    if (exitIds.empty())
      exitIds = partitionIdsOfFwd(ifOp);
    if (!exitIds.empty())
      gpu::setPartition(exitIf, exitIds.getArrayRef());
    gpu::setPartitionOutputs(exitIf, {});
    if (c.releaseOnly)
      continue;
    // Step 2 — acquire-if.
    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(b, loc, TypeRange{ctx.tokenType},
                                     ifOp.getCondition(),
                                     /*withElseRegion=*/true);
    Block *acqBlock = c.branchIsThen ? enterIf.thenBlock() : enterIf.elseBlock();
    c.acquireOp->moveBefore(acqBlock, acqBlock->begin());
    ifOp.getResult(c.tokenResultIdx).replaceAllUsesWith(enterIf.getResult(0));
    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(
        b, loc,
        ValueRange{c.branchIsThen
                       ? Value(c.acquireOp.getToken())
                       : ifOp.thenYield().getOperand(c.tokenResultIdx)});
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(
        b, loc,
        ValueRange{c.branchIsThen
                       ? ifOp.elseYield().getOperand(c.tokenResultIdx)
                       : Value(c.acquireOp.getToken())});
    // Step 3 — middle-if poison (the sanctioned in-loop poison).
    b.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(b, loc, ctx.tokenType).getResult();
    ifOp.thenYield().setOperand(c.tokenResultIdx, poison);
    ifOp.elseYield().setOperand(c.tokenResultIdx, poison);
    // Step 4 — attrs/stage on the acquire-if.
    enterIf->setAttrs(ifOp->getAttrs());
    gpu::StageCluster acquireStage = gpu::getStageCluster(c.acquireOp);
    assignStageIfKnown(b, enterIf, acquireStage);
    // Step 5 — enter/exit partitions.
    SetVector<int> enterExitIds = partitionIdsOfFwd(c.releaseOp);
    for (int p : partitionIdsOfFwd(c.acquireOp))
      enterExitIds.insert(p);
    if (!enterExitIds.empty()) {
      gpu::setPartition(exitIf, enterExitIds.getArrayRef());
      gpu::setPartition(enterIf, enterExitIds.getArrayRef());
      gpu::setPartitionOutputs(exitIf, {});
      SmallVector<SetVector<int>, 1> enterOutputs{enterExitIds};
      gpu::setPartitionOutputs(enterIf, enterOutputs);
    }
    // Step 6 — middle-if partition metadata: PRESERVED, not re-derived
    // (user ruling 11jun26, refining the upstream-9860c26c port). The
    // authored ttg.partition/ttg.partition.outputs stay verbatim: a
    // multi-result middle if still carries replicated scalar
    // pass-throughs that every original partition's clone must keep
    // computing (tightening to the consumer partition deletes them —
    // caught by the partition-outputs verifier). The token slot needs no
    // outputs entry at all: it is dead after the reroute and the
    // post-split eraseDeadTokenSlots sweep drops it, filtering the
    // surviving slots' authored outputs through filterPartitionOutputs.
    // The PARTITION attr is the one thing the split invalidates (the
    // body was peeled), recomputed as the CONTENT UNION: the remainder
    // (original ids minus enter/exit ids) plus the consumers of every
    // surviving result — a multi-result middle if is a replicated
    // scalar mux each consuming partition's clone must keep (use_acc
    // feeding next iteration's mma); with no surviving results it
    // collapses to the body's partition.
    // PRODUCERS ARE INVARIANT UNDER THE SPLIT (user ruling 11jun26):
    // outputs stay the AUTHORED entries — the split cannot make a
    // partition start producing a value it never produced. The dead
    // token slot's entry leaves with the slot (post-split sweep +
    // filterPartitionOutputs). The partition attr is then exactly what
    // the metadata justifies: body-op owners plus the union of the
    // surviving authored outputs (a replicated select slot authored
    // {0,1} is what legitimately keeps a body-less partition in the
    // set — never a fabricated entry).
    SetVector<int> middleIds;
    for (Operation &op : ifOp.getThenRegion().front()) {
      if (isa<scf::YieldOp>(op))
        continue; // yields mirror the region op; they are not evidence
      for (int p : partitionIdsOfFwd(&op))
        middleIds.insert(p);
    }
    if (!ifOp.getElseRegion().empty())
      for (Operation &op : ifOp.getElseRegion().front()) {
        if (isa<scf::YieldOp>(op))
          continue;
        for (int p : partitionIdsOfFwd(&op))
          middleIds.insert(p);
      }
    auto authored = gpu::getPartitionOutputs(ifOp);
    for (auto [i, o] : llvm::enumerate(authored)) {
      if (i == c.tokenResultIdx)
        continue; // dead after the reroute; dropped by the sweep
      for (int p : o)
        middleIds.insert(p);
    }
    if (!middleIds.empty())
      gpu::setPartition(ifOp, middleIds.getArrayRef());
  }
  return success();
}

// ---------------------------------------------------------------------------
// Driver.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// POST-EMIT PARTITION-OUTPUTS VERIFIER (hard error; user ruling 11jun26):
// ttg.partition.outputs is AUTHORED routing metadata — PartitionLoops
// wires each region result from the partition clone the attribute
// names. For every For/If carrying the attribute, each result slot
// whose yielded value has a determinate producing partition must name
// one of those producers. Indeterminate producers (block args, poison,
// constants, attr-less defs) are skipped — never guessed.
// ---------------------------------------------------------------------------
static LogicalResult verifyPartitionOutputs(triton::FuncOp func) {
  constexpr llvm::StringLiteral kOutputs = "ttg.partition.outputs";
  // Producing partitions of a yielded value (one region-op level deep).
  auto producerIds = [&](Value v) -> SetVector<int> {
    Operation *def = v.getDefiningOp();
    if (!def || isa<ub::PoisonOp>(def) ||
        def->hasTrait<OpTrait::ConstantLike>())
      return {};
    // Nested region results: the value is AVAILABLE in every partition
    // the inner op runs in (its ttg.partition ids) — the inner op's own
    // outputs entry is a routing CHOICE, not the availability set, and
    // nesting levels may legitimately choose different routings for a
    // replicated scalar.
    if (!gpu::hasPartition(def))
      return {};
    return gpu::getPartitionIds(def);
  };
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::IfOp>(op) || !op->hasAttr(kOutputs) ||
        failed(result))
      return;
    auto outputs = gpu::getPartitionOutputs(op);
    if (outputs.size() != op->getNumResults()) {
      result = op->emitError("nvws-insert-semas: partition-outputs "
                             "verifier: attribute has ")
               << outputs.size() << " entries for " << op->getNumResults()
               << " results";
      return;
    }
    SmallVector<Operation *, 2> terms;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      terms.push_back(forOp.getBody()->getTerminator());
    } else {
      auto ifOp = cast<scf::IfOp>(op);
      terms.push_back(ifOp.thenYield());
      if (!ifOp.getElseRegion().empty())
        terms.push_back(ifOp.elseYield());
    }
    for (auto [i, outSet] : llvm::enumerate(outputs))
      for (Operation *term : terms) {
        SetVector<int> prod = producerIds(term->getOperand(i));
        if (prod.empty())
          continue; // indeterminate producer: out of scope, never guessed
        if (llvm::none_of(prod,
                          [&](int p) { return outSet.contains(p); })) {
          std::string have;
          llvm::raw_string_ostream os(have);
          for (int p : prod)
            os << p << " ";
          result = op->emitError("nvws-insert-semas: partition-outputs "
                                 "verifier: result ")
                   << i << " is produced by partition(s) " << os.str()
                   << "but ttg.partition.outputs names none of them";
          return;
        }
      }
  });
  return result;
}

// ---------------------------------------------------------------------------
// POST-EMIT TOKEN/VIEW LOCALITY VERIFIER (hard error; user ruling
// 10jun26): every token consumed by semaphore.buffer/release must trace
// (through loop iter_args and if-results) to acquires of the op's own
// partition — cross-partition tokens are invalid. A buffer's views may
// only be consumed by ops of the view's partition. Attr-less acquires
// (root entry seeds) are the one sanctioned exemption.
// ---------------------------------------------------------------------------
static LogicalResult verifyTokenLocality(triton::FuncOp func) {
  auto idsOf = [](Operation *op) -> std::optional<SmallVector<int, 2>> {
    if (!gpu::hasPartition(op))
      return std::nullopt;
    auto set = gpu::getPartitionIds(op);
    SmallVector<int, 2> v(set.begin(), set.end());
    llvm::sort(v);
    return v;
  };
  std::function<LogicalResult(Operation *, Value, DenseSet<Value> &)> trace =
      [&](Operation *consumer, Value tok,
          DenseSet<Value> &seen) -> LogicalResult {
    if (!seen.insert(tok).second)
      return success();
    if (auto ba = dyn_cast<BlockArgument>(tok)) {
      if (auto forOp =
              dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
        unsigned idx = ba.getArgNumber() - 1; // skip induction var
        if (failed(trace(consumer, forOp.getInits()[idx], seen)))
          return failure();
        auto *yield = forOp.getBody()->getTerminator();
        return trace(consumer, yield->getOperand(idx), seen);
      }
      return success(); // other block args: out of scope
    }
    Operation *def = tok.getDefiningOp();
    if (!def || isa<ub::PoisonOp>(def))
      return success();
    if (auto acq = dyn_cast<nvws::SemaphoreAcquireOp>(def)) {
      auto ap = idsOf(acq), cp = idsOf(consumer);
      if (!ap)
        return success(); // root entry seed exemption
      if (cp && *ap != *cp)
        return consumer->emitError(
                   "nvws-insert-semas: token-locality violation: token "
                   "acquired in partition set differs from consumer's")
                   .attachNote(acq.getLoc())
               << "acquired here";
      return success();
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      unsigned idx = cast<OpResult>(tok).getResultNumber();
      if (failed(trace(consumer, ifOp.thenYield().getOperand(idx), seen)))
        return failure();
      if (!ifOp.getElseRegion().empty())
        return trace(consumer, ifOp.elseYield().getOperand(idx), seen);
      return success();
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned idx = cast<OpResult>(tok).getResultNumber();
      if (failed(trace(consumer, forOp.getInits()[idx], seen)))
        return failure();
      auto *yield = forOp.getBody()->getTerminator();
      return trace(consumer, yield->getOperand(idx), seen);
    }
    return success();
  };
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    Value tok;
    if (auto rel = dyn_cast<nvws::SemaphoreReleaseOp>(op))
      tok = rel.getToken();
    else if (auto buf = dyn_cast<nvws::SemaphoreBufferOp>(op))
      tok = buf.getToken();
    else
      return;
    DenseSet<Value> seen;
    if (failed(trace(op, tok, seen)))
      result = failure();
    if (auto buf = dyn_cast<nvws::SemaphoreBufferOp>(op)) {
      auto bp = idsOf(op);
      if (bp)
        for (Value view : buf->getResults())
          for (Operation *user : view.getUsers()) {
            auto up = idsOf(user);
            if (up && *up != *bp) {
              user->emitError("nvws-insert-semas: view-locality violation: "
                              "view consumed outside its partition")
                  .attachNote(op->getLoc())
                  << "view materialized here";
              result = failure();
            }
          }
    }
  });
  return result;
}

// Emission constraints recorded for the hold-rule work (plan M0.3):
//  - protocol ops (semaphore acquire/release/buffer) must never be emitted
//    inside a PARTITION-LESS scf.if: AssignStagePhase's assignStateInIfOp
//    requires partition metadata and aborts on root conditional pre/post-WS
//    code containing protocol ops;
//  - root-side anchors HOIST ABOVE predicates rather than render under
//    them (the entry acquire above a conditional init is what makes the
//    adopted token dominate) — current emitter behavior, load-bearing.

// Verifier (plan M0.2b; spec Addendum B.3(b)): no token may have a buffer
// view AFTER a release in the same block — such an access sits outside its
// hold's span and rides on incidental program order (Addendum A's hazard
// class). Natively emitted protocols cannot contain the shape, so its
// appearance is a hard error, never repaired.
static LogicalResult verifyNoUseAfterRelease(triton::FuncOp funcOp) {
  auto checkToken = [](Value tok) -> LogicalResult {
    llvm::SmallDenseMap<Block *, SmallVector<Operation *, 4>> byBlock;
    for (Operation *u : tok.getUsers())
      byBlock[u->getBlock()].push_back(u);
    for (auto &[block, users] : byBlock) {
      llvm::sort(users, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      bool sawRelease = false;
      for (Operation *u : users) {
        if (isa<nvws::SemaphoreReleaseOp>(u))
          sawRelease = true;
        else if (sawRelease && isa<nvws::SemaphoreBufferOp>(u))
          return u->emitError(
              "nvws-insert-semas: token has a buffer view after its release "
              "(use-after-release; spec fable/semas-report3.md Addendum "
              "B.3(b))");
      }
    }
    return success();
  };
  auto res = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto acq = dyn_cast<nvws::SemaphoreAcquireOp>(op))
      if (failed(checkToken(acq.getToken())))
        return WalkResult::interrupt();
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      for (BlockArgument arg : forOp.getRegionIterArgs())
        if (isa<gpu::AsyncTokenType>(arg.getType()) &&
            failed(checkToken(arg)))
          return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

// Resolve the acquire feeding a value, looking through scf.if results (the
// conditional re-acquire shape yields the token out of a branch). Bounded
// fuel: the chain descends nested ifs only.
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

// Verifier (plan M0.2a; spec Addendum B.3(a)): at most ONE carrier token
// slot per semaphore group per loop. AssignStagePhase scopes its stage
// threading by backing buffer; a second loop-carried slot of one group
// crashes propagateStage (DenseMap::at on an unthreaded key). Assert here,
// where the offending loop and backing are nameable.
static LogicalResult verifySingleCarrierPerGroup(triton::FuncOp funcOp) {
  auto res = funcOp.walk([&](scf::ForOp forOp) -> WalkResult {
    llvm::SmallDenseMap<Value, unsigned> slotsPerBacking;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      if (!isa<gpu::AsyncTokenType>(arg.getType()))
        continue;
      unsigned idx = arg.getArgNumber() - 1; // skip induction variable
      // The live carrier definition is the yielded value; dead/converted
      // slots yield poison and resolve to nothing.
      nvws::SemaphoreAcquireOp acq =
          resolveAcquireThroughIfs(yieldOp.getOperand(idx));
      if (!acq)
        continue;
      auto create =
          acq.getSemaphore().getDefiningOp<nvws::SemaphoreCreateOp>();
      if (!create || create.getBuffers().empty())
        continue;
      Value backing = create.getBuffers().front();
      if (++slotsPerBacking[backing] > 1) {
        forOp.emitError(
            "nvws-insert-semas: two carrier token slots for one semaphore "
            "group in a single loop (spec fable/semas-report3.md Addendum "
            "B.3(a)); AssignStagePhase cannot thread this");
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

LogicalResult emitIR(triton::FuncOp funcOp,
                            MutableArrayRef<GroupDag> groups) {
  EmitCtx ctx;
  ctx.func = funcOp;
  ctx.tokenType = gpu::AsyncTokenType::get(funcOp.getContext());
  {
    OpBuilder b(&funcOp.getBody().front(), funcOp.getBody().front().begin());
    ctx.poison =
        ub::PoisonOp::create(b, funcOp.getLoc(), ctx.tokenType).getResult();
  }
  // Step 1.
  for (GroupDag &g : groups)
    if (!g.semaTable.semas.empty()) {
      nukeGroupTokens(ctx, g);
      forEachNode(g, [&](Node *n) {
        if (n->kind == Node::Access && n->op)
          g.accessRowOps.insert(n->op);
      });
    }
  // Step 1b: erase the dead token slots the nuke leaves behind (fixpoint).
  while (eraseDeadTokenSlots(ctx, groups)) {
  }
  // Step 2.
  DenseMap<Node *, Value> emitted;
  for (GroupDag &g : groups)
    if (failed(emitBackingsAndCreates(ctx, g)))
      return failure();
  for (GroupDag &g : groups)
    if (!g.semaTable.semas.empty())
      emitEntryAcquires(ctx, g, emitted);
  // Step 3.
  if (failed(rewriteSignatures(ctx, groups)))
    return failure();
  // Step 4.
  for (GroupDag &g : groups) {
    if (g.semaTable.semas.empty())
      continue;
    RenderState rs;
    // Seed function-level carriers from entry tokens.
    std::function<void(Node *)> seed = [&](Node *head) {
      for (Node *n = head; n; n = n->next)
        if (n->kind == Node::Acquire && emitted.count(n)) {
          rs.carrier[g.semaTable.semas[n->sema].component] =
              emitted.lookup(n);
          rs.carrierSema[g.semaTable.semas[n->sema].component] =
              g.semaTable.semas[n->sema].create;
        }
    };
    if (!g.root->children.empty())
      seed(g.root->children[0]);
    if (failed(renderChain(ctx, g, g.root->children[0], rs, emitted)))
      return failure();
  }
  // Step 6.
  for (GroupDag &g : groups)
    coalesceBackings(g);
  // Step 7.
  {
    if (failed(workaroundLoopScheduler(ctx)))
      return failure();
    // The split reroutes the middle if's token result to the enter-if and
    // leaves a dead poison slot behind; a result slot is metadata nobody
    // reads (user ruling 11jun26: drop it rather than label it). The same
    // keep-mask rebuild as step 1b carries the surviving slots' AUTHORED
    // partition.outputs through filterPartitionOutputs.
    while (eraseDeadTokenSlots(ctx, groups)) {
    }
  }
  // Erase dead alias-view chains (fixpoint, leaf-first): retargeting left
  // the original memdesc view ops dead, and they pin the original allocs
  // (and carry stale partition stamps that partition-loops rejects as
  // cross-partition SSA edges — gate-2 case 3, 10jun26).
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
  // Erase fully-retargeted original member allocs (uses-before-defs order:
  // these are leaves once their results are unused).
  for (GroupDag &g : groups) {
    if (g.semaTable.semas.empty())
      continue;
    for (const Member &m : g.pieceTable.members)
      if (m.allocOp && m.allocOp->getBlock() && m.allocOp->use_empty())
        m.allocOp->erase();
  }
  // Point-of-use placement is NATIVE as of plan M2 (spec Addendum B.2.2):
  // the SYNC-DAG anchors ungated wrap acquires at the hold's first toucher
  // (applyHoldRulePlacement) and materializes no slot for them. The former
  // post-emission rewrite (rewriteCarriedTokensToPointOfUse) and its M1
  // oracle cross-check are deleted.
  // Poison cleanup: if unused, drop it.
  if (ctx.poison.use_empty())
    ctx.poison.getDefiningOp()->erase();
  // Post-emit token/view locality subpass (user ruling 10jun26).
  if (failed(verifyPartitionOutputs(funcOp)))
    return failure();
  if (failed(verifyTokenLocality(funcOp)))
    return failure();
  // Hold-rule seatbelts (plan M0.2; spec Addendum B.3).
  if (failed(verifyNoUseAfterRelease(funcOp)))
    return failure();
  if (failed(verifySingleCarrierPerGroup(funcOp)))
    return failure();
  return success();
}

} // namespace nvws_semas
} // namespace triton
} // namespace mlir
