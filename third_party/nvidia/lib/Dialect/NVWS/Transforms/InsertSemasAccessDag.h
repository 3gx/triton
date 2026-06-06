#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_

// ---------------------------------------------------------------------------
// Discovery helpers (buffer.id / offset / overlap classes).
// ---------------------------------------------------------------------------

// Layout-correct extent. Delegates to `mlir::triton::getMemDescSize` so
// TMEM allocs report columns (from tensor_memory_encoding) and SMEM
// allocs report bytes (product(shape) * elementBitWidth/8). The
// `buffer.offset` attr on each alloc is in the same native unit.
static int64_t getAllocExtent(MemDescType type) {
  return static_cast<int64_t>(mlir::triton::getMemDescSize(type));
}

static bool intervalsOverlap(int64_t aLo, int64_t aHi, int64_t bLo,
                             int64_t bHi) {
  return aLo < bHi && bLo < aHi;
}

static bool isTmemAlloc(Operation *op) { return isa<TMEMAllocOp>(op); }
static bool isLocalAlloc(Operation *op) { return isa<LocalAllocOp>(op); }

// A local alloc is a candidate semaphore-managed backing buffer if its
// memdesc type has the multi-stage layout that insert-allocas chose.
// Heuristic that matches the prior implementation: the alloc result type
// has a rank greater than the source tensor's rank (the leading dim
// is the multi-stage depth).
static bool isLocalSemaphoreBackingType(MemDescType type) {
  return type.getMutableMemory();
}

// ---------------------------------------------------------------------------
// Alias chain + touch construction (v4 §Access Events).
// ---------------------------------------------------------------------------

static FailureOr<AliasInfo> lookupAlias(BufferGroup &group, Value v) {
  auto it = group.aliases.find(v);
  if (it == group.aliases.end()) return failure();
  return it->second;
}

static LogicalResult addAlias(BufferGroup &group, Operation *op) {
  if (op->getNumResults() != 1) return success();
  if (!isa<MemDescType>(op->getResult(0).getType())) return success();

  std::optional<AliasInfo> source;
  unsigned sourceOperand = 0;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    auto it = group.aliases.find(operand);
    if (it == group.aliases.end()) continue;
    source = it->second;
    sourceOperand = idx;
    break;
  }
  if (!source) return success();

  if (!isSupportedAliasOp(op)) {
    if (group.isTmem())
      return op->emitError("nvws-insert-semas: unsupported memdesc alias use ")
             << op->getName();
    return op->emitError("local semaphore: unsupported SMEM memdesc alias use ")
           << op->getName();
  }

  AliasInfo alias = *source;
  alias.steps.push_back({op, sourceOperand});
  group.aliases.insert({op->getResult(0), alias});
  group.aliasOps.push_back(op);
  return success();
}

static void addTouch(BufferGroup &group, AccessEvent &event, Value v,
                     AccessEffect effect) {
  auto alias = lookupAlias(group, v);
  if (failed(alias)) return;
  BufferMember &member = group.members[alias->memberIdx];
  event.touches.push_back(
      {alias->memberIdx, member.resourceKey, effect, v, *alias});
}

static bool aliasesSameResource(BufferGroup &group, Value a, Value b) {
  auto aAlias = lookupAlias(group, a);
  auto bAlias = lookupAlias(group, b);
  if (failed(aAlias) || failed(bAlias))
    return false;
  if (aAlias->memberIdx >= group.members.size() ||
      bAlias->memberIdx >= group.members.size())
    return false;
  return group.members[aAlias->memberIdx].resourceKey ==
         group.members[bAlias->memberIdx].resourceKey;
}

static bool isAccumulatorImmediatelyOverwritten(BufferGroup &group,
                                                MMAv5OpInterface mma) {
  Operation *op = mma.getOperation();
  if (!op || !op->getBlock() || op->getNumResults() == 0)
    return false;
  Operation *next = op->getNextNode();
  if (!next)
    return false;
  auto store = dyn_cast<TMEMStoreOp>(next);
  if (!store || store.getDep() != op->getResult(0))
    return false;
  return aliasesSameResource(group, mma.getAccumulator(), store.getDst()) &&
         sameOwner(getPartitionId(op), getPartitionId(store));
}

// ---------------------------------------------------------------------------
// Physical conflict key (v4 §Physical Conflict Key).
// Union-find over members of the same group: members whose offset intervals
// overlap share a resourceKey.
// ---------------------------------------------------------------------------

// Union-find members of one buffer.id group by overlap in their native
// (memory-space) interval [offset, offset + size). Works uniformly for
// TMEM (columns) and SMEM (bytes) because `getAllocExtent` selects the
// right unit per memory space.
static void assignResourceKeys(BufferGroup &group) {
  SmallVector<int64_t> parent(group.members.size());
  for (auto [idx, _] : llvm::enumerate(group.members)) parent[idx] = idx;
  std::function<int64_t(int64_t)> find = [&](int64_t i) -> int64_t {
    if (parent[i] == i) return i;
    parent[i] = find(parent[i]);
    return parent[i];
  };
  auto unite = [&](int64_t a, int64_t b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[b] = a;
  };
  for (unsigned i = 0; i < group.members.size(); ++i)
    for (unsigned j = i + 1; j < group.members.size(); ++j) {
      BufferMember &lhs = group.members[i];
      BufferMember &rhs = group.members[j];
      // Plan §Physical Conflict Key: members whose native intervals overlap
      // MUST share a resourceKey (overlap ⇒ same key), with no exception.
      // A reuse handoff between overlapping members of different element
      // types (e.g. an f32 MMA accumulator whose columns are reused to stage
      // an f16 MMA operand) is still a physical conflict and must be
      // synchronized through the shared resource; separating them by element
      // type would silently drop the reuse edge and race.
      if (intervalsOverlap(lhs.offset, lhs.offset + lhs.extent, rhs.offset,
                           rhs.offset + rhs.extent))
        unite(i, j);
    }
  for (auto [idx, member] : llvm::enumerate(group.members))
    member.resourceKey = find(idx);
}

// ---------------------------------------------------------------------------
// Discovery: build one SmallVector<BufferGroup> covering BOTH ttng.tmem_alloc
// and ttg.local_alloc, uniformly. v4 §Uniform Access-DAG Builder.
// ---------------------------------------------------------------------------

template <typename AllocOpT>
static BufferGroup makeGroup(MemorySpaceKind memory, int64_t logicalId,
                             MutableArrayRef<AllocOpT> allocs) {
  BufferGroup group;
  group.memory = memory;
  group.logicalId = logicalId;
  for (auto [idx, allocOp] : llvm::enumerate(allocs)) {
    auto type = cast<MemDescType>(allocOp.getResult().getType());
    BufferMember member;
    member.allocOp = allocOp;
    member.value = allocOp.getResult();
    member.type = type;
    member.offset = getBufferOffset(allocOp);
    member.extent =
        memory == MemorySpaceKind::Local && !type.getShape().empty()
            ? type.getShape().front()
            : getAllocExtent(type);
    group.aliases.insert(
        {allocOp.getResult(), AliasInfo{static_cast<unsigned>(idx), {}}});
    group.members.push_back(member);
  }
  assignResourceKeys(group);
  return group;
}

static SmallVector<BufferGroup, 0>
collectAllBackingGroups(triton::FuncOp funcOp) {
  SmallVector<BufferGroup, 0> groups;
  int64_t nextSyntheticId = 0;

  // TMEM: group by buffer.id.
  llvm::MapVector<int64_t, SmallVector<TMEMAllocOp>> tmemBuckets;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    std::optional<int64_t> id = getBufferId(allocOp);
    int64_t key = id.value_or(nextSyntheticId++);
    tmemBuckets[key].push_back(allocOp);
  });
  for (auto &[id, allocs] : tmemBuckets)
    groups.push_back(makeGroup<TMEMAllocOp>(MemorySpaceKind::Tmem, id, allocs));

  // Local: group by buffer.id (same pattern as TMEM). Allocs without a
  // buffer.id attr get a fresh synthetic id so they remain singleton
  // groups. Members within a shared buffer.id bucket whose
  // [offset, offset+extent) intervals overlap union into one resourceKey.
  llvm::MapVector<int64_t, SmallVector<LocalAllocOp>> localBuckets;
  funcOp.walk([&](LocalAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp)) return;
    if (!isLocalSemaphoreBackingType(cast<MemDescType>(allocOp.getType())))
      return;
    std::optional<int64_t> id = getBufferId(allocOp);
    int64_t key = id.value_or(nextSyntheticId++);
    localBuckets[key].push_back(allocOp);
  });
  for (auto &[id, allocs] : localBuckets)
    groups.push_back(
        makeGroup<LocalAllocOp>(MemorySpaceKind::Local, id, allocs));

  return groups;
}

// ---------------------------------------------------------------------------
// Access-event collection (v4 §Access Events). Walks the function in
// program order; for each terminal access op produces an AccessEvent with
// per-member touches.
// ---------------------------------------------------------------------------

static LogicalResult collectEvents(BufferGroup &group, triton::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (isSemaphoreBackingAlloc(op)) return WalkResult::advance();

    if (failed(addAlias(group, op))) return WalkResult::interrupt();
    if (isSupportedAliasOp(op)) return WalkResult::advance();

    AccessEvent event;
    event.op = op;
    event.owner = getPartitionId(op);
    event.tagSourceOp = event.owner ? getTagSourceOp(op) : nullptr;

    if (group.isTmem()) {
      if (auto allocOp = dyn_cast<TMEMAllocOp>(op)) {
        auto alias = lookupAlias(group, allocOp.getResult());
        if (succeeded(alias) && allocOp.getSrc()) {
          event.sourcefulAllocStore = true;
          addTouch(group, event, allocOp.getResult(), AccessEffect::Write);
        }
      } else if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
        addTouch(group, event, loadOp.getSrc(), AccessEffect::Read);
      } else if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
        addTouch(group, event, storeOp.getDst(), AccessEffect::Write);
      } else if (auto mma = dyn_cast<MMAv5OpInterface>(op)) {
        // v4: mma is always W on its accumulator regardless of
        // useAccumulator. The read-side semantics inside mma are handled
        // by single-partition program order; for cross-owner sync only
        // the overwrite matters.
        bool accumulatorImmediatelyOverwritten =
            isAccumulatorImmediatelyOverwritten(group, mma);
        for (Value operand : op->getOperands()) {
          auto alias = lookupAlias(group, operand);
          if (failed(alias)) continue;
          if (operand == mma.getAccumulator() && accumulatorImmediatelyOverwritten)
            continue;
          AccessEffect effect = operand == mma.getAccumulator()
                                    ? AccessEffect::Write
                                    : AccessEffect::Read;
          addTouch(group, event, operand, effect);
        }
      }
    } else {
      if (auto allocOp = dyn_cast<LocalAllocOp>(op)) {
        auto alias = lookupAlias(group, allocOp.getResult());
        if (succeeded(alias) && allocOp.getSrc()) {
          event.sourcefulAllocStore = true;
          addTouch(group, event, allocOp.getResult(), AccessEffect::Write);
        }
      } else if (auto storeOp = dyn_cast<LocalStoreOp>(op)) {
        addTouch(group, event, storeOp.getDst(), AccessEffect::Write);
      } else if (auto loadOp = dyn_cast<LocalLoadOp>(op)) {
        addTouch(group, event, loadOp.getSrc(), AccessEffect::Read);
      } else if (auto descLoad =
                     dyn_cast<triton::nvws::DescriptorLoadOp>(op)) {
        addTouch(group, event, descLoad.getResult(), AccessEffect::Write);
      } else if (auto descGather =
                     dyn_cast<triton::nvws::DescriptorGatherOp>(op)) {
        addTouch(group, event, descGather.getResult(), AccessEffect::Write);
      } else {
        for (Value operand : op->getOperands())
          addTouch(group, event, operand, AccessEffect::Read);
      }
    }

    if (!event.touches.empty()) group.events.push_back(std::move(event));
    return WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}


using mlir::triton::gpu::getWarpSpecializeTag;
using mlir::triton::gpu::hasWarpSpecializeTag;

std::optional<int> getAnchorWsScopeTag(Operation *op) {
  if (!op)
    return std::nullopt;
  if (isa<scf::ForOp>(op) && hasWarpSpecializeTag(op))
    return *getWarpSpecializeTag(op);
  Operation *p = op->getParentOfType<scf::ForOp>();
  while (p && !hasWarpSpecializeTag(p))
    p = p->getParentOfType<scf::ForOp>();
  if (!p)
    return std::nullopt;
  return *getWarpSpecializeTag(p);
}

std::string ownerStr(Operation *anchor, std::optional<PartitionId> owner) {
  if (!owner)
    return "root";
  std::string s;
  llvm::raw_string_ostream os(s);
  auto anchorTag = anchor ? getAnchorWsScopeTag(anchor) : std::nullopt;
  if (anchorTag && *anchorTag == owner->second)
    os << "{" << owner->first << "}";
  else
    os << "{@" << owner->second << "." << owner->first << "}";
  return s;
}

char accessKindChar(bool reads, bool writes) { return writes ? 'W' : 'R'; }

std::string treePrefix(unsigned depth) {
  std::string s;
  for (unsigned i = 0; i < depth; ++i)
    s += "|  ";
  return s;
}

std::string forOpLabel(scf::ForOp forOp) {
  if (!hasWarpSpecializeTag(forOp))
    return "scf.for";
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "scf.for (WS, tag=" << *getWarpSpecializeTag(forOp) << ")";
  return s;
}

bool accessSubtreeHasEvent(Operation *op,
                           DenseMap<Operation *, unsigned> &eventIdxByOp) {
  bool found = false;
  op->walk([&](Operation *o) -> WalkResult {
    if (eventIdxByOp.count(o)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

void dumpAccessDagBlock(Block &block, BufferGroup &group,
                        DenseMap<Operation *, unsigned> &eventIdxByOp,
                        unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!accessSubtreeHasEvent(&op, eventIdxByOp))
        continue;
      llvm::errs() << treePrefix(depth) << "|- " << forOpLabel(forOp) << "\n";
      for (Block &b : forOp.getRegion())
        dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (!accessSubtreeHasEvent(&op, eventIdxByOp))
        continue;
      llvm::errs() << treePrefix(depth) << "|- scf.if\n";
      for (Block &b : ifOp.getThenRegion())
        dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      if (!ifOp.getElseRegion().empty())
        for (Block &b : ifOp.getElseRegion())
          dumpAccessDagBlock(b, group, eventIdxByOp, depth + 1);
      continue;
    }
    auto it = eventIdxByOp.find(&op);
    if (it == eventIdxByOp.end())
      continue;
    AccessEvent &event = group.events[it->second];
    for (AccessTouch &touch : event.touches) {
      bool reads = hasRead(touch.effect);
      bool writes = hasWrite(touch.effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  "
                   << "m" << touch.memberIdx << "  "
                   << op.getName().getStringRef() << " "
                   << ownerStr(&op, event.owner) << "\n";
    }
  }
}


void dumpBackingGroupHeader(BufferGroup &group) {
  llvm::errs() << "NVWS-SEMA-DAG buffer.id=" << group.logicalId
               << " memory=" << (group.isTmem() ? "tmem" : "local") << "\n";
  llvm::errs() << "  members:";
  for (auto [idx, member] : llvm::enumerate(group.members)) {
    llvm::errs() << " m" << idx << "(offset=" << member.offset
                 << ",extent=" << member.extent
                 << ",resourceKey=" << member.resourceKey << ")";
  }
  llvm::errs() << "\n";
}

void dumpAccessDag(BufferGroup &group, mlir::triton::FuncOp funcOp) {
  DenseMap<Operation *, unsigned> eventIdxByOp;
  for (auto [idx, event] : llvm::enumerate(group.events))
    eventIdxByOp[event.op] = static_cast<unsigned>(idx);
  llvm::errs() << "ACCESS-DAG\n";
  for (Block &b : funcOp.getBody())
    dumpAccessDagBlock(b, group, eventIdxByOp, /*depth=*/0);
}

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
