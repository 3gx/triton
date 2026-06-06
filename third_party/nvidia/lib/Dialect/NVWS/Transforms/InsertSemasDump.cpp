#include "InsertSemasModel.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace mlir::triton::nvws::insert_semas {
namespace {

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

bool regionHasEvents(Region &region, ResourcePlan &plan) {
  auto it = plan.regionOwners.find(&region);
  return it != plan.regionOwners.end() && it->second.hasEventsInSubtree;
}

std::string regionOpLabel(Operation *op, const ResourcePlan &plan) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (hasWarpSpecializeTag(op))
      os << "scf.for (WS, tag=" << *getWarpSpecializeTag(op) << ")";
    else
      os << "scf.for";
  } else if (isa<scf::IfOp>(op)) {
    os << "scf.if";
  }
  auto part = regionOpPartition(op, plan);
  os << " " << ownerStr(op, part);
  return s;
}

void renderEnterRow(unsigned depth, Operation *anchor,
                    std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- ENTER " << ownerStr(anchor, partition)
               << "\n";
}

void renderYieldRow(unsigned depth, Operation *anchor,
                    std::optional<PartitionId> partition) {
  llvm::errs() << treePrefix(depth) << "|- YIELD " << ownerStr(anchor, partition)
               << "\n";
}

void dumpOwnershipBlock(Block &block, ResourcePlan &plan, BufferGroup &group,
                        unsigned depth);

void dumpOwnershipRegion(Region &region, ResourcePlan &plan,
                         BufferGroup &group, Operation *anchorOp,
                         unsigned depth) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end())
    part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region)
    dumpOwnershipBlock(b, plan, group, depth);
  renderYieldRow(depth, anchorOp, part);
}

void dumpOwnershipBlock(Block &block, ResourcePlan &plan, BufferGroup &group,
                        unsigned depth) {
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (!regionHasEvents(forOp.getRegion(), plan))
        continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpOwnershipRegion(forOp.getRegion(), plan, group, &op, depth + 1);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool thenHas = regionHasEvents(ifOp.getThenRegion(), plan);
      bool elseHas = regionHasEvents(ifOp.getElseRegion(), plan);
      if (!thenHas && !elseHas)
        continue;
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpOwnershipRegion(ifOp.getThenRegion(), plan, group, &op, depth + 2);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpOwnershipRegion(ifOp.getElseRegion(), plan, group, &op, depth + 2);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op))
      continue;
    auto useIt = plan.useOwner.find(&op);
    if (useIt == plan.useOwner.end())
      continue;
    unsigned tIdx = plan.useTouchIdx.lookup(&op);
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    if (!event || tIdx >= event->touches.size())
      continue;
    SmallVector<unsigned, 1> fallbackTouchIdx{tIdx};
    auto allTouchIdxs = plan.useTouchIdxs.find(&op);
    ArrayRef<unsigned> touchIdxs =
        allTouchIdxs == plan.useTouchIdxs.end()
            ? ArrayRef<unsigned>(fallbackTouchIdx)
            : ArrayRef<unsigned>(allTouchIdxs->second);
    for (unsigned touchIdx : touchIdxs) {
      if (touchIdx >= event->touches.size())
        continue;
      AccessTouch &touch = event->touches[touchIdx];
      bool reads = hasRead(touch.effect);
      bool writes = hasWrite(touch.effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m" << touch.memberIdx
                   << "  " << op.getName().getStringRef() << "  use "
                   << ownerStr(&op, useIt->second) << "\n";
    }
  }
}

void renderAcquireRow(unsigned depth, Operation *anchor, const SyncEdge &edge,
                      StringRef name) {
  llvm::errs() << treePrefix(depth) << "|- a  " << name << "  acquire  "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

void renderReleaseRow(unsigned depth, Operation *anchor, const SyncEdge &edge,
                      StringRef name) {
  llvm::errs() << treePrefix(depth) << "|- r  " << name << "  release  "
               << ownerStr(anchor, edge.srcOwner) << " -> "
               << ownerStr(anchor, edge.dstOwner) << "\n";
}

StringRef canonicalSemaName(const SyncPlan &sp, unsigned idx) {
  if (sp.semaRep.empty() || idx >= sp.edges.size())
    return sp.edges[idx].name;
  unsigned rep = sp.semaFind(idx);
  return rep < sp.edges.size() ? StringRef(sp.edges[rep].name)
                               : StringRef(sp.edges[idx].name);
}

std::string ownerSetStr(Operation *anchor,
                        ArrayRef<std::optional<PartitionId>> owners);

struct OptDumpCtx {
  const OptSyncDag *dag;
  DenseSet<unsigned> rendered;
};

void printEdgesAt(SmallVector<unsigned, 2> *edgeIdxs, SyncPlan &sp,
                  Operation *anchor, unsigned depth,
                  OptDumpCtx *octx = nullptr) {
  if (!edgeIdxs)
    return;
  if (!octx) {
    for (unsigned idx : *edgeIdxs) {
      StringRef nm = canonicalSemaName(sp, idx);
      renderReleaseRow(depth, anchor, sp.edges[idx], nm);
      renderAcquireRow(depth, anchor, sp.edges[idx], nm);
    }
    return;
  }
  const OptSyncDag &dag = *octx->dag;
  SmallVector<unsigned, 2> faninHere;
  for (unsigned idx : *edgeIdxs) {
    const SyncEdge &edge = sp.edges[idx];
    unsigned gi = dag.edgeToGroup[idx];
    const SyncGroup &g = dag.groups[gi];
    if (g.kind == SyncGroupKind::ReadyFanout) {
      StringRef nm = canonicalSemaName(sp, g.edgeIdxs.front());
      if (octx->rendered.insert(gi).second) {
        SmallVector<std::optional<PartitionId>, 4> dsts;
        for (unsigned ei : g.edgeIdxs)
          dsts.push_back(sp.edges[ei].dstOwner);
        llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                     << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].srcOwner)
                     << " -> " << ownerSetStr(anchor, dsts) << "\n";
      }
      llvm::errs() << treePrefix(depth) << "|- a  " << nm << "  acquire  "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
    } else if (g.kind == SyncGroupKind::DoneFanin) {
      StringRef nm = canonicalSemaName(sp, idx);
      llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                   << ownerStr(anchor, edge.srcOwner) << " -> "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
      if (!llvm::is_contained(faninHere, gi))
        faninHere.push_back(gi);
    } else {
      StringRef nm = canonicalSemaName(sp, idx);
      llvm::errs() << treePrefix(depth) << "|- r  " << nm << "  release  "
                   << ownerStr(anchor, edge.srcOwner) << " -> "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
      llvm::errs() << treePrefix(depth) << "|- a  " << nm << "  acquire  "
                   << ownerStr(anchor, edge.dstOwner) << "\n";
    }
  }
  for (unsigned gi : faninHere) {
    if (!octx->rendered.insert(gi).second)
      continue;
    const SyncGroup &g = dag.groups[gi];
    SmallVector<std::optional<PartitionId>, 4> srcs;
    for (unsigned ei : g.edgeIdxs)
      srcs.push_back(sp.edges[ei].srcOwner);
    llvm::errs() << treePrefix(depth) << "|- a  "
                 << canonicalSemaName(sp, g.edgeIdxs.front())
                 << "  acquire  pending=" << ownerSetStr(anchor, srcs) << "  "
                 << ownerStr(anchor, sp.edges[g.edgeIdxs.front()].dstOwner)
                 << "\n";
  }
}

void dumpRawSyncBlock(Block &block, SyncPlan &sp, const ResourcePlan &plan,
                      BufferGroup &group, unsigned depth,
                      OptDumpCtx *octx = nullptr);

void dumpRawSyncRegion(Region &region, SyncPlan &sp, const ResourcePlan &plan,
                       BufferGroup &group, Operation *anchorOp, unsigned depth,
                       OptDumpCtx *octx = nullptr) {
  auto recIt = plan.regionOwners.find(&region);
  std::optional<PartitionId> part;
  if (recIt != plan.regionOwners.end())
    part = recIt->second.entry;
  renderEnterRow(depth, anchorOp, part);
  for (Block &b : region)
    dumpRawSyncBlock(b, sp, plan, group, depth, octx);
  auto yIt = sp.beforeYield.find(&region);
  if (yIt != sp.beforeYield.end())
    printEdgesAt(&yIt->second, sp, anchorOp, depth, octx);
  renderYieldRow(depth, anchorOp, part);
}

void dumpRawSyncBlock(Block &block, SyncPlan &sp, const ResourcePlan &plan,
                      BufferGroup &group, unsigned depth, OptDumpCtx *octx) {
  for (Operation &op : block) {
    if (&op == sp.initialPermitBeforeOp && !sp.initialPermitName.empty()) {
      StringRef entryName = sp.initialPermitName;
      if (!sp.semaRep.empty()) {
        unsigned seedId = static_cast<unsigned>(sp.edges.size());
        unsigned rep = sp.semaFind(seedId);
        if (rep < sp.edges.size())
          entryName = sp.edges[rep].name;
      }
      llvm::errs() << treePrefix(depth) << "|- a  " << entryName
                   << "  acquire  root\n";
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      bool show = false;
      forOp.walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show && !sp.beforeYield.count(&forOp.getRegion()))
        continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      dumpRawSyncRegion(forOp.getRegion(), sp, plan, group, &op, depth + 1,
                        octx);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      bool show = false;
      ifOp->walk([&](Operation *o) -> WalkResult {
        if (sp.accessOps.contains(o)) {
          show = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (!show)
        continue;
      auto bIt = sp.beforeOp.find(&op);
      if (bIt != sp.beforeOp.end())
        printEdgesAt(&bIt->second, sp, &op, depth, octx);
      llvm::errs() << treePrefix(depth) << "|- " << regionOpLabel(&op, plan)
                   << "\n";
      llvm::errs() << treePrefix(depth + 1) << "|- then\n";
      dumpRawSyncRegion(ifOp.getThenRegion(), sp, plan, group, &op, depth + 2,
                        octx);
      if (!ifOp.getElseRegion().empty()) {
        llvm::errs() << treePrefix(depth + 1) << "|- else\n";
        dumpRawSyncRegion(ifOp.getElseRegion(), sp, plan, group, &op,
                          depth + 2, octx);
      }
      continue;
    }
    if (isa<scf::YieldOp>(op))
      continue;
    if (!sp.accessOps.contains(&op))
      continue;
    auto bIt = sp.beforeOp.find(&op);
    if (bIt != sp.beforeOp.end())
      printEdgesAt(&bIt->second, sp, &op, depth, octx);
    AccessEvent *event = nullptr;
    for (AccessEvent &e : group.events)
      if (e.op == &op) {
        event = &e;
        break;
      }
    SmallVector<const AccessTouch *, 4> touches;
    if (event)
      collectTouchesForResource(*event, sp.resource.second, touches);
    for (const AccessTouch *touch : touches) {
      bool reads = hasRead(touch->effect);
      bool writes = hasWrite(touch->effect);
      llvm::errs() << treePrefix(depth) << "|- "
                   << accessKindChar(reads, writes) << "  m"
                   << touch->memberIdx << "  " << op.getName().getStringRef()
                   << "  " << ownerStr(&op, event->owner) << "\n";
    }
    if (&op == sp.initialPermitReleaseAfterOp && !sp.initialPermitName.empty())
      llvm::errs() << treePrefix(depth) << "|- r  " << sp.initialPermitName
                   << "  release  root\n";
  }
}

std::string ownerSetStr(Operation *anchor,
                        ArrayRef<std::optional<PartitionId>> owners) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "{";
  bool first = true;
  for (const auto &o : owners) {
    if (!first)
      os << ",";
    os << ownerStr(anchor, o);
    first = false;
  }
  os << "}";
  return s;
}

} // namespace

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

void dumpOwnershipDag(ResourcePlan &plan, BufferGroup &group,
                      mlir::triton::FuncOp funcOp) {
  llvm::errs() << "OWNERSHIP-DAG buffer.id=" << plan.resource.first
               << " resourceKey=" << plan.resource.second << " members:";
  for (unsigned idx : plan.memberIndices)
    llvm::errs() << " m" << idx;
  llvm::errs() << "\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpOwnershipBlock(b, plan, group, /*depth=*/1);
}

void dumpRawSyncDag(SyncPlan &sp, const ResourcePlan &plan, BufferGroup &group,
                    mlir::triton::FuncOp funcOp) {
  llvm::errs() << "RAW-SYNC-DAG buffer.id=" << sp.resource.first
               << " resourceKey=" << sp.resource.second << " edges="
               << sp.edges.size() << "\n";
  if (!sp.semaRep.empty()) {
    unsigned seedId = static_cast<unsigned>(sp.edges.size());
    SmallVector<SmallVector<unsigned, 2>, 4> classes(sp.semaRep.size());
    for (unsigned i = 0; i < sp.semaRep.size(); ++i)
      classes[sp.semaFind(i)].push_back(i);
    bool any = false;
    for (auto &cls : classes)
      if (cls.size() >= 2)
        any = true;
    if (any) {
      llvm::errs() << "|  semaphore-classes:";
      for (auto &cls : classes) {
        if (cls.size() < 2)
          continue;
        llvm::errs() << " {";
        bool first = true;
        for (unsigned i : cls) {
          llvm::errs() << (first ? "" : "=")
                       << (i == seedId ? std::string("SEED")
                                       : sp.edges[i].name);
          first = false;
        }
        llvm::errs() << "}";
      }
      llvm::errs() << "\n";
    }
  }
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1);
}

void dumpOptSyncDag(const OptSyncDag &dag, SyncPlan &sp,
                    const ResourcePlan &plan, BufferGroup &group,
                    mlir::triton::FuncOp funcOp) {
  unsigned nInitial = 0, nFanout = 0, nFanin = 0, nSingleton = 0, nLinear = 0;
  for (const SyncGroup &g : dag.groups) {
    switch (g.kind) {
    case SyncGroupKind::InitialEmpty:
      ++nInitial;
      break;
    case SyncGroupKind::ReadyFanout:
      ++nFanout;
      break;
    case SyncGroupKind::DoneFanin:
      ++nFanin;
      break;
    case SyncGroupKind::Singleton:
      ++nSingleton;
      break;
    case SyncGroupKind::LinearChain:
      ++nLinear;
      break;
    }
  }
  llvm::errs() << "OPT-SYNC-DAG buffer.id=" << dag.resource.first
               << " resourceKey=" << dag.resource.second
               << " groups=" << dag.groups.size()
               << " (initial-empty=" << nInitial
               << " singleton=" << nSingleton
               << " ready-fanout=" << nFanout
               << " done-fanin=" << nFanin
               << " linear-chain=" << nLinear << ")\n";
  llvm::errs() << "|- func region @" << funcOp.getName() << "\n";
  OptDumpCtx octx{&dag, {}};
  for (Block &b : funcOp.getBody())
    dumpRawSyncBlock(b, sp, plan, group, /*depth=*/1, &octx);
}

} // namespace mlir::triton::nvws::insert_semas
