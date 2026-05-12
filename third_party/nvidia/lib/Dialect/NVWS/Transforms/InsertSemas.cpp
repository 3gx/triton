#include "Utilities.h"
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTSEMAS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-insert-semas"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

int getWsTag(Operation *op) {
  while (op && !hasWarpSpecializeTag(op)) {
    op = op->getParentOfType<scf::ForOp>();
  }
  assert(op);
  return *getWarpSpecializeTag(op);
}

using PartitionId = std::pair<int /* PartitionId*/, int /* WsTag*/>;
std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0) {
  if (!hasPartition(op))
    return std::nullopt;
  auto partitionIds = getPartitionIds(op);
  if (op->getNumRegions() > 0) {
    partitionIds = getPartitionOutputs(op)[pos];
  }
  assert(partitionIds.size() == 1);
  return std::make_pair(*partitionIds.begin(), getWsTag(op));
}

void assignStage(OpBuilder &b, Operation *op, StageCluster stageCluster) {
  if (stageCluster) {
    op->setAttr(kLoopStageAttrName, b.getI32IntegerAttr(stageCluster->first));
    op->setAttr(kLoopClusterAttrName,
                b.getI32IntegerAttr(stageCluster->second));
  }
}

template <typename OpT, typename... Args>
OpT createInto(
    OpBuilder &b, Location loc,
    std::pair<std::optional<PartitionId>, StageCluster> partitionIdStageCluster,
    Args &&...args) {
  std::optional<SetVector<int>> partitionIds = SetVector<int>();
  std::optional<int> wsTag;
  if (partitionIdStageCluster.first) {
    auto [id, tag] = *partitionIdStageCluster.first;
    wsTag = tag;
    partitionIds->insert(id);
  } else {
    partitionIds = std::nullopt;
  }
  auto op = triton::gpu::createInto<OpT>(b, loc, partitionIds,
                                         partitionIdStageCluster.second,
                                         std::forward<Args>(args)...);
  if (wsTag) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !hasWarpSpecializeTag(forOp)) {
      forOp = forOp->template getParentOfType<scf::ForOp>();
    }
    // only set wsTag if op is outside tt.ws loop
    if (!forOp) {
      setWarpSpecializeTag(op, *wsTag);
    }
  }
  return op;
}

static constexpr StringLiteral kBufferIdAttrName = "buffer.id";
static constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
static constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";

std::optional<int64_t> getBufferId(TMEMAllocOp allocOp) {
  auto attr = allocOp->getAttrOfType<IntegerAttr>(kBufferIdAttrName);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

bool isSupportedAccessOp(Operation *op) {
  return isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op);
}

bool isSupportedMemDescForwardingOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_index" || name == "ttg.memdesc_subslice" ||
         name == "ttg.memdesc_reshape";
}

bool isSupportedSourcefulTokenlessUse(Value value,
                                      llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  for (Operation *user : value.getUsers()) {
    if (isa<TMEMLoadOp>(user))
      continue;
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
      if (mmaOp.getAccumulator() == value)
        return false;
      bool foundOperand = false;
      for (OpOperand &opnd : user->getOpOperands())
        foundOperand |= opnd.get() == value;
      if (foundOperand)
        continue;
    }
    if (isSupportedMemDescForwardingOp(user) && user->getNumResults() == 1) {
      if (!isSupportedSourcefulTokenlessUse(user->getResult(0), visited))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

LogicalResult validateSourcefulTokenlessAlloc(TMEMAllocOp allocOp) {
  if (!allocOp.getSrc() || allocOp.getToken())
    return success();

  if (!allocOp->hasOneUse())
    return allocOp.emitError("buffer reuse: sourceful tmem_alloc must have "
                             "exactly one user");

  llvm::SmallPtrSet<Value, 8> visited;
  if (isSupportedSourcefulTokenlessUse(allocOp.getResult(), visited))
    return success();

  return allocOp.emitError("buffer reuse: sourceful tmem_alloc supports only "
                           "read/consume uses through supported TMEM aliases "
                           "(tmem_load or non-accumulator MMAv5 operand)");
}

struct BufferAccessDag {
  enum EventKind { Access, SourcefulStore, Boundary };

  struct AliasStep {
    Operation *op = nullptr;
    unsigned sourceOperand = 0;
  };

  struct AliasInfo {
    unsigned memberIdx = 0;
    SmallVector<AliasStep> viewChain;
  };

  struct Event {
    EventKind kind = Access;
    Operation *op = nullptr;
    SmallVector<unsigned, 2> memberIdxs;
    std::optional<PartitionId> partitionId;
  };

  struct RegionNode;

  struct Item {
    bool isRegion = false;
    Event event;
    RegionNode *region = nullptr;
  };

  struct BlockNode {
    Block *block = nullptr;
    SmallVector<Item> items;
    SmallVector<std::unique_ptr<RegionNode>> ownedRegions;
  };

  struct RegionNode {
    Operation *op = nullptr;
    llvm::MapVector<unsigned, int> slotMap;
    std::unique_ptr<BlockNode> body;
    std::unique_ptr<BlockNode> thenBlock;
    std::unique_ptr<BlockNode> elseBlock;
  };

  BufferAccessDag(triton::FuncOp funcOp, ArrayRef<TMEMAllocOp> members)
      : funcOp(funcOp), members(members.begin(), members.end()) {
    for (auto [idx, alloc] : llvm::enumerate(this->members)) {
      origMembers.insert({alloc.getResult(), idx});
      aliases.insert({alloc.getResult(), AliasInfo{static_cast<unsigned>(idx),
                                                   {}}});
      if (alloc.getSrc())
        sourcefulAllocs[alloc.getOperation()] = idx;
    }
  }

  LogicalResult build() {
    Region &body = funcOp.getBody();
    if (body.empty())
      return success();
    root = buildBlock(&body.front());
    return aliasFailure ? failure() : success();
  }

  llvm::MapVector<unsigned, int> getSlotMap(Operation *op) const {
    auto it = regionSlotMaps.find(op);
    if (it == regionSlotMaps.end())
      return {};
    return it->second;
  }

  const AliasInfo *lookupAlias(Value value) const {
    auto it = aliases.find(value);
    if (it == aliases.end())
      return nullptr;
    return &it->second;
  }

  bool hasAliasOperand(Operation *op) const {
    return llvm::any_of(op->getOperands(),
                        [&](Value value) { return lookupAlias(value); });
  }

  LogicalResult recordForwardingAlias(Operation *op) {
    SmallVector<unsigned> aliasOperands;
    for (OpOperand &operand : op->getOpOperands())
      if (lookupAlias(operand.get()))
        aliasOperands.push_back(operand.getOperandNumber());

    if (aliasOperands.empty())
      return success();
    if (isSupportedAccessOp(op))
      return success();
    if (!isSupportedMemDescForwardingOp(op))
      return op->emitError("buffer reuse: unsupported TMEM memdesc alias use ")
             << op->getName();
    if (aliasOperands.size() != 1 || op->getNumResults() != 1 ||
        !isa<MemDescType>(op->getResult(0).getType()))
      return op->emitError("buffer reuse: unsupported TMEM memdesc forwarding "
                           "shape for ")
             << op->getName();

    const AliasInfo *sourceAlias =
        lookupAlias(op->getOperand(aliasOperands.front()));
    assert(sourceAlias && "expected alias source");
    AliasInfo alias = *sourceAlias;
    alias.viewChain.push_back({op, aliasOperands.front()});
    aliases.insert({op->getResult(0), std::move(alias)});
    aliasOps.push_back(op);
    return success();
  }

  std::unique_ptr<BlockNode> buildBlock(Block *block) {
    auto blockNode = std::make_unique<BlockNode>();
    blockNode->block = block;

    for (Operation &op : *block) {
      if (isa<scf::YieldOp>(&op)) {
        if (hasAliasOperand(&op)) {
          op.emitError("buffer reuse: unsupported TMEM memdesc alias through "
                       "control-flow yield");
          aliasFailure = true;
        }
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (hasAliasOperand(forOp)) {
          op.emitError("buffer reuse: unsupported TMEM memdesc alias through "
                       "scf.for operands");
          aliasFailure = true;
        }
        auto region = std::make_unique<RegionNode>();
        region->op = forOp;
        regionNodes[forOp] = region.get();
        region->slotMap = getSlotMap(forOp);
        region->body = buildBlock(forOp.getBody());
        if (!region->body->items.empty() || !region->slotMap.empty())
          addRegion(*blockNode, std::move(region));
        continue;
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (hasAliasOperand(ifOp)) {
          op.emitError("buffer reuse: unsupported TMEM memdesc alias through "
                       "scf.if operands");
          aliasFailure = true;
        }
        auto region = std::make_unique<RegionNode>();
        region->op = ifOp;
        regionNodes[ifOp] = region.get();
        region->slotMap = getSlotMap(ifOp);
        region->thenBlock = buildBlock(ifOp.thenBlock());
        if (ifOp.elseBlock())
          region->elseBlock = buildBlock(ifOp.elseBlock());
        if (!region->thenBlock->items.empty() ||
            (region->elseBlock && !region->elseBlock->items.empty()) ||
            !region->slotMap.empty())
          addRegion(*blockNode, std::move(region));
        continue;
      }

      if (failed(recordForwardingAlias(&op))) {
        aliasFailure = true;
        continue;
      }
      if (isSupportedMemDescForwardingOp(&op) && hasAliasOperand(&op))
        continue;

      if (auto event = getEvent(&op))
        addEvent(*blockNode, *event);
    }

    return blockNode;
  }

  void addRegion(BlockNode &blockNode, std::unique_ptr<RegionNode> region) {
    auto *regionPtr = region.get();
    blockNode.ownedRegions.push_back(std::move(region));
    Item item;
    item.isRegion = true;
    item.region = regionPtr;
    blockNode.items.push_back(item);
  }

  void addEvent(BlockNode &blockNode, const Event &event) {
    collectEventSlots(event);
    Item item;
    item.event = event;
    blockNode.items.push_back(item);
  }

  void recordSlot(Operation *regionOp, unsigned memberIdx, int slot) {
    auto &slotMap = regionSlotMaps[regionOp];
    if (!slotMap.count(memberIdx))
      slotMap.insert({memberIdx, slot});
    auto it = regionNodes.find(regionOp);
    if (it != regionNodes.end() && !it->second->slotMap.count(memberIdx))
      it->second->slotMap.insert({memberIdx, slot});
  }

  void collectSlotsFromValue(Value token, unsigned memberIdx,
                             llvm::SmallPtrSetImpl<Value> &visited) {
    if (!token)
      return;
    if (!visited.insert(token).second)
      return;

    if (auto blockArg = dyn_cast<BlockArgument>(token)) {
      auto *ownerOp = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast_or_null<scf::ForOp>(ownerOp)) {
        for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
          if (arg == blockArg) {
            recordSlot(forOp, memberIdx, idx);
            collectSlotsFromValue(forOp->getOperand(idx + 3), memberIdx,
                                  visited);
            collectSlotsFromValue(forOp.getResult(idx), memberIdx, visited);
            return;
          }
        }
      }
    }

    if (auto result = dyn_cast<OpResult>(token)) {
      Operation *defOp = result.getOwner();
      if (isa<scf::ForOp, scf::IfOp>(defOp))
        recordSlot(defOp, memberIdx, result.getResultNumber());
    }

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : token.getUses())
      uses.push_back(&use);
    for (OpOperand *use : uses) {
      Operation *owner = use->getOwner();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        Operation *parentOp = yieldOp->getParentOp();
        unsigned slot = use->getOperandNumber();
        if (isa<scf::ForOp, scf::IfOp>(parentOp)) {
          recordSlot(parentOp, memberIdx, slot);
          collectSlotsFromValue(parentOp->getResult(slot), memberIdx, visited);
        }
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        int slot = use->getOperandNumber() - 3;
        if (slot < 0)
          continue;
        recordSlot(forOp, memberIdx, slot);
        collectSlotsFromValue(forOp.getRegionIterArg(slot), memberIdx,
                              visited);
        collectSlotsFromValue(forOp.getResult(slot), memberIdx, visited);
      }
    }
  }

  void recordSlotForToken(Value token, unsigned memberIdx) {
    llvm::SmallPtrSet<Value, 8> visited;
    collectSlotsFromValue(token, memberIdx, visited);
  }

  void collectEventSlots(const Event &event) {
    Operation *op = event.op;
    if (event.kind == SourcefulStore) {
      auto allocOp = cast<TMEMAllocOp>(op);
      if (allocOp.getToken())
        recordSlotForToken(allocOp.getToken(), event.memberIdxs.front());
      return;
    }

    auto recordIfMember = [&](Value value, Value depToken, Value resultToken) {
      const AliasInfo *alias = lookupAlias(value);
      if (!alias)
        return;
      recordSlotForToken(depToken, alias->memberIdx);
      recordSlotForToken(resultToken, alias->memberIdx);
    };

    if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
      recordIfMember(loadOp.getSrc(), loadOp.getDep(), loadOp.getToken());
    } else if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
      recordIfMember(storeOp.getDst(), storeOp.getDep(), storeOp.getToken());
    } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      recordIfMember(mmaOp.getAccumulator(), mmaOp.getAccDep(),
                     mmaOp.getToken());
    }
  }

  std::optional<Event> getEvent(Operation *op) {
    if (auto it = sourcefulAllocs.find(op); it != sourcefulAllocs.end()) {
      auto allocOp = cast<TMEMAllocOp>(op);
      Event event;
      event.kind = SourcefulStore;
      event.op = op;
      event.memberIdxs.push_back(it->second);
      event.partitionId = getPartitionId(op);
      return event;
    }

    if (!isSupportedAccessOp(op))
      return std::nullopt;

    SmallVector<unsigned, 2> memberIdxs;
    auto addMember = [&](Value value) {
      const AliasInfo *alias = lookupAlias(value);
      if (!alias)
        return;
      if (!llvm::is_contained(memberIdxs, alias->memberIdx))
        memberIdxs.push_back(alias->memberIdx);
    };

    if (auto loadOp = dyn_cast<TMEMLoadOp>(op)) {
      addMember(loadOp.getSrc());
    } else if (auto storeOp = dyn_cast<TMEMStoreOp>(op)) {
      addMember(storeOp.getDst());
    } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      for (Value operand : op->getOperands())
        addMember(operand);
    }

    if (memberIdxs.empty())
      return std::nullopt;

    Event event;
    event.kind = Access;
    event.op = op;
    event.memberIdxs = std::move(memberIdxs);
    event.partitionId = getPartitionId(op);
    return event;
  }

  void collectOwners(const BlockNode *blockNode, bool &hasRootOwner,
                     std::set<PartitionId> &owners) const {
    if (!blockNode)
      return;
    for (const Item &item : blockNode->items) {
      if (item.isRegion) {
        collectOwners(item.region->body.get(), hasRootOwner, owners);
        collectOwners(item.region->thenBlock.get(), hasRootOwner, owners);
        collectOwners(item.region->elseBlock.get(), hasRootOwner, owners);
        continue;
      }
      if (item.event.partitionId)
        owners.insert(*item.event.partitionId);
      else
        hasRootOwner = true;
    }
  }

  std::pair<bool, std::set<PartitionId>> collectOwners() const {
    bool hasRootOwner = false;
    std::set<PartitionId> owners;
    collectOwners(root.get(), hasRootOwner, owners);
    return {hasRootOwner && !owners.empty(), owners};
  }

  void collectEventsForMember(const BlockNode *blockNode, unsigned memberIdx,
                              SmallVectorImpl<Event> &events) const {
    if (!blockNode)
      return;
    for (const Item &item : blockNode->items) {
      if (item.isRegion) {
        auto slotIt = item.region->slotMap.find(memberIdx);
        if (slotIt != item.region->slotMap.end()) {
          Event boundary;
          boundary.kind = Boundary;
          boundary.op = item.region->op;
          boundary.memberIdxs.push_back(memberIdx);
          if (hasPartition(item.region->op))
            boundary.partitionId = getPartitionId(item.region->op,
                                                  slotIt->second);
          events.push_back(boundary);
        }
        collectEventsForMember(item.region->body.get(), memberIdx, events);
        collectEventsForMember(item.region->thenBlock.get(), memberIdx, events);
        collectEventsForMember(item.region->elseBlock.get(), memberIdx, events);
        if (slotIt != item.region->slotMap.end()) {
          Event boundary;
          boundary.kind = Boundary;
          boundary.op = item.region->op;
          boundary.memberIdxs.push_back(memberIdx);
          if (hasPartition(item.region->op))
            boundary.partitionId = getPartitionId(item.region->op,
                                                  slotIt->second);
          events.push_back(boundary);
        }
        continue;
      }
      if (llvm::is_contained(item.event.memberIdxs, memberIdx))
        events.push_back(item.event);
    }
  }

  SmallVector<Event> collectEventsForMember(unsigned memberIdx) const {
    SmallVector<Event> events;
    collectEventsForMember(root.get(), memberIdx, events);
    return events;
  }

  void collectOrderedEvents(const BlockNode *blockNode,
                            SmallVectorImpl<Event> &events) const {
    if (!blockNode)
      return;
    for (const Item &item : blockNode->items) {
      if (item.isRegion) {
        collectOrderedEvents(item.region->body.get(), events);
        collectOrderedEvents(item.region->thenBlock.get(), events);
        collectOrderedEvents(item.region->elseBlock.get(), events);
        continue;
      }
      events.push_back(item.event);
    }
  }

  std::optional<PartitionId>
  getNextDistinctPartitionAfter(Operation *op, PartitionId current) const {
    SmallVector<Event> events;
    collectOrderedEvents(root.get(), events);
    if (events.empty())
      return std::nullopt;

    std::optional<size_t> pos;
    for (auto [idx, event] : llvm::enumerate(events)) {
      if (event.op == op && event.partitionId && *event.partitionId == current)
        pos = idx;
    }
    if (!pos)
      return std::nullopt;

    for (size_t offset = 1; offset <= events.size(); ++offset) {
      const Event &event = events[(*pos + offset) % events.size()];
      if (event.partitionId && *event.partitionId != current)
        return event.partitionId;
    }
    return std::nullopt;
  }

  std::optional<Event> getFirstEvent(const BlockNode *blockNode) const {
    if (!blockNode)
      return std::nullopt;
    for (const Item &item : blockNode->items) {
      if (!item.isRegion)
        return item.event;

      if (!item.region->slotMap.empty()) {
        auto [memberIdx, tokPos] = item.region->slotMap.front();
        (void)memberIdx;
        Event boundary;
        boundary.kind = Boundary;
        boundary.op = item.region->op;
        if (hasPartition(item.region->op))
          boundary.partitionId = getPartitionId(item.region->op, tokPos);
        return boundary;
      }
      if (auto event = getFirstEvent(item.region->body.get()))
        return event;
      if (auto event = getFirstEvent(item.region->thenBlock.get()))
        return event;
      if (auto event = getFirstEvent(item.region->elseBlock.get()))
        return event;
    }
    return std::nullopt;
  }

  std::optional<Event> getFirstEvent() const { return getFirstEvent(root.get()); }

  void printBlock(const BlockNode *blockNode, int indent,
                  llvm::raw_ostream &os) const {
    if (!blockNode)
      return;
    for (const Item &item : blockNode->items) {
      for (int i = 0; i < indent; ++i)
        os << " ";
      if (!item.isRegion) {
        os << "|- member";
        for (unsigned memberIdx : item.event.memberIdxs)
          os << " " << memberIdx;
        os << " " << item.event.op->getName().getStringRef() << "\n";
        continue;
      }
      os << "|- region " << item.region->op->getName().getStringRef()
         << " slots[";
      for (auto [memberIdx, tokPos] : item.region->slotMap)
        os << " " << memberIdx << ":" << tokPos;
      os << " ]\n";
      printBlock(item.region->body.get(), indent + 4, os);
      printBlock(item.region->thenBlock.get(), indent + 4, os);
      printBlock(item.region->elseBlock.get(), indent + 4, os);
    }
  }

  void printDag(llvm::raw_ostream &os) const {
    os << "TMEM BUFFER DAG\n";
    printBlock(root.get(), 2, os);
    os << "\n";
  }

  triton::FuncOp funcOp;
  SmallVector<TMEMAllocOp> members;
  llvm::MapVector<Value, unsigned> origMembers;
  llvm::MapVector<Value, AliasInfo> aliases;
  SmallVector<Operation *> aliasOps;
  DenseMap<Operation *, unsigned> sourcefulAllocs;
  DenseMap<Operation *, llvm::MapVector<unsigned, int>> regionSlotMaps;
  DenseMap<Operation *, RegionNode *> regionNodes;
  std::unique_ptr<BlockNode> root;
  bool aliasFailure = false;
};

struct TMEMSemaphore {
  enum Kind { PING, PONG };

  TMEMSemaphore(Value ping, Value pong, ArrayRef<Value> allocBufs,
                const llvm::MapVector<Value, BufferAccessDag::AliasInfo>
                    &aliases,
                Value replToken, bool extendReleaseToSameOwnerOps)
      : ping(ping), pong(pong), allocBufs(allocBufs.begin(), allocBufs.end()),
        aliases(aliases), replToken(replToken),
        extendReleaseToSameOwnerOps(extendReleaseToSameOwnerOps), kind(PING) {}

  void acquire(OpBuilder &b, Location loc,
               std::pair<std::optional<PartitionId>, StageCluster>
                   partitionIdStageCluster) {
    Value sem = (kind == PING) ? ping : pong;
    auto op = createInto<SemaphoreAcquireOp>(b, loc, partitionIdStageCluster,
                                             sem, b.getType<AsyncTokenType>());
    token = op.getToken();
    partitionId = partitionIdStageCluster.first;
    if (partitionId) {
      stageClusters[*partitionId] = partitionIdStageCluster.second;
      hasSeenPartitionOwner = true;
    }
    buffers.clear();
    viewBuffers.clear();
    active = true;
    openedByRootInitialization = false;
  }

  SemaphoreReleaseOp release(OpBuilder &b, Location loc) {
    assert(active && "cannot release an inactive TMEM semaphore state");
    if (!asyncOp[partitionId])
      asyncOp[partitionId] = AsyncOp::NONE;
    StageCluster stageCluster;
    if (partitionId)
      stageCluster = stageClusters[*partitionId];
    // Cross-release: PING releases pong, PONG releases ping.
    Value sem = (kind == PING) ? pong : ping;
    auto releaseOp = createInto<SemaphoreReleaseOp>(
        b, loc, {partitionId, stageCluster}, sem, token,
        b.getArrayAttr(SmallVector<Attribute>{
            AsyncOpAttr::get(b.getContext(), *asyncOp[partitionId])}));
    // Toggle kind
    kind = (kind == PING) ? PONG : PING;
    openedByRootInitialization = false;
    return releaseOp;
  }

  Value getBuffer(OpBuilder &b, std::optional<PartitionId> pid, Operation *op,
                  unsigned memberIdx) {
    assert(active && "buffer requested before semaphore acquire");
    if (buffers.empty()) {
      auto stageCluster = getStageCluster(op);
      SmallVector<Type> dataBufTypes;
      for (Value allocBuf : allocBufs) {
        auto bufType = cast<MemDescType>(allocBuf.getType());
        dataBufTypes.push_back(getSemaphoreViewBufferType(bufType));
      }
      Value sem = (kind == PING) ? ping : pong;
      auto bufferOp =
          createInto<SemaphoreBufferOp>(b, op->getLoc(), {pid, stageCluster},
                                        sem, TypeRange(dataBufTypes), token);
      buffers.assign(bufferOp.getBuffers().begin(), bufferOp.getBuffers().end());
    }
    assert(memberIdx < buffers.size());
    return buffers[memberIdx];
  }

  const BufferAccessDag::AliasInfo *lookupAlias(Value value) const {
    auto it = aliases.find(value);
    if (it == aliases.end())
      return nullptr;
    return &it->second;
  }

  Value rebuildAliasView(OpBuilder &b, std::optional<PartitionId> pid,
                         Operation *accessOp, Value source,
                         const BufferAccessDag::AliasInfo &alias) {
    if (alias.viewChain.empty())
      return getBuffer(b, pid, accessOp, alias.memberIdx);

    auto cached = viewBuffers.find(source);
    if (cached != viewBuffers.end())
      return cached->second;

    Value current = getBuffer(b, pid, accessOp, alias.memberIdx);
    for (const BufferAccessDag::AliasStep &step : alias.viewChain) {
      Operation *orig = step.op;
      SmallVector<Value> operands(orig->getOperands());
      operands[step.sourceOperand] = current;

      OperationState state(orig->getLoc(), orig->getName());
      state.addOperands(operands);
      state.addTypes(orig->getResultTypes());
      state.addAttributes(orig->getAttrs());
      Operation *clone = b.create(state);
      assert(clone->getNumResults() == 1 &&
             "supported TMEM alias op must have one result");
      current = clone->getResult(0);
    }
    viewBuffers[source] = current;
    return current;
  }

  Value getBufferForValue(OpBuilder &b, std::optional<PartitionId> pid,
                          Operation *op, Value source) {
    const auto *alias = lookupAlias(source);
    assert(alias && "buffer requested for a non-alias value");
    return rebuildAliasView(b, pid, op, source, *alias);
  }

  // --------------------------------------------------------------------------

  Value ping; // semaphore: initially released
  Value pong; // semaphore: initially not released
  SmallVector<Value> allocBufs;
  llvm::MapVector<Value, BufferAccessDag::AliasInfo> aliases;
  Value replToken;
  bool extendReleaseToSameOwnerOps;

  SmallVector<Value> buffers;
  DenseMap<Value, Value> viewBuffers;
  Value token;
  Kind kind;
  bool active = false;
  bool openedByRootInitialization = false;
  bool hasSeenPartitionOwner = false;
  Operation *lastOp = nullptr;
  DenseMap<PartitionId, Operation *> lastAccessOpByPartition;
  std::optional<PartitionId> partitionId;
  llvm::MapVector<std::optional<PartitionId>, std::optional<AsyncOp>> asyncOp;
  DenseMap<PartitionId, StageCluster> stageClusters;
};

int pickCarrierSlot(const llvm::MapVector<unsigned, int> &slotMap) {
  assert(!slotMap.empty());
  int slot = std::numeric_limits<int>::max();
  for (auto [_, tokPos] : slotMap)
    slot = std::min(slot, tokPos);
  return slot;
}

std::optional<PartitionId> getRegionPartition(Operation *op, int tokPos) {
  if (!hasPartition(op))
    return std::nullopt;
  return getPartitionId(op, tokPos);
}

StageCluster getAcquireStageCluster(TMEMSemaphore &state,
                                    std::optional<PartitionId> partitionId,
                                    Operation *op) {
  auto stageCluster = getStageCluster(op);
  if (!stageCluster && partitionId) {
    auto it = state.stageClusters.find(*partitionId);
    if (it != state.stageClusters.end())
      stageCluster = it->second;
  }
  return stageCluster;
}

LogicalResult ensureAcquiredBefore(Operation *op,
                                   std::optional<PartitionId> partitionId,
                                   TMEMSemaphore &state) {
  if (state.active)
    return success();
  OpBuilder b(op);
  b.setInsertionPoint(op);
  state.acquire(b, op->getLoc(),
                {partitionId, getAcquireStageCluster(state, partitionId, op)});
  return success();
}

SemaphoreReleaseOp releaseForTransition(Operation *op, TMEMSemaphore &state) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  if (state.lastOp && state.lastOp->getBlock() == op->getBlock() &&
      state.lastOp->isBeforeInBlock(op)) {
    Operation *anchor = state.lastOp;
    if (state.extendReleaseToSameOwnerOps) {
      for (Operation *cur = state.lastOp->getNextNode(); cur && cur != op;
           cur = cur->getNextNode()) {
        if (isa<SemaphoreCreateOp, SemaphoreAcquireOp, SemaphoreReleaseOp,
                SemaphoreBufferOp>(cur))
          continue;
        if (!hasPartition(cur)) {
          if (!state.partitionId)
            anchor = cur;
          continue;
        }
        if (!state.partitionId)
          continue;
        auto partitionIds = getPartitionIds(cur);
        if (llvm::is_contained(partitionIds, state.partitionId->first) &&
            getWsTag(cur) == state.partitionId->second)
          anchor = cur;
      }
    }
    b.setInsertionPointAfter(anchor);
    loc = anchor->getLoc();
    if (isa<scf::ForOp>(anchor) && state.partitionId)
      state.stageClusters[*state.partitionId] = {};
  } else if (state.lastOp && state.lastOp->getBlock() != op->getBlock()) {
    b.setInsertionPointToStart(op->getBlock());
    loc = state.lastOp->getLoc();
  } else if (!state.lastOp) {
    b.setInsertionPointToStart(op->getBlock());
  } else {
    b.setInsertionPoint(op);
  }
  return state.release(b, loc);
}

LogicalResult transitionTo(Operation *op, std::optional<PartitionId> partitionId,
                           TMEMSemaphore &state) {
  if (failed(ensureAcquiredBefore(op, partitionId, state)))
    return failure();
  if (state.partitionId == partitionId)
    return success();
  if (state.openedByRootInitialization && !state.hasSeenPartitionOwner &&
      !state.partitionId && partitionId) {
    state.partitionId = partitionId;
    state.stageClusters[*partitionId] = getAcquireStageCluster(state,
                                                               partitionId, op);
    state.buffers.clear();
    state.viewBuffers.clear();
    state.openedByRootInitialization = false;
    state.hasSeenPartitionOwner = true;
    return success();
  }

  releaseForTransition(op, state);
  OpBuilder b(op);
  b.setInsertionPoint(op);
  state.acquire(b, op->getLoc(),
                {partitionId, getAcquireStageCluster(state, partitionId, op)});
  return success();
}

LogicalResult reconcileBefore(Operation *op, TMEMSemaphore &state,
                              TMEMSemaphore::Kind targetKind,
                              std::optional<PartitionId> targetPartition) {
  if (!state.active)
    return success();

  for (int i = 0; i < 3 &&
                  (state.kind != targetKind ||
                   state.partitionId != targetPartition);
       ++i) {
    OpBuilder b(op);
    b.setInsertionPoint(op);
    if (i == 0)
      releaseForTransition(op, state);
    else
      state.release(b, op->getLoc());
    state.acquire(
        b, op->getLoc(),
        {targetPartition, getAcquireStageCluster(state, targetPartition, op)});
  }

  if (state.kind != targetKind || state.partitionId != targetPartition)
    return op->emitError("buffer reuse: cannot reconcile semaphore state "
                         "before ")
           << op->getName();
  return success();
}

std::optional<PartitionId>
getPhaseRepairPartition(const TMEMSemaphore &state,
                        std::optional<PartitionId> exactTarget) {
  if (exactTarget)
    return exactTarget;
  return state.partitionId;
}

LogicalResult closeStateBefore(Operation *op, TMEMSemaphore &state,
                               TMEMSemaphore::Kind targetKind,
                               std::optional<PartitionId> phaseRepairTarget) {
  if (!state.active)
    return success();

  OpBuilder b(op);
  b.setInsertionPoint(op);
  if (state.lastOp && isa<scf::ForOp>(state.lastOp) && state.partitionId)
    state.stageClusters[*state.partitionId] = {};
  auto releaseOp = releaseForTransition(op, state);
  b.setInsertionPointAfter(releaseOp);
  while (state.kind != targetKind) {
    auto partitionId = getPhaseRepairPartition(state, phaseRepairTarget);
    state.acquire(b, op->getLoc(), {partitionId, {}});
    state.release(b, op->getLoc());
  }
  state.active = false;
  state.buffers.clear();
  state.token = {};
  return success();
}

LogicalResult closeStateAfter(Operation *op, TMEMSemaphore &state,
                              TMEMSemaphore::Kind targetKind,
                              std::optional<PartitionId> phaseRepairTarget) {
  if (!state.active)
    return success();

  OpBuilder b(op);
  b.setInsertionPointAfter(op);
  state.release(b, op->getLoc());
  while (state.kind != targetKind) {
    auto partitionId = getPhaseRepairPartition(state, phaseRepairTarget);
    state.acquire(b, op->getLoc(), {partitionId, {}});
    state.release(b, op->getLoc());
  }
  state.active = false;
  state.buffers.clear();
  state.token = {};
  return success();
}

void mergeStateMetadata(TMEMSemaphore &dst, const TMEMSemaphore &src) {
  for (auto [partitionId, asyncOp] : src.asyncOp)
    dst.asyncOp[partitionId] = asyncOp;
  for (auto [partitionId, stageCluster] : src.stageClusters)
    dst.stageClusters[partitionId] = stageCluster;
  for (auto [partitionId, op] : src.lastAccessOpByPartition)
    dst.lastAccessOpByPartition[partitionId] = op;
}

bool shouldUseMmaAsyncPayload(Operation *op);

LogicalResult rewriteAccessEvent(const BufferAccessDag::Event &event,
                                 TMEMSemaphore &state) {
  Operation *op = event.op;
  if (failed(transitionTo(op, event.partitionId, state)))
    return failure();

  OpBuilder b(op);
  b.setInsertionPoint(op);
  if (event.kind == BufferAccessDag::SourcefulStore) {
    auto allocOp = cast<TMEMAllocOp>(op);
    unsigned memberIdx = event.memberIdxs.front();
    auto buffer = state.getBuffer(b, event.partitionId, op, memberIdx);
    auto stageCluster = getStageCluster(op);
    auto vTrue = createInto<arith::ConstantIntOp>(
        b, op->getLoc(), {event.partitionId, stageCluster}, true, 1);
    createInto<TMEMStoreOp>(b, op->getLoc(), {event.partitionId, stageCluster},
                            Type(), buffer, Value(), allocOp.getSrc(), vTrue);
    state.asyncOp[event.partitionId] = AsyncOp::NONE;
    state.openedByRootInitialization =
        !event.partitionId && !state.hasSeenPartitionOwner;
    state.lastOp = op;
    if (event.partitionId)
      state.lastAccessOpByPartition[*event.partitionId] = op;
    return success();
  }

  if (auto tmemLoadOp = dyn_cast<TMEMLoadOp>(op)) {
    auto alias = state.lookupAlias(tmemLoadOp.getSrc());
    if (!alias)
      return op->emitError("buffer reuse: tmem_load does not use grouped buffer");
    if (auto id = event.partitionId)
      state.stageClusters[*id] = getStageCluster(op);
    tmemLoadOp.getSrcMutable().assign(
        state.getBufferForValue(b, event.partitionId, op, tmemLoadOp.getSrc()));
    tmemLoadOp.getDepMutable().clear();
    tmemLoadOp.getToken().replaceAllUsesWith(state.replToken);
    state.asyncOp[event.partitionId] = AsyncOp::NONE;
  } else if (auto tmemStoreOp = dyn_cast<TMEMStoreOp>(op)) {
    auto alias = state.lookupAlias(tmemStoreOp.getDst());
    if (!alias)
      return op->emitError("buffer reuse: tmem_store does not use grouped buffer");
    if (auto id = event.partitionId)
      state.stageClusters[*id] = getStageCluster(op);
    tmemStoreOp.getDstMutable().assign(
        state.getBufferForValue(b, event.partitionId, op, tmemStoreOp.getDst()));
    tmemStoreOp.getDepMutable().clear();
    tmemStoreOp.getToken().replaceAllUsesWith(state.replToken);
    state.asyncOp[event.partitionId] = AsyncOp::NONE;
    state.openedByRootInitialization =
        !event.partitionId && !state.hasSeenPartitionOwner;
  } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
    if (auto id = event.partitionId)
      state.stageClusters[*id] = getStageCluster(op);
    bool usesGroupedTmem = false;
    if (state.lookupAlias(mmaOp.getAccumulator())) {
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(state.replToken);
      mmaOp.setAccumulator(state.getBufferForValue(
          b, event.partitionId, op, mmaOp.getAccumulator()));
      usesGroupedTmem = true;
    }
    for (OpOperand &opnd : op->getOpOperands()) {
      if (state.lookupAlias(opnd.get())) {
        opnd.set(state.getBufferForValue(b, event.partitionId, op, opnd.get()));
        usesGroupedTmem = true;
      }
    }
    if (usesGroupedTmem)
      state.asyncOp[event.partitionId] =
          shouldUseMmaAsyncPayload(op) ? AsyncOp::TC5MMA : AsyncOp::NONE;
    else if (!state.asyncOp[event.partitionId])
      state.asyncOp[event.partitionId] = AsyncOp::NONE;
  } else {
    return op->emitError("buffer reuse: unsupported grouped TMEM access op ")
           << op->getName();
  }

  state.lastOp = op;
  if (event.partitionId)
    state.lastAccessOpByPartition[*event.partitionId] = op;
  return success();
}

LogicalResult processBlock(BufferAccessDag::BlockNode *blockNode,
                           TMEMSemaphore &state,
                           const std::set<PartitionId> &partitions);

LogicalResult processLocalBlock(BufferAccessDag::BlockNode *blockNode,
                                Operation *terminator, TMEMSemaphore &parent,
                                const std::set<PartitionId> &partitions) {
  TMEMSemaphore localState = parent;
  localState.active = false;
  localState.buffers.clear();
  localState.token = {};
  localState.lastOp = nullptr;
  auto entryKind = parent.kind;

  if (failed(processBlock(blockNode, localState, partitions)))
    return failure();
  return closeStateBefore(terminator, localState, entryKind,
                          parent.partitionId);
}

LogicalResult processForRegion(BufferAccessDag::RegionNode *region,
                               TMEMSemaphore &state,
                               const std::set<PartitionId> &partitions) {
  auto forOp = cast<scf::ForOp>(region->op);
  if (region->slotMap.empty()) {
    if (failed(processLocalBlock(region->body.get(),
                                 forOp.getBody()->getTerminator(), state,
                                 partitions)))
      return failure();
    state.lastOp = forOp;
    return success();
  }

  int tokPos = pickCarrierSlot(region->slotMap);
  auto boundaryPartition = getRegionPartition(forOp, tokPos);
  if (failed(transitionTo(forOp, boundaryPartition, state)))
    return failure();

  for (auto [_, slot] : region->slotMap)
    forOp->setOperand(slot + 3, slot == tokPos ? state.token : state.replToken);

  auto entryKind = state.kind;
  auto entryPartition = state.partitionId;
  TMEMSemaphore bodyState = state;
  bodyState.token = forOp.getRegionIterArg(tokPos);
  bodyState.buffers.clear();
  bodyState.lastOp = nullptr;

  if (failed(processBlock(region->body.get(), bodyState, partitions)))
    return failure();

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (failed(reconcileBefore(yieldOp, bodyState, entryKind, entryPartition)))
    return failure();

  for (auto [_, slot] : region->slotMap)
    yieldOp.setOperand(slot, slot == tokPos ? bodyState.token : state.replToken);

  mergeStateMetadata(state, bodyState);
  state.kind = entryKind;
  state.partitionId = entryPartition;
  state.token = forOp.getResult(tokPos);
  state.buffers.clear();
  state.active = true;
  state.lastOp = forOp;
  return success();
}

LogicalResult processIfRegion(BufferAccessDag::RegionNode *region,
                              TMEMSemaphore &state,
                              const std::set<PartitionId> &partitions) {
  auto ifOp = cast<scf::IfOp>(region->op);
  if (region->slotMap.empty()) {
    if (failed(processLocalBlock(region->thenBlock.get(), ifOp.thenYield(),
                                 state, partitions)))
      return failure();
    if (region->elseBlock &&
        failed(processLocalBlock(region->elseBlock.get(), ifOp.elseYield(),
                                 state, partitions)))
      return failure();
    state.lastOp = ifOp;
    return success();
  }

  int tokPos = pickCarrierSlot(region->slotMap);
  auto boundaryPartition = getRegionPartition(ifOp, tokPos);
  if (failed(transitionTo(ifOp, boundaryPartition, state)))
    return failure();

  TMEMSemaphore thenState = state;
  thenState.lastOp = nullptr;
  if (failed(processBlock(region->thenBlock.get(), thenState, partitions)))
    return failure();

  TMEMSemaphore elseState = state;
  elseState.lastOp = nullptr;
  if (region->elseBlock &&
      failed(processBlock(region->elseBlock.get(), elseState, partitions)))
    return failure();

  auto targetKind = thenState.kind;
  auto targetPartition = thenState.partitionId;
  if (failed(reconcileBefore(ifOp.thenYield(), thenState, targetKind,
                             targetPartition)))
    return failure();
  if (ifOp.elseBlock() &&
      failed(reconcileBefore(ifOp.elseYield(), elseState, targetKind,
                             targetPartition)))
    return failure();

  for (auto [_, slot] : region->slotMap) {
    ifOp.thenYield().setOperand(slot,
                                slot == tokPos ? thenState.token
                                               : state.replToken);
    if (ifOp.elseBlock())
      ifOp.elseYield().setOperand(slot,
                                  slot == tokPos ? elseState.token
                                                 : state.replToken);
  }

  mergeStateMetadata(state, thenState);
  mergeStateMetadata(state, elseState);
  state.kind = targetKind;
  state.partitionId = targetPartition;
  state.token = ifOp.getResult(tokPos);
  state.buffers.clear();
  state.active = true;
  state.lastOp = ifOp;
  return success();
}

LogicalResult processBlock(BufferAccessDag::BlockNode *blockNode,
                           TMEMSemaphore &state,
                           const std::set<PartitionId> &partitions) {
  if (!blockNode)
    return success();

  for (BufferAccessDag::Item &item : blockNode->items) {
    if (!item.isRegion) {
      if (failed(rewriteAccessEvent(item.event, state)))
        return failure();
      continue;
    }

    if (isa<scf::ForOp>(item.region->op)) {
      if (failed(processForRegion(item.region, state, partitions)))
        return failure();
    } else if (isa<scf::IfOp>(item.region->op)) {
      if (failed(processIfRegion(item.region, state, partitions)))
        return failure();
    } else {
      return item.region->op->emitError("buffer reuse: unsupported region op ")
             << item.region->op->getName();
    }
  }
  return success();
}

bool canDoubleBufferAcc(MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  auto blockM = tmemDesc.getShape()[0];
  auto blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + (blockM * blockN * 2) > numTMEMRows * numTMEMColumns) {
    return false;
  }
  if (isa<TCGen5MMAScaledOp>(mmaOp) && blockN == 256) {
    return false;
  }
  return true;
};

bool hasProducerConsumerPartitioning(const BufferAccessDag &accessDag,
                                     unsigned memberIdx) {
  // TMEM partitioning follows a producer-consumer pattern if it has this
  // structure:
  //
  //      |alloc
  //      |-- ops
  //    loop (tt.ws)
  //      |----  producer @A
  //      |----  consumer @B
  //      |----  producer @A
  //
  // We have root operations, then enter a warp-specialized loop where:
  // - First, partition A owns TMEM and performs producer operations
  // - Then, partition B owns TMEM and performs consumer operations
  // - Possibly, partition A owns TMEM and performs producer operations
  // - Loop repeats with partition A yielding
  //
  // Here is an example where the producer-consumer pattern is not present:
  //   |alloc
  //   |store
  //   |for  (tt.ws)
  //   |  |store @A
  //   |  |for
  //   |  |   mma @B
  //   |  |load @A
  // The partitions @A & @B are both producers.
  //
  // Compare to the following, where we change ownership of TMEM where partition
  // B is the producer and partition A is the consumer:
  //   |alloc
  //   |store
  //   |for  (tt.ws)
  //   |  |store @B
  //   |  |for
  //   |  |   mma @B
  //   |  |load @A
  // Here, we may double-buffer the accumulator.
  //
  // This is a necessary (but not sufficient) condition for enabling TMEM
  // multi-buffering with arefs. Additional validation will verify sufficient
  // conditions for multi-buffering.

  auto events = accessDag.collectEventsForMember(memberIdx);
  bool expectProducer = true;
  int changeGroup = 0;
  bool valid = true;
  SmallVector<BufferAccessDag::Event> partitionedEvents;
  for (const auto &event : events)
    if (event.partitionId)
      partitionedEvents.push_back(event);

  // Count partition transitions: producer-consumer pattern has exactly two
  // transitions (A->B followed by B->A), where 'A' is producer and 'B' is
  // consumer. More than two transitions (e.g., A-A-B-B-A-A-B-B-A-A) indicate a
  // more complex pattern that doesn't fit the producer-consumer model.
  for (size_t i = 0; i + 1 < partitionedEvents.size(); ++i) {
    auto op = partitionedEvents[i].op;
    if (partitionedEvents[i].kind == BufferAccessDag::SourcefulStore) {
      valid = valid && expectProducer;
    } else if (isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op)) {
      valid = valid && (expectProducer
                            ? isa<TMEMStoreOp, MMAv5OpInterface>(op)
                            : isa<TMEMLoadOp>(op));
    }
    if (*partitionedEvents[i].partitionId !=
        *partitionedEvents[i + 1].partitionId) {
      expectProducer = !expectProducer;
      ++changeGroup;
    }
  }
  valid = valid && changeGroup == 2;

  return valid;
}

bool isMultiStagedMember(TMEMAllocOp allocOp, const BufferAccessDag &accessDag,
                         unsigned memberIdx, int numTmemBlocks) {
  if (!hasProducerConsumerPartitioning(accessDag, memberIdx))
    return false;

  bool valid = true;
  for (Operation *user : allocOp.getResult().getUsers()) {
    auto mmaOp = dyn_cast<MMAv5OpInterface>(user);
    if (!mmaOp || mmaOp.getAccumulator() != allocOp.getResult())
      continue;
    if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
      auto wsLoop = getOuterWSLoop(loop);
      valid = valid && !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              isAccMultibufferingPossible(mmaOp, loop) &&
              !getDisallowAccMultiBuffer(wsLoop) &&
              canDoubleBufferAcc(mmaOp, numTmemBlocks);
    }
  }
  return valid;
}

bool shouldUseMmaAsyncPayload(Operation *op) {
  auto loop = op->getParentOfType<scf::ForOp>();
  if (!loop)
    return true;
  auto wsLoop = getOuterWSLoop(loop);
  return !wsLoop || hasPartition(wsLoop);
}

std::optional<PartitionId>
getFinalPhaseRepairTarget(const BufferAccessDag &groupDag,
                          const TMEMSemaphore &state) {
  if (!state.partitionId)
    return std::nullopt;
  auto it = state.lastAccessOpByPartition.find(*state.partitionId);
  if (it == state.lastAccessOpByPartition.end())
    return state.partitionId;
  auto next = groupDag.getNextDistinctPartitionAfter(it->second,
                                                     *state.partitionId);
  return next ? next : state.partitionId;
}

void copyBufferAttrs(TMEMAllocOp src, TMEMAllocOp dst) {
  for (StringRef attrName :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName}) {
    if (Attribute attr = src->getAttr(attrName))
      dst->setAttr(attrName, attr);
  }
}

Operation *getSemaphoreInsertionAnchor(SmallVectorImpl<TMEMAllocOp> &members) {
  Operation *anchor = members.front();
  auto outerWsLoop = anchor->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  return outerWsLoop ? outerWsLoop.getOperation() : anchor;
}

LogicalResult eraseOriginalAllocs(SmallVectorImpl<TMEMAllocOp> &members,
                                  ArrayRef<Operation *> aliasOps) {
  for (Operation *op : llvm::reverse(aliasOps)) {
    if (llvm::any_of(op->getResults(),
                     [](Value result) { return !result.use_empty(); }))
      return op->emitError("buffer reuse: original TMEM alias still has users "
                           "after semaphore insertion");
    op->erase();
  }

  for (TMEMAllocOp allocOp : members) {
    if (!allocOp.getResult().use_empty())
      return allocOp.emitError("buffer reuse: original tmem_alloc result still "
                               "has users after semaphore insertion");
    if (Value token = allocOp.getToken())
      if (!token.use_empty())
        return allocOp.emitError("buffer reuse: original tmem_alloc token still "
                                 "has users after semaphore insertion");
  }
  for (TMEMAllocOp allocOp : members)
    allocOp.erase();
  return success();
}

void eraseUnusedTmemAllocs(triton::FuncOp funcOp) {
  SmallVector<TMEMAllocOp> unusedAllocs;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    for (Value result : allocOp->getResults())
      if (!result.use_empty())
        return;
    unusedAllocs.push_back(allocOp);
  });

  for (TMEMAllocOp allocOp : unusedAllocs)
    allocOp.erase();
}

bool isSemaphoreBackingAlloc(TMEMAllocOp allocOp) {
  return llvm::any_of(allocOp->getUsers(), [](Operation *user) {
    return isa<SemaphoreCreateOp>(user);
  });
}

LogicalResult insertTmemSemaphore(BufferAccessDag &groupDag,
                                  int &numTmemBlocks) {
  auto &members = groupDag.members;
  assert(!members.empty());

  bool isMultiStaged = false;
  for (auto [idx, member] : llvm::enumerate(members))
    isMultiStaged |=
        isMultiStagedMember(member, groupDag, idx, numTmemBlocks);
  auto numStages = 1 + 1 * isMultiStaged;

  bool hasViewAlias = llvm::any_of(groupDag.aliases, [](const auto &entry) {
    return !entry.second.viewChain.empty();
  });

  SmallVector<Type> semBufTypes;
  for (TMEMAllocOp allocOp : members) {
    if (hasViewAlias && allocOp.getType().getShape().size() != 2 &&
        !isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(
            allocOp.getType().getEncoding()))
      return allocOp.emitError("buffer reuse: unsupported TMEM alias on "
                               "non-2D allocation; semaphore backing would "
                               "require an unsupported memdesc rank");
    auto shape = allocOp.getType().getShape();
    if (shape.size() >= 2)
      numTmemBlocks += shape[0] * shape[1] * numStages;
    semBufTypes.push_back(
        getSemaphoreMultiBufferedType(allocOp.getResult().getType(), numStages));
  }

  OpBuilder b(members.front());
  b.setInsertionPoint(getSemaphoreInsertionAnchor(members));

  SmallVector<Value> semAllocResults;
  for (auto [idx, allocOp] : llvm::enumerate(members)) {
    auto semAlloc = cast<TMEMAllocOp>(
        createAlloc(b, allocOp.getLoc(), cast<MemDescType>(semBufTypes[idx]),
                    Value()));
    copyBufferAttrs(allocOp, semAlloc);
    semAllocResults.push_back(semAlloc.getResult());
  }

  auto baseTypes = TypeArrayAttr::get(b.getContext(), semBufTypes);
  auto semaTy = SemaphoreType::get(b.getContext(), baseTypes);
  auto pingSem = SemaphoreCreateOp::create(b, members.front().getLoc(), semaTy,
                                           semAllocResults, true);
  auto pongSem = SemaphoreCreateOp::create(b, members.front().getLoc(), semaTy,
                                           semAllocResults, false);
  auto replToken =
      ub::PoisonOp::create(b, members.front().getLoc(),
                           b.getType<AsyncTokenType>());

  TMEMSemaphore state(pingSem, pongSem, semAllocResults, groupDag.aliases,
                      replToken, members.size() > 1);
  if (members.size() == 1) {
    auto firstEvent = groupDag.getFirstEvent();
    assert(firstEvent && "live semaphore group must have an access event");
    state.acquire(b, members.front().getLoc(),
                  {firstEvent->partitionId,
                   getAcquireStageCluster(state, firstEvent->partitionId,
                                          firstEvent->op)});
  }
  auto [hasRootOwner, partitions] = groupDag.collectOwners();
  (void)hasRootOwner;

  if (failed(processBlock(groupDag.root.get(), state, partitions)))
    return failure();

  bool needsFinalClose = true;
  if (members.size() == 1 && members.front().getSrc()) {
    needsFinalClose = llvm::any_of(groupDag.collectEventsForMember(0),
                                   [](const BufferAccessDag::Event &event) {
                                     return isa<TMEMLoadOp>(event.op);
                                   });
  }

  if (needsFinalClose && state.active && state.lastOp) {
    auto phaseRepairTarget = getFinalPhaseRepairTarget(groupDag, state);
    if (Operation *terminator = state.lastOp->getBlock()->getTerminator()) {
      if (failed(closeStateBefore(terminator, state, TMEMSemaphore::PING,
                                  phaseRepairTarget)))
        return failure();
    } else if (failed(closeStateAfter(state.lastOp, state, TMEMSemaphore::PING,
                                      phaseRepairTarget))) {
      return failure();
    }
  }

  return eraseOriginalAllocs(members, groupDag.aliasOps);
}

Operation *getFirstNonYieldOp(Block *block) {
  for (Operation &op : *block)
    if (!isa<scf::YieldOp>(op))
      return &op;
  return nullptr;
}

Operation *getPostConsumerReleaseInsertionPoint(scf::IfOp ifOp) {
  Operation *insertPt = ifOp.thenYield();
  for (Operation &op : ifOp.thenBlock()->without_terminator()) {
    if (isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op)) {
      insertPt = op.getNextNode();
      break;
    }
  }
  return insertPt;
}

SemaphoreAcquireOp getFirstAcquireAfter(Operation *op) {
  for (Operation *cur = op ? op->getNextNode() : nullptr; cur;
       cur = cur->getNextNode()) {
    if (isa<scf::YieldOp>(cur))
      return {};
    if (auto acquireOp = dyn_cast<SemaphoreAcquireOp>(cur))
      return acquireOp;
  }
  return {};
}

void splitAccessDagConditionalTransfers(triton::FuncOp funcOp) {
  SmallVector<scf::IfOp> ifs;
  funcOp.walk([&](scf::IfOp ifOp) {
    if (!ifOp.elseBlock() || ifOp.getNumResults() != 1 ||
        !isa<AsyncTokenType>(ifOp.getResult(0).getType()))
      return;

    auto thenRelease = dyn_cast_or_null<SemaphoreReleaseOp>(
        getFirstNonYieldOp(ifOp.thenBlock()));
    auto elseRelease = dyn_cast_or_null<SemaphoreReleaseOp>(
        getFirstNonYieldOp(ifOp.elseBlock()));
    if (!thenRelease || !elseRelease)
      return;

    auto thenAcquire = getFirstAcquireAfter(thenRelease);
    auto elseAcquire = getFirstAcquireAfter(elseRelease);
    if (!thenAcquire || !elseAcquire)
      return;

    auto postRelease =
        dyn_cast_or_null<SemaphoreReleaseOp>(ifOp->getNextNode());
    if (!postRelease || postRelease.getToken() != ifOp.getResult(0))
      return;
    auto postAcquire =
        dyn_cast_or_null<SemaphoreAcquireOp>(postRelease->getNextNode());
    if (!postAcquire)
      return;

    ifs.push_back(ifOp);
  });

  for (scf::IfOp ifOp : ifs) {
    auto thenRelease =
        cast<SemaphoreReleaseOp>(getFirstNonYieldOp(ifOp.thenBlock()));
    auto elseRelease =
        cast<SemaphoreReleaseOp>(getFirstNonYieldOp(ifOp.elseBlock()));
    auto thenAcquire = getFirstAcquireAfter(thenRelease);
    auto elseAcquire = getFirstAcquireAfter(elseRelease);
    auto postRelease = cast<SemaphoreReleaseOp>(ifOp->getNextNode());
    auto postAcquire = cast<SemaphoreAcquireOp>(postRelease->getNextNode());
    Value entryToken = thenRelease.getToken();

    ImplicitLocOpBuilder b(ifOp.getLoc(), ifOp);
    auto releaseIf =
        scf::IfOp::create(b, TypeRange{}, ifOp.getCondition(), false);
    releaseIf->setAttrs(ifOp->getAttrs());
    thenRelease->moveBefore(releaseIf.thenBlock(),
                            releaseIf.thenBlock()->begin());

    b.setInsertionPoint(ifOp);
    Value poisonToken =
        ub::PoisonOp::create(b, b.getType<AsyncTokenType>());

    Operation *postReleaseInsertPt = getPostConsumerReleaseInsertionPoint(ifOp);
    postRelease->setOperand(1, thenAcquire.getToken());
    postRelease->moveBefore(postReleaseInsertPt);

    ifOp.thenYield().setOperand(0, poisonToken);
    ifOp.elseYield().setOperand(0, poisonToken);
    elseAcquire.erase();
    elseRelease.erase();

    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(
        b, SmallVector<Type>{b.getType<AsyncTokenType>()}, ifOp.getCondition(),
        true);
    enterIf->setAttrs(ifOp->getAttrs());
    postAcquire->moveBefore(enterIf.thenBlock(), enterIf.thenBlock()->begin());
    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(b, postAcquire.getToken());
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(b, entryToken);

    postAcquire.getToken().replaceUsesWithIf(
        enterIf.getResult(0), [&](OpOperand &use) {
          return use.getOwner()->getBlock() != enterIf.thenBlock();
        });

    if (hasPartition(thenRelease)) {
      auto releaseIds = getPartitionIds(thenRelease);
      setPartition(releaseIf, releaseIds);
      setPartition(enterIf, releaseIds);
      setPartitionOutputs(releaseIf, {});
      setPartitionOutputs(enterIf, {releaseIds});
    }
    if (hasPartition(thenAcquire)) {
      auto acquireIds = getPartitionIds(thenAcquire);
      setPartition(ifOp, acquireIds);
      setPartitionOutputs(ifOp, {acquireIds});
    }
    assignStage(b, releaseIf, getStageCluster(thenRelease));
    assignStage(b, ifOp, getStageCluster(thenAcquire));
    assignStage(b, enterIf, getStageCluster(postAcquire));
  }
}

void workaroundForLoopScheduler(triton::FuncOp funcOp) {
  splitAccessDagConditionalTransfers(funcOp);

  SmallVector<scf::IfOp> ifs;
  funcOp.walk([&](scf::IfOp ifOp) {
    auto firstOp = &*ifOp.thenBlock()->begin();
    auto lastOp = ifOp.thenBlock()->getTerminator()->getPrevNode();
    if (isa<SemaphoreReleaseOp>(firstOp) && isa<SemaphoreAcquireOp>(lastOp)) {
      ifs.push_back(ifOp);
    }
  });

  // Transform if-statements that contain sema.acquire/release pairs to work
  // around loop scheduler limitations. The transformation splits a single if-op
  // with token-producing operations into three separate if-ops to ensure proper
  // scheduling and token handling.
  //
  // Original pattern:
  //   %results, %token, %more = scf.if %condition {
  //     sema.release                     // Release tensor memory
  //     <computation_code>               // User computation
  //     %new_token = sema.acquire        // Acquire tensor memory
  //     scf.yield %values, %new_token, %other_values
  //   } else {
  //     scf.yield %alt_values, %old_token, %alt_other_values
  //   }
  //   ... use %token
  //
  // Transformed pattern:
  //   scf.if %condition {
  //     sema.release                    // Separate release operation
  //   } { .. loop.stage = 1, ttg.partition = {1}, ttg.partition.outputs = [] }
  //   %results, %poison_tok, %more = scf.if %condition {
  //     <computation_code>               // Main computation without token ops
  //     scf.yield %values, %poison_tok, %other_values
  //   } else {
  //     scf.yield %alt_values, %poison_tok, %alt_other_values
  //   } {.. ttg.partition = {0}, ttg.partition.outputs = [{0}, {0}, {0}, ..]}
  //   %token = scf.if %condition {
  //     %new_token = sema.acquire       // Separate acquire operation
  //     scf.yield %new_token
  //   } else {
  //     scf.yield %old_token
  //   } { .. loop.stage = 1, ttg.partition = {1}, ttg.partition.outputs =
  //   [{1}]}
  //   ... use %token

  for (auto ifOp : ifs) {
    ImplicitLocOpBuilder b(ifOp.getLoc(), ifOp);

    // move releaseOp
    b.setInsertionPoint(ifOp);
    auto exitIf =
        scf::IfOp::create(b, SmallVector<Type>{}, ifOp.getCondition(), false);
    auto releaseOp = cast<SemaphoreReleaseOp>(*ifOp.thenBlock()->begin());
    releaseOp->moveBefore(exitIf.thenBlock(), exitIf.thenBlock()->begin());

    // move acquireOp
    b.setInsertionPointAfter(ifOp);
    auto enterIf =
        scf::IfOp::create(b, SmallVector<Type>{b.getType<AsyncTokenType>()},
                          ifOp.getCondition(), true);
    auto acquireOp = cast<SemaphoreAcquireOp>(
        ifOp.thenBlock()->getTerminator()->getPrevNode());
    acquireOp->moveBefore(enterIf.thenBlock(), enterIf.thenBlock()->begin());

    // replace token uses
    auto tok = acquireOp.getToken();
    auto pos = *findValuePosInRange(ifOp.thenYield()->getOperands(), tok);
    ifOp.getResult(pos).replaceAllUsesWith(enterIf.getResult(0));

    // insert yield-ops inside enterIf
    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(b, tok);
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(b, ifOp.elseYield().getOperand(pos));

    // invalidate tokens in main ifOp
    b.setInsertionPoint(ifOp);
    auto poisonToken = ub::PoisonOp::create(b, b.getType<AsyncTokenType>());
    ifOp.thenYield().setOperand(pos, poisonToken);
    ifOp.elseYield().setOperand(pos, poisonToken);

    // patch loop.stage=1
    enterIf->setAttrs(ifOp->getAttrs());
    exitIf->setAttrs(ifOp->getAttrs());
    assignStage(b, enterIf, getStageCluster(acquireOp));
    assignStage(b, exitIf, getStageCluster(releaseOp));

    SetVector<int> enterExitIds, middleIds;
    enterExitIds.insert(1);
    middleIds.insert(0);
    setPartition(enterIf, enterExitIds);
    setPartition(exitIf, enterExitIds);
    setPartition(ifOp, middleIds);

    SetVector<int> p0array, p1array;
    p0array.insert(0);
    p1array.insert(1);
    setPartitionOutputs(exitIf, {});
    setPartitionOutputs(enterIf, {p1array});
    SmallVector<SetVector<int>> outputs(ifOp->getNumResults(), p0array);
    setPartitionOutputs(ifOp, outputs);
  }
}

LogicalResult runOnFunction(triton::FuncOp funcOp) {
  // Skip this function if there is no warp specialized loop.
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  int64_t nextBufferId = 0;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    if (isSemaphoreBackingAlloc(allocOp))
      return;
    if (auto bufferId = getBufferId(allocOp))
      nextBufferId = std::max(nextBufferId, *bufferId + 1);
  });

  DenseMap<Operation *, int64_t> syntheticBufferIds;
  llvm::MapVector<int64_t, SmallVector<TMEMAllocOp>> groupsByBufferId;
  auto allocWalk = funcOp.walk([&](TMEMAllocOp allocOp) -> WalkResult {
    if (isSemaphoreBackingAlloc(allocOp))
      return WalkResult::advance();
    auto bufferId = getBufferId(allocOp);
    if (!bufferId) {
      syntheticBufferIds[allocOp.getOperation()] = nextBufferId++;
      bufferId = syntheticBufferIds[allocOp.getOperation()];
    }

    if (failed(validateSourcefulTokenlessAlloc(allocOp)))
      return WalkResult::interrupt();

    groupsByBufferId[*bufferId].push_back(allocOp);
    return WalkResult::advance();
  });
  if (allocWalk.wasInterrupted())
    return failure();

  int numTmemBlocks = 0;
  for (auto &[bufferId, members] : groupsByBufferId) {
    (void)bufferId;
    BufferAccessDag groupDag(funcOp, members);
    if (failed(groupDag.build()))
      return failure();
    LLVM_DEBUG({ groupDag.printDag(llvm::dbgs()); });

    auto [hasRootPartition, partitions] = groupDag.collectOwners();
    auto totalOwners = static_cast<int>(hasRootPartition) + partitions.size();
    if (totalOwners <= 1)
      continue;

    if (failed(insertTmemSemaphore(groupDag, numTmemBlocks)))
      return failure();
  }

  workaroundForLoopScheduler(funcOp);
  eraseUnusedTmemAllocs(funcOp);

  return success();
}

} // namespace

class NVWSInsertSemas
    : public triton::impl::NVWSInsertSemasBase<
          NVWSInsertSemas> {
public:
  void runOnOperation() override {
    getOperation().walk([&](triton::FuncOp funcOp) {
      if (failed(runOnFunction(funcOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
  }
};

} // namespace triton
} // namespace mlir
