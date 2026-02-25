#include "Utilities.h"
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
#include <optional>

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSINSERTTMEMSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-insert-tmem-semaphore"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;
using namespace triton::nvws;

// ---------------------------------------------------------------------------
// Helpers for TMEM semaphore insertion
// ---------------------------------------------------------------------------

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

// TmemAccessDag — DAG for TMEM access patterns
struct TmemAccessDag {
  struct Node {
    std::unique_ptr<Node> user;
    SmallVector<std::unique_ptr<Node>> subDags;
    Node(Operation *op, OpOperand *tokOperand,
         std::optional<PartitionId> partitionId, Node *parent)
        : op(op), tokOperand(tokOperand), partitionId(partitionId),
          parent(parent), parentDag(nullptr) {}

    Operation *op;
    OpOperand *tokOperand;
    Node *parent;
    Node *parentDag;
    std::optional<int> tokPos;
    std::optional<PartitionId> partitionId;
  };

  TmemAccessDag(std::unique_ptr<Node> dag) : dag(std::move(dag)) {}

  Node *getRootNode() { return dag.get(); }
  TMEMAllocOp getAllocOp() { return cast<TMEMAllocOp>(dag->op); }

  Value addIfOp(Value tok, Node *node) {
    SmallVector<OpOperand *> uses;
    for (auto &use : tok.getUses())
      uses.push_back(&use);
    assert(uses.size() == 2 && "expecting two uses of a token");
    auto useThen = uses[0];
    auto useElse = uses[1];

    auto ifOp = cast<scf::IfOp>(useThen->getOwner()->getParentOp());
    node->user.reset(new Node(ifOp, nullptr, {}, node));
    auto ifOpNode = node->user.get();

    if (ifOp.thenBlock() != useThen->getOwner()->getBlock())
      std::swap(useThen, useElse);
    assert(ifOp.thenBlock() == useThen->getOwner()->getBlock());
    assert(ifOp.elseBlock() == useElse->getOwner()->getBlock());

    auto thenDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto elseDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto thenTok = addOp(*useThen, thenDag.get());
    addOp(*useElse, elseDag.get());

    auto tokPos =
        *findValuePosInRange(ifOp.thenYield()->getOperands(), thenTok);
    ifOpNode->partitionId = getPartitionId(ifOp, tokPos);

    Node *finalThenNode = thenDag.get();
    while (finalThenNode->user)
      finalThenNode = finalThenNode->user.get();
    auto thenYieldOp = ifOp.thenYield();
    finalThenNode->user =
        std::make_unique<Node>(thenYieldOp, &thenYieldOp->getOpOperand(tokPos),
                               ifOpNode->partitionId, finalThenNode);
    finalThenNode->user->parentDag = thenDag->user.get();

    Node *finalElseNode = elseDag.get();
    while (finalElseNode->user)
      finalElseNode = finalElseNode->user.get();
    auto elseYieldOp = ifOp.elseYield();
    finalElseNode->user =
        std::make_unique<Node>(elseYieldOp, &elseYieldOp->getOpOperand(tokPos),
                               ifOpNode->partitionId, finalElseNode);
    finalElseNode->user->parentDag = elseDag->user.get();

    thenDag->user->parent = nullptr;
    elseDag->user->parent = nullptr;
    thenDag->user->parentDag = ifOpNode;
    elseDag->user->parentDag = ifOpNode;

    ifOpNode->subDags.push_back(std::move(thenDag->user));
    ifOpNode->subDags.push_back(std::move(elseDag->user));

    ifOpNode->tokPos = tokPos;

    auto newTok = ifOp.getResult(tokPos);
    assert(newTok.hasOneUse());
    return addOp(*newTok.getUses().begin(), ifOpNode);
  }

  Value addForOp(OpOperand &tokOperand, Node *forOpNode) {
    auto forOp = cast<scf::ForOp>(tokOperand.getOwner());
    auto tokPos = tokOperand.getOperandNumber() - 3;
    auto tokDefOp = forOp.getYieldedValues()[tokPos].getDefiningOp();
    assert(tokDefOp && "expecting a token definition op");

    auto subDag =
        std::make_unique<Node>(nullptr, nullptr, std::nullopt, nullptr);
    auto tokArg = forOp.getRegionIterArg(tokPos);
    assert(tokArg.hasOneUse());
    addOp(*tokArg.getUses().begin(), subDag.get());
    forOpNode->partitionId = getPartitionId(forOp, tokPos);

    Node *finalNode = subDag->user.get();
    while (finalNode->user)
      finalNode = finalNode->user.get();
    auto yieldOp = forOp.getBody()->getTerminator();
    finalNode->user =
        std::make_unique<Node>(yieldOp, &yieldOp->getOpOperand(tokPos),
                               forOpNode->partitionId, finalNode);
    finalNode->user->parentDag = subDag->user.get();
    forOpNode->tokPos = tokPos;

    subDag->user->parent = nullptr;
    subDag->user->parentDag = forOpNode;

    forOpNode->subDags.push_back(std::move(subDag->user));
    return forOp.getResult(tokPos);
  }

  Value addOp(OpOperand &tokOperand, Node *node) {
    if (isa<scf::YieldOp>(tokOperand.getOwner()))
      return tokOperand.get();

    auto op = tokOperand.getOwner();
    std::optional<PartitionId> partitionId;
    if (op->getNumRegions() == 0)
      partitionId = getPartitionId(op);
    node->user.reset(new Node(op, &tokOperand, partitionId, node));
    auto newNode = node->user.get();
    Value newTok;

    if (auto tmemLoad = dyn_cast<TMEMLoadOp>(op)) {
      newTok = tmemLoad.getToken();
    } else if (auto tmemStore = dyn_cast<TMEMStoreOp>(op)) {
      newTok = tmemStore.getToken();
    } else if (auto mmav5 = dyn_cast<MMAv5OpInterface>(op)) {
      newTok = mmav5.getToken();
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      newTok = addForOp(tokOperand, newNode);
    } else {
      llvm_unreachable("unsupported user");
    }

    if (newTok.use_empty())
      return newTok;

    if (newTok.hasOneUse()) {
      auto &use = *newTok.getUses().begin();
      return addOp(use, newNode);
    }

    return addIfOp(newTok, newNode);
  }

  static TmemAccessDag build(TMEMAllocOp allocOp) {
    std::optional<PartitionId> partitionId;
    if (allocOp.getSrc()) {
      partitionId = getPartitionId(allocOp);
    }
    TmemAccessDag accessDag(
        std::make_unique<Node>(allocOp, nullptr, partitionId, nullptr));

    if (allocOp.getSrc() && !allocOp.getToken()) {
      assert(allocOp->hasOneUse());
      auto user = *allocOp->getUsers().begin();
      accessDag.getRootNode()->user.reset(new Node{
          user, nullptr, getPartitionId(user), accessDag.getRootNode()});
    } else {
      auto tok = allocOp.getToken();
      assert(tok && tok.hasOneUse());
      auto &tokUse = *tok.getUses().begin();
      accessDag.addOp(tokUse, accessDag.getRootNode());
    }
    return accessDag;
  }

  void collectPartitions(
      Node *node, bool &hasRootPartition,
      SmallVector<std::pair<PartitionId, Operation *>> &partitions) {
    if (node->partitionId) {
      partitions.push_back(std::make_pair(*node->partitionId, node->op));
    } else {
      hasRootPartition = !partitions.empty();
    }
    for (auto &subDag : node->subDags) {
      if (subDag) {
        collectPartitions(subDag.get(), hasRootPartition, partitions);
      }
    }
    if (node->user) {
      collectPartitions(node->user.get(), hasRootPartition, partitions);
    }
  };

  std::pair<bool, SmallVector<std::pair<PartitionId, Operation *>>>
  collectPartitionsVec() {
    SmallVector<std::pair<PartitionId, Operation *>> partitions;
    bool hasRootPartition = false;
    auto node = getRootNode();
    auto allocOp = getAllocOp();
    if (allocOp.getSrc() && node->partitionId)
      partitions.push_back(std::make_pair(*node->partitionId, node->op));
    collectPartitions(getRootNode()->user.get(), hasRootPartition, partitions);
    return {hasRootPartition, partitions};
  }

  std::pair<bool, std::set<PartitionId>> collectPartitionsSet() {
    auto [hasRootPartition, partitions] = collectPartitionsVec();
    std::set<PartitionId> partitionSet;
    for (auto [partition, _] : partitions) {
      partitionSet.insert(partition);
    }
    return {hasRootPartition, partitionSet};
  }

  void printNode(Node *node, int indent, llvm::raw_ostream &os) {
    if (!node)
      return;
    for (int i = 0; i < indent; i++)
      os << " ";
    std::set<PartitionId> partitions;
    os << "|- [" << node->op << "]";
    bool hasRootPartition = false;
    if (node->partitionId)
      partitions.insert(*node->partitionId);
    else
      hasRootPartition = true;
    if (node->op) {
      os << node->op->getName().getStringRef() << " ";
      if (auto tmemAlloc = dyn_cast<TMEMAllocOp>(node->op)) {
        if (tmemAlloc.getSrc())
          os << " %src ";
        else
          std::tie(hasRootPartition, partitions) = collectPartitionsSet();
      }
      os << "  ";
    }
    os << "[" << (hasRootPartition ? "root" : "");
    for (auto partition : partitions) {
      auto [id, tag] = partition;
      os << " @" << tag << "." << id << " ";
    }
    os << "]";
    os << " prev[" << (node->parent ? node->parent->op : nullptr) << "]";
    os << "\n";
    for (auto &subDag : node->subDags) {
      for (int i = 0; i < indent + 4; i++)
        os << " ";
      os << "|- subDag\n";
      if (subDag)
        printNode(subDag.get(), indent + 8, os);
    }
    if (node->user) {
      printNode(node->user.get(), indent, os);
    }
  };
  void printDag(llvm::raw_ostream &os) {
    os << "TMEMDAG\n";
    printNode(dag.get(), 2, os);
    os << "\n";
  }

  std::unique_ptr<Node> dag;
};

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
    if (!forOp) {
      setWarpSpecializeTag(op, *wsTag);
    }
  }
  return op;
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

bool hasProducerConsumerPartitioning(TmemAccessDag &accessDag) {
  auto [hasRootPartition, partitions] = accessDag.collectPartitionsVec();
  bool expectProducer = true;
  int changeGroup = 0;
  bool valid = true;

  for (size_t i = 0; i < partitions.size() - 1; ++i) {
    auto op = partitions[i].second;
    if (isa<TMEMLoadOp, TMEMStoreOp, MMAv5OpInterface>(op)) {
      valid = valid && (expectProducer ? isa<TMEMStoreOp, MMAv5OpInterface>(op)
                                       : isa<TMEMLoadOp>(op));
    }
    if (partitions[i].first != partitions[i + 1].first) {
      expectProducer = !expectProducer;
      ++changeGroup;
    }
  }
  valid = valid && changeGroup == 2;

  return valid;
}

// ---------------------------------------------------------------------------
// TMEMSemaphore struct
// ---------------------------------------------------------------------------

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true, type.getAllocShape());
}

struct TMEMSemaphore {
  enum Kind { PING, PONG };

  TMEMSemaphore(Value empty, Value full, Value allocBuf, Value origBuffer,
                Value replToken)
      : empty(empty), full(full), allocBuf(allocBuf), origBuffer(origBuffer),
        replToken(replToken), kind(PING) {}

  void acquire(OpBuilder &b, Location loc,
               std::pair<std::optional<PartitionId>, StageCluster>
                   partitionIdStageCluster) {
    // PING acquires empty, PONG acquires full
    Value sem = (kind == PING) ? empty : full;
    auto op = createInto<SemaphoreAcquireOp>(
        b, loc, partitionIdStageCluster, sem,
        b.getType<AsyncTokenType>());
    token = op.getToken();
    partitionId = partitionIdStageCluster.first;
    if (partitionId)
      stageClusters[*partitionId] = partitionIdStageCluster.second;
    buffer = {};
  }

  void release(OpBuilder &b, Location loc) {
    assert(asyncOp[partitionId]);
    StageCluster stageCluster;
    if (partitionId)
      stageCluster = stageClusters[*partitionId];
    // Cross-release: PING releases full, PONG releases empty
    Value sem = (kind == PING) ? full : empty;
    createInto<SemaphoreReleaseOp>(
        b, loc, {partitionId, stageCluster}, sem, token,
        b.getArrayAttr(SmallVector<Attribute>{
            AsyncOpAttr::get(b.getContext(), *asyncOp[partitionId])}));
    // Toggle kind
    kind = (kind == PING) ? PONG : PING;
  }

  Value getBuffer(OpBuilder &b, std::optional<PartitionId> pid,
                  Operation *op) {
    if (!buffer) {
      auto stageCluster = getStageCluster(op);
      auto bufType = cast<MemDescType>(allocBuf.getType());
      Type dataBufType = getSemaphoreViewBufferType(bufType);
      // Make mutable
      if (auto memTy = dyn_cast<MemDescType>(dataBufType))
        dataBufType = getAsMutable(memTy);
      Value sem = (kind == PING) ? empty : full;
      auto bufferOp = createInto<SemaphoreBufferOp>(
          b, op->getLoc(), {pid, stageCluster}, sem,
          TypeRange{dataBufType}, token);
      buffer = bufferOp.getBuffers()[0];
    }
    return buffer;
  }

  // --------------------------------------------------------------------------

  Value empty;       // semaphore: initially released (producer acquires)
  Value full;        // semaphore: initially not released (consumer acquires)
  Value allocBuf;    // underlying TMEM buffer allocation
  Value origBuffer;
  Value replToken;

  Value buffer;
  Value token;
  Kind kind;
  std::optional<PartitionId> partitionId;
  llvm::MapVector<std::optional<PartitionId>, std::optional<AsyncOp>> asyncOp;
  DenseMap<PartitionId, StageCluster> stageClusters;
};

// ---------------------------------------------------------------------------
// DAG traversal for TMEM semaphore insertion
// ---------------------------------------------------------------------------

TmemAccessDag::Node *
insertTmemSemaphoreImpl(TmemAccessDag::Node *node,
                        std::optional<PartitionId> curPartitionId,
                        TMEMSemaphore &state) {
  if (curPartitionId && node->partitionId != curPartitionId) {
    OpBuilder b(node->op);
    Operation *prevOp = nullptr;
    if (node->parent) {
      prevOp = node->parent->op;
      b.setInsertionPointAfter(prevOp);
    } else {
      prevOp = node->parentDag->op;
      b.setInsertionPointToStart(node->op->getBlock());
    }
    state.release(b, prevOp->getLoc());

    auto curOp = node->op;
    auto partitionId = node->partitionId;
    b.setInsertionPoint(curOp);

    if (isa<scf::YieldOp>(curOp)) {
      curOp = node->parentDag->op;
    }
    auto stageCluster = getStageCluster(curOp);
    if (!stageCluster && partitionId)
      stageCluster = state.stageClusters[*partitionId];
    state.acquire(b, curOp->getLoc(), {partitionId, stageCluster});
  }

  for (auto &subDag : node->subDags) {
    auto subdagState = state;
    if (auto forOp = dyn_cast<scf::ForOp>(node->op)) {
      if (node->tokOperand) {
        subdagState.token =
            forOp.getRegionIterArg(node->tokOperand->getOperandNumber() - 3);
        subdagState.buffer = {};
      }
    }
    insertTmemSemaphoreImpl(subDag.get(), node->partitionId, subdagState);

    state.asyncOp = subdagState.asyncOp;
    state.partitionId = subdagState.partitionId;
  }

  if (isa<MMAv5OpInterface>(node->op)) {
    state.asyncOp[node->partitionId] = AsyncOp::TC5MMA;
  } else if (isa<TMEMLoadOp, TMEMStoreOp>(node->op)) {
    state.asyncOp[node->partitionId] = AsyncOp::NONE;
  }

  OpBuilder b(node->op);
  if (auto tmemLoadOp = dyn_cast<TMEMLoadOp>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    tmemLoadOp.getSrcMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemLoadOp.getDepMutable().clear();
    tmemLoadOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto tmemStoreOp = dyn_cast<TMEMStoreOp>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    tmemStoreOp.getDstMutable().assign(
        state.getBuffer(b, node->partitionId, node->op));
    tmemStoreOp.getDepMutable().clear();
    tmemStoreOp.getToken().replaceAllUsesWith(state.replToken);
  } else if (auto mmaOp = dyn_cast<MMAv5OpInterface>(node->op)) {
    if (auto id = node->partitionId)
      state.stageClusters[*id] = getStageCluster(node->op);
    if (mmaOp.getAccumulator() == state.origBuffer) {
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(state.replToken);
    }
    for (auto &opnd : mmaOp->getOpOperands()) {
      if (opnd.get() == state.origBuffer)
        opnd.set(state.getBuffer(b, node->partitionId, node->op));
    }
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(node->op)) {
    yieldOp.setOperand(node->tokOperand->getOperandNumber(), state.token);
  } else if (isa<scf::IfOp, scf::ForOp>(node->op)) {
    if (node->tokPos) {
      if (isa<scf::ForOp>(node->op))
        node->op->setOperand(node->tokOperand->getOperandNumber(), state.token);
      state.token = node->op->getResult(*node->tokPos);
      state.buffer = {};
    }
  } else {
    llvm_unreachable("unsupported tmem op");
  }

  if (node->user)
    return insertTmemSemaphoreImpl(node->user.get(), node->partitionId, state);
  return node;
}

// ---------------------------------------------------------------------------
// Main TMEM semaphore insertion function
// ---------------------------------------------------------------------------

int insertTmemSemaphore(TmemAccessDag &accessDag, int numTmemBlocks) {
  auto rootNode = accessDag.getRootNode();
  auto allocOp = cast<TMEMAllocOp>(rootNode->op);

  auto isMultiStaged = hasProducerConsumerPartitioning(accessDag);
  if (isMultiStaged) {
    for (auto user : allocOp.getResult().getUsers()) {
      if (auto mmaOp = dyn_cast<MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          auto wsLoop = getOuterWSLoop(loop);
          bool accIsMultiBuffered =
              !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              isAccMultibufferingPossible(mmaOp, loop) &&
              !getDisallowAccMultiBuffer(wsLoop) &&
              canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiStaged = isMultiStaged && accIsMultiBuffered;
        }
      }
    }
  }
  auto numStages = 1 + isMultiStaged;

  auto allocShape = allocOp.getType().getShape();
  numTmemBlocks += allocShape[0] * allocShape[1] * numStages;
  auto semBufType =
      getSemaphoreMultiBufferedType(allocOp.getResult().getType(), numStages);
  OpBuilder b(allocOp);

  auto outerWsLoop = allocOp->getParentOfType<scf::ForOp>();
  while (outerWsLoop && !outerWsLoop->hasAttr(triton::kWarpSpecializeAttrName))
    outerWsLoop = outerWsLoop->getParentOfType<scf::ForOp>();
  if (outerWsLoop)
    b.setInsertionPoint(outerWsLoop);

  auto semAlloc =
      cast<TMEMAllocOp>(createAlloc(b, allocOp.getLoc(), semBufType, Value()));

  // Create semaphore pair
  auto baseTypes =
      TypeArrayAttr::get(b.getContext(), {semBufType});
  auto depth = getSemaphoreDepth(semBufType);
  auto semaTy = SemaphoreType::get(b.getContext(), baseTypes, depth);
  auto emptySem = SemaphoreCreateOp::create(b, allocOp.getLoc(), semaTy,
                                            semAlloc->getResults(), true);
  auto fullSem = SemaphoreCreateOp::create(b, allocOp.getLoc(), semaTy,
                                           semAlloc->getResults(), false);

  auto stageCluster = getStageCluster(allocOp);
  auto partitionId = accessDag.getRootNode()->partitionId;
  if (!allocOp.getSrc() && outerWsLoop) {
    partitionId = accessDag.getRootNode()->user->partitionId;
  }

  TMEMSemaphore state(
      emptySem, fullSem, semAlloc->getResult(0), allocOp.getResult(),
      ub::PoisonOp::create(b, allocOp.getLoc(), b.getType<AsyncTokenType>()));
  b.setInsertionPoint(allocOp);
  state.acquire(b, allocOp.getLoc(), {partitionId, stageCluster});

  if (!state.partitionId) {
    auto node = rootNode->user.get();
    do {
      state.partitionId = node->partitionId;
      node = node->user.get();
    } while (node && !state.partitionId);
  }

  if (auto src = allocOp.getSrc()) {
    auto buffer = state.getBuffer(b, partitionId, allocOp);
    state.asyncOp[partitionId] = AsyncOp::NONE;
    auto vTrue = createInto<arith::ConstantIntOp>(
        b, allocOp.getLoc(), {partitionId, stageCluster}, true, 1);
    createInto<TMEMStoreOp>(b, allocOp.getLoc(), {partitionId, stageCluster},
                            Type(), buffer, Value(), src, vTrue);
  }

  auto node = insertTmemSemaphoreImpl(rootNode->user.get(), partitionId, state);

  if (outerWsLoop) {
    b.setInsertionPointAfter(node->op);
  } else {
    auto op1 = emptySem->getBlock()->findAncestorOpInBlock(*node->op);
    if (auto id = node->partitionId)
      state.stageClusters[*id] = {};
    b.setInsertionPointAfter(op1);
  }
  state.release(b, node->op->getLoc());

  if (state.kind == TMEMSemaphore::PONG) {
    auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
    std::optional<PartitionId> otherPartitionId;
    for (auto partitionId : partitions) {
      if (partitionId != state.partitionId) {
        otherPartitionId = partitionId;
        break;
      }
    }
    state.acquire(b, node->op->getLoc(), {otherPartitionId, {}});
    state.release(b, node->op->getLoc());
  }

  return numTmemBlocks;
}

// ---------------------------------------------------------------------------
// workaroundForLoopScheduler — adapted for semaphore ops
// ---------------------------------------------------------------------------

void workaroundForLoopScheduler(triton::FuncOp funcOp) {
  SmallVector<scf::IfOp> ifs;
  funcOp.walk([&](scf::IfOp ifOp) {
    auto firstOp = &*ifOp.thenBlock()->begin();
    auto lastOp = ifOp.thenBlock()->getTerminator()->getPrevNode();
    if (isa<SemaphoreReleaseOp>(firstOp) && isa<SemaphoreAcquireOp>(lastOp)) {
      ifs.push_back(ifOp);
    }
  });

  for (auto ifOp : ifs) {
    ImplicitLocOpBuilder b(ifOp.getLoc(), ifOp);

    // move releaseOp (was putExitOp)
    b.setInsertionPoint(ifOp);
    auto exitIf =
        scf::IfOp::create(b, SmallVector<Type>{}, ifOp.getCondition(), false);
    auto releaseOp = cast<SemaphoreReleaseOp>(*ifOp.thenBlock()->begin());
    releaseOp->moveBefore(exitIf.thenBlock(), exitIf.thenBlock()->begin());

    // move acquireOp (was putEnterOp)
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

    // patch loop.stage
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

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

LogicalResult runOnFunction(triton::FuncOp funcOp) {
  auto walkResult = funcOp.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(kWarpSpecializeAttrName))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  SmallVector<TmemAccessDag> tmemDags;
  funcOp.walk([&](TMEMAllocOp allocOp) {
    tmemDags.push_back(TmemAccessDag::build(allocOp));
  });

  int numTmemBlocks = 0;
  for (auto &accessDag : tmemDags) {
    LLVM_DEBUG({ accessDag.printDag(llvm::dbgs()); });
    auto [hasRootPartition, partitions] = accessDag.collectPartitionsSet();
    assert(partitions.size() <= 2 && "expecting at most 2 partitions");
    auto totalOwners = hasRootPartition + partitions.size();
    if (totalOwners > 1) {
      numTmemBlocks = insertTmemSemaphore(accessDag, numTmemBlocks);
    }
  }

  workaroundForLoopScheduler(funcOp);

  return success();
}

} // namespace

// ---------------------------------------------------------------------------
// Pass class
// ---------------------------------------------------------------------------

class NVWSInsertTmemSemaphore
    : public triton::impl::NVWSInsertTmemSemaphoreBase<
          NVWSInsertTmemSemaphore> {
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
