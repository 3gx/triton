/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "WSUtility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#define DEBUG_TYPE "nvws-memory-planner"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton {

#define GEN_PASS_DEF_NVWSMEMORYPLANNER
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

using OperationListT = SmallVector<Operation *>;

enum class ChannelKind { LocalPost, TMEMPost };

struct TmemDataChannelPost {
  int producer;
  SmallVector<int> consumers;
  Operation *allocOp;
  Operation *explicitSrcOp = nullptr;
  SmallVector<Operation *> explicitDstOps;
  bool isOperandD;
  bool isOperandDNoAcc;
  bool isSameIterGuard = false;
  bool isPlannerOnly = false;
  unsigned uniqID;

  TmemDataChannelPost(int producer, ArrayRef<int> consumers,
                      Operation *allocOp, bool isOperandD,
                      bool isOperandDNoAcc, unsigned uniqID)
      : producer(producer), consumers(consumers.begin(), consumers.end()),
        allocOp(allocOp), isOperandD(isOperandD),
        isOperandDNoAcc(isOperandDNoAcc), uniqID(uniqID) {}

  TmemDataChannelPost(int producer, ArrayRef<int> consumers,
                      Operation *allocOp, Operation *srcOp,
                      ArrayRef<Operation *> dstOps, bool isOperandD,
                      bool isOperandDNoAcc, bool isPlannerOnly,
                      unsigned uniqID)
      : producer(producer), consumers(consumers.begin(), consumers.end()),
        allocOp(allocOp), explicitSrcOp(srcOp),
        explicitDstOps(dstOps.begin(), dstOps.end()), isOperandD(isOperandD),
        isOperandDNoAcc(isOperandDNoAcc), isPlannerOnly(isPlannerOnly),
        uniqID(uniqID) {}

  Operation *getAllocOp() const { return allocOp; }
  Operation *getSrcOp() const;
  Operation *getDstOp() const;
  void getDstOps(SmallVectorImpl<Operation *> &dsts) const;
};

struct LocalDataChannelPost {
  int producer;
  SmallVector<int> consumers;
  Operation *allocOp;
  Operation *explicitSrcOp = nullptr;
  SmallVector<Operation *> explicitDstOps;
  unsigned uniqID;

  LocalDataChannelPost(int producer, ArrayRef<int> consumers,
                       Operation *allocOp, Operation *srcOp,
                       ArrayRef<Operation *> dstOps, unsigned uniqID)
      : producer(producer), consumers(consumers.begin(), consumers.end()),
        allocOp(allocOp), explicitSrcOp(srcOp),
        explicitDstOps(dstOps.begin(), dstOps.end()), uniqID(uniqID) {}

  Operation *getAllocOp() const { return allocOp; }
  Operation *getSrcOp() const { return explicitSrcOp; }
  Operation *getDstOp() const {
    if (explicitDstOps.empty())
      return nullptr;
    return explicitDstOps.back();
  }
  void getDstOps(SmallVectorImpl<Operation *> &dsts) const {
    dsts.append(explicitDstOps.begin(), explicitDstOps.end());
  }
};

struct TmemBuffer {
  Operation *owner = nullptr;
  size_t rowSize = 0;
  size_t colSize = 0;
  size_t rowOffset = std::numeric_limits<size_t>::max();
  size_t colOffset = std::numeric_limits<size_t>::max();
  bool isOwnerOfSpace = false;
  TmemBuffer *reuseOwner = nullptr;
};

enum class SmemBufferPriority {
  Lowest = 0,
  HostToDevice = 1,
  CrossStage = 2,
  Epilogue = 3,
};

struct LocalBuffer {
  ttg::LocalAllocOp alloc;
  LocalDataChannelPost *channel = nullptr;
  unsigned bufferId = 0;
  unsigned numCopies = 1;
  unsigned offset = 0;
  unsigned sizeInBytes = 0;
  bool pinned = false;
  bool isInnermost = false;
  bool isTMA = false;
  bool isCrossStage = false;
  bool isCircular = false;
  unsigned circularStart = 0;
  SmemBufferPriority priority = SmemBufferPriority::Lowest;
  Interval<size_t> liveness = Interval<size_t>(0, 0);
};

struct ChannelAnnotation {
  std::string operand;
  std::string memType;
  unsigned numCopies;
  unsigned bufferId;
};

static void setI32Attr(Operation *op, StringRef name, int32_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     value));
}

static void setUnitAttr(Operation *op, StringRef name) {
  op->setAttr(name, UnitAttr::get(op->getContext()));
}

static void eraseAttr(Operation *op, StringRef name) {
  if (op->hasAttr(name))
    op->removeAttr(name);
}

static bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations())
    if (isa<scf::ForOp>(nestedOp))
      return false;
  return true;
}

static void buildOperationIdMap(Operation *operation,
                                DenseMap<Operation *, size_t> &operationId) {
  operation->walk<WalkOrder::PostOrder>(
      [&](Operation *op) { operationId[op] = operationId.size(); });
}

static Interval<size_t> intervalFromOps(ArrayRef<Operation *> liveOps,
                                        DenseMap<Operation *, size_t> &opId) {
  if (liveOps.empty())
    return Interval<size_t>(0, 0);
  size_t minId = std::numeric_limits<size_t>::max();
  size_t maxId = std::numeric_limits<size_t>::min();
  for (Operation *liveOp : liveOps) {
    auto it = opId.find(liveOp);
    if (it == opId.end())
      continue;
    minId = std::min(minId, it->second);
    maxId = std::max(maxId, it->second + 1);
  }
  if (minId == std::numeric_limits<size_t>::max())
    return Interval<size_t>(0, 0);
  return Interval<size_t>(minId, maxId);
}

static Interval<size_t>
getIntervalForCtrlOp(Operation *ctrlOp,
                     DenseMap<Operation *, size_t> &operationId) {
  auto forOp = dyn_cast_or_null<scf::ForOp>(ctrlOp);
  if (!forOp)
    return Interval<size_t>(0, 0);
  for (Operation &op : forOp.getBody()->without_terminator())
    return Interval<size_t>(operationId[&op], operationId[ctrlOp]);
  return Interval<size_t>(operationId[ctrlOp], operationId[ctrlOp]);
}

static Operation *skipIdxOp(Operation *op) {
  if (auto idx = dyn_cast_or_null<ttg::MemDescIndexOp>(op)) {
    Operation *first = nullptr;
    unsigned numUsers = 0;
    for (Operation *user : idx->getUsers()) {
      first = user;
      ++numUsers;
    }
    if (numUsers <= 1)
      return first;
  }
  return op;
}

static bool isConstFalse(Value v) {
  if (!v)
    return false;
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    Attribute value = constOp.getValue();
    if (auto boolAttr = dyn_cast<BoolAttr>(value))
      return !boolAttr.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(value))
      return intAttr.getInt() == 0;
  }
  return false;
}

static bool isLoopCarriedInitConstFalse(Value v, scf::ForOp forOp) {
  // A loop-carried use_acc flag initialized to false means the first MMA
  // iteration fully overwrites operand D, so it can seed producer tracking.
  auto blockArg = dyn_cast<BlockArgument>(v);
  if (!blockArg || blockArg.getOwner() != forOp.getBody())
    return false;

  unsigned argNum = blockArg.getArgNumber();
  unsigned numInductionVars = forOp.getNumInductionVars();
  if (argNum < numInductionVars)
    return false;

  return isConstFalse(forOp.getInitArgs()[argNum - numInductionVars]);
}

static Value getTmemAllocValue(Operation *allocOp) {
  auto alloc = cast<ttng::TMEMAllocOp>(allocOp);
  return alloc.getResult();
}

static bool isMmaAccumulatorUse(Value allocOrSubview, Operation *user) {
  auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user);
  return mmaOp && mmaOp.getAccumulator() == allocOrSubview;
}

static bool isTmemProducer(Value allocOrSubview, Operation *user) {
  if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user))
    return mmaOp.getAccumulator() == allocOrSubview;
  return isa<ttng::TMEMStoreOp>(user);
}

static SmallVector<int> getTaskIds(Operation *op) {
  return nvws::getAsyncTaskIds(op);
}

static std::map<std::pair<Operation *, unsigned>, ChannelAnnotation>
parseChannelAnnotations(Operation *parentOp) {
  std::map<std::pair<Operation *, unsigned>, ChannelAnnotation> result;
  std::map<unsigned, unsigned> bufferIdToCopies;

  parentOp->walk([&](Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>("tt.autows");
    if (!attr)
      return;

    auto parsed = llvm::json::parse(attr.getValue());
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return;
    }

    auto *obj = parsed->getAsObject();
    if (!obj)
      return;
    auto *channels = obj->getArray("channels");
    if (!channels)
      return;

    for (auto &elem : *channels) {
      auto str = elem.getAsString();
      if (!str)
        continue;

      SmallVector<StringRef, 4> parts;
      StringRef(*str).split(parts, ',');
      if (parts.size() != 4)
        continue;

      ChannelAnnotation ann;
      ann.operand = parts[0].str();
      ann.memType = parts[1].str();
      ann.numCopies = 0;
      ann.bufferId = 0;
      if (parts[2].getAsInteger(10, ann.numCopies) ||
          parts[3].getAsInteger(10, ann.bufferId))
        continue;

      if (ann.operand != "opndA" && ann.operand != "opndB" &&
          ann.operand != "opndD")
        continue;
      if (ann.memType != "smem" && ann.memType != "tmem")
        continue;
      if (ann.operand == "opndD")
        ann.memType = "tmem";

      auto it = bufferIdToCopies.find(ann.bufferId);
      if (it != bufferIdToCopies.end()) {
        ann.numCopies = std::max(ann.numCopies, it->second);
        it->second = ann.numCopies;
      } else {
        bufferIdToCopies[ann.bufferId] = ann.numCopies;
      }

      unsigned operandIdx = ann.operand == "opndA"   ? 0
                            : ann.operand == "opndB" ? 1
                                                     : 2;
      result[{op, operandIdx}] = ann;
    }
  });

  return result;
}

static Operation *traceBackToAlloc(Value value) {
  DenseSet<Value> visited;
  SmallVector<Value> worklist = {value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      continue;
    if (isa<ttg::LocalAllocOp, ttng::TMEMAllocOp>(defOp))
      return defOp;

    for (Value operand : defOp->getOperands())
      worklist.push_back(operand);
  }
  return nullptr;
}

static DenseMap<Operation *, ChannelAnnotation> buildAllocToAnnotationMap(
    const std::map<std::pair<Operation *, unsigned>, ChannelAnnotation>
        &annotations) {
  DenseMap<Operation *, ChannelAnnotation> result;
  for (const auto &[key, ann] : annotations) {
    auto [op, operandIdx] = key;
    auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op);
    if (!mmaOp)
      continue;

    Value operand;
    if (operandIdx == 0)
      operand = mmaOp.getA();
    else if (operandIdx == 1)
      operand = mmaOp.getB();
    else
      operand = mmaOp.getAccumulator();

    Operation *allocOp = traceBackToAlloc(operand);
    if (!allocOp)
      continue;

    if ((isa<ttng::TMEMAllocOp>(allocOp) && ann.memType == "tmem") ||
        (isa<ttg::LocalAllocOp>(allocOp) && ann.memType == "smem"))
      result.try_emplace(allocOp, ann);
  }
  return result;
}

static bool needsChannel(int producer, ArrayRef<int> consumers) {
  return !llvm::all_of(consumers, [producer](int consumerId) {
    return consumerId == producer;
  });
}

static SmallVector<int> getUniqueTaskIds(ArrayRef<Operation *> ops) {
  SmallVector<int> taskIds;
  DenseSet<int> seenTaskIds;
  for (Operation *op : ops) {
    for (int id : getTaskIds(op)) {
      if (seenTaskIds.insert(id).second)
        taskIds.push_back(id);
    }
  }
  return taskIds;
}

static void setTmemChannelAttr(Operation *op, int channelId,
                               StringRef attrName) {
  SmallVector<int> ids;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>(attrName))
    ids.append(attr.asArrayRef().begin(), attr.asArrayRef().end());
  ids.push_back(channelId);
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  op->setAttr(attrName, DenseI32ArrayAttr::get(op->getContext(), ids));
}

static Operation *findTmemStartEnd(const TmemDataChannelPost *ch,
                                   StringRef attrName) {
  auto alloc = cast<ttng::TMEMAllocOp>(ch->allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    DenseSet<int> channelIds;
    if (auto attr = user->getAttrOfType<DenseI32ArrayAttr>(attrName)) {
      for (int asyncTaskId : attr.asArrayRef())
        channelIds.insert(asyncTaskId);
      if (channelIds.contains(ch->uniqID))
        return user;
    }
  }
  return nullptr;
}

Operation *TmemDataChannelPost::getSrcOp() const {
  if (explicitSrcOp)
    return explicitSrcOp;

  if (isOperandD)
    return findTmemStartEnd(this, "tmem.start");

  auto alloc = cast<ttng::TMEMAllocOp>(allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value producerValue = user == usr ? getTmemAllocValue(allocOp)
                                      : usr->getResult(0);
    if (isTmemProducer(producerValue, user))
      return user;
  }
  return nullptr;
}

static void getAllConsumers(const TmemDataChannelPost *ch,
                            SmallVectorImpl<Operation *> &consumers) {
  auto alloc = cast<ttng::TMEMAllocOp>(ch->allocOp);
  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value producerValue = user == usr ? getTmemAllocValue(ch->allocOp)
                                      : usr->getResult(0);
    if (!isTmemProducer(producerValue, user))
      consumers.push_back(user);
  }
}

Operation *TmemDataChannelPost::getDstOp() const {
  if (!explicitDstOps.empty())
    return explicitDstOps.back();

  if (isOperandD)
    return findTmemStartEnd(this, "tmem.end");

  SmallVector<Operation *> allConsumers;
  getAllConsumers(this, allConsumers);
  if (allConsumers.empty())
    return nullptr;
  return allConsumers.back();
}

void TmemDataChannelPost::getDstOps(
    SmallVectorImpl<Operation *> &dsts) const {
  if (!explicitDstOps.empty()) {
    dsts.append(explicitDstOps.begin(), explicitDstOps.end());
    return;
  }

  if (isOperandD) {
    if (Operation *dst = getDstOp())
      dsts.push_back(dst);
    return;
  }
  getAllConsumers(this, dsts);
}

static void
createChannelsForProducers(SmallVectorImpl<Operation *> &currentProds,
                           int producerTaskId, ArrayRef<int> consumerIds,
                           Operation *allocOp, Operation *consumerOp,
                           SmallVectorImpl<
                               std::unique_ptr<TmemDataChannelPost>> &channels,
                           bool isSameIterGuard = false) {
  for (Operation *prod : currentProds) {
    auto channelId = channels.size();
    auto channel = std::make_unique<TmemDataChannelPost>(
        producerTaskId, consumerIds, allocOp, true /*isOperandD*/,
        true /*isOperandDNoAcc*/, channelId);
    channel->isSameIterGuard = isSameIterGuard;
    channels.push_back(std::move(channel));
    setTmemChannelAttr(prod, channelId, "tmem.start");
    setTmemChannelAttr(consumerOp, channelId, "tmem.end");
  }
}

static LogicalResult
handleOperandD(ttng::TMEMAllocOp tmemAllocOp,
               ttng::MMAv5OpInterface representativeMma,
               SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>>
                   &channels) {
  DenseSet<Operation *> users;
  DenseSet<Operation *> handledUsers;
  for (Operation *user : tmemAllocOp.getResult().getUsers())
    users.insert(skipIdxOp(user));

  auto forOp = representativeMma->getParentOfType<scf::ForOp>();
  if (!forOp)
    return representativeMma->emitError(
        "NVWS memory planner expected operand-D MMA inside scf.for");

  SmallVector<Operation *> currentProds;
  SmallVector<int> channelsToUpdate;
  Operation *firstProducer = nullptr;
  Operation *lastConsumer = nullptr;
  unsigned numChannelsCreated = 0;

  for (Operation *user : tmemAllocOp.getResult().getUsers()) {
    Operation *actual = skipIdxOp(user);
    if (auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(actual)) {
      if (!forOp->isProperAncestor(storeOp)) {
        currentProds.clear();
        currentProds.push_back(storeOp);
        handledUsers.insert(storeOp);
      }
    }
  }

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!users.contains(&op))
      continue;
    handledUsers.insert(&op);

    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
      if (mmaOp.getAccumulator() == tmemAllocOp.getResult()) {
        if (currentProds.empty() &&
            (isConstFalse(mmaOp.useAccumulator()) ||
             isLoopCarriedInitConstFalse(mmaOp.useAccumulator(), forOp))) {
          currentProds.push_back(&op);
          continue;
        }
        if (currentProds.empty())
          return op.emitError("NVWS memory planner found no producer for MMA "
                              "operand-D accumulator");

        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          currentProds.push_back(&op);
          continue;
        }

        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
          currentProds.clear();
          currentProds.push_back(&op);
        } else {
          currentProds.push_back(&op);
        }
      } else {
        if (currentProds.empty())
          return op.emitError("NVWS memory planner found no producer for "
                              "TMEM MMA consumer");

        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          currentProds.push_back(&op);
          continue;
        }

        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
        } else {
          currentProds.push_back(&op);
        }
      }
    } else if (isa<ttng::TMEMStoreOp>(&op)) {
      currentProds.clear();
      currentProds.push_back(&op);
    } else if (isa<ttng::TMEMLoadOp>(&op)) {
      if (!currentProds.empty()) {
        SmallVector<int> producerIds = getTaskIds(currentProds.front());
        SmallVector<int> consumerIds = getTaskIds(&op);
        if (producerIds.size() != 1) {
          currentProds.push_back(&op);
          continue;
        }
        int producerId = producerIds.front();
        if (needsChannel(producerId, consumerIds)) {
          if (!firstProducer)
            firstProducer = currentProds.front();
          lastConsumer = &op;
          ++numChannelsCreated;
          createChannelsForProducers(currentProds, producerId, consumerIds,
                                     tmemAllocOp.getOperation(), &op,
                                     channels);
        } else {
          currentProds.push_back(&op);
        }
      } else {
        unsigned channelId = channels.size();
        channelsToUpdate.push_back(channelId);
        channels.push_back(std::make_unique<TmemDataChannelPost>(
            -1, getTaskIds(&op), tmemAllocOp.getOperation(),
            true /*isOperandD*/, true /*isOperandDNoAcc*/, channelId));
        setTmemChannelAttr(&op, channelId, "tmem.end");
      }
    } else {
      return op.emitError("NVWS memory planner found unsupported TMEM user");
    }
  }

  for (int idx : channelsToUpdate) {
    if (currentProds.empty())
      return representativeMma->emitError(
          "NVWS memory planner found no producer for deferred TMEM channel");
    Operation *lastProd = currentProds.back();
    SmallVector<int> producerIds = getTaskIds(lastProd);
    if (producerIds.size() != 1)
      continue;
    channels[idx]->producer = producerIds.front();
    setTmemChannelAttr(lastProd, channels[idx]->uniqID, "tmem.start");
  }

  for (Operation *user : users) {
    if (handledUsers.contains(user))
      continue;
    if (!isa_and_nonnull<ttng::TMEMLoadOp>(user))
      return user->emitError(
          "NVWS memory planner found unsupported post-loop TMEM user");
    if (currentProds.empty())
      return user->emitError(
          "NVWS memory planner found no producer for post-loop TMEM load");

    SmallVector<int> producerIds = getTaskIds(currentProds.front());
    SmallVector<int> consumerIds = getTaskIds(user);
    if (producerIds.size() != 1)
      continue;
    int producerId = producerIds.front();
    if (needsChannel(producerId, consumerIds)) {
      if (!firstProducer)
        firstProducer = currentProds.front();
      lastConsumer = user;
      ++numChannelsCreated;
      createChannelsForProducers(currentProds, producerId, consumerIds,
                                 tmemAllocOp.getOperation(), user, channels);
    }
  }

  if (numChannelsCreated >= 2 && firstProducer && lastConsumer &&
      firstProducer->getBlock() == lastConsumer->getBlock()) {
    SmallVector<int> firstProducerIds = getTaskIds(firstProducer);
    SmallVector<int> lastConsumerIds = getTaskIds(lastConsumer);
    if (firstProducerIds.size() == 1 &&
        needsChannel(firstProducerIds.front(), lastConsumerIds)) {
      SmallVector<Operation *> producers = {firstProducer};
      createChannelsForProducers(producers, firstProducerIds.front(),
                                 lastConsumerIds, tmemAllocOp.getOperation(),
                                 lastConsumer, channels);
    }

    if (lastConsumerIds.size() == 1 && isa<ttng::TMEMLoadOp>(lastConsumer) &&
        isa<ttng::TMEMStoreOp>(firstProducer) &&
        needsChannel(lastConsumerIds.front(), firstProducerIds)) {
      unsigned channelId = channels.size();
      auto guard = std::make_unique<TmemDataChannelPost>(
          lastConsumerIds.front(), firstProducerIds, tmemAllocOp.getOperation(),
          true /*isOperandD*/, false /*isOperandDNoAcc*/, channelId);
      guard->isSameIterGuard = true;
      channels.push_back(std::move(guard));
      setTmemChannelAttr(lastConsumer, channelId, "tmem.start");
      setTmemChannelAttr(firstProducer, channelId, "tmem.end");
    }
  }

  return success();
}

static LogicalResult
createTmemChannelPost(ttng::TMEMAllocOp alloc,
                      SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>>
                          &channels) {
  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
  ttng::MMAv5OpInterface operandDMma;
  bool isOperandD = false;
  bool isOperandDNoAcc = false;

  for (Operation *usr : alloc.getResult().getUsers()) {
    Operation *user = skipIdxOp(usr);
    if (!user)
      continue;
    Value accessValue =
        user == usr ? Value(alloc.getResult()) : Value(usr->getResult(0));
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      if (mmaOp.getAccumulator() == accessValue) {
        if (user != usr)
          return user->emitError("NVWS memory planner does not support "
                                 "partial-view TMEM producer modeling");
        if (!isConstFalse(mmaOp.useAccumulator())) {
          operandDMma = mmaOp;
          isOperandD = true;
        } else {
          isOperandDNoAcc = true;
          producers.push_back(user);
        }
      } else {
        consumers.push_back(user);
      }
    } else if (isa<ttng::TMEMStoreOp>(user)) {
      if (user != usr)
        return user->emitError("NVWS memory planner does not support "
                               "partial-view TMEM producer modeling");
      producers.push_back(user);
    } else if (isa<ttng::TMEMLoadOp>(user)) {
      consumers.push_back(user);
    } else {
      return user->emitError("NVWS memory planner found unsupported TMEM user");
    }
  }

  if (isOperandD)
    return handleOperandD(alloc, operandDMma, channels);

  if (producers.empty()) {
    if (!alloc.getSrc())
      return success();
    if (consumers.empty())
      return success();

    SmallVector<int> producerIds = getTaskIds(alloc.getOperation());
    if (producerIds.size() != 1)
      return alloc.emitError("NVWS memory planner expected sourceful "
                             "ttng.tmem_alloc to have exactly one partition");

    SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
    consumerTaskIds.erase(std::remove(consumerTaskIds.begin(),
                                      consumerTaskIds.end(),
                                      producerIds.front()),
                          consumerTaskIds.end());

    // Sourceful ttng.tmem_alloc is planner-equivalent to a hoisted storage
    // allocation plus an init store at the real alloc op. Keep that producer
    // record internal to the planner; InsertTmemSemaphore must not see it as an
    // extra real channel.
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerIds.front(), consumerTaskIds, alloc.getOperation(),
        alloc.getOperation(), consumers, false /*isOperandD*/,
        false /*isOperandDNoAcc*/, true /*isPlannerOnly*/, channels.size()));
    return success();
  }

  Operation *producerOp = producers.front();
  if (producers.size() > 1 && !consumers.empty()) {
    producerOp = nullptr;
    for (Operation *prod : producers) {
      if (prod->getBlock() == consumers.front()->getBlock()) {
        if (producerOp)
          return prod->emitError(
              "NVWS memory planner found ambiguous TMEM producers");
        producerOp = prod;
      }
    }
    if (!producerOp)
      producerOp = producers.front();
  }

  SmallVector<int> producerIds = getTaskIds(producerOp);
  if (producerIds.size() != 1)
    return success();
  int producerId = producerIds.front();

  SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
  consumerTaskIds.erase(
      std::remove(consumerTaskIds.begin(), consumerTaskIds.end(), producerId),
      consumerTaskIds.end());

  if (needsChannel(producerId, consumerTaskIds)) {
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerId, consumerTaskIds, alloc.getOperation(),
        false /*isOperandD*/, isOperandDNoAcc, channels.size()));
  } else if (!consumers.empty()) {
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerId, consumerTaskIds, alloc.getOperation(), producerOp,
        consumers, false /*isOperandD*/, isOperandDNoAcc,
        true /*isPlannerOnly*/, channels.size()));
  }

  return success();
}

static LogicalResult collectTmemPostChannels(
    SmallVectorImpl<std::unique_ptr<TmemDataChannelPost>> &channels,
    FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](ttng::TMEMAllocOp alloc) {
    if (failed(createTmemChannelPost(alloc, channels)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static TmemDataChannelPost *
findChannelForAlloc(Operation *allocOp,
                    ArrayRef<std::unique_ptr<TmemDataChannelPost>> channels) {
  for (const auto &channel : channels) {
    if (channel->allocOp == allocOp) {
      if (channel->isSameIterGuard)
        continue;
      return channel.get();
    }
  }
  return nullptr;
}

static bool isTransparentMemdescViewOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

static void collectTransitiveMemdescUsers(Value value,
                                          DenseSet<Operation *> &users) {
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      Operation *actual = skipIdxOp(user);
      if (!actual)
        continue;
      users.insert(actual);
      if (actual == user && isTransparentMemdescViewOp(actual))
        for (Value result : actual->getResults())
          worklist.push_back(result);
    }
  }
}

static void collectLocalMemdescUsers(Value value,
                                     DenseSet<Operation *> &users) {
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      Operation *actual = skipIdxOp(user);
      if (!actual)
        continue;
      users.insert(actual);

      if (actual == user && isTransparentMemdescViewOp(actual)) {
        for (Value result : actual->getResults())
          worklist.push_back(result);
        continue;
      }

      if (isa<nvws::DescriptorLoadOp, nvws::DescriptorGatherOp>(actual)) {
        for (Value result : actual->getResults())
          if (isa<ttg::MemDescType>(result.getType()))
            worklist.push_back(result);
      }
    }
  }
}

static bool isLocalProducer(Operation *op) {
  return isa<ttg::LocalStoreOp, nvws::DescriptorLoadOp,
             nvws::DescriptorGatherOp>(op);
}

static Operation *producerForSourcefulLocalAlloc(ttg::LocalAllocOp alloc) {
  if (!alloc.getSrc())
    return nullptr;
  return alloc.getOperation();
}

static LogicalResult createLocalChannelPost(
    ttg::LocalAllocOp alloc,
    SmallVectorImpl<std::unique_ptr<LocalDataChannelPost>> &channels) {
  if (!alloc.isSharedMemoryAlloc())
    return success();

  DenseSet<Operation *> users;
  collectLocalMemdescUsers(alloc.getResult(), users);

  SmallVector<Operation *> producers;
  SmallVector<Operation *> consumers;
  if (Operation *producer = producerForSourcefulLocalAlloc(alloc))
    producers.push_back(producer);

  for (Operation *user : users) {
    if (isTransparentMemdescViewOp(user))
      continue;
    if (isLocalProducer(user)) {
      producers.push_back(user);
      continue;
    }
    consumers.push_back(user);
  }

  if (producers.empty() && consumers.empty())
    return success();

  Operation *producerOp =
      producers.empty() ? alloc.getOperation() : producers.front();
  SmallVector<int> producerIds = getTaskIds(producerOp);
  int producerId = producerIds.size() == 1 ? producerIds.front() : -1;

  SmallVector<int> consumerTaskIds = getUniqueTaskIds(consumers);
  if (producerId >= 0)
    consumerTaskIds.erase(std::remove(consumerTaskIds.begin(),
                                      consumerTaskIds.end(), producerId),
                          consumerTaskIds.end());

  channels.push_back(std::make_unique<LocalDataChannelPost>(
      producerId, consumerTaskIds, alloc.getOperation(), producerOp, consumers,
      channels.size()));
  return success();
}

static LogicalResult collectLocalPostChannels(
    SmallVectorImpl<std::unique_ptr<LocalDataChannelPost>> &channels,
    FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](ttg::LocalAllocOp alloc) {
    if (failed(createLocalChannelPost(alloc, channels)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static LocalDataChannelPost *
findChannelForAlloc(Operation *allocOp,
                    ArrayRef<std::unique_ptr<LocalDataChannelPost>> channels) {
  for (const auto &channel : channels)
    if (channel->allocOp == allocOp)
      return channel.get();
  return nullptr;
}

static OperationListT
livenessForLocalAlloc(ttg::LocalAllocOp alloc,
                      ArrayRef<std::unique_ptr<LocalDataChannelPost>>
                          channels) {
  OperationListT liveOps;
  DenseSet<Operation *> users;
  if (LocalDataChannelPost *channel =
          findChannelForAlloc(alloc.getOperation(), channels)) {
    if (Operation *src = channel->getSrcOp())
      users.insert(src);
    SmallVector<Operation *> dsts;
    channel->getDstOps(dsts);
    for (Operation *dst : dsts)
      users.insert(dst);
  } else {
    collectLocalMemdescUsers(alloc.getResult(), users);
  }
  liveOps.append(users.begin(), users.end());
  if (liveOps.empty())
    liveOps.push_back(alloc.getOperation());
  return liveOps;
}

static LogicalResult getAllTmemUsers(TmemDataChannelPost *channel,
                                     DenseSet<Operation *> &users,
                                     Operation *allocOp) {
  if (!channel) {
    collectTransitiveMemdescUsers(cast<ttng::TMEMAllocOp>(allocOp).getResult(),
                                  users);
    return success();
  }

  if (Operation *src = channel->getSrcOp())
    users.insert(src);

  if (channel->isOperandD) {
    collectTransitiveMemdescUsers(
        cast<ttng::TMEMAllocOp>(channel->allocOp).getResult(), users);
  } else {
    SmallVector<Operation *> dsts;
    channel->getDstOps(dsts);
    for (Operation *dst : dsts)
      users.insert(dst);
  }
  return success();
}

static OperationListT
livenessForTmemAlloc(ttng::TMEMAllocOp alloc,
                     ArrayRef<std::unique_ptr<TmemDataChannelPost>> channels) {
  OperationListT liveOps;
  DenseSet<Operation *> users;
  if (failed(getAllTmemUsers(findChannelForAlloc(alloc, channels), users,
                             alloc.getOperation())))
    return liveOps;
  liveOps.append(users.begin(), users.end());
  if (liveOps.empty())
    liveOps.push_back(alloc.getOperation());
  return liveOps;
}

static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  auto parent = op->getParentOfType<scf::ForOp>();
  while (parent) {
    ++depth;
    parent = parent->getParentOfType<scf::ForOp>();
  }
  return depth;
}

static bool isDataDependent(Operation *srcOp, Operation *dstOp) {
  if (!srcOp || !dstOp)
    return false;
  SmallVector<Operation *, 16> worklist;
  DenseSet<Operation *> visited;
  auto enqueueUsers = [&](Operation *op) {
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        if (visited.insert(user).second)
          worklist.push_back(user);
    if (isa<ttg::LocalStoreOp, ttng::TMEMStoreOp>(op))
      for (Value operand : op->getOperands())
        if (isa<ttg::MemDescType>(operand.getType()))
          for (Operation *user : operand.getUsers())
            if (user != op && visited.insert(user).second)
              worklist.push_back(user);
  };
  enqueueUsers(srcOp);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (op == dstOp)
      return true;
    enqueueUsers(op);
  }
  return false;
}

static unsigned getLocalAllocSizeBytes(ttg::LocalAllocOp alloc) {
  auto allocType = alloc.getType();
  int64_t numElems = 1;
  if (auto paddedEnc =
          dyn_cast<ttg::PaddedSharedEncodingAttr>(allocType.getEncoding())) {
    SmallVector<int64_t> unpaddedShape = ttg::getShapePerCTA(allocType);
    numElems = paddedEnc.getPaddedSize(unpaddedShape);
  } else {
    SmallVector<int64_t> shapePerCTA =
        ttg::getAllocationShapePerCTA(allocType);
    for (int64_t dim : shapePerCTA)
      numElems *= dim;
  }
  return static_cast<unsigned>(numElems * allocType.getElementTypeBitWidth() /
                               8);
}

static bool hasLoopStage(Operation *op) {
  if (!op)
    return false;
  if (op->getAttrOfType<IntegerAttr>("loop.stage"))
    return true;
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    if (Value src = alloc.getSrc())
      return hasLoopStage(src.getDefiningOp());
  if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
    return hasLoopStage(store.getSrc().getDefiningOp());
  return false;
}

static bool isDescriptorLoadProducer(Operation *op) {
  if (!op)
    return false;
  if (isa<nvws::DescriptorLoadOp, nvws::DescriptorGatherOp,
          tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
    return true;
  if (auto store = dyn_cast<ttg::LocalStoreOp>(op))
    return isDescriptorLoadProducer(store.getSrc().getDefiningOp());
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op))
    if (Value src = alloc.getSrc())
      return isDescriptorLoadProducer(src.getDefiningOp());
  return false;
}

static Operation *findOriginalLoadOp(Value value, DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return nullptr;
  Operation *def = value.getDefiningOp();
  if (!def)
    return nullptr;
  if (isa<ttng::TMEMLoadOp>(def))
    return def;
  for (Value operand : def->getOperands()) {
    if (isa<ttg::MemDescType>(operand.getType()))
      continue;
    if (Operation *load = findOriginalLoadOp(operand, visited))
      return load;
  }
  return nullptr;
}

static Operation *findOriginalLoadOp(Value value) {
  DenseSet<Value> visited;
  return findOriginalLoadOp(value, visited);
}

static Operation *findOriginalLoadOp(LocalDataChannelPost *channel) {
  if (!channel)
    return nullptr;
  Operation *srcOp = channel->getSrcOp();
  if (auto store = dyn_cast_or_null<ttg::LocalStoreOp>(srcOp))
    return findOriginalLoadOp(store.getSrc());
  if (auto alloc = dyn_cast_or_null<ttg::LocalAllocOp>(srcOp))
    if (Value src = alloc.getSrc())
      return findOriginalLoadOp(src);
  return nullptr;
}

static bool usersInInnermostLoop(LocalDataChannelPost *channel) {
  if (!channel)
    return false;
  SmallVector<Operation *> users;
  if (Operation *src = channel->getSrcOp())
    users.push_back(src);
  channel->getDstOps(users);
  if (users.empty())
    return false;

  Operation *first = users.front();
  for (Operation *user : users)
    if (user->getBlock() != first->getBlock())
      return false;
  auto parentLoop = first->getParentOfType<scf::ForOp>();
  return parentLoop && isInnermostLoop(parentLoop);
}

class LocalSmemAllocator {
public:
  LocalSmemAllocator(FuncOp funcOp,
                     SmallVector<std::unique_ptr<LocalDataChannelPost>>
                         &channels,
                     unsigned numBuffers, int smemAllocAlgo,
                     unsigned smemBudget, bool smemCircularReuse)
      : funcOp(funcOp), channels(channels),
        numBuffers(std::max(1u, numBuffers)), smemAllocAlgo(smemAllocAlgo),
        smemBudget(smemBudget), smemCircularReuse(smemCircularReuse) {}

  LogicalResult run(unsigned &nextBufferId) {
    buildOperationIdMap(funcOp, operationId);
    collectAllocs();
    nextBufferId = 0;
    if (buffers.empty())
      return success();

    auto annotations = parseChannelAnnotations(funcOp);
    smemAnnotations = buildAllocToAnnotationMap(annotations);

    if (smemAllocAlgo == 1) {
      if (smemBudget == 0)
        return funcOp.emitError("NVWS memory planner requires smem-budget for "
                                "smem-alloc-algo=1");
      runWSBufferPlan(nextBufferId);
    } else {
      runLegacyPlan(nextBufferId);
    }

    emitAttrs();
    return success();
  }

private:
  void collectAllocs() {
    funcOp.walk<WalkOrder::PreOrder>([&](ttg::LocalAllocOp alloc) {
      if (!alloc.isSharedMemoryAlloc())
        return;
      auto buffer = std::make_unique<LocalBuffer>();
      buffer->alloc = alloc;
      buffer->channel = findChannelForAlloc(alloc.getOperation(), channels);
      buffer->sizeInBytes = getLocalAllocSizeBytes(alloc);
      buffer->isInnermost = usersInInnermostLoop(buffer->channel);
      buffer->isTMA = isDescriptorLoadProducer(
          buffer->channel ? buffer->channel->getSrcOp() : alloc.getOperation());
      buffer->isCrossStage =
          hasLoopStage(buffer->channel ? buffer->channel->getSrcOp()
                                       : alloc.getOperation());
      buffer->liveness =
          intervalFromOps(livenessForLocalAlloc(alloc, channels), operationId);
      buffers.push_back(std::move(buffer));
    });
  }

  bool isTwoDimensional(ttg::LocalAllocOp alloc) {
    return alloc.getType().getShape().size() >= 2;
  }

  void runLegacyPlan(unsigned &nextBufferId) {
    DenseMap<Type, unsigned> innermostBufferIds;
    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      buffer.pinned = false;
      buffer.offset = 0;

      auto annIt = smemAnnotations.find(buffer.alloc.getOperation());
      if (annIt != smemAnnotations.end()) {
        const ChannelAnnotation &ann = annIt->second;
        buffer.pinned = true;
        buffer.bufferId = ann.bufferId;
        buffer.numCopies = std::max(1u, ann.numCopies);
        nextBufferId = std::max(nextBufferId, ann.bufferId + 1);
        continue;
      }

      if (buffer.isInnermost && isTwoDimensional(buffer.alloc)) {
        Type elementType = buffer.alloc.getType().getElementType();
        auto it = innermostBufferIds.find(elementType);
        if (it == innermostBufferIds.end()) {
          it = innermostBufferIds.insert({elementType, nextBufferId++}).first;
        }
        buffer.bufferId = it->second;
        buffer.numCopies = numBuffers;
        buffer.isCircular = true;
      } else {
        buffer.bufferId = nextBufferId++;
        buffer.numCopies = 1;
      }
    }

    enforceMinCopyForSharedIds(/*cyclicOnly=*/false);
    fuseEpilogueBuffers();
  }

  void runWSBufferPlan(unsigned &nextBufferId) {
    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      buffer.offset = 0;
      auto annIt = smemAnnotations.find(buffer.alloc.getOperation());
      if (annIt != smemAnnotations.end()) {
        const ChannelAnnotation &ann = annIt->second;
        buffer.pinned = true;
        buffer.bufferId = ann.bufferId;
        buffer.numCopies = std::max(1u, ann.numCopies);
        nextBufferId = std::max(nextBufferId, ann.bufferId + 1);
      } else {
        buffer.pinned = false;
        buffer.bufferId = nextBufferId++;
        buffer.numCopies = 1;
      }
      buffer.priority = classifyPriority(buffer);
    }

    fuseEpilogueBuffers();

    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      if (buffer.pinned)
        continue;
      unsigned targetCopies = buffer.priority == SmemBufferPriority::Lowest
                                  ? 1
                                  : numBuffers;
      growCopiesWithinBudget(buffer, targetCopies);
    }

    if (smemCircularReuse)
      coalesceCircularReuseCandidates();
    enforceMinCopyForSharedIds(/*cyclicOnly=*/true);
  }

  SmemBufferPriority classifyPriority(const LocalBuffer &buffer) const {
    if (findOriginalLoadOp(buffer.channel))
      return SmemBufferPriority::Epilogue;
    if (buffer.isCrossStage)
      return SmemBufferPriority::CrossStage;
    if (buffer.isInnermost || buffer.isTMA)
      return SmemBufferPriority::HostToDevice;
    return SmemBufferPriority::Lowest;
  }

  unsigned computeTotalSmem() const {
    DenseMap<unsigned, std::pair<unsigned, unsigned>> idInfo;
    for (const auto &bufferPtr : buffers) {
      const LocalBuffer &buffer = *bufferPtr;
      auto it = idInfo.find(buffer.bufferId);
      if (it == idInfo.end()) {
        idInfo[buffer.bufferId] = {buffer.sizeInBytes, buffer.numCopies};
      } else {
        it->second.first = std::max(it->second.first, buffer.sizeInBytes);
        it->second.second = std::max(it->second.second, buffer.numCopies);
      }
    }

    unsigned total = 0;
    for (const auto &[id, sizeAndCopies] : idInfo)
      total += sizeAndCopies.first * sizeAndCopies.second;
    return total;
  }

  void growCopiesWithinBudget(LocalBuffer &buffer, unsigned targetCopies) {
    while (buffer.numCopies < targetCopies) {
      unsigned oldCopies = buffer.numCopies;
      ++buffer.numCopies;
      if (computeTotalSmem() <= smemBudget)
        continue;
      buffer.numCopies = oldCopies;
      break;
    }
  }

  void enforceMinCopyForSharedIds(bool cyclicOnly) {
    DenseMap<unsigned, unsigned> idToCount;
    for (const auto &bufferPtr : buffers) {
      const LocalBuffer &buffer = *bufferPtr;
      if (!cyclicOnly || buffer.isInnermost || buffer.isCrossStage)
        ++idToCount[buffer.bufferId];
    }

    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      auto it = idToCount.find(buffer.bufferId);
      if (it == idToCount.end())
        continue;
      buffer.numCopies = std::max(buffer.numCopies, it->second);
    }
  }

  bool compatibleForReuse(const LocalBuffer &a, const LocalBuffer &b) const {
    auto aType = cast<ttg::MemDescType>(a.alloc->getResult(0).getType());
    auto bType = cast<ttg::MemDescType>(b.alloc->getResult(0).getType());
    return aType.getElementType() == bType.getElementType() &&
           a.sizeInBytes == b.sizeInBytes;
  }

  void fuseEpilogueBuffers() {
    DenseMap<Operation *, LocalBuffer *> originalLoadToOwner;
    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      Operation *originalLoad = findOriginalLoadOp(buffer.channel);
      if (!originalLoad)
        continue;
      auto it = originalLoadToOwner.find(originalLoad);
      if (it == originalLoadToOwner.end()) {
        originalLoadToOwner[originalLoad] = &buffer;
        continue;
      }

      LocalBuffer *owner = it->second;
      if (!compatibleForReuse(*owner, buffer))
        continue;
      if (owner->liveness.intersects(buffer.liveness))
        continue;
      buffer.bufferId = owner->bufferId;
      buffer.numCopies = std::max(buffer.numCopies, owner->numCopies);
    }
  }

  void coalesceCircularReuseCandidates() {
    for (size_t i = 0; i < buffers.size(); ++i) {
      LocalBuffer &owner = *buffers[i];
      if (owner.pinned || owner.priority == SmemBufferPriority::Lowest)
        continue;
      for (size_t j = i + 1; j < buffers.size(); ++j) {
        LocalBuffer &candidate = *buffers[j];
        if (candidate.pinned || candidate.priority != owner.priority)
          continue;
        if (!compatibleForReuse(owner, candidate))
          continue;
        if (owner.liveness.intersects(candidate.liveness))
          continue;

        unsigned oldId = candidate.bufferId;
        candidate.bufferId = owner.bufferId;
        unsigned oldCopies = candidate.numCopies;
        candidate.numCopies = std::max(candidate.numCopies, owner.numCopies);
        if (computeTotalSmem() <= smemBudget) {
          owner.isCircular = true;
          candidate.isCircular = true;
          continue;
        }
        candidate.bufferId = oldId;
        candidate.numCopies = oldCopies;
      }
    }
  }

  void assignCircularStarts() {
    llvm::MapVector<unsigned, SmallVector<LocalBuffer *>> groups;
    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      if (!buffer.isCircular)
        continue;
      groups[buffer.bufferId].push_back(&buffer);
    }

    for (auto &entry : groups) {
      SmallVector<LocalBuffer *> &group = entry.second;
      if (group.size() < 2) {
        group.front()->isCircular = false;
        group.front()->circularStart = 0;
        continue;
      }

      unsigned requiredCopies = static_cast<unsigned>(group.size());
      for (LocalBuffer *buffer : group)
        buffer->numCopies = std::max(buffer->numCopies, requiredCopies);

      for (auto [idx, buffer] : llvm::enumerate(group))
        buffer->circularStart = static_cast<unsigned>(idx);
    }
  }

  void emitAttrs() {
    assignCircularStarts();
    for (auto &bufferPtr : buffers) {
      LocalBuffer &buffer = *bufferPtr;
      eraseAttr(buffer.alloc, "buffer.id");
      eraseAttr(buffer.alloc, "buffer.copy");
      eraseAttr(buffer.alloc, "buffer.offset");
      eraseAttr(buffer.alloc, "buffer.circular");
      eraseAttr(buffer.alloc, "buffer.start");
      setI32Attr(buffer.alloc, "buffer.id", buffer.bufferId);
      setI32Attr(buffer.alloc, "buffer.copy", buffer.numCopies);
      if (buffer.offset != 0)
        setI32Attr(buffer.alloc, "buffer.offset", buffer.offset);
      if (buffer.isCircular) {
        setUnitAttr(buffer.alloc, "buffer.circular");
        setI32Attr(buffer.alloc, "buffer.start", buffer.circularStart);
      }
    }
  }

  FuncOp funcOp;
  SmallVector<std::unique_ptr<LocalDataChannelPost>> &channels;
  unsigned numBuffers;
  int smemAllocAlgo;
  unsigned smemBudget;
  bool smemCircularReuse;
  DenseMap<Operation *, size_t> operationId;
  DenseMap<Operation *, ChannelAnnotation> smemAnnotations;
  SmallVector<std::unique_ptr<LocalBuffer>> buffers;
};

class TmemAllocator {
public:
  TmemAllocator(FuncOp funcOp,
                SmallVector<std::unique_ptr<TmemDataChannelPost>> &channels,
                unsigned firstBufferId)
      : funcOp(funcOp), channels(channels), nextBufferId(firstBufferId) {}

  LogicalResult run() {
    buildOperationIdMap(funcOp, operationId);

    funcOp.walk<WalkOrder::PreOrder>([&](ttng::TMEMAllocOp alloc) {
      allocs.push_back(alloc);
    });
    if (allocs.empty())
      return success();

    for (ttng::TMEMAllocOp alloc : allocs) {
      auto allocSize = ttng::getTmemAllocSizes(alloc.getType());
      auto buffer = std::make_unique<TmemBuffer>();
      buffer->owner = alloc.getOperation();
      buffer->rowSize = allocSize.numRows;
      buffer->colSize = allocSize.numCols;
      TmemBuffer *bufferPtr = buffer.get();
      buffers.push_back(std::move(buffer));
      allocToBuffer[alloc.getOperation()] = bufferPtr;
      auto liveOps = livenessForTmemAlloc(alloc, channels);
      allocToIntervals[alloc.getOperation()] =
          intervalFromOps(liveOps, operationId);
      allocToChannel[alloc.getOperation()] =
          findChannelForAlloc(alloc.getOperation(), channels);
      eraseAttr(alloc, "buffer.id");
      eraseAttr(alloc, "buffer.copy");
      eraseAttr(alloc, "buffer.offset");
      LLVM_DEBUG({
        auto interval = allocToIntervals[alloc.getOperation()];
        LDBG("alloc size rows=" << bufferPtr->rowSize
                                << " cols=" << bufferPtr->colSize
                                << " live=[" << interval.start() << ","
                                << interval.end() << ")");
        alloc->dump();
      });
    }

    llvm::sort(allocs, [&](ttng::TMEMAllocOp a, ttng::TMEMAllocOp b) {
      TmemDataChannelPost *aCh = allocToChannel.lookup(a.getOperation());
      TmemDataChannelPost *bCh = allocToChannel.lookup(b.getOperation());
      if (aCh && bCh && aCh->isOperandD != bCh->isOperandD)
        return aCh->isOperandD;
      if (aCh && !bCh)
        return true;
      if (!aCh && bCh)
        return false;
      TmemBuffer *aBuf = getBuffer(a.getOperation());
      TmemBuffer *bBuf = getBuffer(b.getOperation());
      if (aBuf->rowSize * aBuf->colSize != bBuf->rowSize * bBuf->colSize)
        return aBuf->rowSize * aBuf->colSize > bBuf->rowSize * bBuf->colSize;
      auto aInt = allocToIntervals[a.getOperation()];
      auto bInt = allocToIntervals[b.getOperation()];
      if (aInt.start() != bInt.start())
        return aInt.start() < bInt.start();
      return getLoopDepth(a) > getLoopDepth(b);
    });

    DenseSet<Operation *> handledAllocs;
    auto annotations = parseChannelAnnotations(funcOp);
    if (!annotations.empty()) {
      auto tmemAnnotations = buildAllocToAnnotationMap(annotations);
      preAssignAnnotatedAllocs(tmemAnnotations, handledAllocs);
    }

    SmallVector<Operation *> innermostLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (isInnermostLoop(forOp))
        innermostLoops.push_back(forOp.getOperation());
    });

    unsigned ctrlIdx = 0;
    for (Operation *ctrlOp : innermostLoops) {
      SmallVector<ttng::TMEMAllocOp> loopAllocs;
      auto ctrlInterval = getIntervalForCtrlOp(ctrlOp, operationId);
      for (ttng::TMEMAllocOp alloc : allocs) {
        if (handledAllocs.contains(alloc.getOperation()))
          continue;
        auto allocInterval = allocToIntervals[alloc.getOperation()];
        if (ctrlInterval.intersects(allocInterval) ||
            ctrlIdx == innermostLoops.size() - 1) {
          loopAllocs.push_back(alloc);
          handledAllocs.insert(alloc.getOperation());
        }
      }
      if (!loopAllocs.empty() && failed(allocateTmemAllocs(loopAllocs, ctrlOp)))
        return failure();
      ++ctrlIdx;
    }

    SmallVector<ttng::TMEMAllocOp> remainingAllocs;
    for (ttng::TMEMAllocOp alloc : allocs)
      if (!handledAllocs.contains(alloc.getOperation()))
        remainingAllocs.push_back(alloc);
    if (!remainingAllocs.empty() &&
        failed(allocateTmemAllocs(remainingAllocs, nullptr)))
      return failure();

    maximizeLoopCarriedCopies();
    return success();
  }

private:
  struct AllocationState {
    DenseMap<TmemBuffer *, std::pair<TmemBuffer *, size_t>> assignment;
    DenseSet<TmemBuffer *> owners;
    size_t usedRows = 0;
  };

  TmemBuffer *getBuffer(Operation *op) {
    auto it = allocToBuffer.find(op);
    return it == allocToBuffer.end() ? nullptr : it->second;
  }

  bool sameLoop(TmemBuffer *buffer, Operation *ctrlOp) {
    if (!ctrlOp)
      return false;
    return allocToIntervals[buffer->owner].intersects(
        getIntervalForCtrlOp(ctrlOp, operationId));
  }

  SmallVector<int> getCombinedTasks(TmemBuffer *buffer) {
    SmallVector<int> combinedTasks;
    TmemDataChannelPost *channel = allocToChannel.lookup(buffer->owner);
    if (!channel)
      return combinedTasks;
    DenseSet<Operation *> users;
    if (failed(getAllTmemUsers(channel, users, buffer->owner)))
      return combinedTasks;
    DenseSet<int> combinedSet;
    for (Operation *user : users) {
      for (int task : getTaskIds(user)) {
        if (combinedSet.insert(task).second)
          combinedTasks.push_back(task);
      }
    }
    llvm::sort(combinedTasks);
    return combinedTasks;
  }

  bool isSourcefulOperandD(TmemBuffer *buffer) {
    auto alloc = dyn_cast<ttng::TMEMAllocOp>(buffer->owner);
    TmemDataChannelPost *channel = allocToChannel.lookup(buffer->owner);
    return alloc && alloc.getSrc() && channel && channel->isOperandD;
  }

  bool samePartition(TmemBuffer *a, TmemBuffer *b,
                     unsigned partitionCondition) {
    if (partitionCondition == 0)
      return true;
    TmemDataChannelPost *aCh = allocToChannel.lookup(a->owner);
    TmemDataChannelPost *bCh = allocToChannel.lookup(b->owner);
    if (!aCh || !bCh)
      return false;
    if (partitionCondition == 1) {
      Operation *aDst = aCh->getDstOp();
      Operation *bSrc = bCh->getSrcOp();
      if (!aDst || !bSrc)
        return false;
      return getTaskIds(bSrc) == getTaskIds(aDst);
    }
    return getCombinedTasks(a) == getCombinedTasks(b);
  }

  bool alongDependencyChain(Operation *src, Operation *dst) {
    TmemDataChannelPost *srcCh = allocToChannel.lookup(src);
    TmemDataChannelPost *dstCh = allocToChannel.lookup(dst);
    if (!srcCh || !dstCh)
      return false;
    Operation *srcDst = srcCh->getDstOp();
    Operation *dstSrc = dstCh->getSrcOp();
    if (!srcDst || !dstSrc)
      return false;
    if (getTaskIds(dstSrc) == getTaskIds(srcDst))
      return true;
    return isDataDependent(srcDst, dstSrc) || isDataDependent(dstSrc, srcDst);
  }

  int hasPotentialReuse(TmemBuffer *owner, TmemBuffer *candidate,
                        Operation *ctrlOp) {
    if (isSourcefulOperandD(owner) || isSourcefulOperandD(candidate))
      return 0;

    if (candidate->colSize > owner->colSize)
      return 0;
    if (allocToIntervals[owner->owner].intersects(
            allocToIntervals[candidate->owner]))
      return 0;

    TmemDataChannelPost *ownerCh = allocToChannel.lookup(owner->owner);
    TmemDataChannelPost *candidateCh =
        allocToChannel.lookup(candidate->owner);
    if (!ownerCh || !candidateCh)
      return 0;

    Operation *ownerDst = ownerCh->getDstOp();
    Operation *candidateSrc = candidateCh->getSrcOp();
    Operation *candidateDst = candidateCh->getDstOp();
    Operation *ownerSrc = ownerCh->getSrcOp();
    bool hasDependency =
        isDataDependent(ownerDst, candidateSrc) ||
        isDataDependent(candidateDst, ownerSrc) ||
        (sameLoop(owner, ctrlOp) &&
         alongDependencyChain(owner->owner, candidate->owner));
    if (!hasDependency)
      return 0;

    return candidate->colSize == owner->colSize ? 2 : 1;
  }

  size_t computeColOffset(TmemBuffer *candidate, TmemBuffer *owner,
                          const AllocationState &state, Operation *ctrlOp) {
    size_t maxColOffset = 0;
    for (const auto &[reuser, assignment] : state.assignment) {
      auto [reuseOwner, reuserColOffset] = assignment;
      if (reuseOwner != owner)
        continue;

      bool canShareColumns =
          hasPotentialReuse(reuser, candidate, ctrlOp) > 0 ||
          hasPotentialReuse(candidate, reuser, ctrlOp) > 0;
      if (!canShareColumns)
        maxColOffset =
            std::max(maxColOffset, reuserColOffset + reuser->colSize);
    }

    if (maxColOffset + candidate->colSize > owner->colSize)
      return std::numeric_limits<size_t>::max();
    return maxColOffset;
  }

  bool tryAllocateBacktracking(ArrayRef<ttng::TMEMAllocOp> toAllocate,
                               size_t idx, AllocationState &state,
                               Operation *ctrlOp) {
    if (idx == toAllocate.size())
      return true;

    ttng::TMEMAllocOp candidateAlloc = toAllocate[idx];
    TmemBuffer *candidate = getBuffer(candidateAlloc.getOperation());
    SmallVector<std::pair<TmemBuffer *, int>> reuseCandidates;
    for (TmemBuffer *owner : state.owners) {
      int priority = hasPotentialReuse(owner, candidate, ctrlOp);
      if (priority > 0)
        reuseCandidates.push_back({owner, priority});
    }
    llvm::sort(reuseCandidates, [](const auto &a, const auto &b) {
      return a.second > b.second;
    });

    for (auto &[owner, priority] : reuseCandidates) {
      size_t colOffset = computeColOffset(candidate, owner, state, ctrlOp);
      if (colOffset == std::numeric_limits<size_t>::max())
        continue;

      AllocationState nextState = state;
      nextState.assignment[candidate] = {owner, colOffset};
      if (tryAllocateBacktracking(toAllocate, idx + 1, nextState, ctrlOp)) {
        state = std::move(nextState);
        return true;
      }
    }

    constexpr size_t maxTmemRows = 512;
    if (state.usedRows + candidate->rowSize <= maxTmemRows) {
      AllocationState nextState = state;
      nextState.owners.insert(candidate);
      nextState.usedRows += candidate->rowSize;
      if (tryAllocateBacktracking(toAllocate, idx + 1, nextState, ctrlOp)) {
        state = std::move(nextState);
        return true;
      }
    }

    return false;
  }

  LogicalResult allocateTmemAllocsBacktracking(
      ArrayRef<ttng::TMEMAllocOp> toAllocate, Operation *ctrlOp) {
    AllocationState state;
    if (!tryAllocateBacktracking(toAllocate, 0, state, ctrlOp)) {
      ttng::TMEMAllocOp firstAlloc = toAllocate.front();
      return firstAlloc.emitError(
          "can't find tmem space: failed backtracking TMEM allocation");
    }

    size_t rowOffset = 0;
    DenseMap<TmemBuffer *, unsigned> ownerToBufferId;
    for (ttng::TMEMAllocOp alloc : toAllocate) {
      TmemBuffer *buffer = getBuffer(alloc.getOperation());
      if (!state.owners.contains(buffer))
        continue;

      buffer->rowOffset = rowOffset;
      buffer->colOffset = 0;
      buffer->isOwnerOfSpace = true;
      buffer->reuseOwner = buffer;
      ownerToBufferId[buffer] = nextBufferId;
      setI32Attr(alloc, "buffer.id", nextBufferId++);
      setI32Attr(alloc, "buffer.copy", 1);
      setI32Attr(alloc, "buffer.offset", 0);
      rowOffset += buffer->rowSize;
    }

    for (ttng::TMEMAllocOp alloc : toAllocate) {
      TmemBuffer *buffer = getBuffer(alloc.getOperation());
      if (state.owners.contains(buffer))
        continue;

      auto it = state.assignment.find(buffer);
      assert(it != state.assignment.end());
      auto [owner, colOffset] = it->second;
      buffer->rowOffset = owner->rowOffset;
      buffer->colOffset = colOffset;
      buffer->isOwnerOfSpace = false;
      buffer->reuseOwner = owner;
      auto ownerId = ownerToBufferId.lookup(owner);
      setI32Attr(alloc, "buffer.id", ownerId);
      setI32Attr(alloc, "buffer.copy", 1);
      setI32Attr(alloc, "buffer.offset", colOffset);
    }

    return success();
  }

  bool checkOtherReuses(TmemBuffer *candidate, TmemBuffer *reuseOwner,
                        size_t colOffset) {
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (!buffer.isOwnerOfSpace && buffer.reuseOwner == reuseOwner) {
        Interval<size_t> candRange(colOffset,
                                   colOffset + candidate->colSize);
        Interval<size_t> bufferRange(buffer.colOffset,
                                     buffer.colOffset + buffer.colSize);
        if (allocToIntervals[buffer.owner].intersects(
                allocToIntervals[candidate->owner]) &&
            bufferRange.intersects(candRange))
          return false;
      }
    }
    return true;
  }

  void preAssignAnnotatedAllocs(
      const DenseMap<Operation *, ChannelAnnotation> &annotations,
      DenseSet<Operation *> &handledAllocs) {
    if (annotations.empty())
      return;

    std::map<unsigned, SmallVector<ttng::TMEMAllocOp>> groups;
    for (ttng::TMEMAllocOp alloc : allocs) {
      auto it = annotations.find(alloc.getOperation());
      if (it != annotations.end())
        groups[it->second.bufferId].push_back(alloc);
    }

    size_t rowOffset = 0;
    for (auto &[bufferId, group] : groups) {
      if (group.empty())
        continue;

      nextBufferId = std::max(nextBufferId, bufferId + 1);

      ttng::TMEMAllocOp ownerAlloc = group.front();
      TmemBuffer *owner = getBuffer(ownerAlloc.getOperation());
      owner->rowOffset = rowOffset;
      owner->colOffset = 0;
      owner->isOwnerOfSpace = true;
      owner->reuseOwner = owner;
      setI32Attr(ownerAlloc, "buffer.id", bufferId);
      setI32Attr(ownerAlloc, "buffer.copy",
                 annotations.lookup(ownerAlloc.getOperation()).numCopies);
      setI32Attr(ownerAlloc, "buffer.offset", 0);
      handledAllocs.insert(ownerAlloc.getOperation());
      rowOffset += owner->rowSize;

      for (ttng::TMEMAllocOp reuserAlloc : ArrayRef(group).drop_front()) {
        TmemBuffer *reuser = getBuffer(reuserAlloc.getOperation());
        bool canReuseAnnotatedOwner =
            hasPotentialReuse(owner, reuser, nullptr) > 0 ||
            hasPotentialReuse(reuser, owner, nullptr) > 0;
        size_t colOffset =
            canReuseAnnotatedOwner ? findReuseSpace(reuser, owner, nullptr)
                                   : std::numeric_limits<size_t>::max();

        if (colOffset != std::numeric_limits<size_t>::max() &&
            checkOtherReuses(reuser, owner, colOffset)) {
          reuser->rowOffset = owner->rowOffset;
          reuser->colOffset = colOffset;
          reuser->isOwnerOfSpace = false;
          reuser->reuseOwner = owner;
          setI32Attr(reuserAlloc, "buffer.id", bufferId);
          setI32Attr(reuserAlloc, "buffer.copy",
                     annotations.lookup(reuserAlloc.getOperation()).numCopies);
          setI32Attr(reuserAlloc, "buffer.offset", colOffset);
        } else {
          // Autotuning annotations may encode an intended physical packing
          // group. Preserve the pinned id only when the planner can prove the
          // same semantic reuse relation; otherwise split the alloc into its
          // own semantic TMEM group.
          reuser->rowOffset = rowOffset;
          reuser->colOffset = 0;
          reuser->isOwnerOfSpace = true;
          reuser->reuseOwner = reuser;
          setI32Attr(reuserAlloc, "buffer.id", nextBufferId++);
          setI32Attr(reuserAlloc, "buffer.copy",
                     annotations.lookup(reuserAlloc.getOperation()).numCopies);
          setI32Attr(reuserAlloc, "buffer.offset", 0);
          rowOffset += reuser->rowSize;
        }
        handledAllocs.insert(reuserAlloc.getOperation());
      }
    }
  }

  size_t findUsesInCtrlOp(TmemBuffer *owner, TmemBuffer *candidate,
                          Operation *ctrlOp) {
    size_t maxColOffset = 0;
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (!buffer.isOwnerOfSpace && buffer.reuseOwner == owner->reuseOwner &&
          &buffer != owner &&
          (sameLoop(&buffer, ctrlOp) ||
           allocToIntervals[buffer.owner].intersects(
               allocToIntervals[candidate->owner]))) {
        maxColOffset = std::max(maxColOffset, buffer.colOffset + buffer.colSize);
      }
    }
    return maxColOffset;
  }

  size_t findReuseSpace(TmemBuffer *candidate, TmemBuffer *reuseOwner,
                        Operation *ctrlOp) {
    size_t maxColOffset = 0;
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (!buffer.isOwnerOfSpace && buffer.reuseOwner == reuseOwner) {
        if (sameLoop(&buffer, ctrlOp) ||
            allocToIntervals[buffer.owner].intersects(
                allocToIntervals[candidate->owner]))
          maxColOffset =
              std::max(buffer.colOffset + buffer.colSize, maxColOffset);
      }
    }
    if (maxColOffset + candidate->colSize <= reuseOwner->colSize)
      return maxColOffset;

    if (!sameLoop(reuseOwner, ctrlOp)) {
      for (auto &bufferPtr : buffers) {
        TmemBuffer &buffer = *bufferPtr;
        if (!buffer.isOwnerOfSpace && buffer.reuseOwner == reuseOwner &&
            buffer.colOffset == 0 && sameLoop(&buffer, ctrlOp) &&
            alongDependencyChain(buffer.owner, candidate->owner)) {
          size_t offset = findUsesInCtrlOp(&buffer, candidate, ctrlOp);
          if (offset + candidate->colSize <= buffer.colSize)
            return offset;
        }
      }
    }
    return std::numeric_limits<size_t>::max();
  }

  TmemBuffer *findReuseChannel(TmemBuffer *candidate, Operation *ctrlOp,
                               unsigned partitionCondition) {
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (!buffer.isOwnerOfSpace)
        continue;
      if (isSourcefulOperandD(&buffer) || isSourcefulOperandD(candidate))
        continue;
      if (allocToIntervals[buffer.owner].intersects(
              allocToIntervals[candidate->owner]) ||
          buffer.colSize < candidate->colSize)
        continue;

      if (!allocToChannel.lookup(buffer.owner) ||
          !allocToChannel.lookup(candidate->owner))
        continue;

      bool compatible =
          (!sameLoop(&buffer, ctrlOp) &&
           samePartition(&buffer, candidate, partitionCondition)) ||
          (sameLoop(&buffer, ctrlOp) &&
           alongDependencyChain(buffer.owner, candidate->owner));
      if (!compatible)
        continue;

      size_t colOffset = findReuseSpace(candidate, &buffer, ctrlOp);
      if (colOffset == std::numeric_limits<size_t>::max())
        continue;
      if (!checkOtherReuses(candidate, &buffer, colOffset))
        continue;

      candidate->isOwnerOfSpace = false;
      candidate->rowOffset = buffer.rowOffset;
      candidate->colOffset = colOffset;
      candidate->reuseOwner = &buffer;
      return &buffer;
    }
    return nullptr;
  }

  bool allInterfere(TmemBuffer *candidate) {
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (buffer.rowOffset != std::numeric_limits<size_t>::max() &&
          !allocToIntervals[buffer.owner].intersects(
              allocToIntervals[candidate->owner]))
        return false;
    }
    return true;
  }

  bool allocateNewSpace(TmemBuffer *candidate, bool apply) {
    size_t maxRowOffset = 0;
    for (auto &bufferPtr : buffers) {
      TmemBuffer &buffer = *bufferPtr;
      if (buffer.rowOffset != std::numeric_limits<size_t>::max())
        maxRowOffset = std::max(maxRowOffset, buffer.rowOffset + buffer.rowSize);
    }
    if (maxRowOffset + candidate->rowSize > 512)
      return false;
      if (apply) {
      candidate->rowOffset = maxRowOffset;
      candidate->colOffset = 0;
      candidate->isOwnerOfSpace = true;
      candidate->reuseOwner = candidate;
      setI32Attr(candidate->owner, "buffer.id", nextBufferId++);
      setI32Attr(candidate->owner, "buffer.copy", 1);
      setI32Attr(candidate->owner, "buffer.offset", 0);
      LLVM_DEBUG({
        LDBG("allocate new buffer.id=" << nextBufferId - 1
                                       << " rowOffset=" << candidate->rowOffset
                                       << " rows=" << candidate->rowSize
                                       << " cols=" << candidate->colSize);
        candidate->owner->dump();
      });
    }
    return true;
  }

  LogicalResult allocateTmemAllocs(ArrayRef<ttng::TMEMAllocOp> toAllocate,
                                   Operation *ctrlOp) {
    if (getTmemAllocAlgo(ctrlOp) == 2)
      return allocateTmemAllocsBacktracking(toAllocate, ctrlOp);

    for (ttng::TMEMAllocOp alloc : toAllocate) {
      TmemBuffer *candidate = getBuffer(alloc.getOperation());
      if (!candidate)
        return alloc.emitError("NVWS memory planner lost TMEM buffer state");

      if (allInterfere(candidate)) {
        if (!allocateNewSpace(candidate, true))
          return alloc.emitError(
              "can't find tmem space: no new space for TMEM alloc");
      } else {
        TmemBuffer *reuse = findReuseChannel(candidate, ctrlOp, 2);
        if (!reuse)
          reuse = findReuseChannel(candidate, ctrlOp, 1);
        if (reuse) {
          auto idAttr = reuse->owner->getAttrOfType<IntegerAttr>("buffer.id");
          assert(idAttr && "reuse owner must have buffer.id");
          setI32Attr(alloc, "buffer.id", idAttr.getInt());
          setI32Attr(alloc, "buffer.copy", 1);
          setI32Attr(alloc, "buffer.offset", candidate->colOffset);
          LLVM_DEBUG({
            LDBG("reuse buffer.id=" << idAttr.getInt()
                                    << " colOffset=" << candidate->colOffset);
            alloc->dump();
          });
        } else if (allocateNewSpace(candidate, false)) {
          allocateNewSpace(candidate, true);
        } else {
          return alloc.emitError(
              "can't find tmem space: failed to allocate new TMEM space");
        }
      }
    }
    return success();
  }

  int getTmemAllocAlgo(Operation *ctrlOp) {
    int algo = 1;
    if (!ctrlOp)
      return algo;
    if (auto attr = ctrlOp->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo"))
      algo = attr.getInt();
    for (auto parent = ctrlOp->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>()) {
      if (auto attr =
              parent->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo")) {
        if (!ctrlOp->getAttrOfType<IntegerAttr>("tt.tmem_alloc_algo"))
          algo = attr.getInt();
      }
    }
    return algo;
  }

  bool hasLoopCarriedAccToken(ttng::TMEMAllocOp alloc, scf::ForOp forOp) {
    for (Operation *user : alloc.getResult().getUsers()) {
      auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(skipIdxOp(user));
      if (!mmaOp || !forOp->isProperAncestor(mmaOp))
        continue;
      Value accDep = mmaOp.getAccDep();
      auto blockArg = dyn_cast_or_null<BlockArgument>(accDep);
      if (!blockArg || blockArg.getOwner() != forOp.getBody())
        continue;
      unsigned argIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      Value token = mmaOp.getToken();
      if (token && yieldOp.getOperand(argIdx) == token)
        return true;
    }
    return false;
  }

  void maximizeLoopCarriedCopies() {
    constexpr unsigned tmemColLimit = 512;
    struct AllocInfo {
      ttng::TMEMAllocOp alloc;
      unsigned baseCols;
      unsigned copy;
    };
    SmallVector<AllocInfo> allocInfos;
    unsigned totalCols = 0;
    for (ttng::TMEMAllocOp alloc : allocs) {
      auto allocSize = ttng::getTmemAllocSizes(alloc.getType());
      unsigned baseCols = allocSize.numCols;
      unsigned copy = 1;
      if (auto copyAttr = alloc->getAttrOfType<IntegerAttr>("buffer.copy"))
        copy = copyAttr.getInt();
      totalCols += baseCols * copy;

      bool hasLoopCarriedMMA = false;
      for (Operation *user : alloc.getResult().getUsers()) {
        Operation *actual = skipIdxOp(user);
        if (!actual)
          continue;
        if (auto forOp = actual->getParentOfType<scf::ForOp>()) {
          if (hasLoopCarriedAccToken(alloc, forOp)) {
            hasLoopCarriedMMA = true;
            break;
          }
        }
      }
      if (hasLoopCarriedMMA)
        allocInfos.push_back({alloc, baseCols, copy});
    }

    while (totalCols < tmemColLimit && !allocInfos.empty()) {
      bool added = false;
      for (AllocInfo &info : allocInfos) {
        if (totalCols + info.baseCols <= tmemColLimit) {
          ++info.copy;
          totalCols += info.baseCols;
          added = true;
        }
      }
      if (!added)
        break;
    }

    for (AllocInfo &info : allocInfos)
      setI32Attr(info.alloc, "buffer.copy", info.copy);
  }

  FuncOp funcOp;
  SmallVector<std::unique_ptr<TmemDataChannelPost>> &channels;
  unsigned nextBufferId;
  DenseMap<Operation *, size_t> operationId;
  SmallVector<ttng::TMEMAllocOp> allocs;
  SmallVector<std::unique_ptr<TmemBuffer>> buffers;
  DenseMap<Operation *, TmemBuffer *> allocToBuffer;
  DenseMap<Operation *, Interval<size_t>> allocToIntervals;
  DenseMap<Operation *, TmemDataChannelPost *> allocToChannel;
};

struct EffectiveSmemOptions {
  int allocAlgo = 0;
  unsigned budget = 0;
  bool circularReuse = false;
};

static EffectiveSmemOptions getEffectiveSmemOptions(FuncOp funcOp,
                                                    int passAllocAlgo,
                                                    unsigned passBudget,
                                                    bool passCircularReuse) {
  EffectiveSmemOptions options{passAllocAlgo, passBudget, passCircularReuse};
  funcOp.walk([&](scf::ForOp forOp) {
    if (!forOp->hasAttr("tt.warp_specialize"))
      return;

    SmallVector<scf::ForOp> loopChain;
    loopChain.push_back(forOp);
    for (auto parent = forOp->getParentOfType<scf::ForOp>(); parent;
         parent = parent->getParentOfType<scf::ForOp>())
      loopChain.push_back(parent);

    for (auto it = loopChain.rbegin(); it != loopChain.rend(); ++it) {
      scf::ForOp loop = *it;
      if (auto attr = loop->getAttrOfType<IntegerAttr>("tt.smem_alloc_algo"))
        options.allocAlgo = attr.getInt();
      if (auto attr = loop->getAttrOfType<IntegerAttr>("tt.smem_budget"))
        options.budget = static_cast<unsigned>(attr.getInt());
      if (auto attr = loop->getAttrOfType<BoolAttr>("tt.smem_circular_reuse"))
        options.circularReuse = attr.getValue();
    }
  });
  return options;
}

static LogicalResult doMemoryPlanning(FuncOp funcOp, unsigned numBuffers,
                                      int smemAllocAlgo, unsigned smemBudget,
                                      bool smemCircularReuse) {
  SmallVector<std::unique_ptr<LocalDataChannelPost>> localChannels;
  if (failed(collectLocalPostChannels(localChannels, funcOp)))
    return failure();

  EffectiveSmemOptions effectiveSmemOptions = getEffectiveSmemOptions(
      funcOp, smemAllocAlgo, smemBudget, smemCircularReuse);

  unsigned firstTmemBufferId = 0;
  LocalSmemAllocator localAllocator(
      funcOp, localChannels, numBuffers, effectiveSmemOptions.allocAlgo,
      effectiveSmemOptions.budget, effectiveSmemOptions.circularReuse);
  if (failed(localAllocator.run(firstTmemBufferId)))
    return failure();

  SmallVector<std::unique_ptr<TmemDataChannelPost>> channels;
  if (failed(collectTmemPostChannels(channels, funcOp)))
    return failure();
  TmemAllocator allocator(funcOp, channels, firstTmemBufferId);
  return allocator.run();
}

class NVWSMemoryPlanner
    : public impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner> {
public:
  using impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner>::NVWSMemoryPlannerBase;

  void runOnOperation() override {
    getOperation()->walk([&](FuncOp funcOp) {
      unsigned effectiveNumBuffers =
          numBuffers <= 0 ? 1u : static_cast<unsigned>(numBuffers);
      unsigned effectiveSmemBudget =
          smemBudget <= 0 ? 0u : static_cast<unsigned>(smemBudget);
      if (failed(doMemoryPlanning(funcOp, effectiveNumBuffers, smemAllocAlgo,
                                  effectiveSmemBudget, smemCircularReuse)))
        signalPassFailure();
    });
  }
};

} // namespace

} // namespace mlir::triton
