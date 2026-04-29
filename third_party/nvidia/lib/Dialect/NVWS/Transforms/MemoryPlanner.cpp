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
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>
#include <memory>

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

enum class ChannelKind { TMEMPost };

struct TmemDataChannelPost {
  int producer;
  SmallVector<int> consumers;
  Operation *allocOp;
  bool isOperandD;
  bool isOperandDNoAcc;
  bool isSameIterGuard = false;
  unsigned uniqID;

  TmemDataChannelPost(int producer, ArrayRef<int> consumers,
                      Operation *allocOp, bool isOperandD,
                      bool isOperandDNoAcc, unsigned uniqID)
      : producer(producer), consumers(consumers.begin(), consumers.end()),
        allocOp(allocOp), isOperandD(isOperandD),
        isOperandDNoAcc(isOperandDNoAcc), uniqID(uniqID) {}

  Operation *getAllocOp() const { return allocOp; }
  Operation *getSrcOp() const;
  Operation *getDstOp() const;
  void getDstOps(SmallVectorImpl<Operation *> &dsts) const;
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

static void setI32Attr(Operation *op, StringRef name, int32_t value) {
  op->setAttr(name, IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     value));
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

static bool needsChannel(int producer, ArrayRef<int> consumers) {
  return !llvm::all_of(consumers, [producer](int consumerId) {
    return consumerId == producer;
  });
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
        if (currentProds.empty() && isConstFalse(mmaOp.useAccumulator())) {
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
    // ttng.tmem_alloc %src is initialized storage. It must not become an
    // extra producer channel; consumers still participate in liveness below.
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

  SmallVector<int> consumerTaskIds;
  DenseSet<int> seenTaskIds;
  for (Operation *consumer : consumers) {
    for (int id : getTaskIds(consumer))
      if (seenTaskIds.insert(id).second)
        consumerTaskIds.push_back(id);
  }
  consumerTaskIds.erase(
      std::remove(consumerTaskIds.begin(), consumerTaskIds.end(), producerId),
      consumerTaskIds.end());

  if (needsChannel(producerId, consumerTaskIds)) {
    channels.push_back(std::make_unique<TmemDataChannelPost>(
        producerId, consumerTaskIds, alloc.getOperation(),
        false /*isOperandD*/, isOperandDNoAcc, channels.size()));
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
      if (actual == user && isa<ttg::MemDescIndexOp, ttg::MemDescReinterpretOp,
                               ttg::MemDescTransOp>(actual))
        for (Value result : actual->getResults())
          worklist.push_back(result);
    }
  }
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
    if (isa<ttng::TMEMStoreOp>(op))
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
      TmemBuffer buffer;
      buffer.owner = alloc.getOperation();
      buffer.rowSize = allocSize.numRows;
      buffer.colSize = allocSize.numCols;
      buffers.push_back(buffer);
      allocToBuffer[alloc.getOperation()] = &buffers.back();
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
        LDBG("alloc size rows=" << buffer.rowSize << " cols=" << buffer.colSize
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

    SmallVector<Operation *> innermostLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (isInnermostLoop(forOp))
        innermostLoops.push_back(forOp.getOperation());
    });

    DenseSet<Operation *> handledAllocs;
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

  bool checkOtherReuses(TmemBuffer *candidate, TmemBuffer *reuseOwner,
                        size_t colOffset) {
    for (TmemBuffer &buffer : buffers) {
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

  size_t findUsesInCtrlOp(TmemBuffer *owner, TmemBuffer *candidate,
                          Operation *ctrlOp) {
    size_t maxColOffset = 0;
    for (TmemBuffer &buffer : buffers) {
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
    for (TmemBuffer &buffer : buffers) {
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
      for (TmemBuffer &buffer : buffers) {
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
    for (TmemBuffer &buffer : buffers) {
      if (!buffer.isOwnerOfSpace)
        continue;
      if (allocToIntervals[buffer.owner].intersects(
              allocToIntervals[candidate->owner]) ||
          buffer.colSize < candidate->colSize)
        continue;

      bool hasBothChannels = allocToChannel.lookup(buffer.owner) &&
                             allocToChannel.lookup(candidate->owner);
      bool compatible =
          !hasBothChannels ||
          ((!sameLoop(&buffer, ctrlOp) &&
            samePartition(&buffer, candidate, partitionCondition)) ||
           (sameLoop(&buffer, ctrlOp) &&
            alongDependencyChain(buffer.owner, candidate->owner)));
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
    for (TmemBuffer &buffer : buffers) {
      if (buffer.rowOffset != std::numeric_limits<size_t>::max() &&
          !allocToIntervals[buffer.owner].intersects(
              allocToIntervals[candidate->owner]))
        return false;
    }
    return true;
  }

  bool allocateNewSpace(TmemBuffer *candidate, bool apply) {
    size_t maxRowOffset = 0;
    for (TmemBuffer &buffer : buffers) {
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
  SmallVector<TmemBuffer> buffers;
  DenseMap<Operation *, TmemBuffer *> allocToBuffer;
  DenseMap<Operation *, Interval<size_t>> allocToIntervals;
  DenseMap<Operation *, TmemDataChannelPost *> allocToChannel;
};

static LogicalResult doTmemMemoryPlanning(FuncOp funcOp,
                                          unsigned firstBufferId) {
  SmallVector<std::unique_ptr<TmemDataChannelPost>> channels;
  if (failed(collectTmemPostChannels(channels, funcOp)))
    return failure();
  TmemAllocator allocator(funcOp, channels, firstBufferId);
  return allocator.run();
}

class NVWSMemoryPlanner
    : public impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner> {
public:
  using impl::NVWSMemoryPlannerBase<NVWSMemoryPlanner>::NVWSMemoryPlannerBase;

  void runOnOperation() override {
    getOperation()->walk([&](FuncOp funcOp) {
      if (failed(doTmemMemoryPlanning(funcOp, numBuffers)))
        signalPassFailure();
    });
  }
};

} // namespace

} // namespace mlir::triton
