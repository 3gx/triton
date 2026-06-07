#ifndef NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_MODEL_H_
#define NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_MODEL_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::triton::nvws::insert_semas {

using gpu::MemDescType;
using gpu::StageCluster;

inline constexpr StringLiteral kBufferIdAttrName = "buffer.id";
inline constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
inline constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";

enum class MemorySpaceKind { Tmem, Local };
enum class AccessEffect { Read, Write };

inline bool hasRead(AccessEffect e) { return e == AccessEffect::Read; }
inline bool hasWrite(AccessEffect e) { return e == AccessEffect::Write; }

// Effective owner. std::nullopt is root/external, not partition 0.
using PartitionId = std::pair<int /*partition*/, int /*wsTag*/>;

inline bool sameOwner(const std::optional<PartitionId> &a,
                      const std::optional<PartitionId> &b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return *a == *b;
}

struct AliasStep {
  Operation *op = nullptr;
  unsigned sourceOperand = 0;
};

struct AliasInfo {
  unsigned memberIdx = 0;
  SmallVector<AliasStep> steps;
};

struct BufferMember {
  Operation *allocOp = nullptr;
  Value value;
  MemDescType type;
  int64_t offset = 0;
  int64_t extent = 1;
  int64_t resourceKey = 0;
};

struct AccessTouch {
  unsigned memberIdx = 0;
  int64_t resourceKey = 0;
  AccessEffect effect = AccessEffect::Read;
  Value accessValue;
  AliasInfo alias;
};

struct AccessEvent {
  Operation *op = nullptr;
  std::optional<PartitionId> owner;
  Operation *tagSourceOp = nullptr;
  SmallVector<AccessTouch, 2> touches;
  bool sourcefulAllocStore = false;
};

struct BufferGroup {
  MemorySpaceKind memory = MemorySpaceKind::Tmem;
  int64_t logicalId = 0;
  SmallVector<BufferMember> members;
  DenseMap<Value, AliasInfo> aliases;
  SmallVector<Operation *> aliasOps;
  SmallVector<AccessEvent, 0> events;

  bool isTmem() const { return memory == MemorySpaceKind::Tmem; }
};

std::optional<int> tryGetWsTag(Operation *op);
Operation *getTagSourceOp(Operation *op);
std::optional<PartitionId> getPartitionId(Operation *op, int pos = 0);
std::optional<int64_t> getI64Attr(Operation *op, StringRef name);
std::optional<int64_t> getBufferId(Operation *op);
int64_t getBufferOffset(Operation *op);
bool isSemaphoreBackingAlloc(Operation *op);
bool isSupportedAliasOp(Operation *op);
AsyncOp getAsyncPayload(Operation *op);
RankedTensorType getTensorTypeFromScalar(OpBuilder &builder, Value scalar);
int getTxCount(Operation *descOp);
bool isEventInScopeForRegion(Operation *tagSourceOp, Operation *eventOp,
                             Region *region);
void buildProgramOrderRank(mlir::triton::FuncOp funcOp,
                           DenseMap<Operation *, unsigned> &rank);
unsigned maxRankInSubtree(Operation *op,
                          const DenseMap<Operation *, unsigned> &rank);

inline bool touchReads(const AccessTouch &t) { return hasRead(t.effect); }
inline bool touchWrites(const AccessTouch &t) { return hasWrite(t.effect); }

inline void
collectTouchesForResource(const AccessEvent &event, int64_t resourceKey,
                          SmallVectorImpl<const AccessTouch *> &touches) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey)
      touches.push_back(&t);
}

inline bool eventTouchesResource(const AccessEvent &event,
                                 int64_t resourceKey) {
  return llvm::any_of(event.touches, [&](const AccessTouch &t) {
    return t.resourceKey == resourceKey;
  });
}

inline bool eventConsumes(const AccessEvent &event, int64_t resourceKey) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey && touchReads(t))
      return true;
  return false;
}

inline bool eventProduces(const AccessEvent &event, int64_t resourceKey) {
  for (const AccessTouch &t : event.touches)
    if (t.resourceKey == resourceKey && touchWrites(t))
      return true;
  return false;
}

using ResourceId = std::pair<int64_t /*logicalGroupId*/,
                             int64_t /*resourceKey*/>;

struct RegionOwnership {
  std::optional<PartitionId> entry;
  std::optional<PartitionId> exit;
  bool carried = false;
  bool hasEventsInSubtree = false;
};

struct ResourcePlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  DenseMap<Region *, RegionOwnership> regionOwners;
  DenseMap<Operation *, std::optional<PartitionId>> useOwner;
  DenseMap<Operation *, Operation *> useTagSource;
  DenseMap<Operation *, unsigned> useTouchIdx;
  DenseMap<Operation *, SmallVector<unsigned, 2>> useTouchIdxs;
  DenseMap<Operation *, std::optional<PartitionId>> yieldOwner;
};

inline std::optional<PartitionId>
regionOpPartition(Operation *op, const ResourcePlan &rp) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto it = rp.regionOwners.find(&forOp.getRegion());
    if (it == rp.regionOwners.end())
      return std::nullopt;
    if (!it->second.hasEventsInSubtree)
      return std::nullopt;
    return it->second.entry;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    auto it = rp.regionOwners.find(&ifOp.getThenRegion());
    if (it != rp.regionOwners.end() && it->second.hasEventsInSubtree)
      return it->second.entry;
    auto it2 = rp.regionOwners.find(&ifOp.getElseRegion());
    if (it2 != rp.regionOwners.end() && it2->second.hasEventsInSubtree)
      return it2->second.entry;
    return std::nullopt;
  }
  return std::nullopt;
}

enum class SyncEdgeKind { Ready, Done, Handoff };

struct ProducedVersionKey {
  int64_t resourceKey = 0;
  unsigned epoch = 0;
  bool operator==(const ProducedVersionKey &other) const {
    return resourceKey == other.resourceKey && epoch == other.epoch;
  }
};

enum class CarrierTokenChoice { None, Source, Destination };

struct SyncEdge {
  std::string name;
  SyncEdgeKind kind = SyncEdgeKind::Handoff;
  ProducedVersionKey producedVersion;
  Operation *srcOp = nullptr;
  Operation *dstOp = nullptr;
  Region *srcYieldRegion = nullptr;
  Region *dstYieldRegion = nullptr;
  std::optional<PartitionId> srcOwner;
  std::optional<PartitionId> dstOwner;
  unsigned srcEpoch = 0;
  RegionOwnership srcRegionOwner;
  RegionOwnership dstRegionOwner;
  std::optional<PartitionId> preOwner;
  std::optional<PartitionId> postOwner;
  bool releaseOutsideControlFlow = false;
  bool acquireOutsideControlFlow = false;
  bool forceFullSemaphore = false;
  CarrierTokenChoice carrierChoice = CarrierTokenChoice::None;
  StageCluster releaseStageCluster;
  StageCluster acquireStageCluster;
  AsyncOp asyncPayload = AsyncOp::NONE;
  SmallVector<AccessTouch, 2> touches;
};

struct SyncPlan {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncEdge, 0> edges;
  DenseMap<Operation *, SmallVector<unsigned, 2>> beforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> beforeYield;
  DenseSet<Operation *> accessOps;
  std::string initialPermitName;
  Operation *initialPermitBeforeOp = nullptr;
  Operation *initialPermitReleaseAfterOp = nullptr;
  int initialPermitEdgeIdx = -1;
  SmallVector<unsigned, 8> semaRep;
  unsigned semaFind(unsigned x) const {
    while (semaRep[x] != x)
      x = semaRep[x];
    return x;
  }
};

enum class SyncGroupKind {
  InitialEmpty,
  Singleton,
  ReadyFanout,
  DoneFanin,
  LinearChain
};

struct SyncGroup {
  std::string name;
  SyncGroupKind kind = SyncGroupKind::Singleton;
  SmallVector<unsigned, 4> edgeIdxs;
  Operation *initialOp = nullptr;
  std::optional<PartitionId> initialOwner;
};

struct ReadyFanoutKey {
  std::optional<PartitionId> owner;
  unsigned epoch = 0;
  Operation *srcOp = nullptr;
  Region *srcYield = nullptr;
  bool operator==(const ReadyFanoutKey &o) const {
    return owner == o.owner && epoch == o.epoch && srcOp == o.srcOp &&
           srcYield == o.srcYield;
  }
};

struct DoneFaninKey {
  std::optional<PartitionId> owner;
  Operation *dstOp = nullptr;
  Region *dstYield = nullptr;
  bool operator==(const DoneFaninKey &o) const {
    return owner == o.owner && dstOp == o.dstOp && dstYield == o.dstYield;
  }
};

struct PlannedRelease {
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> edgeIdxs;
};

inline bool operator==(const PlannedRelease &lhs,
                       const PlannedRelease &rhs) {
  return lhs.groupIdx == rhs.groupIdx && lhs.edgeIdxs == rhs.edgeIdxs;
}

struct OptSyncDag {
  ResourceId resource{0, 0};
  unsigned groupIdx = 0;
  SmallVector<unsigned, 4> memberIndices;
  SmallVector<SyncGroup> groups;
  SmallVector<unsigned> edgeToGroup;
  SmallVector<std::pair<unsigned, std::optional<PartitionId>>, 2>
      releasedSemaphores;
  DenseMap<Operation *, SmallVector<PlannedRelease, 2>> releaseBeforeOp;
  DenseMap<Region *, SmallVector<PlannedRelease, 2>> releaseBeforeYield;
  DenseMap<Operation *, SmallVector<PlannedRelease, 2>> releaseAfterOp;
  DenseMap<Region *, SmallVector<PlannedRelease, 2>> releaseAfterYield;
  DenseMap<Operation *, SmallVector<unsigned, 2>> acquireBeforeOp;
  DenseMap<Region *, SmallVector<unsigned, 2>> acquireBeforeYield;
  DenseMap<unsigned, Operation *> ifYieldJoinAccess;
  DenseMap<unsigned, Operation *> dstBranchEntryAnchor;
  DenseMap<unsigned, Operation *> tmemLoopExitRead;
  DenseMap<unsigned, Operation *> loopEntryHandoffAccess;
  DenseMap<unsigned, Region *> skippedInitialLoopCarrierRegion;
  DenseSet<unsigned> edgesDeferringToSkippedLoopExit;
  DenseSet<unsigned> terminalLoopReadEdgesDeferringToExit;
  DenseSet<unsigned> srcYieldParentWarpFor;
  DenseSet<Operation *> accessOps;
  DenseSet<Operation *> threadForOps;
  DenseSet<Operation *> threadIfOps;
};

struct GroupBacking {
  SmallVector<unsigned, 4> memberIndices;
  DenseMap<unsigned, unsigned> memberToBackingIndex;
  SmallVector<Value, 4> buffers;
  SmallVector<Type, 4> bufferTypes;
};

using BackingKey = std::pair<unsigned /*groupIdx*/, int64_t /*resourceKey*/>;

struct ResourceSemaphores {
  DenseMap<unsigned, Value> byClass;
  unsigned seedId = 0;
  Value forClass(const SyncPlan &sp, std::optional<unsigned> edgeIdx) const {
    unsigned id = edgeIdx ? sp.semaFind(*edgeIdx) : sp.semaFind(seedId);
    return byClass.lookup(id);
  }
};

enum class SyncAnchorKind {
  AcquireBeforeOp,
  AcquireBeforeYield,
  ReleaseBeforeOp,
  ReleaseBeforeYield,
  ReleaseAfterOp,
  ReleaseAfterYield
};

struct AcquireRecord {
  Value semaphore;
  Value token;
  std::optional<PartitionId> owner;
};

struct EmittedSyncRecord {
  unsigned groupIdx = 0;
  SyncAnchorKind kind = SyncAnchorKind::AcquireBeforeOp;
  Operation *anchor = nullptr;
  Region *yieldRegion = nullptr;
  SmallVector<unsigned, 4> edgeIdxs;
};

struct PoisonTokenRecord {
  Operation *op = nullptr;
  Operation *insertBefore = nullptr;
};

} // namespace mlir::triton::nvws::insert_semas

#endif // NVIDIA_NVWS_TRANSFORMS_INSERT_SEMAS_MODEL_H_
