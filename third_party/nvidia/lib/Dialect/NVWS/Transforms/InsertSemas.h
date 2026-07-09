// Shared model and traversal utilities; see sema-docs/insert-semas/overview.md.
#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_H_
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir::triton::nvws_semas {
using namespace mlir;
using triton::nvws::AsyncOp;
namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;
namespace nvws = triton::nvws;
using MemberId = unsigned;
using PieceId = unsigned;
using SemaId = unsigned;
using PartitionId = std::pair<int /*ttg.partition*/, int /*ws tag*/>;
using Owner = std::optional<PartitionId>;
inline int64_t ownerKey(const Owner &o) {
  return o ? (static_cast<int64_t>(o->second) << 32) |
                 static_cast<uint32_t>(o->first)
           : -1;
}
inline bool sameOwner(const Owner &a, const Owner &b) { return a == b; }

enum class Effect : uint8_t { R, W };
inline Effect joinEffect(Effect a, Effect b) {
  return (a == Effect::W || b == Effect::W) ? Effect::W : Effect::R;
}

struct PieceInfo {
  Owner owner;
  Effect effect = Effect::R;
};

struct AliasStep {
  Operation *op = nullptr;
  unsigned operandIdx = 0;
  Type resultType;
};

struct Touch {
  MemberId member = 0;
  Effect effect = Effect::R;
  Value accessValue;
  Type accessType;
  SmallVector<AliasStep, 2> alias;
};

struct Node;

// Static identity of one token.  The group is implicit in the GroupDag being
// processed; producer identifies the token SSA (an acquire or a region
// carrier), while that token preserves its dynamic physical-slot lineage.
// `sema` is the selected render channel; region results preserve an incoming
// channel when one exists, otherwise use a deterministic fallback. The exact
// producer may have acquired another channel in the same GroupDag.
struct TokenRef {
  Node *producer = nullptr;
  SemaId sema = 0;
  Owner owner;
};
struct CompletionSummary {
  bool valid = true, usesInput = true, hasFallback = false;
  gpu::StageCluster schedule;
};
// A region that threads a token: each exit path names the boundary owner's
// last token producer inside that path, or nullptr to pass the ENTER token
// through unchanged.
struct RegionFlow {
  Owner owner;
  SmallVector<Node *, 2> exits; // nullptr means pass the ENTER token
  // Concrete exit channel used when no non-entry token reaches the region.
  std::optional<SemaId> concreteSema;
};

struct Node {
  enum Kind { Func, For, If, Enter, Exit, Access, Acquire, Release };
  Kind kind = Access;
  Operation *op = nullptr;
  Operation *completionAnchor = nullptr;
  Node *parent = nullptr, *prev = nullptr, *next = nullptr;
  SmallVector<Node *, 2> children;
  Owner owner;
  SmallVector<Touch, 2> touches;
  DenseMap<PieceId, PieceInfo> pieceInfo;
  SemaId sema = 0;
  unsigned count = 0;
  SmallVector<AsyncOp, 1> payloads;
  gpu::StageCluster stageCluster;
  std::optional<int64_t> stageOffset;
  std::optional<int64_t> bufferStageOffset;
  bool postLoopAcquire = false;
  // The SYNC-DAG proved that this node can reuse its owner's earlier token in
  // the same chain without a fresh handoff. EmitIR renders this fact; it does
  // not infer token-reuse eligibility on its own.
  std::optional<int64_t> reuseTokenOwner;
  std::optional<RegionFlow> flow;
  std::optional<Owner> completionOwner;
  CompletionSummary completion;
  SmallVector<int, 2> requiredParts;
  bool isRegion() const { return kind == For || kind == If; }
  bool isProtocol() const { return kind == Acquire || kind == Release; }
  bool isSlotEvent() const {
    return kind == Access || (isRegion() && !pieceInfo.empty());
  }
  bool isLinked() const {
    return parent &&
           (prev || next || llvm::is_contained(parent->children, this));
  }
  bool isDirectLoopNode() const { return isLinked() && parent->kind == For; }
};
struct ProtocolArc {
  Node *release = nullptr, *acquire = nullptr;
  Node *producer = nullptr, *consumer = nullptr;
  Node *wait = nullptr;
};
inline void markTokenReuse(Node *n, const Owner &owner) {
  assert(owner && n->owner && sameOwner(n->owner, owner));
  n->reuseTokenOwner = ownerKey(owner);
}
inline bool nodeReusesToken(const Node *n, const Owner &owner) {
  return owner && n->reuseTokenOwner && *n->reuseTokenOwner == ownerKey(owner);
}
inline CompletionSummary joinCompletion(CompletionSummary a,
                                        CompletionSummary b) {
  if (!a.valid || !b.valid ||
      (a.hasFallback && b.hasFallback && a.schedule != b.schedule))
    return {false};
  return {true, a.usesInput || b.usesInput, a.hasFallback || b.hasFallback,
          a.hasFallback ? a.schedule : b.schedule};
}
inline CompletionSummary applyCompletion(CompletionSummary flow,
                                         CompletionSummary input) {
  if (!flow.valid || !input.valid ||
      (flow.usesInput && flow.hasFallback && input.hasFallback &&
       flow.schedule != input.schedule))
    return {false};
  if (!flow.usesInput)
    return {true, false, true, flow.schedule};
  return {true, input.usesInput, flow.hasFallback || input.hasFallback,
          input.hasFallback ? input.schedule : flow.schedule};
}
inline CompletionSummary completionAfterChain(Node *head, const Owner &owner,
                                              CompletionSummary state = {}) {
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Access && sameOwner(n->owner, owner)) {
      Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
      state = {true, false, true, gpu::getStageCluster(completion)};
    } else if (n->isRegion() && n->completionOwner &&
               !(n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op)) &&
               sameOwner(*n->completionOwner, owner)) {
      state = applyCompletion(n->completion, state);
    }
  }
  return state;
}
inline SmallVector<std::pair<PieceId, PieceInfo>, 4>
sortedPieceInfo(const Node *n) {
  SmallVector<std::pair<PieceId, PieceInfo>, 4> v(n->pieceInfo.begin(),
                                                  n->pieceInfo.end());
  llvm::sort(v, [](const auto &a, const auto &b) { return a.first < b.first; });
  return v;
}
// Outer optional: the node has one uniform owner. Inner Owner may still be
// root; an empty outer optional means no pieces or mixed owners.
inline std::optional<Owner> uniformPieceOwner(const Node *n) {
  bool fresh = true;
  Owner owner;
  for (auto &[p, pi] : sortedPieceInfo(n)) {
    if (fresh) {
      owner = pi.owner;
      fresh = false;
    } else if (!sameOwner(owner, pi.owner)) {
      return std::nullopt;
    }
  }
  if (fresh)
    return std::nullopt;
  return std::optional<Owner>(std::in_place, owner);
}

struct Member {
  Operation *allocOp = nullptr;
  gpu::MemDescType type;
  int64_t offset = 0;
  int64_t extent = 1;
  int64_t circularStart = 0;
};

struct Piece {
  int64_t lo = 0, hi = 0;
  SmallVector<MemberId, 2> cover;
};

struct PieceTable {
  SmallVector<Member> members;
  SmallVector<Piece> pieces;
  SmallVector<SmallVector<PieceId, 2>> footprint;
};

struct Sema {
  std::string name;
  unsigned count = 0;
  bool isEntry = false;
  Owner entryTokenOwner;
  Value create;
};

enum class MemKind { Tmem, Local };

struct GroupDag {
  int64_t bufferId = 0;
  bool synthetic = false;
  bool mixedDepthPhysicalAlias = false;
  MemKind memory = MemKind::Tmem;
  bool circular = false;
  PieceTable pieceTable;
  DenseMap<Value, std::pair<MemberId, SmallVector<AliasStep, 2>>> aliases;
  SmallVector<Operation *, 1> ttDescriptorFedMembers;
  DenseSet<Operation *> accessNodeOps;
  SmallVector<std::unique_ptr<Node>> nodes;
  Node *root = nullptr;
  SmallVector<Sema> semas;
  SmallVector<ProtocolArc> protocolArcs;
  int numCopies = 1, numSemaphoreCopies = 1;
  SmallVector<Value> backing;
  bool isTmem() const { return memory == MemKind::Tmem; }
  bool isLocal() const { return memory == MemKind::Local; }
  bool isCircular() const { return circular; }
  Node *newNode(Node::Kind k, Operation *op, Node *parent) {
    nodes.push_back(std::make_unique<Node>());
    Node *n = nodes.back().get();
    n->kind = k;
    n->op = op;
    n->completionAnchor = op;
    n->parent = parent;
    return n;
  }
};
inline Sema &getSema(GroupDag &group, const Node *node) {
  return group.semas[node->sema];
}
inline const Sema &getSema(const GroupDag &group, const Node *node) {
  return group.semas[node->sema];
}
inline bool canOwnMixedDepthTmem(const GroupDag &owner,
                                 const GroupDag &reuser) {
  if (!owner.isTmem() || !reuser.isTmem() ||
      owner.pieceTable.members.size() != 1 ||
      reuser.pieceTable.members.size() != 1)
    return false;
  const Member &ownerMember = owner.pieceTable.members.front();
  const Member &reuserMember = reuser.pieceTable.members.front();
  unsigned ownerWidth = ownerMember.type.getElementTypeBitWidth();
  unsigned reuserWidth = reuserMember.type.getElementTypeBitWidth();
  if (ownerWidth != reuserWidth && ownerWidth != 2 * reuserWidth)
    return false;
  int64_t ownerSpan = ownerMember.extent * owner.numCopies;
  int64_t reuserSpan = reuserMember.extent * reuser.numCopies;
  return reuserMember.offset >= ownerMember.offset &&
         reuserMember.offset + reuserSpan <= ownerMember.offset + ownerSpan;
}
inline constexpr StringLiteral kBufferIdAttrName = "buffer.id";
inline constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
inline constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";
inline constexpr StringLiteral kBufferCircularAttrName = "buffer.circular";
inline constexpr StringLiteral kBufferStartAttrName = "buffer.start";
inline std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}
inline InFlightDiagnostic semaError(Operation *op) {
  return op->emitError() << "nvws-insert-semas: ";
}
template <typename OpTy> inline InFlightDiagnostic semaError(OpTy op) {
  return semaError(op.getOperation());
}
inline Owner resolveOwner(Operation *op) {
  if (!gpu::hasPartition(op))
    return std::nullopt;
  auto ids = gpu::getPartitionIds(op);
  if (ids.size() != 1)
    return std::nullopt;
  Operation *tagSrc = op;
  while (tagSrc && !gpu::hasWarpSpecializeTag(tagSrc))
    tagSrc = tagSrc->getParentOfType<scf::ForOp>();
  if (!tagSrc)
    return std::nullopt;
  return PartitionId{*ids.begin(), *gpu::getWarpSpecializeTag(tagSrc)};
}
inline std::string ownerStr(Operation *anchor, const Owner &owner) {
  if (!owner)
    return "root";
  std::string s;
  llvm::raw_string_ostream os(s);
  Operation *scope = anchor;
  while (scope && !(isa<scf::ForOp>(scope) && gpu::hasWarpSpecializeTag(scope)))
    scope = scope->getParentOfType<scf::ForOp>();
  std::optional<int> tag =
      scope ? gpu::getWarpSpecializeTag(scope) : std::nullopt;
  if (tag && *tag == owner->second)
    os << "{" << owner->first << "}";
  else
    os << "{@" << owner->second << "." << owner->first << "}";
  return s;
}
inline AsyncOp asyncPayloadOf(Operation *op) {
  if (!op)
    return AsyncOp::NONE;
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op))
    if (Value src = localAlloc.getSrc())
      if (Operation *def = src.getDefiningOp())
        return asyncPayloadOf(def);
  if (isa<nvidia_gpu::MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  StringRef name = op->getName().getStringRef();
  if (name == "nvws.descriptor_load" || name == "nvws.descriptor_gather")
    return AsyncOp::TMALoad;
  return AsyncOp::NONE;
}
inline bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}
inline bool shouldDumpDag() {
  const char *env = ::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  return env && StringRef(env) == "1";
}
template <typename Fn> inline void forEachNode(Node *head, Fn &&fn) {
  for (Node *n = head; n; n = n->next) {
    fn(n);
    if (n->isRegion())
      for (Node *child : n->children)
        forEachNode(child, fn);
  }
}
template <typename Fn> inline void forEachNode(GroupDag &g, Fn &&fn) {
  if (!g.root->children.empty())
    forEachNode(g.root->children[0], std::forward<Fn>(fn));
}
template <typename Fn> inline void forEachRegionPostOrder(Node *head, Fn &&fn) {
  for (Node *n = head; n; n = n->next) {
    if (!n->isRegion())
      continue;
    for (Node *child : n->children)
      forEachRegionPostOrder(child, fn);
    fn(n);
  }
}
template <typename Fn>
inline void forEachTouchedPiece(const GroupDag &g, const Node *node, Fn &&fn) {
  for (const Touch &touch : node->touches)
    for (PieceId piece : g.pieceTable.footprint[touch.member])
      fn(piece, touch.effect);
}
inline bool touchesPiece(const GroupDag &g, const Node *node, PieceId piece) {
  bool found = false;
  forEachTouchedPiece(g, node, [&](PieceId p, Effect) { found |= p == piece; });
  return found;
}
inline bool nodeTouchesGroup(const GroupDag &g, const Node *node) {
  bool found = false;
  forEachTouchedPiece(g, node, [&](PieceId, Effect) { found = true; });
  return found;
}
template <typename Map>
inline void mergeEffect(Map &effects, PieceId piece, Effect effect) {
  auto [it, inserted] = effects.try_emplace(piece, effect);
  if (!inserted)
    it->second = joinEffect(it->second, effect);
}

FailureOr<SmallVector<GroupDag, 0>> collectGroups(triton::FuncOp funcOp);
LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp);
LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner,
                           int lowerSemaphoreNumStages, int &numTmemBlocks);
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups);
LogicalResult emitIR(triton::FuncOp funcOp, MutableArrayRef<GroupDag> groups);
} // namespace mlir::triton::nvws_semas
#endif // NVWS_TRANSFORMS_INSERT_SEMAS_H_
