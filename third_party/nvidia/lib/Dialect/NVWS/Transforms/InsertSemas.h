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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace triton {
namespace nvws_semas {

using namespace mlir;
using triton::nvws::AsyncOp;
namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;
namespace nvws = triton::nvws;

// Shared data model and helpers for the nvws-insert-semas pass.
// Spec: fable/semas-report3.md (section 2 — the Node model);
// plan: fable/new-insert-semas-plan-2.md (section 2 — every type defined).
//
// Self-contained shared header for the per-stage translation units
// (plan section 1 addendum, post-closure refactor). All identifiers are
// indices into per-group tables, allocated in program/discovery order —
// never hash order (plan ground rule 6: hash containers are lookup-only).

// ---------------------------------------------------------------------------
// Identifiers.
// ---------------------------------------------------------------------------
using MemberId = unsigned; // index into PieceTable::members
using PieceId = unsigned;  // index into PieceTable::pieces
using SemaId = unsigned;   // index into SemaTable::semas
using CompId = unsigned;   // connected component of pieces = one token game

// ---------------------------------------------------------------------------
// Owner: who executes an op / who is planned to hold a piece.
// root/external (std::nullopt) is distinct from partition 0.
// ---------------------------------------------------------------------------
using PartitionId = std::pair<int /*ttg.partition*/, int /*ws tag*/>;
using Owner = std::optional<PartitionId>;

inline int64_t ownerKey(const Owner &o) {
  if (!o)
    return -1;
  return (static_cast<int64_t>(o->second) << 32) |
         static_cast<uint32_t>(o->first);
}

inline bool sameOwner(const Owner &a, const Owner &b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return *a == *b;
}

// ---------------------------------------------------------------------------
// Access classification (spec section 1.1): R = provably load-only
// (local_load, tmem_load, MMA A/B operand touches); W = everything else
// (stores, sourceful allocs, descriptor loads/gathers, MMA accumulator).
// ---------------------------------------------------------------------------
enum class Effect : uint8_t { R, W };

inline Effect joinEffect(Effect a, Effect b) {
  return (a == Effect::W || b == Effect::W) ? Effect::W : Effect::R;
}

// Region-row boundary plan, per piece (spec section 4). The effect half is
// computed at stage 1 on For/If nodes (subtree OR); the owner half is
// assigned at stage 2 (loop: first toucher; if: first in-branch toucher).
struct PieceInfo {
  Owner owner;
  Effect effect = Effect::R;
};

// ---------------------------------------------------------------------------
// One step of a memdesc alias chain (stage-1 fact; replayed at emit to
// rebuild the view from the semaphore buffer).
// ---------------------------------------------------------------------------
struct AliasStep {
  Operation *op = nullptr; // ttg.memdesc_{index,subview,trans,reinterpret,reshape}
  unsigned operandIdx = 0; // which operand carries the source memdesc
  Type resultType;         // captured at stage 1 (context-owned — survives
                           // op erasure during render)
};

// One buffer-touch of one access op. Pieces are DERIVED, never stored:
// footprint(touch) = PieceTable::footprint[touch.member].
struct Touch {
  MemberId member = 0;
  Effect effect = Effect::R;
  Value accessValue;                // the memdesc SSA value the op uses
  Type accessType;                  // its type, captured at stage 1
                                    // (erase-proof; values may dangle after
                                    // sourceful allocs are replaced)
  SmallVector<AliasStep, 2> alias;  // chain: member alloc -> accessValue
};

// ---------------------------------------------------------------------------
// The DAG node — one type end to end (ACCESS -> OWNER -> SYNC snapshots).
// prev/next = program-order chain within the parent region (NOT dependency
// edges); children = region-chain heads of For/If rows; sat = the single
// dependency relation, injected only at the SYNC stage.
// ---------------------------------------------------------------------------
struct Node;

// Carrier-threading facts, computed at stage 3 and stored ON the For/If
// node (the DAG is the authority; "ThreadingPlan" is only the derived
// emission-time aggregation across groups).
struct Crossing {
  enum class HoldKind {
    GATED,
    POINT_OF_USE,
    PASSTHROUGH_DROP,
  };

  CompId comp = 0;     // which token game (group implicit: the node's DAG)
  Owner slotOwner;     // partition stamped on this slot's
                       // ttg.partition.outputs entry and yield attrs —
                       // the owner of every chain's final carrier
  SmallVector<Node *, 2> finals; // PER CHAIN, parallel to children: the
                       // chain's last carrier-producing row (an Acquire of
                       // the component, or a nested region row with its
                       // own crossing). Its token is what that chain's
                       // yield returns — the NEW token. nullptr =
                       // PASS-THROUGH: that chain yields the INCOMING
                       // carrier unchanged (the OLD token).
  // HOLD-RULE gate facts. Computed at stage 3 for For-row crossings and
  // printed in the SYNC dump. holdGated is retained as the emission-facing
  // boolean: both POINT_OF_USE and PASSTHROUGH_DROP materialize no carrier slot.
  HoldKind holdKind = HoldKind::GATED;
  bool holdGated = true;            // true: keep the rotated boundary device
  const char *holdGateReason = "";  // dump tag; empty when UNGATED
  Node *holdFirstToucher = nullptr; // UNGATED: the point-of-use target row
  Node *holdFeedAcquire = nullptr;  // UNGATED: the slot's feeding entry-
                                    // instance acquire (unlinked at M2 —
                                    // iteration 1 pairs with the initial
                                    // permit instead)
  bool holdRegionTail = false;      // POINT_OF_USE: finals[0] is a region row
  // v5 uniform hold-builder side-band. M1 computes these fields beside the
  // legacy gate; M2/M3 make them the emission authority.
  HoldKind uniformHoldKind = HoldKind::GATED;
  const char *uniformHoldReason = "";
  Node *uniformFirstToucher = nullptr;
  Node *uniformFeedAcquire = nullptr;
  bool uniformRegionTail = false;
};

struct Node {
  enum Kind { Func, For, If, Enter, Exit, Access, Acquire, Release };
  Kind kind = Access;
  Operation *op = nullptr;     // For/If/Access anchor; null otherwise
  Node *parent = nullptr;
  Node *prev = nullptr, *next = nullptr;
  SmallVector<Node *, 2> children; // For: body head; If: then head[, else head]
  Owner owner;                     // Access/Acquire/Release: executing partition
  SmallVector<Touch, 2> touches;   // Access: one per touched member
  DenseMap<PieceId, PieceInfo> pieceInfo; // Enter/Exit/For/If only; iterate
                                          // sorted by PieceId (determinism)
  SemaId sema = 0;
  unsigned count = 0;              // Acquire pending count
  SmallVector<AsyncOp, 1> payloads; // Release payload(s): the source
                                    // holder's last real access payload —
                                    // a UNION after dedupe merges (emitted
                                    // as the release's async_ops array)
  Node *sat = nullptr;             // Release -> the ONE Acquire it satisfies
  SmallVector<Crossing, 1> crossings; // For/If only (stage 3): carrier
                                      // slots crossing this region
  SmallVector<int, 2> requiredParts;  // For/If only (stage 3): sorted union
                                      // of subtree row partitions = the
                                      // clone set after partition-loops
                                      // (the C10 ttg.partition array)
};

// Deterministic iteration over a node's pieceInfo.
inline SmallVector<std::pair<PieceId, PieceInfo>, 4>
sortedPieceInfo(const Node *n) {
  SmallVector<std::pair<PieceId, PieceInfo>, 4> v(n->pieceInfo.begin(),
                                                  n->pieceInfo.end());
  llvm::sort(v, [](const auto &a, const auto &b) { return a.first < b.first; });
  return v;
}

// ---------------------------------------------------------------------------
// Stage-1 side tables (plan section 2).
// ---------------------------------------------------------------------------
struct Member {
  Operation *allocOp = nullptr; // original ttng.tmem_alloc / ttg.local_alloc
                                // (buffer.* attrs live on it, preserved)
  gpu::MemDescType type;
  int64_t offset = 0; // [offset, offset+extent) in the group's native unit:
  int64_t extent = 1; // TMEM = columns (getTmemAllocSizes().numCols);
                      // local = leading dim of the memdesc shape (the memory
                      // planner's offset unit — corpus-proven: 128x128xf16
                      // members at offsets 0/64 overlap, 0/256 do not).
};

struct Piece { // cut-point interval (spec section 3 item 2)
  int64_t lo = 0, hi = 0;            // [lo, hi) in native units
  SmallVector<MemberId, 2> cover;    // members containing this piece, ascending
};

struct PieceTable { // one per group
  SmallVector<Member> members;                    // discovery order
  SmallVector<Piece> pieces;                      // ascending by lo
  SmallVector<SmallVector<PieceId, 2>> footprint; // per member, ascending
  SmallVector<CompId> pieceComp;                  // per piece: its component
};

// ---------------------------------------------------------------------------
// Stage-3 side tables (defined now per plan section 2; filled at commit 3).
// ---------------------------------------------------------------------------
struct Sema {
  std::string name;     // "S<k>" per group in creation order; "E<k>" for
                        // dedicated entry semaphores (deterministic)
  CompId component = 0;
  SmallVector<PieceId, 2> pieces;
  unsigned count = 0; // pending count of the PRIMARY acquire (the first
                      // destination group); merged For-row groups add their
                      // own acquire instances with their own counts
  unsigned expectedReleases = 0; // total release sites across all groups
                                 // sharing this semaphore (balance check)
  bool isEntry = false; // first event in chain order is an acquire
                        //   => nvws.semaphore.create ... true
  Owner inheritStamp;   // entry semas: the placement-point holder (carrier
                        // inherit). The entry-acquire NODE is owned by
                        // ROOT; emission stamps the op with THIS fact
                        // (partition+tag, or no attrs for root) — matching
                        // the previous pass's emitted IR.
  Value create;         // backpatch slot, filled at emit step 2
};

struct SemaTable {
  SmallVector<Sema> semas; // index = SemaId; allocation order = program order
};

struct BackingPlan {
  int numStages = 1;                 // local: always 1; TMEM: 1 or 2
  Operation *hoistAnchor = nullptr;  // function scope, before outermost WS loop
  SmallVector<Value> backing;        // per member; backpatch slot (emit step 2)
};

// NOTE: there is no stored ThreadingPlan — crossings live on For/If Nodes
// (Crossing above). The emission-time slot numbering across groups is a
// derived aggregation (plan section 2).

// ---------------------------------------------------------------------------
// Per group: the whole artifact handed from stage to stage.
// ---------------------------------------------------------------------------
enum class MemKind { Tmem, Local };

struct GroupDag {
  unsigned groupIdx = 0;
  int64_t bufferId = 0;   // buffer.id attr value, or synthetic (negative)
  bool synthetic = false; // no buffer.id attr on the alloc
  MemKind memory = MemKind::Tmem;
  PieceTable pieceTable;
  // Alias map: tracked memdesc SSA value -> (member, view chain from the
  // member's alloc). Lookup-only (never iterated for output).
  DenseMap<Value, std::pair<MemberId, SmallVector<AliasStep, 2>>> aliases;
  SmallVector<Operation *, 1> ttDescriptorFedMembers; // tt-form descriptor-
                                          // fed sourceful allocs (pipeline-
                                          // invariant guard, contract D)
  DenseSet<Operation *> accessRowOps;     // emit-time: every Access-row
                                          // anchor — the sourceful-alloc
                                          // RAUW must not steal their
                                          // operands (each row retargets
                                          // itself with its own owner's
                                          // view). Lookup-only.
  SmallVector<std::unique_ptr<Node>> nodes; // pool; pointer-stable
  Node *root = nullptr;                     // Func node of current snapshot
  SemaTable semaTable;
  BackingPlan backingPlan;

  bool isTmem() const { return memory == MemKind::Tmem; }
  Node *newNode(Node::Kind k, Operation *op, Node *parent) {
    nodes.push_back(std::make_unique<Node>());
    Node *n = nodes.back().get();
    n->kind = k;
    n->op = op;
    n->parent = parent;
    return n;
  }
};

// ---------------------------------------------------------------------------
// Helpers re-derived from the spec (section 1.1) — not copied from the old
// implementation.
// ---------------------------------------------------------------------------
inline constexpr StringLiteral kBufferIdAttrName = "buffer.id";
inline constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";

inline std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

// Effective owner: (partition, wsTag) from the op's ttg.partition annotation
// plus the nearest enclosing (or intrinsic) warp-specialize tag; an op with
// no annotation, a multi-partition annotation, or no reachable tag is root.
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

// The WS tag of the scope an anchor row sits in (the nearest enclosing
// WS-tagged loop, including the anchor itself), if any.
inline std::optional<int> anchorScopeTag(Operation *anchor) {
  Operation *p = anchor;
  while (p && !(isa<scf::ForOp>(p) && gpu::hasWarpSpecializeTag(p)))
    p = p->getParentOfType<scf::ForOp>();
  if (!p)
    return std::nullopt;
  return *gpu::getWarpSpecializeTag(p);
}

// Owner display (spec dump format): `{p}` inside the owning WS loop,
// `{@tag.p}` outside it, `root` for no owner.
inline std::string ownerStr(Operation *anchor, const Owner &owner) {
  if (!owner)
    return "root";
  std::string s;
  llvm::raw_string_ostream os(s);
  auto tag = anchor ? anchorScopeTag(anchor) : std::nullopt;
  if (tag && *tag == owner->second)
    os << "{" << owner->first << "}";
  else
    os << "{@" << owner->second << "." << owner->first << "}";
  return s;
}

// Release payload classification (spec section 1.1): a table lookup, never
// a derivation. Used from commit 3 on.
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

// Supported memdesc view ops (alias chain steps). Anything else that
// forwards a tracked memdesc is a hard diagnostic.
inline bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_trans" || name == "ttg.memdesc_reinterpret" ||
         name == "ttg.memdesc_reshape";
}

// Native-unit extent of a member allocation (see Member::extent).
inline int64_t memberExtent(MemKind memory, gpu::MemDescType type) {
  if (memory == MemKind::Tmem)
    return static_cast<int64_t>(mlir::triton::getMemDescSize(type));
  return type.getShape().empty() ? 1 : type.getShape().front();
}

// Dump tree prefix.
inline std::string treePrefix(unsigned depth) {
  std::string s;
  for (unsigned i = 0; i < depth; ++i)
    s += "|  ";
  return s;
}

inline bool shouldDumpDag() {
  const char *env = ::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  return env && StringRef(env) == "1";
}

// Pre-order walk over every node of a chain, descending into For/If
// region children — the one traversal shape shared by the stages.
template <typename Fn>
inline void forEachNode(Node *head, Fn &&fn) {
  for (Node *n = head; n; n = n->next) {
    fn(n);
    if (n->kind == Node::For || n->kind == Node::If)
      for (Node *child : n->children)
        forEachNode(child, fn);
  }
}

template <typename Fn>
inline void forEachNode(GroupDag &g, Fn &&fn) {
  if (!g.root->children.empty())
    forEachNode(g.root->children[0], std::forward<Fn>(fn));
}

} // namespace nvws_semas
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_H_
