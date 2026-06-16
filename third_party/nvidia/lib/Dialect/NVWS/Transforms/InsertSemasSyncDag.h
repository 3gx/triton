#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_

// Stage 3 of nvws-insert-semas: SYNC-DAG (spec section 5; plan commit 3).
//
// One ownership walk over the OWNER-DAG plays the token game per piece and
// records every handoff as an edge; edges are deduped (payload union),
// grouped by destination (fan-in), materialized as Acquire/Release nodes
// spliced into the chains, then completed with per-component entry
// acquires, carrier crossings (threading facts ON the For/If nodes),
// requiredParts clone sets, and the BackingPlan. Pure analysis: the IR is
// not touched.
//
// Walk rules (spec section 5.1):
//   1. W by p: edge from every holder != p; piece becomes Exclusive(p).
//   2. R by p: producer/holder reread updates in place; a new reader takes
//      an edge from the current primary holder and joins the readers.
//   3. Same-owner touches move lastRow/lastPayload, no edge.
//   4. EXIT closes in-body holders != carried owner — except holders the
//      carried owner already synchronized behind (transitive-sync skip),
//      and only when load-bearing (under a loop, or the piece is touched
//      later) — never as drains.
//   5. Region bodies walk FRESH local games: Exclusive(carried owner)
//      seeded at ENTER, versionProducer imported from the parent game, and
//      the payload seed IMPORTED when the carried owner is the parent
//      game's producer (rule A makes producer-bracketed branches the
//      normal conditional shape — their Enter-sourced release must carry
//      the producer's async payload).
//   6. A region row is one super-node touch per piece in the parent game
//      (touch first, then recurse); WS-For rows ADOPT root-held pieces
//      without an edge; after recursion the region row's holder carries
//      its games' final payloads (union for If).

// ---------------------------------------------------------------------------
// Walk state.
// ---------------------------------------------------------------------------
#include "InsertSemas.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

CompId compOfMember(GroupDag &g, MemberId m);

LogicalResult computeBackingPlan(GroupDag &g, triton::FuncOp funcOp,
                                 bool useMetaPartitioner, int &numTmemBlocks);

LogicalResult buildSyncDag(GroupDag &g, triton::FuncOp funcOp,
                           bool useMetaPartitioner, int &numTmemBlocks);

void dumpGroupSyncDag(GroupDag &g, triton::FuncOp funcOp);

} // namespace nvws_semas
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_SYNC_DAG_H_
