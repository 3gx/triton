#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_

// Stage 2 — OWNER-DAG (spec fable/semas-report3.md section 4; plan
// fable/new-insert-semas-plan-2.md commit 2). Extends the ACCESS-DAG in
// place: splices Enter/Exit brackets onto every For/If region chain
// (including the VIRTUAL else chain when the IR has no else region — spec
// section 4 else rule; the Func chain gets neither), and fills the OWNER
// half of pieceInfo:
//   - loop body: carried owner := owner of the body's FIRST TOUCHER of the
//     piece (a nested region row counts as a toucher of the pieces in its
//     summary, with its own carried owner — hence post-order),
//   - scf.if: branch owner := owner of the FIRST IN-BRANCH TOUCHER (then
//     chain first, then else chain; no fallbacks),
// then copies the full per-piece record onto the bracket nodes.
// Invariants (asserted, never repaired): For == Enter == Exit per piece;
// If == then.Enter == then.Exit == else.Enter == else.Exit; effects are
// stage-1 copies, never recomputed. Pure analysis; no IR mutation.

// Wrap a region chain (possibly empty) with Enter/Exit bracket nodes; the
// Exit sits where the region terminator (scf.yield) is. Returns the new
// chain head (the Enter node).
#include "InsertSemas.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

LogicalResult buildOwnerDag(GroupDag &g);

void printPieceRecord(llvm::raw_ostream &os, const Node *n,
                      Operation *anchor);

void dumpGroupOwnerDag(GroupDag &g, triton::FuncOp funcOp);

} // namespace nvws_semas
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_OWNER_DAG_H_
