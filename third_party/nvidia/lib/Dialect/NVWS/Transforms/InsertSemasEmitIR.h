#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_IR_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_IR_H_

// Stage 4 of nvws-insert-semas: EMIT-IR (spec section 6; plan commit 4).
// Strict order: (1) token-nuke pre-process, (2) backings + creates + entry
// acquires, (3) aggregated signature rewrites, (4) render walk per group,
// (5) post-emit verifier, (6) TMEM backing coalescing, (7) loop-scheduler
// workaround. The emitter transcribes the SYNC-DAG; it decides nothing.

// ---------------------------------------------------------------------------
// Shared emission state.
// ---------------------------------------------------------------------------
#include "InsertSemas.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

LogicalResult emitIR(triton::FuncOp funcOp,
                     MutableArrayRef<GroupDag> groups);

} // namespace nvws_semas
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_EMIT_IR_H_
