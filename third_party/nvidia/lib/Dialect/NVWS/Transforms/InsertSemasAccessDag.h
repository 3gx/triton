#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_

// Stage 1 — ACCESS-DAG (spec fable/semas-report3.md section 3; plan
// fable/new-insert-semas-plan-2.md commit 1). Pure analysis: discovery of
// buffer groups, cut-point pieces/footprints, access events with R/W
// touches, region effect summaries, and the ACCESS-DAG dump. No IR mutation.

// ---------------------------------------------------------------------------
// Discovery: bucket allocs by buffer.id (synthetic id when absent), uniform
// over TMEM and local. TMEM = every ttng.tmem_alloc; local = every
// mutable-memdesc ttg.local_alloc.
// ---------------------------------------------------------------------------
#include "InsertSemas.h"

namespace mlir {
namespace triton {
namespace nvws_semas {

SmallVector<GroupDag, 0> collectGroups(triton::FuncOp funcOp);

LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp);

void dumpGroupAccessDag(GroupDag &g, triton::FuncOp funcOp);

} // namespace nvws_semas
} // namespace triton
} // namespace mlir

#endif // NVWS_TRANSFORMS_INSERT_SEMAS_ACCESS_DAG_H_
