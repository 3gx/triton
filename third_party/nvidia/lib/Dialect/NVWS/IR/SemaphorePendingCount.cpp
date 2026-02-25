/* Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "nvidia/include/Dialect/NVWS/IR/SemaphorePendingCount.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::triton::nvws {
namespace {

std::optional<int>
getReleaseAsyncContribution(ArrayAttr asyncOps,
                            std::optional<AsyncOp> &unsupportedAsyncOp) {
  int contribution = 0;
  for (Attribute asyncOp : asyncOps) {
    auto kind = cast<AsyncOpAttr>(asyncOp).getValue();
    switch (kind) {
    case AsyncOp::TC5MMA:
    case AsyncOp::TMALoad:
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
    case AsyncOp::TMEMCopy:
      ++contribution;
      break;
    default:
      unsupportedAsyncOp = kind;
      return std::nullopt;
    }
  }

  return contribution;
}

} // namespace

SemaphorePendingCountAnalysis analyzeSemaphorePendingCount(SemaphoreCreateOp op) {
  SemaphorePendingCountAnalysis analysis;
  llvm::DenseMap<int, int> partitionContrib;
  int pendingCount = 0;

  for (Operation *user : op->getUsers()) {
    auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user);
    if (!releaseOp || !gpu::hasPartition(user))
      continue;

    auto partitionIds = gpu::getPartitionIds(user);
    if (partitionIds.size() != 1) {
      analysis.invalidPartitionArity = partitionIds.size();
      return analysis;
    }

    std::optional<AsyncOp> unsupportedAsyncOp;
    auto contribution =
        getReleaseAsyncContribution(releaseOp.getAsyncOps(), unsupportedAsyncOp);
    if (!contribution) {
      analysis.unsupportedAsyncOp = unsupportedAsyncOp;
      return analysis;
    }

    int partitionId = partitionIds.front();
    auto [it, inserted] =
        partitionContrib.try_emplace(partitionId, contribution.value());
    if (inserted) {
      pendingCount += contribution.value();
      continue;
    }

    if (it->second != contribution.value()) {
      analysis.inconsistentPartitionId = partitionId;
      analysis.expectedContribution = it->second;
      analysis.actualContribution = contribution.value();
      return analysis;
    }
  }

  analysis.pendingCount = pendingCount == 0 ? 1 : pendingCount;
  return analysis;
}

} // namespace mlir::triton::nvws
