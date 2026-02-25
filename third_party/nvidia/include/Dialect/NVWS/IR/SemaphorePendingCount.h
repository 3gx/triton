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

#ifndef DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_
#define DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_

#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include <optional>

namespace mlir::triton::nvws {

struct SemaphorePendingCountAnalysis {
  int pendingCount = 1;
  std::optional<unsigned> invalidPartitionArity;
  std::optional<AsyncOp> unsupportedAsyncOp;
  std::optional<int> inconsistentPartitionId;
  int expectedContribution = 0;
  int actualContribution = 0;

  bool hasError() const {
    return invalidPartitionArity.has_value() || unsupportedAsyncOp.has_value() ||
           inconsistentPartitionId.has_value();
  }
};

SemaphorePendingCountAnalysis analyzeSemaphorePendingCount(SemaphoreCreateOp op);

} // namespace mlir::triton::nvws

#endif // DIALECT_NVWS_IR_SEMAPHOREPENDINGCOUNT_H_
