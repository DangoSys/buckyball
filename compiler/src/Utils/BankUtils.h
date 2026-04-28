//===- BankUtils.h - Utilities for Bank operations -----------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Utility functions for generating Bank-SSA operations.
//
//===----------------------------------------------------------------------===//

#ifndef BUCKYBALL_CONVERSION_BANKUTILS_H
#define BUCKYBALL_CONVERSION_BANKUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "Buckyball/BuckyballOps.h"

namespace buddy {
namespace buckyball {

/// Create i64 constant.
static inline mlir::Value createI64Const(mlir::OpBuilder &b, mlir::Location loc,
                                         int64_t val) {
  return b.create<mlir::arith::ConstantOp>(loc, b.getI64Type(),
                                           b.getI64IntegerAttr(val));
}

/// Create i64 constant from uint64_t.
static inline mlir::Value createI64ConstU(mlir::OpBuilder &b,
                                          mlir::Location loc, uint64_t val) {
  return b.create<mlir::arith::ConstantOp>(loc, b.getI64Type(),
                                           b.getI64IntegerAttr(val));
}

/// Pack bit field: extract bits [startBit, endBit] from val and shift to
/// position.
static inline uint64_t packBits(uint64_t val, int startBit, int endBit) {
  uint64_t width = endBit - startBit + 1;
  uint64_t mask = (1ULL << width) - 1;
  return (val & mask) << startBit;
}

/// Allocate a bank with given row/col dimensions.
static inline mlir::Value allocBank(mlir::OpBuilder &b, mlir::Location loc,
                                    int64_t row, int64_t col) {
  auto i64Type = b.getI64Type();
  return b.create<BankAllocOp>(loc, i64Type, b.getI64IntegerAttr(row),
                               b.getI64IntegerAttr(col));
}

/// Release a bank.
static inline void releaseBank(mlir::OpBuilder &b, mlir::Location loc,
                               mlir::Value bank) {
  b.create<BankReleaseOp>(loc, bank);
}

/// Move data from memref into bank.
static inline mlir::Value mvinBank(mlir::OpBuilder &b, mlir::Location loc,
                                   mlir::Value memref, mlir::Value bank,
                                   int64_t depth, int64_t stride = 1) {
  mlir::Value depthVal = createI64Const(b, loc, depth);
  mlir::Value strideVal = createI64Const(b, loc, stride);
  return b.create<BankMvinOp>(loc, bank.getType(), memref, bank, depthVal,
                              strideVal);
}

/// Move data from bank to memref.
static inline mlir::Value mvoutBank(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value memref, mlir::Value bank,
                                    int64_t depth, int64_t stride = 1) {
  mlir::Value depthVal = createI64Const(b, loc, depth);
  mlir::Value strideVal = createI64Const(b, loc, stride);
  return b.create<BankMvoutOp>(loc, bank.getType(), memref, bank, depthVal,
                               strideVal);
}

/// Create mset operation for bank allocation/release.
static inline MsetOp createMset(mlir::OpBuilder &b, mlir::Location loc,
                                uint64_t bankId, bool alloc, uint64_t row,
                                uint64_t col) {
  auto op = b.create<MsetOp>(loc, createI64ConstU(b, loc, bankId));
  op->setAttr("alloc", b.getBoolAttr(alloc));
  op->setAttr("row", b.getI64IntegerAttr(row));
  op->setAttr("col", b.getI64IntegerAttr(col));
  return op;
}

} // namespace buckyball
} // namespace buddy

#endif // BUCKYBALL_CONVERSION_BANKUTILS_H
