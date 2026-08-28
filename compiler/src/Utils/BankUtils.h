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

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace buddy {
namespace buckyball {

class BankSMatMulOp;

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
  if (val > mask) {
    std::fprintf(stderr, "packBits: value %llu does not fit in [%d:%d]\n",
                 (unsigned long long)val, startBit, endBit);
    std::abort();
  }
  return val << startBit;
}

static inline uint64_t matrixRs2(uint64_t rows, uint64_t cols, uint64_t k) {
  if (rows == 0 || cols == 0 || cols > 0xfff || k == 0 || rows > 0xfff ||
      k > 0xfff || rows % 16 || cols % 16 || k % 16) {
    std::fprintf(
        stderr,
        "matrix rs2: rows/cols/k must fit in 12 bits (got %llu %llu %llu)\n",
        (unsigned long long)rows, (unsigned long long)cols,
        (unsigned long long)k);
    std::abort();
  }
  return packBits(rows, 0, 11) | packBits(cols, 12, 23) | packBits(k, 24, 35);
}

static inline mlir::Value createBankSMatMul(mlir::OpBuilder &b,
                                            mlir::Location loc, mlir::Type wrTy,
                                            mlir::Value a, mlir::Value opB,
                                            mlir::Value wr, mlir::Value cfg) {
  return b.create<BankSMatMulOp>(loc, wrTy, a, opB, wr, cfg,
                                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                                 b.getI64IntegerAttr(0));
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
