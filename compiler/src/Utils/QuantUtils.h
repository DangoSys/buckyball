//===- QuantUtils.h - Shared f32 quant helpers for bank-SSA ---------------===//

#ifndef BUCKYBALL_CONVERSION_QUANTUTILS_H
#define BUCKYBALL_CONVERSION_QUANTUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace buddy {
namespace buckyball {

static inline mlir::Value cstF32(mlir::OpBuilder &b, mlir::Location loc,
                                 float v) {
  return b.create<mlir::arith::ConstantOp>(loc, b.getF32Type(),
                                           b.getF32FloatAttr(v));
}

static inline mlir::Value
packF32BitsAsI64(mlir::OpBuilder &b, mlir::Location loc, mlir::Value f32Val) {
  mlir::Value i32Bits =
      b.create<mlir::arith::BitcastOp>(loc, b.getI32Type(), f32Val);
  return b.create<mlir::arith::ExtUIOp>(loc, b.getI64Type(), i32Bits);
}

// Abs-max over memref<?x?>f32 with static rows/cols.
static inline mlir::Value absMaxF32(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value mem, int64_t rows,
                                    int64_t cols) {
  auto maxTy = mlir::MemRefType::get({1}, b.getF32Type());
  mlir::Value maxBuf = b.create<mlir::memref::AllocOp>(loc, maxTy);
  mlir::Value zero = b.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value one = b.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value rowsV = b.create<mlir::arith::ConstantIndexOp>(loc, rows);
  mlir::Value colsV = b.create<mlir::arith::ConstantIndexOp>(loc, cols);
  b.create<mlir::memref::StoreOp>(loc, cstF32(b, loc, 0.0f), maxBuf,
                                  mlir::ValueRange{zero});
  auto rL = b.create<mlir::scf::ForOp>(loc, zero, rowsV, one);
  b.setInsertionPointToStart(rL.getBody());
  auto cL = b.create<mlir::scf::ForOp>(loc, zero, colsV, one);
  b.setInsertionPointToStart(cL.getBody());
  mlir::Value elem = b.create<mlir::memref::LoadOp>(
      loc, mem, mlir::ValueRange{rL.getInductionVar(), cL.getInductionVar()});
  mlir::Value neg = b.create<mlir::arith::NegFOp>(loc, elem);
  mlir::Value abs = b.create<mlir::arith::MaximumFOp>(loc, elem, neg);
  mlir::Value cur =
      b.create<mlir::memref::LoadOp>(loc, maxBuf, mlir::ValueRange{zero});
  b.create<mlir::memref::StoreOp>(
      loc, b.create<mlir::arith::MaximumFOp>(loc, cur, abs), maxBuf,
      mlir::ValueRange{zero});
  b.setInsertionPointAfter(rL);
  mlir::Value result =
      b.create<mlir::memref::LoadOp>(loc, maxBuf, mlir::ValueRange{zero});
  b.create<mlir::memref::DeallocOp>(loc, maxBuf);
  return result;
}

// Quant scale = 127/maxAbs (or 1 if maxAbs==0).
static inline mlir::Value quantScale(mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::Value maxAbs) {
  mlir::Value zero = cstF32(b, loc, 0.0f);
  mlir::Value one = cstF32(b, loc, 1.0f);
  mlir::Value qmax = cstF32(b, loc, 127.0f);
  mlir::Value has = b.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OGT, maxAbs, zero);
  mlir::Value scaled = b.create<mlir::arith::DivFOp>(loc, qmax, maxAbs);
  return b.create<mlir::arith::SelectOp>(loc, has, scaled, one);
}

static inline mlir::Value dequantScale(mlir::OpBuilder &b, mlir::Location loc,
                                       mlir::Value scaleA, mlir::Value scaleB) {
  mlir::Value one = cstF32(b, loc, 1.0f);
  mlir::Value prod = b.create<mlir::arith::MulFOp>(loc, scaleA, scaleB);
  return b.create<mlir::arith::DivFOp>(loc, one, prod);
}

} // namespace buckyball
} // namespace buddy

#endif // BUCKYBALL_CONVERSION_QUANTUTILS_H
