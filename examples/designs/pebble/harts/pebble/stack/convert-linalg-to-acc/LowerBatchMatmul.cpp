//===- LowerBatchMatmul.cpp - Batched integer matrix products ------------===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

LogicalResult pebble::lowerBatchMatmul(::buddy::acc::CustomOp op,
                                       Value &result) {
  auto a = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto w = cast<RankedTensorType>(op.getInputs()[1].getType());
  auto out = cast<RankedTensorType>(op.getResult(0).getType());
  if (!a.hasStaticShape() || !w.hasStaticShape() || a.getRank() != 3 ||
      w.getRank() != 3 || !a.getElementType().isInteger(8) ||
      !w.getElementType().isInteger(8) || !out.getElementType().isInteger(32) ||
      a.getDimSize(0) != w.getDimSize(0) ||
      a.getDimSize(2) != w.getDimSize(1) || w.getDimSize(2) % 16 ||
      out.getShape() !=
          ArrayRef<int64_t>{a.getDimSize(0), a.getDimSize(1), w.getDimSize(2)})
    return op.emitOpError(
        "requires BxMxK/BxKxN INT8 and BxMxN INT32, N divisible by 16");
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value input = tensorBuffer(b, loc, op.getInputs()[0]);
  Value weight = tensorBuffer(b, loc, op.getInputs()[1]);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(out.getShape(), b.getI32Type()));
  Value bias = memref::AllocOp::create(
      b, loc, MemRefType::get({1, w.getDimSize(2)}, b.getI32Type()));
  linalg::FillOp::create(
      b, loc, arith::ConstantIntOp::create(b, loc, 0, 32).getResult(), bias);
  repeat(b, loc, 0, a.getDimSize(0), 1, false, {},
         [&](OpBuilder &b, Value batch, ValueRange) {
           SmallVector<Value> matrices;
           for (Value memory : {input, weight, output}) {
             auto type = cast<MemRefType>(memory.getType());
             SmallVector<OpFoldResult> sizes{
                 b.getIndexAttr(1), b.getIndexAttr(type.getDimSize(1)),
                 b.getIndexAttr(type.getDimSize(2))};
             auto viewType = memref::SubViewOp::inferRankReducedResultType(
                 ArrayRef<int64_t>{type.getDimSize(1), type.getDimSize(2)},
                 type,
                 ArrayRef<OpFoldResult>{batch, b.getIndexAttr(0),
                                        b.getIndexAttr(0)},
                 sizes, SmallVector<OpFoldResult>(3, b.getIndexAttr(1)));
             matrices.push_back(memref::SubViewOp::create(
                 b, loc, cast<MemRefType>(viewType), memory,
                 ArrayRef<OpFoldResult>{batch, b.getIndexAttr(0),
                                        b.getIndexAttr(0)},
                 sizes, SmallVector<OpFoldResult>(3, b.getIndexAttr(1))));
           }
           linearTiles(b, loc, matrices[0], Value(), 0, Value(), matrices[1],
                       bias, Value(), false, matrices[2]);
           return SmallVector<Value>{};
         });
  result = bufferization::ToTensorOp::create(b, loc, out, output, true, true);
  return success();
}
