//===- LowerChannelOps.cpp - Per-channel INT8 operations and transpose ---===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "int8mulOps.h"
#include "lutOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "transposeOps.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult pebble::lowerChannelOp(::buddy::acc::CustomOp op, Value &result) {
  bool lookup = op.getKind() == "lookup_i8";
  auto type = cast<RankedTensorType>(op.getInputs()[lookup ? 0 : 1].getType());
  auto otherType =
      cast<RankedTensorType>(op.getInputs()[lookup ? 1 : 0].getType());
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isInteger(8) ||
      !otherType.getElementType().isInteger(8) ||
      op.getResult(0).getType() != type ||
      (lookup ? (otherType.getRank() != 1 ||
                 (otherType.getDimSize(0) != 256 &&
                  otherType.getDimSize(0) != 4096 &&
                  otherType.getDimSize(0) !=
                      ((type.getDimSize(1) + 15) / 16) * 4096))
              : otherType.getShape() !=
                    ArrayRef<int64_t>({1, type.getDimSize(1)})))
    return op.emitOpError(
        "requires an INT8 matrix and an INT8 table or per-channel gate");
  int64_t rows = type.getDimSize(0), channels = type.getDimSize(1);
  Value storage = op.getInputs()[lookup ? 0 : 1];
  while (auto slice = storage.getDefiningOp<tensor::ExtractSliceOp>())
    storage = slice.getSource();
  int64_t storageChannels =
      cast<RankedTensorType>(storage.getType()).getShape().back();
  // A strided DMA may touch one extra 4-KiB base page at an unaligned start.
  // Bound both source and destination by the prepared translation capacity.
  int64_t capacity =
      std::min(target::getTarget().bankDepth,
               (target::getTarget().dmaMaxRanges - 1) * 4096 / storageChannels);
  if (capacity <= 0)
    return op.emitOpError(
        "channel stride exceeds the prepared DMA range capacity");
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](OpBuilder &b, int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(b, 0), one = index(b, 1);
  Type i64 = b.getI64Type();
  Value input = bufferization::ToBufferOp::create(
      b, loc,
      MemRefType::get(type.getShape(), b.getI8Type(),
                      StridedLayoutAttr::get(b.getContext(),
                                             ShapedType::kDynamic,
                                             {ShapedType::kDynamic, 1})),
      op.getInputs()[lookup ? 0 : 1], true);
  Value inputStride =
      memref::ExtractStridedMetadataOp::create(b, loc, input).getStrides()[0];
  Value other = tensorBuffer(b, loc, op.getInputs()[lookup ? 1 : 0]);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(type.getShape(), b.getI8Type()));
  Value ratio;
  if (!lookup)
    ratio = arith::ConstantOp::create(b, loc,
                                      cast<FloatAttr>(op->getAttr("ratio")));
  pebble::repeat(
      b, loc, 0, channels, 16, false, {},
      [&](OpBuilder &b, Value c, ValueRange) {
        Value lanesEnd = arith::MinSIOp::create(
            b, loc, index(b, 16),
            arith::SubIOp::create(b, loc, index(b, channels), c));
        Value tableBank, gateBank;
        if (lookup) {
          int64_t entries = otherType.getDimSize(0);
          int64_t groups = entries == 256 ? 1 : 4;
          int64_t bytes = entries == 256 ? 256 : 1024;
          tableBank = BankAllocOp::create(b, loc, i64, 1, groups);
          for (int64_t group = 0; group < groups; ++group) {
            Value offset = index(b, group * bytes);
            if (entries > 4096)
              offset = arith::AddIOp::create(
                  b, loc, offset,
                  arith::MulIOp::create(b, loc, c, index(b, 256)));
            Value table = memref::SubViewOp::create(
                b, loc, other, ArrayRef<OpFoldResult>{offset},
                ArrayRef<OpFoldResult>{b.getIndexAttr(bytes)},
                ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
            tableBank = BankMvinGroupOp::create(b, loc, i64, table, tableBank,
                                                group, bytes / 16, 1);
          }
        } else {
          Value gate = memref::SubViewOp::create(
              b, loc, other, ArrayRef<OpFoldResult>{zero, c},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          gateBank = BankAllocOp::create(b, loc, i64, 1, 1);
          gateBank = BankMvinStridedOp::create(
              b, loc, i64, gate, gateBank, zero, one, index(b, channels), 0, 1,
              lanesEnd, arith::ConstantIndexOp::create(b, loc, 0));
        }
        // Keep all output row blocks in one loop, including the final tail.
        auto chunk = [&](OpBuilder &b, Value start, int64_t count) {
          Value memory = memref::SubViewOp::create(
              b, loc, input, ArrayRef<OpFoldResult>{start, c},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          Value source = BankAllocOp::create(b, loc, i64, 1, 1);
          source = BankMvinStridedOp::create(
              b, loc, i64, memory, source, zero, index(b, count), inputStride,
              0, 1, lanesEnd, arith::ConstantIndexOp::create(b, loc, 0));
          Value destination = BankAllocOp::create(b, loc, i64, 1, 1);
          destination =
              lookup ? Value(::buddy::lut_ball::BankLookupI8Op::create(
                           b, loc, i64, source, tableBank, destination, count))
                     : Value(::buddy::int8mul_ball::BankGateMulI8Op::create(
                           b, loc, i64, gateBank, source, destination, ratio,
                           count));
          Value target = memref::SubViewOp::create(
              b, loc, output, ArrayRef<OpFoldResult>{start, c},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)},
              ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1)});
          BankMvoutStridedOp::create(b, loc, i64, target, destination, zero,
                                     index(b, count), index(b, channels), 0, 1,
                                     lanesEnd,
                                     arith::ConstantIndexOp::create(b, loc, 0));
          BankReleaseOp::create(b, loc, source);
          BankReleaseOp::create(b, loc, destination);
        };
        int64_t full = rows / capacity * capacity;
        pebble::repeat(b, loc, 0, rows, capacity, false, {},
                       [&](OpBuilder &b, Value start, ValueRange) {
                         if (rows == full) {
                           chunk(b, start, capacity);
                         } else {
                           Value isFull = arith::CmpIOp::create(
                               b, loc, arith::CmpIPredicate::ult, start,
                               index(b, full));
                           scf::IfOp::create(
                               b, loc, isFull,
                               [&](OpBuilder &b, Location) {
                                 chunk(b, start, capacity);
                                 scf::YieldOp::create(b, loc);
                               },
                               [&](OpBuilder &b, Location) {
                                 chunk(b, start, rows - full);
                                 scf::YieldOp::create(b, loc);
                               });
                         }
                         return SmallVector<Value>{};
                       });
        BankReleaseOp::create(b, loc, lookup ? tableBank : gateBank);
        return SmallVector<Value>{};
      });
  result = bufferization::ToTensorOp::create(b, loc, type, output, true, true);
  return success();
}

LogicalResult pebble::lowerTranspose(::buddy::acc::CustomOp op, Value &result) {
  auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
  int64_t rank = type.getRank();
  if (!type.hasStaticShape() || rank < 2 || !type.getElementType().isInteger(8))
    return op.emitOpError("transpose_i8 requires static INT8 matrices");
  SmallVector<int64_t> shape(type.getShape());
  std::swap(shape[rank - 2], shape[rank - 1]);
  if (outputType.getShape() != ArrayRef<int64_t>(shape))
    return op.emitOpError("transpose_i8 swaps the last two axes");
  int64_t rows = type.getDimSize(rank - 2), columns = type.getDimSize(rank - 1);
  const auto &target = target::getTarget();
  if (rows > target.bankDepth || columns % 16 ||
      rows * columns > (target.dmaMaxRanges - 1) * 4096)
    return op.emitOpError("transpose_i8 exceeds bank or DMA geometry");
  int64_t tileColumns = std::min<int64_t>(31, (target.bankNum - 1) / 2) * 16;
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](OpBuilder &b, int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  Value zero = index(b, 0);
  SmallVector<int64_t> strides(rank, ShapedType::kDynamic);
  strides.back() = 1;
  Value input = bufferization::ToBufferOp::create(
      b, loc,
      MemRefType::get(type.getShape(), b.getI8Type(),
                      StridedLayoutAttr::get(b.getContext(),
                                             ShapedType::kDynamic, strides)),
      op.getInputs()[0], true);
  Value inputStride = memref::ExtractStridedMetadataOp::create(b, loc, input)
                          .getStrides()[rank - 2];
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(outputType.getShape(), b.getI8Type()));
  {
    OpBuilder::InsertionGuard guard(b);
    SmallVector<OpFoldResult> prefix;
    for (int64_t axis = 0; axis < rank - 2; ++axis) {
      auto loop = scf::ForOp::create(
          b, loc, zero, index(b, type.getDimSize(axis)), index(b, 1));
      b.setInsertionPointToStart(loop.getBody());
      prefix.push_back(loop.getInductionVar());
    }
    auto tile = [&](OpBuilder &b, Value column, int64_t count) {
      int64_t groups = count / 16;
      Value source = BankAllocOp::create(b, loc, b.getI64Type(), 1, groups);
      for (int64_t group = 0; group < groups; ++group) {
        SmallVector<OpFoldResult> at(prefix);
        at.push_back(zero);
        at.push_back(arith::AddIOp::create(b, loc, column, index(b, group * 16))
                         .getResult());
        Value view = memref::SubViewOp::create(
            b, loc, input, at,
            SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
            SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
        source = BankMvinStridedOp::create(b, loc, b.getI64Type(), view, source,
                                           zero, index(b, rows), inputStride,
                                           group, 1, index(b, 16), zero);
      }
      Value destination =
          BankAllocOp::create(b, loc, b.getI64Type(), 1, groups);
      destination = ::buddy::transpose_ball::BankTransposeI8Op::create(
          b, loc, b.getI64Type(), source, destination, rows, count);
      SmallVector<OpFoldResult> at(prefix);
      at.append({column, zero});
      Value view = memref::SubViewOp::create(
          b, loc, output, at,
          SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)),
          SmallVector<OpFoldResult>(rank, b.getIndexAttr(1)));
      BankMvoutOp::create(b, loc, b.getI64Type(), view, destination, rows, 1);
      BankReleaseOp::create(b, loc, source);
      BankReleaseOp::create(b, loc, destination);
    };
    int64_t full = columns / tileColumns * tileColumns;
    repeat(b, loc, 0, full, tileColumns, false, {},
           [&](OpBuilder &b, Value column, ValueRange) {
             tile(b, column, tileColumns);
             return SmallVector<Value>{};
           });
    if (full != columns)
      tile(b, index(b, full), columns - full);
  }
  result =
      bufferization::ToTensorOp::create(b, loc, outputType, output, true, true);
  return success();
}
