//===- LowerChannelOps.cpp - Per-channel INT8 operations and transpose ---===//
#include "Buckyball/BuckyballOps.h"
#include "Lowering.h"
#include "Target/BuckyballTargetRegistry.h"
#include "int8mulOps.h"
#include "lutOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "transposeOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult pebble::lowerChannelOp(::buddy::custom::CustomOp op,
                                     Value &result) {
  bool lookup = op.getKind() == "lookup_i8";
  auto type = cast<RankedTensorType>(op.getInputs()[lookup ? 0 : 1].getType());
  auto otherType =
      cast<RankedTensorType>(op.getInputs()[lookup ? 1 : 0].getType());
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isInteger(8) ||
      !otherType.getElementType().isInteger(8) ||
      op.getResult(0).getType() != type ||
      (lookup ? (otherType.getRank() != 1 || (otherType.getDimSize(0) != 256 &&
                                              otherType.getDimSize(0) != 4096))
              : otherType.getShape() !=
                    ArrayRef<int64_t>({1, type.getDimSize(1)})))
    return op.emitOpError(
        "requires an INT8 matrix and an INT8 table or per-channel gate");
  int64_t rows = type.getDimSize(0), channels = type.getDimSize(1);
  int64_t capacity = buckyball_target::getBuckyballTarget().bankDepth;
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto index = [&](OpBuilder &b, int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(b, 0), one = index(b, 1);
  Type i64 = b.getI64Type();
  Value input = tensorBuffer(b, loc, op.getInputs()[lookup ? 0 : 1]);
  Value other = tensorBuffer(b, loc, op.getInputs()[lookup ? 1 : 0]);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(type.getShape(), b.getI8Type()));
  Value izero = arith::ConstantIntOp::create(b, loc, 0, 8);
  Value tableBank, ratio;
  if (lookup) {
    int64_t columns = otherType.getDimSize(0) == 256 ? 1 : 4;
    if (columns == 4) {
      Value packed = memref::AllocOp::create(
          b, loc, MemRefType::get({64, 64}, b.getI8Type()));
      auto rows = scf::ForOp::create(b, loc, zero, index(b, 64), one);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToStart(rows.getBody());
        Value row = rows.getInductionVar();
        auto cols = scf::ForOp::create(b, loc, zero, index(b, 64), one);
        b.setInsertionPointToStart(cols.getBody());
        Value col = cols.getInductionVar();
        Value channel = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(
                b, loc, arith::DivUIOp::create(b, loc, col, index(b, 16)),
                index(b, 4)),
            arith::DivUIOp::create(b, loc, row, index(b, 16)));
        Value entry = arith::AddIOp::create(
            b, loc,
            arith::MulIOp::create(
                b, loc, arith::RemUIOp::create(b, loc, row, index(b, 16)),
                index(b, 16)),
            arith::RemUIOp::create(b, loc, col, index(b, 16)));
        Value offset = arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, channel, index(b, 256)),
            entry);
        Value value = memref::LoadOp::create(b, loc, other, offset);
        memref::StoreOp::create(b, loc, value, packed, ValueRange{row, col});
      }
      other = packed;
    }
    tableBank = BankAllocOp::create(b, loc, i64, 1, columns);
    tableBank = BankMvinOp::create(b, loc, i64, other, tableBank,
                                   otherType.getDimSize(0) / columns / 16, 1);
  } else {
    ratio = arith::ConstantOp::create(b, loc,
                                      cast<FloatAttr>(op->getAttr("ratio")));
  }
  pebble::repeat(
      b, loc, 0, channels, 16, false, {},
      [&](OpBuilder &b, Value c, ValueRange) {
        Value lanesEnd = arith::MinSIOp::create(
            b, loc, index(b, 16),
            arith::SubIOp::create(b, loc, index(b, channels), c));
        Value gateBank;
        if (!lookup) {
          Value gate = memref::AllocOp::create(
              b, loc, MemRefType::get({16}, b.getI8Type()));
          linalg::FillOp::create(b, loc, izero, gate);
          auto lanes = scf::ForOp::create(b, loc, zero, lanesEnd, one);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(lanes.getBody());
            Value lane = lanes.getInductionVar();
            Value value = memref::LoadOp::create(
                b, loc, other,
                ValueRange{zero, arith::AddIOp::create(b, loc, lane, c)});
            memref::StoreOp::create(b, loc, value, gate, lane);
          }
          gateBank = BankAllocOp::create(b, loc, i64, 1, 1);
          gateBank = BankMvinOp::create(b, loc, i64, gate, gateBank, 1, 1);
        }
        // Full chunks of `capacity` rows share one loop; the remaining rows
        // follow.
        auto chunk = [&](OpBuilder &b, Value start, int64_t count) {
          Value pack = memref::AllocOp::create(
              b, loc, MemRefType::get({count, 16}, b.getI8Type()));
          linalg::FillOp::create(b, loc, izero, pack);
          auto spatial = scf::ForOp::create(b, loc, zero, index(b, count), one);
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(spatial.getBody());
            Value r = spatial.getInductionVar();
            auto lanes = scf::ForOp::create(b, loc, zero, lanesEnd, one);
            b.setInsertionPointToStart(lanes.getBody());
            Value lane = lanes.getInductionVar();
            Value value = memref::LoadOp::create(
                b, loc, input,
                ValueRange{arith::AddIOp::create(b, loc, r, start),
                           arith::AddIOp::create(b, loc, lane, c)});
            memref::StoreOp::create(b, loc, value, pack, ValueRange{r, lane});
          }
          Value source = BankAllocOp::create(b, loc, i64, 1, 1);
          source = BankMvinOp::create(b, loc, i64, pack, source, count, 1);
          Value destination = BankAllocOp::create(b, loc, i64, 1, 1);
          destination =
              lookup ? Value(::buddy::lut_ball::BankLookupI8Op::create(
                           b, loc, i64, source, tableBank, destination, count))
                     : Value(::buddy::int8mul_ball::BankGateMulI8Op::create(
                           b, loc, i64, gateBank, source, destination, ratio,
                           count));
          BankMvoutOp::create(b, loc, i64, pack, destination, count, 1);
          BankReleaseOp::create(b, loc, source);
          BankReleaseOp::create(b, loc, destination);
          auto outputRows =
              scf::ForOp::create(b, loc, zero, index(b, count), one);
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(outputRows.getBody());
          Value r = outputRows.getInductionVar();
          auto lanes = scf::ForOp::create(b, loc, zero, lanesEnd, one);
          b.setInsertionPointToStart(lanes.getBody());
          Value lane = lanes.getInductionVar();
          Value value =
              memref::LoadOp::create(b, loc, pack, ValueRange{r, lane});
          memref::StoreOp::create(
              b, loc, value, output,
              ValueRange{arith::AddIOp::create(b, loc, r, start),
                         arith::AddIOp::create(b, loc, lane, c)});
        };
        int64_t full = rows / capacity * capacity;
        pebble::repeat(b, loc, 0, full, capacity, false, {},
                       [&](OpBuilder &b, Value start, ValueRange) {
                         chunk(b, start, capacity);
                         return SmallVector<Value>{};
                       });
        if (rows > full)
          chunk(b, index(b, full), rows - full);
        if (!lookup)
          BankReleaseOp::create(b, loc, gateBank);
        return SmallVector<Value>{};
      });
  if (lookup)
    BankReleaseOp::create(b, loc, tableBank);
  result = bufferization::ToTensorOp::create(b, loc, type, output, true, true);
  return success();
}

LogicalResult pebble::lowerTranspose(::buddy::custom::CustomOp op,
                                     Value &result) {
  auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto outputType = cast<RankedTensorType>(op.getResult(0).getType());
  if (!type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isInteger(8) ||
      outputType.getShape() !=
          ArrayRef<int64_t>({type.getDimSize(1), type.getDimSize(0)}))
    return op.emitOpError("transpose_i8 requires a static INT8 matrix");
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value input = tensorBuffer(b, loc, op.getInputs()[0]);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(outputType.getShape(), b.getI8Type()));
  Value source =
      BankAllocOp::create(b, loc, b.getI64Type(), 1, type.getDimSize(1) / 16);
  Value destination =
      BankAllocOp::create(b, loc, b.getI64Type(), 1, type.getDimSize(1) / 16);
  source = BankMvinOp::create(b, loc, b.getI64Type(), input, source,
                              type.getDimSize(0), 1);
  destination = ::buddy::transpose_ball::BankTransposeI8Op::create(
      b, loc, b.getI64Type(), source, destination, type.getDimSize(0),
      type.getDimSize(1));
  BankMvoutOp::create(b, loc, b.getI64Type(), output, destination,
                      type.getDimSize(0), 1);
  BankReleaseOp::create(b, loc, source);
  BankReleaseOp::create(b, loc, destination);
  result =
      bufferization::ToTensorOp::create(b, loc, outputType, output, true, true);
  return success();
}
