//===- LowerPool.cpp - Overlapping quantized pooling tiles ----------------===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "lutOps.h"
#include "maxpoolOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "toint8Ops.h"
#include "transposeOps.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <limits>

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult pebble::lowerPool(CustomOp op, Value &result) {
  auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  bool direct = op.getKind() == "maxpool_i8";
  auto outputType =
      direct
          ? resultType
          : RankedTensorType::get(
                cast<DenseI64ArrayAttr>(op->getAttr("pool_shape")).asArrayRef(),
                resultType.getElementType());
  int64_t kernel = cast<IntegerAttr>(op->getAttr("kernel")).getInt();
  int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
  bool relu = !direct && cast<BoolAttr>(op->getAttr("relu")).getValue();
  auto padding = cast<DenseI64ArrayAttr>(op->getAttr("padding")).asArrayRef();
  int64_t maxSide = std::min<int64_t>(
      15, std::sqrt(target::getTarget().bankDepth / (direct ? 1 : 4)));
  int64_t side = std::min({(maxSide - kernel) / stride + 1,
                           outputType.getDimSize(1), outputType.getDimSize(2)});
  if (side <= 0)
    return op.emitOpError("pooling window exceeds the input bank capacity");
  if (direct) {
    int64_t best = std::numeric_limits<int64_t>::max();
    for (int64_t candidate = side; candidate > 0; --candidate) {
      int64_t input = (candidate - 1) * stride + kernel;
      int64_t tiles = llvm::divideCeil(outputType.getDimSize(1), candidate) *
                      llvm::divideCeil(outputType.getDimSize(2), candidate);
      int64_t cost =
          tiles * (candidate * candidate * kernel * kernel + input * input);
      if (cost < best) {
        best = cost;
        side = candidate;
      }
    }
  }
  int64_t inputSide = (side - 1) * stride + kernel;
  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i64 = b.getI64Type();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Value input = bufferization::ToBufferOp::create(
      b, loc,
      MemRefType::get(type.getShape(), type.getElementType(),
                      StridedLayoutAttr::get(
                          b.getContext(), ShapedType::kDynamic,
                          SmallVector<int64_t>(4, ShapedType::kDynamic))),
      op.getInputs()[0], true);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(outputType.getShape(), b.getI8Type()));
  Value multipliers, paddingMemory;
  if (direct) {
    Value fill = arith::ConstantOp::create(
        b, loc,
        DenseElementsAttr::get(
            RankedTensorType::get({inputSide * inputSide, 16}, b.getI8Type()),
            b.getIntegerAttr(b.getI8Type(), -128)));
    paddingMemory = tensorBuffer(b, loc, fill);
  } else {
    multipliers = tensorBuffer(b, loc, op.getInputs()[1]);
  }
  Value pixelStride = arith::MulIOp::create(
      b, loc,
      memref::ExtractStridedMetadataOp::create(b, loc, input).getStrides()[2],
      index(direct ? 1 : 4));
  Value table;
  if (relu) {
    SmallVector<int8_t> entries(256);
    for (int i = 0; i < 128; ++i)
      entries[i] = i;
    Value data = arith::ConstantOp::create(
        b, loc,
        DenseIntElementsAttr::get(RankedTensorType::get({256}, b.getI8Type()),
                                  ArrayRef<int8_t>(entries)));
    table = BankAllocOp::create(b, loc, i64, 1, 1);
    table = BankMvinOp::create(b, loc, i64, tensorBuffer(b, loc, data), table,
                               16, 1);
  }
  auto channels =
      scf::ForOp::create(b, loc, zero, index(type.getDimSize(3)), index(16));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(channels.getBody());
    Value c = channels.getInductionVar();
    Value scales;
    if (!direct) {
      Value factors = memref::SubViewOp::create(
          b, loc, multipliers, ArrayRef<OpFoldResult>{zero, zero, zero, c},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                 b.getIndexAttr(1), b.getIndexAttr(16)},
          SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
      scales = BankAllocOp::create(b, loc, i64, 1, 1);
      scales = BankMvinOp::create(b, loc, i64, factors, scales, 4, 1);
    }
    auto ys = scf::ForOp::create(b, loc, zero, index(outputType.getDimSize(1)),
                                 index(side));
    b.setInsertionPointToStart(ys.getBody());
    Value y = ys.getInductionVar();
    auto xs = scf::ForOp::create(b, loc, zero, index(outputType.getDimSize(2)),
                                 index(side));
    b.setInsertionPointToStart(xs.getBody());
    Value x = xs.getInductionVar();
    Value iy = arith::SubIOp::create(
        b, loc, arith::MulIOp::create(b, loc, y, index(stride)),
        index(padding[0]));
    Value ix = arith::SubIOp::create(
        b, loc, arith::MulIOp::create(b, loc, x, index(stride)),
        index(padding[1]));
    Value yBegin = arith::MaxSIOp::create(
        b, loc, zero, arith::SubIOp::create(b, loc, zero, iy));
    Value xBegin = arith::MaxSIOp::create(
        b, loc, zero, arith::SubIOp::create(b, loc, zero, ix));
    Value height = arith::MinSIOp::create(
        b, loc, index(inputSide),
        arith::SubIOp::create(b, loc, index(type.getDimSize(1)), iy));
    Value width = arith::MinSIOp::create(
        b, loc, index(inputSide),
        arith::SubIOp::create(b, loc, index(type.getDimSize(2)), ix));
    Value values = BankAllocOp::create(b, loc, i64, 1, 1);
    values = direct
                 ? Value(BankMvinOp::create(b, loc, i64, paddingMemory, values,
                                            inputSide * inputSide, 1))
                 : Value(BankClearOp::create(b, loc, i64, values, 1, 1));
    width = arith::SubIOp::create(b, loc, width, xBegin);
    auto rows =
        scf::ForOp::create(b, loc, yBegin, height, one, ValueRange{values});
    b.setInsertionPointToStart(rows.getBody());
    Value row = rows.getInductionVar();
    Value bank = rows.getRegionIterArgs()[0];
    Value memory = memref::SubViewOp::create(
        b, loc, input,
        ArrayRef<OpFoldResult>{
            zero, arith::AddIOp::create(b, loc, iy, row).getResult(),
            arith::AddIOp::create(b, loc, ix, xBegin).getResult(), c},
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
    Value line = arith::MulIOp::create(
        b, loc, index(direct ? 1 : 4),
        arith::AddIOp::create(
            b, loc, xBegin,
            arith::MulIOp::create(b, loc, row, index(inputSide))));
    bank = BankMvinStridedOp::create(
        b, loc, i64, memory, bank, line,
        arith::MulIOp::create(b, loc, width, index(direct ? 1 : 4)),
        pixelStride, 0, direct ? 1 : 4, index(16), index(direct ? 0 : 2));
    scf::YieldOp::create(b, loc, bank);
    b.setInsertionPointAfter(rows);
    values = rows.getResult(0);
    Value codes = values;
    if (!direct) {
      codes = BankAllocOp::create(b, loc, i64, 1, 1);
      codes = type.getElementType().isF32()
                  ? Value(::buddy::toint8_ball::BankFp32ToInt8Op::create(
                        b, loc, i64, values, scales, codes,
                        inputSide * inputSide * 4))
                  : Value(::buddy::toint8_ball::BankInt32ToInt8Op::create(
                        b, loc, i64, values, scales, codes,
                        inputSide * inputSide, false));
      BankReleaseOp::create(b, loc, values);
    }
    Value pooled = BankAllocOp::create(b, loc, i64, 1, 1);
    pooled = ::buddy::maxpool_ball::BankMaxpoolI8Op::create(
        b, loc, i64, codes, pooled, inputSide, kernel, stride, 0);
    BankReleaseOp::create(b, loc, codes);
    if (relu) {
      Value clamped = BankAllocOp::create(b, loc, i64, 1, 1);
      clamped = ::buddy::lut_ball::BankLookupI8Op::create(
          b, loc, i64, pooled, table, clamped, side * side);
      BankReleaseOp::create(b, loc, pooled);
      pooled = clamped;
    }
    height = arith::MinSIOp::create(
        b, loc, index(side),
        arith::SubIOp::create(b, loc, index(outputType.getDimSize(1)), y));
    width = arith::MinSIOp::create(
        b, loc, index(side),
        arith::SubIOp::create(b, loc, index(outputType.getDimSize(2)), x));
    auto stores =
        scf::ForOp::create(b, loc, zero, height, one, ValueRange{pooled});
    b.setInsertionPointToStart(stores.getBody());
    row = stores.getInductionVar();
    memory = memref::SubViewOp::create(
        b, loc, output,
        ArrayRef<OpFoldResult>{
            zero, arith::AddIOp::create(b, loc, y, row).getResult(), x, c},
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
    Value stored = BankMvoutStridedOp::create(
        b, loc, i64, memory, stores.getRegionIterArgs()[0],
        arith::MulIOp::create(b, loc, row, index(side)), width,
        index(outputType.getDimSize(3)), 0, 1, index(16),
        arith::ConstantIndexOp::create(b, loc, 0));
    scf::YieldOp::create(b, loc, stored);
    b.setInsertionPointAfter(stores);
    BankReleaseOp::create(b, loc, stores.getResult(0));
    b.setInsertionPointAfter(ys);
    if (scales)
      BankReleaseOp::create(b, loc, scales);
  }
  if (table)
    BankReleaseOp::create(b, loc, table);
  if (resultType.getRank() == 2) {
    auto shape = outputType.getShape();
    int64_t pixels = shape[1] * shape[2];
    if (pixels > target::getTarget().bankDepth)
      return op.emitOpError(
          "flattened pooling image exceeds one transpose bank");
    Value flat = memref::AllocOp::create(
        b, loc, MemRefType::get(resultType.getShape(), b.getI8Type()));
    auto blocks = scf::ForOp::create(b, loc, zero, index(shape[3]), index(16));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(blocks.getBody());
      Value c = blocks.getInductionVar();
      Value source = memref::SubViewOp::create(
          b, loc, output, ArrayRef<OpFoldResult>{zero, zero, zero, c},
          SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
          SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
      Value bank = BankAllocOp::create(b, loc, i64, 1, 1);
      bank = BankMvinStridedOp::create(b, loc, i64, source, bank, zero,
                                       index(pixels), index(shape[3]), 0, 1,
                                       index(16), zero);
      Value transposed = BankAllocOp::create(b, loc, i64, 1, 1);
      transposed = ::buddy::transpose_ball::BankTransposeI8Op::create(
          b, loc, i64, bank, transposed, pixels, 16);
      BankReleaseOp::create(b, loc, bank);
      Value offset = arith::MulIOp::create(b, loc, c, index(pixels));
      Value destination = memref::SubViewOp::create(
          b, loc, flat, ArrayRef<OpFoldResult>{zero, offset},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1),
                                 b.getIndexAttr(pixels * 16)},
          SmallVector<OpFoldResult>(2, b.getIndexAttr(1)));
      transposed =
          BankMvoutOp::create(b, loc, i64, destination, transposed, pixels, 1);
      BankReleaseOp::create(b, loc, transposed);
    }
    output = flat;
  }
  result =
      bufferization::ToTensorOp::create(b, loc, resultType, output, true, true);
  return success();
}
