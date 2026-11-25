//===- LowerMaxPool.cpp - Exact maximum of nonnegative I32 activations
//-----===//
#include "Acc/AccOps.h"
#include "Lowering.h"
#include "Target/TargetRegistry.h"
#include "fixedOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult pebble::lowerMaxPool(CustomOp op, Value &result) {
  auto type = cast<RankedTensorType>(op.getInputs()[0].getType());
  auto outType = cast<RankedTensorType>(op.getResult(0).getType());
  int64_t kernel = cast<IntegerAttr>(op->getAttr("kernel")).getInt();
  int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
  auto padding = cast<DenseI64ArrayAttr>(op->getAttr("padding")).asArrayRef();
  int64_t pixels =
      std::min(outType.getDimSize(2), target::getTarget().bankDepth / 4);
  OpBuilder b(op);
  Location loc = op.getLoc();
  Type i64 = b.getI64Type();
  auto index = [&](int64_t n) -> Value {
    return arith::ConstantIndexOp::create(b, loc, n);
  };
  Value zero = index(0), one = index(1);
  Value input = bufferization::ToBufferOp::create(
      b, loc,
      MemRefType::get(type.getShape(), b.getI32Type(),
                      StridedLayoutAttr::get(
                          b.getContext(), ShapedType::kDynamic,
                          SmallVector<int64_t>(4, ShapedType::kDynamic))),
      op.getInputs()[0], true);
  Value output = memref::AllocOp::create(
      b, loc, MemRefType::get(outType.getShape(), b.getI32Type()));
  Value negative = arith::ConstantOp::create(
      b, loc,
      DenseIntElementsAttr::get(RankedTensorType::get({16}, b.getI32Type()),
                                int32_t(-(1 << 24))));
  Value minusOne = BankAllocOp::create(b, loc, i64, 1, 1);
  minusOne = BankMvinOp::create(b, loc, i64, tensorBuffer(b, loc, negative),
                                minusOne, 4, 1);
  Value memoryStride = arith::MulIOp::create(
      b, loc,
      memref::ExtractStridedMetadataOp::create(b, loc, input).getStrides()[2],
      index(4 * stride));
  auto channels =
      scf::ForOp::create(b, loc, zero, index(type.getDimSize(3)), index(16));
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(channels.getBody());
    Value c = channels.getInductionVar();
    auto ys =
        scf::ForOp::create(b, loc, zero, index(outType.getDimSize(1)), one);
    b.setInsertionPointToStart(ys.getBody());
    Value y = ys.getInductionVar();
    auto xs = scf::ForOp::create(b, loc, zero, index(outType.getDimSize(2)),
                                 index(pixels));
    b.setInsertionPointToStart(xs.getBody());
    Value x = xs.getInductionVar();
    Value count = arith::MinSIOp::create(
        b, loc, index(pixels),
        arith::SubIOp::create(b, loc, index(outType.getDimSize(2)), x));
    Value maximum;
    for (int64_t ky = 0; ky < kernel; ++ky)
      for (int64_t kx = 0; kx < kernel; ++kx) {
        Value iy = arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, y, index(stride)),
            index(ky - padding[0]));
        Value ix = arith::AddIOp::create(
            b, loc, arith::MulIOp::create(b, loc, x, index(stride)),
            index(kx - padding[1]));
        Value begin = arith::CeilDivUIOp::create(
            b, loc,
            arith::MaxSIOp::create(b, loc, zero,
                                   arith::SubIOp::create(b, loc, zero, ix)),
            index(stride));
        Value end = arith::MinSIOp::create(
            b, loc, count,
            arith::CeilDivUIOp::create(
                b, loc,
                arith::MaxSIOp::create(
                    b, loc, zero,
                    arith::SubIOp::create(b, loc, index(type.getDimSize(2)),
                                          ix)),
                index(stride)));
        Value valid = arith::AndIOp::create(
            b, loc,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, iy, zero),
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, iy,
                                  index(type.getDimSize(1))));
        valid = arith::AndIOp::create(
            b, loc, valid,
            arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, begin,
                                  end));
        Value data = BankAllocOp::create(b, loc, i64, 1, 1);
        data = BankClearOp::create(b, loc, i64, data, 1, 1);
        auto load = scf::IfOp::create(b, loc, TypeRange{i64}, valid, true);
        {
          OpBuilder::InsertionGuard branch(b);
          b.setInsertionPointToStart(&load.getThenRegion().front());
          Value loaded = data;
          Value start = arith::AddIOp::create(
              b, loc, ix, arith::MulIOp::create(b, loc, begin, index(stride)));
          Value memory = memref::SubViewOp::create(
              b, loc, input, ArrayRef<OpFoldResult>{zero, iy, start, c},
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
          Value first = arith::MulIOp::create(b, loc, begin, index(4));
          Value rows = arith::MulIOp::create(
              b, loc, index(4), arith::SubIOp::create(b, loc, end, begin));
          loaded = BankMvinStridedOp::create(b, loc, i64, memory, loaded, first,
                                             rows, memoryStride, 0, 4,
                                             index(16), index(2));
          scf::YieldOp::create(b, loc, loaded);
          b.setInsertionPointToStart(&load.getElseRegion().front());
          scf::YieldOp::create(b, loc, data);
        }
        data = load.getResult(0);
        if (!maximum) {
          maximum = data;
          continue;
        }
        // Nonnegative I32 values make a-b and relu(a-b)+b exact without
        // saturation. Multiplication by -Q supplies the exact negation.
        Value negated = BankAllocOp::create(b, loc, i64, 1, 1);
        negated = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, i64, data, minusOne, negated, pixels * 4, 3, 24, true,
            false);
        Value difference = BankAllocOp::create(b, loc, i64, 1, 1);
        difference = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, i64, maximum, negated, difference, pixels * 4, 5, 24, false,
            false);
        BankReleaseOp::create(b, loc, maximum);
        BankReleaseOp::create(b, loc, negated);
        Value positive = BankAllocOp::create(b, loc, i64, 1, 1);
        positive = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, i64, difference, minusOne, positive, pixels * 4, 4, 24,
            false, false);
        BankReleaseOp::create(b, loc, difference);
        maximum = BankAllocOp::create(b, loc, i64, 1, 1);
        maximum = ::buddy::fixed_ball::BankFixedOp::create(
            b, loc, i64, positive, data, maximum, pixels * 4, 5, 24, false,
            false);
        BankReleaseOp::create(b, loc, positive);
        BankReleaseOp::create(b, loc, data);
      }
    Value memory = memref::SubViewOp::create(
        b, loc, output, ArrayRef<OpFoldResult>{zero, y, x, c},
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)),
        SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
    maximum = BankMvoutStridedOp::create(
        b, loc, i64, memory, maximum, zero,
        arith::MulIOp::create(b, loc, count, index(4)),
        index(outType.getDimSize(3) * 4), 0, 4, index(16), index(2));
    BankReleaseOp::create(b, loc, maximum);
  }
  BankReleaseOp::create(b, loc, minusOne);
  result =
      bufferization::ToTensorOp::create(b, loc, outType, output, true, true);
  return success();
}
