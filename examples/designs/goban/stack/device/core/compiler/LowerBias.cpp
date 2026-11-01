//===- LowerBias.cpp - Goban bias decoding on the int2fp Ball -------------===//
#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "int2fpOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
using namespace mlir;
using namespace ::buddy::acc;
using namespace ::buddy::frontend;

LogicalResult lowerGobanBiasOutputs(ModuleOp module) {
  SmallVector<Int32ToFP32Op> ops;
  module.walk([&](Int32ToFP32Op op) { ops.push_back(op); });
  for (auto op : ops) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = op.getResult().getType();
    if (type.getRank() != 1 || !type.hasStaticShape() ||
        !op.getResult().hasOneUse())
      return op.emitError("goban bias decoding needs a static vector output");
    auto destination = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *op.getResult().getUsers().begin());
    if (!destination)
      return op.emitError("bias must write its core subgraph destination");
    auto c = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    int64_t count = type.getDimSize(0), padded = (count + 15) / 16 * 16;
    Value values = memref::AllocOp::create(
        b, loc, MemRefType::get({padded}, b.getI32Type()));
    Value scales = memref::AllocOp::create(
        b, loc, MemRefType::get({padded}, b.getF32Type()));
    linalg::FillOp::create(
        b, loc, ValueRange{arith::ConstantIntOp::create(b, loc, 0, 32)},
        values);
    linalg::FillOp::create(
        b, loc,
        ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(1))},
        scales);
    auto fill = scf::ForOp::create(b, loc, c(0), c(count), c(1));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(fill.getBody());
      Value i = fill.getInductionVar();
      Value scaleIndex =
          cast<RankedTensorType>(op.getStep().getType()).getDimSize(0) == 1
              ? c(0)
              : i;
      memref::StoreOp::create(
          b, loc, tensor::ExtractOp::create(b, loc, op.getInput(), i), values,
          i);
      memref::StoreOp::create(
          b, loc, tensor::ExtractOp::create(b, loc, op.getStep(), scaleIndex),
          scales, i);
    }
    Value final = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get({count}, b.getF32Type()),
        destination.getDest());
    Value output = count == padded
                       ? final
                       : memref::AllocOp::create(
                             b, loc, MemRefType::get({padded}, b.getF32Type()))
                             .getResult();
    auto loop = scf::ForOp::create(b, loc, c(0), c(padded / 16), c(1));
    loop->setAttr("core.stream", b.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value offset =
          arith::MulIOp::create(b, loc, loop.getInductionVar(), c(16));
      auto tile = [&](Value memory) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{b.getIndexAttr(16)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      Value v = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value s = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value out = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      v = BankMvinOp::create(b, loc, b.getI64Type(), tile(values), v, 4, 1);
      s = BankMvinOp::create(b, loc, b.getI64Type(), tile(scales), s, 4, 1);
      out = ::buddy::int2fp_ball::BankInt32ToFp32Op::create(
          b, loc, b.getI64Type(), v, s, out, 1, false);
      BankMvoutOp::create(b, loc, b.getI64Type(), tile(output), out, 4, 1);
      BankReleaseOp::create(b, loc, v);
      BankReleaseOp::create(b, loc, s);
      BankReleaseOp::create(b, loc, out);
    }
    if (count != padded) {
      Value logical = memref::SubViewOp::create(
          b, loc, output, ArrayRef<OpFoldResult>{b.getIndexAttr(0)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(count)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      auto copy = memref::CopyOp::create(b, loc, logical, final);
      copy->setAttr("core.after", b.getUnitAttr());
    }
    Value written =
        bufferization::ToTensorOp::create(b, loc, type, final, true, true);
    destination.getResult().replaceAllUsesWith(written);
    destination.erase();
    op.erase();
  }
  return success();
}
