#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mxint8unpackOps.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult lowerReducePackedDecode(ModuleOp module) {
  SmallVector<::buddy::frontend::MXINT8ToFP32Op> operations;
  module.walk(
      [&](::buddy::frontend::MXINT8ToFP32Op op) { operations.push_back(op); });
  for (auto op : operations) {
    auto type = op.getResult().getType();
    auto inputType = op.getInput().getType();
    int64_t rank = type.getRank(), width = type.getShape().back();
    int64_t packed = width + width / 32;
    if (int64_t(op.getAxis()) != rank - 1 || width <= 0 || width % 32 ||
        width / 32 > 128 || width / 4 > target::getTarget().bankDepth ||
        inputType.getShape().back() != packed)
      return op.emitError("packed decode requires aligned rows of at most 4096 "
                          "values on the last axis");
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t n) -> Value {
      return arith::ConstantIndexOp::create(b, loc, n);
    };
    Value rows = index(1);
    SmallVector<Value> dynamic;
    for (int64_t axis = 0; axis < rank - 1; ++axis) {
      Value size = tensor::DimOp::create(b, loc, op.getInput(), axis);
      rows = b.createOrFold<arith::MulIOp>(loc, rows, size);
      if (type.isDynamicDim(axis))
        dynamic.push_back(size);
    }
    Value input = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(inputType.getShape(), b.getI8Type()),
        op.getInput(), true);
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(type.getShape(), b.getF32Type()), dynamic);
    output.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
    auto flatten = [&](Value value) -> Value {
      auto memory = cast<MemRefType>(value.getType());
      if (memory.getRank() == 1)
        return value;
      ReassociationIndices axes;
      for (int64_t axis = 0; axis < memory.getRank(); ++axis)
        axes.push_back(axis);
      auto flat =
          MemRefType::get({memory.hasStaticShape() ? memory.getNumElements()
                                                   : ShapedType::kDynamic},
                          memory.getElementType());
      return memref::CollapseShapeOp::create(
          b, loc, flat, value, ArrayRef<ReassociationIndices>{axes});
    };
    Value source = flatten(input), destination = flatten(output);
    auto loop = scf::ForOp::create(b, loc, index(0), rows, index(1));
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loop.getBody());
      Value row = loop.getInductionVar();
      Value from = arith::MulIOp::create(b, loc, row, index(packed));
      Value to = arith::MulIOp::create(b, loc, row, index(width));
      auto window = [&](Value memory, Value offset, int64_t length) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{b.getIndexAttr(length)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value result = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      int64_t whole = packed / 16 * 16;
      a = BankMvinOp::create(b, loc, b.getI64Type(),
                             window(source, from, whole), a, whole / 16, 1);
      if (packed % 16) {
        Value tail = arith::AddIOp::create(b, loc, from, index(whole));
        a = BankMvinStridedOp::create(
            b, loc, b.getI64Type(), window(source, tail, packed % 16), a,
            index(whole / 16), index(1), index(packed % 16), 0, 1,
            index(packed % 16), index(0));
      }
      Value blocks = arith::ConstantIntOp::create(b, loc, width / 32, 64);
      result = ::buddy::mxint8unpack_ball::BankUnpackOp::create(
          b, loc, b.getI64Type(), a, result, blocks);
      result = BankMvoutOp::create(b, loc, b.getI64Type(),
                                   window(destination, to, width), result,
                                   width / 4, 1);
      BankReleaseOp::create(b, loc, a);
      BankReleaseOp::create(b, loc, result);
    }
    Value result =
        bufferization::ToTensorOp::create(b, loc, type, output, true, true);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}
