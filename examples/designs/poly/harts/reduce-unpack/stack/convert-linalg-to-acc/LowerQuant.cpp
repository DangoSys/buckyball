#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mxint8quantizeOps.h"

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult lowerReduceQuantizedRows(ModuleOp module) {
  SmallVector<::buddy::frontend::FP32ToMXINT8Op> operations;
  module.walk([&](::buddy::frontend::FP32ToMXINT8Op op) {
    if (int64_t(op.getAxis()) == op.getInput().getType().getRank() - 1)
      operations.push_back(op);
  });
  for (auto op : operations) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto type = op.getInput().getType();
    auto outputType = op.getResult().getType();
    int64_t k = type.getShape().back(), packed = k + k / 32;
    if (k <= 0 || k % 32 || k / 4 > target::getTarget().bankDepth)
      return op.emitError(
          "reduce quantization requires a static aligned row fitting one bank");
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    Value rows = index(1);
    SmallVector<Value> dynamic;
    for (int64_t axis = 0; axis < type.getRank() - 1; ++axis) {
      Value size = tensor::DimOp::create(b, loc, op.getInput(), axis);
      rows = b.createOrFold<arith::MulIOp>(loc, rows, size);
      if (type.isDynamicDim(axis))
        dynamic.push_back(size);
    }
    Value input = bufferization::ToBufferOp::create(
        b, loc, MemRefType::get(type.getShape(), b.getF32Type()), op.getInput(),
        true);
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(outputType.getShape(), b.getI8Type()), dynamic);
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
      Value from = arith::MulIOp::create(b, loc, row, index(k));
      Value to = arith::MulIOp::create(b, loc, row, index(packed));
      auto window = [&](Value memory, Value offset, int64_t length) -> Value {
        return memref::SubViewOp::create(
            b, loc, memory, ArrayRef<OpFoldResult>{offset},
            ArrayRef<OpFoldResult>{b.getIndexAttr(length)},
            ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
      };
      Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      Value q = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
      a = BankMvinOp::create(b, loc, b.getI64Type(), window(source, from, k), a,
                             k / 4, 1);
      q = ::buddy::mxint8quantize_ball::BankQuantMxint8Op::create(
          b, loc, b.getI64Type(), a, q, 1, k);
      if (packed % 16 == 0) {
        q = BankMvoutOp::create(b, loc, b.getI64Type(),
                                window(destination, to, packed), q, packed / 16,
                                1);
      } else {
        q = BankMvoutOp::create(b, loc, b.getI64Type(),
                                window(destination, to, k), q, k / 16, 1);
        if (k / 32 >= 16) {
          Value offset = arith::AddIOp::create(b, loc, to, index(k));
          q = BankMvoutStridedOp::create(
              b, loc, b.getI64Type(), window(destination, offset, k / 512 * 16),
              q, index(k / 16), index(k / 512), index(16), 0, 1, index(16),
              index(0));
        }
        Value offset =
            arith::AddIOp::create(b, loc, to, index(k + k / 512 * 16));
        q = BankMvoutStridedOp::create(
            b, loc, b.getI64Type(), window(destination, offset, k / 32 % 16), q,
            index(k / 16 + k / 512), index(1), index(k / 32 % 16), 0, 1,
            index(k / 32 % 16), index(0));
      }
      BankReleaseOp::create(b, loc, a);
      BankReleaseOp::create(b, loc, q);
    }
    Value result = bufferization::ToTensorOp::create(b, loc, outputType, output,
                                                     true, true);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}
