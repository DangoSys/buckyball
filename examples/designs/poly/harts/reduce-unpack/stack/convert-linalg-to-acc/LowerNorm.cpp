#include "Acc/AccOps.h"
#include "Frontend/FrontendOps.h"
#include "Target/TargetRegistry.h"
#include "intnormOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <numeric>

using namespace mlir;
using namespace ::buddy::acc;

LogicalResult lowerReduceIntegerNorm(ModuleOp module) {
  SmallVector<::buddy::frontend::FixedNormMXINT8Op> operations;
  module.walk([&](::buddy::frontend::FixedNormMXINT8Op op) {
    operations.push_back(op);
  });
  for (auto op : operations) {
    auto inputType = op.getInput().getType();
    auto coefficientType = op.getCoefficients().getType();
    auto outputType = op.getResult().getType();
    if (inputType.getRank() < 1)
      return op.emitError("integer normalization requires rows");
    int64_t k = inputType.getShape().back(), packed = k + k / 32;
    int64_t depth = target::getTarget().bankDepth;
    if (k <= 0 || k % 512 || coefficientType.getRank() != 1 ||
        coefficientType.getDimSize(0) != 16 + k * 8 ||
        outputType.getRank() != inputType.getRank() ||
        outputType.getShape().drop_back() != inputType.getShape().drop_back() ||
        outputType.getShape().back() != packed || k / 4 > depth ||
        1 + k / 2 > depth || packed / 16 + k / 4 > depth)
      return op.emitError(
          "integer normalization shape or coefficients exceed the bank");
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(b, loc, value);
    };
    Value rows = index(1);
    SmallVector<Value> dynamic;
    for (int64_t axis = 0; axis < inputType.getRank() - 1; ++axis) {
      Value size = tensor::DimOp::create(b, loc, op.getInput(), axis);
      rows = b.createOrFold<arith::MulIOp>(loc, rows, size);
      if (inputType.isDynamicDim(axis))
        dynamic.push_back(size);
    }

    // Preserve borrowed input and parameter offsets through DMA lowering.
    SmallVector<Value> buffers;
    SmallVector<Value> tensors{op.getInput(), op.getCoefficients()};
    for (Value tensor : tensors) {
      auto type = cast<RankedTensorType>(tensor.getType());
      SmallVector<int64_t> strides(type.getRank());
      int64_t pitch = 1;
      for (int64_t axis = type.getRank() - 1; axis >= 0; --axis) {
        strides[axis] = pitch;
        pitch = pitch < 0 || type.isDynamicDim(axis)
                    ? ShapedType::kDynamic
                    : pitch * type.getDimSize(axis);
      }
      auto layout =
          StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, strides);
      auto memory =
          MemRefType::get(type.getShape(), type.getElementType(), layout);
      buffers.push_back(
          bufferization::ToBufferOp::create(b, loc, memory, tensor, true));
    }
    Value output = memref::AllocOp::create(
        b, loc, MemRefType::get(outputType.getShape(), b.getI8Type()), dynamic);
    output.getDefiningOp()->setAttr("alignment", b.getI64IntegerAttr(64));
    SmallVector<Value> flat;
    SmallVector<Value> memories{buffers[0], output};
    for (Value memory : memories) {
      auto type = cast<MemRefType>(memory.getType());
      if (type.getRank() == 1) {
        flat.push_back(memory);
      } else {
        ReassociationIndices axes(type.getRank());
        std::iota(axes.begin(), axes.end(), 0);
        flat.push_back(memref::CollapseShapeOp::create(
            b, loc, memory, ArrayRef<ReassociationIndices>{axes}));
      }
    }
    auto window = [&](Value memory, Value offset, int64_t length) -> Value {
      return memref::SubViewOp::create(
          b, loc, memory, ArrayRef<OpFoldResult>{offset},
          ArrayRef<OpFoldResult>{b.getIndexAttr(length)},
          ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
    };
    Value a = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    Value p = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    Value c = BankAllocOp::create(b, loc, b.getI64Type(), 1, 1);
    p = BankMvinOp::create(b, loc, b.getI64Type(), buffers[1], p, 1 + k / 2, 1);
    int64_t tile = 1;
    while (tile * 2 * k / 4 <= depth && tile * 2 * packed / 16 + k / 4 <= depth)
      tile *= 2;
    Value start = index(0);
    SmallVector<Value> held{a, c};
    for (int64_t height = tile; height > 0; height /= 2) {
      Value remaining = b.createOrFold<arith::SubIOp>(loc, rows, start);
      Value count =
          b.createOrFold<arith::DivUIOp>(loc, remaining, index(height));
      Value end = b.createOrFold<arith::AddIOp>(
          loc, start, b.createOrFold<arith::MulIOp>(loc, count, index(height)));
      auto loop = scf::ForOp::create(b, loc, start, end, index(height), held);
      b.setInsertionPointToStart(loop.getBody());
      Value row = loop.getInductionVar();
      Value from = arith::MulIOp::create(b, loc, row, index(k));
      Value to = arith::MulIOp::create(b, loc, row, index(packed));
      Value input = BankMvinOp::create(
          b, loc, b.getI64Type(), window(flat[0], from, height * k),
          loop.getRegionIterArgs()[0], height * k / 4, 1);
      Value normalized = ::buddy::intnorm_ball::BankNormMxint8Op::create(
          b, loc, b.getI64Type(), input, p, loop.getRegionIterArgs()[1], height,
          k, 0, 0, 0);
      Value stored = BankMvoutOp::create(b, loc, b.getI64Type(),
                                         window(flat[1], to, height * packed),
                                         normalized, height * packed / 16, 1);
      scf::YieldOp::create(b, loc, ValueRange{input, stored});
      b.setInsertionPointAfter(loop);
      held.assign(loop.getResults().begin(), loop.getResults().end());
      start = end;
    }
    BankReleaseOp::create(b, loc, held[0]);
    BankReleaseOp::create(b, loc, held[1]);
    BankReleaseOp::create(b, loc, p);
    Value result = bufferization::ToTensorOp::create(b, loc, outputType, output,
                                                     true, true);
    op.getResult().replaceAllUsesWith(result);
    op.erase();
  }
  return success();
}
