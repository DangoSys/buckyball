//===- QuantizeTensorToBankSSAPatterns.cpp - Input quantization ---------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

class QuantizeTensorToBankSSAPattern
    : public OpRewritePattern<QuantizeTensorF32ToI8Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeTensorF32ToI8Op op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inputTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputTy || !outputTy || !inputTy.hasStaticShape() ||
        !outputTy.hasStaticShape() || inputTy.getRank() != outputTy.getRank() ||
        (inputTy.getRank() != 2 && inputTy.getRank() != 4) ||
        !inputTy.getElementType().isF32() ||
        !outputTy.getElementType().isInteger(8))
      return op.emitError(
          "input quantization requires static rank-2/rank-4 memrefs");
    ArrayRef<int64_t> inputShape = inputTy.getShape();
    ArrayRef<int64_t> outputShape = outputTy.getShape();
    bool nchwToNhwc = op.getNchwToNhwcAttr().getValue();
    if ((!nchwToNhwc && inputShape != outputShape) ||
        (nchwToNhwc && inputTy.getRank() != 4) ||
        (nchwToNhwc &&
         (inputShape[0] != outputShape[0] || inputShape[1] != outputShape[3] ||
          inputShape[2] != outputShape[1] || inputShape[3] != outputShape[2])))
      return op.emitError("input quantization layout shape mismatch");

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < 4 ||
        target.bankDepth % 4)
      return op.emitError("input quantization requires 128-bit banks with "
                          "depth divisible by 4");
    int64_t elements = 1;
    for (int64_t dim : outputShape) {
      if (dim <= 0)
        return op.emitError("input quantization requires positive dimensions");
      elements *= dim;
    }

    Value inputBank = allocBank(b, loc, 1, 1);
    Value outputBank = allocBank(b, loc, 1, 1);
    Value scale =
        b.create<arith::ConstantOp>(loc, b.getF32Type(), op.getScaleAttr());
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value four = b.create<arith::ConstantIndexOp>(loc, 4);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, 16);
    Value zeroF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                                b.getF32FloatAttr(0.0));
    int64_t maxElements = target.bankDepth * 4;

    for (int64_t offset = 0; offset < elements; offset += maxElements) {
      int64_t validElements = std::min(maxElements, elements - offset);
      int64_t paddedElements = (validElements + 15) / 16 * 16;
      int64_t inputRows = paddedElements / 4;
      int64_t outputRows = paddedElements / 16;
      auto inputPackTy = MemRefType::get({inputRows, 4}, b.getF32Type());
      auto outputPackTy = MemRefType::get({outputRows, 16}, b.getI8Type());
      Value inputPack = b.create<memref::AllocOp>(loc, inputPackTy);
      Value outputPack = b.create<memref::AllocOp>(loc, outputPackTy);
      b.create<linalg::FillOp>(loc, zeroF32, inputPack);

      Value end = b.create<arith::ConstantIndexOp>(loc, validElements);
      Value offsetValue = b.create<arith::ConstantIndexOp>(loc, offset);
      auto copyIn = b.create<scf::ForOp>(loc, zero, end, one);
      b.setInsertionPointToStart(copyIn.getBody());
      Value i = copyIn.getInductionVar();
      Value linear = b.create<arith::AddIOp>(loc, offsetValue, i);
      SmallVector<Value, 4> outputIndices(outputTy.getRank());
      Value remaining = linear;
      for (int64_t dim = outputTy.getRank() - 1; dim >= 0; --dim) {
        Value extent = b.create<arith::ConstantIndexOp>(loc, outputShape[dim]);
        outputIndices[dim] = b.create<arith::RemUIOp>(loc, remaining, extent);
        remaining = b.create<arith::DivUIOp>(loc, remaining, extent);
      }
      SmallVector<Value, 4> indices(outputTy.getRank());
      if (nchwToNhwc) {
        indices[0] = outputIndices[0];
        indices[1] = outputIndices[3];
        indices[2] = outputIndices[1];
        indices[3] = outputIndices[2];
      } else {
        indices = outputIndices;
      }
      Value value = b.create<memref::LoadOp>(loc, op.getInput(), indices);
      Value packRow = b.create<arith::DivUIOp>(loc, i, four);
      Value packColumn = b.create<arith::RemUIOp>(loc, i, four);
      b.create<memref::StoreOp>(loc, value, inputPack,
                                ValueRange{packRow, packColumn});
      b.setInsertionPointAfter(copyIn);

      inputBank = mvinBank(b, loc, inputPack, inputBank, inputRows);
      outputBank = b.create<BankQuantF32ToI8Op>(
          loc, outputBank.getType(), inputBank, outputBank,
          createI64Const(b, loc, inputRows), scale);
      outputBank = mvoutBank(b, loc, outputPack, outputBank, outputRows);
      b.create<FenceOp>(loc);

      auto copyOut = b.create<scf::ForOp>(loc, zero, end, one);
      b.setInsertionPointToStart(copyOut.getBody());
      i = copyOut.getInductionVar();
      linear = b.create<arith::AddIOp>(loc, offsetValue, i);
      remaining = linear;
      for (int64_t dim = outputTy.getRank() - 1; dim >= 0; --dim) {
        Value extent = b.create<arith::ConstantIndexOp>(loc, outputShape[dim]);
        indices[dim] = b.create<arith::RemUIOp>(loc, remaining, extent);
        remaining = b.create<arith::DivUIOp>(loc, remaining, extent);
      }
      packRow = b.create<arith::DivUIOp>(loc, i, sixteen);
      packColumn = b.create<arith::RemUIOp>(loc, i, sixteen);
      value = b.create<memref::LoadOp>(loc, outputPack,
                                       ValueRange{packRow, packColumn});
      b.create<memref::StoreOp>(loc, value, op.getOutput(), indices);
      b.setInsertionPointAfter(copyOut);
      b.create<memref::DeallocOp>(loc, inputPack);
      b.create<memref::DeallocOp>(loc, outputPack);
    }

    releaseBank(b, loc, inputBank);
    releaseBank(b, loc, outputBank);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populatePebbleQuantizeTensorToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<QuantizeTensorToBankSSAPattern>(patterns.getContext());
}
} // namespace mlir::buddy
