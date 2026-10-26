//===- QuantizePatterns.cpp - marked FP32-to-INT8 input quantization -----===//
//
// Claims the generics the frontend marked `buckyball.quant_f32_to_i8`.
//
//===----------------------------------------------------------------------===//

#include <cmath>

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class QuantF32ToI8Lowering : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("buckyball.quant_f32_to_i8"))
      return failure();
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return op.emitError("FP32-to-INT8 quant requires one input and output");
    Value input = op.getInputs()[0];
    Value output = op.getOutputs()[0];
    auto inputType = dyn_cast<MemRefType>(input.getType());
    auto outputType = dyn_cast<MemRefType>(output.getType());
    auto nchwToNhwc = op->getAttrOfType<BoolAttr>("nchw_to_nhwc");
    auto scale = op->getAttrOfType<FloatAttr>("scale");
    if (!inputType || !outputType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() ||
        inputType.getRank() != outputType.getRank() ||
        (inputType.getRank() != 2 && inputType.getRank() != 4) ||
        !inputType.getElementType().isF32() ||
        !outputType.getElementType().isInteger(8) || !nchwToNhwc || !scale ||
        !std::isfinite(scale.getValueAsDouble()) ||
        scale.getValueAsDouble() <= 0.0)
      return op.emitError("FP32-to-INT8 quant requires matching static "
                          "rank-2/rank-4 memrefs and a positive scale");
    ArrayRef<int64_t> in = inputType.getShape();
    ArrayRef<int64_t> out = outputType.getShape();
    if (nchwToNhwc.getValue()
            ? inputType.getRank() != 4 || in[0] != out[0] || in[1] != out[3] ||
                  in[2] != out[1] || in[3] != out[2]
            : in != out)
      return op.emitError("FP32-to-INT8 quant layout shape mismatch");
    rewriter.create<QuantizeTensorF32ToI8Op>(op.getLoc(), input, output, scale,
                                             nchwToNhwc);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populateToInt8BallLinalgPatterns(RewritePatternSet &patterns) {
  patterns.add<QuantF32ToI8Lowering>(patterns.getContext());
}
} // namespace mlir::buddy
