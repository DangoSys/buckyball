//===- Lowering.cpp - MaxPoolBall in lower-buckyball ----------------------===//
//
// maxpool_i8: the source holds a square feature map of `input_side` pixels
// per side; group g is channel block g, line y * input_side + x holds the
// sixteen i8 channels of pixel (y, x). The destination, with the same groups,
// receives the `kernel` x `kernel` maxima with `stride` and `padding` in the
// same layout, output_side = (input_side + 2 * padding - kernel) / stride + 1.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class MaxpoolI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "maxpool_i8")
      return failure();
    if (op.getSources().size() != 1 ||
        op.getSources()[0] == op.getDestination())
      return op.emitOpError(
          "maxpool_i8 requires distinct input and output banks");
    int64_t inputSide = cast<IntegerAttr>(op->getAttr("input_side")).getInt();
    int64_t kernel = cast<IntegerAttr>(op->getAttr("kernel")).getInt();
    int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
    int64_t padding = cast<IntegerAttr>(op->getAttr("padding")).getInt();
    int64_t span = inputSide + 2 * padding - kernel;
    if (inputSide <= 0 || inputSide > 15 || kernel <= 0 || kernel > 15 ||
        stride <= 0 || stride > 15 || padding < 0 || padding > 15 || span < 0 ||
        span % stride || span / stride + 1 > 15 ||
        inputSide * inputSide >
            buckyball_target::getBuckyballTarget().bankDepth)
      return op.emitOpError("maxpool_i8 geometry exceeds the ISA field ranges");
    int64_t outputSide = span / stride + 1;
    // Both maps start at the base of their groups; output rows are packed.
    uint64_t rs2 = mlir::buddy::field(inputSide, 0, 3) |
                   mlir::buddy::field(outputSide, 4, 7) |
                   mlir::buddy::field(kernel, 8, 11) |
                   mlir::buddy::field(stride, 12, 15) |
                   mlir::buddy::field(padding, 16, 19) |
                   mlir::buddy::field(outputSide, 32, 37);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(), buckyball_target::getBuckyballFunct7("MAXPOOL"),
        mlir::buddy::constantI64(
            rewriter, op.getLoc(),
            mlir::buddy::bankFields(op.getSources(), op.getDestination(), false,
                                    outputSide * outputSide) |
                (1ULL << 63)),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<MaxpoolI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
