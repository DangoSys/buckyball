//===- Lowering.cpp - MaxPoolBall in lower-buckyball ----------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include <algorithm>

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
    int64_t outputSide = cast<IntegerAttr>(op->getAttr("output_side")).getInt();
    int64_t kernel = cast<IntegerAttr>(op->getAttr("kernel")).getInt();
    int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
    int64_t padding = cast<IntegerAttr>(op->getAttr("padding")).getInt();
    int64_t inputBase = cast<IntegerAttr>(op->getAttr("op1_base")).getInt();
    int64_t outputBase = cast<IntegerAttr>(op->getAttr("wr_base")).getInt();
    int64_t outputStride =
        cast<IntegerAttr>(op->getAttr("output_stride")).getInt();
    int64_t startRow = cast<IntegerAttr>(op->getAttr("start_row")).getInt();
    int64_t startCol = cast<IntegerAttr>(op->getAttr("start_col")).getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || inputSide <= 0 || inputSide > 15 ||
        outputSide <= 0 || outputSide > 15 || kernel <= 0 || kernel > 15 ||
        stride <= 0 || stride > 15 || padding < 0 || padding > 15 ||
        inputBase < 0 || inputBase > 63 || outputBase < 0 || outputBase > 63 ||
        outputStride < outputSide || outputStride > 63 || startRow < 0 ||
        startRow > 15 || startCol < 0 || startCol > 15)
      return op.emitOpError("maxpool_i8 geometry exceeds the ISA field ranges");
    int64_t lastRow = startRow + (outputSide - 1) * stride + kernel - 1;
    int64_t lastCol = startCol + (outputSide - 1) * stride + kernel - 1;
    int64_t inputEnd = padding + inputSide;
    if (lastRow >= inputSide + 2 * padding ||
        lastCol >= inputSide + 2 * padding ||
        outputBase + (outputSide - 1) * outputStride + outputSide >
            target.bankDepth)
      return op.emitOpError(
          "maxpool_i8 window or output bank footprint is invalid");
    if (startRow < inputEnd && startCol < inputEnd && lastRow >= padding &&
        lastCol >= padding &&
        inputBase + (std::min(lastRow, inputEnd - 1) - padding) * inputSide +
                std::min(lastCol, inputEnd - 1) - padding >=
            target.bankDepth)
      return op.emitOpError("maxpool_i8 input bank footprint is invalid");
    uint64_t rs2 = mlir::buddy::field(inputSide, 0, 3) |
                   mlir::buddy::field(outputSide, 4, 7) |
                   mlir::buddy::field(kernel, 8, 11) |
                   mlir::buddy::field(stride, 12, 15) |
                   mlir::buddy::field(padding, 16, 19) |
                   mlir::buddy::field(inputBase, 20, 25) |
                   mlir::buddy::field(outputBase, 26, 31) |
                   mlir::buddy::field(outputStride, 32, 37) |
                   mlir::buddy::field(startRow, 38, 41) |
                   mlir::buddy::field(startCol, 42, 45);
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
