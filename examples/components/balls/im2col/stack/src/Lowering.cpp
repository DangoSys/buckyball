//===- Lowering.cpp - Im2colBall in lower-buckyball -----------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class Im2colI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "im2col_i8")
      return failure();
    if (op.getSources().size() != 1 ||
        op.getSources()[0] == op.getDestination())
      return op.emitOpError(
          "im2col_i8 requires distinct input and output banks");
    int64_t side = cast<IntegerAttr>(op->getAttr("input_side")).getInt();
    int64_t kernel = cast<IntegerAttr>(op->getAttr("kernel")).getInt();
    int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
    int64_t padding = cast<IntegerAttr>(op->getAttr("padding")).getInt();
    int64_t startRow = cast<IntegerAttr>(op->getAttr("start_row")).getInt();
    int64_t startCol = cast<IntegerAttr>(op->getAttr("start_col")).getInt();
    int64_t base = cast<IntegerAttr>(op->getAttr("op1_base")).getInt();
    int64_t lane = cast<IntegerAttr>(op->getAttr("lane")).getInt();
    int64_t windowStart =
        cast<IntegerAttr>(op->getAttr("window_start")).getInt();
    int64_t windowCount =
        cast<IntegerAttr>(op->getAttr("window_count")).getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || side <= 0 || side >= (1LL << 33) ||
        side > buckyball_target::getBuckyballBallParam("IM2COL", "maxIter") ||
        kernel <= 0 || kernel > 255 ||
        kernel >
            buckyball_target::getBuckyballBallParam("IM2COL", "maxKSize") ||
        stride <= 0 || stride > 255 || padding < 0 || padding > 255 ||
        padding >
            buckyball_target::getBuckyballBallParam("IM2COL", "maxPadding") ||
        startRow < 0 || startRow > padding || startCol < 0 ||
        startCol > padding || base < 0 || base > 63 || lane < 0 || lane > 15 ||
        windowStart < 0 || windowStart > 63 || windowCount <= 0 ||
        windowCount > 64)
      return op.emitOpError("im2col_i8 geometry exceeds the ISA field ranges");
    int64_t rowNumerator = side + 2 * padding - kernel - startRow;
    int64_t colNumerator = side + 2 * padding - kernel - startCol;
    if (rowNumerator < 0 || colNumerator < 0 || rowNumerator % stride ||
        colNumerator % stride ||
        windowStart + windowCount >
            (rowNumerator / stride + 1) * (colNumerator / stride + 1) ||
        base + side * side > target.bankDepth ||
        ((windowCount + 15) / 16) * ((kernel * kernel + 15) / 16) * 16 >
            target.bankDepth)
      return op.emitOpError("im2col_i8 window or bank footprint is invalid");
    uint64_t rs2 =
        mlir::buddy::field(kernel, 0, 7) | mlir::buddy::field(stride, 8, 15) |
        mlir::buddy::field(padding, 16, 23) |
        mlir::buddy::field(startCol, 24, 31) |
        mlir::buddy::field(startRow, 32, 39) |
        mlir::buddy::field(base, 40, 45) | mlir::buddy::field(lane, 46, 49) |
        mlir::buddy::field(windowStart, 50, 55) |
        mlir::buddy::field(windowCount, 56, 62);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(), buckyball_target::getBuckyballFunct7("IM2COL"),
        mlir::buddy::constantI64(rewriter, op.getLoc(),
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, side)),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs2));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<Im2colI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
