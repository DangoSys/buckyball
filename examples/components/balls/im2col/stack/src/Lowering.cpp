//===- Lowering.cpp - Im2colBall in lower-buckyball -----------------------===//
//
// im2col_i8: the source holds a square feature map of `input_side` pixels per
// side; group g is channel block g, line y * input_side + x holds the sixteen
// channels of pixel (y, x). The destination receives the windows of the
// `kernel` x `kernel` convolution with `stride` and zero `padding`, every
// window in row-major order, as the matmul_i8 A operand: K is ordered (kernel
// row, kernel column, channel), sixteen columns (one source line) per K tile,
// K tile t = (kr * kernel + kc) * groups + g. The destination groups divide
// the K tiles; group j holds the next K / 16 / groups tiles in M tiles of
// `tile_rows` rows (the layout of the matmul that consumes it), windows
// beyond the last zero up to a multiple of tile_rows.
//
//===----------------------------------------------------------------------===//

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
    int64_t tileRows = cast<IntegerAttr>(op->getAttr("tile_rows")).getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t span = side + 2 * padding - kernel;
    if (target.bankWidthBits != 128 || side <= 0 ||
        side > buckyball_target::getBuckyballBallParam("IM2COL", "maxIter") ||
        side * side > target.bankDepth || kernel <= 0 ||
        kernel >
            buckyball_target::getBuckyballBallParam("IM2COL", "maxKSize") ||
        stride <= 0 || stride > 15 || padding < 0 ||
        padding >
            buckyball_target::getBuckyballBallParam("IM2COL", "maxPadding") ||
        span < 0 || span % stride || tileRows <= 0 || tileRows > 16 ||
        (tileRows & (tileRows - 1)))
      return op.emitOpError("im2col_i8 geometry exceeds the ISA field ranges");
    int64_t windows = (span / stride + 1) * (span / stride + 1);
    if (windows > 1023)
      return op.emitOpError("im2col_i8 has more windows than one instruction");
    uint64_t rs2 = mlir::buddy::field(kernel, 0, 3) |
                   mlir::buddy::field(stride, 4, 7) |
                   mlir::buddy::field(padding, 8, 11) |
                   mlir::buddy::field(windows, 36, 45) |
                   mlir::buddy::field(tileRows, 46, 50);
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
