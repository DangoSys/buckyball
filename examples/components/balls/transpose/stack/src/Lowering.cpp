//===- Lowering.cpp - TransposeBall in lower-buckyball --------------------===//
//
// Both banks have cols / (bankWidthBits / 8) columns. The instruction packs
// the transposed bytes into the same bank geometry, with `rows` bank rows.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class TransposeI8Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "transpose_i8")
      return failure();
    if (op.getSources().size() != 1 ||
        op.getSources()[0] == op.getDestination())
      return op.emitOpError(
          "transpose_i8 requires distinct input and output banks");
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t cols = cast<IntegerAttr>(op->getAttr("cols")).getInt();
    const auto &target = buckyball_target::getBuckyballTarget();
    int64_t lanes = target.bankWidthBits / 8;
    if (lanes <= 0 || target.bankWidthBits % 8 || rows <= 0 ||
        rows >= (1LL << 33) || rows > target.bankDepth || cols <= 0 ||
        cols % lanes || cols / lanes > 31)
      return op.emitOpError(
          "transpose_i8 shape exceeds the bank or ISA geometry");
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(),
        buckyball_target::getBuckyballFunct7("TRANSPOSE"),
        mlir::buddy::constantI64(rewriter, op.getLoc(),
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        mlir::buddy::constantI64(rewriter, op.getLoc(), 8));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<TransposeI8Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
