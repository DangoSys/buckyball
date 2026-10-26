//===- Lowering.cpp - Int2FpBall in lower-buckyball -----------------------===//
//
// int32_to_fp32: source 0 holds `rows` rows of sixteen i32 (four bank rows
// each), source 1 the sixteen f32 scales of the columns (four bank rows); the
// destination receives the f32 value * scale of every element.
// int32_to_fp32_relu: the same, with negative results replaced by zero.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class Int32ToFp32Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    bool relu = op.getKind() == "int32_to_fp32_relu";
    if (op.getKind() != "int32_to_fp32" && !relu)
      return failure();
    if (op.getSources().size() != 2)
      return op.emitOpError("int32_to_fp32 reads the values and the scales");
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("INT32_TO_FP32"),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows * 4)),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::field(relu, 0, 0)));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<Int32ToFp32Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
