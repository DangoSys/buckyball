//===- Lowering.cpp - ReluBall in lower-buckyball -------------------------===//
//
// relu_i32: max(x, 0) on the first `rows` rows of four i32 of a one-group
// bank, in place: the one source is the destination.
//
// The Ball updates whole groups of sixteen rows: it rounds `rows` up to a
// multiple of sixteen, which only touches rows past the valid data, and runs
// one segment as deep as the bank.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class ReluI32Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "relu_i32")
      return failure();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
    if (op.getSources().size() != 1 ||
        op.getSources()[0] != int64_t(op.getDestination()))
      return op.emitOpError("relu_i32 updates its one source bank in place");
    if (rows <= 0 || rows > depth || depth % 16)
      return op.emitOpError("relu_i32 rows exceed the bank");
    // Group 0 of the bank in the bank1 field; bit 63 marks the in-place
    // update. rs2 is the segment stride.
    uint64_t rs1 = mlir::buddy::bankFields(op.getSources(), op.getDestination(),
                                           false, (rows + 15) / 16 * 16) |
                   (1ULL << 63);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(), buckyball_target::getBuckyballFunct7("RELU"),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs1),
        mlir::buddy::constantI64(rewriter, op.getLoc(), depth));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<ReluI32Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
