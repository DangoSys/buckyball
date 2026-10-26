//===- Lowering.cpp - ReluBall in lower-buckyball -------------------------===//
//
// relu_i32: max(x, 0) on the i32 lanes of physical bank `group` of the bank,
// in place: the one source is the destination. In every `stride` rows of the
// physical bank the first `rows` rows are updated; `rows` is a multiple of
// sixteen no larger than `stride`, and `stride` divides the bank depth.
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
    int64_t stride = cast<IntegerAttr>(op->getAttr("stride")).getInt();
    int64_t group = cast<IntegerAttr>(op->getAttr("group")).getInt();
    int64_t depth = buckyball_target::getBuckyballTarget().bankDepth;
    if (op.getSources().size() != 1 ||
        op.getSources()[0] != int64_t(op.getDestination()))
      return op.emitOpError("relu_i32 updates its one source bank in place");
    if (rows <= 0 || rows % 16 || rows > stride || depth % stride ||
        group < 0 || group >= 1024)
      return op.emitOpError("relu_i32 rows, stride or group exceed the ISA");
    // The bank1 field selects the physical bank; bit 63 marks the in-place
    // update.
    uint64_t rs1 = mlir::buddy::bankFields(op.getSources(), op.getDestination(),
                                           false, rows) |
                   mlir::buddy::field(group, 10, 19) | (1ULL << 63);
    mlir::buddy::emitInstruction(
        rewriter, op.getLoc(), buckyball_target::getBuckyballFunct7("RELU"),
        mlir::buddy::constantI64(rewriter, op.getLoc(), rs1),
        mlir::buddy::constantI64(rewriter, op.getLoc(), stride));
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
