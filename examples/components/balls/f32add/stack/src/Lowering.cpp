//===- Lowering.cpp - F32AddBall in lower-buckyball -----------------------===//
//
// add_f32: C = B + A[group] over `rows` rows of four f32, RNE with canonical
// NaN. A may have several groups; B and C have one. With `first` the sum
// starts from +0 instead of reading B, which still names an allocated bank.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class AddF32Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "add_f32")
      return failure();
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    int64_t group = cast<IntegerAttr>(op->getAttr("group")).getInt();
    bool first = cast<BoolAttr>(op->getAttr("first")).getValue();
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        group < 0 || group > 31 || op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "add_f32 requires distinct banks, rows within a bank and a group "
          "in [0, 31]");
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("F32ADD"),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::field(group, 0, 4) |
                                     mlir::buddy::field(first, 5, 5)));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<AddF32Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
