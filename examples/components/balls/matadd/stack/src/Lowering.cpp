//===- Lowering.cpp - MatAddBall in lower-buckyball -----------------------===//
//
// add_i32: C = A + B modulo 2^32 over `rows` rows of four i32 in every group
// of the banks; A, B and C have the same number of groups.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/BuckyballOps.h"
#include "Conversion/Buckyball.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class AddI32Lowering : public OpRewritePattern<CustomOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != "add_i32")
      return failure();
    Location loc = op.getLoc();
    int64_t rows = cast<IntegerAttr>(op->getAttr("rows")).getInt();
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError(
          "add_i32 requires distinct banks and rows within a bank");
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("MATADD"),
        mlir::buddy::constantI64(rewriter, loc,
                                 mlir::buddy::bankFields(op.getSources(),
                                                         op.getDestination(),
                                                         false, rows)),
        mlir::buddy::constantI64(rewriter, loc, 0));
    rewriter.eraseOp(op);
    return success();
  }
};

void populate(RewritePatternSet &patterns) {
  patterns.add<AddI32Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
