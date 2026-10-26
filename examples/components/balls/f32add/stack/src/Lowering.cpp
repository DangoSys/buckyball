//===- Lowering.cpp - F32AddBall in lower-buckyball -----------------------===//
//
// add_f32: C = A + B over `rows` rows of four f32, RNE with canonical NaN.
// A, B and C are distinct one-group banks.
//
// The Ball reads both addends from its source banks and cannot read the old
// destination, so it takes no read_destination: a sum of more terms is a
// chain of add_f32 into fresh banks. The Ball's input group is group 0 and it
// always reads B.
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
    if (rows <= 0 || rows > buckyball_target::getBuckyballTarget().bankDepth ||
        op.getReadDestination() || op.getSources().size() != 2 ||
        op.getSources()[0] == op.getSources()[1] ||
        llvm::is_contained(op.getSources(), op.getDestination()))
      return op.emitOpError("add_f32 requires distinct A, B and C banks, rows "
                            "within a bank and no read_destination");
    // rs2: group 0 in [4:0]; bit 5 clear reads B.
    mlir::buddy::emitInstruction(
        rewriter, loc, buckyball_target::getBuckyballFunct7("F32ADD"),
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
  patterns.add<AddF32Lowering>(patterns.getContext());
}

[[maybe_unused]] const bool registered =
    (mlir::buddy::addBallLowering(populate), true);
} // namespace
