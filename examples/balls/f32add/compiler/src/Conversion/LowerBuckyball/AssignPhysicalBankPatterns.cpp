#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
class BankF32AddPattern : public OpRewritePattern<BankF32AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BankF32AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<F32AddOp>(op.getLoc(), op.getA(), op.getB(), op.getC(),
                              op.getRowsAttr(), op.getGroupAttr(),
                              op.getFirstAttr());
    rewriter.replaceOp(op, op.getC());
    return success();
  }
};
} // namespace

namespace mlir::buddy {
void populateF32AddBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                  PhysicalBankState &) {
  patterns.add<BankF32AddPattern>(patterns.getContext());
}
} // namespace mlir::buddy
