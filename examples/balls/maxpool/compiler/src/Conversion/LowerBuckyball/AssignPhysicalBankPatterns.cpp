#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateMaxPoolBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                   PhysicalBankState &state);
} // namespace mlir::buddy

namespace {
class BankMaxPoolPattern : public OpRewritePattern<BankMaxPoolOp> {
public:
  using OpRewritePattern<BankMaxPoolOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMaxPoolOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MaxPoolOp>(
        op.getLoc(), op.getInBank(), op.getOutBank(), op.getIter(),
        op.getInputSideAttr(), op.getOutputSideAttr(), op.getKernelAttr(),
        op.getStrideAttr(), op.getPaddingAttr(), op.getInputBaseAttr(),
        op.getOutputBaseAttr(), op.getOutputStrideAttr());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateMaxPoolBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  (void)state;
  patterns.add<BankMaxPoolPattern>(patterns.getContext());
}
