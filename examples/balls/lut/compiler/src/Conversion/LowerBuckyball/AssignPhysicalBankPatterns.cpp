#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateLutBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                               PhysicalBankState &state);
} // namespace mlir::buddy

namespace {
class BankLutPattern : public OpRewritePattern<BankLutOp> {
public:
  using OpRewritePattern<BankLutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankLutOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<LutOp>(op.getLoc(), op.getInBank(), op.getLutBank(),
                           op.getOutBank(), op.getIter());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateLutBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  (void)state;
  patterns.add<BankLutPattern>(patterns.getContext());
}
