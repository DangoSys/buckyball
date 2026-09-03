#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateInt8AddBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                   PhysicalBankState &state);
}

namespace {
class BankInt8AddPattern : public OpRewritePattern<BankInt8AddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BankInt8AddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Int8AddOp>(
        op.getLoc(), op.getLhsBank(), op.getRhsBank(), op.getOutputBank(),
        op.getIter(), op.getLhsRatio(), op.getRhsRatio(), op.getReluAttr());
    rewriter.replaceOp(op, op.getOutputBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateInt8AddBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  (void)state;
  patterns.add<BankInt8AddPattern>(patterns.getContext());
}
