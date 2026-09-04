#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateInt8MulBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                   PhysicalBankState &state);
}

namespace {
class BankInt8MulPattern : public OpRewritePattern<BankInt8MulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BankInt8MulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Int8MulOp>(op.getLoc(), op.getGateBank(), op.getInputBank(),
                               op.getOutputBank(), op.getIter(), op.getRatio(),
                               op.getGateRow());
    rewriter.replaceOp(op, op.getOutputBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateInt8MulBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  (void)state;
  patterns.add<BankInt8MulPattern>(patterns.getContext());
}
