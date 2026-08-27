#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateReluBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                PhysicalBankState &state);
}

namespace {
class BankReluPattern : public OpRewritePattern<BankReluOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BankReluOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<ReluOp>(op.getLoc(), op.getBank(), op.getGroup(),
                            op.getIter(), op.getStride());
    rewriter.replaceOp(op, op.getBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateReluBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankReluPattern>(patterns.getContext());
}
