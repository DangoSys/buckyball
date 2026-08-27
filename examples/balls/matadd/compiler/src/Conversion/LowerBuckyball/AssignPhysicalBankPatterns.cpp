#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateMatAddAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
}

namespace {
class BankMatAddPattern : public OpRewritePattern<BankMatAddOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMatAddOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MatAddOp>(op.getLoc(), op.getABank(), op.getBBank(),
                              op.getCBank(), op.getIter());
    rewriter.replaceOp(op, op.getCBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateMatAddAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankMatAddPattern>(patterns.getContext());
}
