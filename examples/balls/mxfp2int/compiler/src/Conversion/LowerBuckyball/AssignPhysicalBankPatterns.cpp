#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateMxfp2IntBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state);
} // namespace mlir::buddy

namespace {
class BankMxfp2IntPattern : public OpRewritePattern<BankMxfp2IntOp> {
public:
  using OpRewritePattern<BankMxfp2IntOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMxfp2IntOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Mxfp2IntOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                                op.getIter(), op.getSpecial());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateMxfp2IntBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankMxfp2IntPattern>(patterns.getContext());
}
