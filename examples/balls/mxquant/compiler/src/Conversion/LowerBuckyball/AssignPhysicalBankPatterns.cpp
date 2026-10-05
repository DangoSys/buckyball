#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateMxquantBallAssignPhysicalBankPatterns(RewritePatternSet &,
                                                   PhysicalBankState &);
}

namespace {
class BankMxquantPattern : public OpRewritePattern<BankMxquantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BankMxquantOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MxquantOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                               op.getIter());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};
} // namespace

void mlir::buddy::populateMxquantBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &) {
  patterns.add<BankMxquantPattern>(patterns.getContext());
}
