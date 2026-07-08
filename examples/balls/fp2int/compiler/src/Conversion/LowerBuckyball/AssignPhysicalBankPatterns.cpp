//===- AssignPhysicalBankPatterns.cpp - Fp2Int bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateFp2IntAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankFp2IntPattern : public OpRewritePattern<BankFp2IntOp> {
public:
  using OpRewritePattern<BankFp2IntOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankFp2IntOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Fp2IntOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                              op.getIter(), op.getScale());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateFp2IntAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankFp2IntPattern>(patterns.getContext());
}
