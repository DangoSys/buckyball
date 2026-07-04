//===- AssignPhysicalBankPatterns.cpp - Transpose bank assignment patterns ===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateTransposeAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                 PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankTransposePattern : public OpRewritePattern<BankTransposeOp> {
public:
  using OpRewritePattern<BankTransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankTransposeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<TransposeOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                                 op.getIter(), op.getMode());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateTransposeAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankTransposePattern>(patterns.getContext());
}
