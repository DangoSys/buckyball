//===- AssignPhysicalBankPatterns.cpp - Matrix bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateMatrixAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankMatrixPattern : public OpRewritePattern<BankMatrixOp> {
public:
  using OpRewritePattern<BankMatrixOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMatrixOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MatrixOp>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                              op.getWrBank(), op.getConfig());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateMatrixAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankMatrixPattern>(patterns.getContext());
}
