//===- AssignPhysicalBankPatterns.cpp - Vector bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateVecBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankVecMat16Pattern : public OpRewritePattern<BankVecMat16Op> {
public:
  using OpRewritePattern<BankVecMat16Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankVecMat16Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<VecMat16Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                                op.getWrBank(), op.getIter(), op.getMode());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateVecBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankVecMat16Pattern>(patterns.getContext());
}
