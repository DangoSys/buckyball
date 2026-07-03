//===- AssignPhysicalBankPatterns.cpp - Vector bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateVectorAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankMulWarp16Pattern : public OpRewritePattern<BankMulWarp16Op> {
public:
  using OpRewritePattern<BankMulWarp16Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMulWarp16Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MulWarp16Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                                 op.getWrBank(), op.getIter(), op.getMode());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateVectorAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankMulWarp16Pattern>(patterns.getContext());
}
