//===- AssignPhysicalBankPatterns.cpp - Im2col bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateIm2colAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankIm2colPattern : public OpRewritePattern<BankIm2colOp> {
public:
  using OpRewritePattern<BankIm2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankIm2colOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Im2colOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                              op.getKrow(), op.getKcol(), op.getInrow(),
                              op.getIncol(), op.getStartrow(), op.getStartcol(),
                              op.getColStep());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateIm2colAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankIm2colPattern>(patterns.getContext());
}
