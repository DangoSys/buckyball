//===- AssignPhysicalBankPatterns.cpp - Im2col bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateIm2colBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                  PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankIm2colPattern : public OpRewritePattern<BankIm2colOp> {
public:
  using OpRewritePattern<BankIm2colOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankIm2colOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Im2colOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                              op.getIter(), op.getKsize(), op.getStride(),
                              op.getPadding(), op.getStartRowAttr(),
                              op.getStartColAttr());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateIm2colBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankIm2colPattern>(patterns.getContext());
}
