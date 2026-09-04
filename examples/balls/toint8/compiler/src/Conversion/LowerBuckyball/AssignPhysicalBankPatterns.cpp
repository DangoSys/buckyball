//===- AssignPhysicalBankPatterns.cpp - ToInt8 bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateToInt8BallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                  PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankQuantF32ToI8Pattern : public OpRewritePattern<BankQuantF32ToI8Op> {
public:
  using OpRewritePattern<BankQuantF32ToI8Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankQuantF32ToI8Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<QuantF32ToI8Op>(op.getLoc(), op.getInBank(),
                                    op.getOutBank(), op.getIter(),
                                    op.getScale());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

class BankQuantI32ToI8Pattern : public OpRewritePattern<BankQuantI32ToI8Op> {
public:
  using OpRewritePattern<BankQuantI32ToI8Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankQuantI32ToI8Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<QuantI32ToI8Op>(
        op.getLoc(), op.getInBank(), op.getScaleBank(), op.getOutBank(),
        op.getIter(), op.getOutputBase(), op.getInputBase(),
        op.getOutputWidthAttr(), op.getOutputHeightAttr(),
        op.getOutputStrideAttr(), op.getReluAttr());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateToInt8BallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankQuantF32ToI8Pattern, BankQuantI32ToI8Pattern>(
      patterns.getContext());
}
