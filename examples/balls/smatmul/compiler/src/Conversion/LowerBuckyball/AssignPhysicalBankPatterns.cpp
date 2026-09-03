//===- AssignPhysicalBankPatterns.cpp - Matrix bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateSMatMulBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                   PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankSMatMulPattern : public OpRewritePattern<BankSMatMulOp> {
public:
  using OpRewritePattern<BankSMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankSMatMulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<SMatMulOp>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                               op.getWrBank(), op.getConfig(),
                               op.getFirstAttr(), op.getLastAttr());
    rewriter.replaceOp(op, op.getWrBank());
    return success();
  }
};

class BankSMatMulBiasPattern : public OpRewritePattern<BankSMatMulBiasOp> {
public:
  using OpRewritePattern<BankSMatMulBiasOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankSMatMulBiasOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<SMatMulBiasOp>(op.getLoc(), op.getBiasBank());
    rewriter.replaceOp(op, op.getBiasBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateSMatMulBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankSMatMulPattern, BankSMatMulBiasPattern>(
      patterns.getContext());
}
