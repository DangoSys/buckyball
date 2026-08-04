//===- AssignPhysicalBankPatterns.cpp - Int2Fp bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateInt2FpAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                              PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankInt2FpPattern : public OpRewritePattern<BankInt2FpOp> {
public:
  using OpRewritePattern<BankInt2FpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankInt2FpOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Int2FpOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                              op.getIter(), op.getScale());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateInt2FpAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankInt2FpPattern>(patterns.getContext());
}
