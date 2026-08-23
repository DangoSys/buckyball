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

template <typename BankOp, typename PhysicalOp>
class BankInt2FpPattern : public OpRewritePattern<BankOp> {
public:
  using OpRewritePattern<BankOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<PhysicalOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                                op.getIter(), op.getDaAddr(), op.getDwAddr());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateInt2FpAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankInt2FpPattern<BankInt2FpTensorOp, Int2FpTensorOp>,
               BankInt2FpPattern<BankInt2FpChannelOp, Int2FpChannelOp>>(
      patterns.getContext());
}
