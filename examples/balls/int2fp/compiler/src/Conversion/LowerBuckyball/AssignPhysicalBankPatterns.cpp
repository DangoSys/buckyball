//===- AssignPhysicalBankPatterns.cpp - Int2Fp bank assignment patterns ---===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateInt2FpBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                  PhysicalBankState &state);
} // namespace mlir::buddy

namespace {

class BankInt32ToFp32Pattern : public OpRewritePattern<BankInt32ToFp32Op> {
public:
  using OpRewritePattern<BankInt32ToFp32Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankInt32ToFp32Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Int32ToFp32Op>(op.getLoc(), op.getInBank(),
                                   op.getScaleBank(), op.getOutBank(),
                                   op.getIter(), op.getReluAttr());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

} // namespace

void mlir::buddy::populateInt2FpBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankInt32ToFp32Pattern>(patterns.getContext());
}
