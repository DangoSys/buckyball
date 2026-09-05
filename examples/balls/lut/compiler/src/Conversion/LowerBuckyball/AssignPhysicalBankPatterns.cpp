#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateLutBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                               PhysicalBankState &state);
} // namespace mlir::buddy

namespace {
class BankLutPattern : public OpRewritePattern<BankLutOp> {
public:
  BankLutPattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankLutOp>(context), state(state) {}

  LogicalResult matchAndRewrite(BankLutOp op,
                                PatternRewriter &rewriter) const override {
    auto input = state.getSlot(op.getInBank());
    auto lut = state.getSlot(op.getLutBank());
    auto output = state.getSlot(op.getOutBank());
    if (!input || !lut || !output)
      return failure();
    if (input->row != 1 || input->col != 1 || output->row != 1 ||
        output->col != 1 || lut->row != 1 || (lut->col != 1 && lut->col != 4))
      return op.emitError(
          "LUT requires col=1 input/output and col=1 or col=4 table");
    auto overlaps = [](const BankSlot &lhs, const BankSlot &rhs) {
      return lhs.base < rhs.base + rhs.row * rhs.col &&
             rhs.base < lhs.base + lhs.row * lhs.col;
    };
    if (overlaps(*input, *lut) || overlaps(*input, *output) ||
        overlaps(*lut, *output))
      return op.emitError("LUT bank groups must not overlap");
    rewriter.create<LutOp>(op.getLoc(), op.getInBank(), op.getLutBank(),
                           op.getOutBank(), op.getIter());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }

private:
  PhysicalBankState &state;
};
} // namespace

void mlir::buddy::populateLutBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  patterns.add<BankLutPattern>(patterns.getContext(), state);
}
