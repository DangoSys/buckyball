#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateNliBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                               PhysicalBankState &state);
} // namespace mlir::buddy

namespace {
class BankNliPattern : public OpRewritePattern<BankNliOp> {
public:
  BankNliPattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankNliOp>(context), state(state) {}

  LogicalResult matchAndRewrite(BankNliOp op,
                                PatternRewriter &rewriter) const override {
    auto input = state.getSlot(op.getInBank());
    auto table = state.getSlot(op.getTableBank());
    auto output = state.getSlot(op.getOutBank());
    if (!input || !table || !output)
      return failure();
    if (input->row != 1 || input->col != 1 || output->row != 1 ||
        output->col != 1 || table->row != 1 || table->col != 1)
      return op.emitError("NLI requires col=1 input/table/output");
    auto overlaps = [](const BankSlot &lhs, const BankSlot &rhs) {
      return lhs.base < rhs.base + rhs.row * rhs.col &&
             rhs.base < lhs.base + lhs.row * lhs.col;
    };
    if (overlaps(*input, *table) || overlaps(*input, *output) ||
        overlaps(*table, *output))
      return op.emitError("NLI bank groups must not overlap");
    rewriter.create<NliOp>(op.getLoc(), op.getInBank(), op.getTableBank(),
                           op.getOutBank(), op.getIter());
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }

private:
  PhysicalBankState &state;
};
} // namespace

void mlir::buddy::populateNliBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, PhysicalBankState &state) {
  patterns.add<BankNliPattern>(patterns.getContext(), state);
}
