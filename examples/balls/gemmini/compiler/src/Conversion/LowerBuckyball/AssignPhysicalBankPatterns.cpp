#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace mlir::buddy {
void populateGemminiBallAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                                   PhysicalBankState &state);
}

namespace {
Value shift(PatternRewriter &rewriter, Location loc, Value value,
            int64_t bits) {
  return rewriter.create<arith::ShLIOp>(
      loc, value, rewriter.create<arith::ConstantIntOp>(loc, bits, 64));
}

Value rs1(PatternRewriter &rewriter, Location loc, Value a, Value b, Value c,
          Value iter) {
  Value ab = rewriter.create<arith::OrIOp>(loc, a, shift(rewriter, loc, b, 10));
  Value ci = rewriter.create<arith::OrIOp>(loc, shift(rewriter, loc, c, 20),
                                           shift(rewriter, loc, iter, 30));
  return rewriter.create<arith::OrIOp>(loc, ab, ci);
}

Value rs2(PatternRewriter &rewriter, Location loc, Value a, Value b, Value c,
          int64_t subcommand) {
  Value packed = rewriter.create<arith::ConstantIntOp>(loc, subcommand, 64);
  packed =
      rewriter.create<arith::OrIOp>(loc, packed, shift(rewriter, loc, a, 6));
  packed =
      rewriter.create<arith::OrIOp>(loc, packed, shift(rewriter, loc, b, 16));
  return rewriter.create<arith::OrIOp>(loc, packed,
                                       shift(rewriter, loc, c, 26));
}

class BankGemminiPreloadPattern
    : public OpRewritePattern<BankGemminiPreloadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BankGemminiPreloadOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    rewriter.create<GemminiInstructionOp>(
        op.getLoc(), rewriter.getStringAttr("GEMMINI_PRELOAD"),
        rs1(rewriter, op.getLoc(), op.getInBank(), zero, op.getOutBank(),
            op.getIter()),
        rs2(rewriter, op.getLoc(), op.getInBase(), zero, op.getOutBase(), 1));
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }
};

template <typename Op>
class BankGemminiComputePattern : public OpRewritePattern<Op> {
public:
  BankGemminiComputePattern(MLIRContext *context, StringRef mnemonic,
                            int64_t subcommand)
      : OpRewritePattern<Op>(context), mnemonic(mnemonic),
        subcommand(subcommand) {}
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<GemminiInstructionOp>(
        op.getLoc(), rewriter.getStringAttr(mnemonic),
        rs1(rewriter, op.getLoc(), op.getABank(), op.getBBank(),
            op.getOutBank(), op.getIter()),
        rs2(rewriter, op.getLoc(), op.getABase(), op.getBBase(),
            op.getOutBase(), subcommand));
    rewriter.replaceOp(op, op.getOutBank());
    return success();
  }

private:
  StringRef mnemonic;
  int64_t subcommand;
};
} // namespace

void mlir::buddy::populateGemminiBallAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  (void)state;
  patterns.add<BankGemminiPreloadPattern>(patterns.getContext());
  patterns.add<BankGemminiComputePattern<BankGemminiComputePreloadedOp>>(
      patterns.getContext(), "GEMMINI_COMPUTE_PRELOADED", 2);
  patterns.add<BankGemminiComputePattern<BankGemminiComputeAccumulatedOp>>(
      patterns.getContext(), "GEMMINI_COMPUTE_ACCUMULATED", 3);
}
