#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

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

// Computes A x B into the output bank; accumulation sets rs1 bit 29.
template <typename Op>
LogicalResult compute(Op op, PatternRewriter &rewriter, StringRef mnemonic,
                      int64_t subcommand) {
  Value operands = rs1(rewriter, op.getLoc(), op.getABank(), op.getBBank(),
                       op.getOutBank(), op.getIter());
  if (subcommand == 3)
    operands = rewriter.create<arith::OrIOp>(
        op.getLoc(), operands,
        rewriter.create<arith::ConstantIntOp>(op.getLoc(), 1LL << 29, 64));
  rewriter.create<GemminiInstructionOp>(
      op.getLoc(), rewriter.getStringAttr(mnemonic), operands,
      rs2(rewriter, op.getLoc(), op.getABase(), op.getBBase(), op.getOutBase(),
          subcommand));
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
} // namespace

LogicalResult
BankGemminiPreloadOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                          mlir::buddy::PhysicalBankState &) {
  Value zero = rewriter.create<arith::ConstantIntOp>(getLoc(), 0, 64);
  rewriter.create<GemminiInstructionOp>(
      getLoc(), rewriter.getStringAttr("GEMMINI_PRELOAD"),
      rs1(rewriter, getLoc(), getInBank(), zero, getOutBank(), getIter()),
      rs2(rewriter, getLoc(), getInBase(), zero, getOutBase(), 1));
  rewriter.replaceOp(*this, getOutBank());
  return success();
}

LogicalResult BankGemminiComputePreloadedOp::assignPhysicalBanks(
    PatternRewriter &rewriter, mlir::buddy::PhysicalBankState &) {
  return compute(*this, rewriter, "GEMMINI_COMPUTE_PRELOADED", 2);
}

LogicalResult BankGemminiComputeAccumulatedOp::assignPhysicalBanks(
    PatternRewriter &rewriter, mlir::buddy::PhysicalBankState &) {
  return compute(*this, rewriter, "GEMMINI_COMPUTE_ACCUMULATED", 3);
}
