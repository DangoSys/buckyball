#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankMXFP8Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                 mlir::buddy::PhysicalBankState &) {
  BankMXFP8Op op = *this;
  rewriter.create<MXFP8Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                           op.getWrBank(), op.getConfig(), op.getFirst(),
                           op.getLast(), op.getOutputBase());
  rewriter.replaceOp(op, op.getWrBank());
  return success();
}

LogicalResult
BankMXFP8WindowOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                       mlir::buddy::PhysicalBankState &) {
  BankMXFP8WindowOp op = *this;
  rewriter.create<MXFP8WindowOp>(
      op.getLoc(), op.getOp1Bank(), op.getOp2Bank(), op.getWrBank(),
      op.getRows(), op.getCols(), op.getCount(), op.getFullK(), op.getStartK(),
      op.getFirst(), op.getLast(), op.getOutputBase());
  rewriter.replaceOp(op, op.getWrBank());
  return success();
}

LogicalResult
BankFP32Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                mlir::buddy::PhysicalBankState &) {
  BankFP32Op op = *this;
  rewriter.create<FP32Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                          op.getWrBank(), op.getConfig(), op.getFirst(),
                          op.getLast(), op.getOutputBase(), op.getFusedAttr());
  rewriter.replaceOp(op, op.getWrBank());
  return success();
}
