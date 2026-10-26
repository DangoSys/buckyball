#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankInt8MulOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                   mlir::buddy::PhysicalBankState &) {
  BankInt8MulOp op = *this;
  rewriter.create<Int8MulOp>(op.getLoc(), op.getGateBank(), op.getInputBank(),
                             op.getOutputBank(), op.getIter(), op.getRatio(),
                             op.getGateRow());
  rewriter.replaceOp(op, op.getOutputBank());
  return success();
}
