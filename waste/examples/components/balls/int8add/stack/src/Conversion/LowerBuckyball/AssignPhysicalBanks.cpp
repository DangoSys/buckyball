#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankInt8AddOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                   mlir::buddy::PhysicalBankState &) {
  BankInt8AddOp op = *this;
  rewriter.create<Int8AddOp>(op.getLoc(), op.getLhsBank(), op.getRhsBank(),
                             op.getOutputBank(), op.getIter(), op.getLhsRatio(),
                             op.getRhsRatio(), op.getReluAttr());
  rewriter.replaceOp(op, op.getOutputBank());
  return success();
}
