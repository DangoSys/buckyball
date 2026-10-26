#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankMxquantOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                   mlir::buddy::PhysicalBankState &) {
  BankMxquantOp op = *this;
  rewriter.create<MxquantOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                             op.getIter());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
