#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankMatAddOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                  mlir::buddy::PhysicalBankState &) {
  BankMatAddOp op = *this;
  rewriter.create<MatAddOp>(op.getLoc(), op.getABank(), op.getBBank(),
                            op.getCBank(), op.getIter());
  rewriter.replaceOp(op, op.getCBank());
  return success();
}
