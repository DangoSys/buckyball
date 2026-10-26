#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankMxfp2IntOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                    mlir::buddy::PhysicalBankState &) {
  BankMxfp2IntOp op = *this;
  rewriter.create<Mxfp2IntOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                              op.getIter(), op.getSpecial());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
