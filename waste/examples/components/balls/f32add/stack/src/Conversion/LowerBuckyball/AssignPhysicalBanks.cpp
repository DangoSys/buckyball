#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankF32AddOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                  mlir::buddy::PhysicalBankState &) {
  BankF32AddOp op = *this;
  rewriter.create<F32AddOp>(op.getLoc(), op.getA(), op.getB(), op.getC(),
                            op.getRowsAttr(), op.getGroupAttr(),
                            op.getFirstAttr());
  rewriter.replaceOp(op, op.getC());
  return success();
}
