#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankRedmuleF16Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                      mlir::buddy::PhysicalBankState &) {
  BankRedmuleF16Op op = *this;
  rewriter.create<RedmuleF16Op>(op.getLoc(), op.getA(), op.getB(), op.getC(),
                                op.getRowsAttr(), op.getKAttr(),
                                op.getOp1BaseAttr(), op.getOp2BaseAttr(),
                                op.getWrBaseAttr(), op.getFirstAttr());
  rewriter.replaceOp(op, op.getC());
  return success();
}
