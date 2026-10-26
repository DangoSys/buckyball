#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankMaxPoolOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                   mlir::buddy::PhysicalBankState &) {
  BankMaxPoolOp op = *this;
  rewriter.create<MaxPoolOp>(
      op.getLoc(), op.getInBank(), op.getOutBank(), op.getIter(),
      op.getInputSideAttr(), op.getOutputSideAttr(), op.getKernelAttr(),
      op.getStrideAttr(), op.getPaddingAttr(), op.getInputBase(),
      op.getOutputBase(), op.getOutputStride(), op.getStartRowAttr(),
      op.getStartColAttr());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
