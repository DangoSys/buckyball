//===- AssignPhysicalBanks.cpp - Im2col bank assignment -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankIm2colOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                  mlir::buddy::PhysicalBankState &) {
  BankIm2colOp op = *this;
  rewriter.create<Im2colOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                            op.getIter(), op.getKsize(), op.getStride(),
                            op.getPadding(), op.getInputBase(), op.getLane(),
                            op.getStartRowAttr(), op.getStartColAttr(),
                            op.getWindowStartAttr(), op.getWindowCountAttr());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
