//===- AssignPhysicalBanks.cpp - ToInt8 bank assignment -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankQuantF32ToI8Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                        mlir::buddy::PhysicalBankState &) {
  BankQuantF32ToI8Op op = *this;
  rewriter.create<QuantF32ToI8Op>(op.getLoc(), op.getInBank(), op.getOutBank(),
                                  op.getIter(), op.getScale());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}

LogicalResult
BankQuantI32ToI8Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                        mlir::buddy::PhysicalBankState &) {
  BankQuantI32ToI8Op op = *this;
  rewriter.create<QuantI32ToI8Op>(
      op.getLoc(), op.getInBank(), op.getScaleBank(), op.getOutBank(),
      op.getIter(), op.getOutputBase(), op.getInputBase(),
      op.getOutputWidthAttr(), op.getOutputHeightAttr(),
      op.getOutputStrideAttr(), op.getReluAttr());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
