//===- AssignPhysicalBanks.cpp - Transpose bank assignment ----------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankTransposeOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                     mlir::buddy::PhysicalBankState &) {
  BankTransposeOp op = *this;
  rewriter.create<TransposeOp>(op.getLoc(), op.getInBank(), op.getOutBank(),
                               op.getIter(), op.getElemBits());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
