//===- AssignPhysicalBanks.cpp - Vector bank assignment -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankVecMat16Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                    mlir::buddy::PhysicalBankState &) {
  BankVecMat16Op op = *this;
  rewriter.create<VecMat16Op>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                              op.getWrBank(), op.getIter(), op.getMode());
  rewriter.replaceOp(op, op.getWrBank());
  return success();
}
