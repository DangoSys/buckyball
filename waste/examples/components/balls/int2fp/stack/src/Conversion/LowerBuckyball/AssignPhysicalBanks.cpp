//===- AssignPhysicalBanks.cpp - Int2Fp bank assignment -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankInt32ToFp32Op::assignPhysicalBanks(PatternRewriter &rewriter,
                                       mlir::buddy::PhysicalBankState &) {
  BankInt32ToFp32Op op = *this;
  rewriter.create<Int32ToFp32Op>(op.getLoc(), op.getInBank(), op.getScaleBank(),
                                 op.getOutBank(), op.getIter(),
                                 op.getReluAttr());
  rewriter.replaceOp(op, op.getOutBank());
  return success();
}
