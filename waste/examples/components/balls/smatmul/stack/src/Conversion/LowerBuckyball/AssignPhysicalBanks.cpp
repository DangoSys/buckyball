//===- AssignPhysicalBanks.cpp - Matrix bank assignment -------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankSMatMulOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                   mlir::buddy::PhysicalBankState &) {
  BankSMatMulOp op = *this;
  rewriter.create<SMatMulOp>(op.getLoc(), op.getOp1Bank(), op.getOp2Bank(),
                             op.getWrBank(), op.getConfig(), op.getFirst(),
                             op.getLast(), op.getOutputBase());
  rewriter.replaceOp(op, op.getWrBank());
  return success();
}

LogicalResult
BankSMatMulBiasOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                       mlir::buddy::PhysicalBankState &) {
  BankSMatMulBiasOp op = *this;
  rewriter.create<SMatMulBiasOp>(op.getLoc(), op.getBiasBank(),
                                 op.getInputBase());
  rewriter.replaceOp(op, op.getBiasBank());
  return success();
}
