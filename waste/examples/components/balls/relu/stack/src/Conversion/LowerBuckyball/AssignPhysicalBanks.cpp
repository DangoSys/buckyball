#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;

LogicalResult
BankReluOp::assignPhysicalBanks(PatternRewriter &rewriter,
                                mlir::buddy::PhysicalBankState &) {
  BankReluOp op = *this;
  rewriter.create<ReluOp>(op.getLoc(), op.getBank(), op.getGroup(),
                          op.getIter(), op.getStride());
  rewriter.replaceOp(op, op.getBank());
  return success();
}
