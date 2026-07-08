//===- BaseAssignPhysicalBankPatterns.cpp - Base bank assignment ----------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace mlir::buddy;
using namespace ::buddy::buckyball;

namespace {

class BankAllocPattern : public OpRewritePattern<BankAllocOp> {
public:
  BankAllocPattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankAllocOp>(context), state(state) {}

  LogicalResult matchAndRewrite(BankAllocOp op,
                                PatternRewriter &rewriter) const override {
    int64_t row = op.getRow();
    int64_t col = op.getCol();
    if (row <= 0 || col <= 0)
      return op.emitError("assign-physical-banks: invalid bank shape");

    auto base = state.tryAlloc(row, col);
    if (!base)
      return op.emitError("assign-physical-banks: out of physical banks");

    state.createMset(rewriter, op.getLoc(), static_cast<uint64_t>(*base), true,
                     row, col);
    state.remember(*base, row, col);
    rewriter.replaceOp(op, state.cstI64(rewriter, op.getLoc(), *base));
    return success();
  }

private:
  PhysicalBankState &state;
};

class BankReleasePattern : public OpRewritePattern<BankReleaseOp> {
public:
  BankReleasePattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankReleaseOp>(context), state(state) {}

  LogicalResult matchAndRewrite(BankReleaseOp op,
                                PatternRewriter &rewriter) const override {
    auto bank = state.getConstI64(op.getBank());
    if (!bank)
      return op.emitError(
          "assign-physical-banks: release bank id is not constant");
    if (failed(state.release(op, *bank)))
      return failure();

    state.createMset(rewriter, op.getLoc(), static_cast<uint64_t>(*bank), false,
                     0, 0);
    rewriter.eraseOp(op);
    return success();
  }

private:
  PhysicalBankState &state;
};

class BankMvinPattern : public OpRewritePattern<BankMvinOp> {
public:
  using OpRewritePattern<BankMvinOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMvinOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MvinOp>(op.getLoc(), op.getInput(), op.getBank(),
                            op.getDepth(), op.getStride());
    rewriter.replaceOp(op, op.getBank());
    return success();
  }
};

class BankMvoutPattern : public OpRewritePattern<BankMvoutOp> {
public:
  using OpRewritePattern<BankMvoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMvoutOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<MvoutOp>(op.getLoc(), op.getOutput(), op.getBank(),
                             op.getDepth(), op.getStride());
    rewriter.replaceOp(op, op.getBank());
    return success();
  }
};

} // namespace

namespace mlir::buddy {

LogicalResult verifyNoBankSSAOps(Operation *root) {
  Operation *badOp = nullptr;
  root->walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("buckyball.bank_")) {
      badOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!badOp)
    return success();
  return badOp->emitError("assign-physical-banks: unsupported bank op");
}

void addBaseAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                       PhysicalBankState &state) {
  patterns.add<BankAllocPattern, BankReleasePattern>(patterns.getContext(),
                                                     state);
  patterns.add<BankMvinPattern, BankMvoutPattern>(patterns.getContext());
}

} // namespace mlir::buddy
