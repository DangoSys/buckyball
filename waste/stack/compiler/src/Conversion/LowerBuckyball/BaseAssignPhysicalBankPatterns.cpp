//===- BaseAssignPhysicalBankPatterns.cpp - Base bank assignment ----------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "Buckyball/BuckyballOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

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

    auto bank = state.tryAlloc(row, col);
    if (!bank) {
      InFlightDiagnostic diagnostic =
          op.emitError("assign-physical-banks: unavailable bank resources");
      diagnostic << " (request=" << row << "x" << col
                 << ", used=" << state.getUsedCount() << "/"
                 << state.getBankNum()
                 << ", private ID max=" << state.getPrivateBankMax() << ")";
      return failure();
    }

    state.createMset(rewriter, op.getLoc(), static_cast<uint64_t>(*bank), true,
                     row, col);
    rewriter.replaceOp(op, state.cstI64(rewriter, op.getLoc(), *bank));
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
    if (!bank) {
      return op.emitError(
          "assign-physical-banks: release bank id is not constant");
    }
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

class BankTransferPattern : public OpRewritePattern<BankTransferOp> {
public:
  BankTransferPattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankTransferOp>(context), state(state) {}

  LogicalResult matchAndRewrite(BankTransferOp op,
                                PatternRewriter &rewriter) const override {
    auto source = state.getConstI64(op.getSource());
    auto target = state.getConstI64(op.getTarget());
    if (!source || !target)
      return op.emitError("bank transfer requires constant virtual IDs");
    auto assigned = op.getSource().getDefiningOp<arith::ConstantOp>();
    if (!assigned || assigned->getBlock() != op->getBlock())
      return op.emitError(
          "bank transfer cannot consume a control-flow bank alias");
    for (Operation *user : op.getSource().getUsers()) {
      if (user == op.getOperation())
        continue;
      while (user && user->getBlock() != op->getBlock())
        user = user->getParentOp();
      if (!user || !user->isBeforeInBlock(op))
        return op.emitError(
            "source bank handle is used after transfer or outside its block");
    }
    if (failed(state.transfer(op, *source, *target)))
      return failure();
    rewriter.create<MsetTransferOp>(op.getLoc(), op.getSource(),
                                    op.getTarget());
    rewriter.replaceOp(op, state.cstI64(rewriter, op.getLoc(), *target));
    return success();
  }

private:
  PhysicalBankState &state;
};

class BankKernelPattern : public OpRewritePattern<BankKernelOp> {
public:
  BankKernelPattern(MLIRContext *context, PhysicalBankState &state)
      : OpRewritePattern<BankKernelOp>(context), state(state) {}
  LogicalResult matchAndRewrite(BankKernelOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(
            state.verifyKernelHandles(op, op.getReadBank(), op.getWriteBank())))
      return failure();
    auto module = op->getParentOfType<ModuleOp>();
    StringRef name = op.getCallee();
    SmallVector<Type> types(op.getOperandTypes());
    auto signature = rewriter.getFunctionType(types, TypeRange{});
    auto callee = module.lookupSymbol<func::FuncOp>(name);
    if (callee && callee.getFunctionType() != signature)
      return op.emitError(
          "bank kernel callee type does not match its operands");
    if (!callee) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto function =
          rewriter.create<func::FuncOp>(op.getLoc(), name, signature);
      function.setPrivate();
      function->setAttr("llvm.emit_c_interface", rewriter.getUnitAttr());
    }
    rewriter.create<func::CallOp>(op.getLoc(), name, TypeRange{},
                                  op.getOperands());
    rewriter.replaceOp(op, ValueRange{op.getReadBank(), op.getWriteBank()});
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
                            op.getDepth(), op.getStride(), op.getGroupAttr());
    rewriter.replaceOp(op, op.getBank());
    return success();
  }
};

class BankMvin2dPattern : public OpRewritePattern<BankMvin2dOp> {
public:
  using OpRewritePattern<BankMvin2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BankMvin2dOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<Mvin2dOp>(op.getLoc(), op.getInput(), op.getBank(),
                              op.getHeight(), op.getPixelBytes(),
                              op.getSourceWidth(), op.getDstBase(),
                              op.getWidth(), op.getValidBytes());
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
                             op.getDepth(), op.getStride(), op.getGroupAttr());
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
  patterns.add<BankAllocPattern, BankReleasePattern, BankTransferPattern,
               BankKernelPattern>(patterns.getContext(), state);
  patterns.add<BankMvinPattern, BankMvin2dPattern, BankMvoutPattern>(
      patterns.getContext());
}

} // namespace mlir::buddy
