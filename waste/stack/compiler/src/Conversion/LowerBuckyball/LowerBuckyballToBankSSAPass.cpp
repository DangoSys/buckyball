//===- LowerBuckyballToBankSSAPass.cpp - Buckyball bank-SSA lowering -----===//
//
// Lowers every operation implementing BankSSALowering to bank-SSA operations.
//
//===----------------------------------------------------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Trace/TraceDialect.h"

using namespace mlir;

namespace {
class BankSSALoweringPattern
    : public OpInterfaceRewritePattern<::buddy::buckyball::BankSSALowering> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(::buddy::buckyball::BankSSALowering op,
                                PatternRewriter &rewriter) const override {
    return op.lowerToBankSSA(rewriter);
  }
};

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Buckyball Buckyball ops to explicit bank-SSA ops.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::trace::BuddyTraceDialect, arith::ArithDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect, scf::SCFDialect,
                    ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::buddy::populateBankSSALoweringPatterns(patterns);
    bool emittedError = false;
    ScopedDiagnosticHandler diagnostics(
        &getContext(), [&](Diagnostic &message) {
          emittedError |= message.getSeverity() == DiagnosticSeverity::Error;
          return failure();
        });
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))) ||
        emittedError)
      signalPassFailure();
  }
};

} // namespace

void mlir::buddy::populateBankSSALoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<BankSSALoweringPattern>(patterns.getContext());
}

void mlir::buddy::registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
