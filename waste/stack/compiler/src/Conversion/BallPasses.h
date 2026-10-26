//===- BallPasses.h - Passes a Ball maintains ----------------------------===//
//
// A Ball owns two passes: one recognizes the Linalg operations it executes,
// the other lowers its operations to LLVM intrinsics. Bank lowering belongs to
// the framework, which a Ball joins through the interfaces in
// BuckyballInterfaces.td.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_CONVERSION_BALL_PASSES_H
#define BUDDY_CONVERSION_BALL_PASSES_H

#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"

#include <functional>
#include <memory>

namespace mlir::buddy {

// Claims the Linalg operations a Ball executes. Derived provides
// `void populate(RewritePatternSet &, int64_t bankWidthBytes,
// int64_t bankDepth, int64_t bankNum) const`. Operations it leaves stay in
// Linalg for the CPU; a pattern that rejects an operation it claims emits an
// error, which fails the pass.
template <typename Derived>
class LinalgToBallPass : public PassWrapper<Derived, OperationPass<ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::buckyball::BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    const auto &config = buckyball_target::getBuckyballTarget();
    MLIRContext *context = &this->getContext();
    RewritePatternSet patterns(context);
    static_cast<const Derived *>(this)->populate(
        patterns, config.bankWidthBits / 8, config.bankDepth, config.bankNum);
    // Rewrite only what the Ball claims; leave the rest of the IR as it is.
    GreedyRewriteConfig rewrite;
    rewrite.enableFolding(false);
    rewrite.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    bool emittedError = false;
    ScopedDiagnosticHandler diagnostics(context, [&](Diagnostic &message) {
      emittedError |= message.getSeverity() == DiagnosticSeverity::Error;
      return failure();
    });
    if (failed(applyPatternsGreedily(this->getOperation(), std::move(patterns),
                                     rewrite)) ||
        emittedError)
      this->signalPassFailure();
  }
};

// Lowers a Ball's operations to LLVM intrinsics. Derived provides
// `void populate(LLVMTypeConverter &, RewritePatternSet &,
// LLVMConversionTarget &, bool stable, int64_t bankDepth) const` and marks its
// operations illegal.
template <typename Derived>
class LowerBallPass : public PassWrapper<Derived, OperationPass<ModuleOp>> {
public:
  LowerBallPass() = default;
  LowerBallPass(const LowerBallPass &other)
      : PassWrapper<Derived, OperationPass<ModuleOp>>(other) {}

  Pass::Option<bool> stable{*this, "stable",
                            llvm::cl::desc("Use stable LLVM Buckyball "
                                           "intrinsics."),
                            llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cf::ControlFlowDialect, LLVM::LLVMDialect,
                    arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    LLVMTypeConverter converter(context);
    RewritePatternSet patterns(context);
    LLVMConversionTarget target(*context);
    // Intrinsics and the operations of other Balls stay; Derived marks its
    // own operations illegal.
    target.addLegalDialect<cf::ControlFlowDialect, func::FuncDialect,
                           scf::SCFDialect, memref::MemRefDialect,
                           arith::ArithDialect,
                           ::buddy::buckyball::BuckyballDialect>();
    static_cast<const Derived *>(this)->populate(
        converter, patterns, target, stable,
        buckyball_target::getBuckyballTarget().bankDepth);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(this->getOperation(), target,
                                      std::move(patterns), config)))
      this->signalPassFailure();
  }
};

// `lower-buckyball` runs the framework lowering and then every Ball lowering
// pass registered here.
void addBallLowering(std::function<std::unique_ptr<Pass>(bool stable)> create);

template <typename BallPass> void registerBallLoweringPass() {
  PassRegistration<BallPass>();
  addBallLowering([](bool stable) {
    auto pass = std::make_unique<BallPass>();
    pass->stable = stable;
    return pass;
  });
}

} // namespace mlir::buddy

#endif // BUDDY_CONVERSION_BALL_PASSES_H
