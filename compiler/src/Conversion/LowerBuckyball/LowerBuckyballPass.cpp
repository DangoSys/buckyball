#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"

using namespace mlir;

namespace {
class LowerBuckyballToLLVMPass
    : public PassWrapper<LowerBuckyballToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToLLVMPass)
  LowerBuckyballToLLVMPass() = default;
  LowerBuckyballToLLVMPass(const LowerBuckyballToLLVMPass &) {}

  StringRef getArgument() const final { return "lower-buckyball"; }
  StringRef getDescription() const final {
    return "Lower Buckyball operations for the selected target.";
  }

  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{*this, "rushb",
                     llvm::cl::desc("Lower DMA operations to the rushB ABI."),
                     llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect,
                scf::SCFDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    const buckyball_target::BuckyballTargetConfig &targetConfig =
        buckyball_target::getBuckyballTarget();
    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    configureBuckyballLegalizeForExportTarget(target, stable);
    target.addLegalDialect<cf::ControlFlowDialect, func::FuncDialect,
                           scf::SCFDialect>();
    populateBuckyballLegalizeForLLVMExportPatterns(
        converter, patterns, targetConfig.bankWidthBits / 8,
        targetConfig.bankDepth, targetConfig.bankNum,
        /*includeFuncOperandForwarding=*/false, stable, rushB);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      signalPassFailure();
  }
};

class LowerBankSSAToIntrinsicsPass
    : public PassWrapper<LowerBankSSAToIntrinsicsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBankSSAToIntrinsicsPass)
  LowerBankSSAToIntrinsicsPass() = default;
  LowerBankSSAToIntrinsicsPass(const LowerBankSSAToIntrinsicsPass &) {}

  StringRef getArgument() const final { return "lower-bank-ssa-to-intrinsics"; }
  StringRef getDescription() const final {
    return "Lower bank-SSA operations for the selected Buckyball target.";
  }

  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{*this, "rushb",
                     llvm::cl::desc("Lower DMA operations to the rushB ABI."),
                     llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect,
                scf::SCFDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    const buckyball_target::BuckyballTargetConfig &targetConfig =
        buckyball_target::getBuckyballTarget();
    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    configureBuckyballLegalizeForExportTarget(target, stable);
    target.addLegalDialect<func::FuncDialect, scf::SCFDialect>();
    populateBuckyballLegalizeForLLVMExportPatterns(
        converter, patterns, targetConfig.bankWidthBits / 8,
        targetConfig.bankDepth, targetConfig.bankNum,
        /*includeFuncOperandForwarding=*/false, stable, rushB);
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerBuckyballPass() {
  PassRegistration<LowerBuckyballToLLVMPass>();
}

void mlir::buddy::registerLowerBankSSAToIntrinsicsPass() {
  PassRegistration<LowerBankSSAToIntrinsicsPass>();
}
