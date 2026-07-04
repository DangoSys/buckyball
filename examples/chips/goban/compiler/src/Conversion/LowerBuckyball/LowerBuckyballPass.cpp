//===- LowerBuckyballPass.cpp - Goban Buckyball lowering pass -------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

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
    return "Lower Goban Buckyball dialect ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(4096)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect,
                scf::SCFDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    LLVMTypeConverter converter(context);
    RewritePatternSet patterns(context);
    LLVMConversionTarget target(*context);

    configureBuckyballLegalizeForExportTarget(target, stable);
    target.addLegalDialect<cf::ControlFlowDialect, func::FuncDialect,
                           scf::SCFDialect>();
    populateBuckyballLegalizeForLLVMExportPatterns(
        converter, patterns, bankWidthBytes, bankDepth, bankNum,
        /*includeFuncOperandForwarding=*/false, stable);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config)))
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
    return "Lower Goban bank-SSA and Buckyball ops to intrinsic ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(4096)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, arith::ArithDialect, memref::MemRefDialect,
                scf::SCFDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    LLVMTypeConverter converter(context);
    RewritePatternSet patterns(context);
    LLVMConversionTarget target(*context);

    configureBuckyballLegalizeForExportTarget(target, stable);
    target.addLegalDialect<func::FuncDialect, scf::SCFDialect>();
    populateBuckyballLegalizeForLLVMExportPatterns(
        converter, patterns, bankWidthBytes, bankDepth, bankNum,
        /*includeFuncOperandForwarding=*/false, stable);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config)))
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
