//===- LowerBuckyballPass.cpp - Pebble Buckyball lowering pass
//-------------===//

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
#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

static FlatSymbolRefAttr getOrInsertRushBFunction(OpBuilder &builder,
                                                  ModuleOp module,
                                                  StringRef name,
                                                  LLVM::LLVMFunctionType type) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return FlatSymbolRefAttr::get(builder.getContext(), name);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  LLVM::LLVMFuncOp::create(builder, module.getLoc(), name, type,
                           LLVM::Linkage::External, false, LLVM::CConv::C);
  return FlatSymbolRefAttr::get(builder.getContext(), name);
}

class LowerBuckyballToLLVMPass
    : public PassWrapper<LowerBuckyballToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToLLVMPass)
  LowerBuckyballToLLVMPass() = default;
  LowerBuckyballToLLVMPass(const LowerBuckyballToLLVMPass &) {}

  StringRef getArgument() const final { return "lower-buckyball"; }
  StringRef getDescription() const final {
    return "Lower Pebble Buckyball dialect ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{*this, "rushb",
                     llvm::cl::desc("Emit rushB host DMA carriers."),
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
        /*includeFuncOperandForwarding=*/false, stable, rushB);

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
    return "Lower Pebble bank-SSA and Buckyball ops to intrinsic ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(8)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{*this, "rushb",
                     llvm::cl::desc("Emit rushB host DMA carriers."),
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
        /*includeFuncOperandForwarding=*/false, stable, rushB);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(module, target, std::move(patterns),
                                      config)))
      signalPassFailure();
  }
};

class LowerBuckyballIntrinsicsToRushBPass
    : public PassWrapper<LowerBuckyballIntrinsicsToRushBPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerBuckyballIntrinsicsToRushBPass)

  StringRef getArgument() const final {
    return "lower-buckyball-intrinsics-to-rushb";
  }
  StringRef getDescription() const final {
    return "Lower Buckyball intrinsic ops to the rushB host ABI.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> intrinsicOps;
    module.walk([&](Operation *op) {
      if (isa<MsetIntrOp, RushBMvinOp, RushBMvoutOp, CustomIntrOp, FenceIntrOp>(
              op))
        intrinsicOps.push_back(op);
    });

    OpBuilder builder(&getContext());
    Type i32Type = IntegerType::get(&getContext(), 32);
    Type voidType = LLVM::LLVMVoidType::get(&getContext());

    auto call = [&](Operation *op, StringRef name, ValueRange operands) {
      SmallVector<Type> argumentTypes;
      argumentTypes.reserve(operands.size());
      for (Value operand : operands)
        argumentTypes.push_back(operand.getType());
      auto type = LLVM::LLVMFunctionType::get(voidType, argumentTypes);
      auto callee = getOrInsertRushBFunction(builder, module, name, type);
      LLVM::CallOp::create(builder, op->getLoc(), TypeRange{}, callee,
                           operands);
      op->erase();
    };

    for (Operation *op : intrinsicOps) {
      builder.setInsertionPoint(op);
      if (isa<MsetIntrOp>(op)) {
        call(op, "rushb_mset", op->getOperands());
        continue;
      }
      if (isa<RushBMvinOp, RushBMvoutOp>(op)) {
        call(op, isa<RushBMvinOp>(op) ? "rushb_mvin" : "rushb_mvout",
             op->getOperands());
        continue;
      }
      if (auto custom = dyn_cast<CustomIntrOp>(op)) {
        Value funct7 = LLVM::ConstantOp::create(
            builder, op->getLoc(), i32Type,
            builder.getI32IntegerAttr(custom.getFunct7()));
        SmallVector<Value> operands{custom.getRs1(), custom.getRs2(), funct7};
        call(op, "rushb_custom", operands);
        continue;
      }
      if (auto fence = dyn_cast<FenceIntrOp>(op)) {
        Value funct7 = LLVM::ConstantOp::create(builder, op->getLoc(), i32Type,
                                                builder.getI32IntegerAttr(0));
        SmallVector<Value> operands{fence->getOperand(0), fence->getOperand(1),
                                    funct7};
        call(op, "rushb_custom", operands);
      }
    }
  }
};

} // namespace

void mlir::buddy::registerLowerBuckyballPass() {
  PassRegistration<LowerBuckyballToLLVMPass>();
}

void mlir::buddy::registerLowerBankSSAToIntrinsicsPass() {
  PassRegistration<LowerBankSSAToIntrinsicsPass>();
  PassRegistration<LowerBuckyballIntrinsicsToRushBPass>();
}
