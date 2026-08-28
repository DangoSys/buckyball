//===- LowerBuckyballPass.cpp - Decode Buckyball lowering pass ------------===//

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

class LowerBuckyballToLLVMPass
    : public PassWrapper<LowerBuckyballToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToLLVMPass)
  LowerBuckyballToLLVMPass() = default;
  LowerBuckyballToLLVMPass(const LowerBuckyballToLLVMPass &) {}

  StringRef getArgument() const final { return "lower-buckyball"; }
  StringRef getDescription() const final {
    return "Lower Decode Buckyball dialect ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(20)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{
      *this, "rushb",
      llvm::cl::desc("Lower DMA operations to the rushB host ABI."),
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
    return "Lower Decode bank-SSA and Buckyball ops to intrinsic ops.";
  }

  Option<int64_t> bankWidthBytes{
      *this, "bank_width", llvm::cl::desc("Physical bank width in bytes."),
      llvm::cl::init(16)};
  Option<int64_t> bankDepth{*this, "bank_depth",
                            llvm::cl::desc("Depth of each bank."),
                            llvm::cl::init(1024)};
  Option<int64_t> bankNum{*this, "bank_num", llvm::cl::desc("Number of banks."),
                          llvm::cl::init(20)};
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
  Option<bool> rushB{
      *this, "rushb",
      llvm::cl::desc("Lower DMA operations to the rushB host ABI."),
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

class LowerBuckyballIntrinsicsToRushBPass
    : public PassWrapper<LowerBuckyballIntrinsicsToRushBPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LowerBuckyballIntrinsicsToRushBPass)
  LowerBuckyballIntrinsicsToRushBPass() = default;
  LowerBuckyballIntrinsicsToRushBPass(
      const LowerBuckyballIntrinsicsToRushBPass &) {}

  StringRef getArgument() const final {
    return "lower-buckyball-intrinsics-to-rushb";
  }
  StringRef getDescription() const final {
    return "Lower Decode Buckyball intrinsic ops to the rushB host ABI.";
  }

  Option<int64_t> coreId{
      *this, "core_id",
      llvm::cl::desc("Bind generated rushB calls to a tile Core."),
      llvm::cl::init(-1)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, BuckyballDialect>();
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
    FlatSymbolRefAttr selectCallee;
    if (coreId >= 0) {
      auto selectType =
          LLVM::LLVMFunctionType::get(voidType, {i32Type, i32Type});
      selectCallee = getOrInsertRushBFunction(
          builder, module, "rushb_select_accelerator", selectType);
    }
    llvm::SmallPtrSet<func::FuncOp, 8> boundFunctions;
    for (Operation *op : intrinsicOps) {
      if (coreId >= 0) {
        if (auto function = op->getParentOfType<func::FuncOp>();
            function && boundFunctions.insert(function).second) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&function.front());
          auto selected = LLVM::ConstantOp::create(
              builder, function.getLoc(), i32Type,
              builder.getI32IntegerAttr(static_cast<int32_t>(coreId)));
          auto chip =
              LLVM::ConstantOp::create(builder, function.getLoc(), i32Type,
                                       builder.getI32IntegerAttr(0));
          LLVM::CallOp::create(builder, function.getLoc(), TypeRange{},
                               selectCallee, ValueRange{selected, chip});
        }
      }
      builder.setInsertionPoint(op);
      SmallVector<Value> operands;
      StringRef name;
      if (isa<MsetIntrOp>(op)) {
        name = "rushb_mset";
        operands.append(op->getOperands().begin(), op->getOperands().end());
      } else if (isa<RushBMvinOp>(op) || isa<RushBMvoutOp>(op)) {
        name = isa<RushBMvinOp>(op) ? "rushb_mvin" : "rushb_mvout";
        operands.append(op->getOperands().begin(), op->getOperands().end());
      } else {
        name = "rushb_custom";
        if (auto custom = dyn_cast<CustomIntrOp>(op)) {
          operands.append(custom.getOperands().begin(),
                          custom.getOperands().end());
          operands.push_back(LLVM::ConstantOp::create(
              builder, op->getLoc(), i32Type,
              builder.getI32IntegerAttr(custom.getFunct7())));
        } else {
          auto fence = cast<FenceIntrOp>(op);
          operands.append(fence->getOperands().begin(),
                          fence->getOperands().end());
          operands.push_back(LLVM::ConstantOp::create(
              builder, op->getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        }
      }

      SmallVector<Type> argumentTypes;
      for (Value operand : operands)
        argumentTypes.push_back(operand.getType());
      auto type = LLVM::LLVMFunctionType::get(voidType, argumentTypes);
      auto callee = getOrInsertRushBFunction(builder, module, name, type);
      LLVM::CallOp::create(builder, op->getLoc(), TypeRange{}, callee,
                           operands);
      op->erase();
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
