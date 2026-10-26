#include "Conversion/BallPasses.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace {
// Lowers the framework operations: bank configuration, data movement and raw
// instructions. Ball operations are lowered by their own passes.
class LowerFrameworkPass
    : public mlir::buddy::LowerBallPass<LowerFrameworkPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFrameworkPass)

  StringRef getArgument() const final { return "lower-buckyball-framework"; }
  StringRef getDescription() const final {
    return "Lower Buckyball framework operations to LLVM intrinsics.";
  }

  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool, int64_t) const {
    ::buddy::buckyball::legalize::populateBaseLegalizeForLLVMExportPatterns(
        converter, patterns, /*includeFuncOperandForwarding=*/false);
    ::buddy::buckyball::legalize::configureBaseLegalizeForExportTarget(target);
  }
};

// Turns the Buckyball intrinsic operations into calls of the RISC-V LLVM
// intrinsics, so any LLVM IR translation handles the result.
class LowerIntrinsicsPass
    : public PassWrapper<LowerIntrinsicsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIntrinsicsPass)

  StringRef getArgument() const final { return "lower-buckyball-intrinsics"; }
  StringRef getDescription() const final {
    return "Lower buckyball.intr.* to llvm.call_intrinsic.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> intrinsics;
    getOperation().walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("buckyball.intr."))
        intrinsics.push_back(op);
    });
    OpBuilder builder(&getContext());
    for (Operation *op : intrinsics) {
      StringRef name = op->getName().getStringRef().drop_front(15);
      SmallVector<Value> operands(op->getOperands());
      builder.setInsertionPoint(op);
      std::string intrinsic = ("llvm.riscv.bb." + name).str();
      // The custom instruction carries its funct7 as an immediate argument.
      if (name == "custom") {
        intrinsic = "llvm.riscv.buckyball.custom";
        operands.push_back(builder.create<LLVM::ConstantOp>(
            op->getLoc(), builder.getI32Type(),
            cast<IntegerAttr>(op->getAttr("funct7"))));
      }
      builder.create<LLVM::CallIntrinsicOp>(
          op->getLoc(), builder.getStringAttr(intrinsic), operands);
      op->erase();
    }
  }
};

struct LowerBuckyballOptions
    : public PassPipelineOptions<LowerBuckyballOptions> {
  Option<bool> stable{*this, "stable",
                      llvm::cl::desc("Use stable LLVM Buckyball intrinsics."),
                      llvm::cl::init(false)};
};

std::vector<std::function<std::unique_ptr<Pass>(bool)>> &ballLowerings() {
  static std::vector<std::function<std::unique_ptr<Pass>(bool)>> lowerings;
  return lowerings;
}
} // namespace

void mlir::buddy::addBallLowering(
    std::function<std::unique_ptr<Pass>(bool stable)> create) {
  ballLowerings().push_back(std::move(create));
}

void mlir::buddy::registerBuckyballPasses() {
  registerLowerLinalgToBuckyballPass();
  registerLowerBuckyballToBankSSAPass();
  registerAssignPhysicalBanksPass();
  PassRegistration<LowerFrameworkPass>();
  PassRegistration<LowerIntrinsicsPass>();
  PassPipelineRegistration<LowerBuckyballOptions>(
      "lower-buckyball",
      "Lower Buckyball framework and Ball operations to LLVM intrinsics.",
      [](OpPassManager &pm, const LowerBuckyballOptions &options) {
        auto framework = std::make_unique<LowerFrameworkPass>();
        framework->stable = options.stable;
        pm.addPass(std::move(framework));
        for (auto &create : ballLowerings())
          pm.addPass(create(options.stable));
        pm.addPass(std::make_unique<LowerIntrinsicsPass>());
      });
}
