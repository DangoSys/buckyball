//===- AssignPhysicalBanksPass.cpp - Toy physical bank assignment ---------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"

using namespace mlir;

namespace {

class AssignPhysicalBanksPass
    : public PassWrapper<AssignPhysicalBanksPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignPhysicalBanksPass)
  AssignPhysicalBanksPass() = default;
  AssignPhysicalBanksPass(const AssignPhysicalBanksPass &) {}

  StringRef getArgument() const final { return "assign-physical-banks"; }
  StringRef getDescription() const final {
    return "Assign physical banks for Toy bank-SSA ops.";
  }

  Option<int64_t> bankNum{*this, "bank_num",
                          llvm::cl::desc("Number of physical banks."),
                          llvm::cl::init(16)};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (bankNum <= 0) {
      func.emitError("assign-physical-banks: bank_num must be > 0");
      signalPassFailure();
      return;
    }

    mlir::buddy::PhysicalBankState state(bankNum);
    RewritePatternSet patterns(&getContext());
    mlir::buddy::addBaseAssignPhysicalBankPatterns(patterns, state);
    mlir::buddy::populateToyAssignPhysicalBankPatterns(patterns, state);

    walkAndApplyPatterns(func, std::move(patterns));
    if (failed(mlir::buddy::verifyNoBankSSAOps(func))) {
      signalPassFailure();
      return;
    }

    if (!state.empty()) {
      func.emitError("assign-physical-banks: leaked virtual bank handles");
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::buddy::registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
