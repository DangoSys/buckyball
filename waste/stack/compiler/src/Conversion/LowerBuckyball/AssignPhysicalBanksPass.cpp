#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"

#include <functional>

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;

namespace {
class PhysicalBankAssignmentPattern
    : public OpInterfaceRewritePattern<
          ::buddy::buckyball::PhysicalBankAssignment> {
public:
  PhysicalBankAssignmentPattern(MLIRContext *context,
                                mlir::buddy::PhysicalBankState &state)
      : OpInterfaceRewritePattern(context), state(state) {}

  LogicalResult matchAndRewrite(::buddy::buckyball::PhysicalBankAssignment op,
                                PatternRewriter &rewriter) const override {
    return op.assignPhysicalBanks(rewriter, state);
  }

private:
  mlir::buddy::PhysicalBankState &state;
};

class AssignPhysicalBanksPass
    : public PassWrapper<AssignPhysicalBanksPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignPhysicalBanksPass)

  StringRef getArgument() const final { return "assign-physical-banks"; }
  StringRef getDescription() const final {
    return "Assign physical banks for the selected Buckyball target.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, ::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    const buckyball_target::BuckyballTargetConfig &target =
        buckyball_target::getBuckyballTarget();
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      mlir::buddy::PhysicalBankState state(target.bankNum,
                                           target.privateBankMax);
      RewritePatternSet patterns(&getContext());
      mlir::buddy::addBaseAssignPhysicalBankPatterns(patterns, state);
      patterns.add<PhysicalBankAssignmentPattern>(&getContext(), state);
      FrozenRewritePatternSet frozen(std::move(patterns));
      PatternApplicator applicator(frozen);
      applicator.applyDefaultCostModel();
      PatternRewriter rewriter(&getContext());
      bool assignmentFailed = false;
      std::function<void(Region &)> visitRegion = [&](Region &region) {
        for (Block &block : region) {
          for (auto it = block.begin(); it != block.end();) {
            if (assignmentFailed)
              return;
            Operation *op = &*it++;
            if (succeeded(applicator.matchAndRewrite(op, rewriter)))
              continue;
            if (op->getName().getStringRef().starts_with("buckyball.bank_")) {
              assignmentFailed = true;
              return;
            }
            for (Region &nested : op->getRegions())
              visitRegion(nested);
          }
        }
      };
      for (Region &region : func->getRegions())
        visitRegion(region);
      if (assignmentFailed ||
          mlir::failed(mlir::buddy::verifyNoBankSSAOps(func)) ||
          !state.empty()) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

void mlir::buddy::registerAssignPhysicalBanksPass() {
  PassRegistration<AssignPhysicalBanksPass>();
}
