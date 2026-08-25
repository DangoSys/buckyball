#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;

namespace mlir::buddy {
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                        \
  void populate##BALL##LowerBuckyballToBankSSAPatterns(RewritePatternSet &);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
#define BUCKYBALL_CORE_BANK_SSA_HOOK(CORE, NAME)                            \
  void populate##CORE##CoreBankSSALoweringPatterns(RewritePatternSet &);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_CORE_BANK_SSA_HOOK
} // namespace mlir::buddy

namespace {
class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)

  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower Buckyball operations to explicit bank-SSA operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<::buddy::buckyball::BuckyballDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    for (llvm::StringRef ball :
         buckyball_target::getBuckyballTarget().balls) {
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                        \
  if (ball == #BALL)                                                          \
    mlir::buddy::populate##BALL##LowerBuckyballToBankSSAPatterns(patterns);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
    }
#define BUCKYBALL_CORE_BANK_SSA_HOOK(CORE, NAME)                            \
    if (buckyball_target::getBuckyballTarget().core == NAME)                \
      mlir::buddy::populate##CORE##CoreBankSSALoweringPatterns(patterns);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_CORE_BANK_SSA_HOOK
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

void mlir::buddy::registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
