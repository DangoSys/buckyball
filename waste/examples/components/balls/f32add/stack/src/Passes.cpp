//===- Passes.cpp - F32AddBall passes -------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateF32AddBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                     RewritePatternSet &, bool,
                                                     int64_t);
void configureF32AddBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerF32AddPass : public mlir::buddy::LowerBallPass<LowerF32AddPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerF32AddPass)
  StringRef getArgument() const final { return "lower-f32add-ball"; }
  StringRef getDescription() const final {
    return "Lower F32AddBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateF32AddBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureF32AddBallLegalizeForExportTarget(target,
                                                                       stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerF32AddPass>();
  return true;
}();
