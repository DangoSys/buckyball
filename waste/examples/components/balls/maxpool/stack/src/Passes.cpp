//===- Passes.cpp - MaxPoolBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateMaxPoolBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureMaxPoolBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerMaxPoolPass : public mlir::buddy::LowerBallPass<LowerMaxPoolPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMaxPoolPass)
  StringRef getArgument() const final { return "lower-maxpool-ball"; }
  StringRef getDescription() const final {
    return "Lower MaxPoolBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateMaxPoolBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureMaxPoolBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerMaxPoolPass>();
  return true;
}();
