//===- Passes.cpp - Im2colBall passes -------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateIm2colBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                     RewritePatternSet &, bool,
                                                     int64_t);
void configureIm2colBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerIm2colPass : public mlir::buddy::LowerBallPass<LowerIm2colPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerIm2colPass)
  StringRef getArgument() const final { return "lower-im2col-ball"; }
  StringRef getDescription() const final {
    return "Lower Im2colBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateIm2colBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureIm2colBallLegalizeForExportTarget(target,
                                                                       stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerIm2colPass>();
  return true;
}();
