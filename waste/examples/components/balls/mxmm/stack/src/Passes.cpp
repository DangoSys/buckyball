//===- Passes.cpp - MxmmBall passes ---------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
void registerNormQuantWindowPass();
namespace buckyball {
void populateMxmmBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                   RewritePatternSet &, bool,
                                                   int64_t);
void configureMxmmBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerMxmmPass : public mlir::buddy::LowerBallPass<LowerMxmmPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMxmmPass)
  StringRef getArgument() const final { return "lower-mxmm-ball"; }
  StringRef getDescription() const final {
    return "Lower MxmmBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateMxmmBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureMxmmBallLegalizeForExportTarget(target,
                                                                     stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerMxmmPass>();
  mlir::buddy::registerNormQuantWindowPass();
  return true;
}();
