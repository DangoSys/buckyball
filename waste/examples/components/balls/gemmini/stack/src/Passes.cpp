//===- Passes.cpp - GemminiBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
void populateGemminiBallLinalgPatterns(RewritePatternSet &patterns);
namespace buckyball {
void populateGemminiBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureGemminiBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LinalgToGemminiPass
    : public mlir::buddy::LinalgToBallPass<LinalgToGemminiPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToGemminiPass)
  StringRef getArgument() const final {
    return "convert-linalg-to-gemmini-ball";
  }
  StringRef getDescription() const final {
    return "Lower signed INT8 Linalg contractions to GemminiBall.";
  }
  void populate(RewritePatternSet &patterns, int64_t, int64_t, int64_t) const {
    mlir::buddy::populateGemminiBallLinalgPatterns(patterns);
  }
};

class LowerGemminiPass : public mlir::buddy::LowerBallPass<LowerGemminiPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGemminiPass)
  StringRef getArgument() const final { return "lower-gemmini-ball"; }
  StringRef getDescription() const final {
    return "Lower GemminiBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateGemminiBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureGemminiBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerGemminiPass>();
  PassRegistration<LinalgToGemminiPass>();
  return true;
}();
