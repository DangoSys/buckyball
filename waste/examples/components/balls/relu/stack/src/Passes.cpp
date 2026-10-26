//===- Passes.cpp - ReluBall passes ---------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
void populateReluBallLinalgPatterns(RewritePatternSet &patterns);
namespace buckyball {
void populateReluBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                   RewritePatternSet &, bool,
                                                   int64_t);
void configureReluBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LinalgToReluPass
    : public mlir::buddy::LinalgToBallPass<LinalgToReluPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToReluPass)
  StringRef getArgument() const final { return "convert-linalg-to-relu-ball"; }
  StringRef getDescription() const final {
    return "Lower the Linalg operations ReluBall executes.";
  }
  void populate(RewritePatternSet &patterns, int64_t, int64_t bankDepth,
                int64_t) const {
    mlir::buddy::populateReluBallLinalgPatterns(patterns);
  }
};

class LowerReluPass : public mlir::buddy::LowerBallPass<LowerReluPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerReluPass)
  StringRef getArgument() const final { return "lower-relu-ball"; }
  StringRef getDescription() const final {
    return "Lower ReluBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateReluBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureReluBallLegalizeForExportTarget(target,
                                                                     stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerReluPass>();
  PassRegistration<LinalgToReluPass>();
  return true;
}();
