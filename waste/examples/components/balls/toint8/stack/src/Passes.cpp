//===- Passes.cpp - ToInt8Ball passes -------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
void populateToInt8BallLinalgPatterns(RewritePatternSet &patterns);
namespace buckyball {
void populateToInt8BallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                     RewritePatternSet &, bool,
                                                     int64_t);
void configureToInt8BallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LinalgToToInt8Pass
    : public mlir::buddy::LinalgToBallPass<LinalgToToInt8Pass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToToInt8Pass)
  StringRef getArgument() const final {
    return "convert-linalg-to-toint8-ball";
  }
  StringRef getDescription() const final {
    return "Lower the Linalg operations ToInt8Ball executes.";
  }
  void populate(RewritePatternSet &patterns, int64_t, int64_t bankDepth,
                int64_t) const {
    mlir::buddy::populateToInt8BallLinalgPatterns(patterns);
  }
};

class LowerToInt8Pass : public mlir::buddy::LowerBallPass<LowerToInt8Pass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToInt8Pass)
  StringRef getArgument() const final { return "lower-toint8-ball"; }
  StringRef getDescription() const final {
    return "Lower ToInt8Ball operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateToInt8BallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureToInt8BallLegalizeForExportTarget(target,
                                                                       stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerToInt8Pass>();
  PassRegistration<LinalgToToInt8Pass>();
  return true;
}();
