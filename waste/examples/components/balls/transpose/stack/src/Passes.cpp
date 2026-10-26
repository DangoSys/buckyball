//===- Passes.cpp - TransposeBall passes ----------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
void populateTransposeBallLinalgPatterns(RewritePatternSet &patterns);
namespace buckyball {
void populateTransposeBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                        RewritePatternSet &,
                                                        bool, int64_t);
void configureTransposeBallLegalizeForExportTarget(LLVMConversionTarget &,
                                                   bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LinalgToTransposePass
    : public mlir::buddy::LinalgToBallPass<LinalgToTransposePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToTransposePass)
  StringRef getArgument() const final {
    return "convert-linalg-to-transpose-ball";
  }
  StringRef getDescription() const final {
    return "Lower the Linalg operations TransposeBall executes.";
  }
  void populate(RewritePatternSet &patterns, int64_t, int64_t bankDepth,
                int64_t) const {
    mlir::buddy::populateTransposeBallLinalgPatterns(patterns);
  }
};

class LowerTransposePass
    : public mlir::buddy::LowerBallPass<LowerTransposePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTransposePass)
  StringRef getArgument() const final { return "lower-transpose-ball"; }
  StringRef getDescription() const final {
    return "Lower TransposeBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateTransposeBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureTransposeBallLegalizeForExportTarget(
        target, stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerTransposePass>();
  PassRegistration<LinalgToTransposePass>();
  return true;
}();
