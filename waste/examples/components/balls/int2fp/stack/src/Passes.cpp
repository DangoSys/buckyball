//===- Passes.cpp - Int2FpBall passes -------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateInt2FpBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                     RewritePatternSet &, bool,
                                                     int64_t);
void configureInt2FpBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerInt2FpPass : public mlir::buddy::LowerBallPass<LowerInt2FpPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerInt2FpPass)
  StringRef getArgument() const final { return "lower-int2fp-ball"; }
  StringRef getDescription() const final {
    return "Lower Int2FpBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateInt2FpBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureInt2FpBallLegalizeForExportTarget(target,
                                                                       stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerInt2FpPass>();
  return true;
}();
