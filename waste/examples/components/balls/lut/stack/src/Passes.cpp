//===- Passes.cpp - LutBall passes ----------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateLutBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                  RewritePatternSet &, bool,
                                                  int64_t);
void configureLutBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerLutPass : public mlir::buddy::LowerBallPass<LowerLutPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLutPass)
  StringRef getArgument() const final { return "lower-lut-ball"; }
  StringRef getDescription() const final {
    return "Lower LutBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateLutBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureLutBallLegalizeForExportTarget(target,
                                                                    stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerLutPass>();
  return true;
}();
