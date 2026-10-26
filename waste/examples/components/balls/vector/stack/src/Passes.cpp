//===- Passes.cpp - VecBall passes ----------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateVecBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                  RewritePatternSet &, bool,
                                                  int64_t);
void configureVecBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerVecPass : public mlir::buddy::LowerBallPass<LowerVecPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVecPass)
  StringRef getArgument() const final { return "lower-vector-ball"; }
  StringRef getDescription() const final {
    return "Lower VecBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateVecBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureVecBallLegalizeForExportTarget(target,
                                                                    stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerVecPass>();
  return true;
}();
