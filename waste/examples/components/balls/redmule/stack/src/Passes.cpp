//===- Passes.cpp - RedmuleBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateRedmuleBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureRedmuleBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerRedmulePass : public mlir::buddy::LowerBallPass<LowerRedmulePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRedmulePass)
  StringRef getArgument() const final { return "lower-redmule-ball"; }
  StringRef getDescription() const final {
    return "Lower RedmuleBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateRedmuleBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureRedmuleBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerRedmulePass>();
  return true;
}();
