//===- Passes.cpp - MxquantBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateMxquantBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureMxquantBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerMxquantPass : public mlir::buddy::LowerBallPass<LowerMxquantPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMxquantPass)
  StringRef getArgument() const final { return "lower-mxquant-ball"; }
  StringRef getDescription() const final {
    return "Lower MxquantBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateMxquantBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureMxquantBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerMxquantPass>();
  return true;
}();
