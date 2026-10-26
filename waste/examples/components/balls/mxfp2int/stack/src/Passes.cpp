//===- Passes.cpp - Mxfp2IntBall passes -----------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateMxfp2IntBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                       RewritePatternSet &,
                                                       bool, int64_t);
void configureMxfp2IntBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerMxfp2IntPass : public mlir::buddy::LowerBallPass<LowerMxfp2IntPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMxfp2IntPass)
  StringRef getArgument() const final { return "lower-mxfp2int-ball"; }
  StringRef getDescription() const final {
    return "Lower Mxfp2IntBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateMxfp2IntBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureMxfp2IntBallLegalizeForExportTarget(
        target, stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerMxfp2IntPass>();
  return true;
}();
