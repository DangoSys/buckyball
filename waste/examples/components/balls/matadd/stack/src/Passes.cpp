//===- Passes.cpp - MatAddBall passes -------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateMatAddBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                     RewritePatternSet &, bool,
                                                     int64_t);
void configureMatAddBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerMatAddPass : public mlir::buddy::LowerBallPass<LowerMatAddPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMatAddPass)
  StringRef getArgument() const final { return "lower-matadd-ball"; }
  StringRef getDescription() const final {
    return "Lower MatAddBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateMatAddBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureMatAddBallLegalizeForExportTarget(target,
                                                                       stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerMatAddPass>();
  return true;
}();
