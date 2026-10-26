//===- Passes.cpp - Int8AddBall passes ------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateInt8AddBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                      RewritePatternSet &, bool,
                                                      int64_t);
void configureInt8AddBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerInt8AddPass : public mlir::buddy::LowerBallPass<LowerInt8AddPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerInt8AddPass)
  StringRef getArgument() const final { return "lower-int8add-ball"; }
  StringRef getDescription() const final {
    return "Lower Int8AddBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateInt8AddBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureInt8AddBallLegalizeForExportTarget(target,
                                                                        stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerInt8AddPass>();
  return true;
}();
