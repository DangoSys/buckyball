//===- Passes.cpp - TraceBall passes --------------------------------------===//

#include "Conversion/BallPasses.h"

using namespace mlir;

namespace mlir::buddy {
namespace buckyball {
void populateTraceBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &,
                                                    RewritePatternSet &, bool,
                                                    int64_t);
void configureTraceBallLegalizeForExportTarget(LLVMConversionTarget &, bool);
} // namespace buckyball
} // namespace mlir::buddy

namespace {
class LowerTracePass : public mlir::buddy::LowerBallPass<LowerTracePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTracePass)
  StringRef getArgument() const final { return "lower-trace-ball"; }
  StringRef getDescription() const final {
    return "Lower TraceBall operations to LLVM intrinsics.";
  }
  void populate(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                LLVMConversionTarget &target, bool stable,
                int64_t bankDepth) const {
    mlir::buddy::buckyball::populateTraceBallLegalizeForLLVMExportPatterns(
        converter, patterns, stable, bankDepth);
    mlir::buddy::buckyball::configureTraceBallLegalizeForExportTarget(target,
                                                                      stable);
  }
};
} // namespace

// Registered when the compiler plugin is loaded.
[[maybe_unused]] static const bool registered = [] {
  mlir::buddy::registerBallLoweringPass<LowerTracePass>();
  return true;
}();
