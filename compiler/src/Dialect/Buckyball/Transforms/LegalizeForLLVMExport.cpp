#include "Buckyball/Transform.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball::legalize;

namespace mlir::buddy::buckyball {
#define BUCKYBALL_LEGALIZE_HOOK(BALL)                                          \
  void populate##BALL##LegalizeForLLVMExportPatterns(                          \
      LLVMTypeConverter &, RewritePatternSet &, bool, int64_t, bool);          \
  void configure##BALL##LegalizeForExportTarget(LLVMConversionTarget &, bool);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_LEGALIZE_HOOK
} // namespace mlir::buddy::buckyball

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t bankWidthBytes, int64_t bankDepth, int64_t bankNum,
    bool includeFuncOperandForwarding, bool stable, bool rushB) {
  populateBaseLegalizeForLLVMExportPatterns(
      converter, patterns, includeFuncOperandForwarding, rushB);
  for (llvm::StringRef ball : buckyball_target::getBuckyballTarget().balls) {
#define BUCKYBALL_LEGALIZE_HOOK(BALL)                                          \
  if (ball == #BALL)                                                           \
    buddy::buckyball::populate##BALL##LegalizeForLLVMExportPatterns(           \
        converter, patterns, stable, bankDepth, rushB);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_LEGALIZE_HOOK
  }
  (void)bankWidthBytes;
  (void)bankNum;
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  for (llvm::StringRef ball : buckyball_target::getBuckyballTarget().balls) {
#define BUCKYBALL_LEGALIZE_HOOK(BALL)                                          \
  if (ball == #BALL)                                                           \
    buddy::buckyball::configure##BALL##LegalizeForExportTarget(target, stable);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_LEGALIZE_HOOK
  }
}
