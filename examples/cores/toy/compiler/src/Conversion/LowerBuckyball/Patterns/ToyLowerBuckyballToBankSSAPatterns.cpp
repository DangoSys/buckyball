//===- ToyLowerBuckyballToBankSSAPatterns.cpp - Toy bank SSA hooks --------===//

#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;

namespace mlir::buddy {
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                          \
  void populate##BALL##LowerBuckyballToBankSSAPatterns(RewritePatternSet &);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
} // namespace mlir::buddy

void mlir::buddy::populateToyLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  for (llvm::StringRef ball : buckyball_target::getBuckyballTarget().balls) {
#define BUCKYBALL_BANK_SSA_HOOK(BALL)                                          \
  if (ball == #BALL)                                                           \
    populate##BALL##LowerBuckyballToBankSSAPatterns(patterns);
#include "BuckyballBallLoweringHooks.inc"
#undef BUCKYBALL_BANK_SSA_HOOK
  }
}
