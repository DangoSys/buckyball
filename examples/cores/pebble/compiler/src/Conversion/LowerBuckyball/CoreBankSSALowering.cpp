#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleMegaKernelToBankSSAPatterns(RewritePatternSet &patterns,
                                               bool traceMegaStages,
                                               int64_t traceMegaStageStart,
                                               int64_t traceMegaStageLimit);
void populatePebbleMegaConv2dToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleQuantizeTensorToBankSSAPatterns(RewritePatternSet &patterns);

void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns,
                                               bool traceMegaStages,
                                               int64_t traceMegaStageStart,
                                               int64_t traceMegaStageLimit) {
  populatePebbleMegaKernelToBankSSAPatterns(
      patterns, traceMegaStages, traceMegaStageStart, traceMegaStageLimit);
  populatePebbleMegaConv2dToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
  populatePebbleQuantizeTensorToBankSSAPatterns(patterns);
}
} // namespace mlir::buddy
