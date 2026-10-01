#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleMegaConv2dToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);

void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns,
                                               bool traceMegaStages,
                                               int64_t traceMegaStageStart,
                                               int64_t traceMegaStageLimit) {
  populateMatmulRegionToBankSSAPatterns(
      patterns, traceMegaStages, traceMegaStageStart, traceMegaStageLimit);
  populatePebbleMegaConv2dToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
  populateQuantizeTensorToBankSSAPatterns(patterns);
}
} // namespace mlir::buddy
