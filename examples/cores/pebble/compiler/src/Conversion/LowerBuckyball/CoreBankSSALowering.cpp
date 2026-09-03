#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleMegaKernelToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMegaConv2dToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleQuantizeTensorToBankSSAPatterns(RewritePatternSet &patterns);

void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns) {
  populatePebbleMegaKernelToBankSSAPatterns(patterns);
  populatePebbleMegaConv2dToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
  populatePebbleQuantizeTensorToBankSSAPatterns(patterns);
}
} // namespace mlir::buddy
