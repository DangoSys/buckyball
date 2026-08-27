#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;

namespace mlir::buddy {
void populateSMatMulBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns);
void populateReluBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns);
void populatePebbleIm2colMatmulToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);

void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns) {
  populateSMatMulBallLowerBuckyballToBankSSAPatterns(patterns);
  populateReluBallLowerBuckyballToBankSSAPatterns(patterns);
  populatePebbleIm2colMatmulToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
}
} // namespace mlir::buddy
