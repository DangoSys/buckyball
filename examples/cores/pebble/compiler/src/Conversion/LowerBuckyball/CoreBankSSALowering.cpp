#include "Conversion/LowerBuckyball/LowerBuckyball.h"

using namespace mlir;

namespace mlir::buddy {
void populatePebbleIm2colMatmulToBankSSAPatterns(RewritePatternSet &patterns);
void populatePebbleMemTransposeToBankSSAPatterns(RewritePatternSet &patterns);

void populatePebbleCoreBankSSALoweringPatterns(RewritePatternSet &patterns) {
  populatePebbleIm2colMatmulToBankSSAPatterns(patterns);
  populatePebbleMemTransposeToBankSSAPatterns(patterns);
}
} // namespace mlir::buddy
