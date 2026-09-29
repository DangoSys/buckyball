//===- ToyLowerBuckyballToBankSSAPatterns.cpp - Toy bank SSA hooks --------===//

#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"

using namespace mlir;

namespace mlir::buddy {
void populateGemminiBallLowerBuckyballToBankSSAPatterns(RewritePatternSet &);
}

void mlir::buddy::populateToyLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  populateGemminiBallLowerBuckyballToBankSSAPatterns(patterns);
}
