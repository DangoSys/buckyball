//===- ToyAssignPhysicalBankPatterns.cpp - Toy bank assignment hooks ------===//

#include "Conversion/LowerBuckyball/Patterns/ToyLowerBuckyballPatterns.h"

using namespace mlir;

void mlir::buddy::populateToyAssignPhysicalBankPatterns(
    RewritePatternSet &patterns, mlir::buddy::PhysicalBankState &state) {
  populateVectorAssignPhysicalBankPatterns(patterns, state);
  populateTransposeAssignPhysicalBankPatterns(patterns, state);
  populateIm2colAssignPhysicalBankPatterns(patterns, state);
  populateFp2IntAssignPhysicalBankPatterns(patterns, state);
  populateInt2FpAssignPhysicalBankPatterns(patterns, state);
}
