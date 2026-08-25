//===- ToyLowerBuckyballPatterns.h - Toy lowering pattern hooks -*- C++ -*-===//

#ifndef TOY_CONVERSION_LOWER_BUCKYBALL_PATTERNS_H
#define TOY_CONVERSION_LOWER_BUCKYBALL_PATTERNS_H

#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"

namespace mlir {
namespace buddy {

void populateToyAssignPhysicalBankPatterns(RewritePatternSet &patterns,
                                           PhysicalBankState &state);

void populateToyLowerBuckyballToLLVMPatterns(RewritePatternSet &patterns);
void configureToyLowerBuckyballToLLVMTarget(LLVMConversionTarget &target);
void populateToyLowerBuckyballToBankSSAPatterns(RewritePatternSet &patterns);

} // namespace buddy
} // namespace mlir

#endif // TOY_CONVERSION_LOWER_BUCKYBALL_PATTERNS_H
