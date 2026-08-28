//===- LegalizeForLLVMExport.cpp - Toy Buckyball LLVM lowering ------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/Transform.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball::legalize;

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t bankWidthBytes, int64_t bankDepth, int64_t bankNum,
    bool includeFuncOperandForwarding, bool stable, bool rushB) {
  (void)bankWidthBytes;
  (void)bankDepth;
  (void)bankNum;

  populateBaseLegalizeForLLVMExportPatterns(
      converter, patterns, includeFuncOperandForwarding, rushB);
  (void)stable;
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  (void)stable;
}
