//===- LegalizeForLLVMExport.cpp - Pebble Buckyball LLVM lowering
//----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "Buckyball/Transform.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball::legalize;

namespace mlir::buddy::buckyball {
void populateTransposeLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable);
void configureTransposeLegalizeForExportTarget(LLVMConversionTarget &target,
                                               bool stable);
void populateSystolicLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable);
void configureSystolicLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool stable);
} // namespace mlir::buddy::buckyball

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t bankWidthBytes, int64_t bankDepth, int64_t bankNum,
    bool includeFuncOperandForwarding, bool stable) {
  (void)bankWidthBytes;
  (void)bankDepth;
  (void)bankNum;

  populateBaseLegalizeForLLVMExportPatterns(converter, patterns,
                                            includeFuncOperandForwarding);
  mlir::buddy::buckyball::populateTransposeLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateSystolicLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  mlir::buddy::buckyball::configureTransposeLegalizeForExportTarget(target,
                                                                    stable);
  mlir::buddy::buckyball::configureSystolicLegalizeForExportTarget(target,
                                                                   stable);
}
