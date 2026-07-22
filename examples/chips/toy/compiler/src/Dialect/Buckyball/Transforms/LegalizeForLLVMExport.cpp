//===- LegalizeForLLVMExport.cpp - Toy Buckyball LLVM lowering ------------===//
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/Transform.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball::legalize;

namespace mlir::buddy::buckyball {
void populateVectorLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable);
void configureVectorLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable);
void populateTransposeLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable);
void configureTransposeLegalizeForExportTarget(LLVMConversionTarget &target,
                                               bool stable);
void populateIm2colLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable);
void configureIm2colLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable);
void populateFp2IntLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable);
void configureFp2IntLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable);
void populateInt2FpLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable);
void configureInt2FpLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable);
void populateReluLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns,
                                               bool stable);
void configureReluLegalizeForExportTarget(LLVMConversionTarget &target,
                                          bool stable);
void populateMatrixMatmulLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable);
void configureMatrixMatmulLegalizeForExportTarget(LLVMConversionTarget &target,
                                                  bool stable);
void populateMatrixLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable);
void configureMatrixLegalizeForExportTarget(LLVMConversionTarget &target,
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
  mlir::buddy::buckyball::populateMatrixMatmulLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateVectorLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateTransposeLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateIm2colLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateFp2IntLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateInt2FpLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateReluLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
  mlir::buddy::buckyball::populateMatrixLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  mlir::buddy::buckyball::configureMatrixMatmulLegalizeForExportTarget(target,
                                                                       stable);
  mlir::buddy::buckyball::configureVectorLegalizeForExportTarget(target,
                                                                 stable);
  mlir::buddy::buckyball::configureTransposeLegalizeForExportTarget(target,
                                                                    stable);
  mlir::buddy::buckyball::configureIm2colLegalizeForExportTarget(target,
                                                                 stable);
  mlir::buddy::buckyball::configureFp2IntLegalizeForExportTarget(target,
                                                                 stable);
  mlir::buddy::buckyball::configureInt2FpLegalizeForExportTarget(target,
                                                                 stable);
  mlir::buddy::buckyball::configureReluLegalizeForExportTarget(target, stable);
  mlir::buddy::buckyball::configureMatrixLegalizeForExportTarget(target,
                                                                 stable);
}
