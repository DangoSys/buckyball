//===- LegalizeForLLVMExport.cpp - Pebble Buckyball LLVM lowering ---------===//
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
void populateTransposeBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureTransposeBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                   bool stable);
void populateSMatMulBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureSMatMulBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool stable);
void populateIm2colBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureIm2colBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable);
void populateFp2IntBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureFp2IntBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable);
void populateInt2FpBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureInt2FpBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable);
void populateMatAddBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB);
void configureMatAddBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable);
void populateReluBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable);
void configureReluBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool stable);
} // namespace mlir::buddy::buckyball

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t bankWidthBytes, int64_t bankDepth, int64_t bankNum,
    bool includeFuncOperandForwarding, bool stable, bool rushB) {
  (void)bankWidthBytes;
  (void)bankNum;

  populateBaseLegalizeForLLVMExportPatterns(
      converter, patterns, includeFuncOperandForwarding, rushB);
  mlir::buddy::buckyball::populateSMatMulBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateTransposeBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateIm2colBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateFp2IntBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateInt2FpBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateMatAddBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable, bankDepth, rushB);
  mlir::buddy::buckyball::populateReluBallLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  mlir::buddy::buckyball::configureSMatMulBallLegalizeForExportTarget(target,
                                                                      stable);
  mlir::buddy::buckyball::configureTransposeBallLegalizeForExportTarget(target,
                                                                        stable);
  mlir::buddy::buckyball::configureIm2colBallLegalizeForExportTarget(target,
                                                                     stable);
  mlir::buddy::buckyball::configureFp2IntBallLegalizeForExportTarget(target,
                                                                     stable);
  mlir::buddy::buckyball::configureInt2FpBallLegalizeForExportTarget(target,
                                                                     stable);
  mlir::buddy::buckyball::configureMatAddBallLegalizeForExportTarget(target,
                                                                     stable);
  mlir::buddy::buckyball::configureReluBallLegalizeForExportTarget(target,
                                                                   stable);
}
