//===- LegalizeForLLVMExportBase.h - Buckyball base LLVM lowering -*- C++
//-*-===//
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

#ifndef BUDDY_BUCKYBALL_LEGALIZE_FOR_LLVM_EXPORT_BASE_H
#define BUDDY_BUCKYBALL_LEGALIZE_FOR_LLVM_EXPORT_BASE_H

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include <cstdint>

namespace buddy {
namespace buckyball {
namespace legalize {

uint64_t fieldBits(uint64_t val, int startBit, int endBit);

mlir::Value cstI64(mlir::OpBuilder &b, mlir::Location loc, uint64_t v);
mlir::Value extractPtr(mlir::OpBuilder &b, mlir::Location loc,
                       mlir::Value memref);
mlir::Value packRs1BanksIter(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value rBank0, mlir::Value rBank1,
                             mlir::Value wBank, mlir::Value iter);
mlir::Value packRs1BankIter(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value bankId, mlir::Value depth);
mlir::Value packRs2MemStride(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value memAddr, mlir::Value stride);
void emitMset(mlir::OpBuilder &b, mlir::Location loc, uint64_t bankId,
              uint64_t row, uint64_t col, uint64_t alloc);

void populateBaseLegalizeForLLVMExportPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns,
    bool includeFuncOperandForwarding);

void configureBaseLegalizeForExportTarget(mlir::LLVMConversionTarget &target);

} // namespace legalize
} // namespace buckyball
} // namespace buddy

#endif // BUDDY_BUCKYBALL_LEGALIZE_FOR_LLVM_EXPORT_BASE_H
