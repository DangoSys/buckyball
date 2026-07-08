//======- BuckyballToLLVMIRTranslation.cpp - Translate Buckyball to LLVM
// IR--====//
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
//
// This file implements a translation between the Buckyball dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Target/LLVMIR/Dialect/Buckyball/BuckyballToLLVMIRTranslation.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace buddy;

namespace {
static llvm::Intrinsic::ID lookupStableIntrinsic(StringRef opName) {
  return llvm::StringSwitch<llvm::Intrinsic::ID>(opName)
      .Case("buckyball.intr.fence", llvm::Intrinsic::riscv_bb_fence)
      .Case("buckyball.intr.fp2int", llvm::Intrinsic::riscv_bb_fp2int)
      .Case("buckyball.intr.im2col", llvm::Intrinsic::riscv_bb_im2col)
      .Case("buckyball.intr.int2fp", llvm::Intrinsic::riscv_bb_int2fp)
      .Case("buckyball.intr.mset", llvm::Intrinsic::riscv_bb_mset)
      .Case("buckyball.intr.mul_warp16", llvm::Intrinsic::riscv_bb_mul_warp16)
      .Case("buckyball.intr.mvin", llvm::Intrinsic::riscv_bb_mvin)
      .Case("buckyball.intr.mvout", llvm::Intrinsic::riscv_bb_mvout)
      .Case("buckyball.intr.relu", llvm::Intrinsic::riscv_bb_relu)
      .Case("buckyball.intr.bbfp.mul", llvm::Intrinsic::riscv_bb_bbfp_mul)
      .Case("buckyball.intr.transpose", llvm::Intrinsic::riscv_bb_transpose)
      .Default(llvm::Intrinsic::not_intrinsic);
}

static LogicalResult
convertStableIntrinsic(Operation &opInst, llvm::IRBuilderBase &builder,
                       LLVM::ModuleTranslation &moduleTranslation) {
  llvm::Intrinsic::ID id =
      lookupStableIntrinsic(opInst.getName().getStringRef());
  if (id == llvm::Intrinsic::not_intrinsic)
    return failure();

  SmallVector<llvm::Value *> operands;
  operands.reserve(opInst.getNumOperands());
  for (Value operand : opInst.getOperands()) {
    llvm::Value *llvmOperand = moduleTranslation.lookupValue(operand);
    if (!llvmOperand)
      return failure();
    operands.push_back(llvmOperand);
  }
  LLVM::detail::createIntrinsicCall(builder, id, operands);
  return success();
}

/// Implementation of the dialect interface that converts operations belonging
/// to the Buckyball dialect to LLVM IR.
class BuckyballDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  /// Translates the given operation to LLVM IR using the provided IR builder
  /// and saving the state in `moduleTranslation`.
  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    Operation &opInst = *op;
#include "BuckyballConversions.inc"

    if (succeeded(convertStableIntrinsic(opInst, builder, moduleTranslation)))
      return success();

    return failure();
  }
};
} // end namespace

void buddy::registerBuckyballDialectTranslation(DialectRegistry &registry) {
  registry.insert<buckyball::BuckyballDialect>();
  registry.addExtension(
      +[](MLIRContext *ctx, buckyball::BuckyballDialect *dialect) {
        dialect->addInterfaces<BuckyballDialectLLVMIRTranslationInterface>();
      });
}

void buddy::registerBuckyballDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerBuckyballDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
