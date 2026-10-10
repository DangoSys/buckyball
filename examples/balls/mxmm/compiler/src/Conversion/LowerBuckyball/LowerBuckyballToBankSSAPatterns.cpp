#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerBuckyball/LowerBuckyball.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <algorithm>
#include <cstdint>
#include <tuple>
using namespace mlir;
using namespace ::buddy::buckyball;
namespace {
#include "FP32Patterns.inc"
#include "MXFP8Patterns.inc"
#include "SharedPanelPatterns.inc"
#include "SharedWindowPatterns.inc"
} // namespace
namespace mlir::buddy {
void populateMxmmBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MXFP8ToBanks, MXFP8SharedPanelToBanks, MXFP8SharedWindowToBanks,
               FP32ToBanks>(patterns.getContext());
}

} // namespace mlir::buddy
