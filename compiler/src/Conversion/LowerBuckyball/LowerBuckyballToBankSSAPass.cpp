//====- LowerBuckyballToBankSSAPass.cpp - Expand to bank-SSA ops
//-----------===//
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"

using namespace mlir;
using namespace buddy;

namespace {

static Value cstI64(OpBuilder &b, Location loc, uint64_t v) {
  return b.create<arith::ConstantOp>(loc, b.getI64Type(),
                                     b.getI64IntegerAttr(v));
}

static LogicalResult getStaticRowStrideDiv16(MemRefType ty, uint64_t &out) {
  SmallVector<int64_t, 4> strides;
  int64_t off = 0;
  if (failed(ty.getStridesAndOffset(strides, off)) || strides.size() < 2)
    return failure();
  if (ShapedType::isDynamic(strides[0]) || strides[0] <= 0 ||
      strides[0] % 16 != 0)
    return failure();
  if (ShapedType::isDynamic(strides[1]) || strides[1] != 1)
    return failure();
  out = static_cast<uint64_t>(strides[0] / 16);
  return success();
}

class MatMulToBankSSAPattern : public OpRewritePattern<buckyball::MatMulOp> {
public:
  using OpRewritePattern<buckyball::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(buckyball::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = dyn_cast<MemRefType>(aMem.getType());
    auto bTy = dyn_cast<MemRefType>(bMem.getType());
    auto cTy = dyn_cast<MemRefType>(cMem.getType());
    if (!aTy || !bTy || !cTy || !aTy.hasStaticShape() ||
        !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "bb_matmul requires static rank-2 memrefs");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");
    if (k % 16 != 0 || n % 16 != 0)
      return rewriter.notifyMatchFailure(op, "K and N must be multiples of 16");

    uint64_t strideA = 1;
    uint64_t strideB = 0;
    uint64_t strideC = 0;
    if (failed(getStaticRowStrideDiv16(bTy, strideB)))
      return rewriter.notifyMatchFailure(
          op, "B requires static strided<[row,1]> and row % 16 == 0");
    if (failed(getStaticRowStrideDiv16(cTy, strideC)))
      return rewriter.notifyMatchFailure(
          op, "C requires static strided<[row,1]> and row % 16 == 0");

    // Depth calculations for fp32 input (cols=4 mode)
    // fp32: each chunk is 16 elements * 4 bytes = 64 bytes
    uint64_t depthA_fp32 = m * (k / 16);
    uint64_t depthB_fp32 = k * (n / 16);
    uint64_t depthC_fp32 = m * (n / 16);

    // After quant fp32→i8, depth stays the same (4 fp32 banks → 1 i8 bank)
    uint64_t depthA_i8 = depthA_fp32;
    uint64_t depthB_i8 = depthB_fp32;
    uint64_t depthC_i8 = depthC_fp32;

    // Scale for quantization: hardcoded 1.0f for now
    // TODO: get scale from MatMulOp attributes or global config
    uint32_t scale_bits = 0x3f800000; // 1.0f in IEEE 754
    auto scale = cstI64(rewriter, loc, scale_bits);

    // Allocate fp32 banks for input (cols=4)
    auto a0_fp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    a0_fp32->setAttr("col", rewriter.getI64IntegerAttr(4));
    auto b0_fp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    b0_fp32->setAttr("col", rewriter.getI64IntegerAttr(4));

    // Allocate i8 banks for quantized input (cols=1)
    auto a0_i8 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    auto b0_i8 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());

    // Allocate i32 accumulator bank (cols=4)
    auto c0_i32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    c0_i32->setAttr("col", rewriter.getI64IntegerAttr(4));

    // Allocate fp32 bank for dequantized output (cols=4)
    auto c0_fp32 =
        rewriter.create<buckyball::BankAllocOp>(loc, rewriter.getI64Type());
    c0_fp32->setAttr("col", rewriter.getI64IntegerAttr(4));

    // Load fp32 A and B from DRAM
    auto a1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), aMem, a0_fp32.getBank(),
        cstI64(rewriter, loc, depthA_fp32), cstI64(rewriter, loc, strideA));
    auto b1 = rewriter.create<buckyball::BankMvinOp>(
        loc, rewriter.getI64Type(), bMem, b0_fp32.getBank(),
        cstI64(rewriter, loc, depthB_fp32), cstI64(rewriter, loc, strideB));

    // Quantize fp32 → i8
    auto a2 = rewriter.create<buckyball::BankQuantOp>(
        loc, rewriter.getI64Type(), a1.getBankOut(), a0_i8.getBank(),
        cstI64(rewriter, loc, depthA_i8), scale);
    auto b2 = rewriter.create<buckyball::BankQuantOp>(
        loc, rewriter.getI64Type(), b1.getBankOut(), b0_i8.getBank(),
        cstI64(rewriter, loc, depthB_i8), scale);

    // Matrix multiply i8 × i8 → i32
    auto c1 = rewriter.create<buckyball::BankMulWarp16Op>(
        loc, rewriter.getI64Type(), a2.getOutBankOut(), b2.getOutBankOut(),
        c0_i32.getBank(), cstI64(rewriter, loc, k), cstI64(rewriter, loc, 0));

    // Dequantize i32 → fp32
    auto c2 = rewriter.create<buckyball::BankDequantOp>(
        loc, rewriter.getI64Type(), c1.getWrBankOut(), c0_fp32.getBank(),
        cstI64(rewriter, loc, depthC_i8), scale);

    // Store fp32 C back to DRAM
    auto c3 = rewriter.create<buckyball::BankMvoutOp>(
        loc, rewriter.getI64Type(), cMem, c2.getOutBankOut(),
        cstI64(rewriter, loc, depthC_fp32), cstI64(rewriter, loc, strideC));

    // Release all banks
    rewriter.create<buckyball::BankReleaseOp>(loc, a1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, a2.getOutBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, b1.getBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, b2.getOutBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, c1.getWrBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, c2.getOutBankOut());
    rewriter.create<buckyball::BankReleaseOp>(loc, c3.getBankOut());

    rewriter.eraseOp(op);
    return success();
  }
};

// TransposeToBankSSAPattern removed - TransposeOp no longer has memref operands

// Im2colToBankSSAPattern removed - Im2colOp no longer has memref operands

// QuantToBankSSAPattern removed - QuantOp no longer has memref operands

// DequantToBankSSAPattern removed - DequantOp no longer has memref operands

class LowerBuckyballToBankSSAPass
    : public PassWrapper<LowerBuckyballToBankSSAPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBuckyballToBankSSAPass)
  StringRef getArgument() const final { return "lower-buckyball-to-bank-ssa"; }
  StringRef getDescription() const final {
    return "Lower bb_matmul to explicit bank-SSA ops.";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulToBankSSAPattern>(&getContext());
    // TransposeToBankSSAPattern, Im2colToBankSSAPattern, QuantToBankSSAPattern,
    // DequantToBankSSAPattern removed - these operations no longer have memref
    // operands
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace buddy {
void registerLowerBuckyballToBankSSAPass() {
  PassRegistration<LowerBuckyballToBankSSAPass>();
}
} // namespace buddy
} // namespace mlir
