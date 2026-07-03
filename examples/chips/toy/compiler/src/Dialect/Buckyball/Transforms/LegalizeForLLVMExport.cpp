//===- LegalizeForLLVMExport.cpp - Toy Buckyball LLVM lowering ------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Buckyball/Transform.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
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
void populateSystolicLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable);
void configureSystolicLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool stable);
} // namespace mlir::buddy::buckyball

namespace {
struct MatMulLowering : public ConvertOpToLLVMPattern<MatMulOp> {
  MatMulLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<MatMulOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value aMem = op.getAMemArray();
    Value bMem = op.getBMemArray();
    Value cMem = op.getCMemArray();

    auto aTy = cast<MemRefType>(aMem.getType());
    auto bTy = cast<MemRefType>(bMem.getType());
    auto cTy = cast<MemRefType>(cMem.getType());
    if (!aTy.hasStaticShape() || !bTy.hasStaticShape() || !cTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "buckyball.matmul requires static memref shapes");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb)
      return rewriter.notifyMatchFailure(op, "inner dimensions must match");
    if (m % 16 != 0 || k % 16 != 0 || n % 16 != 0)
      return rewriter.notifyMatchFailure(op,
                                         "M, K and N must be multiples of 16");

    const uint64_t aBank = 0;
    const uint64_t bBank = 1;
    const uint64_t cBank = 2;
    uint64_t depthA = m * (k / 16);
    uint64_t depthB = k * (n / 16);
    uint64_t depthC = m * (n / 16);

    emitMset(rewriter, loc, aBank, 1, 1, 1);
    emitMset(rewriter, loc, bBank, 1, 1, 1);
    emitMset(rewriter, loc, cBank, 1, 4, 1);

    Value aPtr = extractPtr(rewriter, loc, aMem);
    Value bPtr = extractPtr(rewriter, loc, bMem);
    Value cPtr = extractPtr(rewriter, loc, cMem);

    SmallVector<int64_t, 4> bStrides;
    SmallVector<int64_t, 4> cStrides;
    int64_t bOff = 0;
    int64_t cOff = 0;
    if (failed(bTy.getStridesAndOffset(bStrides, bOff)) || bStrides.size() < 2)
      return rewriter.notifyMatchFailure(op, "B memref needs static strides");
    if (failed(cTy.getStridesAndOffset(cStrides, cOff)) || cStrides.size() < 2)
      return rewriter.notifyMatchFailure(op, "C memref needs static strides");
    if (ShapedType::isDynamic(bStrides[0]) ||
        ShapedType::isDynamic(cStrides[0]) || bStrides[0] % 16 != 0 ||
        cStrides[0] % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "B/C row stride must be static and divisible by 16");

    Value rs1A = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, aBank),
                                 cstI64(rewriter, loc, depthA));
    Value rs2A =
        packRs2MemStride(rewriter, loc, aPtr, cstI64(rewriter, loc, 1));
    rewriter.create<MvinIntrOp>(loc, rs1A, rs2A);

    Value rs1B = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, bBank),
                                 cstI64(rewriter, loc, depthB));
    Value rs2B = packRs2MemStride(
        rewriter, loc, bPtr,
        cstI64(rewriter, loc, static_cast<uint64_t>(bStrides[0] / 16)));
    rewriter.create<MvinIntrOp>(loc, rs1B, rs2B);

    uint64_t rs1Mul = fieldBits(aBank, 0, 9) | fieldBits(bBank, 10, 19) |
                      fieldBits(cBank, 20, 29) | fieldBits(k, 30, 63);
    if (stable)
      rewriter.create<MulWarp16IntrOp>(loc, cstI64(rewriter, loc, rs1Mul),
                                       cstI64(rewriter, loc, 0));
    else
      rewriter.create<CustomIntrOp>(loc, cstI64(rewriter, loc, rs1Mul),
                                    cstI64(rewriter, loc, 0),
                                    rewriter.getI32IntegerAttr(64));

    Value rs1C = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, cBank),
                                 cstI64(rewriter, loc, depthC));
    Value rs2C = packRs2MemStride(
        rewriter, loc, cPtr,
        cstI64(rewriter, loc, static_cast<uint64_t>(cStrides[0] / 16)));
    rewriter.create<MvoutIntrOp>(loc, rs1C, rs2C);

    Value zero = cstI64(rewriter, loc, 0);
    rewriter.create<FenceIntrOp>(loc, zero, zero);
    emitMset(rewriter, loc, aBank, 0, 0, 0);
    emitMset(rewriter, loc, bBank, 0, 0, 0);
    emitMset(rewriter, loc, cBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }

private:
  bool stable = false;
};
} // namespace

void mlir::populateBuckyballLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    int64_t bankWidthBytes, int64_t bankDepth, int64_t bankNum,
    bool includeFuncOperandForwarding, bool stable) {
  (void)bankWidthBytes;
  (void)bankDepth;
  (void)bankNum;

  populateBaseLegalizeForLLVMExportPatterns(converter, patterns,
                                            includeFuncOperandForwarding);
  patterns.add<MatMulLowering>(converter, stable);
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
  mlir::buddy::buckyball::populateSystolicLegalizeForLLVMExportPatterns(
      converter, patterns, stable);
}

void mlir::configureBuckyballLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  configureBaseLegalizeForExportTarget(target);
  target.addIllegalOp<MatMulOp>();
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
  mlir::buddy::buckyball::configureSystolicLegalizeForExportTarget(target,
                                                                   stable);
}
