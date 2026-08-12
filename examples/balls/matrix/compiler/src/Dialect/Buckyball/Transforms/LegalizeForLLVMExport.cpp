//===- LegalizeForLLVMExport.cpp - MatrixBall LLVM lowering --------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {

uint64_t mnkConfig(uint64_t m, uint64_t n, uint64_t k, uint64_t mode) {
  return fieldBits(m, 0, 11) | fieldBits(n, 12, 23) | fieldBits(k, 24, 35) |
         fieldBits(mode, 36, 36);
}

uint64_t matrixRs1(uint64_t op1, uint64_t op2, uint64_t wr) {
  return fieldBits(op1, 0, 9) | fieldBits(op2, 10, 19) | fieldBits(wr, 20, 29);
}

struct MatrixMatmulLowering : public ConvertOpToLLVMPattern<MatrixMatmulOp> {
  MatrixMatmulLowering(LLVMTypeConverter &converter, bool /*stable*/,
                       bool rushB)
      : ConvertOpToLLVMPattern<MatrixMatmulOp>(converter), rushB(rushB) {}

  LogicalResult
  matchAndRewrite(MatrixMatmulOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
          op, "buckyball.matrix_matmul requires static memref shapes");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb || cTy.getShape()[0] != (int64_t)m ||
        cTy.getShape()[1] != (int64_t)n)
      return rewriter.notifyMatchFailure(op, "matmul shapes mismatch");
    if (m == 0 || n == 0 || k == 0 || m > 16 || n > 16 || k > 16)
      return rewriter.notifyMatchFailure(
          op, "matrix_matmul legalize supports one tile with M,N,K in 1..16");
    if (!aTy.getElementType().isInteger(8) ||
        !bTy.getElementType().isInteger(8) ||
        !cTy.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "Matrix Ball supports only i8 A/B with i32 C");

    const uint64_t aBank = 0;
    const uint64_t bBank = 1;
    const uint64_t cBank = 2;
    uint64_t depthA = 16;
    uint64_t depthB = 16;
    uint64_t depthC = m;

    emitMset(rewriter, loc, aBank, 1, 1, 1);
    emitMset(rewriter, loc, bBank, 1, 1, 1);
    emitMset(rewriter, loc, cBank, 1, 4, 1);

    Value aPtr = extractPtr(rewriter, loc, aMem);
    Value bPtr = extractPtr(rewriter, loc, bMem);
    Value cPtr = extractPtr(rewriter, loc, cMem);

    Value rs1A = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, aBank),
                                 cstI64(rewriter, loc, depthA));
    Value rs2A =
        packRs2MemStride(rewriter, loc, aPtr, cstI64(rewriter, loc, 1));
    if (rushB) {
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      rewriter.create<RushBMvinOp>(
          loc, rs1A, rs2A,
          LLVM::IntToPtrOp::create(rewriter, loc, ptrType, aPtr));
    } else {
      rewriter.create<MvinIntrOp>(loc, rs1A, rs2A);
    }

    Value rs1B = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, bBank),
                                 cstI64(rewriter, loc, depthB));
    Value rs2B =
        packRs2MemStride(rewriter, loc, bPtr, cstI64(rewriter, loc, 1));
    if (rushB) {
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      rewriter.create<RushBMvinOp>(
          loc, rs1B, rs2B,
          LLVM::IntToPtrOp::create(rewriter, loc, ptrType, bPtr));
    } else {
      rewriter.create<MvinIntrOp>(loc, rs1B, rs2B);
    }

    uint64_t mode = op.getWs() ? 1ull : 0ull;
    rewriter.create<CustomIntrOp>(
        loc, cstI64(rewriter, loc, matrixRs1(aBank, bBank, cBank)),
        cstI64(rewriter, loc, mnkConfig(m, n, k, mode)),
        rewriter.getI32IntegerAttr(65));

    Value rs1C = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, cBank),
                                 cstI64(rewriter, loc, depthC));
    Value rs2C =
        packRs2MemStride(rewriter, loc, cPtr, cstI64(rewriter, loc, 1));
    if (rushB) {
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      rewriter.create<RushBMvoutOp>(
          loc, rs1C, rs2C,
          LLVM::IntToPtrOp::create(rewriter, loc, ptrType, cPtr));
    } else {
      rewriter.create<MvoutIntrOp>(loc, rs1C, rs2C);
    }

    Value zero = cstI64(rewriter, loc, 0);
    rewriter.create<FenceIntrOp>(loc, zero, zero);
    emitMset(rewriter, loc, aBank, 0, 0, 0);
    emitMset(rewriter, loc, bBank, 0, 0, 0);
    emitMset(rewriter, loc, cBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }

private:
  bool rushB;
};

struct MatrixLowering : public ConvertOpToLLVMPattern<MatrixOp> {
  MatrixLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<MatrixOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(MatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (stable)
      return rewriter.notifyMatchFailure(
          op, "MatrixBall stable intrinsic is not available; use custom path");

    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(
        rewriter, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
        adaptor.getResultBankId(), cstI64(rewriter, loc, 0));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getConfig(),
                                              rewriter.getI32IntegerAttr(65));
    return success();
  }

private:
  bool stable = false;
};

} // namespace

namespace mlir::buddy::buckyball {
void populateMatrixMatmulLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    bool rushB) {
  patterns.add<MatrixMatmulLowering>(converter, stable, rushB);
}

void configureMatrixMatmulLegalizeForExportTarget(LLVMConversionTarget &target,
                                                  bool /*stable*/) {
  target.addIllegalOp<MatrixMatmulOp>();
}

void populateMatrixLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<MatrixLowering>(converter, stable);
}

void configureMatrixLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool /*stable*/) {
  target.addIllegalOp<MatrixOp, BankMatrixOp>();
}
} // namespace mlir::buddy::buckyball
