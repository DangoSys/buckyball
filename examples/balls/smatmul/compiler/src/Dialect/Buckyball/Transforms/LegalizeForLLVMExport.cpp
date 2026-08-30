//===- LegalizeForLLVMExport.cpp - SMatMulBall LLVM lowering --------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {

uint64_t matrixCfg(uint64_t rows, uint64_t cols, uint64_t k) {
  if (rows == 0 || cols == 0 || k == 0 || rows > 0xfff || k > 0xfff ||
      cols > 16)
    llvm::report_fatal_error("matrix cfg: rows/k in 1..4095, cols in 1..16");
  return fieldBits(rows, 0, 11) | fieldBits(cols, 12, 23) |
         fieldBits(k, 24, 35);
}

uint64_t matrixRs1(uint64_t op1, uint64_t op2, uint64_t wr) {
  return fieldBits(op1, 0, 9) | fieldBits(op2, 10, 19) | fieldBits(wr, 20, 29);
}

struct SMatMulMatmulLowering : public ConvertOpToLLVMPattern<SMatMulMatmulOp> {
  SMatMulMatmulLowering(LLVMTypeConverter &converter, bool /*stable*/,
                        bool rushB)
      : ConvertOpToLLVMPattern<SMatMulMatmulOp>(converter), rushB(rushB) {}

  LogicalResult
  matchAndRewrite(SMatMulMatmulOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("SMatMulBall");
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
          op, "buckyball.smatmul_matmul requires static memref shapes");

    uint64_t m = aTy.getShape()[0];
    uint64_t k = aTy.getShape()[1];
    uint64_t kb = bTy.getShape()[0];
    uint64_t n = bTy.getShape()[1];
    if (k != kb || cTy.getShape()[0] != (int64_t)m ||
        cTy.getShape()[1] != (int64_t)n)
      return rewriter.notifyMatchFailure(op, "matmul shapes mismatch");
    // Bank-unit 8-bit path: N/K are one bank lane; M fills bank depth.
    if (m == 0 || n != 16 || k != 16 || m > 1024 || m % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "SMatMul requires M/K exactly 16 and N exactly 16");
    if (!aTy.getElementType().isInteger(8) ||
        !bTy.getElementType().isInteger(8) ||
        !cTy.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "SMatMul Ball supports only 8-bit A/B with 32-bit C");

    const uint64_t aBank = 0;
    const uint64_t bBank = 1;
    const uint64_t cBank = 2;
    uint64_t depthA = m; // K <= 16 => one bank row per M row
    uint64_t depthB = k;
    uint64_t depthC = 2 * m;

    emitMset(rewriter, loc, aBank, 1, 1, 1);
    emitMset(rewriter, loc, bBank, 1, 1, 1);
    emitMset(rewriter, loc, cBank, 1, 2, 1);

    Value aPtr = extractPtr(rewriter, loc, aMem);
    Value bPtr = extractPtr(rewriter, loc, bMem);
    auto packedType = MemRefType::get({static_cast<int64_t>(depthC), 8},
                                      cTy.getElementType());
    Value packed = rewriter.create<memref::AllocOp>(loc, packedType);
    Value packedPtr = extractPtr(rewriter, loc, packed);

    Value rs1A = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, aBank),
                                 cstI64(rewriter, loc, depthA));
    Value rs2A =
        packRs2MemStride(rewriter, loc, aPtr, cstI64(rewriter, loc, 1));
    if (!rushB)
      emitDmaCacheFlush(rewriter, loc);
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
    if (!rushB)
      emitDmaCacheFlush(rewriter, loc);
    if (rushB) {
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      rewriter.create<RushBMvinOp>(
          loc, rs1B, rs2B,
          LLVM::IntToPtrOp::create(rewriter, loc, ptrType, bPtr));
    } else {
      rewriter.create<MvinIntrOp>(loc, rs1B, rs2B);
    }

    rewriter.create<CustomIntrOp>(
        loc, cstI64(rewriter, loc, matrixRs1(aBank, bBank, cBank)),
        cstI64(rewriter, loc, matrixCfg(m, n, k)),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("SMATMUL_OS")));

    Value rs1C = packRs1BankIter(rewriter, loc, cstI64(rewriter, loc, cBank),
                                 cstI64(rewriter, loc, depthC));
    Value rs2C =
        packRs2MemStride(rewriter, loc, packedPtr, cstI64(rewriter, loc, 1));
    if (!rushB)
      emitDmaCacheFlush(rewriter, loc);
    if (rushB) {
      Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      rewriter.create<RushBMvoutOp>(
          loc, rs1C, rs2C,
          LLVM::IntToPtrOp::create(rewriter, loc, ptrType, packedPtr));
    } else {
      rewriter.create<MvoutIntrOp>(loc, rs1C, rs2C);
    }

    Value zero = cstI64(rewriter, loc, 0);
    rewriter.create<FenceIntrOp>(loc, zero, zero);
    if (!rushB)
      emitDmaCacheFence(rewriter, loc);

    Value indexZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value indexOne = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value indexTwo = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value indexEight = rewriter.create<arith::ConstantIndexOp>(loc, 8);
    Value indexM = rewriter.create<arith::ConstantIndexOp>(loc, m);
    Value indexN = rewriter.create<arith::ConstantIndexOp>(loc, n);
    auto rowLoop =
        rewriter.create<scf::ForOp>(loc, indexZero, indexM, indexOne);
    rewriter.setInsertionPointToStart(rowLoop.getBody());
    Value row = rowLoop.getInductionVar();
    auto columnLoop =
        rewriter.create<scf::ForOp>(loc, indexZero, indexN, indexOne);
    rewriter.setInsertionPointToStart(columnLoop.getBody());
    Value column = columnLoop.getInductionVar();
    Value packedRow = rewriter.create<arith::AddIOp>(
        loc, rewriter.create<arith::MulIOp>(loc, row, indexTwo),
        rewriter.create<arith::DivUIOp>(loc, column, indexEight));
    Value packedColumn =
        rewriter.create<arith::RemUIOp>(loc, column, indexEight);
    Value value = rewriter.create<memref::LoadOp>(
        loc, packed, ValueRange{packedRow, packedColumn});
    rewriter.create<memref::StoreOp>(loc, value, cMem, ValueRange{row, column});
    rewriter.setInsertionPointAfter(rowLoop);
    rewriter.create<memref::DeallocOp>(loc, packed);

    emitMset(rewriter, loc, aBank, 0, 0, 0);
    emitMset(rewriter, loc, bBank, 0, 0, 0);
    emitMset(rewriter, loc, cBank, 0, 0, 0);

    rewriter.eraseOp(op);
    return success();
  }

private:
  bool rushB;
};

struct SMatMulLowering : public ConvertOpToLLVMPattern<SMatMulOp> {
  SMatMulLowering(LLVMTypeConverter &converter, bool, int64_t bankDepth)
      : ConvertOpToLLVMPattern<SMatMulOp>(converter), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(SMatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("SMatMulBall");
    if (bankDepth <= 1)
      return op.emitError("smatmul lowering requires bank_depth > 1");
    int64_t aw = addrBitsForDepth(bankDepth);
    if (aw < 1 || 3 * aw > 34)
      return op.emitError(
          "smatmul iter cannot hold three bases from bank_depth");
    uint64_t mask = (1ULL << aw) - 1;
    if ((uint64_t)op.getOp1Base() > mask || (uint64_t)op.getOp2Base() > mask ||
        (uint64_t)op.getWrBase() > mask)
      return op.emitError("smatmul base exceeds log2(bank_depth)");

    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(
        rewriter, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
        adaptor.getResultBankId(),
        cstI64(rewriter, loc,
               smatmulIterBits(op.getOp1Base(), op.getOp2Base(), op.getWrBase(),
                               aw)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getConfig(),
        rewriter.getI32IntegerAttr(buckyball_target::getBuckyballFunct7(
            op.getWs() ? "SMATMUL_WS" : "SMATMUL_OS")));
    return success();
  }

private:
  int64_t bankDepth = 0;
};

} // namespace

namespace mlir::buddy::buckyball {
void populateSMatMulBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  patterns.add<SMatMulMatmulLowering>(converter, stable, rushB);
  patterns.add<SMatMulLowering>(converter, stable, bankDepth);
}

void configureSMatMulBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool /*stable*/) {
  target.addIllegalOp<SMatMulMatmulOp>();
  target.addIllegalOp<SMatMulOp, BankSMatMulOp>();
}
} // namespace mlir::buddy::buckyball
