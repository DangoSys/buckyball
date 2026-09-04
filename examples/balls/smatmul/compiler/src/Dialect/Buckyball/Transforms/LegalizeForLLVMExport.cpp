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

uint64_t matrixCfg(uint64_t rows, uint64_t cols, bool first = true,
                   bool last = true) {
  if (rows == 0 || rows > 0xfff || (rows != 1 && rows % 16) || cols != 16)
    llvm::report_fatal_error(
        "matrix cfg: rows must be 1 or a multiple of 16 and cols must be 16");
  return fieldBits(rows, 0, 11) | fieldBits(cols, 12, 23) |
         (uint64_t(first) << 24) | (uint64_t(last) << 25);
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
    const int64_t bankDepth = buckyball_target::getBuckyballTarget().bankDepth;
    if (bankDepth <= 0)
      return op.emitError("smatmul lowering requires target bankDepth > 0");
    // A occupies m lines, B occupies k lines, and packed int32 C occupies 2m.
    if (m == 0 || n != 16 || k != 16 || m % 16 != 0 ||
        m > static_cast<uint64_t>(bankDepth / 2) ||
        k > static_cast<uint64_t>(bankDepth))
      return rewriter.notifyMatchFailure(
          op, "SMatMul requires K/N exactly 16 and A/B/C to fit their banks");
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
        loc,
        packRs1BanksIter(rewriter, loc, cstI64(rewriter, loc, aBank),
                         cstI64(rewriter, loc, bBank),
                         cstI64(rewriter, loc, cBank),
                         cstI64(rewriter, loc, k)),
        cstI64(rewriter, loc, matrixCfg(m, n)),
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
    Location loc = op.getLoc();
    auto config = op.getConfig().getDefiningOp<arith::ConstantOp>();
    if (!config)
      return op.emitError("smatmul config must be a constant");
    auto configAttr = dyn_cast<IntegerAttr>(config.getValue());
    if (!configAttr)
      return op.emitError("smatmul config must be an integer constant");
    uint64_t cfg = configAttr.getValue().getZExtValue();
    uint64_t rows = cfg & 0xfff;
    uint64_t cols = (cfg >> 12) & 0xfff;
    uint64_t k = (cfg >> 24) & 0xfff;
    if (rows == 0 || (rows != 1 && rows % 16) || cols != 16 || k == 0 ||
        k % 16 || rows * 4 > static_cast<uint64_t>(bankDepth))
      return op.emitError("SMatMul matrix shape or C footprint is invalid");
    auto base = adaptor.getOutputBase().getDefiningOp<arith::ConstantOp>();
    auto baseAttr =
        base ? dyn_cast<IntegerAttr>(base.getValue()) : IntegerAttr();
    if (!baseAttr || baseAttr.getInt() < 0 ||
        baseAttr.getInt() + rows * 4 > bankDepth)
      return op.emitError(
          "SMatMul outputBase must be constant and fit C in bank");
    Value rs1 = packRs1BanksIter(
        rewriter, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
        adaptor.getResultBankId(), cstI64(rewriter, loc, k));
    Value rs2 = cstI64(rewriter, loc, matrixCfg(rows, cols, false, false));
    Value first = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(),
                                                  adaptor.getFirst());
    Value last = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(),
                                                 adaptor.getLast());
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, first, cstI64(rewriter, loc, 24)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, last, cstI64(rewriter, loc, 25)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputBase(),
                                       cstI64(rewriter, loc, 26)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("SMATMUL_OS")));
    return success();
  }

private:
  int64_t bankDepth;
};

struct SMatMulBiasLowering : public ConvertOpToLLVMPattern<SMatMulBiasOp> {
  SMatMulBiasLowering(LLVMTypeConverter &converter, bool)
      : ConvertOpToLLVMPattern<SMatMulBiasOp>(converter) {}

  LogicalResult
  matchAndRewrite(SMatMulBiasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("SMatMulBall");
    Location loc = op.getLoc();
    auto base = adaptor.getInputBase().getDefiningOp<arith::ConstantOp>();
    auto baseAttr =
        base ? dyn_cast<IntegerAttr>(base.getValue()) : IntegerAttr();
    if (!baseAttr || baseAttr.getInt() < 0 ||
        baseAttr.getInt() + 4 >
            static_cast<int64_t>(
                buckyball_target::getBuckyballTarget().bankDepth))
      return op.emitError(
          "SMatMul bias inputBase must be constant and fit in bank");
    Value rs1 = packRs1BankIter(rewriter, loc, adaptor.getBiasBankId(),
                                cstI64(rewriter, loc, 4));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getInputBase(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("SMATMUL_BIAS")));
    return success();
  }
};

} // namespace

namespace mlir::buddy::buckyball {
void populateSMatMulBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  patterns.add<SMatMulMatmulLowering>(converter, stable, rushB);
  patterns.add<SMatMulLowering>(converter, stable, bankDepth);
  patterns.add<SMatMulBiasLowering>(converter, stable);
}

void configureSMatMulBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool /*stable*/) {
  target.addIllegalOp<SMatMulMatmulOp>();
  target.addIllegalOp<SMatMulOp, BankSMatMulOp, SMatMulBiasOp,
                      BankSMatMulBiasOp>();
}
} // namespace mlir::buddy::buckyball
