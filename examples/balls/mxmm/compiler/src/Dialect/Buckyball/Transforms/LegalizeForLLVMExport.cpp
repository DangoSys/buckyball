#include <optional>
//===- LegalizeForLLVMExport.cpp - MxmmBall LLVM lowering --------------===//

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
using namespace ::buddy::buckyball;
using namespace ::buddy::buckyball::legalize;

namespace {
uint64_t matrixCfg(uint64_t rows, uint64_t cols, bool first, bool last) {
  return fieldBits(rows, 0, 11) | fieldBits(cols, 12, 23) |
         (uint64_t(first) << 24) | (uint64_t(last) << 25);
}
struct MXFP8Lowering : public ConvertOpToLLVMPattern<MXFP8Op> {
  MXFP8Lowering(LLVMTypeConverter &converter, bool, int64_t bankDepth)
      : ConvertOpToLLVMPattern<MXFP8Op>(converter), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(MXFP8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("MxmmBall");
    Location loc = op.getLoc();
    auto config = op.getConfig().getDefiningOp<arith::ConstantOp>();
    if (!config)
      return op.emitError("mxmm config must be a constant");
    auto configAttr = dyn_cast<IntegerAttr>(config.getValue());
    if (!configAttr)
      return op.emitError("mxmm config must be an integer constant");
    uint64_t cfg = configAttr.getValue().getZExtValue();
    uint64_t rows = cfg & 0xfff;
    uint64_t cols = (cfg >> 12) & 0xfff;
    uint64_t k = (cfg >> 24) & 0xfff;
    if (cfg >> 36 || (rows != 1 && (rows == 0 || rows % 16)) || cols == 0 ||
        cols % 16 || k == 0 || k % 32 ||
        (rows * k * 33 / 32 + 15) / 16 > static_cast<uint64_t>(bankDepth) ||
        cols * k * 33 / 32 / 16 > static_cast<uint64_t>(bankDepth) ||
        rows * cols / 4 > static_cast<uint64_t>(bankDepth))
      return op.emitError("Mxmm matrix shape or C footprint is invalid");
    auto base = adaptor.getOutputBase().getDefiningOp<arith::ConstantOp>();
    auto baseAttr =
        base ? dyn_cast<IntegerAttr>(base.getValue()) : IntegerAttr();
    if (!baseAttr || baseAttr.getInt() < 0 || baseAttr.getInt() > 63 ||
        baseAttr.getInt() + rows * cols / 4 > bankDepth)
      return op.emitError("Mxmm outputBase must be constant and fit C in bank");
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
            buckyball_target::getBuckyballFunct7("MXMM_MXFP8")));
    return success();
  }

private:
  int64_t bankDepth;
};

struct FP32Lowering : public ConvertOpToLLVMPattern<FP32Op> {
  FP32Lowering(LLVMTypeConverter &converter, bool, int64_t depth)
      : ConvertOpToLLVMPattern<FP32Op>(converter), depth(depth) {}

  LogicalResult matchAndRewrite(FP32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    buckyball_target::requireBuckyballBall("MxmmBall");
    auto constant = op.getConfig().getDefiningOp<arith::ConstantOp>();
    auto shape =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
    if (!shape)
      return op.emitError("FP32 matmul requires a constant shape");
    uint64_t cfg = shape.getValue().getZExtValue();
    uint64_t rows = cfg & 0xfff, cols = (cfg >> 12) & 0xfff,
             k = (cfg >> 24) & 0xfff;
    if (cfg >> 36 || (rows != 1 && (rows == 0 || rows % 16)) || cols == 0 ||
        cols % 16 || !k || k % 4 || rows * k / 4 > uint64_t(depth) ||
        cols * k / 4 > uint64_t(depth) ||
        buckyball_target::getBuckyballTarget().bankWidthBits != 128)
      return op.emitError(
          "FP32 matmul shape does not fit the configured banks");
    auto baseOp = op.getOutputBase().getDefiningOp<arith::ConstantOp>();
    auto base =
        baseOp ? dyn_cast<IntegerAttr>(baseOp.getValue()) : IntegerAttr();
    if (!base || base.getInt() < 0 || base.getInt() > 63 ||
        base.getInt() + rows * cols / 4 > uint64_t(depth))
      return op.emitError("FP32 output base does not fit the configured bank");
    Location loc = op.getLoc();
    Value rs1 =
        packRs1BanksIter(b, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
                         adaptor.getResultBankId(), cstI64(b, loc, k));
    Value rs2 = cstI64(b, loc, matrixCfg(rows, cols, false, false));
    Value first =
        b.create<arith::ExtUIOp>(loc, b.getI64Type(), adaptor.getFirst());
    Value last =
        b.create<arith::ExtUIOp>(loc, b.getI64Type(), adaptor.getLast());
    rs2 = b.create<arith::OrIOp>(
        loc, rs2, b.create<arith::ShLIOp>(loc, first, cstI64(b, loc, 24)));
    rs2 = b.create<arith::OrIOp>(
        loc, rs2, b.create<arith::ShLIOp>(loc, last, cstI64(b, loc, 25)));
    rs2 = b.create<arith::OrIOp>(
        loc, rs2,
        b.create<arith::ShLIOp>(loc, adaptor.getOutputBase(),
                                cstI64(b, loc, 26)));
    b.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        b.getI32IntegerAttr(buckyball_target::getBuckyballFunct7(
            op.getFused() ? "MXMM_FMA32" : "MXMM_F32")));
    return success();
  }
  int64_t depth;
};

#include "WindowPatterns.inc"

} // namespace
namespace mlir::buddy::buckyball {
void populateMxmmBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable, int64_t depth) {
  patterns.add<MXFP8Lowering, MXFP8WindowLowering, FP32Lowering>(converter,
                                                                 stable, depth);
}
void configureMxmmBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool) {
  target.addIllegalOp<MXFP8Op, BankMXFP8Op, MXFP8WindowOp, BankMXFP8WindowOp,
                      FP32Op, BankFP32Op>();
}
} // namespace mlir::buddy::buckyball
