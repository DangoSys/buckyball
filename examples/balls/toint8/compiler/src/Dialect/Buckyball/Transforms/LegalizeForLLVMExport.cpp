#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct QuantF32ToI8Lowering : public ConvertOpToLLVMPattern<QuantF32ToI8Op> {
  using ConvertOpToLLVMPattern<QuantF32ToI8Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QuantF32ToI8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("ToInt8Ball");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    Value scaleBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), adaptor.getScale());
    Value rs2 =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), scaleBits);
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("QUANT_F32_TO_I8")));
    return success();
  }
};

struct QuantI32ToI8Lowering : public ConvertOpToLLVMPattern<QuantI32ToI8Op> {
  QuantI32ToI8Lowering(LLVMTypeConverter &converter, int64_t bankDepth)
      : ConvertOpToLLVMPattern<QuantI32ToI8Op>(converter),
        bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(QuantI32ToI8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("ToInt8Ball");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 adaptor.getScaleBankId(),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    auto iterOp = op.getIter().getDefiningOp<arith::ConstantOp>();
    auto iterAttr =
        iterOp ? dyn_cast<IntegerAttr>(iterOp.getValue()) : IntegerAttr();
    if (!iterAttr || iterAttr.getInt() <= 0)
      return op.emitError("INT32 to INT8 iter must be a positive constant");
    if (iterAttr.getInt() % 4 != 0 || iterAttr.getInt() > bankDepth)
      return op.emitError(
          "INT32 to INT8 iter must be a multiple of four within bank depth");
    int64_t outputRows = iterAttr.getInt() / 4;
    int64_t outputWidth = op.getOutputWidthAttr().getInt();
    int64_t outputHeight = op.getOutputHeightAttr().getInt();
    int64_t outputStride = op.getOutputStrideAttr().getInt();
    auto outputBaseOp =
        adaptor.getOutputBase().getDefiningOp<arith::ConstantOp>();
    auto inputBaseOp =
        adaptor.getInputBase().getDefiningOp<arith::ConstantOp>();
    auto outputBaseAttr = outputBaseOp
                              ? dyn_cast<IntegerAttr>(outputBaseOp.getValue())
                              : IntegerAttr();
    auto inputBaseAttr = inputBaseOp
                             ? dyn_cast<IntegerAttr>(inputBaseOp.getValue())
                             : IntegerAttr();
    if (outputWidth <= 0 || outputHeight <= 0 || outputStride < outputWidth ||
        outputWidth * outputHeight != outputRows || outputWidth >= 128 ||
        outputHeight >= 128 || outputStride >= 128 || !outputBaseAttr ||
        !inputBaseAttr || outputBaseAttr.getInt() < 0 ||
        inputBaseAttr.getInt() < 0 ||
        outputBaseAttr.getInt() + (outputHeight - 1) * outputStride +
                outputWidth >
            bankDepth ||
        inputBaseAttr.getInt() + iterAttr.getInt() > bankDepth)
      return op.emitError(
          "INT32 to INT8 output tile is invalid for the physical bank");
    Value rs2 = rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputBase(),
                                               cstI64(rewriter, loc, 1));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        cstI64(rewriter, loc,
               (outputWidth << 8) | (outputHeight << 15) |
                   (outputStride << 22) | (op.getRelu() ? 1 : 0)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getInputBase(),
                                       cstI64(rewriter, loc, 29)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("QUANT_I32_TO_I8")));
    return success();
  }

private:
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateToInt8BallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool) {
  (void)stable;
  patterns.add<QuantF32ToI8Lowering>(converter);
  patterns.add<QuantI32ToI8Lowering>(converter, bankDepth);
}

void configureToInt8BallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable) {
  (void)stable;
  target.addIllegalOp<QuantF32ToI8Op, BankQuantF32ToI8Op, QuantI32ToI8Op,
                      BankQuantI32ToI8Op>();
}
} // namespace mlir::buddy::buckyball
