#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
static LogicalResult
validateIm2colEncoding(Operation *op, Value iterValue, Value ksizeValue,
                       Value strideValue, Value paddingValue, int64_t inputBase,
                       int64_t lane, int64_t startRow, int64_t startCol,
                       int64_t windowStart, int64_t windowCount,
                       int64_t bankDepth) {
  auto getConst = [](Value value) -> std::optional<int64_t> {
    if (auto constant = value.getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
        return attr.getInt();
    return std::nullopt;
  };
  auto iter = getConst(iterValue);
  auto ksize = getConst(ksizeValue);
  auto stride = getConst(strideValue);
  auto padding = getConst(paddingValue);
  if (!iter || !ksize || !stride || !padding)
    return op->emitError(
        "Im2col instruction requires constant iter/ksize/stride/padding");
  if (*iter <= 0 || *ksize <= 0 || *ksize > 255 || *stride <= 0 ||
      *stride > 255 || *padding < 0 || *padding > 255 || startRow < 0 ||
      startRow > 255 || startCol < 0 || startCol > 255 || inputBase < 0 ||
      inputBase >= 64 || lane < 0 || lane >= 16)
    return op->emitError("Im2col instruction field out of range");
  int64_t inputRows = *iter * *iter;
  if (inputBase + inputRows > bankDepth)
    return op->emitError("Im2col input range exceeds the physical bank depth");
  if (startRow > *padding || startCol > *padding)
    return op->emitError("Im2col startRow/startCol must not exceed padding");
  if (windowStart < 0 || windowStart >= 64 || windowCount <= 0 ||
      windowCount > 64)
    return op->emitError(
        "Im2col requires windowStart in [0, 63] and windowCount in [1, 64]");

  int64_t padded = *iter + 2 * *padding;
  if (padded < *ksize + startRow || padded < *ksize + startCol ||
      (padded - *ksize - startRow) % *stride ||
      (padded - *ksize - startCol) % *stride)
    return op->emitError("Im2col shape does not produce integral windows");
  int64_t rows = (padded - *ksize - startRow) / *stride + 1;
  int64_t cols = (padded - *ksize - startCol) / *stride + 1;
  if (windowStart + windowCount > rows * cols)
    return op->emitError("Im2col window range exceeds output shape");
  int64_t kernelTiles = (*ksize * *ksize + 15) / 16;
  int64_t outputRows = (windowCount + 15) / 16 * kernelTiles * 16;
  if (outputRows > bankDepth)
    return op->emitError("Im2col output range exceeds the physical bank depth");
  return success();
}

struct Im2colLowering : public ConvertOpToLLVMPattern<Im2colOp> {
  Im2colLowering(LLVMTypeConverter &converter, bool stable, int64_t bankDepth)
      : ConvertOpToLLVMPattern<Im2colOp>(converter), stable(stable),
        bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(Im2colOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Im2colBall");
    if (failed(validateIm2colEncoding(
            op, op.getIter(), op.getKsize(), op.getStride(), op.getPadding(),
            op.getInputBase(), op.getLane(), op.getStartRow(), op.getStartCol(),
            op.getWindowStart(), op.getWindowCount(), bankDepth)))
      return failure();
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());

    Value rs2 = adaptor.getKsize();
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getStride(),
                                       cstI64(rewriter, loc, 8)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getPadding(),
                                       cstI64(rewriter, loc, 16)));
    Value startCol = cstI64(rewriter, loc, op.getStartCol());
    Value startRow = cstI64(rewriter, loc, op.getStartRow());
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startCol,
                                       cstI64(rewriter, loc, 24)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startRow,
                                       cstI64(rewriter, loc, 32)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc,
                                       cstI64(rewriter, loc, op.getInputBase()),
                                       cstI64(rewriter, loc, 40)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, cstI64(rewriter, loc, op.getLane()),
                                       cstI64(rewriter, loc, 46)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(
            loc, cstI64(rewriter, loc, op.getWindowStart()),
            cstI64(rewriter, loc, 50)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(
            loc, cstI64(rewriter, loc, op.getWindowCount()),
            cstI64(rewriter, loc, 56)));

    if (stable) {
      rewriter.replaceOpWithNewOp<Im2colIntrOp>(op, rs1, rs2);
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("IM2COL")));
    return success();
  }

private:
  bool stable = false;
  int64_t bankDepth;
};
struct BankIm2colLowering : public ConvertOpToLLVMPattern<BankIm2colOp> {
  BankIm2colLowering(LLVMTypeConverter &converter, bool stable,
                     int64_t bankDepth)
      : ConvertOpToLLVMPattern<BankIm2colOp>(converter), stable(stable),
        bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(BankIm2colOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Im2colBall");
    if (failed(validateIm2colEncoding(
            op, op.getIter(), op.getKsize(), op.getStride(), op.getPadding(),
            op.getInputBase(), op.getLane(), op.getStartRow(), op.getStartCol(),
            op.getWindowStart(), op.getWindowCount(), bankDepth)))
      return failure();
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInBank(),
                                 cstI64(rewriter, loc, 0), adaptor.getOutBank(),
                                 adaptor.getIter());

    Value rs2 = adaptor.getKsize();
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getStride(),
                                       cstI64(rewriter, loc, 8)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getPadding(),
                                       cstI64(rewriter, loc, 16)));
    Value startCol = cstI64(rewriter, loc, op.getStartCol());
    Value startRow = cstI64(rewriter, loc, op.getStartRow());
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startCol,
                                       cstI64(rewriter, loc, 24)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, startRow,
                                       cstI64(rewriter, loc, 32)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc,
                                       cstI64(rewriter, loc, op.getInputBase()),
                                       cstI64(rewriter, loc, 40)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, cstI64(rewriter, loc, op.getLane()),
                                       cstI64(rewriter, loc, 46)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(
            loc, cstI64(rewriter, loc, op.getWindowStart()),
            cstI64(rewriter, loc, 50)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(
            loc, cstI64(rewriter, loc, op.getWindowCount()),
            cstI64(rewriter, loc, 56)));

    if (stable) {
      rewriter.replaceOpWithNewOp<Im2colIntrOp>(op, rs1, rs2);
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("IM2COL")));
    return success();
  }

private:
  bool stable = false;
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateIm2colBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool) {
  patterns.add<Im2colLowering>(converter, stable, bankDepth);
  patterns.add<BankIm2colLowering>(converter, stable, bankDepth);
}

void configureIm2colBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable) {
  if (stable)
    target.addLegalOp<Im2colIntrOp>();
  else
    target.addIllegalOp<Im2colIntrOp>();
  target.addIllegalOp<Im2colMatmulOp, Im2colFatMatmulOp, Im2colOp,
                      BankIm2colOp>();
}
} // namespace mlir::buddy::buckyball
