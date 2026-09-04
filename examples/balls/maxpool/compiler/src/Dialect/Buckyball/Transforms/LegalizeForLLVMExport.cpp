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
struct MaxPoolLowering : public ConvertOpToLLVMPattern<MaxPoolOp> {
  MaxPoolLowering(LLVMTypeConverter &converter, int64_t bankDepth)
      : ConvertOpToLLVMPattern<MaxPoolOp>(converter), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(MaxPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("MaxPoolBall");
    int64_t inputSide = op.getInputSide();
    int64_t outputSide = op.getOutputSide();
    int64_t kernel = op.getKernel();
    int64_t stride = op.getStride();
    int64_t padding = op.getPadding();
    int64_t startRow = op.getStartRow();
    int64_t startCol = op.getStartCol();
    auto iterOp = op.getIter().getDefiningOp<arith::ConstantOp>();
    auto iterAttr =
        iterOp ? dyn_cast<IntegerAttr>(iterOp.getValue()) : IntegerAttr();
    if (!iterAttr || inputSide <= 0 || inputSide > 15 || outputSide <= 0 ||
        outputSide > 15 || kernel <= 0 || kernel > 15 || stride <= 0 ||
        stride > 15 || padding < 0 || padding > 15 || startRow < 0 ||
        startRow > 15 || startCol < 0 || startCol > 15 ||
        inputSide * inputSide > bankDepth ||
        outputSide * outputSide != iterAttr.getInt() ||
        inputSide + 2 * padding < kernel + startRow ||
        inputSide + 2 * padding < kernel + startCol ||
        startRow + (outputSide - 1) * stride + kernel >
            inputSide + 2 * padding ||
        startCol + (outputSide - 1) * stride + kernel > inputSide + 2 * padding)
      return op.emitError("MaxPool shape does not fit one physical bank tile");

    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    uint64_t geometry = static_cast<uint64_t>(inputSide) |
                        (static_cast<uint64_t>(outputSide) << 4) |
                        (static_cast<uint64_t>(kernel) << 8) |
                        (static_cast<uint64_t>(stride) << 12) |
                        (static_cast<uint64_t>(padding) << 16) |
                        (static_cast<uint64_t>(startRow) << 38) |
                        (static_cast<uint64_t>(startCol) << 42);
    Value rs2 = cstI64(rewriter, loc, geometry);
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getInputBase(),
                                       cstI64(rewriter, loc, 20)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputBase(),
                                       cstI64(rewriter, loc, 26)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputStride(),
                                       cstI64(rewriter, loc, 32)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("MAXPOOL")));
    return success();
  }

private:
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateMaxPoolBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  (void)stable;
  (void)rushB;
  patterns.add<MaxPoolLowering>(converter, bankDepth);
}

void configureMaxPoolBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool stable) {
  (void)stable;
  target.addIllegalOp<MaxPoolOp, BankMaxPoolOp>();
}
} // namespace mlir::buddy::buckyball
