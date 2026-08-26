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
struct Im2colLowering : public ConvertOpToLLVMPattern<Im2colOp> {
  Im2colLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<Im2colOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(Im2colOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Im2colBall");
    Location loc = op.getLoc();
    IntegerType i64 = rewriter.getI64Type();
    Value rs1 = rewriter.create<arith::OrIOp>(
        loc, i64, adaptor.getInputBankId(),
        rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputBankId(),
                                       cstI64(rewriter, loc, 20)));
    rs1 = rewriter.create<arith::OrIOp>(
        loc, rs1,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getIter(),
                                       cstI64(rewriter, loc, 30)));

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
};
} // namespace

namespace mlir::buddy::buckyball {
void populateIm2colBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t, bool) {
  patterns.add<Im2colLowering>(converter, stable);
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
