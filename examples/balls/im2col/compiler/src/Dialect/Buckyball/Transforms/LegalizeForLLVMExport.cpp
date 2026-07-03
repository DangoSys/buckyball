#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

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
    Location loc = op.getLoc();
    IntegerType i64 = rewriter.getI64Type();
    Value rs1 = rewriter.create<arith::OrIOp>(
        loc, i64, adaptor.getInputBankId(),
        rewriter.create<arith::ShLIOp>(loc, adaptor.getOutputBankId(),
                                       cstI64(rewriter, loc, 20)));

    Value rs2 = adaptor.getKcol();
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getKrow(),
                                       cstI64(rewriter, loc, 8)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getIncol(),
                                       cstI64(rewriter, loc, 16)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getInrow(),
                                       cstI64(rewriter, loc, 24)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getStartcol(),
                                       cstI64(rewriter, loc, 32)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getStartrow(),
                                       cstI64(rewriter, loc, 40)));
    rs2 = rewriter.create<arith::OrIOp>(
        loc, rs2,
        rewriter.create<arith::ShLIOp>(loc, adaptor.getColStep(),
                                       cstI64(rewriter, loc, 48)));

    if (stable) {
      rewriter.replaceOpWithNewOp<Im2colIntrOp>(op, rs1, rs2);
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, rs2,
                                              rewriter.getI32IntegerAttr(48));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateIm2colLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<Im2colLowering>(converter, stable);
}

void configureIm2colLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<Im2colIntrOp>();
  else
    target.addIllegalOp<Im2colIntrOp>();
  target.addIllegalOp<Im2colOp, BankIm2colOp>();
}
} // namespace mlir::buddy::buckyball
