#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct Int2FpLowering : public ConvertOpToLLVMPattern<Int2FpOp> {
  Int2FpLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<Int2FpOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(Int2FpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<Int2FpIntrOp>(op, rs1, adaptor.getScale());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getScale(),
                                              rewriter.getI32IntegerAttr(52));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateInt2FpLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<Int2FpLowering>(converter, stable);
}

void configureInt2FpLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<Int2FpIntrOp>();
  else
    target.addIllegalOp<Int2FpIntrOp>();
  target.addIllegalOp<Int2FpOp, BankInt2FpOp>();
}
} // namespace mlir::buddy::buckyball
