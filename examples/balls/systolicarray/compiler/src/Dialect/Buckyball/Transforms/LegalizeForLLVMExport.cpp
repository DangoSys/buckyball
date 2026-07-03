#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct SystolicLowering : public ConvertOpToLLVMPattern<SystolicOp> {
  SystolicLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<SystolicOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(SystolicOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(
        rewriter, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
        adaptor.getResultBankId(), cstI64(rewriter, loc, 0));
    if (stable) {
      rewriter.replaceOpWithNewOp<SystolicIntrOp>(op, rs1, adaptor.getConfig());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getConfig(),
                                              rewriter.getI32IntegerAttr(65));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateSystolicLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable) {
  patterns.add<SystolicLowering>(converter, stable);
}

void configureSystolicLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool stable) {
  if (stable)
    target.addLegalOp<SystolicIntrOp>();
  else
    target.addIllegalOp<SystolicIntrOp>();
  target.addIllegalOp<SystolicOp>();
}
} // namespace mlir::buddy::buckyball
