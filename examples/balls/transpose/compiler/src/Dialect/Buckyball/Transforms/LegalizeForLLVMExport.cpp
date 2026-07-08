#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct TransposeLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  TransposeLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<TransposeOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<TransposeIntrOp>(op, rs1, adaptor.getMode());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getMode(),
                                              rewriter.getI32IntegerAttr(49));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateTransposeLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable) {
  patterns.add<TransposeLowering>(converter, stable);
}

void configureTransposeLegalizeForExportTarget(LLVMConversionTarget &target,
                                               bool stable) {
  if (stable)
    target.addLegalOp<TransposeIntrOp>();
  else
    target.addIllegalOp<TransposeIntrOp>();
  target.addIllegalOp<TransposeOp, BankTransposeOp>();
}
} // namespace mlir::buddy::buckyball
