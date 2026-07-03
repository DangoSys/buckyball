#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct ReluLowering : public ConvertOpToLLVMPattern<ReluOp> {
  ReluLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<ReluOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getDepth());
    if (stable) {
      rewriter.replaceOpWithNewOp<ReluIntrOp>(op, rs1, adaptor.getStride());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getStride(),
                                              rewriter.getI32IntegerAttr(50));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateReluLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns,
                                               bool stable) {
  patterns.add<ReluLowering>(converter, stable);
}

void configureReluLegalizeForExportTarget(LLVMConversionTarget &target,
                                          bool stable) {
  if (stable)
    target.addLegalOp<ReluIntrOp>();
  else
    target.addIllegalOp<ReluIntrOp>();
  target.addIllegalOp<ReluOp>();
}
} // namespace mlir::buddy::buckyball
