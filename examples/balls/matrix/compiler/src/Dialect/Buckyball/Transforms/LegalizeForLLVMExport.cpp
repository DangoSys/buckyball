#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct MatrixLowering : public ConvertOpToLLVMPattern<MatrixOp> {
  MatrixLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<MatrixOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(MatrixOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(
        rewriter, loc, adaptor.getOp1BankId(), adaptor.getOp2BankId(),
        adaptor.getResultBankId(), cstI64(rewriter, loc, 0));
    if (stable) {
      rewriter.replaceOpWithNewOp<MatrixIntrOp>(op, rs1, adaptor.getConfig());
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
void populateMatrixLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<MatrixLowering>(converter, stable);
}

void configureMatrixLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<MatrixIntrOp>();
  else
    target.addIllegalOp<MatrixIntrOp>();
  target.addIllegalOp<MatrixOp>();
}
} // namespace mlir::buddy::buckyball
