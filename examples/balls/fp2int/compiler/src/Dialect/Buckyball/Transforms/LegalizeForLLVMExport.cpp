#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct Fp2IntLowering : public ConvertOpToLLVMPattern<Fp2IntOp> {
  Fp2IntLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<Fp2IntOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(Fp2IntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<Fp2IntIntrOp>(op, rs1, adaptor.getScale());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getScale(),
                                              rewriter.getI32IntegerAttr(51));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateFp2IntLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<Fp2IntLowering>(converter, stable);
}

void configureFp2IntLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<Fp2IntIntrOp>();
  else
    target.addIllegalOp<Fp2IntIntrOp>();
  target.addIllegalOp<Fp2IntOp, BankFp2IntOp>();
}
} // namespace mlir::buddy::buckyball
