#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct MulWarp16Lowering : public ConvertOpToLLVMPattern<MulWarp16Op> {
  MulWarp16Lowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<MulWarp16Op>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(MulWarp16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getOp1BankId(),
                                 adaptor.getOp2BankId(), adaptor.getWrBankId(),
                                 adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<MulWarp16IntrOp>(op, rs1, adaptor.getMode());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, adaptor.getMode(),
                                              rewriter.getI32IntegerAttr(64));
    return success();
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateVectorLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<MulWarp16Lowering>(converter, stable);
}

void configureVectorLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<MulWarp16IntrOp>();
  else
    target.addIllegalOp<MulWarp16IntrOp>();
  target.addIllegalOp<MulWarp16Op, BankMulWarp16Op>();
}
} // namespace mlir::buddy::buckyball
