#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "ballISA.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct VecMat16Lowering : public ConvertOpToLLVMPattern<VecMat16Op> {
  VecMat16Lowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<VecMat16Op>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(VecMat16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getOp1BankId(),
                                 adaptor.getOp2BankId(), adaptor.getWrBankId(),
                                 adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<VecMat16IntrOp>(op, rs1, adaptor.getMode());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getMode(),
        rewriter.getI32IntegerAttr(BB_FUNC7_VECMAT16));
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
  patterns.add<VecMat16Lowering>(converter, stable);
}

void configureVectorLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<VecMat16IntrOp>();
  else
    target.addIllegalOp<VecMat16IntrOp>();
  target.addIllegalOp<VecMat16Op, BankVecMat16Op>();
}
} // namespace mlir::buddy::buckyball
