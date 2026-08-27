#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct MatAddLowering : public ConvertOpToLLVMPattern<MatAddOp> {
  using ConvertOpToLLVMPattern<MatAddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("MatAddBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getABankId(),
                                 adaptor.getBBankId(), adaptor.getCBankId(),
                                 adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, cstI64(rewriter, loc, 0),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("MATADD")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateMatAddLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool) {
  patterns.add<MatAddLowering>(converter);
}

void configureMatAddLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool) {
  target.addIllegalOp<MatAddOp, BankMatAddOp>();
}
} // namespace mlir::buddy::buckyball
