#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct Int32ToFp32Lowering : public ConvertOpToLLVMPattern<Int32ToFp32Op> {
  using ConvertOpToLLVMPattern<Int32ToFp32Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Int32ToFp32Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Int2FpBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 adaptor.getScaleBankId(),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, cstI64(rewriter, loc, op.getRelu() ? 1 : 0),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("INT32_TO_FP32")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateInt2FpBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t, bool) {
  (void)stable;
  patterns.add<Int32ToFp32Lowering>(converter);
}

void configureInt2FpBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable) {
  (void)stable;
  target.addIllegalOp<Int32ToFp32Op, BankInt32ToFp32Op>();
}
} // namespace mlir::buddy::buckyball
