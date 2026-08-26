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
struct ReluLowering : public ConvertOpToLLVMPattern<ReluOp> {
  using ConvertOpToLLVMPattern<ReluOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("ReluBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getDepth());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getStride(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("RELU")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateReluBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool stable, int64_t, bool) {
  (void)stable;
  patterns.add<ReluLowering>(converter);
}

void configureReluBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool stable) {
  (void)stable;
  target.addIllegalOp<ReluOp>();
}
} // namespace mlir::buddy::buckyball
