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
struct VecMat16Lowering : public ConvertOpToLLVMPattern<VecMat16Op> {
  using ConvertOpToLLVMPattern<VecMat16Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(VecMat16Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("VecBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getOp1BankId(),
                                 adaptor.getOp2BankId(), adaptor.getWrBankId(),
                                 adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getMode(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("VECMAT16")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateVecBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns,
                                                  bool stable, int64_t, bool) {
  (void)stable;
  patterns.add<VecMat16Lowering>(converter);
}

void configureVecBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                             bool stable) {
  (void)stable;
  target.addIllegalOp<VecMat16Op, BankVecMat16Op>();
}
} // namespace mlir::buddy::buckyball
