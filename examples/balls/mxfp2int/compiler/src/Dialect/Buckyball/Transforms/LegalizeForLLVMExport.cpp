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
class Mxfp2IntLowering : public ConvertOpToLLVMPattern<Mxfp2IntOp> {
public:
  using ConvertOpToLLVMPattern<Mxfp2IntOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Mxfp2IntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Mxfp2IntBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getSpecial(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("MXFP2INT")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateMxfp2IntBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  (void)stable;
  (void)bankDepth;
  (void)rushB;
  patterns.add<Mxfp2IntLowering>(converter);
}

void configureMxfp2IntBallLegalizeForExportTarget(
    LLVMConversionTarget &target, bool stable) {
  (void)stable;
  target.addIllegalOp<Mxfp2IntOp, BankMxfp2IntOp>();
}
} // namespace mlir::buddy::buckyball
