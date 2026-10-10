#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
class F32AddLowering : public ConvertOpToLLVMPattern<F32AddOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(F32AddOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("F32AddBall");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || op.getRows() > target.bankDepth)
      return op.emitError(
          "F32ADD rows exceed the target 128-bit bank capacity");
    Location loc = op.getLoc();
    Value rs1 =
        packRs1BanksIter(rewriter, loc, operands.getA(), operands.getB(),
                         operands.getC(), cstI64(rewriter, loc, op.getRows()));
    Value rs2 = cstI64(rewriter, loc, op.getGroup() | (op.getFirst() ? 32 : 0));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("F32ADD")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateF32AddBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool, int64_t) {
  patterns.add<F32AddLowering>(converter);
}

void configureF32AddBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool) {
  target.addIllegalOp<F32AddOp, BankF32AddOp>();
}
} // namespace mlir::buddy::buckyball
