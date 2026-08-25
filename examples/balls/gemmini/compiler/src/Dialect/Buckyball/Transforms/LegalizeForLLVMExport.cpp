#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;

namespace {
class GemminiInstructionLowering
    : public ConvertOpToLLVMPattern<GemminiInstructionOp> {
public:
  using ConvertOpToLLVMPattern<GemminiInstructionOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemminiInstructionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("GemminiBall");
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, adaptor.getRs1(), adaptor.getRs2(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7(op.getMnemonic())));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateGemminiBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  (void)stable;
  (void)bankDepth;
  (void)rushB;
  patterns.add<GemminiInstructionLowering>(converter);
}

void configureGemminiBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                  bool stable) {
  (void)stable;
  target.addIllegalOp<GemminiInstructionOp>();
}
} // namespace mlir::buddy::buckyball
