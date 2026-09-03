#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct LutLowering : public ConvertOpToLLVMPattern<LutOp> {
  LutLowering(LLVMTypeConverter &converter, int64_t bankDepth)
      : ConvertOpToLLVMPattern<LutOp>(converter), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(LutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("LutBall");
    auto iterOp = op.getIter().getDefiningOp<arith::ConstantOp>();
    auto iterAttr =
        iterOp ? dyn_cast<IntegerAttr>(iterOp.getValue()) : IntegerAttr();
    if (!iterAttr || iterAttr.getInt() <= 0 || iterAttr.getInt() > bankDepth)
      return op.emitError(
          "LUT iter must be a positive constant within one bank");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 adaptor.getLutBankId(),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, cstI64(rewriter, loc, 0),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("LUT")));
    return success();
  }

private:
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateLutBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                  RewritePatternSet &patterns,
                                                  bool stable,
                                                  int64_t bankDepth,
                                                  bool rushB) {
  (void)stable;
  (void)rushB;
  patterns.add<LutLowering>(converter, bankDepth);
}

void configureLutBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                             bool stable) {
  (void)stable;
  target.addIllegalOp<LutOp, BankLutOp>();
}
} // namespace mlir::buddy::buckyball
