#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct MxquantLowering : public ConvertOpToLLVMPattern<MxquantOp> {
  MxquantLowering(LLVMTypeConverter &converter, int64_t bankDepth)
      : ConvertOpToLLVMPattern<MxquantOp>(converter), bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(MxquantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("MxquantBall");
    if (buckyball_target::getBuckyballTarget().bankWidthBits != 128)
      return op.emitError("mxquant: requires 16-byte bank rows");
    if (auto constant = op.getIter().getDefiningOp<arith::ConstantOp>()) {
      int64_t count = cast<IntegerAttr>(constant.getValue()).getInt();
      if (count <= 0 || count % 32)
        return op.emitError("mxquant: count must be a positive multiple of 32");
      if (count > bankDepth * 4 || count + count / 32 > bankDepth * 16)
        return op.emitError("mxquant: bank footprint exceeds capacity");
    }
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, cstI64(rewriter, loc, 0),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("MXQUANT")));
    return success();
  }

private:
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateMxquantBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool,
    int64_t bankDepth) {
  patterns.add<MxquantLowering>(converter, bankDepth);
}
void configureMxquantBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool) {
  target.addIllegalOp<MxquantOp, BankMxquantOp>();
}
} // namespace mlir::buddy::buckyball
