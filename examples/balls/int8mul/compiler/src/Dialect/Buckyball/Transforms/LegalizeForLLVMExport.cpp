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
struct Int8MulLowering : public ConvertOpToLLVMPattern<Int8MulOp> {
  using ConvertOpToLLVMPattern<Int8MulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Int8MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Int8MulBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getGateBankId(),
                                 adaptor.getInputBankId(),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    Value ratioBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), adaptor.getRatio());
    Value ratio =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), ratioBits);
    Value gateRow = rewriter.create<arith::ShLIOp>(loc, adaptor.getGateRow(),
                                                   cstI64(rewriter, loc, 32));
    Value rs2 = rewriter.create<arith::OrIOp>(loc, ratio, gateRow);
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("INT8MUL")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateInt8MulBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool, int64_t,
    bool) {
  patterns.add<Int8MulLowering>(converter);
}

void configureInt8MulBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool) {
  target.addIllegalOp<Int8MulOp, BankInt8MulOp>();
}
} // namespace mlir::buddy::buckyball
