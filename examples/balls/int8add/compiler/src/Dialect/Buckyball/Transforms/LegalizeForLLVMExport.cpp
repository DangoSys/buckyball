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
struct Int8AddLowering : public ConvertOpToLLVMPattern<Int8AddOp> {
  using ConvertOpToLLVMPattern<Int8AddOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Int8AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Int8AddBall");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getLhsBankId(),
                                 adaptor.getRhsBankId(),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    Value lhsBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), adaptor.getLhsRatio());
    Value rhsBits = rewriter.create<arith::BitcastOp>(
        loc, rewriter.getI32Type(), adaptor.getRhsRatio());
    Value lhs64 =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), lhsBits);
    Value rhs64 =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), rhsBits);
    Value rs2 = rewriter.create<arith::OrIOp>(
        loc, lhs64,
        rewriter.create<arith::ShLIOp>(loc, rhs64, cstI64(rewriter, loc, 32)));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(buckyball_target::getBuckyballFunct7(
            op.getRelu() ? "INT8ADD_RELU" : "INT8ADD")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateInt8AddBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool, int64_t,
    bool) {
  patterns.add<Int8AddLowering>(converter);
}

void configureInt8AddBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool) {
  target.addIllegalOp<Int8AddOp, BankInt8AddOp>();
}
} // namespace mlir::buddy::buckyball
