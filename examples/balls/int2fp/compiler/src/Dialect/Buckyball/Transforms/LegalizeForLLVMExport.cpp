#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {

Value packInt2FpRs2(ConversionPatternRewriter &rewriter, Location loc,
                    Value scale, uint64_t outputMode) {
  Value scaleLow = rewriter.create<arith::AndIOp>(
      loc, scale, cstI64(rewriter, loc, 0xffffffffULL));
  if (outputMode == 0)
    return scaleLow;
  Value mode = cstI64(rewriter, loc, outputMode << 32);
  return rewriter.create<arith::OrIOp>(loc, scaleLow, mode);
}

template <typename OpTy>
LogicalResult lowerIntConvert(OpTy op, typename OpTy::Adaptor adaptor,
                              ConversionPatternRewriter &rewriter, bool stable,
                              uint64_t outputMode) {
  Location loc = op.getLoc();
  Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                               cstI64(rewriter, loc, 0),
                               adaptor.getOutputBankId(), adaptor.getIter());
  Value rs2 = packInt2FpRs2(rewriter, loc, adaptor.getScale(), outputMode);
  if (stable) {
    rewriter.replaceOpWithNewOp<Int2FpIntrOp>(op, rs1, rs2);
    return success();
  }
  rewriter.replaceOpWithNewOp<CustomIntrOp>(op, rs1, rs2,
                                            rewriter.getI32IntegerAttr(52));
  return success();
}

struct Int2FpLowering : public ConvertOpToLLVMPattern<Int2FpOp> {
  Int2FpLowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<Int2FpOp>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(Int2FpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerIntConvert(op, adaptor, rewriter, stable, /*outputMode=*/0);
  }

private:
  bool stable = false;
};

struct Int32ToInt8Lowering : public ConvertOpToLLVMPattern<Int32ToInt8Op> {
  Int32ToInt8Lowering(LLVMTypeConverter &converter, bool stable)
      : ConvertOpToLLVMPattern<Int32ToInt8Op>(converter), stable(stable) {}

  LogicalResult
  matchAndRewrite(Int32ToInt8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerIntConvert(op, adaptor, rewriter, stable, /*outputMode=*/1);
  }

private:
  bool stable = false;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateInt2FpLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  patterns.add<Int2FpLowering, Int32ToInt8Lowering>(converter, stable);
}

void configureInt2FpLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  if (stable)
    target.addLegalOp<Int2FpIntrOp>();
  else
    target.addIllegalOp<Int2FpIntrOp>();
  target
      .addIllegalOp<Int2FpOp, BankInt2FpOp, Int32ToInt8Op, BankInt32ToInt8Op>();
}
} // namespace mlir::buddy::buckyball
