#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {
struct Fp2IntLowering : public ConvertOpToLLVMPattern<Fp2IntOp> {
  using ConvertOpToLLVMPattern<Fp2IntOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Fp2IntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Fp2IntBall");
    llvm::APInt daAddr(64, 0);
    if (!matchPattern(op.getDaAddr(), m_ConstantInt(&daAddr)) ||
        daAddr.getSExtValue() != 0)
      return op.emitError("FP2INT Da address must be 0");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getDaAddr(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("FP2INT")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateFp2IntBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t, bool) {
  (void)stable;
  patterns.add<Fp2IntLowering>(converter);
}

void configureFp2IntBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable) {
  (void)stable;
  target.addIllegalOp<Fp2IntOp, BankFp2IntOp>();
}
} // namespace mlir::buddy::buckyball
