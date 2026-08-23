#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <type_traits>

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "ballISA.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {

template <typename Op>
struct Int2FpLowering : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::APInt daAddr(64, 0);
    if (!matchPattern(op.getDaAddr(), m_ConstantInt(&daAddr)) ||
        daAddr.getSExtValue() != 0)
      return op.emitError("INT2FP Da address must be 0");
    llvm::APInt dwAddr(64, 0);
    if (!matchPattern(op.getDwAddr(), m_ConstantInt(&dwAddr)) ||
        dwAddr.getSExtValue() < 16 || dwAddr.getSExtValue() % 4 != 0)
      return op.emitError("INT2FP Dw address must be >= 16 and 4-byte aligned");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    Value dw = rewriter.create<arith::ShLIOp>(loc, adaptor.getDwAddr(),
                                              cstI64(rewriter, loc, 13));
    Value rs2 = rewriter.create<arith::OrIOp>(loc, adaptor.getDaAddr(), dw);
    constexpr int funct7 = std::is_same_v<Op, Int2FpTensorOp>
                               ? BB_FUNC7_INT2FP_TENSOR
                               : BB_FUNC7_INT2FP_CHANNEL;
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2, rewriter.getI32IntegerAttr(funct7));
    return success();
  }
};

} // namespace

namespace mlir::buddy::buckyball {
void populateInt2FpLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 bool stable) {
  (void)stable;
  patterns.add<Int2FpLowering<Int2FpTensorOp>, Int2FpLowering<Int2FpChannelOp>>(
      converter);
}

void configureInt2FpLegalizeForExportTarget(LLVMConversionTarget &target,
                                            bool stable) {
  (void)stable;
  target.addIllegalOp<Int2FpTensorOp, Int2FpChannelOp, BankInt2FpTensorOp,
                      BankInt2FpChannelOp>();
}
} // namespace mlir::buddy::buckyball
