#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/PatternMatch.h"
#include <cstdint>

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::buckyball::legalize;

namespace {
struct ReluLowering : public ConvertOpToLLVMPattern<ReluOp> {
  using ConvertOpToLLVMPattern<ReluOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value rs1 =
        packRs1BanksIter(rewriter, loc, adaptor.getBankId(), adaptor.getGroup(),
                         adaptor.getBankId(), adaptor.getIter());
    rs1 =
        rewriter.create<LLVM::OrOp>(loc, rs1, cstI64(rewriter, loc, INT64_MIN));
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getStride(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("RELU")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateReluBallLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                                   RewritePatternSet &patterns,
                                                   bool, int64_t) {
  patterns.add<ReluLowering>(converter);
}

void configureReluBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                              bool) {
  target.addIllegalOp<ReluOp, BankReluOp>();
}
} // namespace mlir::buddy::buckyball
