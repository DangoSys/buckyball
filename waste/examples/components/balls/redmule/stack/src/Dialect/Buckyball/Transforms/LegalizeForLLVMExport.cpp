#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"

using namespace mlir;
using namespace ::buddy::buckyball;
using namespace ::buddy::buckyball::legalize;

namespace {
class RedmuleLowering : public ConvertOpToLLVMPattern<RedmuleF16Op> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(RedmuleF16Op op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value rs1 =
        packRs1BanksIter(rewriter, loc, operands.getA(), operands.getB(),
                         operands.getC(), cstI64(rewriter, loc, op.getK()));
    uint64_t special =
        uint64_t(op.getRows()) | uint64_t(op.getOp1Base()) << 12 |
        uint64_t(op.getOp2Base()) << 22 | uint64_t(op.getWrBase()) << 32 |
        uint64_t(op.getFirst()) << 42;
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, cstI64(rewriter, loc, special),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("REDMULE_F16")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateRedmuleBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool, int64_t) {
  patterns.add<RedmuleLowering>(converter);
}
void configureRedmuleBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                 bool) {
  target.addIllegalOp<RedmuleF16Op, BankRedmuleF16Op>();
}
} // namespace mlir::buddy::buckyball
