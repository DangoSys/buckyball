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
class BdbCounterLowering : public ConvertOpToLLVMPattern<BdbCounterOp> {
public:
  using ConvertOpToLLVMPattern<BdbCounterOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BdbCounterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("TraceBall");
    Location loc = op.getLoc();
    Value counter = rewriter.create<arith::ShLIOp>(loc, adaptor.getCounterId(),
                                                   cstI64(rewriter, loc, 4));
    Value payload = rewriter.create<arith::ShLIOp>(loc, adaptor.getPayload(),
                                                   cstI64(rewriter, loc, 8));
    Value rs2 =
        rewriter.create<arith::OrIOp>(loc, adaptor.getSubcommand(), counter);
    rs2 = rewriter.create<arith::OrIOp>(loc, rs2, payload);
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, cstI64(rewriter, loc, 0), rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("BDB_COUNTER")));
    return success();
  }
};
} // namespace

namespace mlir::buddy::buckyball {
void populateTraceBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool rushB) {
  (void)stable;
  (void)bankDepth;
  (void)rushB;
  patterns.add<BdbCounterLowering>(converter);
}

void configureTraceBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                               bool stable) {
  (void)stable;
  target.addIllegalOp<BdbCounterOp>();
}
} // namespace mlir::buddy::buckyball
