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
static LogicalResult validateTranspose(Operation *op, Value elemBits,
                                       Value iter, int64_t bankDepth) {
  auto constant = elemBits.getDefiningOp<arith::ConstantOp>();
  auto value =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  if (!value || (value.getInt() != 8 && value.getInt() != 32))
    return op->emitError("Transpose elemBits must be constant 8 or 32");
  auto iterOp = iter.getDefiningOp<arith::ConstantOp>();
  auto iterAttr =
      iterOp ? dyn_cast<IntegerAttr>(iterOp.getValue()) : IntegerAttr();
  if (!iterAttr || iterAttr.getInt() <= 0 || iterAttr.getInt() > bankDepth)
    return op->emitError(
        "Transpose iter must be a positive constant within the bank depth");
  return success();
}

struct TransposeLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  TransposeLowering(LLVMTypeConverter &converter, bool stable,
                    int64_t bankDepth)
      : ConvertOpToLLVMPattern<TransposeOp>(converter), stable(stable),
        bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("TransposeBall");
    Location loc = op.getLoc();
    if (failed(
            validateTranspose(op, op.getElemBits(), op.getIter(), bankDepth)))
      return failure();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<TransposeIntrOp>(op, rs1,
                                                   adaptor.getElemBits());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getElemBits(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("TRANSPOSE")));
    return success();
  }

private:
  bool stable = false;
  int64_t bankDepth;
};
struct BankTransposeLowering : public ConvertOpToLLVMPattern<BankTransposeOp> {
  BankTransposeLowering(LLVMTypeConverter &converter, bool stable,
                        int64_t bankDepth)
      : ConvertOpToLLVMPattern<BankTransposeOp>(converter), stable(stable),
        bankDepth(bankDepth) {}

  LogicalResult
  matchAndRewrite(BankTransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("TransposeBall");
    Location loc = op.getLoc();
    if (failed(
            validateTranspose(op, op.getElemBits(), op.getIter(), bankDepth)))
      return failure();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInBank(),
                                 cstI64(rewriter, loc, 0), adaptor.getOutBank(),
                                 adaptor.getIter());
    if (stable) {
      rewriter.replaceOpWithNewOp<TransposeIntrOp>(op, rs1,
                                                   adaptor.getElemBits());
      return success();
    }
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, adaptor.getElemBits(),
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7("TRANSPOSE")));
    return success();
  }

private:
  bool stable = false;
  int64_t bankDepth;
};
} // namespace

namespace mlir::buddy::buckyball {
void populateTransposeBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t bankDepth, bool) {
  patterns.add<TransposeLowering>(converter, stable, bankDepth);
  patterns.add<BankTransposeLowering>(converter, stable, bankDepth);
}

void configureTransposeBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                   bool stable) {
  if (stable)
    target.addLegalOp<TransposeIntrOp>();
  else
    target.addIllegalOp<TransposeIntrOp>();
  target.addIllegalOp<MemTransposeOp, TransposeOp, BankTransposeOp>();
}
} // namespace mlir::buddy::buckyball
