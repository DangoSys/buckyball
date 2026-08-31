#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <type_traits>

#include "Buckyball/BuckyballOps.h"
#include "Dialect/Buckyball/Transforms/LegalizeForLLVMExportBase.h"
#include "Target/BuckyballTargetRegistry.h"

using namespace mlir;
using namespace buddy::buckyball;
using namespace buddy::buckyball::legalize;

namespace {

static std::optional<int64_t> getConstantInt(Value value) {
  llvm::APInt constant(64, 0);
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

// Channel lowering advances Dw by four bytes for each output channel.  Keep
// accepting literal addresses, but also recognize the bounded affine form
// emitted by the im2col/matmul-to-bank-SSA lowering:
//
//   dwBase + index_cast(scf.for induction variable) * byteStride
//
// This is deliberately narrower than accepting an arbitrary runtime value: it
// proves the minimum address, alignment, and the 13-bit instruction field.
static bool isValidDynamicChannelDwAddr(Value value) {
  auto add = value.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  Value offset;
  std::optional<int64_t> base = getConstantInt(add.getLhs());
  if (base)
    offset = add.getRhs();
  else {
    base = getConstantInt(add.getRhs());
    offset = add.getLhs();
  }
  constexpr int64_t maxDwAddr = (1 << 13) - 1;
  if (!base || *base < 16 || *base > maxDwAddr || *base % 4 != 0)
    return false;

  auto mul = offset.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value induction;
  std::optional<int64_t> byteStride = getConstantInt(mul.getLhs());
  if (byteStride)
    induction = mul.getRhs();
  else {
    byteStride = getConstantInt(mul.getRhs());
    induction = mul.getLhs();
  }
  if (!byteStride || *byteStride <= 0 || *byteStride % 4 != 0)
    return false;

  auto cast = induction.getDefiningOp<arith::IndexCastOp>();
  if (!cast)
    return false;
  auto blockArg = dyn_cast<BlockArgument>(cast.getIn());
  if (!blockArg)
    return false;
  auto loop = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!loop || blockArg != loop.getInductionVar())
    return false;

  std::optional<int64_t> lower = getConstantInt(loop.getLowerBound());
  std::optional<int64_t> upper = getConstantInt(loop.getUpperBound());
  std::optional<int64_t> step = getConstantInt(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *upper <= *lower || *step <= 0)
    return false;

  // Dw occupies special[25:13], hence its byte address must fit in 13 bits.
  // Use upper - 1 as a conservative bound for the induction variable.
  if (*upper - 1 > (maxDwAddr - *base) / *byteStride)
    return false;
  return true;
}

template <typename Op>
struct Int2FpLowering : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    buckyball_target::requireBuckyballBall("Int2FpBall");
    llvm::APInt daAddr(64, 0);
    if (!matchPattern(op.getDaAddr(), m_ConstantInt(&daAddr)) ||
        daAddr.getSExtValue() != 0)
      return op.emitError("INT2FP Da address must be 0");
    std::optional<int64_t> dwAddr = getConstantInt(op.getDwAddr());
    bool validConstant = dwAddr && *dwAddr >= 16 &&
                         *dwAddr <= ((1 << 13) - 1) && *dwAddr % 4 == 0;
    bool validDynamic = std::is_same_v<Op, Int2FpChannelOp> &&
                        isValidDynamicChannelDwAddr(op.getDwAddr());
    if (!validConstant && !validDynamic)
      return op.emitError("INT2FP Dw address must be >= 16 and 4-byte aligned");
    if (adaptor.getInputBankId() == adaptor.getOutputBankId())
      return op.emitError("INT2FP forbids in-place dequantization");
    Location loc = op.getLoc();
    Value rs1 = packRs1BanksIter(rewriter, loc, adaptor.getInputBankId(),
                                 cstI64(rewriter, loc, 0),
                                 adaptor.getOutputBankId(), adaptor.getIter());
    Value dw = rewriter.create<arith::ShLIOp>(loc, adaptor.getDwAddr(),
                                              cstI64(rewriter, loc, 13));
    Value rs2 = rewriter.create<arith::OrIOp>(loc, adaptor.getDaAddr(), dw);
    llvm::StringRef mnemonic =
        std::is_same_v<Op, Int2FpTensorOp> ? "INT2FP_TENSOR" : "INT2FP_CHANNEL";
    rewriter.replaceOpWithNewOp<CustomIntrOp>(
        op, rs1, rs2,
        rewriter.getI32IntegerAttr(
            buckyball_target::getBuckyballFunct7(mnemonic)));
    return success();
  }
};

} // namespace

namespace mlir::buddy::buckyball {
void populateInt2FpBallLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, bool stable,
    int64_t, bool) {
  (void)stable;
  patterns.add<Int2FpLowering<Int2FpTensorOp>, Int2FpLowering<Int2FpChannelOp>>(
      converter);
}

void configureInt2FpBallLegalizeForExportTarget(LLVMConversionTarget &target,
                                                bool stable) {
  (void)stable;
  target.addIllegalOp<Int2FpTensorOp, Int2FpChannelOp, BankInt2FpTensorOp,
                      BankInt2FpChannelOp>();
}
} // namespace mlir::buddy::buckyball
