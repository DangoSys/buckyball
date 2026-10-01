#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
using namespace mlir;
using namespace buddy::buckyball;
namespace {
struct ConvLowering : OpRewritePattern<GemminiMemConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GemminiMemConvOp op,
                                PatternRewriter &b) const override {
    buckyball_target::requireBuckyballBall("GemminiBall");
    auto input = cast<MemRefType>(op.getInput().getType()),
         weight = cast<MemRefType>(op.getWeight().getType()),
         bias = cast<MemRefType>(op.getBias().getType()),
         output = cast<MemRefType>(op.getOutput().getType());
    if (input.getRank() != 4 || weight.getRank() != 2 || bias.getRank() != 1 ||
        output.getRank() != 4 || !input.getElementType().isF32() ||
        !weight.getElementType().isInteger(8) ||
        !bias.getElementType().isF32() || !output.getElementType().isF32())
      return op.emitError("Gemmini model ABI expects F32 NCHW, I8 matrix "
                          "weights and F32 bias/output");
    auto dynamic = ShapedType::kDynamic;
    auto matrix = [&](int rank, Type element) {
      return MemRefType::get(
          SmallVector<int64_t>(rank, dynamic), element,
          StridedLayoutAttr::get(b.getContext(), dynamic,
                                 SmallVector<int64_t>(rank, dynamic)));
    };
    auto a = matrix(4, b.getF32Type()), w = matrix(2, b.getI8Type()),
         d = matrix(1, b.getF32Type());
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op.getLoc();
    if (!module.lookupSymbol<func::FuncOp>("gemmini_conv")) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(module.getBody());
      auto fn = b.create<func::FuncOp>(
          loc, "gemmini_conv",
          b.getFunctionType({a, w, d, a, b.getI64Type(), b.getI64Type(),
                             b.getI64Type(), b.getI64Type(), b.getF32Type(),
                             b.getF32Type()},
                            {}));
      fn.setPrivate();
      fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
    }
    b.create<func::CallOp>(
        loc, "gemmini_conv", TypeRange{},
        ValueRange{b.create<memref::CastOp>(loc, a, op.getInput()),
                   b.create<memref::CastOp>(loc, w, op.getWeight()),
                   b.create<memref::CastOp>(loc, d, op.getBias()),
                   b.create<memref::CastOp>(loc, a, op.getOutput()),
                   b.create<arith::ConstantIntOp>(loc, op.getKh(), 64),
                   b.create<arith::ConstantIntOp>(loc, op.getKw(), 64),
                   b.create<arith::ConstantIntOp>(loc, op.getStride(), 64),
                   b.create<arith::ConstantIntOp>(loc, op.getPadding(), 64),
                   b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               op.getInputScaleAttr()),
                   b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               op.getWeightScaleAttr())});
    b.eraseOp(op);
    return success();
  }
};
} // namespace
namespace mlir::buddy {
void populateGemminiBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ConvLowering>(patterns.getContext());
}
} // namespace mlir::buddy
