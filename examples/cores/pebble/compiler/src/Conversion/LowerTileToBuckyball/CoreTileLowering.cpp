//===- CoreTileLowering.cpp - Pebble Tile to Buckyball lowering ----------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Buckyball/BuckyballDialect.h"
#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Tile/TileDialect.h"
#include "Tile/TileOps.h"

using namespace mlir;
using namespace ::buddy::buckyball;
namespace tile = ::buddy::tile;

namespace mlir::buddy {
void populateTransposeBallTileLoweringPatterns(RewritePatternSet &patterns,
                                               int64_t bankWidthBytes,
                                               int64_t bankDepth,
                                               int64_t bankNum);
} // namespace mlir::buddy

namespace {

class QuantF32ToI8Lowering : public OpRewritePattern<tile::TileQuantF32ToI8Op> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileQuantF32ToI8Op op,
                                PatternRewriter &rewriter) const override {
    auto input = cast<MemRefType>(op.getInput().getType());
    auto output = cast<MemRefType>(op.getOutput().getType());
    if (!input.hasStaticShape() || !output.hasStaticShape() ||
        input.getRank() != output.getRank() ||
        (input.getRank() != 2 && input.getRank() != 4))
      return op.emitError("input quantization requires static shapes");
    ArrayRef<int64_t> inShape = input.getShape();
    ArrayRef<int64_t> outShape = output.getShape();
    bool nchwToNhwc = op.getNchwToNhwcAttr().getValue();
    if ((!nchwToNhwc && inShape != outShape) ||
        (nchwToNhwc && input.getRank() != 4) ||
        (nchwToNhwc &&
         (inShape[0] != outShape[0] || inShape[1] != outShape[3] ||
          inShape[2] != outShape[1] || inShape[3] != outShape[2])))
      return op.emitError("input quantization layout shape mismatch");
    rewriter.create<QuantizeTensorF32ToI8Op>(op.getLoc(), op.getInput(),
                                             op.getOutput(), op.getScaleAttr(),
                                             op.getNchwToNhwcAttr());
    rewriter.eraseOp(op);
    return success();
  }
};

class CpuMatmulLowering : public OpRewritePattern<tile::TileMatMulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileMatMulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<linalg::MatmulOp>(
        op.getLoc(), ValueRange{op.getAMemArray(), op.getBMemArray()},
        ValueRange{op.getCMemArray()});
    rewriter.eraseOp(op);
    return success();
  }
};

class MegaKernelLowering : public OpRewritePattern<tile::TileMegaKernelOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tile::TileMegaKernelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBody().empty())
      return op.emitError("MegaKernel region must contain one block");
    SmallVector<Operation *> stages;
    for (Operation &child : op.getBody().front().without_terminator())
      stages.push_back(&child);
    if (stages.empty())
      return op.emitError("MegaKernel region must contain at least one stage");
    if (!isa<tile::TileMegaYieldOp>(op.getBody().front().getTerminator()))
      return op.emitError("MegaKernel region must end with tile.mega_yield");

    rewriter.setInsertionPoint(op);
    auto kernel = rewriter.create<MegaKernelOp>(op.getLoc(), op.getInput(),
                                                op.getOutput());
    kernel.getBody().emplaceBlock();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&kernel.getBody().front());

    for (auto [index, stage] : llvm::enumerate(stages)) {
      bool last = index + 1 == stages.size();
      if (auto matmul = dyn_cast<tile::TileMegaMatmulOp>(stage)) {
        if (last ? matmul.getOutput() != op.getOutput()
                 : matmul.getOutput() == op.getOutput())
          return matmul.emitError(
              "MegaKernel MatMul output boundary is invalid");
        auto input = cast<MemRefType>(matmul.getInput().getType());
        auto weight = cast<MemRefType>(matmul.getWeight().getType());
        auto bias = cast<MemRefType>(matmul.getBias().getType());
        auto scale = cast<MemRefType>(matmul.getScale().getType());
        auto lut = cast<MemRefType>(matmul.getLut().getType());
        auto output = cast<MemRefType>(matmul.getOutput().getType());
        if (!input.hasStaticShape() || !weight.hasStaticShape() ||
            !bias.hasStaticShape() || !scale.hasStaticShape() ||
            !lut.hasStaticShape() || !output.hasStaticShape())
          return matmul.emitError("MegaKernel MatMul requires static shapes");
        int64_t m = input.getShape()[0];
        int64_t k = input.getShape()[1];
        int64_t n = weight.getShape()[1];
        if (m <= 0 || k <= 0 || n <= 0 || weight.getShape()[0] != k ||
            output.getShape()[0] != m || output.getShape()[1] != n ||
            bias.getShape()[0] != n || scale.getShape()[0] != n ||
            lut.getShape()[0] != (matmul.getActivation() == 2 ? 256 : 1) ||
            matmul.getActivation() < 0 || matmul.getActivation() > 2 ||
            matmul.getOutputScale().convertToDouble() <= 0.0 ||
            (last ? !output.getElementType().isF32()
                  : !output.getElementType().isInteger(8)))
          return matmul.emitError(
              "MegaKernel MatMul shape or output type mismatch");
        rewriter.create<MegaMatmulOp>(
            matmul.getLoc(), matmul.getInput(), matmul.getWeight(),
            matmul.getBias(), matmul.getScale(), matmul.getLut(),
            matmul.getOutput(), matmul.getActivationAttr(),
            matmul.getOutputScaleAttr());
        continue;
      }

      if (auto globalAvg = dyn_cast<tile::TileMegaGlobalAvgPoolOp>(stage)) {
        auto input = cast<MemRefType>(globalAvg.getInput().getType());
        auto output = cast<MemRefType>(globalAvg.getOutput().getType());
        if ((last ? globalAvg.getOutput() != op.getOutput()
                  : globalAvg.getOutput() == op.getOutput()) ||
            !input.hasStaticShape() || !output.hasStaticShape() ||
            input.getShape()[0] != output.getShape()[0] ||
            input.getShape()[3] != output.getShape()[3] ||
            output.getShape()[1] != 1 || output.getShape()[2] != 1 ||
            globalAvg.getInputScale().convertToDouble() <= 0.0 ||
            globalAvg.getOutputScale().convertToDouble() <= 0.0)
          return globalAvg.emitError("Mega global-average contract is invalid");
        rewriter.create<MegaGlobalAvgPoolOp>(
            globalAvg.getLoc(), globalAvg.getInput(), globalAvg.getOutput(),
            globalAvg.getInputScaleAttr(), globalAvg.getOutputScaleAttr());
        continue;
      }

      if (auto maxPool = dyn_cast<tile::TileMegaMaxPool2dOp>(stage)) {
        auto input = cast<MemRefType>(maxPool.getInput().getType());
        auto output = cast<MemRefType>(maxPool.getOutput().getType());
        if (!input.hasStaticShape() || !output.hasStaticShape() ||
            input.getRank() != 4 || output.getRank() != 4 ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) ||
            maxPool.getFinalOutput() != last || maxPool.getKernel() <= 0 ||
            maxPool.getStride() <= 0 || maxPool.getPadding() < 0)
          return maxPool.emitError("Mega MaxPool2D contract is invalid");
        rewriter.create<MegaMaxPool2dOp>(
            maxPool.getLoc(), maxPool.getInput(), maxPool.getOutput(),
            maxPool.getKernelAttr(), maxPool.getStrideAttr(),
            maxPool.getPaddingAttr(), maxPool.getFinalOutputAttr());
        continue;
      }

      auto int8Mul = dyn_cast<tile::TileMegaInt8MulOp>(stage);
      auto int8Add = dyn_cast<tile::TileMegaInt8AddOp>(stage);
      if (int8Mul || int8Add) {
        Value lhs = int8Mul ? int8Mul.getLhs() : int8Add.getLhs();
        Value rhs = int8Mul ? int8Mul.getRhs() : int8Add.getRhs();
        Value outputValue = int8Mul ? int8Mul.getOutput() : int8Add.getOutput();
        auto lhsType = cast<MemRefType>(lhs.getType());
        auto rhsType = cast<MemRefType>(rhs.getType());
        auto outputType = cast<MemRefType>(outputValue.getType());
        double lhsScale = int8Mul ? int8Mul.getLhsScale().convertToDouble()
                                  : int8Add.getLhsScale().convertToDouble();
        double rhsScale = int8Mul ? int8Mul.getRhsScale().convertToDouble()
                                  : int8Add.getRhsScale().convertToDouble();
        double outputScale = int8Mul
                                 ? int8Mul.getOutputScale().convertToDouble()
                                 : int8Add.getOutputScale().convertToDouble();
        int64_t activation =
            int8Mul ? int8Mul.getActivation() : int8Add.getActivation();
        if ((last ? outputValue != op.getOutput()
                  : outputValue == op.getOutput()) ||
            !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
            !outputType.hasStaticShape() || lhsScale <= 0.0 ||
            rhsScale <= 0.0 || outputScale <= 0.0 || activation < 0 ||
            activation > 1)
          return stage->emitError("Mega INT8 elementwise contract is invalid");
        for (int64_t dimension = 0; dimension < 4; ++dimension) {
          int64_t out = outputType.getShape()[dimension];
          if ((lhsType.getShape()[dimension] != 1 &&
               lhsType.getShape()[dimension] != out) ||
              (rhsType.getShape()[dimension] != 1 &&
               rhsType.getShape()[dimension] != out))
            return stage->emitError(
                "Mega INT8 elementwise shapes do not broadcast");
        }
        if (int8Add && lhsType != rhsType)
          return int8Add.emitError(
              "Mega residual add requires equal input types");
        if (int8Mul)
          rewriter.create<MegaInt8MulOp>(
              int8Mul.getLoc(), lhs, rhs, outputValue,
              int8Mul.getLhsScaleAttr(), int8Mul.getRhsScaleAttr(),
              int8Mul.getOutputScaleAttr(), int8Mul.getActivationAttr());
        else
          rewriter.create<MegaInt8AddOp>(
              int8Add.getLoc(), lhs, rhs, outputValue,
              int8Add.getLhsScaleAttr(), int8Add.getRhsScaleAttr(),
              int8Add.getOutputScaleAttr(), int8Add.getActivationAttr());
        continue;
      }

      auto conv = dyn_cast<tile::TileMegaConv2dOp>(stage);
      auto depthwise = dyn_cast<tile::TileMegaConv2dDepthwiseOp>(stage);
      if (!conv && !depthwise)
        return stage->emitError("unsupported operation in MegaKernel region");

      Value inputValue = conv ? conv.getInput() : depthwise.getInput();
      Value filterValue = conv ? conv.getFilter() : depthwise.getFilter();
      Value biasValue = conv ? conv.getBias() : depthwise.getBias();
      Value scaleValue = conv ? conv.getScale() : depthwise.getScale();
      Value lutValue = conv ? conv.getLut() : depthwise.getLut();
      Value outputValue = conv ? conv.getOutput() : depthwise.getOutput();
      if (last ? outputValue != op.getOutput() : outputValue == op.getOutput())
        return stage->emitError(
            "MegaKernel convolution output boundary is invalid");
      auto input = cast<MemRefType>(inputValue.getType());
      auto filter = cast<MemRefType>(filterValue.getType());
      auto bias = cast<MemRefType>(biasValue.getType());
      auto scale = cast<MemRefType>(scaleValue.getType());
      auto lut = cast<MemRefType>(lutValue.getType());
      auto output = cast<MemRefType>(outputValue.getType());
      if (!input.hasStaticShape() || !filter.hasStaticShape() ||
          !bias.hasStaticShape() || !scale.hasStaticShape() ||
          !output.hasStaticShape())
        return stage->emitError(
            "MegaKernel convolution requires static shapes");

      int64_t stride = conv ? conv.getStride() : depthwise.getStride();
      int64_t padLow = conv ? conv.getPadLow() : depthwise.getPadLow();
      int64_t padHigh = conv ? conv.getPadHigh() : depthwise.getPadHigh();
      int64_t activation =
          conv ? conv.getActivation() : depthwise.getActivation();
      auto is = input.getShape();
      auto fs = filter.getShape();
      auto os = output.getShape();
      int64_t outChannels = last ? os[1] : os[3];
      int64_t outHeight = last ? os[2] : os[1];
      int64_t outWidth = last ? os[3] : os[2];
      int64_t expectedChannels = depthwise ? is[3] : fs[3];
      if (is[0] <= 0 || is[1] <= 0 || is[2] <= 0 || is[3] <= 0 || fs[0] <= 0 ||
          fs[1] <= 0 || fs[2] != is[3] || fs[3] <= 0 ||
          (depthwise && fs[3] != 1) || os[0] != is[0] ||
          outChannels != expectedChannels ||
          bias.getShape()[0] != expectedChannels ||
          scale.getShape()[0] != expectedChannels || stride <= 0 ||
          lut.getRank() != 1 || !lut.getElementType().isInteger(8) ||
          lut.getShape()[0] != (activation == 2 ? 256 : 1) || activation < 0 ||
          activation > 2 ||
          (conv ? conv.getOutputScale() : depthwise.getOutputScale())
                  .convertToDouble() <= 0.0 ||
          padLow < 0 || padHigh < 0 || is[1] + padLow + padHigh < fs[0] ||
          is[2] + padLow + padHigh < fs[1] ||
          outHeight != (is[1] + padLow + padHigh - fs[0]) / stride + 1 ||
          outWidth != (is[2] + padLow + padHigh - fs[1]) / stride + 1 ||
          (last ? !output.getElementType().isF32()
                : !output.getElementType().isInteger(8)))
        return stage->emitError(
            "MegaKernel convolution shape or output type mismatch");

      if (depthwise)
        rewriter.create<MegaConv2dDepthwiseOp>(
            depthwise.getLoc(), inputValue, filterValue, biasValue, scaleValue,
            lutValue, outputValue, depthwise.getStrideAttr(),
            depthwise.getPadLowAttr(), depthwise.getPadHighAttr(),
            depthwise.getActivationAttr(), depthwise.getOutputScaleAttr());
      else
        rewriter.create<MegaConv2dOp>(
            conv.getLoc(), inputValue, filterValue, biasValue, scaleValue,
            lutValue, outputValue, conv.getStrideAttr(), conv.getPadLowAttr(),
            conv.getPadHighAttr(), conv.getActivationAttr(),
            conv.getOutputScaleAttr());
    }
    rewriter.create<MegaYieldOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

class LowerTileToBuckyballPass
    : public PassWrapper<LowerTileToBuckyballPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTileToBuckyballPass)

  StringRef getArgument() const final { return "convert-tile-to-buckyball"; }
  StringRef getDescription() const final {
    return "Convert explicit Pebble Tile kernels to Buckyball";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tile::TileDialect, BuckyballDialect, func::FuncDialect,
                    memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect,
                    linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    const auto &targetConfig = buckyball_target::getBuckyballTarget();
    ConversionTarget target(getContext());
    target.addLegalDialect<BuckyballDialect, memref::MemRefDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           func::FuncDialect, linalg::LinalgDialect>();
    target.addIllegalDialect<tile::TileDialect>();

    RewritePatternSet patterns(&getContext());
    mlir::buddy::populateTransposeBallTileLoweringPatterns(
        patterns, targetConfig.bankWidthBits / 8, targetConfig.bankDepth,
        targetConfig.bankNum);
    patterns.add<QuantF32ToI8Lowering, CpuMatmulLowering, MegaKernelLowering>(
        &getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::buddy {
void registerLowerTileToBuckyballPass() {
  PassRegistration<LowerTileToBuckyballPass>();
}
} // namespace mlir::buddy
