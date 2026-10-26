//===- MegaKernelPatterns.cpp - marked MegaKernel regions ----------------===//
//
// Collects the generics the frontend marked as one cross-layer resident
// quantized MegaKernel into buckyball.mega_kernel.
//
//===----------------------------------------------------------------------===//

#include <cmath>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Conversion/LowerLinalgToBuckyball/LowerLinalgToBuckyball.h"

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {
bool positive(FloatAttr attr) {
  return attr && std::isfinite(attr.getValueAsDouble()) &&
         attr.getValueAsDouble() > 0.0;
}

bool staticInt8(Value value, int64_t rank) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() == rank &&
         type.getElementType().isInteger(8);
}

// The stages of one MegaKernel are consecutive generics sharing
// `mega_kernel_id`; each stage names its kind with a `buckyball.mega_*` marker.
class MegaKernelLowering : public OpRewritePattern<linalg::GenericOp> {
public:
  MegaKernelLowering(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context, 2) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto first = op->getAttrOfType<IntegerAttr>("mega_kernel_stage");
    if (!op->hasAttr("buckyball.mega_kernel") || !first || first.getInt() != 0)
      return failure();
    auto size = op->getAttrOfType<IntegerAttr>("mega_kernel_size");
    auto kernelId = op->getAttrOfType<StringAttr>("mega_kernel_id");
    if (!size || size.getInt() <= 0 || !kernelId || kernelId.getValue().empty())
      return op.emitError("MegaKernel requires a positive size and an ID");

    SmallVector<linalg::GenericOp> stages(size.getInt());
    for (Operation &candidate : *op->getBlock()) {
      auto generic = dyn_cast<linalg::GenericOp>(candidate);
      if (!generic ||
          generic->getAttrOfType<StringAttr>("mega_kernel_id") != kernelId)
        continue;
      auto index = generic->getAttrOfType<IntegerAttr>("mega_kernel_stage");
      auto stageSize = generic->getAttrOfType<IntegerAttr>("mega_kernel_size");
      if (!generic->hasAttr("buckyball.mega_kernel") || !index || !stageSize ||
          stageSize.getInt() != size.getInt() || index.getInt() < 0 ||
          index.getInt() >= size.getInt())
        return generic.emitError("MegaKernel stage metadata is malformed");
      if (stages[index.getInt()])
        return generic.emitError("MegaKernel stage index is duplicated");
      stages[index.getInt()] = generic;
    }
    if (llvm::any_of(stages, [](linalg::GenericOp stage) { return !stage; }))
      return op.emitError("MegaKernel stage sequence is incomplete");

    Value kernelOutput = stages.back().getOutputs()[0];
    rewriter.setInsertionPointAfter(stages.back());
    auto kernel = rewriter.create<MegaKernelOp>(
        op.getLoc(), stages.front().getInputs()[0], kernelOutput);
    kernel.getBody().emplaceBlock();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&kernel.getBody().front());
    for (auto [index, current] : llvm::enumerate(stages)) {
      bool last = index + 1 == stages.size();
      bool normal = current->hasAttr("buckyball.mega_conv2d");
      bool depthwise = current->hasAttr("buckyball.mega_conv2d_depthwise");
      bool matmul = current->hasAttr("buckyball.mega_matmul");
      bool maxPool = current->hasAttr("buckyball.mega_max_pool2d");
      bool globalAvg = current->hasAttr("buckyball.mega_global_avg_pool");
      bool channelSlice = current->hasAttr("buckyball.mega_channel_slice");
      bool channelConcat = current->hasAttr("buckyball.mega_channel_concat");
      bool resizeNearest = current->hasAttr("buckyball.mega_resize_nearest");
      bool int8Mul = current->hasAttr("buckyball.mega_int8_mul");
      bool int8Add = current->hasAttr("buckyball.mega_int8_add");
      if (int(normal) + int(depthwise) + int(matmul) + int(globalAvg) +
              int(maxPool) + int(channelSlice) + int(channelConcat) +
              int(resizeNearest) + int(int8Mul) + int(int8Add) !=
          1)
        return current.emitError("MegaKernel stage kind is not unique");
      Location loc = current.getLoc();
      ValueRange inputs = current.getInputs();
      if (current.getOutputs().size() != 1)
        return current.emitError("MegaKernel stage needs one output");
      Value output = current.getOutputs()[0];
      auto outputType = dyn_cast<MemRefType>(output.getType());
      if (!outputType || !outputType.hasStaticShape() ||
          (outputType.getRank() != 4 && !matmul))
        return current.emitError("MegaKernel stage needs a static output");
      ArrayRef<int64_t> out = outputType.getShape();
      // Only the last stage writes the kernel output.
      bool boundary = last ? output == kernelOutput : output != kernelOutput;

      if (channelSlice) {
        auto offset = current->getAttrOfType<IntegerAttr>("offset");
        if (inputs.size() != 1 || !staticInt8(inputs[0], 4) ||
            !staticInt8(output, 4) || !offset || offset.getInt() < 0 ||
            offset.getInt() % 16 != 0 || out[3] % 16 != 0)
          return current.emitError("Mega channel slice contract is invalid");
        auto in = cast<MemRefType>(inputs[0].getType()).getShape();
        if (in[0] != out[0] || in[1] != out[1] || in[2] != out[2] ||
            offset.getInt() + out[3] > in[3])
          return current.emitError("Mega channel slice contract is invalid");
        rewriter.create<MegaChannelSliceOp>(loc, inputs[0], output, offset);
        continue;
      }

      if (channelConcat) {
        auto segments = current->getAttrOfType<DenseI64ArrayAttr>("segments");
        if (inputs.empty() || !staticInt8(output, 4) || !segments ||
            segments.size() != inputs.size())
          return current.emitError("Mega channel concat has the wrong arity");
        int64_t channels = 0;
        for (auto [input, segment] : llvm::zip(inputs, segments.asArrayRef())) {
          if (!staticInt8(input, 4) || segment <= 0 || segment % 16 != 0)
            return current.emitError("Mega channel concat contract is invalid");
          auto in = cast<MemRefType>(input.getType()).getShape();
          if (in[0] != out[0] || in[1] != out[1] || in[2] != out[2] ||
              in[3] != segment)
            return current.emitError("Mega channel concat contract is invalid");
          channels += segment;
        }
        if (out[3] != channels)
          return current.emitError(
              "Mega channel concat output channels are invalid");
        rewriter.create<MegaChannelConcatOp>(loc, inputs, output, segments);
        continue;
      }

      if (resizeNearest) {
        auto scaleH = current->getAttrOfType<IntegerAttr>("scale_h");
        auto scaleW = current->getAttrOfType<IntegerAttr>("scale_w");
        if (inputs.size() != 1 || !staticInt8(inputs[0], 4) ||
            !staticInt8(output, 4) || !scaleH || !scaleW ||
            scaleH.getInt() <= 0 || scaleW.getInt() <= 0)
          return current.emitError("Mega resize-nearest contract is invalid");
        auto in = cast<MemRefType>(inputs[0].getType()).getShape();
        if (in[0] != out[0] || in[3] != out[3] ||
            out[1] != in[1] * scaleH.getInt() ||
            out[2] != in[2] * scaleW.getInt())
          return current.emitError("Mega resize-nearest contract is invalid");
        rewriter.create<MegaResizeNearestOp>(loc, inputs[0], output, scaleH,
                                             scaleW);
        continue;
      }

      if (maxPool) {
        auto kernelSize = current->getAttrOfType<IntegerAttr>("kernel");
        auto stride = current->getAttrOfType<IntegerAttr>("stride");
        auto padding = current->getAttrOfType<IntegerAttr>("padding");
        auto finalOutput = current->getAttrOfType<BoolAttr>("final_output");
        if (inputs.size() != 1 || !staticInt8(inputs[0], 4) ||
            !staticInt8(output, 4) || !kernelSize || !stride || !padding ||
            !finalOutput || kernelSize.getInt() <= 0 || stride.getInt() <= 0 ||
            padding.getInt() < 0 || (finalOutput.getValue() && !last))
          return current.emitError("Mega MaxPool2D contract is invalid");
        auto in = cast<MemRefType>(inputs[0].getType()).getShape();
        bool nchw = finalOutput.getValue();
        int64_t outH = nchw ? out[2] : out[1];
        int64_t outW = nchw ? out[3] : out[2];
        int64_t outC = nchw ? out[1] : out[3];
        int64_t window = 2 * padding.getInt() - kernelSize.getInt();
        if (out[0] != in[0] || outC != in[3] ||
            outH != (in[1] + window) / stride.getInt() + 1 ||
            outW != (in[2] + window) / stride.getInt() + 1)
          return current.emitError("Mega MaxPool2D shape is invalid");
        rewriter.create<MegaMaxPool2dOp>(loc, inputs[0], output, kernelSize,
                                         stride, padding, finalOutput);
        continue;
      }

      if (globalAvg) {
        auto inputScale = current->getAttrOfType<FloatAttr>("input_scale");
        auto outputScale = current->getAttrOfType<FloatAttr>("output_scale");
        if (!boundary || inputs.size() != 1 || !staticInt8(inputs[0], 4) ||
            !staticInt8(output, 4) || !positive(inputScale) ||
            !positive(outputScale))
          return current.emitError("Mega global-average contract is invalid");
        auto in = cast<MemRefType>(inputs[0].getType()).getShape();
        if (in[0] != out[0] || in[3] != out[3] || out[1] != 1 || out[2] != 1)
          return current.emitError("Mega global-average contract is invalid");
        rewriter.create<MegaGlobalAvgPoolOp>(loc, inputs[0], output, inputScale,
                                             outputScale);
        continue;
      }

      if (int8Mul || int8Add) {
        auto lhsScale = current->getAttrOfType<FloatAttr>("lhs_scale");
        auto rhsScale = current->getAttrOfType<FloatAttr>("rhs_scale");
        auto outputScale = current->getAttrOfType<FloatAttr>("output_scale");
        auto activation = current->getAttrOfType<IntegerAttr>("activation");
        if (!boundary || inputs.size() != 2 || !staticInt8(inputs[0], 4) ||
            !staticInt8(inputs[1], 4) || !staticInt8(output, 4) ||
            !positive(lhsScale) || !positive(rhsScale) ||
            !positive(outputScale) || !activation || activation.getInt() < 0 ||
            activation.getInt() > 1)
          return current.emitError("Mega INT8 elementwise contract is invalid");
        auto lhs = cast<MemRefType>(inputs[0].getType());
        auto rhs = cast<MemRefType>(inputs[1].getType());
        for (int64_t dimension = 0; dimension < 4; ++dimension)
          if ((lhs.getShape()[dimension] != 1 &&
               lhs.getShape()[dimension] != out[dimension]) ||
              (rhs.getShape()[dimension] != 1 &&
               rhs.getShape()[dimension] != out[dimension]))
            return current.emitError(
                "Mega INT8 elementwise shapes do not broadcast");
        if (int8Add && lhs != rhs)
          return current.emitError(
              "Mega residual add requires equal input types");
        if (int8Mul)
          rewriter.create<MegaInt8MulOp>(loc, inputs[0], inputs[1], output,
                                         lhsScale, rhsScale, outputScale,
                                         activation);
        else
          rewriter.create<MegaInt8AddOp>(loc, inputs[0], inputs[1], output,
                                         lhsScale, rhsScale, outputScale,
                                         activation);
        continue;
      }

      // Compute stages: input, weight, bias, scale and activation LUT.
      auto activation = current->getAttrOfType<IntegerAttr>("activation");
      auto finalOutput = current->getAttrOfType<BoolAttr>("final_output");
      auto outputScale = current->getAttrOfType<FloatAttr>("output_scale");
      if (!boundary || inputs.size() != 5 || !activation ||
          activation.getInt() < 0 || activation.getInt() > 2 || !finalOutput ||
          !positive(outputScale) || (finalOutput.getValue() && !last))
        return current.emitError("Mega compute stage contract is invalid");
      Value input = inputs[0], weight = inputs[1], bias = inputs[2],
            scale = inputs[3], lut = inputs[4];
      auto inputType = cast<MemRefType>(input.getType());
      auto weightType = cast<MemRefType>(weight.getType());
      auto biasType = cast<MemRefType>(bias.getType());
      auto scaleType = cast<MemRefType>(scale.getType());
      auto lutType = cast<MemRefType>(lut.getType());
      if (!inputType.hasStaticShape() || !weightType.hasStaticShape() ||
          !biasType.hasStaticShape() || !scaleType.hasStaticShape() ||
          !staticInt8(lut, 1))
        return current.emitError("Mega compute stage requires static shapes");
      // A 4096-entry LUT holds one table per lane of a 16-channel Conv whose
      // only consumer is the depthwise stage after it.
      int64_t lutSize = lutType.getShape()[0];
      if (lutSize == 4096) {
        int64_t consumers = 0;
        for (linalg::GenericOp candidate : stages)
          consumers += llvm::is_contained(candidate.getInputs(), output);
        if (!normal || index != 0 || activation.getInt() != 2 || out[3] != 16 ||
            stages.size() < 2 ||
            !stages[1]->hasAttr("buckyball.mega_conv2d_depthwise") ||
            stages[1].getInputs()[0] != output || consumers != 1)
          return current.emitError(
              "4096-entry lane LUT is only legal on stage 0 before one "
              "16-channel depthwise Conv");
      } else if (lutSize != (activation.getInt() == 2 ? 256 : 1)) {
        return current.emitError(
            "MegaKernel activation LUT has the wrong shape");
      }
      // Only FP32 marks the model-visible result; INT8 stays inside.
      Type element = outputType.getElementType();
      if (last ? !element.isF32() && !element.isInteger(8)
               : !element.isInteger(8))
        return current.emitError("MegaKernel stage output type is invalid");

      if (matmul) {
        int64_t m = inputType.getShape()[0], k = inputType.getShape()[1];
        int64_t n = weightType.getShape()[1];
        if (m <= 0 || k <= 0 || n <= 0 || weightType.getShape()[0] != k ||
            out[0] != m || out[1] != n || biasType.getShape()[0] != n ||
            scaleType.getShape()[0] != n || (last && !element.isF32()))
          return current.emitError("MegaKernel MatMul shape mismatch");
        rewriter.create<MegaMatmulOp>(loc, input, weight, bias, scale, lut,
                                      output, activation, outputScale);
        continue;
      }

      auto kernelSize = current->getAttrOfType<IntegerAttr>("kernel");
      auto stride = current->getAttrOfType<IntegerAttr>("stride");
      auto padLow = current->getAttrOfType<IntegerAttr>("pad_low");
      auto padHigh = current->getAttrOfType<IntegerAttr>("pad_high");
      if (!kernelSize || !stride || !padLow || !padHigh)
        return current.emitError(
            "MegaKernel convolution attributes are missing");
      int64_t k = kernelSize.getInt(), s = stride.getInt();
      int64_t pads = padLow.getInt() + padHigh.getInt();
      auto in = inputType.getShape();
      auto filter = weightType.getShape();
      auto outputNhwc = current->getAttrOfType<BoolAttr>("output_nhwc");
      bool nchw =
          last && element.isF32() && !(outputNhwc && outputNhwc.getValue());
      int64_t outC = nchw ? out[1] : out[3];
      int64_t outH = nchw ? out[2] : out[1];
      int64_t outW = nchw ? out[3] : out[2];
      int64_t channels = depthwise ? in[3] : outC;
      bool filterValid =
          depthwise ? filter == ArrayRef<int64_t>({k, k, in[3], 1})
                    : filter == ArrayRef<int64_t>({(outC + 15) / 16, in[3],
                                                   (k * k + 15) / 16 * 16, 16});
      if (in[0] <= 0 || in[1] <= 0 || in[2] <= 0 || in[3] <= 0 || k <= 0 ||
          k > 7 || s <= 0 || padLow.getInt() < 0 || padHigh.getInt() < 0 ||
          !filterValid || out[0] != in[0] || outC != channels ||
          biasType.getShape()[0] != channels ||
          scaleType.getShape()[0] != channels || in[1] + pads < k ||
          in[2] + pads < k || outH != (in[1] + pads - k) / s + 1 ||
          outW != (in[2] + pads - k) / s + 1)
        return current.emitError("MegaKernel convolution shape mismatch");
      Operation *lowered;
      if (depthwise)
        lowered = rewriter.create<MegaConv2dDepthwiseOp>(
            loc, input, weight, bias, scale, lut, output, kernelSize, stride,
            padLow, padHigh, activation, outputScale);
      else
        lowered = rewriter.create<MegaConv2dOp>(
            loc, input, weight, bias, scale, lut, output, kernelSize, stride,
            padLow, padHigh, activation, outputScale);
      if (outputNhwc)
        lowered->setAttr("output_nhwc", outputNhwc);
    }
    for (auto [source, target] :
         llvm::zip_equal(stages, kernel.getBody().front()))
      if (auto trace = source->getAttr("trace_node"))
        target.setAttr("trace_node", trace);
    rewriter.create<MegaYieldOp>(op.getLoc());
    for (int64_t index = stages.size(); index > 0; --index)
      rewriter.eraseOp(stages[index - 1]);
    return success();
  }
};
} // namespace

void mlir::buddy::populateMegaKernelLinalgLoweringPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MegaKernelLowering>(patterns.getContext());
}
