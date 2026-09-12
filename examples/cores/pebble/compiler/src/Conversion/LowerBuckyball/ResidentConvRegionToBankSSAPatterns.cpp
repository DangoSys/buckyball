//===- ResidentConvRegionToBankSSAPatterns.cpp ---------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Trace/TraceOps.h"
#include "Utils/BankUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <algorithm>
#include <cmath>
#include <functional>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;

class ResidentConvRegionPattern : public OpRewritePattern<MegaKernelOp> {
public:
  ResidentConvRegionPattern(MLIRContext *context, bool traceMegaStages,
                            int64_t traceMegaStageStart,
                            int64_t traceMegaStageLimit)
      : OpRewritePattern<MegaKernelOp>(context, 4),
        traceMegaStages(traceMegaStages),
        traceMegaStageStart(traceMegaStageStart),
        traceMegaStageLimit(traceMegaStageLimit) {}

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    if (body.without_terminator().empty() || isa<MegaMatmulOp>(body.front()))
      return failure();

    struct Stage {
      Operation *op;
      Value input;
      Value rhs;
      ValueRange inputs;
      Value output;
      Value weight;
      Value weightChannelStride;
      Value bias;
      Value scale;
      Value lut;
      int64_t inputHeight;
      int64_t inputWidth;
      int64_t inputChannels;
      int64_t outputHeight;
      int64_t outputWidth;
      int64_t outputChannels;
      int64_t kernel;
      int64_t stride;
      int64_t padding;
      int64_t activation;
      float lhsScale;
      float rhsScale;
      float outputScale;
      bool pool;
      bool add;
      bool multiply;
      bool average;
      bool depthwise;
      bool channelSlice;
      bool channelConcat;
      bool resizeNearest;
      int64_t channelOffset;
      ArrayRef<int64_t> channelSegments;
      int64_t scaleH;
      int64_t scaleW;
      bool finalOutput;
      int64_t lutEntries;
      SmallVector<float> lutOutputScales;
    };

    SmallVector<Stage, 0> stages;
    DenseMap<Value, int64_t> producer;
    for (Operation &operation : body.without_terminator()) {
      Stage stage{};
      stage.op = &operation;
      if (auto conv = dyn_cast<MegaConv2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(conv.getInput().getType());
        auto weight = dyn_cast<MemRefType>(conv.getWeight().getType());
        auto bias = dyn_cast<MemRefType>(conv.getBias().getType());
        auto scale = dyn_cast<MemRefType>(conv.getScale().getType());
        auto lut = dyn_cast<MemRefType>(conv.getLut().getType());
        auto output = dyn_cast<MemRefType>(conv.getOutput().getType());
        // A one-stage MegaKernel can still produce an INT8 value for the
        // next kernel.  Region/value identity alone does not mean this is
        // the FP32 boundary; the output element type is the contract.
        bool finalOutput = output && output.getElementType().isF32();
        if (!input || !weight || !bias || !scale || !lut || !output ||
            !input.hasStaticShape() || !weight.hasStaticShape() ||
            !bias.hasStaticShape() || !scale.hasStaticShape() ||
            !lut.hasStaticShape() || !output.hasStaticShape() ||
            input.getRank() != 4 || weight.getRank() != 4 ||
            lut.getRank() != 1 || !lut.getElementType().isInteger(8) ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !weight.getElementType().isInteger(8) ||
            !bias.getElementType().isInteger(32) ||
            !scale.getElementType().isF32() ||
            (finalOutput ? !output.getElementType().isF32()
                         : !output.getElementType().isInteger(8)) ||
            conv.getActivation() < 0 || conv.getActivation() > 2 ||
            conv.getStride() <= 0 || conv.getPadLow() < 0 ||
            conv.getPadLow() != conv.getPadHigh())
          return conv.emitError("unsupported Conv stage in resident region");
        auto in = input.getShape();
        auto weights = weight.getShape();
        auto out = output.getShape();
        int64_t kernelSize = conv.getKernel();
        int64_t outputChannels = bias.getShape()[0];
        int64_t outputHeight = finalOutput ? out[2] : out[1];
        int64_t outputWidth = finalOutput ? out[3] : out[2];
        int64_t paddedKernel =
            ((kernelSize * kernelSize + kTile - 1) / kTile) * kTile;
        if (kernelSize <= 0 || kernelSize > 7 ||
            weights != ArrayRef<int64_t>({(outputChannels + kTile - 1) / kTile,
                                          in[3], paddedKernel, kTile}) ||
            bias.getShape() != ArrayRef<int64_t>({outputChannels}) ||
            scale.getShape() != ArrayRef<int64_t>({outputChannels}) ||
            (conv.getActivation() == 2
                 ? (lut.getShape()[0] != 256 && lut.getShape()[0] != 4096)
                 : lut.getShape()[0] != 1) ||
            (lut.getShape()[0] == 4096 &&
             (conv.getActivation() != 2 || outputChannels != 16)) ||
            (in[1] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                outputHeight ||
            (in[2] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                outputWidth ||
            (finalOutput ? (out[1] != outputChannels)
                         : (out[3] != outputChannels)))
          return conv.emitError("Conv shape is inconsistent");
        stage.input = conv.getInput();
        stage.output = conv.getOutput();
        stage.weight = conv.getWeight();
        stage.bias = conv.getBias();
        stage.scale = conv.getScale();
        stage.lut = conv.getLut();
        stage.inputHeight = in[1];
        stage.inputWidth = in[2];
        stage.inputChannels = in[3];
        stage.outputHeight = outputHeight;
        stage.outputWidth = outputWidth;
        stage.outputChannels = outputChannels;
        stage.kernel = kernelSize;
        stage.stride = conv.getStride();
        stage.padding = conv.getPadLow();
        stage.activation = conv.getActivation();
        stage.outputScale = conv.getOutputScale().convertToFloat();
        stage.finalOutput = finalOutput;
        stage.lutEntries = lut.getShape()[0];
        if (auto scales = operation.getAttrOfType<DenseF32ArrayAttr>(
                "lane_output_scales"))
          stage.lutOutputScales.assign(scales.asArrayRef().begin(),
                                       scales.asArrayRef().end());
        if (!stage.lutOutputScales.empty() &&
            static_cast<int64_t>(stage.lutOutputScales.size()) !=
                outputChannels)
          return conv.emitError(
              "lane_output_scales count does not match output channels");
      } else if (auto conv = dyn_cast<MegaConv2dDepthwiseOp>(operation)) {
        auto input = dyn_cast<MemRefType>(conv.getInput().getType());
        auto weight = dyn_cast<MemRefType>(conv.getWeight().getType());
        auto bias = dyn_cast<MemRefType>(conv.getBias().getType());
        auto scale = dyn_cast<MemRefType>(conv.getScale().getType());
        auto lut = dyn_cast<MemRefType>(conv.getLut().getType());
        auto output = dyn_cast<MemRefType>(conv.getOutput().getType());
        bool finalOutput = output && output.getElementType().isF32();
        if (!input || !weight || !bias || !scale || !lut || !output ||
            !input.hasStaticShape() || !weight.hasStaticShape() ||
            !bias.hasStaticShape() || !scale.hasStaticShape() ||
            !lut.hasStaticShape() || !output.hasStaticShape() ||
            input.getRank() != 4 || weight.getRank() != 4 ||
            lut.getRank() != 1 || !lut.getElementType().isInteger(8) ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !weight.getElementType().isInteger(8) ||
            !bias.getElementType().isInteger(32) ||
            !scale.getElementType().isF32() ||
            (finalOutput ? !output.getElementType().isF32()
                         : !output.getElementType().isInteger(8)) ||
            conv.getActivation() < 0 || conv.getActivation() > 2 ||
            conv.getStride() <= 0 || conv.getPadLow() < 0 ||
            conv.getPadLow() != conv.getPadHigh())
          return conv.emitError(
              "unsupported Depthwise Conv stage in resident region");
        auto in = input.getShape();
        auto weights = weight.getShape();
        auto out = output.getShape();
        int64_t kernelSize = conv.getKernel();
        int64_t outputChannels = bias.getShape()[0];
        int64_t outputHeight = finalOutput ? out[2] : out[1];
        int64_t outputWidth = finalOutput ? out[3] : out[2];
        if (kernelSize <= 0 || kernelSize > 7 ||
            weights != ArrayRef<int64_t>({kernelSize, kernelSize, in[3], 1}) ||
            bias.getShape() != ArrayRef<int64_t>({outputChannels}) ||
            scale.getShape() != ArrayRef<int64_t>({outputChannels}) ||
            (conv.getActivation() == 2
                 ? (lut.getShape()[0] != 256 && lut.getShape()[0] != 4096)
                 : lut.getShape()[0] != 1) ||
            lut.getShape()[0] == 4096 || in[3] != outputChannels ||
            (in[1] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                outputHeight ||
            (in[2] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                outputWidth ||
            (finalOutput ? (out[1] != outputChannels)
                         : (out[3] != outputChannels)))
          return conv.emitError("Depthwise Conv shape is inconsistent");
        stage.input = conv.getInput();
        stage.output = conv.getOutput();
        stage.weight = conv.getWeight();
        stage.bias = conv.getBias();
        stage.scale = conv.getScale();
        stage.lut = conv.getLut();
        stage.inputHeight = in[1];
        stage.inputWidth = in[2];
        stage.inputChannels = in[3];
        stage.outputHeight = outputHeight;
        stage.outputWidth = outputWidth;
        stage.outputChannels = outputChannels;
        stage.kernel = kernelSize;
        stage.stride = conv.getStride();
        stage.padding = conv.getPadLow();
        stage.activation = conv.getActivation();
        stage.outputScale = conv.getOutputScale().convertToFloat();
        stage.depthwise = true;
        stage.finalOutput = finalOutput;
        stage.lutEntries = lut.getShape()[0];
        if (auto scales = operation.getAttrOfType<DenseF32ArrayAttr>(
                "lane_output_scales"))
          stage.lutOutputScales.assign(scales.asArrayRef().begin(),
                                       scales.asArrayRef().end());
        if (!stage.lutOutputScales.empty() &&
            static_cast<int64_t>(stage.lutOutputScales.size()) !=
                outputChannels)
          return conv.emitError(
              "lane_output_scales count does not match output channels");
      } else if (auto pool = dyn_cast<MegaMaxPool2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(pool.getInput().getType());
        auto output = dyn_cast<MemRefType>(pool.getOutput().getType());
        bool finalOutput = pool.getFinalOutput();
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || pool.getKernel() <= 0 ||
            pool.getKernel() > 8 || pool.getStride() <= 0 ||
            pool.getPadding() < 0)
          return pool.emitError("unsupported MaxPool stage in resident region");
        auto in = input.getShape();
        auto out = output.getShape();
        int64_t outputHeight = finalOutput ? out[2] : out[1];
        int64_t outputWidth = finalOutput ? out[3] : out[2];
        int64_t outputChannels = finalOutput ? out[1] : out[3];
        if (in[3] != outputChannels ||
            (in[1] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                outputHeight ||
            (in[2] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                outputWidth)
          return pool.emitError("MaxPool shape is inconsistent");
        stage.input = pool.getInput();
        stage.output = pool.getOutput();
        stage.inputHeight = in[1];
        stage.inputWidth = in[2];
        stage.inputChannels = in[3];
        stage.outputHeight = outputHeight;
        stage.outputWidth = outputWidth;
        stage.outputChannels = outputChannels;
        stage.kernel = pool.getKernel();
        stage.stride = pool.getStride();
        stage.padding = pool.getPadding();
        stage.pool = true;
        stage.finalOutput = finalOutput;
      } else if (auto add = dyn_cast<MegaInt8AddOp>(operation)) {
        auto lhs = dyn_cast<MemRefType>(add.getLhs().getType());
        auto rhs = dyn_cast<MemRefType>(add.getRhs().getType());
        auto output = dyn_cast<MemRefType>(add.getOutput().getType());
        float lhsScale = add.getLhsScale().convertToFloat();
        float rhsScale = add.getRhsScale().convertToFloat();
        float outputScale = add.getOutputScale().convertToFloat();
        if (!lhs || !rhs || !output || !lhs.hasStaticShape() ||
            !rhs.hasStaticShape() || !output.hasStaticShape() ||
            lhs.getRank() != 4 || rhs != lhs || output != lhs ||
            lhs.getShape()[0] != 1 || !lhs.getElementType().isInteger(8) ||
            add.getActivation() < 0 || add.getActivation() > 1 ||
            !std::isfinite(lhsScale) || !std::isfinite(rhsScale) ||
            !std::isfinite(outputScale) || lhsScale <= 0.0f ||
            rhsScale <= 0.0f || outputScale <= 0.0f)
          return add.emitError("unsupported INT8 Add stage in resident region");
        auto shape = lhs.getShape();
        stage.input = add.getLhs();
        stage.rhs = add.getRhs();
        stage.output = add.getOutput();
        stage.inputHeight = stage.outputHeight = shape[1];
        stage.inputWidth = stage.outputWidth = shape[2];
        stage.inputChannels = stage.outputChannels = shape[3];
        stage.kernel = stage.stride = 1;
        stage.activation = add.getActivation();
        stage.lhsScale = lhsScale;
        stage.rhsScale = rhsScale;
        stage.outputScale = outputScale;
        stage.add = true;
      } else if (auto multiply = dyn_cast<MegaInt8MulOp>(operation)) {
        auto gate = dyn_cast<MemRefType>(multiply.getLhs().getType());
        auto input = dyn_cast<MemRefType>(multiply.getRhs().getType());
        auto output = dyn_cast<MemRefType>(multiply.getOutput().getType());
        float gateScale = multiply.getLhsScale().convertToFloat();
        float inputScale = multiply.getRhsScale().convertToFloat();
        float outputScale = multiply.getOutputScale().convertToFloat();
        if (!gate || !input || !output || !gate.hasStaticShape() ||
            !input.hasStaticShape() || !output.hasStaticShape() ||
            gate.getRank() != 4 || input.getRank() != 4 || output != input ||
            gate.getShape()[0] != 1 || gate.getShape()[1] != 1 ||
            gate.getShape()[2] != 1 || input.getShape()[0] != 1 ||
            gate.getShape()[3] != input.getShape()[3] ||
            input.getShape()[3] <= 0 ||
            input.getShape()[3] >
                buckyball_target::getBuckyballTarget().bankDepth * kTile ||
            !gate.getElementType().isInteger(8) ||
            !input.getElementType().isInteger(8) ||
            multiply.getActivation() != 0 || !std::isfinite(gateScale) ||
            !std::isfinite(inputScale) || !std::isfinite(outputScale) ||
            gateScale <= 0.0f || inputScale <= 0.0f || outputScale <= 0.0f)
          return multiply.emitError(
              "INT8 Mul requires [1,1,1,C] gate and [1,H,W,C] input, "
              "1 <= C <= bankDepth*16, and activation=0");
        auto shape = input.getShape();
        stage.input = multiply.getRhs();
        stage.rhs = multiply.getLhs();
        stage.output = multiply.getOutput();
        stage.inputHeight = stage.outputHeight = shape[1];
        stage.inputWidth = stage.outputWidth = shape[2];
        stage.inputChannels = stage.outputChannels = shape[3];
        stage.kernel = stage.stride = 1;
        stage.lhsScale = gateScale;
        stage.rhsScale = inputScale;
        stage.outputScale = outputScale;
        stage.multiply = true;
      } else if (auto average = dyn_cast<MegaGlobalAvgPoolOp>(operation)) {
        auto input = dyn_cast<MemRefType>(average.getInput().getType());
        auto output = dyn_cast<MemRefType>(average.getOutput().getType());
        float inputScale = average.getInputScale().convertToFloat();
        float outputScale = average.getOutputScale().convertToFloat();
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            input.getShape()[3] <= 0 ||
            input.getShape()[3] >
                buckyball_target::getBuckyballTarget().bankDepth * kTile ||
            output.getShape() !=
                ArrayRef<int64_t>({1, 1, 1, input.getShape()[3]}) ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) ||
            !std::isfinite(inputScale) || !std::isfinite(outputScale) ||
            inputScale <= 0.0f || outputScale <= 0.0f)
          return average.emitError(
              "unsupported GlobalAvgPool stage in resident region");
        auto shape = input.getShape();
        stage.input = average.getInput();
        stage.output = average.getOutput();
        stage.inputHeight = shape[1];
        stage.inputWidth = shape[2];
        stage.inputChannels = shape[3];
        stage.outputHeight = stage.outputWidth = 1;
        stage.outputChannels = shape[3];
        stage.kernel = stage.stride = 1;
        stage.lhsScale = inputScale;
        stage.outputScale = outputScale;
        stage.average = true;
      } else if (auto slice = dyn_cast<MegaChannelSliceOp>(operation)) {
        auto input = dyn_cast<MemRefType>(slice.getInput().getType());
        auto output = dyn_cast<MemRefType>(slice.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || slice.getOffset() < 0 ||
            slice.getOffset() % kTile != 0 ||
            output.getShape()[3] % kTile != 0 ||
            input.getShape()[1] != output.getShape()[1] ||
            input.getShape()[2] != output.getShape()[2] ||
            slice.getOffset() + output.getShape()[3] > input.getShape()[3])
          return slice.emitError(
              "unsupported channel slice in resident region");
        stage.input = slice.getInput();
        stage.output = slice.getOutput();
        stage.inputHeight = stage.outputHeight = input.getShape()[1];
        stage.inputWidth = stage.outputWidth = input.getShape()[2];
        stage.inputChannels = input.getShape()[3];
        stage.outputChannels = output.getShape()[3];
        stage.kernel = stage.stride = 1;
        stage.channelSlice = true;
        stage.channelOffset = slice.getOffset();
      } else if (auto concat = dyn_cast<MegaChannelConcatOp>(operation)) {
        auto output = dyn_cast<MemRefType>(concat.getOutput().getType());
        ArrayRef<int64_t> segments = concat.getSegments();
        if (!output || !output.hasStaticShape() || output.getRank() != 4 ||
            output.getShape()[0] != 1 ||
            !output.getElementType().isInteger(8) ||
            concat.getInputs().empty() ||
            segments.size() != concat.getInputs().size())
          return concat.emitError(
              "unsupported channel concat in resident region");
        int64_t channels = 0;
        for (auto [inputValue, segment] :
             llvm::zip(concat.getInputs(), segments)) {
          auto input = dyn_cast<MemRefType>(inputValue.getType());
          if (!input || !input.hasStaticShape() || input.getRank() != 4 ||
              input.getShape()[0] != 1 ||
              !input.getElementType().isInteger(8) || segment <= 0 ||
              segment % kTile != 0 ||
              input.getShape()[1] != output.getShape()[1] ||
              input.getShape()[2] != output.getShape()[2] ||
              input.getShape()[3] != segment)
            return concat.emitError("channel concat shape is inconsistent");
          channels += segment;
        }
        if (channels != output.getShape()[3])
          return concat.emitError("channel concat channels are inconsistent");
        stage.inputs = concat.getInputs();
        stage.output = concat.getOutput();
        stage.outputHeight = output.getShape()[1];
        stage.outputWidth = output.getShape()[2];
        stage.outputChannels = output.getShape()[3];
        stage.kernel = stage.stride = 1;
        stage.channelConcat = true;
        stage.channelSegments = segments;
      } else if (auto resize = dyn_cast<MegaResizeNearestOp>(operation)) {
        auto input = dyn_cast<MemRefType>(resize.getInput().getType());
        auto output = dyn_cast<MemRefType>(resize.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || resize.getScaleH() <= 0 ||
            resize.getScaleW() <= 0 ||
            input.getShape()[3] != output.getShape()[3] ||
            output.getShape()[1] != input.getShape()[1] * resize.getScaleH() ||
            output.getShape()[2] != input.getShape()[2] * resize.getScaleW())
          return resize.emitError(
              "unsupported resize-nearest in resident region");
        stage.input = resize.getInput();
        stage.output = resize.getOutput();
        stage.inputHeight = input.getShape()[1];
        stage.inputWidth = input.getShape()[2];
        stage.inputChannels = input.getShape()[3];
        stage.outputHeight = output.getShape()[1];
        stage.outputWidth = output.getShape()[2];
        stage.outputChannels = output.getShape()[3];
        stage.kernel = stage.stride = 1;
        stage.resizeNearest = true;
        stage.scaleH = resize.getScaleH();
        stage.scaleW = resize.getScaleW();
      } else {
        return operation.emitError(
            "resident Conv region supports Conv, Depthwise Conv, MaxPool, "
            "INT8 Add, INT8 Mul, GlobalAvgPool, channel slice, channel concat, "
            "and nearest resize stages");
      }

      // A MegaKernel may have multiple external activation inputs, e.g. the
      // projection branch of a residual block.  Such values are captured by
      // the region directly; only values produced inside this region enter
      // `producer`.
      if (producer.contains(stage.output))
        return operation.emitError("region tensor has multiple producers");
      producer[stage.output] = stages.size();
      stages.push_back(stage);
    }
    if (stages.back().output != kernel.getOutput())
      return kernel.emitError("final stage must produce MegaKernel output");
    for (auto [index, stage] : llvm::enumerate(stages)) {
      if (stage.lutEntries != 4096)
        continue;
      int64_t consumers = 0;
      for (const Stage &candidate : stages)
        consumers +=
            candidate.input == stage.output || candidate.rhs == stage.output;
      if (index != 0 || stages.size() < 2 || !stages[1].depthwise ||
          stages[1].input != stage.output || consumers != 1)
        return stage.op->emitError(
            "4096-entry lane LUT is only legal on stage 0 before one "
            "16-channel depthwise Conv");
    }

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < 4 ||
        target.bankDepth % 4 != 0 || target.bankNum <= 0)
      return kernel.emitError(
          "resident Conv region requires 128-bit rows, bankDepth divisible "
          "by 4, and a positive bank count");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return kernel.emitError("resident Conv region requires SMatMul outBW=1");

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value minI8 = b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                              b.getI8IntegerAttr(-128));
    Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                b.getI32IntegerAttr(0));
    Value oneF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               b.getF32FloatAttr(1.0));
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> hostPacks;
    Value zeroPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
    Value minPack = b.create<memref::AllocOp>(
        loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
    hostPacks.push_back(zeroPack);
    hostPacks.push_back(minPack);
    b.create<linalg::FillOp>(loc, zeroI8, zeroPack);
    b.create<linalg::FillOp>(loc, minI8, minPack);

    DenseMap<Operation *, Value> packedBiases;
    DenseMap<Operation *, Value> packedScales;
    DenseMap<Operation *, Value> packedDepthwiseWeights;
    DenseMap<Operation *, Value> packedLuts;
    DenseMap<Operation *, Value> packedLutOutputScales;
    for (Stage &stage : stages) {
      if (stage.pool || stage.add || stage.multiply || stage.average ||
          stage.channelSlice || stage.channelConcat || stage.resizeNearest)
        continue;
      if (!stage.depthwise && stage.kernel == 1 && stage.stride == 1 &&
          stage.padding == 0 && stage.inputChannels % kTile == 0) {
        auto metadata =
            b.create<memref::ExtractStridedMetadataOp>(loc, stage.weight);
        Value channelStride = metadata.getStrides()[1];
        Value contiguousLanes = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, metadata.getStrides()[3], one);
        Value strideLowerBound = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, channelStride,
            b.create<arith::ConstantIndexOp>(loc, kTile));
        Value strideUpperBound = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sle, channelStride,
            b.create<arith::ConstantIndexOp>(loc, 1016));
        Value strideAligned = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq,
            b.create<arith::RemSIOp>(loc, channelStride,
                                     b.create<arith::ConstantIndexOp>(loc, 8)),
            zero);
        b.create<cf::AssertOp>(
            loc,
            b.create<arith::AndIOp>(
                loc,
                b.create<arith::AndIOp>(loc, contiguousLanes, strideLowerBound),
                b.create<arith::AndIOp>(loc, strideUpperBound, strideAligned)),
            "1x1 Conv weight requires contiguous lanes and a channel stride "
            "between 16 and 1016 bytes divisible by 8");
        stage.weightChannelStride =
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), channelStride);
      }
      int64_t outputPanels = (stage.outputChannels + kTile - 1) / kTile;
      Value biasPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getI32Type()));
      Value scalePack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getF32Type()));
      b.create<linalg::FillOp>(loc, zeroI32, biasPack);
      b.create<linalg::FillOp>(loc, oneF32, scalePack);
      hostPacks.append({biasPack, scalePack});
      packedBiases[stage.op] = biasPack;
      packedScales[stage.op] = scalePack;
      if (stage.depthwise) {
        int64_t paddedK =
            (stage.kernel * stage.kernel + kTile - 1) / kTile * kTile;
        Value weightPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({outputPanels, kTile, paddedK, kTile},
                                 b.getI8Type()));
        b.create<linalg::FillOp>(loc, zeroI8, weightPack);
        hostPacks.push_back(weightPack);
        packedDepthwiseWeights[stage.op] = weightPack;
      }
      if (stage.finalOutput && stage.activation == 2) {
        Value outputScalePack = b.create<memref::AllocOp>(
            loc, MemRefType::get({outputPanels, 4, 4}, b.getF32Type()));
        b.create<linalg::FillOp>(loc, oneF32, outputScalePack);
        hostPacks.push_back(outputScalePack);
        packedLutOutputScales[stage.op] = outputScalePack;
        for (int64_t panel = 0; panel < outputPanels; ++panel) {
          for (int64_t lane = 0; lane < kTile; ++lane) {
            int64_t channel = panel * kTile + lane;
            if (channel >= stage.outputChannels)
              continue;
            float outputScale = stage.lutOutputScales.empty()
                                    ? stage.outputScale
                                    : stage.lutOutputScales[channel];
            if (!std::isfinite(outputScale) || outputScale <= 0.0f)
              return stage.op->emitError(
                  "LUT output scales must be finite and positive");
            b.create<memref::StoreOp>(
                loc,
                b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                            b.getF32FloatAttr(outputScale)),
                outputScalePack,
                ValueRange{b.create<arith::ConstantIndexOp>(loc, panel),
                           b.create<arith::ConstantIndexOp>(loc, lane / 4),
                           b.create<arith::ConstantIndexOp>(loc, lane % 4)});
          }
        }
      }
      auto outputPanelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, outputPanels), one);
      b.setInsertionPointToStart(outputPanelLoop.getBody());
      Value outputPanel = outputPanelLoop.getInductionVar();
      Value channelBase = b.create<arith::MulIOp>(
          loc, outputPanel, b.create<arith::ConstantIndexOp>(loc, kTile));
      auto outputLaneLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(outputLaneLoop.getBody());
      Value outputLane = outputLaneLoop.getInductionVar();
      Value outputChannel =
          b.create<arith::AddIOp>(loc, channelBase, outputLane);
      auto validChannel = b.create<scf::IfOp>(
          loc,
          b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, outputChannel,
              b.create<arith::ConstantIndexOp>(loc, stage.outputChannels)),
          false);
      b.setInsertionPointToStart(&validChannel.getThenRegion().front());
      Value group = b.create<arith::DivUIOp>(
          loc, outputLane, b.create<arith::ConstantIndexOp>(loc, 4));
      Value groupLane = b.create<arith::RemUIOp>(
          loc, outputLane, b.create<arith::ConstantIndexOp>(loc, 4));
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, stage.bias, outputChannel),
          biasPack, ValueRange{outputPanel, group, groupLane});
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, stage.scale, outputChannel),
          scalePack, ValueRange{outputPanel, group, groupLane});
      if (stage.depthwise) {
        for (int64_t ky = 0; ky < stage.kernel; ++ky) {
          for (int64_t kx = 0; kx < stage.kernel; ++kx) {
            Value weight = b.create<memref::LoadOp>(
                loc, stage.weight,
                ValueRange{b.create<arith::ConstantIndexOp>(loc, ky),
                           b.create<arith::ConstantIndexOp>(loc, kx),
                           outputChannel, zero});
            b.create<memref::StoreOp>(
                loc, weight, packedDepthwiseWeights.lookup(stage.op),
                ValueRange{outputPanel, outputLane,
                           b.create<arith::ConstantIndexOp>(
                               loc, ky * stage.kernel + kx),
                           outputLane});
          }
        }
      }
      b.setInsertionPointAfter(validChannel);
      b.setInsertionPointAfter(outputPanelLoop);

      if (stage.activation == 2) {
        Value lutPack;
        if (stage.lutEntries == 4096)
          lutPack = b.create<memref::AllocOp>(
              loc,
              MemRefType::get({target.bankDepth, 4 * kTile}, b.getI8Type()));
        else
          lutPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({kTile, kTile}, b.getI8Type()));
        hostPacks.push_back(lutPack);
        packedLuts[stage.op] = lutPack;
        auto lutLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.lutEntries),
            one);
        b.setInsertionPointToStart(lutLoop.getBody());
        Value index = lutLoop.getInductionVar();
        Value lutValue = b.create<memref::LoadOp>(loc, stage.lut, index);
        if (stage.lutEntries == 4096) {
          Value group = b.create<arith::DivUIOp>(
              loc, index,
              b.create<arith::ConstantIndexOp>(loc, target.bankDepth * kTile));
          Value withinGroup = b.create<arith::RemUIOp>(
              loc, index,
              b.create<arith::ConstantIndexOp>(loc, target.bankDepth * kTile));
          b.create<memref::StoreOp>(
              loc, lutValue, lutPack,
              ValueRange{
                  b.create<arith::DivUIOp>(
                      loc, withinGroup,
                      b.create<arith::ConstantIndexOp>(loc, kTile)),
                  b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(
                          loc, group,
                          b.create<arith::ConstantIndexOp>(loc, kTile)),
                      b.create<arith::RemUIOp>(
                          loc, withinGroup,
                          b.create<arith::ConstantIndexOp>(loc, kTile)))});
        } else {
          b.create<memref::StoreOp>(
              loc, lutValue, lutPack,
              ValueRange{
                  b.create<arith::DivUIOp>(
                      loc, index, b.create<arith::ConstantIndexOp>(loc, kTile)),
                  b.create<arith::RemUIOp>(
                      loc, index,
                      b.create<arith::ConstantIndexOp>(loc, kTile))});
        }
        b.setInsertionPointAfter(lutLoop);
      }
    }

    // One resident zero tile is reused for all invalid-edge masking.
    Value zeroBank = allocBank(b, loc, 1, 1);
    zeroBank = mvinBank(b, loc, zeroPack, zeroBank, target.bankDepth);

    DenseSet<int64_t> materialized;

    struct TileBanks {
      SmallVector<Value> banks;
      int64_t panelRows;
      int64_t panelsPerBank;
      int64_t panelCount;
    };

    struct CachedTile {
      int64_t stage = -1;
      TileBanks tile{};
      int64_t firstPanel = 0;
      Value y;
      Value x;
      int64_t width = 0;
    };
    DenseMap<int64_t, TileBanks> gateCaches;
    CachedTile residualCache;

    auto allocateTile = [&](int64_t panelCount, int64_t panelRows, Value fill) {
      TileBanks tile{{}, panelRows, target.bankDepth / panelRows, panelCount};
      if (panelRows <= 0 || panelRows > target.bankDepth ||
          tile.panelsPerBank <= 0) {
        kernel.emitError("resident tile does not fit one bank");
        return tile;
      }
      int64_t bankCount =
          (panelCount + tile.panelsPerBank - 1) / tile.panelsPerBank;
      for (int64_t index = 0; index < bankCount; ++index) {
        Value bank = allocBank(b, loc, 1, 1);
        tile.banks.push_back(mvinBank(b, loc,
                                      fill == minI8 ? minPack : zeroPack, bank,
                                      target.bankDepth));
      }
      return tile;
    };

    auto releaseTile = [&](TileBanks &tile) {
      for (Value bank : tile.banks)
        releaseBank(b, loc, bank);
      tile.banks.clear();
    };

    auto loadInt8Tile = [&](Value input, int64_t inputHeight,
                            int64_t inputWidth, int64_t inputChannels, Value y0,
                            Value x0, Value firstPanel, int64_t panelCount,
                            int64_t height,
                            int64_t width) -> FailureOr<TileBanks> {
      TileBanks tile = allocateTile(panelCount, height * width, zeroI8);
      if (tile.banks.empty())
        return failure();
      Value yBegin = b.create<arith::MaxSIOp>(
          loc, zero, b.create<arith::SubIOp>(loc, zero, y0));
      Value xBegin = b.create<arith::MaxSIOp>(
          loc, zero, b.create<arith::SubIOp>(loc, zero, x0));
      Value yEnd = b.create<arith::MinSIOp>(
          loc, b.create<arith::ConstantIndexOp>(loc, height),
          b.create<arith::SubIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, inputHeight), y0));
      Value xEnd = b.create<arith::MinSIOp>(
          loc, b.create<arith::ConstantIndexOp>(loc, width),
          b.create<arith::SubIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, inputWidth), x0));
      auto inputType = cast<MemRefType>(input.getType());
      SmallVector<int64_t> inputStrides;
      int64_t inputOffset;
      bool directMvin2d =
          inputType.getElementType().isInteger(8) &&
          succeeded(inputType.getStridesAndOffset(inputStrides, inputOffset)) &&
          inputStrides[3] == 1 && inputStrides[2] == inputChannels &&
          inputStrides[1] == inputWidth * inputChannels &&
          inputChannels % 8 == 0 && inputChannels <= 1016 && inputWidth <= 1023;
      if (directMvin2d) {
        Value eight = b.create<arith::ConstantIndexOp>(loc, 8);
        Value pixelBytes = createI64Const(b, loc, inputChannels);
        Value sourceWidth = createI64Const(b, loc, inputWidth);
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t bankIndex = localPanel / tile.panelsPerBank;
          int64_t slot = localPanel % tile.panelsPerBank;
          Value channel = b.create<arith::MulIOp>(
              loc,
              b.create<arith::AddIOp>(
                  loc, firstPanel,
                  b.create<arith::ConstantIndexOp>(loc, localPanel)),
              b.create<arith::ConstantIndexOp>(loc, kTile));
          Value validBytes = b.create<arith::MinSIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, kTile),
              b.create<arith::SubIOp>(
                  loc, b.create<arith::ConstantIndexOp>(loc, inputChannels),
                  channel));
          Value validBytesI64 =
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), validBytes);
          auto yLoop = b.create<scf::ForOp>(loc, yBegin, yEnd, one,
                                            ValueRange{tile.banks[bankIndex]});
          b.setInsertionPointToStart(yLoop.getBody());
          Value localY = yLoop.getInductionVar();
          auto xLoop = b.create<scf::ForOp>(loc, xBegin, xEnd, eight,
                                            yLoop.getRegionIterArgs());
          b.setInsertionPointToStart(xLoop.getBody());
          Value localX = xLoop.getInductionVar();
          Value globalY = b.create<arith::AddIOp>(loc, y0, localY);
          Value globalX = b.create<arith::AddIOp>(loc, x0, localX);
          Value copyWidth = b.create<arith::MinSIOp>(
              loc, eight, b.create<arith::SubIOp>(loc, xEnd, localX));
          Value slice = b.create<memref::SubViewOp>(
              loc, input,
              SmallVector<OpFoldResult>{b.getIndexAttr(0), globalY, globalX,
                                        channel},
              SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                        copyWidth, validBytes},
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
          Value row = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, slot * tile.panelRows),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::MulIOp>(
                      loc, localY,
                      b.create<arith::ConstantIndexOp>(loc, width)),
                  localX));
          Value loaded = b.create<BankMvin2dOp>(
              loc, b.getI64Type(), slice, xLoop.getRegionIterArgs().front(),
              createI64Const(b, loc, 1), pixelBytes, sourceWidth,
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), row),
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), copyWidth),
              validBytesI64);
          b.create<scf::YieldOp>(loc, loaded);
          b.setInsertionPointAfter(xLoop);
          b.create<scf::YieldOp>(loc, xLoop.getResults());
          b.setInsertionPointAfter(yLoop);
          tile.banks[bankIndex] = yLoop.getResult(0);
        }
        return tile;
      }
      for (size_t bankIndex = 0; bankIndex < tile.banks.size(); ++bankIndex) {
        Value pack = b.create<memref::AllocOp>(
            loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
        b.create<linalg::FillOp>(loc, zeroI8, pack);
        int64_t panelBegin = bankIndex * tile.panelsPerBank;
        int64_t panelEnd =
            std::min<int64_t>(panelCount, panelBegin + tile.panelsPerBank);
        for (int64_t localPanel = panelBegin; localPanel < panelEnd;
             ++localPanel) {
          auto yLoop = b.create<scf::ForOp>(loc, yBegin, yEnd, one);
          b.setInsertionPointToStart(yLoop.getBody());
          Value localY = yLoop.getInductionVar();
          auto xLoop = b.create<scf::ForOp>(loc, xBegin, xEnd, one);
          b.setInsertionPointToStart(xLoop.getBody());
          Value localX = xLoop.getInductionVar();
          Value globalY = b.create<arith::AddIOp>(loc, y0, localY);
          Value globalX = b.create<arith::AddIOp>(loc, x0, localX);
          auto laneLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
          b.setInsertionPointToStart(laneLoop.getBody());
          Value lane = laneLoop.getInductionVar();
          Value channel = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc,
                  b.create<arith::AddIOp>(
                      loc, firstPanel,
                      b.create<arith::ConstantIndexOp>(loc, localPanel)),
                  b.create<arith::ConstantIndexOp>(loc, kTile)),
              lane);
          int64_t slot = localPanel % tile.panelsPerBank;
          Value row = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, slot * tile.panelRows),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::MulIOp>(
                      loc, localY,
                      b.create<arith::ConstantIndexOp>(loc, width)),
                  localX));
          auto channelValid = b.create<scf::IfOp>(
              loc,
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, channel,
                  b.create<arith::ConstantIndexOp>(loc, inputChannels)),
              false);
          b.setInsertionPointToStart(&channelValid.getThenRegion().front());
          Value value = b.create<memref::LoadOp>(
              loc, input, ValueRange{zero, globalY, globalX, channel});
          b.create<memref::StoreOp>(loc, value, pack, ValueRange{row, lane});
          b.setInsertionPointAfter(channelValid);
          b.setInsertionPointAfter(laneLoop);
          b.setInsertionPointAfter(xLoop);
          b.setInsertionPointAfter(yLoop);
        }
        tile.banks[bankIndex] =
            mvinBank(b, loc, pack, tile.banks[bankIndex], target.bankDepth);
        b.create<memref::DeallocOp>(loc, pack);
      }
      return tile;
    };

    auto copyTile = [&](TileBanks &source, TileBanks &destination,
                        int64_t panelCount, int64_t height, int64_t width,
                        int64_t destinationBase,
                        int64_t destinationStride) -> LogicalResult {
      if (source.panelCount != panelCount || source.panelRows != height * width)
        return failure();
      SmallVector<Value> destinationStates(destination.banks.begin(),
                                           destination.banks.end());
      for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
        int64_t sourceBank = localPanel / source.panelsPerBank;
        int64_t sourceSlot = localPanel % source.panelsPerBank;
        int64_t destinationBank = localPanel / destination.panelsPerBank;
        int64_t destinationSlot = localPanel % destination.panelsPerBank;
        destinationStates[destinationBank] =
            b.create<BankMaxPoolOp>(
                 loc, destinationStates[destinationBank].getType(),
                 source.banks[sourceBank], destinationStates[destinationBank],
                 createI64Const(b, loc, height * width),
                 b.getI64IntegerAttr(height), b.getI64IntegerAttr(width),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(0),
                 createI64Const(b, loc, sourceSlot * source.panelRows),
                 createI64Const(b, loc,
                                destinationSlot * destination.panelRows +
                                    destinationBase),
                 createI64Const(b, loc, destinationStride),
                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                .getOutBankOut();
      }
      destination.banks.assign(destinationStates.begin(),
                               destinationStates.end());
      return success();
    };

    std::function<LogicalResult(int64_t, Value, Value, int64_t, int64_t, Value,
                                int64_t, TileBanks &, int64_t, int64_t)>
        emitInto;
    emitInto = [&](int64_t stageIndex, Value y0, Value x0, int64_t height,
                   int64_t width, Value firstPanel, int64_t panelCount,
                   TileBanks &destination, int64_t destinationBase,
                   int64_t destinationStride) -> LogicalResult {
      // Keep nested stage emission before this anchor and resume after it.
      Operation *insertionAnchor =
          b.create<arith::ConstantIndexOp>(loc, 0).getOperation();
      b.setInsertionPoint(insertionAnchor);
      llvm::scope_exit eraseInsertionAnchor([&]() {
        b.setInsertionPointAfter(insertionAnchor);
        b.eraseOp(insertionAnchor);
      });
      Stage &stage = stages[stageIndex];
      const bool fp32Output = stage.finalOutput && !stage.pool;
      const bool externalInput = !producer.contains(stage.input);
      int64_t totalPanels = (stage.outputChannels + kTile - 1) / kTile;
      if (height <= 0 || width <= 0 || panelCount <= 0 ||
          panelCount > totalPanels || destination.panelCount != panelCount ||
          destinationBase < 0 || destinationStride < width ||
          destinationBase + (height - 1) * destinationStride + width >
              destination.panelRows)
        return stage.op->emitError("invalid resident tile request");

      if (stage.channelSlice) {
        IntegerAttr::ValueType requestedFirstPanel;
        if (!matchPattern(firstPanel, m_ConstantInt(&requestedFirstPanel)))
          return stage.op->emitError(
              "resident channel slice panel offset must be constant");
        Value sourceFirstPanel = b.create<arith::ConstantIndexOp>(
            loc,
            requestedFirstPanel.getSExtValue() + stage.channelOffset / kTile);
        TileBanks source = allocateTile(panelCount, height * width, zeroI8);
        if (source.banks.empty())
          return failure();
        if (producer.contains(stage.input)) {
          if (failed(emitInto(producer.lookup(stage.input), y0, x0, height,
                              width, sourceFirstPanel, panelCount, source, 0,
                              width)))
            return failure();
        } else {
          FailureOr<TileBanks> loaded =
              loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                           stage.inputChannels, y0, x0, sourceFirstPanel,
                           panelCount, height, width);
          if (failed(loaded))
            return stage.op->emitError(
                "failed to load INT8 channel slice input");
          releaseTile(source);
          source = std::move(*loaded);
        }
        LogicalResult copied =
            copyTile(source, destination, panelCount, height, width,
                     destinationBase, destinationStride);
        releaseTile(source);
        return copied;
      }

      if (stage.channelConcat) {
        IntegerAttr::ValueType requestedFirstPanel;
        if (!matchPattern(firstPanel, m_ConstantInt(&requestedFirstPanel)))
          return stage.op->emitError(
              "resident channel concat panel offset must be constant");
        int64_t requestedPanel = requestedFirstPanel.getSExtValue();
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t outputPanel = requestedPanel + localPanel;
          int64_t sourcePanelBase = 0;
          int64_t sourceIndex = -1;
          for (auto [index, segment] : llvm::enumerate(stage.channelSegments)) {
            int64_t panels = segment / kTile;
            if (outputPanel >= sourcePanelBase &&
                outputPanel < sourcePanelBase + panels) {
              sourceIndex = index;
              break;
            }
            sourcePanelBase += panels;
          }
          if (sourceIndex < 0)
            return stage.op->emitError(
                "resident channel concat source is missing");
          Value sourceFirstPanel = b.create<arith::ConstantIndexOp>(
              loc, outputPanel - sourcePanelBase);
          TileBanks source = allocateTile(1, height * width, zeroI8);
          if (source.banks.empty())
            return failure();
          Value sourceValue = stage.inputs[sourceIndex];
          if (producer.contains(sourceValue)) {
            if (failed(emitInto(producer.lookup(sourceValue), y0, x0, height,
                                width, sourceFirstPanel, 1, source, 0, width)))
              return failure();
          } else {
            auto sourceType = cast<MemRefType>(sourceValue.getType());
            FailureOr<TileBanks> loaded =
                loadInt8Tile(sourceValue, sourceType.getShape()[1],
                             sourceType.getShape()[2], sourceType.getShape()[3],
                             y0, x0, sourceFirstPanel, 1, height, width);
            if (failed(loaded))
              return stage.op->emitError(
                  "failed to load INT8 channel concat input");
            releaseTile(source);
            source = std::move(*loaded);
          }
          int64_t destinationBank = localPanel / destination.panelsPerBank;
          int64_t destinationSlot = localPanel % destination.panelsPerBank;
          destination.banks[destinationBank] =
              b.create<BankMaxPoolOp>(
                   loc, destination.banks[destinationBank].getType(),
                   source.banks.front(), destination.banks[destinationBank],
                   createI64Const(b, loc, height * width),
                   b.getI64IntegerAttr(height), b.getI64IntegerAttr(width),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                   createI64Const(b, loc,
                                  destinationSlot * destination.panelRows +
                                      destinationBase),
                   createI64Const(b, loc, destinationStride),
                   b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                  .getOutBankOut();
          releaseTile(source);
        }
        return success();
      }

      if (stage.resizeNearest) {
        for (int64_t localY = 0; localY < height; ++localY) {
          Value sourceY = b.create<arith::DivUIOp>(
              loc,
              b.create<arith::AddIOp>(
                  loc, y0, b.create<arith::ConstantIndexOp>(loc, localY)),
              b.create<arith::ConstantIndexOp>(loc, stage.scaleH));
          for (int64_t localX = 0; localX < width; ++localX) {
            Value sourceX = b.create<arith::DivUIOp>(
                loc,
                b.create<arith::AddIOp>(
                    loc, x0, b.create<arith::ConstantIndexOp>(loc, localX)),
                b.create<arith::ConstantIndexOp>(loc, stage.scaleW));
            if (producer.contains(stage.input)) {
              if (failed(emitInto(
                      producer.lookup(stage.input), sourceY, sourceX, 1, 1,
                      firstPanel, panelCount, destination,
                      destinationBase + localY * destinationStride + localX,
                      destinationStride)))
                return failure();
            } else {
              TileBanks source = allocateTile(panelCount, 1, zeroI8);
              if (source.banks.empty())
                return failure();
              FailureOr<TileBanks> loaded =
                  loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                               stage.inputChannels, sourceY, sourceX,
                               firstPanel, panelCount, 1, 1);
              if (failed(loaded))
                return stage.op->emitError(
                    "failed to load INT8 resize-nearest input");
              releaseTile(source);
              source = std::move(*loaded);
              LogicalResult copied = copyTile(
                  source, destination, panelCount, 1, 1,
                  destinationBase + localY * destinationStride + localX,
                  destinationStride);
              releaseTile(source);
              if (failed(copied))
                return failure();
            }
          }
        }
        return success();
      }

      CachedTile *cache = nullptr;
      if (residualCache.stage == stageIndex)
        cache = &residualCache;
      if (cache) {
        IntegerAttr::ValueType requestedFirstPanel;
        if (!matchPattern(firstPanel, m_ConstantInt(&requestedFirstPanel)))
          cache = nullptr;
        else if (requestedFirstPanel.getSExtValue() < cache->firstPanel ||
                 requestedFirstPanel.getSExtValue() + panelCount >
                     cache->firstPanel + cache->tile.panelCount)
          cache = nullptr;
      }
      if (cache) {
        IntegerAttr::ValueType requestedFirstPanel;
        if (!matchPattern(firstPanel, m_ConstantInt(&requestedFirstPanel)))
          return stage.op->emitError(
              "cached tile panel offset must be constant");
        SmallVector<Value> destinationStates(destination.banks.begin(),
                                             destination.banks.end());
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t sourcePanel = requestedFirstPanel.getSExtValue() -
                                cache->firstPanel + localPanel;
          int64_t sourceBank = sourcePanel / cache->tile.panelsPerBank;
          int64_t sourceSlot = sourcePanel % cache->tile.panelsPerBank;
          int64_t destinationBank = localPanel / destination.panelsPerBank;
          int64_t destinationSlot = localPanel % destination.panelsPerBank;
          Value sourceBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::ConstantIndexOp>(
                      loc, sourceSlot * cache->tile.panelRows),
                  b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(
                          loc, b.create<arith::SubIOp>(loc, y0, cache->y),
                          b.create<arith::ConstantIndexOp>(loc, cache->width)),
                      b.create<arith::SubIOp>(loc, x0, cache->x))));
          destinationStates[destinationBank] =
              b.create<BankMaxPoolOp>(
                   loc, destinationStates[destinationBank].getType(),
                   cache->tile.banks[sourceBank],
                   destinationStates[destinationBank],
                   createI64Const(b, loc, height * width),
                   b.getI64IntegerAttr(cache->width),
                   b.getI64IntegerAttr(width), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(0), sourceBase,
                   createI64Const(b, loc,
                                  destinationSlot * destination.panelRows +
                                      destinationBase),
                   createI64Const(b, loc, destinationStride),
                   b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                  .getOutBankOut();
        }
        destination.banks.assign(destinationStates.begin(),
                                 destinationStates.end());
        return success();
      }

      if (materialized.contains(stageIndex)) {
        auto inputType = cast<MemRefType>(stage.output.getType());
        SmallVector<int64_t> inputStrides;
        int64_t inputOffset;
        bool directMvin2d =
            inputType.getElementType().isInteger(8) &&
            succeeded(
                inputType.getStridesAndOffset(inputStrides, inputOffset)) &&
            inputStrides[3] == 1 && inputStrides[2] == stage.outputChannels &&
            inputStrides[1] == stage.outputWidth * stage.outputChannels &&
            stage.outputChannels % 8 == 0 && stage.outputChannels <= 1016 &&
            stage.outputWidth <= 1023;
        if (directMvin2d) {
          for (Value &bank : destination.banks)
            bank = mvinBank(b, loc, zeroPack, bank, target.bankDepth);
          Value yBegin = b.create<arith::MaxSIOp>(loc, y0, zero);
          Value yEnd = b.create<arith::MinSIOp>(
              loc,
              b.create<arith::AddIOp>(
                  loc, y0, b.create<arith::ConstantIndexOp>(loc, height)),
              b.create<arith::ConstantIndexOp>(loc, stage.outputHeight));
          Value xBegin = b.create<arith::MaxSIOp>(loc, x0, zero);
          Value xEnd = b.create<arith::MinSIOp>(
              loc,
              b.create<arith::AddIOp>(
                  loc, x0, b.create<arith::ConstantIndexOp>(loc, width)),
              b.create<arith::ConstantIndexOp>(loc, stage.outputWidth));
          Value eight = b.create<arith::ConstantIndexOp>(loc, 8);
          Value pixelBytes = createI64Const(b, loc, stage.outputChannels);
          Value sourceWidth = createI64Const(b, loc, stage.outputWidth);
          for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
            int64_t bankIndex = localPanel / destination.panelsPerBank;
            int64_t bankSlot = localPanel % destination.panelsPerBank;
            Value channel = b.create<arith::MulIOp>(
                loc,
                b.create<arith::AddIOp>(
                    loc, firstPanel,
                    b.create<arith::ConstantIndexOp>(loc, localPanel)),
                b.create<arith::ConstantIndexOp>(loc, kTile));
            Value validBytes = b.create<arith::MinSIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, kTile),
                b.create<arith::SubIOp>(
                    loc,
                    b.create<arith::ConstantIndexOp>(loc, stage.outputChannels),
                    channel));
            Value validBytesI64 =
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), validBytes);
            auto yLoop =
                b.create<scf::ForOp>(loc, yBegin, yEnd, one,
                                     ValueRange{destination.banks[bankIndex]});
            b.setInsertionPointToStart(yLoop.getBody());
            Value globalY = yLoop.getInductionVar();
            auto xLoop = b.create<scf::ForOp>(loc, xBegin, xEnd, eight,
                                              yLoop.getRegionIterArgs());
            b.setInsertionPointToStart(xLoop.getBody());
            Value globalX = xLoop.getInductionVar();
            Value copyWidth = b.create<arith::MinSIOp>(
                loc, eight, b.create<arith::SubIOp>(loc, xEnd, globalX));
            Value slice = b.create<memref::SubViewOp>(
                loc, stage.output,
                SmallVector<OpFoldResult>{b.getIndexAttr(0), globalY, globalX,
                                          channel},
                SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                          copyWidth, validBytes},
                SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
            Value destinationRow = b.create<arith::AddIOp>(
                loc,
                b.create<arith::ConstantIndexOp>(
                    loc, bankSlot * destination.panelRows + destinationBase),
                b.create<arith::AddIOp>(
                    loc,
                    b.create<arith::MulIOp>(
                        loc, b.create<arith::SubIOp>(loc, globalY, y0),
                        b.create<arith::ConstantIndexOp>(loc,
                                                         destinationStride)),
                    b.create<arith::SubIOp>(loc, globalX, x0)));
            Value loaded = b.create<BankMvin2dOp>(
                loc, b.getI64Type(), slice, xLoop.getRegionIterArgs().front(),
                createI64Const(b, loc, 1), pixelBytes, sourceWidth,
                b.create<arith::IndexCastOp>(loc, b.getI64Type(),
                                             destinationRow),
                b.create<arith::IndexCastOp>(loc, b.getI64Type(), copyWidth),
                validBytesI64);
            b.create<scf::YieldOp>(loc, loaded);
            b.setInsertionPointAfter(xLoop);
            b.create<scf::YieldOp>(loc, xLoop.getResults());
            b.setInsertionPointAfter(yLoop);
            destination.banks[bankIndex] = yLoop.getResult(0);
          }
          return success();
        }
        SmallVector<Value> packs;
        for (Value bank : destination.banks) {
          Value pack = b.create<memref::AllocOp>(
              loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
          b.create<linalg::FillOp>(loc, zeroI8, pack);
          packs.push_back(pack);
        }
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t bankIndex = localPanel / destination.panelsPerBank;
          int64_t bankSlot = localPanel % destination.panelsPerBank;
          auto yLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, height), one);
          b.setInsertionPointToStart(yLoop.getBody());
          Value localY = yLoop.getInductionVar();
          auto xLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, width), one);
          b.setInsertionPointToStart(xLoop.getBody());
          Value localX = xLoop.getInductionVar();
          Value globalY = b.create<arith::AddIOp>(loc, y0, localY);
          Value globalX = b.create<arith::AddIOp>(loc, x0, localX);
          Value yValid = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, globalY,
                                      zero),
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, globalY,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputHeight)));
          Value xValid = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, globalX,
                                      zero),
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, globalX,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputWidth)));
          auto valid = b.create<scf::IfOp>(
              loc, b.create<arith::AndIOp>(loc, yValid, xValid), false);
          b.setInsertionPointToStart(&valid.getThenRegion().front());
          auto laneLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
          b.setInsertionPointToStart(laneLoop.getBody());
          Value lane = laneLoop.getInductionVar();
          Value panel = b.create<arith::AddIOp>(
              loc, firstPanel,
              b.create<arith::ConstantIndexOp>(loc, localPanel));
          Value channel = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, panel, b.create<arith::ConstantIndexOp>(loc, kTile)),
              lane);
          auto channelValid = b.create<scf::IfOp>(
              loc,
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, channel,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputChannels)),
              false);
          b.setInsertionPointToStart(&channelValid.getThenRegion().front());
          Value value = b.create<memref::LoadOp>(
              loc, stage.output, ValueRange{zero, globalY, globalX, channel});
          Value row = b.create<arith::AddIOp>(
              loc,
              b.create<arith::ConstantIndexOp>(
                  loc, bankSlot * destination.panelRows + destinationBase),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::MulIOp>(
                      loc, localY,
                      b.create<arith::ConstantIndexOp>(loc, destinationStride)),
                  localX));
          b.create<memref::StoreOp>(loc, value, packs[bankIndex],
                                    ValueRange{row, lane});
          b.setInsertionPointAfter(channelValid);
          b.setInsertionPointAfter(laneLoop);
          b.setInsertionPointAfter(valid);
          b.setInsertionPointAfter(yLoop);
        }
        for (size_t index = 0; index < destination.banks.size(); ++index) {
          mvinBank(b, loc, packs[index], destination.banks[index],
                   target.bankDepth);
          b.create<memref::DeallocOp>(loc, packs[index]);
        }
        return success();
      }

      auto maskInvalidOutput = [&]() {
        for (size_t bankIndex = 0; bankIndex < destination.banks.size();
             ++bankIndex) {
          int64_t panelBegin = bankIndex * destination.panelsPerBank;
          int64_t panelEnd = std::min<int64_t>(
              panelCount, panelBegin + destination.panelsPerBank);
          auto panelLoop = b.create<scf::ForOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, panelBegin),
              b.create<arith::ConstantIndexOp>(loc, panelEnd), one,
              ValueRange{destination.banks[bankIndex]});
          b.setInsertionPointToStart(panelLoop.getBody());
          Value localPanel = panelLoop.getInductionVar();
          Value panelState = panelLoop.getRegionIterArgs().front();
          auto yLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, height), one,
              ValueRange{panelState});
          b.setInsertionPointToStart(yLoop.getBody());
          Value localY = yLoop.getInductionVar();
          Value yState = yLoop.getRegionIterArgs().front();
          auto xLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, width), one,
              ValueRange{yState});
          b.setInsertionPointToStart(xLoop.getBody());
          Value localX = xLoop.getInductionVar();
          Value xState = xLoop.getRegionIterArgs().front();
          Value globalY = b.create<arith::AddIOp>(loc, y0, localY);
          Value globalX = b.create<arith::AddIOp>(loc, x0, localX);
          Value yInvalid = b.create<arith::OrIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalY,
                                      zero),
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge, globalY,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputHeight)));
          Value xInvalid = b.create<arith::OrIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalX,
                                      zero),
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge, globalX,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputWidth)));
          auto invalid = b.create<scf::IfOp>(
              loc, b.create<arith::OrIOp>(loc, yInvalid, xInvalid), false);
          b.setInsertionPointToStart(&invalid.getThenRegion().front());
          Value bankSlot = b.create<arith::SubIOp>(
              loc, localPanel,
              b.create<arith::ConstantIndexOp>(loc, panelBegin));
          Value outputBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(loc, bankSlot,
                                              b.create<arith::ConstantIndexOp>(
                                                  loc, destination.panelRows)),
                      b.create<arith::ConstantIndexOp>(loc, destinationBase)),
                  b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(loc, localY,
                                              b.create<arith::ConstantIndexOp>(
                                                  loc, destinationStride)),
                      localX)));
          b.create<BankMaxPoolOp>(
              loc, xState.getType(), zeroBank, xState,
              createI64Const(b, loc, 1), b.getI64IntegerAttr(1),
              b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
              b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
              createI64Const(b, loc, 0), outputBase, createI64Const(b, loc, 1),
              b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
          b.setInsertionPointAfter(invalid);
          b.create<scf::YieldOp>(loc, xState);
          b.setInsertionPointAfter(xLoop);
          b.create<scf::YieldOp>(loc, xLoop.getResult(0));
          b.setInsertionPointAfter(yLoop);
          b.create<scf::YieldOp>(loc, yLoop.getResult(0));
          b.setInsertionPointAfter(panelLoop);
          destination.banks[bankIndex] = panelLoop.getResult(0);
        }
      };

      if (stage.add) {
        Value lhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale / stage.outputScale));
        Value rhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.rhsScale / stage.outputScale));

        bool mainBranchExternal = !producer.contains(stage.input);
        if (!mainBranchExternal) {
          for (int64_t cursor = producer.lookup(stage.input);
               stages[cursor].input != kernel.getInput();) {
            if (!producer.contains(stages[cursor].input)) {
              mainBranchExternal = true;
              break;
            }
            int64_t next = producer.lookup(stages[cursor].input);
            if (next >= cursor)
              return stage.op->emitError("INT8 Add branch is not acyclic");
            cursor = next;
          }
        }
        const bool usePanelAdd =
            mainBranchExternal || !producer.contains(stage.rhs);
        if (usePanelAdd) {
          IntegerAttr::ValueType requestedFirstPanel;
          if (!matchPattern(firstPanel, m_ConstantInt(&requestedFirstPanel)))
            return stage.op->emitError(
                "external INT8 Add panel offset must be constant");
          int64_t panelRows = height * width;
          for (int64_t panelBegin = 0; panelBegin < panelCount;) {
            int64_t destinationBank = panelBegin / destination.panelsPerBank;
            int64_t firstDestinationSlot =
                panelBegin % destination.panelsPerBank;
            int64_t chunkCount = std::min<int64_t>(
                {panelCount - panelBegin, target.bankDepth / panelRows,
                 destination.panelsPerBank - firstDestinationSlot});
            Value panel = b.create<arith::ConstantIndexOp>(
                loc, requestedFirstPanel.getSExtValue() + panelBegin);
            TileBanks lhs = allocateTile(chunkCount, panelRows, zeroI8);
            TileBanks rhs = allocateTile(chunkCount, panelRows, zeroI8);
            if (lhs.banks.size() != 1 || rhs.banks.size() != 1)
              return failure();
            if (producer.contains(stage.input)) {
              if (failed(emitInto(producer.lookup(stage.input), y0, x0, height,
                                  width, panel, chunkCount, lhs, 0, width)))
                return failure();
            } else {
              auto input = cast<MemRefType>(stage.input.getType());
              FailureOr<TileBanks> loaded =
                  loadInt8Tile(stage.input, input.getShape()[1],
                               input.getShape()[2], input.getShape()[3], y0, x0,
                               panel, chunkCount, height, width);
              if (failed(loaded))
                return stage.op->emitError("failed to load INT8 Add lhs");
              releaseTile(lhs);
              lhs = std::move(*loaded);
            }
            if (producer.contains(stage.rhs)) {
              if (failed(emitInto(producer.lookup(stage.rhs), y0, x0, height,
                                  width, panel, chunkCount, rhs, 0, width)))
                return failure();
            } else {
              auto input = cast<MemRefType>(stage.rhs.getType());
              FailureOr<TileBanks> loaded =
                  loadInt8Tile(stage.rhs, input.getShape()[1],
                               input.getShape()[2], input.getShape()[3], y0, x0,
                               panel, chunkCount, height, width);
              if (failed(loaded))
                return stage.op->emitError("failed to load INT8 Add rhs");
              releaseTile(rhs);
              rhs = std::move(*loaded);
            }
            TileBanks sum = allocateTile(chunkCount, panelRows, zeroI8);
            if (sum.banks.size() != 1)
              return failure();
            sum.banks.front() = b.create<BankInt8AddOp>(
                loc, sum.banks.front().getType(), lhs.banks.front(),
                rhs.banks.front(), sum.banks.front(),
                createI64Const(b, loc, target.bankDepth), lhsRatio, rhsRatio,
                b.getBoolAttr(stage.activation == 1));
            for (int64_t localPanel = 0; localPanel < chunkCount;
                 ++localPanel) {
              int64_t destinationSlot = firstDestinationSlot + localPanel;
              destination.banks[destinationBank] =
                  b.create<BankMaxPoolOp>(
                       loc, destination.banks[destinationBank].getType(),
                       sum.banks.front(), destination.banks[destinationBank],
                       createI64Const(b, loc, height * width),
                       b.getI64IntegerAttr(height), b.getI64IntegerAttr(width),
                       b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(0),
                       createI64Const(b, loc, localPanel * panelRows),
                       createI64Const(b, loc,
                                      destinationSlot * destination.panelRows +
                                          destinationBase),
                       createI64Const(b, loc, destinationStride),
                       b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                      .getOutBankOut();
            }
            releaseTile(lhs);
            releaseTile(rhs);
            releaseTile(sum);
            panelBegin += chunkCount;
          }
          return success();
        }

        int64_t branchStage = producer.lookup(stage.input);
        DenseSet<int64_t> branchAncestors;
        for (int64_t cursor = branchStage;;) {
          branchAncestors.insert(cursor);
          Value input = stages[cursor].input;
          if (input == kernel.getInput())
            break;
          if (!producer.contains(input))
            return stage.op->emitError(
                "INT8 Add main branch has no region producer");
          int64_t next = producer.lookup(input);
          if (next >= cursor)
            return stage.op->emitError("INT8 Add branch is not acyclic");
          cursor = next;
        }
        int64_t residualStage = producer.lookup(stage.rhs);
        bool residualHasCommonProducer = true;
        while (!branchAncestors.contains(residualStage)) {
          Value input = stages[residualStage].input;
          if (!producer.contains(input)) {
            residualHasCommonProducer = false;
            break;
          }
          int64_t next = producer.lookup(input);
          if (next >= residualStage)
            return stage.op->emitError("INT8 Add branch is not acyclic");
          residualStage = next;
        }
        Value residualY = y0;
        Value residualX = x0;
        int64_t residualHeight = height;
        int64_t residualWidth = width;
        for (int64_t cursor = branchStage;
             residualHasCommonProducer && cursor != residualStage;) {
          Stage &branch = stages[cursor];
          if (branch.add || branch.average ||
              branch.input == kernel.getInput() ||
              !producer.contains(branch.input))
            return stage.op->emitError(
                "INT8 Add main branch must be a linear Conv chain from rhs");
          residualY = b.create<arith::SubIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, residualY,
                  b.create<arith::ConstantIndexOp>(loc, branch.stride)),
              b.create<arith::ConstantIndexOp>(loc, branch.padding));
          residualX = b.create<arith::SubIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, residualX,
                  b.create<arith::ConstantIndexOp>(loc, branch.stride)),
              b.create<arith::ConstantIndexOp>(loc, branch.padding));
          residualHeight = (residualHeight - 1) * branch.stride + branch.kernel;
          residualWidth = (residualWidth - 1) * branch.stride + branch.kernel;
          int64_t next = producer.lookup(branch.input);
          if (next >= cursor)
            return stage.op->emitError("INT8 Add branch is not acyclic");
          cursor = next;
        }
        int64_t residualPanelRows = residualHeight * residualWidth;
        int64_t residualPanels =
            (stages[residualStage].outputChannels + kTile - 1) / kTile;
        int64_t residualPanelsPerBank =
            residualPanelRows <= target.bankDepth
                ? target.bankDepth / residualPanelRows
                : 0;
        if (residualPanelsPerBank <= 0)
          return stage.op->emitError(
              "INT8 Add residual tile does not fit one bank");
        int64_t chunkPanels = std::min<int64_t>(
            residualPanels,
            std::min(destination.panelsPerBank, residualPanelsPerBank));
        if (chunkPanels <= 0)
          return stage.op->emitError("INT8 Add panel chunk is empty");

        for (int64_t panelBegin = 0; panelBegin < panelCount;
             panelBegin += chunkPanels) {
          int64_t chunkCount =
              std::min<int64_t>(chunkPanels, panelCount - panelBegin);
          Value chunkFirstPanel = firstPanel;
          if (panelBegin != 0) {
            IntegerAttr::ValueType firstPanelValue;
            if (!matchPattern(firstPanel, m_ConstantInt(&firstPanelValue)))
              return stage.op->emitError(
                  "INT8 Add panel offset must be constant");
            chunkFirstPanel = b.create<arith::ConstantIndexOp>(
                loc, firstPanelValue.getSExtValue() + panelBegin);
          }

          TileBanks lhs =
              allocateTile(chunkCount, destination.panelRows, zeroI8);
          TileBanks rhs =
              allocateTile(chunkCount, destination.panelRows, zeroI8);
          if (lhs.banks.size() != 1 || rhs.banks.size() != 1)
            return stage.op->emitError(
                "INT8 Add branch panel chunk must fit one bank");
          if (failed(emitInto(producer.lookup(stage.input), y0, x0, height,
                              width, chunkFirstPanel, chunkCount, lhs, 0,
                              destinationStride)))
            return failure();

          TileBanks residual =
              allocateTile(chunkCount, residualPanelRows, zeroI8);
          if (residual.banks.empty() ||
              failed(emitInto(residualStage, residualY, residualX,
                              residualHeight, residualWidth, chunkFirstPanel,
                              chunkCount, residual, 0, residualWidth)))
            return failure();
          residualCache.stage = residualStage;
          residualCache.tile = residual;
          IntegerAttr::ValueType cachedFirstPanel;
          if (!matchPattern(chunkFirstPanel, m_ConstantInt(&cachedFirstPanel)))
            return stage.op->emitError(
                "INT8 Add residual panel offset must be constant");
          residualCache.firstPanel = cachedFirstPanel.getSExtValue();
          residualCache.y = residualY;
          residualCache.x = residualX;
          residualCache.width = residualWidth;
          if (failed(emitInto(producer.lookup(stage.rhs), y0, x0, height, width,
                              chunkFirstPanel, chunkCount, rhs, 0,
                              destinationStride)))
            return failure();
          residualCache.stage = -1;
          residualCache.tile.banks.clear();
          releaseTile(residual);

          TileBanks sum =
              allocateTile(chunkCount, destination.panelRows, zeroI8);
          if (sum.banks.size() != 1)
            return stage.op->emitError(
                "INT8 Add output panel chunk must fit one bank");
          b.create<BankInt8AddOp>(
              loc, sum.banks.front().getType(), lhs.banks.front(),
              rhs.banks.front(), sum.banks.front(),
              createI64Const(b, loc, target.bankDepth), lhsRatio, rhsRatio,
              b.getBoolAttr(stage.activation == 1));
          int64_t destinationBank = panelBegin / destination.panelsPerBank;
          Value destinationState = destination.banks[destinationBank];
          // BankInt8Add packs all panels in the chunk into one bank.  Copy
          // each panel's spatial rows into its destination slot explicitly.
          for (int64_t localPanel = 0; localPanel < chunkCount; ++localPanel) {
            int64_t destinationSlot =
                (panelBegin + localPanel) % destination.panelsPerBank;
            destinationState =
                b.create<BankMaxPoolOp>(
                     loc, destinationState.getType(), sum.banks.front(),
                     destinationState, createI64Const(b, loc, height * width),
                     b.getI64IntegerAttr(height), b.getI64IntegerAttr(width),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                     b.getI64IntegerAttr(0),
                     createI64Const(b, loc, localPanel * destination.panelRows),
                     createI64Const(b, loc,
                                    destinationSlot * destination.panelRows +
                                        destinationBase),
                     createI64Const(b, loc, destinationStride),
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                    .getOutBankOut();
          }
          destination.banks[destinationBank] = destinationState;
          releaseTile(lhs);
          releaseTile(rhs);
          releaseTile(sum);
        }
        maskInvalidOutput();
        return success();
      }

      if (stage.multiply) {
        if (height * width > target.bankDepth)
          return stage.op->emitError("INT8 Mul tile exceeds one bank");
        Value ratio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale * stage.rhsScale /
                              stage.outputScale));
        int64_t gateStage = producer.lookup(stage.rhs);
        auto gateIt = gateCaches.find(gateStage);
        if (gateIt == gateCaches.end()) {
          InFlightDiagnostic diagnostic =
              stage.op->emitError("INT8 Mul gate was not precomputed");
          diagnostic << " (multiplyStage=" << stageIndex
                     << ", gateStage=" << gateStage << ", cachedStages=";
          for (const auto &entry : gateCaches)
            diagnostic << entry.first << " ";
          diagnostic << ")";
          return failure();
        }
        TileBanks &gate = gateIt->second;
        if (gate.panelCount < panelCount || gate.banks.size() != 1)
          return stage.op->emitError("INT8 Mul gate cache layout mismatch");

        for (size_t destinationBank = 0;
             destinationBank < destination.banks.size(); ++destinationBank) {
          int64_t panelBegin = destinationBank * destination.panelsPerBank;
          int64_t panelEnd = std::min<int64_t>(
              panelCount, panelBegin + destination.panelsPerBank);
          auto panelLoop = b.create<scf::ForOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, panelBegin),
              b.create<arith::ConstantIndexOp>(loc, panelEnd), one,
              ValueRange{destination.banks[destinationBank]});
          b.setInsertionPointToStart(panelLoop.getBody());
          Value localPanel = panelLoop.getInductionVar();
          Value destinationState = panelLoop.getRegionIterArgs().front();
          Value panel = b.create<arith::AddIOp>(loc, firstPanel, localPanel);
          TileBanks input;
          if (producer.contains(stage.input)) {
            input = allocateTile(1, height * width, zeroI8);
            if (failed(emitInto(producer.lookup(stage.input), y0, x0, height,
                                width, panel, 1, input, 0, width)))
              return failure();
          } else {
            FailureOr<TileBanks> loaded = loadInt8Tile(
                stage.input, stage.inputHeight, stage.inputWidth,
                stage.inputChannels, y0, x0, panel, 1, height, width);
            if (failed(loaded))
              return stage.op->emitError("failed to load INT8 Mul input");
            input = std::move(*loaded);
          }
          Value multiplied = allocBank(b, loc, 1, 1);
          multiplied =
              b.create<BankInt8MulOp>(
                   loc, multiplied.getType(), gate.banks.front(),
                   input.banks.front(), multiplied,
                   createI64Const(b, loc, height * width), ratio,
                   b.create<arith::IndexCastOp>(loc, b.getI64Type(), panel))
                  .getOutputBankOut();

          Value destinationSlot = b.create<arith::SubIOp>(
              loc, localPanel,
              b.create<arith::ConstantIndexOp>(loc, panelBegin));
          Value outputBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::MulIOp>(loc, destinationSlot,
                                          b.create<arith::ConstantIndexOp>(
                                              loc, destination.panelRows)),
                  b.create<arith::ConstantIndexOp>(loc, destinationBase)));
          Value destinationNext =
              b.create<BankMaxPoolOp>(
                   loc, destinationState.getType(), multiplied,
                   destinationState, createI64Const(b, loc, height * width),
                   b.getI64IntegerAttr(width), b.getI64IntegerAttr(width),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                   outputBase, createI64Const(b, loc, destinationStride),
                   b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                  .getOutBankOut();
          releaseTile(input);
          releaseBank(b, loc, multiplied);
          b.create<scf::YieldOp>(loc, destinationNext);
          b.setInsertionPointAfter(panelLoop);
          destination.banks[destinationBank] = panelLoop.getResult(0);
        }
        maskInvalidOutput();
        return success();
      }

      if (stage.average) {
        int64_t inputRows = stage.inputHeight * stage.inputWidth;
        bool fitsOneBank = inputRows <= target.bankDepth;
        if (!fitsOneBank && !externalInput &&
            !materialized.contains(producer.lookup(stage.input)))
          return stage.op->emitError(
              "large GlobalAvgPool requires a materialized input");
        int64_t sumK = fitsOneBank ? target.bankDepth : kTile;
        Value oneI8 = b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                                  b.getI8IntegerAttr(1));
        Value ratio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale /
                              (inputRows * stage.outputScale)));
        Value onesPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({sumK / kTile, kTile}, b.getI8Type()));
        Value biasPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getI32Type()));
        Value scalePack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getF32Type()));
        b.create<linalg::FillOp>(loc, oneI8, onesPack);
        b.create<linalg::FillOp>(loc, zeroI32, biasPack);
        b.create<linalg::FillOp>(loc, ratio, scalePack);

        auto panelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, panelCount), one,
            ValueRange{destination.banks.front()});
        b.setInsertionPointToStart(panelLoop.getBody());
        Value localPanel = panelLoop.getInductionVar();
        Value destinationState = panelLoop.getRegionIterArgs().front();
        Value panel = b.create<arith::AddIOp>(loc, firstPanel, localPanel);

        TileBanks fullSource;
        if (fitsOneBank) {
          fullSource = allocateTile(1, target.bankDepth, zeroI8);
          if (externalInput) {
            FailureOr<TileBanks> loaded =
                loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                             stage.inputChannels, zero, zero, panel, 1,
                             stage.inputHeight, stage.inputWidth);
            if (failed(loaded))
              return stage.op->emitError(
                  "failed to load INT8 GlobalAvgPool input");
            releaseTile(fullSource);
            fullSource = std::move(*loaded);
          } else if (failed(emitInto(producer.lookup(stage.input), zero, zero,
                                     stage.inputHeight, stage.inputWidth, panel,
                                     1, fullSource, 0, stage.inputWidth))) {
            return failure();
          }
        }

        Value onesBank = allocBank(b, loc, 1, 1);
        Value onesLoaded = mvinBank(b, loc, onesPack, onesBank, sumK / kTile);
        Value biasBank = allocBank(b, loc, 1, 1);
        Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
        Value biasState = b.create<BankSMatMulBiasOp>(
            loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
        releaseBank(b, loc, biasState);
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value result = allocBank(b, loc, 1, 1);

        Value resultState;
        if (fitsOneBank) {
          resultState =
              b.create<BankSMatMulOp>(
                   loc, result.getType(), onesLoaded, fullSource.banks.front(),
                   result,
                   createI64ConstU(b, loc,
                                   matrixRs2(1, kTile, target.bankDepth)),
                   createI1Const(b, loc, true), createI1Const(b, loc, true),
                   createI64Const(b, loc, 0))
                  .getWrBankOut();
          releaseTile(fullSource);
        } else {
          Value four = b.create<arith::ConstantIndexOp>(loc, 4);
          auto yLoop = b.create<scf::ForOp>(
              loc, zero,
              b.create<arith::ConstantIndexOp>(loc, stage.inputHeight), four,
              ValueRange{result});
          b.setInsertionPointToStart(yLoop.getBody());
          Value y = yLoop.getInductionVar();
          Value yState = yLoop.getRegionIterArgs().front();
          auto xLoop = b.create<scf::ForOp>(
              loc, zero,
              b.create<arith::ConstantIndexOp>(loc, stage.inputWidth), four,
              ValueRange{yState});
          b.setInsertionPointToStart(xLoop.getBody());
          Value x = xLoop.getInductionVar();
          Value xState = xLoop.getRegionIterArgs().front();
          TileBanks source = allocateTile(1, kTile, zeroI8);
          if (externalInput) {
            FailureOr<TileBanks> loaded =
                loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                             stage.inputChannels, y, x, panel, 1, 4, 4);
            if (failed(loaded))
              return stage.op->emitError(
                  "failed to load INT8 GlobalAvgPool input");
            releaseTile(source);
            source = std::move(*loaded);
          } else if (failed(emitInto(producer.lookup(stage.input), y, x, 4, 4,
                                     panel, 1, source, 0, 4))) {
            return failure();
          }
          Value first = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, y, zero),
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, x, zero));
          Value last = b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge,
                  b.create<arith::AddIOp>(loc, y, four),
                  b.create<arith::ConstantIndexOp>(loc, stage.inputHeight)),
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::sge,
                  b.create<arith::AddIOp>(loc, x, four),
                  b.create<arith::ConstantIndexOp>(loc, stage.inputWidth)));
          Value resultNext =
              b.create<BankSMatMulOp>(
                   loc, xState.getType(), onesLoaded, source.banks.front(),
                   xState, createI64ConstU(b, loc, matrixRs2(1, kTile, kTile)),
                   first, last, createI64Const(b, loc, 0))
                  .getWrBankOut();
          releaseTile(source);
          b.create<scf::YieldOp>(loc, resultNext);
          b.setInsertionPointAfter(xLoop);
          b.create<scf::YieldOp>(loc, xLoop.getResult(0));
          b.setInsertionPointAfter(yLoop);
          resultState = yLoop.getResult(0);
        }
        releaseBank(b, loc, onesLoaded);

        Value quantized = allocBank(b, loc, 1, 1);
        quantized = b.create<BankQuantI32ToI8Op>(
                         loc, quantized.getType(), resultState, scaleLoaded,
                         quantized, createI64Const(b, loc, 4),
                         createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                         b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                         b.getI64IntegerAttr(1), b.getBoolAttr(false))
                        .getOutBankOut();
        releaseBank(b, loc, resultState);
        releaseBank(b, loc, scaleLoaded);
        Value outputBase =
            b.create<arith::IndexCastOp>(loc, b.getI64Type(), localPanel);
        Value destinationNext =
            b.create<BankMaxPoolOp>(
                 loc, destinationState.getType(), quantized, destinationState,
                 createI64Const(b, loc, 1), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
                 createI64Const(b, loc, 0), outputBase,
                 createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                 b.getI64IntegerAttr(0))
                .getOutBankOut();
        releaseBank(b, loc, quantized);
        b.create<scf::YieldOp>(loc, destinationNext);
        b.setInsertionPointAfter(panelLoop);
        destination.banks.front() = panelLoop.getResult(0);
        b.create<memref::DeallocOp>(loc, onesPack);
        b.create<memref::DeallocOp>(loc, biasPack);
        b.create<memref::DeallocOp>(loc, scalePack);
        return success();
      }

      bool streamMaterializedInput =
          !stage.pool && !stage.depthwise && !externalInput &&
          materialized.contains(producer.lookup(stage.input));
      int64_t maxSide = std::min<int64_t>({4, height, width});
      int64_t lastInputBanks = -1;
      int64_t lastPanelsPerBank = -1;
      int64_t lastInputSide = -1;
      while (maxSide > 0) {
        int64_t inputSide = (maxSide - 1) * stage.stride + stage.kernel;
        int64_t inputPanelRows = inputSide * inputSide;
        int64_t inputPanels = stage.depthwise
                                  ? panelCount
                                  : (stage.inputChannels + kTile - 1) / kTile;
        int64_t panelsPerBank = inputPanelRows <= target.bankDepth
                                    ? target.bankDepth / inputPanelRows
                                    : 0;
        lastInputSide = inputSide;
        lastPanelsPerBank = panelsPerBank;
        int64_t inputBanks =
            panelsPerBank == 0 ? target.bankNum + 1
            : externalInput ? (inputPanels + panelsPerBank - 1) / panelsPerBank
            : streamMaterializedInput ? 1
            : stage.depthwise
                ? 1
                : (inputPanels + panelsPerBank - 1) / panelsPerBank;
        lastInputBanks = inputBanks;
        int64_t reservedBanks = externalInput ? 6 : 11;
        if (stage.activation == 2)
          reservedBanks += 2;
        if (inputBanks + static_cast<int64_t>(destination.banks.size()) +
                reservedBanks <=
            target.bankNum)
          break;
        --maxSide;
      }
      if (maxSide == 0) {
        InFlightDiagnostic diagnostic = stage.op->emitError(
            "resident stage tile exceeds physical bank capacity");
        diagnostic << " (request=" << height << "x" << width
                   << ", panels=" << panelCount
                   << ", destinationBanks=" << destination.banks.size()
                   << ", bankDepth=" << target.bankDepth
                   << ", bankNum=" << target.bankNum
                   << ", inputSide=" << lastInputSide
                   << ", inputBanks=" << lastInputBanks
                   << ", panelsPerBank=" << lastPanelsPerBank << ")";
        return failure();
      }
      int64_t side = std::min<int64_t>({maxSide, height, width});
      if (height != side || width != side) {
        if (failed(emitInto(stageIndex, y0, x0, side, side, firstPanel,
                            panelCount, destination, destinationBase,
                            destinationStride)))
          return failure();
        Value nextX = b.create<arith::AddIOp>(
            loc, x0, b.create<arith::ConstantIndexOp>(loc, side));
        if (width > side &&
            failed(emitInto(stageIndex, y0, nextX, side, width - side,
                            firstPanel, panelCount, destination,
                            destinationBase + side * (fp32Output ? 4 : 1),
                            destinationStride)))
          return failure();
        Value nextY = b.create<arith::AddIOp>(
            loc, y0, b.create<arith::ConstantIndexOp>(loc, side));
        if (height > side &&
            failed(emitInto(stageIndex, nextY, x0, height - side, width,
                            firstPanel, panelCount, destination,
                            destinationBase + side * destinationStride,
                            destinationStride)))
          return failure();
        return success();
      }

      int64_t inputSide = (side - 1) * stage.stride + stage.kernel;
      Value sourceY = b.create<arith::SubIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, y0, b.create<arith::ConstantIndexOp>(loc, stage.stride)),
          b.create<arith::ConstantIndexOp>(loc, stage.padding));
      Value sourceX = b.create<arith::SubIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, x0, b.create<arith::ConstantIndexOp>(loc, stage.stride)),
          b.create<arith::ConstantIndexOp>(loc, stage.padding));
      int64_t inputPanelCount = (stage.pool || stage.depthwise)
                                    ? panelCount
                                    : (stage.inputChannels + kTile - 1) / kTile;
      TileBanks source;
      if (externalInput && !stage.pool) {
        source.panelRows = inputSide * inputSide;
        // Keep the external input layout simple: one channel panel per bank.
        // This also makes the bank lifetime explicit when Cin is large.
        source.panelsPerBank = target.bankDepth / source.panelRows;
        source.panelCount = inputPanelCount;
        if (source.panelRows <= 0 || source.panelRows > target.bankDepth)
          return stage.op->emitError(
              "resident region input panel does not fit one bank");
      } else if (stage.pool) {
        // Pooling preserves channels. Stream input panels one bank at a time
        // instead of pinning the whole channel tile (e.g. 16 banks for 256 C).
        int64_t sourcePanelsPerBank =
            target.bankDepth / (inputSide * inputSide);
        if (sourcePanelsPerBank <= 0)
          return stage.op->emitError(
              "resident pool input panel does not fit bank");
        for (int64_t sourcePanelBegin = 0; sourcePanelBegin < inputPanelCount;
             sourcePanelBegin += sourcePanelsPerBank) {
          int64_t sourcePanelEnd = std::min<int64_t>(
              inputPanelCount, sourcePanelBegin + sourcePanelsPerBank);
          TileBanks sourceChunk = allocateTile(
              sourcePanelEnd - sourcePanelBegin, inputSide * inputSide, minI8);
          if (sourceChunk.banks.size() != 1)
            return stage.op->emitError(
                "resident pool input chunk must fit one bank");
          if (externalInput) {
            Value pack = b.create<memref::AllocOp>(
                loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
            b.create<linalg::FillOp>(loc, minI8, pack);
            auto yLoop = b.create<scf::ForOp>(
                loc, zero, b.create<arith::ConstantIndexOp>(loc, inputSide),
                one);
            b.setInsertionPointToStart(yLoop.getBody());
            Value localY = yLoop.getInductionVar();
            auto xLoop = b.create<scf::ForOp>(
                loc, zero, b.create<arith::ConstantIndexOp>(loc, inputSide),
                one);
            b.setInsertionPointToStart(xLoop.getBody());
            Value globalY = b.create<arith::AddIOp>(loc, sourceY, localY);
            Value globalX =
                b.create<arith::AddIOp>(loc, sourceX, xLoop.getInductionVar());
            Value yValid = b.create<arith::AndIOp>(
                loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, globalY,
                                        zero),
                b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, globalY,
                    b.create<arith::ConstantIndexOp>(loc, stage.inputHeight)));
            Value xValid = b.create<arith::AndIOp>(
                loc,
                b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, globalX,
                                        zero),
                b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, globalX,
                    b.create<arith::ConstantIndexOp>(loc, stage.inputWidth)));
            auto valid = b.create<scf::IfOp>(
                loc, b.create<arith::AndIOp>(loc, yValid, xValid), false);
            b.setInsertionPointToStart(&valid.getThenRegion().front());
            auto laneLoop = b.create<scf::ForOp>(
                loc,
                b.create<arith::ConstantIndexOp>(loc, sourcePanelBegin * kTile),
                b.create<arith::ConstantIndexOp>(loc, sourcePanelEnd * kTile),
                one);
            b.setInsertionPointToStart(laneLoop.getBody());
            Value channel = laneLoop.getInductionVar();
            Value inputChannel = b.create<arith::AddIOp>(
                loc, channel,
                b.create<arith::MulIOp>(
                    loc, firstPanel,
                    b.create<arith::ConstantIndexOp>(loc, kTile)));
            auto channelValid = b.create<scf::IfOp>(
                loc,
                b.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, inputChannel,
                    b.create<arith::ConstantIndexOp>(loc, stage.inputChannels)),
                false);
            b.setInsertionPointToStart(&channelValid.getThenRegion().front());
            Value value = b.create<memref::LoadOp>(
                loc, stage.input,
                ValueRange{zero, globalY, globalX, inputChannel});
            Value row = b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc,
                    b.create<arith::DivUIOp>(
                        loc,
                        b.create<arith::SubIOp>(
                            loc, channel,
                            b.create<arith::ConstantIndexOp>(
                                loc, sourcePanelBegin * kTile)),
                        b.create<arith::ConstantIndexOp>(loc, kTile)),
                    b.create<arith::ConstantIndexOp>(loc,
                                                     inputSide * inputSide)),
                b.create<arith::AddIOp>(
                    loc,
                    b.create<arith::MulIOp>(
                        loc, localY,
                        b.create<arith::ConstantIndexOp>(loc, inputSide)),
                    xLoop.getInductionVar()));
            b.create<memref::StoreOp>(
                loc, value, pack,
                ValueRange{row,
                           b.create<arith::RemUIOp>(
                               loc, channel,
                               b.create<arith::ConstantIndexOp>(loc, kTile))});
            b.setInsertionPointAfter(channelValid);
            b.setInsertionPointAfter(laneLoop);
            b.setInsertionPointAfter(valid);
            b.setInsertionPointAfter(xLoop);
            b.setInsertionPointAfter(yLoop);
            sourceChunk.banks.front() = mvinBank(
                b, loc, pack, sourceChunk.banks.front(), target.bankDepth);
            b.create<memref::DeallocOp>(loc, pack);
          } else if (failed(emitInto(producer.lookup(stage.input), sourceY,
                                     sourceX, inputSide, inputSide,
                                     b.create<arith::AddIOp>(
                                         loc, firstPanel,
                                         b.create<arith::ConstantIndexOp>(
                                             loc, sourcePanelBegin)),
                                     sourcePanelEnd - sourcePanelBegin,
                                     sourceChunk, 0, inputSide)))
            return failure();
          for (int64_t localPanel = sourcePanelBegin;
               localPanel < sourcePanelEnd; ++localPanel) {
            int64_t sourceSlot = localPanel - sourcePanelBegin;
            int64_t destinationBank = localPanel / destination.panelsPerBank;
            int64_t destinationSlot = localPanel % destination.panelsPerBank;
            destination.banks[destinationBank] =
                b.create<BankMaxPoolOp>(
                     loc, destination.banks[destinationBank].getType(),
                     sourceChunk.banks.front(),
                     destination.banks[destinationBank],
                     createI64Const(b, loc, side * side),
                     b.getI64IntegerAttr(inputSide), b.getI64IntegerAttr(side),
                     b.getI64IntegerAttr(stage.kernel),
                     b.getI64IntegerAttr(stage.stride), b.getI64IntegerAttr(0),
                     createI64Const(b, loc, sourceSlot * sourceChunk.panelRows),
                     createI64Const(b, loc,
                                    destinationSlot * destination.panelRows +
                                        destinationBase),
                     createI64Const(b, loc, destinationStride),
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                    .getOutBankOut();
          }
          releaseTile(sourceChunk);
        }
        maskInvalidOutput();
        return success();
      } else if (!externalInput && !stage.depthwise &&
                 !streamMaterializedInput) {
        source = allocateTile(inputPanelCount, inputSide * inputSide, zeroI8);
        if (failed(emitInto(producer.lookup(stage.input), sourceY, sourceX,
                            inputSide, inputSide, zero, inputPanelCount, source,
                            0, inputSide)))
          return failure();
      }

      if (stage.depthwise) {
        int64_t paddedK =
            (stage.kernel * stage.kernel + kTile - 1) / kTile * kTile;
        Value lutLoaded;
        if (stage.activation == 2) {
          Value lutBank = allocBank(b, loc, 1, 1);
          lutLoaded =
              mvinBank(b, loc, packedLuts.lookup(stage.op), lutBank, kTile);
        }
        SmallVector<Value> destinationStates(destination.banks.begin(),
                                             destination.banks.end());
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t destinationBank = localPanel / destination.panelsPerBank;
          int64_t destinationSlot = localPanel % destination.panelsPerBank;
          Value outputPanel = b.create<arith::AddIOp>(
              loc, firstPanel,
              b.create<arith::ConstantIndexOp>(loc, localPanel));
          TileBanks panelSource;
          if (externalInput) {
            auto loaded =
                loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                             stage.inputChannels, sourceY, sourceX, outputPanel,
                             1, inputSide, inputSide);
            if (failed(loaded))
              return stage.op->emitError("failed to load INT8 Depthwise input");
            panelSource = std::move(*loaded);
          } else {
            panelSource = allocateTile(1, inputSide * inputSide, zeroI8);
            if (panelSource.banks.size() != 1)
              return stage.op->emitError(
                  "Depthwise Conv input panel must fit one bank");
            if (failed(emitInto(producer.lookup(stage.input), sourceY, sourceX,
                                inputSide, inputSide, outputPanel, 1,
                                panelSource, 0, inputSide)))
              return failure();
          }
          SmallVector<OpFoldResult> parameterOffsets = {
              outputPanel, b.getIndexAttr(0), b.getIndexAttr(0)};
          SmallVector<OpFoldResult> parameterSizes = {
              b.getIndexAttr(1), b.getIndexAttr(4), b.getIndexAttr(4)};
          SmallVector<OpFoldResult> parameterStrides(3, b.getIndexAttr(1));
          Value biasSlice = b.create<memref::SubViewOp>(
              loc, packedBiases.lookup(stage.op), parameterOffsets,
              parameterSizes, parameterStrides);
          Value biasPack = b.create<memref::CollapseShapeOp>(
              loc, biasSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});
          Value scaleSlice = b.create<memref::SubViewOp>(
              loc, packedScales.lookup(stage.op), parameterOffsets,
              parameterSizes, parameterStrides);
          Value scalePack = b.create<memref::CollapseShapeOp>(
              loc, scaleSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});

          Value biasBank = allocBank(b, loc, 1, 1);
          Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
          Value biasState = b.create<BankSMatMulBiasOp>(
              loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
          releaseBank(b, loc, biasState);
          Value scaleBank = allocBank(b, loc, 1, 1);
          Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
          Value patchState = allocBank(b, loc, 1, 1);
          Value weightState = allocBank(b, loc, 1, 1);
          Value resultState = allocBank(b, loc, 1, 1);
          auto laneLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one,
              ValueRange{patchState, weightState, resultState});
          b.setInsertionPointToStart(laneLoop.getBody());
          Value lane = laneLoop.getInductionVar();
          ValueRange laneStates = laneLoop.getRegionIterArgs();
          Value laneI64 =
              b.create<arith::IndexCastOp>(loc, b.getI64Type(), lane);
          Value patchNext =
              b.create<BankIm2colOp>(
                   loc, laneStates[0].getType(), panelSource.banks.front(),
                   laneStates[0], createI64Const(b, loc, inputSide),
                   createI64Const(b, loc, stage.kernel),
                   createI64Const(b, loc, stage.stride),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   laneI64, b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                   b.getI64IntegerAttr(0), b.getI64IntegerAttr(side * side))
                  .getOutBankOut();

          Value weightSlice = b.create<memref::SubViewOp>(
              loc, packedDepthwiseWeights.lookup(stage.op),
              SmallVector<OpFoldResult>{outputPanel, lane, b.getIndexAttr(0),
                                        b.getIndexAttr(0)},
              SmallVector<OpFoldResult>{b.getIndexAttr(1), b.getIndexAttr(1),
                                        b.getIndexAttr(paddedK),
                                        b.getIndexAttr(kTile)},
              SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
          Value weightPack = b.create<memref::CollapseShapeOp>(
              loc, weightSlice,
              SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
          Value weightNext =
              mvinBank(b, loc, weightPack, laneStates[1], paddedK);
          Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                lane, zero);
          Value last = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, lane,
              b.create<arith::ConstantIndexOp>(loc, kTile - 1));
          Value resultNext =
              b.create<BankSMatMulOp>(
                   loc, laneStates[2].getType(), patchNext, weightNext,
                   laneStates[2],
                   createI64ConstU(b, loc, matrixRs2(kTile, kTile, paddedK)),
                   first, last, createI64Const(b, loc, 0))
                  .getWrBankOut();
          b.create<scf::YieldOp>(loc,
                                 ValueRange{patchNext, weightNext, resultNext});
          b.setInsertionPointAfter(laneLoop);
          patchState = laneLoop.getResult(0);
          weightState = laneLoop.getResult(1);
          resultState = laneLoop.getResult(2);
          releaseBank(b, loc, patchState);
          releaseBank(b, loc, weightState);

          Value outputBank = allocBank(b, loc, 1, 1);
          Value outputState;
          if (fp32Output && stage.activation == 2) {
            Value quantizedState =
                b.create<BankQuantI32ToI8Op>(
                     loc, outputBank.getType(), resultState, scaleLoaded,
                     outputBank, createI64Const(b, loc, side * side * 4),
                     createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                     b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                     b.getI64IntegerAttr(side), b.getBoolAttr(false))
                    .getOutBankOut();
            releaseBank(b, loc, resultState);
            releaseBank(b, loc, scaleLoaded);
            Value lutOutput = allocBank(b, loc, 1, 1);
            Value transformed = b.create<BankLutOp>(
                loc, lutOutput.getType(), quantizedState, lutLoaded, lutOutput,
                createI64Const(b, loc, side * side));
            releaseBank(b, loc, quantizedState);
            Value destinationState = destinationStates[destinationBank];
            int64_t pixels = side * side;
            for (int64_t pixel = 0; pixel < pixels; ++pixel) {
              Value outputBase =
                  createI64Const(b, loc,
                                 destinationSlot * destination.panelRows +
                                     destinationBase + pixel);
              destinationState =
                  b.create<BankMaxPoolOp>(
                       loc, destinationState.getType(), transformed,
                       destinationState, createI64Const(b, loc, 1),
                       b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(0), createI64Const(b, loc, pixel),
                       outputBase, createI64Const(b, loc, 1),
                       b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                      .getOutBankOut();
            }
            destinationStates[destinationBank] = destinationState;
            releaseBank(b, loc, transformed);
          } else if (fp32Output) {
            outputState =
                b.create<BankInt32ToFp32Op>(
                     loc, outputBank.getType(), resultState, scaleLoaded,
                     outputBank, createI64Const(b, loc, side * side * 4),
                     b.getBoolAttr(stage.activation == 1))
                    .getOutBankOut();
            releaseBank(b, loc, resultState);
            releaseBank(b, loc, scaleLoaded);
            Value destinationState = destinationStates[destinationBank];
            int64_t pixels = side * side;
            for (int64_t pixel = 0; pixel < pixels; ++pixel) {
              for (int64_t group = 0; group < 4; ++group) {
                int64_t sourceRow = pixel * 4 + group;
                Value outputBase =
                    createI64Const(b, loc,
                                   destinationSlot * destination.panelRows +
                                       destinationBase + pixel * 4 + group);
                destinationState =
                    b.create<BankMaxPoolOp>(
                         loc, destinationState.getType(), outputState,
                         destinationState, createI64Const(b, loc, 1),
                         b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                         b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                         b.getI64IntegerAttr(0),
                         createI64Const(b, loc, sourceRow), outputBase,
                         createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                         b.getI64IntegerAttr(0))
                        .getOutBankOut();
              }
            }
            destinationStates[destinationBank] = destinationState;
            releaseBank(b, loc, outputState);
          } else {
            Value quantizedState =
                b.create<BankQuantI32ToI8Op>(
                     loc, outputBank.getType(), resultState, scaleLoaded,
                     outputBank, createI64Const(b, loc, side * side * 4),
                     createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                     b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                     b.getI64IntegerAttr(side),
                     b.getBoolAttr(stage.activation == 1))
                    .getOutBankOut();
            releaseBank(b, loc, resultState);
            releaseBank(b, loc, scaleLoaded);
            Value transformed;
            outputState = quantizedState;
            if (stage.activation == 2) {
              Value lutOutput = allocBank(b, loc, 1, 1);
              transformed = b.create<BankLutOp>(
                  loc, lutOutput.getType(), quantizedState, lutLoaded,
                  lutOutput, createI64Const(b, loc, side * side));
              outputState = transformed;
            }
            Value outputBase = createI64Const(
                b, loc,
                destinationSlot * destination.panelRows + destinationBase);
            destinationStates[destinationBank] =
                b.create<BankMaxPoolOp>(
                     loc, destinationStates[destinationBank].getType(),
                     outputState, destinationStates[destinationBank],
                     createI64Const(b, loc, side * side),
                     b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                     b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                     outputBase, createI64Const(b, loc, destinationStride),
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                    .getOutBankOut();
            releaseBank(b, loc, quantizedState);
            if (transformed)
              releaseBank(b, loc, transformed);
          }
          releaseTile(panelSource);
        }
        if (lutLoaded)
          releaseBank(b, loc, lutLoaded);
        maskInvalidOutput();
        return success();
      }

      int64_t kernelElements = stage.kernel * stage.kernel;
      int64_t paddedK = (kernelElements + kTile - 1) / kTile * kTile;
      Value lutLoaded;
      if (stage.activation == 2) {
        Value lutBank = allocBank(b, loc, 1, stage.lutEntries == 4096 ? 4 : 1);
        lutLoaded =
            mvinBank(b, loc, packedLuts.lookup(stage.op), lutBank,
                     stage.lutEntries == 4096 ? target.bankDepth : kTile);
      }
      for (size_t destinationBank = 0;
           destinationBank < destination.banks.size(); ++destinationBank) {
        int64_t panelBegin = destinationBank * destination.panelsPerBank;
        int64_t panelEnd = std::min<int64_t>(
            panelCount, panelBegin + destination.panelsPerBank);
        auto outputPanelLoop = b.create<scf::ForOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, panelBegin),
            b.create<arith::ConstantIndexOp>(loc, panelEnd), one,
            ValueRange{destination.banks[destinationBank]});
        b.setInsertionPointToStart(outputPanelLoop.getBody());
        Value localPanel = outputPanelLoop.getInductionVar();
        Value destinationState = outputPanelLoop.getRegionIterArgs().front();
        Value outputPanel =
            b.create<arith::AddIOp>(loc, firstPanel, localPanel);
        SmallVector<OpFoldResult> parameterOffsets = {
            outputPanel, b.getIndexAttr(0), b.getIndexAttr(0)};
        SmallVector<OpFoldResult> parameterSizes = {
            b.getIndexAttr(1), b.getIndexAttr(4), b.getIndexAttr(4)};
        SmallVector<OpFoldResult> parameterStrides(3, b.getIndexAttr(1));
        Value biasSlice = b.create<memref::SubViewOp>(
            loc, packedBiases.lookup(stage.op), parameterOffsets,
            parameterSizes, parameterStrides);
        Value biasPack = b.create<memref::CollapseShapeOp>(
            loc, biasSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});
        Value scaleSlice = b.create<memref::SubViewOp>(
            loc, packedScales.lookup(stage.op), parameterOffsets,
            parameterSizes, parameterStrides);
        Value scalePack = b.create<memref::CollapseShapeOp>(
            loc, scaleSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});

        SmallVector<Value> states;
        auto ensureStates = [&]() {
          if (states.empty()) {
            states = {allocBank(b, loc, 1, 1), allocBank(b, loc, 1, 1),
                      allocBank(b, loc, 1, 1)};
            if (stage.weightChannelStride)
              states[0] = mvinBank(b, loc, zeroPack, states[0], kTile);
          }
        };
        auto accumulateSource = [&](Value sourceBank, Value sourcePanelBegin,
                                    Value sourcePanelEnd,
                                    int64_t sourcePanelRows) {
          Value tileSize = b.create<arith::ConstantIndexOp>(loc, kTile);
          Value channelBegin =
              b.create<arith::MulIOp>(loc, sourcePanelBegin, tileSize);
          Value channelEnd = b.create<arith::MinUIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, stage.inputChannels),
              b.create<arith::MulIOp>(loc, sourcePanelEnd, tileSize));
          int64_t channelStep = stage.weightChannelStride ? kTile : 1;
          auto channelLoop =
              b.create<scf::ForOp>(loc, channelBegin, channelEnd,
                                   stage.weightChannelStride ? tileSize : one,
                                   ValueRange{states[0], states[1], states[2]});
          b.setInsertionPointToStart(channelLoop.getBody());
          Value inputChannel = channelLoop.getInductionVar();
          ValueRange iterStates = channelLoop.getRegionIterArgs();
          Value sourceSlot = b.create<arith::SubIOp>(
              loc,
              b.create<arith::DivUIOp>(
                  loc, inputChannel,
                  b.create<arith::ConstantIndexOp>(loc, kTile)),
              sourcePanelBegin);
          Value inputBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::MulIOp>(
                  loc, sourceSlot,
                  b.create<arith::ConstantIndexOp>(loc, sourcePanelRows)));
          Value patchNext;
          Value weightNext;
          if (stage.weightChannelStride) {
            patchNext = b.create<BankMaxPoolOp>(
                loc, iterStates[0].getType(), sourceBank, iterStates[0],
                createI64Const(b, loc, side * side), b.getI64IntegerAttr(side),
                b.getI64IntegerAttr(side), b.getI64IntegerAttr(1),
                b.getI64IntegerAttr(1), b.getI64IntegerAttr(0), inputBase,
                createI64Const(b, loc, 0), createI64Const(b, loc, side),
                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
            weightNext = iterStates[1];
            for (int64_t half = 0; half < 2; ++half) {
              Value channel = b.create<arith::AddIOp>(
                  loc, inputChannel,
                  b.create<arith::ConstantIndexOp>(loc, half * 8));
              Value weightSlice = b.create<memref::SubViewOp>(
                  loc, stage.weight,
                  SmallVector<OpFoldResult>{outputPanel, channel,
                                            b.getIndexAttr(0),
                                            b.getIndexAttr(0)},
                  SmallVector<OpFoldResult>{
                      b.getIndexAttr(1), b.getIndexAttr(8), b.getIndexAttr(1),
                      b.getIndexAttr(kTile)},
                  SmallVector<OpFoldResult>(4, b.getIndexAttr(1)));
              weightNext = b.create<BankMvin2dOp>(
                  loc, weightNext.getType(), weightSlice, weightNext,
                  createI64Const(b, loc, 1), stage.weightChannelStride,
                  createI64Const(b, loc, 8), createI64Const(b, loc, half * 8),
                  createI64Const(b, loc, 8), createI64Const(b, loc, kTile));
            }
          } else {
            Value inputLane = b.create<arith::IndexCastOp>(
                loc, b.getI64Type(),
                b.create<arith::RemUIOp>(
                    loc, inputChannel,
                    b.create<arith::ConstantIndexOp>(loc, kTile)));
            patchNext =
                b.create<BankIm2colOp>(
                     loc, iterStates[0].getType(), sourceBank, iterStates[0],
                     createI64Const(b, loc, inputSide),
                     createI64Const(b, loc, stage.kernel),
                     createI64Const(b, loc, stage.stride),
                     createI64Const(b, loc, 0), inputBase, inputLane,
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(side * side))
                    .getOutBankOut();
            SmallVector<OpFoldResult> weightOffsets = {
                outputPanel, inputChannel, b.getIndexAttr(0),
                b.getIndexAttr(0)};
            SmallVector<OpFoldResult> weightSizes = {
                b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(paddedK),
                b.getIndexAttr(kTile)};
            SmallVector<OpFoldResult> weightStrides(4, b.getIndexAttr(1));
            Value weightSlice = b.create<memref::SubViewOp>(
                loc, stage.weight, weightOffsets, weightSizes, weightStrides);
            Value weightPack = b.create<memref::CollapseShapeOp>(
                loc, weightSlice,
                SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
            weightNext = mvinBank(b, loc, weightPack, iterStates[1], paddedK);
          }
          Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                inputChannel, zero);
          Value last = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, inputChannel,
              b.create<arith::ConstantIndexOp>(loc, stage.inputChannels -
                                                        channelStep));
          Value resultNext =
              b.create<BankSMatMulOp>(
                   loc, iterStates[2].getType(), patchNext, weightNext,
                   iterStates[2],
                   createI64ConstU(
                       b, loc,
                       matrixRs2(side == 1 && paddedK == kTile ? 1 : kTile,
                                 kTile, paddedK)),
                   first, last, createI64Const(b, loc, 0))
                  .getWrBankOut();
          b.create<scf::YieldOp>(loc,
                                 ValueRange{patchNext, weightNext, resultNext});
          b.setInsertionPointAfter(channelLoop);
          states.assign(channelLoop.getResults().begin(),
                        channelLoop.getResults().end());
        };
        Value biasBank = allocBank(b, loc, 1, 1);
        Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
        Value biasState = b.create<BankSMatMulBiasOp>(
            loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
        releaseBank(b, loc, biasState);

        if (externalInput) {
          ensureStates();
          auto inputPanelLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, inputPanelCount),
              one, ValueRange{states[0], states[1], states[2]});
          b.setInsertionPointToStart(inputPanelLoop.getBody());
          Value panel = inputPanelLoop.getInductionVar();
          states.assign(inputPanelLoop.getRegionIterArgs().begin(),
                        inputPanelLoop.getRegionIterArgs().end());
          FailureOr<TileBanks> loaded =
              loadInt8Tile(stage.input, stage.inputHeight, stage.inputWidth,
                           stage.inputChannels, sourceY, sourceX, panel, 1,
                           inputSide, inputSide);
          if (failed(loaded))
            return stage.op->emitError("failed to load external Conv input");
          accumulateSource(loaded->banks.front(), panel,
                           b.create<arith::AddIOp>(loc, panel, one),
                           loaded->panelRows);
          releaseTile(*loaded);
          b.create<scf::YieldOp>(loc,
                                 ValueRange{states[0], states[1], states[2]});
          b.setInsertionPointAfter(inputPanelLoop);
          states.assign(inputPanelLoop.getResults().begin(),
                        inputPanelLoop.getResults().end());
        } else if (streamMaterializedInput) {
          ensureStates();
          int64_t streamedPanelsPerBank =
              target.bankDepth / (inputSide * inputSide);
          if (streamedPanelsPerBank <= 0)
            return stage.op->emitError(
                "streamed Conv input panel does not fit one bank");
          for (int64_t inputPanel = 0; inputPanel < inputPanelCount;
               inputPanel += streamedPanelsPerBank) {
            int64_t panelCountInBank = std::min<int64_t>(
                streamedPanelsPerBank, inputPanelCount - inputPanel);
            TileBanks streamedSource =
                allocateTile(panelCountInBank, inputSide * inputSide, zeroI8);
            if (streamedSource.banks.size() != 1)
              return stage.op->emitError(
                  "streamed Conv input panel group must fit one bank");
            if (failed(
                    emitInto(producer.lookup(stage.input), sourceY, sourceX,
                             inputSide, inputSide,
                             b.create<arith::ConstantIndexOp>(loc, inputPanel),
                             panelCountInBank, streamedSource, 0, inputSide)))
              return failure();
            accumulateSource(streamedSource.banks.front(),
                             b.create<arith::ConstantIndexOp>(loc, inputPanel),
                             b.create<arith::ConstantIndexOp>(
                                 loc, inputPanel + panelCountInBank),
                             streamedSource.panelRows);
            releaseTile(streamedSource);
          }
        } else {
          ensureStates();
          for (size_t sourceBank = 0; sourceBank < source.banks.size();
               ++sourceBank) {
            int64_t sourcePanelBegin = sourceBank * source.panelsPerBank;
            int64_t sourcePanelEnd = std::min<int64_t>(
                inputPanelCount, sourcePanelBegin + source.panelsPerBank);
            accumulateSource(
                source.banks[sourceBank],
                b.create<arith::ConstantIndexOp>(loc, sourcePanelBegin),
                b.create<arith::ConstantIndexOp>(loc, sourcePanelEnd),
                source.panelRows);
          }
        }
        releaseBank(b, loc, states[0]);
        releaseBank(b, loc, states[1]);
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value destinationSlot = b.create<arith::SubIOp>(
            loc, localPanel, b.create<arith::ConstantIndexOp>(loc, panelBegin));
        Value destinationOffset = b.create<arith::MulIOp>(
            loc, destinationSlot,
            b.create<arith::ConstantIndexOp>(loc, destination.panelRows));
        Value outputBase = b.create<arith::IndexCastOp>(
            loc, b.getI64Type(),
            b.create<arith::AddIOp>(
                loc, destinationOffset,
                b.create<arith::ConstantIndexOp>(loc, destinationBase)));
        Value outputBank = allocBank(b, loc, 1, 1);
        Value outputState;
        Value destinationNext = destinationState;
        if (fp32Output && stage.activation == 2) {
          Value quantizedState =
              b.create<BankQuantI32ToI8Op>(
                   loc, outputBank.getType(), states[2], scaleLoaded,
                   outputBank, createI64Const(b, loc, side * side * 4),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                   b.getI64IntegerAttr(side), b.getBoolAttr(false))
                  .getOutBankOut();
          releaseBank(b, loc, states[2]);
          releaseBank(b, loc, scaleLoaded);
          Value lutOutput = allocBank(b, loc, 1, 1);
          Value transformed = b.create<BankLutOp>(
              loc, lutOutput.getType(), quantizedState, lutLoaded, lutOutput,
              createI64Const(b, loc, side * side));
          releaseBank(b, loc, quantizedState);
          int64_t pixels = side * side;
          for (int64_t pixel = 0; pixel < pixels; ++pixel) {
            Value sourceBase = createI64Const(b, loc, pixel);
            Value outputBase = b.create<arith::AddIOp>(
                loc,
                b.create<arith::IndexCastOp>(
                    loc, b.getI64Type(),
                    b.create<arith::AddIOp>(
                        loc,
                        b.create<arith::MulIOp>(
                            loc, destinationSlot,
                            b.create<arith::ConstantIndexOp>(
                                loc, destination.panelRows)),
                        b.create<arith::ConstantIndexOp>(loc,
                                                         destinationBase))),
                createI64Const(b, loc, pixel));
            destinationNext =
                b.create<BankMaxPoolOp>(
                     loc, destinationNext.getType(), transformed,
                     destinationNext, createI64Const(b, loc, 1),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                     b.getI64IntegerAttr(0), sourceBase, outputBase,
                     createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                     b.getI64IntegerAttr(0))
                    .getOutBankOut();
          }
          releaseBank(b, loc, transformed);
        } else if (fp32Output) {
          outputState =
              b.create<BankInt32ToFp32Op>(
                   loc, outputBank.getType(), states[2], scaleLoaded,
                   outputBank, createI64Const(b, loc, side * side * 4),
                   b.getBoolAttr(stage.activation == 1))
                  .getOutBankOut();
          releaseBank(b, loc, states[2]);
          releaseBank(b, loc, scaleLoaded);
          int64_t pixels = side * side;
          for (int64_t pixel = 0; pixel < pixels; ++pixel) {
            for (int64_t group = 0; group < 4; ++group) {
              Value sourceBase = createI64Const(b, loc, pixel * 4 + group);
              Value outputBase = createI64Const(b, loc, pixel * 4 + group);
              outputBase = b.create<arith::AddIOp>(
                  loc, outputBase,
                  b.create<arith::IndexCastOp>(
                      loc, b.getI64Type(),
                      b.create<arith::AddIOp>(
                          loc,
                          b.create<arith::MulIOp>(
                              loc, destinationSlot,
                              b.create<arith::ConstantIndexOp>(
                                  loc, destination.panelRows)),
                          b.create<arith::ConstantIndexOp>(loc,
                                                           destinationBase))));
              destinationNext =
                  b.create<BankMaxPoolOp>(
                       loc, destinationNext.getType(), outputState,
                       destinationNext, createI64Const(b, loc, 1),
                       b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(0), sourceBase, outputBase,
                       createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                       b.getI64IntegerAttr(0))
                      .getOutBankOut();
            }
          }
          releaseBank(b, loc, outputState);
        } else {
          Value quantizedState =
              b.create<BankQuantI32ToI8Op>(
                   loc, outputBank.getType(), states[2], scaleLoaded,
                   outputBank, createI64Const(b, loc, side * side * 4),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                   b.getI64IntegerAttr(side),
                   b.getBoolAttr(stage.activation == 1))
                  .getOutBankOut();
          releaseBank(b, loc, states[2]);
          releaseBank(b, loc, scaleLoaded);
          Value transformed;
          outputState = quantizedState;
          if (stage.activation == 2) {
            Value lutOutput = allocBank(b, loc, 1, 1);
            transformed = b.create<BankLutOp>(
                loc, lutOutput.getType(), quantizedState, lutLoaded, lutOutput,
                createI64Const(b, loc, side * side));
            outputState = transformed;
          }
          destinationNext =
              b.create<BankMaxPoolOp>(
                   loc, destinationState.getType(), outputState,
                   destinationState, createI64Const(b, loc, side * side),
                   b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                   outputBase, createI64Const(b, loc, destinationStride),
                   b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                  .getOutBankOut();
          releaseBank(b, loc, quantizedState);
          if (transformed)
            releaseBank(b, loc, transformed);
        }
        b.create<scf::YieldOp>(loc, destinationNext);
        b.setInsertionPointAfter(outputPanelLoop);
        destination.banks[destinationBank] = outputPanelLoop.getResult(0);
      }
      if (lutLoaded)
        releaseBank(b, loc, lutLoaded);
      releaseTile(source);
      maskInvalidOutput();
      return success();
    };

    auto materializeStage = [&](int64_t stageIndex) -> LogicalResult {
      Stage &stage = stages[stageIndex];
      const bool fp32Output = stage.finalOutput && !stage.pool;
      const int64_t outputStorageFactor = fp32Output ? 4 : 1;
      int64_t side =
          std::min<int64_t>((stage.add || stage.average) ? 1 : 2,
                            std::min(stage.outputHeight, stage.outputWidth));
      int64_t panelCount = (stage.outputChannels + kTile - 1) / kTile;
      auto yLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.outputHeight),
          b.create<arith::ConstantIndexOp>(loc, side));
      b.setInsertionPointToStart(yLoop.getBody());
      Value y = yLoop.getInductionVar();
      auto xLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.outputWidth),
          b.create<arith::ConstantIndexOp>(loc, side));
      b.setInsertionPointToStart(xLoop.getBody());
      Value x = xLoop.getInductionVar();
      auto emitOutput = [&](Value firstPanel,
                            int64_t requestedPanels) -> LogicalResult {
        const bool fp32LutOutput = fp32Output && stage.activation == 2;
        if (fp32LutOutput && !packedLutOutputScales.lookup(stage.op))
          return stage.op->emitError(
              "missing LUT output scales for FP32 final output");
        TileBanks output = allocateTile(
            requestedPanels, side * side * outputStorageFactor, zeroI8);
        if (output.banks.size() != 1) {
          return failure();
        }
        if (failed(emitInto(stageIndex, y, x, side, side, firstPanel,
                            requestedPanels, output, 0,
                            side * outputStorageFactor))) {
          return failure();
        }
        Type outputElementType =
            fp32LutOutput
                ? Type(b.getI8Type())
                : (fp32Output ? Type(b.getF32Type()) : Type(b.getI8Type()));
        int64_t packedRows = fp32LutOutput
                                 ? target.bankDepth
                                 : target.bankDepth / outputStorageFactor;
        Value pack = b.create<memref::AllocOp>(
            loc, MemRefType::get({packedRows, kTile}, outputElementType));
        mvoutBank(b, loc, pack, output.banks.front(), target.bankDepth);
        b.create<FenceOp>(loc);

        auto panelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, requestedPanels),
            one);
        b.setInsertionPointToStart(panelLoop.getBody());
        Value panelInRequest = panelLoop.getInductionVar();
        Value panel = b.create<arith::AddIOp>(loc, firstPanel, panelInRequest);
        auto localYLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, side), one);
        b.setInsertionPointToStart(localYLoop.getBody());
        Value localY = localYLoop.getInductionVar();
        auto localXLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, side), one);
        b.setInsertionPointToStart(localXLoop.getBody());
        Value localX = localXLoop.getInductionVar();
        Value globalY = b.create<arith::AddIOp>(loc, y, localY);
        Value globalX = b.create<arith::AddIOp>(loc, x, localX);
        Value yValid = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, globalY,
            b.create<arith::ConstantIndexOp>(loc, stage.outputHeight));
        Value xValid = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, globalX,
            b.create<arith::ConstantIndexOp>(loc, stage.outputWidth));
        auto valid = b.create<scf::IfOp>(
            loc, b.create<arith::AndIOp>(loc, yValid, xValid), false);
        b.setInsertionPointToStart(&valid.getThenRegion().front());
        auto laneLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
        b.setInsertionPointToStart(laneLoop.getBody());
        Value lane = laneLoop.getInductionVar();
        Value channel = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, panel, b.create<arith::ConstantIndexOp>(loc, kTile)),
            lane);
        auto channelValid = b.create<scf::IfOp>(
            loc,
            b.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, channel,
                b.create<arith::ConstantIndexOp>(loc, stage.outputChannels)),
            false);
        b.setInsertionPointToStart(&channelValid.getThenRegion().front());
        Value row = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, panelInRequest,
                b.create<arith::ConstantIndexOp>(loc, side * side)),
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, localY, b.create<arith::ConstantIndexOp>(loc, side)),
                localX));
        Value packedLane = lane;
        if (fp32Output && !fp32LutOutput) {
          Value group = b.create<arith::DivUIOp>(
              loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
          Value laneInGroup = b.create<arith::RemUIOp>(
              loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
          packedLane = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(loc, group,
                                      b.create<arith::ConstantIndexOp>(loc, 4)),
              laneInGroup);
        }
        Value value =
            b.create<memref::LoadOp>(loc, pack, ValueRange{row, packedLane});
        if (fp32LutOutput) {
          Value valueF32 =
              b.create<arith::SIToFPOp>(loc, b.getF32Type(), value);
          Value outputScale = b.create<memref::LoadOp>(
              loc, packedLutOutputScales.lookup(stage.op),
              ValueRange{
                  panel,
                  b.create<arith::DivUIOp>(
                      loc, lane, b.create<arith::ConstantIndexOp>(loc, 4)),
                  b.create<arith::RemUIOp>(
                      loc, lane, b.create<arith::ConstantIndexOp>(loc, 4))});
          value = b.create<arith::MulFOp>(loc, valueF32, outputScale);
        }
        if (stage.finalOutput)
          b.create<memref::StoreOp>(
              loc, value, stage.output,
              ValueRange{zero, channel, globalY, globalX});
        else
          b.create<memref::StoreOp>(
              loc, value, stage.output,
              ValueRange{zero, globalY, globalX, channel});
        b.setInsertionPointAfter(channelValid);
        b.setInsertionPointAfter(laneLoop);
        b.setInsertionPointAfter(valid);
        b.setInsertionPointAfter(localYLoop);
        b.setInsertionPointAfter(panelLoop);
        b.create<memref::DeallocOp>(loc, pack);
        releaseTile(output);
        return success();
      };

      int64_t panelsPerBank =
          target.bankDepth / (side * side * outputStorageFactor);
      if (panelsPerBank <= 0)
        return stage.op->emitError("resident output tile does not fit bank");
      if (stage.depthwise) {
        auto depthwisePanelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, panelCount), one);
        b.setInsertionPointToStart(depthwisePanelLoop.getBody());
        if (failed(emitOutput(depthwisePanelLoop.getInductionVar(), 1)))
          return failure();
        b.setInsertionPointAfter(depthwisePanelLoop);
      } else {
        for (int64_t panelBegin = 0; panelBegin < panelCount;
             panelBegin += panelsPerBank) {
          int64_t chunkCount =
              std::min<int64_t>(panelCount - panelBegin, panelsPerBank);
          if (failed(
                  emitOutput(b.create<arith::ConstantIndexOp>(loc, panelBegin),
                             chunkCount)))
            return failure();
        }
      }
      b.setInsertionPointAfter(xLoop);
      b.setInsertionPointAfter(yLoop);
      // A zero limit is the debug mode for the normal resident path: do not
      // force any stage to materialize, but trace every stage that the normal
      // scheduler materializes at a pool/add/final boundary.
      if (traceMegaStages && stageIndex >= traceMegaStageStart &&
          (traceMegaStageLimit == 0 || traceMegaStageLimit < 0 ||
           stageIndex < traceMegaStageLimit)) {
        auto id = b.getI64IntegerAttr(stageIndex);
        auto trace = b.create<::buddy::trace::EndOp>(
            loc, stage.output.getType(), stage.output, id,
            b.getStringAttr("mega-stage"));
        trace->setAttr("id_path", b.getArrayAttr({id}));
        trace->setAttr("buckyball.stage_trace", b.getUnitAttr());
      }
      materialized.insert(stageIndex);
      return success();
    };

    int64_t traceLimit = 0;
    if (traceMegaStages) {
      if (traceMegaStageStart < 0 ||
          traceMegaStageStart > static_cast<int64_t>(stages.size()))
        return kernel.emitError("trace-mega-stage-start is out of range");
      traceLimit = traceMegaStageLimit < 0
                       ? static_cast<int64_t>(stages.size())
                       : std::min<int64_t>(traceMegaStageLimit,
                                           static_cast<int64_t>(stages.size()));
      if (traceLimit < 0)
        return kernel.emitError("trace-mega-stage-limit must be non-negative");
      for (int64_t stageIndex = 0; stageIndex < traceLimit; ++stageIndex) {
        Stage &stage = stages[stageIndex];
        if (stage.multiply) {
          int64_t gateStage = producer.lookup(stage.rhs);
          int64_t gatePanels =
              (stages[gateStage].outputChannels + kTile - 1) / kTile;
          TileBanks gate = allocateTile(gatePanels, 1, zeroI8);
          if (gate.banks.size() != 1 ||
              failed(emitInto(gateStage, zero, zero, 1, 1, zero, gatePanels,
                              gate, 0, 1)))
            return stage.op->emitError(
                "INT8 Mul gate must fit one complete bank");
          gateCaches.try_emplace(gateStage, std::move(gate));
        }
        if (failed(materializeStage(stageIndex)))
          return failure();
        if (stage.multiply) {
          for (auto &entry : gateCaches)
            releaseTile(entry.second);
          gateCaches.clear();
        }
      }
    }
    for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
      if (static_cast<int64_t>(stageIndex) < traceLimit)
        continue;
      if (stage.multiply) {
        if (producer.contains(stage.input)) {
          int64_t inputStage = producer.lookup(stage.input);
          if (!materialized.contains(inputStage) &&
              failed(materializeStage(inputStage)))
            return failure();
        }
        if (!producer.contains(stage.rhs))
          return stage.op->emitError(
              "INT8 Mul gate must have a region producer");
        int64_t gateStage = producer.lookup(stage.rhs);
        if (!gateCaches.contains(gateStage)) {
          int64_t gatePanels =
              (stages[gateStage].outputChannels + kTile - 1) / kTile;
          TileBanks gate = allocateTile(gatePanels, 1, zeroI8);
          if (gate.banks.size() != 1 ||
              failed(emitInto(gateStage, zero, zero, 1, 1, zero, gatePanels,
                              gate, 0, 1)))
            return stage.op->emitError(
                "INT8 Mul gate must fit one complete bank");
          gateCaches.try_emplace(gateStage, std::move(gate));
        }
      }
      if ((stage.pool || stage.add) && !materialized.contains(stageIndex)) {
        if (failed(materializeStage(stageIndex)))
          return failure();
        for (auto &entry : gateCaches)
          releaseTile(entry.second);
        gateCaches.clear();
      }
    }

    int64_t finalStage = stages.size() - 1;
    if (!materialized.contains(finalStage) &&
        failed(materializeStage(finalStage)))
      return failure();
    for (auto &entry : gateCaches)
      releaseTile(entry.second);
    releaseBank(b, loc, zeroBank);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.eraseOp(kernel);
    return success();
  }

private:
  bool traceMegaStages;
  int64_t traceMegaStageStart;
  int64_t traceMegaStageLimit;
};

} // namespace

namespace mlir::buddy {
void populatePebbleResidentConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns, bool traceMegaStages,
    int64_t traceMegaStageStart, int64_t traceMegaStageLimit) {
  patterns.add<ResidentConvRegionPattern>(patterns.getContext(),
                                          traceMegaStages, traceMegaStageStart,
                                          traceMegaStageLimit);
}
} // namespace mlir::buddy
