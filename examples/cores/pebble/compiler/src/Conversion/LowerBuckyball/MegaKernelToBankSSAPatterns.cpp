//===- MegaKernelToBankSSAPatterns.cpp - MegaKernel to bank SSA ----------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;
constexpr int64_t kInt32Rows = 64;

class LinearConvPoolMegaKernelPattern : public OpRewritePattern<MegaKernelOp> {
public:
  LinearConvPoolMegaKernelPattern(MLIRContext *context)
      : OpRewritePattern<MegaKernelOp>(context, 3) {}

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    if (body.without_terminator().empty() ||
        !isa<MegaConv2dOp, MegaConv2dDepthwiseOp>(body.front()))
      return failure();

    struct Stage {
      Operation *op;
      Value input;
      Value rhs;
      Value output;
      Value weight;
      Value bias;
      Value scale;
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
      bool depthwise;
      bool pool;
      bool add;
      bool globalAvg;
    };

    SmallVector<Stage> stages;
    for (auto [index, operation] : llvm::enumerate(body.without_terminator())) {
      bool last =
          index + 1 ==
          static_cast<size_t>(std::distance(body.without_terminator().begin(),
                                            body.without_terminator().end()));
      if (auto pool = dyn_cast<MegaMaxPool2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(pool.getInput().getType());
        auto output = dyn_cast<MemRefType>(pool.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || pool.getFinalOutput() != last ||
            input.getShape()[0] != 1 || output.getShape()[0] != 1 ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || pool.getKernel() <= 0 ||
            pool.getKernel() > 8 || pool.getStride() <= 0 ||
            pool.getPadding() < 0)
          return pool.emitError("unsupported MaxPool stage in Conv MegaKernel");
        int64_t inH = input.getShape()[1];
        int64_t inW = input.getShape()[2];
        int64_t channels = input.getShape()[3];
        int64_t outH =
            pool.getFinalOutput() ? output.getShape()[2] : output.getShape()[1];
        int64_t outW =
            pool.getFinalOutput() ? output.getShape()[3] : output.getShape()[2];
        int64_t outC =
            pool.getFinalOutput() ? output.getShape()[1] : output.getShape()[3];
        if (channels <= 0 || outC != channels ||
            (inH + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                outH ||
            (inW + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                outW)
          return pool.emitError("MaxPool shape is inconsistent");
        stages.push_back({&operation,
                          pool.getInput(),
                          {},
                          pool.getOutput(),
                          {},
                          {},
                          {},
                          inH,
                          inW,
                          channels,
                          outH,
                          outW,
                          channels,
                          static_cast<int64_t>(pool.getKernel()),
                          static_cast<int64_t>(pool.getStride()),
                          static_cast<int64_t>(pool.getPadding()),
                          0,
                          0.0f,
                          0.0f,
                          0.0f,
                          false,
                          true,
                          false,
                          false});
        continue;
      }

      if (auto add = dyn_cast<MegaInt8AddOp>(operation)) {
        auto lhs = dyn_cast<MemRefType>(add.getLhs().getType());
        auto rhs = dyn_cast<MemRefType>(add.getRhs().getType());
        auto output = dyn_cast<MemRefType>(add.getOutput().getType());
        if (!lhs || !rhs || !output || !lhs.hasStaticShape() ||
            !rhs.hasStaticShape() || !output.hasStaticShape() ||
            lhs.getRank() != 4 || rhs != lhs || output != lhs ||
            lhs.getShape()[0] != 1 || !lhs.getElementType().isInteger(8) ||
            add.getActivation() < 0 || add.getActivation() > 1 ||
            add.getLhsScale().convertToFloat() <= 0.0f ||
            add.getRhsScale().convertToFloat() <= 0.0f ||
            add.getOutputScale().convertToFloat() <= 0.0f)
          return add.emitError("unsupported INT8 Add stage in Conv MegaKernel");
        auto shape = lhs.getShape();
        stages.push_back({&operation,
                          add.getLhs(),
                          add.getRhs(),
                          add.getOutput(),
                          {},
                          {},
                          {},
                          shape[1],
                          shape[2],
                          shape[3],
                          shape[1],
                          shape[2],
                          shape[3],
                          1,
                          1,
                          0,
                          static_cast<int64_t>(add.getActivation()),
                          add.getLhsScale().convertToFloat(),
                          add.getRhsScale().convertToFloat(),
                          add.getOutputScale().convertToFloat(),
                          false,
                          false,
                          true,
                          false});
        continue;
      }

      if (auto average = dyn_cast<MegaGlobalAvgPoolOp>(operation)) {
        auto input = dyn_cast<MemRefType>(average.getInput().getType());
        auto output = dyn_cast<MemRefType>(average.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape() !=
                ArrayRef<int64_t>({1, 1, 1, input.getShape()[3]}) ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) ||
            average.getInputScale().convertToFloat() <= 0.0f ||
            average.getOutputScale().convertToFloat() <= 0.0f)
          return average.emitError(
              "unsupported GlobalAvgPool stage in Conv MegaKernel");
        auto shape = input.getShape();
        stages.push_back({&operation, average.getInput(),
                          {},         average.getOutput(),
                          {},         {},
                          {},         shape[1],
                          shape[2],   shape[3],
                          1,          1,
                          shape[3],   1,
                          1,          0,
                          0,          average.getInputScale().convertToFloat(),
                          0.0f,       average.getOutputScale().convertToFloat(),
                          false,      false,
                          false,      true});
        continue;
      }

      auto conv = dyn_cast<MegaConv2dOp>(operation);
      auto depthwise = dyn_cast<MegaConv2dDepthwiseOp>(operation);
      if (!conv && !depthwise)
        return operation.emitError(
            "Conv MegaKernel currently accepts only Conv2D, depthwise Conv2D, "
            "MaxPool2D, GlobalAvgPool, and INT8 Add stages");
      Value inputValue = conv ? conv.getInput() : depthwise.getInput();
      Value weightValue = conv ? conv.getWeight() : depthwise.getWeight();
      Value biasValue = conv ? conv.getBias() : depthwise.getBias();
      Value scaleValue = conv ? conv.getScale() : depthwise.getScale();
      Value lutValue = conv ? conv.getLut() : depthwise.getLut();
      Value outputValue = conv ? conv.getOutput() : depthwise.getOutput();
      int64_t stride = conv ? conv.getStride() : depthwise.getStride();
      int64_t padLow = conv ? conv.getPadLow() : depthwise.getPadLow();
      int64_t padHigh = conv ? conv.getPadHigh() : depthwise.getPadHigh();
      int64_t activation =
          conv ? conv.getActivation() : depthwise.getActivation();
      auto input = dyn_cast<MemRefType>(inputValue.getType());
      auto weight = dyn_cast<MemRefType>(weightValue.getType());
      auto bias = dyn_cast<MemRefType>(biasValue.getType());
      auto scale = dyn_cast<MemRefType>(scaleValue.getType());
      auto lut = dyn_cast<MemRefType>(lutValue.getType());
      auto output = dyn_cast<MemRefType>(outputValue.getType());
      if (!input || !weight || !bias || !scale || !lut || !output ||
          !input.hasStaticShape() || !weight.hasStaticShape() ||
          !bias.hasStaticShape() || !scale.hasStaticShape() ||
          !lut.hasStaticShape() || !output.hasStaticShape() ||
          input.getRank() != 4 || weight.getRank() != 4 ||
          output.getRank() != 4 || input.getShape()[0] != 1 ||
          output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
          !weight.getElementType().isInteger(8) ||
          !bias.getElementType().isInteger(32) ||
          !scale.getElementType().isF32() ||
          !output.getElementType().isInteger(8) || activation < 0 ||
          activation > 1 || stride <= 0 || padLow < 0 || padLow != padHigh)
        return operation.emitError("unsupported Conv stage in Conv MegaKernel");
      int64_t inH = input.getShape()[1];
      int64_t inW = input.getShape()[2];
      int64_t inC = input.getShape()[3];
      int64_t kernelSize = conv ? conv.getKernel() : depthwise.getKernel();
      int64_t outH = output.getShape()[1];
      int64_t outW = output.getShape()[2];
      int64_t outC = output.getShape()[3];
      bool isDepthwise = static_cast<bool>(depthwise);
      int64_t paddedKernel =
          (kernelSize * kernelSize + kTile - 1) / kTile * kTile;
      bool weightShapeMatches =
          isDepthwise
              ? weight.getShape() ==
                        ArrayRef<int64_t>({kernelSize, kernelSize, inC, 1}) &&
                    outC == inC
              : weight.getShape() ==
                    ArrayRef<int64_t>(
                        {(outC + kTile - 1) / kTile, inC, paddedKernel, kTile});
      if (inH <= 0 || inW <= 0 || inC <= 0 || outH <= 0 || outW <= 0 ||
          outC <= 0 || kernelSize <= 0 || kernelSize > 7 ||
          !weightShapeMatches || bias.getShape() != ArrayRef<int64_t>({outC}) ||
          scale.getShape() != ArrayRef<int64_t>({outC}) || lut.getRank() != 1 ||
          lut.getShape()[0] != 1 ||
          (inH + 2 * padLow - kernelSize) / stride + 1 != outH ||
          (inW + 2 * padLow - kernelSize) / stride + 1 != outW)
        return operation.emitError(
            "Conv shape or quantization data is inconsistent");
      stages.push_back({&operation,
                        inputValue,
                        {},
                        outputValue,
                        weightValue,
                        biasValue,
                        scaleValue,
                        inH,
                        inW,
                        inC,
                        outH,
                        outW,
                        outC,
                        kernelSize,
                        stride,
                        padLow,
                        activation,
                        0.0f,
                        0.0f,
                        conv ? conv.getOutputScale().convertToFloat()
                             : depthwise.getOutputScale().convertToFloat(),
                        isDepthwise,
                        false,
                        false,
                        false});
    }
    if (stages.empty() || stages.back().output != kernel.getOutput())
      return kernel.emitError(
          "Conv MegaKernel final stage must produce the kernel output");

    DenseMap<Value, int64_t> producer;
    for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
      if (producer.contains(stage.output))
        return stage.op->emitError(
            "Conv MegaKernel has multiple producers for one tensor");
      for (Value input : {stage.input, stage.rhs}) {
        if (input && input != kernel.getInput() && !producer.contains(input))
          return stage.op->emitError(
              "Conv MegaKernel input is not produced by an earlier stage");
      }
      producer[stage.output] = stageIndex;
    }

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth != 64 ||
        target.bankNum != 24)
      return kernel.emitError(
          "resident Conv MegaKernel requires the Pebble 24x64x128 bank layout");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return kernel.emitError(
          "resident Conv MegaKernel requires SMatMul outBW=1");

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    SmallVector<Value> hostPacks;
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value minI8 = b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                              b.getI8IntegerAttr(-128));
    Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                b.getI32IntegerAttr(0));
    Value oneF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               b.getF32FloatAttr(1.0));

    auto makeFilledBank = [&](Value fill) {
      Value pack = b.create<memref::AllocOp>(
          loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
      hostPacks.push_back(pack);
      b.create<linalg::FillOp>(loc, fill, pack);
      Value bank = allocBank(b, loc, 1, 1);
      return mvinBank(b, loc, pack, bank, target.bankDepth);
    };

    std::function<Value(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                        Value, int64_t, int64_t)>
        emitStage;
    emitStage = [&](int64_t stageIndex, int64_t y0, int64_t x0, int64_t height,
                    int64_t width, int64_t channelBase, Value destination,
                    int64_t destinationBase,
                    int64_t destinationStride) -> Value {
      Stage &stage = stages[stageIndex];
      if (y0 < 0 || x0 < 0 || height <= 0 || width <= 0 ||
          y0 + height > stage.outputHeight || x0 + width > stage.outputWidth ||
          channelBase < 0 || channelBase >= stage.outputChannels ||
          destinationBase < 0 || destinationStride < width ||
          destinationBase + (height - 1) * destinationStride + width >
              target.bankDepth) {
        stage.op->emitError("requested resident output tile is invalid");
        return {};
      }

      if (stage.add) {
        if (stage.input == kernel.getInput() ||
            stage.rhs == kernel.getInput()) {
          stage.op->emitError("INT8 Add directly consuming the MegaKernel "
                              "input is unsupported");
          return {};
        }
        Value lhs = makeFilledBank(zeroI8);
        lhs = emitStage(producer.lookup(stage.input), y0, x0, height, width,
                        channelBase, lhs, destinationBase, destinationStride);
        if (!lhs)
          return {};
        Value rhs = makeFilledBank(zeroI8);
        rhs = emitStage(producer.lookup(stage.rhs), y0, x0, height, width,
                        channelBase, rhs, destinationBase, destinationStride);
        if (!rhs)
          return {};
        Value lhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale / stage.outputScale));
        Value rhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.rhsScale / stage.outputScale));
        destination = b.create<BankInt8AddOp>(
                           loc, destination.getType(), lhs, rhs, destination,
                           createI64Const(b, loc, target.bankDepth), lhsRatio,
                           rhsRatio, b.getBoolAttr(stage.activation == 1))
                          .getOutputBankOut();
        releaseBank(b, loc, lhs);
        releaseBank(b, loc, rhs);
        return destination;
      }

      if (stage.globalAvg) {
        if (destinationBase != 0 || destinationStride != 1 || y0 != 0 ||
            x0 != 0 || height != 1 || width != 1 ||
            stage.input == kernel.getInput()) {
          stage.op->emitError("invalid resident GlobalAvgPool request");
          return {};
        }
        int64_t inputElements = stage.inputHeight * stage.inputWidth;
        if (inputElements > target.bankDepth) {
          stage.op->emitError(
              "GlobalAvgPool spatial extent exceeds one physical bank");
          return {};
        }
        Value source = makeFilledBank(zeroI8);
        source = emitStage(producer.lookup(stage.input), 0, 0,
                           stage.inputHeight, stage.inputWidth, channelBase,
                           source, 0, stage.inputWidth);
        if (!source)
          return {};

        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        Value sixteen = b.create<arith::ConstantIndexOp>(loc, kTile);
        Value onesPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, kTile}, b.getI8Type()));
        Value biasPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getI32Type()));
        Value scalePack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getF32Type()));
        hostPacks.append({onesPack, biasPack, scalePack});
        b.create<linalg::FillOp>(loc, zeroI8, onesPack);
        b.create<linalg::FillOp>(loc, zeroI32, biasPack);
        Value ratio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale /
                              (inputElements * stage.outputScale)));
        b.create<linalg::FillOp>(loc, ratio, scalePack);
        auto onesLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, inputElements),
            one);
        b.setInsertionPointToStart(onesLoop.getBody());
        Value index = onesLoop.getInductionVar();
        b.create<memref::StoreOp>(
            loc,
            b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                        b.getI8IntegerAttr(1)),
            onesPack,
            ValueRange{b.create<arith::DivUIOp>(loc, index, sixteen),
                       b.create<arith::RemUIOp>(loc, index, sixteen)});
        b.setInsertionPointAfter(onesLoop);

        Value onesBank = allocBank(b, loc, 1, 1);
        Value onesLoaded = mvinBank(b, loc, onesPack, onesBank, 4);
        Value biasBank = allocBank(b, loc, 1, 1);
        Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
        Value biasState = b.create<BankSMatMulBiasOp>(
            loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value result = allocBank(b, loc, 1, 1);
        result =
            b.create<BankSMatMulOp>(
                 loc, result.getType(), onesLoaded, source, result,
                 createI64ConstU(b, loc, matrixRs2(1, kTile, target.bankDepth)),
                 createI1Const(b, loc, true), createI1Const(b, loc, true),
                 createI64Const(b, loc, 0))
                .getWrBankOut();
        destination = b.create<BankQuantI32ToI8Op>(
                           loc, destination.getType(), result, scaleLoaded,
                           destination, createI64Const(b, loc, 4),
                           createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                           b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                           b.getI64IntegerAttr(1), b.getBoolAttr(false))
                          .getOutBankOut();
        releaseBank(b, loc, source);
        releaseBank(b, loc, onesLoaded);
        releaseBank(b, loc, biasState);
        releaseBank(b, loc, scaleLoaded);
        releaseBank(b, loc, result);
        return destination;
      }

      if (stage.pool) {
        int64_t maxSide = (8 - stage.kernel) / stage.stride + 1;
        int64_t side = std::min({height, width, maxSide});
        if (side <= 0) {
          stage.op->emitError("MaxPool input tile does not fit one bank");
          return {};
        }
        if (height != side || width != side) {
          destination =
              emitStage(stageIndex, y0, x0, side, side, channelBase,
                        destination, destinationBase, destinationStride);
          if (!destination)
            return {};
          if (width > side) {
            destination = emitStage(stageIndex, y0, x0 + side, side,
                                    width - side, channelBase, destination,
                                    destinationBase + side, destinationStride);
            if (!destination)
              return {};
          }
          if (height > side) {
            destination = emitStage(stageIndex, y0 + side, x0, height - side,
                                    width, channelBase, destination,
                                    destinationBase + side * destinationStride,
                                    destinationStride);
          }
          return destination;
        }

        int64_t inputSide = (side - 1) * stage.stride + stage.kernel;
        int64_t sourceY = y0 * stage.stride - stage.padding;
        int64_t sourceX = x0 * stage.stride - stage.padding;
        Value source = makeFilledBank(minI8);
        int64_t validY = std::max<int64_t>(0, sourceY);
        int64_t validX = std::max<int64_t>(0, sourceX);
        int64_t validYEnd = std::min(stage.inputHeight, sourceY + inputSide);
        int64_t validXEnd = std::min(stage.inputWidth, sourceX + inputSide);
        if (validY < validYEnd && validX < validXEnd) {
          if (stage.input == kernel.getInput()) {
            stage.op->emitError("MaxPool directly consuming the MegaKernel "
                                "input is unsupported");
            return {};
          }
          source = emitStage(
              producer.lookup(stage.input), validY, validX, validYEnd - validY,
              validXEnd - validX, channelBase, source,
              (validY - sourceY) * inputSide + validX - sourceX, inputSide);
          if (!source)
            return {};
        }
        destination =
            b.create<BankMaxPoolOp>(
                 loc, destination.getType(), source, destination,
                 createI64Const(b, loc, side * side),
                 b.getI64IntegerAttr(inputSide), b.getI64IntegerAttr(side),
                 b.getI64IntegerAttr(stage.kernel),
                 b.getI64IntegerAttr(stage.stride), b.getI64IntegerAttr(0),
                 createI64Const(b, loc, 0),
                 createI64Const(b, loc, destinationBase),
                 createI64Const(b, loc, destinationStride),
                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                .getOutBankOut();
        releaseBank(b, loc, source);
        return destination;
      }

      int64_t maxSide =
          std::min<int64_t>(4, (8 - stage.kernel) / stage.stride + 1);
      int64_t side = std::min({height, width, maxSide});
      if (side <= 0) {
        stage.op->emitError("Conv input tile does not fit one bank");
        return {};
      }
      if (height != side || width != side) {
        destination =
            emitStage(stageIndex, y0, x0, side, side, channelBase, destination,
                      destinationBase, destinationStride);
        if (!destination)
          return {};
        if (width > side) {
          destination = emitStage(stageIndex, y0, x0 + side, side, width - side,
                                  channelBase, destination,
                                  destinationBase + side, destinationStride);
          if (!destination)
            return {};
        }
        if (height > side) {
          destination = emitStage(stageIndex, y0 + side, x0, height - side,
                                  width, channelBase, destination,
                                  destinationBase + side * destinationStride,
                                  destinationStride);
        }
        return destination;
      }
      int64_t outputPanels = (stage.outputChannels + kTile - 1) / kTile;
      if (channelBase % kTile || channelBase / kTile >= outputPanels) {
        stage.op->emitError("Conv output channel panel is invalid");
        return {};
      }

      int64_t inputSide = (side - 1) * stage.stride + stage.kernel;
      int64_t sourceY = y0 * stage.stride - stage.padding;
      int64_t sourceX = x0 * stage.stride - stage.padding;
      int64_t panelRows = inputSide * inputSide;
      int64_t inputPanels =
          stage.depthwise ? 1 : (stage.inputChannels + kTile - 1) / kTile;
      int64_t panelsPerBank = target.bankDepth / panelRows;
      if (panelsPerBank <= 0) {
        stage.op->emitError("Conv source panel does not fit one bank");
        return {};
      }
      int64_t sourceBankCount =
          (inputPanels + panelsPerBank - 1) / panelsPerBank;
      SmallVector<Value> sourceBanks;
      for (int64_t bankIndex = 0; bankIndex < sourceBankCount; ++bankIndex) {
        Value pack = b.create<memref::AllocOp>(
            loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
        hostPacks.push_back(pack);
        b.create<linalg::FillOp>(loc, zeroI8, pack);
        if (stage.input == kernel.getInput()) {
          int64_t firstPanel = bankIndex * panelsPerBank;
          int64_t lastPanel = std::min(inputPanels, firstPanel + panelsPerBank);
          for (int64_t panel = firstPanel; panel < lastPanel; ++panel) {
            int64_t rootChannelBase =
                stage.depthwise ? channelBase : panel * kTile;
            int64_t base = (panel - firstPanel) * panelRows;
            int64_t localYBegin = std::max<int64_t>(0, -sourceY);
            int64_t localYEnd =
                std::min<int64_t>(inputSide, stage.inputHeight - sourceY);
            int64_t localXBegin = std::max<int64_t>(0, -sourceX);
            int64_t localXEnd =
                std::min<int64_t>(inputSide, stage.inputWidth - sourceX);
            int64_t validLanes =
                std::min<int64_t>(kTile, stage.inputChannels - rootChannelBase);
            int64_t validRows = localYEnd - localYBegin;
            int64_t validColumns = localXEnd - localXBegin;
            if (validRows > 0 && validColumns > 0 && validLanes > 0) {
              Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
              Value one = b.create<arith::ConstantIndexOp>(loc, 1);
              auto copyY = b.create<scf::ForOp>(
                  loc, zero, b.create<arith::ConstantIndexOp>(loc, validRows),
                  one);
              b.setInsertionPointToStart(copyY.getBody());
              Value localY = b.create<arith::AddIOp>(
                  loc, copyY.getInductionVar(),
                  b.create<arith::ConstantIndexOp>(loc, localYBegin));
              auto copyX = b.create<scf::ForOp>(
                  loc, zero,
                  b.create<arith::ConstantIndexOp>(loc, validColumns), one);
              b.setInsertionPointToStart(copyX.getBody());
              Value localX = b.create<arith::AddIOp>(
                  loc, copyX.getInductionVar(),
                  b.create<arith::ConstantIndexOp>(loc, localXBegin));
              auto copyLane = b.create<scf::ForOp>(
                  loc, zero, b.create<arith::ConstantIndexOp>(loc, validLanes),
                  one);
              b.setInsertionPointToStart(copyLane.getBody());
              Value lane = copyLane.getInductionVar();
              Value globalY = b.create<arith::AddIOp>(
                  loc, localY, b.create<arith::ConstantIndexOp>(loc, sourceY));
              Value globalX = b.create<arith::AddIOp>(
                  loc, localX, b.create<arith::ConstantIndexOp>(loc, sourceX));
              Value channel = b.create<arith::AddIOp>(
                  loc, lane,
                  b.create<arith::ConstantIndexOp>(loc, rootChannelBase));
              Value row = b.create<arith::AddIOp>(
                  loc,
                  b.create<arith::AddIOp>(
                      loc,
                      b.create<arith::MulIOp>(
                          loc, localY,
                          b.create<arith::ConstantIndexOp>(loc, inputSide)),
                      localX),
                  b.create<arith::ConstantIndexOp>(loc, base));
              Value value = b.create<memref::LoadOp>(
                  loc, kernel.getInput(),
                  ValueRange{zero, globalY, globalX, channel});
              b.create<memref::StoreOp>(loc, value, pack,
                                        ValueRange{row, lane});
              b.setInsertionPointAfter(copyY);
            }
          }
        }
        Value sourceBank = allocBank(b, loc, 1, 1);
        sourceBanks.push_back(
            mvinBank(b, loc, pack, sourceBank, target.bankDepth));
      }

      if (stage.input != kernel.getInput()) {
        int64_t validY = std::max<int64_t>(0, sourceY);
        int64_t validX = std::max<int64_t>(0, sourceX);
        int64_t validYEnd = std::min(stage.inputHeight, sourceY + inputSide);
        int64_t validXEnd = std::min(stage.inputWidth, sourceX + inputSide);
        if (validY < validYEnd && validX < validXEnd) {
          for (int64_t panel = 0; panel < inputPanels; ++panel) {
            int64_t bankIndex = panel / panelsPerBank;
            int64_t slot = panel % panelsPerBank;
            int64_t sourceChannelBase =
                stage.depthwise ? channelBase : panel * kTile;
            sourceBanks[bankIndex] =
                emitStage(producer.lookup(stage.input), validY, validX,
                          validYEnd - validY, validXEnd - validX,
                          sourceChannelBase, sourceBanks[bankIndex],
                          slot * panelRows + (validY - sourceY) * inputSide +
                              validX - sourceX,
                          inputSide);
            if (!sourceBanks[bankIndex])
              return {};
          }
        }
      }

      int64_t validOutputChannels =
          std::min(kTile, stage.outputChannels - channelBase);
      Value biasPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({4, 4}, b.getI32Type()));
      Value scalePack = b.create<memref::AllocOp>(
          loc, MemRefType::get({4, 4}, b.getF32Type()));
      hostPacks.push_back(biasPack);
      hostPacks.push_back(scalePack);
      b.create<linalg::FillOp>(loc, zeroI32, biasPack);
      b.create<linalg::FillOp>(loc, oneF32, scalePack);
      Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
      Value one = b.create<arith::ConstantIndexOp>(loc, 1);
      auto parameterLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, validOutputChannels),
          one);
      b.setInsertionPointToStart(parameterLoop.getBody());
      Value lane = parameterLoop.getInductionVar();
      Value channel = b.create<arith::AddIOp>(
          loc, lane, b.create<arith::ConstantIndexOp>(loc, channelBase));
      Value group = b.create<arith::DivUIOp>(
          loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
      Value groupLane = b.create<arith::RemUIOp>(
          loc, lane, b.create<arith::ConstantIndexOp>(loc, 4));
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, stage.bias, channel), biasPack,
          ValueRange{group, groupLane});
      b.create<memref::StoreOp>(
          loc, b.create<memref::LoadOp>(loc, stage.scale, channel), scalePack,
          ValueRange{group, groupLane});
      b.setInsertionPointAfter(parameterLoop);
      Value biasBank = allocBank(b, loc, 1, 1);
      Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
      Value biasState = b.create<BankSMatMulBiasOp>(
          loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
      Value scaleBank = allocBank(b, loc, 1, 1);
      Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
      Value patchState = allocBank(b, loc, 1, 1);
      Value weightState = allocBank(b, loc, 1, 1);
      Value resultState = allocBank(b, loc, 1, 1);
      int64_t kernelElements = stage.kernel * stage.kernel;
      int64_t paddedK = (kernelElements + kTile - 1) / kTile * kTile;
      int64_t accumulationCount = stage.depthwise ? kTile : stage.inputChannels;

      for (int64_t accumulation = 0; accumulation < accumulationCount;
           ++accumulation) {
        int64_t inputChannel =
            stage.depthwise ? channelBase + accumulation : accumulation;
        int64_t panel = stage.depthwise ? 0 : inputChannel / kTile;
        int64_t bankIndex = panel / panelsPerBank;
        int64_t slot = panel % panelsPerBank;
        int64_t lane = stage.depthwise ? accumulation : inputChannel % kTile;
        patchState = b.create<BankIm2colOp>(
                          loc, patchState.getType(), sourceBanks[bankIndex],
                          patchState, createI64Const(b, loc, inputSide),
                          createI64Const(b, loc, stage.kernel),
                          createI64Const(b, loc, stage.stride),
                          createI64Const(b, loc, 0),
                          createI64Const(b, loc, slot * panelRows),
                          createI64Const(b, loc, lane), b.getI64IntegerAttr(0),
                          b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                          b.getI64IntegerAttr(side * side))
                         .getOutBankOut();

        Value weightPack;
        if (stage.depthwise) {
          weightPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({paddedK, kTile}, b.getI8Type()));
          hostPacks.push_back(weightPack);
          b.create<linalg::FillOp>(loc, zeroI8, weightPack);
          if (inputChannel < stage.inputChannels &&
              inputChannel < stage.outputChannels) {
            Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
            Value one = b.create<arith::ConstantIndexOp>(loc, 1);
            Value kernelSize =
                b.create<arith::ConstantIndexOp>(loc, stage.kernel);
            auto copyWeightY = b.create<scf::ForOp>(loc, zero, kernelSize, one);
            b.setInsertionPointToStart(copyWeightY.getBody());
            Value ky = copyWeightY.getInductionVar();
            auto copyWeightX = b.create<scf::ForOp>(loc, zero, kernelSize, one);
            b.setInsertionPointToStart(copyWeightX.getBody());
            Value kx = copyWeightX.getInductionVar();
            Value outputLaneBegin =
                b.create<arith::ConstantIndexOp>(loc, accumulation);
            Value outputLaneEnd =
                b.create<arith::ConstantIndexOp>(loc, accumulation + 1);
            auto copyWeightLane =
                b.create<scf::ForOp>(loc, outputLaneBegin, outputLaneEnd, one);
            b.setInsertionPointToStart(copyWeightLane.getBody());
            Value outputLane = copyWeightLane.getInductionVar();
            Value weightRow = b.create<arith::AddIOp>(
                loc, b.create<arith::MulIOp>(loc, ky, kernelSize), kx);
            Value value = b.create<memref::LoadOp>(
                loc, stage.weight,
                ValueRange{ky, kx,
                           b.create<arith::ConstantIndexOp>(loc, inputChannel),
                           zero});
            b.create<memref::StoreOp>(loc, value, weightPack,
                                      ValueRange{weightRow, outputLane});
            b.setInsertionPointAfter(copyWeightY);
          }
        } else {
          SmallVector<OpFoldResult> weightOffsets = {
              b.getIndexAttr(channelBase / kTile), b.getIndexAttr(inputChannel),
              b.getIndexAttr(0), b.getIndexAttr(0)};
          SmallVector<OpFoldResult> weightSizes = {
              b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(paddedK),
              b.getIndexAttr(kTile)};
          SmallVector<OpFoldResult> weightStrides(4, b.getIndexAttr(1));
          Value weightSlice = b.create<memref::SubViewOp>(
              loc, stage.weight, weightOffsets, weightSizes, weightStrides);
          weightPack = b.create<memref::CollapseShapeOp>(
              loc, weightSlice,
              SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
        }
        weightState = mvinBank(b, loc, weightPack, weightState, paddedK);
        resultState =
            b.create<BankSMatMulOp>(
                 loc, resultState.getType(), patchState, weightState,
                 resultState,
                 createI64ConstU(b, loc, matrixRs2(kTile, kTile, paddedK)),
                 createI1Const(b, loc, accumulation == 0),
                 createI1Const(b, loc, accumulation + 1 == accumulationCount),
                 createI64Const(b, loc, 0))
                .getWrBankOut();
      }

      destination = b.create<BankQuantI32ToI8Op>(
                         loc, destination.getType(), resultState, scaleLoaded,
                         destination, createI64Const(b, loc, side * side * 4),
                         createI64Const(b, loc, destinationBase),
                         createI64Const(b, loc, 0), b.getI64IntegerAttr(side),
                         b.getI64IntegerAttr(side),
                         b.getI64IntegerAttr(destinationStride),
                         b.getBoolAttr(stage.activation == 1))
                        .getOutBankOut();
      for (Value source : sourceBanks)
        releaseBank(b, loc, source);
      releaseBank(b, loc, patchState);
      releaseBank(b, loc, weightState);
      releaseBank(b, loc, resultState);
      releaseBank(b, loc, biasState);
      releaseBank(b, loc, scaleLoaded);
      return destination;
    };

    Stage &finalStage = stages.back();
    for (int64_t channelBase = 0; channelBase < finalStage.outputChannels;
         channelBase += kTile) {
      for (int64_t y0 = 0; y0 < finalStage.outputHeight; y0 += 8) {
        int64_t height = std::min<int64_t>(8, finalStage.outputHeight - y0);
        for (int64_t x0 = 0; x0 < finalStage.outputWidth; x0 += 8) {
          int64_t width = std::min<int64_t>(8, finalStage.outputWidth - x0);
          Value outputBank = allocBank(b, loc, 1, 1);
          outputBank = emitStage(stages.size() - 1, y0, x0, height, width,
                                 channelBase, outputBank, 0, width);
          if (!outputBank)
            return failure();
          Value packed = b.create<memref::AllocOp>(
              loc, MemRefType::get({height * width, kTile}, b.getI8Type()));
          Value stored = mvoutBank(b, loc, packed, outputBank, height * width);
          b.create<FenceOp>(loc);
          int64_t validChannels =
              std::min(kTile, finalStage.outputChannels - channelBase);
          Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
          Value one = b.create<arith::ConstantIndexOp>(loc, 1);
          Value heightValue = b.create<arith::ConstantIndexOp>(loc, height);
          Value widthValue = b.create<arith::ConstantIndexOp>(loc, width);
          Value channels = b.create<arith::ConstantIndexOp>(loc, validChannels);
          auto copyOutputY = b.create<scf::ForOp>(loc, zero, heightValue, one);
          b.setInsertionPointToStart(copyOutputY.getBody());
          Value y = copyOutputY.getInductionVar();
          auto copyOutputX = b.create<scf::ForOp>(loc, zero, widthValue, one);
          b.setInsertionPointToStart(copyOutputX.getBody());
          Value x = copyOutputX.getInductionVar();
          auto copyOutputLane = b.create<scf::ForOp>(loc, zero, channels, one);
          b.setInsertionPointToStart(copyOutputLane.getBody());
          Value lane = copyOutputLane.getInductionVar();
          Value position = b.create<arith::AddIOp>(
              loc, b.create<arith::MulIOp>(loc, y, widthValue), x);
          Value value =
              b.create<memref::LoadOp>(loc, packed, ValueRange{position, lane});
          Value outputChannel = b.create<arith::AddIOp>(
              loc, lane, b.create<arith::ConstantIndexOp>(loc, channelBase));
          Value outputY = b.create<arith::AddIOp>(
              loc, y, b.create<arith::ConstantIndexOp>(loc, y0));
          Value outputX = b.create<arith::AddIOp>(
              loc, x, b.create<arith::ConstantIndexOp>(loc, x0));
          if (finalStage.pool &&
              cast<MegaMaxPool2dOp>(finalStage.op).getFinalOutput())
            b.create<memref::StoreOp>(
                loc, value, kernel.getOutput(),
                ValueRange{zero, outputChannel, outputY, outputX});
          else
            b.create<memref::StoreOp>(
                loc, value, kernel.getOutput(),
                ValueRange{zero, outputY, outputX, outputChannel});
          b.setInsertionPointAfter(copyOutputY);
          releaseBank(b, loc, stored);
          b.create<memref::DeallocOp>(loc, packed);
          for (Value pack : hostPacks)
            b.create<memref::DeallocOp>(loc, pack);
          hostPacks.clear();
        }
      }
    }
    b.eraseOp(kernel);
    return success();
  }
};

class ConvMegaKernelPreparationPattern : public OpRewritePattern<MegaKernelOp> {
public:
  ConvMegaKernelPreparationPattern(MLIRContext *context)
      : OpRewritePattern<MegaKernelOp>(context, 2) {}

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    SmallVector<Operation *> stages;
    for (Operation &op : body.without_terminator())
      stages.push_back(&op);
    if (stages.empty())
      return kernel.emitError(
          "MegaKernel region must contain at least one stage");
    if (!isa<MegaConv2dOp, MegaConv2dDepthwiseOp>(stages.front()))
      return failure();

    if (stages.size() == 1) {
      stages.front()->moveBefore(kernel);
      b.eraseOp(kernel);
      return success();
    }

    if (llvm::any_of(stages,
                     [](Operation *op) { return !isa<MegaConv2dOp>(op); }))
      return failure();

    for (auto [index, operation] : llvm::enumerate(stages)) {
      auto stage = dyn_cast<MegaConv2dOp>(operation);
      auto input = cast<MemRefType>(stage.getInput().getType());
      auto weight = cast<MemRefType>(stage.getWeight().getType());
      auto output = cast<MemRefType>(stage.getOutput().getType());
      bool last = index + 1 == stages.size();
      if (!input.hasStaticShape() || !weight.hasStaticShape() ||
          !output.hasStaticShape() || input.getRank() != 4 ||
          weight.getRank() != 4 || output.getRank() != 4 ||
          input.getShape()[0] != 1 || input.getShape()[1] != 1 ||
          input.getShape()[2] != 1 || weight.getShape()[0] != 1 ||
          weight.getShape()[1] != 1 || output.getShape()[0] != 1 ||
          (last ? (output.getShape()[2] != 1 || output.getShape()[3] != 1)
                : (output.getShape()[1] != 1 || output.getShape()[2] != 1)) ||
          stage.getStride() != 1 || stage.getPadLow() != 0 ||
          stage.getPadHigh() != 0)
        return stage.emitError(
            "multi-stage pointwise MegaKernel requires batch=1, spatial=1x1, "
            "stride=1, and no padding");
    }

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    SmallVector<ReassociationIndices> nhwc = {{0, 1, 2}, {3}};
    SmallVector<ReassociationIndices> nchw = {{0}, {1, 2, 3}};
    Value input = b.create<memref::CollapseShapeOp>(
        loc, cast<MegaConv2dOp>(stages.front()).getInput(), nhwc);
    Value output = b.create<memref::CollapseShapeOp>(
        loc, cast<MegaConv2dOp>(stages.back()).getOutput(), nchw);
    SmallVector<Value> weights;
    SmallVector<Value> outputs;
    for (auto [index, operation] : llvm::enumerate(stages)) {
      auto stage = cast<MegaConv2dOp>(operation);
      weights.push_back(
          b.create<memref::CollapseShapeOp>(loc, stage.getWeight(), nhwc));
      outputs.push_back(index + 1 == stages.size()
                            ? output
                            : Value(b.create<memref::CollapseShapeOp>(
                                  loc, stage.getOutput(), nhwc)));
    }
    auto replacement = b.create<MegaKernelOp>(loc, input, output);
    replacement.getBody().emplaceBlock();

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(&replacement.getBody().front());
    Value stageInput = input;
    for (auto [index, operation] : llvm::enumerate(stages)) {
      auto stage = cast<MegaConv2dOp>(operation);
      b.create<MegaMatmulOp>(loc, stageInput, weights[index], stage.getBias(),
                             stage.getScale(), stage.getLut(), outputs[index],
                             stage.getActivationAttr(),
                             stage.getOutputScaleAttr());
      stageInput = outputs[index];
    }
    b.create<MegaYieldOp>(loc);
    b.eraseOp(kernel);
    return success();
  }
};

class ConvMegaKernelToBankSSAPattern : public OpRewritePattern<MegaKernelOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    SmallVector<Operation *> operations;
    for (Operation &operation : body.without_terminator())
      operations.push_back(&operation);
    if (operations.empty() || !isa<MegaConv2dOp>(operations.front()))
      return failure();
    if (operations.size() != 2 && operations.size() != 3 &&
        operations.size() != 4)
      return kernel.emitError(
          "Conv MegaKernel must be Conv-DW, PW-DW-PW, or PW-PW-DW-PW");

    bool finalDepthwise = operations.size() == 2;
    unsigned expansionIndex = finalDepthwise ? 0 : operations.size() - 3;
    auto preparation = operations.size() == 4
                           ? dyn_cast<MegaConv2dOp>(operations[0])
                           : MegaConv2dOp();
    auto expansion = dyn_cast<MegaConv2dOp>(operations[expansionIndex]);
    auto depthwise =
        dyn_cast<MegaConv2dDepthwiseOp>(operations[expansionIndex + 1]);
    auto projection =
        finalDepthwise ? MegaConv2dOp()
                       : dyn_cast<MegaConv2dOp>(operations[expansionIndex + 2]);
    if (!expansion || !depthwise || (!finalDepthwise && !projection) ||
        (operations.size() == 4 && !preparation))
      return kernel.emitError(
          "Conv MegaKernel must be Conv-DW, PW-DW-PW, or PW-PW-DW-PW");

    if ((preparation && preparation.getOutput() != expansion.getInput()) ||
        expansion.getOutput() != depthwise.getInput() ||
        (!finalDepthwise && depthwise.getOutput() != projection.getInput()) ||
        (finalDepthwise ? depthwise.getOutput() != kernel.getOutput()
                        : projection.getOutput() != kernel.getOutput()) ||
        operations.front()->getOperand(0) != kernel.getInput())
      return kernel.emitError("Conv MegaKernel data chain is malformed");

    auto inputTy = dyn_cast<MemRefType>(kernel.getInput().getType());
    auto expansionWeightTy =
        dyn_cast<MemRefType>(expansion.getWeight().getType());
    auto expansionOutputTy =
        dyn_cast<MemRefType>(expansion.getOutput().getType());
    auto depthwiseWeightTy =
        dyn_cast<MemRefType>(depthwise.getWeight().getType());
    auto depthwiseOutputTy =
        dyn_cast<MemRefType>(depthwise.getOutput().getType());
    auto projectionWeightTy =
        finalDepthwise ? MemRefType()
                       : dyn_cast<MemRefType>(projection.getWeight().getType());
    auto outputTy = dyn_cast<MemRefType>(kernel.getOutput().getType());
    if (!inputTy || !expansionWeightTy || !expansionOutputTy ||
        !depthwiseWeightTy || !depthwiseOutputTy ||
        (!finalDepthwise && !projectionWeightTy) || !outputTy ||
        !inputTy.hasStaticShape() || !expansionWeightTy.hasStaticShape() ||
        !expansionOutputTy.hasStaticShape() ||
        !depthwiseWeightTy.hasStaticShape() ||
        !depthwiseOutputTy.hasStaticShape() ||
        (!finalDepthwise && !projectionWeightTy.hasStaticShape()) ||
        !outputTy.hasStaticShape())
      return kernel.emitError("Conv MegaKernel requires static memrefs");

    auto inputShape = inputTy.getShape();
    auto expansionWeightShape = expansionWeightTy.getShape();
    auto expansionOutputShape = expansionOutputTy.getShape();
    auto depthwiseWeightShape = depthwiseWeightTy.getShape();
    auto depthwiseOutputShape = depthwiseOutputTy.getShape();
    auto projectionWeightShape =
        finalDepthwise ? ArrayRef<int64_t>() : projectionWeightTy.getShape();
    auto outputShape = outputTy.getShape();
    int64_t inputSize = inputShape[1];
    int64_t inputChannels = inputShape[3];
    int64_t expandedChannels = expansionWeightShape[3];
    int64_t outputChannels =
        finalDepthwise ? expandedChannels : projectionWeightShape[3];
    int64_t intermediateSize = expansionOutputShape[1];
    int64_t expansionKernel = expansionWeightShape[0];
    int64_t expansionStride = expansion.getStride();
    int64_t expansionPadding = expansion.getPadLow();
    int64_t kernelSize = depthwiseWeightShape[0];
    int64_t stride = depthwise.getStride();
    int64_t padding = depthwise.getPadLow();
    int64_t outputSize = outputShape[2];
    if (inputTy.getRank() != 4 || expansionWeightTy.getRank() != 4 ||
        expansionOutputTy.getRank() != 4 || depthwiseWeightTy.getRank() != 4 ||
        depthwiseOutputTy.getRank() != 4 ||
        (!finalDepthwise && projectionWeightTy.getRank() != 4) ||
        outputTy.getRank() != 4)
      return kernel.emitError("Conv MegaKernel operands must be rank-4");
    if (inputShape[0] != 1 || inputShape[2] != inputSize || inputSize <= 0 ||
        inputChannels <= 0)
      return kernel.emitError("Conv MegaKernel input shape is invalid");
    if (expansionKernel <= 0 || expansionKernel > 7 ||
        expansionWeightShape[1] != expansionKernel ||
        expansionWeightShape[2] != inputChannels || expandedChannels <= 0)
      return kernel.emitError("Conv MegaKernel expansion shape is invalid");
    if (expansionOutputShape !=
        ArrayRef<int64_t>(
            {1, intermediateSize, intermediateSize, expandedChannels}))
      return kernel.emitError(
          "Conv MegaKernel expansion output layout is invalid");
    if (depthwiseWeightShape !=
        ArrayRef<int64_t>({kernelSize, kernelSize, expandedChannels, 1}))
      return kernel.emitError(
          "Conv MegaKernel depthwise weight layout is invalid");
    if (finalDepthwise) {
      if (depthwiseOutputShape != outputShape)
        return kernel.emitError(
            "Conv MegaKernel final depthwise output layout is invalid");
    } else if (depthwiseOutputShape !=
               ArrayRef<int64_t>(
                   {1, outputSize, outputSize, expandedChannels})) {
      return kernel.emitError(
          "Conv MegaKernel depthwise output layout is invalid");
    }
    if (!finalDepthwise &&
        projectionWeightShape !=
            ArrayRef<int64_t>({1, 1, expandedChannels, outputChannels}))
      return kernel.emitError(
          "Conv MegaKernel projection weight layout is invalid");
    if (outputShape !=
        ArrayRef<int64_t>({1, outputChannels, outputSize, outputSize}))
      return kernel.emitError("Conv MegaKernel output layout is invalid");
    if (expansionStride <= 0 || expansionPadding < 0 ||
        expansion.getPadHigh() != expansionPadding ||
        intermediateSize !=
            (inputSize + 2 * expansionPadding - expansionKernel) /
                    expansionStride +
                1)
      return kernel.emitError("Conv MegaKernel expansion geometry is invalid");
    if (!finalDepthwise &&
        (projection.getStride() != 1 || projection.getPadLow() != 0 ||
         projection.getPadHigh() != 0))
      return kernel.emitError("Conv MegaKernel projection geometry is invalid");
    if (depthwise.getPadHigh() != padding || stride <= 0 || padding < 0 ||
        kernelSize <= 0 || kernelSize > 7 ||
        outputSize !=
            (intermediateSize + 2 * padding - kernelSize) / stride + 1)
      return kernel.emitError("Conv MegaKernel depthwise geometry is invalid");

    int64_t preparationChannels = inputChannels;
    if (preparation) {
      auto preparationWeightTy =
          dyn_cast<MemRefType>(preparation.getWeight().getType());
      auto preparationOutputTy =
          dyn_cast<MemRefType>(preparation.getOutput().getType());
      if (!preparationWeightTy || !preparationOutputTy ||
          !preparationWeightTy.hasStaticShape() ||
          !preparationOutputTy.hasStaticShape())
        return preparation.emitError("preparation PW requires static memrefs");
      auto weightShape = preparationWeightTy.getShape();
      preparationChannels = weightShape[3];
      if (weightShape !=
              ArrayRef<int64_t>({1, 1, inputChannels, preparationChannels}) ||
          preparationOutputTy.getShape() !=
              ArrayRef<int64_t>(
                  {1, inputSize, inputSize, preparationChannels}) ||
          preparation.getStride() != 1 || preparation.getPadLow() != 0 ||
          preparation.getPadHigh() != 0 || preparationChannels > 64 ||
          expansionWeightShape[2] != preparationChannels)
        return preparation.emitError("preparation PW must preserve space and "
                                     "produce at most 64 channels");
    }

    auto validStageTypes = [&](auto stage, bool final) {
      auto weight = cast<MemRefType>(stage.getWeight().getType());
      auto bias = cast<MemRefType>(stage.getBias().getType());
      auto scale = cast<MemRefType>(stage.getScale().getType());
      auto lut = cast<MemRefType>(stage.getLut().getType());
      auto output = cast<MemRefType>(stage.getOutput().getType());
      int64_t activation = stage.getActivation();
      int64_t channels = final
                             ? outputChannels
                             : (isa<MegaConv2dDepthwiseOp>(stage.getOperation())
                                    ? expandedChannels
                                    : weight.getShape()[3]);
      return weight.getElementType().isInteger(8) &&
             bias.getElementType().isInteger(32) &&
             scale.getElementType().isF32() &&
             lut.getElementType().isInteger(8) && lut.getRank() == 1 &&
             lut.getShape()[0] == (activation == 2 ? 256 : 1) &&
             activation >= 0 && activation <= 2 &&
             (!final || activation != 2) && bias.hasStaticShape() &&
             scale.hasStaticShape() &&
             bias.getShape() == ArrayRef<int64_t>({channels}) &&
             scale.getShape() == ArrayRef<int64_t>({channels}) &&
             (final ? output.getElementType().isF32()
                    : output.getElementType().isInteger(8));
    };
    if ((preparation && !validStageTypes(preparation, false)) ||
        !validStageTypes(expansion, false) ||
        !validStageTypes(depthwise, finalDepthwise) ||
        (!finalDepthwise && !validStageTypes(projection, true)))
      return kernel.emitError(
          "Conv MegaKernel requires INT8 weights/intermediates, INT32 bias, "
          "FP32 scale, and FP32 final output");

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < 64 ||
        !llvm::isPowerOf2_64(target.bankDepth))
      return kernel.emitError("Conv MegaKernel requires 128-bit power-of-two "
                              "banks with at least 64 rows");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return kernel.emitError("Conv MegaKernel requires SMatMulBall outBW=1");

    int64_t blockSide = 4;
    while (blockSide > 0 && ((blockSide - 1) * stride + kernelSize) *
                                    ((blockSide - 1) * stride + kernelSize) >
                                target.bankDepth)
      --blockSide;
    if (blockSide == 0)
      return kernel.emitError("Depthwise halo does not fit one bank");
    int64_t haloSide = (blockSide - 1) * stride + kernelSize;
    int64_t haloElements = haloSide * haloSide;
    int64_t haloRows = (haloElements + kTile - 1) / kTile * kTile;
    int64_t haloChunks = haloRows / kTile;
    int64_t expandedPanels = (expandedChannels + kTile - 1) / kTile;
    int64_t depthwiseBankCount = finalDepthwise ? 0 : (expandedPanels + 3) / 4;
    int64_t persistentPreparationBanks = preparation ? haloChunks : 0;
    bool hasHardSwish = llvm::any_of(operations, [](Operation *operation) {
      if (auto stage = dyn_cast<MegaConv2dOp>(operation))
        return stage.getActivation() == 2;
      return cast<MegaConv2dDepthwiseOp>(operation).getActivation() == 2;
    });
    if (depthwiseBankCount + persistentPreparationBanks + 8 +
            (hasHardSwish ? 2 : 0) >
        target.bankNum)
      return kernel.emitError("Conv MegaKernel exceeds physical bank capacity");

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kTile);
    Value inputSizeValue = b.create<arith::ConstantIndexOp>(loc, inputSize);
    Value intermediateSizeValue =
        b.create<arith::ConstantIndexOp>(loc, intermediateSize);
    Value outputSizeValue = b.create<arith::ConstantIndexOp>(loc, outputSize);
    Value blockSideValue = b.create<arith::ConstantIndexOp>(loc, blockSide);
    Value strideValue = b.create<arith::ConstantIndexOp>(loc, stride);
    Value paddingValue = b.create<arith::ConstantIndexOp>(loc, padding);
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                b.getI32IntegerAttr(0));
    Value oneF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               b.getF32FloatAttr(1.0));

    SmallVector<Value> lutPacks;
    auto packLut = [&](Value lut) {
      if (!lut)
        return Value();
      Value pack = b.create<memref::AllocOp>(
          loc, MemRefType::get({kTile, kTile}, b.getI8Type()));
      lutPacks.push_back(pack);
      auto loop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, 256), one);
      b.setInsertionPointToStart(loop.getBody());
      Value index = loop.getInductionVar();
      Value row = b.create<arith::DivUIOp>(loc, index, sixteen);
      Value column = b.create<arith::RemUIOp>(loc, index, sixteen);
      Value value = b.create<memref::LoadOp>(loc, lut, index);
      b.create<memref::StoreOp>(loc, value, pack, ValueRange{row, column});
      b.setInsertionPointAfter(loop);
      return pack;
    };
    Value preparationLutPack = preparation && preparation.getActivation() == 2
                                   ? packLut(preparation.getLut())
                                   : Value();
    Value expansionLutPack =
        expansion.getActivation() == 2 ? packLut(expansion.getLut()) : Value();
    Value depthwiseLutPack =
        depthwise.getActivation() == 2 ? packLut(depthwise.getLut()) : Value();

    auto outputYLoop =
        b.create<scf::ForOp>(loc, zero, outputSizeValue, blockSideValue);
    b.setInsertionPointToStart(outputYLoop.getBody());
    Value outputY0 = outputYLoop.getInductionVar();
    auto outputXLoop =
        b.create<scf::ForOp>(loc, zero, outputSizeValue, blockSideValue);
    b.setInsertionPointToStart(outputXLoop.getBody());
    Value outputX0 = outputXLoop.getInductionVar();
    SmallVector<Value> hostPacks;

    auto loadLut = [&](Value lutPack) {
      Value bank = allocBank(b, loc, 1, 1);
      return mvinBank(b, loc, lutPack, bank, 16);
    };
    auto applyLut = [&](Value input, Value lut, int64_t rows) {
      Value output = allocBank(b, loc, 1, 1);
      Value transformed =
          b.create<BankLutOp>(loc, output.getType(), input, lut, output,
                              createI64Const(b, loc, rows));
      releaseBank(b, loc, input);
      return transformed;
    };

    Value globalY0 = b.create<arith::SubIOp>(
        loc, b.create<arith::MulIOp>(loc, outputY0, strideValue), paddingValue);
    Value globalX0 = b.create<arith::SubIOp>(
        loc, b.create<arith::MulIOp>(loc, outputX0, strideValue), paddingValue);
    auto loadBiasAndScale = [&](Value bias, Value scale, int64_t channels,
                                int64_t channelBase) -> SmallVector<Value> {
      Value biasPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({4, 4}, b.getI32Type()));
      Value scalePack = b.create<memref::AllocOp>(
          loc, MemRefType::get({4, 4}, b.getF32Type()));
      hostPacks.push_back(biasPack);
      hostPacks.push_back(scalePack);
      b.create<linalg::FillOp>(loc, zeroI32, biasPack);
      b.create<linalg::FillOp>(loc, oneF32, scalePack);
      int64_t validChannels = std::min(kTile, channels - channelBase);
      auto channelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, validChannels), one);
      b.setInsertionPointToStart(channelLoop.getBody());
      Value channel = channelLoop.getInductionVar();
      Value source = b.create<arith::AddIOp>(
          loc, b.create<arith::ConstantIndexOp>(loc, channelBase), channel);
      Value group = b.create<arith::DivUIOp>(
          loc, channel, b.create<arith::ConstantIndexOp>(loc, 4));
      Value lane = b.create<arith::RemUIOp>(
          loc, channel, b.create<arith::ConstantIndexOp>(loc, 4));
      b.create<memref::StoreOp>(loc,
                                b.create<memref::LoadOp>(loc, bias, source),
                                biasPack, ValueRange{group, lane});
      b.create<memref::StoreOp>(loc,
                                b.create<memref::LoadOp>(loc, scale, source),
                                scalePack, ValueRange{group, lane});
      b.setInsertionPointAfter(channelLoop);
      Value biasBank = allocBank(b, loc, 1, 1);
      Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
      Value biasState = b.create<BankSMatMulBiasOp>(
          loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
      Value scaleBank = allocBank(b, loc, 1, 1);
      Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
      return {biasState, scaleLoaded};
    };

    auto emitPointwiseChunk =
        [&](MegaConv2dOp stage, int64_t channelBase, int64_t chunk,
            Value bankInput, Value outputBank, int64_t outputBase) -> Value {
      auto stageInputTy = cast<MemRefType>(stage.getInput().getType());
      auto stageWeightTy = cast<MemRefType>(stage.getWeight().getType());
      int64_t stageInputSize = stageInputTy.getShape()[1];
      int64_t stageInputChannels = stageInputTy.getShape()[3];
      int64_t stageKernel = stageWeightTy.getShape()[0];
      int64_t stageStride = stage.getStride();
      int64_t stagePadding = stage.getPadLow();
      int64_t logicalK = stageKernel * stageKernel * stageInputChannels;
      int64_t logicalN = stageWeightTy.getShape()[3];
      int64_t paddedK = (logicalK + kTile - 1) / kTile * kTile;
      if (bankInput && paddedK > target.bankDepth) {
        stage.emitError("bank-resident PW input exceeds one bank");
        return {};
      }
      SmallVector<Value> params = loadBiasAndScale(
          stage.getBias(), stage.getScale(), logicalN, channelBase);
      Value resultState = allocBank(b, loc, 1, 1);
      for (int64_t k0 = 0; k0 < paddedK; k0 += target.bankDepth) {
        int64_t rows = std::min(target.bankDepth, paddedK - k0);
        int64_t validK = std::min(rows, logicalK - k0);
        Value activation = bankInput;
        if (!activation) {
          Value inputPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({rows, kTile}, b.getI8Type()));
          hostPacks.push_back(inputPack);
          b.create<linalg::FillOp>(loc, zeroI8, inputPack);
          auto positionLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
          b.setInsertionPointToStart(positionLoop.getBody());
          Value localPosition = positionLoop.getInductionVar();
          Value position = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, chunk * kTile),
              localPosition);
          Value validPosition = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, position,
              b.create<arith::ConstantIndexOp>(loc, haloElements));
          Value localY = b.create<arith::DivUIOp>(
              loc, position, b.create<arith::ConstantIndexOp>(loc, haloSide));
          Value localX = b.create<arith::RemUIOp>(
              loc, position, b.create<arith::ConstantIndexOp>(loc, haloSide));
          Value y = b.create<arith::AddIOp>(loc, globalY0, localY);
          Value x = b.create<arith::AddIOp>(loc, globalX0, localX);
          Value outputInBounds = b.create<arith::AndIOp>(
              loc, validPosition,
              b.create<arith::AndIOp>(
                  loc,
                  b.create<arith::AndIOp>(
                      loc,
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, y,
                                              zero),
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, y,
                                              intermediateSizeValue)),
                  b.create<arith::AndIOp>(
                      loc,
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, x,
                                              zero),
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, x,
                                              intermediateSizeValue))));
          auto kLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validK), one);
          b.setInsertionPointToStart(kLoop.getBody());
          Value localK = kLoop.getInductionVar();
          Value sourceK = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, k0), localK);
          Value sourceChannel = b.create<arith::RemUIOp>(
              loc, sourceK,
              b.create<arith::ConstantIndexOp>(loc, stageInputChannels));
          Value kernelPosition = b.create<arith::DivUIOp>(
              loc, sourceK,
              b.create<arith::ConstantIndexOp>(loc, stageInputChannels));
          Value kernelY = b.create<arith::DivUIOp>(
              loc, kernelPosition,
              b.create<arith::ConstantIndexOp>(loc, stageKernel));
          Value kernelX = b.create<arith::RemUIOp>(
              loc, kernelPosition,
              b.create<arith::ConstantIndexOp>(loc, stageKernel));
          Value sourceY = b.create<arith::AddIOp>(
              loc,
              b.create<arith::SubIOp>(
                  loc,
                  b.create<arith::MulIOp>(
                      loc, y,
                      b.create<arith::ConstantIndexOp>(loc, stageStride)),
                  b.create<arith::ConstantIndexOp>(loc, stagePadding)),
              kernelY);
          Value sourceX = b.create<arith::AddIOp>(
              loc,
              b.create<arith::SubIOp>(
                  loc,
                  b.create<arith::MulIOp>(
                      loc, x,
                      b.create<arith::ConstantIndexOp>(loc, stageStride)),
                  b.create<arith::ConstantIndexOp>(loc, stagePadding)),
              kernelX);
          Value inputInBounds = b.create<arith::AndIOp>(
              loc, outputInBounds,
              b.create<arith::AndIOp>(
                  loc,
                  b.create<arith::AndIOp>(
                      loc,
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              sourceY, zero),
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              sourceY,
                                              b.create<arith::ConstantIndexOp>(
                                                  loc, stageInputSize))),
                  b.create<arith::AndIOp>(
                      loc,
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              sourceX, zero),
                      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              sourceX,
                                              b.create<arith::ConstantIndexOp>(
                                                  loc, stageInputSize)))));
          auto validInput = b.create<scf::IfOp>(loc, inputInBounds, false);
          b.setInsertionPointToStart(&validInput.getThenRegion().front());
          Value row = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, b.create<arith::DivUIOp>(loc, localK, sixteen), sixteen),
              localPosition);
          Value lane = b.create<arith::RemUIOp>(loc, localK, sixteen);
          Value value = b.create<memref::LoadOp>(
              loc, stage.getInput(),
              ValueRange{zero, sourceY, sourceX, sourceChannel});
          b.create<memref::StoreOp>(loc, value, inputPack,
                                    ValueRange{row, lane});
          b.setInsertionPointAfter(validInput);
          b.setInsertionPointAfter(kLoop);
          b.setInsertionPointAfter(positionLoop);
          Value inputBank = allocBank(b, loc, 1, 1);
          activation = mvinBank(b, loc, inputPack, inputBank, rows);
        }

        Value weightPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({rows, kTile}, b.getI8Type()));
        hostPacks.push_back(weightPack);
        b.create<linalg::FillOp>(loc, zeroI8, weightPack);
        auto kLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, validK), one);
        b.setInsertionPointToStart(kLoop.getBody());
        Value localK = kLoop.getInductionVar();
        int64_t validN = std::min(kTile, logicalN - channelBase);
        auto nLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
        b.setInsertionPointToStart(nLoop.getBody());
        Value localN = nLoop.getInductionVar();
        Value sourceK = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, k0), localK);
        Value sourceN = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, channelBase), localN);
        Value weight = b.create<memref::LoadOp>(
            loc, stage.getWeight(),
            ValueRange{
                b.create<arith::DivUIOp>(
                    loc,
                    b.create<arith::DivUIOp>(loc, sourceK,
                                             b.create<arith::ConstantIndexOp>(
                                                 loc, stageInputChannels)),
                    b.create<arith::ConstantIndexOp>(loc, stageKernel)),
                b.create<arith::RemUIOp>(
                    loc,
                    b.create<arith::DivUIOp>(loc, sourceK,
                                             b.create<arith::ConstantIndexOp>(
                                                 loc, stageInputChannels)),
                    b.create<arith::ConstantIndexOp>(loc, stageKernel)),
                b.create<arith::RemUIOp>(
                    loc, sourceK,
                    b.create<arith::ConstantIndexOp>(loc, stageInputChannels)),
                sourceN});
        b.create<memref::StoreOp>(loc, weight, weightPack,
                                  ValueRange{localK, localN});
        b.setInsertionPointAfter(kLoop);
        Value weightBank = allocBank(b, loc, 1, 1);
        Value weightLoaded = mvinBank(b, loc, weightPack, weightBank, rows);
        resultState =
            b.create<BankSMatMulOp>(
                 loc, resultState.getType(), activation, weightLoaded,
                 resultState,
                 createI64ConstU(b, loc, matrixRs2(kTile, kTile, rows)),
                 createI1Const(b, loc, k0 == 0),
                 createI1Const(b, loc, k0 + rows == paddedK),
                 createI64Const(b, loc, 0))
                .getWrBankOut();
        if (!bankInput)
          releaseBank(b, loc, activation);
        releaseBank(b, loc, weightLoaded);
      }
      Value converted = b.create<BankQuantI32ToI8Op>(
          loc, outputBank.getType(), resultState, params[1], outputBank,
          createI64Const(b, loc, 64), createI64Const(b, loc, outputBase),
          createI64Const(b, loc, 0), b.getI64IntegerAttr(16),
          b.getI64IntegerAttr(1), b.getI64IntegerAttr(16),
          b.getBoolAttr(stage.getActivation() == 1));
      releaseBank(b, loc, resultState);
      releaseBank(b, loc, params[0]);
      releaseBank(b, loc, params[1]);
      return converted;
    };

    SmallVector<Value> preparationBanks;
    Value preparationLut = preparation && preparation.getActivation() == 2
                               ? loadLut(preparationLutPack)
                               : Value();
    if (preparation) {
      for (int64_t chunk = 0; chunk < haloChunks; ++chunk) {
        Value prepared = allocBank(b, loc, 1, 1);
        for (int64_t n0 = 0; n0 < preparationChannels; n0 += kTile) {
          prepared =
              emitPointwiseChunk(preparation, n0, chunk, Value(), prepared, n0);
          if (!prepared)
            return failure();
        }
        if (preparationLut)
          prepared =
              applyLut(prepared, preparationLut,
                       (preparationChannels + kTile - 1) / kTile * kTile);
        preparationBanks.push_back(prepared);
      }
    }
    if (preparationLut)
      releaseBank(b, loc, preparationLut);

    SmallVector<Value> depthwiseBanks;
    SmallVector<int64_t> depthwiseK;
    int64_t depthwiseKernelRows =
        (kernelSize * kernelSize + kTile - 1) / kTile * kTile;
    int64_t blockWindows = blockSide * blockSide;
    int64_t planeRows = haloRows / kTile;
    Value expansionLut =
        expansion.getActivation() == 2 ? loadLut(expansionLutPack) : Value();
    Value depthwiseLut =
        depthwise.getActivation() == 2 ? loadLut(depthwiseLutPack) : Value();
    for (int64_t n0 = 0; n0 < expandedPanels * kTile; n0 += kTile) {
      Value expanded = allocBank(b, loc, 1, 1);
      for (int64_t chunk = 0; chunk < haloChunks; ++chunk) {
        Value source = preparation ? preparationBanks[chunk] : Value();
        expanded = emitPointwiseChunk(expansion, n0, chunk, source, expanded,
                                      chunk * kTile);
        if (!expanded)
          return failure();
      }
      if (expansionLut)
        expanded = applyLut(expanded, expansionLut, haloRows);
      Value transposedBank = allocBank(b, loc, 1, 1);
      Value transposed = b.create<BankTransposeOp>(
          loc, transposedBank.getType(), expanded, transposedBank,
          createI64Const(b, loc, haloRows), createI64Const(b, loc, 8));
      releaseBank(b, loc, expanded);

      SmallVector<Value> params = loadBiasAndScale(
          depthwise.getBias(), depthwise.getScale(), expandedChannels, n0);
      Value patchState = allocBank(b, loc, 1, 1);
      Value weightState = allocBank(b, loc, 1, 1);
      Value resultState = allocBank(b, loc, 1, 1);
      for (int64_t channel = 0; channel < kTile; ++channel) {
        Value patch = b.create<BankIm2colOp>(
            loc, patchState.getType(), transposed, patchState,
            createI64Const(b, loc, haloSide),
            createI64Const(b, loc, kernelSize), createI64Const(b, loc, stride),
            createI64Const(b, loc, 0),
            createI64Const(b, loc, channel * planeRows),
            createI64Const(b, loc, 0), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(blockWindows));
        patchState = patch;

        Value weightPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({depthwiseKernelRows, kTile}, b.getI8Type()));
        hostPacks.push_back(weightPack);
        b.create<linalg::FillOp>(loc, zeroI8, weightPack);
        if (n0 + channel < expandedChannels) {
          for (int64_t ky = 0; ky < kernelSize; ++ky) {
            for (int64_t kx = 0; kx < kernelSize; ++kx) {
              Value weight = b.create<memref::LoadOp>(
                  loc, depthwise.getWeight(),
                  ValueRange{
                      b.create<arith::ConstantIndexOp>(loc, ky),
                      b.create<arith::ConstantIndexOp>(loc, kx),
                      b.create<arith::ConstantIndexOp>(loc, n0 + channel),
                      zero});
              b.create<memref::StoreOp>(
                  loc, weight, weightPack,
                  ValueRange{b.create<arith::ConstantIndexOp>(
                                 loc, ky * kernelSize + kx),
                             b.create<arith::ConstantIndexOp>(loc, channel)});
            }
          }
        }
        weightState =
            mvinBank(b, loc, weightPack, weightState, depthwiseKernelRows);
        resultState =
            b.create<BankSMatMulOp>(
                 loc, resultState.getType(), patchState, weightState,
                 resultState,
                 createI64ConstU(b, loc,
                                 matrixRs2(kTile, kTile, depthwiseKernelRows)),
                 createI1Const(b, loc, channel == 0),
                 createI1Const(b, loc, channel + 1 == kTile),
                 createI64Const(b, loc, 0))
                .getWrBankOut();
      }

      int64_t panel = n0 / kTile;
      if (finalDepthwise) {
        Value outputBank = allocBank(b, loc, 1, 1);
        Value converted = b.create<BankInt32ToFp32Op>(
            loc, outputBank.getType(), resultState, params[1], outputBank,
            createI64Const(b, loc, 64),
            b.getBoolAttr(depthwise.getActivation() == 1));
        Value packed = b.create<memref::AllocOp>(
            loc, MemRefType::get({kTile, kTile}, b.getF32Type()));
        hostPacks.push_back(packed);
        Value stored = mvoutBank(b, loc, packed, converted, 64);
        b.create<FenceOp>(loc);

        auto yLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, blockSide), one);
        b.setInsertionPointToStart(yLoop.getBody());
        Value localY = yLoop.getInductionVar();
        auto xLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, blockSide), one);
        b.setInsertionPointToStart(xLoop.getBody());
        Value localX = xLoop.getInductionVar();
        Value globalY = b.create<arith::AddIOp>(loc, outputY0, localY);
        Value globalX = b.create<arith::AddIOp>(loc, outputX0, localX);
        Value validPosition = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalY,
                                    outputSizeValue),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalX,
                                    outputSizeValue));
        auto validOutput = b.create<scf::IfOp>(loc, validPosition, false);
        b.setInsertionPointToStart(&validOutput.getThenRegion().front());
        int64_t validChannels = std::min(kTile, expandedChannels - n0);
        auto channelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, validChannels),
            one);
        b.setInsertionPointToStart(channelLoop.getBody());
        Value localChannel = channelLoop.getInductionVar();
        Value packedRow = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, localY, blockSideValue), localX);
        Value value = b.create<memref::LoadOp>(
            loc, packed, ValueRange{packedRow, localChannel});
        Value globalChannel = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, n0), localChannel);
        b.create<memref::StoreOp>(
            loc, value, kernel.getOutput(),
            ValueRange{zero, globalChannel, globalY, globalX});
        b.setInsertionPointAfter(validOutput);
        b.setInsertionPointAfter(yLoop);
        releaseBank(b, loc, stored);
      } else {
        if (panel % 4 == 0) {
          depthwiseBanks.push_back(allocBank(b, loc, 1, 1));
          depthwiseK.push_back(0);
        }
        depthwiseBanks.back() = b.create<BankQuantI32ToI8Op>(
            loc, depthwiseBanks.back().getType(), resultState, params[1],
            depthwiseBanks.back(), createI64Const(b, loc, 64),
            createI64Const(b, loc, (panel % 4) * kTile),
            createI64Const(b, loc, 0), b.getI64IntegerAttr(16),
            b.getI64IntegerAttr(1), b.getI64IntegerAttr(16),
            b.getBoolAttr(depthwise.getActivation() == 1));
        depthwiseK.back() += kTile;
        if (depthwiseLut && (panel % 4 == 3 || panel + 1 == expandedPanels))
          depthwiseBanks.back() =
              applyLut(depthwiseBanks.back(), depthwiseLut, depthwiseK.back());
      }
      releaseBank(b, loc, resultState);
      releaseBank(b, loc, patchState);
      releaseBank(b, loc, weightState);
      releaseBank(b, loc, params[0]);
      releaseBank(b, loc, params[1]);
      releaseBank(b, loc, transposed);
    }
    if (expansionLut)
      releaseBank(b, loc, expansionLut);
    if (depthwiseLut)
      releaseBank(b, loc, depthwiseLut);

    for (Value prepared : preparationBanks)
      releaseBank(b, loc, prepared);

    if (!finalDepthwise)
      for (int64_t n0 = 0; n0 < outputChannels; n0 += kTile) {
        SmallVector<Value> params = loadBiasAndScale(
            projection.getBias(), projection.getScale(), outputChannels, n0);
        Value resultState = allocBank(b, loc, 1, 1);
        int64_t kBase = 0;
        for (auto [index, activation] : llvm::enumerate(depthwiseBanks)) {
          int64_t rows = depthwiseK[index];
          Value weightPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({rows, kTile}, b.getI8Type()));
          hostPacks.push_back(weightPack);
          b.create<linalg::FillOp>(loc, zeroI8, weightPack);
          int64_t validK = std::min(rows, expandedChannels - kBase);
          int64_t validN = std::min(kTile, outputChannels - n0);
          auto kLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validK), one);
          b.setInsertionPointToStart(kLoop.getBody());
          Value localK = kLoop.getInductionVar();
          auto nLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
          b.setInsertionPointToStart(nLoop.getBody());
          Value localN = nLoop.getInductionVar();
          Value sourceK = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, kBase), localK);
          Value sourceN = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, n0), localN);
          Value weight = b.create<memref::LoadOp>(
              loc, projection.getWeight(),
              ValueRange{zero, zero, sourceK, sourceN});
          b.create<memref::StoreOp>(loc, weight, weightPack,
                                    ValueRange{localK, localN});
          b.setInsertionPointAfter(kLoop);
          Value weightBank = allocBank(b, loc, 1, 1);
          Value weightLoaded = mvinBank(b, loc, weightPack, weightBank, rows);
          resultState =
              b.create<BankSMatMulOp>(
                   loc, resultState.getType(), activation, weightLoaded,
                   resultState,
                   createI64ConstU(b, loc, matrixRs2(kTile, kTile, rows)),
                   createI1Const(b, loc, index == 0),
                   createI1Const(b, loc, index + 1 == depthwiseBanks.size()),
                   createI64Const(b, loc, 0))
                  .getWrBankOut();
          releaseBank(b, loc, weightLoaded);
          kBase += rows;
        }

        Value outputBank = allocBank(b, loc, 1, 1);
        Value converted = b.create<BankInt32ToFp32Op>(
            loc, outputBank.getType(), resultState, params[1], outputBank,
            createI64Const(b, loc, 64),
            b.getBoolAttr(projection.getActivation() == 1));
        releaseBank(b, loc, resultState);
        releaseBank(b, loc, params[0]);
        releaseBank(b, loc, params[1]);
        Value packed = b.create<memref::AllocOp>(
            loc, MemRefType::get({kTile, kTile}, b.getF32Type()));
        hostPacks.push_back(packed);
        Value stored = mvoutBank(b, loc, packed, converted, 64);
        b.create<FenceOp>(loc);

        auto yLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, blockSide), one);
        b.setInsertionPointToStart(yLoop.getBody());
        Value localY = yLoop.getInductionVar();
        auto xLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, blockSide), one);
        b.setInsertionPointToStart(xLoop.getBody());
        Value localX = xLoop.getInductionVar();
        Value globalY = b.create<arith::AddIOp>(loc, outputY0, localY);
        Value globalX = b.create<arith::AddIOp>(loc, outputX0, localX);
        Value validPosition = b.create<arith::AndIOp>(
            loc,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalY,
                                    outputSizeValue),
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, globalX,
                                    outputSizeValue));
        auto validOutput = b.create<scf::IfOp>(loc, validPosition, false);
        b.setInsertionPointToStart(&validOutput.getThenRegion().front());
        int64_t validN = std::min(kTile, outputChannels - n0);
        auto channelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
        b.setInsertionPointToStart(channelLoop.getBody());
        Value localN = channelLoop.getInductionVar();
        Value packedRow = b.create<arith::AddIOp>(
            loc, b.create<arith::MulIOp>(loc, localY, blockSideValue), localX);
        Value value = b.create<memref::LoadOp>(loc, packed,
                                               ValueRange{packedRow, localN});
        Value globalN = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, n0), localN);
        b.create<memref::StoreOp>(loc, value, kernel.getOutput(),
                                  ValueRange{zero, globalN, globalY, globalX});
        b.setInsertionPointAfter(validOutput);
        b.setInsertionPointAfter(yLoop);
        releaseBank(b, loc, stored);
      }

    for (Value activation : depthwiseBanks)
      releaseBank(b, loc, activation);
    b.create<FenceOp>(loc);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.setInsertionPointAfter(outputXLoop);
    b.setInsertionPointAfter(outputYLoop);
    for (Value pack : lutPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.eraseOp(kernel);
    return success();
  }
};

class MegaKernelToBankSSAPattern : public OpRewritePattern<MegaKernelOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");

    Block &body = kernel.getBody().front();
    if (body.without_terminator().empty())
      return kernel.emitError(
          "MegaKernel region must contain at least one stage");
    if (!isa<MegaMatmulOp>(body.front()))
      return failure();

    SmallVector<MegaMatmulOp> stages;
    for (Operation &op : body.without_terminator()) {
      auto matmul = dyn_cast<MegaMatmulOp>(op);
      if (!matmul)
        return kernel.emitError(
            "MatMul MegaKernel cannot contain non-MatMul stages");
      stages.push_back(matmul);
    }

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < kInt32Rows ||
        !llvm::isPowerOf2_64(target.bankDepth))
      return kernel.emitError(
          "MatMul MegaKernel requires 128-bit power-of-two-depth banks with "
          "at least 64 rows");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return kernel.emitError("MatMul MegaKernel requires SMatMulBall outBW=1");

    Value expectedInput = kernel.getInput();
    for (auto [index, stage] : llvm::enumerate(stages)) {
      bool last = index + 1 == stages.size();
      auto inputTy = cast<MemRefType>(stage.getInput().getType());
      auto weightTy = cast<MemRefType>(stage.getWeight().getType());
      auto biasTy = cast<MemRefType>(stage.getBias().getType());
      auto scaleTy = cast<MemRefType>(stage.getScale().getType());
      auto lutTy = cast<MemRefType>(stage.getLut().getType());
      auto outputTy = cast<MemRefType>(stage.getOutput().getType());
      if (stage.getInput() != expectedInput ||
          (last ? stage.getOutput() != kernel.getOutput()
                : stage.getOutput() == kernel.getOutput()))
        return stage.emitError("MatMul stage breaks the MegaKernel data chain");
      if (!inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
          !biasTy.hasStaticShape() || !scaleTy.hasStaticShape() ||
          !lutTy.hasStaticShape() || !outputTy.hasStaticShape())
        return stage.emitError("MatMul MegaKernel requires static shapes");
      int64_t m = inputTy.getShape()[0];
      int64_t k = inputTy.getShape()[1];
      int64_t n = weightTy.getShape()[1];
      if (m != 1 || k <= 0 || n <= 0 || weightTy.getShape()[0] != k ||
          outputTy.getShape()[0] != 1 || outputTy.getShape()[1] != n ||
          biasTy.getShape()[0] != n || scaleTy.getShape()[0] != n ||
          lutTy.getRank() != 1 || !lutTy.getElementType().isInteger(8) ||
          lutTy.getShape()[0] != (stage.getActivation() == 2 ? 256 : 1) ||
          stage.getActivation() < 0 || stage.getActivation() > 2 ||
          (last && stage.getActivation() == 2) ||
          (last ? !outputTy.getElementType().isF32()
                : !outputTy.getElementType().isInteger(8)))
        return stage.emitError("MatMul MegaKernel requires M=1 and matching "
                               "INT8/FP32 stage shapes");
      int64_t inputBanks = index == 0 ? 0 : (k + 4 * kTile - 1) / (4 * kTile);
      int64_t outputBanks = (n + 4 * kTile - 1) / (4 * kTile);
      int64_t requiredBanks = inputBanks + (last ? 5 : outputBanks + 6);
      if (requiredBanks > target.bankNum)
        return stage.emitError("MatMul MegaKernel exceeds bank capacity");
      expectedInput = stage.getOutput();
    }

    Location loc = kernel.getLoc();
    b.setInsertionPoint(kernel);
    SmallVector<Value> hostPacks;
    SmallVector<Value> activationBanks;
    SmallVector<int64_t> activationK;
    SmallVector<Value> initialPacks;
    SmallVector<int64_t> initialK;

    auto firstTy = cast<MemRefType>(stages.front().getInput().getType());
    int64_t firstK = firstTy.getShape()[1];
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kTile);
    int64_t paddedFirstK = (firstK + kTile - 1) / kTile * kTile;
    for (int64_t k0 = 0; k0 < paddedFirstK; k0 += target.bankDepth) {
      int64_t thisK = std::min(target.bankDepth, paddedFirstK - k0);
      int64_t validK = std::min(thisK, firstK - k0);
      Value inputPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({thisK / kTile, kTile}, b.getI8Type()));
      hostPacks.push_back(inputPack);
      b.create<linalg::FillOp>(loc, zeroI8, inputPack);
      auto inputLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, validK), one);
      b.setInsertionPointToStart(inputLoop.getBody());
      Value localK = inputLoop.getInductionVar();
      Value sourceK = b.create<arith::AddIOp>(
          loc, b.create<arith::ConstantIndexOp>(loc, k0), localK);
      Value value = b.create<memref::LoadOp>(loc, stages.front().getInput(),
                                             ValueRange{zero, sourceK});
      Value tile = b.create<arith::DivUIOp>(loc, localK, sixteen);
      Value packedColumn = b.create<arith::RemUIOp>(loc, localK, sixteen);
      b.create<memref::StoreOp>(loc, value, inputPack,
                                ValueRange{tile, packedColumn});
      b.setInsertionPointAfter(inputLoop);
      initialPacks.push_back(inputPack);
      initialK.push_back(thisK);
    }

    for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
      bool last = stageIndex + 1 == stages.size();
      int64_t logicalK =
          cast<MemRefType>(stage.getInput().getType()).getShape()[1];
      int64_t n = cast<MemRefType>(stage.getWeight().getType()).getShape()[1];
      SmallVector<Value> outputs;
      SmallVector<int64_t> outputK;
      SmallVector<Value> finalPacks;
      Value packedOutput;
      int64_t packedRows = 0;
      Value lutBank;
      if (stage.getActivation() == 2) {
        Value lutPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({kTile, kTile}, b.getI8Type()));
        hostPacks.push_back(lutPack);
        auto lutLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, 256), one);
        b.setInsertionPointToStart(lutLoop.getBody());
        Value index = lutLoop.getInductionVar();
        Value row = b.create<arith::DivUIOp>(loc, index, sixteen);
        Value column = b.create<arith::RemUIOp>(loc, index, sixteen);
        Value value = b.create<memref::LoadOp>(loc, stage.getLut(), index);
        b.create<memref::StoreOp>(loc, value, lutPack, ValueRange{row, column});
        b.setInsertionPointAfter(lutLoop);
        lutBank = allocBank(b, loc, 1, 1);
        lutBank = mvinBank(b, loc, lutPack, lutBank, 16);
      }

      for (int64_t n0 = 0; n0 < n; n0 += kTile) {
        int64_t validN = std::min(kTile, n - n0);
        if (!last && n0 % (4 * kTile) == 0) {
          packedOutput = allocBank(b, loc, 1, 1);
          packedRows = 0;
        }
        Value biasPack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getI32Type()));
        Value scalePack = b.create<memref::AllocOp>(
            loc, MemRefType::get({4, 4}, b.getF32Type()));
        hostPacks.push_back(biasPack);
        hostPacks.push_back(scalePack);
        Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                    b.getI32IntegerAttr(0));
        Value oneF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                                   b.getF32FloatAttr(1.0));
        b.create<linalg::FillOp>(loc, zeroI32, biasPack);
        b.create<linalg::FillOp>(loc, oneF32, scalePack);
        auto channelLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
        b.setInsertionPointToStart(channelLoop.getBody());
        Value channel = channelLoop.getInductionVar();
        Value sourceChannel = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, n0), channel);
        Value group = b.create<arith::DivUIOp>(
            loc, channel, b.create<arith::ConstantIndexOp>(loc, 4));
        Value lane = b.create<arith::RemUIOp>(
            loc, channel, b.create<arith::ConstantIndexOp>(loc, 4));
        Value bias =
            b.create<memref::LoadOp>(loc, stage.getBias(), sourceChannel);
        Value scale =
            b.create<memref::LoadOp>(loc, stage.getScale(), sourceChannel);
        b.create<memref::StoreOp>(loc, bias, biasPack, ValueRange{group, lane});
        b.create<memref::StoreOp>(loc, scale, scalePack,
                                  ValueRange{group, lane});
        b.setInsertionPointAfter(channelLoop);

        Value biasBank = allocBank(b, loc, 1, 1);
        Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
        Value biasState = b.create<BankSMatMulBiasOp>(
            loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value weightBank = allocBank(b, loc, 1, 1);
        Value resultState = allocBank(b, loc, 1, 1);

        int64_t kBase = 0;
        size_t chunkCount =
            stageIndex == 0 ? initialPacks.size() : activationBanks.size();
        for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
          int64_t thisK =
              stageIndex == 0 ? initialK[chunk] : activationK[chunk];
          int64_t validK = std::min(thisK, logicalK - kBase);
          Value activation;
          if (stageIndex == 0) {
            Value inputBank = allocBank(b, loc, 1, 1);
            activation =
                mvinBank(b, loc, initialPacks[chunk], inputBank, thisK / kTile);
          } else {
            activation = activationBanks[chunk];
          }
          Value weightPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({thisK, kTile}, b.getI8Type()));
          hostPacks.push_back(weightPack);
          b.create<linalg::FillOp>(loc, zeroI8, weightPack);
          auto kLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validK), one);
          b.setInsertionPointToStart(kLoop.getBody());
          Value localK = kLoop.getInductionVar();
          auto nLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
          b.setInsertionPointToStart(nLoop.getBody());
          Value localN = nLoop.getInductionVar();
          Value sourceK = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, kBase), localK);
          Value sourceN = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, n0), localN);
          Value weight = b.create<memref::LoadOp>(loc, stage.getWeight(),
                                                  ValueRange{sourceK, sourceN});
          b.create<memref::StoreOp>(loc, weight, weightPack,
                                    ValueRange{localK, localN});
          b.setInsertionPointAfter(kLoop);

          weightBank = mvinBank(b, loc, weightPack, weightBank, thisK);
          auto smatmul = b.create<BankSMatMulOp>(
              loc, resultState.getType(), activation, weightBank, resultState,
              createI64ConstU(b, loc, matrixRs2(1, kTile, thisK)),
              createI1Const(b, loc, chunk == 0),
              createI1Const(b, loc, chunk + 1 == chunkCount),
              createI64Const(b, loc, 0));
          resultState = smatmul.getWrBankOut();
          if (stageIndex == 0)
            releaseBank(b, loc, activation);
          kBase += thisK;
        }

        if (last) {
          Value outputBank = allocBank(b, loc, 1, 1);
          Value converted = b.create<BankInt32ToFp32Op>(
              loc, outputBank.getType(), resultState, scaleLoaded, outputBank,
              createI64Const(b, loc, 4),
              b.getBoolAttr(stage.getActivation() == 1));
          releaseBank(b, loc, resultState);
          Value packed = b.create<memref::AllocOp>(
              loc, MemRefType::get({1, kTile}, b.getF32Type()));
          hostPacks.push_back(packed);
          Value stored = mvoutBank(b, loc, packed, converted, 4);
          releaseBank(b, loc, stored);
          finalPacks.push_back(packed);
        } else {
          Value converted = b.create<BankQuantI32ToI8Op>(
              loc, packedOutput.getType(), resultState, scaleLoaded,
              packedOutput, createI64Const(b, loc, 4),
              createI64Const(b, loc, packedRows), createI64Const(b, loc, 0),
              b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
              b.getI64IntegerAttr(1),
              b.getBoolAttr(stage.getActivation() == 1));
          releaseBank(b, loc, resultState);
          packedOutput = converted;
          ++packedRows;
          if (packedRows == 4 || n0 + kTile >= n) {
            if (lutBank) {
              Value lutOutput = allocBank(b, loc, 1, 1);
              Value transformed = b.create<BankLutOp>(
                  loc, lutOutput.getType(), packedOutput, lutBank, lutOutput,
                  createI64Const(b, loc, packedRows));
              releaseBank(b, loc, packedOutput);
              packedOutput = transformed;
            }
            outputs.push_back(packedOutput);
            outputK.push_back(packedRows * kTile);
            packedOutput = {};
          }
        }
        releaseBank(b, loc, biasState);
        releaseBank(b, loc, scaleLoaded);
        releaseBank(b, loc, weightBank);
      }
      if (lutBank)
        releaseBank(b, loc, lutBank);

      for (Value activation : activationBanks)
        releaseBank(b, loc, activation);
      activationBanks = outputs;
      activationK = outputK;

      if (last) {
        b.create<FenceOp>(loc);
        for (auto [panel, packed] : llvm::enumerate(finalPacks)) {
          int64_t n0 = panel * kTile;
          int64_t validN = std::min(kTile, n - n0);
          auto outputLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validN), one);
          b.setInsertionPointToStart(outputLoop.getBody());
          Value localN = outputLoop.getInductionVar();
          Value output =
              b.create<memref::LoadOp>(loc, packed, ValueRange{zero, localN});
          Value outputN = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, n0), localN);
          b.create<memref::StoreOp>(loc, output, kernel.getOutput(),
                                    ValueRange{zero, outputN});
          b.setInsertionPointAfter(outputLoop);
        }
      }
    }

    for (Value bank : activationBanks)
      releaseBank(b, loc, bank);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.eraseOp(kernel);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populatePebbleMegaKernelToBankSSAPatterns(RewritePatternSet &patterns) {
  patterns
      .add<LinearConvPoolMegaKernelPattern, ConvMegaKernelPreparationPattern,
           ConvMegaKernelToBankSSAPattern, MegaKernelToBankSSAPattern>(
          patterns.getContext());
}
} // namespace mlir::buddy
