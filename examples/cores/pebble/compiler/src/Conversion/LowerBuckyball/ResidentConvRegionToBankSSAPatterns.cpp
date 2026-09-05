//===- ResidentConvRegionToBankSSAPatterns.cpp ---------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    if (body.without_terminator().empty() || !isa<MegaConv2dOp>(body.front()))
      return failure();

    struct Stage {
      Operation *op;
      Value input;
      Value rhs;
      Value output;
      Value weight;
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
      bool finalOutput;
      int64_t lutEntries;
    };

    SmallVector<Stage> stages;
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
            !output.getElementType().isInteger(8) || conv.getActivation() < 0 ||
            conv.getActivation() > 2 || conv.getStride() <= 0 ||
            conv.getPadLow() < 0 || conv.getPadLow() != conv.getPadHigh())
          return conv.emitError("unsupported Conv stage in resident region");
        auto in = input.getShape();
        auto weights = weight.getShape();
        auto out = output.getShape();
        int64_t kernelSize = conv.getKernel();
        int64_t paddedKernel =
            ((kernelSize * kernelSize + kTile - 1) / kTile) * kTile;
        if (kernelSize <= 0 || kernelSize > 7 ||
            weights != ArrayRef<int64_t>({(out[3] + kTile - 1) / kTile, in[3],
                                          paddedKernel, kTile}) ||
            bias.getShape() != ArrayRef<int64_t>({out[3]}) ||
            scale.getShape() != ArrayRef<int64_t>({out[3]}) ||
            (conv.getActivation() == 2
                 ? (lut.getShape()[0] != 256 && lut.getShape()[0] != 4096)
                 : lut.getShape()[0] != 1) ||
            (lut.getShape()[0] == 4096 &&
             (conv.getActivation() != 2 || out[3] != 16)) ||
            (in[1] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[1] ||
            (in[2] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[2])
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
        stage.outputHeight = out[1];
        stage.outputWidth = out[2];
        stage.outputChannels = out[3];
        stage.kernel = kernelSize;
        stage.stride = conv.getStride();
        stage.padding = conv.getPadLow();
        stage.activation = conv.getActivation();
        stage.outputScale = conv.getOutputScale().convertToFloat();
        stage.lutEntries = lut.getShape()[0];
      } else if (auto conv = dyn_cast<MegaConv2dDepthwiseOp>(operation)) {
        auto input = dyn_cast<MemRefType>(conv.getInput().getType());
        auto weight = dyn_cast<MemRefType>(conv.getWeight().getType());
        auto bias = dyn_cast<MemRefType>(conv.getBias().getType());
        auto scale = dyn_cast<MemRefType>(conv.getScale().getType());
        auto lut = dyn_cast<MemRefType>(conv.getLut().getType());
        auto output = dyn_cast<MemRefType>(conv.getOutput().getType());
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
            !output.getElementType().isInteger(8) || conv.getActivation() < 0 ||
            conv.getActivation() > 2 || conv.getStride() <= 0 ||
            conv.getPadLow() < 0 || conv.getPadLow() != conv.getPadHigh())
          return conv.emitError(
              "unsupported Depthwise Conv stage in resident region");
        auto in = input.getShape();
        auto weights = weight.getShape();
        auto out = output.getShape();
        int64_t kernelSize = conv.getKernel();
        if (kernelSize <= 0 || kernelSize > 7 ||
            weights != ArrayRef<int64_t>({kernelSize, kernelSize, in[3], 1}) ||
            bias.getShape() != ArrayRef<int64_t>({out[3]}) ||
            scale.getShape() != ArrayRef<int64_t>({out[3]}) ||
            (conv.getActivation() == 2
                 ? (lut.getShape()[0] != 256 && lut.getShape()[0] != 4096)
                 : lut.getShape()[0] != 1) ||
            lut.getShape()[0] == 4096 || in[3] != out[3] ||
            (in[1] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[1] ||
            (in[2] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[2])
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
        stage.outputHeight = out[1];
        stage.outputWidth = out[2];
        stage.outputChannels = out[3];
        stage.kernel = kernelSize;
        stage.stride = conv.getStride();
        stage.padding = conv.getPadLow();
        stage.activation = conv.getActivation();
        stage.outputScale = conv.getOutputScale().convertToFloat();
        stage.depthwise = true;
        stage.lutEntries = lut.getShape()[0];
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
      } else {
        return operation.emitError(
            "resident Conv region supports Conv, Depthwise Conv, MaxPool, "
            "INT8 Add, INT8 Mul, and GlobalAvgPool stages");
      }

      if (stage.input != kernel.getInput() && !producer.contains(stage.input))
        return operation.emitError(
            "stage input is not produced by an earlier region stage");
      if (stage.rhs && stage.rhs != kernel.getInput() &&
          !producer.contains(stage.rhs))
        return operation.emitError(
            "stage rhs is not produced by an earlier region stage");
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
    DenseMap<Operation *, Value> packedLuts;
    for (Stage &stage : stages) {
      if (stage.pool || stage.add || stage.multiply || stage.average)
        continue;
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
      int64_t totalPanels = (stage.outputChannels + kTile - 1) / kTile;
      if (height <= 0 || width <= 0 || panelCount <= 0 ||
          panelCount > totalPanels || destination.panelCount != panelCount ||
          destinationBase < 0 || destinationStride < width ||
          destinationBase + (height - 1) * destinationStride + width >
              destination.panelRows)
        return stage.op->emitError("invalid resident tile request");

      CachedTile *cache = nullptr;
      if (residualCache.stage == stageIndex)
        cache = &residualCache;
      if (cache) {
        IntegerAttr::ValueType firstPanelValue;
        if (!matchPattern(firstPanel, m_ConstantInt(&firstPanelValue)))
          return stage.op->emitError(
              "cached tile panel offset must be constant");
        int64_t firstPanelIndex = firstPanelValue.getSExtValue();
        if (firstPanelIndex < 0 ||
            firstPanelIndex + panelCount > cache->tile.panelCount)
          return stage.op->emitError("cached tile layout mismatch");
        SmallVector<Value> destinationStates(destination.banks.begin(),
                                             destination.banks.end());
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t sourcePanel = firstPanelIndex + localPanel;
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
        Value zeroBank = allocBank(b, loc, 1, 1);
        zeroBank = mvinBank(b, loc, zeroPack, zeroBank, target.bankDepth);
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
        releaseBank(b, loc, zeroBank);
      };

      if (stage.add) {
        Value lhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale / stage.outputScale));
        Value rhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.rhsScale / stage.outputScale));

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
        while (!branchAncestors.contains(residualStage)) {
          Value input = stages[residualStage].input;
          if (input == kernel.getInput() || !producer.contains(input))
            return stage.op->emitError(
                "INT8 Add branches have no resident common producer");
          int64_t next = producer.lookup(input);
          if (next >= residualStage)
            return stage.op->emitError("INT8 Add branch is not acyclic");
          residualStage = next;
        }
        Value residualY = y0;
        Value residualX = x0;
        int64_t residualHeight = height;
        int64_t residualWidth = width;
        for (int64_t cursor = branchStage; cursor != residualStage;) {
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
        // The main branch may need every channel panel of the common
        // residual stage as its input. Keep that complete panel set in the
        // cache for this spatial tile; only the bank packing, not the channel
        // count, may be split.
        int64_t chunkPanels = residualPanels;
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
          int64_t destinationSlot = panelBegin % destination.panelsPerBank;
          b.create<BankMaxPoolOp>(
              loc, destination.banks[destinationBank].getType(),
              sum.banks.front(), destination.banks[destinationBank],
              createI64Const(b, loc, height * width),
              b.getI64IntegerAttr(height), b.getI64IntegerAttr(width),
              b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
              b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
              createI64Const(b, loc,
                             destinationSlot * destination.panelRows +
                                 destinationBase),
              createI64Const(b, loc, destinationStride), b.getI64IntegerAttr(0),
              b.getI64IntegerAttr(0));
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
          TileBanks input = allocateTile(1, height * width, zeroI8);
          if (failed(emitInto(producer.lookup(stage.input), y0, x0, height,
                              width, panel, 1, input, 0, width)))
            return failure();
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
        if (height != 1 || width != 1 || destinationBase != 0 ||
            destinationStride != 1 || destination.banks.size() != 1)
          return stage.op->emitError("invalid GlobalAvgPool output tile");
        int64_t inputRows = stage.inputHeight * stage.inputWidth;
        bool fitsOneBank = inputRows <= target.bankDepth;
        if (!fitsOneBank &&
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
          if (failed(emitInto(producer.lookup(stage.input), zero, zero,
                              stage.inputHeight, stage.inputWidth, panel, 1,
                              fullSource, 0, stage.inputWidth)))
            return failure();
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
          if (failed(emitInto(producer.lookup(stage.input), y, x, 4, 4, panel,
                              1, source, 0, 4)))
            return failure();
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

      int64_t maxSide = std::min<int64_t>({4, height, width});
      while (maxSide > 0) {
        int64_t inputSide = (maxSide - 1) * stage.stride + stage.kernel;
        int64_t inputPanelRows = inputSide * inputSide;
        int64_t inputPanels = stage.depthwise
                                  ? panelCount
                                  : (stage.inputChannels + kTile - 1) / kTile;
        int64_t panelsPerBank = inputPanelRows <= target.bankDepth
                                    ? target.bankDepth / inputPanelRows
                                    : 0;
        int64_t inputBanks =
            stage.depthwise ? (panelsPerBank ? 1 : target.bankNum + 1)
            : panelsPerBank ? (inputPanels + panelsPerBank - 1) / panelsPerBank
                            : target.bankNum + 1;
        int64_t reservedBanks = stage.input == kernel.getInput() ? 6 : 11;
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
                   << ", destinationBanks=" << destination.banks.size() << ")";
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
                            destinationBase + side, destinationStride)))
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
      if (stage.input == kernel.getInput()) {
        if (inputPanelCount != 1)
          return stage.op->emitError(
              "resident region input must fit one 16-channel panel");
        source.panelRows = inputSide * inputSide;
        source.panelsPerBank = target.bankDepth / source.panelRows;
        source.panelCount = inputPanelCount;
        Value pack = b.create<memref::AllocOp>(
            loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
        b.create<linalg::FillOp>(loc, zeroI8, pack);
        auto yLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, inputSide), one);
        b.setInsertionPointToStart(yLoop.getBody());
        Value localY = yLoop.getInductionVar();
        auto xLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, inputSide), one);
        b.setInsertionPointToStart(xLoop.getBody());
        Value localX = xLoop.getInductionVar();
        Value globalY = b.create<arith::AddIOp>(loc, sourceY, localY);
        Value globalX = b.create<arith::AddIOp>(loc, sourceX, localX);
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
            loc, zero,
            b.create<arith::ConstantIndexOp>(loc, stage.inputChannels), one);
        b.setInsertionPointToStart(laneLoop.getBody());
        Value lane = laneLoop.getInductionVar();
        Value row = b.create<arith::AddIOp>(
            loc,
            b.create<arith::MulIOp>(
                loc, localY, b.create<arith::ConstantIndexOp>(loc, inputSide)),
            localX);
        Value value = b.create<memref::LoadOp>(
            loc, kernel.getInput(), ValueRange{zero, globalY, globalX, lane});
        b.create<memref::StoreOp>(loc, value, pack, ValueRange{row, lane});
        b.setInsertionPointAfter(laneLoop);
        b.setInsertionPointAfter(valid);
        b.setInsertionPointAfter(yLoop);
        Value bank = allocBank(b, loc, 1, 1);
        source.banks.push_back(mvinBank(b, loc, pack, bank, target.bankDepth));
        b.create<memref::DeallocOp>(loc, pack);
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
          if (sourceChunk.banks.size() != 1 ||
              failed(emitInto(
                  producer.lookup(stage.input), sourceY, sourceX, inputSide,
                  inputSide,
                  b.create<arith::ConstantIndexOp>(loc, sourcePanelBegin),
                  sourcePanelEnd - sourcePanelBegin, sourceChunk, 0,
                  inputSide)))
            return failure();
          for (int64_t localPanel = sourcePanelBegin;
               localPanel < sourcePanelEnd; ++localPanel) {
            int64_t sourceSlot = localPanel - sourcePanelBegin;
            int64_t destinationBank = localPanel / destination.panelsPerBank;
            int64_t destinationSlot = localPanel % destination.panelsPerBank;
            b.create<BankMaxPoolOp>(
                loc, destination.banks[destinationBank].getType(),
                sourceChunk.banks.front(), destination.banks[destinationBank],
                createI64Const(b, loc, side * side),
                b.getI64IntegerAttr(inputSide), b.getI64IntegerAttr(side),
                b.getI64IntegerAttr(stage.kernel),
                b.getI64IntegerAttr(stage.stride), b.getI64IntegerAttr(0),
                createI64Const(b, loc, sourceSlot * sourceChunk.panelRows),
                createI64Const(b, loc,
                               destinationSlot * destination.panelRows +
                                   destinationBase),
                createI64Const(b, loc, destinationStride),
                b.getI64IntegerAttr(0), b.getI64IntegerAttr(0));
          }
          releaseTile(sourceChunk);
        }
        maskInvalidOutput();
        return success();
      } else if (!stage.depthwise) {
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
          TileBanks panelSource =
              allocateTile(1, inputSide * inputSide, zeroI8);
          if (panelSource.banks.size() != 1)
            return stage.op->emitError(
                "Depthwise Conv input panel must fit one bank");
          if (failed(emitInto(producer.lookup(stage.input), sourceY, sourceX,
                              inputSide, inputSide, outputPanel, 1, panelSource,
                              0, inputSide)))
            return failure();
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

          Value weightPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({paddedK, kTile}, b.getI8Type()));
          b.create<linalg::FillOp>(loc, zeroI8, weightPack);
          Value outputChannel = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc, outputPanel,
                  b.create<arith::ConstantIndexOp>(loc, kTile)),
              lane);
          auto validChannel = b.create<scf::IfOp>(
              loc,
              b.create<arith::CmpIOp>(
                  loc, arith::CmpIPredicate::slt, outputChannel,
                  b.create<arith::ConstantIndexOp>(loc, stage.outputChannels)),
              false);
          b.setInsertionPointToStart(&validChannel.getThenRegion().front());
          for (int64_t ky = 0; ky < stage.kernel; ++ky) {
            for (int64_t kx = 0; kx < stage.kernel; ++kx) {
              Value weight = b.create<memref::LoadOp>(
                  loc, stage.weight,
                  ValueRange{b.create<arith::ConstantIndexOp>(loc, ky),
                             b.create<arith::ConstantIndexOp>(loc, kx),
                             outputChannel, zero});
              b.create<memref::StoreOp>(
                  loc, weight, weightPack,
                  ValueRange{b.create<arith::ConstantIndexOp>(
                                 loc, ky * stage.kernel + kx),
                             lane});
            }
          }
          b.setInsertionPointAfter(validChannel);
          Value weightNext =
              mvinBank(b, loc, weightPack, laneStates[1], paddedK);
          b.create<memref::DeallocOp>(loc, weightPack);
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

          Value quantized = allocBank(b, loc, 1, 1);
          Value quantizedState =
              b.create<BankQuantI32ToI8Op>(
                   loc, quantized.getType(), resultState, scaleLoaded,
                   quantized, createI64Const(b, loc, side * side * 4),
                   createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                   b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                   b.getI64IntegerAttr(side),
                   b.getBoolAttr(stage.activation == 1))
                  .getOutBankOut();
          releaseBank(b, loc, resultState);
          releaseBank(b, loc, scaleLoaded);
          Value transformed;
          Value outputState = quantizedState;
          if (stage.activation == 2) {
            Value lutOutput = allocBank(b, loc, 1, 1);
            transformed = b.create<BankLutOp>(
                loc, lutOutput.getType(), quantizedState, lutLoaded, lutOutput,
                createI64Const(b, loc, side * side));
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
          releaseTile(panelSource);
        }
        destination.banks.assign(destinationStates.begin(),
                                 destinationStates.end());
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
          if (states.empty())
            states = {allocBank(b, loc, 1, 1), allocBank(b, loc, 1, 1),
                      allocBank(b, loc, 1, 1)};
        };
        auto accumulateSource = [&](Value sourceBank, int64_t sourcePanelBegin,
                                    int64_t sourcePanelEnd,
                                    int64_t sourcePanelRows) {
          int64_t channelBegin = sourcePanelBegin * kTile;
          int64_t channelEnd =
              std::min(stage.inputChannels, sourcePanelEnd * kTile);
          auto channelLoop = b.create<scf::ForOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, channelBegin),
              b.create<arith::ConstantIndexOp>(loc, channelEnd), one,
              ValueRange{states[0], states[1], states[2]});
          b.setInsertionPointToStart(channelLoop.getBody());
          Value inputChannel = channelLoop.getInductionVar();
          ValueRange iterStates = channelLoop.getRegionIterArgs();
          Value sourceSlot = b.create<arith::SubIOp>(
              loc,
              b.create<arith::DivUIOp>(
                  loc, inputChannel,
                  b.create<arith::ConstantIndexOp>(loc, kTile)),
              b.create<arith::ConstantIndexOp>(loc, sourcePanelBegin));
          Value inputBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::MulIOp>(
                  loc, sourceSlot,
                  b.create<arith::ConstantIndexOp>(loc, sourcePanelRows)));
          Value inputLane = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::RemUIOp>(
                  loc, inputChannel,
                  b.create<arith::ConstantIndexOp>(loc, kTile)));
          Value patchNext =
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
              outputPanel, inputChannel, b.getIndexAttr(0), b.getIndexAttr(0)};
          SmallVector<OpFoldResult> weightSizes = {
              b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(paddedK),
              b.getIndexAttr(kTile)};
          SmallVector<OpFoldResult> weightStrides(4, b.getIndexAttr(1));
          Value weightSlice = b.create<memref::SubViewOp>(
              loc, stage.weight, weightOffsets, weightSizes, weightStrides);
          Value weightPack = b.create<memref::CollapseShapeOp>(
              loc, weightSlice,
              SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
          Value weightNext =
              mvinBank(b, loc, weightPack, iterStates[1], paddedK);
          Value first = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                inputChannel, zero);
          Value last = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, inputChannel,
              b.create<arith::ConstantIndexOp>(loc, stage.inputChannels - 1));
          Value resultNext =
              b.create<BankSMatMulOp>(
                   loc, iterStates[2].getType(), patchNext, weightNext,
                   iterStates[2],
                   createI64ConstU(b, loc, matrixRs2(kTile, kTile, paddedK)),
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

        if (stage.input == kernel.getInput()) {
          ensureStates();
          accumulateSource(source.banks.front(), 0, inputPanelCount,
                           source.panelRows);
        } else {
          ensureStates();
          for (size_t sourceBank = 0; sourceBank < source.banks.size();
               ++sourceBank) {
            int64_t sourcePanelBegin = sourceBank * source.panelsPerBank;
            int64_t sourcePanelEnd = std::min<int64_t>(
                inputPanelCount, sourcePanelBegin + source.panelsPerBank);
            accumulateSource(source.banks[sourceBank], sourcePanelBegin,
                             sourcePanelEnd, source.panelRows);
          }
        }
        releaseBank(b, loc, states[0]);
        releaseBank(b, loc, states[1]);
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value destinationSlot = b.create<arith::SubIOp>(
            loc, localPanel, b.create<arith::ConstantIndexOp>(loc, panelBegin));
        Value outputBase = b.create<arith::IndexCastOp>(
            loc, b.getI64Type(),
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(loc, destinationSlot,
                                        b.create<arith::ConstantIndexOp>(
                                            loc, destination.panelRows)),
                b.create<arith::ConstantIndexOp>(loc, destinationBase)));
        Value quantized = allocBank(b, loc, 1, 1);
        Value quantizedState =
            b.create<BankQuantI32ToI8Op>(
                 loc, quantized.getType(), states[2], scaleLoaded, quantized,
                 createI64Const(b, loc, side * side * 4),
                 createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                 b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                 b.getI64IntegerAttr(side),
                 b.getBoolAttr(stage.activation == 1))
                .getOutBankOut();
        releaseBank(b, loc, states[2]);
        releaseBank(b, loc, scaleLoaded);
        Value transformed;
        Value outputState = quantizedState;
        if (stage.activation == 2) {
          Value lutOutput = allocBank(b, loc, 1, 1);
          transformed = b.create<BankLutOp>(
              loc, lutOutput.getType(), quantizedState, lutLoaded, lutOutput,
              createI64Const(b, loc, side * side));
          outputState = transformed;
        }
        Value destinationNext =
            b.create<BankMaxPoolOp>(
                 loc, destinationState.getType(), outputState, destinationState,
                 createI64Const(b, loc, side * side), b.getI64IntegerAttr(side),
                 b.getI64IntegerAttr(side), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
                 createI64Const(b, loc, 0), outputBase,
                 createI64Const(b, loc, destinationStride),
                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                .getOutBankOut();
        releaseBank(b, loc, quantized);
        if (transformed)
          releaseBank(b, loc, transformed);
        b.create<scf::YieldOp>(loc, destinationNext);
        b.setInsertionPointAfter(outputPanelLoop);
      }
      if (lutLoaded)
        releaseBank(b, loc, lutLoaded);
      releaseTile(source);
      maskInvalidOutput();
      return success();
    };

    auto materializeStage = [&](int64_t stageIndex) -> LogicalResult {
      Stage &stage = stages[stageIndex];
      int64_t side = std::min<int64_t>(
          stage.add ? 1 : 2, std::min(stage.outputHeight, stage.outputWidth));
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
        TileBanks output = allocateTile(requestedPanels, side * side, zeroI8);
        if (output.banks.size() != 1 ||
            failed(emitInto(stageIndex, y, x, side, side, firstPanel,
                            requestedPanels, output, 0, side)))
          return failure();
        Value pack = b.create<memref::AllocOp>(
            loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
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
                b.create<arith::ConstantIndexOp>(loc, output.panelRows)),
            b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, localY, b.create<arith::ConstantIndexOp>(loc, side)),
                localX));
        Value value =
            b.create<memref::LoadOp>(loc, pack, ValueRange{row, lane});
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

      int64_t panelsPerBank = target.bankDepth / (side * side);
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
      if (traceMegaStages && stageIndex >= traceMegaStageStart &&
          (traceMegaStageLimit < 0 || stageIndex < traceMegaStageLimit)) {
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
      if (stage.multiply && stage.input != kernel.getInput()) {
        int64_t inputStage = producer.lookup(stage.input);
        if (!materialized.contains(inputStage) &&
            failed(materializeStage(inputStage)))
          return failure();
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
