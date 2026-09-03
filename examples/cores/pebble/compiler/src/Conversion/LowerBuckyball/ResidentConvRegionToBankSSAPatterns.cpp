//===- ResidentConvRegionToBankSSAPatterns.cpp ---------------------------===//

#include "Conversion/LowerBuckyball/LowerBuckyball.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

#include "Buckyball/BuckyballOps.h"
#include "Target/BuckyballTargetRegistry.h"
#include "Utils/BankUtils.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <tuple>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;

class ResidentConvRegionPattern : public OpRewritePattern<MegaKernelOp> {
public:
  ResidentConvRegionPattern(MLIRContext *context)
      : OpRewritePattern<MegaKernelOp>(context, 4) {}

  LogicalResult matchAndRewrite(MegaKernelOp kernel,
                                PatternRewriter &b) const override {
    if (kernel.getBody().empty())
      return kernel.emitError("MegaKernel region must contain one block");
    Block &body = kernel.getBody().front();
    if (body.without_terminator().empty() || !isa<MegaConv2dOp>(body.front()))
      return failure();
    if (!isa<MegaGlobalAvgPoolOp>(*std::prev(body.without_terminator().end())))
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
      bool pool;
      bool add;
      bool average;
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
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            output.getShape()[0] != 1 || !input.getElementType().isInteger(8) ||
            !weight.getElementType().isInteger(8) ||
            !bias.getElementType().isInteger(32) ||
            !scale.getElementType().isF32() ||
            !output.getElementType().isInteger(8) || conv.getActivation() < 0 ||
            conv.getActivation() > 1 || conv.getStride() <= 0 ||
            conv.getPadLow() < 0 || conv.getPadLow() != conv.getPadHigh())
          return conv.emitError("unsupported Conv stage in resident region");
        auto in = input.getShape();
        auto weights = weight.getShape();
        auto out = output.getShape();
        int64_t kernelSize = weights[0];
        if (weights !=
                ArrayRef<int64_t>({kernelSize, kernelSize, in[3], out[3]}) ||
            kernelSize <= 0 || kernelSize > 7 ||
            bias.getShape() != ArrayRef<int64_t>({out[3]}) ||
            scale.getShape() != ArrayRef<int64_t>({out[3]}) ||
            lut.getShape() != ArrayRef<int64_t>({1}) ||
            (in[1] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[1] ||
            (in[2] + 2 * conv.getPadLow() - kernelSize) / conv.getStride() +
                    1 !=
                out[2] ||
            out[3] % kTile != 0)
          return conv.emitError("Conv shape is inconsistent");
        stage.input = conv.getInput();
        stage.output = conv.getOutput();
        stage.weight = conv.getWeight();
        stage.bias = conv.getBias();
        stage.scale = conv.getScale();
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
      } else if (auto pool = dyn_cast<MegaMaxPool2dOp>(operation)) {
        auto input = dyn_cast<MemRefType>(pool.getInput().getType());
        auto output = dyn_cast<MemRefType>(pool.getOutput().getType());
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || pool.getFinalOutput() ||
            input.getShape()[0] != 1 || output.getShape()[0] != 1 ||
            !input.getElementType().isInteger(8) ||
            !output.getElementType().isInteger(8) || pool.getKernel() <= 0 ||
            pool.getKernel() > 8 || pool.getStride() <= 0 ||
            pool.getPadding() < 0)
          return pool.emitError("unsupported MaxPool stage in resident region");
        auto in = input.getShape();
        auto out = output.getShape();
        if (in[3] != out[3] ||
            (in[1] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                out[1] ||
            (in[2] + 2 * pool.getPadding() - pool.getKernel()) /
                        pool.getStride() +
                    1 !=
                out[2] ||
            out[3] % kTile != 0)
          return pool.emitError("MaxPool shape is inconsistent");
        stage.input = pool.getInput();
        stage.output = pool.getOutput();
        stage.inputHeight = in[1];
        stage.inputWidth = in[2];
        stage.inputChannels = in[3];
        stage.outputHeight = out[1];
        stage.outputWidth = out[2];
        stage.outputChannels = out[3];
        stage.kernel = pool.getKernel();
        stage.stride = pool.getStride();
        stage.padding = pool.getPadding();
        stage.pool = true;
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
            lhs.getShape()[0] != 1 || lhs.getShape()[3] % kTile != 0 ||
            !lhs.getElementType().isInteger(8) || add.getActivation() < 0 ||
            add.getActivation() > 1 || !std::isfinite(lhsScale) ||
            !std::isfinite(rhsScale) || !std::isfinite(outputScale) ||
            lhsScale <= 0.0f || rhsScale <= 0.0f || outputScale <= 0.0f)
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
      } else if (auto average = dyn_cast<MegaGlobalAvgPoolOp>(operation)) {
        auto input = dyn_cast<MemRefType>(average.getInput().getType());
        auto output = dyn_cast<MemRefType>(average.getOutput().getType());
        float inputScale = average.getInputScale().convertToFloat();
        float outputScale = average.getOutputScale().convertToFloat();
        if (!input || !output || !input.hasStaticShape() ||
            !output.hasStaticShape() || input.getRank() != 4 ||
            output.getRank() != 4 || input.getShape()[0] != 1 ||
            input.getShape()[3] % kTile != 0 ||
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
            "resident Conv region supports Conv, MaxPool, INT8 Add, and "
            "GlobalAvgPool stages");
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

    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth != 64 ||
        target.bankNum != 24)
      return kernel.emitError(
          "resident Conv region requires Pebble 24x64x128 banks");
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

    DenseMap<Operation *, Value> packedWeights;
    DenseMap<Operation *, Value> packedBiases;
    DenseMap<Operation *, Value> packedScales;
    std::map<std::tuple<int64_t, int64_t, int64_t>, Value> packedInputs;
    for (Stage &stage : stages) {
      if (stage.pool || stage.add || stage.average)
        continue;
      int64_t outputPanels = stage.outputChannels / kTile;
      int64_t kernelElements = stage.kernel * stage.kernel;
      int64_t paddedK = (kernelElements + kTile - 1) / kTile * kTile;
      Value weightPack = b.create<memref::AllocOp>(
          loc,
          MemRefType::get({outputPanels, stage.inputChannels, paddedK, kTile},
                          b.getI8Type()));
      Value biasPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getI32Type()));
      Value scalePack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getF32Type()));
      hostPacks.append({weightPack, biasPack, scalePack});
      packedWeights[stage.op] = weightPack;
      packedBiases[stage.op] = biasPack;
      packedScales[stage.op] = scalePack;
      b.create<linalg::FillOp>(loc, zeroI8, weightPack);
      b.create<linalg::FillOp>(loc, zeroI32, biasPack);
      b.create<linalg::FillOp>(loc, oneF32, scalePack);

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
      b.setInsertionPointAfter(outputLaneLoop);

      auto inputChannelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.inputChannels),
          one);
      b.setInsertionPointToStart(inputChannelLoop.getBody());
      Value inputChannel = inputChannelLoop.getInductionVar();
      auto kernelYLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.kernel), one);
      b.setInsertionPointToStart(kernelYLoop.getBody());
      Value kernelY = kernelYLoop.getInductionVar();
      auto kernelXLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, stage.kernel), one);
      b.setInsertionPointToStart(kernelXLoop.getBody());
      Value kernelX = kernelXLoop.getInductionVar();
      auto laneLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
      b.setInsertionPointToStart(laneLoop.getBody());
      Value lane = laneLoop.getInductionVar();
      Value sourceChannel = b.create<arith::AddIOp>(loc, channelBase, lane);
      Value weight = b.create<memref::LoadOp>(
          loc, stage.weight,
          ValueRange{kernelY, kernelX, inputChannel, sourceChannel});
      Value weightRow = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(
              loc, kernelY,
              b.create<arith::ConstantIndexOp>(loc, stage.kernel)),
          kernelX);
      b.create<memref::StoreOp>(
          loc, weight, weightPack,
          ValueRange{outputPanel, inputChannel, weightRow, lane});
      b.setInsertionPointAfter(outputPanelLoop);
    }

    struct TileBanks {
      SmallVector<Value> banks;
      int64_t panelRows;
      int64_t panelsPerBank;
      int64_t panelCount;
    };

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

    std::function<LogicalResult(int64_t, int64_t, int64_t, int64_t, int64_t,
                                Value, int64_t, TileBanks &, int64_t, int64_t)>
        emitInto;
    emitInto = [&](int64_t stageIndex, int64_t y0, int64_t x0, int64_t height,
                   int64_t width, Value firstPanel, int64_t panelCount,
                   TileBanks &destination, int64_t destinationBase,
                   int64_t destinationStride) -> LogicalResult {
      Stage &stage = stages[stageIndex];
      int64_t totalPanels = stage.outputChannels / kTile;
      if (y0 < 0 || x0 < 0 || height <= 0 || width <= 0 ||
          y0 + height > stage.outputHeight || x0 + width > stage.outputWidth ||
          panelCount <= 0 || panelCount > totalPanels ||
          destination.panelCount != panelCount || destinationBase < 0 ||
          destinationStride < width ||
          destinationBase + (height - 1) * destinationStride + width >
              destination.panelRows)
        return stage.op->emitError("invalid resident tile request");

      if (stage.add) {
        Value lhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.lhsScale / stage.outputScale));
        Value rhsRatio = b.create<arith::ConstantOp>(
            loc, b.getF32Type(),
            b.getF32FloatAttr(stage.rhsScale / stage.outputScale));
        TileBanks lhs = allocateTile(panelCount, destination.panelRows, zeroI8);
        if (failed(emitInto(producer.lookup(stage.input), y0, x0, height, width,
                            firstPanel, panelCount, lhs, destinationBase,
                            destinationStride)))
          return failure();
        TileBanks rhs = allocateTile(panelCount, destination.panelRows, zeroI8);
        if (failed(emitInto(producer.lookup(stage.rhs), y0, x0, height, width,
                            firstPanel, panelCount, rhs, destinationBase,
                            destinationStride)))
          return failure();
        if (lhs.banks.size() != destination.banks.size() ||
            rhs.banks.size() != destination.banks.size())
          return stage.op->emitError("INT8 Add bank layouts do not match");
        for (size_t index = 0; index < destination.banks.size(); ++index) {
          destination.banks[index] =
              b.create<BankInt8AddOp>(
                   loc, destination.banks[index].getType(), lhs.banks[index],
                   rhs.banks[index], destination.banks[index],
                   createI64Const(b, loc, target.bankDepth), lhsRatio, rhsRatio,
                   b.getBoolAttr(stage.activation == 1))
                  .getOutputBankOut();
        }
        releaseTile(lhs);
        releaseTile(rhs);
        return success();
      }

      if (stage.average) {
        if (y0 != 0 || x0 != 0 || height != 1 || width != 1 ||
            destinationBase != 0 || destinationStride != 1)
          return stage.op->emitError("invalid GlobalAvgPool output tile");
        int64_t inputRows = stage.inputHeight * stage.inputWidth;
        if (inputRows > target.bankDepth)
          return stage.op->emitError(
              "GlobalAvgPool input panel exceeds one bank");
        constexpr int64_t panelsPerGroup = 4;
        for (int64_t group = 0; group < panelCount; group += panelsPerGroup) {
          int64_t groupPanels = std::min(panelsPerGroup, panelCount - group);
          Value groupFirstPanel = b.create<arith::AddIOp>(
              loc, firstPanel, b.create<arith::ConstantIndexOp>(loc, group));
          TileBanks source = allocateTile(groupPanels, inputRows, zeroI8);
          if (failed(emitInto(producer.lookup(stage.input), 0, 0,
                              stage.inputHeight, stage.inputWidth,
                              groupFirstPanel, groupPanels, source, 0,
                              stage.inputWidth)))
            return failure();
          for (int64_t localPanel = 0; localPanel < groupPanels; ++localPanel) {
            Value onesPack = b.create<memref::AllocOp>(
                loc, MemRefType::get({4, kTile}, b.getI8Type()));
            Value biasPack = b.create<memref::AllocOp>(
                loc, MemRefType::get({4, 4}, b.getI32Type()));
            Value scalePack = b.create<memref::AllocOp>(
                loc, MemRefType::get({4, 4}, b.getF32Type()));
            hostPacks.push_back(onesPack);
            hostPacks.push_back(biasPack);
            hostPacks.push_back(scalePack);
            b.create<linalg::FillOp>(loc, zeroI8, onesPack);
            b.create<linalg::FillOp>(loc, zeroI32, biasPack);
            Value ratio = b.create<arith::ConstantOp>(
                loc, b.getF32Type(),
                b.getF32FloatAttr(stage.lhsScale /
                                  (inputRows * stage.outputScale)));
            b.create<linalg::FillOp>(loc, ratio, scalePack);
            auto onesLoop = b.create<scf::ForOp>(
                loc, zero, b.create<arith::ConstantIndexOp>(loc, inputRows),
                one);
            b.setInsertionPointToStart(onesLoop.getBody());
            Value index = onesLoop.getInductionVar();
            Value sixteen = b.create<arith::ConstantIndexOp>(loc, kTile);
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
                loc, biasLoaded.getType(), biasLoaded);
            Value scaleBank = allocBank(b, loc, 1, 1);
            Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
            Value result = allocBank(b, loc, 1, 1);
            result = b.create<BankSMatMulOp>(
                          loc, result.getType(), onesLoaded,
                          source.banks[localPanel], result,
                          createI64ConstU(
                              b, loc, matrixRs2(1, kTile, target.bankDepth)),
                          b.getBoolAttr(true), b.getBoolAttr(true))
                         .getWrBankOut();
            int64_t destinationPanel = group + localPanel;
            int64_t destinationBank =
                destinationPanel / destination.panelsPerBank;
            int64_t destinationSlot =
                destinationPanel % destination.panelsPerBank;
            destination.banks[destinationBank] =
                b.create<BankQuantI32ToI8Op>(
                     loc, destination.banks[destinationBank].getType(), result,
                     scaleLoaded, destination.banks[destinationBank],
                     createI64Const(b, loc, 4),
                     b.getI64IntegerAttr(destinationSlot),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                     b.getI64IntegerAttr(1), b.getBoolAttr(false))
                    .getOutBankOut();
            releaseBank(b, loc, onesLoaded);
            releaseBank(b, loc, biasState);
            releaseBank(b, loc, scaleLoaded);
            releaseBank(b, loc, result);
          }
          releaseTile(source);
        }
        return success();
      }

      int64_t maxSide = std::min<int64_t>({4, height, width});
      while (maxSide > 0) {
        int64_t inputSide = (maxSide - 1) * stage.stride + stage.kernel;
        int64_t inputPanelRows = inputSide * inputSide;
        int64_t inputPanels = stage.inputChannels / kTile;
        if (stage.inputChannels % kTile)
          ++inputPanels;
        int64_t panelsPerBank = inputPanelRows <= target.bankDepth
                                    ? target.bankDepth / inputPanelRows
                                    : 0;
        int64_t inputBanks =
            panelsPerBank ? (inputPanels + panelsPerBank - 1) / panelsPerBank
                          : target.bankNum + 1;
        int64_t reservedBanks = stage.input == kernel.getInput() ? 6 : 11;
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
        if (width > side &&
            failed(emitInto(stageIndex, y0, x0 + side, side, width - side,
                            firstPanel, panelCount, destination,
                            destinationBase + side, destinationStride)))
          return failure();
        if (height > side &&
            failed(emitInto(stageIndex, y0 + side, x0, height - side, width,
                            firstPanel, panelCount, destination,
                            destinationBase + side * destinationStride,
                            destinationStride)))
          return failure();
        return success();
      }

      int64_t inputSide = (side - 1) * stage.stride + stage.kernel;
      int64_t sourceY = y0 * stage.stride - stage.padding;
      int64_t sourceX = x0 * stage.stride - stage.padding;
      int64_t inputPanelCount =
          stage.pool ? panelCount : (stage.inputChannels + kTile - 1) / kTile;
      TileBanks source;
      if (stage.input == kernel.getInput()) {
        if (inputPanelCount != 1)
          return stage.op->emitError(
              "resident region input must fit one 16-channel panel");
        source.panelRows = inputSide * inputSide;
        source.panelsPerBank = target.bankDepth / source.panelRows;
        source.panelCount = inputPanelCount;
        auto key = std::make_tuple(sourceY, sourceX, inputSide);
        Value pack;
        auto existing = packedInputs.find(key);
        if (existing != packedInputs.end()) {
          pack = existing->second;
        } else {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(kernel);
          pack = b.create<memref::AllocOp>(
              loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
          hostPacks.push_back(pack);
          b.create<linalg::FillOp>(loc, zeroI8, pack);
          int64_t validY = std::max<int64_t>(0, sourceY);
          int64_t validX = std::max<int64_t>(0, sourceX);
          int64_t validYEnd = std::min(stage.inputHeight, sourceY + inputSide);
          int64_t validXEnd = std::min(stage.inputWidth, sourceX + inputSide);
          auto yLoop = b.create<scf::ForOp>(
              loc, zero,
              b.create<arith::ConstantIndexOp>(loc, validYEnd - validY), one);
          b.setInsertionPointToStart(yLoop.getBody());
          Value localY = yLoop.getInductionVar();
          auto xLoop = b.create<scf::ForOp>(
              loc, zero,
              b.create<arith::ConstantIndexOp>(loc, validXEnd - validX), one);
          b.setInsertionPointToStart(xLoop.getBody());
          Value localX = xLoop.getInductionVar();
          auto laneLoop = b.create<scf::ForOp>(
              loc, zero,
              b.create<arith::ConstantIndexOp>(loc, stage.inputChannels), one);
          b.setInsertionPointToStart(laneLoop.getBody());
          Value lane = laneLoop.getInductionVar();
          Value globalY = b.create<arith::AddIOp>(
              loc, localY, b.create<arith::ConstantIndexOp>(loc, validY));
          Value globalX = b.create<arith::AddIOp>(
              loc, localX, b.create<arith::ConstantIndexOp>(loc, validX));
          Value row = b.create<arith::AddIOp>(
              loc,
              b.create<arith::MulIOp>(
                  loc,
                  b.create<arith::AddIOp>(
                      loc, localY,
                      b.create<arith::ConstantIndexOp>(loc, validY - sourceY)),
                  b.create<arith::ConstantIndexOp>(loc, inputSide)),
              b.create<arith::AddIOp>(
                  loc, localX,
                  b.create<arith::ConstantIndexOp>(loc, validX - sourceX)));
          Value value = b.create<memref::LoadOp>(
              loc, kernel.getInput(), ValueRange{zero, globalY, globalX, lane});
          b.create<memref::StoreOp>(loc, value, pack, ValueRange{row, lane});
          b.setInsertionPointAfter(yLoop);
          packedInputs[key] = pack;
        }
        Value bank = allocBank(b, loc, 1, 1);
        source.banks.push_back(mvinBank(b, loc, pack, bank, target.bankDepth));
      } else {
        source = allocateTile(inputPanelCount, inputSide * inputSide,
                              stage.pool ? minI8 : zeroI8);
        int64_t validY = std::max<int64_t>(0, sourceY);
        int64_t validX = std::max<int64_t>(0, sourceX);
        int64_t validYEnd = std::min(stage.inputHeight, sourceY + inputSide);
        int64_t validXEnd = std::min(stage.inputWidth, sourceX + inputSide);
        if (validY < validYEnd && validX < validXEnd &&
            failed(emitInto(
                producer.lookup(stage.input), validY, validX,
                validYEnd - validY, validXEnd - validX,
                stage.pool ? firstPanel
                           : Value(b.create<arith::ConstantIndexOp>(loc, 0)),
                inputPanelCount, source,
                (validY - sourceY) * inputSide + validX - sourceX, inputSide)))
          return failure();
      }

      if (stage.pool) {
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t sourceBank = localPanel / source.panelsPerBank;
          int64_t sourceSlot = localPanel % source.panelsPerBank;
          int64_t destinationBank = localPanel / destination.panelsPerBank;
          int64_t destinationSlot = localPanel % destination.panelsPerBank;
          destination.banks[destinationBank] =
              b.create<BankMaxPoolOp>(
                   loc, destination.banks[destinationBank].getType(),
                   source.banks[sourceBank], destination.banks[destinationBank],
                   createI64Const(b, loc, side * side),
                   b.getI64IntegerAttr(inputSide), b.getI64IntegerAttr(side),
                   b.getI64IntegerAttr(stage.kernel),
                   b.getI64IntegerAttr(stage.stride), b.getI64IntegerAttr(0),
                   b.getI64IntegerAttr(sourceSlot * source.panelRows),
                   b.getI64IntegerAttr(destinationSlot * destination.panelRows +
                                       destinationBase),
                   b.getI64IntegerAttr(destinationStride))
                  .getOutBankOut();
        }
        releaseTile(source);
        return success();
      }

      int64_t kernelElements = stage.kernel * stage.kernel;
      int64_t paddedK = (kernelElements + kTile - 1) / kTile * kTile;
      for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
        Value outputPanel = b.create<arith::AddIOp>(
            loc, firstPanel, b.create<arith::ConstantIndexOp>(loc, localPanel));
        Value channelBase = b.create<arith::MulIOp>(
            loc, outputPanel, b.create<arith::ConstantIndexOp>(loc, kTile));
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
        Value biasState =
            b.create<BankSMatMulBiasOp>(loc, biasLoaded.getType(), biasLoaded);
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value patch = allocBank(b, loc, 1, 1);
        Value weightBank = allocBank(b, loc, 1, 1);
        Value result = allocBank(b, loc, 1, 1);
        for (int64_t inputChannel = 0; inputChannel < stage.inputChannels;
             ++inputChannel) {
          int64_t inputPanel = inputChannel / kTile;
          int64_t sourceBank = inputPanel / source.panelsPerBank;
          int64_t sourceSlot = inputPanel % source.panelsPerBank;
          patch = b.create<BankIm2colOp>(
                       loc, patch.getType(), source.banks[sourceBank], patch,
                       createI64Const(b, loc, inputSide),
                       createI64Const(b, loc, stage.kernel),
                       createI64Const(b, loc, stage.stride),
                       createI64Const(b, loc, 0),
                       createI64Const(b, loc, sourceSlot * source.panelRows),
                       createI64Const(b, loc, inputChannel % kTile),
                       b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                       b.getI64IntegerAttr(0), b.getI64IntegerAttr(side * side))
                      .getOutBankOut();
          SmallVector<OpFoldResult> weightOffsets = {
              outputPanel, b.getIndexAttr(inputChannel), b.getIndexAttr(0),
              b.getIndexAttr(0)};
          SmallVector<OpFoldResult> weightSizes = {
              b.getIndexAttr(1), b.getIndexAttr(1), b.getIndexAttr(paddedK),
              b.getIndexAttr(kTile)};
          SmallVector<OpFoldResult> weightStrides(4, b.getIndexAttr(1));
          Value weightSlice = b.create<memref::SubViewOp>(
              loc, packedWeights.lookup(stage.op), weightOffsets, weightSizes,
              weightStrides);
          Value weightPack = b.create<memref::CollapseShapeOp>(
              loc, weightSlice,
              SmallVector<ReassociationIndices>{{0, 1, 2}, {3}});
          weightBank = mvinBank(b, loc, weightPack, weightBank, paddedK);
          result =
              b.create<BankSMatMulOp>(
                   loc, result.getType(), patch, weightBank, result,
                   createI64ConstU(b, loc, matrixRs2(kTile, kTile, paddedK)),
                   b.getBoolAttr(inputChannel == 0),
                   b.getBoolAttr(inputChannel + 1 == stage.inputChannels))
                  .getWrBankOut();
        }
        int64_t destinationBank = localPanel / destination.panelsPerBank;
        int64_t destinationSlot = localPanel % destination.panelsPerBank;
        destination.banks[destinationBank] =
            b.create<BankQuantI32ToI8Op>(
                 loc, destination.banks[destinationBank].getType(), result,
                 scaleLoaded, destination.banks[destinationBank],
                 createI64Const(b, loc, side * side * 4),
                 b.getI64IntegerAttr(destinationSlot * destination.panelRows +
                                     destinationBase),
                 b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                 b.getI64IntegerAttr(destinationStride),
                 b.getBoolAttr(stage.activation == 1))
                .getOutBankOut();
        releaseBank(b, loc, biasState);
        releaseBank(b, loc, scaleLoaded);
        releaseBank(b, loc, patch);
        releaseBank(b, loc, weightBank);
        releaseBank(b, loc, result);
      }
      releaseTile(source);
      return success();
    };

    Stage &finalStage = stages.back();
    int64_t finalPanels = finalStage.outputChannels / kTile;
    auto panelLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, finalPanels), one);
    b.setInsertionPointToStart(panelLoop.getBody());
    Value panel = panelLoop.getInductionVar();
    TileBanks output = allocateTile(1, 1, zeroI8);
    if (output.banks.size() != 1 ||
        failed(emitInto(stages.size() - 1, 0, 0, 1, 1, panel, 1, output, 0, 1)))
      return failure();
    Value packed = b.create<memref::AllocOp>(
        loc, MemRefType::get({1, kTile}, b.getI8Type()));
    Value stored = mvoutBank(b, loc, packed, output.banks.front(), 1);
    b.create<FenceOp>(loc);
    auto outputLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
    b.setInsertionPointToStart(outputLoop.getBody());
    Value lane = outputLoop.getInductionVar();
    Value channel = b.create<arith::AddIOp>(
        loc,
        b.create<arith::MulIOp>(loc, panel,
                                b.create<arith::ConstantIndexOp>(loc, kTile)),
        lane);
    Value value = b.create<memref::LoadOp>(loc, packed, ValueRange{zero, lane});
    b.create<memref::StoreOp>(loc, value, kernel.getOutput(),
                              ValueRange{zero, zero, zero, channel});
    b.setInsertionPointAfter(outputLoop);
    releaseBank(b, loc, stored);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.create<memref::DeallocOp>(loc, packed);
    b.setInsertionPointAfter(panelLoop);
    b.eraseOp(kernel);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populatePebbleResidentConvRegionToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ResidentConvRegionPattern>(patterns.getContext());
}
} // namespace mlir::buddy
