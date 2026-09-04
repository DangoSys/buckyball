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

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <cmath>
#include <functional>

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
        int64_t kernelSize = conv.getKernel();
        int64_t paddedKernel =
            ((kernelSize * kernelSize + kTile - 1) / kTile) * kTile;
        if (kernelSize <= 0 || kernelSize > 7 ||
            weights != ArrayRef<int64_t>(
                           {out[3] / kTile, in[3], paddedKernel, kTile}) ||
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
    for (Stage &stage : stages) {
      if (stage.pool || stage.add || stage.average)
        continue;
      int64_t outputPanels = stage.outputChannels / kTile;
      Value biasPack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getI32Type()));
      Value scalePack = b.create<memref::AllocOp>(
          loc, MemRefType::get({outputPanels, 4, 4}, b.getF32Type()));
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
      b.setInsertionPointAfter(outputPanelLoop);
    }

    Value zeroBank = allocBank(b, loc, 1, 1);
    zeroBank = mvinBank(b, loc, zeroPack, zeroBank, target.bankDepth);
    DenseSet<int64_t> materialized;

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

    std::function<LogicalResult(int64_t, Value, Value, int64_t, int64_t, Value,
                                int64_t, TileBanks &, int64_t, int64_t)>
        emitInto;
    emitInto = [&](int64_t stageIndex, Value y0, Value x0, int64_t height,
                   int64_t width, Value firstPanel, int64_t panelCount,
                   TileBanks &destination, int64_t destinationBase,
                   int64_t destinationStride) -> LogicalResult {
      Stage &stage = stages[stageIndex];
      int64_t totalPanels = stage.outputChannels / kTile;
      if (height <= 0 || width <= 0 || panelCount <= 0 ||
          panelCount > totalPanels || destination.panelCount != panelCount ||
          destinationBase < 0 || destinationStride < width ||
          destinationBase + (height - 1) * destinationStride + width >
              destination.panelRows)
        return stage.op->emitError("invalid resident tile request");

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
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t bankIndex = localPanel / destination.panelsPerBank;
          int64_t bankSlot = localPanel % destination.panelsPerBank;
          for (int64_t localY = 0; localY < height; ++localY) {
            for (int64_t localX = 0; localX < width; ++localX) {
              Value globalY = b.create<arith::AddIOp>(
                  loc, y0, b.create<arith::ConstantIndexOp>(loc, localY));
              Value globalX = b.create<arith::AddIOp>(
                  loc, x0, b.create<arith::ConstantIndexOp>(loc, localX));
              Value yInvalid = b.create<arith::OrIOp>(
                  loc,
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                          globalY, zero),
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                          globalY,
                                          b.create<arith::ConstantIndexOp>(
                                              loc, stage.outputHeight)));
              Value xInvalid = b.create<arith::OrIOp>(
                  loc,
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                          globalX, zero),
                  b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                          globalX,
                                          b.create<arith::ConstantIndexOp>(
                                              loc, stage.outputWidth)));
              auto invalid = b.create<scf::IfOp>(
                  loc, b.create<arith::OrIOp>(loc, yInvalid, xInvalid), false);
              b.setInsertionPointToStart(&invalid.getThenRegion().front());
              b.create<BankMaxPoolOp>(
                  loc, destination.banks[bankIndex].getType(), zeroBank,
                  destination.banks[bankIndex], createI64Const(b, loc, 1),
                  b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                  b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                  b.getI64IntegerAttr(0), createI64Const(b, loc, 0),
                  createI64Const(b, loc,
                                 bankSlot * destination.panelRows +
                                     destinationBase +
                                     localY * destinationStride + localX),
                  createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                  b.getI64IntegerAttr(0));
              b.setInsertionPointAfter(invalid);
            }
          }
        }
      };

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
          b.create<BankInt8AddOp>(
              loc, destination.banks[index].getType(), lhs.banks[index],
              rhs.banks[index], destination.banks[index],
              createI64Const(b, loc, target.bankDepth), lhsRatio, rhsRatio,
              b.getBoolAttr(stage.activation == 1));
        }
        releaseTile(lhs);
        releaseTile(rhs);
        maskInvalidOutput();
        return success();
      }

      if (stage.average) {
        if (height != 1 || width != 1 || destinationBase != 0 ||
            destinationStride != 1)
          return stage.op->emitError("invalid GlobalAvgPool output tile");
        int64_t inputRows = stage.inputHeight * stage.inputWidth;
        if (inputRows > target.bankDepth)
          return stage.op->emitError(
              "GlobalAvgPool input panel exceeds one bank");
        constexpr int64_t panelsPerGroup = 4;
        if (panelCount % panelsPerGroup || destination.banks.size() != 1)
          return stage.op->emitError("GlobalAvgPool requires groups of four "
                                     "panels in one output bank");
        auto groupLoop = b.create<scf::ForOp>(
            loc, zero, b.create<arith::ConstantIndexOp>(loc, panelCount),
            b.create<arith::ConstantIndexOp>(loc, panelsPerGroup),
            ValueRange{destination.banks.front()});
        b.setInsertionPointToStart(groupLoop.getBody());
        Value group = groupLoop.getInductionVar();
        Value destinationState = groupLoop.getRegionIterArgs().front();
        Value groupFirstPanel = b.create<arith::AddIOp>(loc, firstPanel, group);
        TileBanks source = allocateTile(panelsPerGroup, inputRows, zeroI8);
        if (failed(emitInto(producer.lookup(stage.input), zero, zero,
                            stage.inputHeight, stage.inputWidth,
                            groupFirstPanel, panelsPerGroup, source, 0,
                            stage.inputWidth)))
          return failure();
        for (int64_t localPanel = 0; localPanel < panelsPerGroup;
             ++localPanel) {
          Value onesPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({4, kTile}, b.getI8Type()));
          Value biasPack = b.create<memref::AllocOp>(
              loc, MemRefType::get({4, 4}, b.getI32Type()));
          Value scalePack = b.create<memref::AllocOp>(
              loc, MemRefType::get({4, 4}, b.getF32Type()));
          b.create<linalg::FillOp>(loc, zeroI8, onesPack);
          b.create<linalg::FillOp>(loc, zeroI32, biasPack);
          Value ratio = b.create<arith::ConstantOp>(
              loc, b.getF32Type(),
              b.getF32FloatAttr(stage.lhsScale /
                                (inputRows * stage.outputScale)));
          b.create<linalg::FillOp>(loc, ratio, scalePack);
          auto onesLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, inputRows), one);
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
              loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
          Value scaleBank = allocBank(b, loc, 1, 1);
          Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
          Value result = allocBank(b, loc, 1, 1);
          result = b.create<BankSMatMulOp>(
                        loc, result.getType(), onesLoaded,
                        source.banks[localPanel], result,
                        createI64ConstU(b, loc,
                                        matrixRs2(1, kTile, target.bankDepth)),
                        createI1Const(b, loc, true),
                        createI1Const(b, loc, true), createI64Const(b, loc, 0))
                       .getWrBankOut();
          Value quantized = allocBank(b, loc, 1, 1);
          quantized = b.create<BankQuantI32ToI8Op>(
                           loc, quantized.getType(), result, scaleLoaded,
                           quantized, createI64Const(b, loc, 4),
                           createI64Const(b, loc, 0), createI64Const(b, loc, 0),
                           b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                           b.getI64IntegerAttr(1), b.getBoolAttr(false))
                          .getOutBankOut();
          Value destinationBase = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::AddIOp>(
                  loc, group,
                  b.create<arith::ConstantIndexOp>(loc, localPanel)));
          destinationState =
              b.create<BankMaxPoolOp>(
                   loc, destinationState.getType(), quantized, destinationState,
                   createI64Const(b, loc, 1), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                   b.getI64IntegerAttr(1), b.getI64IntegerAttr(0),
                   createI64Const(b, loc, 0), destinationBase,
                   createI64Const(b, loc, 1), b.getI64IntegerAttr(0),
                   b.getI64IntegerAttr(0))
                  .getOutBankOut();
          releaseBank(b, loc, onesLoaded);
          releaseBank(b, loc, biasState);
          releaseBank(b, loc, scaleLoaded);
          releaseBank(b, loc, result);
          releaseBank(b, loc, quantized);
          b.create<memref::DeallocOp>(loc, onesPack);
          b.create<memref::DeallocOp>(loc, biasPack);
          b.create<memref::DeallocOp>(loc, scalePack);
        }
        releaseTile(source);
        b.create<scf::YieldOp>(loc, destinationState);
        b.setInsertionPointAfter(groupLoop);
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
      } else {
        source = allocateTile(inputPanelCount, inputSide * inputSide,
                              stage.pool ? minI8 : zeroI8);
        if (failed(emitInto(
                producer.lookup(stage.input), sourceY, sourceX, inputSide,
                inputSide,
                stage.pool ? firstPanel
                           : Value(b.create<arith::ConstantIndexOp>(loc, 0)),
                inputPanelCount, source, 0, inputSide)))
          return failure();
      }

      if (stage.pool) {
        for (int64_t localPanel = 0; localPanel < panelCount; ++localPanel) {
          int64_t sourceBank = localPanel / source.panelsPerBank;
          int64_t sourceSlot = localPanel % source.panelsPerBank;
          int64_t destinationBank = localPanel / destination.panelsPerBank;
          int64_t destinationSlot = localPanel % destination.panelsPerBank;
          b.create<BankMaxPoolOp>(
              loc, destination.banks[destinationBank].getType(),
              source.banks[sourceBank], destination.banks[destinationBank],
              createI64Const(b, loc, side * side),
              b.getI64IntegerAttr(inputSide), b.getI64IntegerAttr(side),
              b.getI64IntegerAttr(stage.kernel),
              b.getI64IntegerAttr(stage.stride), b.getI64IntegerAttr(0),
              createI64Const(b, loc, sourceSlot * source.panelRows),
              createI64Const(b, loc,
                             destinationSlot * destination.panelRows +
                                 destinationBase),
              createI64Const(b, loc, destinationStride), b.getI64IntegerAttr(0),
              b.getI64IntegerAttr(0));
        }
        releaseTile(source);
        maskInvalidOutput();
        return success();
      }

      int64_t kernelElements = stage.kernel * stage.kernel;
      int64_t paddedK = (kernelElements + kTile - 1) / kTile * kTile;
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

        Value biasBank = allocBank(b, loc, 1, 1);
        Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
        Value biasState = b.create<BankSMatMulBiasOp>(
            loc, biasLoaded.getType(), biasLoaded, createI64Const(b, loc, 0));
        Value scaleBank = allocBank(b, loc, 1, 1);
        Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);
        Value patch = allocBank(b, loc, 1, 1);
        Value weightBank = allocBank(b, loc, 1, 1);
        Value result = allocBank(b, loc, 1, 1);
        SmallVector<Value> states{patch, weightBank, result};
        for (size_t sourceBank = 0; sourceBank < source.banks.size();
             ++sourceBank) {
          int64_t sourcePanelBegin = sourceBank * source.panelsPerBank;
          int64_t sourcePanelEnd = std::min<int64_t>(
              inputPanelCount, sourcePanelBegin + source.panelsPerBank);
          int64_t channelBegin = sourcePanelBegin * kTile;
          int64_t channelEnd =
              std::min(stage.inputChannels, sourcePanelEnd * kTile);
          auto channelLoop = b.create<scf::ForOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, channelBegin),
              b.create<arith::ConstantIndexOp>(loc, channelEnd), one,
              ValueRange(states));
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
                  b.create<arith::ConstantIndexOp>(loc, source.panelRows)));
          Value inputLane = b.create<arith::IndexCastOp>(
              loc, b.getI64Type(),
              b.create<arith::RemUIOp>(
                  loc, inputChannel,
                  b.create<arith::ConstantIndexOp>(loc, kTile)));
          Value patchNext =
              b.create<BankIm2colOp>(
                   loc, iterStates[0].getType(), source.banks[sourceBank],
                   iterStates[0], createI64Const(b, loc, inputSide),
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
        }
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
        Value destinationNext =
            b.create<BankMaxPoolOp>(
                 loc, destinationState.getType(), quantizedState,
                 destinationState, createI64Const(b, loc, side * side),
                 b.getI64IntegerAttr(side), b.getI64IntegerAttr(side),
                 b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                 b.getI64IntegerAttr(0), createI64Const(b, loc, 0), outputBase,
                 createI64Const(b, loc, destinationStride),
                 b.getI64IntegerAttr(0), b.getI64IntegerAttr(0))
                .getOutBankOut();
        releaseBank(b, loc, biasState);
        releaseBank(b, loc, scaleLoaded);
        releaseBank(b, loc, patch);
        releaseBank(b, loc, weightBank);
        releaseBank(b, loc, result);
        releaseBank(b, loc, quantized);
        b.create<scf::YieldOp>(loc, destinationNext);
        b.setInsertionPointAfter(outputPanelLoop);
      }
      releaseTile(source);
      maskInvalidOutput();
      return success();
    };

    auto materializeStage = [&](int64_t stageIndex) -> LogicalResult {
      Stage &stage = stages[stageIndex];
      constexpr int64_t side = 2;
      int64_t panelCount = stage.outputChannels / kTile;
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
      auto panelLoop = b.create<scf::ForOp>(
          loc, zero, b.create<arith::ConstantIndexOp>(loc, panelCount), one);
      b.setInsertionPointToStart(panelLoop.getBody());
      Value panel = panelLoop.getInductionVar();
      TileBanks output = allocateTile(1, side * side, zeroI8);
      if (failed(emitInto(stageIndex, y, x, side, side, panel, 1, output, 0,
                          side)))
        return failure();

      Value pack = b.create<memref::AllocOp>(
          loc, MemRefType::get({target.bankDepth, kTile}, b.getI8Type()));
      mvoutBank(b, loc, pack, output.banks.front(), target.bankDepth);
      b.create<FenceOp>(loc);
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
          b.create<arith::MulIOp>(loc, panel,
                                  b.create<arith::ConstantIndexOp>(loc, kTile)),
          lane);
      Value row = b.create<arith::AddIOp>(
          loc,
          b.create<arith::MulIOp>(loc, localY,
                                  b.create<arith::ConstantIndexOp>(loc, side)),
          localX);
      Value value = b.create<memref::LoadOp>(loc, pack, ValueRange{row, lane});
      b.create<memref::StoreOp>(loc, value, stage.output,
                                ValueRange{zero, globalY, globalX, channel});
      b.setInsertionPointAfter(laneLoop);
      b.setInsertionPointAfter(valid);
      b.setInsertionPointAfter(localYLoop);
      b.create<memref::DeallocOp>(loc, pack);
      releaseTile(output);
      b.setInsertionPointAfter(panelLoop);
      b.setInsertionPointAfter(xLoop);
      b.setInsertionPointAfter(yLoop);
      materialized.insert(stageIndex);
      return success();
    };

    for (auto [stageIndex, stage] : llvm::enumerate(stages)) {
      if ((stage.pool || stage.add) && failed(materializeStage(stageIndex)))
        return failure();
    }

    Stage &finalStage = stages.back();
    int64_t finalPanels = finalStage.outputChannels / kTile;
    // Generate the region once for all output-channel panels.  Rebuilding the
    // producer tree once per panel is quadratic in stage count and was the
    // source of multi-gigabyte compiler RSS on ResNet.
    TileBanks output = allocateTile(finalPanels, 1, zeroI8);
    if (output.banks.size() != 1 ||
        failed(emitInto(stages.size() - 1, zero, zero, 1, 1, zero, finalPanels,
                        output, 0, 1)))
      return failure();
    Value packed = b.create<memref::AllocOp>(
        loc, MemRefType::get({finalPanels, kTile}, b.getI8Type()));
    Value stored = mvoutBank(b, loc, packed, output.banks.front(), finalPanels);
    b.create<FenceOp>(loc);
    auto panelLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, finalPanels), one);
    b.setInsertionPointToStart(panelLoop.getBody());
    Value panel = panelLoop.getInductionVar();
    auto outputLoop = b.create<scf::ForOp>(
        loc, zero, b.create<arith::ConstantIndexOp>(loc, kTile), one);
    b.setInsertionPointToStart(outputLoop.getBody());
    Value lane = outputLoop.getInductionVar();
    Value channel = b.create<arith::AddIOp>(
        loc,
        b.create<arith::MulIOp>(loc, panel,
                                b.create<arith::ConstantIndexOp>(loc, kTile)),
        lane);
    Value value =
        b.create<memref::LoadOp>(loc, packed, ValueRange{panel, lane});
    b.create<memref::StoreOp>(loc, value, kernel.getOutput(),
                              ValueRange{zero, zero, zero, channel});
    b.setInsertionPointAfter(outputLoop);
    b.setInsertionPointAfter(panelLoop);
    releaseBank(b, loc, stored);
    releaseBank(b, loc, zeroBank);
    for (Value pack : hostPacks)
      b.create<memref::DeallocOp>(loc, pack);
    b.create<memref::DeallocOp>(loc, packed);
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
