//===- MegaConv2dToBankSSAPatterns.cpp - Mega Conv2D to banks -----------===//

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

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;

template <typename MegaOp, bool kDepthwise>
class MegaConv2dToBankSSAPattern : public OpRewritePattern<MegaOp> {
public:
  using OpRewritePattern<MegaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MegaOp op, PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto inputTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto weightTy = dyn_cast<MemRefType>(op.getWeight().getType());
    auto biasTy = dyn_cast<MemRefType>(op.getBias().getType());
    auto scaleTy = dyn_cast<MemRefType>(op.getScale().getType());
    auto lutTy = dyn_cast<MemRefType>(op.getLut().getType());
    auto outputTy = dyn_cast<MemRefType>(op.getOutput().getType());
    if (op->template getParentOfType<MegaKernelOp>())
      return failure();
    bool finalOutput = true;
    if (!inputTy || !weightTy || !biasTy || !scaleTy || !lutTy || !outputTy ||
        !inputTy.hasStaticShape() || !weightTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !scaleTy.hasStaticShape() ||
        !lutTy.hasStaticShape() || !outputTy.hasStaticShape())
      return op.emitError("Mega Conv2D requires static memrefs");
    if (!inputTy.getElementType().isInteger(8) ||
        !weightTy.getElementType().isInteger(8) ||
        !biasTy.getElementType().isInteger(32) ||
        !scaleTy.getElementType().isF32() ||
        !lutTy.getElementType().isInteger(8) || lutTy.getRank() != 1 ||
        lutTy.getShape()[0] != 1 || op.getActivation() < 0 ||
        op.getActivation() > 1 ||
        (finalOutput ? !outputTy.getElementType().isF32()
                     : !outputTy.getElementType().isInteger(8)))
      return op.emitError(
          "Mega Conv2D input/weight must be INT8, bias INT32, scale FP32, "
          "and only the last stage may output FP32");
    if (op.getPadLow() != op.getPadHigh())
      return op.emitError("Mega Conv2D requires symmetric padding");

    auto inputShape = inputTy.getShape();
    auto weightShape = weightTy.getShape();
    auto outputShape = outputTy.getShape();
    int64_t batch = inputShape[0];
    int64_t inputSize = inputShape[1];
    int64_t cin = inputShape[3];
    int64_t kernelSize = weightShape[0];
    int64_t cout = kDepthwise ? cin : weightShape[3];
    int64_t outputSize = finalOutput ? outputShape[2] : outputShape[1];
    int64_t stride = op.getStride();
    int64_t padding = op.getPadLow();
    bool weightShapeMatches =
        weightShape[1] == kernelSize && weightShape[2] == cin &&
        (kDepthwise ? weightShape[3] == 1 : weightShape[3] == cout);
    if (batch <= 0 || inputSize <= 0 || cin <= 0 || kernelSize <= 0 ||
        cout <= 0 || stride <= 0 || padding < 0 || inputShape[2] != inputSize ||
        !weightShapeMatches || outputShape[0] != batch ||
        (finalOutput ? outputShape[3] != outputSize
                     : outputShape[2] != outputSize) ||
        (finalOutput ? outputShape[1] != cout : outputShape[3] != cout) ||
        biasTy.getShape()[0] != cout || scaleTy.getShape()[0] != cout)
      return op.emitError("Mega Conv2D NHWC/HWCF shape mismatch");
    if (kernelSize > 7)
      return op.emitError("Mega Conv2D kernel size exceeds Im2colBall limit 7");
    int64_t numerator = inputSize + 2 * padding - kernelSize;
    if (numerator < 0 || outputSize != numerator / stride + 1)
      return op.emitError("Mega Conv2D output shape mismatch");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < 64 ||
        target.bankDepth % 4)
      return op.emitError(
          "Mega Conv2D requires 128-bit banks with depth divisible by 4");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return op.emitError("Mega Conv2D requires SMatMulBall outBW=1");
    int64_t kernelRows = (kernelSize * kernelSize + kTile - 1) / kTile * kTile;
    int64_t windowCapacity =
        std::min(target.bankDepth / 4, target.bankDepth / kernelRows * kTile);
    windowCapacity = windowCapacity / kTile * kTile;
    if (windowCapacity <= 0)
      return op.emitError("Mega Conv2D kernel tile exceeds one bank");
    int64_t blockSide = 1;
    while ((blockSide + 1) * (blockSide + 1) <= windowCapacity)
      ++blockSide;
    blockSide = std::min(blockSide, outputSize);
    while (blockSide > 0 && (outputSize % blockSide != 0 ||
                             ((blockSide - 1) * stride + kernelSize) *
                                     ((blockSide - 1) * stride + kernelSize) >
                                 target.bankDepth))
      --blockSide;
    if (blockSide == 0)
      return op.emitError("Mega Conv2D input tile exceeds one bank");
    int64_t blockInputSize = (blockSide - 1) * stride + kernelSize;
    int64_t inputRows = blockInputSize * blockInputSize;

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value sixteen = b.create<arith::ConstantIndexOp>(loc, kTile);
    Value outputSizeValue = b.create<arith::ConstantIndexOp>(loc, outputSize);
    Value blockSideValue = b.create<arith::ConstantIndexOp>(loc, blockSide);
    Value blockInputSizeValue =
        b.create<arith::ConstantIndexOp>(loc, blockInputSize);
    Value inputSizeValue = b.create<arith::ConstantIndexOp>(loc, inputSize);
    Value coutValue = b.create<arith::ConstantIndexOp>(loc, cout);
    Value paddedCoutValue = b.create<arith::ConstantIndexOp>(
        loc, (cout + kTile - 1) / kTile * kTile);
    Value strideValue = b.create<arith::ConstantIndexOp>(loc, stride);
    Value paddingValue = b.create<arith::ConstantIndexOp>(loc, padding);
    Value zeroI8 =
        b.create<arith::ConstantOp>(loc, b.getI8Type(), b.getI8IntegerAttr(0));

    auto outputChannelTileLoop =
        b.create<scf::ForOp>(loc, zero, paddedCoutValue, sixteen);
    b.setInsertionPointToStart(outputChannelTileLoop.getBody());
    Value n0 = outputChannelTileLoop.getInductionVar();
    auto biasPackTy = MemRefType::get({4, 4}, b.getI32Type());
    auto scalePackTy = MemRefType::get({4, 4}, b.getF32Type());
    Value biasPack = b.create<memref::AllocOp>(loc, biasPackTy);
    Value scalePack = b.create<memref::AllocOp>(loc, scalePackTy);
    Value zeroI32 = b.create<arith::ConstantOp>(loc, b.getI32Type(),
                                                b.getI32IntegerAttr(0));
    Value oneF32 = b.create<arith::ConstantOp>(loc, b.getF32Type(),
                                               b.getF32FloatAttr(1.0));
    b.create<linalg::FillOp>(loc, zeroI32, biasPack);
    b.create<linalg::FillOp>(loc, oneF32, scalePack);
    for (int64_t i = 0; i < kTile; ++i) {
      Value source = b.create<arith::AddIOp>(
          loc, n0, b.create<arith::ConstantIndexOp>(loc, i));
      Value row = b.create<arith::ConstantIndexOp>(loc, i / 4);
      Value column = b.create<arith::ConstantIndexOp>(loc, i % 4);
      Value validChannel = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, source, coutValue);
      auto storeChannel = b.create<scf::IfOp>(loc, validChannel, false);
      b.setInsertionPointToStart(&storeChannel.getThenRegion().front());
      Value bias = b.create<memref::LoadOp>(loc, op.getBias(), source);
      Value scale = b.create<memref::LoadOp>(loc, op.getScale(), source);
      b.create<memref::StoreOp>(loc, bias, biasPack, ValueRange{row, column});
      b.create<memref::StoreOp>(loc, scale, scalePack, ValueRange{row, column});
      b.setInsertionPointAfter(storeChannel);
    }

    Value biasBank = allocBank(b, loc, 1, 1);
    Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
    Value biasState =
        b.create<BankSMatMulBiasOp>(loc, biasLoaded.getType(), biasLoaded);
    Value scaleBank = allocBank(b, loc, 1, 1);
    Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);

    for (int64_t batchIndex = 0; batchIndex < batch; ++batchIndex) {
      auto outputYLoop =
          b.create<scf::ForOp>(loc, zero, outputSizeValue, blockSideValue);
      b.setInsertionPointToStart(outputYLoop.getBody());
      Value outputY0 = outputYLoop.getInductionVar();
      auto outputXLoop =
          b.create<scf::ForOp>(loc, zero, outputSizeValue, blockSideValue);
      b.setInsertionPointToStart(outputXLoop.getBody());
      Value outputX0 = outputXLoop.getInductionVar();
      int64_t blockWindows = blockSide * blockSide;
      int64_t paddedM = (blockWindows + kTile - 1) / kTile * kTile;
      Value inputBank = allocBank(b, loc, 1, 1);
      Value patchBank = allocBank(b, loc, 1, 1);
      Value weightBank = allocBank(b, loc, 1, 1);
      Value resultBank = allocBank(b, loc, 1, 1);
      Value resultState = resultBank;

      int64_t accumulationBlocks = kDepthwise ? kTile : cin;
      Value accumulationEnd =
          b.create<arith::ConstantIndexOp>(loc, accumulationBlocks);
      auto inputPacksTy = MemRefType::get(
          {accumulationBlocks, inputRows, kTile}, b.getI8Type());
      auto weightPacksTy = MemRefType::get(
          {accumulationBlocks, kernelRows, kTile}, b.getI8Type());
      Value inputPacks = b.create<memref::AllocOp>(loc, inputPacksTy);
      Value weightPacks = b.create<memref::AllocOp>(loc, weightPacksTy);
      b.create<linalg::FillOp>(loc, zeroI8, inputPacks);
      b.create<linalg::FillOp>(loc, zeroI8, weightPacks);

      Value globalInputY0 = b.create<arith::SubIOp>(
          loc, b.create<arith::MulIOp>(loc, outputY0, strideValue),
          paddingValue);
      Value globalInputX0 = b.create<arith::SubIOp>(
          loc, b.create<arith::MulIOp>(loc, outputX0, strideValue),
          paddingValue);
      Value inputBatchValue = b.create<arith::ConstantIndexOp>(loc, batchIndex);
      auto packChannelLoop =
          b.create<scf::ForOp>(loc, zero, accumulationEnd, one);
      b.setInsertionPointToStart(packChannelLoop.getBody());
      Value block = packChannelLoop.getInductionVar();
      Value channelValue =
          kDepthwise ? Value(b.create<arith::AddIOp>(loc, n0, block)) : block;

      auto inputRowLoop =
          b.create<scf::ForOp>(loc, zero, blockInputSizeValue, one);
      b.setInsertionPointToStart(inputRowLoop.getBody());
      Value inputRow = inputRowLoop.getInductionVar();
      auto inputColumnLoop =
          b.create<scf::ForOp>(loc, zero, blockInputSizeValue, one);
      b.setInsertionPointToStart(inputColumnLoop.getBody());
      Value inputColumn = inputColumnLoop.getInductionVar();
      Value globalInputY =
          b.create<arith::AddIOp>(loc, globalInputY0, inputRow);
      Value globalInputX =
          b.create<arith::AddIOp>(loc, globalInputX0, inputColumn);
      Value inBounds = b.create<arith::AndIOp>(
          loc,
          b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                      globalInputY, zero),
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                      globalInputY, inputSizeValue)),
          b.create<arith::AndIOp>(
              loc,
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                      globalInputX, zero),
              b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                      globalInputX, inputSizeValue)));
      if constexpr (kDepthwise)
        inBounds = b.create<arith::AndIOp>(
            loc, inBounds,
            b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                    channelValue, coutValue));
      auto validInput = b.create<scf::IfOp>(loc, inBounds, false);
      b.setInsertionPointToStart(&validInput.getThenRegion().front());
      Value linear = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, inputRow, blockInputSizeValue),
          inputColumn);
      Value inputValue =
          b.create<memref::LoadOp>(loc, op.getInput(),
                                   ValueRange{inputBatchValue, globalInputY,
                                              globalInputX, channelValue});
      b.create<memref::StoreOp>(loc, inputValue, inputPacks,
                                ValueRange{block, linear, zero});
      b.setInsertionPointAfter(inputRowLoop);

      Value kernelEnd = b.create<arith::ConstantIndexOp>(loc, kernelSize);
      auto kernelRowLoop = b.create<scf::ForOp>(loc, zero, kernelEnd, one);
      b.setInsertionPointToStart(kernelRowLoop.getBody());
      Value kernelRow = kernelRowLoop.getInductionVar();
      auto kernelColumnLoop = b.create<scf::ForOp>(loc, zero, kernelEnd, one);
      b.setInsertionPointToStart(kernelColumnLoop.getBody());
      Value kernelColumn = kernelColumnLoop.getInductionVar();
      Value weightRow = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, kernelRow, kernelEnd),
          kernelColumn);
      if constexpr (kDepthwise) {
        Value multiplier = b.create<arith::ConstantIndexOp>(loc, 0);
        Value validChannel = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, channelValue, coutValue);
        auto storeWeight = b.create<scf::IfOp>(loc, validChannel, false);
        b.setInsertionPointToStart(&storeWeight.getThenRegion().front());
        Value weightValue = b.create<memref::LoadOp>(
            loc, op.getWeight(),
            ValueRange{kernelRow, kernelColumn, channelValue, multiplier});
        b.create<memref::StoreOp>(loc, weightValue, weightPacks,
                                  ValueRange{block, weightRow, block});
        b.setInsertionPointAfter(storeWeight);
      } else {
        auto outputChannelLoop = b.create<scf::ForOp>(loc, zero, sixteen, one);
        b.setInsertionPointToStart(outputChannelLoop.getBody());
        Value outputChannel = outputChannelLoop.getInductionVar();
        Value sourceChannel = b.create<arith::AddIOp>(loc, n0, outputChannel);
        Value validChannel = b.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, sourceChannel, coutValue);
        auto storeWeight = b.create<scf::IfOp>(loc, validChannel, false);
        b.setInsertionPointToStart(&storeWeight.getThenRegion().front());
        Value weightValue = b.create<memref::LoadOp>(
            loc, op.getWeight(),
            ValueRange{kernelRow, kernelColumn, channelValue, sourceChannel});
        b.create<memref::StoreOp>(loc, weightValue, weightPacks,
                                  ValueRange{block, weightRow, outputChannel});
        b.setInsertionPointAfter(storeWeight);
      }
      b.setInsertionPointAfter(kernelRowLoop);
      b.setInsertionPointAfter(packChannelLoop);

      auto emitAccumulation = [&](Value channel, Value inputState,
                                  Value patchState, Value weightState,
                                  Value accumulatorState, bool first,
                                  bool last) -> SmallVector<Value> {
        SmallVector<OpFoldResult> inputOffsets = {channel, b.getIndexAttr(0),
                                                  b.getIndexAttr(0)};
        SmallVector<OpFoldResult> inputSizes = {b.getIndexAttr(1),
                                                b.getIndexAttr(inputRows),
                                                b.getIndexAttr(kTile)};
        SmallVector<OpFoldResult> strides(3, b.getIndexAttr(1));
        Value inputSlice = b.create<memref::SubViewOp>(
            loc, inputPacks, inputOffsets, inputSizes, strides);
        Value inputPack = b.create<memref::CollapseShapeOp>(
            loc, inputSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});
        SmallVector<OpFoldResult> weightOffsets = {channel, b.getIndexAttr(0),
                                                   b.getIndexAttr(0)};
        SmallVector<OpFoldResult> weightSizes = {b.getIndexAttr(1),
                                                 b.getIndexAttr(kernelRows),
                                                 b.getIndexAttr(kTile)};
        Value weightSlice = b.create<memref::SubViewOp>(
            loc, weightPacks, weightOffsets, weightSizes, strides);
        Value weightPack = b.create<memref::CollapseShapeOp>(
            loc, weightSlice, SmallVector<ReassociationIndices>{{0, 1}, {2}});

        Value inputLoaded = mvinBank(b, loc, inputPack, inputState, inputRows);
        Value patchLoaded = b.create<BankIm2colOp>(
            loc, patchState.getType(), inputLoaded, patchState,
            createI64Const(b, loc, blockInputSize),
            createI64Const(b, loc, kernelSize), createI64Const(b, loc, stride),
            createI64Const(b, loc, 0), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
            b.getI64IntegerAttr(blockWindows));
        Value weightLoaded =
            mvinBank(b, loc, weightPack, weightState, kernelRows);
        Value accumulated =
            b.create<BankSMatMulOp>(
                 loc, accumulatorState.getType(), patchLoaded, weightLoaded,
                 accumulatorState,
                 createI64ConstU(b, loc, matrixRs2(paddedM, kTile, kernelRows)),
                 b.getBoolAttr(first), b.getBoolAttr(last))
                .getWrBankOut();
        return {inputLoaded, patchLoaded, weightLoaded, accumulated};
      };

      SmallVector<Value> states =
          emitAccumulation(zero, inputBank, patchBank, weightBank, resultState,
                           true, accumulationBlocks == 1);
      if (accumulationBlocks > 2) {
        Value lastChannel =
            b.create<arith::ConstantIndexOp>(loc, accumulationBlocks - 1);
        auto middleLoop = b.create<scf::ForOp>(loc, one, lastChannel, one,
                                               ValueRange(states));
        b.setInsertionPointToStart(middleLoop.getBody());
        ValueRange iterStates = middleLoop.getRegionIterArgs();
        SmallVector<Value> next = emitAccumulation(
            middleLoop.getInductionVar(), iterStates[0], iterStates[1],
            iterStates[2], iterStates[3], false, false);
        b.create<scf::YieldOp>(loc, next);
        b.setInsertionPointAfter(middleLoop);
        states.assign(middleLoop.getResults().begin(),
                      middleLoop.getResults().end());
      }
      if (accumulationBlocks > 1) {
        Value lastChannel =
            b.create<arith::ConstantIndexOp>(loc, accumulationBlocks - 1);
        states = emitAccumulation(lastChannel, states[0], states[1], states[2],
                                  states[3], false, true);
      }
      inputBank = states[0];
      patchBank = states[1];
      weightBank = states[2];
      resultState = states[3];

      Value outputBank = allocBank(b, loc, 1, 1);
      Value converted =
          finalOutput
              ? b.create<BankInt32ToFp32Op>(
                     loc, outputBank.getType(), resultState, scaleLoaded,
                     outputBank, createI64Const(b, loc, paddedM * 4),
                     b.getBoolAttr(op.getActivation() == 1))
                    .getOutBankOut()
              : b.create<BankQuantI32ToI8Op>(
                     loc, outputBank.getType(), resultState, scaleLoaded,
                     outputBank, createI64Const(b, loc, paddedM * 4),
                     b.getI64IntegerAttr(0), b.getI64IntegerAttr(paddedM),
                     b.getI64IntegerAttr(1), b.getI64IntegerAttr(paddedM),
                     b.getBoolAttr(op.getActivation() == 1))
                    .getOutBankOut();
      releaseBank(b, loc, resultState);

      Type outputElementType =
          finalOutput ? Type(b.getF32Type()) : Type(b.getI8Type());
      auto packedOutputTy =
          MemRefType::get({paddedM, kTile}, outputElementType);
      Value packedOutput = b.create<memref::AllocOp>(loc, packedOutputTy);
      Value stored = mvoutBank(b, loc, packedOutput, converted,
                               finalOutput ? paddedM * 4 : paddedM);
      b.create<FenceOp>(loc);
      b.create<memref::DeallocOp>(loc, inputPacks);
      b.create<memref::DeallocOp>(loc, weightPacks);

      Value outputChannelOffset = n0;
      Value outputBatchValue =
          b.create<arith::ConstantIndexOp>(loc, batchIndex);
      auto outputRowLoop = b.create<scf::ForOp>(loc, zero, blockSideValue, one);
      b.setInsertionPointToStart(outputRowLoop.getBody());
      Value outputRow = outputRowLoop.getInductionVar();
      auto outputColumnLoop =
          b.create<scf::ForOp>(loc, zero, blockSideValue, one);
      b.setInsertionPointToStart(outputColumnLoop.getBody());
      Value outputColumn = outputColumnLoop.getInductionVar();
      Value outputPackedRow = b.create<arith::AddIOp>(
          loc, b.create<arith::MulIOp>(loc, outputRow, blockSideValue),
          outputColumn);
      Value outputChannelEnd = sixteen;
      auto outputChannelLoop =
          b.create<scf::ForOp>(loc, zero, outputChannelEnd, one);
      b.setInsertionPointToStart(outputChannelLoop.getBody());
      Value outputChannel = outputChannelLoop.getInductionVar();
      Value globalChannel =
          b.create<arith::AddIOp>(loc, outputChannelOffset, outputChannel);
      Value outputValue = b.create<memref::LoadOp>(
          loc, packedOutput, ValueRange{outputPackedRow, outputChannel});
      Value globalY = b.create<arith::AddIOp>(loc, outputY0, outputRow);
      Value globalX = b.create<arith::AddIOp>(loc, outputX0, outputColumn);
      Value validChannel = b.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, globalChannel, coutValue);
      auto storeOutput = b.create<scf::IfOp>(loc, validChannel, false);
      b.setInsertionPointToStart(&storeOutput.getThenRegion().front());
      if (finalOutput)
        b.create<memref::StoreOp>(
            loc, outputValue, op.getOutput(),
            ValueRange{outputBatchValue, globalChannel, globalY, globalX});
      else
        b.create<memref::StoreOp>(
            loc, outputValue, op.getOutput(),
            ValueRange{outputBatchValue, globalY, globalX, globalChannel});
      b.setInsertionPointAfter(storeOutput);
      b.setInsertionPointAfter(outputRowLoop);

      releaseBank(b, loc, stored);
      b.create<memref::DeallocOp>(loc, packedOutput);
      releaseBank(b, loc, inputBank);
      releaseBank(b, loc, patchBank);
      releaseBank(b, loc, weightBank);
      b.setInsertionPointAfter(outputXLoop);
      b.setInsertionPointAfter(outputYLoop);
    }

    releaseBank(b, loc, biasState);
    releaseBank(b, loc, scaleLoaded);
    b.create<memref::DeallocOp>(loc, biasPack);
    b.create<memref::DeallocOp>(loc, scalePack);
    b.setInsertionPointAfter(outputChannelTileLoop);

    b.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir::buddy {
void populatePebbleMegaConv2dToBankSSAPatterns(RewritePatternSet &patterns) {
  patterns.add<MegaConv2dToBankSSAPattern<MegaConv2dOp, false>,
               MegaConv2dToBankSSAPattern<MegaConv2dDepthwiseOp, true>>(
      patterns.getContext());
}
} // namespace mlir::buddy
