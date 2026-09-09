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
#include "Trace/TraceOps.h"
#include "Utils/BankUtils.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace ::buddy::buckyball;

namespace {

constexpr int64_t kTile = 16;
constexpr int64_t kInt32Rows = 64;

class MatmulMegaKernelToBankSSAPattern : public OpRewritePattern<MegaKernelOp> {
public:
  MatmulMegaKernelToBankSSAPattern(MLIRContext *context, bool traceMegaStages,
                                   int64_t traceMegaStageStart,
                                   int64_t traceMegaStageLimit)
      : OpRewritePattern<MegaKernelOp>(context),
        traceMegaStages(traceMegaStages),
        traceMegaStageStart(traceMegaStageStart),
        traceMegaStageLimit(traceMegaStageLimit) {}

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

      bool traceStage =
          traceMegaStages && !last &&
          static_cast<int64_t>(stageIndex) >= traceMegaStageStart &&
          (traceMegaStageLimit < 0 ||
           static_cast<int64_t>(stageIndex) < traceMegaStageLimit);
      if (traceStage) {
        b.create<FenceOp>(loc);
        int64_t channelBase = 0;
        for (auto [bankIndex, output] : llvm::enumerate(outputs)) {
          int64_t rows = outputK[bankIndex] / kTile;
          Value pack = b.create<memref::AllocOp>(
              loc, MemRefType::get({rows, kTile}, b.getI8Type()));
          hostPacks.push_back(pack);
          outputs[bankIndex] = mvoutBank(b, loc, pack, output, rows);
          b.create<FenceOp>(loc);
          int64_t validChannels = std::min(outputK[bankIndex], n - channelBase);
          auto channelLoop = b.create<scf::ForOp>(
              loc, zero, b.create<arith::ConstantIndexOp>(loc, validChannels),
              one);
          b.setInsertionPointToStart(channelLoop.getBody());
          Value channel = channelLoop.getInductionVar();
          Value row = b.create<arith::DivUIOp>(loc, channel, sixteen);
          Value column = b.create<arith::RemUIOp>(loc, channel, sixteen);
          Value value =
              b.create<memref::LoadOp>(loc, pack, ValueRange{row, column});
          Value outputChannel = b.create<arith::AddIOp>(
              loc, b.create<arith::ConstantIndexOp>(loc, channelBase), channel);
          b.create<memref::StoreOp>(loc, value, stage.getOutput(),
                                    ValueRange{zero, outputChannel});
          b.setInsertionPointAfter(channelLoop);
          channelBase += outputK[bankIndex];
        }
        auto id = b.getI64IntegerAttr(1000 + stageIndex);
        auto trace = b.create<::buddy::trace::EndOp>(
            loc, stage.getOutput().getType(), stage.getOutput(), id,
            b.getStringAttr("mega-matmul-stage"));
        trace->setAttr("id_path", b.getArrayAttr({id}));
        trace->setAttr("buckyball.stage_trace", b.getUnitAttr());
      }

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
        if (traceMegaStages &&
            traceMegaStageStart <= static_cast<int64_t>(stageIndex) &&
            (traceMegaStageLimit < 0 ||
             static_cast<int64_t>(stageIndex) < traceMegaStageLimit)) {
          auto id = b.getI64IntegerAttr(1001);
          auto trace = b.create<::buddy::trace::EndOp>(
              loc, stage.getOutput().getType(), stage.getOutput(), id,
              b.getStringAttr("classifier-output"));
          trace->setAttr("id_path", b.getArrayAttr({id}));
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

private:
  bool traceMegaStages;
  int64_t traceMegaStageStart;
  int64_t traceMegaStageLimit;
};

} // namespace

namespace mlir::buddy {
void populatePebbleMegaKernelToBankSSAPatterns(RewritePatternSet &patterns,
                                               bool traceMegaStages,
                                               int64_t traceMegaStageStart,
                                               int64_t traceMegaStageLimit) {
  patterns.add<MatmulMegaKernelToBankSSAPattern>(
      patterns.getContext(), traceMegaStages, traceMegaStageStart,
      traceMegaStageLimit);
}
} // namespace mlir::buddy
