//===- LowerBuckyballToBankSSAPatterns.cpp - Mega MatMul to banks --------===//

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
constexpr int64_t kInt32RowsPerTile = 64;

class MegaMatmulToBankSSAPattern : public OpRewritePattern<MegaMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MegaMatmulOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto aTy = dyn_cast<MemRefType>(op.getInput().getType());
    auto wTy = dyn_cast<MemRefType>(op.getWeight().getType());
    auto biasTy = dyn_cast<MemRefType>(op.getBias().getType());
    auto scaleTy = dyn_cast<MemRefType>(op.getScale().getType());
    auto lutTy = dyn_cast<MemRefType>(op.getLut().getType());
    auto outTy = dyn_cast<MemRefType>(op.getOutput().getType());
    auto kernel = op->getParentOfType<MegaKernelOp>();
    if (!kernel)
      return op.emitError(
          "Mega MatMul must be nested in buckyball.mega_kernel");
    bool finalOutput = op.getOutput() == kernel.getOutput();
    if (!aTy || !wTy || !biasTy || !scaleTy || !lutTy || !outTy ||
        !aTy.hasStaticShape() || !wTy.hasStaticShape() ||
        !biasTy.hasStaticShape() || !scaleTy.hasStaticShape() ||
        !lutTy.hasStaticShape() || !outTy.hasStaticShape())
      return op.emitError("Mega MatMul requires static memrefs");
    if (!aTy.getElementType().isInteger(8) ||
        !wTy.getElementType().isInteger(8) ||
        !biasTy.getElementType().isInteger(32) ||
        !scaleTy.getElementType().isF32() ||
        !lutTy.getElementType().isInteger(8) || lutTy.getRank() != 1 ||
        lutTy.getShape()[0] != (op.getActivation() == 2 ? 256 : 1) ||
        op.getActivation() < 0 || op.getActivation() > 2 ||
        (finalOutput && op.getActivation() == 2) ||
        (finalOutput ? !outTy.getElementType().isF32()
                     : !outTy.getElementType().isInteger(8)))
      return op.emitError(
          "Mega MatMul input/weight must be INT8, bias INT32, scale FP32, "
          "and only the last stage may output FP32");

    int64_t M = aTy.getShape()[0];
    int64_t K = aTy.getShape()[1];
    int64_t N = wTy.getShape()[1];
    if (M <= 0 || K <= 0 || N <= 0 || wTy.getShape()[0] != K ||
        biasTy.getShape()[0] != N || scaleTy.getShape()[0] != N ||
        outTy.getShape()[0] != M || outTy.getShape()[1] != N)
      return op.emitError("Mega MatMul shape mismatch");
    const auto &target = buckyball_target::getBuckyballTarget();
    if (target.bankWidthBits != 128 || target.bankDepth < kInt32RowsPerTile ||
        target.bankDepth % kInt32RowsPerTile)
      return op.emitError(
          "Mega MatMul requires 128-bit banks whose depth is a positive "
          "multiple of 64 rows");
    if (buckyball_target::getBuckyballBallMapping("SMatMulBall").outBW != 1)
      return op.emitError("Mega MatMul requires SMatMulBall outBW=1");

    int64_t paddedM = (M + kTile - 1) / kTile * kTile;
    int64_t paddedK = (K + kTile - 1) / kTile * kTile;
    int64_t kChunk = target.bankDepth / kTile * kTile;
    if (kChunk <= 0)
      return op.emitError("bank depth cannot hold one K tile");
    int64_t tilesPerResultBank = target.bankDepth / kInt32RowsPerTile;

    Value aBank = allocBank(b, loc, 1, 1);
    Value wBank = allocBank(b, loc, 1, 1);
    for (int64_t n0 = 0; n0 < N; n0 += kTile) {
      int64_t thisN = std::min(kTile, N - n0);
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
      for (int64_t i = 0; i < thisN; ++i) {
        Value source = b.create<arith::ConstantIndexOp>(loc, n0 + i);
        Value row = b.create<arith::ConstantIndexOp>(loc, i / 4);
        Value column = b.create<arith::ConstantIndexOp>(loc, i % 4);
        Value bias = b.create<memref::LoadOp>(loc, op.getBias(), source);
        Value scale = b.create<memref::LoadOp>(loc, op.getScale(), source);
        b.create<memref::StoreOp>(loc, bias, biasPack, ValueRange{row, column});
        b.create<memref::StoreOp>(loc, scale, scalePack,
                                  ValueRange{row, column});
      }

      Value biasBank = allocBank(b, loc, 1, 1);
      Value biasLoaded = mvinBank(b, loc, biasPack, biasBank, 4);
      Value biasState =
          b.create<BankSMatMulBiasOp>(loc, biasLoaded.getType(), biasLoaded);
      Value scaleBank = allocBank(b, loc, 1, 1);
      Value scaleLoaded = mvinBank(b, loc, scalePack, scaleBank, 4);

      for (int64_t m0 = 0; m0 < paddedM; m0 += tilesPerResultBank * kTile) {
        int64_t tileCount =
            std::min(tilesPerResultBank, (paddedM - m0) / kTile);
        Value resultBank = allocBank(b, loc, 1, 1);
        Value resultState = resultBank;
        SmallVector<Value> hostPacks;

        for (int64_t tile = 0; tile < tileCount; ++tile) {
          int64_t row0 = m0 + tile * kTile;
          int64_t validRows = std::min(kTile, std::max<int64_t>(0, M - row0));
          for (int64_t k0 = 0; k0 < paddedK; k0 += kChunk) {
            int64_t thisK = std::min(kChunk, paddedK - k0);
            int64_t validK = std::min(thisK, std::max<int64_t>(0, K - k0));
            auto aPackTy = MemRefType::get({thisK, kTile}, b.getI8Type());
            Value aPack = b.create<memref::AllocOp>(loc, aPackTy);
            hostPacks.push_back(aPack);
            Value zeroI8 = b.create<arith::ConstantOp>(loc, b.getI8Type(),
                                                       b.getI8IntegerAttr(0));
            b.create<linalg::FillOp>(loc, zeroI8, aPack);

            Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
            Value one = b.create<arith::ConstantIndexOp>(loc, 1);
            Value rowEnd = b.create<arith::ConstantIndexOp>(loc, validRows);
            Value kTileEnd = b.create<arith::ConstantIndexOp>(
                loc, (validK + kTile - 1) / kTile);
            auto ktLoop = b.create<scf::ForOp>(loc, zero, kTileEnd, one);
            b.setInsertionPointToStart(ktLoop.getBody());
            Value kt = ktLoop.getInductionVar();
            auto rowLoop = b.create<scf::ForOp>(loc, zero, rowEnd, one);
            b.setInsertionPointToStart(rowLoop.getBody());
            Value row = rowLoop.getInductionVar();
            Value remaining = b.create<arith::SubIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, validK),
                b.create<arith::MulIOp>(
                    loc, kt, b.create<arith::ConstantIndexOp>(loc, kTile)));
            Value columnEnd = b.create<arith::MinUIOp>(
                loc, remaining, b.create<arith::ConstantIndexOp>(loc, kTile));
            auto columnLoop = b.create<scf::ForOp>(loc, zero, columnEnd, one);
            b.setInsertionPointToStart(columnLoop.getBody());
            Value column = columnLoop.getInductionVar();
            Value sourceRow = b.create<arith::AddIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, row0), row);
            Value sourceColumn = b.create<arith::AddIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, k0),
                b.create<arith::AddIOp>(
                    loc,
                    b.create<arith::MulIOp>(
                        loc, kt, b.create<arith::ConstantIndexOp>(loc, kTile)),
                    column));
            Value packedRow = b.create<arith::AddIOp>(
                loc,
                b.create<arith::MulIOp>(
                    loc, kt, b.create<arith::ConstantIndexOp>(loc, kTile)),
                row);
            Value value = b.create<memref::LoadOp>(
                loc, op.getInput(), ValueRange{sourceRow, sourceColumn});
            b.create<memref::StoreOp>(loc, value, aPack,
                                      ValueRange{packedRow, column});
            b.setInsertionPointAfter(ktLoop);

            auto wPackTy = MemRefType::get({thisK, kTile}, b.getI8Type());
            Value wPack = b.create<memref::AllocOp>(loc, wPackTy);
            hostPacks.push_back(wPack);
            b.create<linalg::FillOp>(loc, zeroI8, wPack);
            Value validKEnd = b.create<arith::ConstantIndexOp>(loc, validK);
            Value validNEnd = b.create<arith::ConstantIndexOp>(loc, thisN);
            auto weightRowLoop =
                b.create<scf::ForOp>(loc, zero, validKEnd, one);
            b.setInsertionPointToStart(weightRowLoop.getBody());
            Value weightRow = weightRowLoop.getInductionVar();
            auto weightColumnLoop =
                b.create<scf::ForOp>(loc, zero, validNEnd, one);
            b.setInsertionPointToStart(weightColumnLoop.getBody());
            Value weightColumn = weightColumnLoop.getInductionVar();
            Value sourceWeightRow = b.create<arith::AddIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, k0), weightRow);
            Value sourceWeightColumn = b.create<arith::AddIOp>(
                loc, b.create<arith::ConstantIndexOp>(loc, n0), weightColumn);
            Value weightValue = b.create<memref::LoadOp>(
                loc, op.getWeight(),
                ValueRange{sourceWeightRow, sourceWeightColumn});
            b.create<memref::StoreOp>(loc, weightValue, wPack,
                                      ValueRange{weightRow, weightColumn});
            b.setInsertionPointAfter(weightRowLoop);

            Value aLoaded = mvinBank(b, loc, aPack, aBank, thisK);
            Value wLoaded = mvinBank(b, loc, wPack, wBank, thisK);
            uint64_t cfg = matrixRs2(kTile, kTile, thisK);
            auto smatmul = b.create<BankSMatMulOp>(
                loc, resultState.getType(), aLoaded, wLoaded, resultState,
                createI64Const(b, loc, static_cast<int64_t>(cfg)),
                b.getBoolAttr(k0 == 0), b.getBoolAttr(k0 + thisK == paddedK));
            resultState = smatmul.getWrBankOut();
            aBank = aLoaded;
            wBank = wLoaded;
          }
        }

        Value outputBank = allocBank(b, loc, 1, 1);
        Value converted =
            finalOutput
                ? b.create<BankInt32ToFp32Op>(
                       loc, outputBank.getType(), resultState, scaleLoaded,
                       outputBank,
                       createI64Const(b, loc, tileCount * kInt32RowsPerTile),
                       b.getBoolAttr(op.getActivation() == 1))
                      .getOutBankOut()
                : b.create<BankQuantI32ToI8Op>(
                       loc, outputBank.getType(), resultState, scaleLoaded,
                       outputBank,
                       createI64Const(b, loc, tileCount * kInt32RowsPerTile),
                       b.getI64IntegerAttr(0),
                       b.getI64IntegerAttr(tileCount * kTile),
                       b.getI64IntegerAttr(1),
                       b.getI64IntegerAttr(tileCount * kTile),
                       b.getBoolAttr(op.getActivation() == 1))
                      .getOutBankOut();
        releaseBank(b, loc, resultState);

        Type outputElementType =
            finalOutput ? Type(b.getF32Type()) : Type(b.getI8Type());
        auto packedOutTy =
            MemRefType::get({tileCount * kTile, kTile}, outputElementType);
        Value packedOut = b.create<memref::AllocOp>(loc, packedOutTy);
        Value stored = mvoutBank(b, loc, packedOut, converted,
                                 finalOutput ? tileCount * kInt32RowsPerTile
                                             : tileCount * kTile);
        b.create<FenceOp>(loc);
        for (Value pack : hostPacks)
          b.create<memref::DeallocOp>(loc, pack);

        Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
        Value one = b.create<arith::ConstantIndexOp>(loc, 1);
        int64_t validOutputRows = std::min(tileCount * kTile, M - m0);
        Value rowEnd = b.create<arith::ConstantIndexOp>(loc, validOutputRows);
        Value columnEnd = b.create<arith::ConstantIndexOp>(loc, thisN);
        auto rowLoop = b.create<scf::ForOp>(loc, zero, rowEnd, one);
        b.setInsertionPointToStart(rowLoop.getBody());
        Value row = rowLoop.getInductionVar();
        auto columnLoop = b.create<scf::ForOp>(loc, zero, columnEnd, one);
        b.setInsertionPointToStart(columnLoop.getBody());
        Value column = columnLoop.getInductionVar();
        Value value =
            b.create<memref::LoadOp>(loc, packedOut, ValueRange{row, column});
        Value outputRow = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, m0), row);
        Value outputColumn = b.create<arith::AddIOp>(
            loc, b.create<arith::ConstantIndexOp>(loc, n0), column);
        b.create<memref::StoreOp>(loc, value, op.getOutput(),
                                  ValueRange{outputRow, outputColumn});
        b.setInsertionPointAfter(rowLoop);
        releaseBank(b, loc, stored);
        b.create<memref::DeallocOp>(loc, packedOut);
      }

      releaseBank(b, loc, biasState);
      releaseBank(b, loc, scaleLoaded);
      b.create<memref::DeallocOp>(loc, biasPack);
      b.create<memref::DeallocOp>(loc, scalePack);
    }

    releaseBank(b, loc, aBank);
    releaseBank(b, loc, wBank);
    b.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::buddy::populateSMatMulBallLowerBuckyballToBankSSAPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MegaMatmulToBankSSAPattern>(patterns.getContext());
}
